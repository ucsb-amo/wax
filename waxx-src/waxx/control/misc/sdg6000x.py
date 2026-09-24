"""Siglent SDG6000X arbitrary waveform generator over VXI-11 (LAN).

Link-failure policy
-------------------
The channel methods that kernels reach by RPC (``set``, ``set_output``,
``sweep``, ``init``) never raise on a communications failure: a failed write
is printed loudly (the hardware was NOT updated) and the cached value for
that setting is marked stale, so the next request for it is written again
rather than skipped as "unchanged". Reads (``get_frequency`` /
``fetch_state``) raise, and callers treat that as "value unknown".

The instrument carries a :class:`~waxx.util.link_latch.LinkLatch`: after a
failure every call is skipped (cheaply) for ``T_LINK_RETRY`` seconds, then
tried again. Connect and reply timeouts are ``T_TIMEOUT`` (the python-vxi11
defaults are an unbounded connect and a 10 s reply).
"""

import socket
import time

from artiq.coredevice.core import Core
from artiq.language import now_mu, kernel, delay, portable, TFloat

import vxi11
from vxi11 import rpc
from vxi11 import vxi11 as _vxi11   # CoreClient and the DEVICE_CORE_* ids are not re-exported

from waxx.util.artiq.async_print import aprint
from waxx.util.link_latch import LinkLatch, LinkDownError, T_LINK_RETRY

T_RPC_DELAY = 20.e-3

# Seconds allowed for a TCP connect and for each VXI-11 reply.
T_TIMEOUT = 2.

dv = -0.1

class SDG6000X_Params():
    def __init__(self,
                 frequency=0.,
                 amplitude_vpp=0.,
                 state=0,
                 max_amplitude_vpp=0.,
                 max_frequency=0.,
                 min_frequency=0.):
        self.frequency = frequency
        self.amplitude_vpp = amplitude_vpp
        self.max_amplitude_vpp = max_amplitude_vpp
        self.state = state
        self.max_frequency = max_frequency
        self.min_frequency = min_frequency


class _BoundedConnect:
    """Mixin for python-vxi11 RPC clients: connect with a timeout.

    ``rpc.RawTCPClient.connect`` calls ``socket.connect`` on a socket with no
    timeout, so an unreachable host blocks for the OS default (~20 s on
    Windows) -- and the VXI-11 open does that twice (portmapper, then the
    instrument link). ``_connect_timeout`` must be set before ``__init__``
    runs, because ``RawTCPClient.__init__`` connects immediately.
    """
    _connect_timeout = T_TIMEOUT

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port),
                                             timeout=self._connect_timeout)


class _PortMapperClient(_BoundedConnect, rpc.TCPPortMapperClient):
    def __init__(self, host, connect_timeout):
        self._connect_timeout = connect_timeout
        super().__init__(host)


class _CoreClient(_BoundedConnect, _vxi11.CoreClient):
    def __init__(self, host, port, connect_timeout):
        self._connect_timeout = connect_timeout
        super().__init__(host, port)


class SDG6000X(vxi11.Instrument):
    def __init__(self, ip, timeout=T_TIMEOUT, retry_after=T_LINK_RETRY):
        super().__init__(ip)
        self.ip = ip
        self.timeout = timeout          # VXI-11 reply timeout (vxi11 default 10 s)
        self._connect_timeout = timeout
        self.latch = LinkLatch(f"siglent {ip}", retry_after=retry_after)

    # ---- connection ------------------------------------------------------

    def open(self):
        """As ``vxi11.Device.open`` but with bounded TCP connects."""
        if self.link is None and self.client is None:
            pmap = _PortMapperClient(self.host, self._connect_timeout)
            try:
                port = pmap.get_port((_vxi11.DEVICE_CORE_PROG,
                                      _vxi11.DEVICE_CORE_VERS,
                                      rpc.IPPROTO_TCP, 0))
            finally:
                pmap.close()
            if port == 0:
                raise rpc.RPCError('program not registered')
            self.client = _CoreClient(self.host, port, self._connect_timeout)
        super().open()

    def _drop_connection(self):
        """Forget a (possibly dead) link without talking to the instrument.

        ``Device.close`` sends destroy_link over the socket, which is exactly
        what fails when the LAN is down. The next call reopens from scratch.
        """
        try:
            if self.client is not None:
                self.client.close()
        except Exception:
            pass
        self.client = None
        self.link = None

    def _guarded(self, fn, *args):
        self.latch.raise_if_skipping()
        try:
            result = fn(*args)
        except Exception as e:
            self.latch.trip(e)
            self._drop_connection()
            raise
        self.latch.clear()
        return result

    @property
    def link_down(self) -> bool:
        return self.latch.down

    # ---- I/O (every instrument access goes through these two) ------------

    def write(self, message, encoding='utf-8'):
        self._guarded(super().write, message, encoding)

    def ask(self, message, num=-1, encoding='utf-8'):
        return self._guarded(super().ask, message, num, encoding)

    # ---- commands --------------------------------------------------------

    def _set_amp_command(self,ch,amp):
        self.write(f"C{ch}:BSWV AMP,{amp}")

    def _set_freq_command(self,ch,freq):
        self.write(f"C{ch}:BSWV FRQ,{freq}")

    def _sw_output(self,ch,state=0):
        if state == 1:
            s = "ON"
        else:
            s = "OFF"
        self.write(f"C{ch}:OUTP {s}")

class SDG6000X_CH():
    def __init__(self,
                 ch,ip,
                 frequency,
                 amplitude_vpp,
                 default_state=1,
                 max_amplitude_vpp=1.,
                 min_frequency=0.,
                 max_frequency=500.e6,
                 core=Core):

        self._instr = SDG6000X(ip)
        self.ch = ch

        self._p = SDG6000X_Params(frequency=frequency,
                                amplitude_vpp=amplitude_vpp,
                                state=default_state,
                                max_amplitude_vpp=max_amplitude_vpp,
                                min_frequency=min_frequency,
                                max_frequency=max_frequency)

        # Settings whose cached value in self._p is NOT known to match the
        # hardware (a write failed, or a read-back failed). A stale setting is
        # always written on the next request, even if the value is unchanged.
        self._stale = set()

        # self._frequency_default = 0.
        # self._amplitude_vpp_default = 0.
        self._stash_defaults()

        self.core = core

    @property
    def _label(self) -> str:
        return f"[siglent {self._instr.ip} ch{self.ch}]"

    def _report_failure(self, what, e):
        """Every failed write is printed: the hardware was NOT updated."""
        print(f"{self._label} *** {what} FAILED -- hardware NOT updated "
              f"({type(e).__name__}: {e}) ***")

    def _stash_defaults(self):
        self._frequency_default = self._p.frequency
        self._amplitude_vpp_default = self._p.amplitude_vpp

    def _restore_defaults(self):
        self._p.frequency = self._frequency_default
        self._p.amplitude_vpp = self._amplitude_vpp_default

    def set_output_rpc(self,state=1,init=False):
        # Turns out it's better just to poll the state of the device, then only
        # turn it on if the device was off before. Otherwise, sending the "ON"
        # command again causes a brief interruption of output.

        # Thus the `init` option has been removed.

        # if init:
        #   sw_changed = True else:
        sw_changed = bool(state) != (self._p.state == 1) or 'state' in self._stale

        if sw_changed:
            new_state = state if state >= 0. else self._p.state
            try:
                self._instr._sw_output(self.ch,new_state)
            except Exception as e:
                self._stale.add('state')
                self._report_failure(f"output {'ON' if new_state else 'OFF'}", e)
                return
            self._p.state = new_state
            self._stale.discard('state')

    def get_frequency(self) -> TFloat:
        """Read back the set frequency (Hz). Raises when the link is down."""
        self.fetch_state()
        return self._p.frequency

    def fetch_state(self):
        """Read frequency, amplitude and output state back from the hardware.

        Raises on a communications failure (after marking every cached
        setting stale); callers decide what "unknown" means for them.
        """
        try:
            reply = self._instr.ask(f"C{self.ch}:BSWV?")
            def parse_params(response):
                frq_start = response.find('FRQ,') + 4
                frq_end = response.find('HZ', frq_start)
                if frq_start > 3 and frq_end != -1:
                    freq = float(response[frq_start:frq_end])

                amp_start = response.find('AMP,') + 4
                amp_end = response.find('V', amp_start)
                if amp_start > 3 and amp_end != -1:
                    amp = float(response[amp_start:amp_end])
                return freq, amp
            freq, amp = parse_params(reply)

            reply = self._instr.ask(f"C{self.ch}:OUTP?")
            def parse_output_state(response):
                outp_start = response.find('OUTP ') + 5
                outp_end = response.find(',', outp_start)
                if outp_start > 4 and outp_end != -1:
                    string = response[outp_start:outp_end]
                    if string == 'ON' :
                        state = 1
                    elif string == 'OFF':
                        state = 0
                    return state
                return None
            state = parse_output_state(reply)
        except Exception:
            self._stale.update(('frequency', 'amplitude', 'state'))
            raise

        self._p.frequency = freq
        self._p.amplitude_vpp = amp
        self._p.state = state
        self._stale.clear()

    def set_rpc(self,
            frequency=dv,
            amplitude=dv,
            init=False):

        if init:
            freq_changed = True
            amp_changed = True
            self._restore_defaults()
        else:
            freq_changed = (frequency >= 0.) and (frequency != self._p.frequency
                                                  or 'frequency' in self._stale)
            amp_changed = (amplitude >= 0.) and (amplitude != self._p.amplitude_vpp
                                                 or 'amplitude' in self._stale)
        if freq_changed:
            self._p.frequency = frequency if frequency!=dv else self._p.frequency
            if self._p.frequency > self._p.max_frequency:
                self._p.frequency = self._p.max_frequency
                aprint("Requested siglent freuqency exceeds configured maximum, setting to max.")
            elif self._p.frequency < self._p.min_frequency:
                self._p.frequency = self._p.min_frequency
                aprint("Requested siglent freuqency exceeds configured minimum, setting to min.")
            try:
                self._instr._set_freq_command(self.ch,self._p.frequency)
            except Exception as e:
                self._stale.add('frequency')
                self._report_failure(f"set frequency={self._p.frequency/1.e6:.4f} MHz", e)
            else:
                self._stale.discard('frequency')
        if amp_changed:
            if self._p.amplitude_vpp > self._p.max_amplitude_vpp:
                raise ValueError("Amplitdue requested for this channel is beyond configured maximum.")
            self._p.amplitude_vpp = amplitude if amplitude!=dv else self._p.amplitude_vpp
            try:
                self._instr._set_amp_command(self.ch,self._p.amplitude_vpp)
            except Exception as e:
                self._stale.add('amplitude')
                self._report_failure(f"set amplitude={self._p.amplitude_vpp} Vpp", e)
            else:
                self._stale.discard('amplitude')

    @kernel
    def sweep(self, frequency_end=dv, frequency_step=1.e6, reset=False):
        self.core.wait_until_mu(now_mu())
        self.sweep_rpc(frequency_end, frequency_step, reset)
        self.core.break_realtime()

    def sweep_rpc(self,
                  frequency_end=dv,
                  frequency_step=1.e6,
                  reset=False):
        T = 0.1

        # The sweep starts from the frequency the hardware is actually at. If
        # that cannot be read there is nothing safe to do: report and leave
        # the hardware alone.
        try:
            self.fetch_state()
        except Exception as e:
            if frequency_end == dv or reset:
                target = "default"
            else:
                target = f"{frequency_end/1.e6:.4f} MHz"
            self._report_failure(f"sweep to {target} (could not read the "
                                 f"current frequency)", e)
            return

        if frequency_end == dv or reset == True:
            frequency_end = self._frequency_default

        f0 = self._p.frequency
        ff = frequency_end

        if f0 == ff:
            return

        direction = 1 if ff >= f0 else -1
        df = abs(frequency_step) * direction
        f = f0
        while direction * (ff - f) > abs(df):
            f += df
            time.sleep(T)
            self.set_rpc(f)
            if 'frequency' in self._stale:
                # A step failed (set_rpc reported it). Stop rather than keep
                # stepping a frequency the hardware is not following.
                return
        time.sleep(T)
        self.set_rpc(ff)

    def init_rpc(self):
        self.sweep_rpc(reset=True)
        self.set_output_rpc(state=1)

    @kernel
    def init(self):
        self.core.wait_until_mu(now_mu())
        self.init_rpc()
        self.core.break_realtime()

    @kernel
    def set(self,frequency=dv,amplitude=dv,init=False):
        self.core.wait_until_mu(now_mu())
        self.set_rpc(frequency,amplitude,init)
        # A slow RPC (link timeout) must not leave the timeline behind the
        # wall clock: an RTIOUnderflow here aborts the whole scan.
        self.core.break_realtime()
        delay(T_RPC_DELAY)

    @kernel
    def set_output(self,state=1,init=False):
        self.core.wait_until_mu(now_mu())
        self.set_output_rpc(state,init)
        self.core.break_realtime()
        delay(T_RPC_DELAY)
