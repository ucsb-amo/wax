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

import re
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
        # Check the amplitude that would be written -- before anything is
        # sent. (This used to check the previously cached value, so a request
        # above the maximum was written and only the call after it raised.)
        if amp_changed:
            new_amp = amplitude if amplitude != dv else self._p.amplitude_vpp
            if new_amp > self._p.max_amplitude_vpp:
                raise ValueError(f"{self._label} amplitude {new_amp} Vpp requested, above "
                                 f"the configured maximum {self._p.max_amplitude_vpp} Vpp; "
                                 f"nothing sent.")
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
            self._p.amplitude_vpp = new_amp
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


# ---------------------------------------------------------------------------
# FM channel: fixed carrier, sine FM from the internal source
# ---------------------------------------------------------------------------
# Ported from k-jam/jk/siglent_fm_control.ipynb (2026-11-05).


# Seconds between a write and its read-back.
T_FM_SETTLE = 0.05

_FM_KEYS = ('frequency_carrier', 'amplitude_vpp', 'fm_deviation', 'fm_rate')


def _num(s):
    """'1000HZ' -> 1000.0, '0.5V' -> 0.5; non-numeric -> the string unchanged."""
    m = re.match(r'^\s*([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)', s)
    return float(m.group(1)) if m else s.strip()


def _pairs(tokens):
    return {tokens[i].strip().upper(): _num(tokens[i + 1])
            for i in range(0, len(tokens) - 1, 2)}


def _body(reply):
    return reply.strip().strip('\x00').split(' ', 1)[-1]


def parse_mdwv(reply):
    """'C1:MDWV STATE,ON,FM,MDSP,SINE,SRC,INT,FRQ,100HZ,DEVI,1000HZ,CARR,WVTP,...'
    -> {'STATE': 'ON', 'TYPE': 'FM', 'FM': {...}, 'CARR': {...}, 'raw': ...}"""
    toks = [t.strip() for t in _body(reply).split(',')]
    out = {'raw': reply.strip().strip('\x00')}
    if 'CARR' in toks:
        i = toks.index('CARR')
        toks, carr = toks[:i], toks[i + 1:]
        out['CARR'] = _pairs(carr)
    if len(toks) >= 2 and toks[0].upper() == 'STATE':
        out['STATE'] = toks[1].upper()
        toks = toks[2:]
    if toks and toks[0]:
        out['TYPE'] = toks[0].upper()
        out[out['TYPE']] = _pairs(toks[1:])
    return out


def parse_bswv(reply):
    return _pairs([t.strip() for t in _body(reply).split(',')])


def parse_outp(reply):
    """'C1:OUTP ON,LOAD,50,PLRT,NOR' -> {'STATE': 'ON', 'LOAD': 50.0, 'PLRT': 'NOR'}"""
    toks = [t.strip() for t in _body(reply).split(',')]
    out = _pairs(toks[1:])
    out['STATE'] = toks[0].upper() if toks else None
    return out


class SDG6000X_FM_CH():
    """One SDG6000X channel used as a fixed-carrier FM source (sine FM,
    internal modulation source).

    Experiments set the carrier amplitude (Vpp), the FM deviation and the FM
    rate (Hz). The carrier frequency is fixed at ``frequency_carrier``:
    ``init()`` writes it only if the instrument reads something else, and
    nothing else ever writes it.

    Limits, enforced before anything is sent (``ValueError``, nothing
    written; raised from a kernel this aborts the run):

    - amplitude 0 .. ``max_amplitude_vpp``,
    - FM deviation 0 .. ``max_fm_deviation``, FM rate 0 .. ``max_fm_rate``.

    The amplitude is what the instrument is programmed to deliver into the
    load set on the channel (``OUTP LOAD``); ``init()`` prints that load.

    Behaviour of the instrument this class works around (SDG6022X, seen
    2026-11-05): FM parameter writes are ignored while modulation is off, and
    modulation comes on with the last-used type (AM came up). So FM values
    set while FM is off are kept and written by ``fm_on()``, which switches
    modulation on, selects FM (internal source, sine), writes deviation and
    rate, and reads back; if the result is not FM at the requested values it
    switches modulation off again and says so.

    Every write is read back (the instrument clamps out-of-range values
    silently). A setting that failed to write, could not be read back, or
    reads back different from the request is printed loudly and kept
    *stale*: it is written again at the next call. Communications failures
    never raise (module policy above); they are printed.

    Each call is a host RPC of a few tens of ms (write + ``T_FM_SETTLE`` +
    read-back): not for time-critical parts of a sequence. Nothing resets the
    instrument at the end of a run; it stays at the last values set.
    """

    def __init__(self, ch, ip,
                 frequency_carrier,
                 max_amplitude_vpp,
                 fm_deviation=0.,
                 fm_rate=0.,
                 max_fm_deviation=10.e6,
                 max_fm_rate=1.e6,
                 core=Core):
        if not max_amplitude_vpp > 0.:
            raise ValueError(f"max_amplitude_vpp must be > 0, got {max_amplitude_vpp}")
        self._instr = SDG6000X(ip)
        self.ch = ch
        self.frequency_carrier = frequency_carrier
        self.max_amplitude_vpp = max_amplitude_vpp
        self.max_fm_deviation = max_fm_deviation
        self.max_fm_rate = max_fm_rate
        self._check(dv, fm_deviation, fm_rate)
        self._fm_deviation_default = fm_deviation
        self._fm_rate_default = fm_rate

        # Targets: the values last asked for. A key in self._stale is a target
        # the hardware is not known to hold; it is written at the next chance.
        self.amplitude_vpp = dv      # unknown until init() reads it or set() writes it
        self.fm_deviation = fm_deviation
        self.fm_rate = fm_rate
        self.fm_state = -1           # -1 unknown, 0 FM off, 1 FM on
        self._stale = {'fm_deviation', 'fm_rate'}

        self.core = core

    # ---- helpers -----------------------------------------------------------

    @property
    def _label(self) -> str:
        return f"[siglent {self._instr.ip} ch{self.ch} FM]"

    def _report_failure(self, what, e):
        print(f"{self._label} *** {what} FAILED -- hardware NOT updated "
              f"({type(e).__name__}: {e}) ***")

    def _check(self, amplitude_vpp, fm_deviation, fm_rate) -> dict:
        """The requested values (>= 0; a negative value means 'no change').
        Raises ValueError, before anything is sent, on a value out of limits."""
        req = {}
        for key, v, hi, unit in (('amplitude_vpp', amplitude_vpp, self.max_amplitude_vpp, 'Vpp'),
                                 ('fm_deviation', fm_deviation, self.max_fm_deviation, 'Hz'),
                                 ('fm_rate', fm_rate, self.max_fm_rate, 'Hz')):
            if v != v:
                raise ValueError(f"{self._label} {key} is NaN; nothing sent.")
            if v < 0.:
                continue
            if v > hi:
                raise ValueError(f"{self._label} {key} {v} {unit} requested, above the "
                                 f"limit {hi} {unit}; nothing sent.")
            req[key] = float(v)
        return req

    def _command(self, key) -> str:
        c = self.ch
        if key == 'frequency_carrier':
            return f"C{c}:BSWV FRQ,{self.frequency_carrier}"
        if key == 'amplitude_vpp':
            return f"C{c}:BSWV AMP,{self.amplitude_vpp}"
        if key == 'fm_deviation':
            return f"C{c}:MDWV FM,DEVI,{self.fm_deviation:.6f}"
        if key == 'fm_rate':
            return f"C{c}:MDWV FM,FRQ,{self.fm_rate:.6f}"
        raise KeyError(key)

    @staticmethod
    def _matches(applied, target) -> bool:
        return (isinstance(applied, float)
                and abs(applied - target) <= 1e-6 * max(1., abs(target)))

    def _problems(self, st) -> list:
        """Reasons the read state is outside the channel's limits ([] = OK)."""
        probs = []
        f, a = st['frequency_carrier'], st['amplitude_vpp']
        if not self._matches(f, self.frequency_carrier):
            probs.append(f"carrier frequency reads {f!r} Hz, not {self.frequency_carrier:.0f} Hz")
        if not isinstance(a, float) or a > self.max_amplitude_vpp:
            probs.append(f"amplitude reads {a!r} Vpp, above the {self.max_amplitude_vpp} Vpp "
                         f"limit (or unreadable)")
        return probs

    def _sync_fm_state(self, st):
        self.fm_state = 1 if (st['mod_state'] == 'ON' and st['mod_type'] == 'FM') else 0

    @staticmethod
    def describe(st) -> str:
        def f(x, scale, unit):
            return f"{x / scale:.6g} {unit}" if isinstance(x, float) else repr(x)
        return (f"output {st['output']} (load {st['load']}), "
                f"carrier {f(st['frequency_carrier'], 1e6, 'MHz')}, "
                f"{f(st['amplitude_vpp'], 1., 'Vpp')}, "
                f"modulation {st['mod_state']} ({st['mod_type']}), "
                f"deviation {f(st['fm_deviation'], 1e6, 'MHz')}, "
                f"rate {f(st['fm_rate'], 1e3, 'kHz')}")

    # ---- host side ---------------------------------------------------------

    def read_state(self) -> dict:
        """Read-only snapshot of the channel. Raises on a communications failure.
        The instrument reports FM deviation/rate only while FM is on."""
        mdwv = parse_mdwv(self._instr.ask(f"C{self.ch}:MDWV?"))
        bswv = parse_bswv(self._instr.ask(f"C{self.ch}:BSWV?"))
        outp = parse_outp(self._instr.ask(f"C{self.ch}:OUTP?"))
        fm = mdwv.get('FM', {})
        return dict(output=outp.get('STATE'), load=outp.get('LOAD'),
                    mod_state=mdwv.get('STATE'), mod_type=mdwv.get('TYPE'),
                    fm_deviation=fm.get('DEVI'), fm_rate=fm.get('FRQ'),
                    fm_source=fm.get('SRC'), fm_shape=fm.get('MDSP'),
                    frequency_carrier=bswv.get('FRQ'), amplitude_vpp=bswv.get('AMP'),
                    raw_mdwv=mdwv['raw'])

    def _verify(self, keys):
        """Read back after writing ``keys``; clear those that match."""
        time.sleep(T_FM_SETTLE)
        try:
            st = self.read_state()
        except Exception as e:
            print(f"{self._label} *** read-back after writing {', '.join(keys)} FAILED -- "
                  f"the write was sent but is NOT verified ({type(e).__name__}: {e}) ***")
            return None
        self._sync_fm_state(st)
        for k in keys:
            target = getattr(self, k)
            if self._matches(st[k], target):
                self._stale.discard(k)
            else:
                print(f"{self._label} *** {k}: requested {target}, instrument reads "
                      f"{st[k]!r} -- hardware does NOT hold the request ***")
        for p in self._problems(st):
            print(f"{self._label} *** {p} ***")
        return st

    def _flush(self):
        """Write every stale target the instrument will accept now."""
        pending = [k for k in _FM_KEYS if k in self._stale]
        fm_pending = [k for k in pending if k.startswith('fm_')]
        if fm_pending and self.fm_state == -1:
            try:
                self._sync_fm_state(self.read_state())
            except Exception as e:
                self._report_failure(f"set {', '.join(fm_pending)} (could not read whether FM is on)", e)
        if self.fm_state != 1:
            # The instrument ignores FM writes while FM is off; fm_on() writes them.
            pending = [k for k in pending if not k.startswith('fm_')]
        wrote = []
        for k in pending:
            try:
                self._instr.write(self._command(k))
            except Exception as e:
                self._report_failure(f"set {k} = {getattr(self, k)}", e)
            else:
                wrote.append(k)
        if wrote:
            self._verify(wrote)

    def set_rpc(self, amplitude_vpp=dv, fm_deviation=dv, fm_rate=dv):
        req = self._check(amplitude_vpp, fm_deviation, fm_rate)   # raises; nothing sent
        for k, v in req.items():
            if v != getattr(self, k) or k in self._stale:
                setattr(self, k, v)
                self._stale.add(k)
        self._flush()

    def fm_on_rpc(self):
        if self.fm_state == 1:
            self._flush()            # already on: only changed values are written
            return
        c = self.ch
        cmds = (f"C{c}:MDWV STATE,ON", f"C{c}:MDWV FM", f"C{c}:MDWV FM,SRC,INT",
                f"C{c}:MDWV FM,MDSP,SINE",
                self._command('fm_deviation'), self._command('fm_rate'))
        self._stale.update(('fm_deviation', 'fm_rate'))
        for cmd in cmds:
            try:
                self._instr.write(cmd)
            except Exception as e:
                self.fm_state = -1
                self._report_failure(f"switch FM on ({cmd})", e)
                return
        st = self._verify(['fm_deviation', 'fm_rate'])
        if st is None:
            self.fm_state = -1
            return
        if self.fm_state == 1 and not ({'fm_deviation', 'fm_rate'} & self._stale):
            return
        print(f"{self._label} *** FM did not come up as requested; switching modulation "
              f"OFF again. Read: {st['raw_mdwv']} ***")
        try:
            self._instr.write(f"C{c}:MDWV STATE,OFF")
        except Exception as e:
            self._report_failure("switch modulation off", e)
        self.fm_state = -1

    def fm_off_rpc(self):
        if self.fm_state == 0:
            return
        try:
            self._instr.write(f"C{self.ch}:MDWV STATE,OFF")
        except Exception as e:
            self.fm_state = -1
            self._report_failure("switch FM off", e)
            return
        self._stale.update(('fm_deviation', 'fm_rate'))   # rewritten at the next fm_on()
        st = self._verify([])
        if st is not None and st['mod_state'] != 'OFF':
            print(f"{self._label} *** modulation still reads {st['mod_state']} after OFF ***")
            self.fm_state = -1

    def init_rpc(self, fm_on=True):
        """Read the channel; put the carrier at ``frequency_carrier``; bring an
        amplitude above the limit down to the limit (otherwise keep it); FM
        deviation and rate back to their defaults, FM on (or off). The channel
        output is never switched."""
        try:
            st = self.read_state()
        except Exception as e:
            self.fm_state = -1
            print(f"{self._label} *** init could not read the instrument -- nothing written "
                  f"({type(e).__name__}: {e}) ***")
            return
        self._sync_fm_state(st)
        print(f"{self._label} on init: {self.describe(st)}")
        if st['output'] != 'ON':
            print(f"{self._label} output is {st['output']}; init does not switch it.")
        if st['mod_state'] == 'ON' and st['mod_type'] != 'FM':
            print(f"{self._label} modulation is on as {st['mod_type']}, not FM.")

        if not self._matches(st['frequency_carrier'], self.frequency_carrier):
            print(f"{self._label} carrier reads {st['frequency_carrier']!r} Hz; "
                  f"setting {self.frequency_carrier:.0f} Hz.")
            self._stale.add('frequency_carrier')
        a = st['amplitude_vpp']
        if isinstance(a, float) and 0. <= a <= self.max_amplitude_vpp:
            self.amplitude_vpp = a
            self._stale.discard('amplitude_vpp')
        else:
            print(f"{self._label} *** amplitude reads {a!r} Vpp, outside 0-"
                  f"{self.max_amplitude_vpp} Vpp; setting {self.max_amplitude_vpp} Vpp ***")
            self.amplitude_vpp = self.max_amplitude_vpp
            self._stale.add('amplitude_vpp')

        self.fm_deviation = self._fm_deviation_default
        self.fm_rate = self._fm_rate_default
        self._stale.update(('fm_deviation', 'fm_rate'))
        self._flush()                 # carrier, amplitude (and FM values if FM is on)
        if fm_on:
            self.fm_on_rpc()
        else:
            self.fm_off_rpc()

    # ---- kernel side -------------------------------------------------------

    @kernel
    def init(self, fm_on=True):
        self.core.wait_until_mu(now_mu())
        self.init_rpc(fm_on)
        self.core.break_realtime()
        delay(T_RPC_DELAY)

    @kernel
    def set(self, amplitude_vpp=dv, fm_deviation=dv, fm_rate=dv):
        """Amplitude (Vpp), FM deviation (Hz), FM rate (Hz); a negative value
        (the default) leaves that setting unchanged."""
        self.core.wait_until_mu(now_mu())
        self.set_rpc(amplitude_vpp, fm_deviation, fm_rate)
        self.core.break_realtime()
        delay(T_RPC_DELAY)

    @kernel
    def fm_on(self):
        self.core.wait_until_mu(now_mu())
        self.fm_on_rpc()
        self.core.break_realtime()
        delay(T_RPC_DELAY)

    @kernel
    def fm_off(self):
        self.core.wait_until_mu(now_mu())
        self.fm_off_rpc()
        self.core.break_realtime()
        delay(T_RPC_DELAY)
