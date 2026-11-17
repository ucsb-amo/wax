"""RF consultant (TPI-1005-A signal generator) under ARTIQ control.

One :class:`RFConsultant` per assigned consultant, made by
``waxx.config.rf_consultant_id.rf_consultant_frame`` from the beacon identity
frame.  In an experiment::

    rf = self.rf_consultant.rf_consultant_980
    rf.sweep_to(self.p.frequency_eo_980)   # Hz; walks there 0.5 MHz at a time
    rf.set_frequency(80.e6)                # Hz; jumps
    rf.set_amplitude(-10.)                 # dBm
    rf.on()

``sweep_to`` is the Siglent's (``ry_980.sweep_to`` / ``SDG6000X_CH.sweep``):
same arguments and 0.5 MHz steps, ``T_SWEEP_STEP`` (50 ms) apart -- the Siglent
waits 0.1 s -- starting from the frequency
the device reports, so an ``xvar`` scan of the end frequency moves the laser
in small steps between shots, never in one jump.

Every kernel method waits for the timeline to reach the wall clock
(``wait_until_mu(now_mu())``), makes the host RPC, then ``break_realtime``
and a short delay -- the device changes at the point in the sequence where
the call sits, not at a precise time.  Each call is one or more round trips
over the network to the TPI server (milliseconds), so keep them out of
timing-critical stretches.

Limits are checked before anything is sent.  An amplitude above
``max_amplitude_dbm`` or a frequency outside ``[min_frequency, max_frequency]``
raises ``ValueError`` (from a kernel this aborts the run): the request is
refused, never clipped.  The limits are the ones assigned in
``beacon.tpi.rf_consultant_id`` (``max_amplitude_dbm``, ``min_frequency``,
``max_frequency``), tightened by any :meth:`RFConsultant.limit` call.

The first call finds the device and holds it (``beacon.tpi.client.TpiHold``)
for the rest of the process, so its TPI server keeps the COM port open
through a run; the hold lapses ~15 s after the experiment process ends.

Link-failure policy (as for the Siglent, ``waxx.control.misc.sdg6000x``): a
write that fails on the link or at the server -- server down, port closed by a
person, device error -- does not raise.  It is printed loudly (the hardware
was NOT updated) and the cached value is marked stale, so the next request
for it is written even if unchanged.
"""

from __future__ import annotations

import time

from artiq.coredevice.core import Core
from artiq.language import delay, kernel, now_mu

# What a TPI itself accepts (Hz, dBm).
from beacon.tpi.rf_consultant_base import (F_MAX_TPI, F_MIN_TPI,  # noqa: F401
                                           AMPLITUDE_MAX_TPI_DBM, AMPLITUDE_MIN_TPI_DBM)

# "Leave unchanged" sentinels: a frequency is >= 0, but an amplitude in dBm can be
# negative, so it gets its own.
dv = -0.1
dv_dbm = -1000.

T_RPC_DELAY = 20.e-3
T_SWEEP_STEP = 0.05          # s between sweep steps
F_SWEEP_STEP = 5.e5         # Hz per sweep step, as ry_980.sweep_to


class RFConsultant:
    """One TPI-1005-A, reached through the TPI server that hosts it.

    Built from a ``beacon.tpi.rf_consultant_base.RfConsultantId``; the network
    is not touched until the first call (the device is then found by serial on
    whichever computer's TPI server lists it).
    """

    def __init__(self, ident, core=Core):
        self.core = core
        self.key = ident.key
        self.serial = ident.serial
        self.name = ident.display_name

        self.frequency_default = (dv if ident.frequency is None
                                  else ident.frequency)
        self.amplitude_dbm_default = (dv_dbm if ident.amplitude_dbm is None
                                  else float(ident.amplitude_dbm))
        self._frequency_default = self.frequency_default   # the Siglent channel's name

        self.min_frequency = F_MIN_TPI
        self.max_frequency = F_MAX_TPI
        self.max_amplitude_dbm = float(AMPLITUDE_MAX_TPI_DBM)
        self.limit(max_amplitude_dbm=ident.max_amplitude_dbm,
                   min_frequency=ident.min_frequency,
                   max_frequency=ident.max_frequency)

        # Last values written (unknown until then).  A setting in _stale is
        # not known to match the hardware and is always written next time.
        self.frequency = dv
        self.amplitude_dbm = dv_dbm
        self.state = -1
        self._stale = set()

        self._device = None    # beacon.tpi.client.TpiDevice, found on first use
        self._hold = None

    # ------------------------------------------------------------------ #
    # Configuration (host)
    # ------------------------------------------------------------------ #

    def limit(self, max_amplitude_dbm=None, min_frequency=None, max_frequency=None):
        """Tighten the limits (Hz, dBm).  A looser value than the current one
        is ignored: a limit set in beacon's rf_consultant_id always holds."""
        if max_amplitude_dbm is not None:
            self.max_amplitude_dbm = min(self.max_amplitude_dbm, float(max_amplitude_dbm))
        if min_frequency is not None:
            self.min_frequency = max(self.min_frequency, float(min_frequency))
        if max_frequency is not None:
            self.max_frequency = min(self.max_frequency, float(max_frequency))
        if self.amplitude_dbm_default != dv_dbm and self.amplitude_dbm_default > self.max_amplitude_dbm:
            raise ValueError(f"{self._label} default amplitude {self.amplitude_dbm_default} dBm is above "
                             f"its limit {self.max_amplitude_dbm} dBm")
        return self

    @property
    def _label(self) -> str:
        return f"[rf_consultant {self.key or self.serial}]"

    def __repr__(self) -> str:
        return (f"<RFConsultant {self.key} S/N {self.serial}  "
                f"f {self.min_frequency/1e6:g}-{self.max_frequency/1e6:g} MHz  "
                f"amplitude <= {self.max_amplitude_dbm:g} dBm>")

    # ------------------------------------------------------------------ #
    # Host side (RPC targets)
    # ------------------------------------------------------------------ #

    def _dev(self):
        if self._device is None:
            from beacon.tpi.client import TpiHold, find_device  # noqa: PLC0415 (network, lazily)
            self._device = find_device(self.serial)
            # Hold it for as long as this process runs, so its server does not
            # idle-close the port in a long stretch of a run without calls.
            # (Daemon thread: the hold lapses ~15 s after the process ends.)
            self._hold = TpiHold(self._device).start()
        return self._device

    def _write(self, what: str, setting: str, call) -> bool:
        """Send one write; on any failure print it and mark ``setting`` stale."""
        try:
            reply = call(self._dev())
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error", "unknown error"))
        except Exception as e:
            self._stale.add(setting)
            print(f"{self._label} *** {what} FAILED -- hardware NOT updated "
                  f"({type(e).__name__}: {e}) ***")
            return False
        self._stale.discard(setting)
        return True

    def _check_frequency(self, frequency):
        if not self.min_frequency <= frequency <= self.max_frequency:
            raise ValueError(f"{self._label} frequency {frequency/1e6:.4f} MHz requested, outside "
                             f"{self.min_frequency/1e6:g}-{self.max_frequency/1e6:g} MHz; "
                             f"nothing sent.")

    def set_frequency_rpc(self, frequency):
        if frequency < 0. or (frequency == self.frequency and 'frequency' not in self._stale):
            return
        self._check_frequency(frequency)
        self.frequency = frequency
        self._write(f"set frequency={frequency/1e6:.4f} MHz", 'frequency',
                    lambda d: d.set_frequency(frequency))

    def set_amplitude_rpc(self, amplitude_dbm):
        if amplitude_dbm == dv_dbm or (amplitude_dbm == self.amplitude_dbm and 'amplitude' not in self._stale):
            return
        if amplitude_dbm > self.max_amplitude_dbm:
            raise ValueError(f"{self._label} amplitude {amplitude_dbm} dBm requested, above the "
                             f"configured maximum {self.max_amplitude_dbm} dBm; nothing sent.")
        if amplitude_dbm < AMPLITUDE_MIN_TPI_DBM or amplitude_dbm != round(amplitude_dbm):
            raise ValueError(f"{self._label} amplitude {amplitude_dbm} dBm: the TPI takes whole dBm "
                             f"from {AMPLITUDE_MIN_TPI_DBM}; nothing sent.")
        self.amplitude_dbm = amplitude_dbm
        self._write(f"set amplitude={amplitude_dbm:g} dBm", 'amplitude',
                    lambda d: d.set_amplitude(int(amplitude_dbm)))

    def set_output_rpc(self, state):
        if state == self.state and 'state' not in self._stale:
            return
        self.state = state
        self._write(f"output {'ON' if state else 'OFF'}", 'state',
                    lambda d: d.set_rf(bool(state)))

    def sweep_to_rpc(self, frequency_end=dv, frequency_step=F_SWEEP_STEP,
                     reset=False, t_step=T_SWEEP_STEP):
        """Walk from the frequency the device reports now to ``frequency_end``
        (the assigned default with ``reset=True`` or no end) in
        ``frequency_step`` steps, ``t_step`` s apart."""
        if frequency_end < 0. or reset:
            frequency_end = self.frequency_default
        if frequency_end < 0.:
            raise ValueError(f"{self._label} sweep with no end frequency and no "
                             f"default frequency assigned; nothing sent.")
        self._check_frequency(frequency_end)
        # Start from where the hardware actually is; if that cannot be read
        # there is nothing safe to do but report and leave it alone.
        try:
            f = self._dev().frequency
        except Exception as e:
            print(f"{self._label} *** sweep to {frequency_end/1e6:.4f} MHz FAILED -- could not "
                  f"read the current frequency, hardware NOT updated "
                  f"({type(e).__name__}: {e}) ***")
            return
        if not self.min_frequency <= f <= self.max_frequency:
            # Every step would be refused; say why instead of failing on step 1.
            raise ValueError(f"{self._label} the device is at {f/1e6:.4f} MHz, outside its "
                             f"limits {self.min_frequency/1e6:g}-{self.max_frequency/1e6:g} MHz, "
                             f"so it cannot be swept from there; nothing sent. Set it inside "
                             f"the limits first (TPI GUI, or set_frequency).")
        self.frequency = f
        self._stale.discard('frequency')
        step = abs(frequency_step) if frequency_end >= f else -abs(frequency_step)
        while step != 0. and abs(frequency_end - f) > abs(step):
            f += step
            self.set_frequency_rpc(f)
            if 'frequency' in self._stale:
                return      # a step failed (reported): stop following a dead link
            time.sleep(t_step)
        self.set_frequency_rpc(frequency_end)

    def init_rpc(self):
        """As the Siglent's init: walk to the assigned default frequency (never
        a jump), set the default amplitude, output on.  A default that is not
        assigned is left alone."""
        if self.frequency_default >= 0.:
            self.sweep_to_rpc(reset=True)
        self.set_amplitude_rpc(self.amplitude_dbm_default)
        self.set_output_rpc(1)

    # ---- the Siglent channel's interface (waxx.control.misc.sdg6000x) ---- #
    # What kexp.control.rydberg_lasers calls on its frequency source, so a
    # Rydberg beam can be driven by either.  Same names, same meaning.

    def _stash_defaults(self):
        """Nothing to stash: the defaults come from rf_consultant_id."""

    def get_frequency(self) -> float:
        """Read the frequency back from the device (Hz).  Raises when the link
        is down, as the Siglent's does."""
        return self._dev().frequency

    def set_rpc(self, frequency=dv, amplitude=dv, init=False):
        if amplitude >= 0.:
            # A Siglent amplitude is in Vpp; this device takes dBm.  Passing
            # the number through would set the wrong power.
            raise ValueError(f"{self._label} set(amplitude={amplitude}) is a Siglent amplitude "
                             f"(Vpp); an RF consultant takes dBm -- use set_amplitude(dBm). "
                             f"Nothing sent.")
        if init:
            self.init_rpc()
        self.set_frequency_rpc(frequency)

    # ------------------------------------------------------------------ #
    # Kernel side
    # ------------------------------------------------------------------ #

    @kernel
    def set_frequency(self, frequency):
        """Set the output frequency (Hz)."""
        self.core.wait_until_mu(now_mu())
        self.set_frequency_rpc(frequency)
        self.core.break_realtime()
        delay(T_RPC_DELAY)

    @kernel
    def set_amplitude(self, amplitude_dbm):
        """Set the output amplitude (dBm, whole numbers; refused above the limit)."""
        self.core.wait_until_mu(now_mu())
        self.set_amplitude_rpc(amplitude_dbm)
        self.core.break_realtime()
        delay(T_RPC_DELAY)

    @kernel
    def sweep_to(self, frequency_end=dv, frequency_step=F_SWEEP_STEP, reset=False,
                 t_step=T_SWEEP_STEP):
        """Walk the frequency to ``frequency_end`` (Hz) in ``frequency_step``
        steps ``t_step`` s apart, from wherever the device is.  With no end, or
        ``reset=True``, go to the assigned default.  Blocks (wall clock) until
        the last step is written."""
        self.core.wait_until_mu(now_mu())
        self.sweep_to_rpc(frequency_end, frequency_step, reset, t_step)
        self.core.break_realtime()
        delay(T_RPC_DELAY)

    @kernel
    def sweep(self, frequency_end=dv, frequency_step=1.e6, reset=False):
        """The Siglent channel's ``sweep`` (as ``SDG6000X_CH.sweep``)."""
        self.sweep_to(frequency_end, frequency_step, reset)

    @kernel
    def set(self, frequency=dv, amplitude=dv, init=False):
        """The Siglent channel's ``set``: frequency (Hz) only; a Vpp amplitude
        is refused (use :meth:`set_amplitude` in dBm)."""
        self.core.wait_until_mu(now_mu())
        self.set_rpc(frequency, amplitude, init)
        self.core.break_realtime()
        delay(T_RPC_DELAY)

    @kernel
    def set_output(self, state=1):
        """RF output on (1) or off (0)."""
        self.core.wait_until_mu(now_mu())
        self.set_output_rpc(state)
        self.core.break_realtime()
        delay(T_RPC_DELAY)

    @kernel
    def on(self):
        self.set_output(1)

    @kernel
    def off(self):
        self.set_output(0)

    @kernel
    def init(self):
        """As the Siglent's init: walk to the default frequency, default
        amplitude, output on (defaults from rf_consultant_id)."""
        self.core.wait_until_mu(now_mu())
        self.init_rpc()
        self.core.break_realtime()
        delay(T_RPC_DELAY)
