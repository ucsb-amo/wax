"""One TTL pulse from ARTIQ on request, outside any experiment (TTLPulser).

A host program (the SLM spot finder) that wants an externally triggered camera
frame needs one rising edge on the camera's trigger line per frame, from the
same hardware and channel a run uses -- without being a run.  TTLPulser gives
exactly that: it builds the ``core`` and one ``TTLOut`` from the device
database (``artiq.master.worker_db.DeviceManager``, as ``artiq_run`` does; no
EnvExperiment), compiles ONE kernel once (``Core.precompile``)::

    core.reset()                    # what every run's init_kernel does first
    delay(t_lead)
    ttl.pulse(t_pulse)
    core.wait_until_mu(now_mu())    # return only once the edge is out

and ``pulse()`` runs it, as often as asked, with no compile in between.

The core device runs one kernel at a time: a connection to it ends whatever
kernel is running there -- normally the monitor (``waxx.base.monitor``), which
holds the hardware between runs.  So a TTLPulser behaves like a run towards
the monitor server:

* ``acquire()`` announces itself (``run_pending`` with a token, under
  ``label``: composite ops are refused with that name as the reason, the
  tweezer AWG is released), reads the device state and its trust flag, then
  connects and precompiles.  The monitor dies as it does for a run; the
  server marks the device state UNTRUSTED ("<label> took the core ...").
* ``release()`` closes the connection and, when the machine is otherwise
  free, (a) carries the trust flag through -- the state the server held, with
  this TTL at 0, sent back as this pulser's end state -- provided the state
  was trusted when the core was taken and its version did not move since:
  nothing but this TTL (pulsed, left low) was touched meanwhile, and nobody
  else wrote the hardware (the monitor was down, the core was ours); then
  (b) sends ``run complete`` so the server restarts the monitor.  "Otherwise
  free" means: no other run has announced itself to the monitor server, the
  caller's ``run_check`` (liveOD's run state) says no run, and the core still
  answers us (nobody took it).  A monitor started while a run is compiling
  would kill that run's kernel when it connects, so whenever this is in doubt
  NOTHING is sent and the note says the monitor was left down.
* a pulse whose connection was reset (another program took the core) raises
  :class:`CoreTaken`; ``release()`` then sends nothing to the monitor server.

Everything that talks to the network is best-effort and reported in the
returned notes; only the core connection and the compile raise.  The class is
machine-agnostic: the TTL device name, pulse length and the ``run_check`` come
from the caller (kexp: the spot finder's ``artiq_trigger``).
"""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from typing import Callable, Optional

from artiq.language.core import delay, kernel, now_mu

#: Timeline slack after core.reset() before the edge (the reset itself puts
#: the cursor 125 us ahead of the counter).
T_LEAD_DEFAULT = 1.0e-3
#: Default pulse length when the caller gives none.
T_PULSE_DEFAULT = 200.0e-9
#: The socket to the core device: a probe or a kernel that does not answer in
#: this long counts as the core being gone (the hold's kernel is ~ms).
CORE_SOCKET_TIMEOUT_S = 10.0
#: How long the monitor server may take to answer the announcement (it
#: releases the AWG first, up to connections.RELEASE_TIMEOUT_S).
ANNOUNCE_TIMEOUT_S = 8.0
#: Discovery of the monitor server (its beacon is every 0.5 s).
MONITOR_DISCOVERY_S = 3.0


class PulserError(RuntimeError):
    pass


class PulserNotHeld(PulserError):
    """pulse() without acquire()."""


class CoreTaken(PulserError):
    """The connection to the core device was reset: another program (a run,
    the monitor restarting) took the core."""


class _PulseKernel:
    """The object the kernel is compiled against: exactly the two devices it
    touches, so the compiler embeds nothing else."""

    def __init__(self, core, ttl):
        self.core = core
        self.ttl = ttl

    @kernel
    def pulse(self, t_pulse, t_lead):
        self.core.reset()
        delay(t_lead)
        self.ttl.pulse(t_pulse)
        self.core.wait_until_mu(now_mu())


def _default_core_factory(device_db_path: str, ttl_name: str):
    """(device manager, core, ttl device) from the device database, as
    artiq_run builds them.  Nothing connects here."""
    from artiq.master.databases import DeviceDB
    from artiq.master.worker_db import DeviceManager
    dmgr = DeviceManager(DeviceDB(device_db_path))
    core = dmgr.get("core")
    ttl = dmgr.get(ttl_name)
    return dmgr, core, ttl


def _default_monitor_factory():
    from waxx.util.comms_server.comm_client import MonitorClient
    return MonitorClient(discovery_timeout=MONITOR_DISCOVERY_S)


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return ""


class TTLPulser:
    """One precompiled ``core.reset(); ttl.pulse(t)`` kernel, fired on request.

    Args:
        ttl_name: the TTLOut's device-db name (``"ttl7"``).
        device_db_path: the device database file (default: env var ``db``).
        t_pulse / t_lead: pulse length and the slack before it (seconds).
        label: how this pulser names itself to the monitor server (the fence's
            reason, the untrusted-state reason, the end state's run name).
        ttl_state_name: the TTL's name in the device-state file (the
            ttl_frame attribute, e.g. ``"andor"``); its ``ttl_state`` is set
            to 0 in the carried-through end state.  None: the state is sent
            back unchanged (the pulse leaves the line low either way).
        run_check: ``() -> (free: bool, why: str)``, asked at release before
            the monitor is restarted (the caller's view of runs, e.g. liveOD's
            POLL).  None: only the monitor server's own run fence is checked.
        carry_trust: send the device state back as this pulser's end state
            when that is honest (module docstring).  False: never.
        core_factory / monitor_factory: test hooks.
    """

    def __init__(self, ttl_name: str, *, device_db_path: Optional[str] = None,
                 t_pulse: float = T_PULSE_DEFAULT, t_lead: float = T_LEAD_DEFAULT,
                 label: str = "TTL pulser", ttl_state_name: Optional[str] = None,
                 run_check: Optional[Callable[[], tuple]] = None, carry_trust: bool = True,
                 core_factory: Optional[Callable] = None,
                 monitor_factory: Optional[Callable] = None,
                 socket_timeout_s: float = CORE_SOCKET_TIMEOUT_S):
        self.ttl_name = str(ttl_name)
        self.device_db_path = device_db_path or os.getenv("db") or ""
        self.t_pulse = float(t_pulse)
        self.t_lead = float(t_lead)
        self.label = str(label)
        self.ttl_state_name = ttl_state_name
        self.run_check = run_check
        self.carry_trust = bool(carry_trust)
        self._default_core = core_factory is None
        self._core_factory = core_factory or (
            lambda: _default_core_factory(self.device_db_path, self.ttl_name))
        self._monitor_factory = monitor_factory or _default_monitor_factory
        self._socket_timeout_s = float(socket_timeout_s)

        self._lock = threading.RLock()
        self._dmgr = None
        self._core = None
        self._ttl = None
        self._run_precompiled = None
        self._monitor = None
        self._monitor_error = ""
        self._token = ""
        self._announced = False
        self._state_before = None       # (version, trust dict, config) at acquire
        self._dropped = ""              # why the core was lost during the hold
        self.held = False
        self.n_pulses = 0
        self.n_holds = 0
        self.last_notes: list = []
        self.t_acquired: Optional[float] = None

    # ------------------------------------------------------------------ #
    # the hold
    # ------------------------------------------------------------------ #

    def acquire(self) -> list:
        """Announce, connect (this ends the monitor's kernel), precompile.
        Returns the notes (strings) of what happened on the monitor side.
        Raises on a core device that cannot be reached or a kernel that does
        not compile; nothing is then held."""
        with self._lock:
            if self.held:
                return list(self.last_notes)
            notes: list = []
            self._dropped = ""
            self._announce(notes)
            try:
                self._connect_and_compile()
            except BaseException:
                self._withdraw(notes)
                self._close_core()
                self.last_notes = notes
                raise
            self.held = True
            self.n_holds += 1
            self.t_acquired = time.monotonic()
            notes.append(f"{self.label}: ARTIQ core taken; {self.ttl_name} pulse kernel "
                         f"({self.t_pulse * 1e9:.0f} ns) precompiled")
            self.last_notes = notes
            return list(notes)

    def pulse(self) -> float:
        """One edge.  Returns ``time.monotonic()`` right after the kernel
        returned, i.e. after the pulse was emitted.  Raises :class:`CoreTaken`
        when the connection was reset (another program has the core),
        :class:`PulserNotHeld` without a hold."""
        with self._lock:
            if not self.held:
                raise PulserNotHeld(f"{self.label}: pulse() before acquire()")
            if self._dropped:
                raise CoreTaken(f"{self.label}: {self._dropped}")
            try:
                self._run_precompiled()
            except OSError as e:
                # ConnectionResetError / timeout: the core device dropped us
                self._dropped = (f"the connection to the core device was lost at pulse "
                                 f"{self.n_pulses + 1} ({type(e).__name__}: {e}); another "
                                 f"program has the core")
                raise CoreTaken(f"{self.label}: {self._dropped}") from e
            self.n_pulses += 1
            return time.monotonic()

    def release(self) -> list:
        """Close the core connection; when the machine is free, carry the
        trust flag through and restart the monitor (module docstring).
        Returns the notes.  Never raises; a no-op without a hold."""
        with self._lock:
            if not self.held:
                return []
            notes: list = []
            self.held = False
            try:
                self._probe_core(notes)
                self._close_core()
                self._settle_monitor(notes)
            except Exception as e:                 # bookkeeping must not fail the caller
                notes.append(f"{self.label}: release bookkeeping failed ({type(e).__name__}: "
                             f"{e})")
            finally:
                self._state_before = None
                self._token = ""
                self._announced = False
                self.t_acquired = None
            self.last_notes = notes
            return list(notes)

    @property
    def dropped(self) -> str:
        return self._dropped

    # ------------------------------------------------------------------ #
    # core device
    # ------------------------------------------------------------------ #

    def _connect_and_compile(self):
        if self._core is None:
            if self._default_core and not self.device_db_path:
                raise PulserError(f"{self.label}: no device database (env var 'db' is not "
                                  f"set and no device_db_path was given)")
            self._dmgr, self._core, self._ttl = self._core_factory()
        comm = self._core.comm
        comm.open()
        sock = getattr(comm, "socket", None)
        if sock is not None:
            try:
                sock.settimeout(self._socket_timeout_s)
            except Exception:
                pass
        # The first request: it ends the kernel running on the device (the
        # monitor), and it is where an unreachable core shows up.
        comm.check_system_info()
        if self._run_precompiled is None:
            k = _PulseKernel(self._core, self._ttl)
            self._run_precompiled = self._core.precompile(k.pulse, self.t_pulse, self.t_lead)

    def _probe_core(self, notes: list):
        """Does the core still answer us?  If not, someone took it."""
        if self._dropped:
            notes.append(f"{self.label}: {self._dropped}")
            return
        try:
            self._core.comm.check_system_info()
        except Exception as e:
            self._dropped = (f"the core device did not answer at release ({type(e).__name__}: "
                             f"{e}); another program has probably taken it")
            notes.append(f"{self.label}: {self._dropped}")

    def _close_core(self):
        core, self._core = self._core, None
        self._run_precompiled = None       # the precompiled callable belongs to that session
        self._ttl = None
        dmgr, self._dmgr = self._dmgr, None
        try:
            if dmgr is not None:
                dmgr.close_devices()       # closes the core's socket too
            elif core is not None:
                core.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # monitor server
    # ------------------------------------------------------------------ #

    def _monitor_client(self):
        if self._monitor is None and not self._monitor_error:
            try:
                self._monitor = self._monitor_factory()
            except Exception as e:
                self._monitor_error = f"{type(e).__name__}: {e}"
        return self._monitor

    def _announce(self, notes: list):
        """Fence composite ops under our label; read the device state and
        its trust before the core is taken."""
        self._state_before = None
        self._announced = False
        m = self._monitor_client()
        if m is None:
            notes.append(f"{self.label}: no monitor server found ({self._monitor_error}); "
                         f"the monitor is not told the core is being taken")
            return
        self._token = uuid.uuid4().hex
        try:
            reply = m.announce_run(run_id=None, expt=self.label, client=_hostname(),
                                   token=self._token, timeout=ANNOUNCE_TIMEOUT_S)
        except Exception as e:
            reply = None
            notes.append(f"{self.label}: announcing to the monitor server failed "
                         f"({type(e).__name__}: {e})")
        if reply is None or reply.get("status") != "ok":
            if reply is not None:
                notes.append(f"{self.label}: the monitor server refused the announcement: "
                             f"{reply.get('msg', reply)}")
            self._token = ""
            return
        self._announced = True
        # the state the monitor was holding the hardware at, and whether it
        # was trusted -- read now, right before the core is taken
        try:
            state = m.get_state()
        except Exception as e:
            state = None
            notes.append(f"{self.label}: could not read the device state "
                         f"({type(e).__name__}: {e}); the trust flag will not be carried")
        if isinstance(state, dict) and state.get("status") == "ok":
            trust = state.get("trust") or {}
            self._state_before = (state.get("version"), dict(trust), state.get("config"))
            if trust.get("trusted") is False:
                notes.append(f"{self.label}: the device state was already UNTRUSTED "
                             f"({trust.get('reason', '')}); it stays so")
        notes.append(f"{self.label}: announced to the monitor server (composite ops fenced)")

    def _withdraw(self, notes: list):
        """acquire() failed after the announcement: lift our fence."""
        if not self._announced or self._monitor is None:
            return
        try:
            self._monitor.withdraw_run(self._token, None)
            notes.append(f"{self.label}: announcement withdrawn (the core was not taken)")
        except Exception:
            pass
        self._announced = False
        self._token = ""

    def _machine_free(self, notes: list):
        """(free, why): may the monitor be restarted now?  Fails closed."""
        if self._dropped:
            return False, "another program has the core"
        m = self._monitor
        if m is not None:
            try:
                status = m.get_status()
            except Exception as e:
                status = None
                notes.append(f"{self.label}: monitor server status unavailable "
                             f"({type(e).__name__}: {e})")
            if status is None:
                return False, "the monitor server did not answer"
            pending = status.get("run_pending")
            if isinstance(pending, dict) and pending.get("token") != self._token:
                who = pending.get("run_id") or pending.get("expt") or "a run"
                return False, f"run {who} has announced itself to the monitor server"
        if self.run_check is not None:
            try:
                free, why = self.run_check()
            except Exception as e:
                return False, f"the run check failed ({type(e).__name__}: {e})"
            if not free:
                return False, str(why or "a run is in progress")
        return True, ""

    def _settle_monitor(self, notes: list):
        m = self._monitor
        if m is None:
            notes.append(f"{self.label}: core released; no monitor server to tell")
            return
        free, why = self._machine_free(notes)
        if not free:
            notes.append(f"{self.label}: core released, but the monitor was NOT restarted: "
                         f"{why}. That run's end restarts it (or the Device Control GUI).")
            return
        self._carry_trust(m, notes)
        try:
            m.send_end()
            notes.append(f"{self.label}: core released; the monitor server restarts the "
                         f"monitor")
        except Exception as e:
            notes.append(f"{self.label}: core released, but 'run complete' did not reach the "
                         f"monitor server ({type(e).__name__}: {e}); restart the monitor from "
                         f"the Device Control GUI")

    def _carry_trust(self, m, notes: list):
        """Send the device state back as our end state, when honest."""
        if not self.carry_trust:
            return
        before = self._state_before
        if before is None:
            notes.append(f"{self.label}: device state UNTRUSTED until the next run's end "
                         f"state (no state was read before the core was taken)")
            return
        version, trust, config = before
        if trust.get("trusted") is not True:
            notes.append(f"{self.label}: device state stays UNTRUSTED (it was before)")
            return
        if not (isinstance(config, dict)
                and all(isinstance(config.get(k), dict) for k in ("dds", "ttl", "dac"))):
            notes.append(f"{self.label}: device state UNTRUSTED until the next run's end "
                         f"state (the state read before had no dds/ttl/dac sections)")
            return
        try:
            now = m.get_state()
        except Exception as e:
            now = None
        if not isinstance(now, dict) or now.get("status") != "ok":
            notes.append(f"{self.label}: device state UNTRUSTED until the next run's end "
                         f"state (the monitor server did not answer)")
            return
        if now.get("version") != version:
            notes.append(f"{self.label}: device state UNTRUSTED until the next run's end "
                         f"state: the state file changed during the hold (version {version} "
                         f"-> {now.get('version')})")
            return
        sections = {k: {name: dict(entry) for name, entry in config[k].items()}
                    for k in ("dds", "ttl", "dac")}
        if self.ttl_state_name and self.ttl_state_name in sections["ttl"]:
            sections["ttl"][self.ttl_state_name]["ttl_state"] = 0
        try:
            reply = m.replace_state(sections, run_id=None, expt=self.label)
        except Exception as e:
            reply = None
            notes.append(f"{self.label}: end state not sent ({type(e).__name__}: {e})")
        if reply is None or reply.get("status") != "ok":
            notes.append(f"{self.label}: device state UNTRUSTED until the next run's end "
                         f"state (the monitor server did not accept the end state"
                         + (f": {reply.get('msg')}" if isinstance(reply, dict) else "") + ")")
            return
        notes.append(f"{self.label}: device state trusted again -- it was trusted when the "
                     f"core was taken, only {self.ttl_name} was pulsed (left low), and "
                     f"nothing else wrote the hardware meanwhile")

    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        return (f"{self.label}: {self.ttl_name} from {self.device_db_path or '$db'}, "
                f"pulse {self.t_pulse * 1e9:.0f} ns, lead {self.t_lead * 1e3:.1f} ms")

    def close(self):
        """Release if held; forget the monitor client."""
        self.release()
        m, self._monitor = self._monitor, None
        if m is not None:
            try:
                m.close()
            except Exception:
                pass


__all__ = ["TTLPulser", "PulserError", "PulserNotHeld", "CoreTaken", "T_LEAD_DEFAULT",
           "T_PULSE_DEFAULT", "CORE_SOCKET_TIMEOUT_S"]
