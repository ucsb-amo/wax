"""Start liveOD from the Monitor panel: a button that runs the lab's liveOD
launcher (a ``.bat``) the way a person double-clicking it would, and only when
no liveOD is running.

* :func:`start_bat` -- ``cmd /c start "" /D <folder> <bat>`` with the bat's
  folder as the working directory: the bat gets a console window of its own
  (liveOD's console guard needs one), detached from the caller.  The outer
  ``cmd`` is asked to break away from a job object the caller may be in, so
  closing the dashboard never takes liveOD with it (when the job forbids
  that, it starts without the breakaway).
* :func:`beacon_probe` -- is a liveOD beaconing?  A cache lookup in the
  process's discovery registry (``beacon.discovery.client``; receive only,
  never blocks): "running" (a beacon heard within :data:`BEACON_FRESH_S`),
  "absent" (none, after the registry has listened :data:`LISTEN_FIRST_S`) or
  "unknown" (not listened long enough to tell).
* :class:`LiveODButton` -- "Start liveOD": enabled only when the probe says
  "absent", the host's ``refusal()`` says nothing, and no launch from this
  button is younger than :data:`LAUNCH_GUARD_S` without a beacon since (the
  double-click guard).  A press calls ``start(bat)`` once and hands the record
  (who, where, which file, pid or error) to ``record`` -- the host journals it.

The launcher's path comes from the lab (kexp passes ``_bat/live_od.bat``);
nothing here knows the K machine.  Tests inject ``start`` and ``probe``: the
real bat is never run by them.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import Any, Callable

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import QPushButton

from waxx.util.dashboard import theme

#: liveOD's beacon id (waxx.util.live_od.live_od_server).
LIVE_OD_SERVER_ID = "live_od"
#: A beacon heard within this many seconds means liveOD is running (it
#: beacons every 0.5 s).
BEACON_FRESH_S = 3.0
#: Before the registry has listened this long, no beacon proves nothing.
LISTEN_FIRST_S = 2.0
#: After a launch the button stays off this long, or until the beacon appears.
LAUNCH_GUARD_S = 30.0
#: How often the button looks again while shown.
PROBE_MS = 1000

_CREATE_NO_WINDOW = 0x08000000
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def start_bat(path: str, *, popen=subprocess.Popen) -> int:
    """Start the batch file ``path`` in a console of its own, as a double-click
    would; returns the pid of the outer ``cmd`` (which exits at once).
    Raises FileNotFoundError when there is no such file, OSError when it
    could not be started."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no launcher at {path}")
    folder = os.path.dirname(path)
    argv = ["cmd", "/c", "start", "", "/D", folder, path]
    kw: dict[str, Any] = dict(cwd=folder, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, close_fds=True)
    flags = _CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP
    if os.name != "nt":
        return popen(argv, **kw).pid
    try:
        return popen(argv, creationflags=flags | _CREATE_BREAKAWAY_FROM_JOB, **kw).pid
    except PermissionError:
        # in a job that does not allow breakaway: start it inside it
        return popen(argv, creationflags=flags, **kw).pid


def beacon_probe(server_id: str = LIVE_OD_SERVER_ID) -> tuple[str, str]:
    """``(state, text)``: "running" / "absent" / "unknown" (see the module
    docstring).  Imports the discovery registry on first use (it then
    listens for beacons in this process; receive only)."""
    try:
        from beacon.discovery.client import discover_entries, listening_for  # noqa: PLC0415
    except Exception as exc:                          # noqa: BLE001
        return "unknown", f"cannot listen for beacons ({exc})"
    entry = discover_entries(server_id, max_age=BEACON_FRESH_S).get(server_id)
    if entry is not None:
        return "running", (f"liveOD is running: its beacon came from {entry.ip} "
                           f"{entry.age:.1f} s ago")
    listened = listening_for()
    if listened is None or listened < LISTEN_FIRST_S:
        return "unknown", "listening for liveOD's beacon..."
    return "absent", f"no liveOD beacon heard in the last {BEACON_FRESH_S:.0f} s"


def _css(enabled: bool) -> str:
    bg = theme.HIGHLIGHT if enabled else theme.BG_BUTTON
    return (f"QPushButton {{ background: {bg}; color: white;"
            f" border: 1px solid {bg}; border-radius: 9px;"
            f" min-height: 18px; padding: 0px 10px; font-size: 11px; font-weight: 600; }}"
            f" QPushButton:hover:enabled {{ border: 1px solid white; }}"
            f" QPushButton:disabled {{ color: rgba(255,255,255,0.65); }}")


class LiveODButton(QPushButton):
    """"Start liveOD" (see the module docstring).

    ``bat``: the launcher's path; ``start(bat) -> pid`` (default
    :func:`start_bat`); ``probe() -> (state, text)`` (default
    :func:`beacon_probe`); ``refusal() -> str``: why this PC must not start
    liveOD ("" when it may); ``record(dict)``: the press, for the journal;
    ``say(text)``: a message for the person; ``by``: who presses."""

    #: emitted with the record after every press that called ``start``
    launched = pyqtSignal(dict)

    def __init__(self, bat: str, *, start: Callable[[str], Any] | None = None,
                 probe: Callable[[], tuple[str, str]] | None = None,
                 refusal: Callable[[], str] | None = None,
                 record: Callable[[dict], None] | None = None,
                 say: Callable[[str], None] | None = None, by: str = "",
                 clock: Callable[[], float] = time.monotonic, parent=None):
        super().__init__("Start liveOD", parent)
        self.bat = str(bat)
        self._start = start or start_bat
        self._probe = probe or beacon_probe
        self._refusal = refusal
        self._record = record
        self._say = say
        self.by = by
        self._clock = clock
        #: (monotonic time, wall time) of this button's last launch
        self._launched: tuple[float, float] | None = None
        self.state = "unknown"
        self.why = ""
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clicked.connect(lambda _=False: self.press())
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(PROBE_MS)
        self.refresh()

    # -- state ---------------------------------------------------------------------------

    def _tick(self) -> None:
        if self.isVisible():
            self.refresh()

    def assess(self) -> tuple[bool, str]:
        """``(may start, why / what it does)``."""
        refusal = ""
        if self._refusal is not None:
            try:
                refusal = str(self._refusal() or "")
            except Exception as exc:                  # noqa: BLE001
                refusal = f"cannot tell whether this PC may start liveOD ({exc})"
        if refusal:
            return False, refusal
        try:
            state, text = self._probe()
        except Exception as exc:                      # noqa: BLE001
            state, text = "unknown", f"cannot tell whether liveOD is running ({exc})"
        self.state = state
        if state == "running":
            self._launched = None                     # it is up: the guard is done
            return False, text
        if self._launched is not None:
            since = self._clock() - self._launched[0]
            at = time.strftime("%H:%M:%S", time.localtime(self._launched[1]))
            if since < LAUNCH_GUARD_S:
                return False, (f"started from here at {at}; waiting for its beacon "
                               f"({LAUNCH_GUARD_S - since:.0f} s left before this "
                               "button allows another start)")
            text += (f"\nThe start at {at} has not beaconed in {LAUNCH_GUARD_S:.0f} s: "
                     "look at its console window before starting it again.")
        if state != "absent":
            return False, text
        return True, text

    def refresh(self) -> None:
        ok, why = self.assess()
        self.why = why
        self.setEnabled(ok)
        self.setStyleSheet(_css(ok))
        head = (f"Start liveOD: runs {self.bat} in a console of its own, as a double-click "
                "would." if ok else "Start liveOD is off:")
        self.setToolTip(f"{head}\n{why}")

    # -- the press -----------------------------------------------------------------------

    def press(self) -> bool:
        """Start the launcher once, if allowed now; True when ``start`` was
        called and did not fail."""
        ok, why = self.assess()
        if not ok:
            self.refresh()
            if self._say is not None:
                self._say(f"liveOD not started: {why}")
            return False
        self._launched = (self._clock(), time.time())
        rec = {"bat": self.bat, "by": self.by, "owner": "person",
               "host": socket.gethostname(), "probe": why}
        try:
            pid = self._start(self.bat)
        except Exception as exc:                      # noqa: BLE001
            self._launched = None
            rec.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            text = f"liveOD launcher failed: {exc}"
        else:
            rec.update(ok=True, pid=pid if isinstance(pid, int) else None)
            text = (f"liveOD launcher started ({os.path.basename(self.bat)}); its window "
                    f"should appear, and the button stays off until it beacons "
                    f"(or {LAUNCH_GUARD_S:.0f} s)")
        self.refresh()
        if self._say is not None:
            self._say(text)
        if self._record is not None:
            try:
                self._record(dict(rec))
            except Exception:                         # noqa: BLE001
                pass
        self.launched.emit(dict(rec))
        return bool(rec.get("ok"))

    def shutdown(self) -> None:
        self._timer.stop()


__all__ = ["BEACON_FRESH_S", "LAUNCH_GUARD_S", "LIVE_OD_SERVER_ID", "LiveODButton",
           "beacon_probe", "start_bat"]
