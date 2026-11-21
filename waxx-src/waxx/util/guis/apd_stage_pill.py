"""The APD stage pill in the Device Control GUI's "Connected devices" group.

The APD pickoff stage is a beamsplitter on a PDXC piezo stage
(:mod:`waxx.control.misc.pdxc`): **in** sends the light to the APD and blocks
the Andor, **out** leaves the Andor clear.  The pill talks to the PDXC server
directly (found by its beacon), not through the monitor server:

* it shows the position the server last commanded (``in`` / ``out`` / ``?``)
  and whether the server and its COM port are up, polled every few seconds
  while the pill is visible;
* its menu (left or right click) has **Move in** and **Move out** -- a full
  throw to the end stop (``PDXC_Client.move_to``), sent at once with no
  confirmation dialog.  A move to where the server already has the stage is
  sent with ``force=True`` ("drive in again"), for when it was moved by hand.

Moves are allowed while a run is using the machine.  A run sets the stage
itself in ``Base.__init__``, so moving it under a run can block that run's
camera or APD mid-run: the log line names the run (or says the run state is
unknown when the monitor server is unreachable).

All network I/O runs on daemon threads; results come back to the GUI thread
through queued signals.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QMenu, QMessageBox, QPushButton

from waxx.util.dashboard import theme
from waxx.util.guis.qt_upkeep import delete_later, set_style_if_changed

#: How often the pill asks the PDXC server for its snapshot (while visible).
POLL_MS = 5000
#: Beacon discovery wait for the PDXC server, per attempt.
DISCOVERY_TIMEOUT_S = 1.5

POSITION_IN = "in"
POSITION_OUT = "out"

# colours as the connection bar's pills (waxx.util.live_od.gui.camera_menu)
GREEN, PURPLE, GREY, RED, ORANGE = "#43a047", "#ba68c8", "#9e9e9e", "#c62828", "#ef6c00"

_WHAT = {POSITION_IN: "light to the APD, Andor blocked",
         POSITION_OUT: "Andor clear, no light to the APD"}


def _css(color: str) -> str:
    return (f"QPushButton {{ background-color: {color}; color: white; font-weight: bold; "
            f"border: none; border-radius: 8px; padding: 1px 8px; }} "
            f"QPushButton:disabled {{ color: rgba(255, 255, 255, 140); }}")


def _default_client_factory():
    from waxx.control.misc.pdxc import PDXC_Client  # noqa: PLC0415
    return PDXC_Client(discovery_timeout=DISCOVERY_TIMEOUT_S)


class _Bridge(QObject):
    """Carries thread results to the GUI thread (queued: it lives there)."""

    polled = pyqtSignal(object, str)        # snapshot | None, error
    moved = pyqtSignal(str, str, str)       # state, result, error


class ApdStagePill(QPushButton):
    """See the module docstring.

    * ``run_text() -> str | None``: the run using the machine ("" none, None
      when the host cannot tell).  Default: always "".
    * ``record(line)``: where a move's request and outcome are logged.
    * ``client_factory()``: builds a ``PDXC_Client`` (tests pass a fake).
    """

    def __init__(self, run_text: Callable[[], str | None] | None = None,
                 record: Callable[[str], None] | None = None,
                 client_factory: Callable[[], object] | None = None,
                 poll_ms: int = POLL_MS, parent=None):
        super().__init__("APD stage", parent)
        self._run_text = run_text or (lambda: "")
        self._record = record or (lambda line: None)
        self._client_factory = client_factory or _default_client_factory
        self._client = None
        self._lock = threading.Lock()
        self._polling = False
        self._moving: str | None = None         # target of a move this pill sent
        self._closed = False
        self.snapshot: dict | None = None
        self.error = "not asked yet"
        self.polled_at: float | None = None

        self._bridge = _Bridge(self)
        self._bridge.polled.connect(self._on_polled)
        self._bridge.moved.connect(self._on_moved)

        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda pos: self._show_menu(pos))
        self.clicked.connect(lambda _=False: self._show_menu(self.rect().bottomLeft()))

        self._timer = QTimer(self)
        self._timer.setInterval(int(poll_ms))
        self._timer.timeout.connect(self._poll_if_visible)
        if poll_ms > 0:
            self._timer.start()
            QTimer.singleShot(0, self.poll)
        self.refresh()

    # -- network (daemon threads) ---------------------------------------------------

    def _get_client(self):
        with self._lock:
            client = self._client
        if client is None:
            client = self._client_factory()
            with self._lock:
                self._client = client
        return client

    def _drop_client(self) -> None:
        with self._lock:
            self._client = None

    def _emit(self, signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:            # the pill (and its bridge) is gone
            pass

    def _poll_if_visible(self) -> None:
        if self.isVisible():
            self.poll()

    def poll(self) -> None:
        """Ask the PDXC server for its snapshot (one request at a time)."""
        if self._polling or self._closed:
            return
        self._polling = True

        def work():
            try:
                snap = self._get_client().get_snapshot()
                self._emit(self._bridge.polled, snap if isinstance(snap, dict) else {}, "")
            except Exception as e:                       # noqa: BLE001
                self._drop_client()
                self._emit(self._bridge.polled, None, str(e) or type(e).__name__)

        threading.Thread(target=work, name="apd-stage-poll", daemon=True).start()

    def _on_polled(self, snap, error: str) -> None:
        self._polling = False
        self.snapshot = snap if isinstance(snap, dict) else None
        self.error = error
        self.polled_at = time.time()
        self.refresh()

    # -- what it shows --------------------------------------------------------------

    @property
    def reachable(self) -> bool:
        return self.snapshot is not None

    @property
    def com_status(self) -> str:
        return str(((self.snapshot or {}).get("com") or {}).get("status") or "unknown")

    @property
    def position(self) -> str:
        pos = (self.snapshot or {}).get("position")
        return pos if pos in (POSITION_IN, POSITION_OUT) else "unknown"

    @property
    def move_in_progress(self) -> bool:
        return self._moving is not None or bool((self.snapshot or {}).get("move_in_progress"))

    def look(self) -> tuple[str, str, str]:
        """(colour, short suffix, sentence) for the current state."""
        if self.move_in_progress:
            target = f" {self._moving}" if self._moving else ""
            return PURPLE, " · moving…", f"moving{target}"
        if not self.reachable:
            return GREY, "", f"PDXC server not reachable ({self.error})"
        if self.com_status != "connected":
            com = (self.snapshot or {}).get("com") or {}
            why = com.get("last_error") or ("held by a Disconnect" if com.get("held") else "")
            return RED, " · no COM", (f"the PDXC server's COM port is {self.com_status}"
                                      + (f": {why}" if why else ""))
        if self.position == "unknown":
            return ORANGE, " · ?", ("position unknown (moved by hand, a jog, or a throw that "
                                     "did not finish): move it in or out to set it")
        return GREEN, f" · {self.position}", f"{self.position} ({_WHAT[self.position]})"

    def refresh(self) -> None:
        color, suffix, sentence = self.look()
        self.setText("APD stage" + suffix)
        set_style_if_changed(self, _css(color))
        lines = [f"APD pickoff stage (PDXC): {sentence}."]
        if self.reachable:
            lines.append("The position is the one the PDXC server last commanded; it does not "
                         "sense the stage.")
        if self.polled_at is not None:
            lines.append(f"Checked {time.strftime('%H:%M:%S', time.localtime(self.polled_at))}.")
        lines.append("Click: move in (APD) or out (Andor).")
        self.setToolTip("\n".join(lines))

    # -- moves ----------------------------------------------------------------------

    def move_allowed(self) -> tuple[bool, str]:
        """Whether a move may be sent now (the PDXC server checks its own move lock)."""
        if self.move_in_progress:
            return False, "the stage is moving"
        if not self.reachable:
            return False, "the PDXC server is not reachable"
        if self.com_status != "connected":
            return False, f"the PDXC server's COM port is {self.com_status}"
        return True, ""

    def request_move(self, state: str) -> bool:
        """Drive the stage to ``state`` on a thread, no confirmation.  Returns
        whether the move was sent."""
        allowed, why = self.move_allowed()
        if not allowed:
            QMessageBox.warning(self.window(), "APD stage", f"Not moved: {why}.")
            return False
        force = self.position == state
        run = self._run_text()
        self._moving = state
        self._record(f"[apd stage] moving {state}" + (" (full throw again)" if force else "")
                     + (f" during {run}" if run else "")
                     + (" (run state unknown: monitor server unreachable)" if run is None else ""))
        self.refresh()

        def work():
            try:
                result = self._get_client().move_to(state, force=force)
                self._emit(self._bridge.moved, state, str(result), "")
            except Exception as e:                       # noqa: BLE001
                self._emit(self._bridge.moved, state, "", str(e) or type(e).__name__)

        threading.Thread(target=work, name="apd-stage-move", daemon=True).start()
        return True

    def _on_moved(self, state: str, result: str, error: str) -> None:
        self._moving = None
        if error:
            self._record(f"[apd stage] move {state} FAILED: {error}")
            QMessageBox.warning(self.window(), "APD stage",
                                f"The move {state} failed: {error}\n\nThe stage position is "
                                "now unknown until a move in or out completes.")
        else:
            self._record(f"[apd stage] {result}")
        self.refresh()
        self.poll()

    # -- the menu -------------------------------------------------------------------

    def build_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.setStyleSheet(f"QMenu {{ background: {theme.BG_RAISED}; color: {theme.FG}; }}"
                           f"QMenu::item:disabled {{ color: {theme.FG_MUTED}; }}")
        menu.addAction(f"APD stage: {self.look()[2]}").setEnabled(False)
        menu.addSeparator()
        allowed, why = self.move_allowed()
        for state in (POSITION_IN, POSITION_OUT):
            again = " again" if self.position == state else ""
            act = menu.addAction(f"Move {state}{again} ({_WHAT[state]})")
            act.setObjectName(f"apd_stage_move_{state}")
            act.setEnabled(allowed)
            act.triggered.connect(lambda _=False, s=state: self.request_move(s))
        if not allowed:
            menu.addAction(f"    not now: {why}").setEnabled(False)
        menu.addSeparator()
        check = menu.addAction("Check now")
        check.setObjectName("apd_stage_check")
        check.triggered.connect(self.poll)
        return menu

    def _show_menu(self, pos) -> None:
        menu = self.build_menu()
        menu.exec(self.mapToGlobal(pos))
        delete_later(menu)

    def shutdown(self) -> None:
        """Stop polling.  A move in flight finishes on the PDXC server."""
        self._closed = True
        self._timer.stop()
