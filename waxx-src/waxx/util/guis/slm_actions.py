"""What the SLM pill's menu does (:mod:`waxx.util.guis.slm_pill`), for any host
that shows one: Device Control's status row and the monitor state bar
(:mod:`waxx.util.guis.monitor_state_panel`) behave the same because they both
use :class:`SlmActions`.

* **Re-initialise SLM now** / **Restart SLM server** -- a question, then the
  monitor server's ``slm_reinit`` request (``action`` "reinit" / "restart");
  the server sends it only while the machine is idle, and a refusal shows as
  a warning box;
* **View SLM server log** -- :class:`SlmLogWindow`, following the log through
  the monitor server (``output`` kind "slm") every :data:`SLM_LOG_POLL_MS`
  while it is open;
* **Launch spot finder** -- the host's launcher (the lab's SLM spot finder,
  started on this PC).

It never opens a socket: the host passes ``send(request, callback, **kw)``
(``kw``: ``timeout`` / ``attempts`` for the log polls; a host whose link
decides those itself may drop them).
"""

from __future__ import annotations

import socket
import time
from typing import Any, Callable

from PyQt6.QtCore import QByteArray, QObject, QSettings, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.guis.qt_upkeep import delete_later
from waxx.util.guis.slm_pill import SlmPill, describe_next

SLM_LOG_POLL_MS = 1000               # an open SLM log window asks for new lines this often
SLM_LOG_MAX_LINES = 5000             # lines an SLM log window keeps


def _default_settings() -> QSettings:
    # Device Control's own store: the log window keeps one geometry on a PC,
    # whichever host opened it.
    return QSettings("waxx", "device_control_gui")


def _host_name() -> str:
    try:
        return socket.gethostname()
    except Exception:                                 # noqa: BLE001
        return ""


class SlmLogWindow(QWidget):
    """Pop-out window following the SLM server's log (the SLM pill's "View
    SLM server log…"): every line the server and its supervisor write on the
    SLM PC.

    The monitor server fetches the log from the SLM PC while this window asks
    for it (:class:`SlmActions`, every ``SLM_LOG_POLL_MS``) and keeps its last
    lines, numbered (``OutputLog``), so the window asks only for lines after
    :attr:`after`. Parentless (own taskbar entry); deleted on close -- opened
    again, it loads what the monitor server kept.  ``settings``: a factory
    for the QSettings that keep its geometry.
    """

    closed = pyqtSignal()

    _GEOMETRY_KEY = "ui/slm_log_geometry"

    def __init__(self, settings: Callable[[], Any] | None = None):
        super().__init__(None, Qt.WindowType.Window)
        self._settings = settings or _default_settings
        self.setWindowTitle("SLM server log")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setStyleSheet(f"background: {theme.BG};")
        #: Number of the last line shown (the monitor server's numbering).
        self.after = 0
        #: A request is on its way (the host sends one at a time).
        self.fetching = False
        #: The monitor server cannot send the log (older code): stop asking.
        self.unsupported = False

        box = QVBoxLayout(self)
        box.setContentsMargins(6, 6, 6, 6)
        box.setSpacing(4)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("Asking the monitor server for the SLM server's log…")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._set_status(self.status_label.text(), theme.FG_MUTED)
        head.addWidget(self.status_label, 1)
        self.copy_button = QPushButton("Copy")
        self.copy_button.setToolTip("Copy every line shown to the clipboard.")
        self.copy_button.clicked.connect(self._copy_all)
        head.addWidget(self.copy_button)
        self.save_button = QPushButton("Save…")
        self.save_button.setToolTip("Save the lines shown to a text file.")
        self.save_button.clicked.connect(self._save)
        head.addWidget(self.save_button)
        box.addLayout(head)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(SLM_LOG_MAX_LINES)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.text.setFont(mono)
        self.text.setStyleSheet(
            f"QPlainTextEdit {{ color: {theme.FG}; background: {theme.BG_SUNKEN}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 4px; }}")
        self.text.setPlaceholderText("No lines yet.")
        box.addWidget(self.text, 1)

        self.resize(900, 420)
        try:
            geom = self._settings().value(self._GEOMETRY_KEY, None)
        except Exception:                             # noqa: BLE001
            geom = None
        if isinstance(geom, QByteArray) and not geom.isEmpty():
            try:
                self.restoreGeometry(geom)
            except Exception:                         # noqa: BLE001
                pass

    def request(self) -> dict:
        return {"type": "output", "kind": "slm", "after": self.after}

    def add_output(self, reply: dict) -> bool:
        """Show one ``output`` reply; True when the monitor server has more
        lines waiting."""
        if reply.get("status") != "ok":
            msg = str(reply.get("msg") or "no reply")
            if "unknown output kind" in msg or "unknown type" in msg:
                self.unsupported = True
                self._set_status("This monitor server cannot send the SLM server's log (it "
                                 "is from before 2026-09-28): restart the monitor server, then "
                                 "open this window again.", theme.WARN)
            else:
                self._set_status(f"No answer from the monitor server: {msg}", theme.WARN)
            return False
        try:
            first, nxt = int(reply.get("first", 0)), int(reply.get("next", 0))
        except (TypeError, ValueError):
            return False
        if nxt <= self.after:
            # The monitor server restarted (its numbering started again).
            self.text.clear()
            self.after = 0
            return True
        lines = [str(x) for x in (reply.get("lines") or [])]
        if lines and first > self.after + 1:
            lines.insert(0, "(earlier lines: in the log files on the SLM PC)" if self.after == 0
                         else f"({first - self.after - 1} lines were not kept here)")
        if lines:
            bar = self.text.verticalScrollBar()
            at_end = bar.value() >= bar.maximum() - 2
            self.text.appendPlainText("\n".join(lines))
            if at_end:
                bar.setValue(bar.maximum())
        self.after = max(self.after, nxt - 1)
        self._show_follow(reply.get("follow") or {})
        return bool(reply.get("more"))

    def _show_follow(self, follow: dict) -> None:
        state = follow.get("state")
        detail = str(follow.get("detail") or "")
        if state == "following":
            text = "Following the SLM server's log"
            if follow.get("path"):
                text += f" · {follow['path']} on the SLM PC"
            if isinstance(follow.get("last_fetch"), (int, float)):
                text += (" · fetched "
                         + time.strftime("%H:%M:%S", time.localtime(follow["last_fetch"])))
            self._set_status(text, theme.FG_MUTED)
        elif state == "trouble":
            self._set_status(f"⚠ {detail}", theme.WARN)
        elif state == "waiting":
            self._set_status(f"Waiting: {detail}", theme.FG_MUTED)
        else:
            self._set_status("Asking the SLM server for its log…", theme.FG_MUTED)

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")

    def _copy_all(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.clipboard().setText(self.text.toPlainText())

    def _save(self) -> None:
        default = f"slm_server_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Save the SLM server log", default,
                                              "Text (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.text.toPlainText() + "\n")
        except OSError as e:
            QMessageBox.warning(self, "Save the SLM server log", f"Not saved:\n{e}")

    def closeEvent(self, event):                                # noqa: N802
        try:
            self._settings().setValue(self._GEOMETRY_KEY, self.saveGeometry())
        except Exception:                             # noqa: BLE001
            pass
        self.closed.emit()
        super().closeEvent(event)


class SlmActions(QObject):
    """The SLM pill's menu, carried out (see the module docstring).

    ``pill``: the :class:`~waxx.util.guis.slm_pill.SlmPill` whose signals it
    answers (connected here); ``send(request, callback, **kw)``: the host's
    request path to the monitor server; ``dialog_parent``: the widget the
    questions and warnings belong to; ``record(line)``: the host's log of
    what was done (optional); ``launcher()``: starts the spot finder and
    returns a short description (None: the pill offers no launch);
    ``settings``: the log window's QSettings factory."""

    def __init__(self, pill: SlmPill, send: Callable[..., Any], *,
                 dialog_parent: QWidget | None = None,
                 record: Callable[[str], None] | None = None,
                 launcher: Callable[[], Any] | None = None,
                 settings: Callable[[], Any] | None = None):
        super().__init__(pill)
        self.pill = pill
        self._send = send
        self._parent = dialog_parent
        self._record = record
        self.launcher = launcher
        self._settings = settings
        self.log_window: SlmLogWindow | None = None
        self.log_timer: QTimer | None = None
        pill.reinit_requested.connect(self.request_reinit)
        pill.restart_requested.connect(self.request_restart)
        pill.log_requested.connect(self.open_log)
        pill.spot_finder_requested.connect(self.launch_spot_finder)

    # -- helpers --------------------------------------------------------------------

    def _line(self, text: str) -> None:
        if self._record is not None:
            self._record(text)

    def _ask(self, title: str, text: str, verb: str) -> bool:
        box = QMessageBox(self._parent)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(title)
        box.setText(text)
        yes = box.addButton(verb, QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        delete_later(box)
        return clicked is yes

    # -- reinit / restart -------------------------------------------------------------

    def request_reinit(self) -> None:
        """"Re-initialise SLM now": the monitor server sends it only while the
        machine is idle, and says so if it is not."""
        allowed, why = self.pill.reinit_allowed()
        if not allowed:
            QMessageBox.warning(self._parent, "Re-initialise SLM", f"Not sent: {why}.")
            return
        nxt = describe_next(self.pill.snapshot)
        if not self._ask("Re-initialise SLM",
                         "The SLM server re-initialises the SLM now (a few seconds) and puts the "
                         "pattern it shows back. The monitor server sends it only while no run "
                         "is starting or running."
                         + (f"\n\n{nxt}; the hour restarts after this one." if nxt else ""),
                         "Re-initialise"):
            return
        self._send({"type": "slm_reinit", "action": "reinit", "client": _host_name()},
                   self._on_reinit_reply)

    def _on_reinit_reply(self, reply: dict) -> None:
        if reply.get("status") == "ok":
            self._line("[slm] reinit asked for; the monitor server sends it now")
        else:
            self._line(f"[slm] reinit refused: {reply.get('msg')}")
            QMessageBox.warning(self._parent, "Re-initialise SLM",
                                f"The monitor server did not send it: {reply.get('msg')}")

    def request_restart(self) -> None:
        """"Restart SLM server": the server process exits and its supervisor on
        the SLM PC starts a new one; only while the machine is idle."""
        allowed, why = self.pill.restart_allowed()
        if not allowed:
            QMessageBox.warning(self._parent, "Restart SLM server", f"Not sent: {why}.")
            return
        if not self._ask("Restart SLM server",
                         "The SLM server process exits and its supervisor on the SLM PC starts a "
                         "new one (a few seconds): it re-initialises the SLM and puts the saved "
                         "pattern back. The monitor server sends it only while no run is "
                         "starting or running; a run started in those seconds fails its mask "
                         "write with an error rather than running without its mask.",
                         "Restart"):
            return
        self._send({"type": "slm_reinit", "action": "restart", "client": _host_name()},
                   self._on_restart_reply)

    def _on_restart_reply(self, reply: dict) -> None:
        if reply.get("status") == "ok":
            self._line("[slm] server restart asked for; the monitor server sends it now")
        else:
            self._line(f"[slm] server restart refused: {reply.get('msg')}")
            QMessageBox.warning(self._parent, "Restart SLM server",
                                f"The monitor server did not send it: {reply.get('msg')}")

    # -- the log window -----------------------------------------------------------------

    def open_log(self) -> None:
        """"View SLM server log": the log window, following the log (through
        the monitor server) while it is open."""
        if self.log_window is None:
            win = SlmLogWindow(settings=self._settings)
            if self._parent is not None:
                win.setWindowIcon(self._parent.windowIcon())
            win.closed.connect(self._on_log_window_closed)
            self.log_window = win
            if self.log_timer is None:
                self.log_timer = QTimer(self)
                self.log_timer.timeout.connect(self.poll_log)
            self.log_timer.start(SLM_LOG_POLL_MS)
            self.poll_log()
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def poll_log(self) -> None:
        win = self.log_window
        if win is None:
            if self.log_timer is not None:
                self.log_timer.stop()
            return
        if win.fetching or win.unsupported:
            return
        win.fetching = True
        self._send(win.request(), lambda reply, w=win: self._on_log_reply(w, reply),
                   timeout=2.0, attempts=1)

    def _on_log_reply(self, win: SlmLogWindow, reply: dict) -> None:
        if win is not self.log_window:
            return                  # closed (or replaced) meanwhile
        win.fetching = False
        if win.add_output(reply if isinstance(reply, dict) else {}):
            self.poll_log()         # more lines waiting: ask again now

    def _on_log_window_closed(self) -> None:
        self.log_window = None
        if self.log_timer is not None:
            self.log_timer.stop()

    # -- the spot finder ------------------------------------------------------------------

    def launch_spot_finder(self) -> None:
        """"Launch spot finder": the host's launcher, on this PC."""
        if self.launcher is None:
            return
        try:
            what = self.launcher()
        except Exception as e:                        # noqa: BLE001
            self._line(f"[slm] spot finder did not start: {e}")
            QMessageBox.warning(self._parent, "Launch spot finder",
                                f"The spot finder did not start:\n{e}")
            return
        self._line(f"[slm] spot finder started{f' ({what})' if what else ''}")

    # -- teardown ---------------------------------------------------------------------------

    def shutdown(self) -> None:
        """Close the log window and stop its poll (the host is going)."""
        win = self.log_window
        if win is not None:
            self.log_window = None
            win.close()
        if self.log_timer is not None:
            self.log_timer.stop()


__all__ = ["SLM_LOG_MAX_LINES", "SLM_LOG_POLL_MS", "SlmActions", "SlmLogWindow"]
