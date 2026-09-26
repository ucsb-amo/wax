"""The strip above every tab of the Device Control GUI.

It answers "is anything dangerous on, can I trust what the tabs show, and can
I act right now?" without opening the Composite tab:

* banners -- the monitor is not running (with Start), the device state is
  untrusted (with Trust state), a run is starting or in progress, the
  interlock has tripped, the hardware is busy playing out a ramp, a watchdog
  is about to act;
* hazard chips -- one per device that says it is dangerous to leave on (a
  coil at current), with how long this GUI has seen it so; a click shows the
  card;
* Make safe -- opens :class:`MakeSafeDialog`, which lists each hazardous
  device's safe op with its values and sends the ticked ones.

It only displays what it is given (``set_*``) and emits requests; the host
GUI does the talking.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from waxx.util.comms_server.comm_server import STATES
from waxx.util.dashboard import theme
from waxx.util.guis.card_layout import FlowLayout

HAZARD = "#ff5252"
WARN_TEXT = "#f0c14b"
ERR_TEXT = "#ff6b6b"
INFO_TEXT = "#8cc4ff"

_BANNER_COLORS = {
    "error": ("#4a1c1c", ERR_TEXT),
    "warn": ("#46391a", WARN_TEXT),
    "info": ("#1d3247", INFO_TEXT),
}


def _fmt_s(seconds) -> str:
    if seconds is None:
        return ""
    seconds = max(int(round(float(seconds))), 0)
    if seconds < 60:
        return f"{seconds} s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m} min" if m >= 5 else f"{m}:{s:02d}"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


class _Banner(QFrame):
    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        self.key = key
        self.setObjectName("summary_banner")
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 6, 4)
        row.setSpacing(8)
        self.label = QLabel("")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.label, 1)
        self.button = QPushButton("")
        self.button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button.hide()
        row.addWidget(self.button)
        self._level = ""

    def show_text(self, level: str, text: str, button: str = "") -> None:
        if level != self._level:
            self._level = level
            bg, fg = _BANNER_COLORS[level]
            self.setStyleSheet(f"QFrame#summary_banner {{ background: {bg};"
                               f" border: 1px solid {fg}; border-left: 4px solid {fg};"
                               f" border-radius: 5px; }}"
                               f"QLabel {{ color: {fg}; font-size: 12px; }}"
                               f"QPushButton {{ color: {fg}; background: transparent;"
                               f" border: 1px solid {fg}; border-radius: 10px; padding: 2px 10px; }}"
                               f"QPushButton:hover {{ background: {theme.BG_BUTTON_HOVER}; }}")
        self.label.setText(text)
        self.button.setText(button)
        self.button.setVisible(bool(button))
        self.show()


class SummaryStrip(QWidget):
    make_safe_requested = pyqtSignal()
    trust_requested = pyqtSignal()
    start_monitor_requested = pyqtSignal()
    show_device_requested = pyqtSignal(str)
    clear_fence_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        self.banners: dict[str, _Banner] = {}
        for key in ("monitor", "trust", "run", "interlock", "watchdog", "busy"):
            banner = _Banner(key)
            banner.hide()
            self.banners[key] = banner
            box.addWidget(banner)
        self.banners["monitor"].button.clicked.connect(self.start_monitor_requested.emit)
        self.banners["trust"].button.clicked.connect(self.trust_requested.emit)
        self.banners["run"].button.clicked.connect(self.clear_fence_requested.emit)

        self.hazard_row = QFrame()
        self.hazard_row.setObjectName("hazard_row")
        self.hazard_row.setStyleSheet(f"QFrame#hazard_row {{ background: #3a1818;"
                                      f" border: 1px solid {HAZARD}; border-radius: 5px; }}")
        row = QHBoxLayout(self.hazard_row)
        row.setContentsMargins(8, 3, 6, 3)
        row.setSpacing(6)
        title = QLabel("⚠")
        title.setToolTip("Devices that say they are dangerous to leave like this.")
        title.setStyleSheet(f"color: {HAZARD}; font-weight: 700;")
        row.addWidget(title)
        self._chips_host = QWidget()
        self._chips = FlowLayout(self._chips_host, hspacing=6, vspacing=4)
        row.addWidget(self._chips_host, 1)
        self.make_safe_button = QPushButton("Make safe…")
        self.make_safe_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.make_safe_button.setStyleSheet(
            f"QPushButton {{ color: white; background: #b71c1c; border: 1px solid {HAZARD};"
            f" border-radius: 11px; padding: 3px 12px; font-weight: 700; }}"
            f"QPushButton:hover {{ background: #d32f2f; }}"
            f"QPushButton:disabled {{ background: #5a2a2a; color: #b08080; }}")
        self.make_safe_button.setToolTip("Choose which hazardous devices to switch to their "
                                         "safe state (each card's safe op).")
        self.make_safe_button.clicked.connect(self.make_safe_requested.emit)
        row.addWidget(self.make_safe_button)
        box.addWidget(self.hazard_row)
        self.hazard_row.hide()
        self._chip_buttons: list[QPushButton] = []
        self._hazard_key = None

    # -- inputs -------------------------------------------------------------------

    def set_monitor(self, state, reachable: bool, detail: str = "") -> None:
        b = self.banners["monitor"]
        if not reachable:
            b.show_text("error", "Monitor server unreachable: nothing typed here reaches "
                                 "the hardware, and the tabs may be out of date.")
        elif state == STATES.NOT_READY:
            b.show_text("warn", "The monitor is not running"
                                + (f" ({detail})" if detail else "")
                                + ": channel edits and composite ops are not applied "
                                  "until it runs.", "Start monitor")
        else:
            b.hide()

    def set_trust(self, trust: dict | None) -> None:
        b = self.banners["trust"]
        if trust and trust.get("trusted") is False:
            b.show_text("error", "Device state UNTRUSTED: " + str(trust.get("reason", "")) +
                        ". The tabs show the state file, which may not match the hardware "
                        "until a run ends normally.", "Trust state…")
        else:
            b.hide()

    def set_run(self, pending: dict | None, live_od: dict | None) -> None:
        """A fence (a run announced itself and has not yet taken the core)
        always gets a Clear button: a run that died before taking the core
        still looks "in progress" to liveOD, which is told nothing either."""
        b = self.banners["run"]
        live_od = live_od or {}
        text = ""
        if live_od.get("run_in_progress"):
            shots = ""
            if live_od.get("n_shots_expected"):
                shots = f", shot {live_od.get('n_shots') or 0}/{live_od.get('n_shots_expected')}"
            text = (f"Run {live_od.get('run_id')} "
                    f"({live_od.get('expt_name') or 'experiment'}) in progress{shots} (liveOD).")
        if pending:
            text = (text + " " if text else "") + (
                f"Run {pending.get('run_id')} ({pending.get('expt') or 'experiment'}) announced "
                "itself and has not taken the core yet: composite ops are refused until it "
                "does or ends.")
            b.show_text("info", text, "Clear fence…" if pending.get("token") else "")
        elif text:
            b.show_text("info", text + " The hardware belongs to it.")
        else:
            b.hide()

    def set_interlock(self, state: str | None, magnets_enabled) -> None:
        b = self.banners["interlock"]
        if state is None or state == "ok":
            if magnets_enabled is False:
                b.show_text("warn", "Interlock ok, but the magnets are DISABLED at the relay.")
            else:
                b.hide()
        elif state == "tripped":
            b.show_text("error", "INTERLOCK TRIPPED -- the magnets are disabled. Coil ops "
                                 "cannot drive current until it is reset (Interlock panel).")
        else:
            b.show_text("warn", f"Interlock state: {state}.")

    def set_busy(self, seconds: float) -> None:
        b = self.banners["busy"]
        if seconds and seconds > 0.5:
            b.show_text("info", f"Hardware busy for ~{seconds:.0f} s (a ramp is playing out); "
                                "ops sent now wait for it.")
        else:
            b.hide()

    def set_watchdog_warnings(self, warnings: list[str]) -> None:
        b = self.banners["watchdog"]
        if warnings:
            b.show_text("error", " · ".join(warnings) +
                        " (Composite tab: Keep on, or turn it off)")
        else:
            b.hide()

    def set_hazards(self, hazards: list[dict]) -> None:
        key = tuple((h["key"], h["text"], int((h.get("seen_s") or 0) // 60),
                     bool(h.get("max_on_s") and (h.get("seen_s") or 0) > h["max_on_s"]))
                    for h in hazards)
        if key == self._hazard_key:
            return
        self._hazard_key = key
        for b in self._chip_buttons:
            self._chips.removeWidget(b)
            b.deleteLater()
        self._chip_buttons = []
        for h in hazards:
            seen = h.get("seen_s")
            over = h.get("max_on_s") and seen is not None and seen > h["max_on_s"]
            text = f"{h['title']}: {h['text']}"
            if seen is not None and seen >= 60:
                text += f" · {_fmt_s(seen)}"
            chip = QPushButton(text)
            chip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            chip.setToolTip("Seen hazardous by this GUI for " + (_fmt_s(seen) or "a moment") +
                            (f" -- longer than its {_fmt_s(h['max_on_s'])} limit" if over else "")
                            + ". Click to show its card.")
            color = WARN_TEXT if over else "white"
            chip.setStyleSheet(f"QPushButton {{ color: {color}; background: #5c1f1f;"
                               f" border: 1px solid {HAZARD}; border-radius: 10px;"
                               f" padding: 2px 10px; font-weight: 600; }}"
                               f"QPushButton:hover {{ background: #7a2626; }}")
            chip.clicked.connect(lambda _=False, k=h["key"]: self.show_device_requested.emit(k))
            self._chips.addWidget(chip)
            self._chip_buttons.append(chip)
        self.hazard_row.setVisible(bool(hazards))

    def set_make_safe_enabled(self, enabled: bool, why: str = "") -> None:
        self.make_safe_button.setEnabled(enabled)
        self.make_safe_button.setToolTip(why or "Choose which hazardous devices to switch to "
                                               "their safe state (each card's safe op).")


class MakeSafeDialog(QDialog):
    """Tick the devices to make safe; each line says exactly what will be
    sent."""

    def __init__(self, plans: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Make safe")
        box = QVBoxLayout(self)
        intro = QLabel("Each ticked device gets its safe op, in this order:")
        intro.setWordWrap(True)
        box.addWidget(intro)
        self.checks: dict[str, QCheckBox] = {}
        for plan in plans:
            check = QCheckBox(f"{plan['title']} ({plan['text']}): {plan['action']}")
            check.setChecked(plan.get("action_ok", True))
            check.setEnabled(plan.get("action_ok", True))
            if not plan.get("action_ok", True):
                check.setText(check.text() + " -- no safe op defined")
            box.addWidget(check)
            self.checks[plan["key"]] = check
        buttons = QDialogButtonBox()
        self.go = buttons.addButton("Make safe", QDialogButtonBox.ButtonRole.AcceptRole)
        self.go.setStyleSheet(f"color: {ERR_TEXT}; font-weight: 700;")
        buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box.addWidget(buttons)

    def selected(self) -> list[str]:
        return [k for k, c in self.checks.items() if c.isChecked() and c.isEnabled()]
