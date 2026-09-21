"""One-line run status for the liveOD window.

    [Running] 80545 · hf_tweezer_bec  [andor ▾]  [====> 37/120]  Δt 8.2s · ETA 14:32

The shot's xvar values are not here: they are a plate on the OD image (the
viewer's ``set_shot_xvars``). The state comes from ``LiveODServer.run_state_signal``;
the strip adds a stall watchdog (no shot for several times the usual shot period)
and the elapsed time.
No server, camera or config imports: it only displays what it is told. The camera
button is the window's (``add_camera_widget``); without one the run label names the
camera instead.

The strip's minimum width never changes. Every piece is either a fixed width or
elides its text, so a run starting (long experiment name, badge appearing) cannot
push the window wider. What is cut off is in the tooltip.
"""

import time

from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QWidget

from waxx.util.live_od.gui import theme

# state -> (pill text, background colour). The pill is a fixed width: keep these short.
STATES = {
    "idle":               ("Idle", "#9e9e9e"),
    "waiting_camera":     ("Camera…", "#ba68c8"),
    "waiting_grab_drain": ("Old grab…", "#ba68c8"),
    "running":            ("Running", "#43a047"),
    "saving":             ("Saving", "#1e88e5"),
    "saved":              ("Saved", "#2e7d32"),
    "done":               ("Done", "#2e7d32"),
    "aborting":           ("Aborting", "#fb8c00"),
    "aborted":            ("Aborted", "#fb8c00"),
    "stalled":            ("Stalled", "#f9a825"),
    "error":              ("Error", "#c62828"),
}
STATE_TIPS = {
    "waiting_camera": "Waiting for the camera to report ready",
    "waiting_grab_drain": "Waiting for the previous run's grab loop to let go of the camera",
    "done": "Finished; nothing was saved",
}
ACTIVE_STATES = ("waiting_camera", "waiting_grab_drain", "running", "saving", "aborting")

STALL_FACTOR = 3.0      # no shot for this many average shot periods ...
STALL_MIN_S = 10.0      # ... and at least this long -> "Stalled"

PILL_WIDTH = 78
PROGRESS_WIDTH = 110
TIMING_WIDTH = 176


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class ElidedLabel(QLabel):
    """A label that never asks for more room than it is given: text that does not
    fit ends in an ellipsis, and the whole of it is the tooltip."""

    def __init__(self, min_width=40):
        super().__init__()
        self._full_text = ""
        self.setMinimumWidth(min_width)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def setText(self, text):
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self.update()

    def text(self):
        return self._full_text

    def clear(self):
        self.setText("")

    def minimumSizeHint(self):
        return QSize(self.minimumWidth(), QFontMetrics(self.font()).height())

    def sizeHint(self):
        return self.minimumSizeHint()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))    # follows the style sheet
        metrics = QFontMetrics(self.font())
        shown = metrics.elidedText(self._full_text, Qt.TextElideMode.ElideRight, self.width())
        painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), shown)


class StatusStrip(QWidget):
    title_changed = pyqtSignal(str)     # short text for the window title

    def __init__(self):
        super().__init__()
        self._state = "idle"
        self._detail = ""
        self._run_id = 0
        self._expt_class = ""
        self._camera_key = ""
        self._eta = "--:--"
        self._delta_t = None
        self._next_run_id = None
        self._run_start = None
        self._run_end = None
        self._last_shot_time = None
        self._shot_periods = []
        self._shot = 0
        self._n_shots = 0
        self._stalled = False
        self._camera_widget = None

        bold = QFont()
        bold.setBold(True)

        self.pill = QLabel()
        self.pill.setFont(bold)
        self.pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pill.setFixedWidth(PILL_WIDTH)

        self.run_label = ElidedLabel(min_width=90)
        self.run_label.setFont(bold)

        self._save_data = True

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%v/%m")
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress.setTextVisible(False)     # no "0/1" before there has been a run
        self.progress.setFixedWidth(PROGRESS_WIDTH)
        self.progress.setFixedHeight(18)

        # Δt and ETA are what gets looked at during a run: big bold numbers
        self.timing_label = QLabel()
        self.timing_label.setTextFormat(Qt.TextFormat.RichText)
        self.timing_label.setFixedWidth(TIMING_WIDTH)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.pill)
        layout.addWidget(self.run_label, 5)
        layout.addWidget(self.progress)
        layout.addWidget(self.timing_label)
        self.setLayout(layout)
        self._layout = layout

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)

        self._restyle()
        theme.notifier().changed.connect(self._restyle)
        self._render_pill()
        self._render_run_label()

    def _restyle(self, *_):
        # a quiet bar: it is there to glance at, not to be the brightest thing in the window
        self.progress.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {theme.color('separator')}; border-radius: 3px; "
            f"background: transparent; color: {theme.color('muted')}; text-align: center; }} "
            f"QProgressBar::chunk {{ background-color: {theme.color('progress')}; border-radius: 2px; }}")
        self._render_timing()
        # an unsaved run has no run ID to show; its place says so, in red
        self.run_label.setStyleSheet("" if self._save_data or self._run_start is None
                                     else f"color: {theme.color('error')};")

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def add_camera_widget(self, widget):
        """Put the camera button right after the run label, which then stops naming
        the camera. The widget must be a fixed width (see the module docstring)."""
        self._camera_widget = widget
        self._layout.insertWidget(self._layout.indexOf(self.run_label) + 1, widget)
        self._render_run_label()

    def set_state(self, state: str, detail: str = ""):
        """Slot for ``LiveODServer.run_state_signal``."""
        if state not in STATES:
            state = "idle"
        was_active = self._state in ACTIVE_STATES
        self._state = state
        self._detail = detail or ""
        self._stalled = False
        if was_active and state not in ACTIVE_STATES:
            self._run_end = time.time()
        self._render_pill()
        self._render_timing()
        self._emit_title()

    def state(self) -> str:
        return "stalled" if self._stalled else self._state

    def start_run(self, run_id: int, expt_class: str = "", camera_key: str = "",
                  save_data: bool = True, n_shots: int = 0):
        self._run_id = int(run_id)
        self._expt_class = expt_class
        self._camera_key = camera_key
        self._run_start = time.time()
        self._run_end = None
        self._last_shot_time = None
        self._shot_periods = []
        self._eta = "--:--"
        self._delta_t = None
        self._shot = 0
        self._n_shots = int(n_shots)
        self._save_data = bool(save_data)
        self._restyle()
        self.progress.setRange(0, max(1, self._n_shots))
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self._render_run_label()
        self._render_timing()
        self._emit_title()

    def set_next_run_id(self, run_id):
        """The id the next saved run will get: shown until the first run starts,
        and in the run label's tooltip after that."""
        self._next_run_id = run_id
        self._render_run_label()

    def set_progress(self, shot: int, n_shots: int):
        self._shot, self._n_shots = int(shot), int(n_shots)
        self.progress.setRange(0, max(1, self._n_shots))
        self.progress.setValue(min(self._shot, self._n_shots))
        self._last_shot_time = time.time()
        if self._stalled:
            self._stalled = False
            self._render_pill()
        self._emit_title()

    def set_timing(self, delta_t: float, eta: str):
        self._delta_t, self._eta = float(delta_t), str(eta)
        if self._shot > 1:                      # the first period includes camera start-up
            self._shot_periods = (self._shot_periods + [self._delta_t])[-5:]
        self._render_timing()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _tick(self):
        if self._state == "running" and self._last_shot_time is not None and self._shot_periods:
            age = time.time() - self._last_shot_time
            limit = max(STALL_MIN_S, STALL_FACTOR * sum(self._shot_periods) / len(self._shot_periods))
            stalled = age > limit and self._shot < self._n_shots
            if stalled or stalled != self._stalled:
                self._stalled = stalled
                self._render_pill(f"No shot for {age:.0f} s" if stalled else "")
        if self._state in ACTIVE_STATES:
            self._render_timing()

    def _render_pill(self, stall_text: str = ""):
        state = "stalled" if self._stalled else self._state
        text, color = STATES[state]
        self.pill.setText(text)
        self.pill.setToolTip("\n".join(t for t in (stall_text, STATE_TIPS.get(state, ""), self._detail) if t))
        self.pill.setStyleSheet(f"background-color: {color}; color: white;"
                                "padding: 1px 4px; border-radius: 8px;")

    def _render_run_label(self):
        nxt = f"Next run: {self._next_run_id}" if self._next_run_id is not None else "Next run: (unavailable)"
        if self._run_start is None:
            self.run_label.setText(nxt)
            return
        if not self._save_data:
            parts = ["NOT SAVING"]
        else:
            parts = [str(self._run_id) if self._run_id else "unsaved"]
        parts += [p for p in (self._expt_class, self._camera_key) if p]
        shown = parts if self._camera_widget is None else [p for p in parts if p != self._camera_key]
        self.run_label.setText(" · ".join(shown))
        tip = f"Run {' · '.join(parts)}\n{nxt}"
        if not self._save_data:
            tip = "save_data=False: nothing from this run is written to disk\n" + tip
        self.run_label.setToolTip(tip)

    def _render_timing(self):
        def item(label, value):
            return (f'<span style="color:{theme.color("muted")}">{label}</span>&nbsp;'
                    f'<span style="font-size:12pt; font-weight:bold">{value}</span>')
        delta_t = f"{self._delta_t:.1f}s" if self._delta_t is not None else "–"
        eta = self._eta if self._run_start is not None else "--:--"
        self.timing_label.setText(item("Δt", delta_t) + "&nbsp;&nbsp;&nbsp;" + item("ETA", eta))
        if self._run_start is not None:
            end = self._run_end if self._run_end is not None else time.time()
            self.timing_label.setToolTip(f"Elapsed {_format_duration(end - self._run_start)}")

    def _emit_title(self):
        if self._run_start is None:
            self.title_changed.emit("")
            return
        run = f"Run {self._run_id}" if self._run_id else "unsaved run"
        self.title_changed.emit(f"{run} — {STATES[self._state][0]} — {self._shot}/{self._n_shots}")
