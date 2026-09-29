"""The camera in the liveOD status row, and a drop-down for the others.

    [Running] 80545 · hf_tweezer_bec  [ andor |▾]  [====> 37/120]  Δt 8.2s ...

No liveOD window uses ``CameraMenuButton`` any more: the acquisition window (with
or without the camera host) and the remote viewer all show
``camera_control.CameraControl``, so the camera button looks the same everywhere.
It stays for old imports; ``STATES`` and ``CONNECTED_STATES`` here are the shared
state colours.

The button shows one camera -- the run's, or before there has been a run the first
one connected -- coloured by its state, and connects / disconnects it when clicked.
The arrow drops down the same button for each of the other cameras. This replaces
the row of camera buttons the window used to have.

Only displays and asks: ``toggle_requested(camera_key)`` goes to whoever owns the
cameras (the acquisition window's CamConnBar, or the liveOD server for a remote
viewer), and the states come back through ``set_state`` / ``set_states``. No camera
or config imports, so the remote viewer can use it without ARTIQ.

``set_persist({camera_key: bool})`` (the additive ``persist`` of CAMERA_STATE) marks
a camera whose settings are persisted into runs as the camera host's
``CameraControl`` does: bright red with a white diagonal hatch and an LED in the
state colour; the arrow turns red when a camera not shown has it on, and that
camera's drop-down button is hatched.
"""

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QFontMetrics, QPainter
from PyQt6.QtWidgets import QMenu, QPushButton, QToolButton, QVBoxLayout, QWidget, QWidgetAction

# CameraButton.state -> (colour, what the tooltip calls it)
STATES = {
    "closed":   ("#9e9e9e", "not connected"),
    "loading":  ("#ba68c8", "connecting…"),
    "open":     ("#43a047", "connected"),
    "grabbing": ("#1e88e5", "grabbing"),
    "failed":   ("#c62828", "failed"),
}
CONNECTED_STATES = ("open", "grabbing")
ARROW_WIDTH = 18
LED_ROOM = 14           # the persist LED, left of the name
PERSIST_COLOR = "#ff1744"


def _style(color: str, selector: str) -> str:
    return (f"{selector} {{ background-color: {color}; color: white; font-weight: bold; "
            f"border: none; border-radius: 8px; padding: 1px 8px; }} "
            f"{selector}:disabled {{ color: rgba(255, 255, 255, 140); }}")


def _describe(key: str, state: str, persist: bool = False) -> str:
    action = "disconnect" if state in CONNECTED_STATES else "connect"
    text = f"{key}: {STATES.get(state, STATES['closed'])[1]}. Click to {action}."
    if persist:
        text += (f"\nPERSIST ON: liveOD applies {key}'s persisted settings on top of every "
                 f"run's camera_params (recorded in the run file).")
    return text


def _paint_persist(widget, body: QRectF, state: str, text: str):
    """Hatch, LED and name over a persisted camera's red body (the style sheet
    painted the fill and hides its own text)."""
    from waxx.util.live_od.gui import camera_control as cc     # it imports this module
    painter = QPainter(widget)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    path = cc._rounded(body.adjusted(0.5, 0.5, -0.5, -0.5), 8.0, left=True, right=True)
    cc.paint_hatch(painter, path, body)
    color = STATES.get(state, STATES["closed"])[0]
    cc.paint_led(painter, QPointF(body.left() + 4 + cc.LED / 2.0, body.center().y()), color,
                 "!" if state == "failed" else "")
    bold = widget.font()
    bold.setBold(True)
    cc.paint_outlined_text(painter, body.adjusted(LED_ROOM, 0, -4, 0), text, bold)
    painter.end()


class _PersistPushButton(QPushButton):
    """A drop-down camera button that can be hatched (Persist on)."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.persist = False
        self.state = "closed"

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.persist:
            _paint_persist(self, QRectF(self.rect()), self.state, self.text())


class CameraMenuButton(QToolButton):
    toggle_requested = pyqtSignal(str)      # camera_key

    def __init__(self, camera_keys=(), parent=None):
        super().__init__(parent)
        self._states = {}           # camera_key -> state, in the lab's order
        self._current = None        # the run's camera, once there has been one
        self._disabled = set()      # cameras waiting for an answer (remote viewer)
        self._menu_actions = {}     # camera_key -> (QWidgetAction, QPushButton)
        self._persist = {}          # camera_key -> Persist on (from CAMERA_STATE)

        self.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self.clicked.connect(self._on_clicked)
        for key in camera_keys:
            self._add_camera(key)
        self._render()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def camera_keys(self) -> list:
        return list(self._states)

    def state(self, camera_key: str) -> str:
        return self._states.get(camera_key, "closed")

    def set_state(self, camera_key: str, state: str):
        """Slot for ``CameraButton.state_changed``."""
        if camera_key not in self._states:
            self._add_camera(camera_key)
        self._states[camera_key] = state if state in STATES else "closed"
        self._render()

    def set_states(self, states: dict):
        """Every camera at once (a remote viewer's CAMERA_STATE); unknown cameras are
        added, in the order given."""
        for key, state in dict(states).items():
            if key not in self._states:
                self._add_camera(key)
            self._states[key] = state if state in STATES else "closed"
        self._render()

    def set_persist(self, persist: dict):
        """Which cameras have Persist on (CAMERA_STATE's additive ``persist``);
        cameras not named keep what they had."""
        for key, on in dict(persist or {}).items():
            self._persist[str(key)] = bool(on)
        self._render()

    def persisted(self, camera_key: str) -> bool:
        return bool(self._persist.get(camera_key))

    def hidden_persisted(self) -> list:
        """Cameras with Persist on that are not the one on the button."""
        shown = self.shown_camera()
        return [k for k in self._states if k != shown and self._persist.get(k)]

    def set_current(self, camera_key: str):
        """The camera the run uses: the one on the button."""
        if camera_key and camera_key not in self._states:
            self._add_camera(camera_key)
        self._current = camera_key or None
        self._render()

    def set_camera_enabled(self, camera_key: str, enabled: bool):
        """Grey one camera out while a request for it is on its way."""
        if enabled:
            self._disabled.discard(camera_key)
        else:
            self._disabled.add(camera_key)
        self._render()

    def shown_camera(self):
        """The camera on the button: the run's; before any run, the first connected
        one, else the first there is; None with no cameras."""
        if self._current in self._states:
            return self._current
        for key, state in self._states.items():
            if state in CONNECTED_STATES:
                return key
        return next(iter(self._states), None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _add_camera(self, key: str):
        self._states[key] = "closed"
        button = _PersistPushButton(key)
        button.clicked.connect(lambda _=False, k=key: self.toggle_requested.emit(k))
        holder = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(button)
        holder.setLayout(layout)
        action = QWidgetAction(self._menu)
        action.setDefaultWidget(holder)
        self._menu.addAction(action)
        self._menu_actions[key] = (action, button)
        # wide enough for the longest name, so the button never changes width when
        # the camera it shows does (the status row must not move)
        bold = self.font()
        bold.setBold(True)
        longest = max(QFontMetrics(bold).horizontalAdvance(k) for k in self._states)
        self.setFixedWidth(max(64, longest + 22 + LED_ROOM + ARROW_WIDTH))
        for _action, menu_button in self._menu_actions.values():
            menu_button.setMinimumWidth(self.width() - ARROW_WIDTH)

    def _on_clicked(self):
        key = self.shown_camera()
        if key is not None and key not in self._disabled:
            self.toggle_requested.emit(key)

    def _render(self):
        shown = self.shown_camera()
        for key, (action, button) in self._menu_actions.items():
            state = self._states[key]
            persist = bool(self._persist.get(key))
            action.setVisible(key != shown)
            button.persist, button.state = persist, state
            if persist:
                button.setStyleSheet(_style(PERSIST_COLOR, "QPushButton")
                                     + " QPushButton { color: transparent; }")
            else:
                button.setStyleSheet(_style(STATES[state][0], "QPushButton"))
            button.setToolTip(_describe(key, state, persist))
            button.setEnabled(key not in self._disabled)
            button.update()
        if shown is None:
            self.setText("no camera")
            self.setToolTip("No cameras yet")
            self.setStyleSheet(_style(STATES["closed"][0], "QToolButton"))
            self.setEnabled(False)
            return
        state = self._states[shown]
        persist = bool(self._persist.get(shown))
        hidden = self.hidden_persisted()
        self.setEnabled(True)
        self.setText(shown)
        others = len(self._states) > 1
        tip = _describe(shown, state, persist) + ("\nArrow: the other cameras." if others else "")
        if hidden:
            tip += f"\nPersist is ON for a camera not shown: {', '.join(hidden)}."
        self.setToolTip(tip)
        body = f" QToolButton {{ padding-right: {ARROW_WIDTH + 2}px;"
        body += " color: transparent; }" if persist else " }"
        arrow = (f" QToolButton::menu-button {{ border: none; width: {ARROW_WIDTH}px;"
                 " border-left: 1px solid rgba(255, 255, 255, 110);")
        if hidden:
            arrow += (f" background-color: {PERSIST_COLOR}; border-top-right-radius: 8px;"
                      " border-bottom-right-radius: 8px;")
        arrow += " }"
        self.setStyleSheet(_style(PERSIST_COLOR if persist else STATES[state][0], "QToolButton")
                           + body + arrow)
        self._menu.setEnabled(others)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        shown = self.shown_camera()
        if shown is not None and self._persist.get(shown):
            body = QRectF(self.rect()).adjusted(0, 0, -ARROW_WIDTH, 0)
            _paint_persist(self, body, self._states.get(shown, "closed"), shown)
