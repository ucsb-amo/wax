"""The camera in the liveOD status row, and a drop-down for the others.

    [Running] 80545 · hf_tweezer_bec  [ andor |▾]  [====> 37/120]  Δt 8.2s ...

The button shows one camera -- the run's, or before there has been a run the first
one connected -- coloured by its state, and connects / disconnects it when clicked.
The arrow drops down the same button for each of the other cameras. This replaces
the row of camera buttons the window used to have.

Only displays and asks: ``toggle_requested(camera_key)`` goes to whoever owns the
cameras (the acquisition window's CamConnBar, or the liveOD server for a remote
viewer), and the states come back through ``set_state`` / ``set_states``. No camera
or config imports, so the remote viewer can use it without ARTIQ.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFontMetrics
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


def _style(color: str, selector: str) -> str:
    return (f"{selector} {{ background-color: {color}; color: white; font-weight: bold; "
            f"border: none; border-radius: 8px; padding: 1px 8px; }} "
            f"{selector}:disabled {{ color: rgba(255, 255, 255, 140); }}")


def _describe(key: str, state: str) -> str:
    action = "disconnect" if state in CONNECTED_STATES else "connect"
    return f"{key}: {STATES.get(state, STATES['closed'])[1]}. Click to {action}."


class CameraMenuButton(QToolButton):
    toggle_requested = pyqtSignal(str)      # camera_key

    def __init__(self, camera_keys=(), parent=None):
        super().__init__(parent)
        self._states = {}           # camera_key -> state, in the lab's order
        self._current = None        # the run's camera, once there has been one
        self._disabled = set()      # cameras waiting for an answer (remote viewer)
        self._menu_actions = {}     # camera_key -> (QWidgetAction, QPushButton)

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
        button = QPushButton(key)
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
        self.setFixedWidth(max(64, longest + 22 + ARROW_WIDTH))
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
            action.setVisible(key != shown)
            button.setStyleSheet(_style(STATES[state][0], "QPushButton"))
            button.setToolTip(_describe(key, state))
            button.setEnabled(key not in self._disabled)
        if shown is None:
            self.setText("no camera")
            self.setToolTip("No cameras yet")
            self.setStyleSheet(_style(STATES["closed"][0], "QToolButton"))
            self.setEnabled(False)
            return
        state = self._states[shown]
        self.setEnabled(True)
        self.setText(shown)
        others = len(self._states) > 1
        self.setToolTip(_describe(shown, state) + ("\nArrow: the other cameras." if others else ""))
        self.setStyleSheet(
            _style(STATES[state][0], "QToolButton")
            + f" QToolButton {{ padding-right: {ARROW_WIDTH + 2}px; }}"
            f" QToolButton::menu-button {{ border: none; width: {ARROW_WIDTH}px;"
            " border-left: 1px solid rgba(255, 255, 255, 110); }}")
        self._menu.setEnabled(others)
