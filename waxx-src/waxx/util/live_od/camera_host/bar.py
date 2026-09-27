"""Stand-ins for the camera bar (CamConnBar / CameraButton) when liveOD's
cameras are owned by its camera host.

The window and the server use the old bar's surface -- ``buttons``,
``get_button``, ``get_states``, a button's ``camera_name``, ``camera_params``,
``state``, ``state_changed``, ``camera.is_opened()``, ``open_camera()``,
``close_camera()``, ``button_pressed()``, ``_set_color_*`` -- and this keeps
it.  The difference: the state comes from the host's snapshots
(``apply_snapshot``), never from the button itself, so ``_set_color_*`` do
nothing; and opening or closing is a request to the host, which does it on
its own threads (the GUI thread never waits for a camera).
"""
from __future__ import annotations

import logging

from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget
from PyQt6.QtCore import pyqtSignal

logger = logging.getLogger("waxx.live_od.camera_host")

#: the old buttons' colours, by legacy state word
STATE_COLOURS = {"loading": "orchid", "failed": "red", "open": "green", "grabbing": "blue",
                 "closed": "gray"}


def _key(p) -> str:
    k = getattr(p, "key", "")
    return k.decode() if isinstance(k, bytes) else str(k)


class HostCameraView:
    """What ``button.camera`` is in host mode: asks the host."""

    def __init__(self, host, key: str) -> None:
        self._host = host
        self._key = key

    def is_opened(self) -> bool:
        return self._host.is_open(self._key)

    def Close(self):
        return self._host.request(self._key, "close", origin="liveOD window")

    close = Close

    def stop_grab(self) -> None:
        pass


class HostCameraButton(QPushButton):
    state_changed = pyqtSignal(str, str)   # camera_name, state

    def __init__(self, camera_params, host, output_window=None) -> None:
        super().__init__()
        self.camera_params = camera_params
        self.camera_name = _key(camera_params)
        self.host = host
        self.camera = HostCameraView(host, self.camera_name)
        self.output_window = output_window
        self.state = "closed"
        self.host_state = "closed"
        self.is_grabbing = False
        self.setText(self.camera_name)
        self.setStyleSheet(f"background-color: {STATE_COLOURS['closed']}")
        self.clicked.connect(self.button_pressed)

    def msg(self, txt: str) -> None:
        logger.info(txt)

    def _request(self, action: str):
        try:
            fut = self.host.request(self.camera_name, action, origin="liveOD window")
        except Exception as exc:
            logger.warning(f"{self.camera_name}: {action}: {exc}")
            return None
        fut.add_done_callback(lambda f, a=action: self._report(a, f))
        return fut

    def _report(self, action, fut) -> None:
        exc = fut.exception()
        if exc is not None:
            logger.warning(f"{self.camera_name}: {action} failed: {exc}")

    def button_pressed(self):
        return self._request("toggle")

    def open_camera(self):
        return self._request("open")

    def close_camera(self):
        return self._request("close")

    def toggle_grabbing(self, success_bool=True) -> None:
        pass

    def set_host_state(self, state: str, host_state: str = "") -> None:
        self.host_state = host_state or state
        if state != self.state:
            self.setStyleSheet(f"background-color: {STATE_COLOURS.get(state, 'red')}")
            self.state = state
            try:
                self.state_changed.emit(self.camera_name, state)
            except Exception:
                pass

    # the host's snapshot sets the state; these exist for callers of the old button
    def _set_color_loading(self):
        pass

    def _set_color_failed(self):
        pass

    def _set_color_success(self):
        pass

    def _set_color_grabbing(self):
        pass

    def _set_color_closed(self):
        pass


class HostCameraBar(QWidget):
    """One HostCameraButton per camera in the lab's list, in its order."""

    def __init__(self, host, output_window=None, camera_params_list=None) -> None:
        super().__init__()
        self.host = host
        self.output_window = output_window
        if camera_params_list is None:
            camera_params_list = [s.params for s in host.specs]
        self.buttons = []
        for p in camera_params_list:
            b = HostCameraButton(p, host, output_window)
            self.buttons.append(b)
            setattr(self, f"{b.camera_name}_button", b)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.setSpacing(4)
        for b in self.buttons:
            row.addWidget(b)
        layout.addLayout(row)
        self.setLayout(layout)

    def get_button(self, camera_key: str):
        for b in self.buttons:
            if b.camera_name == camera_key:
                return b
        return None

    def get_states(self) -> dict:
        return {b.camera_name: b.state for b in self.buttons}

    def apply_snapshot(self, snapshot) -> None:
        cams = (snapshot or {}).get("cameras") or {}
        for b in self.buttons:
            c = cams.get(b.camera_name)
            if c is not None:
                b.set_host_state(c.get("state", "failed"), c.get("host_state", ""))


__all__ = ["HostCameraBar", "HostCameraButton", "HostCameraView", "STATE_COLOURS"]
