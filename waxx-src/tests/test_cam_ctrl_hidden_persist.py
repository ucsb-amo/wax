"""A camera with Persist on that is not the one on the button must still be
visible: the ▾ turns red (tooltip names it) and its drop-down row is hatched."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt6.QtGui import QColor  # noqa: E402

from liveod_qt_helpers import delete_widgets, session_app  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return session_app()


def _entry(key, persist=False, host_state="idle"):
    from waxx.util.live_od.gui.camera_control import LEGACY_STATE
    return {"key": key, "category": "basler_usb", "camera_type": "basler",
            "host_state": host_state, "state": LEGACY_STATE[host_state], "persist": persist,
            "persisted": {"gain": 12.0} if persist else {}, "n_subs": 0}


def test_arrow_turns_red_for_a_hidden_persisted_camera(app):
    from waxx.util.live_od.gui.camera_control import CameraControl, PERSIST_FILL
    control = CameraControl(["cam_a", "cam_b", "cam_c"])
    try:
        control.set_current("cam_a")
        control.set_snapshot({"cameras": {"cam_a": _entry("cam_a"),
                                          "cam_b": _entry("cam_b", persist=True),
                                          "cam_c": _entry("cam_c")}})
        control.show()
        app.processEvents()
        assert control.hidden_persisted() == ["cam_b"]
        assert control.arrow_button.is_red()
        assert "cam_b" in control.arrow_button.toolTip()
        assert "Persist is ON" in control.arrow_button.toolTip()
        # its pixels: persist red around the triangle
        image = control.arrow_button.grab().toImage()
        red = QColor(PERSIST_FILL)
        px = image.pixelColor(image.width() // 2, 2)
        assert (px.red(), px.green(), px.blue()) in (
            (red.red(), red.green(), red.blue()), (255, 255, 255))
        # the rows: cam_b hatched, cam_c not, cam_a (shown) hidden
        rows = {k: row for k, (_a, row) in control._rows.items()}
        assert rows["cam_b"].button.look().hatch
        assert not rows["cam_c"].button.look().hatch
        assert {k: a.isVisible() for k, (a, _r) in control._rows.items()} == {
            "cam_a": False, "cam_b": True, "cam_c": True}
        # the persisted camera on the button: the arrow is neutral again
        control.set_current("cam_b")
        assert not control.arrow_button.is_red()
        assert control.main_button.look().hatch
        # Persist off everywhere: nothing red
        control.set_snapshot({"cameras": {k: _entry(k) for k in ("cam_a", "cam_b", "cam_c")}})
        assert not control.arrow_button.is_red() and not control.main_button.look().hatch
        assert "Persist" not in control.arrow_button.toolTip()
    finally:
        delete_widgets(app, [control])


def test_several_hidden_persisted_cameras_are_all_named(app):
    from waxx.util.live_od.gui.camera_control import CameraControl
    control = CameraControl(["cam_a", "cam_b", "cam_c"])
    try:
        control.set_current("cam_a")
        control.set_snapshot({"cameras": {"cam_a": _entry("cam_a"),
                                          "cam_b": _entry("cam_b", persist=True),
                                          "cam_c": _entry("cam_c", persist=True)}})
        assert control.hidden_persisted() == ["cam_b", "cam_c"]
        assert "cam_b, cam_c" in control.arrow_button.toolTip()
    finally:
        delete_widgets(app, [control])


def test_legacy_words_show_persist_like_the_host(app):
    """The remote viewer's control: Persist from CAMERA_STATE's additive persist,
    on legacy words, looks as the host's; a new word keeps it; a host camera's own
    persist is not overridden."""
    from waxx.util.live_od.gui.camera_control import CameraControl, PERSIST_FILL
    control = CameraControl(glyphs=False)
    try:
        control.set_persist({"cam_b": True})                 # before the camera is known
        control.set_states({"cam_a": "open", "cam_b": "closed", "cam_c": "closed"})
        control.set_current("cam_a")
        assert control.persisted("cam_b") and not control.persisted("cam_a")
        assert control.hidden_persisted() == ["cam_b"] and control.arrow_button.is_red()
        rows = {k: row for k, (_a, row) in control._rows.items()}
        assert rows["cam_b"].button.look().hatch
        assert rows["cam_b"].button.look().fill == PERSIST_FILL
        assert "PERSIST ON" in rows["cam_b"].button.toolTip()
        control.set_states({"cam_b": "open"})                # a new word keeps Persist
        assert control.persisted("cam_b")
        control.set_current("cam_b")
        assert control.main_button.look().hatch and not control.arrow_button.is_red()
        control.set_persist({"cam_b": False})
        assert not control.main_button.look().hatch and control.hidden_persisted() == []

        control.set_snapshot({"cameras": {"cam_c": _entry("cam_c")}})
        control.set_persist({"cam_c": True})                 # the host's snapshot decides
        assert not control.persisted("cam_c")
    finally:
        delete_widgets(app, [control])


def test_legacy_menu_button_arrow_and_rows(app):
    """The remote viewer's CameraMenuButton, from CAMERA_STATE's additive persist."""
    from waxx.util.live_od.gui.camera_menu import CameraMenuButton, PERSIST_COLOR
    menu = CameraMenuButton(["cam_a", "cam_b", "cam_c"])
    try:
        menu.set_current("cam_a")
        menu.set_persist({"cam_b": True})
        assert menu.hidden_persisted() == ["cam_b"]
        assert PERSIST_COLOR in menu.styleSheet().split("::menu-button")[1]
        assert "cam_b" in menu.toolTip()
        buttons = {k: b for k, (_a, b) in menu._menu_actions.items()}
        assert buttons["cam_b"].persist and not buttons["cam_c"].persist
        assert "PERSIST ON" in buttons["cam_b"].toolTip()
        menu.set_persist({"cam_b": False})
        assert PERSIST_COLOR not in menu.styleSheet()
    finally:
        delete_widgets(app, [menu])
