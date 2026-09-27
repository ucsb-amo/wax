"""Persist and error must never look alike: Persist is bright red with a white
diagonal hatch and an LED in the camera's state colour; an error is solid dark
red with a "!" LED and no hatch.  Checked on the pixels ``grab()`` renders."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt6.QtGui import QColor  # noqa: E402

from liveod_qt_helpers import delete_widgets, session_app  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return session_app()


def _entry(host_state, persist):
    from waxx.util.live_od.gui.camera_control import LEGACY_STATE
    return {"key": "cam", "category": "basler_usb", "camera_type": "basler",
            "host_state": host_state, "state": LEGACY_STATE[host_state], "persist": persist,
            "persisted": {"gain": 12.0} if persist else {}, "n_subs": 0}


def _near(c: QColor, hex_color: str, tol: int = 40) -> bool:
    ref = QColor(hex_color)
    return (abs(c.red() - ref.red()) <= tol and abs(c.green() - ref.green()) <= tol
            and abs(c.blue() - ref.blue()) <= tol)


def _white(c: QColor) -> bool:
    return c.red() > 225 and c.green() > 225 and c.blue() > 225


def _row(image, y, x0, x1):
    return [image.pixelColor(x, y) for x in range(x0, x1)]


def _render(app, host_state, persist):
    from waxx.util.live_od.gui.camera_control import CameraControl
    control = CameraControl(["cam"])
    control.set_snapshot({"cameras": {"cam": _entry(host_state, persist)}})
    control.show()
    app.processEvents()
    image = control.main_button.grab().toImage()
    return control, image


def _body_band(image):
    """Pixels of a band inside the body, clear of the rounded corners, the LED,
    the name and the badge: the second pixel row (text never reaches it)."""
    from waxx.util.live_od.gui.camera_control import LED, PAD
    return _row(image, 1, PAD + LED + 4, image.width() - 12)


def _led_center(image):
    from waxx.util.live_od.gui.camera_control import LED, PAD
    return image.pixelColor(PAD + LED // 2, image.height() // 2)


def test_persist_is_red_with_a_white_hatch_and_a_state_led(app):
    from waxx.util.live_od.gui.camera_control import HOST_STATES, PERSIST_FILL
    control, image = _render(app, "idle", True)
    try:
        band = _body_band(image)
        red = sum(_near(c, PERSIST_FILL) for c in band)
        white = sum(_white(c) for c in band)
        assert red >= len(band) * 0.3, f"{red}/{len(band)} persist-red pixels"
        assert white >= len(band) * 0.1, f"{white}/{len(band)} hatch pixels"
        # the hatch repeats: white runs start more than once along the band
        starts = sum(1 for a, b in zip(band, band[1:]) if not _white(a) and _white(b))
        assert starts >= 3
        assert _near(_led_center(image), HOST_STATES["idle"][0], tol=30)   # green LED: idle
    finally:
        delete_widgets(app, [control])


def test_persist_led_follows_the_state(app):
    from waxx.util.live_od.gui.camera_control import HOST_STATES
    control, image = _render(app, "acquiring", True)
    try:
        assert _near(_led_center(image), HOST_STATES["acquiring"][0], tol=30)   # blue: a run
        assert sum(_white(c) for c in _body_band(image)) > 0
    finally:
        delete_widgets(app, [control])


def test_error_is_solid_dark_red_without_a_hatch(app):
    from waxx.util.live_od.gui.camera_control import ERROR_FILL, LED, PAD, PERSIST_FILL
    control, image = _render(app, "error", False)
    try:
        band = _body_band(image)
        assert not any(_white(c) for c in band), "an error must never be hatched"
        assert all(_near(c, ERROR_FILL, tol=20) for c in band)
        assert not any(_near(c, PERSIST_FILL, tol=20) for c in band)
        # the "!" LED: a white disc (its left part, clear of the mark)
        led = image.pixelColor(PAD + 2, image.height() // 2)
        assert _white(led)
        assert control.main_button.look().led_mark == "!"
    finally:
        delete_widgets(app, [control])


def test_error_with_persist_keeps_both_signs(app):
    """Persist on a failed camera: hatched (Persist) and a "!" LED (the error)."""
    from waxx.util.live_od.gui.camera_control import LED, PAD
    control, image = _render(app, "faulted", True)
    try:
        assert sum(_white(c) for c in _body_band(image)) > 0
        look = control.main_button.look()
        assert look.hatch and look.led_mark == "!"
        assert _white(image.pixelColor(PAD + 2, image.height() // 2))
    finally:
        delete_widgets(app, [control])


def test_idle_without_persist_is_plain(app):
    from waxx.util.live_od.gui.camera_control import HOST_STATES
    control, image = _render(app, "idle", False)
    try:
        band = _body_band(image)
        assert not any(_white(c) for c in band)
        assert all(_near(c, HOST_STATES["idle"][0], tol=20) for c in band)
    finally:
        delete_widgets(app, [control])


def test_unknown_is_grey_with_a_red_border(app):
    from waxx.util.live_od.gui.camera_control import (LED, PAD, CameraControl, UNKNOWN_BORDER,
                                                      UNKNOWN_FILL)
    now = [0.0]
    control = CameraControl(["cam"], clock=lambda: now[0])
    try:
        control.set_snapshot({"cameras": {"cam": _entry("idle", True)}})
        now[0] = 6.0
        control._tick()
        control.show()
        app.processEvents()
        image = control.main_button.grab().toImage()
        mid = image.width() // 2
        assert _near(image.pixelColor(mid, 1), UNKNOWN_BORDER, tol=60)      # the border
        # the fill, between the LED and the name
        assert _near(image.pixelColor(PAD + LED + 3, image.height() // 2), UNKNOWN_FILL, tol=25)
        assert not any(_white(c) for c in _body_band(image)[:-8])           # no hatch
    finally:
        delete_widgets(app, [control])


def test_legacy_menu_button_hatches_a_persisted_camera(app):
    """The remote viewer's CameraMenuButton shows Persist the same way (sampled on
    its bottom rows: its name fills the rest of this short button)."""
    from waxx.util.live_od.gui.camera_menu import ARROW_WIDTH, CameraMenuButton, PERSIST_COLOR
    menu = CameraMenuButton(["cam_a", "cam_b"])
    try:
        menu.set_states({"cam_a": "open", "cam_b": "closed"})
        menu.set_persist({"cam_a": True})
        menu.show()
        app.processEvents()
        image = menu.grab().toImage()
        band = _row(image, image.height() - 2, 26, image.width() - ARROW_WIDTH - 10)
        assert sum(_white(c) for c in band) > 0
        assert sum(_near(c, PERSIST_COLOR) for c in band) >= len(band) * 0.3
        assert "PERSIST ON" in menu.toolTip()
        menu.set_persist({"cam_a": False})
        app.processEvents()
        image = menu.grab().toImage()
        band = _row(image, image.height() - 2, 26, image.width() - ARROW_WIDTH - 10)
        assert not any(_white(c) for c in band)
    finally:
        delete_widgets(app, [menu])
