"""The camera control's enable/emit table: what the main button, the ⚙ and the 🎥
do for every camera state, and what each click emits."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from liveod_qt_helpers import delete_widgets, session_app  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return session_app()


def _entry(key, host_state, *, andor, **kw):
    from waxx.util.live_od.gui.camera_control import LEGACY_STATE
    e = {"key": key, "category": "andor_emccd" if andor else "basler_usb",
         "camera_type": "andor" if andor else "basler", "host_state": host_state,
         "state": LEGACY_STATE[host_state], "persist": False, "persisted": {}, "n_subs": 0,
         "run_id": 80713, "holder": {"label": "Camera Viewer", "host": "kong"}}
    e.update(kw)
    return e


# host_state -> (andor action, basler action, main enabled, cog enabled, 🎥 enabled)
TABLE = {
    "closed":         ("connect", "connect", True, True, True),
    "held_elsewhere": ("take", "take", True, True, False),
    "reserved":       ("", "", False, True, False),
    "claiming":       ("", "", False, True, False),
    "opening":        ("", "", False, True, False),
    "returning":      ("", "", False, True, False),
    "idle":           ("close_sdk", "give_back", True, True, True),
    "streaming":      ("close_sdk", "give_back", True, True, True),
    "run_locked":     ("", "", False, True, False),
    "arming":         ("", "", False, True, False),
    "acquiring":      ("", "", False, True, False),
    "draining":       ("", "", False, True, False),
    "error":          ("retry", "retry", True, True, False),
    "faulted":        ("retry", "retry", True, True, False),
    "run_fault":      ("", "", False, True, False),
    "hung":           ("", "", False, True, False),
    "absent":         ("retry", "retry", True, True, False),
    "virtual":        ("", "", False, False, False),
}
CONFIRMED = ("close_sdk", "take")


def test_the_table_covers_every_host_state():
    from waxx.util.live_od.gui.camera_control import HOST_STATES, LEGACY_STATE
    assert set(TABLE) == set(HOST_STATES) == set(LEGACY_STATE)


def test_legacy_words_match_the_camera_host():
    """The control keeps its own copy (no host import in the GUI); it must be the host's."""
    from waxx.util.live_od.camera_host.host import LEGACY_STATE as HOST_LEGACY, RUN_PHASES
    from waxx.util.live_od.gui import camera_control
    assert camera_control.LEGACY_STATE == HOST_LEGACY
    assert tuple(camera_control.RUN_PHASES) == tuple(RUN_PHASES)


def test_every_action_maps_onto_a_host_request():
    from waxx.util.live_od.gui.camera_control import ACTIONS, HOST_REQUEST
    assert set(ACTIONS) == {"connect", "close_sdk", "give_back", "take", "retry"}
    assert set(HOST_REQUEST.values()) <= {"open", "close", "toggle"}      # CameraHost.request


@pytest.mark.parametrize("andor", [True, False], ids=["andor", "basler"])
@pytest.mark.parametrize("host_state", sorted(TABLE))
@pytest.mark.parametrize("answer", [True, False], ids=["confirmed", "cancelled"])
def test_clicks_follow_the_table(app, host_state, andor, answer):
    from waxx.util.live_od.gui.camera_control import CameraControl
    a_action, b_action, main_on, cog_on, live_on = TABLE[host_state]
    action = a_action if andor else b_action
    asked = []
    control = CameraControl(["cam", "other"], confirm=lambda t, x: asked.append((t, x)) or answer)
    got = {"action": [], "settings": [], "live": [], "toggle": []}
    control.action_requested.connect(lambda k, a: got["action"].append((k, a)))
    control.settings_requested.connect(lambda k: got["settings"].append(k))
    control.live_view_requested.connect(lambda k, on: got["live"].append((k, on)))
    control.toggle_requested.connect(lambda k: got["toggle"].append(k))
    try:
        control.set_current("cam")
        control.set_snapshot({"cameras": {"cam": _entry("cam", host_state, andor=andor),
                                          "other": _entry("other", "idle", andor=False)}})
        d = control.decision("cam")
        assert (d.action, d.main_enabled, d.cog_enabled, d.live_enabled) == (
            action, main_on, cog_on, live_on)
        assert control.main_button.isEnabled() == main_on
        assert control.cog_button.isEnabled() == cog_on
        assert control.live_button.isEnabled() == live_on

        control.main_button.click()
        control.cog_button.click()
        control.live_button.click()
        if action in CONFIRMED:
            assert len(asked) == 1 and "cam" in asked[0][1]
            assert got["action"] == ([("cam", action)] if answer else [])
        else:
            assert asked == []
            assert got["action"] == ([("cam", action)] if main_on and action else [])
        assert got["settings"] == (["cam"] if cog_on else [])
        assert got["live"] == ([("cam", True)] if live_on else [])
        assert got["toggle"] == []
        assert not control.live_button.isChecked()      # until the owner says it is open
    finally:
        delete_widgets(app, [control])


def test_run_lock_says_which_run_and_how_to_free_it(app):
    from waxx.util.live_od.gui.camera_control import CameraControl
    control = CameraControl(["andor"])
    try:
        control.set_snapshot({"cameras": {"andor": _entry("andor", "acquiring", andor=True)}})
        assert "Run 80713 uses andor — Abort to free it." in control.main_button.toolTip()
        d = control.decision("andor")
        assert d.cog_run_tab and d.cog_enabled and not d.live_enabled
        control.set_snapshot({"cameras": {"andor": _entry("andor", "run_locked", andor=True,
                                                          run_id=0)}})
        assert "An unsaved run uses andor" in control.main_button.toolTip()
        control.set_snapshot({"cameras": {"andor": _entry("andor", "run_fault", andor=True)}})
        assert "Run 80713 hit a camera fault on andor" in control.main_button.toolTip()
    finally:
        delete_widgets(app, [control])


def test_an_open_live_view_can_be_closed_even_during_a_run(app):
    from waxx.util.live_od.gui.camera_control import CameraControl
    control = CameraControl(["andor"])
    got = []
    control.live_view_requested.connect(lambda k, on: got.append((k, on)))
    try:
        control.set_snapshot({"cameras": {"andor": _entry("andor", "idle", andor=True)}})
        control.set_live_view_open("andor", True)
        control.set_snapshot({"cameras": {"andor": _entry("andor", "acquiring", andor=True)}})
        assert control.live_button.isEnabled() and control.live_button.isChecked()
        control.live_button.click()
        assert got == [("andor", False)]
        control.set_live_view_open("andor", False)
        assert not control.live_button.isEnabled()
    finally:
        delete_widgets(app, [control])


def test_take_names_the_holder(app):
    from waxx.util.live_od.gui.camera_control import CameraControl
    asked = []
    control = CameraControl(["xy"], confirm=lambda t, x: asked.append(x) or False)
    try:
        control.set_snapshot({"cameras": {"xy": _entry("xy", "held_elsewhere", andor=False,
                                                       holder={"label": "spot finder",
                                                               "host": "kong"})}})
        assert "spot finder on kong" in control.main_button.toolTip()
        control.main_button.click()
        assert "spot finder on kong" in asked[0]
    finally:
        delete_widgets(app, [control])


def test_drop_down_rows_emit_for_their_own_camera(app):
    from waxx.util.live_od.gui.camera_control import CameraControl
    control = CameraControl(["andor", "xy"], confirm=lambda t, x: True)
    got = []
    control.action_requested.connect(lambda k, a: got.append(("action", k, a)))
    control.settings_requested.connect(lambda k: got.append(("settings", k)))
    control.live_view_requested.connect(lambda k, on: got.append(("live", k, on)))
    try:
        control.set_current("andor")
        control.set_snapshot({"cameras": {"andor": _entry("andor", "idle", andor=True),
                                          "xy": _entry("xy", "closed", andor=False)}})
        _action, row = control._rows["xy"]
        row.button.click()
        row.cog.click()
        row.live.click()
        assert got == [("action", "xy", "connect"), ("settings", "xy"), ("live", "xy", True)]
    finally:
        delete_widgets(app, [control])


def test_legacy_cameras_toggle_like_the_old_button(app):
    """A camera known only by its state word (no host) behaves as CameraMenuButton."""
    from waxx.util.live_od.gui.camera_control import CameraControl
    from waxx.util.live_od.gui.camera_menu import STATES
    control = CameraControl(["cam_a", "cam_b"])
    got = []
    control.toggle_requested.connect(got.append)
    control.action_requested.connect(lambda k, a: got.append((k, a)))
    try:
        assert control.shown_camera() == "cam_a"
        control.set_state("cam_b", "open")
        assert control.shown_camera() == "cam_b" and control.state("cam_b") == "open"
        assert control.main_button.look().fill == STATES["open"][0]
        control.main_button.click()
        assert got == ["cam_b"]
        assert not control.cog_button.isEnabled() and not control.live_button.isEnabled()
        control.set_camera_enabled("cam_b", False)
        control.main_button.click()
        assert got == ["cam_b"]
        control.set_camera_enabled("cam_b", True)
        empty = CameraControl()
        assert empty.shown_camera() is None and not empty.main_button.isEnabled()
        delete_widgets(app, [empty])
    finally:
        delete_widgets(app, [control])


def test_a_legacy_word_never_overwrites_a_host_snapshot(app):
    """A CameraButton's state_changed wired here by mistake must not erase the
    host's state (the host's snapshots are the authority for its cameras)."""
    from waxx.util.live_od.gui.camera_control import CameraControl
    control = CameraControl(["cam"])
    try:
        control.set_snapshot({"cameras": {"cam": _entry("cam", "acquiring", andor=False)}})
        control.set_state("cam", "closed")
        control.set_states({"cam": "open", "new": "open"})
        assert control.host_state("cam") == "acquiring" and control.state("cam") == "grabbing"
        assert control.state("new") == "open"
    finally:
        delete_widgets(app, [control])


def test_the_shown_camera_follows_the_run_then_the_first_connected(app):
    from waxx.util.live_od.gui.camera_control import CameraControl
    control = CameraControl(["a", "b", "c"])
    try:
        control.set_snapshot({"cameras": {"a": _entry("a", "closed", andor=False),
                                          "b": _entry("b", "held_elsewhere", andor=False),
                                          "c": _entry("c", "streaming", andor=False)}})
        assert control.shown_camera() == "c"
        control.set_current("a")
        assert control.shown_camera() == "a"
        assert control.state("c") == "open" and control.host_state("c") == "streaming"
    finally:
        delete_widgets(app, [control])


def test_subscriber_badge(app):
    from waxx.util.live_od.gui.camera_control import CameraControl, badge_text
    assert [badge_text(n) for n in (0, 1, 12, 99, 100, 999, None, "x")] == [
        "", "1", "12", "99", "99+", "99+", "", ""]
    control = CameraControl(["cam"])
    try:
        control.set_snapshot({"cameras": {"cam": _entry("cam", "streaming", andor=False,
                                                        n_subs=3)}})
        assert control.main_button.look().badge == "3"
        assert "3 subscriber(s)" in control.main_button.toolTip()
    finally:
        delete_widgets(app, [control])
