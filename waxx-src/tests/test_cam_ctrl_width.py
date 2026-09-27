"""CameraControl never changes the width of the liveOD status row: whatever the
cameras do (every host state, Persist, subscriber counts, the hidden-persist and
unknown markers, a live view open), the control's width and the strip's minimum
width stay what they were.  The window must never change width when a run starts."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from liveod_qt_helpers import delete_widgets, session_app  # noqa: E402

KEYS = ("andor", "xy_basler", "a_long_camera_name")


@pytest.fixture(scope="module")
def app():
    return session_app()


def _entry(key, host_state, persist=False, n_subs=0):
    from waxx.util.live_od.gui.camera_control import LEGACY_STATE
    andor = key == "andor"
    return {"key": key, "category": "andor_emccd" if andor else "basler_usb",
            "camera_type": "andor" if andor else "basler", "host_state": host_state,
            "state": LEGACY_STATE[host_state], "persist": persist,
            "persisted": {"gain": 30} if persist else {}, "persist_since": "2026-09-26T14:02:11",
            "n_subs": n_subs, "run_id": 80713, "error": "boom" if host_state == "error" else None,
            "holder": {"label": "Camera Viewer", "host": "kong"}}


def test_width_is_fixed_across_every_state_persist_count_and_marker(app):
    from waxx.util.live_od.gui.camera_control import CameraControl, HOST_STATES, total_width
    from waxx.util.live_od.gui.status_strip import StatusStrip

    now = [0.0]
    control = CameraControl(KEYS, clock=lambda: now[0], expect_snapshots=True)
    strip = StatusStrip()
    strip.add_camera_widget(control)
    strip.show()
    app.processEvents()
    width = control.width()
    hint = control.sizeHint().width(), control.minimumSizeHint().width()
    strip_min = strip.minimumSizeHint().width()
    assert width == total_width(KEYS, control.font())
    seen = set()
    try:
        for current in (None, "andor", "a_long_camera_name"):
            control.set_current(current)
            for host_state in HOST_STATES:
                for persist in (False, True):
                    for n_subs in (0, 12, 999):
                        for hidden_persist in (False, True):
                            snap = {k: _entry(k, host_state, persist and k == "andor", n_subs)
                                    for k in KEYS}
                            if hidden_persist:
                                snap["xy_basler"]["persist"] = True
                            control.set_snapshot({"cameras": snap})
                            control.set_live_view_open("andor", n_subs == 12)
                            app.processEvents()
                            seen.add((control.main_button.look().badge,
                                      control.arrow_button.is_red()))
                            assert control.width() == width, (current, host_state, persist, n_subs)
                            assert (control.sizeHint().width(),
                                    control.minimumSizeHint().width()) == hint
                            assert strip.minimumSizeHint().width() == strip_min
            # the unknown marker (no snapshot for > 5 s)
            now[0] += 60.0
            control._tick()
            app.processEvents()
            assert control.is_unknown()
            assert control.width() == width and strip.minimumSizeHint().width() == strip_min
            control.set_snapshot({"cameras": {k: _entry(k, "idle") for k in KEYS}})
        # a run starting with a long experiment name next to it changes nothing either
        strip.start_run(80713, "a_very_long_experiment_file_name_indeed", "andor",
                        save_data=True, n_shots=10000)
        strip.set_state("running")
        control.set_snapshot({"cameras": {k: _entry(k, "acquiring", True, 999) for k in KEYS}})
        app.processEvents()
        assert control.width() == width and strip.minimumSizeHint().width() == strip_min
        # every badge the loop set actually showed
        assert {"", "12", "99+"} <= {b for b, _red in seen}
        assert {True, False} == {red for _b, red in seen}
    finally:
        strip.hide()
        delete_widgets(app, [strip])


def test_legacy_words_keep_the_width_too(app):
    from waxx.util.live_od.gui.camera_control import CameraControl
    from waxx.util.live_od.gui.camera_menu import STATES
    control = CameraControl(KEYS)
    control.show()
    width = control.width()
    try:
        for word in STATES:
            control.set_states({k: word for k in KEYS})
            control.set_camera_enabled("andor", word != "loading")
            app.processEvents()
            assert control.width() == width
    finally:
        delete_widgets(app, [control])


def test_width_grows_only_when_a_camera_is_added(app):
    """Adding a camera (the host lists one more) is the one thing that may widen it."""
    from waxx.util.live_od.gui.camera_control import CameraControl, total_width
    control = CameraControl(["cam"])
    try:
        narrow = control.width()
        control.set_snapshot({"cameras": {"cam": _entry("cam", "idle"),
                                          "a_much_longer_camera_name": _entry(
                                              "a_much_longer_camera_name", "idle")}})
        assert control.width() > narrow
        assert control.width() == total_width(["cam", "a_much_longer_camera_name"], control.font())
    finally:
        delete_widgets(app, [control])
