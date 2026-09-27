"""No snapshot from the camera host for 5 s: every camera is shown as unknown
(grey, red border), nothing can be clicked but the ⚙; a snapshot brings it back."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from liveod_qt_helpers import delete_widgets, session_app  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return session_app()


def _snap(host_state="idle", persist=False):
    from waxx.util.live_od.gui.camera_control import LEGACY_STATE
    return {"cameras": {k: {"key": k, "category": "basler_usb", "camera_type": "basler",
                            "host_state": host_state, "state": LEGACY_STATE[host_state],
                            "persist": persist, "n_subs": 2} for k in ("cam_a", "cam_b")}}


def test_unknown_after_five_seconds_without_a_snapshot(app):
    from waxx.util.live_od.gui.camera_control import (CameraControl, STALE_AFTER_S,
                                                      UNKNOWN_BORDER, UNKNOWN_FILL)
    now = [100.0]
    control = CameraControl(["cam_a", "cam_b"], clock=lambda: now[0])
    got = []
    control.action_requested.connect(lambda k, a: got.append(a))
    try:
        control.set_snapshot(_snap(persist=True))
        now[0] += STALE_AFTER_S - 0.1
        control._tick()
        assert not control.is_unknown()
        assert control.main_button.isEnabled()
        now[0] += 0.2
        control._tick()
        assert control.is_unknown()
        look = control.main_button.look()
        assert (look.fill, look.border, look.led_mark, look.hatch) == (
            UNKNOWN_FILL, UNKNOWN_BORDER, "?", False)
        assert "unknown" in control.main_button.toolTip()
        assert not control.main_button.isEnabled() and not control.live_button.isEnabled()
        assert control.cog_button.isEnabled()
        for _action, row in control._rows.values():
            assert row.button.look().border == UNKNOWN_BORDER
        control.main_button.click()
        assert got == []
        # the next snapshot: known again, persist and all
        control.set_snapshot(_snap(persist=True))
        assert not control.is_unknown() and control.main_button.look().hatch
    finally:
        delete_widgets(app, [control])


def test_expecting_snapshots_that_never_come(app):
    """The owner said snapshots would come (host mode); none did."""
    from waxx.util.live_od.gui.camera_control import CameraControl
    now = [0.0]
    control = CameraControl(["cam_a"], clock=lambda: now[0], expect_snapshots=True)
    try:
        now[0] = 4.0
        control._tick()
        assert not control.is_unknown()
        now[0] = 6.0
        control._tick()
        assert control.is_unknown()
    finally:
        delete_widgets(app, [control])


def test_legacy_mode_is_never_unknown(app):
    """Without a host (legacy state words) there are no snapshots to miss."""
    from waxx.util.live_od.gui.camera_control import CameraControl
    now = [0.0]
    control = CameraControl(["cam_a"], clock=lambda: now[0])
    try:
        control.set_state("cam_a", "open")
        now[0] = 1000.0
        control._tick()
        assert not control.is_unknown()
    finally:
        delete_widgets(app, [control])


def test_the_timer_runs_once_a_second(app):
    from waxx.util.live_od.gui.camera_control import CameraControl
    control = CameraControl(["cam_a"])
    try:
        assert control._timer.isActive() and control._timer.interval() == 1000
    finally:
        delete_widgets(app, [control])
