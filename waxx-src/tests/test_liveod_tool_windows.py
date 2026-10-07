"""liveOD toolbar additions: the lab's tool windows (LiveODConfig.tool_windows,
each its own process) and the frame-fixed fit centre in the scalar plot.
The tools here are short `python -c` programs; their logs go to a temp dir.
Offscreen Qt, nothing on the network."""
import logging
import os
import sys

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def launcher_tmp(monkeypatch, tmp_path):
    import waxx.util.live_od.gui.tool_windows as tw
    monkeypatch.setattr(tw.tempfile, "gettempdir", lambda: str(tmp_path))
    return tw


def test_config_has_no_tools_by_default():
    from waxx.util.live_od.config import LiveODConfig
    assert LiveODConfig().tool_windows == []


def test_one_process_per_tool_and_a_button_each(qapp, launcher_tmp):
    from waxx.util.live_od.config import ToolWindow
    msgs = []
    tool = ToolWindow("Sleeper", [sys.executable, "-c", "import time; time.sleep(30)"], "tip")
    launcher = launcher_tmp.ToolLauncher([tool], lambda t, level=logging.INFO: msgs.append(t))
    (button,) = launcher.buttons()
    assert button.text() == "Sleeper" and button.toolTip() == "tip"
    proc = launcher.launch(tool)
    try:
        assert proc is not None and proc.poll() is None
        assert launcher.launch(tool) is proc                  # no second copy
        assert any("already open" in m for m in msgs)
    finally:
        proc.kill()
        proc.wait(10)
    proc2 = launcher.launch(tool)                             # closed: a new one
    try:
        assert proc2 is not proc
    finally:
        proc2.kill()
        proc2.wait(10)


def test_a_tool_that_dies_at_start_is_reported(qapp, launcher_tmp):
    from waxx.util.live_od.config import ToolWindow
    msgs = []
    tool = ToolWindow("Broken", [sys.executable, "-c", "print('no such module'); raise SystemExit(3)"])
    launcher = launcher_tmp.ToolLauncher([tool], lambda t, level=logging.INFO: msgs.append((t, level)))
    proc = launcher.launch(tool)
    proc.wait(30)
    launcher._check_early_exit("Broken", proc)
    text, level = msgs[-1]
    assert "code 3" in text and "no such module" in text and level == logging.ERROR


def test_a_missing_program_is_reported_not_raised(qapp, launcher_tmp):
    from waxx.util.live_od.config import ToolWindow
    msgs = []
    tool = ToolWindow("Ghost", [os.path.join(str(launcher_tmp.tempfile.gettempdir()), "nope.exe")])
    launcher = launcher_tmp.ToolLauncher([tool], lambda t, level=logging.INFO: msgs.append(t))
    assert launcher.launch(tool) is None and "could not start" in msgs[-1]


def test_fit_centre_is_fixed_to_the_camera_frame():
    from waxx.util.live_od.gui.live_scalar_plot_window import LiveScalarPlotWindow, METRICS
    keys = [m[1] for m in METRICS]
    assert "fit_center_frame_x" in keys and "fit_center_frame_y" in keys
    px = 2e-6
    # the same cloud (frame px 130, 75) fitted in two different crops
    a = {"px_calibrated": True, "px_size_m": px, "crop_origin_px": (100, 50),
         "fit_center_x": 30 * px, "fit_center_y": 25 * px, "shot_idx": 0}
    b = dict(a, crop_origin_px=(120, 70), fit_center_x=10 * px, fit_center_y=5 * px, shot_idx=1)
    xs, ys = LiveScalarPlotWindow._series([a, b], "fit_center_frame_x", 1e6, "shot index")
    assert ys == pytest.approx([260., 260.])                 # 130 px x 2 um
    _, ys = LiveScalarPlotWindow._series([a, b], "fit_center_frame_y", 1e6, "shot index")
    assert ys == pytest.approx([150., 150.])
    # no pixel calibration: skipped (NaN), never pixels on a um axis
    c = dict(a, px_calibrated=False, px_size_m=1.0)
    assert LiveScalarPlotWindow._series([c], "fit_center_frame_x", 1e6, "shot index") == ([], [])
