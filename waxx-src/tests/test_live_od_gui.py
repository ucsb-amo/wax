"""liveOD GUI pieces that need no cameras, ARTIQ or data drive: the viewer's crop
region / profile overlays / shot history / ROI file, the log panel and its Qt log
handler, the status strip, and the run states the server reports.

Nothing here touches QSettings (the viewer only stores state in a QSettings it is
handed) or the real ``~/.waxx`` (the ROI file test redirects it to tmp_path).
"""

import logging
import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make_shot(cx=256.0, cy=200.0, n=512, peak_light=3000):
    y, x = np.mgrid[0:n, 0:n]
    od = 2.0 * np.exp(-((x - cx) ** 2 / (2 * 30.0 ** 2) + (y - cy) ** 2 / (2 * 20.0 ** 2)))
    light = np.full((n, n), peak_light, np.uint16)
    atoms = (light * np.exp(-od)).astype(np.uint16)
    dark = np.full((n, n), 100, np.uint16)
    return atoms, light, dark, od, od.sum(0), od.sum(1)


_keep_alive = []    # pyqtgraph scenes torn down by the garbage collector mid-session crash Qt


@pytest.fixture
def viewer(app):
    from waxx.util.live_od.gui.viewer import LiveODViewer
    v = LiveODViewer()
    _keep_alive.append(v)
    v.resize(740, 800)
    v.show()
    app.processEvents()
    yield v
    v.hide()


# ----------------------------------------------------------------------
# Viewer
# ----------------------------------------------------------------------

def test_profiles_are_overlays_on_the_od_plot(viewer, app):
    assert not hasattr(viewer, "sumodx_panel") and not hasattr(viewer, "sumody_panel")
    viewer.handle_plot_data(make_shot(cx=300, cy=180))
    app.processEvents()
    cols, heights = viewer._sumx_curve.getData()
    widths, rows = viewer._sumy_curve.getData()
    assert viewer._sumx_curve.isVisible() and viewer._sumy_curve.isVisible()
    assert cols[np.argmax(heights)] == pytest.approx(300.5, abs=1)
    assert rows[np.argmax(widths)] == pytest.approx(180.5, abs=1)
    # drawn from the image's bottom edge, a fifth of the visible image tall
    assert heights.min() == pytest.approx(0.0, abs=1e-6)
    assert heights.max() == pytest.approx(0.2 * 512, rel=0.02)

    viewer.profiles_checkbox.setChecked(False)
    assert not viewer._sumx_curve.isVisible()


def test_crop_region_is_the_roi_and_else_the_view(viewer, app):
    viewer.handle_plot_data(make_shot())
    viewer.od_plot.getViewBox().setRange(xRange=(100, 300), yRange=(150, 250), padding=0)
    app.processEvents()
    x0, x1, y0, y1 = viewer.get_crop_bounds((512, 512))
    # the plot keeps the image's aspect ratio, so one axis may show more than asked
    assert x0 <= 100 and x1 >= 300 and y0 <= 150 and y1 >= 250
    assert (x1 - x0) < 512 or (y1 - y0) < 512

    viewer.roi_button.setChecked(True)
    viewer._roi_item.setPos([200, 150]); viewer._roi_item.setSize([100, 80])
    assert viewer.get_crop_bounds((512, 512)) == (200, 300, 150, 230)
    # the profiles follow the ROI, as the Analyzer's numbers do
    cols, _ = viewer._sumx_curve.getData()
    assert cols[0] == 200.5 and cols[-1] == 299.5

    viewer.roi_button.setChecked(False)
    assert viewer.get_od_roi_rect() is None


def test_analyzer_crops_with_the_viewers_region(viewer, app):
    from waxx.util.live_od.gui.analyzer import Analyzer
    from queue import Queue
    viewer.handle_plot_data(make_shot())
    viewer.roi_button.setChecked(True)
    viewer._roi_item.setPos([10, 20]); viewer._roi_item.setSize([30, 40])
    analyzer = Analyzer(Queue(), viewer)
    cropped, x_slice, y_slice = analyzer.crop_od_to_view_range(np.zeros((512, 512)))
    assert cropped.shape == (40, 30)
    assert (x_slice.start, y_slice.start) == (10, 20)


def test_fit_overlay_only_for_the_region_it_was_fitted_on(viewer, app):
    viewer.handle_plot_data(make_shot())
    viewer.roi_button.setChecked(True)
    viewer._roi_item.setPos([128, 128]); viewer._roi_item.setSize([256, 256])
    scalars = {'atom_number': 1.2e5, 'atom_cross_section_m2': 1e-13, 'px_calibrated': True,
               'px_size_m': 2e-6, 'fit_sd_x': 60e-6, 'fit_sd_y': 40e-6,
               'fit_amp_x': 50.0, 'fit_amp_y': 70.0,
               'fit_center_x': 2e-6 * 128, 'fit_center_y': 2e-6 * 72,
               'fit_offset_x': 0.0, 'fit_offset_y': 0.0,
               'crop_origin_px': (128, 128), 'crop_shape_px': (256, 256)}
    viewer.on_shot_scalars(scalars)
    assert viewer._fitx_curve.isVisible()
    assert viewer.um_checkbox.isEnabled()
    # pixels until the µm button is pressed, then everything in µm at the atoms
    text = viewer.readout_label.text()
    assert "1.2e5" in text and "30.0 × 20.0 px" in text and "256, 200 px" in text
    viewer.um_checkbox.setChecked(True)
    text = viewer.readout_label.text()
    assert "60.0 × 40.0 µm" in text and "512, 400 µm" in text
    assert viewer.od_plot.getAxis('bottom').scale == pytest.approx(2.0)
    assert "µm" in viewer.roi_button.toolTip() and "512×512 µm" in viewer.roi_button.toolTip()
    viewer.um_checkbox.setChecked(False)
    assert "256, 200 px" in viewer.readout_label.text()
    assert viewer.od_plot.getAxis('bottom').scale == pytest.approx(1.0)

    viewer._roi_item.setSize([200, 256])        # the fit no longer describes this region
    assert not viewer._fitx_curve.isVisible()


def test_fit_belongs_to_its_shot(viewer, app):
    """A new shot shows no fit until its own is done, and stepping through the
    history brings each shot's fit and readout back with it."""
    from PyQt6.QtCore import Qt
    from waxx.util.live_od.gui.plotter import ShotPlotData
    viewer.on_new_run()

    def scalars(i):
        x0, x1, y0, y1 = viewer.get_crop_bounds((512, 512))
        return {'shot_idx': i, 'atom_number': 1000.0 + i, 'px_calibrated': False,
                'px_size_m': 1.0, 'fit_sd_x': 30.0, 'fit_sd_y': 20.0,
                'fit_amp_x': 50.0, 'fit_amp_y': 70.0,
                'fit_center_x': 256.0 - x0, 'fit_center_y': 200.0 - y0,
                'fit_offset_x': 0.0, 'fit_offset_y': 0.0,
                'crop_origin_px': (x0, y0), 'crop_shape_px': (x1 - x0, y1 - y0)}

    def shown():
        return None if viewer._fit is None else viewer._fit['atom_number'] - 1000

    viewer.handle_plot_data(ShotPlotData(make_shot(), 0))
    viewer.on_shot_scalars(scalars(0))
    assert shown() == 0 and viewer._fitx_curve.isVisible()
    assert viewer._fitx_curve.opts['pen'].style() == Qt.PenStyle.SolidLine
    viewer.handle_plot_data(ShotPlotData(make_shot(), 1))
    assert shown() is None
    assert not viewer._fitx_curve.isVisible() and not viewer.readout_label.isVisible()
    viewer.on_shot_scalars(scalars(1))
    assert shown() == 1
    viewer.on_shot_scalars(scalars(2))                   # scalars ahead of their image
    assert shown() == 1
    viewer.handle_plot_data(ShotPlotData(make_shot(), 2))
    assert shown() == 2
    viewer.on_shot_scalars(scalars(3))                   # shot 3's image dropped by the plotter
    viewer.handle_plot_data(ShotPlotData(make_shot(), 4))
    assert shown() is None and not viewer._early_scalars
    viewer.show_previous_shot()
    assert shown() == 2 and "1e3" in viewer.readout_label.text()
    viewer.show_previous_shot()
    viewer.show_previous_shot()
    assert shown() == 0 and viewer._fitx_curve.isVisible()
    viewer.pause_button.click()
    assert shown() is None                              # shot 4 is still unfitted

    # a server that predates shot_idx on the images: scalars go with the latest shot
    viewer.on_new_run()
    viewer.handle_plot_data(make_shot())
    viewer.on_shot_scalars(scalars(0))
    viewer.handle_plot_data(make_shot())
    assert shown() is None
    viewer.on_shot_scalars({k: v for k, v in scalars(1).items() if k != 'shot_idx'})
    assert shown() == 1
    viewer.show_previous_shot()
    assert shown() == 0


def test_um_button_needs_the_pixel_size(viewer, app):
    assert not viewer.um_checkbox.isEnabled()
    viewer.set_pixel_size_m(16e-6 / 8.0)                 # pixel size / magnification
    assert viewer.um_checkbox.isEnabled()
    viewer.um_checkbox.setChecked(True)
    assert viewer._length_unit() == ("µm", pytest.approx(2.0))
    viewer.set_pixel_size_m(None)                        # a camera with no calibration
    assert not viewer.um_checkbox.isEnabled()
    assert viewer._length_unit() == ("px", 1.0)


def test_units_plate_is_the_um_button(viewer, app):
    """The px/µm plate in the image's corner switches the units when clicked."""
    assert not viewer.units_label.testAttribute(
        Qt.WidgetAttribute.WA_TransparentForMouseEvents)     # the other plates are
    viewer.units_label.clicked.emit()                        # no pixel size yet: nothing happens
    assert viewer._length_unit() == ("px", 1.0)
    viewer.set_pixel_size_m(16e-6 / 8.0)
    viewer.units_label.clicked.emit()
    assert viewer.um_checkbox.isChecked() and viewer._length_unit()[0] == "µm"
    assert viewer.units_label.text() == "µm"
    viewer.units_label.clicked.emit()
    assert not viewer.um_checkbox.isChecked() and viewer.units_label.text() == "px"


def test_right_clicking_roi_runs_auto_roi(viewer, app):
    calls = []
    viewer.auto_roi = lambda *a, **k: calls.append((a, k))
    assert (viewer.roi_button.contextMenuPolicy()
            == Qt.ContextMenuPolicy.CustomContextMenu)       # no stock menu in the way
    viewer.roi_button.customContextMenuRequested.emit(QPoint(3, 3))
    assert calls == [((), {})]
    assert not viewer.roi_button.isChecked()                 # the right click did not toggle it


def test_overlays_are_readable_plates_in_the_image_corners(viewer, app):
    viewer.handle_plot_data(make_shot())
    assert viewer.readout_label.isHidden() and viewer.cursor_label.isHidden()    # nothing to say yet
    viewer.on_shot_scalars({'atom_number': 5.0e4, 'px_calibrated': False, 'px_size_m': 1.0,
                            'fit_sd_x': 30.0, 'fit_sd_y': 20.0, 'fit_center_x': 10.0,
                            'fit_center_y': 12.0, 'crop_origin_px': (0, 0), 'crop_shape_px': (512, 512)})
    app.processEvents()
    label = viewer.readout_label
    assert not label.isHidden() and "ΣOD" in label.text()
    assert "font-weight: bold" in label.styleSheet() and "rgba(0, 0, 0, 170)" in label.styleSheet()
    image_area = viewer.od_plot.mapFromScene(
        viewer.od_plot.getViewBox().sceneBoundingRect()).boundingRect()
    # anchored inside the top-right corner (the plate is a fixed width, and the
    # offscreen test font is far wider than a real one, so not "fits inside")
    assert label.geometry().right() <= image_area.right()
    assert image_area.top() <= label.geometry().top() and label.geometry().bottom() <= image_area.bottom()
    assert label.geometry().right() > image_area.center().x()       # top right
    assert label.geometry().top() < image_area.center().y()


def test_xvars_are_a_plate_top_left_in_units_fixed_for_the_run(viewer, app):
    viewer.on_new_run()
    viewer.set_xvar_ranges({"t_tof": [5e-6, 2e-3], "f_raman": [1.0e6, 1.5e6]})
    assert viewer.xvar_label.isHidden()
    viewer.set_shot_xvars(0, {"t_tof": 1.2e-3, "f_raman": 1.25e6, "n_thing": 3})
    viewer.handle_plot_data(_tagged(make_shot(), 0))
    app.processEvents()
    text = viewer.xvar_label.text()
    assert "1.2 ms" in text and "1.25 MHz" in text and "n_thing" in text
    assert "rgba(0, 0, 0, 170)" in viewer.xvar_label.styleSheet()
    image_area = viewer.od_plot.mapFromScene(
        viewer.od_plot.getViewBox().sceneBoundingRect()).boundingRect()
    geometry = viewer.xvar_label.geometry()
    assert geometry.left() >= image_area.left() and geometry.top() >= image_area.top()
    assert geometry.left() < image_area.center().x() and geometry.top() < image_area.center().y()

    # a small value stays in the run's unit: no switching to µs mid-scan
    viewer.set_shot_xvars(1, {"t_tof": 5e-6, "f_raman": 1.0e6, "n_thing": 4})
    viewer.handle_plot_data(_tagged(make_shot(), 1))
    assert "0.005 ms" in viewer.xvar_label.text()
    # stepping back through the history brings back that shot's xvars
    viewer.show_previous_shot()
    assert "1.2 ms" in viewer.xvar_label.text()
    viewer.show_next_shot()

    # an older experiment process sends no ranges: the first value decides, for the run
    viewer.on_new_run()
    assert viewer.xvar_label.isHidden()
    viewer.set_shot_xvars(0, {"t_tof": 20e-6})
    assert "20 µs" in viewer.xvar_label.text()       # no image yet: the latest is shown
    viewer.set_shot_xvars(1, {"t_tof": 2e-3})
    assert "2000 µs" in viewer.xvar_label.text()


def _tagged(shot, shot_idx):
    from waxx.util.live_od.gui.plotter import ShotPlotData
    return ShotPlotData(shot, shot_idx)


def test_toolbar_width_does_not_depend_on_the_run(viewer, app):
    before = viewer.minimumSizeHint().width()
    for i in range(12):
        viewer.handle_plot_data(make_shot())
    viewer.update_image_count(1234, 5678)
    viewer.show_previous_shot()                          # "11/12", paused
    viewer.roi_button.setChecked(True)
    app.processEvents()
    assert viewer.minimumSizeHint().width() == before


def test_saturation_is_flagged_on_the_raw_frame(viewer, app):
    atoms, light, dark, od, sx, sy = make_shot()
    light[:4, :4] = 4095
    viewer.handle_plot_data((atoms, light, dark, od, sx, sy))
    assert "SATURATED" in viewer._raw_labels['light'].text
    assert "SATURATED" not in viewer._raw_labels['atoms'].text
    assert "max 4095" in viewer._raw_labels['light'].text


def test_history_pause_and_step(viewer, app):
    for i in range(6):
        viewer.handle_plot_data(make_shot(cx=200 + 10 * i))
    assert viewer.history_label.toolTip() == "shot 6 · live"
    viewer.show_previous_shot()
    viewer.show_previous_shot()
    # stepping back is not live: the pause button says so, and is the way back
    assert viewer.pause_button.isChecked() and viewer.pause_button.text() == "Live"
    assert "background-color" in viewer.pause_button.styleSheet()
    assert viewer.history_label.text() == "4/6"
    shown = viewer._displayed_od
    viewer.handle_plot_data(make_shot(cx=400))          # the run carries on underneath
    assert viewer._displayed_od is shown
    assert viewer.history_label.text() == "4/7"
    viewer.pause_button.click()                          # "Live": back to the latest shot
    assert viewer.history_label.text() == "7"
    assert not viewer.pause_button.isChecked() and viewer.pause_button.text() != "Live"
    assert viewer.pause_button.styleSheet() == ""
    assert viewer._displayed_od is viewer._history[-1][0][3]

    viewer.avg_spinner.setValue(3)
    expected = np.mean([viewer._history[i][0][3] for i in (-3, -2, -1)], axis=0)
    assert np.allclose(viewer._displayed_od, expected)

    viewer.on_new_run()
    assert len(viewer._history) == 0 and viewer.history_label.toolTip() == "no shots yet"


def test_history_is_capped_by_memory(viewer, app):
    from waxx.util.live_od.gui import viewer as viewer_module
    big = make_shot(n=1024)
    nbytes = sum(a.nbytes for a in big)
    viewer.handle_plot_data(big)
    assert viewer._history.maxlen == max(5, min(50, int(viewer_module.HISTORY_MAX_BYTES // nbytes)))


def test_od_levels_spinner_and_colorbar_agree(viewer, app):
    viewer.handle_plot_data(make_shot())
    viewer.od_max_spinner.setValue(1.5)
    assert tuple(viewer.od_colorbar.levels()) == (0.0, 1.5)
    assert tuple(viewer.od_img_item.levels) == (0.0, 1.5)
    viewer.od_colorbar.setLevels((0.2, 3.0))             # what dragging the bar does ...
    viewer.od_colorbar.sigLevelsChanged.emit(viewer.od_colorbar)     # ... and then reports
    assert viewer.od_min_spinner.value() == pytest.approx(0.2)
    assert viewer.od_max_spinner.value() == pytest.approx(3.0)
    viewer.auto_od_levels()
    assert 1.7 <= viewer.od_max_spinner.value() <= 2.0   # just under the cloud's peak OD of 2
    assert viewer.od_min_spinner.value() == 0.0


def test_holding_a_colorbar_handle_holds_the_level(viewer, app, monkeypatch):
    """The bar's handles are a rubber band: the level is where the drag started plus
    a function of the handle's displacement. Writing the level back into the bar
    mid-drag resets that start point, and the level then climbs with every mouse move
    (2.5 -> 35 over 40 moves) -- the 'far too sensitive' drag."""
    saves = []
    monkeypatch.setattr(viewer, "_save_od_levels", lambda: saves.append(1))
    viewer.handle_plot_data(make_shot())
    bar = viewer.od_colorbar
    top = []
    for i in range(40):                                  # held ~30 px up, hand jittering
        bar.region.lines[1].setValue(191 + 16 + (i % 2))     # moves, no release
        top.append(bar.levels()[1])
    assert max(top) - min(top) <= 0.1 + 1e-9             # one rounding step of jitter
    assert top[-1] == pytest.approx(2.7, abs=0.11)
    assert viewer.od_max_spinner.value() == pytest.approx(top[-1])
    assert tuple(viewer.od_img_item.levels) == pytest.approx(bar.levels())
    assert saves == []                                   # nothing stored mid-drag

    bar.region.sigRegionChangeFinished.emit(bar.region)  # release
    assert saves == [1]
    assert viewer._od_max == pytest.approx(bar.levels()[1])
    # and the next drag starts from the released level, not from 2.5
    bar.region.lines[1].setValue(191 + 16)
    assert bar.levels()[1] > top[-1]


def test_reset_zoom_uses_the_image_shape(viewer, app):
    viewer.handle_plot_data(make_shot(n=300))
    viewer.od_plot.getViewBox().setRange(xRange=(10, 20), yRange=(10, 20), padding=0)
    viewer.reset_zoom()
    app.processEvents()
    (x0, x1), (y0, y1) = viewer.get_od_view_range()
    assert x0 <= 0 and x1 >= 300 and y0 <= 0 and y1 >= 300
    assert min(x1 - x0, y1 - y0) == pytest.approx(300, rel=0.02)     # not the old hardcoded 512


def test_roi_switched_off_is_remembered_per_camera(viewer, app, tmp_path, monkeypatch):
    from waxx.util.live_od.gui.viewer import LiveODViewer
    monkeypatch.setattr(LiveODViewer, "_STATE_DIR", str(tmp_path))
    viewer.set_camera_key("cam_a")
    viewer.handle_plot_data(make_shot())
    viewer.roi_button.setChecked(True)
    viewer._roi_item.setPos([40, 50]); viewer._roi_item.setSize([60, 70])
    viewer._on_roi_changed()
    viewer.set_camera_key("cam_a")                       # next run: comes back
    assert viewer.get_od_roi_rect() == (40.0, 50.0, 100.0, 120.0)
    assert viewer.roi_button.isChecked()

    viewer.roi_button.setChecked(False)
    viewer.set_camera_key("cam_a")                       # stays off ...
    assert viewer.get_od_roi_rect() is None and not viewer.roi_button.isChecked()
    viewer.roi_button.setChecked(True)                   # ... and returns where it was
    assert viewer.get_od_roi_rect() == (40.0, 50.0, 100.0, 120.0)

    # a file from before the ROI could be switched off has no "enabled"
    (tmp_path / "live_od_cam_b_rect.json").write_text('{"rect": [1, 2, 11, 22]}')
    viewer.set_camera_key("cam_b")
    assert viewer.get_od_roi_rect() == (1.0, 2.0, 11.0, 22.0)


def test_toolbar_hosts_the_windows_button(viewer, app):
    from PyQt6.QtWidgets import QPushButton
    button = QPushButton("Adjust")
    viewer.add_window_button(button)
    assert viewer._windows_layout.indexOf(button) > viewer._windows_layout.indexOf(viewer.live_plot_button)
    assert not hasattr(viewer, "fk_tof_button")          # not in use for now
    assert hasattr(viewer, "fk_tof_requested")           # the windows still connect to it


def test_od_scale_limits(viewer, app):
    assert viewer._od_limits == (-2.0, 10.0)
    viewer.handle_plot_data(make_shot())
    bar = viewer.od_colorbar
    for _ in range(30):                                  # drag the top handle up, again and again
        bar.region.lines[1].setValue(255)
        bar.region.sigRegionChangeFinished.emit(bar.region)
    assert bar.levels()[1] == pytest.approx(10.0)
    viewer.set_od_levels(-50, 50)
    assert (viewer._od_min, viewer._od_max) == (-2.0, 10.0)

    # levels remembered from before there were limits, entirely outside them
    viewer.set_od_levels(-13.0, -3.0)
    assert (viewer._od_min, viewer._od_max) == (0.0, 2.5)
    assert tuple(bar.levels()) == (0.0, 2.5)

    viewer.apply_settings(od_limits=(0.0, 4.0))
    viewer.set_od_levels(-1.0, 6.0)
    assert (viewer._od_min, viewer._od_max) == (0.0, 4.0)
    assert viewer.od_max_spinner.maximum() == 4.0 and bar.hi_lim == 4.0


def test_dark_mode_switches_and_restores(viewer, app):
    from waxx.util.live_od.gui import theme
    from PyQt6.QtGui import QPalette
    light_window = app.palette().color(QPalette.ColorRole.Window).name()
    viewer.output_window.append_record(logging.WARNING, "data drive slow")
    try:
        viewer.apply_settings(dark=True)
        assert theme.is_dark()
        assert app.palette().color(QPalette.ColorRole.Window).lightness() < 80
        assert theme.color("warning") in viewer.output_window._text.document().toHtml()
    finally:
        viewer.apply_settings(dark=False)
    assert not theme.is_dark()
    assert app.palette().color(QPalette.ColorRole.Window).name() == light_window


# ----------------------------------------------------------------------
# Log panel and handler
# ----------------------------------------------------------------------

def test_log_panel_levels_filter_and_banner(app):
    from waxx.util.live_od.gui.log_panel import LogPanel
    panel = LogPanel()
    panel.append_record(logging.INFO, "camera ready")
    panel.append_record(logging.WARNING, "data drive slow")
    assert panel.banner.isHidden()
    panel.append_record(logging.ERROR, "save failed\n    how_to_finish_it(r'x')")
    assert not panel.banner.isHidden()
    assert "how_to_finish_it" in panel.banner._label.text()      # the hint stays in view

    assert panel._text.toPlainText().count("\n") >= 3
    panel._filter.setCurrentIndex(3)                     # errors only
    shown = panel._text.toPlainText()
    assert "save failed" in shown and "camera ready" not in shown
    panel._filter.setCurrentIndex(0)
    assert "camera ready" in panel._text.toPlainText()

    panel.banner.dismiss()
    assert panel.banner.isHidden()
    panel.clear()
    assert panel.toPlainText() == ""


def test_log_panel_is_a_drop_in_for_the_old_text_box(app):
    from waxx.util.live_od.gui.log_panel import LogPanel
    panel = LogPanel()
    panel.setReadOnly(True)
    panel.appendPlainText("Failed to open camera andor")
    panel.appendPlainText("Camera grabbing... Expecting 30 images.")
    levels = [levelno for levelno, _, _ in panel._records]
    assert levels == [logging.ERROR, logging.INFO]

    panel.set_collapsed(True)
    assert panel._text.isHidden() and panel.maximumHeight() < 200
    panel.set_collapsed(False)
    assert not panel._text.isHidden()


def test_log_header_has_only_the_level_filter_and_copy_clear_float_over_the_text(app):
    from waxx.util.live_od.gui.log_panel import LogPanel
    from PyQt6.QtWidgets import QAbstractButton
    panel = LogPanel()
    panel.resize(500, 200)
    panel.show()
    app.processEvents()
    header_buttons = [b for b in panel._header.findChildren(QAbstractButton) if b is not panel._toggle]
    assert header_buttons == []
    for button in (panel._copy_button, panel._clear_button):
        assert button.parent() is panel._text and button.isVisible()
    viewport = panel._text.viewport().geometry()
    assert panel._clear_button.geometry().right() <= viewport.right()
    assert panel._clear_button.geometry().top() < viewport.center().y()          # top right
    assert panel._copy_button.geometry().right() < panel._clear_button.geometry().left()
    panel.append_record(logging.INFO, "camera ready")
    panel._clear_button.click()
    assert panel.toPlainText() == ""
    panel.set_collapsed(True)
    assert not panel._copy_button.isVisible() and not panel._clear_button.isVisible()
    panel.set_collapsed(False)
    app.processEvents()
    assert panel._copy_button.isVisible()
    panel.hide()


def test_splitter_cannot_shrink_the_log_to_nothing(viewer):
    assert not viewer.top_splitter.isCollapsible(0)
    viewer.top_splitter.setSizes([0, 1000])
    assert viewer.top_splitter.sizes()[0] > 0
    # collapsed by its arrow, the log still keeps its header row
    viewer.output_window.set_collapsed(True)
    viewer.top_splitter.setSizes([0, 1000])
    assert viewer.top_splitter.sizes()[0] > 0
    viewer.output_window.set_collapsed(False)


def test_qt_log_handler_delivers_from_a_worker_thread(app):
    from waxx.util.live_od.log import QtLogHandler
    logger = logging.getLogger("waxx.live_od.test_handler")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = QtLogHandler(logging.INFO)
    logger.addHandler(handler)
    received = []
    handler.record_signal.connect(lambda levelno, text, created: received.append(
        (levelno, text, threading.current_thread() is threading.main_thread())))
    try:
        def work():
            logger.debug("per-shot noise")               # below the GUI's level
            try:
                raise ValueError("boom")
            except ValueError:
                logger.exception("SaveWorker: write error")
        worker = threading.Thread(target=work)
        worker.start(); worker.join()
        app.processEvents()
    finally:
        logger.removeHandler(handler)
    assert len(received) == 1
    levelno, text, on_main_thread = received[0]
    assert levelno == logging.ERROR and on_main_thread
    assert "SaveWorker: write error" in text and "ValueError: boom" in text


def test_setup_logging_writes_the_file_and_is_idempotent(app, tmp_path, monkeypatch):
    from waxx.util.live_od import log
    monkeypatch.setattr(log, "_qt_handler", None)
    logger = log.get_logger()
    before = list(logger.handlers)
    try:
        handler = log.setup_logging(log_dir=str(tmp_path))
        assert log.setup_logging(log_dir=str(tmp_path)) is handler
        log.get_logger("server").debug("shot 3/120")     # DEBUG: file yes, GUI no
        for h in logger.handlers:
            h.flush()
        assert "shot 3/120" in (tmp_path / "live_od.log").read_text(encoding="utf-8")
    finally:
        for h in [h for h in logger.handlers if h not in before]:
            logger.removeHandler(h)
            h.close()


# ----------------------------------------------------------------------
# Status strip
# ----------------------------------------------------------------------

def test_status_strip_run_and_stall(app, monkeypatch):
    from waxx.util.live_od.gui import status_strip
    strip = status_strip.StatusStrip()
    titles = []
    strip.title_changed.connect(titles.append)
    strip.set_next_run_id(80545)
    assert strip.run_label.text() == "Next run: 80545"

    strip.start_run(80545, "hf_tweezer_bec", "andor", save_data=False, n_shots=120)
    strip.set_state("waiting_camera")
    # save_data=False: no run ID to show, and its place says so
    assert strip.run_label.text() == "NOT SAVING · hf_tweezer_bec · andor"
    assert "color" in strip.run_label.styleSheet()
    assert not hasattr(strip, "adjust_badge")            # the Adjust button already shows the count
    assert strip.pill.text() == "Camera…" and "camera" in strip.pill.toolTip()

    strip.set_state("running")
    for shot in (1, 2, 3):
        strip.set_progress(shot, 120)
        strip.set_timing(8.0, "14:32")
    assert strip.progress.value() == 3 and strip.progress.maximum() == 120
    assert "8.0s" in strip.timing_label.text() and "14:32" in strip.timing_label.text()
    assert "font-weight:bold" in strip.timing_label.text()       # the numbers stand out
    assert strip.timing_label.toolTip().startswith("Elapsed")
    assert not hasattr(strip, "xvar_label")      # the xvars are a plate on the OD image
    assert titles[-1] == "Run 80545 — Running — 3/120"
    strip.start_run(80546, "hf_tweezer_bec", "andor", save_data=True, n_shots=120)
    assert strip.run_label.text() == "80546 · hf_tweezer_bec · andor"
    assert strip.run_label.styleSheet() == ""
    strip.set_state("running")
    for shot in (1, 2, 3):
        strip.set_progress(shot, 120)
        strip.set_timing(8.0, "14:32")

    now = status_strip.time.time()
    monkeypatch.setattr(status_strip.time, "time", lambda: now + 60)     # > 3 x 8 s
    strip._tick()
    assert strip.state() == "stalled" and strip.pill.text() == "Stalled"
    assert strip.pill.toolTip().startswith("No shot for")
    strip.set_progress(4, 120)
    assert strip.state() == "running"

    strip.start_run(0, "scope_test", "", save_data=True, n_shots=5)
    assert strip.run_label.text() == "unsaved · scope_test"


def test_status_strip_never_asks_for_more_width(app):
    """A run starting must not widen the window: whatever the strip is told, the
    least width it will accept stays what it was when idle."""
    from waxx.util.live_od.gui.status_strip import StatusStrip
    strip = StatusStrip()
    strip.show()
    app.processEvents()
    idle = strip.minimumSizeHint().width()
    strip.start_run(80545, "a_very_long_experiment_class_name_indeed", "basler_2dmot",
                    save_data=False, n_shots=10000)
    strip.set_state("waiting_grab_drain", "a long detail string " * 5)
    strip.set_progress(9999, 10000)
    strip.set_timing(123.4, "23:59")
    app.processEvents()
    assert strip.minimumSizeHint().width() == idle
    assert strip.run_label.minimumSizeHint().width() == 90
    assert "a_very_long_experiment_class_name_indeed" in strip.run_label.toolTip()
    strip.hide()


def test_status_strip_with_a_camera_button(app):
    """The camera is a button after the run label, not text in it, and the strip
    still never widens."""
    from waxx.util.live_od.gui.status_strip import StatusStrip
    from waxx.util.live_od.gui.camera_menu import CameraMenuButton
    strip = StatusStrip()
    menu = CameraMenuButton(["cam_a", "a_long_camera_name"])
    strip.add_camera_widget(menu)
    layout = strip.layout()
    assert layout.indexOf(menu) == layout.indexOf(strip.run_label) + 1
    strip.show()
    app.processEvents()
    idle = strip.minimumSizeHint().width()
    strip.start_run(80545, "hf_tweezer_bec", "cam_a", save_data=True, n_shots=10)
    menu.set_current("cam_a")
    assert strip.run_label.text() == "80545 · hf_tweezer_bec"
    assert "cam_a" in strip.run_label.toolTip()
    menu.set_current("a_long_camera_name")
    app.processEvents()
    assert strip.minimumSizeHint().width() == idle
    strip.hide()


# ----------------------------------------------------------------------
# Camera button
# ----------------------------------------------------------------------

def test_camera_button_shows_one_camera_and_drops_down_the_others(app):
    from waxx.util.live_od.gui.camera_menu import CameraMenuButton, STATES
    menu = CameraMenuButton(["cam_a", "cam_b", "cam_c"])
    asked = []
    menu.toggle_requested.connect(asked.append)
    # before any run: the first camera, until one is connected
    assert menu.shown_camera() == "cam_a" and menu.text() == "cam_a"
    menu.set_state("cam_b", "open")
    assert menu.shown_camera() == "cam_b"
    assert STATES["open"][0] in menu.styleSheet()
    # the run's camera wins, whatever its state
    menu.set_current("cam_c")
    assert menu.text() == "cam_c" and STATES["closed"][0] in menu.styleSheet()
    menu.set_state("cam_c", "failed")
    assert STATES["failed"][0] in menu.styleSheet() and "failed" in menu.toolTip()
    width = menu.width()

    menu.click()                                         # the button: its own camera
    assert asked == ["cam_c"]
    shown = {k: a.isVisible() for k, (a, _b) in menu._menu_actions.items()}
    assert shown == {"cam_a": True, "cam_b": True, "cam_c": False}     # the others
    menu._menu_actions["cam_b"][1].click()               # a camera in the drop-down
    assert asked == ["cam_c", "cam_b"]
    assert STATES["open"][0] in menu._menu_actions["cam_b"][1].styleSheet()

    menu.set_camera_enabled("cam_c", False)              # waiting for the server
    menu.click()
    assert asked == ["cam_c", "cam_b"]
    menu.set_camera_enabled("cam_c", True)
    assert menu.width() == width                         # never moves the row

    # a remote viewer learns its cameras from the server's broadcast
    remote = CameraMenuButton()
    assert remote.text() == "no camera" and not remote.isEnabled()
    remote.set_states({"cam_x": "grabbing", "cam_y": "closed"})
    assert remote.camera_keys() == ["cam_x", "cam_y"] and remote.text() == "cam_x"


def test_camera_button_stylesheets_parse(app):
    """Qt drops a stylesheet it can't parse and only says so on the console."""
    from PyQt6.QtCore import qInstallMessageHandler
    from waxx.util.live_od.gui.camera_menu import CameraMenuButton, STATES
    messages = []
    previous = qInstallMessageHandler(lambda _mode, _ctx, msg: messages.append(msg))
    try:
        menu = CameraMenuButton(["cam_a", "cam_b"])
        for state in STATES:
            menu.set_state("cam_a", state)
            menu.set_state("cam_b", state)
            menu.show()
            menu.ensurePolished()
            for _action, button in menu._menu_actions.values():
                button.ensurePolished()
            app.processEvents()
        menu.hide()
        empty = CameraMenuButton()
        empty.ensurePolished()
    finally:
        qInstallMessageHandler(previous)
    assert not [m for m in messages if "stylesheet" in m.lower()]


# ----------------------------------------------------------------------
# Server run states
# ----------------------------------------------------------------------

@pytest.fixture
def server(app):
    """A server that is never started (no socket, no beacon) and has no data saver:
    its handlers are called directly, and a run that asks to save fails at file
    creation, which is the path test_server_reports_... wants."""
    from waxx.util.live_od.live_od_server import LiveODServer
    srv = LiveODServer(server_talk=None, data_saver=None)
    states = []
    srv.run_state_signal.connect(lambda state, detail: states.append(state))
    return srv, states


def _init_msg(**kw):
    msg = {"tag": "INIT_RUN", "save_data": False, "capture_images": False, "camera_key": "",
           "params": {"N_img": 3}, "N_shots_with_repeats": 2, "expt_class": "scope_test"}
    msg.update(kw)
    return msg


def test_server_states_for_an_unsaved_run(server):
    srv, states = server
    assert srv._handle_init_run(_init_msg())["ok"]
    for i in range(2):
        srv._handle_shot_complete({"shot_idx": i, "N_shots_total": 2})
    assert srv._handle_end_run({})["ok"]
    assert states == ["running", "done"]
    assert srv._handle_poll({})["run_in_progress"] is False


def test_server_states_waiting_for_a_camera(server):
    srv, states = server
    srv._handle_init_run(_init_msg(capture_images=True, camera_key="andor"))
    reply = srv._handle_wait_cam_ready({"timeout": 0.01})        # a slice that times out
    assert reply["timed_out"]
    srv.on_cam_ready()
    assert srv._handle_wait_cam_ready({"timeout": 0.01})["ready"]
    assert states == ["waiting_camera", "running"]


def test_server_states_for_an_aborted_run(server):
    srv, states = server
    srv._handle_init_run(_init_msg())
    srv._handle_reset({})
    assert srv._handle_shot_complete({"shot_idx": 0, "N_shots_total": 2})["reset_requested"]
    srv._handle_abort_run({})
    assert states == ["running", "aborting", "aborted"]


def test_server_reports_a_data_file_it_could_not_create(server):
    srv, states = server
    reply = srv._handle_init_run(_init_msg(save_data=True))
    assert not reply["ok"] and reply["error"].startswith("Data file creation failed")
    assert states == ["error"]
    assert srv._run_in_progress is False


def test_run_is_named_after_its_experiment_file(server):
    srv, _states = server
    srv._handle_init_run(_init_msg(expt_class="HFTweezerBEC", expt_file="hf_tweezer_bec"))
    assert srv._current_expt_name == "hf_tweezer_bec"
    srv._handle_end_run({})
    srv._handle_init_run(_init_msg(expt_class="HFTweezerBEC"))       # an older experiment process
    assert srv._current_expt_name == "HFTweezerBEC"


# ----------------------------------------------------------------------
# Units plate and fixed-width readouts
# ----------------------------------------------------------------------

def test_units_are_on_a_corner_plate_not_the_axes(viewer, app):
    viewer.handle_plot_data(make_shot())
    app.processEvents()
    assert viewer.units_label.text() == "px" and not viewer.units_label.isHidden()
    image_area = viewer.od_plot.mapFromScene(
        viewer.od_plot.getViewBox().sceneBoundingRect()).boundingRect()
    plate = viewer.units_label.geometry()
    assert plate.left() < image_area.center().x() and plate.bottom() > image_area.center().y()
    viewer.set_pixel_size_m(2e-6)
    viewer.um_checkbox.setChecked(True)
    assert viewer.units_label.text() == "µm"
    for name in ('bottom', 'left'):
        assert not viewer.od_plot.getAxis(name).label.isVisible()


def test_readout_plates_keep_their_width_across_units(viewer, app):
    from PyQt6.QtCore import QPointF
    viewer.handle_plot_data(make_shot())
    viewer.on_shot_scalars({'atom_number': 1.2e5, 'atom_cross_section_m2': 1e-13, 'px_calibrated': True,
                            'px_size_m': 2e-6, 'fit_sd_x': 60e-6, 'fit_sd_y': 40e-6,
                            'fit_center_x': 2e-4, 'fit_center_y': 1e-4,
                            'crop_origin_px': (0, 0), 'crop_shape_px': (512, 512)})

    def hover():
        viewer._on_mouse_moved((viewer.od_plot.getViewBox().mapViewToScene(QPointF(250, 200)),))
    hover()
    widths = (viewer.readout_label.width(), viewer.cursor_label.width())
    assert "center" in viewer.readout_label.text() and "centre" not in viewer.readout_label.text()
    viewer.um_checkbox.setChecked(True)
    hover()
    assert "µm" in viewer.readout_label.text() and "µm" in viewer.cursor_label.text()
    assert (viewer.readout_label.width(), viewer.cursor_label.width()) == widths


# ----------------------------------------------------------------------
# Markers
# ----------------------------------------------------------------------

def test_markers_add_reshape_delete_and_report(viewer, app):
    viewer.handle_plot_data(make_shot())
    changes = []
    viewer.markers_changed.connect(lambda key, markers: changes.append((key, markers)))
    viewer.add_marker((100.0, 120.0))
    viewer.add_marker((300.5, 40.0), shape="o", size=12.0, color="#00ff00", label="trap")
    first, second = viewer.get_markers()
    assert (first["x"], first["y"], first["shape"], first["label"]) == (100.0, 120.0, "crosshair", "")
    assert first["size"] > 4 and first["color"].startswith("#")      # sized for what is on screen
    assert second == {"x": 300.5, "y": 40.0, "shape": "o", "size": 12.0,
                      "color": "#00ff00", "label": "trap", "hidden": False}
    assert len(changes) == 2 and changes[-1][1] == viewer.get_markers()

    viewer.update_marker(0, shape="star", label="MOT", size=50.0, color="#112233")
    assert viewer.get_markers()[0] == {"x": 100.0, "y": 120.0, "shape": "star", "size": 50.0,
                                       "color": "#112233", "label": "MOT", "hidden": False}
    viewer._marker_items[1].setPos(310.0, 45.0)          # a drag ...
    viewer._marker_items[1].sigMoved.emit(viewer._marker_items[1])   # ... let go
    assert (changes[-1][1][1]["x"], changes[-1][1][1]["y"]) == (310.0, 45.0)
    viewer.delete_marker(0)
    assert [m["label"] for m in viewer.get_markers()] == ["trap"]

    n = len(changes)
    viewer.set_markers([{"x": 1, "y": 2, "shape": "nonsense", "size": 1e9, "color": "red"},
                        {"x": "bad"}])                   # from the server, or an old file
    assert viewer.get_markers() == [{"x": 1.0, "y": 2.0, "shape": "crosshair", "size": 5000.0,
                                     "color": "#ff4081", "label": "", "hidden": False}]
    assert len(changes) == n                             # showing them is not an edit


def test_new_markers_get_random_colors_apart_from_the_others(viewer, app):
    from PyQt6.QtGui import QColor
    viewer.handle_plot_data(make_shot())
    for i in range(4):
        viewer.add_marker((50.0 * i, 50.0))
    colors = [m["color"] for m in viewer.get_markers()]
    hues = sorted(QColor(c).hsvHueF() for c in colors)
    gaps = [b - a for a, b in zip(hues, hues[1:])] + [1 - hues[-1] + hues[0]]
    assert min(gaps) > 0.03                              # no two alike
    viewer.add_marker((10.0, 10.0), color="#123456")     # asked for: kept
    assert viewer.get_markers()[-1]["color"] == "#123456"


def test_double_click_adds_a_marker_where_clicked(viewer, app):
    from PyQt6.QtCore import QPointF, Qt
    viewer.handle_plot_data(make_shot())
    app.processEvents()
    vb = viewer.od_plot.getViewBox()

    class Click:
        def __init__(self, scene_pos, double=True, button=Qt.MouseButton.LeftButton):
            self._pos, self._double, self._button = scene_pos, double, button
        def button(self): return self._button
        def double(self): return self._double
        def scenePos(self): return self._pos

    at = vb.mapViewToScene(QPointF(150.0, 220.0))
    viewer._on_image_clicked(vb, Click(at, double=False))       # a single click: nothing
    assert viewer.get_markers() == []
    viewer._on_image_clicked(vb, Click(at))
    (marker,) = viewer.get_markers()
    assert (marker["x"], marker["y"]) == pytest.approx((150.0, 220.0), abs=0.5)

    viewer._marker_items[0]._hovered = True              # on a marker: no second one on top
    viewer._on_image_clicked(vb, Click(at))
    assert len(viewer.get_markers()) == 1
    off_image = vb.sceneBoundingRect().bottomRight() + QPointF(5, 5)    # the axes, the colour bar
    viewer._on_image_clicked(vb, Click(off_image))
    assert len(viewer.get_markers()) == 1


def test_delete_or_backspace_deletes_the_hovered_marker(viewer, app):
    from PyQt6.QtGui import QShortcut
    viewer.handle_plot_data(make_shot())
    viewer.add_marker((100.0, 100.0), label="a")
    viewer.add_marker((200.0, 200.0), label="b")
    keys = {s.key().toString() for s in viewer.findChildren(QShortcut)}
    assert {"Del", "Backspace"} <= keys
    changes = []
    viewer.markers_changed.connect(lambda key, markers: changes.append(markers))
    viewer.delete_hovered_marker()                       # nothing under the cursor: nothing
    assert len(viewer.get_markers()) == 2 and not changes
    viewer._marker_items[1]._hovered = True
    viewer.delete_hovered_marker()                       # no confirmation
    assert [m["label"] for m in viewer.get_markers()] == ["a"] and len(changes) == 1
    viewer.update_marker(0, hidden=True)
    viewer._marker_items[0]._hovered = True              # a hidden marker cannot be under it
    viewer.delete_hovered_marker()
    assert len(viewer.get_markers()) == 1


def test_markers_can_be_hidden_from_the_panel_and_dialog(viewer, app):
    from waxx.util.live_od.gui.markers import MarkerDialog
    viewer.handle_plot_data(make_shot())
    viewer.add_marker((100.0, 100.0), label="a")
    viewer.add_marker((200.0, 200.0), label="b")
    viewer.open_marker_panel()
    panel = viewer._marker_panel
    eye = panel._rows[1][1].visible_button
    assert eye.isCheckable() and eye.isChecked() and not eye.icon().isNull()
    eye.click()                                          # hide
    assert viewer.get_markers()[1]["hidden"] is True
    assert [item.isVisible() for item in viewer._marker_items] == [True, False]
    assert len(panel._rows) == 2                         # still listed, eye closed
    assert not panel._rows[1][1].visible_button.isChecked()
    dialog = MarkerDialog(viewer, 1)
    assert not dialog.fields.visible_button.isChecked()
    dialog.fields.visible_button.click()                 # and back
    assert viewer.get_markers()[1]["hidden"] is False and viewer._marker_items[1].isVisible()
    dialog.close()
    panel.close()


def test_marker_size_is_a_length_on_the_image(viewer, app):
    """30 image pixels is 30 image pixels at any zoom: the marker grows on screen as
    you zoom in, like the cloud beside it, rather than staying a fixed screen size."""
    viewer.handle_plot_data(make_shot())
    viewer.add_marker((256.0, 256.0), size=30.0)
    item = viewer._marker_items[0]
    vb = viewer.od_plot.getViewBox()

    def on_screen_width():
        vb.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)
        app.processEvents()
        return item.mapRectToDevice(item._path.boundingRect()).width()
    ranges = ((0, 512), (0, 512))
    wide = on_screen_width()
    ranges = ((192, 320), (192, 320))                    # 4x zoom
    assert on_screen_width() == pytest.approx(4 * wide, rel=0.05)
    assert item._path.boundingRect().width() == pytest.approx(30.0)
    # a marker too small to see can still be grabbed
    viewer.update_marker(0, size=1.0)
    assert viewer._marker_items[0].shape().boundingRect().width() > 1.0


def test_marker_dialog_and_panel_edit_in_the_viewers_units(viewer, app):
    from waxx.util.live_od.gui.markers import MarkerDialog
    viewer.handle_plot_data(make_shot())
    viewer.set_pixel_size_m(2e-6)
    viewer.add_marker((100.0, 50.0), size=30.0, label="a")
    viewer.add_marker((200.0, 60.0), size=10.0, label="b")

    dialog = MarkerDialog(viewer, 0)                     # what right-clicking a marker opens
    assert dialog.fields.size_spin.value() == 30.0 and dialog.fields.size_spin.suffix() == " px"
    viewer.um_checkbox.setChecked(True)
    dialog._refresh()
    assert dialog.fields.size_spin.value() == 60.0 and dialog.fields.size_spin.suffix() == " µm"
    assert dialog.position_label.text() == "200, 100 µm"
    dialog.fields.size_spin.setValue(80.0)               # 80 µm = 40 px, applied at once
    dialog.fields.label_edit.setText("MOT"); dialog.fields.label_edit.textEdited.emit("MOT")
    assert viewer.get_markers()[0]["size"] == 40.0 and viewer.get_markers()[0]["label"] == "MOT"

    viewer.open_marker_panel()
    panel = viewer._marker_panel
    assert len(panel._rows) == 2 and panel._rows[1][1].label_edit.text() == "b"
    assert panel._rows[0][1].size_spin.value() == 80.0   # follows the dialog's edit, in µm

    # hovering a row lights its marker up on the image
    panel._rows[1][0].hovered.emit(1)
    assert [item._highlighted for item in viewer._marker_items] == [False, True]
    panel._rows[1][0].hovered.emit(None)
    assert not any(item._highlighted for item in viewer._marker_items)

    panel._rows[0][1].shape_combo.setCurrentIndex(panel._rows[0][1].shape_combo.findData("s"))
    panel._rows[0][1].changed.emit()
    assert viewer.get_markers()[0]["shape"] == "s"
    viewer.delete_marker(0)
    assert len(panel._rows) == 1 and panel._rows[0][1].label_edit.text() == "b"
    panel.close()


def test_roi_is_grabbed_by_its_edges_and_resized_by_its_corners(viewer, app):
    from PyQt6.QtCore import QPointF
    viewer.handle_plot_data(make_shot())
    viewer.roi_button.setChecked(True)
    roi = viewer._roi_item
    roi.setPos([100, 100]); roi.setSize([200, 150])
    app.processEvents()
    assert len(roi.getHandles()) == 4                    # one per corner, no duplicates
    bx, by = roi._edge_band()
    assert 0 < bx < 20 and 0 < by < 20                   # a few screen pixels, in image px
    shape = roi.shape()
    assert not shape.contains(QPointF(100, 75))          # the middle: pans the image
    for point in ((0, 75), (200, 75), (100, 0), (100, 150),     # on each edge
                  (-0.5 * bx, 75), (200 + 0.5 * bx, 75)):       # and just outside
        assert shape.contains(QPointF(*point)), point
    # (pyqtgraph sends mouse events to an item only where its shape() contains the
    # point: GraphicsScene.itemsNearEvent)
    viewer.roi_button.setChecked(False)


def test_one_colormap_for_all_images_from_the_context_menu(viewer, app):
    menus = [viewer.od_plot.getViewBox().menu] + [
        v.getView().menu for v in (viewer.img_atoms_view, viewer.img_light_view, viewer.img_dark_view)]
    for menu in menus:
        # nothing of pyqtgraph's (View All, X / Y axis, Mouse Mode) ...
        titles = [a.text() for a in menu.actions() if a.isVisible()]
        assert titles == ["Add marker here", "Markers…", "Colormap"]
    # ... and the menu pops up without the plot's Plot Options or the scene's Export...
    from PyQt6.QtCore import QPointF
    popped = []
    menus[0].popup = popped.append
    event = type("Ev", (), {"screenPos": lambda self: QPointF(0, 0)})()
    viewer.od_plot.getViewBox().raiseContextMenu(event)
    assert popped and [a.text() for a in menus[0].actions() if a.isVisible()] == titles
    del menus[0].popup
    cmap_menu = [a for a in menus[1].actions() if a.isVisible()][2].menu()             # on a raw frame
    {a.text(): a for a in cmap_menu.actions()}["magma"].trigger()
    assert viewer._cmap_name == "magma"
    luts = [np.asarray(v.imageItem.lut) for v in
            (viewer.img_atoms_view, viewer.img_light_view, viewer.img_dark_view)]
    assert all(np.array_equal(lut, luts[0]) for lut in luts)
    assert viewer.od_colorbar.colorMap().name == "magma"
    checked = [a.text() for a in viewer._cmap_actions if a.isChecked()]
    assert checked == ["magma"] * 4                      # every menu shows the same choice
    viewer.set_all_colormaps("viridis")


def test_raw_levels_come_from_the_first_shot_of_a_series(viewer, app):
    atoms, light, dark, od, sx, sy = make_shot(peak_light=3000)
    viewer.handle_plot_data((atoms, light, dark, od, sx, sy))
    shared = (float(min(atoms.min(), light.min())), 3000.0)
    assert tuple(viewer.img_atoms_view.imageItem.levels) == shared       # atoms and light: one range
    assert tuple(viewer.img_light_view.imageItem.levels) == shared
    assert tuple(viewer.img_dark_view.imageItem.levels) == (100.0, 100.0)    # dark: its own
    viewer.handle_plot_data(make_shot(peak_light=1500))  # a dimmer shot looks dimmer
    assert tuple(viewer.img_light_view.imageItem.levels) == shared
    viewer.on_new_run()                                  # a new series: taken afresh
    viewer.handle_plot_data(make_shot(peak_light=1500))
    assert tuple(viewer.img_light_view.imageItem.levels)[1] == 1500.0


def test_colorbar_does_not_waste_the_right_hand_edge(viewer, app):
    assert viewer.od_colorbar.axis.width() <= 30         # pyqtgraph's default is 45
    app.processEvents()
    bar_right = viewer.od_colorbar.sceneBoundingRect().right()
    assert viewer.od_plot.width() - bar_right <= 4


def test_auto_roi_boxes_the_cloud_over_the_last_n_shots(viewer, app, tmp_path, monkeypatch):
    from waxx.util.live_od.gui.viewer import LiveODViewer
    monkeypatch.setattr(LiveODViewer, "_STATE_DIR", str(tmp_path))
    viewer.set_camera_key("cam_a")
    assert viewer.auto_roi() is None                     # no shots: nothing to do
    rng = np.random.default_rng(0)

    def noisy_shot(cx, cy):
        atoms, light, dark, od, sx, sy = make_shot(cx=cx, cy=cy)
        noise = lambda a: (a + rng.normal(0, 20, a.shape)).astype(np.float32)
        return noise(atoms), noise(light), dark, od, sx, sy

    viewer.handle_plot_data(noisy_shot(100, 100))        # an old shot, far away
    for _ in range(3):
        viewer.handle_plot_data(noisy_shot(350, 300))
    viewer.auto_roi_spinner.setValue(3)
    x1, y1, x2, y2 = viewer.auto_roi()
    assert x1 < 350 < x2 and y1 < 300 < y2
    assert x1 > 150 and y1 > 150                         # the old shot was left out
    assert viewer.roi_button.isChecked()
    assert viewer.get_od_roi_rect() == (x1, y1, x2, y2)
    assert (tmp_path / "live_od_cam_a_rect.json").exists()

    viewer.auto_roi(n_shots=4)                           # now it spans both clouds
    x1, y1, _, _ = viewer.get_od_roi_rect()
    assert x1 < 100 and y1 < 100


def test_markers_are_cleared_when_the_camera_changes(viewer, app, tmp_path, monkeypatch):
    from waxx.util.live_od.gui.viewer import LiveODViewer
    monkeypatch.setattr(LiveODViewer, "_STATE_DIR", str(tmp_path))
    viewer.set_camera_key("cam_a")
    viewer.set_markers([{"x": 5, "y": 6, "shape": "x"}])
    viewer.set_camera_key("cam_a")                       # the next run on the same camera
    assert len(viewer.get_markers()) == 1
    viewer.set_camera_key("cam_b")
    assert viewer.get_markers() == []


def test_marker_store_is_per_camera_and_survives_a_restart(tmp_path):
    from waxx.util.live_od.marker_store import MarkerStore
    path = str(tmp_path / "markers.json")
    store = MarkerStore(path)
    assert store.get("cam_a") == []
    store.set("cam_a", [{"x": 10, "y": 20, "shape": "o"}])
    store.set("cam_b", [{"x": 1, "y": 2}])
    again = MarkerStore(path)
    defaults = {"size": 30.0, "color": "#ff4081", "label": "", "hidden": False}
    assert again.get("cam_a") == [{"x": 10.0, "y": 20.0, "shape": "o", **defaults}]
    assert again.get("cam_b") == [{"x": 1.0, "y": 2.0, "shape": "crosshair", **defaults}]
    assert again.get("cam_c") == []
    # a marker from before size, color and label existed loads with the defaults;
    # the new fields round-trip
    store.set("cam_c", [{"x": 0, "y": 0, "shape": "s", "size": 12.5, "color": "#00FF00", "label": "MOT",
                         "hidden": True}])
    assert MarkerStore(path).get("cam_c") == [
        {"x": 0.0, "y": 0.0, "shape": "s", "size": 12.5, "color": "#00ff00", "label": "MOT",
         "hidden": True}]


def test_server_keeps_markers_for_remote_viewers(app, tmp_path):
    from waxx.util.live_od.live_od_server import LiveODServer
    srv = LiveODServer(server_talk=None, data_saver=None, marker_path=str(tmp_path / "m.json"))
    edits = []
    srv.markers_changed_signal.connect(lambda key, markers: edits.append((key, markers)))
    assert not srv._handle_set_markers({"markers": [{"x": 1, "y": 1}]})["ok"]       # no camera yet
    srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    reply = srv._handle_set_markers({"markers": [{"x": 3, "y": 4, "shape": "d"}]})
    assert reply["ok"] and reply["camera_key"] == "cam_a"
    stored = [{"x": 3.0, "y": 4.0, "shape": "d", "size": 30.0, "color": "#ff4081", "label": "",
               "hidden": False}]
    assert edits == [("cam_a", stored)]
    assert srv._handle_get_markers({})["markers"] == stored
    assert srv._handle_get_markers({"camera_key": "cam_b"})["markers"] == []


# ----------------------------------------------------------------------
# Live plot: a second metric on the right-hand axis
# ----------------------------------------------------------------------

def test_live_plot_second_metric_on_the_right_axis(app):
    from waxx.util.live_od.gui.live_scalar_plot_window import LiveScalarPlotWindow
    win = LiveScalarPlotWindow()
    _keep_alive.append(win)
    tiers = []
    win.subscription_changed_signal.connect(lambda old, new: tiers.append((old, new)))
    win.show()
    app.processEvents()
    win.on_new_run(80545, ["t_tof"])
    for i in range(5):
        win.on_shot_scalars({"shot_idx": i, "xvar_values": {"t_tof": 1e-3 * i},
                             "atom_number": 1e5 + i, "fit_sd_x": (10 + i) * 1e-6})
    right_axis = win.plot_widget.getPlotItem().getAxis('right')
    assert not right_axis.isVisible() and len(win._scatter_item2.getData()[0]) == 0

    win.metric2_combo.setCurrentText("fit σ x")
    assert right_axis.isVisible() and "fit σ x" in right_axis.labelText
    _x2, y2 = win._scatter_item2.getData()
    assert list(y2) == pytest.approx([10, 11, 12, 13, 14])           # µm, on its own axis
    assert list(win._scatter_item.getData()[1]) == pytest.approx([1e5 + i for i in range(5)])
    assert win._scatter_item2.getViewBox() is win._vb2
    assert tiers[-1] == ("atom_number", "fits")                       # the fits are needed now

    win.metric2_combo.setCurrentIndex(0)
    assert not right_axis.isVisible() and len(win._scatter_item2.getData()[0]) == 0
    win.close()


def test_enter_in_the_marker_dialog_commits_the_size_and_presses_no_button(viewer, app, monkeypatch):
    """Enter used to open the color picker (the dialog's first auto-default button)."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor
    from PyQt6.QtTest import QTest
    from waxx.util.live_od.gui import markers
    picked = []
    monkeypatch.setattr(markers.QColorDialog, "getColor",
                        staticmethod(lambda *a, **k: picked.append(a) or QColor()))
    viewer.handle_plot_data(make_shot())
    viewer.add_marker((100.0, 100.0), size=30.0)
    dialog = markers.MarkerDialog(viewer, 0)
    dialog.show()
    spin = dialog.fields.size_spin
    spin.setFocus()
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin, "44")
    QTest.keyClick(spin, Qt.Key.Key_Return)
    assert not picked
    assert viewer.get_markers()[0]["size"] == 44.0 and len(viewer.get_markers()) == 1
    assert dialog.isVisible()
    dialog.close()
