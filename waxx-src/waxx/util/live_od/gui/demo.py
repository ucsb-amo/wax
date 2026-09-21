"""The liveOD window's status strip, log panel and viewer fed with made-up shots,
for looking at the layout without cameras, ARTIQ or a lab config:

    python -m waxx.util.live_od.gui.demo

Nothing is saved, no server is started, and no settings are stored.
"""

import logging
import sys

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from waxx.util.live_od.gui.camera_menu import CameraMenuButton, CONNECTED_STATES
from waxx.util.live_od.gui.status_strip import StatusStrip
from waxx.util.live_od.gui.viewer import LiveODViewer
from waxx.util.live_od.log import get_logger, setup_logging

N_SHOTS = 40
PX_SIZE_M = 2.0e-6
logger = get_logger("demo")


def fake_shot(i, n=512, rng=np.random.default_rng(0)):
    y, x = np.mgrid[0:n, 0:n]
    sx, sy = 25 + i, 18 + 0.5 * i
    cx, cy = 256 + 3 * rng.standard_normal(), 230 + 3 * rng.standard_normal()
    od = 2.2 * np.exp(-((x - cx) ** 2 / (2 * sx ** 2) + (y - cy) ** 2 / (2 * sy ** 2)))
    od = od + 0.03 * rng.standard_normal(od.shape)
    light = (3000 * np.exp(-((x - 256) ** 2 + (y - 256) ** 2) / (2 * 300 ** 2))).astype(np.uint16)
    atoms = (light * np.exp(-od)).astype(np.uint16)
    dark = rng.integers(90, 110, (n, n)).astype(np.uint16)
    return (atoms, light, dark, od, od.sum(0), od.sum(1)), (cx, cy, sx, sy)


class DemoWindow(QWidget):
    def __init__(self):
        super().__init__()
        handler = setup_logging(log_dir=None)
        self.strip = StatusStrip()
        self.strip.set_next_run_id(80545)
        # made-up cameras that connect / disconnect at a click, nothing behind them
        self.camera_menu = CameraMenuButton(["andor", "xy_basler", "z_basler", "basler_2dmot"])
        self.camera_menu.toggle_requested.connect(lambda key: self.camera_menu.set_state(
            key, "closed" if self.camera_menu.state(key) in CONNECTED_STATES else "open"))
        self.camera_menu.set_state("andor", "open")
        self.strip.add_camera_widget(self.camera_menu)
        self.viewer = LiveODViewer()
        self.viewer.image_count_label.hide()
        handler.record_signal.connect(self.viewer.output_window.append_record)

        start = QPushButton("Start fake run")
        start.clicked.connect(self.start_run)
        fail = QPushButton("Log an error")
        fail.clicked.connect(lambda: logger.error(
            "END_RUN: save of run 80545 failed: [Errno 22] data drive unavailable\n"
            "Final params are preserved at C:\\...\\pending_80545.pkl. Finish the save with:\n"
            "    retry_pending_save(r'C:\\...\\pending_80545.pkl')"))
        top = QHBoxLayout()
        top.addWidget(self.strip, 1)
        top.addWidget(start)
        top.addWidget(fail)
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.viewer, 1)
        self.setLayout(layout)

        self._i = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.next_shot)

    def start_run(self):
        self._i = 0
        self.viewer.on_new_run()
        self.strip.start_run(80545, "hf_tweezer_bec", "andor", save_data=False, n_shots=N_SHOTS)
        self.camera_menu.set_current("andor")
        self.strip.set_state("waiting_camera")
        self.viewer.output_window.append_separator("Run 80545 · hf_tweezer_bec")
        logger.info(f"INIT_RUN: run_id=80545, hf_tweezer_bec, {N_SHOTS} shots, save=False, camera=andor")
        QTimer.singleShot(1200, lambda: (self.strip.set_state("running"), self._timer.start(700)))

    def next_shot(self):
        to_plot, (cx, cy, sx, sy) = fake_shot(self._i)
        self.viewer.handle_plot_data(to_plot)
        x0, x1, y0, y1 = self.viewer.get_crop_bounds(to_plot[3].shape)
        crop = to_plot[3][y0:y1, x0:x1]
        self.viewer.on_shot_scalars({
            'atom_number': float(crop.sum()) * PX_SIZE_M ** 2 / 1.4e-13, 'atom_cross_section_m2': 1.4e-13,
            'px_calibrated': True, 'px_size_m': PX_SIZE_M,
            'fit_sd_x': sx * PX_SIZE_M, 'fit_sd_y': sy * PX_SIZE_M,
            'fit_amp_x': float(crop.sum(0).max()), 'fit_amp_y': float(crop.sum(1).max()),
            'fit_center_x': (cx - x0) * PX_SIZE_M, 'fit_center_y': (cy - y0) * PX_SIZE_M,
            'fit_offset_x': 0.0, 'fit_offset_y': 0.0,
            'crop_origin_px': (x0, y0), 'crop_shape_px': (x1 - x0, y1 - y0)})
        self._i += 1
        self.strip.set_progress(self._i, N_SHOTS)
        self.strip.set_timing(0.7, "14:32")
        self.strip.set_xvars({'t_tof': 1e-3 * self._i})
        if self._i % 10 == 0:
            logger.log(logging.INFO, f"shot {self._i}/{N_SHOTS} (Δt=0.7s | ETA 14:32)")
        if self._i == N_SHOTS:
            self._timer.stop()
            logger.info("END_RUN: save_data=False, nothing written.")
            self.strip.set_state("done")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    from waxx.util.live_od.gui import theme
    theme.apply_theme(True)
    win = DemoWindow()
    win.setWindowTitle("LiveOD layout demo")
    win.resize(760, 860)
    win.show()
    sys.exit(app.exec())
