"""The liveOD window's status strip, log panel and viewer fed with made-up shots,
for looking at the layout without cameras, ARTIQ or a lab config:

    python -m waxx.util.live_od.gui.demo            the camera host's CameraControl
    python -m waxx.util.live_od.gui.demo --legacy   the old CameraMenuButton

The camera control is fed by ``DemoHost``, a made-up camera host in this process
(snapshots once a second, as ``HostQtBridge`` delivers them): its ⚙ opens the real
settings dialog and its 🎥 the real live view window, on noise frames.  The buttons
turn Persist on, add subscribers and make the host go quiet (the control then shows
"unknown" after 5 s).  Nothing is saved, no server is started, no camera is opened,
nothing is sent on the network, and no settings are stored.
"""

import concurrent.futures
import logging
import sys
import threading
import time
import types

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from waxx.util.live_od.gui.camera_menu import CameraMenuButton, CONNECTED_STATES
from waxx.util.live_od.gui.status_strip import StatusStrip
from waxx.util.live_od.gui.viewer import LiveODViewer
from waxx.util.live_od.log import get_logger, setup_logging

N_SHOTS = 40
PX_SIZE_M = 2.0e-6
DEMO_RUN_ID = 80545
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


# ---------------------------------------------------------------------------
# A made-up camera host (the names the GUI uses on liveOD's CameraHost)
# ---------------------------------------------------------------------------

def _done(result=None, exc=None):
    fut = concurrent.futures.Future()
    if exc is not None:
        fut.set_exception(exc)
    else:
        fut.set_result(result)
    return fut


class DemoHost:
    """In-process cameras that only exist in this demo: states change when asked,
    frames are noise with a blob, made when the live view asks for one."""

    hostname = "demo"
    server_id = "camera_server:demo:liveod"
    CAMERAS = (("andor", "andor_emccd", "andor"), ("xy_basler", "basler_usb", "basler"),
               ("z_basler", "basler_usb", "basler"), ("basler_2dmot", "basler_usb", "basler"))

    def __init__(self):
        from beacon.camera.schema import get_category
        self._lock = threading.Lock()
        self._seq = 0
        self.quiet = False                  # stop answering snapshots (the "unknown" look)
        self.cams = {}
        for key, cat, ctype in self.CAMERAS:
            category = get_category(cat)
            settings = {s.key: s.default for s in category.settings
                        if s.default is not None and s.group != "hidden"}
            self.cams[key] = {"key": key, "camera_id": f"{cat}:{key}", "camera_type": ctype,
                              "category": cat, "serial": key, "host_state": "closed",
                              "state": "closed", "persist": False, "persisted": {},
                              "persist_since": None, "holder": None, "n_subs": 0, "error": None,
                              "settings": settings, "settings_rev": 1, "locked": False,
                              "run_id": None, "run_tag": None, "list_state": "closed"}
        self.cams["andor"]["settings"].update(gain=30, temperature=-60.0,
                                              cooler_status="stabilized")
        self.cams["z_basler"]["host_state"] = "held_elsewhere"
        self.cams["z_basler"]["holder"] = {"label": "Camera Viewer", "host": "lab-pc"}
        self.set_state("andor", "idle")
        self.core = types.SimpleNamespace(attach=self._attach, detach=self._detach)

    # -- demo controls -------------------------------------------------------------

    def set_state(self, key, host_state, **kw):
        from waxx.util.live_od.gui.camera_control import LEGACY_STATE, RUN_PHASES
        with self._lock:
            cam = self.cams[key]
            cam.update(host_state=host_state, state=LEGACY_STATE.get(host_state, "failed"),
                       locked=host_state in RUN_PHASES, **kw)

    def _key_of(self, camera_id):
        return next(k for k, c in self.cams.items() if c["camera_id"] == camera_id)

    def _attach(self, camera_id, key, kind, label=""):
        self.cams[self._key_of(camera_id)]["n_subs"] += 1
        return _done(True)

    def _detach(self, camera_id, key):
        cam = self.cams[self._key_of(camera_id)]
        cam["n_subs"] = max(0, cam["n_subs"] - 1)
        return _done(True)

    def _category(self, key):
        from beacon.camera.schema import get_category
        return get_category(self.cams[key]["category"])

    # -- the host's API ---------------------------------------------------------------

    def snapshot(self):
        with self._lock:
            return {"server_id": self.server_id, "t": time.time(),
                    "cameras": {k: {**c, "settings": dict(c["settings"]),
                                    "persisted": dict(c["persisted"])}
                                for k, c in self.cams.items()}}

    def on_snapshot(self, cb):
        return lambda: None

    def request(self, key, action, origin=""):
        hs = self.cams[key]["host_state"]
        if action == "toggle":
            action = "close" if hs in ("idle", "streaming") else "open"
        self.set_state(key, "idle" if action == "open" else "closed", holder=None)
        logger.info(f"demo host: {key} -> {action} ({origin})")
        return _done({"ok": True})

    def start_stream(self, key):
        self.set_state(key, "streaming")
        return _done(True)

    def stop_stream(self, key):
        self.set_state(key, "idle")
        return _done(True)

    def set_live(self, key, values, timeout_s=15.0):
        from beacon.camera.backend import ApplyRefused, Readback
        cam = self.cams[key]
        if cam["locked"]:
            raise RuntimeError(f"{key}: run {cam['run_id']} holds the camera")
        values = dict(values)
        unlocked = values.pop("em_gain_unlocked", False)
        if key == "andor" and values.get("gain", 0) > 100 and not unlocked:
            raise ApplyRefused("gain", "live EM gain above 100 needs 'unlock'")
        with self._lock:
            cam["settings"].update(values)
            cam["settings_rev"] += 1
        return {k: Readback(v, "hw") for k, v in values.items()}

    def persist(self, key):
        cam = self.cams[key]
        return types.SimpleNamespace(on=cam["persist"], values=dict(cam["persisted"]),
                                     since_iso=cam["persist_since"])

    def set_persist(self, key, on, values=None):
        cam = self.cams[key]
        if cam["locked"]:
            raise RuntimeError(f"persist for {key} cannot change while run {cam['run_id']} "
                               f"holds the camera")
        keys = self._category(key).persistable_keys()
        with self._lock:
            if on:
                cam.update(persist=True, persist_since=time.strftime("%Y-%m-%dT%H:%M:%S"),
                           persisted={k: cam["settings"][k] for k in keys if k in cam["settings"]})
            else:
                cam.update(persist=False, persisted={}, persist_since=None)
        return self.persist(key)

    def describe(self, key):
        from beacon.camera.schema import to_wire
        cam = self.cams[key]
        cat = self._category(key)
        dynamic = ({"hs_speed": {"choices": [0, 1, 2, 3],
                                 "labels": ["17 MHz", "10 MHz", "5 MHz", "1 MHz"]},
                    "preamp": {"choices": [0, 1, 2], "labels": ["1x", "2x", "4.5x"]},
                    "vs_speed": {"choices": [0, 1, 2, 3, 4],
                                 "labels": ["0.3 us", "0.5 us", "0.9 us", "1.7 us", "3.3 us"]}}
                   if key == "andor" else {"gain": {"range": [0.0, 36.0]}})
        return {"ok": True, "schema": to_wire(cat), "dynamic": dynamic,
                "settings": dict(cam["settings"]), "settings_rev": cam["settings_rev"],
                "live_profile": {k: v for k, v in cam["settings"].items()
                                 if cat.setting(k).group != "status"},
                "persist": cam["persist"], "persisted": dict(cam["persisted"]),
                "persist_since": cam["persist_since"],
                "persistable": list(cat.persistable_keys())}

    # -- what LocalHostStream uses -----------------------------------------------------

    def spec(self, key):
        cam = self.cams[key]
        return types.SimpleNamespace(key=key, camera_id=cam["camera_id"], serial=cam["serial"],
                                     category=self._category(key), camera_type=cam["camera_type"])

    def worker(self, key):
        return types.SimpleNamespace(settings_rev=self.cams[key]["settings_rev"])

    def wait_frame(self, key, after_seq=None, timeout_s=1.0, sources=("live", "snap", "run")):
        cam = self.cams[key]
        run = cam["host_state"] in ("acquiring", "draining")
        if cam["host_state"] != "streaming" and not run:
            time.sleep(min(float(timeout_s), 0.1))
            return None
        time.sleep(min(float(timeout_s), 0.05))
        with self._lock:
            self._seq += 1
            seq = self._seq
        rng = np.random.default_rng(seq)
        y, x = np.mgrid[0:120, 0:160]
        blob = 900 * np.exp(-((x - 80 - 10 * np.sin(seq / 9)) ** 2 + (y - 60) ** 2) / 300.0)
        image = (500 + 20 * rng.standard_normal((120, 160)) + blob).astype(np.uint16)
        image.flags.writeable = False
        return types.SimpleNamespace(
            image=image, source="run" if run else "live", seq=seq, t_host=time.time(),
            run_tag=f"{cam['run_id']}:demo0000" if run else None, settings=dict(cam["settings"]))


class DemoBridge(QObject):
    """HostQtBridge's stand-in: the demo host's snapshot once a second."""
    snapshot_changed = pyqtSignal(object)

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.tick)
        self._timer.start(1000)

    def tick(self):
        if not self.host.quiet:
            self.snapshot_changed.emit(self.host.snapshot())


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

class DemoWindow(QWidget):
    def __init__(self, legacy: bool = False):
        super().__init__()
        handler = setup_logging(log_dir=None)
        self.legacy = legacy
        self.strip = StatusStrip()
        self.strip.set_next_run_id(DEMO_RUN_ID)
        self.host = None
        self.dialogs = {}
        self.live_window = None
        extra = []
        if legacy:
            # made-up cameras that connect / disconnect at a click, nothing behind them
            self.camera_menu = CameraMenuButton(["andor", "xy_basler", "z_basler", "basler_2dmot"])
            self.camera_menu.toggle_requested.connect(lambda key: self.camera_menu.set_state(
                key, "closed" if self.camera_menu.state(key) in CONNECTED_STATES else "open"))
            self.camera_menu.set_state("andor", "open")
        else:
            from waxx.util.live_od.gui.camera_control import CameraControl
            self.host = DemoHost()
            self.bridge = DemoBridge(self.host, self)
            self.camera_menu = CameraControl([k for k, _c, _t in DemoHost.CAMERAS],
                                             expect_snapshots=True)
            self.camera_menu.set_snapshot(self.host.snapshot())
            self.bridge.snapshot_changed.connect(self.camera_menu.set_snapshot)
            self.camera_menu.action_requested.connect(self._on_action)
            self.camera_menu.settings_requested.connect(self._open_settings)
            self.camera_menu.live_view_requested.connect(self._on_live_view)
            persist = QPushButton("Persist xy_basler")
            persist.setCheckable(True)
            persist.toggled.connect(lambda on: (self.host.set_persist("xy_basler", on),
                                                self.bridge.tick()))
            subs = QPushButton("+ subscribers")
            subs.clicked.connect(self._more_subscribers)
            quiet = QPushButton("Host goes quiet")
            quiet.setCheckable(True)
            quiet.toggled.connect(lambda on: setattr(self.host, "quiet", on))
            extra = [persist, subs, quiet]
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
        if extra:
            row = QHBoxLayout()
            for b in extra:
                row.addWidget(b)
            row.addStretch(1)
            layout.addLayout(row)
        layout.addWidget(self.viewer, 1)
        self.setLayout(layout)

        self._i = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.next_shot)

    # -- the camera control's requests -------------------------------------------------

    def _on_action(self, key, action):
        from waxx.util.live_od.gui.camera_control import HOST_REQUEST
        self.host.request(key, HOST_REQUEST[action], origin=f"demo: {action}")
        self.bridge.tick()

    def _open_settings(self, key):
        from waxx.util.live_od.gui.camera_settings_dialog import CameraSettingsDialog
        dlg = self.dialogs.get(key)
        if dlg is None:
            dlg = CameraSettingsDialog(key, self.host, self,
                                       snapshot_signal=self.bridge.snapshot_changed)
            dlg.persist_changed.connect(lambda *_: self.bridge.tick())
            self.dialogs[key] = dlg
        dlg.show()
        dlg.raise_()

    def _on_live_view(self, key, on):
        from waxx.util.live_od.gui.live_view_window import LiveViewWindow
        if self.live_window is None:
            self.live_window = LiveViewWindow(self.host,
                                              snapshot_signal=self.bridge.snapshot_changed)
            self.live_window.view_opened.connect(
                lambda k: self.camera_menu.set_live_view_open(k, True))
            self.live_window.view_closed.connect(
                lambda k: self.camera_menu.set_live_view_open(k, False))
        if on:
            self.live_window.show_camera(key)
        else:
            self.live_window.close_camera(key)
        self.bridge.tick()

    def _more_subscribers(self):
        for cam in self.host.cams.values():
            cam["n_subs"] = {0: 3, 3: 12, 12: 150}.get(cam["n_subs"], 0)
        self.bridge.tick()

    # -- the fake run ----------------------------------------------------------------------

    def start_run(self):
        self._i = 0
        self.viewer.on_new_run()
        self.strip.start_run(DEMO_RUN_ID, "hf_tweezer_bec", "andor", save_data=False,
                             n_shots=N_SHOTS)
        self.camera_menu.set_current("andor")
        if self.host is not None:
            self.host.set_state("andor", "run_locked", run_id=DEMO_RUN_ID)
            if self.live_window is not None:
                self.live_window.set_run_active(True)
            self.bridge.tick()
        self.strip.set_state("waiting_camera")
        self.viewer.output_window.append_separator(f"Run {DEMO_RUN_ID} · hf_tweezer_bec")
        logger.info(f"INIT_RUN: run_id={DEMO_RUN_ID}, hf_tweezer_bec, {N_SHOTS} shots, "
                    f"save=False, camera=andor")
        QTimer.singleShot(1200, self._running)

    def _running(self):
        self.strip.set_state("running")
        if self.host is not None:
            self.host.set_state("andor", "acquiring", run_id=DEMO_RUN_ID)
            self.bridge.tick()
        self._timer.start(700)

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
        # the xvars are a plate on the OD image (the strip no longer has them)
        self.viewer.set_shot_xvars(self._i, {'t_tof': 1e-3 * (self._i + 1)})
        self._i += 1
        self.strip.set_progress(self._i, N_SHOTS)
        self.strip.set_timing(0.7, "14:32")
        if self._i % 10 == 0:
            logger.log(logging.INFO, f"shot {self._i}/{N_SHOTS} (Δt=0.7s | ETA 14:32)")
        if self._i == N_SHOTS:
            self._timer.stop()
            logger.info("END_RUN: save_data=False, nothing written.")
            self.strip.set_state("done")
            if self.host is not None:
                self.host.set_state("andor", "idle", run_id=None)   # stays at the run's settings
                if self.live_window is not None:
                    self.live_window.set_run_active(False)
                self.bridge.tick()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    from waxx.util.live_od.gui import theme
    theme.apply_theme(True)
    win = DemoWindow(legacy="--legacy" in sys.argv)
    win.setWindowTitle("LiveOD layout demo")
    win.resize(760, 860)
    win.show()
    sys.exit(app.exec())
