"""The liveOD window with LiveODConfig.use_camera_host: it builds the host (here
on FakeBackend cameras, through LiveODConfig.camera_host_factory), shows the
host's cameras on the old bar's surface, reports them in POLL, runs a camera
run through the real camera thread on a HostNanny, and closes the host when it
shuts down. The server, broadcaster and plotter are never started (no socket,
no beacon); data files live in tmp_path.
"""
import os
import sys
import threading
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

import cam_host_helpers as h
from cam_host_helpers import BASLER, Fakes, basler_params, wait_for
from live_od_data_fakes import FakeSaver, patch_payload_stash
import liveod_qt_helpers as qt


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


def _window(app, tmp_path, monkeypatch, use_camera_host):
    import gc
    from waxx.util.live_od import config as live_od_config
    from waxx.util.live_od.data import run_file
    from waxx.util.live_od.gui import main_window as mw
    from waxx.util.live_od.gui import theme
    from waxx.util.live_od.gui.plotter import LiveODPlotter
    from waxx.util.live_od.live_od_broadcaster import LiveODBroadcaster
    from waxx.util.live_od.live_od_server import LiveODServer
    from PyQt6 import sip

    h.quiet_network(monkeypatch, tmp_path)
    slot_errors = qt.SlotErrors()
    monkeypatch.setattr(sys, "excepthook", slot_errors)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    gc.collect()
    app.processEvents()
    views_before = qt.pyqtgraph_views()
    threads_before = set(threading.enumerate())
    for cls in (LiveODServer, LiveODBroadcaster, LiveODPlotter):
        monkeypatch.setattr(cls, "start", lambda self, *a: None)
    patch_payload_stash(monkeypatch, run_file, [])
    if theme._notifier is not None and sip.isdeleted(theme._notifier):
        monkeypatch.setattr(theme, "_notifier", None)
    spawned = []
    for name in ("CameraBaby", "DataHandler"):
        real = getattr(mw, name)
        monkeypatch.setattr(mw, name, lambda *a, _real=real, **k: spawned.append(_real(*a, **k)) or spawned[-1])

    fakes = Fakes()
    hosts = []

    def factory(cfg):
        hosts.append(h.make_host(cfg, fakes))
        return hosts[-1]
    other = types.SimpleNamespace(key="cam_c", camera_type="apd", serial_no="")
    config = live_od_config.LiveODConfig(
        data_saver=FakeSaver(tmp_path),
        run_id_source=types.SimpleNamespace(get_run_id=lambda: 1, update_run_id=lambda *a, **k: None,
                                            check_for_mapped_data_dir=lambda *a, **k: True),
        camera_params_list=[BASLER, other], use_camera_host=use_camera_host,
        camera_host_factory=factory)
    monkeypatch.setattr(live_od_config, "_active", None)
    win = mw.LiveODWindow(config, settings=None, log_dir=None)
    monkeypatch.setattr(sys, "excepthook", slot_errors)
    n_before = len(slot_errors.seen)

    def teardown():
        problems = []
        for obj in spawned:
            if hasattr(obj, "request_stop"):
                obj.request_stop()
            if hasattr(obj, "grab_finished"):
                obj.grab_finished()
        for obj in spawned:
            problems.append(qt.join_or_keep(obj))
            writer = getattr(obj, "writer", None)
            if writer is not None and getattr(writer, "started", False):
                problems.append(qt.join_or_keep(writer._worker))
        win.shutdown("test teardown")
        win.run_id_timer.stop()
        win._camera_state_timer.stop()
        for host in hosts:
            problems += h.stop_host(host)
        for t in qt.new_threads(threads_before):
            problems.append(qt.join_or_keep(t))
        qt.forget_views_since(views_before)
        qt.delete_widgets(app, [getattr(win, "live_scalar_plot_window", None),
                                getattr(win, "fk_tof_window", None),
                                getattr(win, "_adjust_panel", None), win])
        spawned.clear()
        qt.delete_widgets(app, [])
        problems = [p for p in problems if p]
        errors = slot_errors.seen[n_before:]
        assert not problems, problems
        assert not errors, f"exceptions in Qt slots while the window existed: {errors}"
    return win, fakes, hosts, teardown


@pytest.fixture
def host_window(app, tmp_path, monkeypatch):
    win, fakes, hosts, teardown = _window(app, tmp_path, monkeypatch, True)
    yield win, fakes, hosts[0]
    teardown()


def call_spinning(app, fn, *args, timeout=15.0):
    """``fn(*args)`` on a helper thread, as on the server's own thread, while this
    (the GUI) thread processes events."""
    out = {}

    def run():
        try:
            out["value"] = fn(*args)
        except BaseException as exc:
            out["exc"] = exc
    t = threading.Thread(target=run, name="test-server-call")
    t.start()
    deadline = time.monotonic() + timeout
    while t.is_alive() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    t.join(1.0)
    assert not t.is_alive()
    if "exc" in out:
        raise out["exc"]
    return out["value"]


def test_the_window_builds_the_host_and_its_bar(host_window, app):
    from waxx.util.live_od.camera_host.bar import HostCameraBar
    win, fakes, host = host_window
    assert win.camera_host is host and host.started
    assert win.live_od_server.camera_host is host
    assert isinstance(win.camera_conn_bar, HostCameraBar)
    assert [b.camera_name for b in win.camera_conn_bar.buttons] == ["cam_b", "cam_c"]
    cams = win.live_od_server._handle_poll({})["cameras"]
    assert cams["cam_b"]["state"] == "closed" and cams["cam_b"]["host_state"] == "closed"
    assert cams["cam_c"]["host_state"] == "virtual"
    # the status row's camera button, under the run rule, asks the host
    win._on_camera_toggle_requested("cam_b")
    wait_for(lambda: host.is_open("cam_b"))

    def shown_open():
        app.processEvents()
        return win.camera_conn_bar.get_button("cam_b").state == "open"
    wait_for(shown_open, what="the bar following the host's snapshot")
    assert win.camera_menu.state("cam_b") == "open"             # and the status row's button


def test_a_camera_run_through_the_window(host_window, app):
    win, fakes, host = host_window
    srv = win.live_od_server
    msg = {"tag": "INIT_RUN", "save_data": True, "capture_images": True, "camera_key": "cam_b",
           "camera_params": basler_params(), "params": {"N_img": 3}, "N_shots_with_repeats": 1,
           "N_pwa_per_shot": 3, "expt_class": "cam_host_window", "images_shape": (3, 4, 6),
           "images_dtype": "uint16"}
    reply = call_spinning(app, srv._handle_init_run, msg)
    token = reply["run_token"]
    assert reply["ok"] and host.worker("cam_b").locked_by == token
    app.processEvents()
    assert win.the_baby is not None
    assert win.the_baby.camera_nanny.token == token             # the run's own nanny

    def ready():
        r = srv._handle_wait_cam_ready({"run_token": token, "timeout": 0.1})
        return r if r.get("ready") else None
    r = call_spinning(app, lambda: wait_for(ready, timeout=10.0))
    assert r["ok"]
    fakes["cam_b"].trigger(3)
    call_spinning(app, srv._handle_shot_complete, {"run_token": token, "shot_idx": 0, "N_shots_total": 1})

    def all_in():
        app.processEvents()
        return srv._images_received_now() == 3
    wait_for(all_in, timeout=10.0)
    end = call_spinning(app, srv._handle_end_run, {"run_token": token})
    assert end["ok"] and "incomplete" not in end
    assert host.worker("cam_b").locked_by is None
    assert win.config.data_saver.saved[-1][0] == reply["filepath"]


def test_shutdown_closes_the_host(host_window, app):
    win, fakes, host = host_window
    host.request("cam_b", "open").result(5)
    assert fakes["cam_b"].opened
    win.shutdown("test")
    assert not fakes["cam_b"].opened
    assert host.shutdown(1.0) == {"already": True}


def test_with_the_flag_off_the_window_has_no_host(app, tmp_path, monkeypatch):
    from waxx.util.live_od.camera_connection_widget import CamConnBar
    win, fakes, hosts, teardown = _window(app, tmp_path, monkeypatch, False)
    try:
        assert win.camera_host is None and hosts == []
        assert win.live_od_server.camera_host is None
        assert type(win.camera_conn_bar) is CamConnBar
        assert win._nanny_for_run() is win.camera_nanny
        assert win._needs_grab_drain("xy_basler")
    finally:
        teardown()
