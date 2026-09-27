"""The liveOD server with its camera host (LiveODConfig.use_camera_host), through
the server's handlers (the server is never started: no socket, no beacon) and
the REAL camera thread and image dispatcher (CameraBaby, DataHandler) on a
HostNanny, with beacon FakeBackend cameras. Data files live in tmp_path
(FakeSaver).

INIT_RUN locks before a run id is used (and refuses without using one), the
first WAIT_CAM_READY arms (an arm that fails fails fast), RESET keeps the lock
and ABORT_RUN clears it, POLL keeps its keys, Persist is announced and
recorded, CAMERA_CONTROL goes to the host.
"""
import json
import logging
import os
import threading
import time
from queue import Queue

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import pytest
from PyQt6.QtCore import Qt

import cam_host_helpers as h
from cam_host_helpers import ANDOR, APD, BASLER, Fakes, andor_params, basler_params, wait_for
from live_od_data_fakes import FakeSaver, patch_payload_stash
import liveod_qt_helpers as qt

DIRECT = Qt.ConnectionType.DirectConnection


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


@pytest.fixture
def records():
    got = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            got.append(record)
    handler = ListHandler(logging.INFO)
    log = logging.getLogger("waxx.live_od")
    level = log.level
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    yield got
    log.removeHandler(handler)
    log.setLevel(level)


@pytest.fixture
def served(app, tmp_path, monkeypatch):
    h.quiet_network(monkeypatch, tmp_path)
    from waxx.util.live_od import config as live_od_config
    from waxx.util.live_od.config import LiveODConfig
    from waxx.util.live_od.data import run_file
    from waxx.util.live_od.live_od_server import LiveODServer
    from beacon.camera.schema import Constraint
    patch_payload_stash(monkeypatch, run_file, [])
    du897 = Constraint(when=(("vs_speed", "==", 0), ("vs_amp", "==", 0)), level="refuse",
                       reason="0.3 us vertical clock with Normal amplitude transfers no charge")
    cfg = LiveODConfig(camera_params_list=[BASLER, ANDOR, APD], use_camera_host=True,
                       camera_constraints={"andor_emccd": (du897,)})
    monkeypatch.setattr(live_od_config, "_active", cfg)
    before = set(threading.enumerate())
    fakes = Fakes(cam_a={"ranges": {"exposure_time": (2e-5, 10.0), "gain": (0, 300)}})
    host = h.make_host(cfg, fakes)
    host.start()
    saver = FakeSaver(tmp_path)
    srv = LiveODServer(server_talk=None, data_saver=saver)      # never started
    srv.set_camera_host(host)
    spawned = []
    yield srv, host, fakes, saver, spawned
    problems = []
    for baby, handler in spawned:
        baby.request_stop()
        handler.grab_finished()
    for baby, handler in spawned:
        problems.append(qt.join_or_keep(baby))
        problems.append(qt.join_or_keep(handler))
        writer = getattr(handler, "writer", None)
        if writer is not None and getattr(writer, "started", False):
            problems.append(qt.join_or_keep(writer._worker))
    problems += h.stop_host(host)
    problems += h.join_new_threads(before)
    problems = [p for p in problems if p]
    assert not problems, problems


def init_msg(**kw):
    msg = {"tag": "INIT_RUN", "save_data": False, "capture_images": True, "camera_key": "cam_b",
           "camera_params": basler_params(), "params": {"N_img": 3},
           "N_shots_with_repeats": 1, "N_pwa_per_shot": 3, "expt_class": "cam_host_test",
           "images_shape": (3, 4, 6), "images_dtype": "uint16"}
    msg.update(kw)
    return msg


def spawn(served, token, filepath="", save_data=False, camera_key="cam_b", n_img=3,
          camera_params=None):
    """What LiveODWindow.spawn_baby wires, with the host's nanny."""
    from waxx.util.live_od.camera_mother import CameraBaby, DataHandler
    from waxx.util.live_od.camera_host import HostNanny
    srv, host, _, _, spawned = served
    queue = Queue()
    handler = DataHandler(queue, filepath, save_data=save_data, imaging_type=0,
                          camera_key=camera_key, camera_params=camera_params or basler_params(),
                          params_payload={"N_img": n_img}, n_img=n_img, n_shots=1,
                          n_pwa_per_shot=n_img)
    baby = CameraBaby(handler, "Tester", queue, HostNanny(host, token))
    statuses = []
    baby.cam_status_signal.connect(statuses.append, DIRECT)
    baby.cam_status_signal.connect(lambda s: srv.on_cam_ready() if s == 2 else None, DIRECT)
    baby.camera_grab_start.connect(handler.get_img_number, DIRECT)
    baby.camera_grab_start.connect(lambda *a: handler.start(), DIRECT)
    baby.grab_failed_signal.connect(srv.on_grab_failed, DIRECT)
    baby.camera_overrides_signal.connect(srv.on_camera_overrides, DIRECT)
    handler.got_image_from_queue.connect(srv.on_image_received, DIRECT)
    handler.done_writing_signal.connect(srv.on_data_handler_done, DIRECT)
    spawned.append((baby, handler))
    baby.start()
    return baby, handler, statuses


def call_spinning(app, fn, *args, timeout=15.0):
    """``fn(*args)`` on a helper thread (as the server's own thread would), with
    this, the GUI thread, processing events meanwhile: the image writer's "done"
    reaches the server through the GUI thread's event loop."""
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
    assert not t.is_alive(), f"{getattr(fn, '__name__', fn)} did not return in {timeout} s"
    if "exc" in out:
        raise out["exc"]
    return out["value"]


def wait_ready(srv, token, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = srv._handle_wait_cam_ready({"run_token": token, "timeout": 0.1})
        if r.get("ready") or not r.get("timed_out"):
            return r
    raise AssertionError("camera never ready")


# ----------------------------------------------------------------------
# a whole run through the real camera thread
# ----------------------------------------------------------------------

def test_a_run_through_the_real_camera_thread_and_writer(served, app):
    srv, host, fakes, saver, _ = served
    reply = srv._handle_init_run(init_msg(save_data=True))
    token, path = reply["run_token"], reply["filepath"]
    assert host.worker("cam_b").locked_by == token and os.path.exists(path)
    baby, handler, statuses = spawn(served, token, filepath=path, save_data=True)
    r = wait_ready(srv, token)
    assert r["ok"] and r["ready"]
    wait_for(lambda: statuses[:4] == [0, 1, 2, 3])            # 3 follows 2 on the camera thread
    fakes["cam_b"].trigger(3)
    srv._handle_shot_complete({"run_token": token, "shot_idx": 0, "N_shots_total": 1})
    wait_for(lambda: srv._images_received_now() == 3, what="3 frames at the dispatcher")
    end = call_spinning(app, srv._handle_end_run, {"run_token": token})
    assert end["ok"] and "incomplete" not in end
    assert [p for p, _ in saver.saved] == [path]
    assert qt.join_or_keep(baby) == "" and statuses[-1] == -1
    assert host.worker("cam_b").locked_by is None                   # END_RUN released it
    assert host.snapshot()["cameras"]["cam_b"]["host_state"] == "idle"
    handler.writer._worker.wait(5000)
    with h5py.File(path, "r") as f:
        assert [int(f["data/images"][i, 0, 1]) for i in range(3)] == [0, 1, 2]   # hw index = slot


def test_a_lost_frame_makes_the_run_incomplete(served, app):
    srv, host, fakes, saver, _ = served
    reply = srv._handle_init_run(init_msg(save_data=True))
    token, path = reply["run_token"], reply["filepath"]
    baby, handler, statuses = spawn(served, token, filepath=path, save_data=True)
    assert wait_ready(srv, token)["ready"]
    fb = fakes["cam_b"]
    fb.trigger(1)
    fb.lose_next(1)
    fb.trigger(2)
    wait_for(lambda: srv._grab_failure, what="the camera thread's grab failure")
    end = call_spinning(app, srv._handle_end_run, {"run_token": token})
    assert end["ok"] and end["incomplete"]
    assert "lost" in srv._grab_failure
    assert saver.incomplete                                           # the file says so


def test_a_camera_fault_during_the_run_makes_it_incomplete(served, app):
    srv, host, fakes, saver, _ = served
    reply = srv._handle_init_run(init_msg(save_data=True))
    token, path = reply["run_token"], reply["filepath"]
    baby, handler, statuses = spawn(served, token, filepath=path, save_data=True)
    assert wait_ready(srv, token)["ready"]
    fb = fakes["cam_b"]
    fb.trigger(1)
    wait_for(lambda: srv._images_received_now() == 1)
    fb.fail_next("retrieve", RuntimeError("USB transfer failed"))
    wait_for(lambda: srv._grab_failure, what="the camera thread's grab failure")
    end = call_spinning(app, srv._handle_end_run, {"run_token": token})
    assert end["ok"] and end["incomplete"]
    assert "USB transfer failed" in srv._grab_failure and "camera error" in srv._grab_failure
    assert saver.incomplete


# ----------------------------------------------------------------------
# INIT_RUN / WAIT_CAM_READY
# ----------------------------------------------------------------------

def test_an_arm_that_fails_fails_fast(served):
    srv, host, fakes, saver, _ = served
    fakes.per_key["cam_b"] = {"run_free_runs": True, "cycle_s": 0.005}
    token = srv._handle_init_run(init_msg())["run_token"]
    t0 = time.monotonic()
    r = srv._handle_wait_cam_ready({"run_token": token, "timeout": 30.0})
    assert time.monotonic() - t0 < 5.0
    assert r["ok"] is False and r["ready"] is False and "timed_out" not in r
    assert "free-running" in r["error"]
    # and the client stops at once
    from waxx.util.live_od.live_od_client import LiveODClient
    client = LiveODClient.__new__(LiveODClient)
    client.last_reset_requested = False
    client._run_token = token
    client._send_recv = lambda payload, rcvtimeo_ms=None: srv._handle_wait_cam_ready(dict(payload))
    with pytest.raises(ValueError, match="free-running"):
        client.wait_cam_ready(timeout=30.0)


def test_a_refused_profile_uses_no_run_id(served, records):
    srv, host, fakes, saver, _ = served
    next_id = saver._next_run_id
    reply = srv._handle_init_run(init_msg(save_data=True, camera_key="cam_a",
                                          camera_params=andor_params(vs_speed=0, vs_amp=0)))
    assert reply["ok"] is False
    assert "no run id used" in reply["error"] and "no charge" in reply["error"]
    assert saver._next_run_id == next_id
    assert [f for f in os.listdir(saver.folder) if f.endswith(".hdf5")] == []
    assert host.worker("cam_a").locked_by is None
    assert srv._run_in_progress is False


def test_reset_keeps_the_lock_and_abort_clears_it(served):
    srv, host, fakes, saver, _ = served
    token = srv._handle_init_run(init_msg())["run_token"]
    assert host.worker("cam_b").locked_by == token
    srv._handle_reset({})
    assert host.worker("cam_b").locked_by == token                   # T10
    srv._handle_abort_run({"run_token": token})
    assert host.worker("cam_b").locked_by is None


def test_a_superseding_init_run_releases_the_old_lock(served):
    srv, host, fakes, saver, _ = served
    old = srv._handle_init_run(init_msg())["run_token"]
    new = srv._handle_init_run(init_msg(capture_images=False, camera_key=""))["run_token"]
    assert old != new and host.worker("cam_b").locked_by is None
    assert srv._handle_end_run({"run_token": old})["stale_run"]


def test_a_superseded_runs_camera_thread_reports_no_failure_to_the_new_run(served):
    srv, host, fakes, saver, _ = served
    old = srv._handle_init_run(init_msg())["run_token"]
    baby, handler, statuses = spawn(served, old)
    assert wait_ready(srv, old)["ready"]
    fakes["cam_b"].trigger(1)
    failures = []
    baby.grab_failed_signal.connect(failures.append, Qt.ConnectionType.DirectConnection)
    new = srv._handle_init_run(init_msg())["run_token"]            # takes over, lock and all
    assert host.worker("cam_b").locked_by == new
    time.sleep(0.3)
    assert baby.isRunning()                                        # waiting to be stopped, quietly
    baby.request_stop()                                            # what the window's spawn does
    assert qt.join_or_keep(baby) == ""
    assert failures == [] and srv._grab_failure == ""
    assert statuses[-1] == -1


def test_no_camera_and_apd_runs_take_no_lock(served):
    srv, host, fakes, saver, _ = served
    srv._handle_init_run(init_msg(capture_images=False))
    assert host.worker("cam_b").locked_by is None
    token = srv._handle_init_run(init_msg(camera_key="apd", camera_params={"key": "apd"}))["run_token"]
    assert host.worker("cam_b").locked_by is None and host.worker("cam_a").locked_by is None
    r = srv._handle_wait_cam_ready({"run_token": token, "timeout": 1.0})
    assert r["ok"] is False and "timed_out" not in r and "apd" in r["error"]


# ----------------------------------------------------------------------
# POLL, CAMERA_CONTROL
# ----------------------------------------------------------------------

def test_poll_keeps_its_keys_and_words(served, app, tmp_path):
    srv, host, fakes, saver, _ = served
    from waxx.util.live_od.live_od_server import LiveODServer
    legacy = LiveODServer(server_talk=None, data_saver=FakeSaver(tmp_path / "x"))
    assert set(srv._handle_poll({})) == set(legacy._handle_poll({}))
    cams = srv._handle_poll({})["cameras"]
    assert list(cams) == ["cam_b", "cam_a", "apd"]
    for c in cams.values():
        assert {"state", "camera_type", "serial_no"} <= set(c)
        assert c["state"] in ("closed", "loading", "open", "grabbing", "failed")
    assert cams["cam_b"]["serial_no"] == "s1" and cams["cam_b"]["camera_type"] == "basler"
    assert {"host_state", "persist", "holder", "n_subs"} <= set(cams["cam_b"])
    token = srv._handle_init_run(init_msg())["run_token"]
    assert srv._handle_poll({})["cameras"]["cam_b"]["state"] == "grabbing"
    assert srv._handle_poll({})["run_camera_key"] == "cam_b"
    srv._handle_abort_run({"run_token": token})


def test_camera_control_goes_to_the_host_under_the_run_rule(served):
    srv, host, fakes, saver, _ = served
    assert srv._handle_camera_control({"camera_key": "cam_b", "action": "open"})["ok"]
    wait_for(lambda: srv._handle_poll({})["cameras"]["cam_b"]["state"] == "open")
    token = srv._handle_init_run(init_msg())["run_token"]
    r = srv._handle_camera_control({"camera_key": "cam_b", "action": "close"})
    assert r["ok"] is False and r["run_in_progress"]                 # the run's own camera
    assert srv._handle_camera_control({"camera_key": "cam_a", "action": "close"})["ok"]
    assert srv._handle_camera_control({"camera_key": "nope", "action": "close"})["ok"] is False
    srv._handle_abort_run({"run_token": token})
    assert srv._handle_camera_control({"camera_key": "cam_b", "action": "close"})["ok"]
    wait_for(lambda: srv._handle_poll({})["cameras"]["cam_b"]["state"] == "closed")


# ----------------------------------------------------------------------
# Persist and clamps: announced, replied, recorded
# ----------------------------------------------------------------------

def test_persist_is_announced_replied_and_recorded_with_clamps(served, records, app):
    srv, host, fakes, saver, _ = served
    host.start_stream("cam_a").result(5)
    host.set_live("cam_a", {"gain": 30})
    ps = host.set_persist("cam_a", True)
    # 1e-5 s is below the fake camera's 2e-5 s minimum: clamped at the arm
    params = andor_params(gain=300, exposure_time=1e-5)
    reply = srv._handle_init_run(init_msg(save_data=True, camera_key="cam_a", camera_params=params))
    assert reply["ok"]
    rec = reply["camera_overrides"]
    assert rec["schema"] == 1 and rec["camera_key"] == "cam_a"
    assert rec["persist_since"] == ps.since_iso
    assert rec["fields"] == {"gain": {"requested": 300, "applied": 30, "origin": "persist"}}
    warned = [r for r in records if "PERSISTED CAMERA SETTINGS" in r.getMessage()]
    assert len(warned) == 1 and warned[0].levelno == logging.WARNING
    text = warned[0].getMessage()
    assert f"run {reply['run_id']} on cam_a" in text
    assert "gain  300 -> 30 (persisted)" in text and ps.since_iso in text
    token = reply["run_token"]
    baby, handler, statuses = spawn(served, token, filepath=reply["filepath"], save_data=True,
                                    camera_key="cam_a", camera_params=params)
    assert wait_ready(srv, token)["ready"]
    fakes["cam_a"].trigger(3)
    wait_for(lambda: srv._images_received_now() == 3)
    assert call_spinning(app, srv._handle_end_run, {"run_token": token})["ok"]
    texts = saver.payloads[-1]["extra_file_texts"]
    fields = json.loads(texts["camera_overrides"])["fields"]
    assert fields["gain"] == {"requested": 300, "applied": 30, "origin": "persist"}
    assert fields["exposure_time"] == {"requested": 1e-5, "applied": 2e-5, "origin": "clamped"}
    assert json.loads(texts["camera_overrides"])["persist_since"] == ps.since_iso


def test_persist_on_with_nothing_different_is_only_info(served, records):
    srv, host, fakes, saver, _ = served
    host.start_stream("cam_a").result(5)
    host.set_live("cam_a", {"gain": 30})
    host.set_persist("cam_a", True)
    reply = srv._handle_init_run(init_msg(camera_key="cam_a", camera_params=andor_params(gain=30)))
    assert "camera_overrides" not in reply
    assert not [r for r in records if "PERSISTED CAMERA SETTINGS" in r.getMessage()]
    assert any("nothing overridden" in r.getMessage() and r.levelno == logging.INFO for r in records)
    srv._handle_abort_run({"run_token": reply["run_token"]})
    # a camera other than the run's says nothing
    records.clear()
    srv._handle_init_run(init_msg())
    assert not [r for r in records if "Persist" in r.getMessage() or "PERSIST" in r.getMessage()]
