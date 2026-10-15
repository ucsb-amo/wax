"""Queue phase 2 in liveOD (2026-10-14): RESERVE_RUN (creates nothing, one at a
time, ttl and dead-pid expiry, dry runs), START_RUN (refusals before any run
id, the server's own start time in the file name), RELEASE_RESERVATION, POLL's
features / reservation, the M0 guard on the queue's plain INIT_RUN, and the
client's new methods against a new and an old server.

The server is never started (no socket, no beacon); its handlers are called
directly (or through _dispatch, the way its loop calls them) and every data
file lives in pytest's tmp_path. No process id used here is looked up for
real: pid_alive / process_started are replaced wherever a test's client is on
this host.
"""
import os
import pickle
import socket
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from live_od_data_fakes import FakeSaver, patch_payload_stash, tmp_data_saver
import liveod_qt_helpers as qt

HERE = socket.gethostname()


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


def _server(tmp_path, monkeypatch, saver):
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    from waxx.util.live_od.live_od_server import LiveODServer
    from waxx.util.live_od.log import get_log_buffer
    get_log_buffer().clear()
    return LiveODServer(server_talk=None, data_saver=saver)      # never started


def _close(server):
    server._run_file.finish_writer()
    assert server._run_file.writer_done.wait(10), "the run's ImageWriter did not finish"


@pytest.fixture
def srv(app, tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch, FakeSaver(tmp_path))
    yield server
    _close(server)


@pytest.fixture
def real(app, tmp_path, monkeypatch):
    """A server whose saver is the real DataSaver, rooted in tmp_path."""
    saver = tmp_data_saver(tmp_path)
    server = _server(tmp_path, monkeypatch, saver)
    yield server, saver
    _close(server)


def _alive(monkeypatch, alive):
    from waxx.util.device_state import run_gate
    monkeypatch.setattr(run_gate, "pid_alive", lambda pid: alive)


def _msg(tag="INIT_RUN", **kw):
    msg = {"tag": tag, "save_data": True, "capture_images": False, "camera_key": "",
           "params": {"N_img": 3, "t_tof": 1e-3}, "N_shots_with_repeats": 2,
           "expt_class": "reserve_test", "xvarnames": ["t_tof"],
           "run_date_str": "2001-02-03", "run_datetime_str": "2001-02-03_04-05-06"}
    msg.update(kw)
    return msg


def _reserve(srv, **kw):
    kw.setdefault("client_pid", 7001)
    kw.setdefault("client_host", "elsewhere")
    return srv._handle_reserve_run(_msg("RESERVE_RUN", **kw))


def _files(folder):
    out = []
    for root, _, names in os.walk(str(folder)):
        out += [os.path.join(root, n) for n in names]
    return sorted(out)


def _state(srv):
    return (srv._current_run_id, srv._run_token, srv._run_in_progress, srv._run_state,
            srv._reset_requested, dict(srv._current_client), srv._run_file.filepath)


# ---------------------------------------------------------------- RESERVE_RUN

def test_reserve_creates_nothing_and_changes_no_run_state(real, tmp_path):
    srv, saver = real
    run = srv._handle_init_run(_msg(client_pid=1, client_host="elsewhere"))
    files, state = _files(tmp_path), _state(srv)
    next_id, sets = saver.server_talk.next_id, saver.server_talk.set_calls
    from waxx.util.live_od.log import get_log_buffer
    runs_before = len(get_log_buffer().runs())
    reply = _reserve(srv, client_pid=2)
    assert reply["ok"] and len(reply["reservation"]) == 32
    assert "run_id" not in reply and "filepath" not in reply        # never a predicted id
    assert reply["features"]["reserve"] is True
    assert _files(tmp_path) == files                                 # no file
    assert (saver.server_talk.next_id, saver.server_talk.set_calls) == (next_id, sets)
    assert _state(srv) == state                                      # the run is untouched
    assert len(get_log_buffer().runs()) == runs_before
    # its log line is not in the current run's GET_LOG, but is kept under no run
    current = srv._handle_get_log({"tag": "GET_LOG"})
    assert current["run"]["run_id"] == run["run_id"]
    assert not any("RESERVE_RUN" in r["msg"] for r in current["records"])
    everything = srv._handle_get_log({"tag": "GET_LOG", "run_id": "all"})["records"]
    assert any("RESERVE_RUN" in r["msg"] for r in everything)


def test_reserve_with_no_run_at_all_changes_nothing(srv, tmp_path):
    state = _state(srv)
    assert _reserve(srv)["ok"]
    assert _state(srv) == state and _files(tmp_path) == []
    assert srv._handle_poll({"tag": "POLL"})["run_state"] == "idle"


def test_one_reservation_at_a_time_the_same_process_renews(srv):
    first = _reserve(srv, client_pid=10, client_started=100.0)
    other = _reserve(srv, client_pid=11, client_started=200.0)
    assert other["ok"] is False and other["refused"] and "one at a time" in other["error"]
    assert other["holder"]["pid"] == 10 and "reservation" not in other
    again = _reserve(srv, client_pid=10, client_started=100.0)       # the same process
    assert again["ok"] and again["reservation"] != first["reservation"]
    # the old token is gone: releasing it releases nothing
    assert srv._handle_release_reservation({"reservation": first["reservation"]})["released"] \
        is False
    assert srv._handle_poll({"tag": "POLL"})["reservation"]["pid"] == 10


def test_a_reused_pid_is_not_the_same_process(srv):
    _reserve(srv, client_pid=10, client_started=100.0)
    assert _reserve(srv, client_pid=10, client_started=900.0)["ok"] is False


def test_the_ttl_expires_a_reservation(srv):
    assert _reserve(srv, client_pid=10, ttl_s=60)["ttl_s"] == 60
    srv._reservation["mono"] -= 61                                   # 61 s later
    assert srv._handle_poll({"tag": "POLL"})["reservation"] is None
    assert _reserve(srv, client_pid=11)["ok"]


@pytest.mark.parametrize("asked, got", [(0.1, 5.0), (1e9, 3600.0), ("x", 900.0),
                                        (float("nan"), 900.0)])
def test_the_ttl_is_clamped(srv, asked, got):
    assert _reserve(srv, ttl_s=asked)["ttl_s"] == got


def test_a_dead_process_expires_its_reservation(srv, monkeypatch):
    _alive(monkeypatch, True)
    _reserve(srv, client_pid=10, client_host=HERE)
    assert _reserve(srv, client_pid=11, client_host=HERE)["ok"] is False
    from waxx.util.device_state import run_gate
    monkeypatch.setattr(run_gate, "pid_alive", lambda pid: pid != 10)  # the first one died
    assert _reserve(srv, client_pid=11, client_host=HERE)["ok"]
    assert srv._handle_poll({"tag": "POLL"})["reservation"]["pid"] == 11


def test_another_hosts_process_expires_by_ttl_only(srv, monkeypatch):
    _alive(monkeypatch, False)
    _reserve(srv, client_pid=10, client_host="other-pc")
    assert _reserve(srv, client_pid=11, client_host="other-pc")["ok"] is False


def test_release_is_idempotent(srv):
    token = _reserve(srv)["reservation"]
    assert srv._handle_release_reservation({"reservation": token}) == {"ok": True,
                                                                       "released": True}
    assert srv._handle_release_reservation({"reservation": token}) == {"ok": True,
                                                                       "released": False}
    assert srv._handle_release_reservation({}) == {"ok": True, "released": False}
    assert srv._handle_poll({"tag": "POLL"})["reservation"] is None


def test_poll_shows_features_and_the_reservation_without_its_token(srv):
    poll = srv._handle_poll({"tag": "POLL"})
    assert poll["features"]["reserve"] is True and poll["reservation"] is None
    token = _reserve(srv, client_pid=10, queue_job="42:abc", ttl_s=120,
                     expt_file="hf_bec.py")["reservation"]
    view = srv._handle_poll({"tag": "POLL"})["reservation"]
    assert (view["expt"], view["pid"], view["host"], view["queue_job"], view["ttl"]) == \
        ("hf_bec.py", 10, "elsewhere", "42:abc", 120.0)
    assert abs(view["since"] - time.time()) < 5 and 0 < view["expires_in_s"] <= 120
    assert token not in repr(view)


def test_the_dry_run_finds_what_the_real_start_finds(real, tmp_path):
    srv, _ = real
    bad = {"params": {"N_img": 1, "nothing": None, "a/b": 1.0},
           "camera_params": {"sensor_roi": None},
           "datavault_shapes": {"bad": {"shape": [2], "dtype": "float64", "fill": "x"}}}
    reserve = _reserve(srv, **bad)
    assert reserve["ok"] and len(reserve["warnings"]) == 4
    assert reserve["preview"] == {"camera_overrides": {}, "refused": []}
    assert _files(tmp_path) == []
    start = srv._handle_start_run(_msg("START_RUN", reservation=reserve["reservation"],
                                       client_pid=7001, client_host="elsewhere", **bad))
    assert start["ok"] and start["warnings"] == reserve["warnings"]


def test_the_dry_run_names_what_would_refuse_the_start(real, tmp_path):
    srv, _ = real
    reply = _reserve(srv, datavault_shapes={"k": {"shape": [2], "dtype": "nonsense"}})
    assert reply["ok"] and reply["preview"]["refused"]
    assert "fill would fail" in reply["preview"]["refused"][0]
    assert _files(tmp_path) == []


def test_an_unreachable_data_dir_is_named(app, tmp_path, monkeypatch):
    srv = _server(tmp_path, monkeypatch, tmp_data_saver(tmp_path / "not_there"))
    reply = _reserve(srv)
    assert reply["ok"] and any("not found" in r for r in reply["preview"]["refused"])
    assert not (tmp_path / "not_there").exists()


def test_a_hung_data_dir_check_does_not_hold_the_server(app, tmp_path, monkeypatch):
    import threading
    from waxx.util.live_od import live_od_server
    monkeypatch.setattr(live_od_server, "RESERVE_DIR_CHECK_S", 0.2)
    release = threading.Event()

    class HungSaver(FakeSaver):
        def data_dir_reachable(self):
            release.wait(10)                    # a share that does not answer
            return True, ""
    srv = _server(tmp_path, monkeypatch, HungSaver(tmp_path))
    try:
        t0 = time.monotonic()
        reply = _reserve(srv)
        assert time.monotonic() - t0 < 2.0
        assert reply["ok"] and "did not answer" in reply["preview"]["refused"][0]
    finally:
        release.set()
        _close(srv)


class _Host:
    """A camera host stand-in: the dry run only; locking would fail the test."""

    def __init__(self):
        self.previews = []

    def preview_run(self, camera_key, capture_images, camera_params):
        self.previews.append((camera_key, capture_images, dict(camera_params)))
        return {"camera_key": camera_key, "refusals": ["em_gain 900 > 300 cap"],
                "not_carried": {"exposure_time": "never persisted"},
                "overrides": {"em_gain": {"requested": 100, "applied": 200,
                                          "origin": "persist"}}}

    def begin_run(self, *a, **k):
        raise AssertionError("RESERVE_RUN must not lock a camera")

    def poll_cameras(self):
        return {}


def test_reserve_previews_the_camera_without_locking_it(srv):
    host = _Host()
    srv.set_camera_host(host)
    reply = _reserve(srv, capture_images=True, camera_key="andor",
                     camera_params={"em_gain": 100})
    assert host.previews == [("andor", True, {"em_gain": 100})]
    assert reply["preview"]["camera_overrides"]["fields"]["em_gain"]["applied"] == 200
    assert reply["preview"]["refused"] == ["camera andor: em_gain 900 > 300 cap"]
    assert any("exposure_time not carried" in w for w in reply["warnings"])
    assert srv._host_run_token == "" and srv._host_pending is None


# ---------------------------------------------------------------- START_RUN

def _start(srv, **kw):
    kw.setdefault("client_pid", 7001)
    kw.setdefault("client_host", "elsewhere")
    return srv._handle_start_run(_msg("START_RUN", **kw))


def test_start_stamps_the_servers_time_into_the_file_name_and_folder(real, tmp_path):
    srv, _ = real
    t0 = time.time()
    reply = _start(srv, reservation=_reserve(srv)["reservation"])
    assert reply["ok"] and reply["reservation_known"] is True
    assert reply["run_date_str"] != "2001-02-03"                      # not the client's stamp
    stamp = time.mktime(time.strptime(reply["run_datetime_str"], "%Y-%m-%d_%H-%M-%S"))
    assert t0 - 1 <= stamp <= time.time() + 1 and abs(reply["run_started_at"] - t0) < 5
    folder, name = os.path.split(reply["filepath"])
    assert folder == os.path.join(str(tmp_path), reply["run_date_str"])
    assert name == f"{reply['run_id']:07d}_{reply['run_datetime_str']}_reserve_test.hdf5"
    import h5py
    with h5py.File(reply["filepath"], "r") as f:                     # the run_info agrees
        assert f["run_info/run_datetime_str"][()].decode() == reply["run_datetime_str"]
    assert srv._handle_poll({"tag": "POLL"})["reservation"] is None    # spent


def test_plain_init_run_keeps_the_clients_stamp(real, tmp_path):
    srv, _ = real
    reply = srv._handle_init_run(_msg())
    assert os.path.dirname(reply["filepath"]) == os.path.join(str(tmp_path), "2001-02-03")
    assert "run_date_str" not in reply


def test_an_unknown_reservation_is_a_plain_start(srv):
    reply = _start(srv, reservation="f" * 32)
    assert reply["ok"] and reply["reservation_known"] is False and reply["run_id"]


def test_a_start_leaves_another_processes_reservation_alone(srv):
    _reserve(srv, client_pid=10)
    srv._handle_init_run(_msg(client_pid=99, client_host="elsewhere"))
    assert srv._handle_poll({"tag": "POLL"})["reservation"]["pid"] == 10


def test_a_missing_creation_time_is_never_the_same_process(srv):
    """N1: same pid and host but a creation time unknown on either side is
    not "the same process" (Windows reuses pids): no renewal, and the
    holder's INIT_RUN does not spend it (it lapses by release / dead pid /
    ttl)."""
    a = {"client_pid": 10, "client_host": "elsewhere", "client_started": 5.0}
    for missing in ({"client_started": None},):
        assert srv._same_process(a, dict(a, **missing)) is False
        assert srv._same_process(dict(a, **missing), a) is False
        assert srv._same_process(dict(a, **missing), dict(a, **missing)) is False
    assert srv._same_process(a, dict(a)) is True
    _reserve(srv, client_pid=10)                                     # no creation time
    assert _reserve(srv, client_pid=10)["ok"] is False               # not renewed
    srv._handle_init_run(_msg(client_pid=10, client_host="elsewhere"))
    assert srv._handle_poll({"tag": "POLL"})["reservation"] is not None


def test_the_holders_own_plain_init_run_spends_its_reservation(srv):
    _reserve(srv, client_pid=10, client_started=5.0)
    srv._handle_init_run(_msg(client_pid=10, client_host="elsewhere", client_started=5.0))
    assert srv._handle_poll({"tag": "POLL"})["reservation"] is None


def _assert_refused_before_an_id(srv, reply, files_before, state_before, folder, reason):
    assert reply["ok"] is False and reply["refused"] and reply["refused_reason"] == reason
    assert "no run id used" in reply["error"] or reason == "save_in_progress"
    assert "run_id" not in reply or reply["run_id"] == state_before[0]
    assert _files(folder) == files_before
    assert _state(srv) == state_before


def test_start_refuses_a_missing_feature_before_an_id(srv, tmp_path):
    files, state = _files(tmp_path), _state(srv)
    reply = _start(srv, require_features=["reserve", "teleport"])
    _assert_refused_before_an_id(srv, reply, files, state, tmp_path, "feature")
    assert reply["missing"] == ["teleport"]


def test_start_refuses_after_a_reset_since_go(srv, tmp_path):
    go = srv._handle_poll({"tag": "POLL"})["reset_count"]
    srv.request_reset("person")                         # Abort pressed between runs
    files, state = _files(tmp_path), _state(srv)
    reply = _start(srv, go_reset_count=go)
    _assert_refused_before_an_id(srv, reply, files, state, tmp_path, "reset_since_go")
    assert srv._handle_poll({"tag": "POLL"})["reset_requested"] is True   # left for a person


def test_start_with_no_reset_since_go_goes(srv):
    go = srv._handle_poll({"tag": "POLL"})["reset_count"]
    assert _start(srv, go_reset_count=go)["ok"]


def test_start_never_supersedes_a_live_run(srv, tmp_path, monkeypatch):
    _alive(monkeypatch, True)
    run = srv._handle_init_run(_msg(client_pid=4242, client_host=HERE))
    files, state = _files(tmp_path), _state(srv)
    reply = _start(srv)
    _assert_refused_before_an_id(srv, reply, files, state, tmp_path, "run_in_progress")
    assert reply["run_id"] == run["run_id"] and "pid 4242" in reply["error"]
    # nor with an Abort pending on it (the experiment has not answered yet)
    srv._handle_reset({"tag": "RESET"})
    reply = _start(srv)
    _assert_refused_before_an_id(srv, reply, files, _state(srv), tmp_path, "run_in_progress")
    assert "Abort is pending" in reply["error"] and os.path.exists(run["filepath"])


@pytest.mark.parametrize("client", [{"client_host": HERE}, {"client_pid": 4242,
                                                            "client_host": "other-pc"}])
def test_a_run_whose_client_cannot_be_known_dead_is_not_superseded(srv, monkeypatch, client):
    _alive(monkeypatch, False)
    srv._handle_init_run(_msg(**client))
    assert _start(srv)["refused_reason"] == "run_in_progress"


def test_start_supersedes_a_run_whose_client_is_known_dead_as_init_run_does(srv, monkeypatch):
    run = srv._handle_init_run(_msg(client_pid=4242, client_host=HERE))
    _alive(monkeypatch, False)
    reply = _start(srv)
    assert reply["ok"] and reply["run_id"] != run["run_id"]
    last = srv._handle_poll({"tag": "POLL"})["last_outcome"]
    assert last["outcome"] == "superseded" and "new START_RUN" in last["detail"]
    assert os.path.exists(run["filepath"])                           # not deleted


def test_start_refuses_while_the_previous_save_will_not_end(srv, tmp_path, monkeypatch):
    run = srv._handle_init_run(_msg(client_pid=1, client_host="elsewhere"))
    from waxx.util.live_od.data.run_file import RunFile
    monkeypatch.setattr(RunFile, "saving", property(lambda self: True))
    monkeypatch.setattr(RunFile, "wait_save", lambda self, timeout: False)
    files, state = _files(tmp_path), _state(srv)
    reply = _start(srv)
    _assert_refused_before_an_id(srv, reply, files, state, tmp_path, "save_in_progress")
    assert srv._current_run_id == run["run_id"]


def _exited_camera_run(srv):
    run = srv._handle_init_run(_msg(capture_images=True, camera_key="andor",
                                    client_pid=4242, client_host="elsewhere"))
    srv._cam_ready_event.set()                           # armed: frames could still come
    out = srv._handle_run_exited({"tag": "RUN_EXITED", "run_token": run["run_token"],
                                  "reason": "test"})
    assert out["ok"] and srv.exited_run_pending()
    return run


def test_start_refuses_while_an_exited_runs_frames_may_still_come(srv, tmp_path, monkeypatch):
    from waxx.util.live_od import live_od_server
    monkeypatch.setattr(live_od_server, "START_EXITED_WAIT_S", 0.3)
    monkeypatch.setattr(live_od_server, "EXITED_FRAME_GRACE_S", 60.0)
    run = _exited_camera_run(srv)
    files, state = _files(tmp_path), _state(srv)
    reply = _start(srv)
    _assert_refused_before_an_id(srv, reply, files, state, tmp_path, "exited_frames_due")
    assert srv.exited_run_pending() and srv._current_run_id == run["run_id"]


def test_start_waits_for_an_exited_run_to_close_then_goes(srv, monkeypatch):
    from waxx.util.live_od import live_od_server
    monkeypatch.setattr(live_od_server, "EXITED_FRAME_GRACE_S", 0.2)
    run = _exited_camera_run(srv)
    reply = _start(srv)
    assert reply["ok"] and reply["run_id"] != run["run_id"]
    assert os.path.exists(run["filepath"])                           # left as it was
    assert any(r["outcome"] == "exited" and r["run_id"] == run["run_id"]
               for r in srv._handle_get_log({"tag": "GET_LOG", "runs": True})["runs"])


def test_a_refused_starts_log_line_is_not_the_running_runs(srv, monkeypatch):
    _alive(monkeypatch, True)
    srv._handle_init_run(_msg(client_pid=4242, client_host=HERE))
    _start(srv)
    records = srv._handle_get_log({"tag": "GET_LOG"})["records"]
    assert not any("START_RUN refused" in r["msg"] for r in records)


# ---------------------------------------------------------------- M0: the queue's INIT_RUN

def test_the_queues_init_run_never_supersedes_a_live_run(srv, tmp_path, monkeypatch):
    _alive(monkeypatch, True)
    run = srv._handle_init_run(_msg(client_pid=4242, client_host=HERE, launcher="run_lock"))
    files, state = _files(tmp_path), _state(srv)
    reply = srv._handle_init_run(_msg(client_pid=5151, client_host=HERE, launcher="kq"))
    _assert_refused_before_an_id(srv, reply, files, state, tmp_path, "run_in_progress")
    assert srv._run_in_progress and srv._current_run_id == run["run_id"]


def test_a_persons_init_run_still_supersedes_as_before(srv, monkeypatch):
    _alive(monkeypatch, True)
    run = srv._handle_init_run(_msg(client_pid=4242, client_host=HERE, launcher="kq"))
    reply = srv._handle_init_run(_msg(client_pid=5151, client_host=HERE))     # a person
    assert reply["ok"] and reply["run_id"] != run["run_id"]
    assert srv._handle_poll({"tag": "POLL"})["last_outcome"]["outcome"] == "superseded"


def test_the_queues_init_run_supersedes_a_dead_clients_run(srv, monkeypatch):
    run = srv._handle_init_run(_msg(client_pid=4242, client_host=HERE))
    _alive(monkeypatch, False)
    reply = srv._handle_init_run(_msg(client_pid=5151, client_host=HERE, launcher="kq"))
    assert reply["ok"] and reply["run_id"] != run["run_id"]


def test_the_queues_init_run_takes_over_a_run_whose_experiment_said_it_exited(srv, monkeypatch):
    """run_gate waives such a run (dead_client: liveOD heard it exit, pid or
    not); the guard agrees, so a phase-1 queue job is not refused for it."""
    from waxx.util.live_od import live_od_server
    monkeypatch.setattr(live_od_server, "EXITED_FRAME_GRACE_S", 60.0)
    run = _exited_camera_run(srv)                         # client on another host
    reply = srv._handle_init_run(_msg(client_pid=5151, client_host=HERE, launcher="kq"))
    assert reply["ok"] and reply["run_id"] != run["run_id"]
    assert os.path.exists(run["filepath"])


def test_init_run_replies_carry_the_fills_warnings(real):
    srv, _ = real
    reply = srv._handle_init_run(_msg(params={"N_img": 1, "nothing": None}))
    assert reply["ok"] and len(reply["warnings"]) == 1
    assert "params/nothing not stored" in reply["warnings"][0]
    clean = srv._handle_init_run(_msg())
    assert clean["ok"] and "warnings" not in clean


# ---------------------------------------------------------------- through the dispatcher

def test_the_new_tags_go_through_the_dispatcher(srv):
    d = lambda m: srv._dispatch([pickle.dumps(m)])
    token = d(_msg("RESERVE_RUN", client_pid=3, client_host="elsewhere"))["reservation"]
    assert d({"tag": "POLL"})["reservation"]["pid"] == 3
    assert d(_msg("START_RUN", reservation=token, client_pid=3,
                  client_host="elsewhere"))["reservation_known"] is True
    assert d({"tag": "RELEASE_RESERVATION", "reservation": token})["released"] is False


# ---------------------------------------------------------------- the client

def _client(transport, once=None):
    from waxx.util.live_od.live_od_client import LiveODClient
    c = LiveODClient.__new__(LiveODClient)
    c._ip, c._port = "127.0.0.1", 1
    c._run_token = ""
    c.server_features = {}
    c._sender = None
    c._run_open = False
    c._exit_hook_registered = True          # no atexit hook from a test
    c.last_reset_requested = False
    c._send_recv = transport
    c._send_once = once or (lambda payload, timeout_ms: transport(payload))
    return c


def _wire(srv, old=False):
    """The client's request/reply straight into the server's dispatcher; ``old``:
    a server from before queue phase 2 (it knows none of the new tags)."""
    new_tags = ("RESERVE_RUN", "START_RUN", "RELEASE_RESERVATION")

    def transport(payload, rcvtimeo_ms=None):
        if old and payload.get("tag") in new_tags:
            return {"ok": False, "error": f"Unknown tag: {payload['tag']}"}
        return pickle.loads(pickle.dumps(srv._dispatch([pickle.dumps(payload)])))
    return transport


def test_the_client_round_trip(srv):
    c = _client(_wire(srv))
    payload = _msg(client_pid=8, client_host="elsewhere")
    payload.pop("tag")
    res = c.reserve_run(payload, ttl_s=60)
    assert res["ok"] and res["ttl_s"] == 60 and "tag" not in payload
    reply = c.start_run(payload, reservation=res["reservation"], require_features=["reserve"],
                        go_reset_count=0)
    assert reply["ok"] and reply["reservation_known"]
    assert c._run_token == reply["run_token"] and c._run_open and c.supports("reserve")
    assert c.release_reservation(res["reservation"]) == {"ok": True, "released": False}


def test_a_refused_start_raises_and_leaves_the_run_closed(srv, monkeypatch):
    from waxx.util.live_od.live_od_client import LiveODStartRefused
    _alive(monkeypatch, True)
    srv._handle_init_run(_msg(client_pid=4242, client_host=HERE))
    c = _client(_wire(srv))
    payload = _msg(client_pid=1, client_host="elsewhere")
    with pytest.raises(LiveODStartRefused) as err:
        c.start_run(payload, reservation=None)
    assert err.value.reply["refused_reason"] == "run_in_progress"
    assert not c._run_open and c._run_token == ""


def test_the_new_client_against_an_old_server(srv):
    c = _client(_wire(srv, old=True))
    payload = _msg()
    for reply in (c.reserve_run(payload), c.start_run(payload, reservation="x"),
                  c.release_reservation("x")):
        # the words Expt's fallback looks for ("unknown tag") stay in the error
        assert reply["ok"] is False and reply["unknown_tag"] is True
        assert "unknown tag" in reply["error"].lower()
    assert not c._run_open
    reply = c.init_run(dict(payload))                       # the fallback
    assert reply["ok"] and c._run_open


def test_an_old_client_against_the_new_server(srv):
    """An INIT_RUN with none of the new keys, and a POLL reader that knows none
    of the new fields, work as before."""
    reply = srv._dispatch([pickle.dumps({"tag": "INIT_RUN", "save_data": False,
                                         "params": {"N_img": 1}})])
    assert reply["ok"] and reply["run_id"] == 0 and "warnings" not in reply
    poll = srv._dispatch([pickle.dumps({"tag": "POLL"})])
    assert poll["run_in_progress"] and poll["reset_count"] == 0


def test_reserve_and_release_never_raise(srv):
    def broken(payload, rcvtimeo_ms=None):
        raise ConnectionError("no liveOD")

    def broken_once(payload, timeout_ms):
        raise TimeoutError("no answer")
    c = _client(broken, once=broken_once)
    assert c.reserve_run(_msg())["ok"] is False
    assert c.release_reservation("tok")["ok"] is False
    with pytest.raises(ConnectionError):                    # start_run raises as init_run
        c.start_run(_msg(), reservation="tok")


def test_the_client_prints_the_fills_warnings(capsys, app, tmp_path, monkeypatch):
    srv2 = _server(tmp_path, monkeypatch, tmp_data_saver(tmp_path / "d"))
    (tmp_path / "d").mkdir()
    try:
        c = _client(_wire(srv2))
        c.init_run(_msg(params={"N_img": 1, "nothing": None}))
        out = capsys.readouterr().out
        assert "!! liveOD, run 1: 1 warning(s):" in out and "params/nothing" in out
    finally:
        _close(srv2)
