"""The run's client in INIT_RUN and POLL (2026-10-09): the experiment sends its
pid, host and launcher (WAXX_LAUNCHER); liveOD keeps them for the run and shows
them in POLL and in the last run's outcome. An older client sends none and
gets None/"". The server is never started (no socket, no beacon); its
handlers are called directly and its data file lives in pytest's tmp_path.
"""
import os
import socket
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from live_od_data_fakes import FakeSaver, patch_payload_stash
import liveod_qt_helpers as qt


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


@pytest.fixture
def srv(app, tmp_path, monkeypatch):
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    from waxx.util.live_od.live_od_server import LiveODServer
    server = LiveODServer(server_talk=None, data_saver=FakeSaver(tmp_path))  # never started
    yield server
    # the last run's ImageWriter thread holds its file open: let it close it
    # (each INIT_RUN finished the one before)
    server._run_file.finish_writer()
    assert server._run_file.writer_done.wait(10), "the run's ImageWriter did not finish"


def _init_msg(**kw):
    msg = {"tag": "INIT_RUN", "save_data": True, "capture_images": False, "camera_key": "",
           "params": {"N_img": 3}, "N_shots_with_repeats": 2, "expt_class": "identity_test"}
    msg.update(kw)
    return msg


def test_a_superseded_run_gets_an_outcome_naming_its_client(srv):
    """Review S4: a run in progress replaced by the next INIT_RUN (the gate's
    dead_client waiver) had no outcome record at all."""
    first = srv._handle_init_run(_init_msg(client_pid=4242, client_host="KONG",
                                           launcher="run_lock"))
    second = srv._handle_init_run(_init_msg(client_pid=5151, client_host="KONG"))
    assert second["ok"] and second["run_id"] != first["run_id"]
    last = srv._handle_poll({"tag": "POLL"})["last_outcome"]
    assert last["run_id"] == first["run_id"] and last["outcome"] == "superseded"
    assert "pid 4242 on KONG, launched by run_lock" in last["detail"]
    assert last["client_pid"] == 4242                       # the superseded run's client
    assert os.path.exists(first["filepath"])                # not deleted


def test_a_superseded_run_from_an_older_client_says_so(srv):
    first = srv._handle_init_run(_init_msg())
    srv._handle_init_run(_init_msg())
    last = srv._handle_poll({"tag": "POLL"})["last_outcome"]
    assert last["run_id"] == first["run_id"] and last["outcome"] == "superseded"
    assert "client not recorded" in last["detail"]


def test_poll_shows_the_runs_client(srv):
    reply = srv._handle_init_run(_init_msg(client_pid=4242, client_host="KONG",
                                           launcher="run_lock"))
    assert reply["ok"]
    poll = srv._handle_poll({"tag": "POLL"})
    assert poll["run_in_progress"] and poll["run_id"] == reply["run_id"]
    assert (poll["client_pid"], poll["client_host"], poll["launcher"]) == (4242, "KONG",
                                                                           "run_lock")


def test_an_older_client_gets_none_and_empty(srv):
    poll = srv._handle_poll({"tag": "POLL"})                 # before any run
    assert (poll["client_pid"], poll["client_host"], poll["launcher"]) == (None, "", "")
    srv._handle_init_run(_init_msg())                        # no client fields
    poll = srv._handle_poll({"tag": "POLL"})
    assert (poll["client_pid"], poll["client_host"], poll["launcher"]) == (None, "", "")


def test_a_malformed_pid_is_dropped_not_raised(srv):
    assert srv._handle_init_run(_init_msg(client_pid="nope", client_host=None,
                                          launcher=None))["ok"]
    poll = srv._handle_poll({"tag": "POLL"})
    assert (poll["client_pid"], poll["client_host"], poll["launcher"]) == (None, "", "")


def test_each_run_replaces_the_client(srv):
    srv._handle_init_run(_init_msg(client_pid=1, client_host="a", launcher="run_loop"))
    srv._handle_init_run(_init_msg(client_pid=2, client_host="b"))
    poll = srv._handle_poll({"tag": "POLL"})
    assert (poll["client_pid"], poll["client_host"], poll["launcher"]) == (2, "b", "")


def test_the_last_outcome_names_the_client(srv):
    reply = srv._handle_init_run(_init_msg(client_pid=4242, client_host="KONG",
                                           launcher="run_loop"))
    out = srv._handle_run_exited({"tag": "RUN_EXITED", "run_id": reply["run_id"],
                                  "reason": "launcher: child exited (code 1)"})
    assert out["ok"]
    poll = srv._handle_poll({"tag": "POLL"})
    assert not poll["run_in_progress"]
    last = poll["last_outcome"]
    assert last["outcome"] == "exited" and last["run_id"] == reply["run_id"]
    assert (last["client_pid"], last["client_host"], last["launcher"]) == (4242, "KONG",
                                                                           "run_loop")
    # POLL keeps showing the last run's client until the next INIT_RUN
    assert poll["client_pid"] == 4242


# -- the experiment's side ------------------------------------------------------------

class _Stub:
    """Just enough of an Expt for _serialize_init_payload."""

    def __init__(self):
        self.camera_params = SimpleNamespace(key="")
        self.data = SimpleNamespace(keys=[])
        self.setup_camera = False
        self.run_info = SimpleNamespace(save_data=False, run_date_str="", run_datetime_str="",
                                        expt_class="Stub", imaging_type=0)
        self.xvarnames, self.xvardims = [], []
        self.sort_idx, self.sort_N = [], []
        self.params = SimpleNamespace(N_img=1)
        self._adjust_specs = []

    def _expt_file_stem(self):
        return "stub"

    def _xvar_ranges(self):
        return {}


@pytest.mark.parametrize("env, launcher", [(None, ""), ("run_lock", "run_lock")])
def test_the_experiment_sends_its_pid_host_and_launcher(monkeypatch, env, launcher):
    from waxx.base.expt import Expt
    if env is None:
        monkeypatch.delenv("WAXX_LAUNCHER", raising=False)
    else:
        monkeypatch.setenv("WAXX_LAUNCHER", env)
    payload = Expt._serialize_init_payload(_Stub())
    assert payload["client_pid"] == os.getpid()
    assert payload["client_host"] == socket.gethostname()
    assert payload["launcher"] == launcher
