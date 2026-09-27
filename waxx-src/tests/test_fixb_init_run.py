"""FIX-B findings whose fix is in FIX-B's guarded edits (scratchpad
guarded/fixb/edits.md): they skip until the lead has applied them.

* m10 (E1+E2): a data file that cannot be created refuses the INIT_RUN and
  leaves the run in progress as it was -- its token, its file, its camera run,
  its camera lock (host mode) -- and the refused INIT_RUN's own lock is given back.
* m1 (E3-E6): a replaced run's image writer cannot open the new run's END_RUN save.

The server is never started (no socket, no beacon); every data file lives in
pytest's tmp_path.
"""
import inspect
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from live_od_data_fakes import FakeSaver, patch_payload_stash
import liveod_qt_helpers as qt

from waxx.util.live_od.live_od_server import LiveODServer

needs_m10 = pytest.mark.skipif(
    not hasattr(LiveODServer, "_host_abandon_run"),
    reason="waits for FIX-B edits.md E1+E2 (m10, guarded INIT_RUN reorder)")
needs_writer_token = pytest.mark.skipif(
    "run_token" not in inspect.signature(LiveODServer.on_data_handler_done).parameters,
    reason="waits for FIX-B edits.md E3-E6 (m1, the writer's done)")


@pytest.fixture(scope="module")
def app():
    return qt.session_app()


@pytest.fixture
def server(app, tmp_path, monkeypatch):
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    saver = FakeSaver(tmp_path)
    srv = LiveODServer(server_talk=None, data_saver=saver)   # never started
    states = []
    srv.run_state_signal.connect(lambda state, detail: states.append(state))
    return srv, saver, states


def _init_msg(**kw):
    msg = {"tag": "INIT_RUN", "save_data": True, "capture_images": False, "camera_key": "",
           "params": {"N_img": 3}, "N_shots_with_repeats": 1, "expt_class": "fixb_init"}
    msg.update(kw)
    return msg


class FakeHost:
    def __init__(self):
        self.begun, self.ended = [], []

    def begin_run(self, token, camera_key, capture_images, camera_params=None, images_shape=None):
        self.begun.append(token)
        return types.SimpleNamespace(camera_key=camera_key, refused={}, persist_on=False,
                                     overrides={}, persist_since=None)

    def note_run_id(self, token, run_id):
        pass

    def end_run(self, token, reason="END_RUN"):
        self.ended.append((token, reason))
        return types.SimpleNamespace(overrides={}, problems=[], camera_key="", persist_since=None)

    def poll_cameras(self):
        return {}


@needs_m10
def test_a_data_file_failure_leaves_the_run_in_progress_as_it_was(server):
    srv, saver, states = server
    first = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    token, spawn = first["run_token"], list(srv._spawn_tokens)
    srv.on_image_received(None)
    saver.fail_reserve = True                                  # the data drive is gone
    reply = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_b"))
    assert reply == {"ok": False, "error": "Data file creation failed: drive not mapped"}
    # nothing of the first run changed
    assert srv._run_token == token and srv._run_in_progress is True
    assert srv._current_camera_key == "cam_a" and srv._current_capture_images is True
    assert srv._images_received_now() == 1 and list(srv._spawn_tokens) == spawn
    assert states[-1] != "error"
    # and it is saved into its own file at its END_RUN
    saver.fail_reserve = False
    srv.on_data_handler_done()
    assert srv._handle_shot_complete({"run_token": token, "shot_idx": 0, "N_shots_total": 1})["ok"]
    assert srv._handle_end_run({"run_token": token})["ok"]
    assert [p for p, _ in saver.saved] == [first["filepath"]]


@needs_m10
def test_a_data_file_failure_with_no_run_in_progress_shows_the_error(server):
    srv, saver, states = server
    saver.fail_reserve = True
    reply = srv._handle_init_run(_init_msg())
    assert reply["ok"] is False and states[-1] == "error"
    assert srv._run_token == "" and srv._run_in_progress is False


@needs_m10
def test_host_mode_gives_back_the_refused_runs_lock_and_keeps_the_running_ones(server):
    srv, saver, _ = server
    host = FakeHost()
    srv.set_camera_host(host)
    first = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))
    saver.fail_reserve = True
    assert srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_b"))["ok"] is False
    refused = host.begun[-1]
    assert refused != first["run_token"]
    assert host.ended == [(refused, "INIT_RUN refused: the data file was not created")]
    assert srv._host_run_token == first["run_token"]


@needs_m10
def test_a_pending_reset_is_still_finalized_by_the_next_accepted_init_run(server):
    """The reset run's file is deleted through its own RunFile state, before the
    new run's begin() forgets it."""
    srv, saver, _ = server
    first = srv._handle_init_run(_init_msg())
    srv._handle_reset({})
    second = srv._handle_init_run(_init_msg())
    assert second["ok"] and not os.path.exists(first["filepath"])
    assert os.path.exists(second["filepath"])
    assert srv._handle_end_run({"run_token": second["run_token"]})["ok"]
    assert [p for p, _ in saver.saved] == [second["filepath"]]


@needs_writer_token
def test_a_replaced_runs_writer_cannot_open_the_new_runs_save_gate(server):
    srv, _, _ = server
    old = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))["run_token"]
    new = srv._handle_init_run(_init_msg(capture_images=True, camera_key="cam_a"))["run_token"]
    srv.on_data_handler_done(run_token=old)
    assert srv.wait_for_image_writer(0) is False               # the new run's writer is still due
    srv.on_data_handler_done(run_token=new)
    assert srv.wait_for_image_writer(0) is True
