"""Arrays pushed into the run's file during the run (PUT_DATA), the writer that
takes them, and the END_RUN save that runs on its own thread (SAVE_STATUS).

Everything here runs against the test saver in a temp folder; no socket, no
network.
"""
import pickle
import threading
import time

import h5py
import numpy as np
import pytest
import liveod_qt_helpers as qt
from live_od_data_fakes import FakeSaver, patch_payload_stash


@pytest.fixture
def app():
    return qt.session_app()


@pytest.fixture
def server(app, tmp_path, monkeypatch):
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    from waxx.util.live_od.live_od_server import LiveODServer
    saver = FakeSaver(tmp_path)
    srv = LiveODServer(server_talk=None, data_saver=saver)   # never started
    return srv, saver


def _init_msg(**kw):
    msg = {"tag": "INIT_RUN", "save_data": True, "capture_images": False, "camera_key": "",
           "params": {"N_img": 3}, "N_shots_with_repeats": 2, "expt_class": "put_test"}
    msg.update(kw)
    return msg


def _put_msg(token, items):
    """(header, buffers) as the server's loop hands them to the handler."""
    specs, buffers = [], []
    for key, index, arr, full_shape, fill in items:
        arr = np.ascontiguousarray(arr)
        specs.append({"key": key, "index": index, "shape": list(arr.shape),
                      "dtype": arr.dtype.str, "full_shape": full_shape, "fill": fill})
        buffers.append(arr.tobytes())
    return {"tag": "PUT_DATA", "run_token": token, "items": specs}, buffers


def _wait(cond, timeout=5.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise AssertionError("timed out")
        time.sleep(0.02)


def test_pushed_arrays_land_in_their_slots_and_the_rest_keeps_the_fill(server):
    srv, saver = server
    init = srv._handle_init_run(_init_msg())
    token, path = init["run_token"], init["filepath"]
    assert init["features"] == {"put_data": True, "async_save": True}
    frame = (np.arange(20, dtype=np.uint8) + 3).reshape(4, 5)
    meta = np.array([1., 2., 3.])
    msg, bufs = _put_msg(token, [
        ("img_a", [1], frame, [2, 4, 5], 0),
        ("img_a_meta", [1], meta, [2, 3], float("nan")),
        ("scope_data/PD/t", None, np.linspace(0, 1, 6, dtype=np.float32), [6], None),
    ])
    reply = srv._handle_put_data(msg, bufs)
    assert reply == {"ok": True, "queued": 3, "written": True}
    writer = srv._run_file.writer
    _wait(lambda: sum(writer.put_report.snapshot()[0].values()) == 3)
    # the file is still the writer's: END_RUN closes it, then saves
    assert not writer.finished
    assert srv._handle_end_run({"run_token": token})["ok"]
    assert writer.finished and srv._run_file.writer_done.is_set()
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["data/img_a"][1], frame)
        assert not f["data/img_a"][0].any()
        assert np.array_equal(f["data/img_a_meta"][1], meta)
        assert np.all(np.isnan(f["data/img_a_meta"][0]))
        assert f["data/scope_data/PD/t"].shape == (6,)
    assert srv._last_outcome["outcome"] == "saved"


def test_a_slice_fills_part_of_a_dataset(server):
    srv, saver = server
    init = srv._handle_init_run(_init_msg())
    token, path = init["run_token"], init["filepath"]
    whole = np.arange(5 * 3, dtype=np.int32).reshape(5, 3)
    msg, bufs = _put_msg(token, [("big", None, whole[:2], [5, 3], -1)])
    msg["items"][0]["offset"] = 0
    assert srv._handle_put_data(msg, bufs)["ok"]
    msg, bufs = _put_msg(token, [("big", None, whole[2:], [5, 3], -1)])
    msg["items"][0]["offset"] = 2
    assert srv._handle_put_data(msg, bufs)["ok"]
    assert srv._handle_end_run({"run_token": token})["ok"]
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["data/big"][()], whole)


def test_put_data_is_refused_for_a_stale_run_no_run_or_no_file(server):
    srv, saver = server
    msg, bufs = _put_msg("nobody", [("x", [0], np.zeros(2), [1, 2], 0)])
    assert srv._handle_put_data(msg, bufs)["ok"] is False
    old = srv._handle_init_run(_init_msg())
    new = srv._handle_init_run(_init_msg())
    msg, bufs = _put_msg(old["run_token"], [("x", [0], np.zeros(2), [1, 2], 0)])
    assert srv._handle_put_data(msg, bufs).get("stale_run")
    msg, bufs = _put_msg(new["run_token"], [("x", [0], np.zeros(2), [1, 2], 0)])
    assert srv._handle_put_data(msg, bufs)["ok"]
    assert srv._handle_end_run({"run_token": new["run_token"]})["ok"]
    # save_data=False: taken (for the broadcast), written nowhere
    init = srv._handle_init_run(_init_msg(save_data=False))
    msg, bufs = _put_msg(init["run_token"], [("x", [0], np.zeros(2), [1, 2], 0)])
    reply = srv._handle_put_data(msg, bufs)
    assert reply["ok"] is True and reply["written"] is False


def test_a_run_that_saves_nothing_takes_pushes_keeps_the_latest_and_broadcasts(server):
    srv, saver = server
    got = []
    srv.aux_data_signal.connect(got.append)
    init = srv._handle_init_run(_init_msg(save_data=False))
    token = init["run_token"]
    frame = np.arange(6, dtype=np.uint8).reshape(2, 3)
    msg, bufs = _put_msg(token, [("img_a", [1], frame, [2, 2, 3], 0),
                                 ("img_a_meta", [1], np.ones(3), [2, 3], float("nan"))])
    reply = srv._handle_put_data(msg, bufs)
    assert reply == {"ok": True, "queued": 0, "written": False}
    assert srv._run_file.writer is None             # nothing is written anywhere
    latest = srv._handle_get_aux_data({"run_token": token})
    assert latest["ok"] and set(latest["items"]) == {"img_a", "img_a_meta"}
    rec = latest["items"]["img_a"]
    assert rec["index"] == [1] and np.array_equal(rec["array"], frame) and rec["run_id"] == init["run_id"]
    assert srv._handle_get_aux_data({"keys": ["img_a"]})["items"].keys() == {"img_a"}
    assert [g["key"] for g in got] == ["img_a", "img_a_meta"]
    # a second shot replaces the latest
    msg, bufs = _put_msg(token, [("img_a", [0], frame * 2, [2, 2, 3], 0)])
    assert srv._handle_put_data(msg, bufs)["ok"]
    assert np.array_equal(srv._handle_get_aux_data({})["items"]["img_a"]["array"], frame * 2)
    reply = srv._handle_end_run({"run_token": token})
    assert reply["ok"] and srv._last_outcome["outcome"] == "nothing_written"
    # the next run starts with no leftovers
    srv._handle_init_run(_init_msg())
    assert srv._handle_get_aux_data({})["items"] == {}


def test_a_saving_run_also_broadcasts_what_it_writes_but_not_slices(server):
    srv, saver = server
    got = []
    srv.aux_data_signal.connect(got.append)
    init = srv._handle_init_run(_init_msg())
    token = init["run_token"]
    msg, bufs = _put_msg(token, [("img_a", [0], np.zeros((2, 3), np.uint8), [2, 2, 3], 0)])
    assert srv._handle_put_data(msg, bufs) == {"ok": True, "queued": 1, "written": True}
    msg, bufs = _put_msg(token, [("big", None, np.zeros((2, 3), np.uint8), [2, 2, 3], 0)])
    msg["items"][0]["offset"] = 0
    assert srv._handle_put_data(msg, bufs)["written"]
    assert [g["key"] for g in got] == ["img_a"]
    assert srv._handle_end_run({"run_token": token})["ok"]


def test_a_mismatched_item_count_is_refused_whole(server):
    srv, saver = server
    init = srv._handle_init_run(_init_msg())
    msg, bufs = _put_msg(init["run_token"], [("x", [0], np.zeros(2), [1, 2], 0)])
    assert srv._handle_put_data(msg, [])["ok"] is False


def test_an_item_that_cannot_be_written_makes_the_run_incomplete(server):
    srv, saver = server
    init = srv._handle_init_run(_init_msg())
    token = init["run_token"]
    # no dataset and nothing to make one from
    msg, bufs = _put_msg(token, [("ghost", [0], np.zeros(2), None, None)])
    assert srv._handle_put_data(msg, bufs)["ok"]
    writer = srv._run_file.writer
    _wait(lambda: writer.put_report.snapshot()[1])
    reply = srv._handle_end_run({"run_token": token})
    assert reply["ok"] and "pushed array" in reply["incomplete"]["reason"]
    assert saver.incomplete["reason"].startswith("1 pushed array(s) not written")


def test_the_message_loop_dispatches_multipart_put_data(server):
    srv, saver = server
    init = srv._handle_init_run(_init_msg())
    msg, bufs = _put_msg(init["run_token"], [("x", [0], np.ones(3, np.uint8), [2, 3], 0)])
    reply = srv._dispatch([pickle.dumps(msg)] + bufs)
    assert reply == {"ok": True, "queued": 1, "written": True}
    assert srv._dispatch([pickle.dumps({"tag": "POLL"})])["ok"]
    assert srv._dispatch([pickle.dumps({"tag": "NOPE"})])["ok"] is False
    assert srv._handle_end_run({"run_token": init["run_token"]})["ok"]


# ----------------------------------------------------------------------
# the save on its own thread
# ----------------------------------------------------------------------

def test_async_end_run_acknowledges_then_reports_saved(server):
    srv, saver = server
    init = srv._handle_init_run(_init_msg())
    token = init["run_token"]
    reply = srv._handle_end_run({"run_token": token, "async_save": True})
    assert reply == {"ok": True, "saving": True}
    _wait(lambda: srv._handle_save_status({"run_token": token})["state"] == "saved")
    st = srv._handle_save_status({"run_token": token})
    assert st["ok"] and st["incomplete"] is None and st["elapsed_s"] >= 0
    _wait(lambda: not srv._run_in_progress)
    assert srv._last_outcome["outcome"] == "saved" and len(saver.saved) == 1


def test_async_end_run_reports_a_failed_save(server):
    srv, saver = server
    saver.fail_save = True
    init = srv._handle_init_run(_init_msg())
    token = init["run_token"]
    assert srv._handle_end_run({"run_token": token, "async_save": True})["saving"]
    _wait(lambda: srv._handle_save_status({"run_token": token})["state"] == "failed")
    st = srv._handle_save_status({"run_token": token})
    assert "drive went away" in st["error"]
    _wait(lambda: not srv._run_in_progress)
    assert srv._last_outcome["outcome"] == "save_failed"


def test_init_run_waits_for_a_save_still_running(server, monkeypatch):
    srv, saver = server
    init = srv._handle_init_run(_init_msg())
    token = init["run_token"]
    real = saver.save_data_from_payload

    def slow(*a, **k):
        time.sleep(0.4)
        return real(*a, **k)
    monkeypatch.setattr(saver, "save_data_from_payload", slow)
    assert srv._handle_end_run({"run_token": token, "async_save": True})["saving"]
    t0 = time.monotonic()
    nxt = srv._handle_init_run(_init_msg())
    assert nxt["ok"] and time.monotonic() - t0 >= 0.3
    assert len(saver.saved) == 1
    assert srv._handle_end_run({"run_token": nxt["run_token"]})["ok"]


def test_a_sync_end_run_from_an_older_client_is_as_before(server):
    srv, saver = server
    init = srv._handle_init_run(_init_msg())
    reply = srv._handle_end_run({"run_token": init["run_token"]})
    assert reply == {"ok": True} and len(saver.saved) == 1


# ----------------------------------------------------------------------
# the writer by itself
# ----------------------------------------------------------------------

def test_the_servers_writer_takes_images_and_pushed_arrays_until_finished(app, tmp_path):
    from waxx.util.live_od.data.image_writer import ImageWriter, PutItem, writer_for
    path = str(tmp_path / "w.hdf5")
    with h5py.File(path, "x") as f:
        f.create_group("data")
    done = threading.Event()
    w = ImageWriter(path, server_owned=True)
    w.start_for_server(n_img=2, on_done=done.set)
    assert writer_for(path) is None             # registration is RunFile's job
    frames = [np.full((3, 4), i + 1, np.uint16) for i in range(2)]
    w.put(frames[0], 0, 1.0)
    assert w.put_data(PutItem("aux", (1,), np.arange(6, dtype=np.uint8), full_shape=(2, 6), fill=0))
    w.put(frames[1], 1, 2.0)
    w.images_done(False)                        # the camera is done; the file stays open
    assert not w.finished and not done.is_set()
    assert w.put_data(PutItem("aux", (0,), np.arange(6, dtype=np.uint8) * 2))
    w.finish()
    assert done.wait(5) and w.finished
    assert not w.put_data(PutItem("aux", (0,), np.zeros(6, np.uint8)))
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["data/images"][1], frames[1])
        assert np.array_equal(f["data/aux"][1], np.arange(6))
        assert np.array_equal(f["data/aux"][0], np.arange(6) * 2)
    written, lost, created = w.put_report.snapshot()
    assert written == {"aux": 2} and lost == [] and created == ["aux"]


def test_a_writer_of_the_data_handlers_own_still_closes_when_the_images_are_in(app, tmp_path):
    from waxx.util.live_od.data.image_writer import ImageWriter
    from PyQt6.QtCore import QObject, pyqtSignal

    class Sig(QObject):
        done = pyqtSignal()
        failed = pyqtSignal(str)
    path = str(tmp_path / "w.hdf5")
    with h5py.File(path, "x") as f:
        f.create_group("data")
    sig = Sig()
    got = threading.Event()
    sig.done.connect(got.set)
    w = ImageWriter(path)
    w.start(n_img=1, check_interrupt_method=lambda: False, done_signal=sig.done,
            failed_signal=sig.failed)
    w.put(np.zeros((2, 2), np.uint8), 0, 0.0)
    w.images_done(False)
    assert w.finished and w.wait_done(5)
    for _ in range(200):                # the handler's signal is queued to this thread
        app.processEvents()
        if got.is_set():
            break
        time.sleep(0.01)
    assert got.is_set()
