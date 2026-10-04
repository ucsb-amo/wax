"""The whole pushed-data path over real ZMQ on localhost: the server's message
loop (beacon off), the real client's sender from several threads, the polled
END_RUN, and the real end-of-run saver on a temp file.

No beacon is sent, no discovery is done: the client is pointed at the
server's port by hand, so the real liveOD on the lab network is never
touched.
"""
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


class RealSaveFakeReserve(FakeSaver):
    """Reserves run files in the temp folder like FakeSaver; the END_RUN
    save is the real one (pre-allocated containers, unshuffle, finalize)."""

    def __init__(self, folder):
        super().__init__(folder)
        from waxa.data.data_saver import DataSaver
        self._real = DataSaver.__new__(DataSaver)
        self.incomplete = None

    def reserve_run_id_and_path(self, msg):
        run_id = self._next_run_id
        self._next_run_id += 1
        filepath = f"{self.folder}/{run_id:07d}.hdf5"
        with h5py.File(filepath, "x") as f:
            self._real._populate_data_file(f, msg, run_id)
        return run_id, filepath

    def save_data_from_payload(self, msg, filepath, shot_timestamps=None, incomplete=None):
        self.saved.append((filepath, list(shot_timestamps or [])))
        self.payloads.append(msg)
        self.incomplete = incomplete
        self._real.save_data_from_payload(msg, filepath, shot_timestamps=shot_timestamps,
                                          incomplete=incomplete)


@pytest.fixture
def served(app, tmp_path, monkeypatch):
    """A running server loop on a random localhost port, and a client on it."""
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    from waxx.util.live_od.live_od_server import LiveODServer
    from waxx.util.live_od.live_od_client import LiveODClient
    saver = RealSaveFakeReserve(tmp_path)
    srv = LiveODServer(server_talk=None, data_saver=saver)
    monkeypatch.setattr(srv, "_start_beacon", lambda: None)      # never on the lab network
    srv.start()
    t0 = time.monotonic()
    while getattr(srv, "_port", None) in (None, 0):
        assert time.monotonic() - t0 < 10, "server loop did not start"
        time.sleep(0.02)
    c = LiveODClient.__new__(LiveODClient)
    c._ip, c._port, c._timeout_ms = "127.0.0.1", int(srv._port), 5000
    c._context = c._socket = None
    c._run_token, c._run_open, c._exit_notified, c._exit_hook_registered = "", False, False, True
    c._grab_failure_warned = False
    c.last_adjust_values, c.last_reset_requested, c.last_end_run_reply = {}, False, {}
    c.server_features, c._sender = {}, None
    yield srv, saver, c
    if c._sender is not None:
        c._sender.close()
    if c._socket is not None:
        c._socket.close()
    srv.stop()
    srv.wait(5000)


SORT = [3, 0, 4, 1, 2]          # scan position j lands at slot SORT[j]
N = 5


def _init_payload():
    return {
        "save_data": True, "capture_images": False, "camera_key": "", "camera_params": {},
        "run_date_str": "2026-10-03", "run_datetime_str": "2026-10-03 03:00:00",
        "expt_class": "roundtrip", "expt_file": "roundtrip", "imaging_type": 0,
        "save_data_flag": 1, "xvarnames": ["dummy"], "xvardims": [N], "xvar_ranges": {},
        "sort_idx": [SORT], "sort_N": [N], "images_shape": (0,), "images_dtype": "uint16",
        "image_timestamps_shape": (0,),
        "datavault_shapes": {
            "img_mot": {"shape": (N, 8, 6), "dtype": "uint8", "external": False, "fill": 0},
            "img_mot_meta": {"shape": (N, 7), "dtype": "float64", "external": False,
                             "fill": float("nan")},
            "apd": {"shape": (N,), "dtype": "float64", "external": False},
        },
        "params": {"dummy": [0] * N, "N_img": 1}, "N_shots_with_repeats": N,
        "N_pwa_per_shot": 1, "save_on_underflow": 0, "adjust_specs": [],
    }


def _end_payload(apd):
    return {
        "params": {"dummy": [0] * N},
        "datavault": {
            "img_mot": {"data": None, "data_gotten": True, "external": True, "final_order": True},
            "img_mot_meta": {"data": None, "data_gotten": True, "external": True,
                             "final_order": True},
            "apd": {"data": apd, "data_gotten": True, "external": False},
        },
        "sort_idx": [SORT], "sort_N": [N], "xvardims": [N], "N_shots_with_repeats": N,
        "N_pwa_per_shot": 1, "capture_images": False, "scope_data_taken": True,
        "scope_data": [{"label": "PD", "data": None, "pushed": True}],
        "expt_filepath": "", "expt_file_text": "", "params_file_text": "",
        "base_class_texts": {}, "extra_file_texts": {},
    }


def test_a_run_pushes_its_data_over_zmq_and_the_file_is_right(served):
    srv, saver, c = served
    reply = c.init_run(_init_payload())
    assert c.supports("put_data") and c.supports("async_save")
    run_id, path = reply["run_id"], reply["filepath"]

    npts = 250_000                     # a 1 MB float32 trace per shot
    t_axis = np.linspace(0, 1e-3, npts, dtype=np.float32)
    frames = [np.full((8, 6), j + 1, np.uint8) for j in range(N)]
    traces = [np.full((1, npts), float(j + 1), np.float32) for j in range(N)]
    errors = []

    def camera_thread():
        # what the stream worker does: the frame and its record, one message
        try:
            for j in range(N):
                r = c.put_data([
                    {"key": "img_mot", "index": (SORT[j],), "array": frames[j],
                     "full_shape": (N, 8, 6), "fill": 0},
                    {"key": "img_mot_meta", "index": (SORT[j],),
                     "array": np.array([j, 1., 2., 3., 4., 5., 6.]),
                     "full_shape": (N, 7), "fill": float("nan")},
                ])
                assert r == {"ok": True, "queued": 2, "written": True}, r
        except Exception as e:
            errors.append(e)

    def scope_thread():
        # what the scope sender does: t once, then v per shot
        try:
            specs = [{"key": "scope_data/PD/t", "index": None, "array": t_axis,
                      "full_shape": (npts,), "fill": None}]
            assert c.put_data(specs)["ok"]
            for j in range(N):
                r = c.put_data([{"key": "scope_data/PD/v", "index": (SORT[j],),
                                 "array": traces[j], "full_shape": (N, 1, npts),
                                 "fill": float("nan")}])
                assert r["ok"], r
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=camera_thread), threading.Thread(target=scope_thread)]
    for th in threads:
        th.start()
    for j in range(N):                  # the kernel's own channel meanwhile
        assert c.shot_complete(j, N, {"dummy": 0}) is False
    for th in threads:
        th.join(30)
    assert not errors and not any(th.is_alive() for th in threads)

    import waxx.util.live_od.live_od_client as m
    m.SAVE_POLL_S = 0.05
    t0 = time.monotonic()
    apd = np.arange(N, dtype=float) * 10
    assert c.end_run(_end_payload(apd)) is True
    assert c.last_end_run_reply.get("incomplete") is None
    assert time.monotonic() - t0 < 10
    assert saver.incomplete is None and [p for p, _ in saver.saved] == [path]

    with h5py.File(path, "r") as f:
        assert f.attrs["run_complete"] and f.attrs["data_complete"] and f.attrs["unshuffle_applied"]
        for j in range(N):
            assert np.all(f["data/img_mot"][SORT[j]] == j + 1)
            assert f["data/img_mot_meta"][SORT[j]][0] == j
            assert np.all(f["data/scope_data/PD/v"][SORT[j]] == j + 1)
            assert f["data/apd"][SORT[j]] == apd[j]          # unshuffled by the saver
        assert np.array_equal(f["data/scope_data/PD/t"][()], t_axis)
        v = f["data/scope_data/PD/v"]
        assert v.shape == (N, 1, npts) and v.compression == "gzip" and v.chunks == (1, 1, npts)
        assert v.id.get_storage_size() < v.nbytes / 10         # flat traces compress well
        assert f["data/scope_data/PD/t"].compression == "gzip"
    assert srv._last_outcome["outcome"] == "saved" and not srv._run_in_progress


def test_an_old_style_end_run_still_works_and_a_late_push_is_refused(served):
    srv, saver, c = served
    c.init_run(_init_payload())
    c.server_features = {}                      # the client of an older checkout
    assert c.end_run(_end_payload(np.zeros(N))) is True
    assert [p for p, _ in saver.saved] and srv._last_outcome["outcome"] == "saved"
    # the run is over: a push now has no run to go to
    r = c.put_data([{"key": "img_mot", "index": (0,), "array": np.zeros((8, 6), np.uint8)}])
    assert r["ok"] is False and "no run" in r["error"]
