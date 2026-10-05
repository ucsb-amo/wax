"""The stream-only path for per-shot auxiliary data (long-run stability,
Phase A): containers that keep no copy in the experiment process, the
bounded shot queue that carries their shots to liveOD, what the streams
count, what END_RUN carries, and the aux_frames_dropped record.

Offline: PUT_DATA goes to an in-memory stand-in (FakePutData), never a
socket; no beacon, no server, no camera.
"""
import json
import pickle
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from waxa.data.camera_frames import N_META
from waxx.base.shot_data_queue import ShotDataQueue
from waxx.config.data_vault import DataVault, HostDataContainer

from test_camera_stream_client import (F, FakeExpt, make_client, make_triggered,
                                       wait_drained, wait_settled)


# ----------------------------------------------------------------------
# stream-only containers
# ----------------------------------------------------------------------

def _vault(xvardims):
    expt = SimpleNamespace(xvardims=list(xvardims))
    return DataVault(expt=expt)


def test_a_stream_only_container_has_the_full_shape_and_no_memory():
    d = _vault([1000])
    d.frame = d.add_host_data_container((480, 640), np.uint8, fill_value=0,
                                        keep_run_data=False)
    d.meta = d.add_host_data_container((N_META,), np.float64, fill_value=np.nan,
                                       keep_run_data=False)
    d.one = d.add_host_data_container((1,), np.int32, fill_value=-1, keep_run_data=False)
    d.init()
    frame = d.frame._run_data
    assert frame.shape == (1000, 480, 640) and frame.dtype == np.uint8
    # one 0-d fill value seen everywhere: 300 MB of shape, one byte of memory
    assert frame.strides == (0, 0, 0) and frame.base.nbytes == 1
    assert not frame.flags.writeable and np.all(frame[17] == 0)
    assert d.meta._run_data.shape == (1000, N_META) and np.isnan(d.meta._run_data).all()
    # trailing size-1 axes squeezed like any container
    assert d.one._run_data.shape == (1000,) and d.one._cell_shape == ()
    assert d.frame.stream_only and d.frame._cell_shape == (480, 640)
    with pytest.raises(RuntimeError, match="stream-only"):
        d.frame.put_shot_data_host((0,), np.zeros((480, 640), np.uint8))


def test_a_kept_host_container_is_unchanged():
    d = _vault([3])
    d.kept = d.add_host_data_container((2,), np.float64, fill_value=np.nan)
    d.init()
    assert not d.kept.stream_only
    d.kept.put_shot_data_host((1,), [1., 2.])
    assert d.kept._run_data.flags.writeable and d.kept._run_data[1, 1] == 2.
    assert np.isnan(d.kept._run_data[0]).all()


# ----------------------------------------------------------------------
# the queue on its own
# ----------------------------------------------------------------------

def _spec(key, slot, arr):
    return {"key": key, "index": (slot,), "array": arr, "full_shape": (10,) + arr.shape,
            "fill": 0}


class Gate:
    """A push that waits for ``release`` and records what it was given."""

    def __init__(self, fail_first=0, raise_=False):
        self.go = threading.Event()
        self.got = []
        self.fail_first = fail_first
        self.raise_ = raise_
        self.attempts = 0

    def __call__(self, specs):
        self.go.wait(10)
        self.attempts += 1
        if self.attempts <= self.fail_first:
            if self.raise_:
                raise ConnectionError("no reply (test)")
            return False
        self.got.append(specs)
        return True


def _outcomes():
    out = []
    return out, (lambda ok, reason: out.append((ok, reason)))


def test_a_full_queue_drops_the_incoming_item_and_keeps_order():
    gate = Gate()
    q = ShotDataQueue(gate, max_bytes=3000, retry_backoff_s=0.01)
    res, done = _outcomes()
    arr = lambda v: np.full(1000, v, np.uint8)
    assert q.put([_spec("k", 0, arr(0))], on_done=done)
    time.sleep(0.1)                         # the sender holds item 0
    assert q.put([_spec("k", 1, arr(1))], on_done=done)
    assert q.put([_spec("k", 2, arr(2))], on_done=done)
    assert not q.put([_spec("k", 3, arr(3))], on_done=done)   # 4000 > 3000
    assert res == [(False, "queue_full")]
    # a clear goes in whatever the queue holds
    assert q.put([_spec("k", 3, arr(0))], clear=True)
    gate.go.set()
    assert q.drain(5)
    assert [s[0]["index"] for s in gate.got] == [(0,), (1,), (2,), (3,)]
    assert [int(s[0]["array"][0]) for s in gate.got[:3]] == [0, 1, 2]
    t = q.report()["keys"]["k"]
    assert t["sent"] == 3 and t["dropped"] == 1 and t["reasons"] == {"queue_full": 1}
    assert t["first_dropped_slots"] == [[3]] and t["clears_sent"] == 1
    assert q.report()["queue_max_bytes"] == 3000 and q.report()["peak_bytes"] >= 3000


def test_queued_arrays_are_copies_counted_at_their_own_size():
    gate = Gate()
    q = ShotDataQueue(gate, max_bytes=10_000)
    sensor = np.zeros((100, 100), np.uint8)
    crop = sensor[10:20, 10:20]              # a view into the 10 kB frame
    assert q.put([_spec("k", 0, crop)])
    assert q.queued_bytes == 100             # the crop, not the sensor frame
    sensor[:] = 9                            # the camera's buffer is reused
    gate.go.set()
    assert q.drain(5)
    sent = gate.got[0][0]["array"]
    assert sent.base is None or sent.base is not sensor
    assert np.all(sent == 0) and sent.flags.c_contiguous


def test_a_refused_push_is_retried_then_succeeds():
    gate = Gate(fail_first=2)
    gate.go.set()
    q = ShotDataQueue(gate, max_bytes=10_000, retries=2, retry_backoff_s=0.01)
    res, done = _outcomes()
    q.put([_spec("k", 0, np.ones(3))], on_done=done)
    assert q.drain(5)
    assert gate.attempts == 3 and res == [(True, None)]
    assert q.report()["keys"]["k"]["sent"] == 1


def test_a_push_that_keeps_failing_is_given_up_and_counted(capsys):
    gate = Gate(fail_first=99, raise_=True)
    gate.go.set()
    q = ShotDataQueue(gate, max_bytes=10_000, retries=2, retry_backoff_s=0.01)
    res, done = _outcomes()
    q.put([_spec("k", 4, np.ones(3)), _spec("k_meta", 4, np.ones(2))], on_done=done)
    assert q.drain(5)
    assert gate.attempts == 3 and res == [(False, "push_failed")]
    rep = q.report()["keys"]
    for key in ("k", "k_meta"):
        assert rep[key]["dropped"] == 1 and rep[key]["reasons"] == {"push_failed": 1}
        assert rep[key]["first_dropped_slots"] == [[4]]
    out = capsys.readouterr().out
    assert "push_failed" in out and "no reply (test)" in out


def test_close_reports_what_was_left_and_refuses_later_items():
    gate = Gate()                            # never released before close
    q = ShotDataQueue(gate, max_bytes=10_000)
    res, done = _outcomes()
    for i in range(3):
        q.put([_spec("k", i, np.ones(4))], on_done=done)
    time.sleep(0.1)
    assert not q.drain(0.1)
    left = q.close(0.1)
    assert left == {"not_sent_at_end": 2, "unconfirmed_at_end": 1}
    assert sorted(r for ok, r in res) == ["not_sent_at_end", "not_sent_at_end",
                                          "unconfirmed_at_end"]
    assert not q.put([_spec("k", 9, np.ones(4))], on_done=done)
    assert res[-1] == (False, "not_sent_at_end")
    gate.go.set()                            # the in-flight push ends after the report
    time.sleep(0.2)
    t = q.report()["keys"]["k"]
    assert t["sent"] == 0 and t["dropped"] == 4
    assert t["reasons"] == {"not_sent_at_end": 3, "unconfirmed_at_end": 1}


def test_a_callback_that_raises_never_reaches_the_caller():
    gate = Gate()
    gate.go.set()
    q = ShotDataQueue(gate, max_bytes=10_000)

    def bad(ok, reason):
        raise ValueError("callback bug")
    assert q.put([_spec("k", 0, np.ones(2))], on_done=bad)
    assert q.drain(5)
    q.drop([_spec("k", 1, np.ones(2))], "no_put_data", on_done=bad)
    assert q.report()["keys"]["k"]["reasons"] == {"no_put_data": 1}


# ----------------------------------------------------------------------
# the streams through the queue
# ----------------------------------------------------------------------

def test_a_frame_is_ok_only_once_liveod_took_it():
    fe, stream, cs = make_triggered()
    fe.data.init()
    gate = threading.Event()
    real = fe.live_od_client.put_data
    fe.live_od_client.put_data = lambda specs: (gate.wait(10), real(specs))[1]
    try:
        cs.announce_trigger_mu(0, 0)
        stream.edge(value=4)
        deadline = time.monotonic() + 5
        while fe.shot_data_queue.n_queued == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        # queued, not yet in the file: not ok, slot not ok
        assert cs.counts["ok"] == 0 and cs._slot_ok[((0,), 0)] is False
        gate.set()
        wait_settled(cs)
        assert cs.counts["ok"] == 1 and cs._slot_ok[((0,), 0)] is True
        assert np.all(F(fe, "img_test")[0] == 4)
    finally:
        gate.set()
        cs.finish()


def test_a_full_queue_costs_the_frame_and_says_why():
    fe, stream, cs = make_triggered()
    fe.PUSH_QUEUE_MAX_BYTES = 1              # anything beyond the item being sent
    fe.data.init()
    gate = threading.Event()
    real = fe.live_od_client.put_data
    fe.live_od_client.put_data = lambda specs: (gate.wait(10), real(specs))[1]
    try:
        cs.announce_trigger_mu(0, 0)
        stream.edge(value=1)
        deadline = time.monotonic() + 5
        while fe.shot_data_queue.n_queued == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        fe.scan_xvars[0].counter = 1
        cs.announce_trigger_mu(0, 0)
        stream.edge(value=2)
        deadline = time.monotonic() + 5
        while not cs.counts.get("queue_full") and time.monotonic() < deadline:
            time.sleep(0.01)
        gate.set()
        wait_settled(cs)
        assert cs.counts["queue_full"] == 1 and cs.counts["ok"] == 1
        assert cs._slot_ok == {((0,), 0): True, ((1,), 0): False}
        assert np.all(F(fe, "img_test")[1] == 0)          # the fill, not a frame
    finally:
        gate.set()
        cs.finish()
    rec = json.loads(fe._extra_file_texts["camera_stream_img_test"])
    assert rec["frames_not_delivered"] == {"queue_full": 1}
    assert rec["n_shots_missing_frame"] == 1 and rec["storage"] == "stream_only"


def test_a_refused_frame_is_failed_and_nothing_is_kept_in_memory():
    fe, stream, cs = make_triggered()
    fe.data.init()
    fe.live_od_client.refuse = lambda specs: True
    fe._shot_queue = ShotDataQueue(fe._push_queued, fe.PUSH_QUEUE_MAX_BYTES,
                                   retries=2, retry_backoff_s=0.01)
    try:
        cs.announce_trigger_mu(0, 0)
        stream.edge(value=5)
        wait_settled(cs)
        assert cs.counts["ok"] == 0 and cs.counts["push_failed"] == 1
        assert cs.counts["failed"] == 1 and cs._n_missing() == 1
        assert len(fe.live_od_client.calls) == 1 + fe.PUSH_QUEUE_RETRIES
        assert fe.data.img_test._run_data.strides == (0, 0, 0)
    finally:
        cs.finish()


def test_a_warm_up_frame_arriving_after_its_slot_was_requested_again_is_not_queued():
    """The warm-up's own frame comes in only after the real shot asked for
    the same slot: it must not land after that slot's clear."""
    fe, stream, cs = make_triggered(t_match_window=2.0)
    fe.data.init()
    try:
        cs.announce_trigger_mu(0, 0)        # warm-up
        cs.announce_trigger_mu(0, 0)        # the real shot, same slot, before any frame
        stream.edge(value=1)                # answers the warm-up's edge
        stream.edge(value=2)                # answers the real one
        wait_settled(cs)
        assert cs.counts["superseded_frames"] == 1
        assert np.all(F(fe, "img_test")[0] == 2)
        assert cs.counts["slot_clears_sent"] == 1 and cs._n_missing() == 0
    finally:
        cs.finish()


def test_a_failed_slot_clear_is_reported_loudly(capsys):
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        # liveOD refuses everything from now on: the clear and the new frame
        fe.live_od_client.refuse = lambda specs: True
        fe.shot_data_queue.retries = 0
        stream.snap_hook = lambda rev: RuntimeError("boom")
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        assert cs.counts["slot_clears_failed"] == 1
    finally:
        cs.finish()
    out = capsys.readouterr().out
    assert "slot clear(s)" in out and "may remain" in out
    fe._close_shot_data_queue()
    rec = json.loads(fe._extra_file_texts["aux_frames_dropped"])
    assert rec["keys"]["img_test"]["clears_failed"] == 1
    assert rec["keys"]["img_test"]["first_clear_failed_slots"] == [[0]]


def test_no_put_data_drops_every_frame_and_says_so_once(capsys):
    fe, stream, cs = make_client()
    fe.live_od_client.enabled = False        # an older liveOD: no PUT_DATA
    fe.data.init()
    try:
        for shot in (0, 1):
            fe.scan_xvars[0].counter = shot
            cs.request_snap_mu(0, 0.0)
            wait_drained(cs)
        assert cs.counts["no_put_data"] == 2 and cs.counts["ok"] == 0
        assert cs._n_missing() == 2 and fe.live_od_client.calls == []
    finally:
        cs.finish()
    out = capsys.readouterr().out
    assert out.count("takes no PUT_DATA") == 1 and "NOT saved" in out
    fe._close_shot_data_queue()
    rec = json.loads(fe._extra_file_texts["aux_frames_dropped"])
    assert rec["keys"]["img_test"]["reasons"] == {"no_put_data": 2}


# ----------------------------------------------------------------------
# the end of the run
# ----------------------------------------------------------------------

def _end_ready(fe):
    """The few attributes _serialize_end_payload reads beyond the fakes."""
    fe.scope_data = SimpleNamespace(_scope_trace_taken=False, scopes=[])
    fe.ds = SimpleNamespace(_read_text_file_safe=lambda *a: "", _expt_params_path="",
                            _base_class_dir="")
    fe.params = SimpleNamespace(N_shots_with_repeats=4)
    return fe


def test_end_run_never_carries_stream_data_even_after_a_failed_push():
    fe, stream, cs = make_client()
    fe.data.small = fe.data.add_data_container(1, np.float64)
    fe.data.init()
    fe.live_od_client.refuse = lambda specs: True
    fe.shot_data_queue.retries = 0
    try:
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
        assert cs.counts["push_failed"] == 1
        # even if something put the key in _push_failed, it stays out of END_RUN
        fe._push_failed.add("img_test")
    finally:
        cs.finish()
    payload = _end_ready(fe)._serialize_end_payload("")
    for key in ("img_test", "img_test_meta"):
        assert payload["datavault"][key] == {"data": None, "data_gotten": True,
                                             "external": True, "final_order": True}
    # a kernel container still travels in END_RUN as before
    assert payload["datavault"]["small"]["data"].shape == (4,)
    assert len(pickle.dumps(payload)) < 100_000


def test_aux_frames_dropped_lists_every_stream_key_with_zeros_too(capsys):
    fe, stream, cs = make_client()
    fe.data.init()
    try:
        cs.request_snap_mu(0, 0.0)
        wait_drained(cs)
    finally:
        cs.finish()
    fe._close_shot_data_queue()
    rec = json.loads(fe._extra_file_texts["aux_frames_dropped"])
    assert rec["queue_max_bytes"] == fe.PUSH_QUEUE_MAX_BYTES
    assert set(rec["keys"]) == {"img_test", "img_test_meta"}
    for t in rec["keys"].values():
        assert t["sent"] == 1 and t["dropped"] == 0 and t["reasons"] == {}
    assert "never reached" not in capsys.readouterr().out
    # a run with no stream-only data writes no record
    fe2 = FakeExpt()
    fe2.data.k = fe2.data.add_host_data_container((2,), np.float64)
    fe2.data.init()
    fe2._close_shot_data_queue()
    assert "aux_frames_dropped" not in fe2._extra_file_texts


def test_leftovers_at_close_are_dropped_and_the_stream_record_corrected(capsys):
    fe, stream, cs = make_client()
    fe.data.init()
    fe.T_PUSH_QUEUE_CLOSE_S = 0.1
    gate = threading.Event()
    real = fe.live_od_client.put_data
    fe.live_od_client.put_data = lambda specs: (gate.wait(10), real(specs))[1]
    import waxx.control.cameras.camera_stream_client as csc
    old, csc.T_QUEUE_DRAIN_S = csc.T_QUEUE_DRAIN_S, 0.1
    try:
        for shot in (0, 1):
            fe.scan_xvars[0].counter = shot
            cs.request_snap_mu(0, 0.0)
            deadline = time.monotonic() + 5
            while cs._busy or not cs._queue.empty():
                if time.monotonic() > deadline:
                    break
                time.sleep(0.01)
        cs.finish()                          # the queue does not drain in time
        first = json.loads(fe._extra_file_texts["camera_stream_img_test"])
        assert first["shots"]["ok"] == 0
        fe._close_shot_data_queue()
    finally:
        csc.T_QUEUE_DRAIN_S = old
        gate.set()
    rec = json.loads(fe._extra_file_texts["camera_stream_img_test"])
    assert rec["frames_not_delivered"] == {"not_sent_at_end": 1, "unconfirmed_at_end": 1}
    assert rec["n_shots_missing_frame"] == 2
    aux = json.loads(fe._extra_file_texts["aux_frames_dropped"])
    assert aux["keys"]["img_test"]["dropped"] == 2
    out = capsys.readouterr().out
    assert "record updated" in out and "never reached the run's file" in out


def test_the_sender_drops_its_socket_on_any_failure_so_a_retry_starts_clean():
    import zmq
    from test_push_data_client import FakeSock, client_with
    from waxx.util.live_od.live_od_client import LiveODDataSender

    class Broken(FakeSock):
        def recv(self):
            raise zmq.ZMQError(156384763, "Operation cannot be accomplished in current state")
    s = LiveODDataSender(client_with(lambda p, rcvtimeo_ms=None: {"ok": True}))
    s._sock = Broken([])
    with pytest.raises(zmq.ZMQError):
        s.put([{"key": "x", "index": (0,), "array": np.zeros(1)}])
    assert s._sock is None


# ----------------------------------------------------------------------
# a non-stream container whose per-shot push failed (A6)
# ----------------------------------------------------------------------

def test_a_big_container_with_a_failed_shot_push_goes_whole_in_slices():
    from test_push_data_client import _expt, _dc
    e = _expt()
    e.BULK_PUSH_BYTES = 1024
    e.PUT_DATA_SLICE_BYTES = 2 * 1024 * 1024
    dc = _dc("big", (4, 1024, 1024), np.uint8)
    for i in range(4):
        dc._run_data[i] = i + 1
    e.live_od_client.put_data = lambda specs: {"ok": False, "error": "busy"}
    assert not e.push_shot_data([(dc, dc._run_data[0])])
    assert "big" in e._push_failed
    e.live_od_client.put_data = lambda specs: (e.sent.append(specs) or {"ok": True})
    assert e._push_whole_container(dc)
    assert "big" not in e._push_failed and dc._pushed
    got = np.concatenate([s[0]["array"] for s in e.sent])
    assert [int(g[0, 0]) for g in got] == [1, 2, 3, 4]
    # a stream-only container is never pushed whole (no copy to push)
    so = SimpleNamespace(key="so", _run_data=np.broadcast_to(np.uint8(0), (4, 1024, 1024)),
                         stream_only=True, _data_gotten=True, _external_data_bool=False)
    assert not e._push_whole_container(so)
