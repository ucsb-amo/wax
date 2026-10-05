"""The experiment's side of data pushed during the run: the client's sender
and polled END_RUN, Expt's final-slot mapping and END_RUN payload, the camera
stream's push, the scope sender. No socket is opened anywhere here."""
import pickle
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from waxa.data.camera_frames import N_META


# ----------------------------------------------------------------------
# LiveODClient: the sender and the polled END_RUN
# ----------------------------------------------------------------------

class FakeSock:
    def __init__(self, replies):
        self.sent = []
        self.replies = list(replies)

    def send_multipart(self, frames, copy=True):
        self.sent.append([bytes(f) if not isinstance(f, bytes) else f for f in frames])

    def recv(self):
        return pickle.dumps(self.replies.pop(0))

    def close(self):
        pass


def client_with(transport, features=None):
    """A LiveODClient whose request/reply is ``transport(payload, rcvtimeo_ms)``."""
    from waxx.util.live_od.live_od_client import LiveODClient
    c = LiveODClient.__new__(LiveODClient)
    c._ip, c._port = "127.0.0.1", 1
    c._run_token = "tok"
    c.server_features = dict(features or {})
    c._sender = None
    c._run_open = True
    c.last_end_run_reply = {}
    c._send_recv = transport
    return c


def test_the_sender_frames_arrays_raw_and_names_their_slots():
    from waxx.util.live_od.live_od_client import LiveODDataSender
    c = client_with(lambda p, rcvtimeo_ms=None: {"ok": True})
    s = LiveODDataSender(c)
    sock = FakeSock([{"ok": True, "queued": 2}])
    s._socket = lambda: sock
    frame = np.arange(6, dtype=np.uint8).reshape(2, 3)[:, ::2]      # not contiguous
    reply = s.put([{"key": "img", "index": (1, 0), "array": frame, "full_shape": (2, 2, 2, 2),
                    "fill": 0},
                   {"key": "s/v", "offset": 4, "index": None, "array": np.zeros(3, np.float32),
                    "full_shape": None, "fill": None}])
    assert reply == {"ok": True, "queued": 2}
    head = pickle.loads(sock.sent[0][0])
    assert head["tag"] == "PUT_DATA" and head["run_token"] == "tok"
    assert head["items"][0] == {"key": "img", "index": [1, 0], "offset": None, "shape": [2, 2],
                                "dtype": "|u1", "full_shape": [2, 2, 2, 2], "fill": 0}
    assert head["items"][1]["offset"] == 4 and head["items"][1]["index"] is None
    assert sock.sent[0][1] == np.ascontiguousarray(frame).tobytes()
    assert len(sock.sent[0]) == 3


def test_put_data_without_a_reply_raises_connection_error_and_drops_the_socket():
    import zmq
    from waxx.util.live_od.live_od_client import LiveODDataSender

    class Dead(FakeSock):
        def recv(self):
            raise zmq.Again()
    c = client_with(lambda p, rcvtimeo_ms=None: {"ok": True})
    s = LiveODDataSender(c)
    s._sock = Dead([])
    with pytest.raises(ConnectionError):
        s.put([{"key": "x", "index": (0,), "array": np.zeros(1)}])
    assert s._sock is None


def test_end_run_polls_the_servers_save_when_it_can():
    sent = []
    status = iter([{"ok": True, "state": "saving", "phase": "writer"},
                   {"ok": True, "state": "saving", "phase": "write"},
                   {"ok": True, "state": "saved", "incomplete": None, "elapsed_s": 1.5}])

    def transport(payload, rcvtimeo_ms=None):
        sent.append((dict(payload), rcvtimeo_ms))
        if payload["tag"] == "END_RUN":
            return {"ok": True, "saving": True}
        return next(status)
    import waxx.util.live_od.live_od_client as m
    m.SAVE_POLL_S = 0.0
    c = client_with(transport, features={"async_save": True, "put_data": True})
    assert c.end_run({}) is True
    assert sent[0][0]["tag"] == "END_RUN" and sent[0][0]["async_save"] is True
    assert sent[0][1] == 60_000
    assert [p["tag"] for p, _ in sent[1:]] == ["SAVE_STATUS"] * 3
    assert c.last_end_run_reply["save_s"] == 1.5 and c._run_open is False


def test_end_run_against_an_older_server_is_the_one_long_request():
    sent = []

    def transport(payload, rcvtimeo_ms=None):
        sent.append((dict(payload), rcvtimeo_ms))
        return {"ok": True}
    c = client_with(transport)
    assert c.end_run({}) is True
    assert len(sent) == 1 and "async_save" not in sent[0][0] and sent[0][1] == 600_000


def test_a_failed_asynchronous_save_raises_like_a_failed_end_run():
    def transport(payload, rcvtimeo_ms=None):
        if payload["tag"] == "END_RUN":
            return {"ok": True, "saving": True}
        return {"ok": True, "state": "failed", "error": "drive went away"}
    import waxx.util.live_od.live_od_client as m
    m.SAVE_POLL_S = 0.0
    c = client_with(transport, features={"async_save": True})
    with pytest.raises(RuntimeError, match="drive went away"):
        c.end_run({})


def test_a_save_that_stops_making_progress_is_given_up():
    def transport(payload, rcvtimeo_ms=None):
        if payload["tag"] == "END_RUN":
            return {"ok": True, "saving": True}
        return {"ok": True, "state": "saving", "phase": "write"}
    import waxx.util.live_od.live_od_client as m
    m.SAVE_POLL_S = 0.0
    old = m.SAVE_STALL_S
    m.SAVE_STALL_S = 0.05
    try:
        c = client_with(transport, features={"async_save": True})
        with pytest.raises(RuntimeError, match="phase 'write'"):
            c.end_run({})
    finally:
        m.SAVE_STALL_S = old


# ----------------------------------------------------------------------
# Expt: the final slot, the pushes, the END_RUN payload
# ----------------------------------------------------------------------

def _expt(sort_idx=(), xvardims=(4,), push=True):
    """A stand-in with Expt's push methods bound to it."""
    from waxx.base.expt import Expt
    e = SimpleNamespace()
    e.sort_idx = [np.array(s) for s in sort_idx]
    e.sort_N = [len(s) for s in sort_idx]
    e.xvardims = list(xvardims)
    e.scan_xvars = [SimpleNamespace(counter=0) for _ in xvardims]
    e.run_info = SimpleNamespace(save_data=True)
    e.live_od_client = SimpleNamespace(
        supports=lambda f: push, put_data=lambda specs: {"ok": True}, last_end_run_reply={})
    e.sent = []
    e.live_od_client.put_data = lambda specs: (e.sent.append(specs) or {"ok": True})
    e._push_failed, e._n_pushed, e._n_push_failed = set(), 0, 0
    for name in ("current_shot_index", "final_shot_index", "push_shot_data", "push_raw",
                 "_push", "_note_push_failure", "_push_whole_container"):
        setattr(e, name, getattr(Expt, name).__get__(e))
    e.push_data_enabled = Expt.push_data_enabled.fget(e)
    e.BULK_PUSH_BYTES = Expt.BULK_PUSH_BYTES
    e.PUT_DATA_SLICE_BYTES = Expt.PUT_DATA_SLICE_BYTES
    return e


def _dc(key, shape, dtype=np.float64, fill=0):
    return SimpleNamespace(key=key, _run_data=np.full(shape, fill, dtype), _fill_value=fill,
                           _pushed=False, _data_gotten=True, _external_data_bool=False)


def test_final_shot_index_is_the_slot_the_saver_unshuffles_into():
    from waxa.data.data_saver import DataSaver
    rng = np.random.default_rng(3)
    perm0, perm1 = rng.permutation(5), rng.permutation(3)
    e = _expt(sort_idx=(perm0, perm1), xvardims=(5, 3))
    shuffled = rng.random((5, 3, 2))          # (*xvardims, per shot)
    final = DataSaver._unshuffle_single_array(shuffled, [perm0.tolist(), perm1.tolist()],
                                              [5, 3], exclude_dims=1)
    for i in range(5):
        for j in range(3):
            assert np.array_equal(final[e.final_shot_index((i, j))], shuffled[i, j])
    assert _expt(xvardims=(4,)).final_shot_index((2,)) == (2,)


def test_push_shot_data_sends_the_shots_slot_and_marks_the_container():
    e = _expt(sort_idx=([3, 0, 1, 2],))
    e.scan_xvars[0].counter = 1
    img, meta = _dc("img", (4, 2, 2), np.uint8), _dc("img_meta", (4, 3), fill=np.nan)
    assert e.push_shot_data([(img, np.ones((2, 2), np.uint8)), (meta, np.zeros(3))])
    (specs,) = e.sent
    assert [s["key"] for s in specs] == ["img", "img_meta"]
    assert specs[0]["index"] == (0,) and specs[0]["full_shape"] == (4, 2, 2)
    assert np.isnan(specs[1]["fill"])
    assert img._pushed and meta._pushed and e._n_pushed == 2


def test_a_refused_push_sends_the_container_with_end_run_instead(capsys):
    e = _expt()
    e.live_od_client.put_data = lambda specs: {"ok": False, "error": "no run"}
    dc = _dc("img", (4, 2))
    assert not e.push_shot_data([(dc, np.zeros(2))])
    assert "img" in e._push_failed and not dc._pushed and e._n_push_failed == 1
    assert "did not take" in capsys.readouterr().out
    e2 = _expt(push=False)
    assert not e2.push_shot_data([(dc, np.zeros(2))]) and e2.sent == []


def test_a_big_unpushed_container_goes_in_unshuffled_slices_before_end_run():
    perm = [2, 0, 3, 1]
    e = _expt(sort_idx=(perm,))
    e.PUT_DATA_SLICE_BYTES = 3 * 1024 * 1024
    e.BULK_PUSH_BYTES = 1024
    dc = _dc("big", (4, 1024, 1024), np.uint8)           # 4 MB: two slices of 3 and 1 rows
    for i in range(4):
        dc._run_data[i] = i + 1
    assert e._push_whole_container(dc) and dc._pushed
    assert [s[0]["offset"] for s in e.sent] == [0, 3]
    assert [len(s[0]["array"]) for s in e.sent] == [3, 1]
    got = np.concatenate([s[0]["array"] for s in e.sent])
    # row j of the shuffled array lands at perm[j]
    for j in range(4):
        assert got[perm[j]][0, 0] == j + 1
    small = _dc("small", (4, 2))
    assert not e._push_whole_container(small) and not small._pushed


# ----------------------------------------------------------------------
# the camera stream pushes each frame it stores
# ----------------------------------------------------------------------

def test_a_stored_frame_is_pushed_with_its_record():
    from test_camera_stream_client import make_triggered, wait_settled
    fe, stream, cs = make_triggered()
    fe.data.init()
    try:
        cs.announce_trigger_mu(0, 0)
        stream.edge(value=7)
        wait_settled(cs)
        assert cs.counts["ok"] == 1
        # the frame and its record in one PUT_DATA message, at the shot's slot
        (call,) = fe.live_od_client.calls
        assert call == [("img_test", (0,)), ("img_test_meta", (0,))]
        (_, _, frame), (_, _, row) = fe.live_od_client.history
        assert np.all(frame == 7) and row.shape == (N_META,) and row[0] > 0
    finally:
        cs.finish()


def test_a_failed_push_is_counted_and_no_frame_kept_in_memory():
    """Stream frames have no in-memory copy and no END_RUN fallback: a frame
    liveOD did not take is failed (with its reason), not ok."""
    from test_camera_stream_client import make_triggered, wait_settled
    fe, stream, cs = make_triggered()
    fe.data.init()
    fe.live_od_client.refuse = lambda specs: True
    fe.shot_data_queue.retry_backoff_s = 0.01
    try:
        cs.announce_trigger_mu(0, 0)
        stream.edge(value=5)
        wait_settled(cs)
        assert cs.counts.get("push_failed") == 1 and cs.counts["ok"] == 0
        assert cs._n_missing() == 1
        assert fe.data.img_test._run_data.strides == (0, 0, 0)
    finally:
        cs.finish()


# ----------------------------------------------------------------------
# the scope sender
# ----------------------------------------------------------------------

def _scope_expt():
    e = SimpleNamespace(push_data_enabled=True, xvardims=[3], sent=[])
    e.current_shot_index = lambda: (1,)
    e.final_shot_index = lambda idx: (2,)
    e.push_raw = lambda specs: (e.sent.append(specs) or True)
    return e


class _Scope:
    """A scope with the real trace store and push flags, no instrument."""

    def __init__(self, label="PD"):
        from waxx.control.misc.oscilloscopes import ScopeTraces
        self.label = label
        self.traces = ScopeTraces()
        self._t_pushed = False
        self.pushed_any = False
        self.push_failed = False

    @property
    def t_varies(self):
        return self.traces.t_varies

    def store(self, arr):
        self.traces.store(arr)


def _traces(t, *vs):
    """``(channels, 2, points)`` float32 from one time axis and per-channel v."""
    return np.stack([np.stack([t, v]) for v in vs]).astype(np.float32)


def test_each_shots_traces_go_in_at_their_slot_and_the_time_axis_once():
    from waxx.control.misc.oscilloscopes import ScopeData
    sd = ScopeData()
    e = _scope_expt()
    sd.attach_expt(e)
    sc = _Scope()
    t = np.linspace(0, 1, 5, dtype=np.float32)
    arr = _traces(t, t * 2, t * 3)                  # 2 channels
    for _ in range(2):
        sc.store(arr)
        sd.push(sc, arr)
    sd.close()
    assert len(e.sent) == 2 and sc.pushed_any and not sc.push_failed and not sc.t_varies
    first, second = e.sent
    assert [s["key"] for s in first] == ["scope_data/PD/t", "scope_data/PD/v"]
    assert first[0]["index"] is None and first[0]["full_shape"] == (5,)
    assert np.array_equal(first[0]["array"], t)
    assert first[1]["index"] == (2,) and first[1]["full_shape"] == (3, 2, 5)
    assert np.array_equal(first[1]["array"], arr[:, 1, :])
    assert [s["key"] for s in second] == ["scope_data/PD/v"]


def test_a_changed_time_axis_hands_the_traces_back_to_end_run():
    from waxx.control.misc.oscilloscopes import ScopeData, GenericWaxxScope
    sd = ScopeData()
    e = _scope_expt()
    sd.attach_expt(e)
    sc = _Scope()
    t = np.linspace(0, 1, 5, dtype=np.float32)
    for arr in (_traces(t, t), _traces(t * 2, t)):
        sc.store(arr)
        sd.push(sc, arr)
    sd.close()
    assert sc.t_varies and sc.pushed_any
    assert GenericWaxxScope.pushed_ok.fget(sc) is False
    # END_RUN's fallback gets every shot's own axis back
    full = sc.traces.full()
    assert np.array_equal(full[0, 0, 0], t) and np.array_equal(full[1, 0, 0], t * 2)


def test_nothing_is_queued_when_the_run_does_not_push():
    from waxx.control.misc.oscilloscopes import ScopeData
    sd = ScopeData()
    e = _scope_expt()
    e.push_data_enabled = False
    sd.attach_expt(e)
    sd.push(_Scope(), np.zeros((1, 2, 3), np.float32))
    assert sd._sender is None
    sd.close()


def test_scope_traces_keep_one_time_axis_and_give_the_full_array_back():
    from waxx.control.misc.oscilloscopes import ScopeTraces
    tr = ScopeTraces()
    t = np.linspace(0, 1, 4, dtype=np.float32)
    assert tr.store(_traces(t, t + 1, t + 2)) == 0
    assert tr.store(_traces(t, t + 3, t + 4)) == 1
    assert tr.n == 2 and not tr.t_varies and tr.t_extra == {}
    assert tr.t_ref.shape == (2, 4) and len(tr.v) == 2 and tr.v[1].shape == (2, 4)
    full = tr.full()
    assert full.shape == (2, 2, 2, 4) and full.dtype == np.float32
    assert np.array_equal(full[1, 1, 1], t + 4) and np.array_equal(full[1, 1, 0], t)
    tr.pad(4)                                        # save_on_underflow: zero shots, zero axes
    assert tr.n == 4 and tr.t_varies and np.all(tr.full()[3] == 0)
    tr.clear()
    assert tr.n == 0 and tr.t_ref is None and tr.full().shape == (0, 0, 2, 0)


def test_a_run_that_saves_nothing_still_pushes_but_sends_no_bulk_at_the_end():
    e = _expt()
    e.run_info.save_data = False
    assert e.push_data_enabled
    dc = _dc("img", (4, 2), np.uint8)
    assert e.push_shot_data([(dc, np.zeros(2, np.uint8))]) and dc._pushed
    e.BULK_PUSH_BYTES = 1
    big = _dc("big", (4, 1024), np.uint8)
    assert not e._push_whole_container(big) and len(e.sent) == 1
