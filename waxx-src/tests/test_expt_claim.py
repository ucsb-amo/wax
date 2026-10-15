"""waxx Expt: the claim (queue phase 2) and the source-text snapshot (M0).

Host-only, offline: a stand-in object with the real Expt methods bound, a
recording fake liveOD client and monitor, queue_gate with a stand-in
transport that gives GO at once. No network, no servers, no core device.
"""
import json
import os
from types import SimpleNamespace

import pytest

from waxa.data.data_saver import DataSaver
from waxx.base.expt import Expt
from waxx.util.device_state import queue_gate


class _GoAtOnce:
    """The run queue as queue_gate sees it: GO at the first ask (reset_count
    unknown).  Never the real monitor server."""

    def request(self, obj, timeout=None, attempts=None):
        return {"status": "ok", "go": obj.get("stage") != "warming", "reset_count": None}


@pytest.fixture(autouse=True)
def _fresh_gate(monkeypatch):
    monkeypatch.delenv("WAXX_QUEUE_WARM", raising=False)
    queue_gate._reset_for_tests()
    queue_gate.set_transport(_GoAtOnce())
    yield
    queue_gate._reset_for_tests()


class LiveODStartRefused(RuntimeError):
    """Stands in for D's client exception (a RuntimeError)."""


RESERVED = {"ok": True, "reservation": "r9", "features": {"reserve": True, "put_data": True},
            "warnings": ["w"]}
REFUSED = {"ok": False, "refused": True, "error": "another process holds the slot",
           "holder": {"pid": 1}, "features": {"reserve": True, "put_data": True}}
UNKNOWN = {"ok": False, "unknown_tag": True, "error": "Unknown tag: RESERVE_RUN (old liveOD)"}


class FakeLiveOD:
    """LiveODClient as Expt uses it (D's API: reserve/release never raise,
    start_run raises LiveODStartRefused on a refusal, an old server answers
    unknown_tag)."""

    def __init__(self, events, has_phase2=True, start_reply=None, reserve_reply=None,
                 poll_reply=None, start_raises=None):
        self.events = events
        self.server_features = {}
        self._start_reply = start_reply
        self._reserve_reply = reserve_reply
        self._poll_reply = poll_reply
        self._start_raises = start_raises
        if not has_phase2:
            self.reserve_run = self.start_run = None
        self.released = []
        self.init_payload_times = []

    def supports(self, f):
        return bool(self.server_features.get(f))

    def poll(self):
        self.events.append("poll")
        if isinstance(self._poll_reply, Exception):
            raise self._poll_reply
        return self._poll_reply or {"ok": True, "features": {}}

    def init_run(self, payload):
        self.events.append("init_run")
        self.init_payload_times.append(payload.get("run_datetime_str"))
        self.server_features = {"put_data": True}
        return {"ok": True, "run_id": 7, "filepath": "x.hdf5"}

    def reserve_run(self, payload, ttl_s=900.0):
        self.events.append("reserve_run")
        r = self._reserve_reply if self._reserve_reply is not None else RESERVED
        if isinstance(r, Exception):            # (D's never raises; an older client might)
            raise r
        return r

    def start_run(self, payload, *, reservation, require_features=(), go_reset_count=None):
        self.events.append(("start_run", reservation, tuple(require_features), go_reset_count))
        if self._start_raises is not None:
            raise self._start_raises
        if self._start_reply is not None:
            return self._start_reply
        return {"ok": True, "run_id": 8, "filepath": "y.hdf5",
                "run_date_str": "2026-10-14", "run_datetime_str": "2026-10-14_09-08-07"}

    def release_reservation(self, r):
        self.released.append(r)
        return {"ok": True}


class Stand(SimpleNamespace):
    pass


_METHODS = ("claim", "claim_hardware", "claim_after_start", "pre_init_run",
            "_start_liveod_run", "_adopt_start_time", "_reserve_warm_run",
            "_release_reservation", "_liveod_has_reserve", "_restamp_run_time_now",
            "_decide_subscan", "_compile_subscan_switch",
            "_expt_file_stem", "_serialize_init_payload", "_xvar_ranges",
            "_source_texts_for_end", "_source_files", "_read_source",
            "_snapshot_source_texts", "_expt_file_path")


def _stand(events, client):
    from waxa.data.run_info import RunInfo
    e = Stand()
    for n in _METHODS:
        setattr(e, n, getattr(Expt, n).__get__(e))
    e.push_data_enabled = True
    e._claimed = False
    e._run_started = False
    e._reservation = None
    e._liveod_features = None
    e._subscan = None
    e._extra_file_texts = {}
    e.live_od_client = client
    e.run_info = RunInfo(None, save_data=False)
    e.claim_hardware = lambda: events.append("hardware")
    e.claim_after_start = lambda: events.append("after_start")
    e._serialize_init_payload = lambda: {"payload": True,
                                         "run_datetime_str": e.run_info.run_datetime_str}
    e._expt_file_stem = lambda: "stand"
    e.monitor = SimpleNamespace(announce_run=lambda **kw: events.append(("announce",
                                                                         kw["run_id"])))
    return e


@pytest.fixture
def warm(monkeypatch):
    monkeypatch.setenv("WAXX_QUEUE_WARM", "tok")
    assert queue_gate.is_warm()


def _stale(e):
    """The warm process's start time, long before the claim."""
    e.run_info.run_date_str = "2026-01-01"
    e.run_info.run_datetime_str = "2026-01-01_00-00-00"


def _starts(ev):
    return [x for x in ev if isinstance(x, tuple) and x[0] == "start_run"]


def test_cold_claim_order_and_once():
    ev = []
    e = _stand(ev, FakeLiveOD(ev))
    e.claim()
    e.claim()
    assert ev == ["hardware", "init_run", "after_start", ("announce", 7)]
    assert e.run_info.run_id == 7 and e._run_started


def test_the_stamp_records_the_trust_from_before_the_runs_own_announcement():
    """The monitor server marks the device state untrusted on a run's
    announcement (run_pending, M0): the stamp (claim_after_start) is taken
    before the announcement, so it records the state the run starts from,
    not the run's own "untrusted"."""
    server = {"trusted": True, "reason": "end state of the previous run"}
    ev = []
    e = _stand(ev, FakeLiveOD(ev))

    def announce_run(**kw):
        ev.append(("announce", kw["run_id"]))
        server.update(trusted=False, reason=f"run {kw['run_id']} announced")

    e.monitor = SimpleNamespace(announce_run=announce_run)
    stamped = []
    e.claim_after_start = lambda: stamped.append(dict(server))
    e.claim()
    assert stamped == [{"trusted": True, "reason": "end state of the previous run"}]
    assert server["trusted"] is False                   # the announcement did go out


def test_cold_never_uses_start_run_even_with_a_reservation():
    ev = []
    e = _stand(ev, FakeLiveOD(ev))
    e._reservation = "r1"
    e.claim()
    assert "init_run" in ev and _starts(ev) == []


def test_warm_reserve_then_start_with_the_reservation_and_server_time(warm, capsys):
    ev = []
    client = FakeLiveOD(ev)
    e = _stand(ev, client)
    e._reserve_warm_run()
    assert ev == ["reserve_run"] and e._reservation == "r9"
    assert "reservation warning: w" in capsys.readouterr().out      # printed once, by Expt
    e.claim()
    assert ev[1:] == ["hardware", ("start_run", "r9", ("reserve",), None), "after_start",
                      ("announce", 8)]
    assert e.run_info.run_id == 8 and e._reservation is None
    assert e.run_info.run_datetime_str == "2026-10-14_09-08-07"
    assert e.run_info.run_date_str == "2026-10-14"
    assert e.run_info.run_datetime.tm_hour == 9
    e._release_reservation()                     # consumed: nothing to release
    assert client.released == []


def test_warm_reserve_refused_still_starts_with_start_run_and_no_reservation(warm):
    ev = []
    e = _stand(ev, FakeLiveOD(ev, reserve_reply=REFUSED))
    e._reserve_warm_run()
    assert e._reservation is None
    e.claim()
    assert _starts(ev) == [("start_run", None, ("reserve",), None)]
    assert "init_run" not in ev and "poll" not in ev
    assert e.run_info.run_datetime_str == "2026-10-14_09-08-07"      # liveOD's start time


def test_warm_reserve_failed_asks_poll_for_the_feature(warm):
    ev = []
    e = _stand(ev, FakeLiveOD(ev, reserve_reply=ConnectionError("no reply"),
                              poll_reply={"ok": True, "features": {"reserve": True}}))
    e._reserve_warm_run()
    e.claim()
    assert "poll" in ev and _starts(ev) == [("start_run", None, ("reserve",), None)]
    assert "init_run" not in ev


def test_warm_without_the_reserve_feature_restamps_then_init_run(warm):
    ev = []
    client = FakeLiveOD(ev, reserve_reply=ConnectionError("no reply"),
                        poll_reply={"ok": True, "features": {"put_data": True}})
    e = _stand(ev, client)
    e._reserve_warm_run()
    _stale(e)
    e.claim()
    assert _starts(ev) == [] and "init_run" in ev
    assert client.init_payload_times[0] != "2026-01-01_00-00-00"     # re-stamped before it
    assert e.run_info.run_date_str != "2026-01-01"
    assert e.run_info.run_datetime_str == client.init_payload_times[0]


def test_warm_against_an_old_liveod_restamps_then_init_run(warm):
    ev = []
    client = FakeLiveOD(ev, reserve_reply=UNKNOWN)
    e = _stand(ev, client)
    e._reserve_warm_run()
    assert e._reservation is None and e._liveod_features == {}
    _stale(e)
    e.claim()
    assert _starts(ev) == [] and "poll" not in ev and "init_run" in ev
    assert client.init_payload_times[0] != "2026-01-01_00-00-00"


def test_warm_with_an_old_client_without_the_methods_gets_init_run(warm):
    ev = []
    client = FakeLiveOD(ev, has_phase2=False)
    e = _stand(ev, client)
    e._reserve_warm_run()
    _stale(e)
    e.claim()
    assert "init_run" in ev and e.run_info.run_id == 7
    assert e.run_info.run_date_str != "2026-01-01"


def test_warm_start_unknown_tag_falls_back_to_a_restamped_init_run(warm):
    ev = []
    client = FakeLiveOD(ev, start_reply=dict(UNKNOWN, error="Unknown tag: START_RUN"))
    e = _stand(ev, client)
    e._reserve_warm_run()
    _stale(e)
    e.claim()
    assert len(_starts(ev)) == 1 and "init_run" in ev
    assert client.init_payload_times[0] != "2026-01-01_00-00-00"


def test_a_start_refusal_propagates_before_the_announce(warm):
    ev = []
    e = _stand(ev, FakeLiveOD(ev, start_raises=LiveODStartRefused("reset since GO")))
    e._reserve_warm_run()
    with pytest.raises(LiveODStartRefused, match="reset since GO"):
        e.claim()
    assert not any(isinstance(x, tuple) and x[0] == "announce" for x in ev)
    assert "init_run" not in ev and e.run_info.run_id == 0


def test_a_non_raising_refusal_reply_is_a_refusal_too(warm):
    ev = []
    e = _stand(ev, FakeLiveOD(ev, start_reply={"ok": False, "error": "a run is in progress"}))
    e._reserve_warm_run()
    with pytest.raises(RuntimeError, match="START_RUN refused: a run is in progress"):
        e.claim()
    assert "init_run" not in ev


def test_an_unused_reservation_is_released():
    ev = []
    client = FakeLiveOD(ev)
    e = _stand(ev, client)
    e._reserve_warm_run()
    e._release_reservation()
    assert client.released == ["r9"]


def test_claim_waits_for_go_first(monkeypatch):
    ev = []
    e = _stand(ev, FakeLiveOD(ev))
    monkeypatch.setenv("WAXX_QUEUE_WARM", "tok")
    assert queue_gate.is_warm() and not queue_gate.go_passed()
    real = queue_gate.ensure_go
    monkeypatch.setattr(queue_gate, "ensure_go", lambda: (ev.append("go"), real()))
    e.claim()
    assert ev[0] == "go" and ev[1] == "hardware"


def test_the_kernel_switch_is_fixed_before_the_claim():
    ev = []
    e = _stand(ev, FakeLiveOD(ev))
    e.sort_idx = []
    sub = SimpleNamespace(enabled=True, require_push=True, quiet=False, name="cal",
                          disabled_reason="")
    sub.disable = lambda why: (setattr(sub, "enabled", False),
                               setattr(sub, "disabled_reason", why))
    sub.describe = lambda: "cal"
    e._subscan = sub
    e._compile_subscan_switch()
    assert e._subscan_enabled
    e.push_data_enabled = False
    e._decide_subscan()                           # after the start: disabled, switch untouched
    assert not sub.enabled and e._subscan_enabled


# ---- the source-text snapshot (M0) ------------------------------------------

def _src_stand(tmp_path):
    expt = tmp_path / "my_expt.py"
    params = tmp_path / "params.py"
    base = tmp_path / "base"
    base.mkdir()
    expt.write_text("expt v1\n")
    params.write_text("params v1\n")
    (base / "cooling.py").write_text("cooling v1\n")
    (base / "image.py").write_text("image v1\n")
    e = Stand()
    for n in ("_source_texts_for_end", "_source_files", "_read_source",
              "_snapshot_source_texts"):
        setattr(e, n, getattr(Expt, n).__get__(e))
    e._expt_file_path = lambda: str(expt)
    e._extra_file_texts = {}
    e._source_snapshot = {}
    e.ds = DataSaver.__new__(DataSaver)          # no server_talk: the reader only
    e.ds._expt_params_path = str(params)
    e.ds._base_class_dir = str(base)
    e._snapshot_source_texts()
    return e, expt, params, base


def test_unchanged_sources_are_saved_and_nothing_is_flagged(tmp_path):
    e, expt, params, base = _src_stand(tmp_path)
    t_expt, t_params, t_base = e._source_texts_for_end(str(expt))
    assert t_expt == "expt v1\n" and t_params == "params v1\n"
    assert t_base == {"base_class_cooling": "cooling v1\n", "base_class_image": "image v1\n"}
    assert "source_changed_during_run" not in e._extra_file_texts


def test_a_file_edited_during_the_run_keeps_its_start_text_and_is_flagged(tmp_path, capsys):
    e, expt, params, base = _src_stand(tmp_path)
    params.write_text("params v2 (a write-back)\n")
    (base / "image.py").unlink()
    (base / "new_mixin.py").write_text("new\n")
    t_expt, t_params, t_base = e._source_texts_for_end(str(expt))
    assert t_params == "params v1\n"                       # what ran
    assert t_base["base_class_image"] == "image v1\n"
    assert t_base["base_class_new_mixin"] == "new\n"
    changed = {os.path.basename(c["file"]): c
               for c in json.loads(e._extra_file_texts["source_changed_during_run"])}
    assert set(changed) == {"params.py", "image.py", "new_mixin.py"}
    assert changed["params.py"]["sha256_start"] != changed["params.py"]["sha256_end"]
    assert changed["image.py"]["sha256_end"] is None
    assert changed["new_mixin.py"]["sha256_start"] is None
    assert "!! source files changed during the run" in capsys.readouterr().out


def test_the_snapshot_is_taken_once_per_file(tmp_path):
    e, expt, params, base = _src_stand(tmp_path)
    expt.write_text("expt v2\n")
    e._snapshot_source_texts()                              # a second call reads nothing again
    assert e._source_snapshot["expt_file"]["text"] == "expt v1\n"


def test_a_warm_run_that_never_claimed_cannot_end_as_if_it_ran(monkeypatch):
    ev = []
    e = _stand(ev, FakeLiveOD(ev))
    e.end_wax = Expt.end_wax.__get__(e)
    monkeypatch.setenv("WAXX_QUEUE_WARM", "tok")
    with pytest.raises(RuntimeError, match="never claimed"):
        e.end_wax("x.py")
    assert ev == []
