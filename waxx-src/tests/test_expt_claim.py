"""waxx Expt: the claim (queue phase 2) and the source-text snapshot (M0).

Host-only, offline: a stand-in object with the real Expt methods bound, a
recording fake liveOD client and monitor, the placeholder queue_gate. No
network, no servers, no core device.
"""
import json
import os
from types import SimpleNamespace

import pytest

from waxa.data.data_saver import DataSaver
from waxx.base.expt import Expt
from waxx.util.device_state import queue_gate


@pytest.fixture(autouse=True)
def _fresh_gate(monkeypatch):
    monkeypatch.delenv("WAXX_QUEUE_WARM", raising=False)
    queue_gate._reset_for_tests()
    yield
    queue_gate._reset_for_tests()


class FakeLiveOD:
    def __init__(self, events, has_phase2=True, start_reply=None, reserve_reply=None):
        self.events = events
        self.server_features = {}
        self._start_reply = start_reply
        self._reserve_reply = reserve_reply
        if not has_phase2:
            self.reserve_run = self.start_run = None
        self.released = []

    def supports(self, f):
        return bool(self.server_features.get(f))

    def init_run(self, payload):
        self.events.append("init_run")
        self.server_features = {"put_data": True}
        return {"ok": True, "run_id": 7, "filepath": "x.hdf5"}

    def reserve_run(self, payload, ttl_s=900.0):
        self.events.append("reserve_run")
        return self._reserve_reply or {"ok": True, "reservation": "r9", "warnings": ["w"]}

    def start_run(self, payload, *, reservation, require_features=(), go_reset_count=None):
        self.events.append(("start_run", reservation, tuple(require_features), go_reset_count))
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
            "_release_reservation", "_decide_subscan", "_compile_subscan_switch",
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
    e._subscan = None
    e._extra_file_texts = {}
    e.live_od_client = client
    e.run_info = RunInfo(None, save_data=False)
    e.claim_hardware = lambda: events.append("hardware")
    e.claim_after_start = lambda: events.append("after_start")
    e._serialize_init_payload = lambda: {"payload": True}
    e._expt_file_stem = lambda: "stand"
    e.monitor = SimpleNamespace(announce_run=lambda **kw: events.append(("announce",
                                                                         kw["run_id"])))
    return e


def test_cold_claim_order_and_once():
    ev = []
    e = _stand(ev, FakeLiveOD(ev))
    e.claim()
    e.claim()
    assert ev == ["hardware", "init_run", ("announce", 7), "after_start"]
    assert e.run_info.run_id == 7 and e._run_started


def test_warm_reserve_then_start_with_the_reservation_and_server_time():
    ev = []
    client = FakeLiveOD(ev)
    e = _stand(ev, client)
    e._reserve_warm_run()
    assert ev == ["reserve_run"] and e._reservation == "r9"
    e.claim()
    assert ev[1:] == ["hardware", ("start_run", "r9", ("reserve",), None), ("announce", 8),
                      "after_start"]
    assert e.run_info.run_id == 8 and e._reservation is None
    assert e.run_info.run_datetime_str == "2026-10-14_09-08-07"
    assert e.run_info.run_date_str == "2026-10-14"
    assert e.run_info.run_datetime.tm_hour == 9
    e._release_reservation()                     # consumed: nothing to release
    assert client.released == []


def test_a_start_refusal_stops_the_claim_before_the_announce():
    ev = []
    e = _stand(ev, FakeLiveOD(ev, start_reply={"ok": False, "error": "a run is in progress"}))
    e._reserve_warm_run()
    with pytest.raises(RuntimeError, match="START_RUN refused: a run is in progress"):
        e.claim()
    assert not any(isinstance(x, tuple) and x[0] == "announce" for x in ev)
    assert "init_run" not in ev


def test_an_old_liveod_gets_init_run():
    ev = []
    e = _stand(ev, FakeLiveOD(ev, has_phase2=False))
    e._reserve_warm_run()
    assert e._reservation is None
    e.claim()
    assert "init_run" in ev and e.run_info.run_id == 7


def test_unknown_tag_replies_fall_back_to_init_run():
    ev = []
    unknown = {"ok": False, "error": "Unknown tag: RESERVE_RUN"}
    e = _stand(ev, FakeLiveOD(ev, reserve_reply=unknown))
    e._reserve_warm_run()
    assert e._reservation is None
    ev2 = []
    e2 = _stand(ev2, FakeLiveOD(ev2, start_reply={"ok": False, "error": "Unknown tag"}))
    e2._reservation = "r1"
    e2.claim()
    assert "init_run" in ev2


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
