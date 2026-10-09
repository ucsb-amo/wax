"""waxx.calibration: CalResult + evaluate (the flag rules), the policy loader, the ledger."""
import json
import math
import sys
import threading

import pytest

from waxx.calibration.ledger import Ledger
from waxx.calibration.policy import PolicyError, load_policy, policy_for
from waxx.calibration.record import CalResult, evaluate
from waxx.calibration.writeback import WritebackReport


def res(**kw):
    base = dict(key="t_pi", value=6.6e-6, unc=2e-8, unit="s", run_id=100, n_used=60,
                fit={"ok": True})
    base.update(kw)
    return CalResult(**base)


def codes(flags):
    return sorted(f["code"] for f in flags)


# ---- CalResult ---------------------------------------------------------------------

def test_json_roundtrip_keeps_every_field_and_nan():
    r = res(value=math.nan, unc=None, excluded={"count": 3, "reason": "non-finite signal"},
            fit={"ok": False, "reason": "x", "params": {"omega": 1.0}}, figure_path="f.png")
    r.set_old_value(6.5e-6)
    back = CalResult.from_json(r.to_json())
    assert math.isnan(back.value) and back.unc is None
    assert back.excluded == {"count": 3, "reason": "non-finite signal"}
    assert back.fit["params"] == {"omega": 1.0}
    assert back.old_value == 6.5e-6 and back.rel_change is None      # NaN value: no change
    assert json.loads(r.to_json())["value"] == "nan"                  # valid JSON
    assert back.timestamp == r.timestamp


def test_numpy_values_serialise():
    np = pytest.importorskip("numpy")
    r = res(value=np.float64(1.5), unc=np.float64(0.1), fit={"ok": True, "cov": np.eye(2)})
    d = json.loads(r.to_json())
    assert d["value"] == 1.5 and d["fit"]["cov"] == [[1.0, 0.0], [0.0, 1.0]]


def test_rel_change():
    r = res(value=110.0)
    r.set_old_value(100.0)
    assert r.rel_change == pytest.approx(0.1)
    r.set_old_value(-100.0)
    assert r.rel_change == pytest.approx(2.1)
    r.set_old_value(0.0)
    assert r.rel_change is None


# ---- evaluate ------------------------------------------------------------------------

def test_no_policy_entry_no_hard_checks():
    r = res(value=1e9, n_used=1)          # absurd, but no policy says so
    r.set_old_value(1.0)
    assert evaluate(r, None) == [] and evaluate(r, {}) == []


def test_rules_that_need_no_policy():
    assert codes(evaluate(res(fit={"ok": False, "reason": "no fit"}), {})) == ["analysis_failed"]
    assert codes(evaluate(res(value=math.nan), {})) == ["nonfinite"]
    assert codes(evaluate(res(value=math.inf), {})) == ["nonfinite"]
    assert codes(evaluate(res(unc=math.nan), {})) == ["nonfinite"]
    assert codes(evaluate(res(unc=-1.0), {})) == ["negative_unc"]
    assert codes(evaluate(res(unc=0.0), {})) == ["zero_unc"]
    assert codes(evaluate(res(unc=None), {})) == ["no_unc"]
    assert evaluate(res(unc=None), {}, allow_no_unc=True) == []
    d = res(value=math.nan, unc=None, deferred=True, fit={"ok": False, "reason": "budget"})
    assert codes(evaluate(d, {}, allow_no_unc=True)) == ["analysis_failed", "nonfinite"]


def test_policy_rules():
    pol = {"max_frac_change": 0.05, "min": 1e-6, "max": 2e-5, "min_shots": 20,
           "max_rel_unc": 0.01}
    r = res()
    r.set_old_value(6.5e-6)
    assert evaluate(r, pol) == []
    r = res(value=7.0e-6)
    r.set_old_value(6.5e-6)
    assert codes(evaluate(r, pol)) == ["change"]
    assert codes(evaluate(res(value=5e-7, unc=1e-9), pol)) == ["below_min"]
    assert codes(evaluate(res(value=3e-5, unc=1e-8), pol)) == ["above_max"]
    assert codes(evaluate(res(n_used=19), pol)) == ["few_shots"]
    assert codes(evaluate(res(unc=1e-7), pol)) == ["rel_unc"]
    # no old value: the change rule cannot fire
    assert evaluate(res(value=7.0e-6), pol) == []
    # every flag has its text
    r = res(value=3e-5, unc=1e-6, n_used=1)
    r.set_old_value(6.5e-6)
    fl = evaluate(r, pol)
    assert codes(fl) == ["above_max", "change", "few_shots", "rel_unc"]
    assert all(f["text"] for f in fl) and r.flags == fl


# ---- policy --------------------------------------------------------------------------

def test_policy_from_mapping_and_module(tmp_path, monkeypatch):
    assert load_policy(None) == {}
    pol = load_policy({"t_pi": {"max_frac_change": "0.05", "min_shots": 20.0}})
    assert pol == {"t_pi": {"max_frac_change": 0.05, "min_shots": 20}}
    assert policy_for(pol, "t_pi")["max_frac_change"] == 0.05
    assert policy_for(pol, "other") == {} and policy_for(None, "x") == {}
    (tmp_path / "calpol_mod.py").write_text("POLICY = {'a': {'min': 1}}\nALT = {'b': {}}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert load_policy("calpol_mod") == {"a": {"min": 1.0}}
        assert load_policy("calpol_mod:ALT") == {"b": {}}
    finally:
        sys.modules.pop("calpol_mod", None)


def test_policy_refuses_typos_and_negatives():
    with pytest.raises(PolicyError, match="unknown field"):
        load_policy({"t_pi": {"max_frac_chang": 0.05}})
    with pytest.raises(PolicyError):
        load_policy({"t_pi": {"max_rel_unc": -1}})
    with pytest.raises(PolicyError):
        load_policy(42)


# ---- ledger --------------------------------------------------------------------------

def test_ledger_needs_a_directory():
    with pytest.raises(ValueError):
        Ledger(None)
    with pytest.raises(ValueError):
        Ledger("")


def test_ledger_records_history_and_apply_state(tmp_path):
    led = Ledger(tmp_path / "led")
    for rid, v in [(100, 6.6e-6), (101, 6.7e-6), (102, 6.8e-6)]:
        led.write_record(res(run_id=rid, value=v))
    assert (tmp_path / "led" / "t_pi" / "101.json").exists()
    assert led.load("t_pi", 101).value == 6.7e-6
    assert [r.run_id for r in led.history("t_pi", 2)] == [101, 102]
    rep = WritebackReport(True, "apply", "t_pi", file="p.py", line_no=12, old_line="a",
                          commented_line="# a", new_line="b", written=True)
    led.note_writeback(rep, run_id=101, by="test")
    assert led.load("t_pi", 101).applied and led.load("t_pi", 101).applied_line == 12
    hist = {r.run_id: r for r in led.history("t_pi", None)}
    assert hist[101].applied and not hist[100].applied
    events = [e["event"] for e in led.events()]
    assert events == ["emit", "emit", "emit", "apply"]
    with pytest.raises(LookupError):
        led.load("t_pi", 999)
    with pytest.raises(ValueError):
        led.record_path("../evil", 1)


def test_ledger_jsonl_is_append_only_and_survives_a_bad_line(tmp_path):
    led = Ledger(tmp_path)
    led.write_record(res(run_id=1))
    with open(led.jsonl, "a") as f:
        f.write("{not json\n")
    led.write_record(res(run_id=2))
    lines = led.jsonl.read_text().splitlines()
    assert len(lines) == 3 and lines[1] == "{not json"
    assert [r.run_id for r in led.history("t_pi")] == [1, 2]
    (led.dir / "t_pi" / "2.json").unlink()            # the jsonl alone is enough to load
    assert led.load("t_pi", 2).run_id == 2


def test_ledger_concurrent_appends_do_not_interleave(tmp_path):
    led = Ledger(tmp_path)

    def worker(base):
        for i in range(25):
            led.append_event({"event": "emit", "key": "k", "run_id": base + i, "pad": "x" * 500})
    ts = [threading.Thread(target=worker, args=(1000 * j,)) for j in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    rows = [json.loads(l) for l in led.jsonl.read_text().splitlines()]
    assert len(rows) == 100 and len({r["run_id"] for r in rows}) == 100
    assert not (tmp_path / "calibrations.jsonl.lock").exists()


def test_ledger_key_must_match_exactly(tmp_path):
    led = Ledger(tmp_path)
    for bad in ("t_pi\n", "t pi", "tp\u03c0", "1x", "", "a.b"):
        with pytest.raises(ValueError):
            led.record_path(bad, 1)
    assert led.record_path("t_pi_2", 1).name == "1.json"


def test_ledger_history_line_goes_first(tmp_path, monkeypatch):
    import waxx.calibration.ledger as ledger_mod
    led = Ledger(tmp_path)

    def fail(*a, **k):
        raise PermissionError("per-run JSON not writable")
    monkeypatch.setattr(ledger_mod, "replace_bytes", fail)
    with pytest.raises(PermissionError):
        led.write_record(res(run_id=7))
    assert [e["run_id"] for e in led.events()] == [7]          # the history has it


def test_a_stale_lock_fails_at_once(tmp_path):
    import os
    import time
    from waxx.calibration._lock import LockTimeout
    led = Ledger(tmp_path)
    lock = tmp_path / "calibrations.jsonl.lock"
    lock.write_text("pid 4242 since long ago")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    t0 = time.monotonic()
    with pytest.raises(LockTimeout, match="STALE LOCK .*pid 4242"):
        led.append_event({"event": "emit", "key": "k", "run_id": 1})
    assert time.monotonic() - t0 < 1.0
    assert lock.exists() and not led.jsonl.exists()             # left for a person; nothing written
