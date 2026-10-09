"""The end-of-run emit: Expt.calibrates / _emit_calibrations / emit_calibration /
end_wax, bound onto a stand-in run (no liveOD, no monitor, no hardware).

Everything lives in tmp_path: a synthetic params module, a registry package of
fake analyses, the ledger, and a sandbox HDF5 file standing in for the saved
run. The loader is injected, so no run is ever looked up."""
import datetime
import importlib
import itertools
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import waxx.base.expt as expt_mod
from waxx.base.expt import Expt
from waxx.calibration import emit
from waxx.calibration.config import CalibrationConfig

PARAMS = '''\
class Params:
    def __init__(self):
        self.N_repeats = 1
        self.t_pi = 6.6403e-06 #85412, 2026-10-07
        self.amp = 0.41
'''

ANALYSES = {
    "__init__": "",
    "fake_pi": '''\
"""Fake Rabi pi time (tests)."""
from waxx.calibration.record import CalResult


def calibrate(ad, key, **opts):
    fp = opts.get("figure_path")
    if fp:
        open(fp, "wb").write(b"not really a png")
    return CalResult(key=key, value=opts.get("value", 6.612e-06), unc=opts.get("unc", 2.1e-08),
                     unit="s", n_used=ad.n, excluded={"count": ad.bad, "reason": "non-finite signal"},
                     method="fake damped cosine", figure_path=fp,
                     fit={"ok": True, "params": {"omega": 4.75e5}, "errors": {"omega": 1.5e3},
                          "goodness": {"chi2r": 1.1}, "warnings": opts.get("warnings", [])})
''',
    "boom": '''\
def calibrate(ad, key, **opts):
    raise RuntimeError("the fit exploded")
''',
    "slow": '''\
import threading
GATE = threading.Event()


def calibrate(ad, key, **opts):
    GATE.wait(10)
    raise RuntimeError("too late")
''',
}

_n = itertools.count()
TODAY = datetime.date.today().isoformat()


class FakeRun:
    """Expt's real calibration / end methods on a bare object."""
    _expt_name_from_filepath = staticmethod(Expt._expt_name_from_filepath)


for _name in ("calibrates", "emit_calibration", "_emit_calibrations", "end_wax",
              "_run_done_printout", "start_param_override_recording",
              "_stop_param_override_recording"):
    setattr(FakeRun, _name, vars(Expt)[_name])


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A sandbox: params module, registry, run file, ledger dir."""
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv(emit.VETO_ENV, raising=False)
    monkeypatch.setattr(expt_mod, "_arm_exit_hang_dump", lambda *a, **k: None)
    i = next(_n)
    pname, rname = f"calemit_params_{i}", f"calemit_reg_{i}"
    (tmp_path / f"{pname}.py").write_text(PARAMS)
    pkg = tmp_path / rname
    pkg.mkdir()
    for mod, text in ANALYSES.items():
        (pkg / f"{mod}.py").write_text(text)
    importlib.invalidate_caches()
    params_mod = importlib.import_module(pname)
    run_file = tmp_path / "sandbox_run_85600.h5"
    with h5py.File(run_file, "w") as f:
        f.attrs["run_complete"] = True
        f.attrs["expt_file"] = "source text"
        f.create_dataset("data/x", data=np.arange(5.0))
    yield SimpleNamespace(tmp=tmp_path, params_mod=params_mod, reg=rname, run_file=run_file,
                          params_file=tmp_path / f"{pname}.py", ledger=tmp_path / "ledger")
    for name in [pname, rname] + [f"{rname}.{m}" for m in ANALYSES]:
        sys.modules.pop(name, None)


def make_run(lab, *, policy=None, budget_s=5.0, loader=None, n_bad=0):
    e = FakeRun()
    e.params = lab.params_mod.Params()
    ad = SimpleNamespace(n=63, bad=n_bad, run_info=SimpleNamespace(filepath=np.array([str(lab.run_file)])))
    e.calibration_config = CalibrationConfig(
        ledger_dir=lab.ledger, policy=policy or {}, registry_modules=[lab.reg],
        params_class=None, loader=loader or (lambda rid: ad), budget_s=budget_s)
    e._cal_declarations, e.calibration_results = [], []
    e._param_overrides = None
    e.run_info = SimpleNamespace(save_data=1, run_id=85600, filepath=str(lab.run_file))
    e.live_od_client = SimpleNamespace(last_end_run_reply={})
    e._run_saved = True
    e._shot_complete_count = e._N_shots_total = 63
    e._expt_file_stem = lambda: "rabi_flop"
    return e


def run_file_records(path):
    with h5py.File(path, "r") as f:
        raw = f.attrs.get(emit.RUN_FILE_ATTR)
    return None if raw is None else json.loads(raw)


# ---- declaring ---------------------------------------------------------------------

def test_calibrates_checks_in_prepare(lab):
    e = make_run(lab)
    e.calibrates("t_pi", "fake_pi", opts={"value": 1.0})
    d = e._cal_declarations[0]
    assert (d.key, d.analysis, d.write_back, d.allow_no_unc, d.opts) == \
        ("t_pi", "fake_pi", True, False, {"value": 1.0})
    with pytest.raises(ValueError, match="declared twice"):
        e.calibrates("t_pi", "fake_pi")
    with pytest.raises(ValueError, match="not a param"):
        e.calibrates("t_nope", "fake_pi")
    with pytest.raises(LookupError, match="no analysis 'nothing'"):
        e.calibrates("amp", "nothing")
    e.calibration_config = None
    with pytest.raises(RuntimeError, match="no calibration_config"):
        e.calibrates("amp", "fake_pi")


# ---- the emit ----------------------------------------------------------------------

def test_simulated_end_of_run_emit(lab, capsys):
    e = make_run(lab)
    e.calibrates("t_pi", "fake_pi")
    e._emit_calibrations()
    out = capsys.readouterr().out
    print(out)                                       # shown with -s: the terminal as it is
    assert out.isascii()                             # console lines are ASCII (cp1252 pipes)
    lines = out.strip().splitlines()
    assert lines[0] == ("[cal] t_pi = 6.612e-06 +/- 2.1e-08 s (#85600, 63 shots, 0 excluded; "
                        "was 6.6403e-06, -0.43 %)")
    assert lines[1] == (f"[cal] applied: {lab.params_file}:5 (kcal revert t_pi undoes it)")
    assert lines[2] == "[cal]   - self.t_pi = 6.6403e-06 #85412, 2026-10-07"
    assert lines[3] == f"[cal]   + self.t_pi = 6.612e-06 #85600, {TODAY}"
    assert len(lines) == 4
    assert lab.params_mod.Params().t_pi == 6.612e-06
    # ledger
    rec = json.loads((lab.ledger / "t_pi" / "85600.json").read_text())
    assert rec["applied"] and rec["applied_line"] == 5 and rec["old_value"] == 6.6403e-06
    assert rec["expt_file"] == "rabi_flop" and rec["analysis"] == "fake_pi"
    assert rec["params_class"].endswith(":Params")
    assert Path(rec["figure_path"]) == lab.ledger / "t_pi" / "85600.png"
    events = [json.loads(l)["event"] for l in (lab.ledger / "calibrations.jsonl").read_text().splitlines()]
    assert events == ["emit", "apply"]
    # the run file: the one attribute added, nothing else touched
    recs = run_file_records(lab.run_file)
    assert len(recs) == 1 and recs[0]["key"] == "t_pi" and recs[0]["value"] == 6.612e-06
    with h5py.File(lab.run_file, "r") as f:
        assert sorted(f.attrs) == ["calibration_emitted", "expt_file", "run_complete"]
        assert list(f["data/x"][:]) == [0, 1, 2, 3, 4]
    assert e.calibration_results[0].applied


def test_veto_records_but_does_not_write(lab, capsys, monkeypatch):
    monkeypatch.setenv(emit.VETO_ENV, "1")
    e = make_run(lab)
    e.calibrates("t_pi", "fake_pi")
    raw = lab.params_file.read_bytes()
    e._emit_calibrations()
    out = capsys.readouterr().out
    assert "[cal] not applied: vetoed by WAXX_CAL_NO_WRITE_BACK; to apply: kcal apply t_pi --run 85600" in out
    assert lab.params_file.read_bytes() == raw
    assert (lab.ledger / "t_pi" / "85600.json").exists()


def test_write_back_false(lab, capsys):
    e = make_run(lab)
    e.calibrates("t_pi", "fake_pi", write_back=False)
    raw = lab.params_file.read_bytes()
    e._emit_calibrations()
    assert "not applied: write_back is off" in capsys.readouterr().out
    assert lab.params_file.read_bytes() == raw


def test_policy_flag_stops_the_write_back(lab, capsys):
    e = make_run(lab, policy={"t_pi": {"max_frac_change": 0.001}})
    e.calibrates("t_pi", "fake_pi")
    raw = lab.params_file.read_bytes()
    e._emit_calibrations()
    out = capsys.readouterr().out
    assert "[cal] not applied: flagged -- change from the value in use -0.426% exceeds" in out
    assert lab.params_file.read_bytes() == raw
    assert json.loads((lab.ledger / "t_pi" / "85600.json").read_text())["flags"][0]["code"] == "change"


def test_exclusions_and_fit_notes_are_printed(lab, capsys):
    e = make_run(lab, n_bad=2)
    e.calibrates("t_pi", "fake_pi", write_back=False, opts={"warnings": ["only 0.9 periods"]})
    e._emit_calibrations()
    out = capsys.readouterr().out
    assert "(#85600, 63 shots, 2 excluded;" in out
    assert "[cal]   excluded 2 shot(s): non-finite signal" in out
    assert "[cal]   fit note: only 0.9 periods" in out


def test_a_failing_analysis_is_recorded_flagged_and_never_raises(lab, capsys):
    e = make_run(lab)
    e.calibrates("t_pi", "boom")
    raw = lab.params_file.read_bytes()
    e._emit_calibrations()
    out = capsys.readouterr().out
    assert "[cal] t_pi = nan" in out
    assert "not applied: flagged -- the analysis failed: the analysis raised RuntimeError: " \
           "the fit exploded" in out
    assert lab.params_file.read_bytes() == raw
    assert json.loads((lab.ledger / "t_pi" / "85600.json").read_text())["value"] == "nan"


def test_a_slow_analysis_is_deferred(lab, capsys):
    e = make_run(lab, budget_s=0.2)
    e.calibrates("t_pi", "slow")
    try:
        e._emit_calibrations()
    finally:
        sys.modules[f"{lab.reg}.slow"].GATE.set()
    out = capsys.readouterr().out
    assert "took longer than 0.2 s and is deferred" in out
    rec = json.loads((lab.ledger / "t_pi" / "85600.json").read_text())
    assert rec["deferred"] is True and not rec["applied"]


@pytest.mark.parametrize("change, why", [
    (dict(_run_saved=False), "the run's save did not complete"),
    (dict(live_od_client=None), "not saved through liveOD"),
    (dict(run_info=SimpleNamespace(save_data=0, run_id=85600, filepath="")), "saved no data"),
    (dict(run_info=SimpleNamespace(save_data=1, run_id=0, filepath="")), "no run id"),
    (dict(_shot_complete_count=40), "ended after 40 of 63 shots"),
    (dict(live_od_client=SimpleNamespace(last_end_run_reply={"incomplete": {"reason": "frames lost"}})),
     "saved INCOMPLETE (frames lost)"),
])
def test_runs_that_must_not_be_calibrated(lab, capsys, change, why):
    e = make_run(lab)
    e.calibrates("t_pi", "fake_pi")
    for k, v in change.items():
        setattr(e, k, v)
    e._emit_calibrations()
    out = capsys.readouterr().out
    assert "[cal] not run (t_pi): " in out and why in out
    assert not lab.ledger.exists() and run_file_records(lab.run_file) is None


def test_a_run_that_cannot_be_loaded_emits_nothing(lab, capsys):
    def no_load(rid):
        raise OSError("drive not mapped")
    e = make_run(lab, loader=no_load)
    e.calibrates("t_pi", "fake_pi")
    e._emit_calibrations()
    assert "[cal] not run (t_pi): could not load run 85600 (OSError: drive not mapped)" in \
        capsys.readouterr().out
    assert not lab.ledger.exists()


def test_run_file_not_complete_gets_no_attribute(lab, capsys):
    with h5py.File(lab.run_file, "r+") as f:
        f.attrs["run_complete"] = False
    e = make_run(lab)
    e.calibrates("t_pi", "fake_pi", write_back=False)
    e._emit_calibrations()
    assert "is not marked run_complete" in capsys.readouterr().out
    assert run_file_records(lab.run_file) is None


def test_emit_calibration_after_end_appends_to_the_run_file(lab, capsys):
    e = make_run(lab)
    e.calibrates("t_pi", "fake_pi", write_back=False)
    e._emit_calibrations()
    r = e.emit_calibration("amp", 0.4234, 0.0021, unit="V", n_used=63, method="mean")
    out = capsys.readouterr().out
    assert "[cal] amp = 4.234e-01 +/- 2.1e-03 V (#85600, 63 shots, 0 excluded; was 0.41, +3.27 %)" in out
    assert "not applied: write_back is off" in out
    assert [x["key"] for x in run_file_records(lab.run_file)] == ["t_pi", "amp"]
    assert r.analysis == "emit_calibration" and e.calibration_results[-1] is r


def test_emit_calibration_before_the_save_is_not_recorded(lab, capsys):
    e = make_run(lab)
    e._run_saved = False
    r = e.emit_calibration("amp", 0.4234, 0.0021)
    assert "amp not recorded: the run's save did not complete" in capsys.readouterr().out
    assert r is not None and not lab.ledger.exists()
    assert e.emit_calibration("amp", 1.0, 0.1, bogus=1) is None      # bad field: printed, None


# ---- end_wax: the emit runs after the save, and never raises -----------------------------

def wire_end(e, order, end_run_raises=None):
    e.scope_data = SimpleNamespace(close=lambda: order.append("scope"))
    e._finish_camera_streams = lambda: None
    e._close_shot_data_queue = lambda: None
    e.cleanup_scanned = lambda: None
    e._serialize_end_payload = lambda fp: {}
    e._n_pushed = e._n_push_failed = 0
    e._run_saved = False

    def end_run(payload):
        order.append("end_run")
        if end_run_raises:
            raise end_run_raises
        return True
    e.live_od_client = SimpleNamespace(end_run=end_run, last_end_run_reply={})
    real = e._emit_calibrations
    e._emit_calibrations = lambda: (order.append("emit"), real())


def test_end_wax_emits_after_the_save(lab, capsys):
    e = make_run(lab)
    e.calibrates("t_pi", "fake_pi")
    order = []
    wire_end(e, order)
    e.end_wax("rabi_flop.py", notify=False)
    assert order == ["scope", "end_run", "emit"]
    assert "[cal] applied:" in capsys.readouterr().out


def test_end_wax_failed_save_raises_as_before_and_does_not_emit(lab):
    e = make_run(lab)
    e.calibrates("t_pi", "fake_pi")
    order = []
    wire_end(e, order, end_run_raises=RuntimeError("END_RUN failed"))
    with pytest.raises(RuntimeError, match="END_RUN failed"):
        e.end_wax("rabi_flop.py", notify=False)
    assert "emit" not in order and not lab.ledger.exists()


def test_end_wax_survives_a_broken_emit(lab, capsys, monkeypatch):
    e = make_run(lab)
    e.calibrates("t_pi", "fake_pi")
    wire_end(e, [])

    def broken(*a, **k):
        raise ZeroDivisionError("bug")
    monkeypatch.setattr(emit, "run_declared", broken)
    e.end_wax("rabi_flop.py", notify=False)
    assert "[cal] WARNING: the calibration step failed (ZeroDivisionError: bug); the run's " \
           "data is saved and unaffected." in capsys.readouterr().out
