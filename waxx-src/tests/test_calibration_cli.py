"""waxx.calibration: the analysis registry and budget, the config loader, and kcal.

The config, registry, params module and ledger are throwaway modules / dirs in
tmp_path; nothing outside it is read or written."""
import importlib
import itertools
import json
import sys
import threading
import time

import pytest

from waxx.calibration import cli
from waxx.calibration.analysis import AnalysisNotFound, list_analyses, resolve, run_analysis
from waxx.calibration.config import ENV_VAR, CalibrationConfig, import_object, load_config
from waxx.calibration.ledger import Ledger
from waxx.calibration.record import CalResult

PARAMS = '''\
class Params:
    def __init__(self):
        # self.t_pi = 6.0e-06 #100, 2026-01-01
        self.t_pi = 6.6403e-06 #85412, 2026-10-07
        self.amp = 0.41
'''

_n = itertools.count()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    i = next(_n)
    names = dict(params=f"calcli_params_{i}", reg=f"calcli_reg_{i}", cfg=f"calcli_cfg_{i}")
    (tmp_path / f"{names['params']}.py").write_text(PARAMS)
    reg = tmp_path / names["reg"]
    reg.mkdir()
    (reg / "__init__.py").write_text("def pkg_level(ad, key, **o):\n    raise RuntimeError\n")
    (reg / "pi_time.py").write_text('"""Pi time from a flop (fake)."""\n'
                                    "def calibrate(ad, key, **opts):\n    pass\n")
    (reg / "_private.py").write_text("def calibrate(ad, key, **o):\n    pass\n")
    (reg / "broken_import.py").write_text("import a_module_that_does_not_exist_xyz\n"
                                          "def calibrate(ad, key, **o):\n    pass\n")
    (reg / "helper.py").write_text("X = 1\n")
    (reg / "fake_emit.py").write_text(
        "from waxx.calibration.record import CalResult\n"
        "def calibrate(ad, key, **opts):\n"
        "    return CalResult(key=key, value=6.612e-06, unc=2.1e-08, unit='s', n_used=63,\n"
        "                     fit={'ok': True, 'saw': ad.tag, 'opts': opts.get('x')})\n")
    ledger = tmp_path / "ledger"
    (tmp_path / f"{names['cfg']}.py").write_text(
        "from types import SimpleNamespace\n"
        "from waxx.calibration.config import CalibrationConfig\n"
        "def _load(rid, needs_images=False):\n"
        "    return SimpleNamespace(tag=rid, params=SimpleNamespace(t_pi=6.6403e-06))\n"
        "def _bad(rid, needs_images=False):\n"
        "    raise OSError('the data drive is not mapped')\n"
        f"CFG = CalibrationConfig(ledger_dir={str(ledger)!r},\n"
        "    policy={'t_pi': {'max_frac_change': 0.05}},\n"
        f"    registry_modules=[{names['reg']!r}],\n"
        f"    params_class='{names['params']}:Params', loader=_load)\n"
        "BAD = CalibrationConfig(ledger_dir=CFG.ledger_dir, registry_modules=CFG.registry_modules,\n"
        "    params_class=CFG.params_class, loader=_bad)\n")
    importlib.invalidate_caches()
    yield type("Env", (), dict(tmp=tmp_path, ledger=ledger, spec=f"{names['cfg']}:CFG",
                               params_file=tmp_path / f"{names['params']}.py", **names))
    for k in list(sys.modules):
        if k.startswith(tuple(names.values())):
            sys.modules.pop(k, None)


def record(key="t_pi", run_id=85600, value=6.612e-06, unc=2.1e-08, **kw):
    r = CalResult(key=key, value=value, unc=unc, unit="s", run_id=run_id, n_used=63,
                  fit={"ok": True}, **kw)
    return r


def kcal(*argv):
    out = []
    code = cli.main(list(argv), out=out.append)
    return code, "\n".join(out)


# ---- registry / budget -----------------------------------------------------------------

def test_resolve(env):
    fn = resolve("pi_time", [env.reg])
    assert fn.__module__ == f"{env.reg}.pi_time"
    assert resolve("pkg_level", [env.reg]).__name__ == "pkg_level"
    with pytest.raises(AnalysisNotFound):
        resolve("nothing_here", [env.reg])
    with pytest.raises(AnalysisNotFound):
        resolve("pi_time", [])
    with pytest.raises(AnalysisNotFound):
        resolve("../x", [env.reg])
    with pytest.raises(ModuleNotFoundError, match="a_module_that_does_not_exist_xyz"):
        resolve("broken_import", [env.reg])          # a broken analysis says so, loudly


def test_list_analyses(env):
    rows = {name: doc for name, _, doc in list_analyses([env.reg])}
    assert rows["pi_time"] == "Pi time from a flop (fake)."
    assert rows["broken_import"].startswith("IMPORT FAILED")
    assert "_private" not in rows and "helper" not in rows


def test_run_analysis_outcomes():
    ok = lambda ad, key, **o: CalResult(key=key, value=1.0, unc=0.1, fit={"ok": True},
                                        figure_path=o.get("figure_path"))
    r = run_analysis(ok, None, "k", budget_s=5, figure_path="f.png", name="ok")
    assert r.value == 1.0 and r.analysis == "ok" and r.figure_path == "f.png"
    assert "elapsed_s" in r.fit

    def boom(ad, key, **o):
        raise KeyError("sum_od_x")
    r = run_analysis(boom, None, "k", name="boom")
    assert not r.fit_ok and "raised KeyError" in r.fit["reason"] and r.analysis == "boom"
    r = run_analysis(lambda ad, key, **o: 3.0, None, "k")
    assert not r.fit_ok and "returned float" in r.fit["reason"]
    r = run_analysis(lambda ad, key, **o: CalResult(key="other", value=1.0, unc=0.1), None, "k")
    assert not r.fit_ok and "'other'" in r.fit["reason"]


def test_budget_defers_and_leaves_the_thread_alone():
    gate, done = threading.Event(), threading.Event()

    def slow(ad, key, **o):
        gate.wait(10)
        done.set()
        return CalResult(key=key, value=1.0, unc=0.1)
    t0 = time.monotonic()
    r = run_analysis(slow, None, "k", budget_s=0.2)
    assert time.monotonic() - t0 < 2
    assert r.deferred and not r.fit_ok and "0.2 s budget" in r.fit["reason"]
    gate.set()
    assert done.wait(5)                                     # it finished on its own


# ---- config ------------------------------------------------------------------------------

def test_config_loading(env, monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match=ENV_VAR):
        load_config(None)
    monkeypatch.setenv(ENV_VAR, env.spec)
    cfg = load_config(None)
    assert isinstance(cfg, CalibrationConfig) and cfg.budget_s == 30.0
    assert cfg.get_ledger_dir() == env.ledger
    assert cfg.get_params_class().__name__ == "Params"
    assert cfg.get_policy() == {"t_pi": {"max_frac_change": 0.05}}
    with pytest.raises(TypeError):
        load_config(f"{env.params}:Params")
    assert import_object("json.dumps") is json.dumps
    lazy = CalibrationConfig(ledger_dir=lambda: env.tmp / "lazy")
    assert lazy.get_ledger_dir() == env.tmp / "lazy"
    with pytest.raises(RuntimeError, match="no ledger directory"):
        CalibrationConfig().get_ledger_dir()


# ---- kcal ----------------------------------------------------------------------------------

def test_kcal_apply_show_check_revert(env, monkeypatch):
    monkeypatch.setenv(ENV_VAR, env.spec)
    Ledger(env.ledger).write_record(record(params_class=f"{env.params}:Params"))
    raw = env.params_file.read_bytes()

    code, out = kcal("check", "t_pi", "--run", "85600")
    assert code == 0 and "no flags" in out and "was 6.6403e-06, -0.43 %" in out

    code, out = kcal("apply", "t_pi", "--run", "85600", "--dry-run")
    assert code == 0 and "would write" in out and env.params_file.read_bytes() == raw

    code, out = kcal("apply", "t_pi", "--run", "85600", "--note", "by hand")
    assert code == 0, out
    assert "+ self.t_pi = 6.612e-06 #85600," in out and "by hand" in out
    assert "self.t_pi = 6.612e-06 #85600" in env.params_file.read_text()
    assert "(kcal revert t_pi undoes it)" in out

    code, out = kcal("show", "t_pi")
    assert code == 0
    assert "#85600" in out and "APPLIED" in out
    assert ">" in out and "# self.t_pi = 6.6403e-06 #85412, 2026-10-07" in out

    code, out = kcal("revert", "t_pi")
    assert code == 0, out
    assert "self.t_pi = 6.6403e-06 #reverted" in env.params_file.read_text()
    events = [e["event"] for e in Ledger(env.ledger).events()]
    assert events == ["emit", "apply", "revert"]


def test_kcal_apply_rechecks_against_the_value_now_in_the_file(env, monkeypatch):
    """The emit's old value was 6.6403e-06; the file has since been edited by
    hand to 7.0e-06: the change is now -5.5 %, over the 5 % policy."""
    monkeypatch.setenv(ENV_VAR, env.spec)
    Ledger(env.ledger).write_record(record())
    text = env.params_file.read_text().replace("self.t_pi = 6.6403e-06", "self.t_pi = 7.0e-06")
    env.params_file.write_text(text)
    sys.modules.pop(env.params, None)
    raw = env.params_file.read_bytes()
    code, out = kcal("apply", "t_pi", "--run", "85600")
    assert code == 2 and "REFUSED: flagged -- change from the value in use -5.543%" in out
    assert env.params_file.read_bytes() == raw
    code, out = kcal("check", "t_pi", "--run", "85600")
    assert code == 1 and "change:" in out


def test_kcal_refuses_a_flagged_or_missing_record(env):
    Ledger(env.ledger).write_record(record(unc=None))
    raw = env.params_file.read_bytes()
    code, out = kcal("--config", env.spec, "apply", "t_pi", "--run", "85600")
    assert code == 2 and "no uncertainty" in out and env.params_file.read_bytes() == raw
    code, out = kcal("--config", env.spec, "apply", "t_pi", "--run", "85600", "--allow-no-unc")
    assert code == 0, out
    code, out = kcal("--config", env.spec, "apply", "t_pi", "--run", "1")
    assert code == 2 and "no ledger record" in out


def test_kcal_analyses_and_missing_config(env, monkeypatch):
    code, out = kcal("--config", env.spec, "analyses")
    assert code == 0 and "pi_time" in out and "Pi time from a flop" in out
    monkeypatch.delenv(ENV_VAR, raising=False)
    code, out = kcal("show", "t_pi")
    assert code == 2 and ENV_VAR in out


# ---- S6: stored flags count; stale records refused; refusals journaled --------------------

def test_kcal_apply_keeps_the_flags_it_was_emitted_with(env, monkeypatch):
    monkeypatch.setenv(ENV_VAR, env.spec)
    r = record()
    r.flags = [{"code": "file_changed", "text": "the params file was edited during the run"}]
    Ledger(env.ledger).write_record(r)
    raw = env.params_file.read_bytes()
    code, out = kcal("apply", "t_pi", "--run", "85600")
    assert code == 2 and "edited during the run" in out
    assert env.params_file.read_bytes() == raw
    code, out = kcal("check", "t_pi", "--run", "85600")
    assert code == 1 and "file_changed:" in out
    ev = [e for e in Ledger(env.ledger).events() if e["event"] == "apply_refused"]
    assert len(ev) == 1 and ev[0]["flags"] == ["file_changed"] and "@" in ev[0]["by"]


def test_kcal_apply_refuses_a_stale_record_unless_forced(env, monkeypatch):
    monkeypatch.setenv(ENV_VAR, env.spec)
    led = Ledger(env.ledger)
    led.write_record(record(run_id=85600))
    led.write_record(record(run_id=85601, value=6.62e-06))
    code, out = kcal("apply", "t_pi", "--run", "85600")
    assert code == 2 and "newer entries for t_pi: emit #85601" in out
    code, out = kcal("apply", "t_pi", "--run", "85600", "--force")
    assert code == 0 and "--force" in out, out
    events = [e["event"] for e in led.events()]
    assert events == ["emit", "emit", "apply_refused", "force", "apply"]
    # an apply after this record's emit makes it stale too
    code, out = kcal("apply", "t_pi", "--run", "85601")
    assert code == 2 and "newer entries" in out and "apply #85600" in out


def test_kcal_refused_apply_is_journaled(env, monkeypatch):
    monkeypatch.setenv(ENV_VAR, env.spec)
    led = Ledger(env.ledger)
    led.write_record(record(key="amp", value=0.4321, unc=0.0011))
    env.params_file.write_text(env.params_file.read_text().replace("self.amp = 0.41",
                                                                   "self.amp = 2 * 0.2"))
    sys.modules.pop(env.params, None)
    code, out = kcal("apply", "amp", "--run", "85600")
    assert code == 2 and "not a plain numeric literal" in out
    ev = [e for e in led.events() if e["event"] == "apply"]
    assert len(ev) == 1 and ev[0]["ok"] is False and "not a plain" in ev[0]["reason"]


# ---- S9: kcal emit ---------------------------------------------------------------------------

def test_kcal_emit_records_but_never_applies(env, monkeypatch):
    monkeypatch.setenv(ENV_VAR, env.spec)
    raw = env.params_file.read_bytes()
    code, out = kcal("emit", "t_pi", "--run", "85600", "--analysis", "fake_emit",
                     "--opts", '{"x": 3}')
    assert code == 0, out
    assert out.startswith("[cal] t_pi = 6.612e-06 +/- 2.1e-08 s (#85600")
    assert "not applied: write_back is off" in out
    assert env.params_file.read_bytes() == raw
    rec = Ledger(env.ledger).load("t_pi", 85600)
    assert rec.fit["saw"] == 85600 and rec.fit["opts"] == 3 and "kcal emit" in rec.fit["emitted_by"]
    assert rec.old_value == 6.6403e-06 and rec.flags == []
    code, out = kcal("apply", "t_pi", "--run", "85600")
    assert code == 0, out


def test_kcal_emit_bad_opts_and_unknown_analysis(env):
    code, out = kcal("--config", env.spec, "emit", "t_pi", "--run", "1", "--analysis", "fake_emit",
                     "--opts", "[1, 2]")
    assert code == 2 and "JSON object" in out
    code, out = kcal("--config", env.spec, "emit", "t_pi", "--run", "1", "--analysis", "nope")
    assert code == 2 and "no analysis 'nope'" in out


# ---- N8: an OSError is one line, exit 2 -------------------------------------------------------

def test_kcal_oserror_is_one_line(env):
    spec = env.spec.replace(":CFG", ":BAD")
    code, out = kcal("--config", spec, "emit", "t_pi", "--run", "1", "--analysis", "fake_emit")
    assert code == 2 and out == "kcal emit: OSError: the data drive is not mapped"
