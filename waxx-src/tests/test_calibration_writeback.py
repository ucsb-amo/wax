"""waxx.calibration.writeback on synthetic params modules written to tmp_path.

Every module here is a throwaway file in a temp dir, imported under a unique
name and removed from sys.modules afterwards. Nothing outside tmp_path is
read or written."""
import contextlib
import importlib
import itertools
import sys
from pathlib import Path

import pytest

from waxx.calibration import writeback
from waxx.calibration.record import CalResult

BASE = '''\
import numpy as np


class Params:
    """K-like params. self.t_tof = 99.0 in a docstring is not an assignment."""

    def __init__(self):
        self.N_repeats = 1
        # self.t_pi = 6.0e-06 #100, 2026-01-01
        self.t_pi = 6.6403e-06 #85412, 2026-10-07
        self.t_tof = 250.e-6  # time of flight (µs scale)
        self.amp = 0.41
        self.N_iter = np.int32(50)
        self.freq = 41.23456789123e6
        self.dup = 1.0
        self.dup = 2.0
        self.expr = 2 * self.amp
        self.derived_x = 0.
        self.tup_a, self.tup_b = 1.0, 2.0
        self.multi = (
            1.0)
        self.aug = 1.0
        self.aug += 1.0
        self.ann: float = 1.0
        self.chain = self.chain2 = 3.0

    def compute_x(self):
        self.derived_x = self.amp * 2
        self.only_in_method = 1.0
'''

SUB = '''\
from {base} import Params


class Sub(Params):
    def __init__(self):
        self.t_tof = 300.e-6   # before super().__init__(): the base overwrites it
        super().__init__()
        self.amp = 0.5 #sub override
'''

_n = itertools.count()


@pytest.fixture
def modules(tmp_path, monkeypatch):
    """make(text, crlf=False, prefix='calp') -> (module, path); cleans up."""
    made = []
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(writeback, "LOCK_TIMEOUT_S", 0.3)

    def make(text, crlf=False, prefix="calp", bom=False):
        name = f"{prefix}_{next(_n)}"
        data = text.replace("\n", "\r\n") if crlf else text
        raw = data.encode("utf-8")
        if bom:
            raw = b"\xef\xbb\xbf" + raw
        path = tmp_path / f"{name}.py"
        path.write_bytes(raw)
        importlib.invalidate_caches()
        mod = importlib.import_module(name)
        made.append(name)
        return mod, path

    yield make
    for name in made:
        sys.modules.pop(name, None)


def result(key="t_pi", value=6.612e-06, unc=2.1e-08, **kw):
    r = CalResult(key=key, value=value, unc=unc, unit="s", run_id=85600, n_used=63,
                  fit={"ok": True})
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def lines(path):
    return path.read_bytes().decode("utf-8-sig").splitlines()


# ---- find ------------------------------------------------------------------------------

def test_find_the_one_active_line(modules):
    mod, path = modules(BASE)
    t = writeback.find_assignment("t_pi", mod.Params)
    assert Path(t.file) == path and t.cls_name == "Params" and t.method == "__init__"
    assert t.line == "        self.t_pi = 6.6403e-06 #85412, 2026-10-07"
    assert t.literal == "6.6403e-06" and t.comment == "#85412, 2026-10-07"
    assert lines(path)[t.line_no - 1] == t.line


@pytest.mark.parametrize("key, why", [
    ("dup", "ambiguous"),
    ("derived_x", "derived quantity"),
    ("only_in_method", "derived quantity"),
    ("expr", "not a plain numeric literal"),
    ("tup_a", "not a plain assignment"),
    ("multi", "not a plain assignment"),
    ("aug", "not a plain assignment"),
    ("ann", "not a plain assignment"),
    ("chain", "not a plain assignment"),
    ("nope", "not a plain assignment"),
])
def test_find_refusals(modules, key, why):
    mod, _ = modules(BASE)
    with pytest.raises(writeback.WritebackRefused, match=why):
        writeback.find_assignment(key, mod.Params)


# ---- apply -----------------------------------------------------------------------------

def test_apply_comments_the_old_line_and_inserts_the_new(modules):
    mod, path = modules(BASE)
    before = lines(path)
    rep = writeback.apply("t_pi", result(), mod.Params, note="kcal test", date="2026-10-09")
    assert rep.ok and rep.written, rep.reason
    assert rep.old_line == "        self.t_pi = 6.6403e-06 #85412, 2026-10-07"
    assert rep.commented_line == "        # self.t_pi = 6.6403e-06 #85412, 2026-10-07"
    assert rep.new_line == "        self.t_pi = 6.612e-06 #85600, 2026-10-09 kcal test"
    after = lines(path)
    i = before.index(rep.old_line)
    assert after[:i] == before[:i] and after[i + 2:] == before[i + 1:]
    assert after[i:i + 2] == [rep.commented_line, rep.new_line]
    assert rep.line_no == i + 2
    assert mod.Params().t_pi == 6.612e-06                     # the module was re-executed
    assert rep.old_value == 6.6403e-06 and rep.new_value == 6.612e-06
    assert not path.with_name(path.name + ".lock").exists()
    assert not list(path.parent.glob("*.tmp"))


def test_dry_run_writes_nothing(modules):
    mod, path = modules(BASE)
    raw = path.read_bytes()
    rep = writeback.apply("t_pi", result(), mod.Params, dry_run=True, date="2026-10-09")
    assert rep.ok and not rep.written and rep.new_line.strip().startswith("self.t_pi = 6.612e-06")
    assert path.read_bytes() == raw


@pytest.mark.parametrize("key, r, why", [
    ("t_pi", result(flags=[{"code": "change", "text": "too big"}]), "flagged: too big"),
    ("t_pi", result(fit={"ok": False, "reason": "no fit"}), "analysis failed"),
    ("t_pi", result(value=float("nan")), "not finite"),
    ("t_pi", result(value=float("inf")), "not finite"),
    ("t_pi", result(unc=float("nan")), "not finite"),
    ("t_pi", result(deferred=True), "deferred"),
    ("t_pi", result(unc=None), "no uncertainty"),
    ("t_pi", result(key="other"), "is for 'other'"),
    ("only_in_method", result(key="only_in_method"), "derived"),
    ("expr", result(key="expr"), "not a plain numeric literal"),
    ("dup", result(key="dup"), "ambiguous"),
    ("derived_x", result(key="derived_x"), "derived quantity"),
    ("N_iter", result(key="N_iter", value=50.5, unc=0.1), "not an integer"),
])
def test_apply_refusals_leave_the_file_alone(modules, key, r, why):
    mod, path = modules(BASE)
    raw = path.read_bytes()
    rep = writeback.apply(key, r, mod.Params)
    assert not rep.ok and why in rep.reason
    assert path.read_bytes() == raw


def test_int_and_wrapper_and_no_unc(modules):
    mod, path = modules(BASE)
    rep = writeback.apply("N_iter", result(key="N_iter", value=64.0, unc=2.0), mod.Params,
                          date="2026-10-09")
    assert rep.ok, rep.reason
    assert rep.new_line.strip() == "self.N_iter = np.int32(64) #85600, 2026-10-09"
    assert mod.Params().N_iter == 64
    rep = writeback.apply("amp", result(key="amp", value=0.4321987, unc=None), mod.Params,
                          allow_no_unc=True, date="2026-10-09")
    assert rep.ok and rep.new_line.strip().startswith("self.amp = 0.4321987 ")


def test_raman_frequency_line_keeps_its_digits(modules):
    mod, path = modules(BASE)
    rep = writeback.apply("freq", result(key="freq", value=41234570.1234, unc=5.0e4),
                          mod.Params, date="2026-10-09")
    assert rep.ok and "self.freq = 41.2345701234e6 #85600" in rep.new_line
    assert mod.Params().freq == 41234570.1234


def test_subclass_override_goes_to_the_subclass_file(modules):
    base, bpath = modules(BASE, prefix="calbase")
    sub, spath = modules(SUB.format(base=base.__name__), prefix="calsub")
    braw = bpath.read_bytes()
    rep = writeback.apply("amp", result(key="amp", value=0.5123, unc=0.0011), sub.Sub,
                          date="2026-10-09")
    assert rep.ok, rep.reason
    assert Path(rep.file) == spath and bpath.read_bytes() == braw
    assert "        # self.amp = 0.5 #sub override" in lines(spath)
    assert sub.Sub().amp == 0.5123
    # a key the subclass does not assign goes to the base
    rep = writeback.apply("t_pi", result(), sub.Sub, date="2026-10-09")
    assert rep.ok and Path(rep.file) == bpath


def test_failed_verification_restores_the_file(modules):
    base, _ = modules(BASE, prefix="calbase")
    sub, spath = modules(SUB.format(base=base.__name__), prefix="calsub")
    raw = spath.read_bytes()
    rep = writeback.apply("t_tof", result(key="t_tof", value=3.1234e-4, unc=1.1e-7), sub.Sub)
    assert not rep.ok and "verification failed" in rep.reason
    assert "original file is back" in rep.reason
    assert spath.read_bytes() == raw
    assert sub.Sub().t_tof == 250.e-6                       # module state restored too


def test_crlf_and_encoding_are_preserved(modules):
    mod, path = modules(BASE, crlf=True, bom=True)
    rep = writeback.apply("t_pi", result(), mod.Params, date="2026-10-09")
    assert rep.ok, rep.reason
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert raw.count(b"\n") == raw.count(b"\r\n")          # no bare LF anywhere
    assert "(µs scale)".encode("utf-8") in raw
    assert mod.Params().t_pi == 6.612e-06


def test_revert_reactivates_the_previous_value(modules):
    mod, path = modules(BASE)
    writeback.apply("t_pi", result(), mod.Params, date="2026-10-09")
    rep = writeback.revert("t_pi", mod.Params, date="2026-10-10")
    assert rep.ok, rep.reason
    assert rep.new_line == ("        self.t_pi = 6.6403e-06 #reverted 2026-10-10; "
                            "was #85412, 2026-10-07")
    assert mod.Params().t_pi == 6.6403e-06
    ls = lines(path)
    i = ls.index(rep.new_line)
    assert ls[i - 3:i] == ["        # self.t_pi = 6.0e-06 #100, 2026-01-01",
                           "        # self.t_pi = 6.6403e-06 #85412, 2026-10-07",
                           "        # self.t_pi = 6.612e-06 #85600, 2026-10-09"]
    # nothing commented above: refused, file unchanged
    raw = path.read_bytes()
    rep = writeback.revert("amp", mod.Params)
    assert not rep.ok and "nothing to revert to" in rep.reason
    assert path.read_bytes() == raw


N_REPEATS_LINE = "        self.N_repeats = 1\n"


def test_an_edit_before_the_lock_is_parsed_not_clobbered(modules, monkeypatch):
    """A person saves the file between the first look and the lock: the edit is
    read under the lock and the right line is changed (S1)."""
    mod, path = modules(BASE)
    real_lock = writeback.file_lock

    @contextlib.contextmanager
    def person_saves_first(target, timeout=1.0):
        text = Path(target).read_text()
        Path(target).write_text(text.replace(N_REPEATS_LINE,
                                             N_REPEATS_LINE + "        # a\n        # b\n"))
        with real_lock(target, timeout=timeout) as lk:
            yield lk
    monkeypatch.setattr(writeback, "file_lock", person_saves_first)
    rep = writeback.apply("t_pi", result(), mod.Params, date="2026-10-09")
    assert rep.ok, rep.reason
    ls = lines(path)
    i = ls.index("        # a")
    assert ls[i:i + 2] == ["        # a", "        # b"]
    assert ls[rep.line_no - 2] == "        # self.t_pi = 6.6403e-06 #85412, 2026-10-07"
    assert ls[rep.line_no - 1] == "        self.t_pi = 6.612e-06 #85600, 2026-10-09"


def test_lines_inserted_between_parse_and_write_refuse(modules, monkeypatch):
    """An editor that ignores the lock inserts lines after the parse: the write
    is refused and the file keeps exactly that edit (S1)."""
    mod, path = modules(BASE)
    real_format = writeback.format_value
    edited = BASE.replace(N_REPEATS_LINE, N_REPEATS_LINE + "        # inserted\n")

    def editor_inserts(*a, **k):
        path.write_text(edited)
        return real_format(*a, **k)
    monkeypatch.setattr(writeback, "format_value", editor_inserts)
    rep = writeback.apply("t_pi", result(), mod.Params)
    assert not rep.ok and "changed since it was read" in rep.reason
    assert path.read_text() == edited
    assert not list(path.parent.glob("*.kcal-backup"))


def test_a_held_lock_refuses(modules):
    mod, path = modules(BASE)
    raw = path.read_bytes()
    lock = path.with_name(path.name + ".lock")
    lock.write_text("pid 1 since earlier")
    rep = writeback.apply("t_pi", result(), mod.Params)
    assert not rep.ok and "is held" in rep.reason and "pid 1" in rep.reason
    assert path.read_bytes() == raw and lock.exists()        # never broken


def test_unimported_module_cannot_be_verified(modules):
    mod, path = modules(BASE)
    raw = path.read_bytes()
    sys.modules.pop(mod.__name__)
    rep = writeback.apply("t_pi", result(), mod.Params)
    assert not rep.ok and ("cannot verify" in rep.reason or "cannot read the source" in rep.reason)
    assert path.read_bytes() == raw


def test_history_lines(modules):
    mod, _ = modules(BASE)
    writeback.apply("t_pi", result(), mod.Params, date="2026-10-09")
    t, hist = writeback.history_lines("t_pi", mod.Params)
    assert [k for _, k, _ in hist] == ["commented", "commented", "active"]
    assert hist[-1][2].strip().startswith("self.t_pi = 6.612e-06")


def test_works_on_a_waxx_params_subclass(modules):
    text = ("from waxx.config.expt_params import ExptParams\n\n\n"
            "class P(ExptParams):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.t_pi = 6.6403e-06 #85412, 2026-10-07\n")
    mod, path = modules(text)
    rep = writeback.apply("t_pi", result(), mod.P, date="2026-10-09")
    assert rep.ok, rep.reason
    assert mod.P().t_pi == 6.612e-06
    # a key only the waxx base assigns lands in waxx's own file: refuse to test
    # that here -- dry run only, the real file is never touched by a test
    rep = writeback.apply("t_apd_slack", result(key="t_apd_slack", value=1.2e-5, unc=1e-7),
                          mod.P, dry_run=True)
    assert rep.ok and rep.file.endswith("expt_params.py") and not rep.written


# ---- S2: restore on any failure, backup until verified --------------------------------

def test_keyboard_interrupt_during_verify_restores(modules, monkeypatch):
    mod, path = modules(BASE)
    raw = path.read_bytes()

    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(writeback, "_verify", interrupted)
    with pytest.raises(KeyboardInterrupt):
        writeback.apply("t_pi", result(), mod.Params)
    assert path.read_bytes() == raw
    assert not list(path.parent.glob("*.kcal-backup")) and not list(path.parent.glob("*.lock"))


def test_a_failed_restore_says_file_left_modified(modules, monkeypatch, capsys):
    mod, path = modules(BASE)
    raw = path.read_bytes()
    real = writeback.replace_bytes
    calls = []

    def flaky(p, data, **k):
        calls.append(Path(p).name)
        if len(calls) == 2:                      # new bytes, then the restore (backup: 'xb')
            raise PermissionError("held open by an editor")
        return real(p, data, **k)
    monkeypatch.setattr(writeback, "replace_bytes", flaky)
    monkeypatch.setattr(writeback, "_verify", lambda *a, **k: "verification failed: test")
    rep = writeback.apply("t_pi", result(), mod.Params)
    assert not rep.ok and rep.left_modified
    assert "FILE LEFT MODIFIED, original at" in rep.reason
    backup = Path(rep.backup)
    assert backup.read_bytes() == raw and path.read_bytes() != raw
    assert "FILE LEFT MODIFIED" in capsys.readouterr().out


def test_backup_removed_after_success(modules):
    mod, path = modules(BASE)
    assert writeback.apply("t_pi", result(), mod.Params).ok
    assert not list(path.parent.glob("*.kcal-backup"))


# ---- S3: derived anywhere in the MRO; compute_derived in the verify ------------------

DERIVING_BASE = '''\
class Base:
    def __init__(self):
        self.amp = 0.41

    def compute_amp(self):
        self.amp = 2 * 0.2
'''

LITERAL_SUB = '''\
from {base} import Base


class Sub(Base):
    def __init__(self):
        super().__init__()
        self.amp = 0.5
'''

SETATTR_DERIVED = '''\
class P:
    def __init__(self):
        self.amp = 0.41

    def compute_derived(self):
        setattr(self, "amp", 1.0)       # invisible to the source scan
'''


def test_a_base_class_method_deriving_the_key_refuses(modules):
    base, _ = modules(DERIVING_BASE, prefix="calderb")
    sub, spath = modules(LITERAL_SUB.format(base=base.__name__), prefix="calders")
    raw = spath.read_bytes()
    rep = writeback.apply("amp", result(key="amp", value=0.5123, unc=0.0011), sub.Sub)
    assert not rep.ok and "Base.compute_amp" in rep.reason and "derived quantity" in rep.reason
    assert spath.read_bytes() == raw


def test_verify_runs_compute_derived(modules):
    mod, path = modules(SETATTR_DERIVED)
    raw = path.read_bytes()
    rep = writeback.apply("amp", result(key="amp", value=0.5123, unc=0.0011), mod.P)
    assert not rep.ok and "after compute_derived" in rep.reason and "original file is back" in rep.reason
    assert path.read_bytes() == raw


# ---- S4: the run's class is checked too ----------------------------------------------

SETATTR_SUB = '''\
from {base} import Params


class Sub(Params):
    def __init__(self):
        super().__init__()
        for k in ("t_pi",):
            setattr(self, k, 1.0)       # the run's class overrides it unseen
'''


def test_the_runs_class_must_see_the_value(modules):
    base, bpath = modules(BASE, prefix="calrunb")
    sub, _ = modules(SETATTR_SUB.format(base=base.__name__), prefix="calruns")
    raw = bpath.read_bytes()
    rep = writeback.apply("t_pi", result(), sub.Sub)
    assert not rep.ok and "Sub().t_pi is 1.0" in rep.reason
    assert bpath.read_bytes() == raw
    assert base.Params().t_pi == 6.6403e-06                 # module state back too


def test_subclass_run_sees_a_base_write(modules):
    base, bpath = modules(BASE, prefix="calrunb")
    sub, _ = modules(SUB.format(base=base.__name__), prefix="calruns")
    rep = writeback.apply("t_pi", result(), sub.Sub, date="2026-10-09")
    assert rep.ok, rep.reason
    assert sub.Sub().t_pi == 6.612e-06 and base.Params().t_pi == 6.612e-06


def test_an_unresolved_backup_blocks_every_write_back(modules, monkeypatch):
    """After a FILE LEFT MODIFIED, the backup holds the true original: no later
    apply or revert may overwrite it."""
    mod, path = modules(BASE)
    raw = path.read_bytes()
    real, real_verify = writeback.replace_bytes, writeback._verify
    calls = []

    def flaky(p, data, **k):
        calls.append(1)
        if len(calls) == 2:                      # new bytes, then the failing restore
            raise PermissionError("held open by an editor")
        return real(p, data, **k)
    monkeypatch.setattr(writeback, "replace_bytes", flaky)
    monkeypatch.setattr(writeback, "_verify", lambda *a, **k: "verification failed: test")
    rep = writeback.apply("t_pi", result(), mod.Params)
    assert rep.left_modified
    backup = Path(rep.backup)
    assert backup.read_bytes() == raw
    monkeypatch.setattr(writeback, "replace_bytes", real)
    monkeypatch.setattr(writeback, "_verify", real_verify)
    modified = path.read_bytes()
    for rep in (writeback.apply("t_pi", result(value=6.7e-06), mod.Params),
                writeback.apply("t_pi", result(value=6.7e-06), mod.Params, dry_run=True),
                writeback.revert("t_pi", mod.Params)):
        assert not rep.ok and "a previous write-back was left unverified" in rep.reason
        assert str(backup) in rep.reason
    assert backup.read_bytes() == raw and path.read_bytes() == modified
    # once a person restores the file and deletes the backup, write-backs work again
    path.write_bytes(raw)
    backup.unlink()
    sys.modules.pop(mod.__name__)
    importlib.invalidate_caches()
    mod = importlib.import_module(mod.__name__)
    assert writeback.apply("t_pi", result(), mod.Params, date="2026-10-09").ok
