"""The emit: what ``Expt.end_wax`` does with a run's declared calibrations.

For each ``self.calibrates(key, analysis, ...)`` of a run that saved completely:
load the run once, run the analysis under the budget, check it (policy + the
fixed rules), write the ledger record, print one ``[cal]`` line, and write the
value back when the experiment declared it, nothing flagged it and the
submitter did not veto (``WAXX_CAL_NO_WRITE_BACK=1``). Finally the run's own
file gets the root attribute ``calibration_emitted`` (the records, as JSON).

Every failure is printed and the run goes on: a calibration can never turn a
saved run into an error.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from waxx.calibration import writeback
from waxx.calibration.analysis import failed_result, resolve, run_analysis
from waxx.calibration.policy import policy_for
from waxx.calibration.record import CalResult, evaluate

VETO_ENV = "WAXX_CAL_NO_WRITE_BACK"
RUN_FILE_ATTR = "calibration_emitted"


@dataclass
class Declaration:
    key: str
    analysis: str
    write_back: bool = True
    allow_no_unc: bool = False
    opts: dict = field(default_factory=dict)


def vetoed() -> bool:
    """The submitter's veto: $WAXX_CAL_NO_WRITE_BACK set to anything but '', 0, false, no."""
    return os.environ.get(VETO_ENV, "").strip().lower() not in ("", "0", "false", "no")


def _say(text: str, out: Callable = print):
    try:
        out(text)
    except UnicodeEncodeError:                     # a pipe in a narrow code page
        out(text.replace("±", "+/-").encode("ascii", "replace").decode("ascii"))


# ---- formatting the [cal] line -------------------------------------------------------------

def _fmt_value_unc(v, u) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return f"{v!r}"
    if not math.isfinite(v):
        return f"{v!r}"
    if u is None:
        return f"{v!r} ± (no uncertainty)"
    try:
        u = float(u)
    except (TypeError, ValueError):
        return f"{v!r} ± {u!r}"
    if not math.isfinite(u) or u <= 0:
        return f"{v!r} ± {u!r}"
    p = math.floor(math.log10(u)) - 1                   # the uncertainty's 2nd digit
    e = math.floor(math.log10(abs(v))) if v != 0 else p + 1
    dec = max(0, min(15, e - p))
    return f"{v:.{dec}e} ± {u:.1e}"


def cal_line(r: CalResult, unit: str = "") -> str:
    unit = r.unit or unit
    exc = int((r.excluded or {}).get("count") or 0)
    head = f"[cal] {r.key} = {_fmt_value_unc(r.value, r.unc)}{(' ' + unit) if unit else ''}"
    was = "was unknown"
    if r.old_value is not None:
        was = f"was {r.old_value!r}"
        if r.rel_change is not None:
            was += f", {100 * r.rel_change:+.2f} %"
    return f"{head} (#{r.run_id}, {r.n_used} shots, {exc} excluded; {was})"


# ---- the pipeline -------------------------------------------------------------------------

def process_result(result: CalResult, config, *, params_cls=None, write_back: bool = False,
                   allow_no_unc: bool = False, out: Callable = print,
                   date: Optional[str] = None) -> CalResult:
    """Check, record, print, and (if allowed) apply one result. Never raises."""
    key = result.key
    try:
        policy = config.get_policy()
    except Exception as e:
        policy = {}
        _say(f"[cal] WARNING: could not load the write-back policy ({e!r}); checking {key} "
             f"without it, and not writing it back", out)
        write_back = False
    pol = policy_for(policy, key)
    if params_cls is not None:
        result.params_class = f"{params_cls.__module__}:{params_cls.__qualname__}"
        result.set_old_value(writeback.current_value(key, params_cls))
    evaluate(result, pol, allow_no_unc=allow_no_unc)

    ledger, recorded = None, False
    try:
        ledger = config.get_ledger()
        ledger.write_record(result)
        recorded = True
    except Exception as e:
        _say(f"[cal] WARNING: could not write the ledger record for {key} ({e!r})", out)

    _say(cal_line(result, pol.get("unit", "")), out)
    for w in (result.fit.get("warnings") or []):
        _say(f"[cal]   fit note: {w}", out)
    if result.excluded and int(result.excluded.get("count") or 0):
        _say(f"[cal]   excluded {result.excluded['count']} shot(s): {result.excluded.get('reason', '')}", out)

    hint = f"kcal apply {key} --run {result.run_id}"
    if result.flags:
        _say(f"[cal] not applied: flagged -- " + "; ".join(f["text"] for f in result.flags)
             + ". A flagged result is never written back.", out)
        return result
    if not write_back:
        _say(f"[cal] not applied: write_back is off for this calibration; to apply: {hint}", out)
        return result
    if vetoed():
        _say(f"[cal] not applied: vetoed by {VETO_ENV}; to apply: {hint}", out)
        return result
    if params_cls is None:
        _say(f"[cal] not applied: no params class to write to; to apply: {hint}", out)
        return result
    if not recorded:
        _say(f"[cal] not applied: the ledger record could not be written; nothing is written "
             f"back without one", out)
        return result
    rep = writeback.apply(key, result, params_cls, allow_no_unc=allow_no_unc, date=date)
    try:
        ledger.note_writeback(rep, run_id=result.run_id, by="emit")
    except Exception as e:
        _say(f"[cal] WARNING: could not record the write-back in the ledger ({e!r})", out)
    if rep.ok:
        result.applied, result.applied_file, result.applied_line = True, rep.file, rep.line_no
        _say(f"[cal] applied: {rep.file}:{rep.line_no} (kcal revert {key} undoes it)", out)
        _say(f"[cal]   - {rep.old_line.strip()}", out)
        _say(f"[cal]   + {rep.new_line.strip()}", out)
    else:
        _say(f"[cal] not applied: {rep.reason}; to apply: {hint}", out)
    return result


def run_declared(expt, declarations, *, out: Callable = print, date: Optional[str] = None) -> list:
    """The emit for one finished run (see the module doc). Returns the results."""
    if not declarations:
        return []
    keys = ", ".join(d.key for d in declarations)
    config = getattr(expt, "calibration_config", None)
    if config is None:
        _say(f"[cal] not run ({keys}): this experiment has no calibration_config", out)
        return []
    why = _why_not_calibrate(expt)
    if why:
        _say(f"[cal] not run ({keys}): {why}", out)
        return []
    rid = int(expt.run_info.run_id)
    try:
        ad = config.load_run(rid)
    except Exception as e:
        _say(f"[cal] not run ({keys}): could not load run {rid} ({type(e).__name__}: {e})", out)
        return []
    params_cls = type(expt.params)
    results = []
    for d in declarations:
        try:
            results.append(_one(expt, config, d, ad, rid, params_cls, out, date))
        except Exception as e:              # belt and braces: process_result never raises
            _say(f"[cal] WARNING: calibrating {d.key} failed ({type(e).__name__}: {e})", out)
    paths = list(_paths(getattr(expt.run_info, "filepath", None)))
    paths += [p for p in _paths(getattr(getattr(ad, "run_info", None), "filepath", None))
              if p not in paths]
    record_in_run_file(paths, results, out=out)
    return results


def _one(expt, config, d: Declaration, ad, rid, params_cls, out, date):
    try:
        func = resolve(d.analysis, config.registry_modules)
    except Exception as e:
        result = failed_result(d.key, f"analysis {d.analysis!r} not found ({e})", analysis=d.analysis)
    else:
        fig = None
        try:
            fig = config.get_ledger().figure_path(d.key, rid)
            fig.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            _say(f"[cal] WARNING: no figure for {d.key} ({e!r})", out)
            fig = None
        result = run_analysis(func, ad, d.key, d.opts, budget_s=config.budget_s,
                              figure_path=fig, name=d.analysis)
    result.run_id = rid
    result.expt_file = result.expt_file or _expt_name(expt)
    if result.deferred:
        _say(f"[cal] {d.key}: the analysis {d.analysis!r} took longer than "
             f"{config.budget_s:g} s and is deferred (no value this run)", out)
    return process_result(result, config, params_cls=params_cls, write_back=d.write_back,
                          allow_no_unc=d.allow_no_unc, out=out, date=date)


def _expt_name(expt) -> str:
    try:
        return str(expt._expt_file_stem())
    except Exception:
        return ""


def _why_not_calibrate(expt) -> str:
    """'' when the run is saved and complete, else why the emit must not run."""
    ri = expt.run_info
    if not getattr(ri, "save_data", False):
        return "this run saved no data"
    if not int(getattr(ri, "run_id", 0) or 0):
        return "this run has no run id"
    client = getattr(expt, "live_od_client", None)
    if client is None:
        return "this run was not saved through liveOD"
    if not getattr(expt, "_run_saved", False):
        return "the run's save did not complete"
    incomplete = (getattr(client, "last_end_run_reply", {}) or {}).get("incomplete")
    if incomplete:
        return f"the run was saved INCOMPLETE ({incomplete.get('reason', '')})"
    n, N = getattr(expt, "_shot_complete_count", 0), getattr(expt, "_N_shots_total", 0)
    if N and n < N:
        return f"the run ended after {n} of {N} shots"
    return ""


def _paths(fp):
    if fp is None:
        return
    if isinstance(fp, (str, os.PathLike)):
        if str(fp):
            yield str(fp)
        return
    try:
        for x in list(fp.ravel()) if hasattr(fp, "ravel") else list(fp):
            yield from _paths(x)
    except TypeError:
        return


# ---- the run file's root attribute ------------------------------------------------------

def record_in_run_file(paths, results, *, out: Callable = print, tries: int = 5,
                       wait: float = 1.0) -> bool:
    """Add (or extend) the root attribute ``calibration_emitted`` of the saved
    run file: a JSON list of the records. Only that attribute is ever written;
    an existing value must be a prefix of the new list (records are only
    appended), and the file must say ``run_complete``. Opened 'r+' (it is never
    created), held only for the write, retried while another process holds it."""
    if not results:
        return False
    path = next((p for p in paths if os.path.isfile(p)), None)
    if path is None:
        _say(f"[cal] WARNING: the run file was not found ({paths}); {RUN_FILE_ATTR} not written "
             f"(the ledger has the records)", out)
        return False
    import h5py
    new = [r.to_dict() for r in results]
    last = None
    for i in range(tries):
        try:
            with h5py.File(path, "r+") as f:
                if not bool(f.attrs.get("run_complete", False)):
                    _say(f"[cal] WARNING: {path} is not marked run_complete; {RUN_FILE_ATTR} "
                         f"not written", out)
                    return False
                old = f.attrs.get(RUN_FILE_ATTR)
                records = []
                if old is not None:
                    old = old.decode() if isinstance(old, bytes) else str(old)
                    records = json.loads(old)
                    if not isinstance(records, list):
                        raise ValueError(f"existing {RUN_FILE_ATTR} is not a list")
                f.attrs[RUN_FILE_ATTR] = json.dumps(records + new)
            return True
        except (OSError, BlockingIOError) as e:
            last = e
            time.sleep(wait)
        except Exception as e:
            _say(f"[cal] WARNING: {RUN_FILE_ATTR} not written to {path} ({type(e).__name__}: {e})",
                 out)
            return False
    _say(f"[cal] WARNING: {RUN_FILE_ATTR} not written to {path} after {tries} tries ({last!r}); "
         f"the ledger has the records", out)
    return False


# ---- emit_calibration: a result the experiment computed itself -------------------------

def emit_direct(expt, key, value, unc, *, write_back=False, allow_no_unc=False,
                out: Callable = print, date: Optional[str] = None, **meta) -> CalResult:
    """``Expt.emit_calibration``: build the CalResult and run it through the same
    checks / ledger / write-back as a declared calibration (after end())."""
    known = set(CalResult.__dataclass_fields__) - {"key", "value", "unc", "flags", "applied",
                                                   "applied_file", "applied_line", "old_value",
                                                   "rel_change", "params_class", "timestamp"}
    unknown = sorted(set(meta) - known)
    if unknown:
        raise TypeError(f"emit_calibration got unknown field(s) {unknown}")
    r = CalResult(key=key, value=value, unc=unc, **meta)
    r.run_id = r.run_id or int(getattr(expt.run_info, "run_id", 0) or 0)
    r.expt_file = r.expt_file or _expt_name(expt)
    r.analysis = r.analysis or "emit_calibration"
    config = getattr(expt, "calibration_config", None)
    why = _why_not_calibrate(expt)
    if config is None or why:
        _say(f"[cal] {key} not recorded: {why or 'this experiment has no calibration_config'} "
             f"(emit_calibration belongs after self.end())", out)
        return r
    process_result(r, config, params_cls=type(expt.params), write_back=write_back,
                   allow_no_unc=allow_no_unc, out=out, date=date)
    record_in_run_file(list(_paths(getattr(expt.run_info, "filepath", None))), [r], out=out)
    return r
