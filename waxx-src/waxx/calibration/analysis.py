"""The analysis contract, the registry, and the time budget.

An analysis is a function ``calibrate(ad, key, **opts) -> CalResult``. It only
reads ``ad``; the one file it may write is its figure, at ``opts["figure_path"]``
when given (the framework passes the ledger's ``<key>/<run_id>.png``). It reports
how many shots it used (``n_used``) and every shot it left out
(``excluded = {"count", "reason"}``), and takes its uncertainty from the fit (no
uncertainty -> ``unc=None``, never a made-up one). On failure it returns a
result with ``fit={"ok": False, "reason": ...}`` or raises; both are recorded.

Analyses are found by name in the machine's registry modules (a config value;
waxx names none): ``resolve("rabi_pi_time", ["kexp.analysis.calibrations"])`` is
``kexp.analysis.calibrations.rabi_pi_time.calibrate``, or an attribute of that
package of the same name if it is callable.
"""

from __future__ import annotations

import importlib
import math
import pkgutil
import sys
import threading
import time
from typing import Callable, Optional, Sequence

from waxx.calibration.record import CalResult


class AnalysisNotFound(LookupError):
    pass


def resolve(name: str, registry_modules: Sequence[str]) -> Callable:
    """The ``calibrate`` function called ``name`` in the first registry module that has it."""
    if not isinstance(name, str) or not name.isidentifier():
        raise AnalysisNotFound(f"analysis name {name!r} is not an identifier")
    tried = []
    for pkg in registry_modules or ():
        try:
            mod = importlib.import_module(f"{pkg}.{name}")
        except ModuleNotFoundError as e:
            if e.name not in (f"{pkg}.{name}", pkg):
                raise                                # the analysis itself failed to import
            mod = None
        if mod is not None:
            fn = getattr(mod, "calibrate", None)
            if callable(fn):
                return fn
            tried.append(f"{pkg}.{name} (no calibrate())")
            continue
        try:
            pmod = importlib.import_module(pkg)
        except ModuleNotFoundError:
            tried.append(f"{pkg} (not importable)")
            continue
        fn = getattr(pmod, name, None)
        if callable(fn):
            return fn
        tried.append(f"{pkg}.{name}")
    raise AnalysisNotFound(f"no analysis {name!r} in the registry ({', '.join(tried) or 'empty'})")


def list_analyses(registry_modules: Sequence[str]) -> list:
    """[(name, module, first docstring line)] of every analysis module in the registry."""
    out = []
    for pkg in registry_modules or ():
        try:
            pmod = importlib.import_module(pkg)
        except ModuleNotFoundError as e:
            out.append(("?", pkg, f"not importable: {e}"))
            continue
        for info in pkgutil.iter_modules(getattr(pmod, "__path__", [])):
            if info.name.startswith("_"):
                continue
            full = f"{pkg}.{info.name}"
            try:
                mod = importlib.import_module(full)
            except Exception as e:
                out.append((info.name, full, f"IMPORT FAILED: {e!r}"))
                continue
            if callable(getattr(mod, "calibrate", None)):
                doc = (mod.__doc__ or mod.calibrate.__doc__ or "").strip().splitlines()
                out.append((info.name, full, doc[0] if doc else ""))
    return out


def failed_result(key: str, reason: str, *, analysis: str = "", deferred: bool = False) -> CalResult:
    return CalResult(key=key, value=math.nan, unc=None, method=analysis, analysis=analysis,
                     fit={"ok": False, "reason": reason}, deferred=deferred)


def call_with_budget(fn: Callable, budget_s: float, name: str = "calibration"):
    """``fn()`` in a daemon thread, waited on for ``budget_s``. Returns
    ``(finished, value, error)``; past the budget the thread is left running
    (never killed). Never raises."""
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:                    # recorded, never re-raised
            box["error"] = e

    th = threading.Thread(target=target, daemon=True, name=name)
    th.start()
    th.join(budget_s)
    if th.is_alive():
        return False, None, None
    return True, box.get("value"), box.get("error")


def needs_images(func: Callable, opts: Optional[dict] = None) -> bool:
    """Whether the analysis needs the camera images (OD, atom number): its
    ``needs_images`` attribute (a bool, or a callable taking the opts), else
    its module's ``NEEDS_IMAGES``, else False -- the run is then loaded without
    them (faster, and no image work inside end())."""
    flag = getattr(func, "needs_images", None)
    if flag is None:
        flag = getattr(sys.modules.get(getattr(func, "__module__", ""), None), "NEEDS_IMAGES", False)
    return bool(flag(dict(opts or {})) if callable(flag) else flag)


def run_analysis(func: Callable, ad, key: str, opts: Optional[dict] = None, *,
                 budget_s: float = 30.0, figure_path=None, name: str = "") -> CalResult:
    """``func(ad, key, **opts)`` in a daemon thread, waited on for ``budget_s``.

    Past the budget the result is ``deferred`` (no value, flagged); the thread
    is left to finish on its own -- it is never killed, and it writes nothing
    but its own figure. An exception, or a return that is not a CalResult,
    becomes a failed result with the reason. Never raises."""
    opts = dict(opts or {})
    if figure_path is not None:
        opts.setdefault("figure_path", str(figure_path))
    t0 = time.monotonic()
    done, res, err = call_with_budget(lambda: func(ad, key, **opts), budget_s,
                                      name=f"calibration:{key}")
    elapsed = time.monotonic() - t0
    if not done:
        return failed_result(key, f"did not finish within the {budget_s:g} s budget; deferred",
                             analysis=name, deferred=True)
    if err is not None:
        return failed_result(key, f"the analysis raised {type(err).__name__}: {err}", analysis=name)
    if not isinstance(res, CalResult):
        return failed_result(key, f"the analysis returned {type(res).__name__}, not a CalResult",
                             analysis=name)
    if res.key != key:
        return failed_result(key, f"the analysis returned a result for {res.key!r}",
                             analysis=name)
    if not res.analysis:
        res.analysis = name
    res.fit.setdefault("elapsed_s", round(elapsed, 3))
    return res
