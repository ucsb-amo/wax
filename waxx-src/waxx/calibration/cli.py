"""``kcal``: look at, emit, apply and revert calibration write-backs.

    kcal show <key> [-n N]                       ledger history + the file's history lines
    kcal emit <key> --run <id> --analysis NAME [--opts JSON] [--budget S]
                                                 run an analysis offline -> ledger record
                                                 (never written back by itself)
    kcal apply <key> --run <id> [--dry-run] [--note TEXT] [--allow-no-unc] [--force]
    kcal revert <key> [--dry-run]
    kcal check <key> --run <id> [--allow-no-unc]  the flags only, nothing written
    kcal analyses                                the registry's analyses

The machine's config comes from ``--config module:attr`` or, without it, the
``WAXX_CALIBRATION_CONFIG`` environment variable in the same form (the K
machine: ``kexp.config.calibration:CALIBRATION_CONFIG``). ``--params
module:Class`` names the params class to write to; otherwise apply uses the
class recorded with the result, and revert / emit the config's.

``apply`` refuses a flagged result: the flags it was emitted with plus the
flags it gets now (current policy, the value now in the file). It also refuses
a record when the ledger holds a newer emit or apply for the key, unless
``--force`` (journaled with who and when). Every refused apply is journaled.
``emit`` is the way to analyse a run whose in-run analysis was deferred or
failed; it reads the run, never writes it. Also runnable as
``python -m waxx.calibration``.
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import json
import socket
import sys
from typing import Optional

from waxx.calibration import writeback
from waxx.calibration.analysis import (list_analyses, needs_images, resolve, run_analysis,
                                       sweep_stale_figure_tmp)
from waxx.calibration.config import ENV_VAR, load_config
from waxx.calibration.emit import _fmt_value_unc, cal_line, offline_run_flags, process_result
from waxx.calibration.policy import policy_for
from waxx.calibration.record import evaluate


def _parser():
    p = argparse.ArgumentParser(prog="kcal", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", help=f"module:attr of the CalibrationConfig (default: ${ENV_VAR})")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="ledger history and the file's history lines")
    s.add_argument("key")
    s.add_argument("-n", type=int, default=10)
    s.add_argument("--params")
    e = sub.add_parser("emit", help="run an analysis on a saved run offline (ledger only)")
    e.add_argument("key")
    e.add_argument("--run", type=int, required=True)
    e.add_argument("--analysis", required=True)
    e.add_argument("--opts", default="{}", help="the analysis options, as a JSON object")
    e.add_argument("--budget", type=float, default=3600.0, help="seconds (default 3600)")
    e.add_argument("--params")
    a = sub.add_parser("apply", help="write a ledger result back into the params file")
    a.add_argument("key")
    a.add_argument("--run", type=int, required=True)
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--note", default="")
    a.add_argument("--allow-no-unc", action="store_true",
                   help="accept a result without an uncertainty (written with its full repr). "
                        "It does NOT clear a no_unc flag the record was emitted with "
                        "(declare allow_no_unc=True in the experiment for that): stored "
                        "flags always refuse")
    a.add_argument("--force", action="store_true",
                   help="apply although the ledger has a newer emit / apply (journaled)")
    a.add_argument("--params")
    r = sub.add_parser("revert", help="re-activate the previous commented value")
    r.add_argument("key")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--params")
    c = sub.add_parser("check", help="the flags a ledger result has (writes nothing)")
    c.add_argument("key")
    c.add_argument("--run", type=int, required=True)
    c.add_argument("--allow-no-unc", action="store_true",
                   help="accept a result without an uncertainty (written with its full repr). "
                        "It does NOT clear a no_unc flag the record was emitted with "
                        "(declare allow_no_unc=True in the experiment for that): stored "
                        "flags always refuse")
    c.add_argument("--params")
    sub.add_parser("analyses", help="the analyses in the registry")
    return p


def _who():
    try:
        user = getpass.getuser()
    except Exception:
        user = "?"
    return f"{user}@{socket.gethostname()}"


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _params_cls(cfg, args, record=None):
    spec = getattr(args, "params", None) or (record.params_class if record is not None
                                            and record.params_class else None)
    return cfg.get_params_class(spec)


def _all_flags(cfg, args, rec, cls):
    """The flags the record was emitted with, plus those it gets now (current
    policy; the value now in the file as the baseline), by code."""
    stored = list(rec.flags)
    rec.set_old_value(writeback.current_value(rec.key, cls))
    now = evaluate(rec, policy_for(cfg.get_policy(), rec.key), allow_no_unc=args.allow_no_unc)
    seen, union = set(), []
    for f in stored + now:
        if f["code"] not in seen:
            seen.add(f["code"])
            union.append(f)
    rec.flags = union
    return union


def _newer_entries(led, key, run_id):
    """Ledger events for ``key`` after this run's last emit that make the record
    stale: another run's emit, or any write-back that was made."""
    events = list(led.events(key))
    idx = max((i for i, e in enumerate(events)
               if e.get("event") == "emit" and e.get("run_id") == run_id), default=None)
    if idx is None:
        return []
    return [e for e in events[idx + 1:]
            if (e.get("event") == "emit" and e.get("run_id") != run_id)
            or (e.get("event") in ("apply", "revert") and e.get("ok") and e.get("written"))]


def cmd_show(cfg, args, out):
    led = cfg.get_ledger()
    hist = led.history(args.key, args.n)
    out(f"ledger ({led.jsonl}): last {len(hist)} result(s) for {args.key}")
    for r in hist:
        flags = ",".join(f["code"] for f in r.flags) or "-"
        state = "APPLIED" if r.applied else ("deferred" if r.deferred else "not applied")
        out(f"  #{r.run_id:<7} {r.timestamp:19}  {_fmt_value_unc(r.value, r.unc):>28}  "
            f"n={r.n_used:<4} flags={flags:<12} {state}")
    try:
        t, lines = writeback.history_lines(args.key, _params_cls(cfg, args))
    except writeback.WritebackRefused as e:
        out(f"file: {e}")
        return 0
    out(f"file ({t.file}, class {t.cls_name}):")
    for n, kind, line in lines:
        mark = {"active": ">", "commented": " ", "other": "?"}[kind]
        out(f"  {mark} {n:5d}  {line.strip()}")
    return 0


def cmd_emit(cfg, args, out):
    opts = json.loads(args.opts)
    if not isinstance(opts, dict):
        raise ValueError("--opts must be a JSON object")
    func = resolve(args.analysis, cfg.registry_modules)
    cls = _params_cls(cfg, args)
    ad = cfg.load_run(args.run, needs_images=needs_images(func, opts))
    led = cfg.get_ledger()
    fig = led.figure_path(args.key, args.run)
    fig.parent.mkdir(parents=True, exist_ok=True)
    sweep_stale_figure_tmp(fig.parent)
    res = run_analysis(func, ad, args.key, opts, budget_s=args.budget, figure_path=fig,
                       name=args.analysis)
    res.run_id = args.run
    res.fit.setdefault("emitted_by", f"kcal emit ({_who()})")
    process_result(res, cfg, params_cls=cls, write_back=False, out=out,
                   run_value=getattr(getattr(ad, "params", None), args.key, None),
                   extra_flags=offline_run_flags(ad, args.key))
    return 0 if (res.fit_ok and not res.deferred) else 1


def _journal_refusal(led, args, reason, flags=()):
    led.append_event({"event": "apply_refused", "key": args.key, "run_id": args.run,
                      "by": _who(), "timestamp": _now(), "reason": reason,
                      "flags": [f["code"] for f in flags]})


def cmd_apply(cfg, args, out):
    led = cfg.get_ledger()
    rec = led.load(args.key, args.run)
    cls = _params_cls(cfg, args, rec)
    flags = _all_flags(cfg, args, rec, cls)
    out(cal_line(rec))
    if flags:
        reason = "flagged -- " + "; ".join(f["text"] for f in flags)
        out(f"REFUSED: {reason}")
        if not args.dry_run:
            _journal_refusal(led, args, reason, flags)
        return 2
    newer = _newer_entries(led, args.key, args.run)
    if newer:
        desc = ", ".join(f"{e.get('event')} #{e.get('run_id')} ({e.get('timestamp', '?')})"
                         for e in newer)
        if not args.force:
            reason = f"the ledger has newer entries for {args.key}: {desc}"
            out(f"REFUSED: {reason} (--force applies this record anyway, journaled)")
            if not args.dry_run:
                _journal_refusal(led, args, reason)
            return 2
        if not args.dry_run:
            led.append_event({"event": "force", "key": args.key, "run_id": args.run,
                              "by": _who(), "timestamp": _now(),
                              "reason": f"applied over newer entries: {desc}"})
        out(f"--force: applying over newer entries ({desc}); journaled")
    rep = writeback.apply(args.key, rec, cls, dry_run=args.dry_run, note=args.note,
                          allow_no_unc=args.allow_no_unc)
    out(str(rep))
    if not args.dry_run:
        led.note_writeback(rep, run_id=args.run, by=f"kcal ({_who()})")
    if rep.ok and not args.dry_run:
        out(f"(kcal revert {args.key} undoes it)")
    return 0 if rep.ok else 2


def cmd_revert(cfg, args, out):
    cls = _params_cls(cfg, args)
    rep = writeback.revert(args.key, cls, dry_run=args.dry_run)
    out(str(rep))
    if not args.dry_run:
        cfg.get_ledger().note_writeback(rep, run_id=None, by=f"kcal ({_who()})")
    return 0 if rep.ok else 2


def cmd_check(cfg, args, out):
    rec = cfg.get_ledger().load(args.key, args.run)
    flags = _all_flags(cfg, args, rec, _params_cls(cfg, args, rec))
    out(cal_line(rec))
    if not flags:
        out("no flags")
    for f in flags:
        out(f"  {f['code']}: {f['text']}")
    return 1 if flags else 0


def cmd_analyses(cfg, args, out):
    rows = list_analyses(cfg.registry_modules)
    if not rows:
        out(f"no analyses in {list(cfg.registry_modules)}")
    for name, mod, doc in rows:
        out(f"  {name:24} {mod:48} {doc}")
    return 0


COMMANDS = {"show": cmd_show, "emit": cmd_emit, "apply": cmd_apply, "revert": cmd_revert,
            "check": cmd_check, "analyses": cmd_analyses}


def main(argv: Optional[list] = None, out=print) -> int:
    args = _parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        return COMMANDS[args.cmd](cfg, args, out)
    except (RuntimeError, LookupError, TypeError, ValueError, ImportError, OSError) as e:
        out(f"kcal {args.cmd}: {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
