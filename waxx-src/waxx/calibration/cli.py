"""``kcal``: look at, apply and revert calibration write-backs.

    kcal show <key> [-n N]                       ledger history + the file's history lines
    kcal apply <key> --run <id> [--dry-run] [--note TEXT] [--allow-no-unc]
    kcal revert <key> [--dry-run]
    kcal check <key> --run <id> [--allow-no-unc]  the flags only, nothing written
    kcal analyses                                the registry's analyses

The machine's config comes from ``--config module:attr`` or, without it, the
``WAXX_CALIBRATION_CONFIG`` environment variable in the same form (the K
machine: ``kexp.config.calibration:CALIBRATION_CONFIG``). ``--params
module:Class`` names the params class to write to; otherwise apply uses the
class recorded with the result, and revert the config's.

``apply`` re-checks the ledger record against the current policy and the value
now in the file before writing; a flagged result is refused. Also runnable as
``python -m waxx.calibration``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from waxx.calibration import writeback
from waxx.calibration.analysis import list_analyses
from waxx.calibration.config import ENV_VAR, load_config
from waxx.calibration.emit import _fmt_value_unc, cal_line
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
    a = sub.add_parser("apply", help="write a ledger result back into the params file")
    a.add_argument("key")
    a.add_argument("--run", type=int, required=True)
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--note", default="")
    a.add_argument("--allow-no-unc", action="store_true")
    a.add_argument("--params")
    r = sub.add_parser("revert", help="re-activate the previous commented value")
    r.add_argument("key")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--params")
    c = sub.add_parser("check", help="the flags a ledger result has now (writes nothing)")
    c.add_argument("key")
    c.add_argument("--run", type=int, required=True)
    c.add_argument("--allow-no-unc", action="store_true")
    c.add_argument("--params")
    sub.add_parser("analyses", help="the analyses in the registry")
    return p


def _params_cls(cfg, args, record=None):
    spec = getattr(args, "params", None) or (record.params_class if record is not None
                                            and record.params_class else None)
    return cfg.get_params_class(spec)


def _recheck(cfg, args, rec, cls):
    rec.set_old_value(writeback.current_value(rec.key, cls))
    return evaluate(rec, policy_for(cfg.get_policy(), rec.key), allow_no_unc=args.allow_no_unc)


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


def cmd_apply(cfg, args, out):
    led = cfg.get_ledger()
    rec = led.load(args.key, args.run)
    cls = _params_cls(cfg, args, rec)
    flags = _recheck(cfg, args, rec, cls)
    out(cal_line(rec))
    if flags:
        out("REFUSED: flagged -- " + "; ".join(f["text"] for f in flags))
        return 2
    rep = writeback.apply(args.key, rec, cls, dry_run=args.dry_run, note=args.note,
                          allow_no_unc=args.allow_no_unc)
    out(str(rep))
    if rep.ok and not args.dry_run:
        led.note_writeback(rep, run_id=args.run, by="kcal")
        out(f"(kcal revert {args.key} undoes it)")
    return 0 if rep.ok else 2


def cmd_revert(cfg, args, out):
    cls = _params_cls(cfg, args)
    rep = writeback.revert(args.key, cls, dry_run=args.dry_run)
    out(str(rep))
    if rep.ok and not args.dry_run:
        cfg.get_ledger().note_writeback(rep, run_id=None, by="kcal")
    return 0 if rep.ok else 2


def cmd_check(cfg, args, out):
    rec = cfg.get_ledger().load(args.key, args.run)
    flags = _recheck(cfg, args, rec, _params_cls(cfg, args, rec))
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


COMMANDS = {"show": cmd_show, "apply": cmd_apply, "revert": cmd_revert, "check": cmd_check,
            "analyses": cmd_analyses}


def main(argv: Optional[list] = None, out=print) -> int:
    args = _parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        return COMMANDS[args.cmd](cfg, args, out)
    except (RuntimeError, LookupError, TypeError, ValueError, ImportError) as e:
        out(f"kcal {args.cmd}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
