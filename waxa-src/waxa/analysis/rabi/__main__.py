"""``python -m waxa.analysis.rabi`` -- fit a Rabi scan from the command line.

    python -m waxa.analysis.rabi 83092
    python -m waxa.analysis.rabi 83092 83091                 # joint fit (shared Rabi parameters)
    python -m waxa.analysis.rabi 83092 83091 --separate      # one fit per run
    python -m waxa.analysis.rabi 0 --compare t_raman_pi_pulse --plot rabi.png
    python -m waxa.analysis.rabi 83092 --model exp --bootstrap 200 --json

Runs are loaded read-only (``atomdata(rid, roi_id=..., lite=False)``); 0 is the latest
complete run.  ``--compare PARAM`` compares the fitted pi time with the value the run
itself used (``ad.p.PARAM``); ``--config-line PARAM`` prints the params-file line.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m waxa.analysis.rabi",
                                 description="Fit a Rabi oscillation (pi time) from one or more runs.")
    ap.add_argument("runs", type=int, nargs="+", help="run ids (0 = latest, -N = Nth most recent)")
    ap.add_argument("--signal", default="atom_number",
                    help="attribute of atomdata (dots allowed, e.g. data.apd) or 'sumod_contrast'")
    ap.add_argument("--xvar", default=None, help="pulse-length xvar (default: the only one)")
    ap.add_argument("--model", default="auto", choices=["auto", "none", "exp", "gauss"])
    ap.add_argument("--noise", default="auto", choices=["auto", "sem", "pooled", "unweighted"])
    ap.add_argument("--sem-floor", type=float, default=0.5)
    ap.add_argument("--separate", action="store_true", help="fit each run on its own instead of jointly")
    ap.add_argument("--roi", default="auto", help="roi_id for atomdata (default 'auto')")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N")
    ap.add_argument("--compare", metavar="PARAM", default=None,
                    help="compare the fitted pi time with ad.p.PARAM (e.g. t_raman_pi_pulse)")
    ap.add_argument("--config-line", metavar="PARAM", default=None, help="print 'self.PARAM = ... #run, date'")
    ap.add_argument("--plot", metavar="PATH", default=None, help="save the figure here")
    ap.add_argument("--show", action="store_true", help="show the figure")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    from waxa.analysis.rabi import rabi
    roi = int(args.roi) if args.roi.lstrip("-").isdigit() else args.roi
    res = rabi(args.runs, signal=args.signal, xvar=args.xvar, model=args.model, noise=args.noise,
               joint=not args.separate, roi_id=roi, sem_floor=args.sem_floor, bootstrap=args.bootstrap)
    fits = res if isinstance(res, list) else [res]
    rc = 0
    for f in fits:
        if args.json:
            print(json.dumps(f.to_dict(), indent=1, default=lambda o: float(o) if np.ndim(o) == 0 else list(o)))
        else:
            print(f.summary())
        rc |= 0 if f.ok else 1
        ref = None
        if args.compare and f.ok and f.params is not None and hasattr(f.params, args.compare):
            ref = float(np.ravel(getattr(f.params, args.compare))[0])
            print("  " + f.compare(ref, args.compare)[2])
        elif args.compare:
            print(f"  (cannot compare: {args.compare} not in the run's params)")
        if args.config_line and f.ok:
            print("  " + f.config_line(args.config_line))
        if args.plot or args.show:
            import matplotlib
            if not args.show:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, _ = f.plot(reference=ref)
            if args.plot:
                path = args.plot if len(fits) == 1 else args.plot.replace(".", f"_{f.run_ids[0]}.", 1)
                fig.savefig(path, dpi=120, bbox_inches="tight")
                print(f"  figure: {path}")
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    return rc


if __name__ == "__main__":
    sys.exit(main())
