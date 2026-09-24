"""``python -m waxa.climate`` - poll the lab climate sensors from a terminal.

    python -m waxa.climate hosts
    python -m waxa.climate items  [--host K]
    python -m waxa.climate now    [--host K] [-F]
    python -m waxa.climate history "Machine Table" --since 24h [--until ...] [--csv out.csv]
    python -m waxa.climate trends  "Machine Table" --since 30d
    python -m waxa.climate room   --since 6h [--csv out.csv]

``--since`` / ``--until`` take a duration (``90m``, ``24h``, ``7d``) measured
back from now, or an absolute local time (``2026-09-24``,
``"2026-09-24 08:00"``).  Temperatures print in degC unless ``-F``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import re
import sys

import numpy as np

from waxa.climate.client import ClimateClient, to_unix

_DUR = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhd])$")
_MULT = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_when(s: str | None, default: float) -> float:
    if s is None:
        return default
    m = _DUR.match(s.strip())
    if m:
        return _dt.datetime.now().timestamp() - float(m.group(1)) * _MULT[m.group(2)]
    return to_unix(s)


def _fmt_t(t: float) -> str:
    return _dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def _write_csv(path: str, header: list[str], rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m waxa.climate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="K", help="Zabbix host name (default K)")
    p.add_argument("-F", "--fahrenheit", action="store_true", help="leave temperatures in degF")
    p.add_argument("--url", default=None, help="override the Zabbix API URL")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("hosts", help="list Zabbix hosts")
    sub.add_parser("items", help="list sensor channels on --host")
    sub.add_parser("now", help="latest reading of every sensor on --host")

    for name, help_ in (("history", "raw 1-min samples of one sensor"),
                        ("trends", "hourly min/avg/max of one sensor")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("item", help="sensor name or key, e.g. 'Machine Table'")
        sp.add_argument("--since", required=True)
        sp.add_argument("--until", default=None)
        sp.add_argument("--csv", default=None, help="write the samples to this file")
        sp.add_argument("-n", type=int, default=20, help="rows to print (default 20, 0 = all)")

    sp = sub.add_parser("room", help="every sensor on --host over a window")
    sp.add_argument("--since", required=True)
    sp.add_argument("--until", default=None)
    sp.add_argument("--csv", default=None, help="write a wide table (one column per sensor)")

    a = p.parse_args(argv)
    kw = {"host": a.host, "temperature_unit": "F" if a.fahrenheit else "C"}
    if a.url:
        kw["url"] = a.url
    cc = ClimateClient(**kw)

    if a.cmd == "hosts":
        for name, hid in sorted(cc.hosts().items()):
            print(f"{hid:>6}  {name}")
        return 0

    if a.cmd == "items":
        for it in cc.items(numeric_only=False):
            print(f"{it.itemid:>6}  {it.name:<24} key={it.key:<24} units={it.units or '-':<3} every {it.delay}")
        return 0

    if a.cmd == "now":
        snap = cc.snapshot()
        units = {it.name: it.units for it in cc.items()}
        for name, (v, t) in sorted(snap.items()):
            u = units.get(name, "")
            if u == "F" and not a.fahrenheit:
                u = "C"
            print(f"{name:<24} {v:8.2f} {u:<3}  ({_fmt_t(t)})")
        return 0

    now = _dt.datetime.now().timestamp()
    t_from = _parse_when(a.since, now - 3600)
    t_to = _parse_when(a.until, now)

    if a.cmd in ("history", "trends"):
        s = cc.history(a.item, t_from, t_to) if a.cmd == "history" else cc.trends(a.item, t_from, t_to)
        print(repr(s))
        st = s.summary()
        print(f"n={st['n']}  mean={st['mean']:.3f}  std={st['std']:.3f}  "
              f"min={st['min']:.3f}  max={st['max']:.3f}  [{s.units}]")
        rows = list(zip(s.t, s.v))
        show = rows if a.n == 0 else rows[-a.n:]
        for t, v in show:
            print(f"{_fmt_t(t)}  {v:9.3f}")
        if a.csv:
            hdr = ["unix_s", "local_time", f"{s.name} [{s.units}]"]
            out = [(f"{t:.3f}", _fmt_t(t), f"{v:.4f}") for t, v in rows]
            if s.kind == "trend":
                hdr += ["min", "max", "num"]
                out = [r + (f"{mn:.4f}", f"{mx:.4f}", str(n)) for r, mn, mx, n in
                       zip(out, s.extra["min"], s.extra["max"], s.extra["num"])]
            _write_csv(a.csv, hdr, out)
        return 0

    if a.cmd == "room":
        room = cc.room(t_from, t_to)
        for name, s in room.items():
            st = s.summary()
            print(f"{name:<24} n={st['n']:<5} mean={st['mean']:8.3f}  std={st['std']:7.3f}  "
                  f"min={st['min']:8.3f}  max={st['max']:8.3f}  [{s.units}]")
        if a.csv:
            # Wide table on the union of sample times; each sensor sampled by
            # nearest neighbour within 90 s so the once-a-minute rows line up.
            t_all = np.unique(np.concatenate([np.round(s.t) for s in room.values() if len(s)]))
            cols = {n: s.at(t_all, max_gap_s=90.0) for n, s in room.items()}
            hdr = ["unix_s", "local_time"] + [f"{n} [{s.units}]" for n, s in room.items()]
            rows = [[f"{t:.0f}", _fmt_t(t)] + ["" if not np.isfinite(cols[n][i]) else f"{cols[n][i]:.4f}"
                                               for n in room] for i, t in enumerate(t_all)]
            _write_csv(a.csv, hdr, rows)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
