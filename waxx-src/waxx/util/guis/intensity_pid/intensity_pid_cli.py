"""Command line for an intensity PID server (the host wraps it with its server id).

    status [--json]          latest meas/set/out of both loops, hold, gains, warnings
    hold [on|off|toggle]     fix both outputs at their current values / release / toggle
    release                  same as hold off
    clear                    zero both integrators
    override LOOP on [VOLT]  loop 1|2 output = VOLT (or the last one) instead of the PID
    override LOOP off        the PID drives it again
    gains [--json]           p1 i1 p2 i2 as the board reads them back
    set NAME VALUE           one gain (p1 i1 p2 i2); the server re-applies it after a board reset
    tail [--seconds S]       print the 100 ms means as they arrive (default 10 s)
    raw LINE                 send one firmware command, print what came back

Exit codes: 0 done; 1 the server refused or the board did not answer; 2 usage;
3 the server was not found or did not answer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Callable, Optional, Sequence

__all__ = ["main", "build_parser"]

EXIT_OK, EXIT_REFUSED, EXIT_USAGE, EXIT_NO_SERVER = 0, 1, 2, 3


def _v(x, spec=".4f") -> str:
    if not isinstance(x, (int, float)) or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:{spec}}"


def build_parser(prog: str, description: str = "") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=description or __doc__.split("\n\n")[0],
                                epilog="exit codes: 0 done, 1 refused / no board reply, 2 usage, "
                                       "3 server not found",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", help="server address (skips discovery; needs --port)")
    p.add_argument("--port", type=int, help="server TCP port (with --host)")
    p.add_argument("--timeout", type=float, default=3.0, help="discovery timeout, s (default 3)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status", help="latest readings, hold, gains, warnings")
    s.add_argument("--json", action="store_true", help="the full snapshot as JSON")
    h = sub.add_parser("hold", help="fix both outputs (on), release (off) or toggle (default)")
    h.add_argument("state", nargs="?", default="toggle", choices=("on", "off", "toggle"))
    sub.add_parser("release", help="same as: hold off")
    sub.add_parser("clear", help="zero both integrators")
    o = sub.add_parser("override", help="force one output to a voltage (on) or give it back (off)")
    o.add_argument("loop", type=int, choices=(1, 2))
    o.add_argument("state", choices=("on", "off"))
    o.add_argument("volt", nargs="?", type=float, help="V (on only; default: the last one)")
    g = sub.add_parser("gains", help="p1 i1 p2 i2 read back from the board")
    g.add_argument("--json", action="store_true")
    st = sub.add_parser("set", help="set one gain")
    st.add_argument("name", choices=("p1", "i1", "p2", "i2"))
    st.add_argument("value", type=float)
    t = sub.add_parser("tail", help="print the 100 ms means as they arrive")
    t.add_argument("--seconds", type=float, default=10.0)
    r = sub.add_parser("raw", help="send one firmware command")
    r.add_argument("line", nargs="+")
    return p


def main(argv: Optional[Sequence[str]] = None, *, server_id: str, prog: Optional[str] = None,
         client_factory: Optional[Callable] = None, out=None) -> int:
    out = out or sys.stdout
    parser = build_parser(prog or server_id)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code not in (0, None) else EXIT_OK

    def say(text: str) -> None:
        print(text, file=out, flush=True)

    try:
        if client_factory is not None:
            client = client_factory()
        else:
            from waxx.util.guis.intensity_pid.intensity_pid_client import IntensityPIDClient  # noqa: PLC0415
            if args.host:
                if not args.port:
                    say("--host needs --port")
                    return EXIT_USAGE
                client = IntensityPIDClient.at(args.host, args.port, server_id)
            else:
                client = IntensityPIDClient(server_id, discovery_timeout=args.timeout)
    except Exception as exc:
        say(f"{server_id}: server not found ({exc})")
        return EXIT_NO_SERVER

    try:
        return _run(args, client, say)
    except OSError as exc:
        say(f"{server_id}: server did not answer ({type(exc).__name__}: {exc})")
        return EXIT_NO_SERVER


def _reply(reply: dict, say, ok_text: Callable[[dict], str]) -> int:
    if reply.get("status") == "ok":
        say(ok_text(reply))
        return EXIT_OK
    say(f"refused: {reply.get('message') or reply.get('status')}")
    return EXIT_REFUSED


def _run(args, client, say) -> int:
    cmd = args.cmd
    if cmd == "status":
        snap = client.get_snapshot()
        if snap.get("status") != "ok":
            return _reply(snap, say, str)
        if args.json:
            say(json.dumps(snap))
            return EXIT_OK
        row = snap.get("latest") or {}
        gains = snap.get("settings") or {}
        names = snap.get("loop_names") or ["loop 1", "loop 2"]
        held = snap.get("held")
        say(f"{names[0]}: meas {_v(row.get('meas1'))} set {_v(row.get('set1'))} out {_v(row.get('out1'))} V | "
            f"{names[1]}: meas {_v(row.get('meas2'))} set {_v(row.get('set2'))} out {_v(row.get('out2'))} V | "
            f"hold {'--' if held is None else int(held)} | "
            + " ".join(f"{k} {_v(gains.get(k), 'g')}" for k in ("p1", "i1", "p2", "i2"))
            + "".join(f" | override{i} {_v(gains.get(f'ovolt{i}'), '.3f')} V" for i in (1, 2)
                      if gains.get(f"override{i}") == 1)
            + f" | com {(snap.get('com') or {}).get('status', '?')}, age {_v(snap.get('data_age_s'), '.2f')} s")
        for w in snap.get("warnings") or []:
            say(f"{w.get('level')}: {w.get('text')}")
        return EXIT_OK
    if cmd in ("hold", "release"):
        state = "off" if cmd == "release" else args.state
        on = {"on": True, "off": False, "toggle": None}[state]
        return _reply(client.hold(on), say,
                      lambda r: "hold on: both outputs fixed" if r.get("on")
                      else "hold off: loops running")
    if cmd == "override":
        on = args.state == "on"
        if not on and args.volt is not None:
            say("override off takes no voltage")
            return EXIT_USAGE
        return _reply(client.override(args.loop, on, args.volt), say,
                      lambda r: f"override{r.get('loop')} on: output {_v(r.get('volt'), '.3f')} V"
                      if r.get("on") else f"override{r.get('loop')} off: the PID drives it")
    if cmd == "clear":
        return _reply(client.clear(), say, lambda r: r.get("message") or "integrators cleared")
    if cmd == "gains":
        reply = client.refresh_settings()
        if reply.get("status") != "ok":
            return _reply(reply, say, str)
        gains = reply.get("settings") or {}
        say(json.dumps(gains) if args.json else
            " ".join(f"{k} {_v(gains.get(k), 'g')}" for k in ("p1", "i1", "p2", "i2")))
        return EXIT_OK if all(gains.get(k) is not None for k in ("p1", "i1", "p2", "i2")) else EXIT_REFUSED
    if cmd == "set":
        return _reply(client.set_value(args.name, args.value), say,
                      lambda r: f"{r.get('name')} = {_v(r.get('value'), 'g')}"
                      + (f" ({r['note']})" if r.get("note") else ""))
    if cmd == "tail":
        since = int(client.get_snapshot().get("seq") or 0)
        say("time      meas1    set1     meas2    set2     out1     out2     hold  (100 ms means)")
        end = time.monotonic() + max(0.0, args.seconds)
        while time.monotonic() < end:
            time.sleep(0.25)
            reply = client.get_history(since=since)
            if reply.get("status") != "ok":
                return _reply(reply, say, str)
            for r in reply.get("rows", []):
                stamp = time.strftime("%H:%M:%S", time.localtime(r.get("t") or 0))
                say(f"{stamp}  " + "  ".join(_v(r.get(k), "+.4f") for k in
                                            ("meas1", "set1", "meas2", "set2", "out1", "out2"))
                    + f"  {_v(r.get('hold_frac'), '.1f')}")
            since = int(reply.get("seq", since))
        return EXIT_OK
    if cmd == "raw":
        reply = client.raw(" ".join(args.line))
        if reply.get("status") == "ok":
            for line in reply.get("lines", []):
                say(line)
            return EXIT_OK
        return _reply(reply, say, str)
    say(f"unknown command {cmd}")
    return EXIT_USAGE
