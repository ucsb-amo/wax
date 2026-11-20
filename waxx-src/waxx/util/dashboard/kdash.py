"""kdash: list and close the dashboards running on the lab network.

Every dashboard advertises a control port by beacon
(:mod:`waxx.util.dashboard.remote_control`).  Usage::

    kdash list                          # every dashboard: host, kind, servers, uptime
    kdash close kong                    # every dashboard on kong (server and client)
    kdash close kong --kind client      # only kong's client dashboard
    kdash close pc1 pc2 --yes           # several hosts, no confirmation prompt
    kdash close --all                   # every dashboard on the network
    kdash close kong --force            # even while a run holds the machine
    kdash kill kong --kind server       # hard kill (frozen window): no graceful stop

``close`` runs each dashboard's own close (as its X button): servers are asked
to stop, COM servers get ``--com-timeout`` seconds (default 30) to release
their ports before they are force-killed, survivors are killed.  A server
dashboard refuses while a run is in progress, or the machine's state cannot be
read, unless ``--force``.  A person at the dashboard's PC can still press
"Cancel close" in its COM dialog.

``kill`` kills the dashboard and every server it started at once, with no
graceful stop (COM ports may stay held until Windows releases them): only for a
dashboard whose window no longer answers.

Agents (``WAXX_OWNER=agent``) may list but not close or kill.  Like kq's
owner, that rule is advisory: the dashboard believes what the request says
(there is no authentication).

``kill`` is accepted while a close is under way (a stuck close is what it is
for), and kdash reports success only once the dashboard stops answering.

Targets are host names (case-insensitive), IP addresses or full beacon ids
(``dashboard:<kind>:<host>:<pid>``).  Every target must match a dashboard, or
nothing is done.

Exit codes: 0 everything asked for was done; 1 something was refused, failed,
was cancelled or did not report back; 2 usage error or no dashboard matched;
3 the confirmation was declined.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from waxx.util.dashboard.remote_control import (
    BEACON_PREFIX,
    COM_TIMEOUT_RANGE_S,
    DEFAULT_COM_TIMEOUT_S,
    parse_beacon_id,
)

OWNER_ENV = "WAXX_OWNER"
#: Beacons older than this are not listed (a closed dashboard's entry lingers
#: in the discovery cache otherwise).
BEACON_MAX_AGE_S = 3.0
CONNECT_TIMEOUT_S = 3.0
#: Time to the first reply line of a close (the server dashboard's busy check
#: asks liveOD and the monitor server first).
FIRST_REPLY_TIMEOUT_S = 25.0
#: How long after "killing" the dashboard's port may still answer (its
#: watchdog ends it within 10 s).
KILL_CONFIRM_S = 15.0

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_DECLINED = 0, 1, 2, 3


@dataclass
class Dashboard:
    sid: str
    kind: str
    host: str
    pid: int
    ip: str
    port: int
    status: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.host} {self.kind}"


def owner_from_env() -> str:
    return "agent" if os.environ.get(OWNER_ENV, "").strip().lower() == "agent" else "person"


def default_by() -> str:
    try:
        user = getpass.getuser()
    except Exception:                                     # noqa: BLE001
        user = "?"
    return f"{user}@{socket.gethostname()}"


# -- network -------------------------------------------------------------------

def request_lines(ip: str, port: int, obj: dict, *, first_timeout: float = CONNECT_TIMEOUT_S,
                  idle_timeout: Optional[float] = None):
    """Send ``obj``; yield each reply line (a dict) until the server closes.
    ``idle_timeout``: the longest wait for a line after the first (None: no
    limit).  Raises OSError / socket.timeout / ValueError."""
    with socket.create_connection((ip, int(port)), timeout=CONNECT_TIMEOUT_S) as sock:
        sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))
        sock.settimeout(first_timeout)
        buf = b""
        first = True
        while True:
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    yield json.loads(line.decode("utf-8"))
                    if first:
                        first = False
                        sock.settimeout(idle_timeout)
            chunk = sock.recv(65536)
            if not chunk:
                if buf.strip():
                    yield json.loads(buf.decode("utf-8"))
                return
            buf += chunk


def request_one(ip: str, port: int, obj: dict, timeout: float = CONNECT_TIMEOUT_S) -> dict:
    for msg in request_lines(ip, port, obj, first_timeout=timeout):
        return msg
    raise ValueError("no reply")


def discover_dashboards(listen_s: float = 1.5) -> list[Dashboard]:
    """Dashboards beaconing now (heard within :data:`BEACON_MAX_AGE_S`)."""
    from beacon.discovery.client import discover_entries, ensure_listened  # noqa: PLC0415
    ensure_listened(listen_s)
    out = []
    for sid, entry in discover_entries(BEACON_PREFIX, max_age=BEACON_MAX_AGE_S).items():
        parsed = parse_beacon_id(sid)
        if parsed is None:
            continue
        kind, host, pid = parsed
        out.append(Dashboard(sid, kind, host, pid, entry.ip, int(entry.port)))
    out.sort(key=lambda d: (d.host.lower(), d.kind, d.pid))
    return out


def fetch_statuses(dashes: Iterable[Dashboard]) -> None:
    def one(d: Dashboard) -> None:
        try:
            d.status = request_one(d.ip, d.port, {"action": "status"})
        except Exception as exc:                          # noqa: BLE001
            d.status = {"ok": False, "error": f"no answer ({exc})"}
    threads = [threading.Thread(target=one, args=(d,), daemon=True) for d in dashes]
    for t in threads:
        t.start()
    for t in threads:
        t.join(CONNECT_TIMEOUT_S * 2 + 1)


def match(dashes: list[Dashboard], targets: list[str], kind: Optional[str]
          ) -> tuple[list[Dashboard], list[str]]:
    """The dashboards the targets name (all when no target), filtered by
    ``kind``; and the targets that named none."""
    pool = [d for d in dashes if kind is None or d.kind == kind]
    if not targets:
        return pool, []
    chosen: list[Dashboard] = []
    unmatched = []
    for t in targets:
        tl = t.strip().lower()
        hits = [d for d in pool if tl in (d.host.lower(), d.host.lower().split(".")[0],
                                          d.ip, d.sid.lower())]
        if not hits:
            unmatched.append(t)
        for d in hits:
            if d not in chosen:
                chosen.append(d)
    return chosen, unmatched


# -- output ----------------------------------------------------------------------

def _fmt_uptime(s) -> str:
    try:
        s = float(s)
    except (TypeError, ValueError):
        return "?"
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    if s < 172800:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} d"


def describe_servers(status: dict) -> str:
    servers = status.get("servers") or {}
    if not servers:
        return "no servers"
    up = sorted(sid for sid, st in servers.items() if st in ("RUNNING", "EXTERNAL", "STARTING"))
    bad = sorted(sid for sid, st in servers.items() if st in ("CRASHED", "FAILED"))
    text = f"{len(up)}/{len(servers)} servers up"
    if bad:
        text += f", crashed: {', '.join(bad)}"
    return text


def print_list(dashes: list[Dashboard], out) -> None:
    if not dashes:
        out.write("no dashboards found on the network\n")
        return
    rows = [("HOST", "KIND", "ADDRESS", "PID", "UP", "STATE")]
    for d in dashes:
        st = d.status or {}
        if not st.get("ok"):
            state = st.get("error") or "no answer"
            up = "?"
        else:
            state = ("CLOSING, " if st.get("closing") else "") + describe_servers(st)
            up = _fmt_uptime(st.get("uptime_s"))
        rows.append((d.host, d.kind, f"{d.ip}:{d.port}", str(d.pid), up, state))
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    for r in rows:
        out.write("  ".join(r[i].ljust(widths[i]) for i in range(5)) + "  " + r[5] + "\n")


def describe_event(msg: dict) -> str:
    ev = msg.get("event")
    if ev == "accepted":
        return "close accepted"
    if ev == "stopping":
        servers = msg.get("servers") or []
        return (f"stopping {len(servers)} server(s): {', '.join(servers)}" if servers
                else "no servers running")
    if ev == "com_wait":
        t = msg.get("timeout_s")
        return (f"waiting for COM server(s) to release their ports: "
                f"{', '.join(msg.get('servers') or [])}"
                + (f" (force-killed after {t:.0f} s)" if isinstance(t, (int, float)) else ""))
    if ev == "com_forced":
        return f"COM server(s) FORCE-KILLED (ports may stay held): {', '.join(msg.get('servers') or [])}"
    if ev == "servers_stopped":
        hk = msg.get("hard_killed") or []
        alive = msg.get("still_running") or []
        text = "servers stopped" + (f"; hard-killed: {', '.join(hk)}" if hk else " cleanly")
        if alive:
            text += f"; STILL RUNNING after the kills: {', '.join(alive)}"
        return text
    if ev == "done":
        res = msg.get("result")
        if res == "closed":
            return "CLOSED"
        if res == "closed_with_survivors":
            return ("CLOSED, BUT these servers were still running after it: "
                    f"{', '.join(msg.get('still_running') or [])} -- stop them at that PC")
        detail = msg.get("detail") or ""
        extra = msg.get("still_running")
        text = f"{str(res).upper()}: {detail}"
        if extra:
            text += f" (still running: {', '.join(extra)})"
        return text
    if ev == "killing":
        return f"killing process tree of pid {msg.get('pid')}"
    return json.dumps(msg)


# -- actions ---------------------------------------------------------------------

def run_on(d: Dashboard, req: dict, emit: Callable[[Dashboard, str], None],
           wait_s: float) -> bool:
    """Send a close/kill to one dashboard and relay its progress; True when it
    reported success."""
    killing = False
    try:
        for msg in request_lines(d.ip, d.port, req, first_timeout=FIRST_REPLY_TIMEOUT_S,
                                 idle_timeout=wait_s):
            if msg.get("ok") is False:
                emit(d, f"REFUSED: {msg.get('refused') or msg.get('error') or msg}")
                return False
            if msg.get("event") == "heartbeat":
                continue
            emit(d, describe_event(msg))
            if msg.get("event") == "done":
                return msg.get("result") == "closed"
            if msg.get("event") == "killing":
                killing = True
                break
    except socket.timeout:
        emit(d, f"no word from the dashboard for {wait_s:.0f} s (it sends a heartbeat every "
                "few seconds while closing): it may be frozen; check with kdash list")
        return False
    except (OSError, ValueError) as exc:
        if not killing:
            emit(d, f"FAILED: {exc} (if it closed a moment ago, kdash list will not show it)")
            return False
    if killing:
        if confirm_gone(d):
            emit(d, "KILLED: its control port no longer answers")
            return True
        emit(d, f"STILL ANSWERING {KILL_CONFIRM_S:.0f} s after the kill: it did not die")
        return False
    emit(d, "the connection ended before the dashboard reported its result "
            "(check with kdash list)")
    return False


def confirm_gone(d: Dashboard, seconds: float = None, poll_s: float = 0.5) -> bool:
    """True once the dashboard's control port stops answering ``status``."""
    seconds = KILL_CONFIRM_S if seconds is None else seconds
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            request_one(d.ip, d.port, {"action": "status"}, timeout=2.0)
        except (OSError, ValueError):
            return True
        time.sleep(poll_s)
    return False


def run_all(dashes: list[Dashboard], req: dict, out, wait_s: float) -> bool:
    lock = threading.Lock()
    results: dict[str, bool] = {}

    def emit(d: Dashboard, text: str) -> None:
        with lock:
            out.write(f"[{d.name}] {text}\n")
            out.flush()

    def one(d: Dashboard) -> None:
        results[d.sid] = run_on(d, req, emit, wait_s)

    threads = [threading.Thread(target=one, args=(d,), daemon=True) for d in dashes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return all(results.get(d.sid) for d in dashes)


def _confirm(prompt: str, inp, out) -> bool:
    out.write(prompt)
    out.flush()
    try:
        answer = inp.readline()
    except Exception:                                     # noqa: BLE001
        return False
    return answer.strip().lower() in ("y", "yes")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kdash", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list", help="list the dashboards on the network")
    pl.add_argument("--kind", choices=("server", "client"))
    pl.add_argument("--json", action="store_true", help="machine-readable output")
    for name, helptext in (("close", "close dashboards (graceful, as their X button)"),
                           ("kill", "hard-kill dashboards and their servers (frozen window)")):
        pc = sub.add_parser(name, help=helptext)
        pc.add_argument("targets", nargs="*", help="host names, IPs or beacon ids")
        pc.add_argument("--kind", choices=("server", "client"))
        pc.add_argument("--force", action="store_true",
                        help="even while a run holds the machine (server dashboards)")
        pc.add_argument("--yes", "-y", action="store_true", help="do not ask to confirm")
        pc.add_argument("--wait", type=float, default=30.0,
                        help="longest silence from a dashboard before giving up on it (s); "
                             "a closing dashboard sends a heartbeat every 5 s")
        if name == "close":
            pc.add_argument("--all", action="store_true", help="every dashboard on the network")
            pc.add_argument("--com-timeout", type=float, default=DEFAULT_COM_TIMEOUT_S,
                            help="seconds COM servers get before they are force-killed "
                                 f"({COM_TIMEOUT_RANGE_S[0]:.0f}-{COM_TIMEOUT_RANGE_S[1]:.0f})")
    return p


def main(argv: Optional[list[str]] = None, *, out=None, inp=None,
         discover: Callable[[], list[Dashboard]] = discover_dashboards) -> int:
    out = out or sys.stdout
    inp = inp or sys.stdin
    args = _parser().parse_args(argv)

    if args.cmd in ("close", "kill"):
        if owner_from_env() == "agent":
            out.write(f"kdash: an agent may not {args.cmd} a dashboard (WAXX_OWNER=agent); "
                      "ask a person\n")
            return EXIT_FAILED
        if args.cmd == "close" and not args.targets and not args.all:
            out.write("kdash close: name the host(s) to close, or --all for every dashboard\n")
            return EXIT_USAGE
        if args.cmd == "close" and args.targets and args.all:
            out.write("kdash close: give host(s) or --all, not both\n")
            return EXIT_USAGE
        if args.cmd == "kill" and not args.targets:
            out.write("kdash kill: name the host(s); there is no kill --all\n")
            return EXIT_USAGE

    dashes = discover()
    if args.cmd == "list":
        dashes = [d for d in dashes if args.kind is None or d.kind == args.kind]
        fetch_statuses(dashes)
        if args.json:
            out.write(json.dumps([{"id": d.sid, "kind": d.kind, "host": d.host, "pid": d.pid,
                                   "ip": d.ip, "port": d.port, "status": d.status}
                                  for d in dashes], indent=2) + "\n")
        else:
            print_list(dashes, out)
        return EXIT_OK

    chosen, unmatched = match(dashes, args.targets, args.kind)
    if unmatched:
        hosts = sorted({d.host for d in dashes})
        out.write(f"kdash: no dashboard matches {', '.join(unmatched)}"
                  + (f" (dashboards are on: {', '.join(hosts)})" if hosts
                     else " (no dashboards found on the network)") + "; nothing done\n")
        return EXIT_USAGE
    if not chosen:
        out.write("kdash: no dashboard matched; nothing done\n")
        return EXIT_USAGE

    fetch_statuses(chosen)
    verb = "Close" if args.cmd == "close" else "HARD-KILL (no graceful stop)"
    out.write(f"{verb}:\n")
    print_list(chosen, out)
    if any(d.kind == "server" for d in chosen):
        out.write("Server dashboards stop every server they run (the interlock and monitor "
                  "servers included, where they run them).\n")
    if not args.yes:
        interactive = bool(getattr(inp, "isatty", lambda: False)())
        if not interactive:
            out.write("kdash: not a terminal; pass --yes to go ahead\n")
            return EXIT_DECLINED
        if not _confirm(f"{verb} {len(chosen)} dashboard(s)? [y/N] ", inp, out):
            out.write("nothing done\n")
            return EXIT_DECLINED

    req = {"action": args.cmd, "by": default_by(), "owner": owner_from_env(),
           "force": bool(args.force)}
    if args.cmd == "close":
        req["com_timeout_s"] = float(args.com_timeout)
    # A closing dashboard sends a heartbeat every few seconds: --wait is the
    # longest silence, not the longest close.
    ok = run_all(chosen, req, out, wait_s=max(5.0, float(args.wait)))
    return EXIT_OK if ok else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
