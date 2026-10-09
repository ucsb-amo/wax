"""kq -- the run queue's command line (:mod:`waxx.util.device_state.run_queue`).

The run queue lives in the monitor server, which launches every job itself
(``artiq_run``, detached, its output in a log file on the server's machine).
``kq`` submits jobs, follows their output and asks the queue about its state;
it never runs, stops or kills an experiment itself, and it never falls back to
a direct run when no queue answers.

Commands::

    kq run <file.py> [ARG...] [options] [-- argv...]   submit, then follow its output
    kq submit <file.py> [ARG...] [options] [-- argv...]   submit only (= run --detach)
    kq list [--all] [--state S ...] [--json]  the jobs and the queue's state
    kq show <id> [--json]                     one job, why it waits, its last lines
    kq tail <id> [-f]                         its log so far; -f follows to the end
    kq cancel <id> [--yes]                    cancel (a running job: liveOD's Abort)
    kq pause [--all] [--reason R]             stop launching agent (or all) jobs
    kq resume [--all]                         lift that pause
    kq hold [reason]                          a person's hold: agents' jobs wait
    kq release                                lift the person's hold
    kq status [--json]                        the queue's state (not occupancy)

Words after the file (``key=value``, as artiq_run takes them) and everything
after ``--`` are passed to the experiment: ``kq run x.py n=3 -- -c MyExpt``.

``kq run`` prints ``[kq] job <id> queued (position k; <why it waits>)``, then
the job's output exactly as the experiment writes it ("Run ID: N" included),
and exits with the experiment's result (see the exit codes).  Ctrl-C while
the job is queued cancels it (and any later jobs of the same submission still
queued).  Ctrl-C while it runs asks once on a terminal "abort the run? it
discards its data file [y/N]": yes asks the queue to cancel it -- the queue
then sends liveOD's Abort (the run stops at its next shot and liveOD discards
its file) -- and kq prints "abort requested; waiting for the run to end" and
follows the job to its end; anything else, or no terminal, leaves the run going and prints how
to follow or cancel it.  A second Ctrl-C while following an abort leaves too.

Owner: kq acts for a person unless the environment has ``WAXX_OWNER=agent``
(the agents' skill sets it; ``--agent`` is a convenience for the same).  Every
request that changes something carries the owner: an agent cannot cancel a
person's job, release a person's hold or resume a person's pause.  Without
``--priority`` the queue gives the owner's default.  ``by`` is ``user@host``.

Output: kq's own lines are ASCII; the experiment's lines are written as they
are, with any character the terminal cannot show replaced.  Errors go to
stderr.

Exit codes:

* 0 -- the job saved (run, tail -f); the request was done; status: queue free
* 1 -- the job failed without an exit code of its own (or with 0)
* N -- the job failed: the experiment's own exit code N
* 2 -- bad command line
* 3 -- the job was cancelled or skipped
* 4 -- no run queue is beaconing (no monitor server, or one without a queue)
* 5 -- status: queue busy -- a queue job is in the slot (launching, running,
  ending) or a person's hold is on.  Queue state only: kq status never asks
  liveOD, the run fence or the run loops, so a direct artiq_run run or a run
  loop does not show; for machine occupancy use occupancy.py (agents) or the
  dashboard
* 6 -- the queue refused the request, or the monitor server did not answer
* 130 -- interrupted (Ctrl-C): the job was left as it was, or cancelled while
  it was queued

An experiment's own exit code is passed through as is, so 3-6 can in principle
also be one; ``kq show <id>`` tells.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time

from waxx.util.device_state.run_queue_client import (
    ENDED, IN_SLOT, STATES, NoRunQueue, RunQueueClient, RunQueueError, owner_from_env,
    safe_write)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_NOT_RUN = 3
EXIT_NO_QUEUE = 4
EXIT_BUSY = 5
EXIT_REFUSED = 6
EXIT_INTERRUPTED = 130

NO_QUEUE_TEXT = ("no run queue is beaconing; use artiq_run --device-db %db% <file> for a "
                 "direct run, or start the Server Dashboard")

#: Lines of ended jobs shown by a plain ``kq list``.
LIST_ENDED = 10

EPILOG = """\
job states: queued -> launching -> running -> ending -> saved | failed;
  or cancelled / skipped.
exit codes: 0 saved / done / free, 1 failed (no exit code of its own),
  N the experiment's own exit code, 2 bad command line, 3 cancelled or skipped,
  4 no run queue beaconing, 5 status: queue busy (a job in the slot or a
  person's hold; queue state only, not machine occupancy: use occupancy.py
  (agents) or the dashboard), 6 refused or no answer,
  130 interrupted (Ctrl-C).
direct runs, outside the queue: artiq_run --device-db %db% <file.py>
"""

_ASCII = str.maketrans({
    "—": "--", "–": "-", "‒": "-", "‐": "-", "−": "-",
    "→": "->", "←": "<-", "⇒": "=>", "↔": "<->",
    "±": "+/-", "µ": "u", "μ": "u", "×": "x", "°": " deg",
    "≤": "<=", "≥": ">=", "≠": "!=", "≈": "~", "…": "...",
    "‘": "'", "’": "'", "“": '"', "”": '"', " ": " ",
    "─": "-", "│": "|", "•": "*", "·": "*", "✓": "ok",
    "✗": "x", "é": "e"})


def ascii_text(text) -> str:
    """``text`` with common symbols spelled in ASCII and anything else as '?'."""
    return str(text).translate(_ASCII).encode("ascii", "replace").decode("ascii")


class _Ctx:
    def __init__(self, client_factory, out, err, ask, isatty):
        self._client_factory = client_factory
        self._client = None
        self.out = out
        self.err = err
        self._ask = ask
        self._isatty = isatty

    @property
    def client(self) -> RunQueueClient:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def say(self, text: str = "") -> None:
        safe_write(self.out, ascii_text(text) + "\n")
        _flush(self.out)

    def warn(self, text: str) -> None:
        safe_write(self.err, ascii_text(text) + "\n")
        _flush(self.err)

    def tty(self) -> bool:
        try:
            return bool(self._isatty())
        except Exception:                             # noqa: BLE001
            return False

    def ask(self, prompt: str) -> bool:
        """y/N on the terminal; anything but y (Ctrl-C and end of input too) is no."""
        try:
            answer = self._ask(ascii_text(prompt))
        except (KeyboardInterrupt, EOFError):
            safe_write(self.out, "\n")
            return False
        return str(answer or "").strip().lower() in ("y", "yes")


def _flush(stream) -> None:
    try:
        stream.flush()
    except Exception:                                 # noqa: BLE001
        pass


def _default_ask(prompt: str) -> str:
    return input(prompt)


def _default_isatty() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:                                 # noqa: BLE001
        return False


# -- formatting -------------------------------------------------------------------------

def _when(t) -> str:
    if t in (None, ""):
        return "-"
    try:
        t = float(t)
        if time.strftime("%Y-%m-%d", time.localtime(t)) == time.strftime("%Y-%m-%d"):
            return time.strftime("%H:%M:%S", time.localtime(t))
        return time.strftime("%m-%d %H:%M", time.localtime(t))
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def _hold_text(hold: dict | None) -> str:
    """"person hold since 14:03:11 by jp@kong (owner person, source request):
    reason" -- owner and source as the server gives them ("unreadable_file":
    the server could not read its hold file and holds until a person releases)."""
    if not hold or not hold.get("active"):
        return ""
    extra = [f"owner {hold['owner']}" if hold.get("owner") else "",
             f"source {hold['source']}" if hold.get("source") else ""]
    extra = ", ".join(e for e in extra if e)
    return (f"person hold since {_when(hold.get('since'))} by {hold.get('by') or '?'}"
            + (f" ({extra})" if extra else "")
            + (f": {hold.get('reason')}" if hold.get("reason") else ""))


def _alarm_text(alarm: dict | None) -> str:
    if not alarm:
        return ""
    waited = alarm.get("waited_s")
    mins = f"{float(waited) / 60.0:.0f} min" if isinstance(waited, (int, float)) else "?"
    return (f"ALARM: job {alarm.get('job')} ready to start for {mins} and nothing launched"
            + (f" -- {alarm.get('why')}" if alarm.get("why") else ""))


def queue_line(info: dict, hold: dict | None = None) -> str:
    """One line for the queue's state (``status_json["run_queue"]``)."""
    parts = [f"queue: {info.get('state', '?')} -- {info.get('text', '')}"]
    nxt = info.get("next") or []
    if nxt:
        parts.append("next: " + ", ".join(str(i) for i in nxt[:10])
                     + (" ..." if len(nxt) > 10 else ""))
    paused = info.get("paused") or {}
    for scope in ("all", "agent"):
        p = paused.get(scope)
        if p:
            parts.append(f"{scope} jobs paused by {p.get('by') or '?'}"
                         + (f" ({p.get('reason')})" if p.get("reason") else ""))
    h = _hold_text(hold if hold is not None else info.get("person_hold"))
    if h:
        parts.append(h)
    a = _alarm_text(info.get("alarm"))
    if a:
        parts.append(a)
    return " | ".join(parts)


def _job_word(job: dict) -> str:
    return f"job {job.get('id')} ({job.get('label')})"


def exit_code_for(job: dict) -> int:
    """The exit code ``kq run`` / ``kq tail -f`` give for an ended job."""
    state = job.get("state")
    if state == "saved":
        return EXIT_OK
    if state in ("cancelled", "skipped"):
        return EXIT_NOT_RUN
    code = job.get("exit_code")
    if isinstance(code, int) and code > 0:
        return code
    return EXIT_FAILED


def _report_end(ctx: _Ctx, job: dict) -> None:
    state = job.get("state")
    run = f"run {job['run_id']}" if job.get("run_id") is not None else "no run id"
    if state == "saved":
        ctx.say(f"[kq] {_job_word(job)} saved ({run})"
                + (f": {job.get('reason')}" if job.get("reason") else ""))
        return
    code = job.get("exit_code")
    extra = f", exit code {code}" if code is not None else ""
    ctx.warn(f"[kq] {_job_word(job)} {str(state).upper()} ({run}{extra})"
             + (f": {job.get('reason')}" if job.get("reason") else ""))


# -- --at -------------------------------------------------------------------------------

_HHMM = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


#: A bare number below this is not taken as epoch seconds (1e9 s is 2001-09-09):
#: ``--at 1430`` is a typo for 14:30, not 1970.
MIN_EPOCH = 1e9


def parse_at(text: str, now: float | None = None) -> float:
    """``HH:MM[:SS]`` (local; the next such time, so a time already past
    today means tomorrow -- by the calendar, so a DST change is no hour off)
    or epoch seconds (at least :data:`MIN_EPOCH`) -> epoch seconds."""
    now = time.time() if now is None else float(now)
    m = _HHMM.match(text.strip())
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        if not (0 <= h < 24 and 0 <= mi < 60 and 0 <= s < 60):
            raise ValueError(f"not a time of day: {text}")
        day = datetime.datetime.fromtimestamp(now).date()
        at = datetime.time(h, mi, s)
        due = datetime.datetime.combine(day, at).timestamp()
        if due <= now:
            due = datetime.datetime.combine(day + datetime.timedelta(days=1), at).timestamp()
        return due
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"--at takes HH:MM or epoch seconds, not {text!r}") from None
    if value < MIN_EPOCH:
        raise ValueError(f"--at {text}: a bare number must be epoch seconds (at least "
                         f"{MIN_EPOCH:.0f}); for a time of day write HH:MM")
    return value


# -- the parser -------------------------------------------------------------------------

def build_parser(out=None, err=None) -> argparse.ArgumentParser:
    """kq's parser; its help goes to ``out`` and its errors to ``err``
    (default: stdout / stderr), in ASCII."""

    class _Parser(argparse.ArgumentParser):
        def _print_message(self, message, file=None):
            if message:
                stream = ((err or sys.stderr) if file is sys.stderr
                          else (out or sys.stdout))
                safe_write(stream, ascii_text(message))

    p = _Parser(prog="kq", description="The run queue (in the monitor server).",
                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", metavar="command", parser_class=_Parser)
    sub.required = True

    def run_options(sp):
        sp.add_argument("file", help="the experiment file (.py)")
        sp.add_argument("args", nargs="*", metavar="ARG",
                        help="the experiment's arguments (key=value), as after the file "
                             "in artiq_run; words starting with - go after --")
        sp.add_argument("--label", help="a short name (default: the file's stem)")
        sp.add_argument("--priority", type=int, default=None,
                        help="higher goes first (default: the queue's, by owner)")
        sp.add_argument("--at", dest="at", metavar="HH:MM|EPOCH",
                        help="not before this time (local HH:MM: the next such time)")
        sp.add_argument("--after", type=int, nargs="+", action="extend", default=[],
                        metavar="ID", help="only after these jobs have saved")
        sp.add_argument("--repeat", type=int, default=1, help="N runs, one job each (a chain)")
        sp.add_argument("--chain", help="chain name (jobs of a chain stop together)")
        sp.add_argument("--no-stop-on-failure", action="store_true",
                        help="a failed job does not cancel the rest of its chain")
        sp.add_argument("--no-write-back", action="store_true",
                        help="veto the experiment's calibration write-back")
        sp.add_argument("--allow-drift", action="store_true",
                        help="run even if the file changed after submit")
        sp.add_argument("--cwd", help="working folder on the server's machine "
                                      "(default: the file's folder)")
        sp.add_argument("--agent", action="store_true",
                        help="an agent's job (also: environment WAXX_OWNER=agent)")

    sp = sub.add_parser("run", help="submit and follow the job's output", epilog=EPILOG,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    run_options(sp)
    sp.add_argument("--detach", action="store_true", help="submit only; do not follow")
    sp = sub.add_parser("submit", help="submit only (= run --detach)")
    run_options(sp)

    sp = sub.add_parser("list", help="the jobs and the queue's state")
    sp.add_argument("--all", action="store_true", help="every job the server keeps")
    sp.add_argument("--state", nargs="+", action="extend", choices=STATES, metavar="STATE",
                    help="only jobs in these states")
    sp.add_argument("--limit", type=int, default=200)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("show", help="one job, why it waits, its last lines")
    sp.add_argument("id", type=int)
    sp.add_argument("--token")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("tail", help="a job's log; -f follows it to the end")
    sp.add_argument("id", type=int)
    sp.add_argument("-f", "--follow", action="store_true")
    sp.add_argument("--token")

    sp = sub.add_parser("cancel", help="cancel a job (a running one: liveOD's Abort)")
    sp.add_argument("id", type=int)
    sp.add_argument("--yes", action="store_true",
                    help="abort a running job without asking (its data file is discarded)")
    sp.add_argument("--token")
    sp.add_argument("--agent", action="store_true", help="cancel as an agent")

    sp = sub.add_parser("pause", help="stop launching agent jobs (--all: every job)")
    sp.add_argument("--all", action="store_true")
    sp.add_argument("--reason", default="")
    sp.add_argument("--agent", action="store_true", help="as an agent")
    sp = sub.add_parser("resume", help="lift a pause")
    sp.add_argument("--all", action="store_true")
    sp.add_argument("--agent", action="store_true", help="as an agent")

    sp = sub.add_parser("hold", help="a person's hold: agents' jobs wait")
    sp.add_argument("reason", nargs="*")
    sp.add_argument("--agent", action="store_true", help="as an agent")
    sp = sub.add_parser("release", help="lift the person's hold")
    sp.add_argument("--agent", action="store_true", help="as an agent")

    sp = sub.add_parser("status", help="the queue's state, not machine occupancy (exit 0 "
                                       "queue free / 5 a job in the slot or a hold)")
    sp.add_argument("--json", action="store_true")
    return p


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """``kq run file.py --label x -- a b`` -> (kq's words, the experiment's)."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _owner(args) -> str:
    """Who kq acts for: ``WAXX_OWNER`` ("agent" -> agent, else person);
    ``--agent`` is a convenience for the same."""
    return "agent" if getattr(args, "agent", False) else owner_from_env()


# -- commands ---------------------------------------------------------------------------

def _cmd_run(ctx: _Ctx, args, expt_argv: list[str]) -> int:
    due = None
    if args.at:
        try:
            due = parse_at(args.at)
        except ValueError as exc:
            ctx.warn(f"kq: {exc}")
            return EXIT_USAGE
    owner = _owner(args)
    try:
        reply = ctx.client.submit(
            args.file, argv=list(args.args) + expt_argv, cwd=args.cwd, label=args.label,
            owner=owner, priority=args.priority, due=due, after=args.after,
            repeat=args.repeat, chain=args.chain,
            stop_on_failure=False if args.no_stop_on_failure else None,
            write_back=False if args.no_write_back else None, allow_drift=args.allow_drift)
    except KeyboardInterrupt:
        safe_write(ctx.out, "\n")
        ctx.warn("[kq] interrupted while submitting: the job may be queued -- check kq list")
        return EXIT_INTERRUPTED
    jobs = reply.get("jobs") or []
    ids = [int(i) for i in reply.get("ids") or [j["id"] for j in jobs]]
    tokens = {int(j["id"]): j.get("token") for j in jobs}
    if not ids:
        ctx.warn("kq: the queue accepted the request but returned no job")
        return EXIT_REFUSED
    first = jobs[0] if jobs else {"id": ids[0]}
    many = (f"jobs {ids[0]}-{ids[-1]} (chain {first.get('chain')})" if len(ids) > 1
            else f"job {ids[0]}")
    due_text = f", due {_when(due)}" if due else ""
    if args.cmd == "submit" or getattr(args, "detach", False):
        ctx.say(f"[kq] {many} queued: {first.get('label')}, owner {owner}{due_text}")
        ctx.say(f"[kq] follow: kq tail {ids[0]} -f    cancel: kq cancel {ids[0]}")
        return EXIT_OK
    # from here on a Ctrl-C anywhere goes to _interrupted: the job ids are
    # known, a queued job is cancelled (queued_only) and its id printed
    n, cursor = 0, {"offset": 0}
    try:
        ctx.say(f"[kq] {many} queued ({_position(ctx, ids[0], tokens.get(ids[0]))})"
                f"{due_text}")
        final = EXIT_OK
        for n, jid in enumerate(ids):
            cursor = {"offset": 0}
            if len(ids) > 1:
                ctx.say(f"[kq] job {jid} ({n + 1} of {len(ids)})")
            state = {"waiting": None}

            def on_wait(described, _state=state):
                why = described.get("waiting") or ""
                if why and why != _state["waiting"]:
                    if _state["waiting"] is not None:
                        ctx.say(f"[kq] waiting: {why}")
                    _state["waiting"] = why

            job = ctx.client.follow(jid, tokens.get(jid), ctx.out, cursor=cursor,
                                    on_wait=on_wait, on_lost=lambda m: ctx.warn(f"[kq] {m}"),
                                    on_abort=lambda j: ctx.say(_abort_text(j)))
            _report_end(ctx, job)
            code = exit_code_for(job)
            if code != EXIT_OK and final == EXIT_OK:
                final = code
        return final
    except KeyboardInterrupt:
        return _interrupted(ctx, ids[n:], tokens, owner, cursor)


def _position(ctx: _Ctx, jid: int, token) -> str:
    try:
        listed = ctx.client.list(states=("queued",) + IN_SLOT)
        described = ctx.client.describe(jid, token)
    except RunQueueError:
        return "position unknown"
    nxt = listed.get("next") or []
    why = described.get("waiting") or ""
    if jid in nxt:
        return f"position {nxt.index(jid) + 1}; {why or 'launching'}"
    return f"not yet eligible; {why}" if why else "state " + str(described["job"].get("state"))


def _cancel_queued(ctx: _Ctx, ids, tokens, owner) -> list[int]:
    """Cancel those of ``ids`` still queued (never a running one)."""
    done = []
    for jid in ids:
        try:
            ctx.client.cancel(jid, token=tokens.get(jid), owner=owner, queued_only=True)
            done.append(jid)
        except RunQueueError:
            pass
    return done


def _abort_text(job: dict) -> str:
    """The line for a job whose cancel has been asked while it runs: the
    queue's tick sends liveOD's Abort; the run ends at its next shot."""
    c = job.get("cancel") or {}
    run = f"run {job['run_id']}" if job.get("run_id") is not None else "no run id yet"
    return (f"[kq] abort requested for {_job_word(job)} ({run})"
            + (f" by {c.get('by')}" if c.get("by") else "")
            + "; waiting for the run to end")


def _leave_text(jid: int) -> str:
    return f"follow: kq tail {jid} -f    cancel: kq cancel {jid}"


def _interrupted(ctx: _Ctx, ids: list[int], tokens: dict, owner: str, cursor: dict) -> int:
    """Ctrl-C while following ``ids[0]`` (the rest: later jobs of the same
    submission)."""
    jid, rest = ids[0], ids[1:]
    safe_write(ctx.out, "\n")
    try:
        job = ctx.client.describe(jid, tokens.get(jid))["job"]
        if job["state"] == "queued":
            try:
                ctx.client.cancel(jid, token=tokens.get(jid), owner=owner, queued_only=True)
                cancelled = [jid] + _cancel_queued(ctx, rest, tokens, owner)
                ctx.say("[kq] interrupted: cancelled job"
                        + ("s " if len(cancelled) > 1 else " ")
                        + ", ".join(map(str, cancelled)) + " (not started)")
                return EXIT_INTERRUPTED
            except RunQueueError:
                job = ctx.client.describe(jid, tokens.get(jid))["job"]  # it just launched
        if job["state"] in ENDED:
            _report_end(ctx, job)
            cancelled = _cancel_queued(ctx, rest, tokens, owner)
            if cancelled:
                ctx.say("[kq] interrupted: cancelled queued job"
                        + ("s " if len(cancelled) > 1 else " ") + ", ".join(map(str, cancelled)))
            return EXIT_INTERRUPTED
        run = f"run {job['run_id']}" if job.get("run_id") is not None else "no run id yet"
        if job["state"] in ("launching", "running") and ctx.tty():
            if ctx.ask(f"[kq] {_job_word(job)} is running ({run}). abort the run? it "
                       "discards its data file [y/N] "):
                reply = ctx.client.cancel(jid, token=tokens.get(jid), owner=owner)
                ctx.say(_abort_text(reply.get("job") or job) + " (Ctrl-C again leaves it)")
                cursor["abort_reported"] = True
                cancelled = _cancel_queued(ctx, rest, tokens, owner)
                if cancelled:
                    ctx.say("[kq] cancelled queued job"
                            + ("s " if len(cancelled) > 1 else " ")
                            + ", ".join(map(str, cancelled)))
                try:
                    final = ctx.client.follow(jid, tokens.get(jid), ctx.out, cursor=cursor,
                                              on_lost=lambda m: ctx.warn(f"[kq] {m}"),
                                              on_abort=lambda j: ctx.say(_abort_text(j)))
                except KeyboardInterrupt:
                    safe_write(ctx.out, "\n")
                    ctx.say(f"[kq] left {_job_word(job)} ending on its own. {_leave_text(jid)}")
                    return EXIT_INTERRUPTED
                _report_end(ctx, final)
                return exit_code_for(final)
        ctx.say(f"[kq] interrupted: {_job_word(job)} is left {job['state']} ({run})"
                + (f"; jobs {', '.join(map(str, rest))} stay queued" if rest else "")
                + f". {_leave_text(jid)}")
        return EXIT_INTERRUPTED
    except KeyboardInterrupt:
        safe_write(ctx.out, "\n")
        ctx.say(f"[kq] interrupted: job {jid} is left as it is. {_leave_text(jid)}")
        return EXIT_INTERRUPTED
    except RunQueueError as exc:
        # still an interrupt: the person asked to stop; what the queue said goes
        # with it, and the job is whatever the queue made of it
        ctx.warn(f"kq: {exc}")
        ctx.say(f"[kq] interrupted: job {jid} may be left as it was. {_leave_text(jid)}")
        return EXIT_INTERRUPTED


def _list_rows(jobs: list[dict]) -> list[list[str]]:
    rows = [["id", "state", "owner", "label", "run_id", "prio", "due", "chain", "reason"]]
    for j in jobs:
        reason = str(j.get("reason") or "")
        if j.get("state") in IN_SLOT and j.get("cancel"):
            c = j["cancel"]
            reason = (f"abort requested by {c.get('by') or '?'}"
                      + (f": {c.get('abort_note')}" if c.get("abort_note") else ""))
        if len(reason) > 70:
            reason = reason[:67] + "..."
        rows.append([str(j.get("id")), str(j.get("state")), str(j.get("owner")),
                     str(j.get("label")), "-" if j.get("run_id") is None else str(j["run_id"]),
                     str(j.get("priority", 0)), _when(j.get("due")),
                     str(j.get("chain") or "-"), reason or "-"])
    return rows


def _cmd_list(ctx: _Ctx, args) -> int:
    reply = ctx.client.list(states=args.state, limit=args.limit)
    jobs = reply.get("jobs") or []
    if not args.all and not args.state:
        ended = [j for j in jobs if j.get("state") in ENDED]
        keep = {j["id"] for j in ended[-LIST_ENDED:]}
        jobs = [j for j in jobs if j.get("state") not in ENDED or j["id"] in keep]
    if args.json:
        ctx.say(json.dumps({"jobs": jobs, "next": reply.get("next"),
                            "run_queue": reply.get("run_queue")}, indent=1, default=str))
        return EXIT_OK
    rows = _list_rows(jobs)
    widths = [max(len(ascii_text(r[i])) for r in rows) for i in range(len(rows[0]) - 1)]
    if len(rows) > 1:
        for r in rows:
            ctx.say("  ".join(c.ljust(w) for c, w in zip(r[:-1], widths)) + "  " + r[-1])
    else:
        ctx.say("(no jobs)")
    ctx.say(queue_line(reply.get("run_queue") or {}))
    return EXIT_OK


def _cmd_show(ctx: _Ctx, args) -> int:
    reply = ctx.client.describe(args.id, args.token)
    if args.json:
        ctx.say(json.dumps(reply, indent=1, default=str))
        return EXIT_OK
    j = reply["job"]
    ctx.say(f"{_job_word(j)}: {j.get('state')}"
            + (f" -- {j.get('reason')}" if j.get("reason") else ""))
    if reply.get("waiting"):
        ctx.say(f"  waiting: {reply['waiting']}")
    for key in ("path", "argv", "cwd", "owner", "priority", "chain", "after", "run_id",
                "exit_code", "log_path", "submitted_by", "token"):
        val = j.get(key)
        if val not in (None, "", []):
            ctx.say(f"  {key}: {val}")
    if j.get("repeat_of", 1) > 1:
        ctx.say(f"  repeat: {j.get('repeat_index')} of {j.get('repeat_of')}")
    for key in ("due", "submitted_at", "launched_at", "ended_at"):
        if j.get(key):
            ctx.say(f"  {key}: {_when(j[key])}")
    if j.get("write_back") is False:
        ctx.say("  write-back: vetoed")
    if j.get("allow_drift"):
        ctx.say("  drift allowed")
    if j.get("outcome"):
        o = j["outcome"]
        ctx.say(f"  outcome: liveOD {o.get('outcome')}"
                + (f" ({o.get('detail')})" if o.get("detail") else "")
                + (f"; {o.get('why')}" if o.get("why") else ""))
    if j.get("cancel"):
        c = j["cancel"]
        ctx.say(f"  cancel: by {c.get('by')} at {_when(c.get('at'))}"
                + (f"; {c.get('abort_note')}" if c.get("abort_note") else ""))
    tail = reply.get("tail") or []
    if tail:
        ctx.say("  last lines:")
        for line in tail:
            ctx.say(f"    | {line}")
    return EXIT_OK


def _cmd_tail(ctx: _Ctx, args) -> int:
    if args.follow:
        cursor = {"offset": 0}
        try:
            job = ctx.client.follow(args.id, args.token, ctx.out, cursor=cursor,
                                    on_lost=lambda m: ctx.warn(f"[kq] {m}"),
                                    on_abort=lambda j: ctx.say(_abort_text(j)))
        except KeyboardInterrupt:
            safe_write(ctx.out, "\n")
            ctx.say(f"[kq] stopped following job {args.id}; it is left as it is")
            return EXIT_INTERRUPTED
        _report_end(ctx, job)
        return exit_code_for(job)
    offset = 0
    while True:
        reply = ctx.client.tail(args.id, args.token, offset)
        for line in reply.get("lines") or []:
            safe_write(ctx.out, line + "\n")
        offset = int(reply.get("offset") or offset)
        if reply.get("done") or not reply.get("lines"):
            break
    _flush(ctx.out)
    if reply.get("state") == "queued":
        ctx.say(f"[kq] job {args.id} is queued: no output yet")
    return EXIT_OK


def _cmd_cancel(ctx: _Ctx, args) -> int:
    owner = _owner(args)
    job = ctx.client.describe(args.id, args.token)["job"]
    if job["state"] == "queued":
        try:
            ctx.client.cancel(args.id, token=args.token, owner=owner, queued_only=True)
            ctx.say(f"[kq] {_job_word(job)} cancelled (it had not started)")
            return EXIT_OK
        except RunQueueError:
            job = ctx.client.describe(args.id, args.token)["job"]
            if job["state"] == "queued":
                raise
            ctx.say(f"[kq] {_job_word(job)} has just started")
    if job["state"] not in ("launching", "running"):
        ctx.warn(f"kq: {_job_word(job)} is {job['state']}: nothing to cancel")
        return EXIT_REFUSED
    run = f"run {job['run_id']}" if job.get("run_id") is not None else "no run id yet"
    if not args.yes:
        if not ctx.tty():
            ctx.warn(f"kq: {_job_word(job)} is running ({run}): cancelling it sends liveOD's "
                     "Abort, which discards its data file. Pass --yes to do that.")
            return EXIT_REFUSED
        if not ctx.ask(f"[kq] {_job_word(job)} is running ({run}). abort the run? it "
                       "discards its data file [y/N] "):
            ctx.say("[kq] nothing cancelled")
            return EXIT_OK
    reply = ctx.client.cancel(args.id, token=args.token, owner=owner)
    ctx.say(_abort_text(reply.get("job") or job)
            + f" (kq show {args.id}, kq tail {args.id} -f)")
    return EXIT_OK


def _cmd_pause(ctx: _Ctx, args) -> int:
    scope = "all" if args.all else "agent"
    reply = ctx.client.pause(scope, args.reason, owner=_owner(args))
    ctx.say(f"[kq] {scope} jobs paused (a running job is not touched)")
    ctx.say(queue_line(reply.get("run_queue") or {}))
    return EXIT_OK


def _cmd_resume(ctx: _Ctx, args) -> int:
    scope = "all" if args.all else "agent"
    reply = ctx.client.resume(scope, owner=_owner(args))
    ctx.say(f"[kq] {scope} jobs resumed")
    ctx.say(queue_line(reply.get("run_queue") or {}))
    return EXIT_OK


def _cmd_hold(ctx: _Ctx, args) -> int:
    reply = ctx.client.hold(" ".join(args.reason), owner=_owner(args))
    h = _hold_text(reply.get("person_hold"))
    if reply.get("already"):
        ctx.say(f"[kq] a hold was already on: {h}")
    else:
        ctx.say(f"[kq] {h}; agents' jobs wait until kq release")
    return EXIT_OK


def _cmd_release(ctx: _Ctx, args) -> int:
    reply = ctx.client.release(owner=_owner(args))
    ctx.say(f"[kq] released ({_hold_text(reply.get('released')) or 'the hold'})")
    return EXIT_OK


#: ``kq status`` reports the queue's state only, never the machine's.
OCCUPANCY_NOTE = ("queue state only, not machine occupancy (liveOD, direct artiq_run "
                  "runs, the run loops): for occupancy use occupancy.py (agents) or the "
                  "dashboard")


def queue_busy(status: dict) -> bool:
    """``kq status``'s exit 5: a queue job is in the slot (launching,
    running, ending) or a person's hold is on.  Queue state only: jobs
    waiting (due later, after others, paused) do not count, and nothing here
    asks liveOD, the run fence or the run loops."""
    rq = status.get("run_queue") or {}
    hold = status.get("person_hold") or rq.get("person_hold") or {}
    return rq.get("current") is not None or bool(hold.get("active"))


def _cmd_status(ctx: _Ctx, args) -> int:
    status = ctx.client.status()
    busy = queue_busy(status)
    rq = status.get("run_queue") or {}
    if args.json:
        ctx.say(json.dumps({"queue_busy": busy, "run_queue": rq,
                            "person_hold": status.get("person_hold"),
                            "note": OCCUPANCY_NOTE}, indent=1, default=str))
        return EXIT_BUSY if busy else EXIT_OK
    ctx.say(("queue busy | " if busy else "queue free | ")
            + queue_line(rq, status.get("person_hold")))
    ctx.say(f"({OCCUPANCY_NOTE})")
    return EXIT_BUSY if busy else EXIT_OK


_COMMANDS = {"list": _cmd_list, "show": _cmd_show, "tail": _cmd_tail, "cancel": _cmd_cancel,
             "pause": _cmd_pause, "resume": _cmd_resume, "hold": _cmd_hold,
             "release": _cmd_release, "status": _cmd_status}


def main(argv=None, *, client_factory=None, out=None, err=None, ask=None,
         isatty=None) -> int:
    """The ``kq`` command; returns the exit code (see the module docstring).
    The keywords are for tests: a client factory (default: discovery), the
    output streams, the y/N prompt and the terminal check."""
    if out is None:
        out = sys.stdout
        try:
            out.reconfigure(errors="replace")
        except Exception:                             # noqa: BLE001
            pass
    if err is None:
        err = sys.stderr
        try:
            err.reconfigure(errors="replace")
        except Exception:                             # noqa: BLE001
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    words, expt_argv = _split_argv(argv)
    parser = build_parser(out=out, err=err)
    try:
        args = parser.parse_args(words)
    except SystemExit as exc:
        return EXIT_OK if exc.code in (0, None) else EXIT_USAGE
    if expt_argv and args.cmd not in ("run", "submit"):
        safe_write(err, "kq: arguments after -- are for kq run / kq submit only\n")
        return EXIT_USAGE
    ctx = _Ctx(client_factory or (lambda: RunQueueClient()), out, err,
               ask or _default_ask, isatty or _default_isatty)
    try:
        if args.cmd in ("run", "submit"):
            if args.repeat < 1:
                ctx.warn("kq: --repeat must be at least 1")
                return EXIT_USAGE
            return _cmd_run(ctx, args, expt_argv)
        return _COMMANDS[args.cmd](ctx, args)
    except NoRunQueue as exc:
        ctx.warn(f"kq: {NO_QUEUE_TEXT}")
        ctx.warn(f"kq: ({exc})")
        return EXIT_NO_QUEUE
    except RunQueueError as exc:
        ctx.warn(f"kq: {exc}")
        return EXIT_REFUSED
    except KeyboardInterrupt:
        safe_write(err, "\n")
        ctx.warn("kq: interrupted")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
