"""kq -- the run queue's command line (:mod:`waxx.util.device_state.run_queue`).

The run queue lives in the monitor server, which launches every job itself
(``artiq_run``, detached, its output in a log file on the server's machine).
``kq`` submits jobs, follows their output and asks the queue about its state;
it never runs, stops or kills an experiment itself, and it never falls back to
a direct run when no queue answers.

Commands::

    kq run <file.py> [ARG...] [options] [-- argv...]   submit, then follow its output
    kq submit <file.py> [ARG...] [options] [-- argv...]   submit only (= run --detach)
    kq insert <file.py> [ARG...] (--at-index N | --before ID | --after ID) [options]
                                              run, placed at that position
    kq move <id> (--to N | --before ID | --after ID)    put a queued job elsewhere
    kq edit <id> [--argv ...] [--label L] [--depends-on ID ... | --depends-on-none]
            [--chain C]
            [--stop-on-failure | --no-stop-on-failure] [--no-write-back |
            --write-back-default] [--at T | --no-at] [--allow-drift |
            --no-allow-drift] [--pause | --unpause]     change a queued job
    kq list [--all] [--state S ...] [--json]  the jobs and the queue's state
    kq show <id> [--json]                     one job, why it waits, its last lines
    kq tail <id> [-f]                         its log so far; -f follows to the end
    kq cancel <id> [--yes]                    cancel (a running job: liveOD's Abort)
    kq pause [--all] [--reason R]             stop launching agent (or all) jobs
    kq resume [--all]                         lift that pause
    kq hold [reason]                          a person's hold: agents' jobs wait
    kq release                                lift the person's hold
    kq status [--json]                        the queue's state (not occupancy)

Words after the file (``key=value``, as artiq_run takes them, before or after
kq's options) and everything after ``--`` are passed to the experiment:
``kq run x.py n=3 --label scan m=2 -- -c MyExpt``.  Words starting with ``-``
must go after ``--``, except negative numbers (``-1``), which are taken as
experiment arguments where they stand.  The queue refuses an argument (or a path) holding any
of ``& | < > ^ % " !`` -- e.g. ``x=50%``, which artiq_run itself accepts --
and one ending in a backslash: such a run goes through artiq_run directly.

``kq run`` prints ``[kq] job <id> queued (position k; <why it waits>)``, then
the job's output exactly as the experiment writes it ("Run ID: N" included),
and exits with the experiment's result (see the exit codes).  Ctrl-C while
the job is queued cancels it (and any later jobs of the same submission still
queued); a Ctrl-C during the submit request itself is held until its reply
(at most the request's timeout) and then does the same, so it never leaves a
job queued that this terminal did not name.  Ctrl-C while it runs asks once on a terminal "abort the run? it
discards its data file [y/N]": yes asks the queue to cancel it -- the queue
then sends liveOD's Abort (the run stops at its next shot and liveOD discards
its file) -- and kq prints "abort requested; waiting for the run to end" and
follows the job to its end; anything else, or no terminal, leaves the run going and prints how
to follow or cancel it.  A second Ctrl-C while following an abort leaves too.

Order: the queue runs the eligible job nearest the front.  A new job goes to
the end, except that a person's job is placed ahead of every queued agent job
(``--at-end`` opts out); ``--priority`` is only a placement hint within its
owner's block.  ``kq insert`` places a job, ``kq move`` moves a queued one.
Positions count from 1 (the front), as ``kq list`` shows them.  ``--before ID``
and ``--after ID`` are positions only (insert, move); a dependency -- "only
after these jobs have saved" -- is ``--depends-on ID [ID ...]`` (run, submit,
insert; on edit ``--depends-on`` replaces it and ``--depends-on-none`` clears
it).

Owner: kq acts for a person unless the environment has ``WAXX_OWNER=agent``
(the agents' skill sets it; ``--agent`` is a convenience for the same); an
agent's jobs carry ``WAXX_AGENT_LABEL`` as their submitter when it is set.
Every request that changes something carries the owner: an agent cannot
cancel, move or edit a person's job, release a person's hold or resume a
person's pause (the server's refusal is printed as it words it).  ``by`` is
``user@host``.

``kq list`` shows each job's position, submitter, experiment class, an
estimate (marked ``est.``: the median of the last saved runs of the same file,
or liveOD's shot count for the running job -- an estimate, not a promise), and
why it waits; a ``*`` after the label marks a queued job whose file changed
since submit (it is skipped at launch unless drift is allowed).

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
import signal
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

#: The commands that submit a job.
RUN_COMMANDS = ("run", "submit", "insert")

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
#: ``--at`` further ahead than this is refused (a typo, not a plan).
MAX_AHEAD_S = 366 * 86400.0


def parse_at(text: str, now: float | None = None) -> float:
    """``HH:MM[:SS]`` (local; the next such time, so a time already past
    today means tomorrow -- by the calendar, so a DST change is no hour off)
    or epoch seconds (at least :data:`MIN_EPOCH`, at most a year ahead) ->
    epoch seconds."""
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
    if value > now + MAX_AHEAD_S:
        raise ValueError(f"--at {text}: more than a year ahead")
    return value


# -- the parser -------------------------------------------------------------------------

def build_parser(out=None, err=None) -> argparse.ArgumentParser:
    """kq's parser; its help goes to ``out`` and its errors to ``err``
    (default: stdout / stderr), in ASCII."""

    class _Parser(argparse.ArgumentParser):
        def error(self, message):
            if "--depends-on" in message and "invalid int value" in message:
                # it takes every word after it: kq run f.py --depends-on 3 a=1
                message += (" (--depends-on takes job ids up to the next option: put "
                            "experiment arguments before it, or another option after it)")
            super().error(message)

        def _print_message(self, message, file=None):
            if message:
                stream = ((err or sys.stderr) if file is sys.stderr
                          else (out or sys.stdout))
                safe_write(stream, ascii_text(message))

    p = _Parser(prog="kq", description="The run queue (in the monitor server).",
                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", metavar="command", parser_class=_Parser)
    sub.required = True

    def run_options(sp, insert=False):
        sp.add_argument("file", help="the experiment file (.py)")
        sp.add_argument("args", nargs="*", metavar="ARG",
                        help="the experiment's arguments (key=value), as after the file "
                             "in artiq_run; words starting with - go after --")
        sp.add_argument("--label", help="a short name (default: the file's stem)")
        sp.add_argument("--priority", type=int, default=None,
                        help="placement hint within the owner's block (higher first)")
        sp.add_argument("--at", dest="at", metavar="HH:MM|EPOCH",
                        help="not before this time (local HH:MM: the next such time)")
        sp.add_argument("--depends-on", dest="after", type=int, nargs="+", action="extend",
                        default=[], metavar="ID", help="only after these jobs have saved")
        if insert:
            where = sp.add_mutually_exclusive_group(required=True)
            where.add_argument("--at-index", type=int, metavar="N",
                               help="at position N (1: the front)")
            where.add_argument("--before", type=int, metavar="ID", help="before queued job ID")
            where.add_argument("--after", dest="after_id", type=int, metavar="ID",
                               help="after queued job ID (a position, not a dependency)")
        else:
            sp.add_argument("--at-end", action="store_true",
                            help="at the end of the queue (a person's job otherwise goes "
                                 "ahead of every queued agent job)")
        sp.add_argument("--repeat", type=int, default=1, help="N runs, one job each (a chain)")
        sp.add_argument("--chain", help="chain name (jobs of a chain stop together)")
        sp.add_argument("--no-stop-on-failure", action="store_true",
                        help="a failed job does not cancel the rest of its chain")
        sp.add_argument("--no-write-back", action="store_true",
                        help="veto the experiment's calibration write-back (the job runs "
                             "with WAXX_CAL_NO_WRITE_BACK=1)")
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
    sp = sub.add_parser("insert", help="run a job placed at a position (follows it, as run)")
    run_options(sp, insert=True)
    sp.add_argument("--detach", action="store_true", help="insert only; do not follow")

    sp = sub.add_parser("move", help="put a queued job elsewhere in the order")
    sp.add_argument("id", type=int)
    where = sp.add_mutually_exclusive_group(required=True)
    where.add_argument("--to", type=int, metavar="N", help="to position N (1: the front)")
    where.add_argument("--before", type=int, metavar="ID", help="before queued job ID")
    where.add_argument("--after", type=int, metavar="ID", help="after queued job ID")
    sp.add_argument("--token")
    sp.add_argument("--agent", action="store_true", help="as an agent")

    sp = sub.add_parser("edit", help="change a queued job")
    sp.add_argument("id", type=int)
    sp.add_argument("--argv", nargs="*", metavar="ARG",
                    help="the experiment's arguments, replaced (none: cleared; words "
                         "starting with - go after --)")
    sp.add_argument("--label")
    flag = sp.add_mutually_exclusive_group()
    flag.add_argument("--depends-on", dest="after", type=int, nargs="+", metavar="ID",
                      help="only after these jobs have saved (replaces the list)")
    flag.add_argument("--depends-on-none", dest="after", action="store_const", const=[],
                      help="no dependencies")
    sp.add_argument("--chain", help="chain name ('' clears it)")
    flag = sp.add_mutually_exclusive_group()
    flag.add_argument("--stop-on-failure", dest="stop_on_failure", action="store_const",
                      const=True)
    flag.add_argument("--no-stop-on-failure", dest="stop_on_failure", action="store_const",
                      const=False)
    flag = sp.add_mutually_exclusive_group()
    flag.add_argument("--no-write-back", dest="write_back", action="store_const",
                      const="veto", help="ask that the write-back be vetoed")
    flag.add_argument("--write-back-default", dest="write_back", action="store_const",
                      const="default", help="drop that request")
    flag = sp.add_mutually_exclusive_group()
    flag.add_argument("--at", dest="at", metavar="HH:MM|EPOCH")
    flag.add_argument("--no-at", dest="at", action="store_const", const="",
                      help="no due time")
    flag = sp.add_mutually_exclusive_group()
    flag.add_argument("--allow-drift", dest="allow_drift", action="store_const", const=True)
    flag.add_argument("--no-allow-drift", dest="allow_drift", action="store_const",
                      const=False)
    flag = sp.add_mutually_exclusive_group()
    flag.add_argument("--pause", dest="paused", action="store_const", const=True,
                      help="this job is not launched until --unpause")
    flag.add_argument("--unpause", dest="paused", action="store_const", const=False)
    sp.add_argument("--token")
    sp.add_argument("--agent", action="store_true", help="as an agent")

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
    # Ctrl-C is held back from the submit request until its reply has been
    # read (bounded by the request's timeout): a job the server queued is then
    # known by id and is cancelled below, never left queued unseen
    guard = _SigintGuard().start()
    try:
        reply = ctx.client.submit(
            args.file, argv=list(args.args) + expt_argv, cwd=args.cwd, label=args.label,
            owner=owner, priority=args.priority, due=due, after=args.after,
            repeat=args.repeat, chain=args.chain,
            stop_on_failure=False if args.no_stop_on_failure else None,
            write_back=False if args.no_write_back else None, allow_drift=args.allow_drift,
            at_end=getattr(args, "at_end", False),
            at_index=(None if getattr(args, "at_index", None) is None
                      else args.at_index - 1),
            before_id=getattr(args, "before", None), after_id=getattr(args, "after_id", None))
        jobs = reply.get("jobs") or []
        ids = [int(i) for i in reply.get("ids") or [j["id"] for j in jobs]]
        tokens = {int(j["id"]): j.get("token") for j in jobs}
    except KeyboardInterrupt:
        # raised inside the request without the guard (it could not be set:
        # not the main thread)
        guard.stop()
        safe_write(ctx.out, "\n")
        ctx.warn("[kq] interrupted while submitting: the job may be queued -- check kq list")
        return EXIT_INTERRUPTED
    except BaseException:
        if guard.stop():
            ctx.warn("[kq] interrupted while submitting")
        raise
    if not ids:
        if guard.stop():
            safe_write(ctx.out, "\n")
        ctx.warn("kq: the queue accepted the request but returned no job")
        return EXIT_REFUSED
    first = jobs[0] if jobs else {"id": ids[0]}
    many = (f"jobs {ids[0]}-{ids[-1]} (chain {first.get('chain')})" if len(ids) > 1
            else f"job {ids[0]}")
    due_text = f", due {_when(due)}" if due else ""
    # from here on a Ctrl-C anywhere -- one held back during the submit
    # included -- goes to _interrupted: the job ids are known, a queued job is
    # cancelled (queued_only) and its id printed
    n, cursor = 0, {"offset": 0}
    try:
        if guard.stop():
            raise KeyboardInterrupt
        if args.cmd == "submit" or getattr(args, "detach", False):
            placed = _position(ctx, ids[0], tokens.get(ids[0]))
            ctx.say(f"[kq] {many} queued: {first.get('label')}, owner {owner}{due_text} "
                    f"({placed})")
            ctx.say(f"[kq] follow: kq tail {ids[0]} -f    cancel: kq cancel {ids[0]}")
            return EXIT_OK
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


class _SigintGuard:
    """Holds Ctrl-C back while it is on: SIGINT only sets ``fired``.
    :meth:`stop` puts the previous handler back and returns ``fired``.  Off
    the main thread (where Python cannot set a handler) it does nothing."""

    def __init__(self):
        self.fired = False
        self._old = None
        self._on = False

    def _handler(self, signum, frame):
        self.fired = True

    def start(self) -> "_SigintGuard":
        try:
            self._old = signal.signal(signal.SIGINT, self._handler)
            self._on = True
        except (ValueError, OSError):
            self._on = False
        return self

    def stop(self) -> bool:
        if self._on:
            self._on = False
            signal.signal(signal.SIGINT, self._old)
        return self.fired


def _position(ctx: _Ctx, jid: int, token) -> str:
    """"position 2 of 5; <why it waits>" for a job just submitted (positions
    count from 1, the front)."""
    try:
        described = ctx.client.describe(jid, token)
    except RunQueueError:
        return "position unknown"
    job = described.get("job") or {}
    why = described.get("waiting") or job.get("waiting") or ""
    pos = job.get("position")
    if pos is not None:
        return f"position {int(pos) + 1}; {why or 'launching'}"
    return f"{job.get('state')}; {why}" if why else f"{job.get('state')}"


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
                    ctx.say(f"[kq] stopped following {_job_word(job)}: its abort was "
                            f"requested and it ends on its own (kq show {jid}). "
                            f"{_leave_text(jid)}")
                    return EXIT_INTERRUPTED
                _report_end(ctx, final)
                return exit_code_for(final)
        ctx.say(f"[kq] interrupted: {_job_word(job)} is left {job['state']} ({run})"
                + (f"; jobs {', '.join(map(str, rest))} stay queued" if rest else "")
                + f". {_leave_text(jid)}")
        return EXIT_INTERRUPTED
    except KeyboardInterrupt:
        # a second Ctrl-C while the interrupt was being handled: a cancel (or
        # an abort) may already have reached the queue
        safe_write(ctx.out, "\n")
        ctx.say(f"[kq] interrupted again: a cancel of job {jid} may or may not have reached "
                f"the queue -- check kq show {jid}. {_leave_text(jid)}")
        return EXIT_INTERRUPTED
    except RunQueueError as exc:
        # still an interrupt: the person asked to stop; what the queue said goes
        # with it, and the job is whatever the queue made of it
        ctx.warn(f"kq: {exc}")
        ctx.say(f"[kq] interrupted: a cancel of job {jid} may or may not have reached the "
                f"queue -- check kq show {jid}. {_leave_text(jid)}")
        return EXIT_INTERRUPTED


def _duration_text(seconds) -> str:
    seconds = float(seconds)
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def est_text(est: dict | None, state: str) -> str:
    """The estimate column, always marked "est." (an estimate from earlier
    runs of the same file, or liveOD's shot count -- not a promise)."""
    est = est or {}
    if state in ENDED or not est:
        return "-"
    if state in IN_SLOT and est.get("eta_end"):
        return f"est. end {_when(est['eta_end'])}"
    if est.get("eta_start"):
        return f"est. start {_when(est['eta_start'])}"
    if est.get("duration_s") is not None:
        return f"est. {_duration_text(est['duration_s'])}"
    return "-"


def _note(view: dict, row: dict | None) -> str:
    """The last column: why a queued job waits, an abort requested for the
    job in the slot, why an ended job ended."""
    state = view.get("state")
    if state in IN_SLOT and view.get("cancel"):
        c = view["cancel"]
        text = (f"abort requested by {c.get('by') or '?'}"
                + (f": {c.get('abort_note')}" if c.get("abort_note") else ""))
    elif state in ENDED:
        text = str(view.get("reason") or "")
    else:
        text = str((row or {}).get("waiting") or view.get("waiting") or "")
        if view.get("paused"):
            text = f"PAUSED by {view.get('paused_by') or '?'}" + (f"; {text}" if text else "")
    return text[:67] + "..." if len(text) > 70 else text


def _list_rows(jobs: list[dict], rows: dict) -> list[list[str]]:
    """The table: the server's ``rows`` (by id) for position, submitter, class
    and estimate, the job views for the rest."""
    table = [["pos", "id", "state", "owner", "submitter", "label", "class", "run_id", "est",
              "waiting / reason"]]
    for j in jobs:
        row = rows.get(j.get("id")) or {}
        pos = row.get("position", j.get("position"))
        label = str(j.get("label")) + ("*" if j.get("source_changed") else "")
        run_id = row.get("run_id", j.get("run_id"))
        table.append([
            "-" if pos is None else str(int(pos) + 1), str(j.get("id")), str(j.get("state")),
            str(j.get("owner")), str(row.get("submitter") or j.get("submitter") or "-"),
            label, str(row.get("expt_class") or j.get("expt_class") or "-"),
            "-" if run_id is None else str(run_id),
            est_text(row.get("est", j.get("estimate")), str(j.get("state"))),
            _note(j, row) or "-"])
    return table


SOURCE_CHANGED_NOTE = ("* source changed since submit: the job is skipped at launch unless "
                       "drift is allowed (kq edit <id> --allow-drift)")


def _cmd_list(ctx: _Ctx, args) -> int:
    reply = ctx.client.list(states=args.state, limit=args.limit)
    jobs = reply.get("jobs") or []
    if not args.all and not args.state:
        ended = [j for j in jobs if j.get("state") in ENDED]
        keep = {j["id"] for j in ended[-LIST_ENDED:]}
        jobs = [j for j in jobs if j.get("state") not in ENDED or j["id"] in keep]
    if args.json:
        ids = {j["id"] for j in jobs}
        ctx.say(json.dumps({"jobs": jobs,
                            "rows": [r for r in reply.get("rows") or [] if r.get("id") in ids],
                            "next": reply.get("next"), "run_queue": reply.get("run_queue")},
                           indent=1, default=str))
        return EXIT_OK
    rows = {r.get("id"): r for r in reply.get("rows") or []}
    table = _list_rows(jobs, rows)
    widths = [max(len(ascii_text(r[i])) for r in table) for i in range(len(table[0]) - 1)]
    if len(table) > 1:
        for r in table:
            ctx.say("  ".join(c.ljust(w) for c, w in zip(r[:-1], widths)) + "  " + r[-1])
    else:
        ctx.say("(no jobs)")
    if any(j.get("source_changed") for j in jobs):
        ctx.say(SOURCE_CHANGED_NOTE)
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
    if j.get("position") is not None:
        ctx.say(f"  position: {int(j['position']) + 1}")
    if reply.get("waiting"):
        ctx.say(f"  waiting: {reply['waiting']}")
    if j.get("paused"):
        ctx.say(f"  PAUSED (this job) by {j.get('paused_by') or '?'} since "
                f"{_when(j.get('paused_since'))}")
    if j.get("source_changed"):
        ctx.say("  source changed since submit: skipped at launch unless drift is allowed")
    for key in ("path", "expt_class", "argv", "cwd", "owner", "submitter", "priority", "chain",
                "after", "run_id", "exit_code", "log_path", "submitted_by", "token"):
        val = j.get(key)
        if val not in (None, "", []):
            ctx.say(f"  {key}: {val}")
    cal = j.get("calibrates_declared")
    if cal is not None:
        ctx.say("  calibrates (declared in the file): " + (", ".join(map(str, cal)) or "none"))
    if j.get("repeat_of", 1) > 1:
        ctx.say(f"  repeat: {j.get('repeat_index')} of {j.get('repeat_of')}")
    for key in ("due", "submitted_at", "launched_at", "ended_at"):
        if j.get(key):
            ctx.say(f"  {key}: {_when(j[key])}")
    est = j.get("estimate")
    if est and j.get("state") not in ENDED:
        ctx.say(f"  {est_text(est, str(j.get('state')))} -- basis: {est.get('basis') or '?'}")
    if j.get("write_back") is False:
        ctx.say("  write-back vetoed (WAXX_CAL_NO_WRITE_BACK=1)")
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


def _cmd_move(ctx: _Ctx, args) -> int:
    reply = ctx.client.move(args.id, to_index=None if args.to is None else args.to - 1,
                            before_id=args.before, after_id=args.after, token=args.token,
                            owner=_owner(args))
    job = reply.get("job") or {"id": args.id}
    pos = reply.get("position")
    ctx.say(f"[kq] {_job_word(job)} moved to position "
            + ("?" if pos is None else str(int(pos) + 1)))
    return EXIT_OK


def _cmd_edit(ctx: _Ctx, args) -> int:
    fields = {}
    if args.argv is not None:
        fields["argv"] = list(args.argv)
    if args.label is not None:
        fields["label"] = args.label
    if args.after is not None:
        fields["after"] = list(args.after)
    if args.chain is not None:
        fields["chain"] = args.chain or None
    if args.stop_on_failure is not None:
        fields["stop_on_failure"] = args.stop_on_failure
    if args.write_back is not None:
        fields["write_back"] = False if args.write_back == "veto" else None
    if args.at is not None:
        if args.at == "":
            fields["due"] = None
        else:
            try:
                fields["due"] = parse_at(args.at)
            except ValueError as exc:
                ctx.warn(f"kq: {exc}")
                return EXIT_USAGE
    if args.allow_drift is not None:
        fields["allow_drift"] = args.allow_drift
    if args.paused is not None:
        fields["paused"] = args.paused
    if not fields:
        ctx.warn("kq: nothing to change (kq edit --help lists the fields)")
        return EXIT_USAGE
    reply = ctx.client.edit(args.id, fields, token=args.token, owner=_owner(args))
    job = reply.get("job") or {"id": args.id}
    changed = reply.get("changed") or []
    ctx.say(f"[kq] {_job_word(job)}: "
            + (f"changed {', '.join(changed)}" if changed else "nothing changed"))
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
    token = args.token or ctx.client.job_token(args.id)
    while True:
        reply = ctx.client.tail(args.id, token, offset)
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
             "move": _cmd_move, "edit": _cmd_edit,
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
        # parse_known_args: experiment words may come after kq's options too
        # (kq run f.py a=1 --label x b=2); anything else left over is an error
        args, extra = parser.parse_known_args(words)
        if extra:
            if args.cmd in RUN_COMMANDS and not any(w.startswith("-") for w in extra):
                args.args = list(args.args) + extra
            else:
                parser.error("unrecognized arguments: " + " ".join(extra)
                             + (" (experiment options go after --)"
                                if args.cmd in RUN_COMMANDS else ""))
    except SystemExit as exc:
        return EXIT_OK if exc.code in (0, None) else EXIT_USAGE
    if expt_argv and args.cmd == "edit":
        args.argv = list(args.argv or []) + expt_argv
    elif expt_argv and args.cmd not in RUN_COMMANDS:
        safe_write(err, "kq: arguments after -- are for kq run / submit / insert / edit "
                        "only\n")
        return EXIT_USAGE
    ctx = _Ctx(client_factory or (lambda: RunQueueClient()), out, err,
               ask or _default_ask, isatty or _default_isatty)
    try:
        if args.cmd in RUN_COMMANDS:
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
