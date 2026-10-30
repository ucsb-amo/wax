"""The K-machine run queue: experiment jobs, one at a time, run by the monitor
server (run-queue plan, phase 1).

The queue generalises the server's run loop (:mod:`~waxx.util.device_state.run_loop`)
from "one file back to back" to "these jobs, in this order": it reuses the
loop's launch command (:func:`~waxx.util.device_state.monitor_manager.ar_command`,
``artiq_run`` directly), its judging of a finished run
(:func:`~waxx.util.device_state.run_loop.judge_run`: exit code 0 and liveOD's
outcome "saved"), its RUN_EXITED-on-behalf rule
(:func:`~waxx.util.device_state.run_loop.tell_live_od_exited`) and the shared
liveOD verdict (:func:`~waxx.util.device_state.run_gate.classify`).  This module
is pure logic: the server passes liveOD, its fence, its monitor state and its
loops in, and calls :meth:`RunQueue.tick` from its watch thread.

**Jobs.**  A job is one run of one experiment file: its absolute path and the
file's SHA-256 at submit, extra ``artiq_run`` arguments, working folder, a
label, its owner (``"person"`` -- the default -- or ``"agent"``), a priority
(higher first), an optional due time, jobs it must wait for (``after``: each
must end *saved* or *done*), an optional chain name, and the submitter's write-back veto.
``repeat`` N submits N jobs in one chain.  States: ``queued`` -> ``running``
-> ``ending`` (its process has exited; the outcome is being read) ->
``saved`` | ``done`` (ran to its end with ``save_data=False``: no file) |
``failed`` | ``aborted`` (an Abort pressed in liveOD); or ``cancelled`` /
``skipped`` (never launched, or launched and cancelled).  ``no_file`` marks a
job whose run had ``save_data=False``, whatever it ended as (from liveOD's
POLL / last_outcome, or "Acquisition for run 0 aborted" in its output).

**Scheduling.**  One slot.  A job is *eligible* when it is queued, its due time
has passed, every job in ``after`` is saved (or done), it is not paused (scope "all", or
"agent" for an agent's job) and -- for an agent's job -- no person's hold is on
(:mod:`~waxx.util.device_state.person_hold`).  The queue has an explicit
order: every job has a ``rank`` and the eligible job with the lowest rank goes
next (due times, dependencies, pauses and the hold only gate a job, they never
reorder the queue).  At submit a job goes to the end, except that a person's job
is placed ahead of every queued agent job (unless ``at_end``); ``priority`` is
only a placement hint within its owner's block (higher first) and is kept for
display.  ``insert`` submits at a position (``at_index`` / ``before_id`` /
``after_id``); ``move`` puts a queued job elsewhere (an agent may move only
agent jobs; launching or running jobs never move).  It is launched only when the previous job's process
has exited and the machine is free: nothing of the server's own holds it
(``server_busy``: a state reset, the monitor starting), no run loop is active,
and liveOD + the run fence say free (:func:`run_gate.classify`; a dead run is
waived as by the loop).  A job ``after`` one that ended failed / cancelled /
skipped is skipped; a job in a chain that fails or is skipped (with
``stop_on_failure``, the default in a chain) cancels the chain's other queued
jobs.

**Launch.**  At launch the file's SHA-256 is computed again: a file changed
since submit is *skipped* ("source changed since submit") unless the job
allows drift.  The command is ``ar_command(path) + argv``, run detached
(:mod:`~waxx.util.device_state.detached`: the job survives a server restart)
with ``WAXX_LAUNCHER=kq``, ``WAXX_QUEUE_JOB=<id>``, ``WAXX_OWNER=<owner>``,
``PYTHONUNBUFFERED=1`` and, for a write-back veto, ``WAXX_CAL_NO_WRITE_BACK=1``;
its output goes to ``<dir>/logs/<id>_<label>.out``.  The job's run is known
from liveOD: the experiment sends ``queue_job`` (WAXX_QUEUE_JOB = "<id>:<token>") at INIT_RUN
and POLL / last_outcome carry it next to ``launcher`` and ``client_pid`` (the
experiment's own pid); the "Run ID:" line in the log (printed whatever
WAX_VERBOSITY when a launcher is set) is the fallback.
``PYTHONIOENCODING=utf-8`` is set too: the child writes UTF-8 bytes into
the log file the launcher opened.  The ``tail`` request serves the log
from a byte offset, so the ``kq`` client
(:mod:`~waxx.util.device_state.kq`) on any PC can follow it.

**Cancel.**  A queued job is cancelled at once.  A running job is never
terminated: the queue sends liveOD's Abort (RESET) for it -- only when liveOD's
run in progress is that job's run id, on a POLL just before; for a run with no
file (``no_file``: run id 0, as every save_data=False run) the POLL must name
the job's ``queue_job`` too, and the RESET carries it, so liveOD refuses it
for any other run -- and the job
stays ``running`` until its process ends; it then ends ``cancelled`` (or
``saved``, if it saved before the Abort reached it).  An Abort is what liveOD's
own button does: the run stops at its next shot and liveOD discards its file.
An agent may not cancel a person's running job.

**Kill.**  For when the Abort cannot work -- liveOD closed, the experiment's
terminal or process hung, the queue out of step with what runs -- a person
may force-kill a launching or running job (``kill``), and so may an agent for
its own job while liveOD says that job's run is stuck (user, 2026-10-30:
wedged, or an Abort pending and unanswered, or its client dead; ``AGENT_KILL_STATES``): the
experiment's process (``client_pid``, only when its creation time matches
the one recorded from liveOD) and the shell the queue followed, each with
everything under it (:func:`~waxx.util.device_state.detached.kill_tree`).
Nothing of the experiment's cleanup runs: the hardware stays as the run left
it until the next run or the monitor sets it.  The job then ends as any run
whose process died: liveOD is told RUN_EXITED on its behalf, which leaves
its file as it was (not saved, not deleted), and the job ends ``cancelled``
("force-killed by ...", or ``saved`` if it had saved).  A pending Abort is
never sent after a kill.  A job with no process left to kill (gone, or never
known) is ended the same way at once, which frees the slot.

**Owners.**  A job's ``owner`` (submit) and the asker's ``owner`` on cancel,
release and resume ("person" | "agent"; required on those three, refused
without) decide what an agent may undo: an agent may not cancel a person's
job, release a person's hold or resume (or replace) a person's pause.  A
hold or pause records who put it on (``owner``, "person" when absent; liveOD's
Reset and the fail-closed hold/pause are a person's).  The kq client sets
``owner`` from the environment variable ``WAXX_OWNER`` ("person" when unset;
the run-experiment skill sets "agent"); the Device Control GUI sends "person".

**Pause / hold.**  ``pause`` with scope "agent" or "all" stops new launches for
those jobs (the running one is not touched); ``resume`` lifts it.  A person's
hold is pause("agent") with provenance (since, by, reason) and is never lifted
by itself.

**Loops.**  When a job is eligible and one of the server's run loops is active
(the TOF loop), the queue asks it to stop gracefully (its run in progress
finishes and saves; no monitor start), remembers it, and runs its jobs; when no
job is eligible or running any more -- with no hold and nothing paused -- it
starts that loop again (not a loop that latched off), with the category it had
and run for the owner it had (``on_behalf_of``).  Which jobs stop which loop
goes by the loop's category (since 2026-10-20;
:data:`~waxx.util.device_state.run_loop.LOOP_CATEGORIES`): a person's job stops
any loop; an agent's job stops a *yielding* loop, whoever started it, and waits
behind a *holding* one.  A loop started with no category yields
(:data:`~waxx.util.device_state.run_loop.DEFAULT_CATEGORY`).  Otherwise, when the
queue has launched jobs and has none left to run, it asks for the monitor (the
queue's jobs do not restart it; see ``WAXX_LAUNCHER`` in ``Expt.end_wax``).

**Alarm.**  An eligible job waiting with no job running for more than
:data:`ALARM_S` raises an alarm: one WARNING (and journal record) per
:data:`ALARM_S`, ``alarm`` in the status, cleared by the next launch.

**Records.**  ``<dir>`` is on local disk (the server's default:
:func:`default_dir`, ``~/.waxx/run_queue``).  ``<dir>/queue.json`` holds every job not yet ended and the last
:data:`KEEP_ENDED` ended ones (atomic replace on every change);
``<dir>/journal.jsonl`` gets one line per transition (append only), and the
server's ops journal the same records (kinds ``run_queue_*``; copied by
:meth:`RunQueue.flush_journal` outside the queue's lock, since the ops journal
appends to the lab's share synchronously).  At a server
restart the queue is read back; a running job whose process is still alive
(same pid, same creation time) is adopted and followed as before; one whose
process is gone is judged from liveOD's record (saved, or failed "server
restarted; process gone").

**Warm start (queue phase 2).**  A stored setting since 2026-10-21: the
monitor server constructs the queue with ``warm_ahead=None`` -- start value
``WAXX_QUEUE_WARM_AHEAD`` when set, else ``<dir>/settings.json``, else
:data:`WARM_AHEAD_DEFAULT` (on); ``kq warm on|off`` and the Monitor panel's
toggle (:meth:`RunQueue.set_warm_ahead`) switch it and store it, journalled.
``RunQueue(warm_ahead=False)`` (the class default, for library use and tests)
takes the value given.
With ``warm_ahead`` on, the queue starts the NEXT job's process while the job
in the slot is still running -- once that job has taken its first shot (the
two compiles never overlap), its end is within :data:`WARM_LEAD_S` (or
unknown), it declares no calibration, and liveOD's POLL advertises
``features.reserve``.  One warm process at a time, for the head of the order
(a job due within :data:`WARM_DUE_LEAD_S` counts).  The warm process gets
``WAXX_QUEUE_WARM=<warm token>`` and BELOW_NORMAL priority; it imports,
prepares and compiles, claims nothing, says "ready" with a manifest of the
files it loaded (:mod:`~waxx.util.device_state.queue_gate`) and polls the
``warm`` request.  The job stays ``queued`` with a ``warm`` sub-record
(``{token, state: warming | ready | aborting, attempt, cmd_pid, cmd_started,
py_pid, py_started, sha256, manifest, spawned_at, ready_at, go, ...}``); an
older queue.json loads (no record), and an older server ignores the field.
GO is decided by the tick and SAVED before it is handed out (the job goes
``launching`` -> ``running`` with the warm process as its process): only
once the slot job's process has exited, the gate is free and the manifest is
fresh (re-stat outside the lock).  The reply carries liveOD's ``reset_count``
as last seen.  A warm launch is discarded -- cooperatively: the process
exits at its next poll -- when the job is no longer the head, is blocked
or paused, is edited (argv, after, due, paused, write_back, cold),
cancelled, its chain stops, its files change, or it has been ready for
:data:`WARM_READY_MAX_S`.  A person hold does NOT discard it: an agent's
warm process stays ready behind the hold (GO waits for the release; the
ready time still counts, so a hold longer than :data:`WARM_READY_MAX_S`
discards it as stale, its attempt given back, and no new warm launch starts
for a held job -- the job then launches cold after the release).  A
discarded process still alive :data:`WARM_KILL_AFTER_S` after it was told
to exit is stopped by the queue (:func:`~waxx.util.device_state.detached.
kill_tree`) only while GO was never delivered to it and its pid and creation
time still match the record; otherwise it raises a warning (``warm_alarm``).
Both leave a journal record.  A re-warm waits for the old process to exit.  Jobs marked
``cold`` (submit / edit), whose class sets ``warm_launch = False``, or whose
``prepare``/``build`` calls a name in :data:`WARM_DENY_NAMES` never warm.  A
warm process that dies before GO (not told to, not exit 78) makes its job
cold and leaves it queued ("warm attempt failed ... it launches cold"); one
that dies after GO was saved but before it was
handed out puts its job back in the queue; after the hand-out, the usual
following applies.  At a restart warm processes are taken back by pid AND
creation time.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import socket
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Mapping

from waxx.util.device_state import queue_gate, run_gate
from waxx.util.device_state.monitor_manager import ar_command, environment_report
from waxx.util.device_state.detached import LAUNCH_TIMEOUT_S, SAME_PROCESS_S
from waxx.util.device_state.person_hold import PersonHold
from waxx.util.device_state.run_loop import (
    ABORTED_RE, RUN_ID_RE, _SHELL_CHARS, effective_owner, info_category, judge_run,
    live_od_outcome, tell_live_od_exited)

log = logging.getLogger(__name__)

STATES = ("queued", "launching", "running", "ending", "saved", "done", "failed", "aborted",
          "cancelled", "skipped")
#: States a job ends in.
ENDED = ("saved", "done", "failed", "aborted", "cancelled", "skipped")
#: The ended states of a job that ran to its end: "saved", or "done" (a run
#: with ``save_data=False``: it went to its end and liveOD wrote nothing).
#: Dependencies, chains and exit codes treat both as success.
SUCCEEDED = ("saved", "done")
#: States of the one job in the slot ("launching": saved before its process is
#: started, so a server that dies meanwhile never starts it twice).
IN_SLOT = ("launching", "running", "ending")
OWNERS = ("person", "agent")
PAUSE_SCOPES = ("agent", "all")

#: The value of ``WAXX_LAUNCHER`` for the queue's jobs (run_gate.KNOWN_LAUNCHERS).
LAUNCHER = "kq"
JOB_ENV = "WAXX_QUEUE_JOB"
OWNER_ENV = "WAXX_OWNER"
#: A submitter's write-back veto (submit "write_back": false) reaches the job as
#: this variable = "1".  It is honoured by the calibration emit on the
#: jep/calibration-writeback branch (waxx/calibration/emit.py: "[cal] not
#: applied: vetoed by WAXX_CAL_NO_WRITE_BACK"); this branch only sets it.
NO_WRITE_BACK_ENV = "WAXX_CAL_NO_WRITE_BACK"

#: An eligible job waiting this long with nothing running raises the alarm.
ALARM_S = 600.0
#: liveOD is polled at most this often for the hold's watch and the status
#: (a launch, a cancel and an outcome poll afresh).
POLL_EVERY_S = 2.0
#: Pause after a job's end before the next launch.
GAP_S = 0.5
#: Ended jobs kept in queue.json (all of them stay in journal.jsonl).
KEEP_ENDED = 500
MAX_REPEAT = 1000
#: Lines of a job's output kept in memory for its judging and ``describe``.
TAIL_LINES = 25
#: Most bytes of a job's log one ``tail`` request returns.
TAIL_CHUNK = 64 * 1024
#: After a launch whose "launching" state could not be saved, launches wait this long.
SAVE_RETRY_S = 30.0
#: A launch taking longer than this is logged (once) while it is waited for.
SPAWN_WARN_S = 60.0
#: A job left "launching" with no process to follow (the server stopped while
#: launching it, or the launcher never answered) is looked for in liveOD this
#: long after its launch (INIT_RUN comes at the end of its prepare), then
#: ended failed.
ORPHAN_WAIT_S = 300.0

#: A warm launch's environment variable (queue_gate pops it at first read).
WARM_ENV = queue_gate.WARM_ENV
#: The exit code of a warm process that stood down before GO.
EXIT_WARM_ABORTED = queue_gate.EXIT_WARM_ABORTED
#: A warm launch starts only when the slot job's expected end is this close
#: (or unknown).
WARM_LEAD_S = 300.0
#: A job due this soon may warm (GO only once it is due).
WARM_DUE_LEAD_S = 60.0
#: A warm process ready for longer than this is discarded (stale).
WARM_READY_MAX_S = 900.0
#: A warm process not ready this long after its launch is discarded.
WARM_PREP_MAX_S = 600.0
#: A discarded warm process still alive this long after it was told to exit
#: is stopped by the queue (GO never delivered, pid + creation time
#: matching), else raises the warm alarm.
WARM_KILL_AFTER_S = 120.0
WARM_EXIT_ALARM_S = WARM_KILL_AFTER_S          # the older name
#: A ready warm process's manifest is re-stat-ed this often while it waits.
WARM_MANIFEST_CHECK_S = 5.0
#: Warm launches per job at most (after that it goes cold).  A launch
#: discarded only because it went stale (ready too long) is not counted.
WARM_MAX_ATTEMPTS = 3
#: The end of the discard reason of a warm process ready too long.
WARM_STALE_MARK = "(stale)"
#: A warm launch whose launcher gave no answer is waited for (its hello)
#: this long before it is forgotten.
WARM_HELLO_WAIT_S = 120.0
#: At most this many manifest entries are kept.
WARM_MANIFEST_MAX = 20000
#: The job fields whose edit discards a warm launch.
WARM_SPEC_FIELDS = ("argv", "after", "due", "paused", "write_back", "cold")
#: Names that make a job cold when its prepare / build / __init__ calls them
#: (hardware or shared lab state touched before GO).  ``RunQueue(
#: warm_deny_names=)`` replaces them.
WARM_DENY_NAMES = ("_announce_to_monitor", "announce_run", "withdraw_run", "picomotor",
                   "stage_dev", "objective_stage")

_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")

#: Environment variable that moves the queue's folder (tests point it at a
#: temporary folder; a lab may point it at another LOCAL folder).
DIR_ENV = "WAXX_RUN_QUEUE_DIR"

#: The queue's settings, beside queue.json (``{"warm_ahead": {"on", "by",
#: "owner", "at"}}``; atomic replace).  Read at the server's start.
SETTINGS_FILE = "settings.json"
#: Warm start when nothing is stored (user, 2026-10-21: on by default).
WARM_AHEAD_DEFAULT = True
#: Environment variable that, when set, overrides the stored warm start at
#: the server's start ("1" / "on" / "true" or "0" / "off" / "false").
WARM_AHEAD_ENV = "WAXX_QUEUE_WARM_AHEAD"


def warm_ahead_from_env() -> bool | None:
    """``WAXX_QUEUE_WARM_AHEAD`` as a bool, None when unset (or not
    understood: a warning, and the stored setting applies)."""
    raw = os.environ.get(WARM_AHEAD_ENV, "").strip().lower()
    if not raw:
        return None
    if raw in ("1", "on", "true", "yes"):
        return True
    if raw in ("0", "off", "false", "no"):
        return False
    log.warning("Run queue: %s=%r not understood (1/on or 0/off); the stored warm start "
                "applies.", WARM_AHEAD_ENV, raw)
    return None


def read_settings(directory: str | None) -> dict:
    """The queue's stored settings (``{}`` when there are none or the file
    cannot be read -- logged)."""
    if not directory:
        return {}
    path = os.path.join(directory, SETTINGS_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.error("Run queue: could not read its settings %s (%s); defaults apply.", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def stored_warm_ahead(directory: str | None) -> bool | None:
    """The stored warm start (None: nothing stored)."""
    rec = read_settings(directory).get("warm_ahead")
    on = rec.get("on") if isinstance(rec, dict) else rec
    return on if isinstance(on, bool) else None


#: Environment variable listing the folders the queue runs files from
#: (os.pathsep-separated); see :func:`default_roots`.
ROOTS_ENV = "WAXX_RUN_QUEUE_ROOTS"


def default_roots() -> list[str]:
    """The folders a job's file must be inside: ``$WAXX_RUN_QUEUE_ROOTS``
    (os.pathsep-separated), else the lab's code folder (``%code%``) and
    ``C:\\lab\\skynet_log`` (agents' work).  A server may pass its own
    (``RunQueue(roots=)``); the lab can override them there later.  One
    definition (:func:`queue_gate.queue_roots`): a warm process builds its
    manifest from the same roots."""
    return queue_gate.queue_roots()


def default_dir() -> str:
    """The queue's folder when the server is given none: ``$WAXX_RUN_QUEUE_DIR``,
    else ``~/.waxx/run_queue`` -- local disk, next to the other ~/.waxx state
    (the job logs are written by the running experiments: a network share
    hiccup there would stall their output)."""
    return os.environ.get(DIR_ENV) or str(Path.home() / ".waxx" / "run_queue")


def _clock_text(t) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(t)))
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def describe_source(path) -> tuple[str, list]:
    """``(expt_class, calibrates_declared)`` from an experiment file's source,
    read with :mod:`ast` (never imported): the single public class that
    subclasses ``EnvExperiment`` or defines ``scan_kernel`` ("" when there is
    none or more than one), and the first argument of every
    ``self.calibrates('<key>', ...)`` call with a string key, in order, once
    each.  ("", []) when the file cannot be read or parsed."""
    try:
        return _read_source(path)
    except Exception as exc:                          # noqa: BLE001
        # a pathological file (RecursionError / MemoryError from ast, ...):
        # the listing goes without; the job is submitted all the same
        log.warning("Run queue: could not read %s's class and calibrations (%r); listed "
                    "without them.", path, exc)
        return "", []


def _read_source(path) -> tuple[str, list]:
    import ast  # noqa: PLC0415
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return "", []
    classes = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        bases = [ast.unparse(b).split(".")[-1] for b in node.bases]
        scan = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == "scan_kernel" for n in node.body)
        if "EnvExperiment" in bases or scan:
            classes.append(node.name)
    keys: list = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "calibrates" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str) and node.args[0].value not in keys):
            keys.append(node.args[0].value)
    return (classes[0] if len(classes) == 1 else ""), keys


#: The methods whose calls decide a job's warm eligibility (they run before GO).
_PRE_GO_METHODS = ("prepare", "build", "__init__", "finish_prepare")


def cold_reason(path, deny_names=WARM_DENY_NAMES) -> str:
    """Why a job of this experiment file must never warm-launch ("" when it
    may), read with :mod:`ast` (never imported): a class body that sets
    ``warm_launch = False``, or a call in ``prepare`` / ``build`` /
    ``__init__`` / ``finish_prepare`` whose dotted name contains one of
    ``deny_names`` (hardware or shared lab state touched before GO).  A file
    that cannot be read or parsed is cold (nothing shows it is safe)."""
    import ast  # noqa: PLC0415
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:                          # noqa: BLE001
        return f"its source could not be read for the warm check ({type(exc).__name__})"
    deny = [d.lower() for d in deny_names if d]
    try:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if (isinstance(item, (ast.Assign, ast.AnnAssign))
                        and isinstance(getattr(item, "value", None), ast.Constant)
                        and item.value.value is False
                        and any(isinstance(t, ast.Name) and t.id == "warm_launch"
                                for t in (item.targets if isinstance(item, ast.Assign)
                                          else [item.target]))):
                    return f"class {node.name} sets warm_launch = False"
                if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name in _PRE_GO_METHODS):
                    for sub in ast.walk(item):
                        if not isinstance(sub, ast.Call):
                            continue
                        name = ast.unparse(sub.func).lower()
                        hit = next((d for d in deny if d in name), None)
                        if hit:
                            return (f"{node.name}.{item.name} calls {ast.unparse(sub.func)} "
                                    f"(touches the lab before GO)")
    except Exception as exc:                          # noqa: BLE001
        return f"its source could not be checked for warm start ({type(exc).__name__})"
    return ""


def _since_text(t) -> str:
    """"15:48", or "Oct 08 15:48" when not today."""
    try:
        t = float(t)
        if time.strftime("%Y%m%d", time.localtime(t)) == time.strftime("%Y%m%d"):
            return time.strftime("%H:%M", time.localtime(t))
        return time.strftime("%b %d %H:%M", time.localtime(t))
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _close(proc) -> None:
    """Close a process watch (its handle); never raises."""
    close = getattr(proc, "close", None)
    if close is not None:
        try:
            close()
        except Exception:                             # noqa: BLE001
            log.exception("Run queue: closing a process watch failed")


def _warm_text(w: Mapping, now: float) -> str:
    """"warming 4 s" / "ready" / "warm process exiting" / "GO given"."""
    if w.get("go"):
        return "GO given" if w["go"].get("delivered") else "GO decided"
    state = w.get("state")
    if state == "ready":
        return "ready"
    if state == "aborting":
        return "warm process exiting"
    try:
        return f"warming {max(0.0, now - float(w.get('spawned_at') or now)):.0f} s"
    except (TypeError, ValueError):
        return "warming"


def _warm_public(w: Mapping | None, now: float) -> dict | None:
    """A warm record as replies show it: no token (it is the warm process's
    credential), the manifest's size instead of the manifest."""
    if not w:
        return None
    out = {k: v for k, v in w.items() if k not in ("token", "manifest", "spec")}
    out["manifest_n"] = len(w.get("manifest") or ())
    out["text"] = _warm_text(w, now)
    return out


def _parse_queue_job(text) -> tuple[int | None, str]:
    """"<id>:<token>" -> (id, token); (None, "") when it is not one."""
    head, _, token = str(text or "").partition(":")
    try:
        return int(head), token
    except ValueError:
        return None, ""


def _quote(arg: str) -> str:
    return f'"{arg}"' if (" " in arg or "\t" in arg) else arg


@dataclass
class Job:
    """One job (see the module docstring).  ``to_dict()`` is its record in
    queue.json and in every reply."""

    id: int
    token: str
    path: str
    sha256: str
    label: str
    argv: list = field(default_factory=list)
    cwd: str = ""
    owner: str = "person"
    priority: int = 0
    due: float | None = None
    after: list = field(default_factory=list)
    chain: str | None = None
    stop_on_failure: bool = False
    write_back: bool | None = None
    allow_drift: bool = False
    repeat_index: int = 1
    repeat_of: int = 1
    submitted_at: float = 0.0
    submitted_by: str = ""
    state: str = "queued"
    #: why it was skipped / cancelled / failed, or "" (saved, or not ended)
    reason: str = ""
    #: the launched shell's pid and creation time (epoch s)
    pid: int | None = None
    pid_started: float | None = None
    #: the experiment process's pid and creation time, from liveOD's POLL
    #: (client_pid, client_started)
    client_pid: int | None = None
    client_started: float | None = None
    run_id: int | None = None
    log_path: str | None = None
    exit_code: int | None = None
    #: {"outcome": liveOD's outcome or None, "detail", "why", "exit_code",
    #:  "aborted", "lost_core", "told_live_od"}
    outcome: dict | None = None
    launched_at: float | None = None
    ended_at: float | None = None
    #: {"by", "at", "abort_sent": bool, "abort_note": str} once a cancel of a
    #: running job was asked for
    cancel: dict | None = None
    #: {"by", "at", "results": [kill_tree replies, each with its "role"],
    #:  "note"} after a person's force kill (the last one)
    kill: dict | None = None
    adopted: bool = False
    #: the job's place in the queue: the scheduler takes the eligible job with
    #: the lowest rank (ties: lowest id); set at submit, changed by `move`
    rank: float | None = None
    #: from the file's source at submit (ast, never imported): its experiment
    #: class ("" when not one clear class) and the keys of its
    #: self.calibrates('<key>', ...) calls -- "declared in source"
    expt_class: str = ""
    calibrates_declared: list = field(default_factory=list)
    #: who submitted it, for listings: a person's "<user>@<pc>" (`by`), an
    #: agent's label (WAXX_AGENT_LABEL) or "agent@<host>"
    submitter: str = ""
    #: this job alone paused (`edit` fields.paused): not launched until unpaused
    paused: bool = False
    paused_by: str = ""
    paused_since: float | None = None
    #: never warm-launched: asked for at submit / edit ("cold")
    cold: bool = False
    #: never warm-launched, from its source at submit (cold_reason): why
    cold_why: str = ""
    #: the warm launch while the job is queued (see the module docstring), or
    #: None; kept on the job once GO was given
    warm: dict | None = None
    #: when its (last) warm process was launched
    warm_started_at: float | None = None
    #: warm launches so far (WARM_MAX_ATTEMPTS at most)
    warm_attempts: int = 0
    #: its run had no file (``save_data=False``): liveOD's POLL showed the
    #: job's run in progress with run id 0, its last_outcome named it, or the
    #: run's output says "Acquisition for run 0 aborted".  Whatever it ended as
    no_file: bool = False

    @property
    def name(self) -> str:
        return f"job {self.id} ({self.label})"

    @property
    def queue_job(self) -> str:
        """"<id>:<token>": the job's WAXX_QUEUE_JOB, sent by its experiment at
        INIT_RUN and reported by liveOD (POLL, last_outcome) as queue_job."""
        return f"{self.id}:{self.token}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "Job":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in known})


def requester_owner(obj: Mapping) -> tuple[str, str]:
    """``(owner, "")`` from a request's required ``owner`` ("person" |
    "agent": who is asking -- the kq client sends WAXX_OWNER, the Device
    Control GUI "person"), or ``("", why it is refused)``."""
    owner = obj.get("owner")
    if owner not in OWNERS:
        return "", (f"owner is required ({' or '.join(OWNERS)}: who is asking)"
                    if owner in (None, "") else
                    f"owner must be one of {', '.join(OWNERS)}, not {owner!r}")
    return str(owner), ""


#: liveOD verdicts on a job's own run under which an agent may force-kill it.
AGENT_KILL_STATES = ("wedged", "reset_pending", "dead_client")


class QueueError(ValueError):
    """A request the queue refuses (the message says why)."""


def _checked_argv(argv) -> list:
    argv = argv or []
    if not isinstance(argv, (list, tuple)) or not all(isinstance(a, str) for a in argv):
        raise QueueError("argv must be a list of strings")
    for a in argv:
        bad = sorted(_SHELL_CHARS & set(a))
        if bad:
            raise QueueError(f"an argument contains {' '.join(bad)} (not allowed): {a}")
        if a.endswith("\\"):
            # quoted, it would escape its own closing quote
            raise QueueError(f"an argument ends in a backslash (not allowed): {a}")
    return list(argv)


def _checked_due(due):
    if due is None:
        return None
    try:
        value = float(due)
    except (TypeError, ValueError):
        raise QueueError(f"due must be epoch seconds or null, not {due!r}")
    if value != value or value in (float("inf"), float("-inf")):
        raise QueueError(f"due must be a finite time, not {due!r}")
    return value


def _checked_after(after) -> list:
    after = after or []
    if not isinstance(after, (list, tuple)):
        after = [after]
    try:
        return [int(a) for a in after]
    except (TypeError, ValueError):
        raise QueueError(f"after must be a list of job ids, not {after!r}")


def _checked_write_back(write_back):
    if write_back is True:
        raise QueueError("write_back true is not a submitter's to give: the experiment "
                         "declares its write-back; a submitter may only veto it "
                         "(write_back false)")
    if write_back not in (None, False):
        raise QueueError(f"write_back must be null or false, not {write_back!r}")
    return write_back


class RunQueue:
    """The queue.  Requests (:meth:`submit`, :meth:`cancel`, :meth:`list`,
    :meth:`describe`, :meth:`pause`, :meth:`resume`, :meth:`hold`,
    :meth:`release`) return reply dicts and may come from any thread;
    :meth:`tick` does all the work and is called by one thread (the server's
    watch).

    Inputs (callables, so it runs without a network in tests):
    ``poll()`` -> liveOD's POLL reply (raises when unreachable);
    ``run_exited(run_id, reason)`` / ``live_od_reset()`` -> liveOD's RUN_EXITED /
    RESET; ``fence()`` -> the server's run fence or None; ``monitor_state()`` ->
    the server's ``status_json`` state; ``server_busy()`` -> why something of the
    server's own holds the machine ("" when nothing); ``loops`` -> the server's
    run loops by key; ``start_monitor(why)``; ``spawn(command, cwd=, env=,
    log_path=)`` -> a process with ``pid``, ``started`` and ``poll()``
    (default: :func:`~waxx.util.device_state.detached.launch`);
    ``adopt(pid, started)`` -> such a process or None; ``kill(pid, started)``
    -> a :func:`~waxx.util.device_state.detached.kill_tree` reply (a person's
    force kill); ``journal`` (the ops
    journal); ``on_change(info)``; ``exclude``: files the queue refuses (the
    monitor experiment)."""

    def __init__(self, directory: str | None, *, poll: Callable[[], dict] | None = None,
                 run_exited: Callable[[int, str], dict] | None = None,
                 live_od_reset: Callable[[], dict] | None = None,
                 fence: Callable[[], dict | None] | None = None,
                 monitor_state: Callable[[], object] | None = None,
                 server_busy: Callable[[], str] | None = None,
                 loops: Mapping | None = None,
                 start_monitor: Callable[[str], None] | None = None,
                 hold: PersonHold | None = None, journal=None,
                 on_change: Callable[[dict], None] | None = None,
                 spawn=None, adopt=None, kill=None, exclude=(), roots=None,
                 clock: Callable[[], float] = time.time,
                 poll_every_s: float = POLL_EVERY_S, gap_s: float = GAP_S,
                 alarm_s: float = ALARM_S, outcome_wait_s: float = 0.5,
                 client_name: str | None = None, warm_ahead: bool | None = False,
                 set_priority=None, warm_deny_names=WARM_DENY_NAMES,
                 code_root: str | None = None):
        self.directory = directory
        #: warm start (see the module docstring).  A bool: that value (the
        #: class's default, off, for library use and tests).  None -- what the
        #: monitor server passes -- the server's start value:
        #: WAXX_QUEUE_WARM_AHEAD when set, else the stored setting
        #: (``<dir>/settings.json``), else WARM_AHEAD_DEFAULT (on).
        self._warm_env: bool | None = None
        if warm_ahead is None:
            self._warm_env = warm_ahead_from_env()
            stored = stored_warm_ahead(directory)
            if self._warm_env is not None:
                warm_ahead, self.warm_ahead_source = self._warm_env, "env"
            elif stored is not None:
                warm_ahead, self.warm_ahead_source = stored, "stored"
            else:
                warm_ahead, self.warm_ahead_source = WARM_AHEAD_DEFAULT, "default"
            self._warm_start_note = (stored, warm_ahead)
        else:
            self.warm_ahead_source = "given"
            self._warm_start_note = None
        self.warm_ahead = bool(warm_ahead)
        self._settings_lock = threading.Lock()
        self._warm_deny = tuple(warm_deny_names)
        if set_priority is None:
            from waxx.util.device_state.detached import set_normal_priority  # noqa: PLC0415
            set_priority = set_normal_priority
        #: set_priority(pid, started) -> bool: the server's backstop at GO
        self._set_priority = set_priority
        self._poll_fn = poll
        self._run_exited = run_exited
        self._live_od_reset = live_od_reset
        self._fence = fence
        self._monitor_state = monitor_state
        self._server_busy = server_busy
        self._loops = loops if loops is not None else {}
        self._start_monitor = start_monitor
        self.hold = hold if hold is not None else PersonHold(
            os.path.join(directory, "person_hold.json") if directory else None,
            journal=journal)
        self._journal = journal
        self._on_change = on_change
        if spawn is None:
            from waxx.util.device_state.detached import launch  # noqa: PLC0415
            spawn = launch
        if adopt is None:
            from waxx.util.device_state.detached import ProcessWatch  # noqa: PLC0415
            adopt = ProcessWatch.adopt
        if kill is None:
            from waxx.util.device_state.detached import kill_tree  # noqa: PLC0415
            kill = kill_tree
        self._spawn = spawn
        self._adopt = adopt
        self._kill = kill
        self._exclude = {self._norm(p) for p in exclude if p}
        #: the folders a job's file must be inside (default_roots())
        self.roots = list(roots) if roots is not None else default_roots()
        #: this machine's workspace (%code%): a submitter's path under ITS
        #: %code% (``client_code_root``) that does not exist here is looked up
        #: at the same place under this one
        self.code_root = code_root if code_root is not None else os.environ.get("code") or ""
        self._clock = clock
        self.poll_every_s = float(poll_every_s)
        self._gap_s = float(gap_s)
        self.alarm_s = float(alarm_s)
        self._outcome_wait_s = float(outcome_wait_s)
        self._host = client_name if client_name is not None else socket.gethostname()

        self._lock = threading.RLock()
        self._tick_lock = threading.Lock()
        self._abort_lock = threading.Lock()
        #: queue.json snapshots: the last numbered, the last written (_save)
        self._save_lock = threading.Lock()
        self._save_seq = 0
        self._saved_seq = 0
        #: an unreadable queue.json that could not be moved aside is never overwritten
        self._no_save = False
        #: the last queue.json write error; launches wait until this time after
        #: a launch whose "launching" state could not be saved
        self._save_error = ""
        self._launch_blocked_until = 0.0
        #: why no job may launch now ("" normally): the server restarting
        self._stop_launching = ""
        #: records waiting to be copied to the ops journal (flush_journal)
        self._mirror: deque = deque()
        self._mirror_lock = threading.Lock()
        self._jobs: dict[int, Job] = {}
        self._next_id = 1
        self._paused: dict[str, dict | None] = {"agent": None, "all": None}
        #: the loop the queue stopped and will start again: {"key", "path", "since"}
        self._resume_loop: dict | None = None
        #: the queue has launched a job since it last ran out: ask for the monitor then
        self._owe_monitor = False
        self._current: int | None = None
        self._procs: dict[int, object] = {}
        self._log_pos: dict[int, int] = {}
        self._log_partial: dict[int, str] = {}
        self._tails: dict[int, deque] = {}
        self._last_poll: dict | None = None
        self._last_poll_t: float | None = None
        self._last_end_t: float | None = None
        self._waiting = ""
        self._waived: tuple | None = None
        self._owed_since: float | None = None
        self._alarm: dict | None = None
        self._own_abort_ids: set = set()
        #: dependencies no longer kept in queue.json: their state from the journal
        self._dep_cache: dict[int, str] = {}
        #: (path, mtime_ns, size) -> sha256, for source_changed on listings
        self._sha_cache: dict = {}
        self._sha_lock = threading.Lock()
        #: the last gate's kind: "free" | "live" | "blocked" (see _gate)
        self._gate_kind = ""
        #: the run id of the job that ended last (its fence may still be up)
        self._last_ended_run_id: int | None = None
        #: the launch in progress: {"job", "thread", "result", "since", "warned"}
        self._spawning: dict | None = None
        #: how long _launch waits for its launch thread before the tick follows it
        self._spawn_join_s = 5.0
        #: warm processes followed: job id -> process watch (the launched
        #: shell, or the experiment's python adopted by pid + creation time)
        self._warm_procs: dict[int, object] = {}
        #: the warm launch in progress: {"job", "token", "thread", "result", "since", "slot"}
        self._warm_spawn: dict | None = None
        #: jobs whose GO is on disk (handed out only then)
        self._go_saved: set = set()
        #: job id -> the slot job's id for which its warm launch failed to
        #: start (not tried again behind that job)
        self._warm_skip: dict[int, int] = {}
        #: job id -> when its manifest was last checked
        self._warm_checked: dict[int, float] = {}
        #: the warm alarm: {"job", "since", "text"} or None
        self._warm_alarm: dict | None = None
        #: (job, token) pairs whose abort reply was logged
        self._warm_abort_logged: set = set()
        self._load()
        if self._warm_start_note is not None:
            stored, on = self._warm_start_note
            log.info("Run queue: warm start %s at start (%s).", "ON" if on else "OFF",
                     {"env": f"{WARM_AHEAD_ENV} is set", "stored": "the stored setting",
                      "default": "nothing stored: the default"}[self.warm_ahead_source])
            if self.warm_ahead_source == "env" and stored is not None and stored != on:
                log.warning("Run queue: %s overrides the stored warm start (%s) with %s.",
                            WARM_AHEAD_ENV, "on" if stored else "off", "on" if on else "off")
                self._record("run_queue_warm_ahead", on=on, was=stored, by=WARM_AHEAD_ENV,
                             owner="env", stored=stored, note="server start")
        self.flush_journal()

    # -- paths ----------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.directory)

    def _path(self, *parts) -> str:
        return os.path.join(self.directory, *parts)

    @staticmethod
    def _norm(path) -> str:
        try:
            return os.path.normcase(str(Path(path).resolve()))
        except OSError:
            return os.path.normcase(str(path))

    # -- requests -------------------------------------------------------------------

    def submit(self, obj: Mapping) -> dict:
        """``{"path", "argv", "cwd", "label", "owner", "priority", "due",
        "after", "repeat", "chain", "stop_on_failure", "write_back",
        "allow_drift", "by", "client_sha256"?, "client_host"?}`` -> ``{"status":
        "ok", "jobs": [...], "ids": [...]}``.  The submitter's own copy must
        match this machine's (:meth:`_check_client_copy`)."""
        try:
            jobs, clamped = self._new_jobs(obj)
        except QueueError as exc:
            self._record("run_queue_refused", what="submit", msg=str(exc),
                         path=str(obj.get("path") or ""), by=str(obj.get("by") or ""))
            log.warning("Run queue: submit refused: %s", exc)
            self.flush_journal()
            return {"status": "error", "msg": str(exc)}
        for job in jobs:
            log.info("Run queue: %s submitted by %s (%s, owner %s, priority %d%s).", job.name,
                     job.submitted_by or "?", job.path, job.owner, job.priority,
                     f", due {_clock_text(job.due)}" if job.due else "")
            self._record("run_queue_submit", job=job.id, label=job.label, path=job.path,
                         sha256=job.sha256, argv=job.argv, cwd=job.cwd, owner=job.owner,
                         priority=job.priority, due=job.due, after=job.after, chain=job.chain,
                         stop_on_failure=job.stop_on_failure, write_back=job.write_back,
                         allow_drift=job.allow_drift, repeat_index=job.repeat_index,
                         repeat_of=job.repeat_of, by=job.submitted_by)
        self._save()
        self._notify()
        # the kq client (run_queue_client.py) follows the job by `tail` from here
        return {"status": "ok", "ids": [j.id for j in jobs], "clamped": clamped,
                "jobs": [j.to_dict() for j in jobs]}

    def _new_jobs(self, obj: Mapping) -> list[Job]:
        if not self.enabled:
            raise QueueError("this monitor server has no run queue folder (no journal_dir / "
                             "run_queue_dir was given): jobs cannot be logged")
        text = str(obj.get("path") or "").strip()
        if not text:
            raise QueueError("no experiment file (path) was given")
        bad = sorted(_SHELL_CHARS & set(text))
        if bad:
            raise QueueError(f"the file's path contains {' '.join(bad)} (not allowed): {text}")
        path = Path(text)
        if not path.is_absolute():
            raise QueueError(f"the path must be absolute (the server's machine): {text}")
        path = self._mapped_path(path.resolve(), obj)
        bad = sorted(_SHELL_CHARS & set(str(path)))
        if bad:
            # a link resolving to such a path: the command line would carry it
            raise QueueError(f"the file's resolved path contains {' '.join(bad)} (not "
                             f"allowed): {path}")
        roots = [Path(r).resolve() for r in self.roots]
        if not any(path == r or r in path.parents for r in roots):
            raise QueueError(f"{path} is outside the folders the queue runs files from ("
                             + ", ".join(str(r) for r in roots) + ")")
        if path.suffix.lower() != ".py":
            raise QueueError(f"not a Python file: {path}")
        if not path.is_file():
            host = str(obj.get("client_host") or "").strip()
            if host and host.lower() != str(self._host).lower():
                raise QueueError(f"no such file on {self._host}: {path} (the file that runs is "
                                 f"{self._host}'s copy: put it in the workspace on both PCs, "
                                 f"or copy it to {self._host} and submit that path)")
            raise QueueError(f"no such file on the monitor server's machine: {path}")
        if self._norm(path) in self._exclude:
            raise QueueError(f"{path.name} is the monitor's own experiment: the monitor server "
                             "runs it itself")
        argv = _checked_argv(obj.get("argv"))
        cwd = str(obj.get("cwd") or path.parent)
        if not Path(cwd).is_dir():
            raise QueueError(f"no such working folder on the monitor server's machine: {cwd}")
        owner = str(obj.get("owner") or "person")
        if owner not in OWNERS:
            raise QueueError(f"owner must be one of {', '.join(OWNERS)}, not {owner!r}")
        try:
            # a placement hint only (higher first within its owner's block at
            # submit), kept for display; the order is the rank
            priority = int(obj.get("priority") or 0)
        except (TypeError, ValueError):
            raise QueueError(f"priority must be an integer, not {obj.get('priority')!r}")
        due = _checked_due(obj.get("due"))
        after = _checked_after(obj.get("after"))
        try:
            repeat = 1 if obj.get("repeat") is None else int(obj.get("repeat"))
        except (TypeError, ValueError):
            raise QueueError(f"repeat must be an integer, not {obj.get('repeat')!r}")
        if not 1 <= repeat <= MAX_REPEAT:
            raise QueueError(f"repeat must be 1 to {MAX_REPEAT}, not {repeat}")
        chain = obj.get("chain")
        chain = str(chain).strip() if chain not in (None, "") else None
        write_back = _checked_write_back(obj.get("write_back"))
        label = _LABEL_RE.sub("_", str(obj.get("label") or path.stem)).strip("_") or path.stem
        label = label[:60]
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "")
        if owner == "agent":
            submitter = str(obj.get("agent_label") or "").strip() or \
                f"agent@{obj.get('host') or obj.get('client') or '?'}"
        else:
            submitter = by or "?"
        sha = file_sha256(path)
        self._check_client_copy(obj, path, sha)
        expt_class, calibrates = describe_source(path)
        cold = obj.get("cold")
        if cold not in (None, True, False):
            raise QueueError(f"cold must be true or false, not {cold!r}")
        cold_why = cold_reason(path, self._warm_deny)
        # TODO(INIT_RUN): the live list of calibrated keys, sent by the
        # experiment at prepare, overrides calibrates_declared once known.
        with self._lock:
            unknown = [a for a in after if a not in self._jobs
                       and self._dep_state(a) == "failed" and a >= self._next_id]
            if unknown:
                raise QueueError(f"after names unknown job(s): {', '.join(map(str, unknown))}")
            first = self._next_id
            if chain is None and repeat > 1:
                chain = f"repeat-{first}"
            stop = obj.get("stop_on_failure")
            stop = bool(chain) if stop is None else bool(stop)
            ranks, clamped = self._place(repeat, owner, priority, obj)
            now = self._clock()
            jobs = []
            for i in range(repeat):
                job = Job(id=self._next_id, token=secrets.token_hex(3), path=str(path),
                          sha256=sha, label=label, argv=list(argv), cwd=cwd, owner=owner,
                          priority=priority, due=due, after=list(after), chain=chain,
                          stop_on_failure=stop, write_back=write_back,
                          allow_drift=bool(obj.get("allow_drift")), repeat_index=i + 1,
                          repeat_of=repeat, submitted_at=now, submitted_by=by,
                          rank=ranks[i], expt_class=expt_class,
                          calibrates_declared=list(calibrates), submitter=submitter,
                          cold=bool(cold), cold_why=cold_why)
                self._next_id += 1
                self._jobs[job.id] = job
                jobs.append(job)
        return jobs, clamped

    def _mapped_path(self, path: Path, obj: Mapping) -> Path:
        """``path`` as given when it exists here; else, when it lies under the
        submitter's workspace (``client_code_root``, its %code%), the same
        place under this machine's (:attr:`code_root`).  The submitter's copy
        is still checked against the one that runs (:meth:`_check_client_copy`).
        Without a usable ``client_code_root`` (an older client, %code% unset
        there), the longest tail of ``path`` that starts with a top-level
        folder of this workspace (``k-exp\\...``) is taken -- only when the
        submitter's ``client_sha256`` matches the file there."""
        if path.exists() or not self.code_root:
            return path
        mine = Path(self.code_root)
        theirs = str(obj.get("client_code_root") or "").strip()
        if theirs:
            try:
                return (mine / path.relative_to(Path(theirs).resolve())).resolve()
            except ValueError:
                pass
        sha = str(obj.get("client_sha256") or "").lower()
        if not sha:
            return path
        parts = path.parts
        for i in range(1, len(parts) - 1):
            if not (mine / parts[i]).is_dir():
                continue
            candidate = mine.joinpath(*parts[i:]).resolve()
            if candidate.is_file() and file_sha256(candidate) == sha:
                return candidate
        return path

    def _check_client_copy(self, obj: Mapping, path: Path, sha: str) -> None:
        """The file that runs is this machine's copy, so it must be the one
        the submitter means: a ``client_sha256`` (the client's hash of its own
        file at that path) must equal this machine's; a client on another host
        (``client_host``) that could not hash its file is refused, since
        nothing shows the two copies agree.  A request without either (an
        older client) is taken as before."""
        theirs = obj.get("client_sha256")
        if theirs:
            if str(theirs).lower() != sha:
                raise QueueError(f"your copy differs from {self._host}'s: {path} (the file "
                                 f"that runs is {self._host}'s; it must match yours)")
            return
        host = str(obj.get("client_host") or "").strip()
        if host and host.lower() != str(self._host).lower():
            raise QueueError(f"{path}: the file that runs is {self._host}'s, and your copy on "
                             f"{host} could not be read to check that it matches")

    # -- the order ---------------------------------------------------------------------

    def _queued_in_order(self, without: Job | None = None) -> list[Job]:
        """The queued jobs in rank order (ties: id)."""
        return sorted((j for j in self._jobs.values()
                       if j.state == "queued" and j is not without),
                      key=lambda j: (j.rank, j.id))

    def _insert_index(self, queued: list[Job], owner: str, priority: int, obj: Mapping,
                      asker: str | None = None) -> tuple[int, bool]:
        """``(index, clamped)``: :meth:`_requested_index`, except that an agent
        (``asker``; the job's ``owner`` when not given) never places a job
        ahead of a queued person job -- the index is moved to just after the
        last one, and ``clamped`` says so.  A person is not restricted."""
        index = self._requested_index(queued, owner, priority, obj)
        if (asker or owner) != "agent":
            return index, False
        floor = max((i + 1 for i, j in enumerate(queued) if j.owner == "person"), default=0)
        if index < floor:
            return floor, True
        return index, False

    def _requested_index(self, queued: list[Job], owner: str, priority: int,
                         obj: Mapping) -> int:
        """Where a job goes among ``queued`` (rank order): the position the
        request names -- ``before_id`` / ``after_id`` (a queued job),
        ``at_index`` / ``to_index`` (0-based; past the end means the end;
        negative is refused), ``at_end`` -- or, by
        default, the owner rule: a person's job ahead of every queued agent
        job (after the person jobs of equal or higher priority), an agent's
        job at the end of the agent jobs of equal or higher priority."""
        given = [k for k in ("before_id", "after_id", "at_index", "to_index")
                 if obj.get(k) is not None]
        if obj.get("at_end"):
            given.append("at_end")
        if len(given) > 1:
            raise QueueError(f"give one position, not {', '.join(given)}")
        ids = [j.id for j in queued]
        for key, offset in (("before_id", 0), ("after_id", 1)):
            if obj.get(key) is not None:
                try:
                    target = int(obj[key])
                except (TypeError, ValueError):
                    raise QueueError(f"{key} must be a job id, not {obj[key]!r}")
                if target not in ids:
                    raise QueueError(f"{key} {target} is not a queued job")
                return ids.index(target) + offset
        for key in ("at_index", "to_index"):
            if obj.get(key) is not None:
                try:
                    index = int(obj[key])
                except (TypeError, ValueError):
                    raise QueueError(f"{key} must be an integer, not {obj[key]!r}")
                if index < 0:
                    raise QueueError(f"{key} must be 0 or more, not {index}")
                return min(index, len(queued))
        if obj.get("at_end"):
            return len(queued)
        for i, other in enumerate(queued):
            if owner == "person" and (other.owner == "agent" or other.priority < priority):
                return i
            if owner == "agent" and other.owner == "agent" and other.priority < priority:
                return i
        return len(queued)

    def _ranks_at(self, queued: list[Job], index: int, n: int) -> list[float]:
        """``n`` ranks between ``queued[index - 1]`` and ``queued[index]``
        (the queued jobs are given new ranks 1, 2, 3... first when the gap is
        too small to split)."""
        def bounds():
            lo = queued[index - 1].rank if index > 0 else None
            hi = queued[index].rank if index < len(queued) else None
            return lo, hi
        lo, hi = bounds()
        if lo is not None and hi is not None and (hi - lo) / (n + 1) < 1e-6:
            for k, other in enumerate(queued):
                other.rank = float(k + 1)
            lo, hi = bounds()
        if lo is None and hi is None:
            top = max((j.rank for j in self._jobs.values() if j.rank is not None),
                      default=0.0)
            return [top + 1.0 + k for k in range(n)]
        if hi is None:
            return [lo + 1.0 + k for k in range(n)]
        if lo is None:
            return [hi - n + k for k in range(n)]
        step = (hi - lo) / (n + 1)
        return [lo + step * (k + 1) for k in range(n)]

    def _place(self, n: int, owner: str, priority: int,
               obj: Mapping) -> tuple[list[float], bool]:
        queued = self._queued_in_order()
        index, clamped = self._insert_index(queued, owner, priority, obj)
        return self._ranks_at(queued, index, n), clamped

    def _positions(self) -> dict[int, int]:
        """Job id -> 0-based position among the queued jobs (rank order)."""
        return {j.id: i for i, j in enumerate(self._queued_in_order())}

    def _medians(self) -> dict:
        """path -> (median launch-to-end seconds, n) over the last 5 saved runs
        of each file (jobs the queue still keeps); one pass over the jobs."""
        by_path: dict = {}
        for j in self._jobs.values():
            if j.state in SUCCEEDED and j.launched_at and j.ended_at:
                by_path.setdefault(j.path, []).append(j)
        out = {}
        for path, runs in by_path.items():
            runs = sorted(runs, key=lambda j: j.ended_at)[-5:]
            times = sorted(j.ended_at - j.launched_at for j in runs)
            mid = len(times) // 2
            out[path] = ((times[mid] if len(times) % 2 else (times[mid - 1] + times[mid]) / 2),
                         len(runs))
        return out

    def _duration_estimate(self, job: Job, medians: dict | None = None
                           ) -> tuple[float | None, str]:
        """(seconds, basis): the median launch-to-end time of the last 5 saved
        runs of the same file (jobs the queue still keeps), or (None, why).
        ``medians``: :meth:`_medians`, when the caller has it already."""
        found = (medians if medians is not None else self._medians()).get(job.path)
        if found is None:
            return None, "no saved run of this file to go by"
        return found[0], f"median of the last {found[1]} saved run(s) of this file"

    def _running_end(self, job: Job, now: float, medians: dict | None = None
                     ) -> tuple[float | None, str]:
        """(epoch, basis): when the job in the slot should end -- from liveOD's
        shots (expected x measured period) once it has taken one, else its
        launch time + the duration estimate."""
        poll = self._last_poll
        if self._is_job_run(job, poll):
            n = poll.get("n_shots") or 0
            want = poll.get("n_shots_expected")
            init_age, last_age = poll.get("init_run_age_s"), poll.get("last_shot_age_s")
            if n and want and init_age is not None and last_age is not None and init_age >= last_age:
                period = (init_age - last_age) / n
                end = now - last_age + max(0, int(want) - int(n)) * period
                return end, (f"liveOD: shot {n} of {want}, {period:.1f} s per shot (the save "
                             "is not included)")
        dur, basis = self._duration_estimate(job, medians)
        if dur is None or not job.launched_at:
            return None, basis
        return job.launched_at + dur, basis

    @staticmethod
    def _stat_key(path: str):
        """(path, mtime_ns, size), or None when the file cannot be read."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (path, st.st_mtime_ns, st.st_size)

    def _fill_source_changed(self, pending) -> None:
        """Called OUTSIDE the lock: for each (view, key, sha at submit), set
        the view's source_changed -- hashing a file again only when its size
        or mtime moved since it was last hashed."""
        for view, key, submitted in pending:
            if key is None:
                view["source_changed"] = None
                continue
            with self._sha_lock:
                sha = self._sha_cache.get(key)
            if sha is None:
                try:
                    sha = file_sha256(key[0])
                except OSError:
                    view["source_changed"] = None
                    continue
                with self._sha_lock:
                    self._sha_cache = {k: v for k, v in self._sha_cache.items()
                                       if k[0] != key[0]}
                    self._sha_cache[key] = sha
            view["source_changed"] = sha != submitted

    def _source_changed(self, job: Job) -> bool | None:
        """Whether a queued job's file differs from the one submitted (None
        when it cannot be read) -- for one job, outside the lock."""
        view: dict = {}
        self._fill_source_changed([(view, self._stat_key(job.path), job.sha256)])
        return view["source_changed"]

    def _views(self, jobs: list[Job], now: float) -> tuple[list[dict], list]:
        """Each job as list/describe show it: its record plus ``position``,
        ``waiting``, ``source_changed`` (queued jobs) and ``estimate`` --
        ``{duration_s, eta_start, eta_end, basis}``, all ESTIMATES: the
        duration from the last saved runs of the same file; for the job in the
        slot its expected end; for a queued job its expected start (now + the
        slot's remaining time + the durations of the ELIGIBLE queued jobs ahead
        of it -- blocked ones (held, paused, due later, waiting) are left out
        and the basis says so; null when any counted duration is unknown).

        Called under the lock; one pass: the order, the positions and the
        per-file medians are computed once.  Returns ``(views, pending)``:
        ``source_changed`` is filled by :meth:`_fill_source_changed` with
        ``pending`` AFTER the lock is released (it may hash files)."""
        queued = self._queued_in_order()
        positions = {j.id: i for i, j in enumerate(queued)}
        order = self._order(now)
        medians = self._medians()
        cur = self._jobs.get(self._current) if self._current is not None else None
        ahead_s: float | None = 0.0
        if cur is not None:
            end, _ = self._running_end(cur, now, medians)
            ahead_s = None if end is None else max(0.0, end - now)
        start_of: dict[int, float | None] = {}
        skipped_ahead: dict[int, bool] = {}
        eligible = {j.id for j in order}
        skipped = False
        for q in queued:
            start_of[q.id] = None if ahead_s is None else now + ahead_s
            skipped_ahead[q.id] = skipped
            if q.id not in eligible:
                # held, paused, due later, waiting on another job: it is not
                # expected to run before the jobs behind it -- left out
                skipped = True
                continue
            dur, _ = self._duration_estimate(q, medians)
            ahead_s = None if (ahead_s is None or dur is None) else ahead_s + dur
        out, pending = [], []
        for job in jobs:
            dur, basis = self._duration_estimate(job, medians)
            estimate = {"duration_s": dur, "eta_start": None, "eta_end": None, "basis": basis}
            view = dict(self._public(job, now), position=positions.get(job.id),
                        waiting=self._why_waiting(job, now, order), source_changed=None,
                        estimate=estimate)
            if job.state == "queued":
                pending.append((view, self._stat_key(job.path), job.sha256))
                estimate["eta_start"] = start_of.get(job.id)
                if estimate["eta_start"] is not None and dur is not None:
                    estimate["eta_end"] = estimate["eta_start"] + dur
                if skipped_ahead.get(job.id):
                    estimate["basis"] += "; est. (ignores blocked jobs ahead)"
            elif job.state in IN_SLOT:
                estimate["eta_end"], estimate["basis"] = self._running_end(job, now, medians)
            if job.warm is not None or job.warm_started_at is not None:
                # a warm job's launch time is its GO: the time it spent warming
                # before (behind the previous job) is not in its duration
                estimate["basis"] += "; GO to end, warm time excluded"
            out.append(view)
        return out, pending

    @staticmethod
    def _row(view: dict) -> dict:
        """The compact listing (kq list's columns)."""
        return {"position": view.get("position"), "id": view["id"], "state": view["state"],
                "owner": view["owner"], "submitter": view.get("submitter") or "",
                "label": view["label"], "expt_class": view.get("expt_class") or "",
                "run_id": view.get("run_id"), "est": view.get("estimate"),
                "waiting": view.get("waiting") or "",
                "warm": ((view.get("warm") or {}).get("text") or "")}

    #: the fields `edit` may change on a queued job
    EDITABLE = ("argv", "label", "after", "chain", "stop_on_failure", "write_back", "due",
                "allow_drift", "paused", "cold")

    def edit(self, obj: Mapping) -> dict:
        """``{"id", "owner", "by", "fields": {argv?, label?, after?, chain?,
        stop_on_failure?, write_back?, due?, allow_drift?, paused?, cold?}}``: change
        a QUEUED job (launching / running refused); each field checked as at
        submit; ``paused`` pauses this job alone.  An agent may edit only
        agent jobs.  Journaled (run_queue_edit) with every changed field's
        value before and after.  The drift check at launch is unchanged."""
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        as_owner, refusal = requester_owner(obj)
        if refusal:
            return {"status": "error", "msg": refusal}
        fields_ = obj.get("fields")
        if not isinstance(fields_, Mapping) or not fields_:
            return {"status": "error", "msg": "edit needs fields: {" +
                    ", ".join(self.EDITABLE) + "}"}
        unknown = sorted(set(fields_) - set(self.EDITABLE))
        if unknown:
            return {"status": "error", "msg": f"not editable: {', '.join(unknown)} (editable: "
                                              f"{', '.join(self.EDITABLE)})"}
        with self._lock:
            try:
                job = self._find(obj)
                if job.state != "queued":
                    raise QueueError(f"{job.name} is {job.state}: only queued jobs are edited")
                if as_owner == "agent" and job.owner != "agent":
                    raise QueueError(f"{job.name} is a person's job: an agent may not edit it")
                new = self._checked_edit(job, fields_)
            except QueueError as exc:
                return {"status": "error", "msg": str(exc)}
            before, after = {}, {}
            for key, value in new.items():
                old = getattr(job, key)
                if old == value:
                    continue
                before[key], after[key] = old, value
                setattr(job, key, value)
            if "paused" in after:
                job.paused_by = by if job.paused else ""
                job.paused_since = self._clock() if job.paused else None
            views, pending = self._views([job], self._clock())
            view = views[0]
        self._fill_source_changed(pending)
        if after:
            log.info("Run queue: %s edited by %s: %s", job.name, by,
                     ", ".join(f"{k} {before[k]!r} -> {after[k]!r}" for k in after))
            self._record("run_queue_edit", job=job.id, by=by, owner=as_owner, before=before,
                         after=after)
            self._save()
        self._notify()
        return {"status": "ok", "job": view, "changed": sorted(after)}

    def _checked_edit(self, job: Job, fields_: Mapping) -> dict:
        """The edit's new values, each checked as at submit (QueueError)."""
        new = {}
        if "argv" in fields_:
            new["argv"] = _checked_argv(fields_["argv"])
        if "label" in fields_:
            label = _LABEL_RE.sub("_", str(fields_["label"] or "")).strip("_")[:60]
            if not label:
                raise QueueError("label is empty")
            new["label"] = label
        if "after" in fields_:
            after = _checked_after(fields_["after"])
            if job.id in after:
                raise QueueError(f"{job.name} cannot wait for itself")
            unknown = [a for a in after if a not in self._jobs
                       and self._dep_state(a) == "failed" and a >= self._next_id]
            if unknown:
                raise QueueError(f"after names unknown job(s): {', '.join(map(str, unknown))}")
            if self._waits_on(after, job.id):
                raise QueueError(f"after {after} would make a cycle back to {job.name}")
            new["after"] = after
        if "chain" in fields_:
            chain = fields_["chain"]
            new["chain"] = str(chain).strip() if chain not in (None, "") else None
        for key in ("stop_on_failure", "allow_drift", "paused", "cold"):
            if key in fields_:
                if not isinstance(fields_[key], bool):
                    raise QueueError(f"{key} must be true or false, not {fields_[key]!r}")
                new[key] = fields_[key]
        if "write_back" in fields_:
            new["write_back"] = _checked_write_back(fields_["write_back"])
        if "due" in fields_:
            new["due"] = _checked_due(fields_["due"])
        return new

    def _waits_on(self, after, target: int) -> bool:
        """Whether any job in ``after`` (transitively, through its own
        `after`) waits for job ``target``."""
        seen, todo = set(), list(after)
        while todo:
            jid = todo.pop()
            if jid == target:
                return True
            if jid in seen:
                continue
            seen.add(jid)
            dep = self._jobs.get(jid)
            if dep is not None:
                todo.extend(dep.after)
        return False

    def insert(self, obj: Mapping) -> dict:
        """``submit`` at a position the request names (``at_index`` |
        ``before_id`` | ``after_id``; one is required)."""
        if all(obj.get(k) is None for k in ("at_index", "before_id", "after_id")):
            return {"status": "error",
                    "msg": "insert needs a position: at_index, before_id or after_id"}
        return self.submit(obj)

    def move(self, obj: Mapping) -> dict:
        """``{"id", "to_index" | "before_id" | "after_id", "owner", "by"}``: put
        a queued job at another place in the order (launching / running jobs
        never move).  An agent may move only agent jobs."""
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        as_owner, refusal = requester_owner(obj)
        if refusal:
            return {"status": "error", "msg": refusal}
        if all(obj.get(k) is None for k in ("to_index", "before_id", "after_id")):
            return {"status": "error", "msg": "move needs to_index, before_id or after_id"}
        with self._lock:
            try:
                job = self._find(obj)
                if job.state != "queued":
                    raise QueueError(f"{job.name} is {job.state}: only queued jobs move")
                if as_owner == "agent" and job.owner != "agent":
                    raise QueueError(f"{job.name} is a person's job: an agent may not move it")
                if job.id in (obj.get("before_id"), obj.get("after_id")):
                    raise QueueError(f"{job.name} cannot be placed next to itself")
                before = self._positions().get(job.id)
                others = self._queued_in_order(without=job)
                index, clamped = self._insert_index(others, job.owner, job.priority, obj,
                                                    asker=as_owner)
                old_rank = job.rank
                job.rank = self._ranks_at(others, index, 1)[0]
                after = self._positions().get(job.id)
            except QueueError as exc:
                return {"status": "error", "msg": str(exc)}
        log.info("Run queue: %s moved by %s from position %s to %s.", job.name, by, before, after)
        self._record("run_queue_move", job=job.id, by=by, owner=as_owner, from_position=before,
                     to_position=after, from_rank=old_rank, to_rank=job.rank)
        self._save()
        self._notify()
        return {"status": "ok", "job": dict(self._public(job), position=after), "position": after,
                "clamped": clamped}

    def _find(self, obj: Mapping) -> Job:
        try:
            job_id = int(obj.get("id"))
        except (TypeError, ValueError):
            raise QueueError(f"no job id given (id: {obj.get('id')!r})")
        job = self._jobs.get(job_id)
        if job is None:
            raise QueueError(f"no job {job_id} in the run queue")
        token = obj.get("token")
        if token and token != job.token:
            raise QueueError(f"job {job_id} has token {job.token}, not {token} (a different "
                             "queue's job?)")
        return job

    def cancel(self, obj: Mapping) -> dict:
        """``{"id", "token"?, "by", "owner", "queued_only"?}``: a queued job is
        cancelled; a launching or running one gets liveOD's Abort for its run
        (never a kill) and ends when its process does.  ``owner`` -- who asks,
        "person" or "agent" -- is required: an agent may not cancel a person's
        job.  ``queued_only``: refuse unless the job is still queued (a
        client's cancel meant for a queued job must not abort one that has
        just launched)."""
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        as_owner, refusal = requester_owner(obj)
        if refusal:
            return {"status": "error", "msg": refusal}
        with self._lock:
            try:
                job = self._find(obj)
            except QueueError as exc:
                return {"status": "error", "msg": str(exc)}
            if job.owner == "person" and as_owner == "agent" and job.state not in ENDED:
                return {"status": "error",
                        "msg": f"{job.name} is a person's job: an agent may not cancel it"}
            if obj.get("queued_only") and job.state != "queued":
                return {"status": "error", "state": job.state,
                        "msg": f"{job.name} is {job.state}, not queued: not cancelled"}
            if job.state == "queued":
                self._end(job, "cancelled", f"cancelled by {by}")
                reply = {"status": "ok", "job": self._public(job)}
                if job.warm is not None and not job.warm.get("go"):
                    self._mark_aborting(job, f"cancelled by {by}")
                    reply["note"] = "not started (warm process told to exit)"
            elif job.state in ("launching", "running"):
                if job.cancel is None:
                    job.cancel = {"by": by, "at": self._clock(), "abort_sent": False,
                                  "abort_note": ""}
                    self._record("run_queue_cancel_requested", job=job.id, by=by,
                                 run_id=job.run_id)
                    log.warning("Run queue: cancel of running %s asked by %s: liveOD's Abort "
                                "goes to run %s (its file is discarded, as for any Abort); "
                                "the job ends when its process does.", job.name, by,
                                job.run_id if job.run_id is not None else
                                "0 (save_data=False: no file)" if job.no_file else
                                "(no run id yet)")
                # the tick sends liveOD's Abort (and retries): a request never
                # waits on liveOD
                reply = {"status": "ok", "job": self._public(job), "pending": True}
                go = (job.warm or {}).get("go")
                if go and not go.get("delivered"):
                    # GO decided, not handed out yet: the warm process is told
                    # to exit at its next poll, nothing of it ran
                    reply["note"] = "not started (warm process told to exit)"
            else:
                return {"status": "error", "msg": f"{job.name} is already {job.state}"}
        self._save()
        self._notify()
        return reply

    def _agent_kill_refusal(self, job: Job) -> str:
        """Why an agent may not force-kill ``job`` ("" when it may): only an
        agent's job, and only while liveOD's verdict on that job's own run is
        one of :data:`AGENT_KILL_STATES` (no cleanup runs, so a run that can
        still end by itself keeps its Abort)."""
        if job.owner != "agent":
            return ("an agent may force-kill only an agent's job (no cleanup runs: the hardware "
                    f"stays as the run left it); {job.name} is a {job.owner}'s: ask a person")
        poll = self._poll_now()
        if poll is None:
            return "liveOD is not reachable: an agent's force-kill needs its verdict on the run"
        fence = self._fence() if self._fence is not None else None
        monitor = self._monitor_state() if self._monitor_state is not None else None
        verdict = run_gate.classify(poll, fence, monitor_state=monitor)
        if verdict.run_id != job.run_id or verdict.state not in AGENT_KILL_STATES:
            return (f"an agent may force-kill its job only when its run is stuck ("
                    f"{', '.join(AGENT_KILL_STATES)}); run {job.run_id}: liveOD says "
                    f"{verdict.state} for run {verdict.run_id}. Try kq cancel first")
        return ""

    def kill(self, obj: Mapping) -> dict:
        """``{"id", "token"?, "by", "owner"}``: force-kill a launching or
        running job (see **Kill** in the module docstring).  A person may
        (``owner`` "person"), and an agent for its own stuck job
        (:meth:`_agent_kill_refusal`).  Kills, in order, the experiment's process
        (``client_pid``, only with a recorded creation time that still
        matches) and the process the queue follows (the shell; the queue's
        handle on it keeps its pid from being reused), each with its tree.
        Never touches liveOD: the job ends on the queue's next tick, which
        tells liveOD the run exited.  -> ``{"status", "job", "results",
        "pending", "note"}``; ``status`` "error" (with ``results``) when a
        process of it may still be alive."""
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        as_owner, refusal = requester_owner(obj)
        if refusal:
            return {"status": "error", "msg": refusal}
        if as_owner != "person":
            with self._lock:
                try:
                    asked = self._find(obj)
                except QueueError as exc:
                    return {"status": "error", "msg": str(exc)}
            refusal = self._agent_kill_refusal(asked)      # outside the lock: it polls liveOD
            if refusal:
                return {"status": "error", "msg": refusal}
        with self._lock:
            try:
                job = self._find(obj)
            except QueueError as exc:
                return {"status": "error", "msg": str(exc)}
            if job.state not in ("launching", "running"):
                return {"status": "error", "state": job.state,
                        "msg": (f"{job.name} is queued: remove it instead" if job.state == "queued"
                                else f"{job.name} is {job.state}: its process has exited"
                                if job.state == "ending" else
                                f"{job.name} is already {job.state}")}
            proc = self._procs.get(job.id)
            targets, unverified, refused = [], [], []
            if job.client_pid and job.client_started is not None and (
                    proc is None or int(job.client_pid) != int(proc.pid)):
                targets.append(("experiment", job.client_pid, job.client_started))
            if proc is not None and proc.poll() is None:
                # the queue holds the handle (or, without one, checks the time)
                vouch = getattr(proc, "_handle", None) is not None
                role = ("experiment" if job.client_pid and int(job.client_pid) == int(proc.pid)
                        else "shell")
                started = None if vouch else (
                    getattr(proc, "started", None)
                    or (job.client_started if role == "experiment" else job.pid_started))
                if vouch or started is not None:
                    targets.append((role, proc.pid, started))
                else:
                    refused.append({"pid": proc.pid, "role": role, "ok": False, "gone": False,
                                    "note": "not killed: no handle on it and no creation time "
                                            "to check it by (its pid may have been reused)"})
            if job.client_pid is not None and job.client_started is None:
                unverified.append(f"experiment pid {job.client_pid} not killed on its own: its "
                                  "creation time was never recorded (it goes with the shell's "
                                  "tree)")
            skipped = "; ".join(unverified)
        log.warning("Run queue: FORCE KILL of %s (run %s) asked by %s: %s.", job.name,
                    job.run_id if job.run_id is not None else "none",
                    by, ", ".join(f"{role} pid {pid}" for role, pid, _ in targets)
                    or "no live process known")
        results = []
        for role, pid, started in targets:
            try:
                r = dict(self._kill(pid, started))
            except Exception as exc:                  # noqa: BLE001
                r = {"pid": pid, "ok": False, "gone": False, "note": f"kill failed: {exc!r}"}
            r["role"] = role
            results.append(r)
            (log.warning if r.get("ok") or r.get("gone") else log.error)(
                "Run queue: kill %s pid %s of %s: %s", role, pid, job.name, r.get("note"))
        results += refused
        alive_left = any(not (r.get("ok") or r.get("gone")) for r in results)
        killed = any(r.get("ok") for r in results)
        note = ("killed" if killed and not alive_left else
                "NOT all killed: " + "; ".join(f"{r['role']} pid {r['pid']}: {r.get('note')}"
                                               for r in results
                                               if not (r.get("ok") or r.get("gone")))
                if alive_left else "no live process of it was found")
        if skipped:
            note += f" ({skipped})"
        with self._lock:
            job.kill = {"by": by, "at": self._clock(), "results": results, "note": note}
            if job.state == "launching" and not alive_left:
                # nothing to wait for: no launch answer is needed any more, and
                # the tick ends it as a run whose process is gone
                if self._spawning is not None and self._spawning.get("job") == job.id:
                    self._abandon_spawn()
                job.state = "running"
                self._procs[job.id] = None
        self._record("run_queue_kill", job=job.id, by=by, run_id=job.run_id,
                     results=results, note=note)
        self._save()
        self._notify()
        reply = {"status": "ok" if not alive_left else "error", "job": self._public(job),
                 "results": results, "pending": True, "note": note}
        if alive_left:
            reply["msg"] = f"{job.name}: {note}"
        return reply

    def list(self, obj: Mapping | None = None) -> dict:
        """``{"states"?: [...], "limit"?: n}`` -> the jobs: the ended ones (by
        id, the last ``limit``), then the one in the slot, then the queued ones
        in rank order (the first ``limit``; ``truncated`` true when there were
        more, ``queued_total`` says how many) -- each with ``position`` (0-based
        among the queued jobs; None for the others); ``next``: the eligible
        jobs in launch order."""
        obj = obj or {}
        states = obj.get("states")
        try:
            limit = int(obj.get("limit") or 200)
        except (TypeError, ValueError):
            limit = 200
        limit = max(1, limit)
        with self._lock:
            ended = sorted((j for j in self._jobs.values() if j.state in ENDED),
                           key=lambda j: j.id)[-limit:]
            slot = sorted((j for j in self._jobs.values() if j.state in IN_SLOT),
                          key=lambda j: j.id)
            queued = self._queued_in_order()
            truncated = len(queued) > limit
            jobs = [j for j in ended + slot + queued[:limit]
                    if not states or j.state in states]
            now = self._clock()
            out, pending = self._views(jobs, now)
            order = [j.id for j in self._order(now)]
        self._fill_source_changed(pending)
        return {"status": "ok", "jobs": out, "rows": [self._row(v) for v in out],
                "next": order, "truncated": truncated, "queued_total": len(queued),
                "run_queue": self.info()}

    def describe(self, obj: Mapping) -> dict:
        """``{"id", "token"?}`` -> the job, why it waits, its last lines."""
        with self._lock:
            try:
                job = self._find(obj)
            except QueueError as exc:
                return {"status": "error", "msg": str(exc)}
            views, pending = self._views([job], self._clock())
            view = views[0]
            out = {"status": "ok", "job": view, "waiting": view["waiting"],
                   "tail": list(self._tails.get(job.id, ()))}
        self._fill_source_changed(pending)
        return out

    def tail(self, obj: Mapping) -> dict:
        """``{"id", "token", "offset"}`` -> ``{"lines", "offset", "done",
        "state", "run_id"}``.  ``token`` is required when the job has one (a
        follower must not read another queue's job that reused the id; the
        client takes it from submit, or once from ``describe``).  The job's
        log file is read on this machine from byte
        ``offset`` (so a client on any PC follows a log that lives on the
        server's disk), at most :data:`TAIL_CHUNK` bytes, whole lines only --
        the last, unterminated line comes once the job has ended (or alone
        when one line is longer than a chunk).  Lines are decoded as UTF-8
        with errors replaced; ``offset`` is the next byte to ask for.
        ``done``: the job has ended and its log is read to the end.  A queued
        job has no log yet (no lines, offset unchanged)."""
        try:
            offset = int(obj.get("offset") or 0)
        except (TypeError, ValueError):
            return {"status": "error", "msg": f"offset must be an integer, not "
                                              f"{obj.get('offset')!r}"}
        if offset < 0:
            return {"status": "error", "msg": f"offset must be 0 or more, not {offset}"}
        with self._lock:
            try:
                job = self._find(obj)
            except QueueError as exc:
                return {"status": "error", "msg": str(exc)}
            if job.token and not obj.get("token"):
                return {"status": "error",
                        "msg": f"tail needs job {job.id}'s token (describe / kq show "
                               f"{job.id} gives it)"}
            # the state before the read: an ended job's log is complete
            state, path, run_id = job.state, job.log_path, job.run_id
        ended = state in ENDED
        reply = {"status": "ok", "lines": [], "offset": offset, "done": False,
                 "state": state, "run_id": run_id}
        if not path:
            reply["done"] = ended
            return reply
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if offset > size:
                    return {"status": "error",
                            "msg": f"offset {offset} is past the end of job {job.id}'s log "
                                   f"({size} bytes)"}
                f.seek(offset)
                data = f.read(TAIL_CHUNK)
        except FileNotFoundError:
            reply["done"] = ended
            return reply
        except OSError as exc:
            return {"status": "error", "msg": f"job {job.id}'s log cannot be read: {exc}"}
        at_end = offset + len(data) >= size
        cut = data.rfind(b"\n") + 1
        if cut == 0 and len(data) == TAIL_CHUNK:
            cut = len(data)                            # one line longer than a chunk
        if ended and at_end:
            cut = len(data)                            # the final, unterminated line
        chunk = data[:cut]
        text = chunk.decode("utf-8", errors="replace")
        if text.endswith("\n"):
            text = text[:-1]
        reply["lines"] = [ln.rstrip("\r") for ln in text.split("\n")] if chunk else []
        reply["offset"] = offset + len(chunk)
        reply["done"] = ended and reply["offset"] >= size
        return reply

    def pause(self, obj: Mapping) -> dict:
        """``{"scope": "agent" | "all", "by", "reason", "owner"}`` (``owner``:
        who pauses, "person" when absent; stored -- an agent may not resume a
        person's pause)."""
        scope = str(obj.get("scope") or "agent")
        if scope not in PAUSE_SCOPES:
            return {"status": "error", "msg": f"scope must be one of {', '.join(PAUSE_SCOPES)}"}
        owner = str(obj.get("owner") or "person")
        if owner not in OWNERS:
            return {"status": "error", "msg": f"owner must be one of {', '.join(OWNERS)}"}
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        reason = str(obj.get("reason") or "")
        with self._lock:
            held = self._paused.get(scope)
            if held and held.get("owner", "person") == "person" and owner == "agent":
                return {"status": "error",
                        "msg": f"{scope} jobs are already paused by a person ({held.get('by')})"
                               ": an agent may not replace that pause"}
            self._paused[scope] = {"by": by, "since": self._clock(), "reason": reason,
                                   "owner": owner}
        log.warning("Run queue: %s jobs paused by %s%s.", "agent" if scope == "agent" else "all",
                    by, f": {reason}" if reason else "")
        self._record("run_queue_pause", scope=scope, by=by, reason=reason, owner=owner)
        self._save()
        self._notify()
        return {"status": "ok", "run_queue": self.info()}

    def resume(self, obj: Mapping) -> dict:
        scope = str(obj.get("scope") or "agent")
        if scope not in PAUSE_SCOPES:
            return {"status": "error", "msg": f"scope must be one of {', '.join(PAUSE_SCOPES)}"}
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        as_owner, refusal = requester_owner(obj)
        if refusal:
            return {"status": "error", "msg": refusal}
        with self._lock:
            paused = self._paused.get(scope)
            if paused is None:
                return {"status": "error", "msg": f"{scope} jobs are not paused"}
            if paused.get("owner", "person") == "person" and as_owner == "agent":
                return {"status": "error",
                        "msg": f"{scope} jobs were paused by a person ({paused.get('by')}): an "
                               "agent may not resume them"}
            self._paused[scope] = None
        log.info("Run queue: %s jobs resumed by %s.", scope, by)
        self._record("run_queue_resume", scope=scope, by=by, owner=as_owner)
        self._save()
        self._notify()
        return {"status": "ok", "run_queue": self.info()}

    def hold_request(self, obj: Mapping) -> dict:
        """``{"reason", "by", "owner"}`` (``owner`` "person" when absent;
        stored -- an agent may not release a person's hold)."""
        owner = str(obj.get("owner") or "person")
        if owner not in OWNERS:
            return {"status": "error", "msg": f"owner must be one of {', '.join(OWNERS)}"}
        reply = self.hold.hold(str(obj.get("reason") or ""),
                               str(obj.get("by") or obj.get("operator") or obj.get("client")
                                   or ""), owner=owner)
        self._notify()
        return reply

    def set_warm_ahead(self, obj: Mapping) -> dict:
        """``{"on": bool | None, "by", "owner"}``: switch warm start on or
        off (``on`` absent: report only).  The value is stored in
        ``<dir>/settings.json`` (atomic replace) and is the server's start
        value from then on -- unless ``WAXX_QUEUE_WARM_AHEAD`` is set where
        the server starts (the reply's ``env_override`` says so).  Switched
        off, a live warm launch is told to exit at the next tick ("warm start
        was switched off").  Either owner may switch it; every change (of the
        value or of what is stored) is journalled with who and when.  Reply:
        ``{"warm_ahead", "was", "persisted", "env_override"}`` (``persisted``
        False with ``persist_error`` when it could not be stored: it then
        holds until the server restarts)."""
        as_owner, refusal = requester_owner(obj)
        if refusal:
            return {"status": "error", "msg": refusal}
        on = obj.get("on")
        env = (f"{WARM_AHEAD_ENV}={'1' if self._warm_env else '0'}"
               if self._warm_env is not None else None)
        if on is None:
            return {"status": "ok", "warm_ahead": self.warm_ahead,
                    "source": self.warm_ahead_source, "env_override": env}
        if not isinstance(on, bool):
            return {"status": "error", "msg": f"on must be true or false, not {on!r}"}
        by = str(obj.get("by") or obj.get("operator") or obj.get("client") or "?")
        with self._lock:
            was, self.warm_ahead = self.warm_ahead, on
        stored_before, persisted, error = self._store_warm_ahead(on, by, as_owner)
        if was != on or (persisted and stored_before != on):
            log.warning("Run queue: warm start switched %s by %s (%s)%s.", "ON" if on else "OFF",
                        by, as_owner, "" if persisted else f"; NOT stored ({error})")
            self._record("run_queue_warm_ahead", on=on, was=was, by=by, owner=as_owner,
                         stored=on if persisted else stored_before,
                         **({} if persisted else {"persist_error": error}))
            self._notify()
        reply = {"status": "ok", "warm_ahead": on, "was": was, "persisted": persisted,
                 "env_override": env}
        if not persisted:
            reply["persist_error"] = error
        return reply

    def _store_warm_ahead(self, on: bool, by: str, owner: str) -> tuple:
        """Store warm start -> ``(stored before, stored now?, error text)``."""
        if not self.enabled:
            return None, False, "the queue has no folder"
        with self._settings_lock:
            data = read_settings(self.directory)
            rec = data.get("warm_ahead")
            before = rec.get("on") if isinstance(rec, dict) else None
            data["warm_ahead"] = {"on": on, "by": by, "owner": owner, "at": self._clock(),
                                  "t": _clock_text(self._clock())}
            try:
                from waxx.util.device_state.state_file_io import atomic_write  # noqa: PLC0415
                os.makedirs(self.directory, exist_ok=True)
                atomic_write(self._path(SETTINGS_FILE), data)
            except Exception as exc:                  # noqa: BLE001
                log.error("Run queue: could not store warm start in %s (%s).",
                          self._path(SETTINGS_FILE), exc)
                return before, False, str(exc)
        if self.warm_ahead_source != "env":
            self.warm_ahead_source = "stored"
        return before, True, ""

    def release_request(self, obj: Mapping) -> dict:
        """``{"by", "owner"}``; ``owner`` -- who asks -- is required."""
        as_owner, refusal = requester_owner(obj)
        if refusal:
            return {"status": "error", "msg": refusal}
        reply = self.hold.release(str(obj.get("by") or obj.get("operator")
                                      or obj.get("client") or ""), owner=as_owner)
        self._notify()
        return reply

    # -- state ----------------------------------------------------------------------

    def eligible_or_running(self) -> str:
        """"" unless a job is in the slot (launching, running, ending) or one
        is eligible to launch now; else which.  Queued jobs that are due
        later, held, paused or waiting on another job do not count: they
        leave a loop's Start alone."""
        with self._lock:
            cur = self._jobs.get(self._current) if self._current is not None else None
            if cur is not None:
                return f"the run queue's {cur.name} is {cur.state}"
            order = self._order(self._clock())
        if order:
            return f"the run queue's {order[0].name} is about to launch"
        return ""

    def monitor_busy(self) -> str:
        """Why a monitor (re)start must wait ("" when it need not): a job is
        in the slot, or one is eligible AND the queue's last gate found the
        machine free or legitimately in use (it launches as soon as that run
        ends).  A blocked gate (liveOD unreachable, a wedged run, the server
        busy...) never defers it: the hardware must not be left without the
        monitor while the queue cannot launch anyway."""
        with self._lock:
            cur = self._jobs.get(self._current) if self._current is not None else None
            if cur is not None:
                return f"the run queue's {cur.name} is {cur.state}"
            order = self._order(self._clock())
            kind = self._gate_kind
        if order and kind in ("free", "live"):
            return f"the run queue's {order[0].name} is about to launch"
        return ""

    def defer_monitor(self, what: str, why: str) -> None:
        """A monitor (re)start asked for while the queue has work: not done
        now; the queue asks for the monitor when it runs out."""
        with self._lock:
            self._owe_monitor = True
        log.warning("Run queue: %s deferred -- %s; the monitor starts when the queue runs out.",
                    what, why)
        self._record("run_queue_monitor_deferred", what=what, why=why)
        self.flush_journal()

    def stop_launching(self, why: str) -> None:
        """No job launches from now on (the server is restarting or shutting
        down); ``why`` is what the gate says.  "" lifts it."""
        with self._lock:
            self._stop_launching = str(why or "")

    def current_job(self) -> dict | None:
        """The job in the slot (launching / running / ending), or None."""
        with self._lock:
            cur = self._jobs.get(self._current) if self._current is not None else None
            return self._public(cur) if cur is not None else None

    def has_work(self) -> bool:
        """Jobs queued or in the slot (a loop's Start is refused then)."""
        with self._lock:
            return any(j.state in ("queued",) + IN_SLOT for j in self._jobs.values())

    def work_text(self) -> str:
        with self._lock:
            queued = sum(j.state == "queued" for j in self._jobs.values())
            cur = self._jobs.get(self._current) if self._current is not None else None
        parts = []
        if cur is not None:
            parts.append(f"{cur.name} is {cur.state}")
        if queued:
            parts.append(f"{queued} job{'s' if queued != 1 else ''} queued")
        return " and ".join(parts)

    def agent_run_ids(self) -> set:
        with self._lock:
            return {j.run_id for j in self._jobs.values()
                    if j.owner == "agent" and j.run_id is not None}

    def own_abort_ids(self) -> set:
        with self._lock:
            return set(self._own_abort_ids)

    def own_run_ids(self) -> set:
        with self._lock:
            return {j.run_id for j in self._jobs.values() if j.run_id is not None}

    def info(self) -> dict:
        """``status_json["run_queue"]``."""
        with self._lock:
            now = self._clock()
            counts = {s: 0 for s in STATES}
            for j in self._jobs.values():
                counts[j.state] += 1
            cur = self._jobs.get(self._current) if self._current is not None else None
            order = self._order(now)
            paused = {k: (dict(v) if v else None) for k, v in self._paused.items()}
            hold = self.hold.info()
            if cur is not None:
                state = "running"
                text = f"{cur.name} {cur.state}" + (f", run {cur.run_id}" if cur.run_id else "")
            elif order:
                state = "waiting"
                text = f"{order[0].name} next: {self._waiting or 'launching'}"
            elif counts["queued"]:
                state = "held" if hold.get("active") else (
                    "paused" if any(paused.values()) else "waiting")
                text = f"{counts['queued']} queued, none can start now"
            else:
                state, text = "idle", "no jobs"
            return {"enabled": self.enabled, "state": state, "text": text,
                    "current": self._public(cur, now) if cur is not None else None,
                    "next": [j.id for j in order[:20]], "counts": counts,
                    "waiting": self._waiting, "alarm": dict(self._alarm) if self._alarm else None,
                    "paused": paused, "person_hold": hold,
                    "resume_loop": dict(self._resume_loop) if self._resume_loop else None,
                    "directory": self.directory, "warm_ahead": self.warm_ahead,
                    "warm_ahead_source": self.warm_ahead_source,
                    "warm_ahead_env": (None if self._warm_env is None else
                                       f"{WARM_AHEAD_ENV}={'1' if self._warm_env else '0'}"),
                    "warm": self._warm_summary(now),
                    "warm_alarm": dict(self._warm_alarm) if self._warm_alarm else None}

    # -- scheduling -------------------------------------------------------------------

    def _blocked_by(self, job: Job, now: float, ignore_hold: bool = False) -> str:
        """Why a queued job may not go now ("" when it is eligible) -- the
        job's own conditions, not the machine's.  ``ignore_hold``: the person
        hold left out (a warm launch stays ready behind it; GO still waits)."""
        if job.state != "queued":
            return f"it is {job.state}"
        if job.paused:
            return (f"paused (this job) by {job.paused_by or '?'} since "
                    f"{_since_text(job.paused_since)}")
        if job.due is not None and job.due > now:
            return f"due {_since_text(job.due)}"
        waits = [a for a in job.after if self._dep_state(a) not in SUCCEEDED]
        if waits:
            return "after " + ", ".join(f"#{a} ({self._dep_state(a)})" for a in waits)
        if self._paused.get("all"):
            p = self._paused["all"]
            return f"all jobs paused by {p.get('by')} since {_since_text(p.get('since'))}"
        if job.owner == "agent" and self._paused.get("agent"):
            p = self._paused["agent"]
            return f"agent jobs paused by {p.get('by')} since {_since_text(p.get('since'))}"
        if job.owner == "agent" and self.hold.active and not ignore_hold:
            return f"held: {self.hold.info().get('reason')}"
        return ""

    def _dep_state(self, job_id: int) -> str:
        """A dependency's state: its record, or -- for a job no longer kept in
        queue.json (beyond KEEP_ENDED, or lost with an unreadable file) -- its
        last end in the journal; a job found nowhere counts as "failed" (its
        dependents are skipped, never left waiting)."""
        dep = self._jobs.get(job_id)
        if dep is not None:
            return dep.state
        cached = self._dep_cache.get(job_id)
        if cached is None:
            cached = "failed"
            try:
                with open(self._path("journal.jsonl"), "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        if rec.get("kind") == "run_queue_end" and rec.get("job") == job_id:
                            cached = str(rec.get("state") or "failed")
            except (OSError, TypeError):
                pass
            self._dep_cache[job_id] = cached
        return cached

    def _order(self, now: float) -> list[Job]:
        """The eligible jobs, the next one first: lowest rank (ties: lowest
        id).  Due times, dependencies, pauses and the hold only gate a job --
        they never reorder the queue."""
        eligible = [j for j in self._jobs.values()
                    if j.state == "queued" and not self._blocked_by(j, now)]
        return sorted(eligible, key=lambda j: (j.rank, j.id))

    def _why_waiting(self, job: Job, now: float, order: list | None = None) -> str:
        """Why a queued job is not launching (``order``: :meth:`_order`, when
        the caller has it already)."""
        if job.state != "queued":
            return ""
        own = self._blocked_by(job, now)
        if own:
            return own
        if self._current is not None:
            cur = self._jobs.get(self._current)
            text = f"after #{self._current} ({cur.state if cur else 'in the slot'})"
            if job.warm is not None:
                text += f"; {_warm_text(job.warm, now)}"
            return text
        order = self._order(now) if order is None else order
        if order and order[0].id != job.id:
            return f"#{order[0].id} goes first"
        return self._waiting or "launching"

    def tick(self) -> None:
        """One step: follow the job in the slot, settle dependencies, stop or
        restart a loop, launch the next job when the machine is free, and the
        alarm.  Called by one thread: the monitor server's watch, every
        ``WATCH_S`` (0.5 s).  liveOD is polled at most every
        ``poll_every_s`` (:data:`POLL_EVERY_S`, 2 s) for the hold's watch and
        the slot's job, and afresh before a launch, an Abort and an outcome."""
        if not self._tick_lock.acquire(blocking=False):
            return
        try:
            now = self._clock()
            if self._last_poll_t is None or now - self._last_poll_t >= self.poll_every_s:
                self._poll_now()
            self.hold.tick()
            if self._current is None:
                self._take_leftover()
            cur = self._jobs.get(self._current) if self._current is not None else None
            if cur is not None:
                self._follow(cur)
            self._settle(now)
            self._warm_tick(self._clock())
            if self._current is None:
                self._schedule(self._clock())
            self._alarm_tick(self._clock())
        finally:
            self._tick_lock.release()
            self.flush_journal()

    def _poll_now(self) -> dict | None:
        """A fresh POLL (fed to the person hold's watch); None when liveOD does
        not answer."""
        self._last_poll_t = self._clock()
        if self._poll_fn is None:
            return None
        try:
            poll = self._poll_fn()
        except Exception:                             # noqa: BLE001
            self._last_poll = None
            return None
        self._last_poll = poll
        self.hold.observe_poll(poll, self.agent_run_ids(), self.own_abort_ids(),
                               on_person_reset=self._claim_person_reset)
        return poll

    def _claim_person_reset(self, last: dict) -> bool:
        """A person pressed liveOD's Reset while the job in the slot has no
        run id yet (it is launching or preparing; liveOD had no run, so the
        Reset names none).  The person's own job: taken as that job's cancel
        -- liveOD would spend that Abort on no run and let the run through, so
        the queue sends its own Abort once the run has an id -- and no hold.
        Anyone else's job (or a Reset naming a run): not claimed, the hold
        goes on as for any person's Reset."""
        if (last or {}).get("run_id") is not None:
            return False
        with self._lock:
            cur = self._jobs.get(self._current) if self._current is not None else None
            if (cur is None or cur.run_id is not None or cur.owner != "person"
                    or cur.state not in ("launching", "running")):
                return False
            if cur.cancel is None:
                cur.cancel = {"by": "a person's Reset in liveOD", "at": self._clock(),
                              "abort_sent": False,
                              "abort_note": "Reset pressed before the run had an id: the queue "
                                            "sends liveOD's Abort once it has one"}
        log.warning("Run queue: a person pressed Reset in liveOD while %s (a person's job) had "
                    "no run yet: taken as its cancel.", cur.name)
        self._record("run_queue_cancel_requested", job=cur.id, by="a person's Reset in liveOD",
                     run_id=None, via="live_od_reset")
        self._save()
        return True

    def _settle(self, now: float) -> None:
        """Skip jobs waiting for a job that will never be saved."""
        changed = True
        while changed:
            changed = False
            with self._lock:
                for job in sorted(self._jobs.values(), key=lambda j: j.id):
                    if job.state != "queued":
                        continue
                    for a in job.after:
                        state = self._dep_state(a)
                        if state in ENDED and state not in SUCCEEDED:
                            dep = self._jobs.get(a)
                            why = (dep.reason if dep is not None else
                                   "no longer in queue.json; from the journal")
                            self._end(job, "skipped", f"job {a} ended {state}"
                                      + (f" ({why})" if why else ""))
                            changed = True
                            break
        if changed:
            self._save()

    def _schedule(self, now: float) -> None:
        with self._lock:
            order = self._order(now)
        loop = self._active_loop()
        if order:
            if loop is not None:
                stopper = next((j for j in order if self._may_stop(j, loop)), None)
                if stopper is not None:
                    self._stop_loop_for_jobs(loop, stopper)
            if self._last_end_t is not None and now - self._last_end_t < self._gap_s:
                return
            head = order[0]
            if head.warm is not None:
                self._go_or_wait(head, self._clock())
                return
            why = self._gate()
            if why:
                if why != self._waiting:
                    log.info("Run queue: %s waits: %s", head.name, why)
                self._waiting = why
                return
            self._waiting = ""
            for job in order:
                if job.warm is not None:
                    # its warm process decides (never a second, cold one)
                    self._go_or_wait(job, self._clock())
                    return
                if self._launch(job):          # else skipped at launch: try the next
                    return
            return
        self._waiting = ""
        self._drained(loop)

    def _active_loop(self):
        for loop in self._loops.values():
            if getattr(loop, "active", False):
                return loop
        return None

    def _own_ended_fence(self, fence: dict) -> bool:
        """The run fence is the one the queue's last ended job left up (it
        exited without lifting it): only that job's run id, and only a real
        one (a positive int -- every save_data=False run is run id 0)."""
        fid = fence.get("run_id")
        last = self._last_ended_run_id
        return (isinstance(fid, int) and not isinstance(fid, bool) and fid > 0
                and fid == last)

    def _gate(self) -> str:
        """Why the next job may not launch now ("" when it may).  Sets
        ``_gate_kind``: "free", "live" (someone's run is legitimately using
        the machine: a run in progress or announced, a loop finishing its run
        after a stop) or "blocked" (anything else -- wedged, an Abort pending,
        liveOD unreachable or unknown, the server busy, a loop that does not
        stop): only "blocked" runs the alarm's clock."""
        self._gate_kind = "blocked"
        if self._stop_launching:
            return self._stop_launching
        if self._clock() < self._launch_blocked_until:
            return ("queue.json could not be saved ("
                    + (self._save_error or "it is not to be overwritten")
                    + "): launches wait until it can be")
        busy = self._server_busy() if self._server_busy is not None else ""
        if busy:
            return busy
        loop = self._active_loop()
        if loop is not None:
            info = loop.info()
            if info.get("state") == "stopping":
                self._gate_kind = "live"
                return (f"{loop.spec.title} is finishing its run in progress (asked to stop "
                        "for the queue)")
            if info_category(info) == "holds":
                self._gate_kind = "live"          # a holding loop: legitimate use
                whose = {"person": "a person", "agent": "an agent"}.get(
                    effective_owner(info), "the run queue")
                return (f"{loop.spec.title} holds (started by {whose}, "
                        f"{info.get('operator') or info.get('client') or '?'}): an agent's "
                        "job waits behind it")
            return f"{loop.spec.title} is running and has not been asked to stop"
        poll = self._poll_now()
        if poll is None:
            return "liveOD is not reachable -- a run could not save"
        fence = self._fence() if self._fence is not None else None
        if fence and self._own_ended_fence(fence):
            fence = None                  # the fence of the queue's own job that just ended
        monitor = self._monitor_state() if self._monitor_state is not None else None
        verdict = run_gate.classify(poll, fence, monitor_state=monitor)
        if verdict.state == "free":
            self._gate_kind = "free"
            return ""
        if verdict.state == "live":
            self._gate_kind = "live"
        if verdict.waivable:
            self._gate_kind = "free"
            key = (verdict.run_id, verdict.state)
            if key != self._waived:
                self._waived = key
                log.warning("Run queue: run %s is %s in liveOD and is waived: %s",
                            verdict.run_id, verdict.state, verdict.reason)
                self._record("run_queue_waived", run_id=verdict.run_id, state=verdict.state,
                             reason=verdict.reason)
            return ""
        return verdict.reason

    # -- warm start ---------------------------------------------------------------------

    def _public(self, job: Job, now: float | None = None) -> dict:
        """The job's record as replies show it (the warm record trimmed:
        :func:`_warm_public`)."""
        d = job.to_dict()
        d["warm"] = _warm_public(job.warm, self._clock() if now is None else now)
        return d

    def _warm_summary(self, now: float) -> dict | None:
        """The live warm launch for ``info()``: ``{"job", "label", "state",
        "text", "attempt"}``, or None."""
        for j in self._warm_live():
            return {"job": j.id, "label": j.label, "state": j.warm.get("state"),
                    "text": _warm_text(j.warm, now), "attempt": j.warm.get("attempt")}
        return None

    def _warm_live(self) -> list[Job]:
        """The jobs with a warm record whose GO was not given (one at most,
        normally), by id."""
        return sorted((j for j in self._jobs.values()
                       if j.warm is not None and not j.warm.get("go")), key=lambda j: j.id)

    def _warm_candidate(self, now: float, ignore_hold: bool = False) -> Job | None:
        """The job a warm launch is for: the first queued job in rank order
        not blocked at ``now + WARM_DUE_LEAD_S`` (so a job due within that
        may warm; its GO waits until it is due).  None when a job ahead of it
        waits for the job in the slot (``after``): that one may well go first
        once the slot job is saved, so nobody warms (never an ``after`` on the
        running job).  ``ignore_hold``: held agent jobs count (a live warm
        launch stays ready behind a person hold); a new warm launch is never
        started for a held job."""
        for j in self._queued_in_order():
            if not self._blocked_by(j, now + WARM_DUE_LEAD_S, ignore_hold=ignore_hold):
                return j
            if self._current is not None and self._current in j.after:
                return None
        return None

    @staticmethod
    def _reserve_advertised(poll) -> bool:
        """liveOD's POLL advertises ``features.reserve`` (absent: no)."""
        f = poll.get("features") if isinstance(poll, dict) else None
        if isinstance(f, dict):
            return bool(f.get("reserve"))
        if isinstance(f, (list, tuple, set)):
            return "reserve" in f
        return False

    def _job_env(self, job: Job, warm_token: str | None = None) -> dict:
        """A launch's environment.  ``WAXX_QUEUE_WARM`` only on a warm
        launch (never inherited from the server's own environment)."""
        # unbuffered: output reaches the log as printed; UTF-8: the log is read
        # as UTF-8, and a console code page cannot fail a print of "µs" or "─"
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env[run_gate.LAUNCHER_ENV] = LAUNCHER
        env[JOB_ENV] = job.queue_job
        env[OWNER_ENV] = job.owner
        if job.write_back is False:
            env[NO_WRITE_BACK_ENV] = "1"
        else:
            env.pop(NO_WRITE_BACK_ENV, None)
        env.pop(WARM_ENV, None)
        if warm_token:
            env[WARM_ENV] = warm_token
        return env

    def _command(self, job: Job) -> str:
        command = ar_command(_quote(job.path))
        if job.argv:
            command += " " + " ".join(_quote(a) for a in job.argv)
        return command

    def _mark_aborting(self, job: Job, why: str) -> bool:
        """Under the lock: tell the job's warm process to stand down (it
        exits at its next poll).  False when there is nothing to tell (no
        live warm launch, GO given, or already told)."""
        w = job.warm
        if w is None or w.get("go") or w.get("state") == "aborting":
            return False
        w["state"] = "aborting"
        w["why"] = why
        w["aborted_at"] = self._clock()
        log.info("Run queue: %s's warm process is told to exit: %s", job.name, why)
        self._record("run_queue_warm_discard", job=job.id, attempt=w.get("attempt"), why=why,
                     py_pid=w.get("py_pid"), cmd_pid=w.get("cmd_pid"))
        return True

    def _discard_warm(self, job: Job, why: str) -> None:
        with self._lock:
            done = self._mark_aborting(job, why)
            if done and why.endswith(WARM_STALE_MARK):
                # only stale (the slot job ran long): its attempt is given
                # back at its exit, so waiting never makes the job cold
                job.warm["refund_attempt"] = True
        if done:
            self._save()
            self._notify()

    def _warm_tick(self, now: float) -> None:
        """Follow the warm launches (their processes, discards, the alarm),
        then start one when it is time (``warm_ahead``)."""
        self._follow_warm_spawn()
        for job in self._warm_live():
            self._follow_warm(job, now)
        self._maybe_warm(now)

    def _follow_warm(self, job: Job, now: float) -> None:
        code = self._warm_exit_code(job, now)
        if code is not None:
            self._warm_exited(job, code)
            return
        w = job.warm
        if w is None:
            return
        if w.get("state") == "aborting":
            waited = now - float(w.get("aborted_at") or now)
            if waited > WARM_KILL_AFTER_S and not w.get("kill_tried"):
                if self._stop_warm(job, waited):
                    return                            # its exit is seen at a later tick
            if waited > WARM_KILL_AFTER_S and not w.get("alarmed"):
                text = (f"{job.name}'s discarded warm process (pid "
                        f"{w.get('py_pid') or w.get('cmd_pid') or '?'}) has not exited "
                        f"{waited:.0f} s after it was told to ({w.get('why')}); "
                        + (f"the queue's stop failed ({w.get('kill_note')})"
                           if w.get("kill_tried") else "the queue does not stop it")
                        + " -- check it")
                with self._lock:
                    w["alarmed"] = now
                    self._warm_alarm = {"job": job.id, "since": now, "text": text}
                log.warning("RUN QUEUE ALARM: %s", text)
                self._record("run_queue_warm_not_exiting", job=job.id, waited_s=round(waited, 1),
                             py_pid=w.get("py_pid"), cmd_pid=w.get("cmd_pid"), why=w.get("why"))
                self._save()
                self._notify()
            return
        why = self._warm_discard_why(job, now)
        if (not why and w.get("state") == "ready"
                and now - self._warm_checked.get(job.id, float("-inf")) >= WARM_MANIFEST_CHECK_S):
            self._warm_checked[job.id] = now
            why = self._warm_stale(job)
        if why:
            self._discard_warm(job, why)

    def _stop_warm(self, job: Job, waited: float) -> bool:
        """A discarded warm process has not exited ``WARM_KILL_AFTER_S``
        after it was told to: stop it -- its python, then the shell the queue
        launched, each with its tree (:func:`~waxx.util.device_state.detached.
        kill_tree`) -- only while it was never given GO (it has claimed
        nothing: no hardware, no core connection, no liveOD start) and only
        a pid whose recorded creation time still matches (Windows reuses
        pids).  Tried once per warm launch.  True when nothing of it is left
        alive; False leaves the alarm to the caller."""
        with self._lock:
            w = job.warm
            if w is None or w.get("state") != "aborting" or w.get("kill_tried"):
                return False
            targets = [(role, w.get(p), w.get(s)) for role, p, s in
                       (("experiment", "py_pid", "py_started"), ("shell", "cmd_pid", "cmd_started"))
                       if w.get(p)]
            if w.get("go"):
                note = "not stopped: GO was decided for it"
                targets = []
            elif not targets:
                note = "not stopped: no pid of it is known"
            elif any(started is None for _, _, started in targets):
                note = ("not stopped: a pid of it has no recorded creation time (it may be "
                        "another process with a reused pid)")
                targets = []
            else:
                note = ""
            w["kill_tried"] = self._clock()
        results = []
        for role, pid, started in targets:
            try:
                r = dict(self._kill(pid, started))
            except Exception as exc:                  # noqa: BLE001
                r = {"pid": pid, "ok": False, "gone": False, "note": f"kill failed: {exc!r}"}
            r["role"] = role
            results.append(r)
        left = [r for r in results if not (r.get("ok") or r.get("gone"))]
        if not note:
            note = ("stopped" if not left else "NOT all stopped: " + "; ".join(
                f"{r['role']} pid {r['pid']}: {r.get('note')}" for r in left))
        with self._lock:
            if job.warm is w:
                w["kill_note"] = note
                w["kill_results"] = results
        stopped = bool(results) and not left
        (log.warning if stopped else log.error)(
            "Run queue: %s's discarded warm process had not exited %.0f s after it was told to "
            "(%s); never given GO: %s.", job.name, waited, w.get("why"), note)
        self._record("run_queue_warm_killed", job=job.id, attempt=w.get("attempt"),
                     waited_s=round(waited, 1), why=w.get("why"), results=results, note=note,
                     stopped=stopped)
        self._save()
        self._notify()
        return stopped

    def _warm_discard_why(self, job: Job, now: float) -> str:
        """Why the job's live warm launch must stand down ("" when it may
        stay)."""
        with self._lock:
            w = job.warm
            if w is None:
                return ""
            if job.state != "queued":
                return f"{job.name} is {job.state}" + (f" ({job.reason})" if job.reason else "")
            if job.cold or job.cold_why:
                return "it was made cold" + (f" ({job.cold_why})" if job.cold_why else "")
            if not self.warm_ahead:
                return "warm start was switched off"
            spec = w.get("spec") or {}
            changed = [k for k in WARM_SPEC_FIELDS if k in spec and spec[k] != getattr(job, k)]
            if changed:
                return "edited (" + ", ".join(changed) + ")"
            # a person hold keeps it ready (user, 2026-10-19): GO still waits
            # for the release (_order / _go check the hold); anything else
            # that blocks the job discards it
            blocked = self._blocked_by(job, now + WARM_DUE_LEAD_S, ignore_hold=True)
            if blocked:
                return blocked
            cand = self._warm_candidate(now, ignore_hold=True)
            if cand is not job:
                return "no longer next" + (f" (#{cand.id} is)" if cand is not None else "")
            if (w.get("state") == "ready"
                    and now - float(w.get("ready_at") or now) > WARM_READY_MAX_S):
                return f"ready for more than {WARM_READY_MAX_S / 60.0:.0f} min {WARM_STALE_MARK}"
            if (w.get("state") == "warming"
                    and now - float(w.get("spawned_at") or now) > WARM_PREP_MAX_S):
                return f"not ready {WARM_PREP_MAX_S / 60.0:.0f} min after its launch"
        return ""

    def _warm_stale(self, job: Job) -> str:
        """Called OUTSIDE the lock: why the ready warm process's code is no
        longer what is on disk -- a manifest file changed or is gone, or the
        experiment file differs from its hash at the warm launch -- or ""."""
        with self._lock:
            w = job.warm
            if w is None:
                return ""
            manifest = list(w.get("manifest") or ())
            sha, path = w.get("sha256"), job.path
        for entry in manifest:
            try:
                p, mtime, size = str(entry[0]), int(entry[1]), int(entry[2])
            except (TypeError, ValueError, IndexError):
                continue
            if queue_gate.is_network_path(p):
                continue                              # never stat a share from the tick
            try:
                st = os.stat(p)
            except OSError:
                return f"a file it loaded is gone: {p}"
            if st.st_mtime_ns != mtime or st.st_size != size:
                return f"a file it loaded changed: {p}"
        view: dict = {}
        self._fill_source_changed([(view, self._stat_key(path), sha)])
        if view.get("source_changed") is None:
            return "its experiment file cannot be read"
        if view["source_changed"]:
            return "its experiment file changed since its warm launch"
        return ""

    def _adopt_warm(self, w: Mapping):
        """A watch on a warm process from its record -- the launched shell,
        else the experiment's python -- by pid AND creation time (a pid with
        no recorded creation time is never taken: Windows reuses pids)."""
        for pid_key, started_key in (("cmd_pid", "cmd_started"), ("py_pid", "py_started")):
            pid, started = w.get(pid_key), w.get(started_key)
            if not pid or started is None:
                continue
            try:
                proc = self._adopt(pid, started)
            except Exception:                         # noqa: BLE001
                proc = None
            if proc is not None:
                return proc
        return None

    def _warm_exit_code(self, job: Job, now: float):
        """None while the job's warm process lives (or may: its launch is
        still being answered); its exit code once it is gone (-1: unknown)."""
        proc = self._warm_procs.get(job.id)
        if proc is None:
            sp = self._warm_spawn
            if sp is not None and sp["job"] == job.id:
                return None
            w = job.warm or {}
            if w.get("py_pid"):
                proc = self._adopt_warm({"py_pid": w.get("py_pid"),
                                         "py_started": w.get("py_started")})
                if proc is None:
                    return -1
                self._warm_procs[job.id] = proc
            else:
                since = max(float(w.get("spawned_at") or 0.0), float(w.get("adopt_from") or 0.0))
                return None if now - since < WARM_HELLO_WAIT_S else -1
        try:
            return proc.poll()
        except Exception:                             # noqa: BLE001
            return None

    def _warm_exited(self, job: Job, code) -> None:
        """A warm process (no GO given) has exited: its record goes.  Told to
        stand down, or it stood down itself (exit 78), or gone with no exit
        code known: the job stays as it is (a launch discarded only for
        being stale gives its attempt back).  Any other exit -- it died
        while warming or ready, before it took the machine -- makes the job
        COLD and leaves it queued: some failures happen only because the
        launch was warm (a host kernel call in prepare refused before GO, a
        lazy device touched before GO, a person ending a stuck warm
        process); a genuine bug then fails the cold launch as before."""
        proc = self._warm_procs.pop(job.id, None)
        _close(proc)
        self._warm_checked.pop(job.id, None)
        with self._lock:
            w = job.warm
            if w is None or w.get("go"):
                return
            state = w.get("state")
            job.warm = None
            refund = bool(w.get("refund_attempt")) and job.warm_attempts > 0
            if refund:
                job.warm_attempts -= 1
            cold = (state != "aborting" and code not in (-1, EXIT_WARM_ABORTED)
                    and job.state == "queued")
            if cold:
                job.cold_why = (f"warm attempt failed (exit {code}): its warm process exited "
                                f"while {state}, before it took the machine; it launches cold")
            if self._warm_alarm is not None and self._warm_alarm.get("job") == job.id:
                self._warm_alarm = None
        if cold:
            log.warning("Run queue: %s: its warm process exited (code %s) while %s, before "
                        "it took the machine; the job stays queued and launches cold.",
                        job.name, code, state)
        else:
            log.info("Run queue: %s's warm process exited (code %s; it was %s%s).", job.name,
                     code, state, f": {w.get('why')}" if w.get("why") else "")
        self._record("run_queue_warm_exited", job=job.id, attempt=w.get("attempt"), code=code,
                     state=state, why=w.get("why") or "", failed=False, made_cold=cold,
                     attempt_refunded=refund)
        if cold:
            self._record("run_queue_warm_made_cold", job=job.id, attempt=w.get("attempt"),
                         code=code, cold_why=job.cold_why)
        self._save()
        self._notify()

    def _maybe_warm(self, now: float) -> None:
        """Start a warm launch when it is time (see the module docstring)."""
        if not self.warm_ahead or self._stop_launching or self._warm_spawn is not None:
            return
        with self._lock:
            if self._warm_live():
                return                                # one warm process at a time
            cur = self._jobs.get(self._current) if self._current is not None else None
            if cur is None or cur.state != "running" or cur.calibrates_declared:
                return
            poll = self._last_poll
            if not self._reserve_advertised(poll):
                return
            if not (self._is_job_run(cur, poll) and poll.get("run_in_progress")
                    and int(poll.get("n_shots") or 0) >= 1):
                return                                # the two compiles never overlap
            end, _ = self._running_end(cur, now)
            if end is None or end - now > WARM_LEAD_S:
                # an unknown ETA waits until liveOD's shot timing (or a
                # duration estimate) gives one: warming at once would go stale
                # on a long run
                return
            cand = self._warm_candidate(now)
            if (cand is None or cand.cold or cand.cold_why
                    or cand.warm_attempts >= WARM_MAX_ATTEMPTS
                    or self._warm_skip.get(cand.id) == cur.id):
                return
            path = cand.path
        try:
            sha = file_sha256(path)
        except OSError:
            return
        if sha != cand.sha256 and not cand.allow_drift:
            return                                    # skipped at its cold launch, as today
        self._warm_launch(cand, cur, sha, now)

    def _warm_launch(self, job: Job, cur: Job, sha: str, now: float) -> None:
        """Launch ``job``'s warm process: its record is SAVED first (a server
        that dies meanwhile knows its token), then the process is started on
        a thread of its own, at BELOW_NORMAL priority, with
        ``WAXX_QUEUE_WARM=<token>``."""
        token = secrets.token_hex(8)
        log_path = self._path("logs", f"{job.id}_{job.label}.out")
        command = self._command(job)
        with self._lock:
            if (job.state != "queued" or job.warm is not None or self._stop_launching
                    or self._warm_candidate(now) is not job):
                return
            job.warm_attempts += 1
            job.warm = {"token": token, "state": "warming", "attempt": job.warm_attempts,
                        "spawned_at": now, "for_slot": cur.id, "sha256": sha,
                        "spec": {k: getattr(job, k) for k in WARM_SPEC_FIELDS},
                        "cmd_pid": None, "cmd_started": None, "py_pid": None,
                        "py_started": None, "manifest": None, "ready_at": None, "go": None,
                        "log_offset": None}
            job.warm_started_at = now
            job.log_path = log_path
        if not self._save():
            with self._lock:
                if job.warm is not None and job.warm.get("token") == token:
                    job.warm = None
                    job.warm_attempts -= 1
            log.error("Run queue: %s not warm-launched: queue.json could not be saved first (%s).",
                      job.name, self._save_error or "it is not to be overwritten")
            return
        self._record("run_queue_warm_launch", job=job.id, attempt=job.warm_attempts,
                     command=command, for_slot=cur.id, log_path=log_path)
        env = self._job_env(job, warm_token=token)
        try:
            os.makedirs(self._path("logs"), exist_ok=True)
            with open(log_path, "ab") as f:
                f.write((f"-- run queue {time.strftime('%Y-%m-%d %H:%M:%S')}: {job.name}, warm "
                         f"launch {job.warm_attempts} (behind job {cur.id}), owner {job.owner}, "
                         f"sha256 {sha[:12]} --\n$ {command}\n").encode("utf-8"))
                offset = f.tell()
        except OSError as exc:
            with self._lock:
                if job.warm is not None and job.warm.get("token") == token:
                    job.warm = None
                self._warm_skip[job.id] = cur.id
            log.error("Run queue: %s not warm-launched: cannot write its log %s (%s).", job.name,
                      log_path, exc)
            self._record("run_queue_warm_launch_failed", job=job.id, error=str(exc))
            self._save()
            return
        with self._lock:
            job.warm["log_offset"] = offset
        result: dict = {}
        record: dict = {}

        def spawn():
            try:
                proc = self._spawn(command, cwd=job.cwd, env=env, log_path=log_path,
                                   below_normal=True)
            except BaseException as exc:              # noqa: BLE001
                result["error"] = exc
                return
            with self._lock:
                result["proc"] = proc
                abandoned = self._warm_spawn is not record
            if abandoned:
                # the queue stopped waiting for this launch: the process is
                # found by its hello (pid + creation time) instead
                _close(proc)
        thread = threading.Thread(target=spawn, daemon=True, name=f"run-queue-warm-{job.id}")
        record.update({"job": job.id, "token": token, "thread": thread, "result": result,
                       "since": self._clock(), "slot": cur.id})
        self._warm_spawn = record
        log.info("Run queue: %s warm-launched behind %s (attempt %d).", job.name, cur.name,
                 job.warm_attempts)
        thread.start()
        thread.join(self._spawn_join_s)
        self._follow_warm_spawn()
        self._notify()

    def _follow_warm_spawn(self) -> None:
        """The warm launch thread's answer: the process to follow, or why
        there is none."""
        sp = self._warm_spawn
        if sp is None:
            return
        job = self._jobs.get(sp["job"])
        if sp["thread"].is_alive():
            if self._clock() - sp["since"] <= LAUNCH_TIMEOUT_S + WARM_HELLO_WAIT_S:
                return
            with self._lock:
                self._warm_spawn = None               # a late answer is closed by the thread
            log.error("Run queue: %s: its warm launch did not answer; the process (if any) is "
                      "followed by its hello.", job.name if job else sp["job"])
            self._record("run_queue_warm_launch_unknown", job=sp["job"], error="no answer")
            return
        self._warm_spawn = None
        result = sp["result"]
        if "proc" in result:
            proc = result["proc"]
            with self._lock:
                w = job.warm if job is not None else None
                keep = w is not None and w.get("token") == sp["token"]
                if keep:
                    w["cmd_pid"] = int(proc.pid)
                    w["cmd_started"] = getattr(proc, "started", None)
                    self._warm_procs[job.id] = proc
            if not keep:
                _close(proc)
                return
            self._record("run_queue_warm_launched", job=job.id, cmd_pid=int(proc.pid))
            self._save()
            return
        exc = result.get("error")
        from waxx.util.device_state.detached import LaunchUnknown  # noqa: PLC0415
        if isinstance(exc, LaunchUnknown):
            log.error("Run queue: %s: warm launch: %s", job.name if job else sp["job"], exc)
            self._record("run_queue_warm_launch_unknown", job=sp["job"], error=str(exc))
            return
        with self._lock:
            if job is not None and job.warm is not None and job.warm.get("token") == sp["token"]:
                job.warm = None
            self._warm_skip[sp["job"]] = sp["slot"]
        log.error("Run queue: could not warm-launch %s: %r (it launches cold).",
                  job.name if job else sp["job"], exc)
        self._record("run_queue_warm_launch_failed", job=sp["job"], error=repr(exc))
        self._save()
        self._notify()

    def _set_waiting(self, why: str, job: Job) -> None:
        if why != self._waiting:
            log.info("Run queue: %s waits: %s", job.name, why)
        self._waiting = why

    def _go_or_wait(self, job: Job, now: float) -> None:
        """The slot is free and ``job`` -- the next to go -- has a warm
        launch: GO when it is ready, the gate is free and its files are as
        it loaded them; otherwise wait (a discarded process first exits:
        never two processes for one job)."""
        with self._lock:
            w = dict(job.warm or {})
        state = w.get("state")
        if state == "aborting":
            self._gate_kind = "live"
            self._set_waiting(f"#{job.id}'s discarded warm process is exiting ({w.get('why')})",
                              job)
            return
        if state != "ready":
            self._gate_kind = "live"
            self._set_waiting(f"#{job.id} is {_warm_text(w, now)}: GO once it is ready", job)
            return
        why = self._gate()
        if why:
            self._set_waiting(why, job)
            return
        stale = self._warm_stale(job)
        if stale:
            self._discard_warm(job, stale)
            self._set_waiting(f"#{job.id}'s warm process is discarded: {stale}", job)
            return
        self._waiting = ""
        self._go(job, now)

    def _go(self, job: Job, now: float) -> None:
        """Give ``job``'s warm process the machine: the decision is SAVED
        (job ``launching``, ``warm.go``) before the ``warm`` request may hand
        it out; then the job is ``running`` with the warm process as its
        process.  The server raises the process's priority too (backstop)."""
        poll = self._last_poll
        reset_count = poll.get("reset_count") if isinstance(poll, dict) else None
        # the liveOD process that count belongs to (absent: an older liveOD)
        instance = poll.get("instance") if isinstance(poll, dict) else None
        proc = self._warm_procs.get(job.id)
        if proc is None:
            with self._lock:
                w = dict(job.warm or {})
            proc = self._adopt_warm(w)
            if proc is None:
                self._discard_warm(job, "its process cannot be followed (no watch, no pid with "
                                        "a creation time)")
                return
            self._warm_procs[job.id] = proc
        with self._lock:
            w = job.warm
            if (job.state != "queued" or w is None or w.get("state") != "ready" or w.get("go")
                    or self._current is not None or self._stop_launching
                    or self._blocked_by(job, now)):
                return
            job.state = "launching"
            job.launched_at = now
            job.reason = ""
            self._current = job.id
            w["go"] = {"at": now, "reset_count": reset_count, "liveod_instance": instance,
                       "delivered": None}
            self._procs[job.id] = None
            self._log_pos[job.id] = int(w.get("log_offset") or 0)
            self._log_partial[job.id] = ""
            self._tails[job.id] = deque(maxlen=TAIL_LINES)
            self._owe_monitor = True
            self._owed_since = None
        if not self._save():                           # GO is on disk before it is handed out
            with self._lock:
                if job.state == "launching" and w.get("go"):
                    job.state = "queued"
                    w["go"] = None
                    job.launched_at = None
                    self._current = None
                    self._procs.pop(job.id, None)
                self._launch_blocked_until = self._clock() + SAVE_RETRY_S
            log.error("Run queue: GO for %s not given: it could not be saved first (%s); "
                      "launches wait %.0f s.", job.name,
                      self._save_error or "queue.json is not to be overwritten", SAVE_RETRY_S)
            self._record("run_queue_launch_refused", job=job.id, reason="GO could not be saved")
            self._notify()
            return
        self._go_saved.add(job.id)
        self._warm_procs.pop(job.id, None)
        spawned, ready = float(w.get("spawned_at") or now), float(w.get("ready_at") or now)
        log.info("Run queue: GO for %s (warm %.0f s, ready for %.0f s).", job.name,
                 now - spawned, now - ready)
        self._record("run_queue_warm_go", job=job.id, attempt=w.get("attempt"),
                     reset_count=reset_count, warm_s=round(now - spawned, 1),
                     ready_s=round(now - ready, 1), py_pid=w.get("py_pid"))
        if w.get("py_pid"):
            try:
                self._set_priority(w.get("py_pid"), w.get("py_started"))
            except Exception:                         # noqa: BLE001
                log.exception("Run queue: raising %s's priority failed", job.name)
        self._launched(job, proc)
        if self._alarm is not None:
            self._clear_alarm("a job was given GO")

    def _warm_requeue(self, job: Job, proc, code) -> None:
        """The job's warm process exited after its GO was decided and saved,
        but before it was handed out: nothing of it ran.  Back in the queue
        (or cancelled, when a cancel came meanwhile)."""
        _close(proc)
        with self._lock:
            w = job.warm or {}
            self._go_saved.discard(job.id)
            self._procs.pop(job.id, None)
            self._current = None
            self._last_end_t = self._clock()
            job.pid = job.pid_started = job.launched_at = job.client_pid = None
            job.warm = None
            if job.cancel is not None:
                self._end(job, "cancelled", f"cancelled by {job.cancel.get('by')} before its "
                                            "warm process was given the machine (nothing ran)",
                          stop_chain=True)
            else:
                job.state = "queued"
                job.reason = (f"its warm process exited (code {code}) after GO was decided, "
                              "before it was handed out: back in the queue (nothing ran)")
        log.warning("Run queue: %s: %s", job.name, job.reason)
        self._record("run_queue_warm_requeued", job=job.id, attempt=w.get("attempt"), code=code,
                     state=job.state)
        self._save()
        self._notify()

    def warm_request(self, obj: Mapping) -> dict:
        """``{"stage": "warming" | "ready" | "poll", "job": "<id>:<token>",
        "warm": <warm token>, "pid", "started", "manifest"?}`` -- the warm
        process's request (:mod:`~waxx.util.device_state.queue_gate`).  Short:
        memory and queue.json only, never liveOD.  ->
        ``{"status": "ok", "go": false, "why"}`` (``resend_ready`` when the
        queue needs the manifest again) | ``{"status": "ok", "go": true,
        "reset_count"}`` | ``{"status": "ok", "abort": why}``.  GO only for
        the matching job, warm token, pid and creation time, and only once it
        is on disk; repeats from the same process are answered the same
        (idempotent); anything else is told to abort."""
        stage = obj.get("stage")
        if stage not in ("warming", "ready", "poll"):
            return {"status": "error", "msg": f"stage must be warming, ready or poll, not "
                                              f"{stage!r}"}
        jid, token = _parse_queue_job(obj.get("job"))
        wtok = str(obj.get("warm") or "")
        try:
            pid = int(obj.get("pid"))
        except (TypeError, ValueError):
            pid = None
        started = obj.get("started")
        try:
            started = None if started is None else float(started)
        except (TypeError, ValueError):
            started = None
        now = self._clock()
        changed, delivered, records = False, False, []

        def abort(why: str) -> dict:
            key = (jid, wtok)
            if key not in self._warm_abort_logged:
                self._warm_abort_logged.add(key)
                records.append(("run_queue_warm_abort_sent", {"job": jid, "pid": pid,
                                                              "why": why}))
            return {"status": "ok", "abort": why}

        with self._lock:
            job = self._jobs.get(jid) if jid is not None else None
            w = job.warm if job is not None else None
            if job is None or not token or job.token != token:
                reply = abort(f"job {obj.get('job')!r} is not a job of this run queue")
            elif w is None or not wtok or w.get("token") != wtok:
                reply = abort(f"{job.name}: this warm launch is not the queue's current one "
                              "(discarded or replaced)")
            elif pid is None:
                reply = abort("no pid given")
            elif started is None:
                # Windows reuses pids: a pid alone never names a process
                reply = abort("no process creation time given (a pid alone is never trusted)")
            elif w.get("py_pid") is not None and (
                    pid != w.get("py_pid") or w.get("py_started") is None
                    or abs(started - float(w["py_started"])) > SAME_PROCESS_S):
                reply = abort(f"{job.name}'s warm launch belongs to pid {w.get('py_pid')}, "
                              f"not {pid}")
            else:
                if w.get("py_pid") is None:
                    w["py_pid"], w["py_started"] = pid, started
                    w.setdefault("hello_at", now)
                    changed = True
                    records.append(("run_queue_warm_hello", {"job": job.id, "py_pid": pid,
                                                             "stage": stage}))
                go = w.get("go")
                if w.get("state") == "aborting":
                    reply = abort(str(w.get("why") or "discarded"))
                elif go and job.id in self._go_saved:
                    if job.cancel is not None and not go.get("delivered"):
                        reply = abort(f"cancelled by {job.cancel.get('by')} before it was "
                                      "given the machine")
                    elif job.state not in ("launching", "running"):
                        reply = abort(f"{job.name} is {job.state}")
                    else:
                        if not go.get("delivered"):
                            go["delivered"] = now
                            delivered = True
                        reply = {"status": "ok", "go": True, "reset_count": go.get("reset_count"),
                                 "liveod_instance": go.get("liveod_instance")}
                elif go:
                    reply = {"status": "ok", "go": False, "why": "GO is being saved"}
                elif job.state != "queued":
                    reply = abort(f"{job.name} is {job.state}"
                                  + (f" ({job.reason})" if job.reason else ""))
                else:
                    if stage == "ready" and w.get("state") != "ready":
                        manifest = []
                        for entry in list(obj.get("manifest") or ())[:WARM_MANIFEST_MAX]:
                            try:
                                e = [str(entry[0]), int(entry[1]), int(entry[2])]
                            except (TypeError, ValueError, IndexError):
                                continue
                            # a network path is never kept: the tick re-stats
                            # the manifest, and a hung share must not stall it
                            if not queue_gate.is_network_path(e[0]):
                                manifest.append(e)
                        w["state"], w["ready_at"], w["manifest"] = "ready", now, manifest
                        changed = True
                        records.append(("run_queue_warm_ready", {
                            "job": job.id, "py_pid": pid, "manifest_n": len(manifest),
                            "warm_s": round(now - float(w.get("spawned_at") or now), 1)}))
                    reply = {"status": "ok", "go": False, "why": self._warm_why(job, now)}
                    if stage == "poll" and w.get("state") != "ready":
                        reply["resend_ready"] = True
        if changed or delivered:
            saved = self._save()
            if delivered and not saved:
                with self._lock:
                    go = (job.warm or {}).get("go")
                    if go and go.get("delivered") == now:
                        go["delivered"] = None
                reply = {"status": "ok", "go": False,
                         "why": "queue.json could not be saved: GO waits until it can be"}
                delivered = False
        if delivered:
            log.info("Run queue: GO handed to %s (pid %s).", job.name, pid)
            records.append(("run_queue_warm_go_delivered", {"job": job.id, "py_pid": pid}))
        for kind, fields_ in records:
            self._record(kind, **fields_)
        if changed or delivered:
            self._notify()
        else:
            self.flush_journal()
        return reply

    def _warm_why(self, job: Job, now: float) -> str:
        """Why a ready warm process waits (under the lock; memory only)."""
        cur = self._jobs.get(self._current) if self._current is not None else None
        if cur is not None:
            return (f"#{cur.id} ({cur.label}) is {cur.state}"
                    + (f", run {cur.run_id}" if cur.run_id else ""))
        blocked = self._blocked_by(job, now)
        if blocked:
            return blocked
        return self._waiting or "the queue's next step"

    # -- launch -----------------------------------------------------------------------

    def _launch(self, job: Job) -> bool:
        """Launch ``job``; False when it was skipped at launch instead (the
        next eligible job may go), True otherwise.

        The job is saved as ``launching`` (with its launch time) BEFORE its
        process is started, and the process is started on a thread of its own
        outside the queue's lock: the tick follows the launch (:meth:`_follow`),
        a hung launcher stalls nothing else, and a server that dies meanwhile
        finds a ``launching`` job at restart -- never launched again, looked
        for in liveOD instead (:meth:`_follow_orphan`)."""
        # (a job with a warm launch never comes here: _schedule hands it to
        # _go_or_wait -- never a second, cold process for one job)
        try:
            sha = file_sha256(job.path)
        except OSError as exc:
            sha, unreadable = None, exc
        else:
            unreadable = None
        with self._lock:
            if job.state != "queued":                  # cancelled meanwhile
                return False
            if self._blocked_by(job, self._clock()):
                # paused, held, edited (paused / due / after) while the gate
                # polled liveOD outside the lock: it stays queued
                return False
            if self._stop_launching:                   # the server is restarting
                return False
            if job.warm is not None:                   # its warm process decides
                return True
            if unreadable is not None:
                self._end(job, "skipped", f"the file cannot be read at launch ({unreadable})")
                skipped = True
            elif sha != job.sha256 and not job.allow_drift:
                self._end(job, "skipped", "source changed since submit")
                skipped = True
            else:
                skipped = False
                log_path = self._path("logs", f"{job.id}_{job.label}.out")
                command = self._command(job)
                job.state = "launching"
                job.launched_at = self._clock()
                job.log_path = log_path
                self._current = job.id
                self._procs[job.id] = None
                self._log_pos[job.id] = 0
                self._log_partial[job.id] = ""
                self._tails[job.id] = deque(maxlen=TAIL_LINES)
                self._owe_monitor = True
                self._owed_since = None
        if skipped:
            self._save()
            self._notify()
            return False
        if not self._save():                           # "launching" is on disk first
            # without it on disk a server restart could launch the job a second
            # time: not launched; back in the queue, launches wait a while
            with self._lock:
                if job.state == "launching":
                    job.state = "queued"
                    job.reason = ("not launched: queue.json could not be saved ("
                                  + (self._save_error or "it is not to be overwritten")
                                  + ")")
                    job.launched_at = None
                    job.log_path = None
                    self._current = None
                    self._procs.pop(job.id, None)
                self._launch_blocked_until = self._clock() + SAVE_RETRY_S
            log.error("Run queue: %s not launched: its launch could not be saved first (%s); "
                      "launches wait %.0f s.", job.name,
                      self._save_error or "queue.json is not to be overwritten", SAVE_RETRY_S)
            self._record("run_queue_launch_refused", job=job.id, reason=job.reason)
            self._notify()
            return True
        self._record("run_queue_launching", job=job.id, command=command, log_path=log_path,
                     owner=job.owner, drift=sha != job.sha256)
        env = self._job_env(job)
        try:
            os.makedirs(self._path("logs"), exist_ok=True)
            with open(log_path, "ab") as f:
                f.write((f"-- run queue {time.strftime('%Y-%m-%d %H:%M:%S')}: {job.name}, "
                         f"owner {job.owner}, sha256 {sha[:12]}"
                         + (" (drift allowed: file changed since submit)"
                            if sha != job.sha256 else "")
                         + f" --\n$ {command}\n").encode("utf-8"))
        except OSError as exc:
            self._launch_failed(job, OSError(f"cannot write its log {log_path}: {exc}"))
            return True
        result: dict = {}
        record: dict = {}

        def spawn():
            try:
                proc = self._spawn(command, cwd=job.cwd, env=env, log_path=log_path)
            except BaseException as exc:              # noqa: BLE001
                result["error"] = exc
                return
            with self._lock:
                result["proc"] = proc
                abandoned = self._spawning is not record
            if abandoned:
                # the queue stopped waiting for this launch (it adopted the run
                # from liveOD, or gave the job up): its watch is not followed
                _close(proc)
        thread = threading.Thread(target=spawn, daemon=True, name=f"run-queue-launch-{job.id}")
        record.update({"job": job.id, "thread": thread, "result": result,
                       "since": self._clock(), "warned": False})
        self._spawning = record
        thread.start()
        thread.join(self._spawn_join_s)                # usually done at once
        self._follow_launch(job)
        if self._alarm is not None:
            self._clear_alarm("a job was launched")
        return True

    def _follow_launch(self, job: Job) -> None:
        """A ``launching`` job whose launch thread this server started: take
        its process when the thread is done."""
        sp = self._spawning
        if sp is None or sp["job"] != job.id:
            self._follow_orphan(job)
            return
        if sp["thread"].is_alive():
            waited = self._clock() - sp["since"]
            if waited > SPAWN_WARN_S and not sp["warned"]:
                sp["warned"] = True
                log.warning("Run queue: %s: its launch has not finished after %.0f s; the slot "
                            "stays taken (the job is looked for in liveOD).", job.name, waited)
            if waited <= LAUNCH_TIMEOUT_S + ORPHAN_WAIT_S:
                self._follow_orphan(job, keep_waiting=True)
                return
            # the launch thread never came back: stop waiting for it (a late
            # answer is closed by the thread itself) and treat the job as an
            # orphan -- found in liveOD, or ended failed now
            self._abandon_spawn()
            self._record("run_queue_launch_abandoned", job=job.id, waited_s=round(waited, 1))
            self._follow_orphan(job)
            return
        self._spawning = None
        result = sp["result"]
        if "proc" in result:
            self._launched(job, result["proc"])
            return
        exc = result.get("error")
        from waxx.util.device_state.detached import LaunchUnknown  # noqa: PLC0415
        if isinstance(exc, LaunchUnknown):
            # it may be running: not failed, not launched again -- looked for
            log.error("Run queue: %s: %s", job.name, exc)
            self._record("run_queue_launch_unknown", job=job.id, error=str(exc))
            self._follow_orphan(job)
            return
        self._launch_failed(job, exc)

    def _abandon_spawn(self) -> None:
        """Stop waiting for the launch thread: a watch it already returned is
        closed here, one it returns later is closed by the thread."""
        with self._lock:
            sp, self._spawning = self._spawning, None
            proc = (sp or {}).get("result", {}).get("proc")
        if proc is not None:
            _close(proc)

    def _launched(self, job: Job, proc) -> None:
        with self._lock:
            if job.state != "launching":
                # ended meanwhile (an orphan check found its run's outcome):
                # the process is not followed twice
                close = True
            else:
                close = False
                job.state = "running"
                job.pid = int(proc.pid)
                job.pid_started = getattr(proc, "started", None)
                self._procs[job.id] = proc
        if close:
            _close(proc)
            return
        log.info("Run queue: %s launched (pid %s).", job.name, job.pid)
        self._record("run_queue_launch", job=job.id, pid=job.pid, log_path=job.log_path,
                     owner=job.owner)
        self._save()
        self._notify()

    def _launch_failed(self, job: Job, exc) -> None:
        log.error("Run queue: could not start %s: %r", job.name, exc)
        for line in environment_report():
            log.error("  %s", line)
        with self._lock:
            if job.state == "launching":
                self._end(job, "failed", f"could not start: {exc!r}")
            self._current = None
            self._procs.pop(job.id, None)
            self._last_end_t = self._clock()
        self._save()
        self._notify()

    def _follow_orphan(self, job: Job, keep_waiting: bool = False) -> None:
        """A ``launching`` job with no process to follow: the server stopped
        while launching it, or its launcher never answered.  It is never
        launched again.  Its run is looked for in liveOD (launcher "kq",
        queue_job = "<id>:<token>"): a run in progress is adopted (its experiment's
        pid), a recorded outcome judges it; after :data:`ORPHAN_WAIT_S` with
        neither it ends failed ("server stopped while launching -- check")."""
        poll = self._last_poll
        if self._is_job_run(job, poll) and poll.get("run_in_progress"):
            proc = None
            try:
                # its creation time too (F3): a reused pid is not adopted
                proc = self._adopt(poll.get("client_pid"), poll.get("client_started"))
            except Exception:                         # noqa: BLE001
                proc = None
            if proc is not None:
                with self._lock:
                    job.run_id = int(poll["run_id"])
                    job.client_pid = poll.get("client_pid")
                    job.client_started = poll.get("client_started")
                    job.adopted = True
                if keep_waiting:
                    self._abandon_spawn()          # the launch thread's answer is not needed
                log.warning("Run queue: %s was launching; its run %s is in liveOD (pid %s): "
                            "followed.", job.name, job.run_id, job.client_pid)
                self._record("run_queue_adopted", job=job.id, pid=job.client_pid,
                             run_id=job.run_id, via="liveOD")
                self._launched(job, proc)
                return
        last = poll.get("last_outcome") if isinstance(poll, dict) else None
        if self._is_job_run(job, last):
            with self._lock:
                job.run_id = int(last["run_id"])
                job.state = "running"
                self._procs[job.id] = None
            self._finish(job, None, -1)
            return
        if keep_waiting:
            return
        started = job.launched_at or job.submitted_at
        if self._clock() - float(started or 0) < ORPHAN_WAIT_S:
            return
        with self._lock:
            if job.state != "launching":
                return
            self._end(job, "failed", "server stopped while launching -- check (no run of this "
                                     f"job appeared in liveOD within {ORPHAN_WAIT_S:.0f} s)")
            self._current = None
            self._last_end_t = self._clock()
        log.error("Run queue: %s ended failed: it was launching when the server stopped (or "
                  "its launcher never answered) and no run of it appeared in liveOD. Check "
                  "that nothing of it is running.", job.name)
        self._save()
        self._notify()

    def _take_leftover(self) -> None:
        """A job left in the slot's states (after a restart several may be):
        the one with a live process first, then the lowest id."""
        with self._lock:
            left = [j for j in self._jobs.values() if j.state in IN_SLOT]
            if not left:
                return
            left.sort(key=lambda j: (self._procs.get(j.id) is None, j.id))
            self._current = left[0].id

    # -- following the job in the slot ---------------------------------------------------

    def _read_log(self, job: Job) -> list[str]:
        if not job.log_path:
            return []
        try:
            with open(job.log_path, "rb") as f:
                f.seek(self._log_pos.get(job.id, 0))
                data = f.read()
        except OSError:
            return []
        self._log_pos[job.id] = self._log_pos.get(job.id, 0) + len(data)
        text = self._log_partial.get(job.id, "") + data.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._log_partial[job.id] = lines.pop()
        return [ln.rstrip("\r") for ln in lines]

    def _take_lines(self, job: Job, lines, final: bool = False) -> None:
        if final and self._log_partial.get(job.id):
            lines = list(lines) + [self._log_partial.pop(job.id)]
        tail = self._tails.setdefault(job.id, deque(maxlen=TAIL_LINES))
        for line in lines:
            if not line.strip():
                continue
            tail.append(line)
            m = RUN_ID_RE.search(line)
            if m and job.run_id is None:
                with self._lock:
                    job.run_id = int(m.group(1))
                log.info("Run queue: %s is run %s.", job.name, job.run_id)
                self._record("run_queue_run_id", job=job.id, run_id=job.run_id, via="output")
                self._save()
                self._notify()

    @staticmethod
    def _is_job_run(job: Job, record) -> bool:
        """A liveOD POLL reply or last_outcome record is about ``job``'s run:
        launched by the queue, with this job's id AND token (``queue_job``
        "<id>:<token>": ids start again at 1 in another queue folder, so an id
        alone could match an old run's record; sent by the
        experiment at INIT_RUN from WAXX_QUEUE_JOB)."""
        return (isinstance(record, dict) and record.get("launcher") == LAUNCHER
                and str(record.get("queue_job") or "") == job.queue_job
                and bool(record.get("run_id")))

    @staticmethod
    def _is_job_unsaved_run(job: Job, record) -> bool:
        """A liveOD last_outcome record is ``job``'s run that had no file
        (``save_data=False``: liveOD gives it run id 0, and it prints no
        "Run ID:" line), matched as :meth:`_is_job_run` does by launcher, job
        id and token."""
        return (isinstance(record, dict) and record.get("launcher") == LAUNCHER
                and bool(job.queue_job)
                and str(record.get("queue_job") or "") == job.queue_job
                and record.get("run_id") == 0)

    def _identify(self, job: Job, poll) -> None:
        """The job's run id and experiment pid from liveOD (``queue_job``) --
        the primary way; the "Run ID:" line in its output is the fallback (an
        older liveOD or experiment that does not send ``queue_job``)."""
        if not self._is_job_run(job, poll):
            return
        changed = False
        with self._lock:
            if job.run_id is None:
                job.run_id = int(poll["run_id"])
                changed = True
            if job.client_pid is None and poll.get("client_pid"):
                job.client_pid = poll.get("client_pid")
                changed = True
            if (job.client_started is None and poll.get("client_started") is not None
                    and poll.get("client_pid") == job.client_pid):
                job.client_started = poll.get("client_started")
                changed = True
        if changed:
            if job.run_id == poll.get("run_id"):
                self._record("run_queue_run_id", job=job.id, run_id=job.run_id, via="liveOD",
                             client_pid=job.client_pid)
            self._save()
            self._notify()

    def _note_no_file(self, job: Job, poll) -> None:
        """liveOD's run in progress is ``job``'s run with no file
        (``save_data=False``): mark the job ``no_file``."""
        if (job.no_file or job.run_id is not None or not isinstance(poll, dict)
                or not poll.get("run_in_progress") or not self._is_job_unsaved_run(job, poll)):
            return
        with self._lock:
            job.no_file = True
        self._save()
        self._notify()

    def _follow(self, job: Job) -> None:
        if job.state == "launching":
            self._take_lines(job, self._read_log(job))
            self._identify(job, self._last_poll)
            self._note_no_file(job, self._last_poll)
            self._follow_launch(job)
            return
        self._take_lines(job, self._read_log(job))
        poll = self._last_poll
        self._identify(job, poll)
        self._note_no_file(job, poll)
        if (job.run_id is not None and job.client_pid is None and isinstance(poll, dict)
                and poll.get("run_id") == job.run_id and poll.get("client_pid")):
            with self._lock:
                job.client_pid = poll.get("client_pid")
                if poll.get("client_started") is not None:
                    job.client_started = poll.get("client_started")
            self._save()
        if job.cancel is not None and not job.cancel.get("abort_sent") and job.kill is None:
            # (never after a kill: an Abort would discard the file liveOD keeps)
            self._send_abort(job)
        proc = self._procs.get(job.id)
        code = proc.poll() if proc is not None else -1
        if code is None:
            return
        go = (job.warm or {}).get("go")
        if go and not go.get("delivered"):
            self._warm_requeue(job, proc, code)
            return
        self._finish(job, proc, code)

    def _finish(self, job: Job, proc, code) -> None:
        with self._lock:
            job.state = "ending"
        self._save()
        self._notify()
        self._take_lines(job, self._read_log(job), final=True)
        unsaved = None
        if job.run_id is None:
            fresh = self._poll_now()
            if isinstance(fresh, dict):
                self._identify(job, fresh)
                self._identify(job, fresh.get("last_outcome"))
                if job.run_id is None and self._is_job_unsaved_run(job, fresh.get("last_outcome")):
                    unsaved = dict(fresh["last_outcome"])
        known = proc is not None and getattr(proc, "exit_code_known", True) and code != -1
        exit_code = int(code) if known else None
        tail = list(self._tails.get(job.id, ()))
        stem = Path(job.path).stem
        told = None
        if self._run_exited is not None and self._poll_fn is not None:
            told = tell_live_od_exited(self._poll_fn, self._run_exited, job.run_id, exit_code,
                                       tail, expt=stem, started=job.launched_at,
                                       now=self._clock(), who=f"Run queue {job.name}",
                                       sender="the monitor server's run queue")
            if told is not None and told["status"] != "not_sent":
                self._record("run_queue_told_live_od_exited", job=job.id,
                             run_id=told["run_id"], ok=told["ok"], status=told["status"])
        if self._poll_fn is not None:
            v = judge_run(exit_code, tail, job.run_id, stem,
                          lambda rid: live_od_outcome(self._poll_fn, rid,
                                                      wait_s=self._outcome_wait_s),
                          unsaved=unsaved)
        else:
            v = judge_run(exit_code, tail, job.run_id, stem, lambda rid: None, unsaved=unsaved)
        outcome = {"outcome": (v.outcome or {}).get("outcome"),
                   "detail": (v.outcome or {}).get("detail") or "", "why": v.why,
                   "exit_code": exit_code, "aborted": v.aborted,
                   "lost_core": not v.start_monitor,
                   "told_live_od": None if told is None else told["status"]}
        aborted_run = ABORTED_RE.search("\n".join(tail))
        with self._lock:
            job.exit_code = exit_code
            job.outcome = outcome
            if (unsaved is not None or v.nothing_written
                    or (job.run_id is None and aborted_run and aborted_run.group(1) == "0")):
                job.no_file = True
            killed_by = (job.kill or {}).get("by")
            if v.saved:
                state = "done" if v.nothing_written else "saved"
                reason = v.why
                if job.kill is not None:
                    reason = (f"{'ran to its end' if v.nothing_written else 'saved'} before "
                              f"the force kill by {killed_by}")
                elif job.cancel is not None:
                    reason = (f"{'ran to its end' if v.nothing_written else 'saved'} before "
                              f"the cancel by {job.cancel.get('by')} reached it"
                              + (f" ({job.cancel['abort_note']})"
                                 if job.cancel.get("abort_note") else ""))
            elif job.kill is not None:
                state = "cancelled"
                reason = (f"force-killed by {killed_by} ({job.kill.get('note')}); no cleanup "
                          f"ran: {v.why}")
            elif job.cancel is not None:
                state = "cancelled"
                reason = f"cancelled by {job.cancel.get('by')} while running: {v.why}"
            elif v.aborted:
                state, reason = "aborted", v.why
            else:
                state, reason = "failed", v.why
            if job.kill is not None:
                pass                                  # the kill's reason stands
            elif proc is None and job.adopted is False and state == "failed":
                reason = f"server restarted; process gone ({v.why})"
            elif proc is None and state == "saved":
                reason = (f"its process ended while the monitor server was down; liveOD "
                          f"recorded run {job.run_id} saved")
            elif proc is None and state == "done":
                reason = ("its process ended while the monitor server was down; liveOD "
                          "recorded its run (save_data=False) as ended with nothing written")
            self._end(job, state, reason,
                      stop_chain=state in ("failed", "aborted", "cancelled"))
            self._current = None
            self._last_end_t = self._clock()
            self._last_ended_run_id = job.run_id
            self._procs.pop(job.id, None)
        if proc is not None and hasattr(proc, "close"):
            try:
                proc.close()
            except Exception:                         # noqa: BLE001
                pass
        ok = state in SUCCEEDED
        (log.info if ok else log.warning)(
            "Run queue: %s %s%s%s", job.name, state if ok else state.upper(),
            f" (run {job.run_id})" if job.run_id is not None else "",
            f": {reason}" if reason else "")
        if not ok and tail:
            for line in tail[-8:]:
                log.warning("    | %s", line)
        self._save()
        self._notify()

    def _send_abort(self, job: Job) -> None:
        """liveOD's Abort for a running job being cancelled -- only when
        liveOD's run in progress is that job's run, and it is not being saved.
        Sent once (the request thread and the tick may both try)."""
        with self._abort_lock:
            self._send_abort_once(job)

    def _cancel_note(self, job: Job, **fields) -> None:
        with self._lock:
            if job.cancel is not None:
                job.cancel.update(fields)

    def _send_abort_once(self, job: Job) -> None:
        with self._lock:
            pending = job.cancel is not None and not job.cancel.get("abort_sent")
        if not pending:
            return
        # a run with no file (save_data=False) is run id 0 in liveOD, like every
        # such run: it is named by the job's queue_job too, on the POLL and in
        # the RESET (a liveOD from before 2026-10-19 ignores it in the RESET:
        # the POLL just before is then the only check)
        no_file = job.run_id is None and job.no_file
        target = 0 if no_file else job.run_id
        name = "its run (save_data=False, run id 0)" if no_file else f"run {target}"
        if target is None:
            self._cancel_note(job, abort_note="no run id yet: the Abort goes once the run has "
                                              "one")
            return
        poll = self._poll_now()
        if poll is None:
            self._cancel_note(job, abort_note="liveOD is not reachable: the Abort is not sent "
                                              "yet")
            return
        if (not poll.get("run_in_progress") or poll.get("run_id") != target
                or (no_file and not self._is_job_unsaved_run(job, poll))):
            current = poll.get("run_id") if poll.get("run_in_progress") else "none"
            if no_file and poll.get("run_in_progress") and current == 0:
                current = f"another run with no file (queue job {poll.get('queue_job') or '?'})"
            self._cancel_note(job, abort_note=f"liveOD's run in progress is {current}, not "
                                              f"{name if no_file else target}: no Abort sent")
            return
        if poll.get("save_in_progress") or poll.get("run_state") == "saving":
            self._cancel_note(job, abort_note=f"liveOD is saving {name}: no Abort sent")
            return
        if poll.get("reset_requested"):
            self._cancel_note(job, abort_sent=True, abort_note="an Abort was already pending")
            return
        if self._live_od_reset is None:
            self._cancel_note(job, abort_note="this queue has no way to send liveOD's Abort")
            return
        if not no_file:
            # (not run id 0: an older liveOD's Reset of any later run-0 run
            # would be taken for the queue's own -- person_hold.observe_poll)
            with self._lock:
                self._own_abort_ids.add(target)
        try:
            if no_file:
                reply = self._live_od_reset(run_id=0, source="queue", queue_job=job.queue_job)
            else:
                reply = self._live_od_reset(run_id=target, source="queue")
        except Exception as exc:                      # noqa: BLE001
            self._cancel_note(job, abort_note=f"sending the Abort failed: {exc}")
            log.error("Run queue: liveOD's Abort for %s (%s) failed: %s", name, job.name, exc)
            return
        ok = bool(isinstance(reply, dict) and reply.get("ok"))
        if isinstance(reply, dict) and reply.get("refused"):
            # liveOD's run is no longer this job's (it ended, or another
            # started): nothing was aborted; tried again while the job runs
            self._cancel_note(job, abort_note=f"liveOD refused the Abort: {reply.get('error')}")
            log.warning("Run queue: liveOD refused the Abort for %s (%s): %s", name, job.name,
                        reply.get("error"))
            return
        self._cancel_note(job, abort_sent=True,
                          abort_note="Abort sent" if ok else f"liveOD refused: {reply}")
        log.warning("Run queue: liveOD's Abort sent for %s (%s): %s", name, job.name,
                    "acknowledged" if ok else reply)
        self._record("run_queue_abort_sent", job=job.id, run_id=target, ok=ok,
                     by=job.cancel.get("by"), no_file=no_file)
        self._save()

    # -- ending jobs and chains ---------------------------------------------------------------

    def _end(self, job: Job, state: str, reason: str, stop_chain: bool | None = None) -> None:
        """Put a job in an ended state (under the lock), journal it, and stop
        its chain (``stop_chain`` None: when it failed or was skipped; a
        running job that was cancelled stops it too)."""
        job.state = state
        job.reason = reason
        job.ended_at = self._clock()
        self._record("run_queue_end", job=job.id, state=state, reason=reason,
                     run_id=job.run_id, exit_code=job.exit_code, outcome=job.outcome)
        if stop_chain is None:
            stop_chain = state in ("failed", "skipped")
        if stop_chain and job.chain and job.stop_on_failure:
            for other in sorted(self._jobs.values(), key=lambda j: j.id):
                if other.chain == job.chain and other.state == "queued" and other.id != job.id:
                    other.state = "cancelled"
                    other.reason = (f"chain {job.chain} stopped: {job.name} {state}"
                                    + (f" ({reason})" if reason else ""))
                    other.ended_at = job.ended_at
                    self._record("run_queue_end", job=other.id, state="cancelled",
                                 reason=other.reason, run_id=None, exit_code=None,
                                 outcome=None)

    # -- loops and the monitor ----------------------------------------------------------------

    @staticmethod
    def _may_stop(job: Job, loop) -> bool:
        """A person's job stops any loop; an agent's job a *yielding* loop,
        whoever started it -- never a *holding* one (run_loop.LOOP_CATEGORIES;
        a loop that names no category yields).  Before 2026-10-20 this went by
        the loop's owner: an agent's job stopped only a loop the queue or an
        agent started."""
        if job.owner == "person":
            return True
        return info_category(loop.info()) == "yields"

    def loop_stopped_by_someone(self, key: str, who: str) -> None:
        """Someone (not the queue) asked a loop to stop: if the queue meant
        to start that loop again once it runs out, it no longer does."""
        with self._lock:
            resume = self._resume_loop
            if resume is None or resume.get("key") != key:
                return
            self._resume_loop = None
        log.info("Run queue: %s stopped the loop %s: the queue will not start it again.", who,
                 key)
        self._record("run_queue_loop_resume_cancelled", loop=key, by=who)
        self._save()
        self._notify()

    def _stop_loop_for_jobs(self, loop, job: Job) -> None:
        info = loop.info()
        if info.get("state") != "running":
            # already stopping: by the queue (asked before), or by a person --
            # whose stop it is, and who did not ask for it to start again
            return
        category, owner = info_category(info), effective_owner(info)
        reply = loop.stop(operator="run queue", client=self._host, start_monitor=False,
                          owner="queue")
        if reply.get("status") != "ok":
            return
        with self._lock:
            # the category and owner it had: the restart keeps them (a holding
            # loop a person's job stopped comes back holding, for that person)
            self._resume_loop = {"key": loop.spec.key,
                                 "path": loop.path if loop.spec.pick else None,
                                 "since": self._clock(), "category": category,
                                 "owner": owner}
        log.info("Run queue: %s (%s) stops after its run in progress for %s; it starts again "
                 "when the queue has run out.", loop.spec.title, category, job.name)
        self._record("run_queue_loop_stop", loop=loop.spec.key, for_job=job.id,
                     job_owner=job.owner, category=category, loop_owner=owner)
        self._save()
        self._notify()

    def _drained(self, active_loop) -> None:
        """No job eligible, none in the slot: start the loop the queue stopped
        (no hold, nothing paused), else ask for the monitor if the queue has
        run jobs since it last ran out."""
        if active_loop is not None or self._stop_launching:
            # (the server is restarting: no loop and no monitor started from here)
            return
        resume = self._resume_loop
        blocked = ""
        if resume is not None:
            blocked = (self.hold.text() if self.hold.active else
                       ("jobs are paused" if any(self._paused.values()) else ""))
            if blocked:
                pass                                  # kept until the hold / pause lifts
            else:
                with self._lock:
                    self._resume_loop = None
                loop = self._loops.get(resume.get("key"))
                info = loop.info() if loop is not None else {}
                text = str(info.get("text") or "")
                if loop is None:
                    self._record("run_queue_loop_not_resumed", loop=resume.get("key"),
                                 why="the server has no such loop now")
                elif info.get("state") != "stopped" or "stopped by run queue" not in text:
                    # it latched off (or someone else ended it): it stays off
                    log.warning("Run queue: %s is not started again: it ended %s (%s).",
                                loop.spec.title, info.get("state"), text)
                    self._record("run_queue_loop_not_resumed", loop=loop.spec.key,
                                 state=info.get("state"), why=text)
                else:
                    # the category it had (a resume record from before
                    # 2026-10-20 has none: the default, "yields")
                    category = resume.get("category") or None
                    reply = loop.start(operator="run queue", client=self._host,
                                       path=resume.get("path"), owner="queue",
                                       category=category,
                                       on_behalf_of=resume.get("owner"))
                    ok = reply.get("status") == "ok"
                    self._record("run_queue_loop_resume", loop=loop.spec.key, ok=ok,
                                 msg=reply.get("msg", ""), category=category,
                                 on_behalf_of=resume.get("owner"))
                    if ok:
                        log.info("Run queue: out of jobs -- %s started again.", loop.spec.title)
                        self._owe_monitor = False
                    else:
                        log.warning("Run queue: out of jobs; %s could not start again: %s",
                                    loop.spec.title, reply.get("msg"))
                self._save()
                self._notify()
        if self._owe_monitor and (self._resume_loop is None or blocked):
            # (a loop waiting on a hold or a pause starts later, taking the core
            # from the monitor as any run does)
            self._owe_monitor = False
            self._record("run_queue_monitor_start")
            if self._start_monitor is not None:
                try:
                    self._start_monitor("the run queue has no job to run")
                except Exception:                     # noqa: BLE001
                    log.exception("Run queue: could not ask for the monitor")

    # -- the alarm ------------------------------------------------------------------------

    def _alarm_tick(self, now: float) -> None:
        with self._lock:
            cur = self._jobs.get(self._current) if self._current is not None else None
            stuck = (cur is not None and cur.state == "launching"
                     and now - float(cur.launched_at or now) > SPAWN_WARN_S)
        if stuck:
            self._raise_alarm(now, cur, now - float(cur.launched_at),
                              f"{cur.name} has been launching for "
                              f"{(now - float(cur.launched_at)) / 60.0:.1f} min (its launcher "
                              "has not answered and no run of it is in liveOD)")
            return
        with self._lock:
            owed = self._current is None and bool(self._order(now))
            nxt = self._order(now)[0] if owed else None
        if owed and self._gate_kind != "blocked":
            # someone's run is legitimately using the machine (or it is free
            # and the job launches): no alarm, and the clock starts again
            self._owed_since = None
            if self._alarm is not None:
                self._clear_alarm("the machine is in legitimate use")
            return
        if not owed:
            self._owed_since = None
            if self._alarm is not None:
                self._clear_alarm("no job is waiting to start")
            return
        if self._owed_since is None:
            self._owed_since = now
            return
        waited = now - self._owed_since
        if waited <= self.alarm_s:
            return
        self._raise_alarm(now, nxt, waited,
                          f"{nxt.name} has been ready to start for {waited / 60.0:.0f} min and "
                          f"nothing has launched -- {self._waiting or 'the reason is not known'}")

    def _raise_alarm(self, now: float, job: Job, waited: float, text: str) -> None:
        """One WARNING and journal record per alarm_s while the alarm holds."""
        if self._alarm is not None and now - self._alarm["warned"] < self.alarm_s:
            return
        first = self._alarm is None
        why = self._waiting if job.state == "queued" else text
        self._alarm = {"since": now - waited, "warned": now, "job": job.id,
                       "waited_s": waited, "why": why}
        log.warning("RUN QUEUE ALARM: %s", text)
        self._record("run_queue_alarm", job=job.id, waited_s=round(waited, 1), why=why,
                     first=first)
        self._notify()

    def _clear_alarm(self, why: str) -> None:
        alarm, self._alarm = self._alarm, None
        log.info("Run queue alarm cleared: %s.", why)
        self._record("run_queue_alarm_cleared", why=why,
                     waited_s=round(self._clock() - alarm["since"], 1) if alarm else None)
        self._notify()

    # -- persistence -----------------------------------------------------------------------

    def _save(self) -> bool:
        """Write queue.json.  Two threads may save at once (a request and the
        tick): each snapshot gets a sequence number under the queue's lock, and
        a snapshot older than the one last written is not written -- the file
        never goes back to an earlier state.  Deadlock-free whatever lock the
        caller holds (the save lock is never held while taking the queue's).
        True when this state is on disk (written now, or a newer one already
        is); False when nothing could be written (no folder, the file is not
        to be overwritten, or the write failed)."""
        if not self.enabled or self._no_save:
            return False
        with self._lock:
            self._save_seq += 1
            seq = self._save_seq
            jobs = sorted(self._jobs.values(), key=lambda j: j.id)
            ended = [j for j in jobs if j.state in ENDED]
            keep = {j.id for j in ended[-KEEP_ENDED:]} | {j.id for j in jobs
                                                          if j.state not in ENDED}
            data = {"version": 1, "saved_at": self._clock(), "next_id": self._next_id,
                    "jobs": [j.to_dict() for j in jobs if j.id in keep],
                    "paused": {k: (dict(v) if v else None) for k, v in self._paused.items()},
                    "resume_loop": dict(self._resume_loop) if self._resume_loop else None,
                    "own_abort_ids": sorted(self._own_abort_ids)[-200:], "seq": seq}
        with self._save_lock:
            if seq <= self._saved_seq:
                return True               # a newer snapshot is already on disk
            try:
                from waxx.util.device_state.state_file_io import atomic_write  # noqa: PLC0415
                os.makedirs(self.directory, exist_ok=True)
                atomic_write(self._path("queue.json"), data)
                self._saved_seq = seq
                return True
            except Exception as exc:                  # noqa: BLE001
                log.error("Run queue: could not store the queue in %s (%s).",
                          self._path("queue.json"), exc)
                self._save_error = str(exc)
                return False

    def _load_failed(self, exc) -> None:
        """queue.json could not be read: fail closed.  The file is moved aside
        (timestamped, never overwritten -- if it cannot be moved, the queue
        never writes queue.json this session), the queue starts empty with
        every scope paused (a person looks, then resumes), and job ids go on
        from the journal's highest."""
        from waxx.util.device_state.person_hold import move_aside  # noqa: PLC0415
        original = self._path("queue.json")
        aside = move_aside(original)
        if not aside:
            self._no_save = True
        why = (f"queue.json could not be read ({exc}); "
               + (f"moved to {aside}" if aside else "left in place, not overwritten")
               + " -- check it, then resume")
        now = self._clock()
        for scope in PAUSE_SCOPES:
            self._paused[scope] = {"by": "monitor server", "since": now, "reason": why,
                                   "owner": "person"}
        self._next_id = self._next_id_from_journal()
        log.error("Run queue: %s. The queue starts EMPTY and PAUSED (all jobs); the journal "
                  "has the jobs' history.", why)
        self._record("run_queue_load_failed", error=str(exc), moved_to=aside or None,
                     next_id=self._next_id)
        self._record("run_queue_pause", scope="all", by="monitor server", reason=why)
        self._save()

    def _next_id_from_journal(self) -> int:
        highest = 0
        try:
            with open(self._path("journal.jsonl"), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        highest = max(highest, int(json.loads(line).get("job") or 0))
                    except (ValueError, TypeError, AttributeError):
                        continue
        except OSError:
            pass
        return highest + 1

    def _load(self) -> None:
        if not self.enabled:
            return
        try:
            with open(self._path("queue.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            jobs = [Job.from_dict(d) for d in data.get("jobs") or []]
            next_id = max([int(data.get("next_id") or 1)] + [j.id + 1 for j in jobs])
        except FileNotFoundError:
            return
        except Exception as exc:                      # noqa: BLE001
            self._load_failed(exc)
            return
        self._jobs = {j.id: j for j in jobs}
        self._next_id = next_id
        if any(j.state == "queued" and j.rank is None for j in jobs):
            # a queue.json from before ranks: the queued jobs keep their id order
            for k, j in enumerate(sorted((j for j in jobs if j.state == "queued"),
                                         key=lambda j: j.id)):
                j.rank = float(k + 1)
        for scope in PAUSE_SCOPES:
            self._paused[scope] = (data.get("paused") or {}).get(scope) or None
        self._resume_loop = data.get("resume_loop") or None
        self._own_abort_ids = set(data.get("own_abort_ids") or ())
        self._load_warm(jobs)
        for job in sorted(jobs, key=lambda j: j.id):
            if job.state not in IN_SLOT or job.id in self._procs:
                continue
            self._log_pos[job.id] = 0
            self._log_partial[job.id] = ""
            self._tails[job.id] = deque(maxlen=TAIL_LINES)
            self._owe_monitor = True
            if job.state == "launching":
                # never launched again: looked for in liveOD (_follow_orphan)
                self._procs[job.id] = None
                log.warning("Run queue: %s was launching when the server stopped: not launched "
                            "again; its run is looked for in liveOD.", job.name)
                self._record("run_queue_launching_at_restart", job=job.id,
                             launched_at=job.launched_at)
                continue
            proc = None
            try:
                proc = self._adopt(job.pid, job.pid_started)
            except Exception:                         # noqa: BLE001
                proc = None
            job.state = "running"
            self._procs[job.id] = proc
            if proc is not None:
                job.adopted = True
                log.warning("Run queue: %s (pid %s, run %s) was running when the server "
                            "stopped and still is: followed again.", job.name, job.pid,
                            job.run_id)
                self._record("run_queue_adopted", job=job.id, pid=job.pid, run_id=job.run_id)
            else:
                # judged when it comes to the slot, from liveOD's record
                log.warning("Run queue: %s (pid %s, run %s) was running when the server "
                            "stopped; its process is gone -- judged from liveOD's record.",
                            job.name, job.pid, job.run_id)
                self._record("run_queue_process_gone", job=job.id, pid=job.pid,
                             run_id=job.run_id)
        self._take_leftover()
        self._save()

    def _load_warm(self, jobs) -> None:
        """At a restart: the warm launches in queue.json.  A live one (no GO)
        is taken back by pid + creation time, or -- recorded but gone -- its
        record goes (the job stays queued; nothing of it ran); one with no
        pid recorded yet waits for the process's hello.  A job whose GO was
        saved: its process taken back runs on (the GO is handed out again if
        it was not); gone before the hand-out, the job goes back in the queue;
        gone after it, the usual orphan / process-gone path judges it."""
        now = self._clock()
        for job in sorted(jobs, key=lambda j: j.id):
            w = job.warm
            if w is None:
                continue
            if not isinstance(w, dict):
                job.warm = None
                continue
            go = w.get("go")
            if go:
                self._go_saved.add(job.id)            # on disk, by definition
                if job.state != "launching":
                    continue                          # running: adopted by job.pid below
                proc = self._adopt_warm(w)
                if proc is not None:
                    job.state = "running"
                    job.pid = int(proc.pid)
                    job.pid_started = getattr(proc, "started", None)
                    job.adopted = True
                    self._procs[job.id] = proc
                    self._log_pos[job.id] = int(w.get("log_offset") or 0)
                    self._log_partial[job.id] = ""
                    self._tails[job.id] = deque(maxlen=TAIL_LINES)
                    self._owe_monitor = True
                    log.warning("Run queue: %s had its GO decided when the server stopped; its "
                                "warm process is alive: followed.", job.name)
                    self._record("run_queue_warm_adopted", job=job.id, go=True, pid=job.pid)
                elif not go.get("delivered"):
                    self._go_saved.discard(job.id)
                    job.state, job.launched_at, job.warm = "queued", None, None
                    job.reason = ("its warm process was gone at a server restart, before its "
                                  "GO was handed out: back in the queue (nothing ran)")
                    log.warning("Run queue: %s: %s", job.name, job.reason)
                    self._record("run_queue_warm_requeued", job=job.id, at_restart=True)
                # else: delivered and gone -- the launching (orphan) path below
                continue
            if job.state in ENDED:
                if w.get("state") != "aborting":
                    w.update(state="aborting", why=f"{job.name} is {job.state}", aborted_at=now)
            w["adopt_from"] = now
            proc = self._adopt_warm(w)
            if proc is not None:
                self._warm_procs[job.id] = proc
                log.warning("Run queue: %s's warm process (%s) is alive after the restart: "
                            "followed.", job.name, w.get("state"))
                self._record("run_queue_warm_adopted", job=job.id, go=False,
                             state=w.get("state"), pid=getattr(proc, "pid", None))
            elif w.get("cmd_pid") or w.get("py_pid"):
                job.warm = None
                log.warning("Run queue: %s's warm process is gone after the restart: its record "
                            "is dropped (nothing of it ran).", job.name)
                self._record("run_queue_warm_gone_at_restart", job=job.id, state=w.get("state"))
            # else: no pid known yet -- its hello is waited for (WARM_HELLO_WAIT_S)

    # -- out ------------------------------------------------------------------------------

    def _record(self, kind: str, **fields) -> None:
        ts = self._clock()
        entry = {"ts": ts, "t": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
                 "kind": kind}
        entry.update(fields)
        if self.enabled:
            try:
                os.makedirs(self.directory, exist_ok=True)
                with open(self._path("journal.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=repr) + "\n")
            except OSError as exc:
                log.error("Run queue: could not append to its journal (%s).", exc)
        if self._journal is not None:
            # the ops journal appends to the lab's share synchronously: the copy
            # is written by flush_journal(), never while the queue's lock is held
            self._mirror.append((kind, fields))

    def flush_journal(self) -> None:
        """Copy the records made since the last flush to the ops journal.
        Called at the end of every request and tick, outside the lock."""
        if self._journal is None:
            return
        is_owned = getattr(self._lock, "_is_owned", None)
        if is_owned is not None and is_owned():
            return                        # under the lock: the next flush writes them
        with self._mirror_lock:
            while self._mirror:
                kind, fields = self._mirror.popleft()
                try:
                    self._journal.record(kind, **fields)
                except Exception:                     # noqa: BLE001
                    log.exception("Could not journal %s", kind)

    def _notify(self) -> None:
        self.flush_journal()
        if self._on_change is None:
            return
        try:
            self._on_change(self.info())
        except Exception:                             # noqa: BLE001
            log.exception("Run queue change notification failed")
