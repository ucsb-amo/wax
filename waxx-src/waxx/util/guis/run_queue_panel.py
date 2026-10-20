"""The run queue panel: what the monitor server's run queue holds, and the
person's controls over it (:mod:`waxx.util.device_state.run_queue`).

:class:`RunQueuePanel` is a reusable widget.  It is fed by its host:

* :meth:`RunQueuePanel.set_state` with the server's ``status_json`` dict (its
  ``run_queue`` and ``person_hold``), ``None`` when the server does not answer;
* :meth:`RunQueuePanel.on_broadcast` with the server's broadcasts
  (``{"type": "run_queue"}``, ``{"type": "person_hold"}``; others are ignored);

and it acts through an injected ``requester(request_dict) -> reply_dict``
(run off the GUI thread by a
:class:`~waxx.util.guis.request_runner.RequestRunner`).  It never discovers the
server or opens a socket itself.  In the monitor server's own window the
requester is a direct call into the server; elsewhere it can be a
MonitorClient-backed callable.

What it shows:

* a top strip: the queue's state pill (idle / running / launching / waiting /
  paused / held), its text and counts; the alarm banner while ``alarm`` is set
  (which job, why, how long); the person hold (since, by, owner, source,
  reason) with Hold / Release; the pause of agent jobs and of all jobs with
  Pause / Resume; the note that the queue will start a run loop it stopped
  (``resume_loop``);
* the jobs, one row each, in the server's order (phase 1b: ended, the slot,
  then the queued jobs by ``position``; before it: the list's order):
  place (the server's 0-based ``position`` shown from 1; "slot" for the job
  in it), id, state ("(paused)" for a job paused alone), owner / submitter,
  label, class (``expt_class``; "[cal]" when it declares calibrations), file
  (elided, full path in the tooltip), run id, priority (a placement hint
  only; the rank is in its tooltip), due, after / chain, the estimate
  (``estimate`` = duration / eta_start / eta_end / basis, marked "est."), a
  source-changed warning, why it waits (``waiting``; derived from the job's
  fields for a server that does not send it), and when / by whom it was
  submitted.  Fields a server does not send are blank.

Layout: one toolbar (the state pill and text, the counts, the filter -- a
dropdown: All; Last N, the default: the last N ended jobs (N from the spin
box beside it, default :data:`DEFAULT_LAST_N`) and every job not ended;
Pending, without the ended jobs --, Follow, Hold / Release, Pause / Resume
agent and all jobs, Refresh); banners only while there is something to say
(the alarm, the person hold, a pause, the loop the queue stopped); the table
(rows tinted: saved green, failed red, the job in the slot blue and bold, an
ended run that had no file -- ``save_data=False``, done / aborted /
cancelled; a failed one stays red -- gray; less-used columns hidden until the
header's right-click menu shows them; a column moves by dragging its
header); a message line.  With Follow on (the default) the table scrolls to
its bottom when a list brings a job the previous list did not have (a newly
queued job).  Given a ``settings`` (a QSettings), the panel remembers the
columns' order, widths and shown set, the filter, N and Follow, on every
change (:func:`panel_settings`).

A job's actions are on its right-click menu (:meth:`RunQueuePanel.job_menu`;
what each allows: :meth:`RunQueuePanel.action_state`).  With several rows
selected, Run again / Run again next, Move to the top / bottom, Remove and
Copy (paths, run ids) act on all of them; the others on the row clicked:

* Show log (also Enter / double-click): a window following the job's log
  through the ``tail`` action -- every 0.5 s while it runs, every 2 s while
  it is queued, until it has ended and its log is read to the end;
* Open source (VS Code, when this PC can read the file) / View source (a
  read-only window).  A ``.py`` is never handed to the OS's default action
  (it can run it).  Both say when the file now differs from the one the job
  was submitted with; what a run ran is in its run file (``expt_file``);
* Run again / Run again next (no question; what to know -- a changed or
  unreadable file, what is not carried over -- goes to the message line):
  the same file as a new person's job (``submit``, or ``insert`` at 0) with
  its argv, working folder, label, write-back veto and drift setting -- not
  its dependencies, chain, due time or repeats; this PC's hash of the file
  goes with it, as kq sends it.  Several jobs go in in table order (next:
  inserted last first, so they run in table order ahead of the queue);
* Edit (``edit`` with ``fields``: argv, label, after, chain, stop_on_failure,
  the write-back veto, due, allow_drift, paused; only the fields changed) and
  Move up / down / to top / to bottom (also Ctrl+Up / Ctrl+Down; ``move``: up
  / down name the neighbour by id -- ``before_id`` / ``after_id`` -- top /
  bottom send ``to_index``; several selected queued jobs move to the top or
  bottom together, keeping their order), for a queued job;
* Remove from queue (also Del; several selected at once; asks first): each
  queued job is cancelled ``queued_only``, so one that launched meanwhile is
  never aborted on a question about a queued job.  The queue keeps the
  records; the Pending filter hides them;
* Abort run, for the job in the slot (asks first: liveOD's Abort discards
  that run's data file -- nothing else is offered until the queue can stop a
  run and keep its data);
* Force kill, for a launching or running job, when the Abort cannot work
  (liveOD closed, the experiment's terminal or process hung, the queue out of
  step): asks first, then ``kill`` -- the server kills the job's processes;
  no cleanup runs, and liveOD leaves the run's file as it was (not saved, not
  deleted).  Offered while a cancel is pending too;
* Copy: a kq command line, the file's path, the run id.

An action the server refuses as unknown (an older server) is disabled.

Every request that changes something carries ``owner`` "person" (a person is
clicking) and ``by`` = ``user@host``.

Requests: a full ``list`` when the panel is first shown and after every
``run_queue`` broadcast (debounced, :data:`LIST_DEBOUNCE_MS`), and after
``set_state`` sees the queue's summary change; the summary between comes from
``status_json``.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import socket
import subprocess
import time
from pathlib import Path, PurePath
from typing import Any, Callable

from PyQt6.QtCore import QAbstractTableModel, QByteArray, QModelIndex, QSettings, QTimer, Qt
from PyQt6.QtGui import QAction, QColor, QFont, QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox,
    QPlainTextEdit, QPushButton, QSizePolicy, QSpinBox, QStyle, QTableView, QToolButton,
    QVBoxLayout, QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.device_state.person_hold import describe as describe_hold
from waxx.util.device_state.run_queue_client import (
    ENDED, IN_SLOT, SUCCEEDED, default_by, local_sha256)
from waxx.util.guis.request_runner import RequestRunner, is_unknown_request
from waxx.util.guis.qt_upkeep import delete_later, set_style_if_changed

#: The coalescing delay between a ``run_queue`` broadcast and the ``list``.
LIST_DEBOUNCE_MS = 250
#: How often a job's log window asks for new lines while the job is in the
#: slot, and while it is queued (it has no log yet).
TAIL_RUNNING_MS = 500
TAIL_QUEUED_MS = 2000
#: Lines a log window keeps.
LOG_MAX_LINES = 20000
#: Jobs asked for in one ``list``.
LIST_LIMIT = 200
#: Who acts in this panel: a person is clicking.
OWNER = "person"

#: The 1b actions a server may not have yet (disabled when refused as unknown).
MOVE_ACTION = "move"
EDIT_ACTION = "edit"
KILL_ACTION = "kill"
#: The panel's four moves (:func:`move_request` turns each into the server's
#: ``before_id`` / ``after_id`` / ``to_index``).
MOVE_WHERE = ("up", "down", "top", "bottom")
_MOVE_TIPS = {"up": "Move it one place earlier", "down": "Move it one place later",
              "top": "Make it the next job", "bottom": "Move it to the end of the queue"}

OK_TEXT = "#5fd38d"
WARN_TEXT = "#f0c14b"
ERR_TEXT = "#ff6b6b"

_PILL_BG = {"on": theme.OK, "off": "#4d4d4d", "partial": "#8a6a12", "warn": "#8a6a12",
            "hazard": "#c62828", "unknown": "#454545"}
#: queue state -> pill level
_QUEUE_LEVEL = {"idle": "off", "running": "on", "launching": "partial", "waiting": "partial",
                "paused": "warn", "held": "warn"}
#: job state -> text colour of its state cell
_STATE_COLOR = {"queued": theme.FG, "launching": WARN_TEXT, "running": OK_TEXT,
                "ending": WARN_TEXT, "saved": OK_TEXT, "done": OK_TEXT, "failed": ERR_TEXT,
                "aborted": WARN_TEXT, "cancelled": theme.FG_MUTED, "skipped": theme.FG_MUTED}
#: job state -> a subtle row tint (translucent, over the row's own base)
ROW_TINT = {"saved": (95, 211, 141, 30), "done": (95, 211, 141, 30),
            "failed": (255, 107, 107, 36), "aborted": (255, 179, 71, 34),
            "launching": (90, 160, 255, 34), "running": (90, 160, 255, 34),
            "ending": (90, 160, 255, 34),
            # an ended run that had no file (save_data=False): repeats, diagnostics
            "no_file": (150, 150, 150, 40)}
#: states whose other cells are drawn muted
_MUTED_STATES = ("cancelled", "skipped")
#: ended states drawn gray when the run had no file (a failed one stays red)
_NO_FILE_GRAY = ("done", "aborted", "cancelled")


# -- small helpers -------------------------------------------------------------------

def _clock(epoch, seconds: bool = False) -> str:
    """"14:03" (or "14:03:11"); a time on another day gets its date; "" for
    no time."""
    if epoch in (None, ""):
        return ""
    try:
        t = float(epoch)
    except (TypeError, ValueError):
        return "?"
    fmt = "%H:%M:%S" if seconds else "%H:%M"
    if time.strftime("%Y-%m-%d", time.localtime(t)) != time.strftime("%Y-%m-%d"):
        fmt = "%m-%d " + fmt
    return time.strftime(fmt, time.localtime(t))


def _dur(seconds) -> str:
    """"35 s", "4.2 min", "1 h 05 min"."""
    try:
        s = max(float(seconds), 0.0)
    except (TypeError, ValueError):
        return "?"
    if s < 90:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    h, rest = divmod(int(round(s)), 3600)
    return f"{h} h {rest // 60:02d} min"


def _pill_css(level: str) -> str:
    bg = _PILL_BG.get(level, "#454545")
    return (f"QLabel {{ background: {bg}; color: white; border-radius: 9px;"
            f" padding: 2px 10px; font-size: 12px; font-weight: 600; }}")


def chip_css(level: str) -> str:
    """A small pill for a one-line status bar."""
    bg = _PILL_BG.get(level, "#454545")
    return (f"QLabel {{ background: {bg}; color: white; border-radius: 8px;"
            f" padding: 1px 8px; font-size: 11px; font-weight: 600; }}")


def _banner_css(bg: str, fg: str = "white") -> str:
    return (f"QLabel {{ background: {bg}; color: {fg}; border-radius: 6px;"
            f" padding: 4px 8px; font-size: 12px; }}")


def _small(text: str = "", color: str = theme.FG_MUTED) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {color}; font-size: 11px;")
    label.setWordWrap(True)
    return label


def _button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return b


def job_name(job: dict) -> str:
    return f"job {job.get('id')} ({job.get('label') or '?'})"


def ran_without_file(job: dict) -> bool:
    """The job's run had no file (``save_data=False``): the server's
    ``no_file``, or the state ``done``.  For a record from before the server
    set ``no_file``: its reason, which the queue words "save_data=False" for
    such a run (:func:`~waxx.util.device_state.run_loop.judge_run`)."""
    return (bool(job.get("no_file")) or job.get("state") == "done"
            or "save_data=False" in str(job.get("reason") or ""))


def gray_no_file(job: dict) -> bool:
    """Drawn gray: an ended run with no file that did not fail."""
    return job.get("state") in _NO_FILE_GRAY and ran_without_file(job)


# -- requests (pure: the tests check them) ---------------------------------------------

def list_request(limit: int = LIST_LIMIT) -> dict:
    return {"type": "run_queue", "action": "list", "limit": int(limit)}


def cancel_request(job: dict, by: str) -> dict:
    """Cancel ``job``.  A queued job is cancelled with ``queued_only``: if it
    launched meanwhile the server refuses, and the person is asked again (an
    Abort that discards a run's data is never sent on a question about a
    queued job)."""
    obj = {"type": "run_queue", "action": "cancel", "id": job.get("id"),
           "token": job.get("token"), "owner": OWNER, "by": by}
    if job.get("state") == "queued":
        obj["queued_only"] = True
    return obj


def kill_request(job: dict, by: str) -> dict:
    """Force-kill ``job``'s processes (a person's request; see
    :meth:`~waxx.util.device_state.run_queue.RunQueue.kill`)."""
    return {"type": "run_queue", "action": KILL_ACTION, "id": job.get("id"),
            "token": job.get("token"), "owner": OWNER, "by": by}


def move_request(job: dict, where: str, by: str, queued: list[dict]) -> dict | None:
    """``move`` for ``job`` (``where`` in :data:`MOVE_WHERE`) given the queued
    jobs in order: up = before the job ahead of it (``before_id``), down =
    after the job behind it (``after_id``), top = ``to_index`` 0, bottom =
    ``to_index`` len(queued) (the server clamps).  Neighbours are named by
    id, so a queue that changed meanwhile never sends it somewhere else.
    None when it cannot go that way (already first / last, not queued)."""
    if where not in MOVE_WHERE:
        raise ValueError(f"where must be one of {MOVE_WHERE}, not {where!r}")
    ids = [j.get("id") for j in queued]
    if job.get("id") not in ids:
        return None
    i = ids.index(job.get("id"))
    obj = {"type": "run_queue", "action": MOVE_ACTION, "id": job.get("id"),
           "token": job.get("token"), "owner": OWNER, "by": by}
    if where == "up":
        if i == 0:
            return None
        obj["before_id"] = ids[i - 1]
    elif where == "down":
        if i == len(ids) - 1:
            return None
        obj["after_id"] = ids[i + 1]
    elif where == "top":
        if i == 0:
            return None
        obj["to_index"] = 0
    else:
        if i == len(ids) - 1:
            return None
        obj["to_index"] = len(ids)
    return obj


def edit_request(job: dict, fields: dict, by: str) -> dict:
    """``edit``: only the ``fields`` that change (see :class:`EditJobDialog`)."""
    return {"type": "run_queue", "action": EDIT_ACTION, "id": job.get("id"),
            "token": job.get("token"), "fields": dict(fields), "owner": OWNER, "by": by}


def tail_request(job_id, token, offset: int) -> dict:
    return {"type": "run_queue", "action": "tail", "id": job_id, "token": token,
            "offset": int(offset)}


def pause_request(scope: str, reason: str, by: str) -> dict:
    return {"type": "run_queue", "action": "pause", "scope": scope, "reason": reason,
            "owner": OWNER, "by": by}


def resume_request(scope: str, by: str) -> dict:
    return {"type": "run_queue", "action": "resume", "scope": scope, "owner": OWNER, "by": by}


def hold_request(reason: str, by: str) -> dict:
    return {"type": "run_queue", "action": "hold", "reason": reason, "owner": OWNER, "by": by}


def release_request(by: str) -> dict:
    return {"type": "run_queue", "action": "release", "owner": OWNER, "by": by}


def again_request(job: dict, by: str, *, first: bool = False,
                  client_sha256: str | None = None, client_host: str | None = None) -> dict:
    """The same file again as a new job of a person's: its argv, working
    folder, label, write-back veto and drift setting.  Its dependencies,
    chain, due time and repeats are NOT carried over (they named the old
    job's place).  ``first``: ``insert`` at ``at_index`` 0 (the next job);
    otherwise ``submit`` (the owner rule: ahead of agents' queued jobs).
    ``client_sha256`` / ``client_host`` let the server refuse a file whose copy
    on its machine differs from this PC's (as kq sends them)."""
    obj = {"type": "run_queue", "action": "insert" if first else "submit",
           "path": job.get("path"), "argv": [str(a) for a in (job.get("argv") or [])],
           "cwd": job.get("cwd") or None, "label": job.get("label") or None,
           "owner": OWNER, "by": by, "allow_drift": bool(job.get("allow_drift")),
           "client_sha256": client_sha256, "client_host": client_host, "host": client_host}
    if job.get("write_back") is False:
        obj["write_back"] = False
    if first:
        obj["at_index"] = 0
    return {k: v for k, v in obj.items() if v is not None}


def not_carried_over(job: dict) -> list[str]:
    """What :func:`again_request` leaves behind, in words."""
    out = []
    if job.get("after"):
        out.append("its dependencies (after " + ", ".join(map(str, job["after"])) + ")")
    if job.get("chain") and not str(job["chain"]).startswith("repeat-"):
        out.append(f"its chain ({job['chain']})")
    if int(job.get("repeat_of") or 1) > 1:
        out.append(f"its repeats (one of {job['repeat_of']})")
    if job.get("due"):
        out.append(f"its due time ({_clock(job['due'])})")
    if job.get("paused"):
        out.append("its pause")
    return out


def find_editor() -> list[str] | None:
    """The command that opens a file in VS Code on this PC (``Code.exe``
    next to the ``code`` CLI's folder, else the CLI itself), or None.  A
    ``.py`` file is never handed to the OS's default action: on Windows that
    can run it."""
    cli = shutil.which("code")
    if not cli:
        return None
    exe = Path(cli).resolve().parent.parent / "Code.exe"
    return [str(exe)] if exe.is_file() else [cli]


def _launch_detached(cmd: list[str]) -> None:
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | \
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True, creationflags=flags)


def _q(arg: str) -> str:
    arg = str(arg)
    return f'"{arg}"' if (" " in arg or "\t" in arg or not arg) else arg


#: the queue's label rule (run_queue._LABEL_RE): the default label is the
#: file's stem through it
_LABEL_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _default_label(path: str) -> str:
    stem = PurePath(str(path).replace("\\", "/")).stem
    return (_LABEL_CHARS.sub("_", stem).strip("_") or stem)[:60]


def kq_commands(job: dict, now: float | None = None) -> dict[str, str]:
    """The kq command lines for a job: ``submit`` (the same submission
    again), ``tail`` (follow its log), ``show``, ``cancel``.

    ``submit`` names only what differs from kq's defaults: ``--label`` when
    it is not the file's stem, ``--priority`` when not 0, ``--at`` for a due
    time still ahead (HH:MM within the next day, else epoch seconds),
    ``--depends-on`` for its dependencies, ``--repeat N`` for one of N
    repeats (the queue names the new chain itself), ``--chain`` for a chain
    of its own, ``--no-stop-on-failure`` when its chain goes on after a
    failure, ``--no-write-back``, ``--allow-drift``, ``--cwd``, ``--agent``;
    the experiment's own words after ``--``."""
    now = time.time() if now is None else float(now)
    jid = job.get("id")
    path = str(job.get("path") or "")
    parts = ["kq", "submit", _q(path)]
    if job.get("label") and job["label"] != _default_label(path):
        parts += ["--label", _q(job["label"])]
    if job.get("priority"):
        parts += ["--priority", str(job["priority"])]
    due = job.get("due")
    if isinstance(due, (int, float)) and due > now:
        parts += ["--at", time.strftime("%H:%M", time.localtime(due)) if due - now < 86000
                  else str(int(due))]
    if job.get("after"):
        parts += ["--depends-on"] + [str(a) for a in job["after"]]
    repeat = int(job.get("repeat_of") or 1)
    chain = job.get("chain") or None
    if repeat > 1:
        parts += ["--repeat", str(repeat)]
        if chain and str(chain).startswith("repeat-"):
            chain = None                  # the queue's own name for a repeat's chain
    if chain:
        parts += ["--chain", _q(chain)]
    if (chain or repeat > 1) and not job.get("stop_on_failure"):
        parts.append("--no-stop-on-failure")
    if job.get("write_back") is False:
        parts.append("--no-write-back")
    if job.get("allow_drift"):
        parts.append("--allow-drift")
    if job.get("cwd"):
        parts += ["--cwd", _q(job["cwd"])]
    if job.get("owner") == "agent":
        parts.append("--agent")
    argv = [str(a) for a in (job.get("argv") or [])]
    if argv:
        parts += ["--"] + [_q(a) for a in argv]
    return {"submit": " ".join(parts), "tail": f"kq tail {jid} -f", "show": f"kq show {jid}",
            "cancel": f"kq cancel {jid}", "kill": f"kq kill {jid}"}


# -- ordering and the table --------------------------------------------------------------

def _group(job: dict) -> int:
    state = job.get("state")
    return 0 if state in ENDED else 1 if state in IN_SLOT else 2 if state == "queued" else 3


def order_jobs(jobs: list[dict]) -> list[dict]:
    """The server's order.  A server that sends ``position`` (0-based among
    the queued jobs, None for the others) lists the ended jobs, then the one
    in the slot, then the queued ones in rank order: that order is kept, the
    queued jobs sorted by ``position`` (stable).  Without ``position``: the
    list's own order."""
    jobs = [j for j in jobs if isinstance(j, dict)]
    if not any(j.get("position") is not None for j in jobs):
        return list(jobs)

    def key(item):
        i, j = item
        pos = j.get("position")
        try:
            pos = float(pos) if pos is not None else float("inf")
        except (TypeError, ValueError):
            pos = float("inf")
        return (_group(j), pos, i)
    return [j for _, j in sorted(enumerate(jobs), key=key)]


def _estimate(job: dict) -> dict:
    """``{duration_s, eta_start, eta_end, basis}`` from the job's
    ``estimate`` (phase 1b), or flat ``estimated_s`` / ``eta_*`` fields."""
    est = job.get("estimate")
    if isinstance(est, dict):
        return est
    return {"duration_s": job.get("estimated_s"), "eta_start": job.get("eta_start"),
            "eta_end": job.get("eta_end"), "basis": ""}


def derived_waiting(job: dict, info: dict, by_id: dict) -> str:
    """Why a queued job waits, for a server that does not say (before phase
    1b): from the job's own fields and the queue's info.  "" when it does not
    wait (or is not queued)."""
    if job.get("state") != "queued":
        return ""
    now = time.time()
    if job.get("paused"):
        return f"paused by {job.get('paused_by') or '?'}"
    due = job.get("due")
    if due is not None:
        try:
            if float(due) > now:
                return f"due at {_clock(due)}"
        except (TypeError, ValueError):
            pass
    # only dependencies in this listing are judged (one beyond it is the
    # server's to judge: it says so through the job's state)
    waits = [a for a in (job.get("after") or [])
             if a in by_id and by_id[a].get("state") not in SUCCEEDED]
    if waits:
        return "waiting for job " + ", ".join(map(str, waits))
    paused = info.get("paused") or {}
    if paused.get("all"):
        return f"all jobs paused by {paused['all'].get('by')}"
    if job.get("owner") == "agent" and paused.get("agent"):
        return f"agent jobs paused by {paused['agent'].get('by')}"
    hold = info.get("person_hold") or {}
    if job.get("owner") == "agent" and hold.get("active"):
        return describe_hold(hold)
    nxt = list(info.get("next") or [])
    current = info.get("current") or None
    if current:
        return f"job {current.get('id')} is in the slot"
    if nxt and nxt[0] != job.get("id"):
        return f"job {nxt[0]} goes first"
    return str(info.get("waiting") or "")


#: (column key, header)
COLUMNS = (
    ("position", "#"), ("id", "id"), ("state", "state"), ("owner", "owner / submitter"),
    ("label", "label"), ("expt_class", "class"), ("path", "file"), ("run_id", "run"),
    ("priority", "prio"), ("due", "due"), ("after", "after / chain"), ("est", "est."),
    ("source_changed", "source"), ("waiting", "waiting"), ("submitted", "submitted"),
)
COLUMN_KEYS = tuple(k for k, _ in COLUMNS)
#: Columns hidden until someone shows them (right-click the header); the
#: label's tooltip carries the file's path.
DEFAULT_HIDDEN = ("expt_class", "path", "priority", "due", "after")
#: Column widths (px) until someone resizes them; the last column stretches.
DEFAULT_WIDTHS = {"position": 40, "id": 44, "state": 100, "owner": 120, "label": 160,
                  "expt_class": 110, "path": 220, "run_id": 60, "priority": 50, "due": 60,
                  "after": 100, "est": 150, "source_changed": 64, "waiting": 200}
#: The table's filters: every listed job; the last N ended ones and every one
#: not ended; only those not ended.
FILTERS = ("all", "last", "pending")
DEFAULT_FILTER = "last"
#: Ended jobs the "last" filter shows until someone changes it.
DEFAULT_LAST_N = 15
_FILTER_TITLES = {"all": "All", "last": "Last", "pending": "Pending"}
#: Where a host's panel remembers its columns and filter (:func:`panel_settings`).
SETTINGS_ORG = "waxx"
SETTINGS_APP = "run_queue_panel"
#: How long after the last column change the layout is written.
SAVE_DEBOUNCE_MS = 300


def panel_settings() -> QSettings:
    """The QSettings a host passes as ``RunQueuePanel(settings=...)`` so the
    table's columns and filter outlive the window."""
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


class JobTableModel(QAbstractTableModel):
    """The jobs of one ``list`` reply, in the server's order
    (:func:`order_jobs`).  ``Qt.ItemDataRole.UserRole`` gives a row's job
    dict."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs: list[dict] = []
        self.info: dict = {}
        self._by_id: dict = {}
        self._all: list[dict] = []
        #: the filter (:data:`FILTERS`) and the "last" filter's N
        self.mode = "all"
        self.last_n = DEFAULT_LAST_N

    # -- feeding ---------------------------------------------------------------------

    def set_jobs(self, jobs: list[dict] | None, info: dict | None = None,
                 nxt: list | None = None) -> None:
        self.beginResetModel()
        all_jobs = [j for j in (jobs or []) if isinstance(j, dict)]
        self._by_id = {j.get("id"): j for j in all_jobs}
        self.info = dict(info or {})
        if nxt is not None:
            self.info["next"] = list(nxt)
        self._all = order_jobs(all_jobs)
        self._filter()
        self.endResetModel()

    def set_mode(self, mode: str, last_n: int | None = None) -> None:
        """The filter (:data:`FILTERS`) and, for "last", how many ended jobs."""
        if mode not in FILTERS:
            raise ValueError(f"filter must be one of {FILTERS}, not {mode!r}")
        self.beginResetModel()
        self.mode = mode
        if last_n is not None:
            self.last_n = max(0, int(last_n))
        self._filter()
        self.endResetModel()

    def set_show_ended(self, show: bool) -> None:
        self.set_mode("all" if show else "pending")

    def _last_ended(self, n: int) -> set:
        """ids of the ``n`` ended jobs that ended last (by ``ended_at``; a job
        without one by its place in the list)."""
        ended = [(i, j) for i, j in enumerate(self._all) if j.get("state") in ENDED]

        def key(item):
            i, j = item
            try:
                t = float(j.get("ended_at"))
            except (TypeError, ValueError):
                t = float("-inf")
            return (t, i)
        return {id(j) for _, j in sorted(ended, key=key)[-n:]} if n > 0 else set()

    def _shown(self, mode: str) -> list[dict]:
        if mode == "all":
            return list(self._all)
        keep = self._last_ended(self.last_n) if mode == "last" else set()
        return [j for j in self._all if j.get("state") not in ENDED or id(j) in keep]

    def _filter(self) -> None:
        self.jobs = self._shown(self.mode)

    def update_job(self, job: dict) -> bool:
        """One job's fresh record (a ``describe`` reply) in place: its row
        repaints, the order stays.  False when the job is not listed."""
        for i, old in enumerate(self._all):
            if old.get("id") == job.get("id") and old.get("token") == job.get("token"):
                self._all[i] = dict(job)
                self._by_id[job.get("id")] = self._all[i]
                break
        else:
            return False
        for row, shown in enumerate(self.jobs):
            if shown.get("id") == job.get("id"):
                self.jobs[row] = self._all[i]
                self.dataChanged.emit(self.index(row, 0),
                                      self.index(row, len(COLUMNS) - 1))
        return True

    def row_of(self, job_id, token=None) -> int:
        for i, j in enumerate(self.jobs):
            if j.get("id") == job_id and (token is None or j.get("token") == token):
                return i
        return -1

    # -- Qt --------------------------------------------------------------------------

    def rowCount(self, parent=QModelIndex()):                 # noqa: N802
        return 0 if parent.isValid() else len(self.jobs)

    def columnCount(self, parent=QModelIndex()):              # noqa: N802
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section][1]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.jobs):
            return None
        job = self.jobs[index.row()]
        key = COLUMN_KEYS[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return self.text(job, key)
        if role == Qt.ItemDataRole.ToolTipRole:
            return self.tooltip(job, key) or None
        state = str(job.get("state"))
        if role == Qt.ItemDataRole.ForegroundRole:
            if key == "source_changed" and job.get("source_changed"):
                return QColor(ERR_TEXT)
            if key == "state":
                return QColor(theme.FG_MUTED if gray_no_file(job)
                              else _STATE_COLOR.get(state, theme.FG))
            if (key == "est" and self.text(job, key)) or state in _MUTED_STATES:
                return QColor(theme.FG_MUTED)
            return QColor(theme.FG)
        if role == Qt.ItemDataRole.BackgroundRole:
            tint = ROW_TINT["no_file"] if gray_no_file(job) else ROW_TINT.get(state)
            return QColor(*tint) if tint else None
        if role == Qt.ItemDataRole.FontRole and state in IN_SLOT:
            font = QFont()
            font.setBold(True)
            return font
        if role == Qt.ItemDataRole.UserRole:
            return job
        return None

    # -- cell text ---------------------------------------------------------------------

    def queued_in_order(self) -> list[dict]:
        """The queued jobs in the server's order (all of them, shown or not)."""
        return [j for j in self._all if j.get("state") == "queued"]

    def filter_counts(self) -> dict[str, int]:
        """Rows each filter would show: ``{"all": n, "last": k, "pending": m}``."""
        return {mode: len(self._shown(mode)) for mode in FILTERS}

    def position_text(self, job: dict) -> str:
        """1-based place in the queue ("slot" for the job in it): the server's
        0-based ``position`` + 1, or (before phase 1b) the place in ``next``."""
        if job.get("state") in IN_SLOT:
            return "slot"
        pos = job.get("position")
        if pos is not None:
            try:
                return str(int(pos) + 1)
            except (TypeError, ValueError):
                return str(pos)
        nxt = list(self.info.get("next") or [])
        if job.get("id") in nxt:
            return str(nxt.index(job.get("id")) + 1)
        return ""

    def text(self, job: dict, key: str) -> str:
        if key == "position":
            return self.position_text(job)
        if key == "id":
            return str(job.get("id", ""))
        if key == "state":
            s = str(job.get("state") or "")
            if job.get("kill") and s in IN_SLOT:
                s += " (killed)"
            elif job.get("cancel") and s in IN_SLOT:
                s += " (cancel asked)"
            elif job.get("paused") and s == "queued":
                s += " (paused)"
            elif s in ENDED and s != "done" and ran_without_file(job):
                s += " (no file)"
            return s
        if key == "owner":
            who = job.get("submitter")
            return " / ".join(str(p) for p in (job.get("owner"), who) if p)
        if key == "label":
            return str(job.get("label") or "")
        if key == "expt_class":
            text = str(job.get("expt_class") or "")
            if job.get("calibrates_declared"):
                text += " [cal]" if text else "[cal]"
            return text
        if key == "path":
            return str(job.get("path") or "")
        if key == "run_id":
            return "" if job.get("run_id") in (None, "") else str(job["run_id"])
        if key == "priority":
            return "" if job.get("priority") is None else str(job["priority"])
        if key == "due":
            return _clock(job.get("due"))
        if key == "after":
            parts = []
            if job.get("after"):
                parts.append("after " + ",".join(str(a) for a in job["after"]))
            if job.get("chain"):
                parts.append(f"chain {job['chain']}")
            return " | ".join(parts)
        if key == "est":
            est = _estimate(job)
            parts = []
            if est.get("duration_s") is not None:
                parts.append(f"~{_dur(est['duration_s'])}")
            if est.get("eta_start") is not None and job.get("state") == "queued":
                parts.append(f"start {_clock(est['eta_start'])}")
            if est.get("eta_end") is not None and job.get("state") not in ENDED:
                parts.append(f"end {_clock(est['eta_end'])}")
            return (" ".join(parts) + " est.") if parts else ""
        if key == "source_changed":
            return "CHANGED" if job.get("source_changed") else ""
        if key == "waiting":
            if "waiting" in job:
                return str(job.get("waiting") or "")
            return derived_waiting(job, self.info, self._by_id)
        if key == "submitted":
            when = _clock(job.get("submitted_at"))
            by = job.get("submitted_by") or ""
            return f"{when} by {by}" if by else when
        return ""

    def tooltip(self, job: dict, key: str) -> str:
        if key == "path":
            sha = str(job.get("sha256") or "")
            lines = [str(job.get("path") or "")]
            if sha:
                lines.append(f"sha256 at submit: {sha[:16]}...")
            if job.get("cwd"):
                lines.append(f"working folder: {job['cwd']}")
            return "\n".join(lines)
        if key == "state":
            lines = []
            if ran_without_file(job):
                lines.append("save_data=False: liveOD wrote no file for this run")
            if job.get("reason"):
                lines.append(str(job["reason"]))
            outcome = job.get("outcome") or {}
            if isinstance(outcome, dict) and outcome:
                lines.append("outcome: " + ", ".join(f"{k}={v}" for k, v in outcome.items()
                                                     if v not in (None, "", False)))
            if job.get("exit_code") is not None:
                lines.append(f"exit code {job['exit_code']}")
            cancel = job.get("cancel")
            if isinstance(cancel, dict):
                lines.append(f"cancel asked by {cancel.get('by')} at {_clock(cancel.get('at'), True)}"
                             + (" (Abort sent)" if cancel.get("abort_sent") else ""))
            kill = job.get("kill")
            if isinstance(kill, dict):
                lines.append(f"force-killed by {kill.get('by')} at {_clock(kill.get('at'), True)}: "
                             f"{kill.get('note')}")
            if job.get("paused"):
                lines.append(f"paused by {job.get('paused_by') or '?'}"
                             + (f" since {_clock(job['paused_since'])}"
                                if job.get("paused_since") else ""))
            return "\n".join(lines)
        if key == "position" and job.get("position") is not None:
            return (f"place {self.position_text(job)} in the queue's order (the server's "
                    f"position {job['position']}, counted from 0)")
        if key == "priority":
            lines = ["priority: only a placement hint at submit (a person's job is placed "
                     "ahead of agents' jobs); the order is the rank"]
            if job.get("rank") is not None:
                lines.append(f"rank {job['rank']:g}" if isinstance(job["rank"], (int, float))
                             else f"rank {job['rank']}")
            return "\n".join(lines)
        if key == "label":
            argv = job.get("argv") or []
            lines = [str(job.get("path") or "")]
            if argv:
                lines.append("argv: " + " ".join(str(a) for a in argv))
            if job.get("expt_class"):
                lines.append(f"class {job['expt_class']}")
            return "\n".join(x for x in lines if x)
        if key == "expt_class":
            cal = job.get("calibrates_declared")
            lines = []
            if cal:
                lines.append(f"declares calibrations: {cal}")
            if job.get("write_back") is False:
                lines.append("write-back vetoed (WAXX_CAL_NO_WRITE_BACK)")
            return "\n".join(lines)
        if key == "source_changed" and job.get("source_changed"):
            return ("The file changed since it was submitted: the job is skipped at launch"
                    + (" -- unless it allows drift (it does)" if job.get("allow_drift")
                       else " (it does not allow drift)") + ".")
        if key == "est":
            basis = _estimate(job).get("basis")
            return ("An estimate, not a promise" + (f": {basis}" if basis else "") + ".")
        if key == "owner":
            return f"submitted by {job.get('submitted_by') or '?'}"
        if key == "after" and job.get("chain"):
            return ("stops with its chain on a failure" if job.get("stop_on_failure")
                    else "its chain goes on after a failure")
        return ""


# -- dialogs -------------------------------------------------------------------------------

def join_argv(argv) -> str:
    """argv as one line: a word holding whitespace (or empty) in double
    quotes -- :func:`split_argv` reads it back to the same list."""
    return " ".join(_q(a) for a in argv)


def split_argv(text: str) -> list[str]:
    """Words of ``text``, shell-style: whitespace separates, "..." or '...'
    keep a word with spaces together.  Backslashes are kept as they are (a
    Windows path is not an escape sequence).  ValueError on an unclosed
    quote."""
    lexer = shlex.shlex(text, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""
    lexer.commenters = ""
    return list(lexer)


def _parse_due(text: str, now: float | None = None) -> float | None:
    """A due time as the dialog takes it -> epoch seconds (None for "").

    * ``HH:MM[:SS]`` or epoch seconds: :func:`kq.parse_at` -- the same rule
      as ``kq --at`` (the next such time, by the calendar: a time already
      past today is tomorrow's, and a DST change is no hour off);
    * ``MM-DD HH:MM[:SS]`` (the form :func:`_clock` shows for another day)
      -- that day this year;
    * ``YYYY-MM-DD HH:MM[:SS]``.

    ValueError when unreadable."""
    import datetime  # noqa: PLC0415

    from waxx.util.device_state.kq import parse_at  # noqa: PLC0415
    text = text.strip()
    if not text:
        return None
    now = time.time() if now is None else float(now)
    year = datetime.datetime.fromtimestamp(now).year
    for fmt, prefix in (("%Y-%m-%d %H:%M", ""), ("%Y-%m-%d %H:%M:%S", ""),
                        ("%Y-%m-%d %H:%M", f"{year}-"), ("%Y-%m-%d %H:%M:%S", f"{year}-")):
        try:
            return datetime.datetime.strptime(prefix + text, fmt).timestamp()
        except ValueError:
            pass
    try:
        return parse_at(text, now)
    except ValueError:
        raise ValueError(f"due: not a time: {text!r} (HH:MM, MM-DD HH:MM or "
                         "YYYY-MM-DD HH:MM)") from None


class EditJobDialog(QDialog):
    """A queued job's editable fields.  :meth:`changes` returns only the
    fields that differ from the job (None when a field is unreadable)."""

    def __init__(self, job: dict, parent=None):
        super().__init__(parent)
        self.job = dict(job)
        self.setWindowTitle(f"Edit {job_name(job)}")
        box = QVBoxLayout(self)
        form = QFormLayout()
        #: argv as first shown (quoted); left as it is, the job's own list stays
        self._argv_shown = join_argv(job.get("argv") or [])
        self.argv = QLineEdit(self._argv_shown)
        self.argv.setPlaceholderText('key=value ... (as artiq_run takes them; "a b" for a '
                                     'word with spaces)')
        self.label = QLineEdit(str(job.get("label") or ""))
        self.after = QLineEdit(" ".join(str(a) for a in (job.get("after") or [])))
        self.after.setPlaceholderText("job ids, e.g. 12 13")
        self.chain = QLineEdit(str(job.get("chain") or ""))
        self.stop_on_failure = QCheckBox("stop the chain when a job fails")
        self.stop_on_failure.setChecked(bool(job.get("stop_on_failure")))
        self.no_write_back = QCheckBox("veto calibration write-back for this job")
        self.no_write_back.setChecked(job.get("write_back") is False)
        #: the due time as first shown: left as it is, the job's own value stays
        self._due_shown = _clock(job.get("due")) if job.get("due") else ""
        self.due = QLineEdit(self._due_shown)
        self.due.setPlaceholderText("blank: no due time; HH:MM (next), MM-DD HH:MM, "
                                    "YYYY-MM-DD HH:MM")
        self.allow_drift = QCheckBox("run it even if the file changes before launch")
        self.allow_drift.setChecked(bool(job.get("allow_drift")))
        self.paused = QCheckBox("paused (this job stays queued until unpaused)")
        self.paused.setChecked(bool(job.get("paused")))
        form.addRow("argv", self.argv)
        form.addRow("label", self.label)
        form.addRow("after", self.after)
        form.addRow("chain", self.chain)
        form.addRow("", self.stop_on_failure)
        form.addRow("", self.no_write_back)
        form.addRow("due", self.due)
        form.addRow("", self.allow_drift)
        form.addRow("", self.paused)
        box.addLayout(form)
        self.problem = _small("", ERR_TEXT)
        box.addWidget(self.problem)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        box.addWidget(self.buttons)
        for w in (self.argv, self.label, self.after, self.chain, self.due):
            w.textChanged.connect(self._update)
        self._update()

    def values(self) -> dict:
        """Every field as the server takes it; ValueError when one is unreadable."""
        try:
            after = [int(x) for x in self.after.text().replace(",", " ").split()]
        except ValueError:
            raise ValueError("after: job ids are whole numbers") from None
        label = self.label.text().strip()
        if not label:
            raise ValueError("label: a label is needed")
        due_text = self.due.text().strip()
        if self.job.get("due") and due_text == self._due_shown:
            due = self.job.get("due")
        else:
            due = _parse_due(due_text)
        if self.argv.text() == self._argv_shown:
            argv = [str(a) for a in (self.job.get("argv") or [])]
        else:
            try:
                argv = split_argv(self.argv.text())
            except ValueError as exc:
                raise ValueError(f"argv: {exc}") from None
        return {"argv": argv, "label": label, "after": after,
                "chain": self.chain.text().strip() or None,
                "stop_on_failure": self.stop_on_failure.isChecked(),
                "write_back": False if self.no_write_back.isChecked() else None,
                "due": due, "allow_drift": self.allow_drift.isChecked(),
                "paused": self.paused.isChecked()}

    def changes(self) -> dict | None:
        try:
            values = self.values()
        except ValueError:
            return None
        job = self.job
        old = {"argv": [str(a) for a in (job.get("argv") or [])],
               "label": str(job.get("label") or ""), "after": list(job.get("after") or []),
               "chain": job.get("chain") or None,
               "stop_on_failure": bool(job.get("stop_on_failure")),
               "write_back": False if job.get("write_back") is False else None,
               "due": job.get("due"), "allow_drift": bool(job.get("allow_drift")),
               "paused": bool(job.get("paused"))}
        return {k: v for k, v in values.items() if v != old[k]}

    def _update(self) -> None:
        try:
            self.values()
        except ValueError as exc:
            self.problem.setText(str(exc))
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return
        self.problem.setText("")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)


class JobLogWindow(QDialog):
    """One job's log, followed through the ``tail`` action with a byte cursor
    (:attr:`offset`): every :data:`TAIL_RUNNING_MS` while the job is in the
    slot, every :data:`TAIL_QUEUED_MS` while it is queued, again at once
    while lines keep coming, and not at all once the server says ``done``
    (the job ended and its log is read to the end).  A server that does not
    answer is asked again; a refusal stops the window (it says why)."""

    def __init__(self, panel: "RunQueuePanel", job: dict):
        super().__init__(panel)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.panel = panel
        self.job_id = job.get("id")
        self.token = job.get("token")
        self.offset = 0
        self.done = False
        self.stopped = False
        self.fetching = False
        self.state = str(job.get("state") or "")
        self.setWindowTitle(f"Run queue: {job_name(job)} -- log")
        self.resize(820, 420)
        box = QVBoxLayout(self)
        self.status = _small("")
        box.addWidget(self.status)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(LOG_MAX_LINES)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.text.setFont(mono)
        self.text.setPlaceholderText("No output yet.")
        box.addWidget(self.text)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.fetch)
        self.timer.start(TAIL_QUEUED_MS if self.state == "queued" else TAIL_RUNNING_MS)
        self._set_status()

    def request(self) -> dict:
        return tail_request(self.job_id, self.token, self.offset)

    def fetch(self) -> None:
        if self.fetching or self.done or self.stopped:
            return
        self.fetching = True
        self.panel.runner.send(self.request(), self.on_reply)

    def on_reply(self, reply: dict) -> None:
        self.fetching = False
        if self.stopped:
            return
        if reply.get("status") != "ok":
            if reply.get("no_reply"):
                self._set_status(f"the monitor server is not answering ({reply.get('msg')}); "
                                 "still asking")
                return
            self.stop(("this monitor server cannot serve job logs (older code)"
                       if is_unknown_request(reply) else str(reply.get("msg") or "refused")))
            return
        lines = [str(x) for x in (reply.get("lines") or [])]
        try:
            new_offset = int(reply.get("offset", self.offset))
        except (TypeError, ValueError):
            new_offset = self.offset
        if new_offset < self.offset:
            # never move the cursor back: a line would be shown twice
            new_offset = self.offset
        self.offset = new_offset
        self.state = str(reply.get("state") or self.state)
        if lines:
            bar = self.text.verticalScrollBar()
            at_end = bar.value() >= bar.maximum() - 2
            self.text.appendPlainText("\n".join(lines))
            if at_end:
                bar.setValue(bar.maximum())
        if reply.get("done"):
            self.done = True
            self.timer.stop()
            run = reply.get("run_id")
            self._set_status(f"job {self.job_id} {self.state}"
                             + (f" (run {run})" if run else "") + "; log read to the end")
            return
        self.timer.setInterval(TAIL_QUEUED_MS if self.state == "queued" else TAIL_RUNNING_MS)
        self._set_status()
        if lines:
            QTimer.singleShot(0, self.fetch)

    def stop(self, why: str) -> None:
        self.stopped = True
        self.timer.stop()
        self._set_status(why, ERR_TEXT)

    def _set_status(self, text: str = "", color: str = theme.FG_MUTED) -> None:
        if not text:
            text = (f"job {self.job_id}: {self.state or '?'} -- "
                    + ("waiting to launch (no log yet)" if self.state == "queued"
                       else "following its log"))
        self.status.setText(text)
        set_style_if_changed(self.status, f"color: {color}; font-size: 11px;")

    def closeEvent(self, event):                                # noqa: N802
        self.stopped = True
        self.timer.stop()
        self.panel.forget_log_window(self)
        super().closeEvent(event)


class SourceWindow(QDialog):
    """A job's experiment file, read-only, as this PC reads it now -- with a
    warning line when it differs from the file the job was submitted with."""

    def __init__(self, panel: "RunQueuePanel", job: dict, text: str, note: str = ""):
        super().__init__(panel)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.panel = panel
        self.job = dict(job)
        self.setWindowTitle(f"{job.get('label') or 'source'} -- {job.get('path')} (read-only)")
        self.resize(900, 640)
        box = QVBoxLayout(self)
        head = QHBoxLayout()
        self.path_label = _small(str(job.get("path") or ""))
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        head.addWidget(self.path_label, 1)
        self.editor_button = _button("Open in VS Code")
        self.editor_button.setEnabled(find_editor() is not None)
        self.editor_button.clicked.connect(lambda _=False: self.panel.open_source(self.job))
        head.addWidget(self.editor_button)
        box.addLayout(head)
        self.note = _small(note, WARN_TEXT)
        self.note.setVisible(bool(note))
        box.addWidget(self.note)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self.text.setFont(mono)
        self.text.setPlainText(text)
        box.addWidget(self.text, 1)


# -- the panel -----------------------------------------------------------------------------

class RunQueuePanel(QWidget):
    """The run queue (see the module docstring).

    ``requester``: ``request_dict -> reply_dict``; ``by``: who is clicking
    (default ``user@host``); ``synchronous``: call the requester on the GUI
    thread (tests).  ``runner``: an existing RequestRunner to share instead.
    ``settings``: a QSettings to remember the columns (order, widths, shown)
    and the filter in, written on every change; None remembers nothing."""

    def __init__(self, requester: Callable[[dict], Any] | None = None, *, by: str | None = None,
                 synchronous: bool = False, runner: RequestRunner | None = None,
                 settings: QSettings | None = None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.by = by or default_by()
        self._own_runner = runner is None
        self.runner = runner or RequestRunner(requester, synchronous=synchronous, parent=self)
        self.info: dict = {}
        self.hold: dict = {}
        self.reachable = False
        self.has_queue = True
        self.unsupported: set[str] = set()
        self._listed_once = False
        self._listing = False
        self._list_again = False
        self._describing = False
        #: queued jobs the last list left out (its `truncated`; -1: count unknown)
        self.not_listed = 0
        self._summary_sig = None
        self._selected: tuple | None = None
        self._log_windows: dict = {}

        #: the table's filter (:data:`FILTERS`)
        saved_filter = self._setting("filter")
        self.filter = saved_filter if saved_filter in FILTERS else DEFAULT_FILTER
        #: ended jobs the "last" filter shows
        try:
            self.last_n = max(1, int(self._setting("last_n") or DEFAULT_LAST_N))
        except (TypeError, ValueError):
            self.last_n = DEFAULT_LAST_N
        #: scroll to the bottom when a list brings a new job (default on)
        saved_follow = self._setting("follow_new")
        self.follow_new = (True if saved_follow in (None, "")
                           else str(saved_follow).lower() in ("true", "1"))
        #: (id, token) of every job in the last good list; None before the first
        self._known_jobs: set | None = None
        #: opens a file in an editor: ``cmd -> None`` (tests replace it)
        self.launch = _launch_detached
        self._source_windows: list = []

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 4, 0, 0)
        box.setSpacing(4)

        # the toolbar: the queue's state, the filter, hold / pause, refresh
        top = QHBoxLayout()
        top.setSpacing(6)
        self.pill = QLabel("?")
        top.addWidget(self.pill)
        self.summary = QLabel("")
        self.summary.setStyleSheet(f"color: {theme.FG}; font-size: 12px;")
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        top.addWidget(self.summary, 1)
        self.counts = _small("")
        self.counts.setWordWrap(False)
        top.addWidget(self.counts)
        top.addSpacing(8)
        self.filter_combo = QComboBox()
        self.filter_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.filter_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for name in FILTERS:
            self.filter_combo.addItem(_FILTER_TITLES[name], name)
        self.filter_combo.setCurrentIndex(FILTERS.index(self.filter))
        self.filter_combo.activated.connect(
            lambda i: self.set_filter(self.filter_combo.itemData(i)))
        top.addWidget(self.filter_combo)
        self.last_spin = QSpinBox()
        self.last_spin.setRange(1, 999)
        self.last_spin.setValue(self.last_n)
        self.last_spin.setToolTip("How many ended jobs the Last filter shows (with every job "
                                  "not ended)")
        self.last_spin.valueChanged.connect(self.set_last_n)
        top.addWidget(self.last_spin)
        self.follow_button = QToolButton()
        self.follow_button.setText("Follow")
        self.follow_button.setCheckable(True)
        self.follow_button.setAutoRaise(True)
        self.follow_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.follow_button.setChecked(self.follow_new)
        self.follow_button.setToolTip("Scroll to the bottom of the table when a job is queued")
        self.follow_button.toggled.connect(self.set_follow_new)
        top.addWidget(self.follow_button)
        top.addSpacing(8)
        self.hold_button = _button("Hold...")
        self.hold_button.clicked.connect(lambda _=False: self.toggle_hold())
        top.addWidget(self.hold_button)
        self.pause_buttons: dict[str, QPushButton] = {}
        for scope in ("agent", "all"):
            button = _button("")
            button.clicked.connect(lambda _=False, s=scope: self.toggle_pause(s))
            self.pause_buttons[scope] = button
            top.addWidget(button)
        self.refresh_button = QToolButton()
        self.refresh_button.setIcon(self.style().standardIcon(
            QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh_button.setToolTip("Ask the server for the list again")
        self.refresh_button.setAutoRaise(True)
        self.refresh_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.refresh_button.clicked.connect(lambda _=False: self.refresh_list())
        top.addWidget(self.refresh_button)
        box.addLayout(top)

        # banners: only while there is something to say
        self.alarm = QLabel("")
        self.alarm.setWordWrap(True)
        self.alarm.setStyleSheet(_banner_css("#c62828"))
        self.alarm.hide()
        box.addWidget(self.alarm)
        self.hold_label = QLabel("")
        self.hold_label.setWordWrap(True)
        self.hold_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.hold_label.hide()
        box.addWidget(self.hold_label)
        self.pause_labels: dict[str, QLabel] = {}
        for scope in ("agent", "all"):
            label = _small("")
            label.hide()
            self.pause_labels[scope] = label
            box.addWidget(label)
        self.loop_note = _small("")
        self.loop_note.hide()
        box.addWidget(self.loop_note)

        # the jobs
        self.model = JobTableModel(self)
        self.model.set_mode(self.filter, self.last_n)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.table.setWordWrap(False)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        header.setHighlightSections(False)
        header.setSectionsMovable(True)
        for key, width in DEFAULT_WIDTHS.items():
            header.resizeSection(COLUMN_KEYS.index(key), width)
        for key in DEFAULT_HIDDEN:
            self.table.setColumnHidden(COLUMN_KEYS.index(key), True)
        self._restore_columns()
        # written SAVE_DEBOUNCE_MS after the last change (a drag resizes many times)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(SAVE_DEBOUNCE_MS)
        self._save_timer.timeout.connect(self.save_columns)
        header.sectionMoved.connect(lambda *_: self._columns_changed())
        header.sectionResized.connect(self._on_section_resized)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._header_menu)
        header.setToolTip("Drag a header to move its column; right-click: show or hide columns")
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.selectionModel().selectionChanged.connect(self._on_selection)
        self.table.doubleClicked.connect(lambda _i: self.show_log())
        self.table.setToolTip("Right-click a job for its actions (log, source, run again, "
                              "edit, move, remove). Ctrl / Shift-click selects several: run "
                              "again, move to top / bottom, remove and copy act on all. Del: "
                              "remove the selected queued jobs; Enter: log; Ctrl+Up / "
                              "Ctrl+Down: move.")
        for keys, slot in ((QKeySequence.StandardKey.Delete, self.remove_selected),
                           (QKeySequence(Qt.Key.Key_Return), self.show_log),
                           (QKeySequence(Qt.Key.Key_Enter), self.show_log),
                           (QKeySequence("Ctrl+Up"), lambda: self.move_selected("up")),
                           (QKeySequence("Ctrl+Down"), lambda: self.move_selected("down"))):
            shortcut = QShortcut(keys, self.table)
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
            shortcut.activated.connect(lambda s=slot: s())
        box.addWidget(self.table, 1)

        self.message = _small("")
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(self.message)

        self._list_timer = QTimer(self)
        self._list_timer.setSingleShot(True)
        self._list_timer.timeout.connect(self.refresh_list)

        self._show_summary()
        self._show_filter_counts()
        self.refresh_buttons()

    # -- inputs from the host -----------------------------------------------------------

    #: set by shutdown(): every input is then a no-op
    _shut = False
    #: the message line shows a failed list (cleared by the next good one)
    _list_error_shown = False
    #: the last list failed: the next status that answers asks again
    _list_failed = False

    def set_state(self, state: dict | None) -> None:
        """The server's ``status_json`` (None: it did not answer)."""
        if self._shut:
            return
        if not isinstance(state, dict):
            self.set_reachable(False)
            return
        self.set_reachable(True)
        if self._list_failed and not self._listing:
            # the last list failed (e.g. the server was not up yet): ask again
            # now that it answers -- an idle queue's summary never changes, so
            # set_info would never ask
            self.schedule_list()
        if "run_queue" not in state or not isinstance(state.get("run_queue"), dict):
            self.has_queue = False
            self._show_summary()
            self.refresh_buttons()
            return
        self.has_queue = True
        if isinstance(state.get("person_hold"), dict):
            self.hold = dict(state["person_hold"])
        self.set_info(state["run_queue"])
        self.refresh_slot()

    def refresh_slot(self) -> None:
        """The job in the slot again (``describe``): its estimated end moves
        with liveOD's shots, so every status poll refreshes that row."""
        cur = self.info.get("current") or {}
        if (not cur or self._describing or self._listing or not self._listed_once
                or self.model.row_of(cur.get("id")) < 0):
            return
        self._describing = True

        def done(reply: dict) -> None:
            self._describing = False
            if reply.get("status") == "ok" and isinstance(reply.get("job"), dict):
                self.model.update_job(reply["job"])
        self.runner.send({"type": "run_queue", "action": "describe", "id": cur.get("id"),
                          "token": cur.get("token")}, done)

    def set_reachable(self, reachable: bool) -> None:
        if bool(reachable) != self.reachable:
            self.reachable = bool(reachable)
            self._show_summary()
            self.refresh_buttons()

    def on_broadcast(self, payload: dict) -> None:
        """A server broadcast: ``run_queue`` refreshes the summary and asks for
        the list (debounced); ``person_hold`` the hold.  Others are ignored
        (all of them after shutdown)."""
        if self._shut or not isinstance(payload, dict):
            return
        kind = payload.get("type")
        if kind == "run_queue" and isinstance(payload.get("run_queue"), dict):
            self.has_queue = True
            self.set_info(payload["run_queue"], ask_list=False)
            self.schedule_list()
        elif kind == "person_hold" and isinstance(payload.get("person_hold"), dict):
            self.hold = dict(payload["person_hold"])
            self._show_summary()
            self.refresh_buttons()

    def set_info(self, info: dict, ask_list: bool = True) -> None:
        """The queue's summary (``status_json["run_queue"]``)."""
        self.info = dict(info or {})
        if isinstance(self.info.get("person_hold"), dict):
            self.hold = dict(self.info["person_hold"])
        self._show_summary()
        self.refresh_buttons()
        sig = self._signature(self.info)
        if sig != self._summary_sig:
            self._summary_sig = sig
            if ask_list and self._listed_once:
                self.schedule_list()

    @staticmethod
    def _signature(info: dict):
        cur = info.get("current") or {}
        counts = info.get("counts") or {}
        return (cur.get("id"), cur.get("state"), cur.get("run_id"), bool(cur.get("cancel")),
                tuple(info.get("next") or ()), tuple(sorted(counts.items())))

    # -- the list -------------------------------------------------------------------------

    def showEvent(self, event):                                 # noqa: N802
        super().showEvent(event)
        if not self._listed_once:
            self.refresh_list()

    def schedule_list(self) -> None:
        """A ``list`` within :data:`LIST_DEBOUNCE_MS` (several calls, one list)."""
        if not self._shut and not self._list_timer.isActive():
            self._list_timer.start(LIST_DEBOUNCE_MS)

    def refresh_list(self) -> None:
        if self._shut:
            return
        if self._listing:
            self._list_again = True
            return
        self._listing = True
        self.runner.send(list_request(), self._on_list)

    def _on_list(self, reply: dict) -> None:
        self._listing = False
        self._list_failed = not (reply.get("status") == "ok" or is_unknown_request(reply))
        if not self._list_failed:
            self._listed_once = True
            if self._list_error_shown:
                self._list_error_shown = False
                self.say("")
        if reply.get("status") == "ok":
            self.has_queue = True
            info = reply.get("run_queue")
            if isinstance(info, dict):
                self.set_info(info, ask_list=False)
                self._summary_sig = self._signature(self.info)
            self.model.set_jobs(reply.get("jobs"), self.info, reply.get("next"))
            self._reselect()
            listed = {(j.get("id"), j.get("token")) for j in self.model._all}
            if (self.follow_new and self._known_jobs is not None
                    and listed - self._known_jobs):
                self.table.scrollToBottom()
            self._known_jobs = listed
            self._show_filter_counts()
            # the server lists the first `limit` queued jobs; how many it left out
            self.not_listed = 0
            if reply.get("truncated"):
                try:
                    listed = len(self.model.queued_in_order())
                    self.not_listed = max(0, int(reply.get("queued_total")) - listed)
                except (TypeError, ValueError):
                    self.not_listed = -1              # truncated, count unknown
        elif is_unknown_request(reply):
            self.has_queue = False
            self.model.set_jobs([], {})
        else:
            self.say(f"list: {reply.get('msg')}", ERR_TEXT)
            self._list_error_shown = True
        self._show_summary()
        self.refresh_buttons()
        if self._list_again:
            self._list_again = False
            self.schedule_list()

    def set_filter(self, name: str) -> None:
        """``"all"``: every listed job; ``"last"``: the last N ended jobs
        (:attr:`last_n`) and every job not ended; ``"pending"``: queued and
        in the slot (the ended ones hidden)."""
        if name not in FILTERS:
            raise ValueError(f"filter must be one of {FILTERS}, not {name!r}")
        self.filter = name
        if self.filter_combo.currentIndex() != FILTERS.index(name):
            self.filter_combo.setCurrentIndex(FILTERS.index(name))
        self.model.set_mode(name, self.last_n)
        self._put_settings({"filter": name})
        self._after_filter()

    def set_last_n(self, n: int) -> None:
        """How many ended jobs the "last" filter shows."""
        self.last_n = max(1, int(n))
        if self.last_spin.value() != self.last_n:
            self.last_spin.setValue(self.last_n)
        self.model.set_mode(self.filter, self.last_n)
        self._put_settings({"last_n": self.last_n})
        self._after_filter()

    def _after_filter(self) -> None:
        self._reselect()
        self._show_filter_counts()
        self.refresh_buttons()

    def set_follow_new(self, on: bool) -> None:
        """Scroll to the bottom when a list brings a new job (or not)."""
        self.follow_new = bool(on)
        if self.follow_button.isChecked() != self.follow_new:
            self.follow_button.setChecked(self.follow_new)
        self._put_settings({"follow_new": self.follow_new})

    def _show_filter_counts(self) -> None:
        counts = self.model.filter_counts()
        tips = {"all": "Every job the server listed: ended ones too (saved green, failed red, "
                       "no file gray)",
                "last": "The last N ended jobs (N: the box beside) and every job not ended",
                "pending": "Only jobs that have not ended: queued, and the one in the slot"}
        for i, name in enumerate(FILTERS):
            text = (_FILTER_TITLES[name] if name == "last"
                    else f"{_FILTER_TITLES[name]} {counts[name]}")
            if self.filter_combo.itemText(i) != text:
                self.filter_combo.setItemText(i, text)
            self.filter_combo.setItemData(i, tips[name], Qt.ItemDataRole.ToolTipRole)
        self.filter_combo.setToolTip(f"{tips[self.filter]} -- {counts[self.filter]} shown")
        self.last_spin.setVisible(self.filter == "last")

    # -- the summary --------------------------------------------------------------------------

    def queue_state(self) -> str:
        cur = self.info.get("current") or {}
        if cur.get("state") == "launching":
            return "launching"
        return str(self.info.get("state") or "idle")

    def _show_summary(self) -> None:
        if not self.reachable:
            self.pill.setText("?")
            set_style_if_changed(self.pill, _pill_css("unknown"))
            self.summary.setText("The monitor server is not answering.")
        elif not self.has_queue:
            self.pill.setText("none")
            set_style_if_changed(self.pill, _pill_css("unknown"))
            self.summary.setText("This monitor server has no run queue (older code).")
        else:
            state = self.queue_state()
            self.pill.setText(state)
            set_style_if_changed(self.pill, _pill_css(_QUEUE_LEVEL.get(state, "unknown")))
            self.summary.setText(str(self.info.get("text") or ""))
        self.summary.setToolTip(self.summary.text())
        counts = self.info.get("counts") or {}
        shown = [f"{counts[s]} {s}" for s in ("queued", "launching", "running", "ending",
                                               "failed", "aborted", "saved", "done")
                 if counts.get(s)]
        if self.not_listed:
            shown.append(f"{self.not_listed} more queued not listed" if self.not_listed > 0
                         else "more queued not listed")
        self.counts.setText(" | ".join(shown) if self.reachable and self.has_queue else "")

        alarm = self.info.get("alarm") if self.reachable else None
        if isinstance(alarm, dict):
            waited = alarm.get("waited_s")
            self.alarm.setText(
                f"RUN QUEUE ALARM: job {alarm.get('job')} has been ready to start for "
                f"{_dur(waited) if waited is not None else '?'} and nothing has launched"
                f" -- {alarm.get('why') or 'no reason given'}")
            self.alarm.show()
        else:
            self.alarm.hide()

        hold = self.hold
        if hold.get("active"):
            details = [describe_hold(hold)]
            extra = []
            if hold.get("owner"):
                extra.append(f"owner {hold['owner']}")
            if hold.get("source"):
                extra.append(f"source {hold['source']}")
            if hold.get("run_id"):
                extra.append(f"run {hold['run_id']}")
            if extra:
                details.append("(" + ", ".join(extra) + ")")
            text = "HELD: " + " ".join(details) + " -- agents' jobs and the run loops wait."
            self.hold_label.setText(text[0].upper() + text[1:])
            set_style_if_changed(self.hold_label, _banner_css(theme.HELD))
            self.hold_label.setVisible(self.reachable)
        else:
            self.hold_label.setText("No person hold: agents may use the machine when it is free.")
            set_style_if_changed(self.hold_label, f"color: {theme.FG_MUTED}; font-size: 11px;")
            self.hold_label.hide()

        paused = self.info.get("paused") or {}
        for scope, label in self.pause_labels.items():
            p = paused.get(scope)
            title = "Agent jobs" if scope == "agent" else "All jobs"
            if isinstance(p, dict):
                why = f": {p['reason']}" if p.get("reason") else ""
                label.setText(f"{title}: PAUSED by {p.get('by')} ({p.get('owner') or 'person'}) "
                              f"since {_clock(p.get('since'))}{why}")
                set_style_if_changed(label, f"color: {WARN_TEXT}; font-size: 11px;")
                label.setVisible(self.reachable)
            else:
                label.setText(f"{title}: not paused")
                set_style_if_changed(label, f"color: {theme.FG_MUTED}; font-size: 11px;")
                label.hide()

        resume = self.info.get("resume_loop")
        if isinstance(resume, dict) and self.reachable:
            self.loop_note.setText(
                f"The queue stopped the loop {resume.get('key')} for its jobs "
                f"(since {_clock(resume.get('since'))}); it starts it again when it has run out"
                + (f" ({resume['category']})." if resume.get("category") else "."))
            self.loop_note.show()
        else:
            self.loop_note.hide()

    # -- selection and buttons ------------------------------------------------------------------

    def selected_job(self) -> dict | None:
        """The job the actions act on: the current row when it is selected,
        else the first selected row."""
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        current = self.table.currentIndex()
        index = rows[0]
        if current.isValid() and any(r.row() == current.row() for r in rows):
            index = self.model.index(current.row(), 0)
        job = self.model.data(index, Qt.ItemDataRole.UserRole)
        return job if isinstance(job, dict) else None

    def selected_jobs(self) -> list[dict]:
        """Every selected job, in table order."""
        rows = sorted(r.row() for r in self.table.selectionModel().selectedRows())
        jobs = [self.model.data(self.model.index(r, 0), Qt.ItemDataRole.UserRole) for r in rows]
        return [j for j in jobs if isinstance(j, dict)]

    def batch_jobs(self, job: dict | None = None) -> list[dict]:
        """The jobs a batch action acts on: the selection, when ``job`` (the
        row clicked; default the selected one) is in it; else ``job`` alone."""
        job = job if job is not None else self.selected_job()
        selected = self.selected_jobs()
        if job is None:
            return selected
        key = (job.get("id"), job.get("token"))
        if any((j.get("id"), j.get("token")) == key for j in selected):
            return selected
        return [job]

    def select_job(self, job_id) -> bool:
        row = self.model.row_of(job_id)
        if row < 0:
            self.table.clearSelection()
            return False
        self.table.selectRow(row)
        return True

    def _on_selection(self, *_):
        job = self.selected_job()
        self._selected = (job.get("id"), job.get("token")) if job else None
        self.refresh_buttons()

    def _reselect(self) -> None:
        if self._selected is None:
            return
        row = self.model.row_of(*self._selected)
        blocker = self.table.selectionModel().blockSignals(True)
        try:
            if row >= 0:
                self.table.selectRow(row)
            else:
                self.table.clearSelection()
        finally:
            self.table.selectionModel().blockSignals(blocker)
        if row < 0:
            self._selected = None

    def supported(self, action: str) -> bool:
        return action not in self.unsupported

    def action_state(self, kind: str, job: dict | None = None) -> tuple[bool, str]:
        """``(enabled, tooltip)`` of a job action for ``job`` (default: the
        selected one).  Kinds: ``log``, ``source``, ``again``, ``again_next``,
        ``edit``, ``up`` / ``down`` / ``top`` / ``bottom``, ``remove`` (the
        selected queued jobs), ``abort`` (the job in the slot), ``kill`` (a
        launching or running job's processes), ``copy``."""
        job = job if job is not None else self.selected_job()
        live = self.reachable and self.has_queue
        if job is None:
            return False, "select a job"
        state = job.get("state")
        if kind in ("log", "copy"):
            return (live or kind == "copy"), ("Follow the job's log" if kind == "log" else
                                               "Copy a command line or a field")
        if kind == "source":
            return bool(job.get("path")), "The experiment file as it is now (read-only view)"
        why_not = ("the monitor server is not answering" if not self.reachable else
                   "this monitor server has no run queue" if not self.has_queue else "")
        if why_not:
            return False, why_not
        if kind in ("again", "again_next"):
            jobs = [j for j in self.batch_jobs(job) if j.get("path")]
            n = len(jobs)
            what = "the same file again as a new job" if n <= 1 else \
                f"the {n} selected jobs' files again as new jobs, in table order"
            return bool(jobs), ("Queue " + what if kind == "again" else
                                "Queue " + what + ", ahead of every queued job")
        if kind == "remove":
            queued = [j for j in self.selected_jobs() or [job] if j.get("state") == "queued"]
            return (bool(queued), "Take the selected queued jobs out of the queue (asks first)"
                    if queued else "only a queued job can be removed (a running one: Abort)")
        if kind == "abort":
            if state not in ("launching", "running"):
                return False, f"it is {state}: no run to abort"
            if job.get("cancel"):
                return False, "its cancel has been asked for; the run ends at its next shot"
            if ran_without_file(job):
                return True, "Stop its run at the next shot (save_data=False: it has no file)"
            return True, "Stop its run at the next shot: liveOD DISCARDS the run's data file"
        if kind == "kill":
            if not self.supported(KILL_ACTION):
                return False, "this monitor server cannot force-kill jobs (older code)"
            if state not in ("launching", "running"):
                return False, f"it is {state}: no process to kill"
            return True, ("Kill its processes now, when the Abort cannot work (liveOD closed, "
                          "its terminal or process hung, the queue out of step): NO cleanup "
                          "runs; liveOD keeps the run's file as it was")
        if kind == "edit":
            if not self.supported(EDIT_ACTION):
                return False, "this monitor server cannot edit jobs (older code)"
            return (state == "queued", "Edit the queued job" if state == "queued"
                    else "only a queued job can be edited")
        if kind in MOVE_WHERE:
            if not self.supported(MOVE_ACTION):
                return False, "this monitor server cannot reorder jobs (older code)"
            queued = [j for j in self.batch_jobs(job) if j.get("state") == "queued"]
            if kind in ("top", "bottom") and len(queued) > 1:
                return True, (f"Move the {len(queued)} selected queued jobs to the "
                              f"{kind}, in their order")
            if state != "queued":
                return False, "only a queued job can be moved"
            if move_request(job, kind, self.by, self.model.queued_in_order()) is None:
                return False, "it is already first" if kind in ("up", "top") else \
                    "it is already last"
            return True, _MOVE_TIPS[kind]
        raise ValueError(f"no such job action: {kind!r}")

    def refresh_buttons(self) -> None:
        live = self.reachable and self.has_queue
        why_not = ("the monitor server is not answering" if not self.reachable else
                   "this monitor server has no run queue" if not self.has_queue else "")
        self.hold_button.setEnabled(live)
        self.hold_button.setText("Release hold" if self.hold.get("active") else "Hold...")
        set_style_if_changed(self.hold_button,
                             theme.held_button_css() if self.hold.get("active") else "")
        self.hold_button.setToolTip(
            why_not if not live else
            ("Release the person hold: agents' queued jobs and the run loops may run again."
             if self.hold.get("active") else
             "Put a person's hold on the machine: agents' jobs wait and the run loops stop "
             "after their run in progress, until someone releases it."))
        paused = self.info.get("paused") or {}
        for scope, b in self.pause_buttons.items():
            on = isinstance(paused.get(scope), dict)
            b.setText(("Resume " if on else "Pause ") + scope)
            set_style_if_changed(b, theme.held_button_css() if on else "")
            b.setEnabled(live)
            b.setToolTip(why_not if not live else
                         (f"Let {scope} jobs launch again" if on else
                          f"Stop launching {scope} jobs (the running one goes on)"))
        self.refresh_button.setEnabled(self.reachable)

    def say(self, text: str, color: str = theme.FG_MUTED) -> None:
        self._list_error_shown = False        # a newer message is not cleared by a list
        self.message.setText(text)
        set_style_if_changed(self.message, f"color: {color}; font-size: 11px;")

    # -- actions ---------------------------------------------------------------------------------

    def _done(self, what: str, on_unknown: str | None = None) -> Callable[[dict], None]:
        def done(reply: dict) -> None:
            if reply.get("status") == "ok" and reply.get("clamped"):
                # the server placed it elsewhere than asked (behind person jobs)
                self.say(f"{what}: done, but the position was clamped behind person jobs",
                         WARN_TEXT)
            elif reply.get("status") == "ok":
                self.say(f"{what}: done" + (" (the run ends at its next shot)"
                                            if reply.get("pending") else ""), OK_TEXT)
            if reply.get("status") == "ok":
                info = reply.get("run_queue")
                if isinstance(info, dict):
                    self.set_info(info, ask_list=False)
                if isinstance(reply.get("person_hold"), dict):
                    self.hold = dict(reply["person_hold"])
                    self._show_summary()
                self.schedule_list()
            elif on_unknown is not None and is_unknown_request(reply):
                self.unsupported.add(on_unknown)
                self.say(f"{what}: this monitor server does not offer it (older code)",
                         WARN_TEXT)
            else:
                self.say(f"{what}: {reply.get('msg')}", ERR_TEXT)
            self.refresh_buttons()
        return done

    def cancel_selected(self) -> bool:
        job = self.selected_job()
        if job is None:
            return False
        state = job.get("state")
        if state == "queued":
            title = f"Cancel {job_name(job)}?"
            text = ("It has not started: it is taken out of the queue and never runs.")
            yes, no = "Cancel the job", "Keep it"
        elif state in ("launching", "running") and ran_without_file(job):
            title = f"Abort {job_name(job)}?"
            text = (f"{job_name(job)} is {state} with save_data=False: liveOD writes no file "
                    "for it.\n\nCancelling it sends liveOD's Abort (Reset) for its run: the run "
                    "stops at its next shot.")
            yes, no = "Abort the run", "Keep it running"
        elif state in ("launching", "running"):
            run = job.get("run_id")
            title = f"Abort {job_name(job)}?"
            text = (f"{job_name(job)} is {state}"
                    + (f" (run {run})" if run else " (no run id yet)") + ".\n\n"
                    "Cancelling it sends liveOD's Abort (Reset) for its run: the run stops at "
                    "its next shot and liveOD DISCARDS THAT RUN'S DATA FILE.\n\n"
                    "There is no way yet to stop a queued run and keep its data.")
            yes, no = "Abort the run and discard its data", "Keep it running"
        else:
            return False
        if not self.confirm(title, text, yes, no):
            return False

        def done(reply: dict) -> None:
            if reply.get("status") != "ok" and state == "queued" and reply.get("state") in IN_SLOT:
                self.say(f"{job_name(job)} launched before the cancel arrived: not cancelled. "
                         "Press Cancel again to abort its run (that discards its data).",
                         WARN_TEXT)
                self.schedule_list()
                return
            self._done(f"cancel {job_name(job)}")(reply)
        self.runner.send(cancel_request(job, self.by), done)
        return True

    def kill_selected(self, job: dict | None = None) -> bool:
        """Force-kill the job's processes (asks first): :func:`kill_request`."""
        job = job or self.selected_job()
        ok, _ = self.action_state("kill", job) if job else (False, "")
        if not ok:
            return False
        run = job.get("run_id")
        text = (f"{job_name(job)} is {job.get('state')}"
                + (f" (run {run})" if run else " (no run id yet)") + ".\n\n"
                "Use this only when Abort run cannot work: liveOD closed, the experiment's "
                "terminal or process hung, or the queue out of step with what runs.\n\n"
                "The monitor server kills the experiment's process and everything under it. "
                "NOTHING of the experiment's cleanup runs: coils, AOMs and shutters stay as the "
                "run left them until the next run or the monitor sets them.\n\n"
                "liveOD is told the run's process exited and leaves its file as it was: not "
                "saved, not deleted.\n\n"
                "Queued jobs go on launching afterwards; pause them first if they should wait.")
        if not self.confirm(f"Force kill {job_name(job)}?", text,
                            "Kill it (no cleanup)", "Don't kill"):
            return False
        what = f"force kill {job_name(job)}"

        def done(reply: dict) -> None:
            if reply.get("status") == "ok":
                self.say(f"{what}: {reply.get('note') or 'done'}; the job ends at the queue's "
                         "next check", WARN_TEXT)
                self.schedule_list()
                self.refresh_buttons()
                return
            self._done(what, on_unknown=KILL_ACTION)(reply)
        self.runner.send(kill_request(job, self.by), done)
        return True

    def move_selected(self, where: str) -> bool:
        """Move the selected job ``where``; to the top / bottom, every selected
        queued job, keeping their order (top: the last sent first)."""
        if not self.supported(MOVE_ACTION):
            return False
        order = self.model.queued_in_order()
        if where in ("top", "bottom"):
            picked = {(j.get("id"), j.get("token")) for j in self.selected_jobs()
                      if j.get("state") == "queued"}
            jobs = [j for j in order if (j.get("id"), j.get("token")) in picked]
            if len(jobs) > 1:
                if where == "top":
                    jobs.reverse()
                requests = [(j, {"type": "run_queue", "action": MOVE_ACTION, "id": j.get("id"),
                                 "token": j.get("token"), "owner": OWNER, "by": self.by,
                                 "to_index": 0 if where == "top" else len(order)})
                            for j in jobs]
                self._send_batch(requests, f"move {len(jobs)} jobs to the {where}",
                                 on_unknown=MOVE_ACTION)
                return True
        job = self.selected_job()
        if job is None or job.get("state") != "queued":
            return False
        request = move_request(job, where, self.by, self.model.queued_in_order())
        if request is None:
            return False
        self.runner.send(request,
                         self._done(f"move {job_name(job)} {where}", on_unknown=MOVE_ACTION))
        return True

    def edit_selected(self) -> bool:
        job = self.selected_job()
        if job is None or job.get("state") != "queued" or not self.supported(EDIT_ACTION):
            return False
        changes = self.ask_edit(job)
        if not changes:
            return False
        self.runner.send(edit_request(job, changes, self.by),
                         self._done(f"edit {job_name(job)}", on_unknown=EDIT_ACTION))
        return True

    def toggle_hold(self) -> bool:
        if self.hold.get("active"):
            request = release_request(self.by)
            what = "release the hold"
        else:
            reason = self.ask_text("Hold: a person has the machine",
                                   "Why (shown to everyone, and to agents waiting for the "
                                   "machine):", "a person has the machine")
            if reason is None:
                return False
            request = hold_request(reason or "a person has the machine", self.by)
            what = "hold"
        self.runner.send(request, self._done(what))
        return True

    def toggle_pause(self, scope: str) -> bool:
        paused = (self.info.get("paused") or {}).get(scope)
        if isinstance(paused, dict):
            self.runner.send(resume_request(scope, self.by), self._done(f"resume {scope} jobs"))
            return True
        reason = self.ask_text(f"Pause {scope} jobs", "Reason (optional):", "")
        if reason is None:
            return False
        self.runner.send(pause_request(scope, reason, self.by), self._done(f"pause {scope} jobs"))
        return True

    def copy_kq(self, kind: str = "submit") -> str:
        job = self.selected_job()
        if job is None:
            return ""
        text = kq_commands(job)[kind]
        try:
            QGuiApplication.clipboard().setText(text)
        except Exception:                             # noqa: BLE001
            pass
        self.say(f"copied: {text}")
        return text

    def copy_text(self, text: str, what: str) -> str:
        try:
            QGuiApplication.clipboard().setText(text)
        except Exception:                             # noqa: BLE001
            pass
        self.say(f"copied {what}: " + text.replace("\n", "; "))
        return text

    def remove_selected(self) -> bool:
        """Take the selected queued jobs out of the queue (asks first): each
        is cancelled ``queued_only``, so one that launched meanwhile is left
        running and named.  The queue keeps their records (as cancelled).  A
        selected job that is not queued is left alone."""
        jobs = [j for j in self.selected_jobs() if j.get("state") == "queued"]
        if not jobs or not (self.reachable and self.has_queue):
            if any(j.get("state") in IN_SLOT for j in self.selected_jobs()):
                self.say("A running job is not removed: right-click it for Abort run (that "
                         "discards its run's data).", WARN_TEXT)
            return False
        names = "\n".join(f"- {job_name(j)}" for j in jobs[:12])
        if len(jobs) > 12:
            names += f"\n- ... and {len(jobs) - 12} more"
        n = len(jobs)
        if not self.confirm(f"Remove {n} job{'s' if n > 1 else ''} from the queue?",
                            f"{names}\n\nNot started: taken out of the queue, never run. The "
                            "queue keeps the record (as cancelled).",
                            "Remove", "Keep"):
            return False
        results: dict = {"ok": [], "launched": [], "failed": []}

        def done_for(job: dict) -> Callable[[dict], None]:
            def done(reply: dict) -> None:
                if reply.get("status") == "ok":
                    results["ok"].append(job)
                elif reply.get("state") in IN_SLOT:
                    results["launched"].append(job)
                else:
                    results["failed"].append((job, reply.get("msg")))
                if sum(len(v) for v in results.values()) == n:
                    self._removed(results)
            return done
        for job in jobs:
            self.runner.send(cancel_request(job, self.by), done_for(job))
        return True

    def _removed(self, results: dict) -> None:
        parts, color = [], OK_TEXT
        if results["ok"]:
            parts.append("removed " + ", ".join(str(j.get("id")) for j in results["ok"]))
        if results["launched"]:
            color = WARN_TEXT
            parts.append("launched before the remove arrived, still running: "
                         + ", ".join(job_name(j) for j in results["launched"]))
        if results["failed"]:
            color = ERR_TEXT
            parts.append("; ".join(f"{job_name(j)}: {msg}" for j, msg in results["failed"]))
        self.say("; ".join(parts), color)
        self.schedule_list()
        self.refresh_buttons()

    def run_again(self, job: dict | list | None = None, first: bool = False) -> bool:
        """The same file as a new job, at once (no question):
        :func:`again_request`.  ``job``: one job, a list of them, or None for
        the selected ones (:meth:`batch_jobs`).  Several go in in table
        order; ``first``: as the next jobs (inserted last first, so they run
        in table order).  The message line says when this PC's copy of a file
        differs from what the job ran or cannot be read, and what is not
        carried over."""
        if isinstance(job, list):
            jobs = job
        else:
            jobs = self.batch_jobs(job) if job is None else [job]
        jobs = [j for j in jobs if isinstance(j, dict) and j.get("path")]
        if not jobs or not self.action_state("again", jobs[0])[0]:
            return False
        host = socket.gethostname()
        notes, requests = [], []
        for j in jobs:
            sha = local_sha256(str(j.get("path")))
            if sha is None:
                notes.append(f"{job_name(j)}: this PC cannot read its file (the queue refuses "
                             "it unless this PC is the queue machine)")
            elif j.get("sha256") and sha != j.get("sha256"):
                notes.append(f"{job_name(j)}: its file has CHANGED since it was submitted "
                             "(the current version runs)")
            dropped = not_carried_over(j)
            if dropped:
                notes.append(f"{job_name(j)}: not carried over: " + "; ".join(dropped))
            requests.append((j, again_request(j, self.by, first=first, client_sha256=sha,
                                              client_host=host)))
        if first:
            requests.reverse()
        n = len(jobs)
        what = ((f"run {job_name(jobs[0])} again" if n == 1 else f"run {n} jobs again")
                + (" next" if first else ""))

        def report(results: list) -> str:
            ids = [i for _, reply in results if reply.get("status") == "ok"
                   for i in (reply.get("ids") or [])]
            if first:
                ids.reverse()                         # sent last first
            text = (f"queued as job{'s' if len(ids) > 1 else ''} "
                    + ", ".join(map(str, ids))) if ids else ""
            if any(reply.get("clamped") for _, reply in results):
                text += " (placed behind person jobs)"
            return text
        self._send_batch(requests, what, notes=notes, report=report)
        return True

    def _send_batch(self, requests: list, what: str, *, on_unknown: str | None = None,
                    notes: list | None = None, report=None) -> None:
        """Send ``requests`` (``[(job, request), ...]``) in order; when every
        reply is in, one message: ``report(results)`` (or "done"), then the
        refusals by job, then ``notes``."""
        results: list = []
        n = len(requests)

        def done_for(job: dict) -> Callable[[dict], None]:
            def done(reply: dict) -> None:
                results.append((job, reply))
                if len(results) < n:
                    return
                ok = [r for r in results if r[1].get("status") == "ok"]
                bad = [r for r in results if r[1].get("status") != "ok"]
                if on_unknown is not None and any(is_unknown_request(r) for _, r in bad):
                    self.unsupported.add(on_unknown)
                parts = []
                if ok:
                    parts.append((report(results) if report else "") or "done")
                parts += [f"{job_name(j)}: {r.get('msg')}" for j, r in bad]
                color = (ERR_TEXT if bad else
                         WARN_TEXT if notes or any(r.get("clamped") for _, r in ok) else OK_TEXT)
                self.say(f"{what}: " + "; ".join(parts + list(notes or [])), color)
                if ok:
                    for _, reply in ok[-1:]:
                        if isinstance(reply.get("run_queue"), dict):
                            self.set_info(reply["run_queue"], ask_list=False)
                    self.schedule_list()
                self.refresh_buttons()
            return done
        for job, request in requests:
            self.runner.send(request, done_for(job))

    def _source_note(self, job: dict) -> str:
        """"" or why the file shown may not be what the job ran."""
        sha = local_sha256(str(job.get("path") or ""))
        if sha is None or not job.get("sha256") or sha == job.get("sha256"):
            return ""
        if job.get("state") == "queued":
            return (f"The file has changed since {job_name(job)} was submitted (the queue skips "
                    "it at launch unless it allows drift).")
        run = f" (run {job['run_id']})" if job.get("run_id") else ""
        return (f"The file has changed since {job_name(job)} was submitted: what ran{run} is "
                "the source text in its run file (attribute expt_file), not this.")

    def open_source(self, job: dict | None = None) -> str:
        """The job's file in VS Code when this PC has it and can read the file,
        else in a read-only window (:meth:`view_source`).  Never the OS's
        default action for a .py file (it can run it).  -> "editor",
        "viewer" or "" (the file cannot be read from this PC)."""
        job = job or self.selected_job()
        if job is None or not job.get("path"):
            return ""
        path = str(job["path"])
        if not os.path.isfile(path):
            self.say(f"{path} cannot be read from this PC (the file that runs is the queue "
                     "machine's copy).", WARN_TEXT)
            return ""
        editor = find_editor()
        if editor is None:
            return "viewer" if self.view_source(job) else ""
        try:
            self.launch(editor + [path])
        except OSError as exc:
            self.say(f"could not start the editor ({exc}); showing it read-only", WARN_TEXT)
            return "viewer" if self.view_source(job) else ""
        note = self._source_note(job)
        self.say(f"opened {path} in VS Code" + (f" -- {note}" if note else ""),
                 WARN_TEXT if note else theme.FG_MUTED)
        return "editor"

    def view_source(self, job: dict | None = None) -> "SourceWindow | None":
        """The job's file, read-only, in a window of this panel's."""
        job = job or self.selected_job()
        if job is None or not job.get("path"):
            return None
        path = str(job["path"])
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(4_000_000)
        except OSError as exc:
            self.say(f"{path} cannot be read from this PC ({exc}).", WARN_TEXT)
            return None
        window = SourceWindow(self, job, text, self._source_note(job))
        for old in [w for w in self._source_windows if not w.isVisible()]:
            self._source_windows.remove(old)
            old.deleteLater()                         # closed by hand earlier
        self._source_windows.append(window)
        window.show()
        return window

    # -- the menus -------------------------------------------------------------------------------

    def job_menu(self, job: dict | None = None) -> tuple[QMenu, dict[str, QAction]]:
        """The right-click menu for ``job`` (default: the selected one) and its
        actions by kind (:meth:`action_state`; ``copy_<what>`` for the copy
        submenu)."""
        job = job if job is not None else self.selected_job()
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        acts: dict[str, QAction] = {}

        def add(target: QMenu, kind: str, text: str, slot, shortcut: str = "") -> QAction:
            act = QAction(text, target)
            ok, tip = self.action_state(kind.split("_")[0] if kind.startswith("copy_") else kind,
                                        job)
            act.setEnabled(ok)
            act.setToolTip(tip)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            act.triggered.connect(lambda _=False: slot())
            target.addAction(act)
            acts[kind] = act
            return act

        if job is None:
            return menu, acts
        add(menu, "log", "Show log", lambda: self.show_log(job), "Return")
        add(menu, "source", "Open source", lambda: self.open_source(job))
        act = QAction("View source (read-only)", menu)
        act.setEnabled(bool(job.get("path")))
        act.triggered.connect(lambda _=False: self.view_source(job))
        menu.addAction(act)
        acts["view_source"] = act
        menu.addSeparator()
        batch = self.batch_jobs(job)
        n_again = sum(1 for j in batch if j.get("path"))
        many = f" {n_again} jobs" if n_again > 1 else ""
        add(menu, "again", f"Run{many} again", lambda: self.run_again(batch))
        add(menu, "again_next", f"Run{many} again next",
            lambda: self.run_again(batch, first=True))
        menu.addSeparator()
        state = job.get("state")
        n_queued = sum(1 for j in batch if j.get("state") == "queued")
        if state == "queued" or n_queued:
            if state == "queued":
                add(menu, "edit", "Edit...", lambda: self.edit_selected())
            if state == "queued" or n_queued > 1:
                move = menu.addMenu(f"Move {n_queued} jobs" if n_queued > 1 else "Move")
                move.setToolTipsVisible(True)
                moves = (("top", "To the top (next)", ""), ("bottom", "To the bottom", ""))
                if n_queued == 1:
                    moves = (("up", "Up", "Ctrl+Up"), ("down", "Down", "Ctrl+Down")) + moves
                for where, text, keys in moves:
                    add(move, where, text, lambda w=where: self.move_selected(w), keys)
            add(menu, "remove", f"Remove {n_queued} jobs from the queue..." if n_queued > 1
                else "Remove from queue...", self.remove_selected, "Del")
        if state in IN_SLOT:
            add(menu, "abort", "Abort run (no file)..." if ran_without_file(job) else
                "Abort run (discards its data)...", self.cancel_selected)
            if state != "ending":
                add(menu, "kill", "Force kill (no cleanup)...", lambda: self.kill_selected(job))
        menu.addSeparator()
        copy = menu.addMenu("Copy")
        commands = kq_commands(job)
        for kind, text in (("submit", "kq submit ... (the same job again)"),
                           ("tail", "kq tail <id> -f"), ("show", "kq show <id>"),
                           ("cancel", "kq cancel <id>"), ("kill", "kq kill <id>")):
            add(copy, f"copy_{kind}", text,
                lambda k=kind: self.copy_text(commands[k], f"kq {k}"))
        add(copy, "copy_path", "File path",
            lambda: self.copy_text(str(job.get("path") or ""), "the path"))
        if job.get("run_id") not in (None, ""):
            add(copy, "copy_run", f"Run id ({job['run_id']})",
                lambda: self.copy_text(str(job["run_id"]), "the run id"))
        if len(batch) > 1:
            copy.addSeparator()
            paths = "\n".join(str(j.get("path") or "") for j in batch)
            add(copy, "copy_paths", f"File paths ({len(batch)} jobs)",
                lambda: self.copy_text(paths, f"{len(batch)} paths"))
            runs = [str(j["run_id"]) for j in batch if j.get("run_id") not in (None, "")]
            if runs:
                add(copy, "copy_runs", f"Run ids ({len(runs)})",
                    lambda: self.copy_text(" ".join(runs), f"{len(runs)} run ids"))
        return menu, acts

    def _context_menu(self, pos) -> None:
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        if not self.table.selectionModel().isRowSelected(index.row(), QModelIndex()):
            self.table.selectRow(index.row())
        self.table.setCurrentIndex(index)
        job = self.model.data(index, Qt.ItemDataRole.UserRole)
        menu, _ = self.job_menu(job)
        menu.exec(self.table.viewport().mapToGlobal(pos))
        delete_later(menu)

    def _header_menu(self, pos) -> None:
        menu = QMenu(self)
        for i, (key, title) in enumerate(COLUMNS):
            act = QAction(title, menu)
            act.setCheckable(True)
            act.setChecked(not self.table.isColumnHidden(i))
            act.toggled.connect(lambda on, c=i: self.set_column_shown(c, on))
            menu.addAction(act)
        menu.addSeparator()
        menu.addAction("Default columns", self.default_columns)
        menu.exec(self.table.horizontalHeader().mapToGlobal(pos))
        delete_later(menu)

    def set_column_shown(self, column: int, shown: bool) -> None:
        self.table.setColumnHidden(column, not shown)
        self._columns_changed()

    def default_columns(self) -> None:
        """The columns as built: their order, widths and shown set."""
        header = self.table.horizontalHeader()
        for logical in range(len(COLUMNS)):
            header.moveSection(header.visualIndex(logical), logical)
        for key, width in DEFAULT_WIDTHS.items():
            header.resizeSection(COLUMN_KEYS.index(key), width)
        for i, key in enumerate(COLUMN_KEYS):
            self.table.setColumnHidden(i, key in DEFAULT_HIDDEN)
        self._columns_changed()

    # -- remembered columns and filter (only with ``settings``) -------------------------------

    def _setting(self, key: str):
        if self.settings is None:
            return None
        try:
            return self.settings.value(key)
        except Exception:                             # noqa: BLE001
            return None

    def _put_settings(self, values: dict) -> None:
        if self.settings is None:
            return
        try:
            for key, value in values.items():
                self.settings.setValue(key, value)
            self.settings.sync()
        except Exception:                             # noqa: BLE001
            pass

    def _restore_columns(self) -> None:
        """The saved order, widths and shown set -- only when they were saved
        for this panel's columns (a version with other columns starts from
        the defaults)."""
        state = self._setting("columns/state")
        if not state or self._setting("columns/keys") != ",".join(COLUMN_KEYS):
            return
        try:
            self.table.horizontalHeader().restoreState(QByteArray(state))
        except Exception:                             # noqa: BLE001
            pass

    def _on_section_resized(self, logical: int, _old: int, _new: int) -> None:
        # the stretched last column follows the window's width: not a choice
        header = self.table.horizontalHeader()
        last = max((i for i in range(len(COLUMNS)) if not header.isSectionHidden(i)),
                   key=header.visualIndex, default=-1)
        if logical != last:
            self._columns_changed()

    def _columns_changed(self) -> None:
        if self.settings is not None and not self._shut:
            self._save_timer.start()

    def save_columns(self) -> None:
        """Write the columns' order, widths and shown set now."""
        self._save_timer.stop()
        self._put_settings({"columns/state": self.table.horizontalHeader().saveState(),
                            "columns/keys": ",".join(COLUMN_KEYS)})

    def show_log(self, job: dict | None = None) -> "JobLogWindow | None":
        job = job or self.selected_job()
        if job is None:
            return None
        key = (job.get("id"), job.get("token"))
        window = self._log_windows.get(key)
        if window is None:
            window = JobLogWindow(self, job)
            self._log_windows[key] = window
            window.show()
            window.fetch()
        else:
            window.raise_()
            window.activateWindow()
        return window

    def forget_log_window(self, window: JobLogWindow) -> None:
        for key, w in list(self._log_windows.items()):
            if w is window:
                del self._log_windows[key]

    # -- dialogs (tests replace them) ----------------------------------------------------------

    def confirm(self, title: str, text: str, yes: str = "OK", no: str = "Cancel") -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        yes_b = box.addButton(yes, QMessageBox.ButtonRole.AcceptRole)
        no_b = box.addButton(no, QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(no_b)
        box.exec()
        clicked = box.clickedButton()
        delete_later(box)
        return clicked is yes_b

    def ask_text(self, title: str, prompt: str, default: str = "") -> str | None:
        text, ok = QInputDialog.getText(self, title, prompt, QLineEdit.EchoMode.Normal, default)
        return text.strip() if ok else None

    def ask_edit(self, job: dict) -> dict | None:
        dialog = EditJobDialog(job, self)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            return dialog.changes()
        finally:
            delete_later(dialog)

    # -- shutdown ----------------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the timers and log windows; from now on state, broadcasts
        and refreshes are ignored (nothing is sent).  A column change not yet
        written is written now."""
        if self._save_timer.isActive():
            self.save_columns()
        self._shut = True
        self._list_timer.stop()
        for window in list(self._log_windows.values()):
            window.stopped = True
            window.timer.stop()
            window.close()
        self._log_windows.clear()
        for window in self._source_windows:
            try:
                window.close()
            except RuntimeError:                      # already deleted (closed by hand)
                pass
        self._source_windows = []
        if self._own_runner:
            self.runner.shutdown()


class QueueSummaryLine(QFrame):
    """One line for hosts that do not show the queue (the Device Control GUI):
    "Queue: <state> (<n> queued) -- open the Monitor panel in the Server
    Dashboard", from ``status_json["run_queue"]`` alone (no requests).
    Hidden while the server reports no queue."""

    HINT = "open the Monitor panel in the Server Dashboard"

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 2, 6, 2)
        self.label = _small("")
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.label, 1)
        self.hide()

    @staticmethod
    def text_for(info: dict) -> str:
        state = str(info.get("state") or "idle")
        cur = info.get("current") or {}
        if cur.get("state") == "launching":
            state = "launching"
        n = int((info.get("counts") or {}).get("queued") or 0)
        alarm = " -- ALARM" if info.get("alarm") else ""
        return f"Queue: {state} ({n} queued){alarm} -- {QueueSummaryLine.HINT}"

    def set_info(self, info: dict | None) -> None:
        if not isinstance(info, dict):
            self.hide()
            return
        self.label.setText(self.text_for(info))
        self.label.setToolTip(str(info.get("text") or ""))
        color = WARN_TEXT if info.get("alarm") else theme.FG_MUTED
        set_style_if_changed(self.label, f"color: {color}; font-size: 11px;")
        self.show()


__all__ = ["COLUMNS", "DEFAULT_HIDDEN", "EditJobDialog", "FILTERS", "JobLogWindow",
           "JobTableModel", "QueueSummaryLine", "RunQueuePanel", "SourceWindow", "again_request",
           "cancel_request", "chip_css", "derived_waiting", "edit_request", "find_editor",
           "hold_request", "kq_commands", "list_request", "move_request", "not_carried_over",
           "order_jobs", "pause_request", "ran_without_file", "release_request",
           "resume_request", "tail_request"]
