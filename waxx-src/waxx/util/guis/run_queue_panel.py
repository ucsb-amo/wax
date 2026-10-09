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

The controls act on the selected job: Cancel (asks first; for a job in the
slot the question says that liveOD's Abort discards that run's data file --
nothing else is offered until the queue can stop a run and keep its data; a
queued job's cancel is sent ``queued_only``, so one that launched meanwhile
is never aborted on a question about a queued job), Move up / down / to top
/ to bottom (``move``: up / down name the neighbour by id -- ``before_id`` /
``after_id`` -- top / bottom send ``to_index``), Edit (``edit`` with
``fields``: argv, label, after, chain, stop_on_failure, the write-back veto,
due, allow_drift, paused; only the fields changed), Show log (a window
following the job's log through the ``tail``
action: every 0.5 s while it runs, every 2 s while it is queued, until it has
ended and its log is read to the end), and Copy kq command.  An action the
server refuses as unknown (an older server) disables its control; so does an
``actions`` list in the queue's info that does not name it.

Every request that changes something carries ``owner`` "person" (a person is
clicking) and ``by`` = ``user@host``.

Requests: a full ``list`` when the panel is first shown and after every
``run_queue`` broadcast (debounced, :data:`LIST_DEBOUNCE_MS`), and after
``set_state`` sees the queue's summary change; the summary between comes from
``status_json``.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import shlex
import time
from typing import Any, Callable

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QGuiApplication
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QFrame, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox, QPlainTextEdit,
    QPushButton, QTableView, QToolButton, QVBoxLayout, QWidget,
)

from waxx.util.dashboard import theme
from waxx.util.device_state.person_hold import describe as describe_hold
from waxx.util.device_state.run_queue_client import ENDED, IN_SLOT, default_by
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
#: Where a ``move`` request sends a job (its ``to`` field).
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
#: job state -> text colour in the table
_STATE_COLOR = {"queued": theme.FG, "launching": WARN_TEXT, "running": OK_TEXT,
                "ending": WARN_TEXT, "saved": theme.FG_MUTED, "failed": ERR_TEXT,
                "cancelled": theme.FG_MUTED, "skipped": theme.FG_MUTED}


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


def _q(arg: str) -> str:
    arg = str(arg)
    return f'"{arg}"' if (" " in arg or "\t" in arg or not arg) else arg


def kq_commands(job: dict) -> dict[str, str]:
    """The kq command lines for a job: ``submit`` (the same job again),
    ``tail`` (follow its log), ``show``, ``cancel``."""
    jid = job.get("id")
    parts = ["kq", "submit", _q(job.get("path") or "")]
    if job.get("label"):
        parts += ["--label", _q(job["label"])]
    if job.get("priority") is not None:
        parts += ["--priority", str(job["priority"])]
    if job.get("chain"):
        parts += ["--chain", _q(job["chain"])]
        if not job.get("stop_on_failure"):
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
            "cancel": f"kq cancel {jid}"}


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
             if a in by_id and by_id[a].get("state") != "saved"]
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
        self.show_ended = True

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

    def set_show_ended(self, show: bool) -> None:
        self.beginResetModel()
        self.show_ended = bool(show)
        self._filter()
        self.endResetModel()

    def _filter(self) -> None:
        self.jobs = [j for j in self._all if self.show_ended or j.get("state") not in ENDED]

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
        if role == Qt.ItemDataRole.ForegroundRole:
            if key == "source_changed" and job.get("source_changed"):
                return QColor(ERR_TEXT)
            if key == "est" and self.text(job, key):
                return QColor(theme.FG_MUTED)
            return QColor(_STATE_COLOR.get(str(job.get("state")), theme.FG))
        if role == Qt.ItemDataRole.UserRole:
            return job
        return None

    # -- cell text ---------------------------------------------------------------------

    def queued_in_order(self) -> list[dict]:
        """The queued jobs in the server's order (all of them, shown or not)."""
        return [j for j in self._all if j.get("state") == "queued"]

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
            if job.get("cancel") and s in IN_SLOT:
                s += " (cancel asked)"
            elif job.get("paused") and s == "queued":
                s += " (paused)"
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
            return ("argv: " + " ".join(str(a) for a in argv)) if argv else ""
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


# -- the panel -----------------------------------------------------------------------------

class RunQueuePanel(QWidget):
    """The run queue (see the module docstring).

    ``requester``: ``request_dict -> reply_dict``; ``by``: who is clicking
    (default ``user@host``); ``synchronous``: call the requester on the GUI
    thread (tests).  ``runner``: an existing RequestRunner to share instead."""

    def __init__(self, requester: Callable[[dict], Any] | None = None, *, by: str | None = None,
                 synchronous: bool = False, runner: RequestRunner | None = None, parent=None):
        super().__init__(parent)
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
        self._summary_sig = None
        self._selected: tuple | None = None
        self._log_windows: dict = {}

        box = QVBoxLayout(self)
        box.setContentsMargins(8, 8, 8, 8)
        box.setSpacing(6)

        # top strip
        top = QHBoxLayout()
        self.pill = QLabel("?")
        top.addWidget(self.pill)
        self.summary = QLabel("")
        self.summary.setStyleSheet(f"color: {theme.FG}; font-size: 12px;")
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(self.summary, 1)
        self.counts = _small("")
        self.counts.setWordWrap(False)
        top.addWidget(self.counts)
        box.addLayout(top)

        self.alarm = QLabel("")
        self.alarm.setWordWrap(True)
        self.alarm.setStyleSheet(_banner_css("#c62828"))
        self.alarm.hide()
        box.addWidget(self.alarm)

        hold_row = QHBoxLayout()
        self.hold_label = QLabel("")
        self.hold_label.setWordWrap(True)
        self.hold_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hold_row.addWidget(self.hold_label, 1)
        self.hold_button = _button("Hold...")
        self.hold_button.clicked.connect(lambda _=False: self.toggle_hold())
        hold_row.addWidget(self.hold_button)
        box.addLayout(hold_row)

        pause_row = QHBoxLayout()
        self.pause_labels: dict[str, QLabel] = {}
        self.pause_buttons: dict[str, QPushButton] = {}
        for scope, title in (("agent", "Agent jobs"), ("all", "All jobs")):
            label = _small("")
            label.setWordWrap(False)
            self.pause_labels[scope] = label
            button = _button("")
            button.clicked.connect(lambda _=False, s=scope: self.toggle_pause(s))
            self.pause_buttons[scope] = button
            pause_row.addWidget(label)
            pause_row.addWidget(button)
            pause_row.addSpacing(12)
        pause_row.addStretch(1)
        box.addLayout(pause_row)

        self.loop_note = _small("")
        self.loop_note.hide()
        box.addWidget(self.loop_note)

        # the jobs
        self.model = JobTableModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.table.setWordWrap(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        for key, width in (("position", 40), ("id", 44), ("state", 110), ("owner", 120),
                           ("label", 120), ("expt_class", 110), ("path", 220), ("run_id", 60),
                           ("priority", 50), ("due", 60), ("after", 100), ("est", 130),
                           ("source_changed", 70), ("waiting", 200)):
            header.resizeSection(COLUMN_KEYS.index(key), width)
        self.table.selectionModel().selectionChanged.connect(self._on_selection)
        self.table.doubleClicked.connect(lambda _i: self.show_log())
        box.addWidget(self.table, 1)

        # actions on the selected job
        actions = QHBoxLayout()
        self.cancel_button = _button("Cancel...")
        self.cancel_button.clicked.connect(lambda _=False: self.cancel_selected())
        actions.addWidget(self.cancel_button)
        self.move_buttons: dict[str, QPushButton] = {}
        for where, text in (("up", "Up"), ("down", "Down"), ("top", "Top"),
                            ("bottom", "Bottom")):
            b = _button(text)
            b.clicked.connect(lambda _=False, w=where: self.move_selected(w))
            self.move_buttons[where] = b
            actions.addWidget(b)
        self.edit_button = _button("Edit...")
        self.edit_button.clicked.connect(lambda _=False: self.edit_selected())
        actions.addWidget(self.edit_button)
        self.log_button = _button("Show log")
        self.log_button.clicked.connect(lambda _=False: self.show_log())
        actions.addWidget(self.log_button)
        self.copy_button = QToolButton()
        self.copy_button.setText("Copy kq command")
        self.copy_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.copy_button)
        for kind, text in (("submit", "kq submit ... (the same job again)"),
                           ("tail", "kq tail <id> -f (follow its log)"),
                           ("show", "kq show <id>"), ("cancel", "kq cancel <id>")):
            menu.addAction(text, lambda k=kind: self.copy_kq(k))
        self.copy_button.setMenu(menu)
        actions.addWidget(self.copy_button)
        actions.addStretch(1)
        self.ended_box = QCheckBox("ended jobs")
        self.ended_box.setChecked(True)
        self.ended_box.toggled.connect(self._on_show_ended)
        actions.addWidget(self.ended_box)
        self.refresh_button = _button("Refresh")
        self.refresh_button.clicked.connect(lambda _=False: self.refresh_list())
        actions.addWidget(self.refresh_button)
        box.addLayout(actions)

        self.message = _small("")
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.addWidget(self.message)

        self._list_timer = QTimer(self)
        self._list_timer.setSingleShot(True)
        self._list_timer.timeout.connect(self.refresh_list)

        self._show_summary()
        self.refresh_buttons()

    # -- inputs from the host -----------------------------------------------------------

    def set_state(self, state: dict | None) -> None:
        """The server's ``status_json`` (None: it did not answer)."""
        if not isinstance(state, dict):
            self.set_reachable(False)
            return
        self.set_reachable(True)
        if "run_queue" not in state or not isinstance(state.get("run_queue"), dict):
            self.has_queue = False
            self._show_summary()
            self.refresh_buttons()
            return
        self.has_queue = True
        if isinstance(state.get("person_hold"), dict):
            self.hold = dict(state["person_hold"])
        self.set_info(state["run_queue"])

    def set_reachable(self, reachable: bool) -> None:
        if bool(reachable) != self.reachable:
            self.reachable = bool(reachable)
            self._show_summary()
            self.refresh_buttons()

    def on_broadcast(self, payload: dict) -> None:
        """A server broadcast: ``run_queue`` refreshes the summary and asks for
        the list (debounced); ``person_hold`` the hold.  Others are ignored."""
        if not isinstance(payload, dict):
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
        actions = self.info.get("actions")
        if isinstance(actions, (list, tuple)):
            for action in (MOVE_ACTION, EDIT_ACTION):
                if action not in actions:
                    self.unsupported.add(action)
                else:
                    self.unsupported.discard(action)
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
        if not self._list_timer.isActive():
            self._list_timer.start(LIST_DEBOUNCE_MS)

    def refresh_list(self) -> None:
        if self._listing:
            self._list_again = True
            return
        self._listing = True
        self.runner.send(list_request(), self._on_list)

    def _on_list(self, reply: dict) -> None:
        self._listing = False
        self._listed_once = True
        if reply.get("status") == "ok":
            self.has_queue = True
            info = reply.get("run_queue")
            if isinstance(info, dict):
                self.set_info(info, ask_list=False)
                self._summary_sig = self._signature(self.info)
            self.model.set_jobs(reply.get("jobs"), self.info, reply.get("next"))
            self._reselect()
        elif is_unknown_request(reply):
            self.has_queue = False
            self.model.set_jobs([], {})
        else:
            self.say(f"list: {reply.get('msg')}", ERR_TEXT)
        self._show_summary()
        self.refresh_buttons()
        if self._list_again:
            self._list_again = False
            self.schedule_list()

    def _on_show_ended(self, show: bool) -> None:
        self.model.set_show_ended(show)
        self._reselect()
        self.refresh_buttons()

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
        counts = self.info.get("counts") or {}
        shown = [f"{counts[s]} {s}" for s in ("queued", "launching", "running", "ending",
                                               "failed", "saved") if counts.get(s)]
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
            set_style_if_changed(self.hold_label, _banner_css("#8a6a12"))
        else:
            self.hold_label.setText("No person hold: agents may use the machine when it is free.")
            set_style_if_changed(self.hold_label, f"color: {theme.FG_MUTED}; font-size: 11px;")

        paused = self.info.get("paused") or {}
        for scope, label in self.pause_labels.items():
            p = paused.get(scope)
            title = "Agent jobs" if scope == "agent" else "All jobs"
            if isinstance(p, dict):
                why = f": {p['reason']}" if p.get("reason") else ""
                label.setText(f"{title}: PAUSED by {p.get('by')} ({p.get('owner') or 'person'}) "
                              f"since {_clock(p.get('since'))}{why}")
                set_style_if_changed(label, f"color: {WARN_TEXT}; font-size: 11px;")
            else:
                label.setText(f"{title}: not paused")
                set_style_if_changed(label, f"color: {theme.FG_MUTED}; font-size: 11px;")

        resume = self.info.get("resume_loop")
        if isinstance(resume, dict) and self.reachable:
            self.loop_note.setText(
                f"The queue stopped the loop {resume.get('key')} for its jobs "
                f"(since {_clock(resume.get('since'))}); it starts it again when it has run out.")
            self.loop_note.show()
        else:
            self.loop_note.hide()

    # -- selection and buttons ------------------------------------------------------------------

    def selected_job(self) -> dict | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        job = self.model.data(rows[0], Qt.ItemDataRole.UserRole)
        return job if isinstance(job, dict) else None

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

    def refresh_buttons(self) -> None:
        live = self.reachable and self.has_queue
        job = self.selected_job()
        state = (job or {}).get("state")
        why_not = ("the monitor server is not answering" if not self.reachable else
                   "this monitor server has no run queue" if not self.has_queue else
                   "select a job" if job is None else "")

        def setup(button, ok: bool, tip_ok: str, tip_no: str) -> None:
            button.setEnabled(bool(live and job is not None and ok))
            button.setToolTip(why_not or (tip_ok if ok else tip_no))

        cancel_asked = bool((job or {}).get("cancel"))
        setup(self.cancel_button, state == "queued" or (state in ("launching", "running")
                                                         and not cancel_asked),
              "Cancel the job (asks first)",
              "its cancel has been asked for; the run ends at its next shot" if cancel_asked
              else f"it is {state}: nothing to cancel")
        queued = self.model.queued_in_order()
        for where, b in self.move_buttons.items():
            if not self.supported(MOVE_ACTION):
                setup(b, False, "", "this monitor server cannot reorder jobs (older code)")
            elif state != "queued":
                setup(b, False, "", "only a queued job can be moved")
            else:
                can = move_request(job, where, self.by, queued) is not None
                setup(b, can, _MOVE_TIPS[where],
                      "it is already first" if where in ("up", "top") else "it is already last")
        if not self.supported(EDIT_ACTION):
            setup(self.edit_button, False, "", "this monitor server cannot edit jobs (older code)")
        else:
            setup(self.edit_button, state == "queued", "Edit the queued job",
                  "only a queued job can be edited")
        setup(self.log_button, True, "Follow the job's log", "")
        setup(self.copy_button, True, "Copy a kq command line for this job", "")
        self.hold_button.setEnabled(live)
        self.hold_button.setText("Release hold" if self.hold.get("active") else "Hold...")
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
            b.setEnabled(live)
            b.setToolTip(why_not if not live else
                         (f"Let {scope} jobs launch again" if on else
                          f"Stop launching {scope} jobs (the running one goes on)"))
        self.refresh_button.setEnabled(self.reachable)

    def say(self, text: str, color: str = theme.FG_MUTED) -> None:
        self.message.setText(text)
        set_style_if_changed(self.message, f"color: {color}; font-size: 11px;")

    # -- actions ---------------------------------------------------------------------------------

    def _done(self, what: str, on_unknown: str | None = None) -> Callable[[dict], None]:
        def done(reply: dict) -> None:
            if reply.get("status") == "ok":
                self.say(f"{what}: done" + (" (the run ends at its next shot)"
                                            if reply.get("pending") else ""), OK_TEXT)
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

    def move_selected(self, where: str) -> bool:
        job = self.selected_job()
        if job is None or job.get("state") != "queued" or not self.supported(MOVE_ACTION):
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
        self._list_timer.stop()
        for window in list(self._log_windows.values()):
            window.stopped = True
            window.timer.stop()
            window.close()
        self._log_windows.clear()
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


__all__ = ["COLUMNS", "EditJobDialog", "JobLogWindow", "JobTableModel", "QueueSummaryLine",
           "RunQueuePanel", "cancel_request", "derived_waiting", "edit_request", "hold_request",
           "kq_commands", "list_request", "move_request", "order_jobs", "pause_request",
           "release_request", "resume_request", "tail_request"]
