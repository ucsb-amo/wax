"""liveOD's "Run again next": the last run that came through the run queue,
queued again as the next job.

The run queue lives in the monitor server (:mod:`waxx.util.device_state.
run_queue`).  A run it launched tells liveOD its job at INIT_RUN
(``queue_job`` = "<id>:<token>", ``launcher`` "kq"); a run started any other
way (``ar_direct``, a run loop, a bare ``artiq_run``) sends ``queue_job`` "".
The windows hand each run to :meth:`RunAgainButton.note_run`; the button keeps
the most recent run that had a queue job, whether it saved, failed or was
aborted.

One press (after a confirmation naming the job and its file) sends the run
queue's ``insert`` at ``at_index`` 0 through
:class:`~waxx.util.device_state.run_queue_client.RunQueueClient` -- the
request the queue panel's "Run again next" sends
(:func:`~waxx.util.guis.run_queue_panel.again_request` with ``first=True``):
the same file as a new person's job, ahead of every queued job (behind the
one running), with the old job's argv, working folder, label, write-back veto
and drift setting; not its dependencies, chain, due time or repeats.  The
file that runs is the one on disk NOW: this PC's hash of it goes with the
request and the queue's own SHA-256 rules apply (it refuses a file whose copy
on the queue machine differs from this PC's, or that this PC cannot read
unless this PC is the queue machine).

The button is enabled only when a last queued job is known, the run queue
answers, and it has described that job (its file); otherwise it is disabled
and its tooltip says why.  While a request is out it stays disabled (one press
= one submission).  Every network call and the file hash run on a worker
thread (:class:`~waxx.util.guis.request_runner.RequestRunner`); the result
(the new job id, or the queue's refusal) goes to the window's log and
:attr:`RunAgainButton.result_ready`.

Nothing is discovered or bound until :meth:`RunAgainButton.start` (the
windows' ``main()`` calls it; tests that build a window never do).

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QPushButton

from waxx.util.device_state.run_queue_client import (
    NoRunQueue, RunQueueError, default_by, local_sha256)
from waxx.util.guis.run_queue_panel import not_carried_over

#: the button's text
BUTTON_TEXT = "Run again next"
#: who a re-submitted job is for: a person pressed the button
OWNER = "person"


def parse_queue_job(text) -> tuple[int, str] | None:
    """``"<id>:<token>"`` (or ``"<id>"``) -> ``(id, token)``; None for "" or
    anything else (a run that did not come through the queue)."""
    text = str(text or "").strip()
    if not text:
        return None
    head, _, token = text.partition(":")
    try:
        job_id = int(head)
    except ValueError:
        return None
    return (job_id, token.strip()) if job_id > 0 else None


def again_kwargs(job: dict) -> dict:
    """:meth:`RunQueueClient.submit` keyword arguments that queue ``job``'s
    file again as the next job -- the fields the queue panel's
    :func:`~waxx.util.guis.run_queue_panel.again_request` carries (``first``):
    argv, working folder, label, write-back veto, drift; at index 0; owner
    person."""
    kw: dict[str, Any] = {
        "argv": [str(a) for a in (job.get("argv") or [])],
        "cwd": job.get("cwd") or None,
        "label": job.get("label") or None,
        "owner": OWNER,
        "allow_drift": bool(job.get("allow_drift")),
        "at_index": 0,
    }
    if job.get("write_back") is False:
        kw["write_back"] = False
    return kw


class LinkTransport:
    """A :class:`RunQueueClient` transport over a monitor link (``request(obj)``
    -> reply dict or None; ``status()`` -> ``status_json`` or None: a
    :class:`~waxx.util.guis.monitor_panel.MonitorLink`, which rediscovers the
    server after a request it did not answer)."""

    def __init__(self, link):
        self.link = link

    def request(self, obj: dict, timeout: float | None = None, attempts: int = 1):
        return self.link.request(obj)

    def get_status(self):
        return self.link.status()


def make_client(by: str):
    """The default client: a RunQueueClient over a MonitorLink (nothing is
    discovered until the first request, on the worker thread)."""
    from waxx.util.device_state.run_queue_client import RunQueueClient  # noqa: PLC0415
    from waxx.util.guis.monitor_panel import MonitorLink  # noqa: PLC0415
    return RunQueueClient(LinkTransport(MonitorLink(discovery_timeout=2.0)), by=by,
                          owner=OWNER)


def _base(path) -> str:
    return os.path.basename(str(path or "")) or "?"


class RunAgainButton(QPushButton):
    """See the module docstring.

    ``client``: a RunQueueClient-like object (``status()``, ``describe(id,
    token)``, ``submit(path, **kw)``; default :func:`make_client`, made at
    :meth:`start`); ``synchronous``: requests on the GUI thread (tests);
    ``log(text, level)``: the window's log; ``confirm(title, text) -> bool``:
    asked before sending (default a Yes / No box); ``by``: who is asking
    (default ``user@host (liveOD)``)."""

    #: How often the run queue is asked whether it answers.
    POLL_MS = 5000

    #: (text, ok) after each press: what the queue answered
    result_ready = pyqtSignal(str, bool)

    def __init__(self, parent=None, *, client=None, synchronous: bool = False,
                 log: Callable[[str, int], Any] | None = None,
                 confirm: Callable[[str, str], bool] | None = None, by: str | None = None):
        super().__init__(BUTTON_TEXT, parent)
        self.client = client
        self._synchronous = bool(synchronous)
        self._log = log
        self._confirm = confirm
        self.by = by or f"{default_by()} (liveOD)"
        self.runner = None
        self._closed = False
        self._polling = False
        self.busy = False
        #: the latest run liveOD saw: {"kind": "queued" | "direct" | "unknown",
        #: "run_id", "expt", "launcher"} or None (no run yet)
        self.latest: dict | None = None
        #: the last run with a queue job: (id, token) and its run
        self.job_ref: tuple[int, str] | None = None
        self.job_run: dict = {}
        #: that job's record from the queue (``describe``), or why there is none
        self.record: dict | None = None
        self.record_error = ""
        #: this PC's SHA-256 of the record's file (None: unreadable), and
        #: whether it has been read
        self.local_sha: str | None = None
        self.sha_checked = False
        #: True / False once the queue has (not) answered; None before
        self.reachable: bool | None = None
        self.unreachable_why = ""
        #: the last press's outcome, for the tooltip
        self.last_result = ""
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedWidth(104)        # constant: a run starting never changes the width
        self.clicked.connect(lambda _=False: self.resubmit())
        self.timer = QTimer(self)
        self.timer.setInterval(self.POLL_MS)
        self.timer.timeout.connect(self.poll)
        self._refresh()

    # -- the link ------------------------------------------------------------------------

    def start(self) -> None:
        """Start asking the run queue (imports the comms package only here:
        importing it starts the discovery listener)."""
        if self._closed:
            return
        from waxx.util.guis.request_runner import RequestRunner  # noqa: PLC0415
        if self.client is None:
            self.client = make_client(self.by)
        if self.runner is None:
            self.runner = RequestRunner(None, synchronous=self._synchronous, parent=self)
        self.timer.start()
        self.poll()

    def poll(self) -> None:
        """Ask (worker thread) whether the queue answers; describe the last
        queued job when its record is not known yet; hash its file."""
        if self._polling or self._closed or self.runner is None:
            return
        self._polling = True
        ref = self.job_ref
        want = ref is not None and self.record is None
        path = (self.record or {}).get("path")
        client = self.client
        self.runner.call(lambda: self._fetch(client, ref, want, path),
                         lambda result, ref=ref: self._on_fetched(ref, result))

    @staticmethod
    def _fetch(client, ref, want_record: bool, path) -> dict:
        """On the worker: ``{"reachable", "why", "record"?, "record_error"?,
        "sha"?}``."""
        out: dict = {"reachable": False, "why": ""}
        try:
            client.status()
            out["reachable"] = True
        except NoRunQueue as exc:
            out["why"] = str(exc)
        except RunQueueError as exc:
            out["why"] = exc.msg
        except Exception as exc:                      # noqa: BLE001
            out["why"] = f"the request failed: {exc}"
        if out["reachable"] and want_record and ref is not None:
            try:
                out["record"] = dict(client.describe(ref[0], ref[1] or None).get("job") or {})
                path = out["record"].get("path")
            except (NoRunQueue, RunQueueError) as exc:
                out["record_error"] = getattr(exc, "msg", None) or str(exc)
            except Exception as exc:                  # noqa: BLE001
                out["record_error"] = f"the request failed: {exc}"
        if path:
            out["sha"] = local_sha256(str(path))
        return out

    def _on_fetched(self, ref, result) -> None:
        self._polling = False
        if self._closed:
            return
        if not isinstance(result, dict):
            result = {"reachable": False, "why": "the request failed"}
        self.reachable = bool(result.get("reachable"))
        self.unreachable_why = str(result.get("why") or "")
        if ref is not None and ref == self.job_ref:   # not a job replaced meanwhile
            if "record" in result:
                record = result["record"]
                if record.get("path"):
                    self.record, self.record_error = record, ""
                else:
                    self.record_error = "the queue's record of it has no file"
            elif "record_error" in result:
                self.record_error = str(result["record_error"])
            if "sha" in result and self.record is not None:
                self.local_sha, self.sha_checked = result["sha"], True
        self._refresh()
        if (self.job_ref is not None and self.job_ref != ref and self.record is None
                and not self.record_error):
            self.poll()                               # a job noted while this was out

    # -- what liveOD saw -----------------------------------------------------------------

    def note_run(self, run_id, expt_name: str = "", queue_job: str | None = None,
                 launcher: str | None = None) -> None:
        """A run started (GUI thread).  ``queue_job``: "<id>:<token>" for a run
        the queue launched, "" for one it did not, None when liveOD did not
        say (an older liveOD server)."""
        try:
            run_id = int(run_id or 0)
        except (TypeError, ValueError):
            run_id = 0
        run = {"run_id": run_id, "expt": str(expt_name or ""), "launcher": str(launcher or "")}
        ref = parse_queue_job(queue_job) if queue_job is not None else None
        if queue_job is None:
            self.latest = dict(run, kind="unknown")
        elif ref is None:
            self.latest = dict(run, kind="direct")
        else:
            self.latest = dict(run, kind="queued")
            if ref != self.job_ref:
                self.job_ref = ref
                self.record, self.record_error = None, ""
                self.local_sha, self.sha_checked = None, False
                self.last_result = ""
            self.job_run = run
            if self.runner is not None:
                self.poll()
        self._refresh()

    # -- the state -----------------------------------------------------------------------

    def file_changed(self) -> bool:
        """This PC's copy of the file differs from the one the job was
        submitted with."""
        rec = self.record or {}
        return bool(self.sha_checked and self.local_sha and rec.get("sha256")
                    and self.local_sha != rec.get("sha256"))

    def file_unreadable(self) -> bool:
        return bool(self.sha_checked and self.local_sha is None)

    def _job_words(self) -> str:
        rec = self.record or {}
        job_id = self.job_ref[0] if self.job_ref else "?"
        label = rec.get("label") or self.job_run.get("expt") or "?"
        return f"job {job_id} ({label})"

    def _latest_direct_note(self) -> str:
        """"" or a note that the latest run was not the queued one offered."""
        latest = self.latest or {}
        if latest.get("kind") not in ("direct", "unknown") or self.job_ref is None:
            return ""
        what = _run_words(latest)
        if latest["kind"] == "direct":
            how = f" (launched by {latest['launcher']})" if latest.get("launcher") else ""
            return (f"The latest run ({what}) was a direct run, not through the queue{how}; "
                    "this re-submits the last queued one.")
        return (f"liveOD did not say whether the latest run ({what}) came through the queue "
                "(an older liveOD server); this re-submits the last queued run it knows of.")

    def state(self) -> tuple[bool, str]:
        """``(enabled, tooltip)``."""
        head = ("Re-submit the last run that came through the run queue, as the NEXT job "
                "(ahead of every queued job, behind the one running), as a person's job.")
        if self._closed:
            return False, head + "\n\nliveOD is closing."
        if self.job_ref is None:
            latest = self.latest or {}
            if latest.get("kind") == "direct":
                how = f", launched by {latest['launcher']}" if latest.get("launcher") else ""
                why = (f"The last run ({_run_words(latest)}) was a direct run, not through "
                       f"the run queue{how}: there is no queue job to re-submit. Runs "
                       "submitted with ar / kq can be run again from here.")
            elif latest.get("kind") == "unknown":
                why = (f"liveOD did not say whether the last run ({_run_words(latest)}) came "
                       "through the run queue (an older liveOD server): nothing to re-submit.")
            else:
                why = ("No run has come through the run queue since this liveOD started: "
                       "nothing to re-submit yet.")
            return False, head + "\n\n" + why
        if self.runner is None:
            return False, head + f"\n\nNot connected to the run queue ({self._job_words()})."
        if self.reachable is None:
            return False, head + "\n\nAsking the run queue (monitor server)..."
        if not self.reachable:
            return False, (head + "\n\nThe run queue (monitor server) is not answering"
                           + (f": {self.unreachable_why}" if self.unreachable_why else "") + ".")
        if self.record is None:
            if self.record_error:
                return False, (head + f"\n\nThe run queue cannot describe job "
                               f"{self.job_ref[0]}: {self.record_error}")
            return False, head + f"\n\nLooking up job {self.job_ref[0]} in the run queue..."
        if self.busy:
            return False, head + f"\n\nSending {self._job_words()} to the queue..."
        return True, head + "\n\n" + self.details()

    def details(self) -> str:
        """What a press re-submits, in words (tooltip and confirmation)."""
        rec = self.record or {}
        lines = [f"Queue job: {self._job_words()}"
                 + (f", {rec['state']}" if rec.get("state") else "")
                 + (f" -- run {self.job_run['run_id']}" if self.job_run.get("run_id") else "")]
        lines.append(f"File: {rec.get('path')}")
        if rec.get("expt_class"):
            lines.append(f"Experiment class: {rec['expt_class']}")
        if rec.get("argv"):
            lines.append("Arguments: " + " ".join(map(str, rec["argv"])))
        notes = ["The file runs as it is on disk NOW (the queue checks its SHA-256 as for "
                 "any submission)."]
        if self.file_changed():
            notes.append(f"It has CHANGED since job {self.job_ref[0]} was submitted: the "
                         "current version runs, not the one that ran.")
        elif self.file_unreadable():
            notes.append("This PC cannot read the file: the queue refuses it unless this PC "
                         "is the queue machine.")
        dropped = not_carried_over(rec)
        if dropped:
            notes.append("Not carried over: " + "; ".join(dropped) + ".")
        direct = self._latest_direct_note()
        if direct:
            notes.append(direct)
        if self.last_result:
            notes.append("Last press: " + self.last_result)
        return "\n".join(lines) + "\n\n" + "\n".join(notes)

    def _refresh(self) -> None:
        enabled, tip = self.state()
        self.setEnabled(enabled)
        self.setToolTip(tip)

    # -- pressing it -----------------------------------------------------------------------

    def resubmit(self) -> bool:
        """Confirm, then send the ``insert`` (worker thread).  False when
        nothing was sent (disabled, a request already out, or declined)."""
        enabled, _tip = self.state()
        if not enabled or self.busy:
            return False
        self.busy = True                              # from here, a second press does nothing
        self._refresh()
        job = dict(self.record or {})
        ref = self.job_ref
        title = f"Run {_base(job.get('path'))} again next?"
        text = (f"Queue {self._job_words()} again as the NEXT job (ahead of every queued "
                f"job, behind the one running), as a person's job.\n\n{self.details()}")
        if not self._ask(title, text):
            self.busy = False
            self._refresh()
            return False
        self._say(f"Run again next: re-submitting {self._job_words()}, "
                  f"{_base(job.get('path'))} ...", 20)
        client = self.client
        self.runner.call(lambda: self._submit(client, job),
                         lambda result, job=job, ref=ref: self._on_submitted(job, ref, result))
        return True

    @staticmethod
    def _submit(client, job: dict) -> dict:
        """On the worker: ``{"ok", "reply" | "msg", "sha"}``."""
        path = str(job.get("path"))
        sha = local_sha256(path)
        try:
            reply = client.submit(path, **again_kwargs(job))
            return {"ok": True, "reply": dict(reply or {}), "sha": sha}
        except (NoRunQueue, RunQueueError) as exc:
            return {"ok": False, "msg": getattr(exc, "msg", None) or str(exc), "sha": sha}
        except Exception as exc:                      # noqa: BLE001
            return {"ok": False, "msg": f"the request failed: {exc}", "sha": sha}

    def _on_submitted(self, job: dict, ref, result) -> None:
        self.busy = False
        if self._closed:
            return
        if not isinstance(result, dict):
            result = {"ok": False, "msg": "the request failed"}
        old = f"job {ref[0] if ref else '?'} ({job.get('label') or '?'})"
        stamp = time.strftime("%H:%M:%S")
        if result.get("ok"):
            reply = result.get("reply") or {}
            ids = [str(i) for i in (reply.get("ids") or [])]
            new = ("job " + ", ".join(ids)) if ids else "a new job"
            text = f"Run again next: {old} re-submitted as {new}, next in the queue"
            if reply.get("clamped"):
                text += " (placed behind person jobs)"
            notes = []
            sha = result.get("sha")
            if sha and job.get("sha256") and sha != job.get("sha256"):
                notes.append("its file has CHANGED since it was submitted (the current version "
                             "runs)")
            dropped = not_carried_over(job)
            if dropped:
                notes.append("not carried over: " + "; ".join(dropped))
            if notes:
                text += " -- " + "; ".join(notes)
            self.last_result = f"{stamp}: queued as {new}"
            self._say(text, 30 if notes else 20)
            self.result_ready.emit(text, True)
        else:
            text = f"Run again next: the run queue refused {old}: {result.get('msg')}"
            self.last_result = f"{stamp}: refused ({result.get('msg')})"
            self._say(text, 40)
            self.result_ready.emit(text, False)
        self._refresh()

    def _ask(self, title: str, text: str) -> bool:
        if self._confirm is not None:
            return bool(self._confirm(title, text))
        box = QMessageBox(QMessageBox.Icon.Question, title, text, parent=self.window())
        go = box.addButton("Queue it next", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(go)
        box.exec()
        return box.clickedButton() is go

    # -- the rest ---------------------------------------------------------------------------

    def _say(self, text: str, level: int = 20) -> None:
        if self._log is not None:
            try:
                self._log(text, level)
                return
            except Exception:                         # noqa: BLE001
                pass
        print(f"[liveOD] {text}")

    def shutdown(self) -> None:
        """Stop asking (safe to call twice)."""
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        if self.runner is not None:
            try:
                self.runner.shutdown()
            except Exception:                         # noqa: BLE001
                pass
        self._refresh()


def _run_words(run: dict) -> str:
    rid = run.get("run_id")
    parts = [f"run {rid}" if rid else "an unsaved run"]
    if run.get("expt"):
        parts.append(str(run["expt"]))
    return ", ".join(parts)


__all__ = ["BUTTON_TEXT", "LinkTransport", "RunAgainButton", "again_kwargs", "make_client",
           "parse_queue_job"]
