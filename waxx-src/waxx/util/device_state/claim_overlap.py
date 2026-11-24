"""A cold queue launch's claim, run while its kernel compiles.

A cold run used to do everything in a row: imports, ``prepare()``, the claim
(APD stage, SLM check, wavemeter, camera connections, the liveOD start, the
device-state stamp, the announcement), THEN the kernel compile, then the
upload.  The claim waits on other machines and the compile is this process's
own CPU, so the two can run side by side:

    cold, as it was:  prepare -> claim -> compile            -> upload -> kernel
    overlapped:       prepare -> compile                     -> upload -> kernel
                                 claim (a thread) --- wait --^

Nothing about the claim itself changes: the same steps in the same order, a
failure before the liveOD start still stops the run before it has a run id,
and the kernel is not sent to the core device until the claim has finished
(``waxx.util.artiq.core.Core`` waits here, at the line where a warm-launched
run waits for the queue's GO).  A warm-launched run has always claimed after
its compile; this is the same claim, started earlier.

* :func:`enabled` -- only a cold launch by the run queue (``WAXX_LAUNCHER=kq``,
  not ``WAXX_QUEUE_WARM``), and only with ``WAXX_CLAIM_OVERLAP`` on (see
  :data:`DEFAULT_ON`; ``WAXX_CLAIM_OVERLAP=0`` always turns it off).  Direct
  ``artiq_run``, the run loop, ``run_lock`` and the monitor are never
  overlapped.
* :func:`run_is_kernel` -- the experiment's ``run`` must be a kernel: then the
  compile starts the moment ``prepare()`` has returned and no host code of the
  experiment runs between the two, so nothing can read the run id (or anything
  else the claim sets) before the claim is done.  An experiment with a host
  ``run`` claims in ``finish_prepare`` as before.
* :func:`background` / :func:`wait` -- a step started early on a thread of its
  own (kexp: the APD stage's throw, begun in ``Base.__init__``); the claim
  waits for it where the step always was, and the Core waits for any that
  nothing else waited for before the first kernel is sent.
* :func:`defer_claim` (``Expt.finish_prepare_wax``) / :func:`start_claim` /
  :func:`finish` (the Core, around the compile).
* :func:`cancel_and_wait` -- the compile raised (a compile error, Ctrl-C):
  the claim is asked to stop before the liveOD start (:func:`check_not_cancelled`,
  called by ``Expt.claim`` just before it), and this waits until its thread
  has ended, so the process leaves with the claim either not started on
  liveOD or complete -- never half-way.

The claim must not set an attribute a kernel reads (it never could: a
warm-launched run's kernel is compiled before its claim).  Import-light: the
standard library only.  Machine-agnostic.
"""

from __future__ import annotations

import os
import threading
import time

#: ``1`` / ``0``: turn the overlap on / off for this process (the submitter's
#: shell: the queue's launcher hands its environment to the run).
ENV = "WAXX_CLAIM_OVERLAP"
#: Used when :data:`ENV` is unset.  False until the overlap has run on the
#: machine (2026-11-20: built, offline tests only).
DEFAULT_ON = False
#: ``WAXX_LAUNCHER`` of a run the run queue launched (run_queue.LAUNCHER).
QUEUE_LAUNCHER = "kq"
#: Seconds between looks at a thread being waited for (so the main thread
#: keeps seeing Ctrl-C on Windows, where a bare join does not).
POLL_S = 0.05

_ON = ("1", "true", "on", "yes")
_OFF = ("0", "false", "off", "no")


class ClaimCancelled(RuntimeError):
    """The claim was told to stop (the compile it ran beside had failed)
    before it started the run on liveOD: no run id was taken."""


class Task:
    """One function on a thread of its own, with what it returned or raised."""

    def __init__(self, name, fn, daemon):
        self.name = name
        self.result = None
        self.exc = None
        self.t_start = time.monotonic()
        self.t_end = None
        self.waited = False
        self._fn = fn
        self.thread = threading.Thread(target=self._run, name=f"claim-overlap:{name}",
                                       daemon=daemon)
        self.thread.start()

    def _run(self):
        try:
            self.result = self._fn()
        except BaseException as exc:            # noqa: BLE001 -- re-raised by wait()
            self.exc = exc
        finally:
            self.t_end = time.monotonic()

    def done(self) -> bool:
        return not self.thread.is_alive()

    @property
    def seconds(self):
        return None if self.t_end is None else self.t_end - self.t_start


_lock = threading.RLock()
_cancel = threading.Event()
_state: dict = {
    "claim": None,        # the deferred claim (a callable), until it is started
    "claim_task": None,   # its Task once started
    "tasks": [],          # background() tasks, in start order
    "report": {},         # what happened, for the run's record
}


def enabled() -> bool:
    """This process overlaps its claim with its compile: a cold launch by the
    run queue with :data:`ENV` on (or unset and :data:`DEFAULT_ON`)."""
    value = os.environ.get(ENV, "").strip().lower()
    if value in _OFF:
        return False
    if os.environ.get("WAXX_LAUNCHER") != QUEUE_LAUNCHER:
        return False
    from waxx.util.device_state import queue_gate  # noqa: PLC0415
    if queue_gate.is_warm():
        return False
    if value in _ON:
        return True
    return bool(DEFAULT_ON)


def run_is_kernel(expt) -> bool:
    """``type(expt).run`` is an ARTIQ kernel (see the module docstring)."""
    run = getattr(type(expt), "run", None)
    info = getattr(run, "artiq_embedded", None)
    return info is not None and getattr(info, "core_name", None) is not None


def applies_to(expt) -> bool:
    """:func:`enabled` and :func:`run_is_kernel`: this experiment's claim (and
    its early steps) may run beside its compile."""
    return enabled() and run_is_kernel(expt)


# -- background steps ---------------------------------------------------------------------

def background(name: str, fn, daemon: bool = True) -> Task:
    """Run ``fn()`` now on a thread of its own.  :func:`wait` for it where the
    step used to be; the Core waits for any task nothing waited for."""
    task = Task(name, fn, daemon)
    with _lock:
        _state["tasks"].append(task)
    return task


def wait(task: Task | None, cancellable: bool = True):
    """Block until ``task`` is done; return what it returned, or raise what it
    raised (in this thread).  ``cancellable``: raises :class:`ClaimCancelled`
    instead once the claim was cancelled (the task's thread is left to end by
    itself).  ``None`` (the step was not started early): returns None."""
    if task is None:
        return None
    while task.thread.is_alive():
        if cancellable and _cancel.is_set():
            raise ClaimCancelled(f"stopped while waiting for {task.name}")
        task.thread.join(POLL_S)
    task.waited = True
    if task.exc is not None:
        raise task.exc
    return task.result


# -- the claim ----------------------------------------------------------------------------

def defer_claim(fn) -> None:
    """``Expt.finish_prepare_wax``: the claim to run beside the next compile."""
    with _lock:
        _state["claim"] = fn
        _state["claim_task"] = None


def claim_pending() -> bool:
    """A deferred claim has not been started yet."""
    return _state["claim"] is not None


def active() -> bool:
    """The Core has something to do here before it sends a kernel: a deferred
    claim to run, or a background task nothing has waited for."""
    with _lock:
        if _state["claim"] is not None or _state["claim_task"] is not None:
            return True
        return any(not t.waited for t in _state["tasks"])


def start_claim() -> Task | None:
    """Start the deferred claim on its thread (None when there is none)."""
    with _lock:
        fn, _state["claim"] = _state["claim"], None
        if fn is None:
            return _state["claim_task"]
        _cancel.clear()
        # not a daemon: the process never exits half-way through a claim
        task = Task("claim", fn, daemon=False)
        _state["claim_task"] = task
        _state["report"] = {"t_compile_start": time.monotonic()}
        return task


def finish() -> None:
    """After the compile: wait for the claim (raising what it raised), then
    for every background task nothing waited for.  Call before the first byte
    goes to the core device."""
    t_compile_end = time.monotonic()
    with _lock:
        task = _state["claim_task"]
        tasks = list(_state["tasks"])
    try:
        if task is not None:
            # the claim stays registered until its thread has ended: a Ctrl-C
            # during this wait still finds it (cancel_and_wait)
            try:
                wait(task, cancellable=False)
            finally:
                if task.done():
                    with _lock:
                        _state["claim_task"] = None
        for t in tasks:
            if not t.waited:
                wait(t, cancellable=False)
    finally:
        report = _state["report"]
        if task is not None and "t_compile_start" in report:
            t0 = report.pop("t_compile_start")
            report.update(compile_s=round(t_compile_end - t0, 3),
                          claim_s=None if task.seconds is None else round(task.seconds, 3),
                          waited_after_compile_s=round(time.monotonic() - t_compile_end, 3))


def cancel_and_wait() -> None:
    """The compile raised: ask the claim to stop before the liveOD start and
    wait until its thread has ended.  Never raises; what the claim raised is
    printed (the compile's error is the one that ends the run)."""
    _cancel.set()
    with _lock:
        task, _state["claim_task"] = _state["claim_task"], None
        _state["claim"] = None
    if task is None:
        return
    try:
        while task.thread.is_alive():
            task.thread.join(POLL_S)
    except BaseException:                        # noqa: BLE001 -- a second Ctrl-C: stop waiting
        return
    task.waited = True
    if task.exc is not None and not isinstance(task.exc, ClaimCancelled):
        print(f"[claim] note: the claim also failed ({type(task.exc).__name__}: {task.exc})")


def cancelled() -> bool:
    return _cancel.is_set()


def check_not_cancelled() -> None:
    """``Expt.claim``, just before the liveOD start: raises
    :class:`ClaimCancelled` when the compile beside it has failed."""
    if _cancel.is_set():
        raise ClaimCancelled("the kernel compile failed: the run is not started on liveOD")


def run_claim_now() -> bool:
    """A deferred claim that was never started (no kernel ran): run it here,
    in this thread.  True when one ran."""
    with _lock:
        fn, _state["claim"] = _state["claim"], None
    if fn is None:
        return False
    fn()
    return True


def report() -> dict:
    """``{"compile_s", "claim_s", "waited_after_compile_s"}`` of the overlapped
    claim (``{}`` when there was none)."""
    return dict(_state["report"])


def _reset_for_tests() -> None:
    with _lock:
        _state.update(claim=None, claim_task=None, tasks=[], report={})
    _cancel.clear()
