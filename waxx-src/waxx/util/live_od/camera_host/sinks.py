"""RunSink: one run's frames, taken off the camera worker as they are published.

The worker calls every tap with every frame it publishes (live, snap and run
frames of every run), on its own thread.  A RunSink keeps only this run's:
``source == "run"``, this run's ``run_tag`` and, once the arm has said which
acquisition it started, that ``acq_gen``.  Anything else is counted and
dropped here, so a frame of another run, or of an earlier acquisition of this
one, never reaches the camera thread that feeds the image writer.

The frames are handed on in the order they came; the sink keeps no more than
it must (they are dropped from it once taken), and at most ``n_frames + 16``
at a time -- the worker stops a run after ``n_frames``, so that bound is never
reached by a working camera.
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Optional

#: frames kept beyond n_frames before new ones are counted instead of kept
EXTRA_BUFFER = 16


class RunSink:
    """The tap for one run (``worker.add_tap(sink)``); ``pop`` takes its frames."""

    def __init__(self, camera_id: str, run_tag: str, n_frames: int) -> None:
        self.camera_id = camera_id
        self.run_tag = str(run_tag)
        self.n_frames = int(n_frames)
        self.max_buffer = self.n_frames + EXTRA_BUFFER
        self.acq_gen: Optional[int] = None        # set once the arm returns
        self._cv = threading.Condition()
        self._q: collections.deque = collections.deque()
        self.accepted = 0          # this run's frames kept for the camera thread
        self.stale = 0             # this run's tag but another acquisition (acq_gen)
        self.foreign = 0           # run frames of another run (another run_tag)
        self.overflow = 0          # beyond max_buffer: counted, not kept
        self.t_first: Optional[float] = None
        self.t_last: Optional[float] = None
        self.closed = False

    # -- worker side ---------------------------------------------------------

    def __call__(self, frame) -> None:
        """The tap: on the worker thread, for every frame.  Quick."""
        if getattr(frame, "source", None) != "run":
            return
        if frame.run_tag != self.run_tag:
            self.foreign += 1
            return
        with self._cv:
            if self.closed:
                return
            if self.acq_gen is not None and frame.acq_gen != self.acq_gen:
                self.stale += 1
                return
            if len(self._q) >= self.max_buffer:
                self.overflow += 1
                return
            self._q.append(frame)
            self.accepted += 1
            now = time.monotonic()
            if self.t_first is None:
                self.t_first = now
            self.t_last = now
            self._cv.notify_all()

    def set_acq_gen(self, acq_gen: int) -> None:
        """The acquisition the arm started (ArmResult.acq_gen)."""
        with self._cv:
            self.acq_gen = int(acq_gen)
            # anything queued from another acquisition goes (none can be: the
            # worker publishes run frames only after the arm returned)
            keep = [f for f in self._q if f.acq_gen == self.acq_gen]
            self.stale += len(self._q) - len(keep)
            self._q = collections.deque(keep)

    # -- consumer side ---------------------------------------------------------

    def pop(self, timeout_s: float = 0.05):
        """The next frame, waiting up to ``timeout_s``; None if none came."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._cv:
            while not self._q:
                left = deadline - time.monotonic()
                if left <= 0 or self.closed:
                    return None
                self._cv.wait(left)
            return self._q.popleft()

    def pending(self) -> int:
        with self._cv:
            return len(self._q)

    def close(self) -> None:
        """No more frames are taken; a waiting ``pop`` returns."""
        with self._cv:
            self.closed = True
            self._cv.notify_all()

    def counts(self) -> dict:
        with self._cv:
            return {"accepted": self.accepted, "stale": self.stale, "foreign": self.foreign,
                    "overflow": self.overflow, "pending": len(self._q)}


__all__ = ["RunSink", "EXTRA_BUFFER"]
