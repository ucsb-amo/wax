"""The camera host behind the camera thread liveOD already has (camera_mother
stays unchanged).

``CameraBaby`` asks its nanny for a camera, has the run's settings applied,
then calls ``camera.start_grab(N_img, queue, check, on_armed=...)`` and
expects ``(img, t, idx)`` on the queue exactly as the drivers put them.  In
host mode its nanny is a ``HostNanny`` for the one run, and its "camera" a
``HostCameraHandle`` on that run:

* ``persistent_get_camera`` attaches to the run and waits, in 0.1 s slices and
  honouring the thread's own stop, for the arm that the server's first
  WAIT_CAM_READY started; an arm that failed gives a ``DummyCamera``, so the
  thread ends "camera not ready" as it always has;
* ``update_params`` applies nothing (the arm applied the whole run profile)
  and reports the camera's clamps for the run's overrides record;
* ``start_grab`` calls ``on_armed`` once (the acquisition is already running),
  then takes the run's frames from its ``RunSink`` and queues each under its
  hardware index.  A frame the camera reported lost raises ``FrameLostError``
  after every frame that did arrive is queued (nothing moves into its slot);
  no frame in time raises the builtin ``TimeoutError``; a camera fault or a
  frame out of place raises and ends the grab -- the run is then saved
  incomplete, as with the drivers;
* ``stop_grab`` only detaches: the camera stays locked until the run ends.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from queue import Queue

import numpy as np

from waxx.control.cameras.dummy_cam import DummyCamera
from waxx.control.cameras.errors import FrameLostError
from waxx.config.timeouts import (CAMERA_GRAB_TIMEOUT_ANDOR, CAMERA_GRAB_TIMEOUT_BASLER_INIT,
                                  CAMERA_GRAB_TIMEOUT_BASLER_RUN)

logger = logging.getLogger("waxx.live_od.camera_host")

#: (first frame, each later frame) timeouts by category, as the drivers have them
GRAB_TIMEOUTS_S = {
    "andor_emccd": (CAMERA_GRAB_TIMEOUT_ANDOR, CAMERA_GRAB_TIMEOUT_ANDOR),
    "basler_usb": (CAMERA_GRAB_TIMEOUT_BASLER_INIT, CAMERA_GRAB_TIMEOUT_BASLER_RUN),
}
DEFAULT_GRAB_TIMEOUTS_S = (CAMERA_GRAB_TIMEOUT_BASLER_INIT, CAMERA_GRAB_TIMEOUT_BASLER_RUN)


class CameraRunFault(RuntimeError):
    """The camera failed during the run (the worker is in run_fault)."""


class FrameIndexError(RuntimeError):
    """A frame's hardware index does not fit the run's slots."""


class HostCameraHandle:
    """The camera thread's camera for one run (see the module docstring)."""

    POLL_S = 0.02
    #: a quietly ended run (see _run_over) waits this long for its thread's stop
    QUIET_WAIT_S = 5.0

    def __init__(self, host, run, *, timeouts_s=None) -> None:
        self._host = host
        self._run = run
        cat = run.category.name if run.category is not None else ""
        self.first_timeout_s, self.next_timeout_s = timeouts_s or GRAB_TIMEOUTS_S.get(
            cat, DEFAULT_GRAB_TIMEOUTS_S)
        self._detached = threading.Event()
        self._over = ""                   # why the run ended under this handle
        self._quiet_until = 0.0
        self._armed_reported = False
        self.frames_queued = 0

    # -- what the camera thread asks ----------------------------------------------

    @property
    def camera_key(self) -> str:
        return self._run.key

    def is_opened(self) -> bool:
        """Armed, and the run still holds the camera."""
        return not self._over and self._host.run_armed(self._run.token)

    def clamps(self) -> dict:
        """``{field: (requested, applied)}`` the camera limited at the arm."""
        return dict(self._run.clamps or {})

    def stop_grab(self) -> None:
        """Only detaches: the camera stays locked (and armed) until the run ends."""
        self._detached.set()

    def Close(self) -> None:
        self.stop_grab()

    close = Close

    def _run_over(self, why: str, quiet: bool = False) -> None:
        """The host ended the run under this handle (or another handle took
        over).  ``quiet``: the camera thread is about to be stopped (a newer run
        took over, liveOD is closing): the grab waits up to ``QUIET_WAIT_S``
        for that stop and returns without a failure, rather than fail at once
        -- a failure it reported could otherwise land on the newer run."""
        self._quiet_until = time.monotonic() + self.QUIET_WAIT_S if quiet else 0.0
        self._over = str(why) or "the run ended"

    def start_grab(self, N_img, output_queue: Queue = None, check_interrupt_method=None,
                   on_armed=None):
        n = int(N_img)
        run = self._run
        key = run.key
        if run.n_img and n != run.n_img:
            raise ValueError(f"{key}: start_grab for {n} frame(s), but the camera was armed "
                             f"for {run.n_img}")
        if not self.is_opened():
            raise RuntimeError(f"{key}: not armed ({self._over or run.arm_error or 'no arm yet'})")
        q = output_queue if output_queue is not None else Queue()
        check = check_interrupt_method or (lambda: False)
        sink = run.sink
        w = self._host.worker(key)
        if on_armed is not None and not self._armed_reported:
            self._armed_reported = True
            on_armed()
        declared = None
        if run.images_shape is not None:
            try:
                declared = tuple(int(v) for v in run.images_shape)[1:]
            except (TypeError, ValueError):
                declared = None
        next_idx, got = 0, 0
        shape = dtype = None
        timeout = self.first_timeout_s
        deadline = time.monotonic() + timeout
        while next_idx < n:
            if check() or self._detached.is_set():
                return None
            f = sink.pop(self.POLL_S)
            if f is not None:
                idx = int(f.hw_idx) if f.hw_idx is not None else next_idx
                if idx < next_idx or idx >= n:
                    raise FrameIndexError(
                        f"{key}: a frame with hardware index {f.hw_idx} arrived when index "
                        f"{next_idx} was due (slots 0..{n - 1}); not queued -- the run's frames "
                        f"cannot be placed in their slots")
                img = f.image
                if shape is None:
                    shape, dtype = img.shape, img.dtype
                    if declared and tuple(declared) != tuple(shape):
                        logger.warning(f"camera host: {key} frames are {tuple(shape)}, the run "
                                       f"declared {tuple(declared)}")
                elif img.shape != shape or img.dtype != dtype:
                    raise FrameIndexError(
                        f"{key}: frame {idx} is {img.shape} {img.dtype}, the run's first frame "
                        f"was {shape} {dtype}; not queued")
                q.put((np.array(img, copy=True), float(f.t_host), idx))
                self.frames_queued += 1
                got += 1
                lost = list(range(next_idx, idx))
                next_idx = idx + 1
                timeout = self.next_timeout_s
                deadline = time.monotonic() + timeout
                if lost:
                    raise FrameLostError(
                        f"{key}: frame(s) {lost} of {n} were reported lost by the camera (got "
                        f"{got}); later frames kept their own index.", lost=lost)
                continue
            # nothing came in this slice
            ws = w.snapshot()
            if self._over or ws["locked_by"] != run.token:
                if time.monotonic() < self._quiet_until:
                    time.sleep(self.POLL_S)         # the window stops this thread shortly
                    continue
                why = self._over or "the run's hold on the camera ended"
                raise TimeoutError(f"{key}: {why} with {got}/{n} frame(s) in")
            if ws["state"] == "run_fault":
                raise CameraRunFault(f"{key}: camera error during the run ({ws['error']}); "
                                     f"{got}/{n} frame(s) in; the camera is not reopened until "
                                     f"the run ends")
            if ws["state"] == "run_locked" and sink.pending() == 0:
                # the camera stopped the acquisition after n frames, counting the lost ones
                missing = list(range(next_idx, n))
                raise FrameLostError(
                    f"{key}: the acquisition ended with frame(s) {missing} of {n} missing "
                    f"(reported lost by the camera; got {got})", lost=missing)
            if time.monotonic() > deadline:
                raise TimeoutError(f"No {key} image within {timeout:.0f} s (got {got}/{n}). "
                                   f"Camera not triggered?")
        return None


class HostNanny:
    """The camera thread's nanny for one run in host mode."""

    POLL_S = 0.1

    def __init__(self, host, token: str) -> None:
        self.host = host
        self.token = str(token)
        self.interrupted = False
        self.handle = None

    def break_check(self) -> bool:
        return self.interrupted

    def persistent_get_camera(self, camera_params, break_check=None):
        check = break_check if break_check is not None else self.break_check
        key = getattr(camera_params, "key", "")
        if isinstance(key, bytes):
            key = key.decode()
        try:
            handle = self.host.attach_run(self.token)
        except Exception as exc:
            logger.warning(f"camera host: {key}: no run to attach to ({exc})")
            return DummyCamera()
        self.handle = handle
        while True:
            if check():
                return DummyCamera()
            fut = self.host.arm_future(self.token)
            if fut is None:
                if not self.host.has_run(self.token):
                    logger.warning(f"camera host: {key}: the run ended before its camera was armed")
                    return DummyCamera()
                time.sleep(self.POLL_S)
                continue
            concurrent.futures.wait([fut], timeout=self.POLL_S)
            if not fut.done():
                continue
            exc = fut.exception()
            if exc is not None:
                logger.error(f"camera host: {key}: the run's camera could not be armed: {exc}")
                return DummyCamera()
            return handle

    def get_camera(self, camera_params):
        return DummyCamera()

    def update_params(self, camera, camera_params, report: dict = None):
        """Nothing to apply (the arm applied the whole run profile); the clamps."""
        if not isinstance(camera, HostCameraHandle):
            return DummyCamera()
        if report is not None:
            report["clamps"] = camera.clamps()
        return camera

    def close_all(self) -> dict:
        return {}


__all__ = ["HostNanny", "HostCameraHandle", "CameraRunFault", "FrameIndexError", "GRAB_TIMEOUTS_S"]
