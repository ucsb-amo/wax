"""The run's data file as the liveOD server sees it.

Moved out of live_od_server.py (2026-09-20) so the server holds no file handling
of its own. The behaviour is the server's, unchanged; the two copies of the
delete (reset at END_RUN, reset finalized later) are now the one ``discard()``.

2026-10-02: owns the run's writer (image_writer.ImageWriter), the one open
handle on the file, started right after the file is made and finished here at
END_RUN, abort or exit; takes the experiment's pushed arrays (``put``); and
can run the END_RUN save on a thread of its own (``save_async`` / ``status``)
so the server's message loop stays free while a long save runs.
"""
import os
import threading
import time

from waxx.config.timeouts import DATA_SAVER_TIMEOUT
from waxa.data.data_saver import clear_end_run_payload, stash_end_run_payload
from waxx.util.live_od.data.image_writer import (
    ImageWriter, register_writer, take_write_report)
from waxx.util.live_od.log import get_logger

logger = get_logger("server")


class RunFileSaveError(Exception):
    """The END_RUN save failed. ``str(err)`` is the full message for the client
    and the log (with the way out, when the payload was stashed); ``cause`` is
    the saver's own error text."""

    def __init__(self, message: str, cause: str):
        super().__init__(message)
        self.cause = cause


class RunFile:
    """One run's data file: reserved, then either saved or discarded.

    Not thread-safe by itself, and does not need to be: every method is called
    from the server's message loop, except ``writer_finished`` (an Event.set),
    ``put`` (a queue put) and the save thread's own bookkeeping.
    """

    def __init__(self, data_saver):
        self._data_saver = data_saver
        self.filepath = ""
        self.save_data = False
        self.writer = None
        self.writer_done = threading.Event()
        self.writer_done.set()      # default: no image writer in flight
        self._save_thread = None
        self._save_status = {"state": "idle"}
        self._status_lock = threading.Lock()

    # ---- the run's writer ------------------------------------------------

    def begin(self, save_data: bool, has_writer: bool):
        """A new run. Gates the END_RUN save on the writer finishing; until
        ``start_writer`` runs (after the file is reserved), a run without a
        camera has no writer, so it is done from the start. A previous run's
        writer still open (its END_RUN never came) is finished first."""
        self.finish_writer()
        self.save_data = bool(save_data)
        self.filepath = ""
        self.writer = None
        self.writer_done = threading.Event()
        if not has_writer:
            self.writer_done.set()

    def start_writer(self, n_img: int = 0):
        """Open the run's one handle on its file, for the whole run: the
        camera's frames (DataHandler attaches to it) and the experiment's
        pushed arrays go through it. Nothing to do without a file."""
        if not (self.save_data and self.filepath) or self.writer is not None:
            return
        writer = ImageWriter(self.filepath, server_owned=True)
        register_writer(self.filepath, writer)
        done = self.writer_done
        done.clear()
        writer.start_for_server(int(n_img), on_done=done.set)
        self.writer = writer

    def finish_writer(self, interrupted: bool = False):
        """No more data for the file: the writer closes it (after what it has
        queued, or at once without writing when ``interrupted``)."""
        if self.writer is not None:
            self.writer.finish(interrupted)

    @property
    def images_done(self) -> bool:
        """The camera's frames are all in (true at once for a run that has no
        writer at all)."""
        if self.writer is not None:
            return self.writer.images_finished.is_set() or self.writer_done.is_set()
        return self.writer_done.is_set()

    def put(self, items) -> bool:
        """The experiment pushed arrays for the file; False when the run has no
        writer to take them (the caller then keeps them for END_RUN)."""
        if self.writer is None or not self.pending:
            return False
        for item in items:
            if not self.writer.put_data(item):
                return False
        return True

    def reserve(self, msg: dict):
        """Create and populate the data file; returns ``(run_id, filepath)``.

        Atomically reserves a unique run_id by exclusively creating the data
        file ('x' mode).  Exclusive create is atomic on the shared filesystem,
        so two liveOD servers driving different hardware but writing to one data
        drive can never collide on a run_id.

        The file is fully populated inside that same exclusive open, so by the
        time the server replies the file is ready for SaveWorker.  Doing this
        synchronously is deliberate: the previous background-thread design could
        fail silently and leave a file with no 'data' group, which cost an
        entire run's images before anything noticed.  Image pre-allocation is
        deferred to SaveWorker, so this is a small write.
        """
        try:
            run_id, filepath = self._data_saver.reserve_run_id_and_path(msg)
        except Exception:
            # No image writer will be spawned, so release the gate begin()
            # cleared — otherwise the next END_RUN / reset blocks on it.
            self.writer_done.set()
            self.filepath = ""
            raise
        self.filepath = filepath
        return run_id, filepath

    def writer_finished(self):
        """The image writer has closed its handle on the file. Safe from any thread."""
        self.writer_done.set()

    @property
    def pending(self) -> bool:
        """There is a file that END_RUN still has to save into."""
        return bool(self.save_data and self.filepath)

    # ---- the END_RUN save -----------------------------------------------

    def save(self, msg: dict, run_id: int, shot_timestamps: list,
             images_expected: int = 0, images_received: int = 0,
             grab_failure: str = ""):
        """The END_RUN save. Raises RunFileSaveError if it fails.

        Returns None when the run's data is all there, else a dict describing
        what is missing (``reason``, ``images_expected``, ``images_received``),
        which the saver has written into the file: ``run_complete`` stays False
        and ``data_complete=False`` / ``incomplete_reason`` say why. A run is
        incomplete when the image writer never finished, when fewer frames
        arrived than the run asked for, when the frames are not the shape or
        dtype the run declared (all written all the same), when a frame that
        arrived did not reach the file (a write error, or a shape or dtype
        change mid-run: the image writer's WriteReport), when an array the
        experiment pushed during the run did not reach the file (PutReport),
        or when the camera grab reported a failure (``grab_failure``, the
        CameraBaby's reason).
        """
        filepath = self.filepath
        writer = self.writer
        writer_done = self.writer_done
        # Everything is in: the writer closes the file. Wait for that before
        # we open the same file for the end-of-run save.  Without this wait the
        # two h5py opens race and either corrupt the file or raise OSError.
        self._set_phase("writer")
        self.finish_writer()
        reasons = []
        if not writer_done.wait(timeout=DATA_SAVER_TIMEOUT):
            logger.warning(f"DataHandler did not finish within {DATA_SAVER_TIMEOUT:.0f} s — proceeding anyway.")
            reasons.append(f"image writer did not finish within {DATA_SAVER_TIMEOUT:.0f} s")
        if images_expected and images_received < images_expected:
            reasons.append(f"{images_received}/{images_expected} images received")
        # what the image writer did (no report: no writer was started for this
        # run -- no camera, or save_data=False)
        report = take_write_report(filepath)
        if report is not None:
            written, not_written, mismatches = report.snapshot()
            reasons += mismatches
            if not_written or written < images_received:
                what = (f"{written}/{images_expected or images_received} images written "
                        f"to the file")
                if not_written:
                    what += f" ({len(not_written)} not written; first: {not_written[0]})"
                reasons.append(what)
        pushed_written = {}
        if writer is not None:
            pushed_written, pushed_lost, _ = writer.put_report.snapshot()
            if pushed_lost:
                reasons.append(f"{len(pushed_lost)} pushed array(s) not written to the file "
                               f"(first: {pushed_lost[0]})")
        if grab_failure:
            reasons.append(str(grab_failure))
        incomplete = None
        if reasons:
            incomplete = {
                "reason": "; ".join(reasons),
                "images_expected": int(images_expected),
                "images_received": int(images_received),
            }
        # Stash the payload on local disk BEFORE touching the data file.
        # The experiment sends its final params exactly once and then
        # drops them, so without this a failed save loses them for good.
        self._set_phase("stash")
        stash_path = stash_end_run_payload(
            msg, filepath, run_id,
            shot_timestamps=shot_timestamps,
            incomplete=incomplete,
        )
        self._set_phase("write")
        try:
            self._data_saver.save_data_from_payload(
                msg, filepath,
                shot_timestamps=shot_timestamps,
                incomplete=incomplete,
            )
            clear_end_run_payload(stash_path)
            # Forget the file so a late RESET cannot delete an already-saved run.
            if self.filepath == filepath:
                self.filepath = ""
            if pushed_written:
                logger.info("END_RUN: pushed arrays written: "
                            + ", ".join(f"{k} x{n}" for k, n in sorted(pushed_written.items())))
            return incomplete
        except Exception as exc:
            logger.debug("END_RUN: save failed", exc_info=True)
            error = str(exc)
            if stash_path:
                hint = (
                    f"Final params for run {run_id} are preserved at "
                    f"{stash_path}. Once the data drive is back, finish the save with:\n"
                    f"    from waxa.data.data_saver import retry_pending_save\n"
                    f"    retry_pending_save(r'{stash_path}')"
                )
                error = f"{error}\n{hint}"
            raise RunFileSaveError(error, cause=str(exc)) from exc

    def save_async(self, msg: dict, run_id: int, shot_timestamps: list, on_done,
                   **kw):
        """``save`` on a thread of its own. ``on_done(result)`` runs on that
        thread when it is over, with ``result`` = ``{"state": "saved" |
        "saved_incomplete" | "failed", "incomplete": ..., "error": ...,
        "cause": ...}``; ``status()`` says how far it is meanwhile."""
        with self._status_lock:
            self._save_status = {"state": "saving", "phase": "start", "run_id": int(run_id),
                                 "t_start": time.time(), "t_phase": time.time()}

        def work():
            try:
                incomplete = self.save(msg, run_id, shot_timestamps, **kw)
                result = {"state": "saved_incomplete" if incomplete else "saved",
                          "incomplete": incomplete, "error": "", "cause": ""}
            except RunFileSaveError as exc:
                result = {"state": "failed", "incomplete": None, "error": str(exc),
                          "cause": exc.cause}
            except Exception as exc:        # not expected: say so rather than hang the client
                logger.exception("END_RUN: the save thread failed")
                result = {"state": "failed", "incomplete": None,
                          "error": f"{type(exc).__name__}: {exc}", "cause": str(exc)}
            with self._status_lock:
                self._save_status = {**self._save_status, **result,
                                     "t_end": time.time(), "run_id": int(run_id)}
            try:
                on_done(result)
            except Exception:
                logger.exception("END_RUN: the save's completion handler failed")

        t = threading.Thread(target=work, name=f"run-save-{run_id}", daemon=True)
        self._save_thread = t
        t.start()

    def _set_phase(self, phase: str):
        with self._status_lock:
            if self._save_status.get("state") == "saving":
                self._save_status["phase"] = str(phase)
                self._save_status["t_phase"] = time.time()

    @property
    def saving(self) -> bool:
        t = self._save_thread
        return t is not None and t.is_alive()

    def wait_save(self, timeout: float) -> bool:
        """True once no save thread is running (at once when there is none)."""
        t = self._save_thread
        if t is None:
            return True
        t.join(timeout)
        return not t.is_alive()

    def status(self) -> dict:
        """The asynchronous save's state, for SAVE_STATUS."""
        with self._status_lock:
            st = dict(self._save_status)
        now = time.time()
        if "t_start" in st:
            st["elapsed_s"] = round(now - st["t_start"], 3)
        if "t_phase" in st:
            st["phase_elapsed_s"] = round(now - st["t_phase"], 3)
        return st

    # ---- a run that is not saved ----------------------------------------

    def discard(self, wait_for_writer: bool = True):
        """Delete the file of a run that was reset or aborted. The only place
        the server's side deletes anything; a saved run's path has already been
        forgotten by ``save()``, so it cannot end up here.

        The file may already be gone, for camera runs whose CameraBaby called
        dishonorable_death.
        """
        # the writer lets go of the file without writing what it still holds
        self.finish_writer(interrupted=True)
        take_write_report(self.filepath)        # the run is gone; so is its report
        if not (self.filepath and os.path.exists(self.filepath)):
            return
        if wait_for_writer:
            # Wait for the image writer to close its HDF5 handle before
            # attempting deletion.  Without this the file is still open and
            # os.remove raises WinError 32 on Windows.
            if not self.writer_done.wait(timeout=DATA_SAVER_TIMEOUT):
                logger.warning(f"DataHandler did not release the file within {DATA_SAVER_TIMEOUT:.0f} s — deletion may fail.")
        try:
            os.remove(self.filepath)
            logger.info(f"Deleted incomplete data file: {self.filepath}")
        except Exception as exc:
            logger.warning(f"Could not delete incomplete data file: {exc}")
        self.filepath = ""

    def leave(self):
        """The run's experiment is gone without END_RUN: the writer closes the
        file with what it has, and the path is forgotten (not saved, not
        deleted; a reset pressed later must not delete it)."""
        self.finish_writer()
        self.filepath = ""
