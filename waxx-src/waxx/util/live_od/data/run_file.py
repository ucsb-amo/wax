"""The run's data file as the liveOD server sees it.

Moved out of live_od_server.py (2026-09-20) so the server holds no file handling
of its own. The behaviour is the server's, unchanged; the two copies of the
delete (reset at END_RUN, reset finalized later) are now the one ``discard()``.
"""
import os
import threading

from waxx.config.timeouts import DATA_SAVER_TIMEOUT
from waxa.data.data_saver import clear_end_run_payload, stash_end_run_payload
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
    from the server's message loop, except ``writer_finished`` (an Event.set).
    """

    def __init__(self, data_saver):
        self._data_saver = data_saver
        self.filepath = ""
        self.save_data = False
        self.writer_done = threading.Event()
        self.writer_done.set()      # default: no image writer in flight

    def begin(self, save_data: bool, has_writer: bool):
        """A new run. Gates the END_RUN save on the image writer finishing; a
        run without a camera has no writer, so it is done from the start."""
        self.save_data = bool(save_data)
        self.filepath = ""
        self.writer_done.clear()
        if not has_writer:
            self.writer_done.set()

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

    def save(self, msg: dict, run_id: int, shot_timestamps: list,
             images_expected: int = 0, images_received: int = 0,
             grab_failure: str = ""):
        """The END_RUN save. Raises RunFileSaveError if it fails.

        Returns None when the run's data is all there, else a dict describing
        what is missing (``reason``, ``images_expected``, ``images_received``),
        which the saver has written into the file: ``run_complete`` stays False
        and ``data_complete=False`` / ``incomplete_reason`` say why. A run is
        incomplete when the image writer never finished, when fewer frames
        arrived than the run asked for, or when the camera grab reported a
        failure (``grab_failure``, the CameraBaby's reason).
        """
        # Wait for the image writer to close its HDF5 handle before we open
        # the same file for the end-of-run save.  Without this wait the
        # two h5py opens race and either corrupt the file or raise OSError.
        reasons = []
        if not self.writer_done.wait(timeout=DATA_SAVER_TIMEOUT):
            logger.warning(f"DataHandler did not finish within {DATA_SAVER_TIMEOUT:.0f} s — proceeding anyway.")
            reasons.append(f"image writer did not finish within {DATA_SAVER_TIMEOUT:.0f} s")
        if images_expected and images_received < images_expected:
            reasons.append(f"{images_received}/{images_expected} images received")
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
        stash_path = stash_end_run_payload(
            msg, self.filepath, run_id,
            shot_timestamps=shot_timestamps,
            incomplete=incomplete,
        )
        try:
            self._data_saver.save_data_from_payload(
                msg, self.filepath,
                shot_timestamps=shot_timestamps,
                incomplete=incomplete,
            )
            clear_end_run_payload(stash_path)
            # Forget the file so a late RESET cannot delete an already-saved run.
            self.filepath = ""
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

    def discard(self, wait_for_writer: bool = True):
        """Delete the file of a run that was reset or aborted. The only place
        the server's side deletes anything; a saved run's path has already been
        forgotten by ``save()``, so it cannot end up here.

        The file may already be gone, for camera runs whose CameraBaby called
        dishonorable_death.
        """
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
