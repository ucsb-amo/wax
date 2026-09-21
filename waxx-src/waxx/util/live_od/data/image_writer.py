"""The run's data file as the camera threads see it: images into it as they
arrive, and the file deleted when the grab dies.

Moved out of camera_mother.py (2026-09-20). SaveWorker is as it was; ImageWriter
is what DataHandler used to do to the file itself, by way of inheriting Scribe.
"""
from queue import Queue

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from waxa.base import Scribe
from waxx.config.timeouts import DATA_SAVER_TIMEOUT
from waxx.util.live_od.log import get_logger

logger = get_logger("camera")


class SaveWorker(QThread):
    """Writes images to HDF5 on a dedicated thread, decoupled from image display.

    On start, waits for the HDF5 file to become available (via ``wait_fn``).
    Images that arrive while waiting are buffered in ``save_queue`` (unbounded)
    and flushed once the file is ready — so DataHandler's display loop is never
    blocked by file I/O or slow NAS pre-allocation.

    Usage
    -----
    1. Construct with ``wait_fn`` (ImageWriter.wait_for_data_available) and call
       ``start()``.  DataHandler begins dispatching images immediately.
    2. For each image, ``save_queue.put((img, idx, img_t))``.
    3. When the grab loop ends, ``save_queue.put(None)`` (sentinel).
    4. SaveWorker closes the file and emits ``done_writing_signal``.

    Interruption
    ------------
    Set ``interrupted = True`` *before* putting the sentinel.  The worker
    will drain the queue without writing, then close the file and emit.
    """

    done_writing_signal = pyqtSignal()
    save_failed_signal = pyqtSignal(str)   # reason — emitted when the HDF5 file is unusable

    def __init__(self, save_queue: Queue, wait_fn, n_img: int,
                 wait_timeout: float = 120.,
                 check_interrupt_method=None):
        super().__init__()
        self._save_queue = save_queue
        self._wait_fn = wait_fn
        self._n_img = n_img
        self._wait_timeout = wait_timeout
        self._check_interrupt = check_interrupt_method if check_interrupt_method is not None else (lambda: False)
        self.interrupted = False

    def run(self):
        # Wait for the HDF5 file to be available.  This blocks only THIS thread;
        # DataHandler's dispatch loop (and therefore image display) continues
        # uninterrupted.  Images accumulate in _save_queue during the wait.
        f = None
        try:
            f = self._wait_fn(timeout=self._wait_timeout,
                              check_interrupt_method=self._check_interrupt)
        except Exception as exc:
            logger.exception(f"SaveWorker: could not open data file: {exc}")
            self.interrupted = True   # drain queue without writing
            # Nothing can be saved for this run.  Tell the GUI so the run is
            # aborted now instead of acquiring an entire scan whose images are
            # all discarded, with the failure only surfacing at END_RUN.
            self.save_failed_signal.emit(str(exc))

        _datasets_created = False
        try:
            while True:
                item = self._save_queue.get()
                if item is None:            # sentinel — we're done
                    break
                if self.interrupted or f is None:
                    continue                # drain without writing
                img, idx, img_t = item
                try:
                    # Lazy dataset creation on the first image received.
                    # images/image_timestamps are deliberately not pre-allocated
                    # at file creation (that heavy NAS write would block the
                    # INIT_RUN reply).  We create them here from the actual
                    # image shape.
                    if not _datasets_created:
                        dgrp = f['data']
                        if 'images' not in dgrp:
                            dgrp.create_dataset(
                                'images',
                                shape=(self._n_img,) + img.shape,
                                dtype=img.dtype,
                            )
                            dgrp.create_dataset(
                                'image_timestamps',
                                shape=(self._n_img,),
                                dtype=np.float64,
                            )
                        _datasets_created = True
                    f['data']['images'][idx] = img
                    f['data']['image_timestamps'][idx] = img_t
                    logger.debug(f"saved {idx + 1}/{self._n_img}")
                except Exception as exc:
                    logger.exception(f"SaveWorker: write error at idx={idx}: {exc}")
        except Exception as exc:
            logger.exception(f"SaveWorker: unexpected error: {exc}")
        finally:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
            self.done_writing_signal.emit()


class ImageWriter(Scribe):
    """One camera run's hold on the data file. Owned by the run's DataHandler,
    which hands it images and tells it when the grab is over.

    Scribe supplies ``wait_for_data_available`` and ``remove_incomplete_data``.
    """

    def __init__(self, data_filepath: str):
        # Deliberately not Scribe.__init__: it builds a DataSaver and a run-id
        # source, which nothing here uses. (DataHandler used to build both on
        # every camera run, through PyQt's cooperative __init__.)
        self.data_filepath = data_filepath
        self._save_queue = None
        self._worker = None

    @property
    def started(self) -> bool:
        return self._worker is not None

    def start(self, n_img: int, check_interrupt_method, done_signal, failed_signal):
        """Start the SaveWorker. Call from the thread that will feed it: it
        waits for the HDF5 file in its own thread while the caller's dispatch
        loop runs unblocked.  Images that arrive before the file is ready queue
        up (unbounded) and are drained once the file becomes available.

        The worker's signals are routed through the caller's (DataHandler's)
        so downstream consumers (the server's writer gate, the GUI's
        abort-on-save-failure slot) are unaffected.
        """
        self._save_queue = Queue()
        self._worker = SaveWorker(
            self._save_queue,
            wait_fn=self.wait_for_data_available,
            n_img=n_img,
            wait_timeout=DATA_SAVER_TIMEOUT,
            check_interrupt_method=check_interrupt_method,
        )
        self._worker.done_writing_signal.connect(done_signal)
        self._worker.save_failed_signal.connect(failed_signal)
        self._worker.start()

    def put(self, img, idx: int, img_t: float):
        """Non-blocking hand-off of one image."""
        self._save_queue.put((img, idx, img_t))

    def finish(self, interrupted: bool):
        """No more images. The worker closes the file and emits done; if
        ``interrupted`` it drains what is queued without writing it."""
        self._worker.interrupted = interrupted
        self._save_queue.put(None)

    def read_groups(self, camera_params, params, run_info):
        """Legacy path, for a run that sent no payloads: fill the three holders
        from the file's groups."""
        # (imported here: atomdata_base pulls in the analysis stack)
        from waxa.atomdata_base import unpack_group
        with self.wait_for_data_available() as f:
            unpack_group(f, 'camera_params', camera_params)
            unpack_group(f, 'params', params)
            unpack_group(f, 'run_info', run_info)

    def discard(self, delete_data: bool = True):
        """The grab died: delete the run's file (unless told not to)."""
        self.remove_incomplete_data(delete_data)
