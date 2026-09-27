"""The run's data file as the camera threads see it: images into it as they
arrive, and the file deleted when the grab dies.

Moved out of camera_mother.py (2026-09-20). SaveWorker is as it was; ImageWriter
is what DataHandler used to do to the file itself, by way of inheriting Scribe.

2026-09-26 (B3): every frame the writer is handed is accounted for in a
WriteReport, which END_RUN's completeness decision reads (RunFile.save). Before,
a failed write was logged and the frame still counted, so the file said complete
with a zero-filled slot. Nothing the camera delivered is dropped for looking
wrong: the images dataset still takes the first frame's shape and dtype, and if
those differ from what the run declared at INIT_RUN (the file's ``images_shape``
/ ``images_dtype``), every frame is still written and the run is marked
incomplete with the mismatch as the reason. Only a frame that cannot be stored
in that dataset (its shape or dtype changed mid-run) is not written, and is
counted.
"""
import os
import threading
from queue import Queue

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from waxa.base import Scribe
from waxx.config.timeouts import DATA_SAVER_TIMEOUT
from waxx.util.live_od.log import get_logger

logger = get_logger("camera")


class WriteReport:
    """What the image writer did with the frames it was handed: how many it
    wrote, one line for each it could not write, and whether the frames were
    what the run declared. Filled by the SaveWorker thread, read by
    RunFile.save once the writer is done. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._written = 0
        self._not_written = []
        self._mismatches = []

    def wrote_one(self):
        with self._lock:
            self._written += 1

    def not_written(self, line: str):
        """A frame that did not reach the file (a write error, or a shape or
        dtype the dataset cannot hold)."""
        with self._lock:
            self._not_written.append(str(line))

    def mismatch(self, line: str):
        """The frames (all written) are not what the run declared."""
        with self._lock:
            self._mismatches.append(str(line))

    def snapshot(self):
        """``(frames written, [one line per frame not written], [mismatches])``."""
        with self._lock:
            return self._written, list(self._not_written), list(self._mismatches)


# The reports of the runs whose writers are in flight, by data file. The writer
# lives with the camera threads and RunFile with the server, and the file path
# is what both know the run by.
_REPORTS = {}
_REPORTS_LOCK = threading.Lock()
_MAX_REPORTS = 64       # runs whose END_RUN never came are dropped oldest first


def _report_key(filepath) -> str:
    return os.path.normcase(os.path.abspath(str(filepath)))


def new_write_report(filepath) -> WriteReport:
    """A fresh report for the run writing ``filepath`` (replacing any older one)."""
    report = WriteReport()
    with _REPORTS_LOCK:
        _REPORTS[_report_key(filepath)] = report
        while len(_REPORTS) > _MAX_REPORTS:
            _REPORTS.pop(next(iter(_REPORTS)))
    return report


def take_write_report(filepath):
    """The report of the run writing ``filepath``, removed from the registry;
    None if no writer was started for it (no camera, save_data=False)."""
    if not filepath:
        return None
    with _REPORTS_LOCK:
        return _REPORTS.pop(_report_key(filepath), None)


def _declared_frame(f):
    """``(frame shape, dtype)`` the run declared at INIT_RUN -- the file's
    ``images_shape`` (N_img, *frame) and ``images_dtype`` attributes -- or None
    for a file that does not declare them (created by an older path)."""
    try:
        shape = f.attrs.get('images_shape', None)
        dtype = f.attrs.get('images_dtype', None)
        if shape is None or dtype is None:
            return None
        if isinstance(dtype, bytes):
            dtype = dtype.decode()
        shape = tuple(int(s) for s in shape)
        if len(shape) < 2:
            return None
        return shape[1:], np.dtype(dtype)
    except Exception as exc:
        logger.warning(f"could not read the run's declared image shape/dtype: {exc}")
        return None


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

    Accounting
    ----------
    ``report`` (a WriteReport) counts every frame written, records every frame
    not written (a write error, or a frame whose shape or dtype differs from
    the dataset's), and records a first frame that differs from what the run
    declared (written all the same).
    """

    done_writing_signal = pyqtSignal()
    save_failed_signal = pyqtSignal(str)   # reason — emitted when the HDF5 file is unusable

    def __init__(self, save_queue: Queue, wait_fn, n_img: int,
                 wait_timeout: float = 120.,
                 check_interrupt_method=None,
                 report: WriteReport = None):
        super().__init__()
        self._save_queue = save_queue
        self._wait_fn = wait_fn
        self._n_img = n_img
        self._wait_timeout = wait_timeout
        self._check_interrupt = check_interrupt_method if check_interrupt_method is not None else (lambda: False)
        self.interrupted = False
        self.report = report if report is not None else WriteReport()

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
        frame_shape, frame_dtype = None, None
        n_refused = 0
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
                        frame_shape = tuple(dgrp['images'].shape[1:])
                        frame_dtype = dgrp['images'].dtype
                        _datasets_created = True
                        # The frames are kept whatever they are; if they are not
                        # what the run declared, the run is marked incomplete
                        # with that as the reason (RunFile.save).
                        declared = _declared_frame(f)
                        if declared is None:
                            logger.warning("the run's file declares no image shape/dtype; "
                                           f"frames are {frame_shape} {frame_dtype}, not checked")
                        elif (tuple(declared[0]) != frame_shape
                              or np.dtype(declared[1]) != frame_dtype):
                            line = (f"frames are {frame_shape} {frame_dtype} but the run "
                                    f"declared {tuple(declared[0])} {np.dtype(declared[1])}")
                            self.report.mismatch(line)
                            logger.error(f"SaveWorker: {line}. Every frame is still written; "
                                         f"the run will be saved incomplete.")
                    if tuple(img.shape) != frame_shape or img.dtype != frame_dtype:
                        # Cannot go in the run's dataset: the frame's shape or
                        # dtype changed mid-run. Not written, and counted.
                        n_refused += 1
                        line = (f"frame {idx}: {tuple(img.shape)} {img.dtype} does not fit the "
                                f"run's images {frame_shape} {frame_dtype}; not written")
                        self.report.not_written(line)
                        if n_refused == 1:
                            logger.error(f"SaveWorker: {line}. The run will be saved incomplete.")
                        else:
                            logger.debug(f"SaveWorker: {line}")
                        continue
                    f['data']['images'][idx] = img
                    f['data']['image_timestamps'][idx] = img_t
                    self.report.wrote_one()
                    logger.debug(f"saved {idx + 1}/{self._n_img}")
                except Exception as exc:
                    logger.exception(f"SaveWorker: write error at idx={idx}: {exc}")
                    self.report.not_written(f"frame {idx}: write failed: {type(exc).__name__}: {exc}")
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
        self.report = None      # WriteReport, from start()

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
        # registered by file path, where the server's END_RUN finds it
        self.report = new_write_report(self.data_filepath)
        self._worker = SaveWorker(
            self._save_queue,
            wait_fn=self.wait_for_data_available,
            n_img=n_img,
            wait_timeout=DATA_SAVER_TIMEOUT,
            check_interrupt_method=check_interrupt_method,
            report=self.report,
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
