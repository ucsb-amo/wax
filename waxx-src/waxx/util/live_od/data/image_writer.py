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

2026-10-02: the writer is the run's ONE open handle on its file, for every run
that saves data (camera or not), started by the server at INIT_RUN
(RunFile.start_writer) and finished by the server at END_RUN / abort / exit.
Besides the run camera's frames it writes the arrays the experiment pushes
during the run (PUT_DATA: auxiliary camera frames, scope traces), each into
its slot of a pre-allocated dataset, or into a dataset it creates on first
write. The DataHandler attaches to the registered writer (writer_for) instead
of opening its own; it only says when the camera's frames are all in
(images_done).
"""
import os
import threading
from queue import Queue

import numpy as np
from PyQt6.QtCore import QThread, Qt, pyqtSignal

from waxa.base import Scribe
from waxa.data.h5_image import COMPRESS_MAIN_IMAGES, MIN_BYTES, create_image_dataset
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


class PutItem:
    """One array pushed during the run (PUT_DATA): ``key`` is the path under
    ``data/`` (``img_mot``, ``scope_data/PD/v``), ``index`` the slot (a tuple
    of ints) or None for the whole dataset. ``full_shape`` / ``dtype`` /
    ``fill`` describe the dataset to create when the file has none yet."""

    __slots__ = ("key", "index", "array", "full_shape", "dtype", "fill", "offset")

    def __init__(self, key, index, array, full_shape=None, dtype=None, fill=None,
                 offset=None):
        self.key = str(key)
        self.index = None if index is None else tuple(int(i) for i in index)
        self.array = np.asarray(array)
        self.full_shape = None if full_shape is None else tuple(int(s) for s in full_shape)
        self.dtype = np.dtype(dtype) if dtype is not None else self.array.dtype
        self.fill = fill
        # a slice of the dataset along its first axis, starting here (with no
        # index): how a whole array goes in pieces
        self.offset = None if offset is None else int(offset)

    def __repr__(self):
        if self.index is not None:
            where = list(self.index)
        elif self.offset is not None:
            where = f"[{self.offset}:{self.offset + len(self.array)}]"
        else:
            where = "[...]"
        return f"PutItem({self.key}{where})"


class PutReport:
    """What the writer did with the pushed arrays: items written per key,
    and one line per item that did not reach the file. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._written = {}
        self._not_written = []
        self._created = []

    def wrote(self, key):
        with self._lock:
            self._written[key] = self._written.get(key, 0) + 1

    def created(self, key):
        with self._lock:
            self._created.append(key)

    def not_written(self, line):
        with self._lock:
            self._not_written.append(str(line))

    def snapshot(self):
        """``({key: items written}, [one line per item not written], [keys created])``."""
        with self._lock:
            return dict(self._written), list(self._not_written), list(self._created)


# The reports of the runs whose writers are in flight, by data file. The writer
# lives with the camera threads and RunFile with the server, and the file path
# is what both know the run by.
_REPORTS = {}
_REPORTS_LOCK = threading.Lock()
_MAX_REPORTS = 64       # runs whose END_RUN never came are dropped oldest first

# The writers the server started, by data file: the DataHandler of a camera run
# attaches to its run's one (writer_for) instead of opening a second handle.
_WRITERS = {}


def _report_key(filepath) -> str:
    return os.path.normcase(os.path.abspath(str(filepath)))


def new_write_report(filepath) -> WriteReport:
    """A fresh report for the run writing ``filepath`` (replacing any older one)."""
    return register_write_report(filepath, WriteReport())


def register_write_report(filepath, report) -> WriteReport:
    """``report`` is the one for the run writing ``filepath`` (replacing any
    older one): what END_RUN's completeness decision reads."""
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


def register_writer(filepath, writer):
    with _REPORTS_LOCK:
        _WRITERS[_report_key(filepath)] = writer
        while len(_WRITERS) > _MAX_REPORTS:
            _WRITERS.pop(next(iter(_WRITERS)))


def writer_for(filepath):
    """The writer the server started for ``filepath``; None when there is none
    (an older server path, or save_data=False)."""
    if not filepath:
        return None
    with _REPORTS_LOCK:
        return _WRITERS.get(_report_key(filepath))


def forget_writer(filepath):
    if not filepath:
        return
    with _REPORTS_LOCK:
        _WRITERS.pop(_report_key(filepath), None)


def _float_storage(full_shape, dtype):
    """Storage options for a big float dataset made on first use (scope
    traces): one chunk per shot and gzip, as the end-of-run scope writer
    stored them (the image helper compresses integer stacks only)."""
    shape = tuple(int(s) for s in full_shape)
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating) or not shape:
        return {}
    if int(np.prod(shape)) * dtype.itemsize < MIN_BYTES:
        return {}
    chunks = True if len(shape) == 1 else (1,) * (len(shape) - 2) + shape[-2:]
    return {"chunks": chunks, "compression": "gzip", "compression_opts": 4}


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
    2. For each image, ``save_queue.put((img, idx, img_t))``; for a pushed
       array, ``save_queue.put(PutItem(...))``.
    3. When the run is over, ``save_queue.put(None)`` (sentinel).
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
    declared (written all the same). ``put_report`` (a PutReport) does the
    same for the pushed arrays.
    """

    done_writing_signal = pyqtSignal()
    save_failed_signal = pyqtSignal(str)   # reason — emitted when the HDF5 file is unusable

    def __init__(self, save_queue: Queue, wait_fn, n_img: int,
                 wait_timeout: float = 120.,
                 check_interrupt_method=None,
                 report: WriteReport = None,
                 put_report: PutReport = None):
        super().__init__()
        self._save_queue = save_queue
        self._wait_fn = wait_fn
        self._n_img = n_img
        self._wait_timeout = wait_timeout
        self._check_interrupt = check_interrupt_method if check_interrupt_method is not None else (lambda: False)
        self.interrupted = False
        self.report = report if report is not None else WriteReport()
        self.put_report = put_report if put_report is not None else PutReport()
        self.open_error = ""        # why the file could not be opened ("" = it could)
        self.file_open = threading.Event()   # the file is open (or the open failed)

    def set_n_img(self, n_img: int):
        self._n_img = int(n_img)

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
            self.open_error = str(exc)
            # Nothing can be saved for this run.  Tell the GUI so the run is
            # aborted now instead of acquiring an entire scan whose images are
            # all discarded, with the failure only surfacing at END_RUN.
            self.save_failed_signal.emit(str(exc))
        finally:
            self.file_open.set()

        _datasets_created = False
        frame_shape, frame_dtype = None, None
        n_refused = 0
        try:
            while True:
                item = self._save_queue.get()
                if item is None:            # sentinel — we're done
                    break
                if self.interrupted or f is None:
                    if isinstance(item, PutItem) and f is None:
                        self.put_report.not_written(f"{item!r}: the file could not be opened")
                    continue                # drain without writing
                if isinstance(item, PutItem):
                    self._write_put(f, item)
                    continue
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
                            # one chunk per frame, gzip (waxa.data.h5_image)
                            # unless COMPRESS_MAIN_IMAGES is off
                            plain = {} if COMPRESS_MAIN_IMAGES else {
                                "chunks": None, "compression": None, "shuffle": False}
                            create_image_dataset(
                                dgrp, 'images',
                                shape=(self._n_img,) + img.shape,
                                dtype=img.dtype, **plain,
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

    def _write_put(self, f, item: PutItem):
        """One pushed array into its slot; the dataset is made on first use."""
        try:
            grp = f['data']
            parts = [p for p in item.key.split('/') if p]
            if not parts:
                raise ValueError("empty key")
            for p in parts[:-1]:
                grp = grp.require_group(p)
            name = parts[-1]
            if name not in grp:
                if item.full_shape is None:
                    raise ValueError("no such dataset and no full_shape to create it")
                kw = {}
                if item.fill is not None:
                    kw["fillvalue"] = item.fill
                kw.update(_float_storage(item.full_shape, item.dtype))
                create_image_dataset(grp, name, shape=item.full_shape, dtype=item.dtype, **kw)
                self.put_report.created(item.key)
            ds = grp[name]
            if item.index is not None:
                ds[item.index] = item.array
            elif item.offset is not None:
                ds[item.offset:item.offset + len(item.array)] = item.array
            else:
                ds[...] = item.array
            self.put_report.wrote(item.key)
        except Exception as exc:
            line = f"{item!r}: {type(exc).__name__}: {exc}"
            logger.error(f"SaveWorker: pushed array not written: {line}")
            self.put_report.not_written(line)


class ImageWriter(Scribe):
    """One run's hold on the data file: the one open handle, for the whole run.

    Started by the server (``start_for_server``, RunFile.start_writer) for
    every run that saves data; the camera run's DataHandler attaches to it
    (``start``) and hands it images; the experiment's pushed arrays come in
    through ``put_data``. The server finishes it (``finish``) at END_RUN,
    abort or exit. A writer the DataHandler made itself (no server-started
    one: ``server_owned`` False) still closes when the camera's frames are in,
    as before.

    Scribe supplies ``wait_for_data_available`` and ``remove_incomplete_data``.
    """

    def __init__(self, data_filepath: str, server_owned: bool = False):
        # Deliberately not Scribe.__init__: it builds a DataSaver and a run-id
        # source, which nothing here uses. (DataHandler used to build both on
        # every camera run, through PyQt's cooperative __init__.)
        self.data_filepath = data_filepath
        self.server_owned = bool(server_owned)
        self._save_queue = None
        self._worker = None
        self._finished = False
        self.report = None          # WriteReport, from the first start
        self.put_report = PutReport()
        self.images_finished = threading.Event()    # the camera's frames are all in

    @property
    def started(self) -> bool:
        return self._worker is not None

    @property
    def finished(self) -> bool:
        return self._finished

    def _start_worker(self, n_img: int, check_interrupt_method):
        self._save_queue = Queue()
        # The frame report is registered by file path, where END_RUN finds
        # it, once a DataHandler feeds this writer (start): a run whose
        # camera frames never came through it has no frame report, as before.
        self.report = WriteReport()
        if not self.server_owned:
            register_write_report(self.data_filepath, self.report)
        self._worker = SaveWorker(
            self._save_queue,
            wait_fn=self.wait_for_data_available,
            n_img=int(n_img),
            wait_timeout=DATA_SAVER_TIMEOUT,
            check_interrupt_method=self._interrupt_check(check_interrupt_method),
            report=self.report,
            put_report=self.put_report,
        )
        self._worker.start()

    def _interrupt_check(self, check_interrupt_method):
        """The worker's "stop waiting for the file" test: an interrupted
        writer (nothing of what it holds will be written: a reset) as well as
        the caller's own check. A writer merely finished still waits for the
        file, to write what it was given."""
        def interrupted():
            w = self._worker
            return bool(w is not None and w.interrupted)
        if check_interrupt_method is None:
            return interrupted
        return lambda: interrupted() or bool(check_interrupt_method())

    def start_for_server(self, n_img: int, on_done=None):
        """The server starts the run's writer right after the file is made.
        ``on_done`` (any callable) runs on the writer thread once the file is
        closed."""
        if self._worker is not None:
            return
        self._start_worker(n_img, None)
        if on_done is not None:
            self._worker.done_writing_signal.connect(on_done, Qt.ConnectionType.DirectConnection)

    def start(self, n_img: int, check_interrupt_method, done_signal, failed_signal):
        """The DataHandler's side: start the writer, or attach to the one the
        server started. Call from the thread that will feed it: a fresh worker
        waits for the HDF5 file in its own thread while the caller's dispatch
        loop runs unblocked.  Images that arrive before the file is ready queue
        up (unbounded) and are drained once the file becomes available.

        The worker's signals are routed through the caller's (DataHandler's)
        so downstream consumers (the server's writer gate, the GUI's
        abort-on-save-failure slot) are unaffected.
        """
        if self._worker is None:
            self._start_worker(n_img, check_interrupt_method)
        else:
            self._worker.set_n_img(n_img)
            if check_interrupt_method is not None:
                self._worker._check_interrupt = self._interrupt_check(check_interrupt_method)
            register_write_report(self.data_filepath, self.report)
        self._worker.done_writing_signal.connect(done_signal)
        self._worker.save_failed_signal.connect(failed_signal)
        if self._worker.open_error:
            failed_signal.emit(self._worker.open_error)

    def put(self, img, idx: int, img_t: float):
        """Non-blocking hand-off of one image."""
        self._save_queue.put((img, idx, img_t))

    def put_data(self, item: PutItem) -> bool:
        """Non-blocking hand-off of one pushed array; False once the writer is
        finished or was never started."""
        if self._worker is None or self._finished:
            return False
        self._save_queue.put(item)
        return True

    def images_done(self, interrupted: bool):
        """The camera's frames are all in (or the grab was interrupted). A
        writer of the server's stays open for the experiment's pushed arrays
        until the server finishes it; one the DataHandler made closes now."""
        self.images_finished.set()
        if self._worker is not None and interrupted:
            self._worker.interrupted = True
        if not self.server_owned:
            self.finish(interrupted)

    def finish(self, interrupted: bool = False):
        """No more data. The worker closes the file and emits done; if
        ``interrupted`` it drains what is queued without writing it."""
        if self._worker is None or self._finished:
            return
        self._finished = True
        if interrupted:
            self._worker.interrupted = True
        self._save_queue.put(None)
        forget_writer(self.data_filepath)

    def wait_done(self, timeout: float) -> bool:
        """True once the worker has closed the file (at once when none ran)."""
        if self._worker is None:
            return True
        return self._worker.wait(int(timeout * 1000))

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
