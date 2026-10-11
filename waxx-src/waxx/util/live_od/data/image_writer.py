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

2026-10-05: what the writers hold and have not written yet is counted
(QueueGauge, POLL's ``writer_queue``; a WARNING past WRITER_QUEUE_WARN_BYTES).
The queue stays unbounded. A frame handed over after the writer was finished,
and anything still queued when the worker stops after an unexpected error, is
reported as not written (it used to sit unseen in a queue nobody read).
"""
import json
import os
import threading
import time
from queue import Empty, Queue

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


# What the writers hold that is not on disk yet: frames and pushed arrays queued
# for a SaveWorker. The queue itself stays unbounded on purpose -- refusing or
# dropping a write would lose data -- so a slow drive shows up as memory. This
# makes that backlog visible (POLL's ``writer_queue``) and loud: one WARNING when
# it crosses WRITER_QUEUE_WARN_BYTES, one INFO when it is back below half of it.
WRITER_QUEUE_WARN_BYTES = 200 * 2**20


def _queued_nbytes(item) -> int:
    """The array bytes one queue item holds (a PutItem, or a frame tuple)."""
    try:
        if isinstance(item, PutItem):
            return int(item.array.nbytes)
        return int(getattr(item[0], "nbytes", 0))
    except Exception:
        return 0


class QueueGauge:
    """Items and bytes waiting in the writers' queues, over every writer in the
    process (a finished run's writer may still be draining when the next run's
    starts). Thread-safe: added on the feeding threads, taken on the workers'."""

    def __init__(self, warn_bytes: int = WRITER_QUEUE_WARN_BYTES):
        self._lock = threading.Lock()
        self.warn_bytes = int(warn_bytes)
        self._items = 0
        self._bytes = 0
        self._peak_bytes = 0
        self._warned = False

    def added(self, item):
        nbytes = _queued_nbytes(item)
        with self._lock:
            self._items += 1
            self._bytes += nbytes
            self._peak_bytes = max(self._peak_bytes, self._bytes)
            crossed = not self._warned and self._bytes >= self.warn_bytes
            if crossed:
                self._warned = True
            items, total = self._items, self._bytes
        if crossed:
            logger.warning(f"Image writer backlog: {items} item(s), {total / 2**20:.0f} MB queued "
                           f"and not yet on disk (warning at {self.warn_bytes / 2**20:.0f} MB). "
                           f"Is the data drive slow? Nothing is dropped; the backlog is held in "
                           f"memory until written.")

    def taken(self, item):
        nbytes = _queued_nbytes(item)
        with self._lock:
            self._items = max(0, self._items - 1)
            self._bytes = max(0, self._bytes - nbytes)
            drained = self._warned and self._bytes < self.warn_bytes / 2
            if drained:
                self._warned = False
            items, total = self._items, self._bytes
        if drained:
            logger.info(f"Image writer backlog down to {items} item(s), {total / 2**20:.0f} MB.")

    def snapshot(self) -> dict:
        """``{items, bytes, peak_bytes (since liveOD started), warn_bytes}``."""
        with self._lock:
            return {"items": self._items, "bytes": self._bytes,
                    "peak_bytes": self._peak_bytes, "warn_bytes": self.warn_bytes}


# the one every writer reports to (POLL reads it)
WRITER_QUEUE = QueueGauge()


def writer_queue_stats() -> dict:
    """What the writers have queued and not yet written (QueueGauge.snapshot)."""
    return WRITER_QUEUE.snapshot()


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


# --- a dead file handle (2026-10-10) -------------------------------------------------
# Runs 86095 (10:42:49) and 86182 (21:46:38): the writer's one long-lived handle on
# the run file on the B: share started failing every access ("file read failed ...
# errno = 22, 'Invalid argument' ... offset = 800") and the worker carried on with it,
# so every frame and pushed array after the fault was lost while a fresh open (the
# END_RUN save) worked. Now a write that fails on a handle that no longer answers
# closes it, reopens the file (the same open liveOD uses: Scribe.wait_for_data_available,
# r+, HDF5's default locking) and writes again what may not have reached the disk.
#
# What a dead handle can lose: everything written since the handle's last flush may
# still be only in HDF5's caches (chunk data, chunk-index entries, the headers of
# datasets created since then). So the writer flushes after every dataset it
# creates, whenever its queue runs empty (about once per shot) and at least every
# FLUSH_EVERY_ITEMS items or FLUSH_EVERY_S seconds, and keeps what it wrote since
# the last flush ("pending") in memory. The reports count an item as written only
# once a flush has succeeded after it. After a reopen every pending item is written
# again (slot writes: the same bytes to the same place), and what was committed
# before is checked (every dataset there with its shape and dtype; each dataset's
# last committed item read back against its sha256). Each reopen is recorded in the
# run file's root attribute ``writer_reopens`` (JSON) and the log; anything the check
# finds wrong makes the run incomplete (``problems``, read by RunFile.save).
#
# A reopen that keeps failing sets ``fault`` and emits ``writer_fault_signal`` -- NOT
# save_failed_signal, whose slot resets the run and deletes its file, with every
# frame written before the fault in it. The server reports ``fault`` at SHOT_COMPLETE
# and in POLL (``writer_fault``). Items that come meanwhile are reported not written,
# and one reopen is tried at most every DEAD_RETRY_S on the next item.

#: Waits (s) before each reopen try; each try is one open bounded by REOPEN_TRY_S.
REOPEN_WAITS_S = (0.0, 5.0, 15.0)
REOPEN_TRY_S = 5.0
#: Flush at least this often (also after each dataset made and when the queue is empty).
FLUSH_EVERY_ITEMS = 32
FLUSH_EVERY_S = 5.0
#: A flush slower than this is reported once per run (WARNING).
SLOW_FLUSH_S = 1.0
#: Once reopening has failed: one more try at most this often.
DEAD_RETRY_S = 60.0
#: Reopen records kept in full (later ones are counted).
MAX_REOPEN_RECORDS = 20


def _item_name(item) -> str:
    return repr(item) if isinstance(item, PutItem) else f"frame {item[1]}"


def _item_key(item) -> str:
    return item.key if isinstance(item, PutItem) else "images"


def _sha(array) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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
    # reason -- the HDF5 file could not be opened at all (the run is reset: its
    # slot deletes the file, which holds nothing yet)
    save_failed_signal = pyqtSignal(str)
    # reason -- the handle died mid-run and reopening failed; the file is KEPT
    # (it holds everything written before the fault), the run is not reset
    writer_fault_signal = pyqtSignal(str)

    def __init__(self, save_queue: Queue, wait_fn, n_img: int,
                 wait_timeout: float = 120.,
                 check_interrupt_method=None,
                 report: WriteReport = None,
                 put_report: PutReport = None,
                 gauge: QueueGauge = None,
                 reopen_fn=None, sleep=time.sleep, clock=time.monotonic, wall=time.time):
        super().__init__()
        self._save_queue = save_queue
        # what the queue holds, counted by the feeder (ImageWriter) and taken
        # off here as each item comes out
        self._gauge = gauge if gauge is not None else WRITER_QUEUE
        self._wait_fn = wait_fn
        self._n_img = n_img
        self._wait_timeout = wait_timeout
        self._check_interrupt = check_interrupt_method if check_interrupt_method is not None else (lambda: False)
        self.interrupted = False
        self.report = report if report is not None else WriteReport()
        self.put_report = put_report if put_report is not None else PutReport()
        self.open_error = ""        # why the file could not be opened ("" = it could)
        self.file_open = threading.Event()   # the file is open (or the open failed)
        # a reopen is the same open as the first one, each try bounded
        self._reopen_fn = reopen_fn if reopen_fn is not None else (
            lambda: wait_fn(timeout=REOPEN_TRY_S, check_interrupt_method=lambda: False))
        self._sleep, self._clock, self._wall = sleep, clock, wall
        self._f = None
        self._pending = []          # [(item, key, where, dtype)] written since the last flush
        self._committed = {}        # key -> (where, sha256, dtype) of its last committed item
        self._layout = {}           # key -> (shape, dtype) of every dataset written into
        self._last_flush = 0.0
        self._slow_flush_warned = False
        self._dead_since = None     # the handle died and reopening failed (clock)
        self._last_dead_try = None
        self._rec = None            # the reopen being made (its record)
        self._n_lost_logged = 0
        self._datasets_created = False
        self._declared_checked = False
        self._frame_shape, self._frame_dtype = None, None
        self._n_refused = 0
        #: "" or why the writer could not keep writing (a reopen failed)
        self.fault = ""
        #: one record per reopen (MAX_REOPEN_RECORDS kept; ``reopens_more`` counts the rest)
        self.reopens = []
        self.reopens_more = 0
        #: what a reopen's check found wrong: the run is saved incomplete
        self.problems = []

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

        self._f = f
        self._last_flush = self._clock()
        try:
            while True:
                item = self._save_queue.get()
                if item is None:            # sentinel — we're done
                    break
                self._gauge.taken(item)
                if self.interrupted or f is None:
                    if isinstance(item, PutItem) and f is None:
                        self.put_report.not_written(f"{item!r}: the file could not be opened")
                    continue                # drain without writing
                self._take(item)
                # flush when the queue has run empty (about once per shot) or
                # when enough has piled up since the last flush
                if self._pending and (self._save_queue.empty() or self._flush_due()):
                    self._flush_or_reopen()
            if self._pending and not self.interrupted:
                self._flush_or_reopen()
        except Exception as exc:
            logger.exception(f"SaveWorker: unexpected error: {exc}")
        finally:
            if self._f is not None:
                if self._pending:
                    # an interrupted run's last items, or an unexpected error:
                    # flushed with the close if the handle allows
                    try:
                        self._flush()
                    except Exception as exc:
                        self._lose_pending(f"the final flush failed ({type(exc).__name__}: {exc})")
                self._close_quietly(self._f)
                self._f = None
            elif self._pending:
                self._lose_pending("the file was not open at the end")
            self._drain_unwritten()
            self.done_writing_signal.emit()

    # -- one item ------------------------------------------------------------------

    def _take(self, item):
        """Write one item; on a dead handle, reopen and write it again."""
        if self._f is None:
            if not self._revive(item):
                self._not_written(item, f"not written: the file's handle failed and the file "
                                        f"could not be reopened ({self.fault})")
                return
        try:
            self._write(item)
            return
        except Exception as exc:
            if self._alive(self._f):
                # the handle is fine: this item's own error, as before
                self._item_error(item, exc)
                return
            first = exc
        if not self._reopen(_item_name(item), first):
            self._not_written(item, f"not written: the file's handle failed and the file could "
                                    f"not be reopened ({self.fault})")
            return
        # the item that met the dead handle, again; a dataset it has to make
        # again goes in the reopen's record
        rec = self.reopens[-1] if self.reopens else None
        n_made = len(rec["recreated"]) if rec else 0
        self._rec = rec
        try:
            self._write(item)
        except Exception as exc:
            self._item_error(item, exc, after_reopen=True)
        finally:
            self._rec = None
        if rec and len(rec["recreated"]) != n_made:
            self._write_records()

    def _item_error(self, item, exc, after_reopen=False):
        how = "write failed after the file was reopened" if after_reopen else "write failed"
        if isinstance(item, PutItem):
            line = f"{item!r}: {type(exc).__name__}: {exc}"
            logger.error(f"SaveWorker: pushed array not written: {line}")
            self.put_report.not_written(line if not after_reopen else f"{line} ({how})")
        else:
            logger.exception(f"SaveWorker: write error at idx={item[1]}: {exc}")
            self.report.not_written(f"frame {item[1]}: {how}: {type(exc).__name__}: {exc}")

    def _not_written(self, item, why):
        """``item`` did not reach the file (or reached it unflushed and was lost)."""
        self._n_lost_logged += 1
        log = logger.error if self._n_lost_logged <= 3 else logger.debug
        log(f"SaveWorker: {_item_name(item)} {why}")
        if isinstance(item, PutItem):
            self.put_report.not_written(f"{item!r}: {why}")
        else:
            self.report.not_written(f"frame {item[1]}: {why}")

    def _write(self, item):
        """One item into the file through the current handle; raises on an I/O
        error. A frame that does not fit the run's images is refused (counted)."""
        f = self._f
        if isinstance(item, PutItem):
            self._write_put(f, item)
            return
        img, idx, img_t = item
        # Lazy dataset creation on the first image received.
        # images/image_timestamps are deliberately not pre-allocated
        # at file creation (that heavy NAS write would block the
        # INIT_RUN reply).  We create them here from the actual
        # image shape.
        if not self._datasets_created:
            dgrp = f['data']
            if 'images' not in dgrp:
                # one chunk per frame, gzip (waxa.data.h5_image)
                # unless COMPRESS_MAIN_IMAGES is off
                plain = {} if COMPRESS_MAIN_IMAGES else {
                    "chunks": None, "compression": None, "shuffle": False}
                shape = (self._n_img,) + img.shape
                if self._frame_shape is not None:     # (re-made after a reopen)
                    shape = (self._layout["images"][0][0],) + tuple(self._frame_shape)
                create_image_dataset(dgrp, 'images', shape=shape,
                                     dtype=img.dtype if self._frame_dtype is None
                                     else self._frame_dtype, **plain)
                dgrp.create_dataset('image_timestamps', shape=(shape[0],), dtype=np.float64)
                if self._rec is not None:
                    self._rec["recreated"].append("images")
                self._flush()                 # a new dataset is on disk before it is used
            self._frame_shape = tuple(dgrp['images'].shape[1:])
            self._frame_dtype = dgrp['images'].dtype
            self._layout["images"] = (tuple(dgrp['images'].shape), dgrp['images'].dtype)
            self._datasets_created = True
            if not self._declared_checked:
                self._declared_checked = True
                # The frames are kept whatever they are; if they are not
                # what the run declared, the run is marked incomplete
                # with that as the reason (RunFile.save).
                declared = _declared_frame(f)
                frame_shape, frame_dtype = self._frame_shape, self._frame_dtype
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
        if tuple(img.shape) != self._frame_shape or img.dtype != self._frame_dtype:
            # Cannot go in the run's dataset: the frame's shape or
            # dtype changed mid-run. Not written, and counted.
            self._n_refused += 1
            line = (f"frame {idx}: {tuple(img.shape)} {img.dtype} does not fit the "
                    f"run's images {self._frame_shape} {self._frame_dtype}; not written")
            self.report.not_written(line)
            if self._n_refused == 1:
                logger.error(f"SaveWorker: {line}. The run will be saved incomplete.")
            else:
                logger.debug(f"SaveWorker: {line}")
            return
        f['data']['images'][idx] = img
        f['data']['image_timestamps'][idx] = img_t
        self._pending.append((item, "images", ("index", (int(idx),)), self._frame_dtype))
        logger.debug(f"saved {idx + 1}/{self._n_img}")

    def _write_put(self, f, item: PutItem):
        """One pushed array into its slot; the dataset is made on first use.
        Raises on an error (the caller decides: the item's own, or a dead
        handle)."""
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
            if self._rec is not None:
                self._rec["recreated"].append(item.key)
            self._flush()                     # a new dataset is on disk before it is used
        ds = grp[name]
        if item.index is not None:
            ds[item.index] = item.array
            where = ("index", item.index)
        elif item.offset is not None:
            ds[item.offset:item.offset + len(item.array)] = item.array
            where = ("slice", item.offset, len(item.array))
        else:
            ds[...] = item.array
            where = ("all",)
        self._layout.setdefault(item.key, (tuple(ds.shape), ds.dtype))
        self._pending.append((item, item.key, where, ds.dtype))

    # -- flushing ------------------------------------------------------------------

    def _flush_due(self) -> bool:
        return (len(self._pending) >= FLUSH_EVERY_ITEMS
                or self._clock() - self._last_flush >= FLUSH_EVERY_S)

    def _flush(self):
        """Flush the handle; then what was written since the last flush counts
        as written. Raises on an I/O error."""
        t0 = self._clock()
        self._f.flush()
        dt = self._clock() - t0
        logger.debug(f"SaveWorker: flush {dt * 1e3:.0f} ms ({len(self._pending)} item(s))")
        if dt > SLOW_FLUSH_S and not self._slow_flush_warned:
            self._slow_flush_warned = True
            logger.warning(f"SaveWorker: a flush of the run file took {dt:.1f} s (> {SLOW_FLUSH_S:.0f} s; "
                           f"said once per run). The data drive is slow; nothing is lost, the "
                           f"backlog shows in POLL's writer_queue.")
        self._commit()
        self._last_flush = self._clock()

    def _commit(self):
        last = {}
        for item, key, where, dtype in self._pending:
            if isinstance(item, PutItem):
                self.put_report.wrote(key)
            else:
                self.report.wrote_one()
            last[key] = (item, where, dtype)
        for key, (item, where, dtype) in last.items():
            arr = item.array if isinstance(item, PutItem) else item[0]
            self._committed[key] = (where, _sha(np.asarray(arr).astype(dtype)), dtype)
        self._pending = []

    def _flush_or_reopen(self):
        if self._f is None or not self._pending:
            return
        try:
            self._flush()
            return
        except Exception as exc:
            first = exc
        self._reopen("the flush", first)

    def _lose_pending(self, why):
        lost, self._pending = self._pending, []
        for item, *_ in lost:
            self._not_written(item, f"written but not flushed when {why}; lost")
        return len(lost)

    # -- the handle ----------------------------------------------------------------

    @staticmethod
    def _alive(f) -> bool:
        """The handle still answers (two small reads)."""
        if f is None:
            return False
        try:
            _ = 'data' in f
            f.attrs.get('images_shape', None)
            return True
        except Exception:
            return False

    def _close_quietly(self, f, rec=None):
        if f is None:
            return
        try:
            f.close()
        except Exception as exc:
            # (a dead handle's close fails on its flush; the OS handle may stay)
            logger.warning(f"SaveWorker: closing the old handle failed: {type(exc).__name__}: {exc}")
            if rec is not None:
                rec["close_error"] = f"{type(exc).__name__}: {exc}"[:300]
            try:
                f.id.close()
            except Exception:
                pass

    def _record(self, rec):
        if len(self.reopens) < MAX_REOPEN_RECORDS:
            self.reopens.append(rec)
        else:
            self.reopens_more += 1

    def records_json(self) -> str:
        """``writer_reopens`` as stored in the run file."""
        return json.dumps({"reopens": self.reopens, "more": self.reopens_more,
                           "fault": self.fault, "problems": self.problems})

    def _write_records(self):
        try:
            self._f.attrs['writer_reopens'] = self.records_json()
            self._f.flush()
        except Exception as exc:
            logger.warning(f"SaveWorker: could not record the reopen in the file now "
                           f"({type(exc).__name__}: {exc}); the END_RUN save records it")

    def _revive(self, item) -> bool:
        """After a failed reopen: one more try, at most every DEAD_RETRY_S."""
        now = self._clock()
        if self._last_dead_try is not None and now - self._last_dead_try < DEAD_RETRY_S:
            return False
        self._last_dead_try = now
        return self._reopen(_item_name(item), None, waits=(0.0,))

    def _reopen(self, what, exc, waits=REOPEN_WAITS_S) -> bool:
        """Close the dead handle, open the file again, check what was committed,
        write again what was pending. False when no handle could be had (then
        ``fault`` says why and the pending items are reported lost)."""
        t = self._wall()
        when = time.strftime("%H:%M:%S", time.localtime(t))
        err = "still unwritable after an earlier failed reopen" if exc is None else \
            f"{type(exc).__name__}: {exc}"
        rec = {"t": t, "time": when, "item": what, "error": err[:400], "attempts": 0,
               "pending": len(self._pending), "rewritten": 0, "recreated": [], "verify": "",
               "outcome": ""}
        if exc is not None:
            logger.error(f"SaveWorker: the run file's handle stopped answering at {what} ({err}); "
                         f"closing it and opening the file again ({len(waits)} tries). "
                         f"{len(self._pending)} item(s) written since the last flush will be "
                         f"written again.")
        self._close_quietly(self._f, rec)
        self._f = None
        last = ""
        for wait in waits:
            if wait:
                self._sleep(wait)
            rec["attempts"] += 1
            try:
                cand = self._reopen_fn()
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                logger.warning(f"SaveWorker: reopen try {rec['attempts']} failed: {last}")
                continue
            if self._alive(cand):
                self._f = cand
                break
            last = "the reopened handle does not answer either"
            logger.warning(f"SaveWorker: reopen try {rec['attempts']}: {last}")
            self._close_quietly(cand)
        if self._f is None:
            rec["outcome"] = f"failed: {last}"[:400]
            self._record(rec)
            n = self._lose_pending(f"the handle failed at {when}")
            self._dead_since = self._dead_since or self._clock()
            self._last_dead_try = self._clock()
            if not self.fault:
                self.fault = (f"the run file's handle failed at {when} ({err[:200]}) and reopening "
                              f"it failed ({last[:200]}); data from then on is not written")
                logger.error(f"SaveWorker: {self.fault}. {n} unflushed item(s) lost. The file is "
                             f"KEPT with what was written before; the run will be saved "
                             f"incomplete. One reopen is tried every {DEAD_RETRY_S:.0f} s.")
                self.writer_fault_signal.emit(self.fault)
            return False
        # a handle again: check what was committed, then write the pending again
        self._dead_since = None
        self._datasets_created = False
        self._rec = rec
        try:
            problems = self._verify()
            pending, self._pending = self._pending, []
            for item, *_ in pending:
                try:
                    self._write(item)
                    rec["rewritten"] += 1
                except Exception as e:
                    self._item_error(item, e, after_reopen=True)
            self._flush()
        except Exception as e:
            # the new handle failed at once: give up on it like a failed reopen
            self._rec = None
            self._close_quietly(self._f)
            self._f = None
            rec["outcome"] = f"reopened, then failed again: {type(e).__name__}: {e}"[:400]
            self._record(rec)
            self._lose_pending(f"the reopened handle failed at once")
            self._dead_since = self._clock()
            self._last_dead_try = self._clock()
            if not self.fault:
                self.fault = (f"the run file's handle failed at {when} ({err[:200]}); the reopened "
                              f"handle failed at once ({type(e).__name__}: {e})")
                logger.error(f"SaveWorker: {self.fault}. The file is KEPT; the run will be saved "
                             f"incomplete.")
                self.writer_fault_signal.emit(self.fault)
            return False
        self._rec = None
        rec["verify"] = "ok" if not problems else "; ".join(problems)[:1000]
        rec["outcome"] = "reopened"
        if problems:
            self.problems += [f"the writer reopened the file at {when}: {p}" for p in problems]
        if self.fault:
            self.fault += f"; reopened at {when}"
        self._record(rec)
        self._write_records()
        log = logger.warning if not problems else logger.error
        log(f"SaveWorker: run file reopened at {when} after {rec['attempts']} try(s); "
            f"{rec['rewritten']} item(s) written again"
            + (f", re-made {rec['recreated']}" if rec["recreated"] else "")
            + f"; check of what was written before: {rec['verify']}")
        return True

    def _dataset(self, key):
        """The dataset ``data/<key>``, or None when it is not there."""
        grp = self._f['data']
        for p in [p for p in key.split('/') if p]:
            if p not in grp:
                return None
            grp = grp[p]
        return grp

    def _verify(self) -> list:
        """After a reopen: every dataset written into is there with its shape
        and dtype, and each one's last committed item reads back the same."""
        problems = []
        for key, (shape, dtype) in self._layout.items():
            ds = self._dataset(key)
            if ds is None:
                if key in self._committed:
                    problems.append(f"data/{key} is missing (it held flushed data)")
                continue
            if tuple(ds.shape) != tuple(shape) or ds.dtype != dtype:
                problems.append(f"data/{key} is {tuple(ds.shape)} {ds.dtype}, was {tuple(shape)} {dtype}")
                continue
            c = self._committed.get(key)
            if c is None:
                continue
            where, sha, cdtype = c
            try:
                if where[0] == "index":
                    back = ds[where[1]]
                elif where[0] == "slice":
                    back = ds[where[1]:where[1] + where[2]]
                else:
                    back = ds[...]
            except Exception as exc:
                problems.append(f"data/{key}: its last flushed item cannot be read back "
                                f"({type(exc).__name__}: {exc})")
                continue
            if _sha(np.asarray(back).astype(cdtype)) != sha:
                problems.append(f"data/{key}{list(where[1:])}: its last flushed item reads back "
                                f"different")
        return problems

    def _drain_unwritten(self):
        """What is still queued once the loop is over (only after an unexpected
        error: the end marker is always last) will never be written. Each item
        is reported as not written, and taken off the backlog gauge."""
        n = 0
        while True:
            try:
                item = self._save_queue.get_nowait()
            except Empty:
                break
            if item is None:
                continue
            self._gauge.taken(item)
            n += 1
            if isinstance(item, PutItem):
                self.put_report.not_written(f"{item!r}: the writer stopped before writing it")
            else:
                self.report.not_written(f"frame {item[1]}: the writer stopped before writing it")
        if n:
            logger.error(f"SaveWorker: {n} queued item(s) were never written: the writer "
                         f"stopped first. The run will be saved incomplete.")


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

    def __init__(self, data_filepath: str, server_owned: bool = False,
                 gauge: QueueGauge = None):
        # Deliberately not Scribe.__init__: it builds a DataSaver and a run-id
        # source, which nothing here uses. (DataHandler used to build both on
        # every camera run, through PyQt's cooperative __init__.)
        self.data_filepath = data_filepath
        self.server_owned = bool(server_owned)
        self._save_queue = None
        self._worker = None
        self._finished = False
        # the backlog gauge (POLL's writer_queue); a test may pass its own
        self._gauge = gauge if gauge is not None else WRITER_QUEUE
        # put / put_data / finish under one lock: nothing is queued behind the
        # end marker, where the worker would never take it
        self._put_lock = threading.Lock()
        self._late_frames = 0
        self.report = None          # WriteReport, from the first start
        self.put_report = PutReport()
        self.images_finished = threading.Event()    # the camera's frames are all in

    @property
    def started(self) -> bool:
        return self._worker is not None

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def fault(self) -> str:
        """"" or why the writer could not keep writing (its handle died and
        reopening failed; the file is kept). POLL / SHOT_COMPLETE ``writer_fault``."""
        return self._worker.fault if self._worker is not None else ""

    @property
    def problems(self) -> list:
        """What a reopen's check found wrong (the run is saved incomplete)."""
        return list(self._worker.problems) if self._worker is not None else []

    @property
    def reopens(self) -> list:
        """The reopen records (empty for a run whose handle never failed)."""
        return list(self._worker.reopens) if self._worker is not None else []

    def records_json(self) -> str:
        """``writer_reopens`` as stored in the run file ("" when nothing happened)."""
        w = self._worker
        if w is None or not (w.reopens or w.fault):
            return ""
        return w.records_json()

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
            gauge=self._gauge,
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
        """Non-blocking hand-off of one image. A frame that comes once the
        writer is finished cannot be written (the worker stops at the end
        marker): it is reported as not written, not left in a queue nobody
        reads."""
        item = (img, idx, img_t)
        with self._put_lock:
            if self._save_queue is None:
                raise RuntimeError("ImageWriter.put before the writer was started")
            if not self._finished:
                self._gauge.added(item)
                self._save_queue.put(item)
                return
            self._late_frames += 1
            n_late = self._late_frames
        line = f"frame {idx}: came after the writer was finished; not written"
        if self.report is not None:
            self.report.not_written(line)
        discarding = bool(self._worker is not None and self._worker.interrupted)
        if n_late == 1 and not discarding:      # (an aborted run's file goes anyway)
            logger.error(f"ImageWriter: {line}.")
        else:
            logger.debug(f"ImageWriter: {line}")

    def put_data(self, item: PutItem) -> bool:
        """Non-blocking hand-off of one pushed array; False once the writer is
        finished or was never started."""
        with self._put_lock:
            if self._worker is None or self._finished:
                return False
            self._gauge.added(item)
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
        with self._put_lock:
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
