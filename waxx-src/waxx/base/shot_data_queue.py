"""A bounded queue of per-shot arrays on their way into the run's file.

Camera streams hand each frame (with its record) to the queue and go straight
back to their camera; one sender thread pushes the items to liveOD
(PUT_DATA), in the order they came. The experiment process keeps no copy of
a stream's frames for the run: an item lives here only until liveOD has it,
and the queue holds at most ``max_bytes`` of them. A slow or stopped liveOD
therefore costs frames, counted, instead of stalling the stream workers or
growing the process until the run dies (the END_RUN fallback that carried
every frame lost run 84979, 2026-10-02).

Every item gets exactly one outcome, given to its ``on_done(ok, reason)``
and tallied per key:

=====================  ==================================================
ok                     liveOD took it (PUT_DATA acknowledged)
``queue_full``         the queue already held ``max_bytes``: the INCOMING
                       item is dropped (the queued ones keep their order)
``push_failed``        liveOD did not take it in 1 + ``retries`` attempts
``not_sent_at_end``    still queued when the queue was closed
``unconfirmed_at_end`` being sent when the queue was closed: it may or may
                       not be in the file
``no_put_data``        this run's liveOD takes no PUT_DATA (Expt drops it)
=====================  ==================================================

A slot whose item was not sent keeps what liveOD pre-allocated it with (the
container's fill value: frame zeros, NaN record), never a stale value -- with
one exception: a *clear* item (a warm-up / duplicate shot resetting its slot
ahead of the new frame) that was not sent leaves whatever the earlier frame
put there. Clears are tallied apart (``clears_failed``) for that reason.
Clears bypass the byte cap (they are rare and must not be dropped for room).

Thread-safe; nothing here raises into its callers.
"""

import collections
import threading
import time

import numpy as np

# give-up reasons (module docstring)
QUEUE_FULL = "queue_full"
PUSH_FAILED = "push_failed"
NOT_SENT_AT_END = "not_sent_at_end"
UNCONFIRMED_AT_END = "unconfirmed_at_end"
NO_PUT_DATA = "no_put_data"

# the tally keeps this many slots of each key's dropped items / failed clears
MAX_DROPPED_SLOTS = 10
# this many give-ups print a line each; later ones only count
MAX_PRINTED_FAILURES = 3


class _Item:
    __slots__ = ("specs", "nbytes", "on_done", "clear", "keys", "slot",
                 "done", "abandoned", "error")

    def __init__(self, specs, on_done, clear):
        self.specs = specs
        self.nbytes = int(sum(int(np.asarray(s["array"]).nbytes) for s in specs))
        self.on_done = on_done
        self.clear = bool(clear)
        self.keys = [str(s["key"]) for s in specs]
        index = specs[0].get("index") if specs else None
        self.slot = None if index is None else [int(i) for i in index]
        self.done = False           # its outcome is being reported by the sender
        self.abandoned = False      # close() reported it unconfirmed
        self.error = ""


def _new_tally():
    return {"sent": 0, "dropped": 0, "reasons": {}, "first_dropped_slots": [],
            "clears_sent": 0, "clears_failed": 0, "clear_fail_reasons": {},
            "first_clear_failed_slots": []}


class ShotDataQueue:
    """FIFO of PUT_DATA items with one sender thread (module docstring).

    ``push(specs)`` sends one item's specs (``[{key, index, array,
    full_shape, fill}]``, as Expt.push_shot_data builds them); it returns
    True when liveOD took them and returns False or raises otherwise.
    """

    def __init__(self, push, max_bytes, retries=2, retry_backoff_s=0.25,
                 name="shot-data-sender", label="[push]"):
        self._push = push
        self.max_bytes = int(max_bytes)
        self.retries = max(0, int(retries))
        self.retry_backoff_s = float(retry_backoff_s)
        self._name = str(name)
        self._label = str(label)
        self._cv = threading.Condition()
        self._items = collections.deque()
        self._bytes = 0             # queued items + the one being sent
        self._peak_bytes = 0
        self._busy = None           # the item being sent
        self._closed = False
        self._thread = None
        self._tallies = {}
        self._n_failures_printed = 0

    # ---- producers -------------------------------------------------------

    def put(self, specs, on_done=None, clear=False) -> bool:
        """Queue one item (its arrays are copied: a crop view would pin the
        whole sensor frame, and the byte count would lie). True when queued;
        otherwise ``on_done(False, reason)`` has been called already."""
        if not clear and self._full_for(specs):
            # no copy made for an item that cannot go in (checked again below)
            self._finish(self._bare_item(specs, on_done, clear), False, QUEUE_FULL)
            return False
        try:
            specs = [dict(s, array=np.array(s["array"], copy=True, order="C"))
                     for s in specs]
            item = _Item(specs, on_done, clear)
        except Exception as e:
            item = self._bare_item(specs, on_done, clear)
            item.error = f"{type(e).__name__}: {e}"
            self._finish(item, False, PUSH_FAILED)
            return False
        with self._cv:
            if self._closed:
                reason = NOT_SENT_AT_END
            elif (not item.clear and self._bytes > 0
                  and self._bytes + item.nbytes > self.max_bytes):
                # an empty queue takes any one item: the cap bounds what
                # waits, not the size of a frame
                reason = QUEUE_FULL
            else:
                self._items.append(item)
                self._bytes += item.nbytes
                self._peak_bytes = max(self._peak_bytes, self._bytes)
                self._ensure_thread()
                self._cv.notify_all()
                return True
        self._finish(item, False, reason)
        return False

    def _full_for(self, specs) -> bool:
        try:
            nbytes = sum(int(np.asarray(s["array"]).nbytes) for s in specs)
        except Exception:
            return False
        with self._cv:
            return (not self._closed and self._bytes > 0
                    and self._bytes + nbytes > self.max_bytes)

    def drop(self, specs, reason, on_done=None, clear=False):
        """Record an item that is not even queued (e.g. ``no_put_data``)."""
        self._finish(self._bare_item(specs, on_done, clear), False, str(reason))

    @staticmethod
    def _bare_item(specs, on_done, clear):
        """An item for the tally only (no copy of the arrays)."""
        try:
            item = _Item([dict(s, array=np.empty(0)) for s in specs], on_done, clear)
        except Exception:
            item = _Item([], on_done, clear)
        return item

    def ensure_keys(self, keys):
        """List ``keys`` in the report even if they never had an item."""
        with self._cv:
            for k in keys:
                self._tallies.setdefault(str(k), _new_tally())

    # ---- the sender thread -------------------------------------------------

    def _ensure_thread(self):
        """(caller holds the condition)"""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name=self._name)
            self._thread.start()

    def _run(self):
        while True:
            with self._cv:
                while not self._items and not self._closed:
                    self._cv.wait()
                if not self._items:
                    return
                item = self._items.popleft()
                self._busy = item
            ok = self._send(item)
            with self._cv:
                abandoned = item.abandoned
                item.done = not abandoned
            if abandoned:
                print(f"{self._label} note: a push still in flight when the run's queue "
                      f"closed then {'succeeded' if ok else 'failed'} "
                      f"({', '.join(item.keys)} at {item.slot}); the record says "
                      f"'{UNCONFIRMED_AT_END}' for it")
            else:
                self._finish(item, ok, None if ok else PUSH_FAILED)
            with self._cv:
                self._busy = None
                self._bytes -= item.nbytes
                self._cv.notify_all()

    def _send(self, item) -> bool:
        for attempt in range(1 + self.retries):
            try:
                if self._push(item.specs):
                    return True
                item.error = "refused"
            except Exception as e:
                item.error = f"{type(e).__name__}: {e}"
            if attempt < self.retries:
                with self._cv:
                    if item.abandoned:
                        return False
                    self._cv.wait(self.retry_backoff_s * (attempt + 1))
                    if item.abandoned:
                        return False
        return False

    # ---- outcomes ----------------------------------------------------------

    def _finish(self, item, ok, reason):
        """Tally one outcome, then tell the producer (outside every lock)."""
        printed = False
        with self._cv:
            for key in item.keys:
                t = self._tallies.setdefault(key, _new_tally())
                if item.clear:
                    if ok:
                        t["clears_sent"] += 1
                    else:
                        t["clears_failed"] += 1
                        t["clear_fail_reasons"][reason] = t["clear_fail_reasons"].get(reason, 0) + 1
                        if (item.slot is not None
                                and len(t["first_clear_failed_slots"]) < MAX_DROPPED_SLOTS):
                            t["first_clear_failed_slots"].append(item.slot)
                elif ok:
                    t["sent"] += 1
                else:
                    t["dropped"] += 1
                    t["reasons"][reason] = t["reasons"].get(reason, 0) + 1
                    if (item.slot is not None
                            and len(t["first_dropped_slots"]) < MAX_DROPPED_SLOTS):
                        t["first_dropped_slots"].append(item.slot)
            if (not ok and reason != NO_PUT_DATA
                    and self._n_failures_printed < MAX_PRINTED_FAILURES):
                self._n_failures_printed += 1
                printed = self._n_failures_printed
        if printed:
            what = "a slot clear" if item.clear else "a shot's data"
            more = " (later ones are only counted)" if printed == MAX_PRINTED_FAILURES else ""
            print(f"{self._label} !! {what} for {', '.join(item.keys)} at {item.slot} did "
                  f"not reach liveOD: {reason}"
                  + (f" ({item.error})" if item.error else "")
                  + f"; that slot keeps its fill value{more}")
        if item.on_done is not None:
            try:
                item.on_done(bool(ok), None if ok else reason)
            except Exception as e:
                print(f"{self._label} WARNING: an on_done callback raised {e!r}")

    # ---- the end of the run ------------------------------------------------

    def drain(self, timeout) -> bool:
        """Wait until every queued item has its outcome; False on timeout."""
        deadline = time.monotonic() + float(timeout)
        with self._cv:
            while self._items or self._busy is not None:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
            return True

    def close(self, timeout) -> dict:
        """Drain for up to ``timeout`` s, then give every item still waiting
        ``not_sent_at_end`` (and the one being sent, ``unconfirmed_at_end``).
        Later puts get ``not_sent_at_end`` at once. Returns those counts."""
        self.drain(timeout)
        with self._cv:
            self._closed = True
            left = list(self._items)
            self._items.clear()
            for item in left:
                self._bytes -= item.nbytes
            busy = self._busy
            if busy is not None and not busy.done:
                busy.abandoned = True
            else:
                busy = None
            self._cv.notify_all()
        for item in left:
            self._finish(item, False, NOT_SENT_AT_END)
        if busy is not None:
            self._finish(busy, False, UNCONFIRMED_AT_END)
        return {NOT_SENT_AT_END: len(left), UNCONFIRMED_AT_END: int(busy is not None)}

    # ---- reporting ---------------------------------------------------------

    @property
    def queued_bytes(self) -> int:
        with self._cv:
            return self._bytes

    @property
    def n_queued(self) -> int:
        with self._cv:
            return len(self._items) + (self._busy is not None)

    def report(self) -> dict:
        """``{"queue_max_bytes", "peak_bytes", "retries", "keys": {key:
        {sent, dropped, reasons, first_dropped_slots, clears_sent,
        clears_failed, clear_fail_reasons, first_clear_failed_slots}}}``."""
        with self._cv:
            keys = {k: {**t, "reasons": dict(t["reasons"]),
                        "first_dropped_slots": [list(s) for s in t["first_dropped_slots"]],
                        "clear_fail_reasons": dict(t["clear_fail_reasons"]),
                        "first_clear_failed_slots": [list(s) for s in
                                                     t["first_clear_failed_slots"]]}
                    for k, t in self._tallies.items()}
            return {"queue_max_bytes": self.max_bytes, "peak_bytes": self._peak_bytes,
                    "retries": self.retries, "keys": keys}
