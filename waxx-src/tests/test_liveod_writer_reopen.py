"""The run file's writer survives a dead handle (image_writer, 2026-10-10; runs 86095,
86182): close, reopen, write again what was not flushed, check what was, record it,
and when reopening fails say so at once and keep the file.

The handle is a real h5py file in a temp folder behind a proxy that fails every access
with the errno-22 error of 86095 / 86182 once it "dies" (after N accesses, or at the
k-th flush). A dead handle's close puts back the file as it was at its last flush, so
what was not flushed is really lost and must be written again. SaveWorker.run() runs
on the test's thread. No liveOD, no socket, no beacon, no camera; temp files only."""
import json
import logging
import os
import shutil
import subprocess
import threading
from queue import Queue

import h5py
import numpy as np
import pytest

import liveod_qt_helpers as qt
from live_od_data_fakes import FakeSaver, new_data_file, patch_payload_stash
from waxx.util.live_od.data import image_writer as iw
from waxx.util.live_od.data.image_writer import PutItem, SaveWorker

ERR22 = ("Unable to synchronously open object (file read failed: time = Sat Oct 10 21:46:38 "
         "2026\n, filename = 'run.hdf5', file descriptor = 5, errno = 22, error message = "
         "'Invalid argument', buf = 0, total read size = 512, bytes this sub-read = 512, offset = 800)")


@pytest.fixture
def app():
    return qt.session_app()


class World:
    """The file system under the writer: opens, deaths, flush snapshots."""

    def __init__(self, path):
        self.path = path
        self.snap = path + ".snap"
        self.dead = False
        self.die_after = None        # accesses after an open
        self.die_at_flush = None     # the k-th flush after an open fails (and kills)
        self.accesses = 0
        self.flushes = 0
        self.total_flushes = 0
        self.open_fails = []         # exceptions raised by the next opens, in turn
        self.opens = 0
        self.close_raises_keep_fd = False
        self.held = []               # real handles a failed close left open
        self.release_held_on_open = None   # open number at which the stuck fd goes away
        self.flush_advance = 0.0     # seconds the fake clock moves during a flush
        self.clock = Clock()
        self.tamper = None           # called on the file between a death and the reopen
        self.partial = False         # cached lookups survive the death (86182)
        self.partial_exc = KeyError(ERR22)

    def open(self, **kw):
        self.opens += 1
        if self.held and self.release_held_on_open == self.opens:
            for real in self.held:
                real.close()
            self.held = []
            shutil.copyfile(self.snap, self.path)       # what the dead handle had flushed
        if self.open_fails and self.opens > 1:          # (reopens only)
            raise self.open_fails.pop(0)
        if self.held:
            raise OSError("Unable to open file (file is already open for write)")
        if self.tamper is not None and self.opens > 1:   # (between a death and the reopen)
            self.tamper(self.path)
            self.tamper = None
        real = h5py.File(self.path, "r+")
        self.dead, self.accesses, self.flushes = False, 0, 0
        self.snapshot(real)
        return Proxy(real, self, root=True)

    def snapshot(self, real):
        """The file as flushed, kept aside (a copy made through h5py: Windows does
        not let another reader copy a file h5py holds open)."""
        real.flush()
        with h5py.File(self.snap, "w") as s:
            for k in real.keys():
                real.copy(real[k], s, name=k)
            for a, v in real.attrs.items():
                s.attrs[a] = v

    def tick(self, cached=False):
        """One access. ``partial``: once dead, what HDF5 answers from its metadata
        cache (``'x' in group``, the root's ``data`` group, attributes) still works,
        and only real reads / writes fail -- 86182's signature."""
        if self.dead:
            if self.partial and cached:
                return
            raise self.partial_exc if self.partial else KeyError(ERR22)
        self.accesses += 1
        if self.die_after is not None and self.accesses > self.die_after:
            self.die_after = None
            self.dead = True
            if self.partial and cached:
                return
            raise self.partial_exc if self.partial else KeyError(ERR22)

    def flush(self, real):
        self.tick()
        self.flushes += 1
        if self.die_at_flush is not None and self.flushes >= self.die_at_flush:
            self.die_at_flush = None
            self.dead = True
            raise OSError(22, ERR22)
        self.clock.t += self.flush_advance
        self.snapshot(real)
        self.total_flushes += 1

    def close(self, real):
        if not self.dead:
            real.close()
            return
        if self.close_raises_keep_fd:
            self.held.append(real)                       # the OS handle stays
            raise OSError(22, "file close failed: Invalid argument")
        real.close()
        shutil.copyfile(self.snap, self.path)            # only what was flushed survives


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _wrap(obj, world, cached=False):
    if isinstance(obj, (h5py.Group, h5py.Dataset, h5py.AttributeManager)):
        return Proxy(obj, world, cached=cached)
    return obj


class Proxy:
    """``cached``: an object whose accesses HDF5 can answer from its metadata
    cache (the root group's ``data`` group, attributes)."""

    def __init__(self, obj, world, root=False, cached=False):
        object.__setattr__(self, "_o", obj)
        object.__setattr__(self, "_w", world)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_cached", cached)

    def __getitem__(self, k):
        cached = self._cached or (self._root and k == "data")
        self._w.tick(cached=cached)
        return _wrap(self._o[k], self._w, cached=False)

    def __setitem__(self, k, v):
        self._w.tick()                                   # a write: never from the cache
        self._o[k] = v

    def __contains__(self, k):
        self._w.tick(cached=True)
        return k in self._o

    def flush(self):
        self._w.flush(self._o)

    def close(self):
        self._w.close(self._o)

    def __getattr__(self, name):
        if name == "attrs":
            self._w.tick(cached=True)
            return Proxy(self._o.attrs, self._w, cached=True)
        attr = getattr(self._o, name)
        if callable(attr):
            def call(*a, **k):
                self._w.tick(cached=self._cached)
                return _wrap(attr(*a, **k), self._w)
            return call
        self._w.tick(cached=self._cached)
        return attr


def new_file(tmp_path, name="run.hdf5"):
    path = str(tmp_path / name)
    with h5py.File(path, "x") as f:
        f.create_group("data")
        f.attrs["images_shape"] = (6, 4, 5)
        f.attrs["images_dtype"] = "uint16"
        f["data"].create_dataset("pre", shape=(6, 3), dtype=np.float64, fillvalue=np.nan)
    return path


def frames(n=6):
    return [np.full((4, 5), 10 * (i + 1), np.uint16) for i in range(n)]


def items(n=6):
    """A run's queue as the camera and the experiment fill it: a frame, then the
    shot's pushed arrays (a diag frame made on first use, a pre-allocated container)."""
    out = []
    for i, fr in enumerate(frames(n)):
        out.append((fr, i, float(i)))
        out.append(PutItem("img_2dmot", (i,), np.full((2, 3), i + 1, np.uint8),
                           full_shape=(n, 2, 3), fill=0))
        out.append(PutItem("pre", (i,), np.arange(3, dtype=np.float64) + i))
    return out


def run_worker(world, queue_items, sleeps=None, **kw):
    q = Queue()
    for it in queue_items:
        q.put(it)
    q.put(None)
    sleeps = [] if sleeps is None else sleeps
    w = SaveWorker(q, wait_fn=lambda timeout, check_interrupt_method: world.open(),
                   n_img=6, reopen_fn=world.open, sleep=sleeps.append,
                   clock=world.clock, wall=lambda: 1_791_700_000.0, **kw)
    faults, failed = [], []
    w.writer_fault_signal.connect(faults.append)      # emitted on this thread: direct
    w.save_failed_signal.connect(failed.append)
    w.run()
    return w, faults, failed


def check_all_written(path, n=6):
    with h5py.File(path, "r") as f:
        for i, fr in enumerate(frames(n)):
            assert np.array_equal(f["data/images"][i], fr), i
            assert np.all(f["data/img_2dmot"][i] == i + 1), i
            assert np.array_equal(f["data/pre"][i], np.arange(3) + i), i


# -- no fault: nothing changes but the flushes ---------------------------------------------

def test_a_run_without_a_fault_is_written_as_before(app, tmp_path):
    path = new_file(tmp_path)
    world = World(path)
    w, faults, failed = run_worker(world, items())
    check_all_written(path)
    written, lost, mism = w.report.snapshot()
    assert written == 6 and lost == [] and mism == []
    pw, plost, created = w.put_report.snapshot()
    assert pw == {"img_2dmot": 6, "pre": 6} and plost == [] and created == ["img_2dmot"]
    assert w.reopens == [] and w.fault == "" and w.problems == [] and not faults and not failed
    assert world.total_flushes >= 2                     # after the creations, and at the end
    with h5py.File(path, "r") as f:
        assert "writer_reopens" not in f.attrs


# -- a dead handle, reopened ----------------------------------------------------------------

@pytest.mark.parametrize("die_after", [30, 41, 57])
def test_dead_handle_is_reopened_and_nothing_is_lost(app, tmp_path, die_after):
    path = new_file(tmp_path)
    world = World(path)
    world.die_after = die_after
    sleeps = []
    w, faults, failed = run_worker(world, items(), sleeps)
    assert len(w.reopens) == 1 and w.reopens[0]["outcome"] == "reopened"
    rec = w.reopens[0]
    assert "errno = 22" in rec["error"] and rec["attempts"] == 1 and rec["verify"] == "ok"
    assert rec["rewritten"] == rec["pending"]
    check_all_written(path)                              # the unflushed items were written again
    written, lost, _ = w.report.snapshot()
    pw, plost, _ = w.put_report.snapshot()
    assert written == 6 and lost == [] and pw == {"img_2dmot": 6, "pre": 6} and plost == []
    assert w.fault == "" and not faults and not failed and w.problems == []
    with h5py.File(path, "r") as f:
        stored = json.loads(f.attrs["writer_reopens"])
    assert stored["reopens"][0]["outcome"] == "reopened"


def test_unflushed_items_really_are_lost_without_the_rewrite(app, tmp_path, monkeypatch):
    """The fake is honest: with the rewrite switched off, the rolled-back slots are
    empty in the file (so the passing test above means something)."""
    path = new_file(tmp_path)
    world = World(path)
    world.die_after = 57
    orig = SaveWorker._reopen

    def reopen_no_rewrite(self, what, exc, waits=iw.REOPEN_WAITS_S):
        self._pending = []                               # drop them instead
        return orig(self, what, exc, waits)
    monkeypatch.setattr(SaveWorker, "_reopen", reopen_no_rewrite)
    run_worker(world, items())
    with h5py.File(path, "r") as f:
        images = f["data/images"][...]
    assert any(not images[i].any() for i in range(6))


# -- a dead handle whose cached lookups still answer (86182) ---------------------------------

@pytest.mark.parametrize("die_after", [30, 41, 57])
def test_cached_probe_passing_on_a_dead_handle_still_reopens(app, tmp_path, die_after, monkeypatch):
    """86182's signature: 'x' in group and attributes answer from HDF5's cache,
    dataset lookups and writes fail with "file read failed ... errno = 22". The
    probe passes; the error's I/O text still marks the handle dead."""
    path = new_file(tmp_path)
    world = World(path)
    world.partial = True
    world.die_after = die_after
    probes = []
    orig_alive = SaveWorker._alive

    def alive(f):
        ok = orig_alive(f)
        probes.append(ok)
        return ok
    monkeypatch.setattr(SaveWorker, "_alive", staticmethod(alive))
    w, faults, failed = run_worker(world, items())
    assert len(w.reopens) == 1 and w.reopens[0]["outcome"] == "reopened"
    assert "file read failed" in w.reopens[0]["error"]
    assert True in probes                               # the probe passed on the dead handle
    check_all_written(path)
    written, lost, _ = w.report.snapshot()
    pw, plost, _ = w.put_report.snapshot()
    assert written == 6 and lost == [] and pw == {"img_2dmot": 6, "pre": 6} and plost == []
    assert not faults and not failed


def test_with_the_probe_alone_the_cached_case_loses_everything_after_it(app, tmp_path, monkeypatch):
    """The 'before' of the fix above (6260011's rule: the probe alone): every item
    after the death is lost as the item's own error until the next flush fails and
    forces a reopen (here: the end of the queue; in a run: the next idle queue,
    32 items or 5 s -- about one shot). Which is why the error's own text decides."""
    monkeypatch.setattr(iw, "_io_error", lambda exc: False)
    monkeypatch.setattr(iw, "MAX_ITEM_ERRORS_IN_A_ROW", 10 ** 6)
    path = new_file(tmp_path)
    world = World(path)
    world.partial = True
    world.die_after = 41
    w, faults, failed = run_worker(world, items())
    assert [r["item"] for r in w.reopens] == ["the flush"]    # only the flush noticed
    lost = w.report.snapshot()[1] + w.put_report.snapshot()[1]
    assert len(lost) >= 10


def test_the_cached_probe_really_passes_on_the_partly_dead_handle(tmp_path):
    """(so the test above is the cached-probe case, not the old one)"""
    path = new_file(tmp_path)
    world = World(path)
    world.partial = True
    f = world.open()
    world.dead = True
    assert SaveWorker._alive(f) is True
    with pytest.raises(KeyError, match="file read failed"):
        f["data"]["pre"]
    world.dead = False
    f.close()


def test_errors_without_io_text_on_a_passing_probe_reopen_after_three(app, tmp_path):
    """A dead handle whose errors carry no I/O text: the third item in a row that
    fails while the probe answers forces a reopen, and the held items are written
    again -- nothing lost."""
    path = new_file(tmp_path)
    world = World(path)
    world.partial = True
    world.partial_exc = RuntimeError("something odd happened")
    world.die_after = 41
    w, faults, failed = run_worker(world, items())
    assert len(w.reopens) == 1 and w.reopens[0]["outcome"] == "reopened"
    assert w.reopens[0]["retried"] == iw.MAX_ITEM_ERRORS_IN_A_ROW
    check_all_written(path)
    assert w.report.snapshot()[1] == [] and w.put_report.snapshot()[1] == []


def test_an_item_error_on_a_healthy_handle_does_not_reopen(app, tmp_path):
    path = new_file(tmp_path)
    world = World(path)
    bad = PutItem("nowhere", (0,), np.zeros(3))          # no such dataset, no full_shape
    w, faults, failed = run_worker(world, items()[:3] + [bad] + items()[3:])
    assert w.reopens == [] and not faults
    pw, plost, _ = w.put_report.snapshot()
    assert len(plost) == 1 and "PutItem(nowhere[0])" in plost[0] and "ValueError" in plost[0]
    check_all_written(path)


def test_a_lone_odd_error_on_a_healthy_handle_stays_the_items_own(app, tmp_path, monkeypatch):
    """A non-ValueError error once, then writes go on: reported as the item's own
    error (held until the next write showed the handle is fine), no reopen."""
    path = new_file(tmp_path)
    world = World(path)
    orig = SaveWorker._write_put
    n = {"k": 0}

    def write_put(self, f, item):
        if item.key == "pre" and item.index == (2,):
            raise RuntimeError("one-off")
        return orig(self, f, item)
    monkeypatch.setattr(SaveWorker, "_write_put", write_put)
    w, faults, failed = run_worker(world, items())
    assert w.reopens == []
    plost = w.put_report.snapshot()[1]
    assert len(plost) == 1 and "pre[2]" in plost[0] and "one-off" in plost[0]


def test_io_error_classification():
    assert iw._io_error(OSError(22, "x"))
    assert iw._io_error(KeyError(ERR22))
    assert iw._io_error(RuntimeError("Can't write data (file write failed)"))
    inner = OSError(9, "Bad file descriptor")
    try:
        try:
            raise inner
        except OSError as e:
            raise KeyError("Unable to synchronously open object") from e
    except KeyError as outer:
        assert iw._io_error(outer)
    assert not iw._io_error(ValueError("no such dataset and no full_shape to create it"))
    assert not iw._io_error(KeyError("Unable to synchronously open object (object 'x' doesn't exist)"))


# -- reopening fails ------------------------------------------------------------------------

def test_reopen_failing_sets_the_fault_and_keeps_the_file(app, tmp_path, caplog):
    path = new_file(tmp_path)
    world = World(path)
    world.die_after = 41
    world.open_fails = [OSError("Unable to open file (errno 22)")] * 50
    sleeps = []
    caplog.set_level(logging.ERROR)
    w, faults, failed = run_worker(world, items(), sleeps)
    assert sleeps[:2] == list(iw.REOPEN_WAITS_S[1:])     # 0 s, 5 s, 15 s before the tries
    assert w.reopens[0]["outcome"].startswith("failed") and w.reopens[0]["attempts"] == 3
    assert w.fault and faults == [w.fault] and failed == []    # NOT save_failed: no reset
    assert os.path.exists(path)                          # the file is kept
    written, lost, _ = w.report.snapshot()
    pw, plost, _ = w.put_report.snapshot()
    assert written + len([l for l in lost]) == 6 and lost
    assert any("could not be reopened" in l or "not flushed" in l for l in lost + plost)
    with h5py.File(path, "r") as f:                      # what was flushed before is there
        for i in range(written):
            assert np.array_equal(f["data/images"][i], frames()[i])
    assert any("KEPT" in r.message for r in caplog.records)


def test_a_later_try_recovers_after_a_failed_reopen(app, tmp_path, monkeypatch):
    path = new_file(tmp_path)
    world = World(path)
    world.die_after = 30
    world.open_fails = [OSError("busy")] * 3            # the first reopen's three tries
    q_items = items()
    sleeps = []

    def advance(s):
        sleeps.append(s)
    # every item moves the clock 20 s: the retry (every 60 s) comes a few items later
    orig_take = SaveWorker._take

    def take(self, item):
        world.clock.t += 20.0
        return orig_take(self, item)
    monkeypatch.setattr(SaveWorker, "_take", take)
    w, faults, failed = run_worker(world, q_items, sleeps)
    assert [r["outcome"].split(":")[0] for r in w.reopens] == ["failed", "reopened"]
    assert "reopened at" in w.fault and faults
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["data/images"][5], frames()[5])   # written after the recovery


# -- (a) the old handle's close fails and its fd stays open ---------------------------------

def test_old_handle_close_fails_and_the_first_reopen_is_refused(app, tmp_path):
    path = new_file(tmp_path)
    world = World(path)
    world.die_after = 41
    world.close_raises_keep_fd = True
    world.release_held_on_open = 3                       # the stuck fd is gone by the 2nd reopen try
    sleeps = []
    w, faults, failed = run_worker(world, items(), sleeps)
    rec = w.reopens[0]
    assert rec["outcome"] == "reopened" and rec["attempts"] == 2
    assert "close failed" in rec["close_error"]
    check_all_written(path)
    assert not faults and not failed


def test_old_fd_never_released_is_a_fault(app, tmp_path):
    path = new_file(tmp_path)
    world = World(path)
    world.die_after = 41
    world.close_raises_keep_fd = True
    w, faults, failed = run_worker(world, items())
    assert w.fault and faults and not failed
    for real in world.held:
        real.close()


# -- (b) a dataset made since the last flush is made again ----------------------------------

def test_dataset_created_since_the_last_flush_is_made_again(app, tmp_path):
    path = new_file(tmp_path)
    world = World(path)
    world.die_at_flush = 2       # 1st flush: after 'images'; 2nd: right after 'img_2dmot' is made
    w, faults, failed = run_worker(world, items())
    rec = w.reopens[0]
    assert rec["outcome"] == "reopened" and "img_2dmot" in rec["recreated"]
    check_all_written(path)
    with h5py.File(path, "r") as f:
        assert f["data/img_2dmot"].fillvalue == 0 and f["data/img_2dmot"].shape == (6, 2, 3)


# -- the check after a reopen ---------------------------------------------------------------

def test_committed_data_that_reads_back_wrong_makes_the_run_incomplete(app, tmp_path):
    path = new_file(tmp_path)
    world = World(path)
    world.die_after = 57

    def tamper(p):
        with h5py.File(p, "r+") as f:
            f["data/images"][0] = 7                      # frame 0 was flushed before the death
    world.tamper = tamper
    w, faults, failed = run_worker(world, items())
    assert w.reopens[0]["outcome"] == "reopened" and w.reopens[0]["verify"] != "ok"
    assert w.problems and "data/images" in w.problems[0] and "reads back different" in w.problems[0]


# -- (c) flush cost ---------------------------------------------------------------------------

def test_a_slow_flush_warns_once(app, tmp_path, caplog):
    path = new_file(tmp_path)
    world = World(path)
    world.flush_advance = 1.5
    caplog.set_level(logging.DEBUG)
    run_worker(world, items())
    slow = [r for r in caplog.records if "flush of the run file took" in r.message]
    assert len(slow) == 1 and slow[0].levelno == logging.WARNING
    assert any(r.levelno == logging.DEBUG and "SaveWorker: flush" in r.message for r in caplog.records)


# -- END_RUN: reasons and the record ---------------------------------------------------------

class _Writer:
    def __init__(self, fault="", problems=(), records=""):
        self.fault, self.problems, self._records = fault, list(problems), records
        self.put_report = iw.PutReport()

    def finish(self, interrupted=False):
        pass

    def records_json(self):
        return self._records


def test_run_file_save_reports_and_records_a_reopen(app, tmp_path, monkeypatch):
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    saver = FakeSaver(tmp_path)
    for sub in "abc":
        (tmp_path / sub).mkdir()
    _, path = new_data_file(tmp_path / "a")      # temp files made by the test fakes
    rf = run_file.RunFile(saver)
    rf.save_data, rf.filepath = True, path
    rec = json.dumps({"reopens": [{"outcome": "failed: busy"}], "more": 0, "fault": "f", "problems": []})
    rf.writer = _Writer(fault="the run file's handle failed at 21:46:38", records=rec)
    incomplete = rf.save({}, 101, [])
    assert "image writer: the run file's handle failed" in incomplete["reason"]
    assert saver.incomplete == incomplete
    with h5py.File(path, "r") as f:
        assert json.loads(f.attrs["writer_reopens"])["reopens"][0]["outcome"] == "failed: busy"
    # a clean reopen: recorded, and the run is not marked incomplete for it
    _, path2 = new_data_file(tmp_path / "b")
    rf2 = run_file.RunFile(saver)
    rf2.save_data, rf2.filepath = True, path2
    rf2.writer = _Writer(records=json.dumps({"reopens": [{"outcome": "reopened", "verify": "ok"}]}))
    assert rf2.save({}, 102, []) is None
    with h5py.File(path2, "r") as f:
        assert "writer_reopens" in f.attrs
    # no reopen: no attribute (as before)
    _, path3 = new_data_file(tmp_path / "c")
    rf3 = run_file.RunFile(saver)
    rf3.save_data, rf3.filepath = True, path3
    rf3.writer = _Writer()
    assert rf3.save({}, 103, []) is None
    with h5py.File(path3, "r") as f:
        assert "writer_reopens" not in f.attrs


# -- (d) server and client, old and new ------------------------------------------------------

@pytest.fixture
def server(app, tmp_path, monkeypatch):
    from waxx.util.live_od.data import run_file
    patch_payload_stash(monkeypatch, run_file, [])
    from waxx.util.live_od.live_od_server import LiveODServer
    return LiveODServer(server_talk=None, data_saver=FakeSaver(tmp_path))


def _init(srv):
    return srv._handle_init_run({"tag": "INIT_RUN", "save_data": True, "capture_images": False,
                                 "camera_key": "", "params": {"N_img": 3},
                                 "N_shots_with_repeats": 2, "expt_class": "reopen_test"})


def test_server_reports_writer_fault_at_shot_complete_and_poll(server):
    init = _init(server)
    msg = {"tag": "SHOT_COMPLETE", "run_token": init["run_token"], "shot_idx": 0,
           "N_shots_total": 2}
    reply = server._handle_shot_complete(dict(msg))
    assert "writer_fault" not in reply                   # no fault: the reply is as before
    assert server._handle_poll({})["writer_fault"] == ""
    server._run_file.writer._worker.fault = "the run file's handle failed at 21:46:38"
    reply = server._handle_shot_complete(dict(msg, shot_idx=1))
    assert reply["writer_fault"].startswith("the run file's handle failed")
    assert server._handle_poll({})["writer_fault"].startswith("the run file's handle failed")
    server._run_file.writer._worker.fault = ""
    server._run_file.finish_writer(interrupted=True)


def _client(module, reply):
    c = module.LiveODClient.__new__(module.LiveODClient)
    c._send_recv = lambda msg: dict(reply)
    c._for_this_run = lambda msg: msg
    c.last_reset_requested = False
    c.last_adjust_values = {}
    c._grab_failure_warned = False
    return c


def test_new_client_warns_once_and_accepts_an_older_server(capsys):
    from waxx.util.live_od import live_od_client as lc
    c = _client(lc, {"ok": True, "reset_requested": False, "adjust_values": {},
                     "writer_fault": "the run file's handle failed at 21:46:38"})
    c._writer_fault_warned = False
    assert c.shot_complete(0, 2, {}) is False
    assert c.shot_complete(1, 2, {}) is False
    out = capsys.readouterr().out
    assert out.count("THE RUN FILE CANNOT BE WRITTEN") == 1
    old_server = _client(lc, {"ok": True, "reset_requested": False, "adjust_values": {}})
    old_server._writer_fault_warned = False
    assert old_server.shot_complete(0, 2, {}) is False
    assert "CANNOT BE WRITTEN" not in capsys.readouterr().out


def test_an_older_client_ignores_writer_fault(tmp_path, capsys):
    """live_od_client.py as it is on wax main (before this change), against a reply
    and a POLL that carry writer_fault."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    src = subprocess.run(["git", "show", "1091b7b:waxx-src/waxx/util/live_od/live_od_client.py"],
                         cwd=here, capture_output=True, text=True, encoding="utf-8")
    if src.returncode != 0:
        pytest.skip("wax main's live_od_client.py not readable from git here")
    old_path = tmp_path / "old_live_od_client.py"
    old_path.write_text(src.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("old_live_od_client", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    assert "writer_fault" not in src.stdout
    c = _client(old, {"ok": True, "reset_requested": True, "adjust_values": {"a": 1.0},
                      "writer_fault": "the run file's handle failed"})
    assert c.shot_complete(0, 2, {}) is True and c.last_adjust_values == {"a": 1.0}
    assert "Traceback" not in capsys.readouterr().out
