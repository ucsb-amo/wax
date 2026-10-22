"""liveOD's "Run again next" (waxx.util.live_od.gui.run_again_button): the last
run that came through the run queue, queued again as the next job.

No network: the button gets a fake RunQueueClient; the windows are built with
their servers' ``start`` patched out (liveod_qt_helpers, as the other liveOD
window tests) and never call ``RunAgainButton.start``.  Nothing is discovered.
"""
import hashlib
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

import liveod_qt_helpers as qt
# the acquisition window, built as test_liveod_shutdown builds it
from test_liveod_shutdown import app, window  # noqa: F401  (fixtures)
from test_liveod_viewer_parity import INFO, viewer  # noqa: F401  (fixture)

from waxx.util.device_state.run_queue_client import (
    NoRunQueue, RunQueueClient, RunQueueError)


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture
def expt_file(tmp_path):
    path = tmp_path / "hf_bec.py"
    path.write_text("class HfBec: pass\n")
    return path


def _record(path, **kw):
    rec = {"id": 41, "token": "abc", "path": str(path), "sha256": _sha(path),
           "label": "hf_bec", "argv": ["--fast"], "cwd": str(os.path.dirname(path)),
           "owner": "agent", "state": "saved", "expt_class": "HfBec", "write_back": False,
           "allow_drift": False, "after": [], "chain": None, "repeat_of": 1, "due": None,
           "paused": False}
    rec.update(kw)
    return rec


class FakeClient:
    """RunQueueClient's surface the button uses."""

    def __init__(self, record=None):
        self.record = record
        self.answer = True
        self.no_queue = False
        self.refuse = None
        self.describe_error = None
        self.calls = []
        self.submits = []
        self.next_id = 57
        self.gate = None           # a threading.Event: submit waits on it
        self.status_gate = None

    def status(self):
        self.calls.append("status")
        if self.status_gate is not None:
            self.status_gate.wait(10)
        if self.no_queue:
            raise NoRunQueue("the monitor server found has no run queue (an older version)")
        if not self.answer:
            raise RunQueueError("the monitor server did not answer status_json")
        return {"run_queue": {"state": "idle"}}

    def describe(self, job_id, token=None):
        self.calls.append(("describe", job_id, token))
        if self.describe_error:
            raise RunQueueError(self.describe_error, {"status": "error"})
        return {"job": dict(self.record), "waiting": "", "tail": []}

    def submit(self, path, **kw):
        if self.gate is not None:
            self.gate.wait(10)
        self.submits.append((path, kw))
        if self.refuse:
            raise RunQueueError(self.refuse, {"status": "error", "msg": self.refuse})
        self.next_id += 1
        return {"status": "ok", "ids": [self.next_id], "jobs": []}


def _button(client, *, synchronous=True, confirm=True, log=None, start=True):
    from waxx.util.live_od.gui.run_again_button import RunAgainButton
    asked = []

    def ask(title, text):
        asked.append((title, text))
        return confirm
    b = RunAgainButton(client=client, synchronous=synchronous, log=log, confirm=ask,
                       by="jp@kong (liveOD)")
    b.asked = asked
    if start:
        b.start()
        b.timer.stop()            # polls are made by hand here
    return b


def _wait(app, cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return cond()


# -- pure helpers ---------------------------------------------------------------------

def test_parse_queue_job():
    from waxx.util.live_od.gui.run_again_button import parse_queue_job
    assert parse_queue_job("41:abc") == (41, "abc")
    assert parse_queue_job("41") == (41, "")
    for bad in ("", None, "x:y", "0:abc", "-3:a"):
        assert parse_queue_job(bad) is None


def test_the_request_is_the_queue_panels_run_again_next(expt_file):
    """What goes on the wire through RunQueueClient.submit equals the panel's
    again_request(first=True) on every field it sets."""
    from waxx.util.guis.run_queue_panel import again_request
    from waxx.util.live_od.gui.run_again_button import again_kwargs

    sent = []

    class Transport:
        def request(self, obj, timeout=None, attempts=1):
            sent.append(dict(obj))
            return {"status": "ok", "ids": [58]}

        def get_status(self):
            return {"run_queue": {}}
    rec = _record(expt_file)
    client = RunQueueClient(Transport(), by="jp@kong (liveOD)", owner="person")
    client.submit(rec["path"], **again_kwargs(rec))
    import socket
    panel = again_request(rec, "jp@kong (liveOD)", first=True,
                          client_sha256=_sha(expt_file), client_host=socket.gethostname())
    got = sent[0]
    assert got["action"] == "insert" and got["at_index"] == 0 and got["owner"] == "person"
    for key, value in panel.items():
        assert got[key] == value, key
    # nothing the panel leaves behind is carried: no dependencies, chain, due, repeats
    assert got["after"] == [] and got["repeat"] == 1
    assert "chain" not in got and "due" not in got


# -- enabled / disabled ---------------------------------------------------------------

def test_disabled_until_a_queued_run_is_known(app, expt_file):
    client = FakeClient(_record(expt_file))
    b = _button(client)
    assert not b.isEnabled() and "No run has come through the run queue" in b.toolTip()
    b.note_run(86001, "auto_tof.py", "", "run_loop")
    assert not b.isEnabled()
    assert "direct run" in b.toolTip() and "run 86001" in b.toolTip()
    assert "run_loop" in b.toolTip()
    b.note_run(86002, "x.py", None)                  # an older liveOD server did not say
    assert not b.isEnabled() and "older liveOD server" in b.toolTip()
    assert not any(c[0] == "describe" for c in client.calls if isinstance(c, tuple))
    b.shutdown()


def test_enabled_for_a_queued_run_naming_the_file_and_job(app, expt_file):
    client = FakeClient(_record(expt_file))
    b = _button(client)
    b.note_run(86003, "hf_bec.py", "41:abc", "kq")
    assert ("describe", 41, "abc") in client.calls
    assert b.isEnabled()
    tip = b.toolTip()
    assert "job 41 (hf_bec)" in tip and str(expt_file) in tip and "run 86003" in tip
    assert "on disk NOW" in tip and "CHANGED" not in tip
    assert "NEXT job" in tip
    b.shutdown()
    assert not b.isEnabled()


def test_queue_unreachable_or_absent_disables_it(app, expt_file):
    client = FakeClient(_record(expt_file))
    b = _button(client)
    b.note_run(86003, "hf_bec.py", "41:abc", "kq")
    assert b.isEnabled()
    client.answer = False
    b.poll()
    assert not b.isEnabled() and "not answering" in b.toolTip()
    client.answer, client.no_queue = True, True
    b.poll()
    assert not b.isEnabled() and "no run queue" in b.toolTip()
    client.no_queue = False
    b.poll()
    assert b.isEnabled()
    b.shutdown()


def test_a_job_the_queue_cannot_describe_is_disabled(app, expt_file):
    client = FakeClient(_record(expt_file))
    client.describe_error = "no job 41 in the run queue"
    b = _button(client)
    b.note_run(86003, "hf_bec.py", "41:abc", "kq")
    assert not b.isEnabled() and "no job 41 in the run queue" in b.toolTip()
    b.shutdown()


def test_not_started_is_disabled(app, expt_file):
    b = _button(FakeClient(_record(expt_file)), start=False)
    b.note_run(86003, "hf_bec.py", "41:abc", "kq")
    assert not b.isEnabled() and "Not connected to the run queue" in b.toolTip()
    b.shutdown()


def test_a_changed_or_unreadable_file_is_named(app, expt_file):
    client = FakeClient(_record(expt_file))
    b = _button(client)
    b.note_run(86003, "hf_bec.py", "41:abc", "kq")
    expt_file.write_text("class HfBec: x = 1\n")
    b.poll()                                          # the file is hashed again each poll
    assert b.isEnabled() and "CHANGED since job 41 was submitted" in b.toolTip()
    os.remove(expt_file)
    b.poll()
    assert "cannot read the file" in b.toolTip()
    b.shutdown()


def test_a_later_direct_run_keeps_the_queued_one_and_says_so(app, expt_file):
    client = FakeClient(_record(expt_file))
    b = _button(client)
    b.note_run(86003, "hf_bec.py", "41:abc", "kq")
    b.note_run(86004, "auto_tof.py", "", "run_loop")
    assert b.isEnabled() and b.job_ref == (41, "abc")
    assert "latest run (run 86004, auto_tof.py) was a direct run" in b.toolTip()
    b.shutdown()


# -- pressing it ------------------------------------------------------------------------

def test_one_press_sends_one_insert_next_as_a_person(app, expt_file):
    lines = []
    results = []
    client = FakeClient(_record(expt_file))
    b = _button(client, log=lambda text, level=20: lines.append((level, text)))
    b.result_ready.connect(lambda text, ok: results.append((text, ok)))
    b.note_run(86003, "hf_bec.py", "41:abc", "kq")
    b.click()
    assert len(client.submits) == 1
    path, kw = client.submits[0]
    assert path == str(expt_file)
    assert kw["at_index"] == 0 and kw["owner"] == "person"
    assert kw["argv"] == ["--fast"] and kw["label"] == "hf_bec" and kw["write_back"] is False
    assert "job 41" in b.asked[0][1] and str(expt_file) in b.asked[0][1]
    assert "hf_bec.py" in b.asked[0][0]
    assert results == [(results[0][0], True)] and "re-submitted as job 58" in results[0][0]
    assert any("re-submitted as job 58" in t for _, t in lines)
    assert b.isEnabled() and "queued as job 58" in b.toolTip()
    b.shutdown()


def test_declining_the_confirmation_sends_nothing(app, expt_file):
    client = FakeClient(_record(expt_file))
    b = _button(client, confirm=False)
    b.note_run(86003, "hf_bec.py", "41:abc", "kq")
    assert b.resubmit() is False
    assert client.submits == [] and b.isEnabled() and not b.busy
    b.shutdown()


def test_a_refusal_is_shown(app, expt_file):
    lines, results = [], []
    client = FakeClient(_record(expt_file))
    client.refuse = "the file differs from the copy on the queue machine"
    b = _button(client, log=lambda text, level=20: lines.append((level, text)))
    b.result_ready.connect(lambda text, ok: results.append((text, ok)))
    b.note_run(86003, "hf_bec.py", "41:abc", "kq")
    b.click()
    assert results and results[0][1] is False and "differs" in results[0][0]
    assert lines[-1][0] == 40 and "refused job 41" in lines[-1][1]
    assert b.isEnabled() and "refused" in b.toolTip()
    b.shutdown()


def test_a_changed_file_is_reported_with_the_result(app, expt_file):
    lines = []
    client = FakeClient(_record(expt_file))
    b = _button(client, log=lambda text, level=20: lines.append((level, text)))
    b.note_run(86003, "hf_bec.py", "41:abc", "kq")
    expt_file.write_text("class HfBec: x = 2\n")
    b.click()
    assert lines[-1][0] == 30 and "CHANGED" in lines[-1][1]
    b.shutdown()


def test_the_press_never_blocks_and_a_second_press_sends_nothing(app, expt_file):
    client = FakeClient(_record(expt_file))
    b = _button(client, synchronous=False)
    b.note_run(86003, "hf_bec.py", "41:abc", "kq")
    assert _wait(app, b.isEnabled)
    client.gate = threading.Event()                   # the queue takes its time
    t0 = time.monotonic()
    assert b.resubmit() is True
    assert time.monotonic() - t0 < 1.0                # returned at once: the call is on a worker
    assert b.busy and not b.isEnabled()
    assert b.resubmit() is False                      # a double click
    b.click()                                         # disabled: no click at all
    client.gate.set()
    assert _wait(app, lambda: not b.busy)
    assert len(client.submits) == 1
    assert b.isEnabled()
    b.shutdown()


def test_a_slow_queue_does_not_block_the_poll(app, expt_file):
    client = FakeClient(_record(expt_file))
    client.status_gate = threading.Event()
    b = _button(client, synchronous=False)
    t0 = time.monotonic()
    b.note_run(86003, "hf_bec.py", "41:abc", "kq")    # polls at once
    assert time.monotonic() - t0 < 1.0
    assert not b.isEnabled() and "Asking the run queue" in b.toolTip()
    client.status_gate.set()
    assert _wait(app, b.isEnabled)
    b.shutdown()


# -- the windows ------------------------------------------------------------------------

def test_the_viewer_follows_run_started_and_catch_up(viewer):
    win, _sent, answers = viewer
    b = win.run_again_button
    assert not b.isEnabled() and b.runner is None     # never started in tests
    win._on_run_started({"run_id": 86010, "xvarnames": [],
                         "info": dict(INFO, queue_job="41:abc", launcher="kq")})
    assert b.job_ref == (41, "abc") and b.latest["kind"] == "queued"
    win._on_run_started({"run_id": 86011, "xvarnames": [],
                         "info": dict(INFO, queue_job="", launcher="")})
    assert b.job_ref == (41, "abc") and b.latest["kind"] == "direct"
    win._on_run_started({"run_id": 86012, "xvarnames": [], "info": INFO})   # older server
    assert b.latest["kind"] == "unknown"
    win._on_run_again_result("Run again next: the run queue refused job 41: no", False)
    assert "refused job 41" in win.status_strip.notice()


def test_a_late_viewer_learns_the_queue_job_from_poll(viewer):
    win, _sent, answers = viewer
    answers["POLL"] = {"ok": True, "run_in_progress": False, "run_id": 86020,
                       "run_state": "saved", "expt_name": "hf_bec.py",
                       "init_run_age_s": 30.0, "queue_job": "42:def", "launcher": "kq"}
    win._catch_up()
    assert win.run_again_button.job_ref == (42, "def")


def test_a_late_viewer_before_any_run_knows_nothing(viewer):
    win, _sent, answers = viewer
    answers["POLL"] = {"ok": True, "run_in_progress": False, "run_id": 0,
                       "run_state": "idle", "init_run_age_s": None, "queue_job": "",
                       "launcher": ""}
    win._catch_up()
    assert win.run_again_button.latest is None
    assert "No run has come through" in win.run_again_button.toolTip()


def test_the_acquisition_window_notes_the_run_and_tells_the_viewers(window):
    srv = window.live_od_server
    srv._current_run_id = 86030
    srv._current_expt_name = "hf_bec.py"
    srv._current_client = {"client_pid": 1, "client_host": "kong", "client_started": None,
                           "launcher": "kq", "queue_job": "43:ghi"}
    told = []
    window.broadcaster.broadcast_run_started = lambda run_id, xv, info=None: told.append(info)
    window.spawn_baby(filepath="", camera_key="", capture_images=False, save_data=True,
                      imaging_type=0, run_info_payload={"xvarnames": []})
    b = window.run_again_button
    assert b.job_ref == (43, "ghi") and b.job_run["run_id"] == 86030
    assert told and told[0]["queue_job"] == "43:ghi" and told[0]["launcher"] == "kq"
    srv._current_client = dict(srv._current_client, queue_job="", launcher="")
    srv._current_run_id = 86031
    window.spawn_baby(filepath="", camera_key="", capture_images=False, save_data=True,
                      imaging_type=0, run_info_payload={"xvarnames": []})
    assert b.latest["kind"] == "direct" and b.job_ref == (43, "ghi")
    assert told[-1]["queue_job"] == ""
    window._on_run_again_result("Run again next: the run queue refused job 43: no", False)
    assert "refused job 43" in window.status_strip.notice()
