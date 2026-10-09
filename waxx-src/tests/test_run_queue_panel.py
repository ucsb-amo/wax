"""The run queue panel (waxx.util.guis.run_queue_panel): the table built from
a list reply (phase-1 fields, and the phase-1b fields present or absent), the
exact request dicts its controls send (owner "person"), the cancel
question, controls disabled by an "unknown action" reply, the log window's
cursor, and the broadcast debounce.

Offscreen Qt.  The requester is a fake that records each request and answers
from a table; requests are answered synchronously.  No socket may be opened
in this module (socket.socket is replaced by one that fails the test), no
server is built, nothing beacons."""
import os
import socket
import time

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from waxx.util.guis import run_queue_panel as rqp
from waxx.util.guis.request_runner import RequestRunner, is_unknown_request, normalize_reply


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class _NoSocket:
    def __init__(self, *a, **k):
        raise AssertionError("the panel opened a socket")


@pytest.fixture(autouse=True)
def no_sockets(monkeypatch):
    monkeypatch.setattr(socket, "socket", _NoSocket)


class FakeServer:
    """Records requests; answers ``answers[action]`` (a dict or a callable of
    the request), else ok."""

    def __init__(self):
        self.requests = []
        self.answers = {}

    def __call__(self, obj):
        self.requests.append(dict(obj))
        answer = self.answers.get(obj.get("action") or obj.get("type"))
        if callable(answer):
            return answer(obj)
        if answer is not None:
            return answer
        return {"status": "ok"}

    def of(self, action):
        return [r for r in self.requests if r.get("action") == action]


NOW = time.time()


def _job(i, state="queued", **kw):
    job = {"id": i, "token": f"t{i}", "path": f"C:/code/kexp/experiments/JP/expt_{i}.py",
           "sha256": "ab" * 32, "label": f"expt_{i}", "argv": [], "cwd": "", "owner": "person",
           "priority": 10, "due": None, "after": [], "chain": None, "stop_on_failure": False,
           "write_back": None, "allow_drift": False, "repeat_index": 1, "repeat_of": 1,
           "submitted_at": NOW - 600, "submitted_by": "jp@kong", "state": state, "reason": "",
           "pid": None, "pid_started": None, "client_pid": None, "run_id": None,
           "log_path": None, "exit_code": None, "outcome": None, "launched_at": None,
           "ended_at": None, "cancel": None, "adopted": False}
    job.update(kw)
    return job


INFO = {"enabled": True, "state": "running", "text": "job 2 (expt_2) running, run 85600",
        "current": {"id": 2, "token": "t2", "label": "expt_2", "state": "running",
                    "run_id": 85600, "owner": "agent", "cancel": None},
        "next": [3], "waiting": "job 2 is in the slot",
        "counts": {"queued": 3, "launching": 0, "running": 1, "ending": 0, "saved": 1,
                   "failed": 0, "cancelled": 0, "skipped": 0},
        "alarm": None, "paused": {"agent": None, "all": None},
        "person_hold": {"active": False}, "resume_loop": None, "directory": "C:/q"}


def _phase1_jobs():
    return [_job(1, "saved", run_id=85599),
            _job(2, "running", run_id=85600, owner="agent"),
            _job(3),
            _job(4, due=NOW + 3600),
            _job(5, after=[3], chain="c1", stop_on_failure=True, argv=["n=3", "-c", "X y"])]


def _list_reply(jobs, info=None, nxt=(3,)):
    return {"status": "ok", "jobs": jobs, "next": list(nxt), "run_queue": dict(info or INFO)}


def _status(info=None, **kw):
    info = dict(info or INFO)
    d = {"state": 0, "state_name": "READY", "run_queue": info,
         "person_hold": info.get("person_hold")}
    d.update(kw)
    return d


@pytest.fixture
def server():
    return FakeServer()


@pytest.fixture
def panel(qapp, server):
    server.answers["list"] = _list_reply(_phase1_jobs())
    p = rqp.RunQueuePanel(server, by="jp@test", synchronous=True)
    p.confirms = []
    p.confirm_answer = True
    p.confirm = lambda title, text, yes="OK", no="Cancel": (
        p.confirms.append((title, text, yes, no)) or p.confirm_answer)
    p.text_answer = "mine"
    p.ask_text = lambda title, prompt, default="": p.text_answer
    p.edit_answer = None
    p.ask_edit = lambda job: p.edit_answer
    p.set_state(_status())
    p.refresh_list()
    yield p
    p.shutdown()


def _row(panel, job_id):
    return panel.model.row_of(job_id)


def _cell(panel, job_id, key, role=Qt.ItemDataRole.DisplayRole):
    index = panel.model.index(_row(panel, job_id), rqp.COLUMN_KEYS.index(key))
    return panel.model.data(index, role)


# --- the table ----------------------------------------------------------------------------

def test_phase1_listing_renders_in_list_order_with_1b_columns_blank(panel):
    assert [j["id"] for j in panel.model.jobs] == [1, 2, 3, 4, 5]
    assert _cell(panel, 2, "position") == "slot"
    assert _cell(panel, 3, "position") == "1"                      # next[0]
    assert _cell(panel, 4, "position") == ""                       # not eligible now
    assert _cell(panel, 2, "state") == "running"
    assert _cell(panel, 2, "run_id") == "85600"
    assert _cell(panel, 2, "owner") == "agent"
    full = "C:/code/kexp/experiments/JP/expt_3.py"
    assert _cell(panel, 3, "path") == full
    assert full in _cell(panel, 3, "path", Qt.ItemDataRole.ToolTipRole)
    for key in ("expt_class", "est", "source_changed"):
        assert _cell(panel, 3, key) == ""
    # why it waits is derived when the server does not say (before 1b)
    assert _cell(panel, 4, "waiting").startswith("due at ")
    assert _cell(panel, 5, "waiting") == "waiting for job 3"
    assert _cell(panel, 5, "after") == "after 3 | chain c1"
    assert _cell(panel, 3, "submitted").endswith("by jp@kong")
    assert "argv: n=3 -c X y" == _cell(panel, 5, "label", Qt.ItemDataRole.ToolTipRole)
    assert panel.table.textElideMode() == Qt.TextElideMode.ElideMiddle


def _est(duration, start=None, end=None, basis="median of 3 saved runs"):
    return {"duration_s": duration, "eta_start": start, "eta_end": end, "basis": basis}


def _phase1b_jobs():
    """As the 1b server lists them: ended, slot, queued in rank order --
    here deliberately shuffled, to check the panel's own ordering."""
    return [_job(3, position=2, rank=3.0, expt_class="Rabi", calibrates_declared=["t_pi"],
                 submitter="agent:codex@kong", estimate=_est(125.0, NOW + 60, NOW + 185),
                 source_changed=True, waiting="job 4 goes first"),
            _job(4, position=0, rank=1.0, expt_class="Tof",
                 estimate=_est(40.0, NOW + 10, NOW + 50), source_changed=False,
                 waiting="launching"),
            _job(2, "running", run_id=85600, owner="agent", position=None,
                 estimate=_est(300.0, None, NOW + 100, "expected end")),
            _job(5, position=1, rank=2.0, waiting="paused by jp@kong", paused=True,
                 paused_by="jp@kong", paused_since=NOW - 60, estimate=_est(None, None, None,
                                                                           "no saved runs")),
            _job(1, "saved", position=None, estimate=_est(30.0))]


def test_phase1b_fields_order_and_render(panel, server):
    server.answers["list"] = _list_reply(_phase1b_jobs(), nxt=(4, 3))
    panel.refresh_list()
    # ended, slot, then queued by position
    assert [j["id"] for j in panel.model.jobs] == [1, 2, 4, 5, 3]
    assert _cell(panel, 4, "position") == "1"                      # position 0, shown 1-based
    assert _cell(panel, 3, "position") == "3"
    assert "counted from 0" in _cell(panel, 3, "position", Qt.ItemDataRole.ToolTipRole)
    assert _cell(panel, 2, "position") == "slot"
    assert _cell(panel, 3, "expt_class") == "Rabi [cal]"
    assert "t_pi" in _cell(panel, 3, "expt_class", Qt.ItemDataRole.ToolTipRole)
    assert _cell(panel, 4, "expt_class", Qt.ItemDataRole.ToolTipRole) is None
    assert "write-back vetoed (WAXX_CAL_NO_WRITE_BACK)" == panel.model.tooltip(
        dict(_job(9, write_back=False)), "expt_class")
    assert _cell(panel, 3, "owner") == "person / agent:codex@kong"
    est = _cell(panel, 3, "est")
    assert est.startswith("~2.1 min") and "start " in est and "end " in est
    assert est.endswith(" est.")
    assert "median of 3 saved runs" in _cell(panel, 3, "est", Qt.ItemDataRole.ToolTipRole)
    assert _cell(panel, 2, "est").startswith("~5.0 min end ")      # the slot: its end
    assert _cell(panel, 5, "est") == ""                             # unknown: blank
    assert _cell(panel, 1, "est") == "~30 s est."                   # ended: no times
    assert _cell(panel, 3, "source_changed") == "CHANGED"
    assert "skipped at launch" in _cell(panel, 3, "source_changed", Qt.ItemDataRole.ToolTipRole)
    assert _cell(panel, 4, "source_changed") == ""
    assert _cell(panel, 3, "waiting") == "job 4 goes first"        # the server's, as sent
    assert _cell(panel, 5, "state") == "queued (paused)"
    assert "paused by jp@kong" in _cell(panel, 5, "state", Qt.ItemDataRole.ToolTipRole)
    assert "rank 3" in _cell(panel, 3, "priority", Qt.ItemDataRole.ToolTipRole)
    assert _cell(panel, 3, "priority") == "10"


def test_ended_jobs_can_be_hidden(panel):
    panel.ended_box.setChecked(False)
    assert [j["id"] for j in panel.model.jobs] == [2, 3, 4, 5]
    panel.ended_box.setChecked(True)
    assert len(panel.model.jobs) == 5


# --- the top strip --------------------------------------------------------------------------

def test_summary_alarm_hold_pause_and_loop_note(panel):
    info = dict(INFO, state="held", alarm={"since": NOW - 700, "job": 3, "waited_s": 700.0,
                                           "why": "liveOD is not reachable"},
                person_hold={"active": True, "since": NOW - 60, "by": "jp@kong",
                             "reason": "aligning", "source": "request", "owner": "person",
                             "run_id": None},
                paused={"agent": {"by": "jp@kong", "since": NOW, "reason": "tea",
                                  "owner": "person"}, "all": None},
                resume_loop={"key": "auto_tof", "path": None, "since": NOW})
    panel.set_state(_status(info))
    assert panel.pill.text() == "held"
    assert not panel.alarm.isHidden()
    assert "job 3" in panel.alarm.text() and "liveOD is not reachable" in panel.alarm.text()
    assert "11.7 min" in panel.alarm.text()
    hold = panel.hold_label.text()
    for word in ("jp@kong", "aligning", "owner person", "source request"):
        assert word in hold
    assert panel.hold_button.text() == "Release hold"
    assert "PAUSED by jp@kong" in panel.pause_labels["agent"].text()
    assert panel.pause_buttons["agent"].text() == "Resume agent"
    assert panel.pause_buttons["all"].text() == "Pause all"
    assert not panel.loop_note.isHidden() and "auto_tof" in panel.loop_note.text()
    # launching shows as such
    cur = _job(2, "launching")
    panel.set_state(_status(dict(INFO, current=cur)))
    assert panel.pill.text() == "launching"


def test_unreachable_and_no_queue(panel):
    panel.set_state(None)
    assert not panel.reachable and not panel.cancel_button.isEnabled()
    assert "not answering" in panel.summary.text()
    panel.set_state({"state": 0, "state_name": "READY"})           # an older server
    assert panel.reachable and not panel.has_queue
    assert "no run queue" in panel.summary.text()
    assert not panel.hold_button.isEnabled()


# --- requests ----------------------------------------------------------------------------

def test_hold_release_pause_resume_send_exact_requests(panel, server):
    panel.toggle_hold()
    assert server.requests[-1] == {"type": "run_queue", "action": "hold", "reason": "mine",
                                   "owner": "person", "by": "jp@test"}
    panel.hold = {"active": True}
    panel.toggle_hold()
    assert server.requests[-1] == {"type": "run_queue", "action": "release",
                                   "owner": "person", "by": "jp@test"}
    panel.text_answer = ""
    panel.toggle_pause("agent")
    assert server.requests[-1] == {"type": "run_queue", "action": "pause", "scope": "agent",
                                   "reason": "", "owner": "person", "by": "jp@test"}
    panel.info["paused"] = {"agent": None, "all": {"by": "x"}}
    panel.toggle_pause("all")
    assert server.requests[-1] == {"type": "run_queue", "action": "resume", "scope": "all",
                                   "owner": "person", "by": "jp@test"}
    n = len(server.requests)
    panel.text_answer = None                                     # the dialog cancelled
    panel.toggle_pause("agent")
    assert len(server.requests) == n


def test_cancel_a_queued_job(panel, server):
    assert panel.select_job(3)
    assert panel.cancel_button.isEnabled()
    assert panel.cancel_selected()
    title, text, yes, no = panel.confirms[-1]
    assert "has not started" in text and "discard" not in text.lower()
    assert server.of("cancel")[-1] == {"type": "run_queue", "action": "cancel", "id": 3,
                                       "token": "t3", "owner": "person", "by": "jp@test",
                                       "queued_only": True}


def test_cancel_a_running_job_says_its_data_is_discarded(panel, server):
    panel.select_job(2)
    panel.confirm_answer = False
    assert not panel.cancel_selected()
    assert server.of("cancel") == []                             # declined: nothing sent
    title, text, yes, no = panel.confirms[-1]
    assert "Abort" in title
    assert "DISCARDS THAT RUN'S DATA FILE" in text and "Reset" in text
    assert "run 85600" in text
    assert "no way yet to stop" in text                          # nothing else offered
    assert yes == "Abort the run and discard its data" and no == "Keep it running"
    panel.confirm_answer = True
    server.answers["cancel"] = {"status": "ok", "pending": True, "job": _job(2, "running")}
    assert panel.cancel_selected()
    assert server.of("cancel")[-1] == {"type": "run_queue", "action": "cancel", "id": 2,
                                       "token": "t2", "owner": "person", "by": "jp@test"}
    assert "next shot" in panel.message.text()


def test_a_queued_cancel_that_meets_a_launch_is_not_turned_into_an_abort(panel, server):
    server.answers["cancel"] = {"status": "error", "state": "running",
                                "msg": "job 3 (expt_3) is running, not queued: not cancelled"}
    panel.select_job(3)
    panel.cancel_selected()
    assert len(server.of("cancel")) == 1                         # never re-sent by itself
    assert "launched before the cancel" in panel.message.text()


def test_cancel_disabled_for_ended_jobs_and_after_a_cancel_was_asked(panel, server):
    panel.select_job(1)
    assert not panel.cancel_button.isEnabled()
    jobs = _phase1_jobs()
    jobs[1]["cancel"] = {"by": "jp", "at": NOW, "abort_sent": True}
    server.answers["list"] = _list_reply(jobs)
    panel.refresh_list()
    panel.select_job(2)
    assert not panel.cancel_button.isEnabled()
    assert _cell(panel, 2, "state") == "running (cancel asked)"


def test_move_names_neighbours_by_id_and_unknown_disables_it(panel, server):
    base = {"type": "run_queue", "action": "move", "id": 4, "token": "t4", "owner": "person",
            "by": "jp@test"}
    panel.select_job(4)                                          # queued: 3, 4, 5
    for where, extra in (("up", {"before_id": 3}), ("down", {"after_id": 5}),
                         ("top", {"to_index": 0}), ("bottom", {"to_index": 3})):
        assert panel.move_buttons[where].isEnabled()
        panel.move_buttons[where].click()
        assert server.of("move")[-1] == dict(base, **extra)
    panel.select_job(3)                                          # first: no up / top
    assert not panel.move_buttons["up"].isEnabled()
    assert not panel.move_buttons["top"].isEnabled()
    assert "already first" in panel.move_buttons["up"].toolTip()
    assert panel.move_buttons["down"].isEnabled()
    panel.select_job(5)                                          # last: no down / bottom
    assert not panel.move_buttons["down"].isEnabled()
    assert not panel.move_buttons["bottom"].isEnabled()
    server.answers["move"] = {"status": "error",
                              "msg": "unknown run_queue action 'move' (known: submit, cancel)"}
    panel.move_buttons["up"].click()
    for b in panel.move_buttons.values():
        assert not b.isEnabled()
        assert "older code" in b.toolTip()
    n = len(server.requests)
    assert not panel.move_selected("up")
    assert len(server.requests) == n
    panel.select_job(2)                                          # a running job: never movable
    assert not panel.move_buttons["up"].isEnabled()


def test_actions_list_in_the_info_gates_1b_controls(panel):
    panel.set_state(_status(dict(INFO, actions=["list", "cancel", "move"])))
    panel.select_job(3)
    assert panel.move_buttons["down"].isEnabled()
    assert not panel.edit_button.isEnabled()
    assert "older code" in panel.edit_button.toolTip()


def test_edit_sends_only_the_changes_and_unknown_disables_it(panel, server):
    panel.select_job(3)
    panel.edit_answer = {"label": "renamed", "argv": ["n=5"]}
    assert panel.edit_selected()
    assert server.of("edit")[-1] == {"type": "run_queue", "action": "edit", "id": 3,
                                     "token": "t3", "fields": {"label": "renamed",
                                                               "argv": ["n=5"]},
                                     "owner": "person", "by": "jp@test"}
    server.answers["edit"] = {"status": "error", "msg": "unknown run_queue action 'edit'"}
    panel.edit_selected()
    assert not panel.edit_button.isEnabled()
    panel.edit_answer = None
    n = len(server.requests)
    assert not panel.edit_selected()
    assert len(server.requests) == n


def test_edit_dialog_reports_only_changed_fields(qapp):
    job = _job(7, after=[3], chain="c", stop_on_failure=True, argv=["a=1"], due=NOW + 7200)
    d = rqp.EditJobDialog(job)
    assert d.changes() == {}
    d.label.setText("other")
    d.after.setText("3, 4")
    d.no_write_back.setChecked(True)
    d.paused.setChecked(True)
    d.allow_drift.setChecked(True)
    assert d.changes() == {"label": "other", "after": [3, 4], "write_back": False,
                           "allow_drift": True, "paused": True}
    d.after.setText("x")
    assert d.changes() is None
    assert not d.buttons.button(d.buttons.StandardButton.Ok).isEnabled()
    d.deleteLater()


def test_parse_due():
    import datetime
    base = datetime.datetime(2026, 10, 9, 12, 0).timestamp()
    assert rqp._parse_due("", base) is None
    assert rqp._parse_due("13:30", base) == base + 5400
    assert rqp._parse_due("11:00", base) == datetime.datetime(2026, 10, 10, 11, 0).timestamp()
    # by the calendar: across the DST change (2026-11-01 in the US) 11:00 is 11:00
    sat = datetime.datetime(2026, 10, 31, 12, 0).timestamp()
    assert rqp._parse_due("11:00", sat) == datetime.datetime(2026, 11, 1, 11, 0).timestamp()
    assert rqp._parse_due("2026-10-12 08:15", base) == \
        datetime.datetime(2026, 10, 12, 8, 15).timestamp()
    assert rqp._parse_due(str(int(base) + 60), base) == int(base) + 60   # epoch, as kq --at
    with pytest.raises(ValueError):
        rqp._parse_due("soon", base)
    # the dialog reads back what it shows for another day ("MM-DD HH:MM")
    later = datetime.datetime.now().replace(second=0, microsecond=0) + \
        datetime.timedelta(days=3)
    shown = rqp._clock(later.timestamp())
    assert len(shown) == len("10-12 08:15")
    assert rqp._parse_due(shown) == later.timestamp()


def test_unknown_request_detection():
    assert is_unknown_request({"status": "error", "msg": "unknown run_queue action 'move'"})
    assert is_unknown_request({"status": "error", "msg": "unknown type get_journal"})
    assert not is_unknown_request({"status": "error", "msg": "job 3 is running"})
    assert not is_unknown_request({"status": "ok"})
    assert normalize_reply(None)["no_reply"] and normalize_reply("x")["status"] == "error"


def test_the_requester_raising_becomes_an_error_reply(qapp):
    def boom(obj):
        raise OSError("down")
    got = []
    RequestRunner(boom, synchronous=True).send({"type": "x"}, got.append)
    assert got[0]["status"] == "error" and "down" in got[0]["msg"] and got[0]["no_reply"]


def test_kq_commands(panel):
    panel.select_job(5)
    cmds = rqp.kq_commands(panel.selected_job())
    # defaults left out: the label is the file's stem; stop-on-failure is a chain's default
    assert cmds["submit"] == ("kq submit C:/code/kexp/experiments/JP/expt_5.py --priority 10 "
                              "--depends-on 3 --chain c1 -- n=3 -c \"X y\"")
    assert cmds["tail"] == "kq tail 5 -f"
    assert panel.copy_kq("tail") == "kq tail 5 -f"
    agent = rqp.kq_commands(_job(9, owner="agent", write_back=False, path="C:/a b/x.py",
                                 priority=0, label="other"))
    assert agent["submit"] == ('kq submit "C:/a b/x.py" --label other --no-write-back '
                               '--agent')
    now = time.time()
    rep = rqp.kq_commands(_job(10, priority=0, due=now + 3600, repeat_of=5, repeat_index=2,
                               chain="repeat-7", stop_on_failure=True,
                               path="C:/x/my scan.py", label="my_scan"), now=now)
    assert rep["submit"] == (f'kq submit "C:/x/my scan.py" --at '
                             f'{time.strftime("%H:%M", time.localtime(now + 3600))} '
                             f'--repeat 5')
    far = rqp.kq_commands(_job(11, priority=0, due=now + 3 * 86400, chain="mine",
                               stop_on_failure=False), now=now)
    assert f"--at {int(now + 3 * 86400)} --chain mine --no-stop-on-failure" in far["submit"]
    past = rqp.kq_commands(_job(12, priority=0, due=now - 60), now=now)
    assert "--at" not in past["submit"]
    # kq itself reads them back
    from waxx.util.device_state.kq import build_parser
    for line, want in ((cmds["submit"], {"after": [3], "chain": "c1", "priority": 10}),
                       (rep["submit"], {"repeat": 5, "chain": None}),
                       (far["submit"], {"chain": "mine", "no_stop_on_failure": True}),
                       (agent["submit"], {"label": "other", "no_write_back": True})):
        args = build_parser().parse_args(rqp.split_argv(line)[1:])
        for k, v in want.items():
            assert getattr(args, k) == v, (line, k)


# --- the log window -------------------------------------------------------------------------

class LogServer(FakeServer):
    """Serves a log the way the queue's tail does: whole lines from a byte
    offset, ``done`` once ended and read to the end."""

    def __init__(self, data: bytes, state="running"):
        super().__init__()
        self.data, self.state, self.silent = data, state, False

    def __call__(self, obj):
        self.requests.append(dict(obj))
        if obj.get("action") != "tail":
            return {"status": "ok"}
        if self.silent:
            return None
        off = obj["offset"]
        chunk = self.data[off:]
        cut = chunk.rfind(b"\n") + 1
        ended = self.state in ("saved", "failed", "cancelled", "skipped")
        if ended:
            cut = len(chunk)
        text = chunk[:cut].decode()
        lines = text[:-1].split("\n") if text.endswith("\n") else (text.split("\n") if text
                                                                     else [])
        new = off + cut
        return {"status": "ok", "lines": lines if cut else [], "offset": new,
                "done": ended and new >= len(self.data), "state": self.state, "run_id": 85600}


def test_log_window_follows_with_a_cursor(qapp):
    server = LogServer(b"Run ID: 85600\nshot 1/2\nshot 2")
    p = rqp.RunQueuePanel(server, by="jp@test", synchronous=True)
    try:
        w = p.show_log(_job(2, "running"))
        assert server.requests[0] == {"type": "run_queue", "action": "tail", "id": 2,
                                      "token": "t2", "offset": 0}
        assert w.text.toPlainText() == "Run ID: 85600\nshot 1/2"
        assert w.offset == len(b"Run ID: 85600\nshot 1/2\n")
        assert w.timer.isActive() and w.timer.interval() == rqp.TAIL_RUNNING_MS
        # silence: asked again, the cursor stays
        server.silent = True
        w.fetch()
        assert w.offset == len(b"Run ID: 85600\nshot 1/2\n") and not w.done
        assert "not answering" in w.status.text() and w.timer.isActive()
        server.silent = False
        server.data += b"\nsaved\n"
        server.state = "saved"
        w.fetch()
        assert server.requests[-1]["offset"] == len(b"Run ID: 85600\nshot 1/2\n")
        assert w.text.toPlainText().splitlines()[-2:] == ["shot 2", "saved"]
        assert w.done and not w.timer.isActive()
        n = len(server.requests)
        w.fetch()                                                  # done: never asked again
        assert len(server.requests) == n
        assert p.show_log(_job(2, "running")) is w                 # one window per job
    finally:
        p.shutdown()


def test_log_window_queued_interval_refusal_and_no_rewind(qapp):
    server = LogServer(b"", state="queued")
    p = rqp.RunQueuePanel(server, by="jp@test", synchronous=True)
    try:
        w = p.show_log(_job(3))
        assert w.timer.interval() == rqp.TAIL_QUEUED_MS and w.offset == 0
        w.on_reply({"status": "ok", "lines": [], "offset": 0, "done": False,
                    "state": "running"})
        assert w.timer.interval() == rqp.TAIL_RUNNING_MS
        w.offset = 100
        w.on_reply({"status": "ok", "lines": [], "offset": 40, "done": False,
                    "state": "running"})
        assert w.offset == 100                                     # never back
        w.on_reply({"status": "error", "msg": "job 3 has token t9, not t3"})
        assert w.stopped and not w.timer.isActive() and "token" in w.status.text()
        w2 = p.show_log(_job(4))
        w2.on_reply({"status": "error", "msg": "unknown run_queue action 'tail'"})
        assert w2.stopped and "older code" in w2.status.text()
    finally:
        p.shutdown()


# --- refreshes --------------------------------------------------------------------------------

def test_first_show_lists_and_broadcasts_are_debounced(qapp, server):
    server.answers["list"] = _list_reply(_phase1_jobs())
    p = rqp.RunQueuePanel(server, by="jp@test", synchronous=True)
    try:
        p.set_state(_status())
        assert server.of("list") == []
        p.show()
        qapp.processEvents()
        assert len(server.of("list")) == 1                          # on first show
        assert server.of("list")[0] == {"type": "run_queue", "action": "list", "limit": 200}
        for _ in range(5):
            p.on_broadcast({"type": "run_queue", "run_queue": dict(INFO, text="x")})
        assert len(server.of("list")) == 1                          # not yet
        assert p.summary.text() == "x"                               # the summary at once
        QTest.qWait(rqp.LIST_DEBOUNCE_MS + 150)
        assert len(server.of("list")) == 2                          # one list for five
        p.on_broadcast({"type": "trust", "trust": {}})               # not the queue's
        p.on_broadcast({"type": "person_hold", "person_hold": {"active": True, "by": "x"}})
        assert p.hold_button.text() == "Release hold"
        QTest.qWait(rqp.LIST_DEBOUNCE_MS + 150)
        assert len(server.of("list")) == 2
        # a status_json whose queue changed asks for the list too (whose
        # reply carries the same, current, info -- as the server's does)
        server.answers["list"] = _list_reply(_phase1_jobs(), dict(INFO, next=[4, 3]), (4, 3))
        p.set_state(_status(dict(INFO, next=[4, 3])))
        QTest.qWait(rqp.LIST_DEBOUNCE_MS + 150)
        assert len(server.of("list")) == 3
        p.set_state(_status(dict(INFO, next=[4, 3])))                # unchanged: no list
        QTest.qWait(rqp.LIST_DEBOUNCE_MS + 150)
        assert len(server.of("list")) == 3
        p.hide()
    finally:
        p.shutdown()


def test_selection_survives_a_refresh(panel, server):
    panel.select_job(4)
    panel.refresh_list()
    assert panel.selected_job()["id"] == 4


def test_queue_summary_line(qapp):
    line = rqp.QueueSummaryLine()
    line.set_info(None)
    assert line.isHidden()
    line.set_info(dict(INFO))
    assert not line.isHidden()
    assert line.label.text() == ("Queue: running (3 queued) -- open the Monitor panel in the "
                                 "Server Dashboard")
    line.set_info(dict(INFO, state="waiting", alarm={"job": 3}))
    assert line.label.text().startswith("Queue: waiting (3 queued) -- ALARM")


def test_edit_dialog_leaves_argv_alone_unless_it_changed(qapp):
    """B-2: argv with spaces in a word survives a label-only edit, and an
    edited argv is split shell-style (quotes keep a word; backslashes stay)."""
    job = _job(8, argv=["n=3", "note=a b", r"path=C:\x\y"])
    d = rqp.EditJobDialog(job)
    assert d.argv.text() == r'n=3 "note=a b" path=C:\x\y'
    d.label.setText("only_the_label")
    assert d.changes() == {"label": "only_the_label"}
    d.argv.setText(r'n=4 "note=a b" path=C:\x\y')
    assert d.changes()["argv"] == ["n=4", "note=a b", r"path=C:\x\y"]
    d.argv.setText('n=4 "unclosed')
    assert d.changes() is None and "argv" in d.problem.text()
    assert rqp.split_argv(rqp.join_argv(["a b", "", "c"])) == ["a b", "", "c"]
    d.deleteLater()
