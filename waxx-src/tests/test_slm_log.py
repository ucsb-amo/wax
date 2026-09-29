"""The SLM server's log in the Device Control GUI (2026-09-28), offline.

* ``slm_protocol.read_log``: the supervisor's daily files read from a cursor --
  the last lines first, only whole lines of the file still being written, the
  rest of a day's file before the next day's, a backlog in pieces, a long gap
  jumped and counted, a cursor that no longer fits starts again;
* run_server.py's ``log`` command (Meadowlark stubbed, 127.0.0.1): answered at
  once, leaves no lines of its own, refused without a supervisor;
* the monitor server's service fetches the log only while a GUI asks for it,
  asks nothing of a server without the command, says each problem once; the
  monitor server serves it as ``output`` kind ``slm``;
* the SLM pill's "View SLM server log…" and the log window.

Nothing touches the lab network (127.0.0.1 only) or a real log folder:
``SLM_STATE_DIR`` points at a temp dir.
"""
import json
import os
import time
import types

import pytest

from waxx.control.slm.server import slm_protocol as sp
from waxx.util.device_state import slm_reinit as sr
from waxx.util.device_state.slm_reinit import SlmReinitConfig, SlmReinitService

from test_slm_supervision import _ask, _free_port, _status, make_server, supervisor_mod  # noqa: F401

DAY1 = "slm_server_2026-09-27.log"
DAY2 = "slm_server_2026-09-28.log"


def _write(folder, name, text):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / name, "ab") as fh:
        fh.write(text.encode("utf-8"))


# --- read_log --------------------------------------------------------------------------

def test_the_first_read_gives_the_last_lines_and_holds_back_a_line_being_written(tmp_path):
    lines = [f"2026-09-28 10:00:{i:02d} | line {i}" for i in range(20)]
    whole = "\n".join(lines) + "\n"
    _write(tmp_path, DAY2, whole + "2026-09-28 10:00:59 | half")
    r = sp.read_log(tmp_path, None, tail=5)
    assert r["status"] == "ok" and r["lines"] == lines[-5:]
    assert (r["more"], r["restarted"], r["skipped_bytes"]) == (False, False, 0)
    assert r["cursor"] == {"file": DAY2, "offset": len(whole.encode())}
    assert r["path"] == str(tmp_path)
    _write(tmp_path, DAY2, " written\nnext\n")          # the half line is finished
    r2 = sp.read_log(tmp_path, r["cursor"])
    assert r2["lines"] == ["2026-09-28 10:00:59 | half written", "next"]
    r3 = sp.read_log(tmp_path, r2["cursor"])
    assert r3["lines"] == [] and r3["cursor"] == r2["cursor"] and r3["more"] is False


def test_the_rest_of_a_day_comes_before_the_next_day(tmp_path):
    _write(tmp_path, DAY1, "a\nb\n")
    r = sp.read_log(tmp_path, None)
    assert r["lines"] == ["a", "b"]
    _write(tmp_path, DAY1, "c\nlast of the day")      # no newline: that file is finished
    _write(tmp_path, DAY2, "d\n")
    r2 = sp.read_log(tmp_path, r["cursor"])
    assert r2["lines"] == ["c", "last of the day", "d"]
    assert r2["cursor"] == {"file": DAY2, "offset": 2}


def test_the_last_lines_reach_into_yesterday_just_after_midnight(tmp_path):
    _write(tmp_path, DAY1, "".join(f"y{i}\n" for i in range(10)))
    _write(tmp_path, DAY2, "t0\nt1\n")
    r = sp.read_log(tmp_path, None, tail=4)
    assert r["lines"] == ["y8", "y9", "t0", "t1"]
    assert r["cursor"] == {"file": DAY2, "offset": 6}


def test_a_backlog_comes_in_pieces(tmp_path):
    _write(tmp_path, DAY2, "start\n")
    cursor = sp.read_log(tmp_path, None)["cursor"]
    _write(tmp_path, DAY2, "".join(f"n{i}\n" for i in range(25)))
    got, calls = [], 0
    while True:
        r = sp.read_log(tmp_path, cursor, max_lines=10)
        got += r["lines"]
        cursor = r["cursor"]
        calls += 1
        if not r["more"]:
            break
    assert got == [f"n{i}" for i in range(25)] and calls == 3


def test_a_gap_nobody_followed_is_jumped_and_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "LOG_SKIP_BYTES", 1000)
    monkeypatch.setattr(sp, "LOG_RESUME_BYTES", 100)
    _write(tmp_path, DAY1, "old\n")
    cursor = sp.read_log(tmp_path, None)["cursor"]
    assert cursor == {"file": DAY1, "offset": 4}
    _write(tmp_path, DAY1, "x" * 500 + "\n")                           # 501 bytes unread
    _write(tmp_path, DAY2, "".join(f"line {i:04d}\n" for i in range(200)))  # 10 bytes each
    r = sp.read_log(tmp_path, cursor)
    assert r["lines"][-1] == "line 0199"
    assert all(len(line) == 9 and line.startswith("line ") for line in r["lines"])  # whole lines
    resumed_at = int(r["lines"][0][5:]) * 10
    assert 2000 - resumed_at <= 100
    assert r["skipped_bytes"] == 501 + resumed_at                      # exactly what was not sent
    assert r["cursor"] == {"file": DAY2, "offset": 2000}


def test_a_cursor_that_no_longer_fits_starts_again_from_the_last_lines(tmp_path):
    _write(tmp_path, DAY2, "a\nb\nc\n")
    for bad in ({"file": "slm_server_2026-01-01.log", "offset": 0},    # removed
                {"file": DAY2, "offset": 999},                          # cut short
                {"file": DAY2, "offset": -1}, {"file": DAY2, "offset": True}, "junk"):
        r = sp.read_log(tmp_path, bad, tail=2)
        assert r["restarted"] is True and r["lines"] == ["b", "c"], bad
        assert r["cursor"] == {"file": DAY2, "offset": 6}


def test_long_lines_are_cut_and_say_so_and_no_files_is_no_lines(tmp_path):
    empty = sp.read_log(tmp_path / "nothing here", None)
    assert empty["status"] == "ok" and empty["lines"] == [] and empty["cursor"] is None
    _write(tmp_path, DAY2, "y" * (sp.LOG_LINE_CHARS + 50) + "\n")
    line = sp.read_log(tmp_path, None)["lines"][0]
    assert line.startswith("y" * sp.LOG_LINE_CHARS) and line.endswith("(50 more characters)")


def test_the_supervisor_writes_where_the_log_is_read(tmp_path, supervisor_mod):
    folder = sp.log_dir(str(tmp_path))
    supervisor_mod.Journal(folder).line("[supervisor] hello")
    assert sp.read_log(folder, None)["lines"][-1].endswith(" | [supervisor] hello")


# --- the server's log command -------------------------------------------------------------

def test_the_log_command_answers_at_once_and_leaves_no_lines(make_server, tmp_path,
                                                            monkeypatch, capsys):
    monkeypatch.setenv("SLM_STATE_DIR", str(tmp_path / "state"))
    folder = tmp_path / "state" / "logs"
    _write(folder, DAY2, "2026-09-28 10:00:00 | [supervisor] SLM server started\n")
    host = make_server()
    assert "log" in _status(host)["capabilities"]
    time.sleep(0.2)
    capsys.readouterr()
    r = _ask(host, {"cmd": "log", "cursor": None, "tail": 50}, ("ok", "error"), control=True)
    assert r["status"] == "ok" and r["path"] == str(folder)
    assert r["lines"] == ["2026-09-28 10:00:00 | [supervisor] SLM server started"]
    time.sleep(0.2)                                      # the connection is over
    assert capsys.readouterr().out == ""                 # nothing of its own for the log
    _status(host)
    time.sleep(0.2)
    out = capsys.readouterr().out
    assert "Connected by" in out and "Received command" in out and "Client disconnected" in out


def test_no_log_without_a_supervisor(make_server, tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_STATE_DIR", str(tmp_path / "state"))
    host = make_server(supervised=False)
    r = _ask(host, {"cmd": "log"}, ("ok", "error"), control=True)
    assert r["status"] == "error" and "supervisor" in r["error"]


# --- the monitor server's service ---------------------------------------------------------

class FakeLoggingSlm:
    """The SLM server as the service sees it, with a log."""

    def __init__(self, capabilities=("seq", "status", "reinit", "restart", "shutdown", "log"),
                 supervised=True):
        self.capabilities = list(capabilities)
        self.supervised = supervised
        self.requests = []
        self.cursors = []
        self.pending = []           # lines the next log reply carries
        self.reply = None           # a fixed reply to "log"
        self.fail = None            # an exception "log" raises

    def __call__(self, host, port, payload, *, control, until, on_sent=None, **kw):
        cmd = payload["cmd"]
        self.requests.append(cmd)
        if on_sent:
            on_sent()
        if cmd == "status":
            return {"status": "ok", "reinit_due": False, "next_due_in_s": 1800.0,
                    "interval_s": 3600, "reinit_in_progress": False, "pattern_epoch": 0,
                    "pattern": {}, "last_reinit_error": "", "supervised": self.supervised,
                    "capabilities": self.capabilities, "instance": "a1", "start_count": 1,
                    "slm_ready": True}
        if cmd == "log":
            self.cursors.append(payload.get("cursor"))
            if self.fail is not None:
                raise self.fail
            if self.reply is not None:
                return dict(self.reply)
            lines, self.pending = self.pending, []
            offset = (payload.get("cursor") or {}).get("offset", 0) + len(lines)
            return {"status": "ok", "lines": lines, "cursor": {"file": "f", "offset": offset},
                    "more": False, "restarted": False, "skipped_bytes": 0,
                    "path": r"C:\slm\logs"}
        raise AssertionError(cmd)


def _svc(slm):
    clock = types.SimpleNamespace(t=0.0)
    svc = SlmReinitService(SlmReinitConfig(host="127.0.0.1", port=1), blocker=lambda: "",
                           link=slm, clock=lambda: clock.t)
    return svc, clock


def _run(svc, clock, until_t, asking=False):
    while clock.t < until_t:
        if asking:
            svc.log_since(0)
        svc.tick()
        clock.t += 0.5


def test_the_log_is_fetched_only_while_a_gui_asks_for_it():
    slm = FakeLoggingSlm()
    slm.pending = ["one", "two"]
    svc, clock = _svc(slm)
    _run(svc, clock, 30.0)
    assert "log" not in slm.requests
    got = svc.log_since(0)
    assert got["lines"] == [] and got["follow"]["state"] == "idle"
    _run(svc, clock, 31.0)
    got = svc.log_since(0)
    assert got["lines"] == ["one", "two"] and got["follow"]["state"] == "following"
    assert got["follow"]["path"] == r"C:\slm\logs" and slm.cursors[0] is None
    slm.pending = ["three"]
    _run(svc, clock, 34.0)
    assert svc.log_since(got["next"] - 1)["lines"] == ["three"]
    assert slm.cursors[1] == {"file": "f", "offset": 2}   # the cursor goes back
    # Nobody asks any more: it stops after LOG_WANTED_S.
    _run(svc, clock, 34.0 + sr.LOG_WANTED_S + 1.0)
    n = slm.requests.count("log")
    assert n <= 2 + sr.LOG_WANTED_S / sr.LOG_POLL_S + 2
    _run(svc, clock, 300.0)
    assert slm.requests.count("log") == n
    assert svc.log_since(0)["follow"]["state"] == "idle"


@pytest.mark.parametrize("kw, word", [
    ({"capabilities": ("seq", "status", "reinit", "restart", "shutdown")}, "no log command"),
    ({"supervised": False}, "supervisor")])
def test_nothing_is_asked_of_a_server_without_a_log(kw, word):
    slm = FakeLoggingSlm(**kw)
    svc, clock = _svc(slm)
    _run(svc, clock, 10.0, asking=True)
    assert "log" not in slm.requests
    got = svc.log_since(0)
    assert got["follow"]["state"] == "trouble" and word in got["follow"]["detail"]
    assert sum(word in line for line in got["lines"]) == 1          # said once


def test_a_failing_fetch_is_said_once_and_retried_after_a_pause():
    slm = FakeLoggingSlm()
    svc, clock = _svc(slm)
    slm.fail = ConnectionRefusedError("refused")
    _run(svc, clock, 25.0, asking=True)
    assert slm.requests.count("log") == 3                           # t = 0, 10, 20
    lines = svc.log_since(0)["lines"]
    assert sum("did not send its log" in line for line in lines) == 1
    slm.fail, slm.pending = None, ["back"]
    _run(svc, clock, 35.0, asking=True)
    lines = svc.log_since(0)["lines"]
    assert lines[-1] == "back" and "comes through again" in lines[-2]
    slm.reply = {"status": "error", "error": "reading the log failed: PermissionError"}
    n = slm.requests.count("log")
    _run(svc, clock, 35.0 + sr.LOG_REFUSED_RETRY_S - 1.0, asking=True)
    assert slm.requests.count("log") == n + 1                        # a refusal waits longer
    got = svc.log_since(0)
    assert "PermissionError" in got["follow"]["detail"]


def test_jumps_and_restarts_are_said_and_more_is_fetched_at_once():
    slm = FakeLoggingSlm()
    svc, clock = _svc(slm)
    slm.reply = {"status": "ok", "lines": ["x"], "cursor": {"file": "f", "offset": 9},
                 "more": True, "restarted": True, "skipped_bytes": 412_000, "path": "P"}
    _run(svc, clock, 0.5, asking=True)
    lines = svc.log_since(0)["lines"]
    assert any("412 kB" in line and "P on the SLM PC" in line for line in lines)
    assert any("files changed" in line for line in lines) and lines[-1] == "x"
    slm.reply = None
    _run(svc, clock, 1.0, asking=True)                               # the next tick, not 2 s
    assert slm.requests.count("log") == 2 and slm.cursors[-1] == {"file": "f", "offset": 9}


def test_no_fetch_while_the_server_restarts():
    slm = FakeLoggingSlm()
    svc, clock = _svc(slm)
    svc.tick()
    svc._restart_t0, svc._restart_instance0 = 0.0, "a1"
    svc._set(sr.RESTARTING, "test")
    _run(svc, clock, 10.0, asking=True)
    assert "log" not in slm.requests
    follow = svc.log_since(0)["follow"]
    assert (follow["state"], follow["detail"]) == ("waiting", "the SLM server is restarting")


# --- the monitor server ------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class Recorder:
    def __init__(self, *a, **k):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


def _monitor_server(monkeypatch, tmp_path, slm_reinit):
    from waxx.util.guis import monitor_server_gui as msg
    monkeypatch.setattr(msg, "StateBroadcaster", Recorder)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"dds": {}, "dac": {}, "ttl": {}}))
    return msg.MonitorUDPServer(config_file_path=str(path), slm_reinit=slm_reinit)


def test_the_monitor_server_serves_the_log_as_output(qapp, monkeypatch, tmp_path):
    server = _monitor_server(monkeypatch, tmp_path,
                             SlmReinitConfig(host="127.0.0.1", port=_free_port()))
    try:
        server.slm_reinit.log.append("hello")
        reply = json.loads(server.generate_reply(json.dumps(
            {"type": "output", "kind": "slm", "after": 0})))
        assert reply["status"] == "ok" and reply["lines"] == ["hello"]
        assert reply["follow"]["state"] == "idle"
        assert server.slm_reinit._log_wanted_until is not None    # asked for: fetched now
    finally:
        server.sock.close()


def test_a_monitor_server_without_an_slm_says_so(qapp, monkeypatch, tmp_path):
    server = _monitor_server(monkeypatch, tmp_path, None)
    try:
        reply = json.loads(server.generate_reply(json.dumps(
            {"type": "output", "kind": "slm", "after": 0})))
        assert reply["status"] == "error" and "no SLM" in reply["msg"]
    finally:
        server.sock.close()


# --- the pill and the window --------------------------------------------------------------

def test_the_pill_always_offers_the_log(qapp):
    from waxx.util.guis.slm_pill import SlmPill
    pill = SlmPill()
    got = []
    pill.log_requested.connect(lambda: got.append(1))
    for reachable in (True, False):
        pill.set_reachable(reachable)
        menu = pill.build_menu()
        action = next(a for a in menu.actions() if a.objectName() == "slm_view_log")
        assert action.isEnabled()
    action.trigger()
    assert got == [1]
    assert "view its log" in pill.toolTip()


@pytest.fixture
def dcgui(qapp, monkeypatch):
    from waxx.util.guis import device_control_gui as dc
    from waxx.util.guis import sequences_panel
    from test_composite_panel import FakeSender
    from test_device_control_gui import FakeSettings
    monkeypatch.setattr(dc, "QSettings", FakeSettings)
    monkeypatch.setattr(sequences_panel, "_OpSender", FakeSender)
    for name in ("_setup_update_sender", "_setup_state_listener", "_setup_state_worker",
                 "setup_status_checker", "setup_timer", "request_state"):
        monkeypatch.setattr(dc.DeviceStateGUI, name, lambda self, *a, **k: None)
    g = dc.DeviceStateGUI()
    g._test_sent = []
    monkeypatch.setattr(g, "_send_request",
                        lambda obj, cb, **kw: g._test_sent.append((obj, cb, kw)))
    yield g
    g.close()


def test_the_window_follows_the_log_while_it_is_open(dcgui):
    dcgui.slm_pill.log_requested.emit()
    win = dcgui._slm_log_window
    assert win is not None and dcgui._slm_log_timer.isActive()
    obj, cb, kw = dcgui._test_sent[-1]
    assert obj == {"type": "output", "kind": "slm", "after": 0}
    assert kw == {"timeout": 2.0, "attempts": 1}
    dcgui._poll_slm_log()                                   # one request at a time
    assert len(dcgui._test_sent) == 1
    cb({"status": "ok", "lines": ["a", "b"], "first": 1, "next": 3, "more": True,
        "follow": {"state": "following", "path": r"C:\slm\logs", "last_fetch": time.time()}})
    assert win.text.toPlainText() == "a\nb" and win.after == 2
    assert len(dcgui._test_sent) == 2 and dcgui._test_sent[-1][0]["after"] == 2  # more: at once
    assert "Following" in win.status_label.text() and r"C:\slm\logs" in win.status_label.text()
    dcgui._test_sent[-1][1]({"status": "ok", "lines": ["d"], "first": 5, "next": 6, "more": False,
                             "follow": {"state": "trouble", "detail": "no answer (refused)"}})
    assert win.text.toPlainText().endswith("(2 lines were not kept here)\nd")
    assert win.status_label.text() == "⚠ no answer (refused)"
    dcgui._slm_log_timer.timeout.emit()                    # the 1 s poll
    assert dcgui._test_sent[-1][0]["after"] == 5
    win.close()
    assert dcgui._slm_log_window is None and not dcgui._slm_log_timer.isActive()
    n = len(dcgui._test_sent)
    dcgui._poll_slm_log()
    assert len(dcgui._test_sent) == n


def test_an_old_monitor_server_is_named_and_not_asked_again(dcgui):
    dcgui._open_slm_log()
    dcgui._test_sent[-1][1]({"status": "error", "msg": "unknown output kind 'slm'"})
    win = dcgui._slm_log_window
    assert win.unsupported and "restart the monitor server" in win.status_label.text()
    dcgui._poll_slm_log()
    assert len(dcgui._test_sent) == 1


def test_a_monitor_server_restart_starts_the_lines_again(dcgui):
    dcgui._open_slm_log()
    dcgui._test_sent[-1][1]({"status": "ok", "lines": ["a", "b", "c"], "first": 1, "next": 4})
    win = dcgui._slm_log_window
    dcgui._poll_slm_log()
    dcgui._test_sent[-1][1]({"status": "ok", "lines": [], "first": 1, "next": 1})
    assert win.after == 0 and win.text.toPlainText() == ""
    assert dcgui._test_sent[-1][0]["after"] == 0           # asked again from the start


def test_a_late_reply_is_dropped_and_the_gui_closes_the_window(dcgui):
    dcgui._open_slm_log()
    first = dcgui._slm_log_window
    late = dcgui._test_sent[-1][1]
    first.close()
    late({"status": "ok", "lines": ["late"], "first": 1, "next": 2})   # nothing to show it in
    dcgui._open_slm_log()
    second = dcgui._slm_log_window
    assert second is not first and second.after == 0 and second.text.toPlainText() == ""
    dcgui.close()
    assert dcgui._slm_log_window is None
