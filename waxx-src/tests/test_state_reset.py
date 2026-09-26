"""Reset state: the monitor server runs a reset experiment whose own end state
is what marks the device state trusted again.

Nothing is launched: the reset's process is a fake whose output and exit code
the test sets (and whose output can be held open until the test releases it,
so the end state can arrive while it "runs").  The server's broadcaster is a
recorder; its journal lives in memory; the state file is in a pytest temp dir.
"""
import json
import os
import threading
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication

from waxx.util.comms_server.comm_server import STATES
from waxx.util.device_state.monitor_manager import ar_command
from waxx.util.device_state.op_journal import OpJournal, describe_entry
from waxx.util.device_state.state_reset import StateReset, describe_expt
from waxx.util.guis import device_control_gui as dc
from waxx.util.guis import monitor_server_gui as msg
from waxx.util.guis.device_summary import SummaryStrip, reset_title


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


RESET_SOURCE = '''"""Leaves the MOT loading: inner coil ON at i_mot."""
from artiq.experiment import *
'''

STATE = {
    "dds": {"imaging": {"frequency": 350e6, "amplitude": 1.0, "v_pd": 0.3, "sw_state": 1,
                        "urukul_idx": 4, "ch": 1, "dac_ch_key": ""}},
    "dac": {"coil": {"ch": 9, "voltage": 0.0}},
    "ttl": {"shutter": {"ch": 22, "ttl_state": 0}},
}


class FakeProc:
    """Prints *lines*, then (with *hold*) keeps its output open until the
    event is set -- a run still going -- and exits with *code*."""

    def __init__(self, lines=(), code=0, hold=None):
        self.pid = 4242
        self.code = code
        self.hold = hold
        self.stdout = self._out(list(lines))

    def _out(self, lines):
        for line in lines:
            yield line + "\n"
        if self.hold is not None:
            self.hold.wait(5)

    def wait(self, timeout=None):
        return self.code


@pytest.fixture
def reset_file(tmp_path):
    path = tmp_path / "mot_observe.py"
    path.write_text(RESET_SOURCE)
    return path


def _reset(reset_file, proc, **kw):
    changes, commands = [], []

    def spawn(command):
        commands.append(command)
        return proc

    r = StateReset(reset_file, on_change=changes.append, journal=OpJournal(None),
                   spawn=spawn, **kw)
    return r, changes, commands


# --- StateReset ------------------------------------------------------------------

def test_a_reset_is_done_only_once_its_end_state_arrived(reset_file):
    release = threading.Event()
    r, changes, commands = _reset(reset_file, FakeProc(["compiling", "Done!"], 0, release))
    reply = r.start(operator="ada", client="pc2")
    assert reply["status"] == "ok" and reply["reset"]["state"] == "running"
    assert commands == [ar_command(reset_file)]
    assert reply["reset"]["expt"] == "mot_observe"
    assert reply["reset"]["about"] == "Leaves the MOT loading: inner coil ON at i_mot."
    assert r.end_state_received("some_other_expt") is False
    assert r.end_state_received("mot_observe") is True
    release.set()
    r.join(2)
    info = r.info()
    assert info["state"] == "done" and info["exit_code"] == 0 and info["end_state"]
    assert "marked it trusted" in info["text"]
    assert [c["state"] for c in changes] == ["running", "running", "done"]
    kinds = [e["kind"] for e in r._journal.tail()]
    assert kinds == ["state_reset_started", "state_reset_done"]
    assert "mot_observe done by ada@pc2" in describe_entry(r._journal.tail()[-1])


def test_a_reset_without_its_end_state_failed_and_says_why(reset_file):
    lines = ["Traceback (most recent call last):", "artiq.compiler ... CompileError: nope"]
    r, changes, _ = _reset(reset_file, FakeProc(lines, 1))
    assert r.start("ada")["status"] == "ok"
    r.join(2)
    info = r.info()
    assert info["state"] == "failed" and info["exit_code"] == 1 and not info["end_state"]
    assert "without reporting its end state" in info["text"]
    assert "COMPILE" in info["text"] and "still untrusted" in info["text"]
    assert info["tail"] == lines
    assert r._journal.tail()[-1]["kind"] == "state_reset_failed"


def test_exit_code_zero_without_an_end_state_is_still_a_failure(reset_file):
    r, _, _ = _reset(reset_file, FakeProc(["Done!"], 0))
    r.start()
    r.join(2)
    assert r.info()["state"] == "failed"


def test_a_hung_reset_is_killed_and_failed(reset_file):
    release = threading.Event()
    killed = []

    def kill(proc):
        killed.append(proc.pid)
        release.set()

    r, _, _ = _reset(reset_file, FakeProc(["compiling"], 1, release), kill_tree=kill,
                     timeout_s=0.05)
    r.start()
    r.join(2)
    assert killed == [4242]
    info = r.info()
    assert info["state"] == "failed" and "was killed" in info["text"]


def test_one_reset_at_a_time(reset_file):
    release = threading.Event()
    r, _, commands = _reset(reset_file, FakeProc([], 0, release))
    assert r.start("ada", "pc2")["status"] == "ok"
    second = r.start("bob")
    assert second["status"] == "error" and "already running" in second["msg"]
    assert "ada@pc2" in second["msg"] and len(commands) == 1
    assert r._journal.tail()[-1]["kind"] == "state_reset_refused"
    release.set()
    r.join(2)


def test_unconfigured_or_missing_file_is_refused(tmp_path):
    r = StateReset(None, spawn=lambda c: pytest.fail("spawned"))
    assert r.info() is None
    assert "no reset experiment" in r.start()["msg"]
    r = StateReset(tmp_path / "gone.py", spawn=lambda c: pytest.fail("spawned"))
    assert "does not exist" in r.start()["msg"]
    assert r.info()["state"] == "idle"


def test_describe_expt_falls_back_to_the_class_docstring(tmp_path):
    p = tmp_path / "x.py"
    p.write_text('import os\nclass X:\n    """Class doc."""\n')
    assert describe_expt(p) == "Class doc."
    assert describe_expt(tmp_path / "missing.py") == ""


def test_describe_expt_unwraps_paragraphs(tmp_path):
    p = tmp_path / "x.py"
    p.write_text('"""One line\n   wrapped.\n\nSecond\nparagraph."""\n')
    assert describe_expt(p) == "One line wrapped.\n\nSecond paragraph."


# --- through the server's request handler -------------------------------------------

class Recorder:
    def __init__(self, *a, **k):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


@pytest.fixture
def server(qapp, monkeypatch, tmp_path, reset_file):
    monkeypatch.setattr(msg, "StateBroadcaster", Recorder)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    path = tmp_path / "state.json"
    path.write_text(json.dumps(STATE))
    s = msg.MonitorUDPServer(config_file_path=str(path), reset_expt_path=str(reset_file))
    yield s
    s.reset.join(2)
    s.sock.close()


def _ask(server, obj):
    return json.loads(server.generate_reply(json.dumps(obj)))


def _end_state(server, run_id, expt):
    return _ask(server, {"type": "replace_state", "run_id": run_id, "expt": expt,
                         "config": {k: STATE[k] for k in ("dds", "ttl", "dac")}})


def test_reset_through_the_server_trusts_only_by_its_end_state(server):
    server._set_trust(False, "run 5 (x) took the core")
    release = threading.Event()
    server.reset._spawn = lambda command: FakeProc(["Done!"], 0, release)
    reply = _ask(server, {"type": "reset_state", "operator": "ada", "client": "pc2"})
    assert reply["status"] == "ok"
    detail = json.loads(server.generate_reply("status_json"))
    assert detail["reset"]["state"] == "running" and detail["reset"]["expt"] == "mot_observe"
    assert detail["trust"]["trusted"] is False            # starting it trusts nothing
    assert _end_state(server, 0, "mot_observe")["status"] == "ok"
    assert server._trust == {"trusted": True, "since": server._trust["since"],
                             "reason": "end state of mot_observe, the state reset "
                                       "requested by ada@pc2"}
    release.set()
    server.reset.join(2)
    assert server.reset.info()["state"] == "done"
    sent = [p["reset"]["state"] for p in server._broadcaster.sent if p.get("type") == "reset_run"]
    assert sent == ["running", "running", "done"]


def test_a_failed_reset_leaves_the_state_untrusted(server):
    server._set_trust(False, "run 5 (x) took the core")
    server.reset._spawn = lambda command: FakeProc(["boom"], 1)
    assert _ask(server, {"type": "reset_state", "operator": "ada"})["status"] == "ok"
    server.reset.join(2)
    assert server.reset.info()["state"] == "failed"
    assert server._trust["trusted"] is False


def test_reset_is_refused_while_a_run_is_starting(server):
    server.reset._spawn = lambda command: pytest.fail("spawned")
    _ask(server, {"type": "run_pending", "run_id": 81000, "expt": "hf_bec", "token": "t"})
    reply = _ask(server, {"type": "reset_state", "operator": "ada"})
    assert reply["status"] == "error" and "81000" in reply["msg"]
    assert server.journal.tail(1)[0]["kind"] == "state_reset_refused"


def test_a_run_without_an_id_is_named_by_its_file(server):
    _ask(server, {"type": "run_pending", "run_id": 0, "expt": "mot_observe", "token": "t"})
    server.on_monitor_state(STATES.NOT_READY, "interrupted_by_run")
    assert server._trust["reason"].startswith("mot_observe took the core")
    _end_state(server, 81001, "hf_bec")
    assert server._trust["reason"] == "end state of run 81001 (hf_bec)"


def test_no_reset_configured(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(msg, "StateBroadcaster", Recorder)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    s = msg.MonitorUDPServer(config_file_path=str(tmp_path / "state.json"))
    try:
        assert json.loads(s.generate_reply("status_json"))["reset"] is None
        reply = _ask(s, {"type": "reset_state"})
        assert reply["status"] == "error" and "no reset experiment" in reply["msg"]
    finally:
        s.sock.close()


# --- GUI ---------------------------------------------------------------------------

UNTRUSTED = {"trusted": False, "reason": "run 5 (x) took the core"}
MOT_ABOUT = "MOT Observe: puts the machine in its MOT-loading idle state.\n\nInner coil ON."


def test_reset_title_is_the_docstring_title_else_the_file_name():
    assert reset_title({"expt": "mot_observe", "about": MOT_ABOUT}) == "MOT Observe"
    assert reset_title({"expt": "mot_observe", "about": "No title here."}) == "mot observe"
    assert reset_title({"expt": "mot_observe"}) == "mot observe"


def test_trust_banner_is_short_and_offers_reset_only_when_the_server_has_one(qapp):
    strip = SummaryStrip()
    b = strip.banners["trust"]
    strip.set_trust(UNTRUSTED, {"expt": "mot_observe", "state": "idle", "about": MOT_ABOUT})
    assert not b.isHidden()
    assert b.label.text() == "Device state untrusted: the tabs may not match the hardware."
    assert "run 5 (x) took the core" in b.label.toolTip()      # why: tooltip only
    assert b.alt_button.text() == "Run MOT Observe" and not b.alt_button.isHidden()
    assert b.button.text() == "Trust state…" and not b.button.isHidden()
    strip.set_trust(UNTRUSTED, None)
    assert b.alt_button.isHidden() and not b.button.isHidden()
    strip.set_trust({"trusted": True}, {"expt": "mot_observe", "state": "idle"})
    assert b.isHidden()


def test_trust_banner_while_a_reset_runs_and_after_it_failed(qapp):
    strip = SummaryStrip()
    b = strip.banners["trust"]
    strip.set_trust(UNTRUSTED, {"expt": "mot_observe", "state": "running", "operator": "ada",
                                "client": "pc2", "started": 0., "about": MOT_ABOUT})
    assert "running MOT Observe (ada@pc2" in b.label.text()
    assert b.alt_button.isHidden() and b.button.isHidden()
    strip.set_trust(UNTRUSTED, {"expt": "mot_observe", "state": "failed", "about": MOT_ABOUT,
                                "text": "mot_observe exited with code 1", "tail": ["boom"]})
    assert b.label.text().endswith("Last MOT Observe failed.")
    assert "mot_observe exited with code 1" in b.label.toolTip()
    assert "boom" in b.label.toolTip() and not b.alt_button.isHidden()


@pytest.fixture
def gui(qapp, monkeypatch):
    monkeypatch.setattr(dc, "QSettings", lambda *a, **k: SimpleNamespace(
        value=lambda key, default=None, type=None: default, setValue=lambda *a: None))
    for name in ("_setup_update_sender", "_setup_state_listener", "_setup_state_worker",
                 "setup_status_checker", "setup_timer", "request_state"):
        monkeypatch.setattr(dc.DeviceStateGUI, name, lambda self, *a, **k: None)
    g = dc.DeviceStateGUI()
    yield g
    g.close()


def test_gui_logs_a_reset_once_per_transition(gui):
    gui._set_reset({"expt": "mot_observe", "state": "done", "started": 1., "text": "old"})
    assert not gui._changes                     # first news of an old reset
    running = {"expt": "mot_observe", "state": "running", "started": 2., "operator": "ada"}
    gui._set_reset(running)
    gui._set_reset(dict(running, text="mot_observe reported its end state"))
    gui._set_reset(dict(running, state="failed", text="it broke"))
    lines = [line.split("  ", 1)[1] for line in gui._changes]
    assert lines == ["[reset] mot_observe started by ada", "[reset] FAILED: it broke"]


class FakeBox:
    """QMessageBox stand-in: records the text, clicks the first button."""
    last = None

    class Icon:
        Warning = 0

    class ButtonRole:
        AcceptRole, RejectRole = 0, 1

    def __init__(self, parent=None):
        self.buttons = []
        FakeBox.last = self

    def setIcon(self, icon):
        pass

    def setWindowTitle(self, title):
        pass

    def setText(self, text):
        self.text = text

    def setDetailedText(self, text):
        self.detailed = text

    def addButton(self, text, role):
        button = SimpleNamespace(text=text)
        self.buttons.append(button)
        return button

    def setDefaultButton(self, button):
        self.default = button

    def exec(self):
        pass

    def clickedButton(self):
        return self.buttons[0]


def test_gui_reset_warns_about_a_live_run_and_sends_the_request(gui, monkeypatch):
    monkeypatch.setattr(dc, "QMessageBox", FakeBox)
    sent = []
    monkeypatch.setattr(gui, "_send_request", lambda obj, cb: sent.append(obj))
    gui._reset = {"expt": "mot_observe", "state": "idle", "about": MOT_ABOUT}
    gui._telemetry_samples = {
        "live_od/run_in_progress": SimpleNamespace(value=True, ok=True, age_s=1.),
        "live_od/run_id": SimpleNamespace(value=81000, ok=True, age_s=1.)}
    gui._reset_state()
    box = FakeBox.last
    assert "run 81000" in box.text and "IN PROGRESS" in box.text
    # the dialog shows the docstring's first paragraph; the rest is under Details
    assert "MOT-loading idle state" in box.text and "Inner coil ON." not in box.text
    assert box.detailed == MOT_ABOUT
    assert box.buttons[0].text == "Run anyway" and box.default is box.buttons[1]
    assert sent and sent[0]["type"] == "reset_state"
    gui._telemetry_samples = {}
    sent.clear()
    gui._reset_state()
    assert FakeBox.last.buttons[0].text == "Run MOT Observe" and sent
    # nothing is sent while a reset runs
    sent.clear()
    gui._reset = {"expt": "mot_observe", "state": "running"}
    gui._reset_state()
    assert not sent


def test_status_row_reset_button_follows_the_server(gui):
    b = gui.reset_button
    assert b.isHidden()                                   # no reset info yet
    gui._set_monitor_state(STATES.READY)
    gui._set_reset({"expt": "mot_observe", "state": "idle", "about": MOT_ABOUT})
    assert not b.isHidden() and b.isEnabled() and b.text() == "Run MOT Observe"
    gui._set_reset({"expt": "mot_observe", "state": "running", "started": 1.,
                    "about": MOT_ABOUT})
    assert b.text() == "Running MOT Observe…" and not b.isEnabled()
    gui._set_reset({"expt": "mot_observe", "state": "done", "started": 1.,
                    "about": MOT_ABOUT})
    assert b.isEnabled()
    gui.on_connection_failed()
    assert not b.isEnabled()
    gui._set_reset(None)
    assert b.isHidden()
