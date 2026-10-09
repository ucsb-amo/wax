"""The monitor state panel (waxx.util.guis.monitor_state_panel) and the
monitor server's own window with its Queue | State | Monitor tabs.

Offscreen Qt.  The panel's requester is a fake (no socket may be opened by
the panel tests: socket.socket is replaced by one that fails the test).  The
window test builds the real MonitorUDPServer in-process -- its broadcaster
faked, its run() (bind, beacon, accept loop) replaced by a no-op, beacon
start made to fail the test, discovery answering "nobody", the monitor
manager a fake -- so nothing listens, beacons or launches."""
import os
import socket
import time

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from waxx.util.comms_server.comm_server import STATES
from waxx.util.guis import monitor_state_panel as msp

NOW = time.time()

LOOP = {"key": "auto_tof", "title": "BEC TOF loop", "expt": "auto_tof",
        "path": "C:/code/auto_tof.py", "about": "BEC TOF loop: t_tof 1-4 ms.",
        "state": "running", "text": "run 85601 in progress (2nd of the loop)", "runs": 1,
        "run_id": 85601, "last": None, "started": NOW - 300, "owner": "person"}

STATUS = {"state": 1, "state_name": "NOT_READY", "sub_state": "interrupted_by_run",
          "reason": "", "since": NOW - 30, "pid": None, "expt_path": "C:/code/monitor.py",
          "trust": {"trusted": False, "reason": "run 85601 (auto_tof) took the core at 14:00 "
                                                "and has not reported its end state",
                    "since": NOW - 30},
          "run_pending": {"run_id": 85602, "expt": "rabi", "client": "kong",
                          "since": NOW - 12, "token": "tok"},
          "connections": {"awg": {"label": "Tweezer AWG", "state": "disconnected",
                                  "detail": "released for run 85601", "since": NOW - 30,
                                  "want": True, "tooltip": ""}},
          "slm_reinit": {"label": "SLM reinit", "state": "due", "detail": "",
                         "blocked_by": "run 85602 (rabi) is starting", "reinit_due": True,
                         "next_due_at": None, "last_reinit": {"at": NOW - 4000}},
          "run_loops": {"auto_tof": LOOP}, "run_queue": {"state": "idle"},
          "person_hold": {"active": False}}


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class _NoSocket:
    def __init__(self, *a, **k):
        raise AssertionError("the panel opened a socket")


class FakeServer:
    def __init__(self):
        self.requests = []
        self.answers = {}

    def __call__(self, obj):
        self.requests.append(dict(obj))
        answer = self.answers.get(obj.get("type"))
        return answer(obj) if callable(answer) else (answer or {"status": "ok"})


@pytest.fixture
def panel(qapp, monkeypatch):
    monkeypatch.setattr(socket, "socket", _NoSocket)
    server = FakeServer()
    server.answers["get_journal"] = {"status": "ok", "path": "C:/logs/ops_journal_x.jsonl",
                                     "entries": [
        {"t": "2026-10-09 14:00:01", "kind": "run_pending", "run_id": 85602, "expt": "rabi"},
        {"t": "2026-10-09 14:00:02", "kind": "trust", "trusted": False, "reason": "took"}]}
    server.answers["run_loop"] = lambda obj: {"status": "ok", "loop": dict(LOOP)}
    p = msp.MonitorStatePanel(server, synchronous=True, by="jp@test")
    p.server = server
    p.sequences.confirm = lambda title, text, verb="Send": True
    yield p
    p.shutdown()


def test_renders_monitor_trust_fence_connections_slm(panel):
    panel.set_state(dict(STATUS))
    assert panel.monitor_pill.text() == "NOT READY"
    assert "interrupted by run" in panel.monitor_line.text()
    assert "UNTRUSTED" in panel.trust_banner.text()
    assert "took the core" in panel.trust_banner.toolTip()
    assert "took the core" in panel.trust_line.text()
    fence = panel.fence_line.text()
    assert "run 85602 (rabi) from kong" in fence and "held 12 s" in fence
    assert "NOT_READY, so it does not lapse" in fence
    pill, line = panel.conn_rows["awg"]
    assert pill.text() == "disconnected" and "released for run 85601" in line.text()
    assert "blocked by: run 85602" in panel.slm_line.text() and "reinit due" in \
        panel.slm_line.text()
    card = panel.sequences.loop_cards["auto_tof"]
    assert card.pill.text() == "RUNNING" and "owner person" in card.status.text()


def test_trusted_ready_no_fence(panel):
    panel.set_state(dict(STATUS, state=0, state_name="READY", sub_state="running",
                         trust={"trusted": True, "reason": "end state of run 85601",
                                "since": NOW}, run_pending=None, connections={}))
    assert panel.monitor_pill.text() == "READY"
    assert panel.trust_banner.text() == "Device state trusted."
    assert panel.fence_line.text().startswith("No run fence")
    assert panel.conn_rows == {} and "holds no connections" in panel.no_conn.text()
    assert msp.fence_text({"run_id": 1, "since": NOW - 5}, "READY", NOW).endswith(
        "it lapses if the run never takes the core.")


def test_unreachable(panel):
    panel.set_state(dict(STATUS))
    panel.set_state(None)
    assert panel.monitor_pill.text() == "?" and "not answering" in panel.monitor_line.text()
    assert not panel.sequences.reachable


def test_loop_start_and_stop_go_through_the_requester_as_a_person(panel):
    panel.set_state(dict(STATUS, run_loops={"auto_tof": dict(LOOP, state="idle")}))
    card = panel.sequences.loop_cards["auto_tof"]
    card.start_button.click()
    req = [r for r in panel.server.requests if r.get("type") == "run_loop"][-1]
    assert (req["action"], req["loop"], req["owner"]) == ("start", "auto_tof", "person")
    assert card.pill.text() == "RUNNING"
    card.stop_button.click()
    req = [r for r in panel.server.requests if r.get("type") == "run_loop"][-1]
    assert (req["action"], req["loop"]) == ("stop", "auto_tof")


def test_broadcasts_update_at_once(panel):
    panel.set_state(dict(STATUS))
    panel.on_broadcast({"type": "trust", "trust": {"trusted": True, "reason": "ok"}})
    assert panel.trust_banner.text() == "Device state trusted."
    panel.on_broadcast({"type": "run_pending", "run_pending": None})
    assert panel.fence_line.text().startswith("No run fence")
    panel.on_broadcast({"type": "connections", "connections": {"awg": dict(
        STATUS["connections"]["awg"], state="connected", detail="")}})
    assert panel.conn_rows["awg"][0].text() == "connected"
    panel.on_broadcast({"type": "run_loop", "loop": dict(LOOP, state="stopped", ended=NOW,
                                                         text="stopped")})
    assert panel.sequences.loop_cards["auto_tof"].pill.text() == "stopped"
    assert panel._journal_soon.isActive()                         # the journal, shortly


def test_journal(panel, qapp):
    panel.set_state(dict(STATUS))
    panel.show()
    qapp.processEvents()
    asked = [r for r in panel.server.requests if r.get("type") == "get_journal"]
    assert asked and asked[-1] == {"type": "get_journal", "n": msp.JOURNAL_LINES}
    text = panel.journal.toPlainText()
    assert "run 85602 (rabi) starting" in text and "UNTRUSTED: took" in text
    assert "ops_journal_x.jsonl" in panel.journal_note.text()
    panel.server.answers["get_journal"] = {"status": "error", "msg": "unknown type get_journal"}
    panel.refresh_journal()
    assert not panel.journal_supported and not panel.journal_button.isEnabled()
    n = len(panel.server.requests)
    panel.refresh_journal()
    assert len(panel.server.requests) == n
    panel.hide()


# --- the monitor server's window ------------------------------------------------------------

class Broadcasts:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


class FakeManager(QObject):
    msg = pyqtSignal(str)
    monitor_stopped = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path, self.starts, self.running = path, 0, False
        self.pid, self.last_stop_kind, self.last_stop_reason = None, None, None

    def preflight_problems(self):
        return []

    def isRunning(self):                                           # noqa: N802
        return self.running

    def start(self):
        self.starts += 1

    def stop(self):
        self.running = False


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    from waxx.util.guis import monitor_server_gui as msg

    def no_beacon(self):
        raise AssertionError("a test started a beacon")
    monkeypatch.setattr(msg, "StateBroadcaster", Broadcasts)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    monkeypatch.setattr(msg, "discover", lambda *a, **k: None)
    monkeypatch.setattr(msg, "MonitorManager", FakeManager)
    monkeypatch.setattr(msg.MonitorUDPServer, "run", lambda self: None)
    monkeypatch.setattr(msg.MonitorUDPServer, "_start_beacon", no_beacon)
    w = msg.MonitorServerGUI(str(tmp_path / "monitor.py"),
                             config_file_path=str(tmp_path / "state.json"),
                             journal_dir=str(tmp_path / "ops_journal"))
    w.request_runner.synchronous = True                            # answers at once
    yield w
    w.close()


def test_the_window_has_queue_state_and_monitor_tabs(window, qapp):
    tabs = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert tabs == ["Queue", "State", "Monitor"]
    assert window.tabs.widget(2).isAncestorOf(window.status_indicator)
    # the existing content still works: NOT READY -> click starts the monitor
    assert window.status_indicator.text() == "NOT READY"
    window.status_indicator.click()
    assert window.monitor_manager.starts == 1


def test_the_window_feeds_its_panels_by_direct_calls(window, qapp):
    window.show()
    qapp.processEvents()
    window.poll_status()
    qp, sp = window.queue_panel, window.state_panel
    assert qp.reachable and qp.has_queue and qp.queue_state() == "idle"
    assert sp.monitor_pill.text() == "NOT READY"
    assert window.direct_request({"type": "run_queue", "action": "list"})["status"] == "ok"
    # a hold put on from the panel reaches the server, and its broadcast the panel
    qp.ask_text = lambda *a, **k: "aligning"
    qp.toggle_hold()
    assert window.udp_server.person_hold.info()["active"]
    assert window.udp_server.person_hold.info()["owner"] == "person"
    assert any(p.get("type") == "person_hold" for p in window.udp_server._broadcaster.sent)
    qapp.processEvents()                                           # the tapped broadcast
    assert qp.hold_button.text() == "Release hold"
    qp.toggle_hold()
    assert not window.udp_server.person_hold.info()["active"]
    window.hide()
