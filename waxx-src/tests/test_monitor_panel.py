"""The Server Dashboard's monitor panel (waxx.util.guis.monitor_panel): its
link to the server (lazy discovery, a dropped client rediscovered, writes
sent once, reads retried), the status poll and broadcasts feeding the Queue,
State and Monitor tabs, the monitor experiment's button, and cleanup.

Offscreen Qt.  The MonitorClient is a fake answering from a table: no
discovery, no socket (socket.socket is replaced by one that fails the test),
no broadcast listener (listener_factory=None, or a fake)."""
import os
import socket

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from waxx.util.guis import monitor_panel as mp

STATUS = {"state": 2, "state_name": "NOT_READY", "sub_state": "never_started", "reason": "",
          "since": 0, "pid": None, "expt_path": "C:/code/monitor.py",
          "trust": {"trusted": True, "reason": "", "since": 0}, "run_pending": None,
          "connections": {}, "run_loops": {}, "slm_reinit": None,
          "person_hold": {"active": False},
          "run_queue": {"enabled": True, "state": "idle", "text": "no jobs", "current": None,
                        "next": [], "counts": {"queued": 0}, "alarm": None,
                        "paused": {"agent": None, "all": None},
                        "person_hold": {"active": False}, "resume_loop": None}}


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


class FakeClient:
    """A MonitorClient's three calls, answered from tables."""

    def __init__(self, n):
        self.n = n
        self.requests, self.texts = [], []
        self.status = dict(STATUS)
        self.answers = {"list": {"status": "ok", "jobs": [], "next": [],
                                 "run_queue": STATUS["run_queue"]},
                        "get_journal": {"status": "ok", "entries": []}}
        self.silent = False

    def request(self, obj, timeout=5.0, attempts=2):
        self.requests.append((dict(obj), attempts))
        if self.silent:
            return None
        return self.answers.get(obj.get("action") or obj.get("type"), {"status": "ok"})

    def get_status(self):
        return None if self.silent else dict(self.status)

    def send_message(self, text, timeout=5.0, attempts=2):
        self.texts.append((text, attempts))
        return None if self.silent else "2"


class Factory:
    def __init__(self, fail_first=0):
        self.made, self.fail_first = [], fail_first

    def __call__(self, discovery_timeout):
        if self.fail_first:
            self.fail_first -= 1
            raise RuntimeError("no monitor server beaconing")
        client = FakeClient(len(self.made))
        self.made.append(client)
        return client


class FakeListener(QObject):
    state_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def wait(self, *a):
        return True


@pytest.fixture
def factory():
    return Factory()


@pytest.fixture
def panel(qapp, factory):
    listeners = []

    def make_listener():
        listeners.append(FakeListener())
        return listeners[-1]
    p = mp.MonitorServerPanel(link=mp.MonitorLink(factory), listener_factory=make_listener,
                              synchronous=True, by="jp@test")
    p.listeners = listeners
    p.confirms = []
    p.confirm = lambda title, text: p.confirms.append((title, text)) or True
    yield p
    p.cleanup()


def test_no_discovery_until_the_first_request(qapp, factory):
    p = mp.MonitorServerPanel(link=mp.MonitorLink(factory), listener_factory=None,
                              synchronous=True)
    assert factory.made == []                                    # built, nothing discovered
    assert [p.tabs.tabText(i) for i in range(p.tabs.count())] == ["Queue", "State", "Monitor"]
    p.cleanup()
    p.cleanup()                                                  # twice is harmless


def test_show_polls_lists_listens_and_feeds_the_tabs(panel, factory, qapp):
    panel.show()
    qapp.processEvents()
    client = factory.made[0]
    assert panel.listeners and panel.listeners[0].started
    assert panel.queue_panel.reachable and panel.queue_panel.queue_state() == "idle"
    assert panel.state_panel.monitor_pill.text() == "NOT READY"
    assert panel.monitor_tab.button.text() == "NOT READY"
    sent = [obj for obj, _ in client.requests]
    assert {"type": "run_queue", "action": "list", "limit": 200} in sent
    # a broadcast from the listener reaches the tabs
    panel.listeners[0].state_received.emit({"type": "person_hold",
                                            "person_hold": {"active": True, "by": "x"}})
    assert panel.queue_panel.hold_button.text() == "Release hold"
    panel.listeners[0].state_received.emit({"type": "trust", "trust": {"trusted": False,
                                                                       "reason": "r"}})
    assert "UNTRUSTED" in panel.state_panel.trust_banner.text()
    panel.hide()


def test_writes_once_reads_retried(panel, factory):
    panel.poll_status(force=True)
    client = factory.made[0]
    panel.queue_panel.ask_text = lambda *a, **k: "mine"
    panel.queue_panel.toggle_hold()
    panel.queue_panel.refresh_list()
    by_action = {obj.get("action"): attempts for obj, attempts in client.requests}
    assert by_action["hold"] == 1 and by_action["list"] == 2
    hold = [obj for obj, _ in client.requests if obj.get("action") == "hold"][-1]
    assert hold == {"type": "run_queue", "action": "hold", "reason": "mine", "owner": "person",
                    "by": "jp@test"}


def test_a_silent_server_is_rediscovered(panel, factory):
    panel.poll_status(force=True)
    first = factory.made[0]
    first.silent = True
    panel.poll_status(force=True)                                # no answer: client dropped
    assert not panel.queue_panel.reachable
    assert panel.monitor_tab.button.text() == "monitor server not answering"
    panel.poll_status(force=True)                                # a fresh discovery
    assert len(factory.made) == 2 and panel.queue_panel.reachable


def test_no_server_found_is_an_error_reply_not_a_crash(qapp):
    link = mp.MonitorLink(Factory(fail_first=1))
    reply = link.request({"type": "run_queue", "action": "list"})
    assert reply["status"] == "error" and "no monitor server found" in reply["msg"]
    assert link.request({"type": "run_queue", "action": "list"})["status"] == "ok"


def test_the_monitor_button(panel, factory):
    panel.poll_status(force=True)
    client = factory.made[0]
    assert panel.monitor_tab.clicked()                            # NOT READY: start, no ask
    assert client.texts == [("reset", 1)] and panel.confirms == []
    client.status = dict(STATUS, state=0, state_name="READY", sub_state="running")
    panel.poll_status(force=True)
    assert panel.monitor_tab.button.text() == "READY"
    panel.confirm = lambda title, text: False
    assert not panel.monitor_tab.clicked()                        # READY: asks; declined
    assert len(client.texts) == 1
    client.status = dict(STATUS, state=1, state_name="LOADING", sub_state="starting")
    panel.poll_status(force=True)
    assert not panel.monitor_tab.button.isEnabled() and not panel.monitor_tab.clicked()


def test_cleanup_stops_the_poll_and_the_listener(panel, qapp):
    panel.show()
    qapp.processEvents()
    listener = panel.listeners[0]
    panel.cleanup()
    assert listener.stopped and not panel.timer.isActive()


def test_constructing_the_panel_binds_nothing():
    """Importing and building the panel (default listener factory, never
    shown) must not import beacon.discovery: its package starts the
    discovery listener, a socket bound to the discovery port, at import.
    Checked in a fresh interpreter (this one has imported it already)."""
    import subprocess
    import sys
    code = (
        "import sys, os\n"
        "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
        "from PyQt6.QtWidgets import QApplication\n"
        "app = QApplication([])\n"
        "from waxx.util.guis import monitor_panel as mp\n"
        "class C:\n"
        "    def request(self, obj, timeout=5.0, attempts=2): return None\n"
        "    def get_status(self): return None\n"
        "    def send_message(self, text, timeout=5.0, attempts=2): return None\n"
        "p = mp.MonitorServerPanel(link=mp.MonitorLink(lambda t: C()), synchronous=True)\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('beacon.discovery')\n"
        "                or m.startswith('waxx.util.comms_server'))\n"
        "p.cleanup()\n"
        "print('LOADED', loaded)\n")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "LOADED []" in out.stdout, out.stdout


@pytest.mark.parametrize("extra, word", [
    ({"sub_state": "interrupted_by_run"}, "interrupted the monitor"),
    ({"run_pending": {"run_id": 85700, "expt": "rabi", "client": "kong", "since": 0}},
     "run 85700 (rabi) has announced itself"),
    ({"run_queue": dict(STATUS["run_queue"], current={"id": 4, "label": "scan",
                                                      "state": "running", "run_id": 85701})},
     "job 4 (scan) is running, run 85701"),
    ({"live_od": {"run_in_progress": True, "run_id": 85702, "expt_name": "tof"}},
     "liveOD says run 85702 (tof) is in progress"),
])
def test_start_asks_whenever_the_core_may_be_held(panel, factory, extra, word):
    """BL-1: NOT READY + any sign the core may be held -> Device Control's
    question first; declined, nothing is sent."""
    panel.poll_status(force=True)
    client = factory.made[0]
    client.status = dict(STATUS, **extra)
    panel.poll_status(force=True)
    asked = []
    panel.confirm = lambda title, text: asked.append((title, text)) or False
    assert not panel.monitor_tab.clicked()
    title, text = asked[-1]
    assert title == "Start monitor"
    assert "An experiment probably holds the core device." in text
    assert "takes the core and cuts that experiment off" in text and word in text
    assert client.texts == []
    panel.confirm = lambda title, text: True
    assert panel.monitor_tab.clicked() and client.texts == [("reset", 1)]


def test_restart_names_what_holds_the_core(panel, factory):
    panel.poll_status(force=True)
    client = factory.made[0]
    client.status = dict(STATUS, state=0, state_name="READY", sub_state="running",
                         run_pending={"run_id": 85703, "expt": "rabi"})
    panel.poll_status(force=True)
    asked = []
    panel.confirm = lambda title, text: asked.append((title, text)) or False
    assert not panel.monitor_tab.clicked()
    assert asked[-1][0] == "Restart monitor"
    assert "this will interrupt it" in asked[-1][1] and "run 85703 (rabi)" in asked[-1][1]


def test_a_double_click_sends_one_reset(panel, factory):
    """S-1: while a reset is in flight the button is disabled and a second
    click sends nothing; the reply re-enables it."""
    panel.poll_status(force=True)
    client = factory.made[0]
    queued = []
    real_call = panel.runner.call
    panel.runner.call = lambda fn, callback=None: queued.append((fn, callback))  # held
    tab = panel.monitor_tab
    tab.button.click()
    tab.button.click()
    assert len(queued) == 1 and tab.in_flight and not tab.button.isEnabled()
    panel._polling = False
    panel.poll_status(force=True)                                 # a poll meanwhile
    assert len(queued) == 2                                       # (the poll, held too)
    panel.runner.call = real_call
    fn, done = queued[0]
    done(fn())                                                    # the reset's reply
    assert client.texts == [("reset", 1)]
    assert not tab.in_flight and tab.button.isEnabled()


def test_a_shut_down_panel_ignores_broadcasts_and_state(panel, factory, qapp):
    panel.show()
    qapp.processEvents()
    client = factory.made[0]
    panel.cleanup()
    n = len(client.requests)
    panel.on_broadcast({"type": "run_queue", "run_queue": STATUS["run_queue"]})
    panel.queue_panel.on_broadcast({"type": "run_queue", "run_queue": STATUS["run_queue"]})
    panel.queue_panel.set_state(dict(STATUS))
    panel.state_panel.on_broadcast({"type": "trust", "trust": {"trusted": False}})
    panel.state_panel.set_state(dict(STATUS))
    panel.state_panel.refresh_journal()
    panel.queue_panel.refresh_list()
    assert not panel.queue_panel._list_timer.isActive()
    assert not panel.state_panel._journal_soon.isActive()
    assert len(client.requests) == n                               # nothing sent
    panel.hide()


def test_a_worker_stuck_in_a_request_is_reported_at_shutdown(qapp, caplog):
    import threading
    import logging
    from waxx.util.guis.request_runner import RequestRunner
    release = threading.Event()
    runner = RequestRunner(lambda obj: release.wait(5) and {"status": "ok"})
    runner.send({"type": "get_version"})
    with caplog.at_level(logging.WARNING, logger="waxx.util.guis.request_runner"):
        runner.shutdown(timeout=0.2)
    release.set()
    assert any("still waiting on a request" in r.getMessage() for r in caplog.records)
