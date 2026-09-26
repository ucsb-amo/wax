"""Host-side connections the monitor server holds between runs
(waxx.util.device_state.connections): the service's rules, the server's
hooks and requests, the monitor's connection_call, and the Composite tab's
connection bar.

Nothing touches the network or hardware: agents are in-process fakes (the
real agent process is covered by test_connection_agent.py), the server's UDP
broadcaster is a recorder, the monitor's server client is a fake, and Qt
runs offscreen.
"""
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication

from waxx.base import monitor as wmon
from waxx.util.comms_server.comm_server import STATES
from waxx.util.device_state import connections as conns
from waxx.util.device_state.composite import Context
from waxx.util.device_state.connection_agent import AgentCommandError, AgentDied
from waxx.util.device_state.connections import Connection, ConnectionService
from waxx.util.guis import monitor_server_gui as msg


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _wait(pred, timeout=3.):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


# --- a fake device behind a fake agent ---------------------------------------------------

class Card:
    def __init__(self):
        self.open = False
        self.opens = self.closes = self.kills = self.agents = 0
        self.fail_open = None            # message the driver raises on open
        self.close_result = True
        self.block_open = False          # open waits until the agent is killed
        self.tones = []


class FakeAgent:
    def __init__(self, card, on_exit):
        self.card = card
        self.on_exit = on_exit
        self.alive = False
        self.expected_exit = False
        self.killed = threading.Event()
        self.state = {"open": False, "detail": ""}
        card.agents += 1

    def _refresh(self):
        c = self.card
        self.state = {"open": c.open,
                      "detail": (f"{len(c.tones)} tone(s)" if c.tones else "no tones")
                      if c.open else ""}

    def start(self, timeout):
        if self.killed.is_set():
            raise AgentDied("stopped")
        self.alive = True

    def call(self, cmd, timeout, **kw):
        c = self.card
        if not self.alive or self.killed.is_set():
            raise AgentDied("not running")
        if cmd == "open":
            if c.block_open and self.killed.wait(timeout):
                raise AgentDied("stopped")
            if c.fail_open:
                raise AgentCommandError(c.fail_open, {"open": False, "detail": ""})
            c.open = True
            c.opens += 1
        elif cmd == "close":
            c.open = False
            c.closes += 1
            self._refresh()
            return c.close_result
        elif cmd == "write_traps":
            if kw["rows"] and kw["rows"][0][1] > 1:
                raise AgentCommandError("amplitudes sum to 2.000 > 1", self.state)
            c.tones = kw["rows"]
        self._refresh()
        return {"tones": len(c.tones)} if cmd == "write_traps" else None

    def exit(self, timeout):
        self.expected_exit = True
        self.alive = False
        return True

    def kill(self):
        self.expected_exit = True
        self.killed.set()
        self.alive = False
        self.card.kills += 1

    def die(self):                               # the process crashed
        self.alive = False
        self.card.open = False
        self.on_exit(self)


AWG = Connection(key="awg", label="Tweezer AWG", driver="fake:Driver", tooltip="netbox",
                 confirm="The card is stopped.")


class Changes:
    def __init__(self):
        self.seen = []

    def __call__(self, snapshot, changed):
        for key in changed:
            self.seen.append((key, snapshot[key]["state"], snapshot[key]["detail"]))

    def states(self):
        return [s for _, s, _ in self.seen]


@pytest.fixture
def svc():
    card, changes = Card(), Changes()
    agents = []

    def factory(conn, log, on_exit):
        agents.append(FakeAgent(card, on_exit))
        return agents[-1]
    s = ConnectionService([AWG], on_change=changes, agent_factory=factory)
    s.card, s.changes, s.agents = card, changes, agents
    s.start()
    yield s
    s.stop()


def _state(s):
    return s.snapshot()["awg"]


def _connected(s):
    s.run_over("the monitor is running", reopen=True)
    assert _wait(lambda: _state(s)["state"] == conns.CONNECTED)


# --- the service -----------------------------------------------------------------------------

def test_nothing_opens_until_the_machine_is_known_idle(svc):
    time.sleep(0.2)
    assert svc.card.agents == 0
    assert _state(svc)["detail"] == "opens when the monitor is running"


def test_opens_when_the_monitor_runs(svc):
    _connected(svc)
    assert svc.card.open and svc.card.opens == 1
    assert _state(svc)["detail"] == "no tones"
    assert svc.changes.states()[-2:] == [conns.CONNECTING, conns.CONNECTED]
    svc.run_over("the monitor is running", reopen=True)          # already open: no-op
    time.sleep(0.1)
    assert svc.card.opens == 1


def test_a_run_announcing_itself_gets_the_card_before_the_reply(svc):
    _connected(svc)
    svc.run_starting("run 81234 (hf_bec)")
    # synchronous: closed by the time run_starting returns
    assert not svc.card.open and svc.card.closes == 1 and svc.card.kills == 0
    s = _state(svc)
    assert s["state"] == conns.DISCONNECTED
    assert s["detail"] == "released for run 81234 (hf_bec), which opens it itself"
    assert not svc.agents[-1].alive


def test_a_release_during_an_open_stops_the_agent_within_the_bound(svc):
    svc.card.block_open = True
    svc.run_over("the monitor is running", reopen=True)
    assert _wait(lambda: _state(svc)["state"] == conns.CONNECTING)
    t0 = time.monotonic()
    svc.run_starting("run 7 (x)", timeout=conns.RELEASE_TIMEOUT_S)
    assert time.monotonic() - t0 < conns.RELEASE_TIMEOUT_S
    assert svc.card.kills == 1 and not svc.card.open
    assert _wait(lambda: _state(svc)["state"] == conns.DISCONNECTED)
    assert "released for run 7 (x)" in _state(svc)["detail"]


def test_a_close_that_does_not_finish_drops_the_connection_and_says_so(svc):
    _connected(svc)
    svc.card.close_result = False
    svc.run_starting("run 8 (y)")
    s = _state(svc)
    assert s["state"] == conns.DISCONNECTED and svc.card.kills == 1
    assert "dropped, the device not stopped" in s["detail"]


def test_after_a_run_ends_it_waits_for_the_monitor(svc):
    _connected(svc)
    svc.run_starting("run 9 (z)")
    svc.run_running("run 9 (z)")
    assert svc.request("awg", "connect") .startswith("run 9 (z) is running")
    svc.run_over("run 9 (z) ended", reopen=False)
    time.sleep(0.2)
    assert not svc.card.open
    assert _state(svc)["detail"] == "the run has ended; opens when the monitor is running"
    svc.run_over("the monitor is running", reopen=True)
    assert _wait(lambda: svc.card.open)


def test_a_reopen_while_open_does_not_linger_past_the_next_run(svc):
    """A reopen (or a GUI connect) that arrives while the device is already
    open must not leave a flag behind that reopens it the moment the next
    run ends -- a run loop's next run would then find it taken."""
    _connected(svc)
    svc.run_over("the monitor is running", reopen=True)         # already open
    svc.request("awg", "connect", "ada")                        # already open
    svc.run_starting("run 3 (loop)")
    svc.run_running("run 3 (loop)")
    svc.run_over("run 3 (loop) ended", reopen=False)
    time.sleep(0.3)
    assert not svc.card.open and svc.card.opens == 1


def test_a_run_that_did_not_announce_is_released_when_it_takes_the_core(svc):
    _connected(svc)
    svc.run_running("an experiment")
    assert _wait(lambda: not svc.card.open)
    assert "released for an experiment, which has the core" in _state(svc)["detail"]


def test_gui_requests(svc):
    _connected(svc)
    assert svc.request("awg", "disconnect", "ada@pc2") == ""
    assert _wait(lambda: not svc.card.open)
    assert _state(svc)["detail"] == "disconnected by ada@pc2" and not _state(svc)["want"]
    svc.run_over("the monitor is running", reopen=True)
    time.sleep(0.2)
    assert not svc.card.open                        # a disconnect is kept
    assert svc.request("awg", "connect", "ada@pc2") == ""
    assert _wait(lambda: svc.card.open)
    assert svc.request("nope", "connect").startswith("the monitor server has no connection")
    assert svc.request("awg", "explode").startswith("unknown action")
    svc.run_starting("run 1 (a)")
    assert "is starting" in svc.request("awg", "connect")


def test_a_failed_open_waits_to_be_asked_again(svc):
    svc.card.fail_open = "card is already in use (held by 192.168.1.90)"
    svc.run_over("the monitor is running", reopen=True)
    assert _wait(lambda: _state(svc)["state"] == conns.FAILED)
    assert "held by 192.168.1.90" in _state(svc)["detail"]
    time.sleep(0.2)
    assert svc.card.agents == 1 and svc.card.kills == 1     # its agent was stopped
    svc.card.fail_open = None
    svc.run_over("the monitor is running", reopen=True)
    assert _wait(lambda: svc.card.open)


def test_calls_go_to_the_open_device(svc):
    with pytest.raises(ConnectionRefusedError, match="not connected"):
        svc.call("awg", "write_traps", {"rows": [[72.e6, 0.1]]})
    _connected(svc)
    assert svc.call("awg", "write_traps", {"rows": [[72.e6, 0.1]]}) == {"tones": 1}
    assert svc.card.tones == [[72.e6, 0.1]] and _state(svc)["detail"] == "1 tone(s)"
    with pytest.raises(RuntimeError, match="sum to"):
        svc.call("awg", "write_traps", {"rows": [[72.e6, 2.0]]})
    assert _state(svc)["state"] == conns.CONNECTED          # a refused write keeps it
    svc.run_starting("run 2 (b)")
    with pytest.raises(ConnectionRefusedError, match="run 2"):
        svc.call("awg", "write_traps", {"rows": []})


def test_an_agent_that_dies_marks_the_connection_failed(svc):
    _connected(svc)
    svc.agents[-1].die()
    s = _state(svc)
    assert s["state"] == conns.FAILED and "exited unexpectedly" in s["detail"]


def test_stop_closes(svc):
    _connected(svc)
    svc.stop()
    assert not svc.card.open and svc.card.closes == 1
    assert _state(svc)["detail"] == "the monitor server is stopping"


def test_definitions_are_validated():
    with pytest.raises(ValueError, match="duplicate"):
        conns.validate_connections([AWG, AWG])
    with pytest.raises(ValueError, match="module:factory"):
        Connection(key="x", label="x", driver="nocolon").validate()
    with pytest.raises(ValueError, match="identifier"):
        Connection(key="a b", label="x", driver="m:f").validate()


# --- the server ----------------------------------------------------------------------------------

class Recorder:
    def __init__(self, *a, **k):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


@pytest.fixture
def server(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(msg, "StateBroadcaster", Recorder)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"dds": {}, "dac": {}, "ttl": {}}))
    s = msg.MonitorUDPServer(config_file_path=str(path))
    card = Card()
    s.connections = ConnectionService(
        [AWG], on_change=s._on_connections_change,
        agent_factory=lambda conn, log, on_exit: FakeAgent(card, on_exit))
    s.connections.start()
    s.card = card
    yield s
    s.connections.stop()
    s.sock.close()


def _ask(server, obj):
    return json.loads(server.generate_reply(json.dumps(obj)))


def _server_connected(server):
    server.on_monitor_state(STATES.NOT_READY, "never_started")
    server.on_monitor_state(STATES.READY, "running")
    assert _wait(lambda: server.card.open)


def test_the_monitor_running_opens_and_status_carries_it(server):
    _server_connected(server)
    assert _wait(lambda: json.loads(server.generate_reply("status_json"))
                 ["connections"]["awg"]["state"] == conns.CONNECTED)
    assert _ask(server, {"type": "get_state"})["connections"]["awg"]["state"] == conns.CONNECTED
    sent = [p for p in server._broadcaster.sent if p.get("type") == "connections"]
    assert sent and sent[-1]["connections"]["awg"]["state"] == conns.CONNECTED
    assert any(e["kind"] == "connection" for e in server.journal.tail(20))


def test_announce_releases_before_replying_and_the_run_keeps_it(server):
    _server_connected(server)
    assert _ask(server, {"type": "run_pending", "run_id": 81234, "expt": "hf_bec",
                         "token": "t"})["status"] == "ok"
    assert not server.card.open                                  # before the reply
    server.on_monitor_state(STATES.NOT_READY, "interrupted_by_run")   # took the core
    assert server._run_pending is None                           # the fence is lifted...
    time.sleep(0.2)
    assert not server.card.open                                  # ...the card stays released
    refused = _ask(server, {"type": "connection", "key": "awg", "action": "connect"})
    assert refused["status"] == "error" and "running" in refused["msg"]
    _ask(server, {"type": "replace_state", "run_id": 81234, "expt": "hf_bec",
                  "config": {"dds": {}, "ttl": {}, "dac": {}}})
    time.sleep(0.2)
    assert not server.card.open                                  # waits for the monitor
    server.on_monitor_state(STATES.LOADING, "starting")
    server.on_monitor_state(STATES.READY, "running")
    assert _wait(lambda: server.card.open)


def test_a_withdrawn_run_gives_the_card_back(server):
    _server_connected(server)
    _ask(server, {"type": "run_pending", "run_id": 5, "token": "tok"})
    assert not server.card.open
    _ask(server, {"type": "run_withdrawn", "token": "tok", "run_id": 5})
    assert _wait(lambda: server.card.open)


def test_a_monitor_restart_without_a_run_keeps_it_open(server):
    _server_connected(server)
    opens = server.card.opens
    server.on_monitor_state(STATES.NOT_READY, "stopped_on_request")
    server.on_monitor_state(STATES.LOADING, "starting")
    server.on_monitor_state(STATES.READY, "running")
    time.sleep(0.2)
    assert server.card.open and server.card.opens == opens and server.card.closes == 0


def test_connection_requests_and_calls(server):
    _server_connected(server)
    bad = _ask(server, {"type": "connection", "key": "awg", "action": "explode"})
    assert bad["status"] == "error"
    assert server.journal.tail(1)[0]["kind"] == "connection_refused"
    reply = _ask(server, {"type": "connection_call", "key": "awg", "cmd": "write_traps",
                          "kwargs": {"rows": [[72.e6, 0.2]]}})
    assert reply == {"status": "ok", "result": {"tones": 1}}
    assert _ask(server, {"type": "connection", "key": "awg", "action": "disconnect",
                         "operator": "ada"})["status"] == "ok"
    assert _wait(lambda: not server.card.open)
    reply = _ask(server, {"type": "connection_call", "key": "awg", "cmd": "write_traps",
                          "kwargs": {"rows": []}})
    assert reply["status"] == "error" and "not connected" in reply["msg"]


def test_bad_connection_definitions_do_not_stop_the_server(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(msg, "StateBroadcaster", Recorder)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    s = msg.MonitorUDPServer(config_file_path=str(tmp_path / "s.json"),
                             connections=[AWG, AWG])
    try:
        assert not s.connections and s.connections.snapshot() == {}
    finally:
        s.sock.close()


# --- the monitor's side ------------------------------------------------------------------------

class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def connection_call(self, key, cmd, kwargs=None, timeout=10.):
        self.calls.append((key, cmd, kwargs))
        return self.reply


def test_monitor_connection_call(monkeypatch, tmp_path):
    fake = FakeClient({"status": "ok", "result": {"tones": 2}})
    monkeypatch.setattr(wmon, "MonitorClient", lambda *a, **k: fake)
    m = wmon.Monitor(SimpleNamespace(), device_state_json_path=str(tmp_path / "s.json"))
    assert m.connection_call("awg", "write_traps", rows=[[72.e6, 0.1]]) == {"tones": 2}
    assert fake.calls == [("awg", "write_traps", {"rows": [[72.e6, 0.1]]})]
    fake.reply = {"status": "error", "msg": "the Tweezer AWG is not connected (failed)"}
    with pytest.raises(RuntimeError, match="not connected"):
        m.connection_call("awg", "write_traps", rows=[])
    fake.reply = None
    with pytest.raises(RuntimeError, match="did not answer"):
        m.connection_call("awg", "write_traps", rows=[])


# --- the connection bar ------------------------------------------------------------------------

# liveOD's camera button colours: connected, not connected, failed
GREEN, GREY, RED = "#43a047", "#9e9e9e", "#c62828"
REPORT = {"awg": {"label": "Tweezer AWG", "state": conns.CONNECTED, "detail": "no tones",
                  "since": 1.0, "want": True, "tooltip": ""}}


@pytest.fixture
def panel(qapp, monkeypatch):
    from test_composite_panel import FakeSender, FakeSettings
    from waxx.util.guis import composite_panel as cp
    monkeypatch.setattr(cp, "QSettings", FakeSettings)
    monkeypatch.setattr(cp, "_OpSender", FakeSender)
    p = cp.CompositePanel([], connections=[AWG], start_sender=False)
    p.answers = []
    p.answer = True
    p.confirm = lambda title, text, verb="Send", danger=False: (
        p.answers.append((title, text, verb)) or p.answer)
    p.set_monitor_state(STATES.READY, reachable=True)
    yield p
    p.shutdown()


def _pill(panel):
    return panel.connection_bar._pills["awg"]


def test_bar_shows_each_connection_before_the_server_reports(panel):
    pill, detail = _pill(panel)
    assert panel.connection_bar.isVisibleTo(panel)
    assert pill.text() == "Tweezer AWG" and GREY in pill.styleSheet()
    assert "not reported" in detail.text()


def test_bar_colours_follow_the_reported_state(panel):
    pill, detail = _pill(panel)
    panel.set_connections(REPORT)
    assert GREEN in pill.styleSheet() and detail.text() == "no tones"
    assert "Click to disconnect" in pill.toolTip() and "netbox" in pill.toolTip()
    panel.set_connections({"awg": dict(REPORT["awg"], state=conns.FAILED, detail="x" * 300)})
    assert RED in pill.styleSheet()
    assert len(detail.text()) <= 90 and detail.text().endswith("…")
    assert detail.toolTip() == "x" * 300
    panel.set_connections({"awg": dict(REPORT["awg"], state=conns.DISCONNECTED)})
    assert GREY in pill.styleSheet()


def test_pill_click_sends_connect_or_confirmed_disconnect(panel):
    pill, _ = _pill(panel)
    pill.click()
    assert panel._sender.requests[-1]["type"] == "connection"
    assert panel._sender.requests[-1]["action"] == "connect"
    panel.set_connections(REPORT)
    panel.answer = False
    n = len(panel._sender.requests)
    pill.click()
    assert len(panel._sender.requests) == n                     # cancelled
    assert "The card is stopped." in panel.answers[-1][1]
    panel.answer = True
    pill.click()
    assert panel._sender.requests[-1]["action"] == "disconnect"
    assert panel._sender.requests[-1]["key"] == "awg"


def test_pills_work_without_the_monitor_but_not_while_a_run_starts(panel):
    pill, detail = _pill(panel)
    panel.set_monitor_state(STATES.NOT_READY, reachable=True)
    assert pill.isEnabled()                                     # the server holds it
    panel.set_run_pending({"run_id": 81234, "expt": "hf_bec"})
    assert not pill.isEnabled() and "run 81234 (hf_bec)" in pill.toolTip()
    panel.set_connections(REPORT)
    assert pill.isEnabled()                                     # disconnect still allowed
    panel.set_run_pending(None)
    panel.set_monitor_state(None, reachable=False)
    assert not pill.isEnabled() and detail.text() == "monitor server unreachable"


def test_status_detail_and_context_carry_the_connections(panel):
    panel.set_monitor_detail({"connections": REPORT})
    assert panel.connections == REPORT
    ctx = panel.context(None, {})
    assert isinstance(ctx, Context) and ctx.connection("awg")["state"] == conns.CONNECTED
    assert ctx.connection("nope") is None
