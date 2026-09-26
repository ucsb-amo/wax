"""Host-side connections the monitor holds (waxx.util.device_state.connections):
the manager's rules, the monitor's poll handling, the server's requests, the
graceful stop, and the Composite tab's connection bar.

Nothing touches the network or hardware: the "device" is a fake with a flag,
the monitor's server client is a fake, the server's UDP broadcaster is a
recorder, the process killer is a recorder, and Qt runs offscreen.
"""
import json
import os
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication

from waxx.base import monitor as wmon
from waxx.util.comms_server.comm_server import STATES
from waxx.util.device_state import connections as conns
from waxx.util.device_state import monitor_manager as mm
from waxx.util.device_state.composite import Context, OpTable
from waxx.util.device_state.connections import Connection, ConnectionManager
from waxx.util.device_state.op_queue import OpQueue
from waxx.util.guis import monitor_server_gui as msg


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


# --- a fake device ------------------------------------------------------------------

class Card:
    """Stands in for the AWG: open or not, with scripted failures."""

    def __init__(self):
        self.open = False
        self.fail_open = None          # exception to raise on open
        self.close_result = True       # False: the close "did not return"
        self.opens = 0
        self.closes = 0


def _connection(card, **kw):
    def connect(expt):
        card.opens += 1
        if card.fail_open is not None:
            raise card.fail_open
        card.open = True

    def close(expt):
        card.closes += 1
        card.open = False
        return card.close_result

    return Connection(key="awg", label="Tweezer AWG", connect=connect, close=close,
                      is_connected=lambda expt: card.open,
                      detail=lambda expt: "no tones loaded", busy_s=7., **kw)


class Reports:
    def __init__(self):
        self.sent = []
        self.busy = []
        self.accept = True

    def report(self, snapshot, timeout=None):
        self.sent.append((json.loads(json.dumps(snapshot)), timeout))
        return self.accept

    def states(self):
        return [s["awg"]["state"] for s, _ in self.sent]


@pytest.fixture
def mgr():
    card, rep = Card(), Reports()
    m = ConnectionManager(SimpleNamespace(), [_connection(card)], report=rep.report,
                          busy=rep.busy.append, log=lambda text: None)
    m.card, m.rep = card, rep
    return m


# --- the manager ------------------------------------------------------------------------

def test_nothing_opens_before_the_server_says_no_run_is_starting(mgr):
    mgr.service()
    assert mgr.card.opens == 0
    assert mgr.snapshot()["awg"]["state"] == conns.DISCONNECTED


def test_opens_on_the_first_clear_poll_and_reports_connecting_first(mgr):
    mgr.set_run_pending(False)
    mgr.service()
    assert mgr.card.open and mgr.card.opens == 1
    mgr.flush_report()
    assert mgr.rep.states() == [conns.CONNECTING, conns.CONNECTED]
    assert mgr.rep.busy == [7.]
    snap = mgr.snapshot()["awg"]
    assert snap["detail"] == "no tones loaded" and snap["want"] is True
    mgr.service()                                   # idempotent
    assert mgr.card.opens == 1


def test_released_while_a_run_is_starting_and_reopened_if_it_never_takes_the_core(mgr):
    mgr.set_run_pending(False)
    mgr.service()
    mgr.set_run_pending({"run_id": 81234, "expt": "hf_bec", "token": "t"})
    mgr.service()
    assert not mgr.card.open and mgr.card.closes == 1
    snap = mgr.snapshot()["awg"]
    assert snap["state"] == conns.DISCONNECTED
    assert "released for run 81234 (hf_bec)" in snap["detail"] and snap["want"] is True
    mgr.service()                                   # still pending: stays closed
    assert mgr.card.opens == 1
    mgr.set_run_pending(None)                       # withdrawn: fence lifted
    mgr.service()
    assert mgr.card.open and mgr.card.opens == 2


def test_a_failed_connect_is_not_retried_until_asked(mgr):
    mgr.card.fail_open = RuntimeError("tweezer awg init failed: card is already in use "
                                      "(held by 192.168.1.90)")
    mgr.set_run_pending(False)
    mgr.service()
    snap = mgr.snapshot()["awg"]
    assert snap["state"] == conns.FAILED and "held by 192.168.1.90" in snap["detail"]
    assert snap["want"] is False
    mgr.service()
    assert mgr.card.opens == 1                      # no retry loop
    mgr.card.fail_open = None
    assert mgr.request("awg", "connect", "ada@pc2") == ""
    mgr.service()
    assert mgr.card.open and mgr.snapshot()["awg"]["state"] == conns.CONNECTED


def test_a_gui_disconnect_closes_it_and_it_stays_closed(mgr):
    mgr.set_run_pending(False)
    mgr.service()
    assert mgr.request("awg", "disconnect", "ada@pc2") == ""
    mgr.service()
    snap = mgr.snapshot()["awg"]
    assert not mgr.card.open and snap["state"] == conns.DISCONNECTED
    assert snap["detail"] == "disconnected by ada@pc2" and snap["want"] is False
    mgr.set_run_pending(None)
    mgr.service()
    assert not mgr.card.open                        # a fence lifting does not reopen it
    assert mgr.request("nope", "connect") .startswith("this monitor has no connection")
    assert mgr.request("awg", "explode").startswith("unknown action")


def test_a_close_that_does_not_finish_blocks_reopening_in_this_process(mgr):
    mgr.set_run_pending(False)
    mgr.service()
    mgr.card.close_result = False
    mgr.request("awg", "disconnect")
    mgr.service()
    snap = mgr.snapshot()["awg"]
    assert snap["state"] == conns.FAILED and "restart the monitor" in snap["detail"]
    refusal = mgr.request("awg", "connect")
    assert "restart the monitor" in refusal
    mgr.service()
    assert mgr.card.opens == 1


def test_close_all_closes_and_nothing_reopens(mgr):
    mgr.set_run_pending(False)
    mgr.service()
    mgr.close_all("the monitor is exiting")
    assert not mgr.card.open
    assert mgr.snapshot()["awg"]["detail"] == "the monitor is exiting"
    mgr.service()
    assert mgr.card.opens == 1
    mgr.close_all()                                 # twice: harmless
    assert mgr.card.closes == 1


def test_a_connection_closed_behind_the_managers_back_is_not_reopened(mgr):
    mgr.set_run_pending(False)
    mgr.service()
    mgr.card.open = False                           # something else closed it
    mgr.refresh()
    snap = mgr.snapshot()["awg"]
    assert snap["state"] == conns.DISCONNECTED and snap["want"] is False
    mgr.service()
    assert mgr.card.opens == 1


def test_an_unaccepted_report_is_sent_again(mgr):
    mgr.rep.accept = False
    assert mgr.flush_report() is False
    mgr.rep.accept = True
    assert mgr.flush_report() is True
    n = len(mgr.rep.sent)
    assert mgr.flush_report() is True and len(mgr.rep.sent) == n     # nothing new


def test_definitions_are_validated():
    card = Card()
    with pytest.raises(ValueError, match="duplicate"):
        conns.validate_connections([_connection(card), _connection(card)])
    with pytest.raises(ValueError, match="identifier"):
        Connection(key="a b", label="x", connect=print, close=print,
                   is_connected=print).validate()


# --- the monitor's host side ---------------------------------------------------------------

class FakeServer:
    def __init__(self):
        self.extras = {"run_pending": False, "connection_requests": [], "exit": False}
        self.reports = []
        self.requests = []

    def poll(self):
        reply = {"status": "ok", "version": 1, "ops": [], "registered": False}
        reply.update(self.extras)
        self.extras["connection_requests"] = []
        return reply

    def report_connections(self, snapshot, timeout=None):
        self.reports.append((snapshot, timeout))
        return {"status": "ok"}

    def report_ops(self, results):
        return {"status": "ok"}

    def request(self, obj):
        self.requests.append(obj)
        return {"status": "ok"}

    def send_ready(self):
        pass


@pytest.fixture
def monitor(monkeypatch, tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"dds": {}, "dac": {}, "ttl": {}}))
    fake = FakeServer()
    monkeypatch.setattr(wmon, "MonitorClient", lambda *a, **k: fake)
    m = wmon.Monitor(SimpleNamespace(), device_state_json_path=str(path))
    m.dds_dict, m.dac_dict, m.ttl_dict = {}, {}, {}
    card = Card()
    m.init_connections([_connection(card)])
    m.fake, m.card = fake, card
    return m


def test_monitor_polls_for_connections_even_without_composite_ops(monitor):
    assert not monitor.composites_enabled
    monitor.signal_ready()
    assert monitor.fake.reports[-1][0]["awg"]["state"] == conns.DISCONNECTED
    assert monitor.poll_changes() == 0
    assert monitor.card.open
    assert monitor.fake.reports[-1][0]["awg"]["state"] == conns.CONNECTED
    assert {"type": "busy", "seconds": 7.} in monitor.fake.requests


def test_monitor_releases_for_a_run_and_serves_gui_requests(monitor):
    monitor.poll_changes()
    monitor.fake.extras["run_pending"] = {"run_id": 81234, "expt": "hf_bec"}
    monitor.poll_changes()
    assert not monitor.card.open
    assert "released for run 81234 (hf_bec)" in monitor.fake.reports[-1][0]["awg"]["detail"]
    monitor.fake.extras["run_pending"] = None
    monitor.poll_changes()
    assert monitor.card.open
    monitor.fake.extras["connection_requests"] = [
        {"key": "awg", "action": "disconnect", "operator": "ada", "client": "pc2"}]
    monitor.poll_changes()
    assert not monitor.card.open
    assert monitor.fake.reports[-1][0]["awg"]["detail"] == "disconnected by ada@pc2"


def test_exit_request_raises_flag_exit_and_opens_nothing_more(monitor):
    monitor.fake.extras["exit"] = True
    assert monitor.poll_changes() & wmon.FLAG_EXIT
    assert monitor.card.opens == 0
    monitor.close_connections()                     # run()'s finally
    assert monitor.fake.reports[-1][1] == wmon.T_EXIT_REPORT_TIMEOUT


def test_an_exiting_monitor_does_not_re_register_retired_ops(monitor):
    registered = []
    monitor._composites_enabled = True
    monitor._op_table = OpTable(())
    monitor._register_ops = lambda: registered.append(1)
    monitor.fake.extras["exit"] = True               # and the poll says registered: False
    monitor.poll_changes()
    assert registered == []


def test_close_connections_at_exit_closes_what_is_open(monitor):
    monitor.poll_changes()
    assert monitor.card.open
    monitor.close_connections()
    assert not monitor.card.open
    snapshot, timeout = monitor.fake.reports[-1]
    assert snapshot["awg"]["state"] == conns.DISCONNECTED
    assert snapshot["awg"]["detail"] == "the monitor is exiting"
    assert timeout == wmon.T_EXIT_REPORT_TIMEOUT


def test_an_old_server_that_does_not_say_when_runs_start_gets_nothing_opened(monitor):
    monitor.fake.extras = {}
    monitor.poll_changes()
    monitor.poll_changes()
    assert monitor.card.opens == 0


def test_ops_refresh_the_connections(monitor):
    monitor.poll_changes()
    monitor.last_config_data = {"dds": {}, "dac": {}, "ttl": {}}
    monitor._snap_dds_keys = monitor._snap_dac_keys = monitor._snap_ttl_keys = []
    monitor.card.open = False                      # an op closed it
    monitor._report_ops(0, [], [], [], [], [], [], [], [])
    assert monitor.fake.reports[-1][0]["awg"]["state"] == conns.DISCONNECTED


# --- the server ------------------------------------------------------------------------------

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
    yield s
    s.sock.close()


def _ask(server, obj):
    return json.loads(server.generate_reply(json.dumps(obj)))


REPORT = {"awg": {"label": "Tweezer AWG", "state": conns.CONNECTED, "detail": "no tones",
                  "since": 1.0, "want": True, "tooltip": ""}}


def test_a_report_is_kept_broadcast_journaled_and_served(server):
    assert _ask(server, {"type": "connections", "connections": REPORT})["status"] == "ok"
    assert server._broadcaster.sent[-1] == {"type": "connections", "connections": REPORT}
    assert server.journal.tail(1)[0]["kind"] == "connection"
    assert json.loads(server.generate_reply("status_json"))["connections"] == REPORT
    assert _ask(server, {"type": "get_state"})["connections"] == REPORT
    bad = dict(REPORT["awg"], state="exploded")
    _ask(server, {"type": "connections", "connections": {"awg": bad}})
    assert server.connections_snapshot()["awg"]["state"] == conns.FAILED


def test_gui_requests_are_checked_then_handed_to_the_next_poll_once(server):
    _ask(server, {"type": "connections", "connections": REPORT})
    req = {"type": "connection", "key": "awg", "action": "disconnect", "operator": "ada"}
    refused = _ask(server, req)
    assert refused["status"] == "error" and "not ready" in refused["msg"]
    server.status.set_state(STATES.READY, "running")
    assert _ask(server, dict(req, key="nope"))["status"] == "error"
    assert _ask(server, dict(req, action="explode"))["status"] == "error"
    assert _ask(server, req)["status"] == "ok"
    poll = _ask(server, {"type": "poll"})
    assert poll["connection_requests"] == [{"key": "awg", "action": "disconnect",
                                            "client": "", "operator": "ada"}]
    assert poll["run_pending"] is None and poll["exit"] is False
    assert _ask(server, {"type": "poll"})["connection_requests"] == []


def test_connect_is_refused_and_polls_say_so_while_a_run_is_starting(server):
    _ask(server, {"type": "connections", "connections": REPORT})
    server.status.set_state(STATES.READY, "running")
    _ask(server, {"type": "run_pending", "run_id": 81234, "expt": "hf_bec", "token": "t"})
    refused = _ask(server, {"type": "connection", "key": "awg", "action": "connect"})
    assert refused["status"] == "error" and "run 81234 (hf_bec)" in refused["msg"]
    assert _ask(server, {"type": "poll"})["run_pending"] == {"run_id": 81234,
                                                            "expt": "hf_bec"}
    assert _ask(server, {"type": "connection", "key": "awg",
                         "action": "disconnect"})["status"] == "ok"


def test_exit_request_only_for_a_ready_monitor_holding_something(server):
    assert server.request_monitor_exit() == []                  # nothing reported
    _ask(server, {"type": "connections", "connections": REPORT})
    assert server.request_monitor_exit() == []                  # not ready
    server.status.set_state(STATES.READY, "running")
    assert server.request_monitor_exit() == ["Tweezer AWG"]
    assert _ask(server, {"type": "poll"})["exit"] is True
    server.clear_monitor_exit()
    assert _ask(server, {"type": "poll"})["exit"] is False


def test_a_monitor_asked_to_exit_is_handed_no_ops(server):
    _ask(server, {"type": "connections", "connections": REPORT})
    table = OpTable(())
    _ask(server, table.registration(session="t"))
    server.status.set_state(STATES.READY, "running")
    e = table.get("monitor.ping")
    seq = _ask(server, {"type": "op", "op": e.name, "sig": e.signature, "args": {}})["seq"]
    server.request_monitor_exit()
    assert _ask(server, {"type": "poll"})["ops"] == []
    server.clear_monitor_exit()
    assert [o["seq"] for o in _ask(server, {"type": "poll"})["ops"]] == [seq]


def test_a_stopped_monitor_holds_nothing(server):
    _ask(server, {"type": "connections", "connections": REPORT})
    server.status.set_state(STATES.READY, "running")
    _ask(server, {"type": "connection", "key": "awg", "action": "disconnect"})
    server.request_monitor_exit()
    server.on_monitor_state(STATES.NOT_READY, "interrupted_by_run")
    snap = server.connections_snapshot()["awg"]
    assert snap["state"] == conns.DISCONNECTED
    assert snap["detail"] == "the monitor is not running (interrupted by run)"

    def connection_broadcasts():
        return [p for p in server._broadcaster.sent if p.get("type") == "connections"]
    assert connection_broadcasts()[-1]["connections"]["awg"] == snap
    n = len(connection_broadcasts())
    server.on_monitor_state(STATES.NOT_READY, "interrupted_by_run")     # 8 Hz: once only
    assert len(connection_broadcasts()) == n
    server.status.set_state(STATES.READY, "running")
    poll = _ask(server, {"type": "poll"})
    assert poll["connection_requests"] == [] and poll["exit"] is False


# --- the graceful stop ----------------------------------------------------------------------

class FakeProc:
    pid = 999999

    def __init__(self, exits_on_request=True):
        self.returncode = None
        self.exits_on_request = exits_on_request
        self.asked = False
        self.waited = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waited.append(timeout)
        if self.asked and self.exits_on_request:
            self.returncode = 0
            return 0
        raise mm.TimeoutExpired("ar", timeout)

    def kill(self):
        self.returncode = 1


@pytest.fixture
def killed(monkeypatch):
    """The process-tree killer is a recorder: a fake pid must never reach taskkill."""
    import waxx.util.dashboard.server_supervisor as sup
    pids = []

    def fake_kill(pid):
        pids.append(pid)
        return True
    monkeypatch.setattr(sup, "_kill_pid_tree", fake_kill)
    return pids


def _manager(proc, holding):
    m = mm.MonitorManager("monitor.py")
    calls = []

    def request():
        calls.append("request")
        proc.asked = bool(holding)
        return holding

    m.set_graceful_exit(request, lambda: calls.append("clear"), grace_s=3.)
    m._proc = proc
    return m, calls


def test_a_monitor_holding_a_connection_is_let_exit_on_its_own(qapp, killed):
    proc = FakeProc()
    m, calls = _manager(proc, ["Tweezer AWG"])
    m.stop()
    assert calls == ["request", "clear"] and proc.waited == [3.]
    assert killed == [] and proc.returncode == 0


def test_a_monitor_that_does_not_exit_in_time_is_killed(qapp, killed):
    proc = FakeProc(exits_on_request=False)
    m, calls = _manager(proc, ["Tweezer AWG"])
    m.stop()
    assert calls == ["request", "clear"] and killed == [proc.pid]


def test_a_monitor_holding_nothing_is_killed_at_once(qapp, killed):
    proc = FakeProc()
    m, calls = _manager(proc, [])
    m.stop()
    assert calls == ["request", "clear"] and proc.waited == [] and killed == [proc.pid]


# --- the connection bar ------------------------------------------------------------------------

# liveOD's camera button colours: connected, not connected, failed
GREEN, GREY, RED = "#43a047", "#9e9e9e", "#c62828"


@pytest.fixture
def panel(qapp, monkeypatch):
    from test_composite_panel import FakeSender, FakeSettings
    from waxx.util.guis import composite_panel as cp
    monkeypatch.setattr(cp, "QSettings", FakeSettings)
    monkeypatch.setattr(cp, "_OpSender", FakeSender)
    card = Card()
    p = cp.CompositePanel([], connections=[_connection(card, tooltip="netbox",
                                                       confirm="The card is stopped.")],
                          start_sender=False)
    p.answers = []
    p.answer = True
    p.confirm = lambda title, text, verb="Send", danger=False: (
        p.answers.append((title, text, verb)) or p.answer)
    p.set_monitor_state(STATES.READY, reachable=True)
    yield p
    p.shutdown()


def _pill(panel):
    return panel.connection_bar._pills["awg"]


def test_bar_shows_each_connection_before_the_monitor_reports(panel):
    pill, detail = _pill(panel)
    assert panel.connection_bar.isVisibleTo(panel)
    assert pill.text() == "Tweezer AWG" and GREY in pill.styleSheet()
    assert "not reported" in detail.text()


def test_bar_colours_follow_the_reported_state(panel):
    pill, detail = _pill(panel)
    panel.set_connections(REPORT)
    assert GREEN in pill.styleSheet() and detail.text() == "no tones"
    assert "Click to disconnect" in pill.toolTip() and "netbox" in pill.toolTip()
    long = dict(REPORT["awg"], state=conns.FAILED, detail="x" * 300)
    panel.set_connections({"awg": long})
    assert RED in pill.styleSheet()
    assert len(detail.text()) <= 90 and detail.text().endswith("…")
    assert detail.toolTip() == "x" * 300                        # the whole text
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


def test_pills_are_disabled_when_the_monitor_cannot_act(panel):
    pill, detail = _pill(panel)
    panel.set_monitor_state(STATES.NOT_READY, reachable=True)
    assert not pill.isEnabled() and "not running" in pill.toolTip()
    panel.set_monitor_state(STATES.READY, reachable=True)
    panel.set_run_pending({"run_id": 81234, "expt": "hf_bec"})
    assert not pill.isEnabled() and "run 81234 (hf_bec)" in pill.toolTip()
    panel.set_run_pending(None)
    panel.set_monitor_state(None, reachable=False)
    assert not pill.isEnabled() and detail.text() == "monitor server unreachable"


def test_status_detail_and_context_carry_the_connections(panel):
    panel.set_monitor_detail({"connections": REPORT})
    assert panel.connections == REPORT
    ctx = panel.context(None, {})
    assert isinstance(ctx, Context) and ctx.connection("awg")["state"] == conns.CONNECTED
    assert ctx.connection("nope") is None
