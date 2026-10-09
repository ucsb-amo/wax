"""The Sequences tab (waxx.util.guis.sequences_panel) and the Device Control
GUI's status row around it: the connection bar moved up there, and the
monitor notice that replaced Start / Restart / Stop.

Offscreen Qt.  The request senders are fakes -- nothing goes on the network
(the development machine may be the lab PC, with the real monitor server
running) -- and QSettings is faked."""
import os
import time

import pytest
from PyQt6.QtWidgets import QApplication, QPushButton

from waxx.util.comms_server.comm_server import STATES
from waxx.util.device_state.connections import Connection
from waxx.util.guis import composite_panel as cp
from waxx.util.guis import device_control_gui as dc
from waxx.util.guis import sequences_panel as sp
from waxx.util.guis.device_summary import (
    NOTICE_INTERRUPTED, NOTICE_NOT_RUNNING, NOTICE_UNREACHABLE)

from test_composite_panel import DEVICE, FakeSender, FakeSettings, _AcceptingBox


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeSettings.store = {}
    monkeypatch.setattr(cp, "QSettings", FakeSettings)
    monkeypatch.setattr(dc, "QSettings", FakeSettings)
    monkeypatch.setattr(cp, "_OpSender", FakeSender)
    monkeypatch.setattr(sp, "_OpSender", FakeSender)
    # A wheel guard cached while another module's QApplication lived died
    # with it: build a fresh one for the composite cards.
    monkeypatch.setattr(cp, "_WHEEL_GUARD", None)


LOOP = {"key": "auto_tof", "title": "BEC TOF loop", "expt": "auto_tof",
        "path": "C:/code/auto_tof.py", "about": "BEC TOF loop: t_tof 1-4 ms.",
        "state": "idle", "text": "not started", "runs": 0, "run_id": None, "last": None}
RESET = {"expt": "mot_observe", "state": "idle",
         "about": "MOT Observe: puts the machine in its MOT-loading idle state."}
AWG = Connection(key="awg", label="Tweezer AWG", driver="fake:Driver")


@pytest.fixture
def panel(qapp):
    p = sp.SequencesPanel(log_line=lambda text: p.lines.append(text))
    p.lines = []
    p.confirm_answers = []
    p.confirm = lambda title, text, verb="Send": p.confirm_answers.append(
        (title, text, verb)) or True
    p.set_reachable(True)
    yield p
    p.shutdown()


def _reply(panel, reply):
    req = panel._sender.requests[-1]
    panel._sender.requested.emit(req["req"], reply)
    return req


# --- cards ------------------------------------------------------------------------------

def test_a_card_is_one_row_buttons_left_and_the_log_arrow_right(panel):
    panel.set_loops({"auto_tof": LOOP})
    card = panel.loop_cards["auto_tof"]
    assert not card.isHidden() and card.title.text() == "BEC TOF loop"
    row = card.layout().itemAt(0).layout()
    order = [row.indexOf(w) for w in (card.start_button, card.stop_button, card.title,
                                      card.pill, card.status, card.toggle)]
    assert -1 not in order and order == sorted(order)
    assert card.log.isHidden() and card.toggle.text() == "▾"
    # the loops come first, the reset experiment last
    panel.set_reset(RESET)
    assert panel._cards.indexOf(card) < panel._cards.indexOf(panel.reset_card)


def test_loop_card_starts_and_stops_through_the_server(panel):
    panel.set_loops({"auto_tof": LOOP})
    card = panel.loop_cards["auto_tof"]
    assert card.pill.text() == "idle"
    assert card.start_button.isEnabled() and not card.stop_button.isEnabled()
    card.start_button.click()
    title, text, verb = panel.confirm_answers[-1]
    assert "t_tof 1-4 ms" in text and "Stop lets the run in progress finish" in text
    assert verb == "Start BEC TOF loop"
    req = _reply(panel, {"status": "ok", "loop": dict(
        LOOP, state="running", run_id=81000, started=time.time(),
        text="run 81000 in progress (1st of the loop)")})
    assert (req["type"], req["action"], req["loop"]) == ("run_loop", "start", "auto_tof")
    assert req["owner"] == "person"                         # a person clicked Start
    assert card.pill.text() == "RUNNING" and "run 81000" in card.status.text()
    assert card.stop_button.isEnabled() and not card.start_button.isEnabled()
    card.stop_button.click()
    assert panel._sender.requests[-1]["action"] == "stop"
    panel.on_run_loop(dict(LOOP, state="latched", runs=1, ended=time.time(),
                           text="run 81001 was aborted in liveOD (Abort): its file was "
                                "discarded"))
    assert card.pill.text() == "LATCHED OFF" and "aborted in liveOD" in card.status.text()
    assert card.start_button.isEnabled()
    assert any(line.startswith("[loop] BEC TOF loop: latched") for line in panel.lines)
    panel.set_reachable(False)
    assert not card.start_button.isEnabled()


def test_a_refused_loop_start_says_why(panel):
    panel.set_loops({"auto_tof": LOOP})
    panel.loop_cards["auto_tof"].start_button.click()
    _reply(panel, {"status": "error", "msg": "run 7 (rabi) is in progress in liveOD"})
    status = panel.loop_cards["auto_tof"].status.text()
    assert "✕ not started: run 7 (rabi) is in progress in liveOD" in status


def test_reset_card_runs_through_the_host_and_shows_how_it_ended(panel):
    card = panel.reset_card
    assert card.isHidden()                                  # until the server has one
    panel.set_reset(RESET)
    assert not card.isHidden() and card.stop_button is None
    assert card.title.text() == "MOT Observe" and card.pill.text() == "idle"
    assert card.start_button.isEnabled() and card.start_button.text() == "Run"
    asked = []
    panel.reset_requested.connect(lambda: asked.append(1))
    card.start_button.click()
    assert asked == [1] and not panel._sender.requests     # the host confirms and sends
    panel.set_reset(dict(RESET, state="running", started=time.time(), operator="",
                         client="kong", text="mot_observe started"))
    assert card.pill.text() == "RUNNING" and "by kong" in card.status.text()
    assert not card.start_button.isEnabled() and card.start_button.text() == "Running…"
    panel.set_reset(dict(RESET, state="failed", ended=time.time(), tail=["boom"],
                         text="mot_observe exited with code 1 without reporting its end state"))
    assert card.pill.text() == "FAILED" and "exited with code 1" in card.status.text()
    assert card.start_button.isEnabled()
    panel.set_reset(None)
    assert card.isHidden()


def test_the_tab_says_when_there_is_nothing_to_show(qapp):
    p = sp.SequencesPanel()
    assert not p.empty.isHidden() and "Waiting" in p.empty.text()
    p.set_reachable(True)
    assert "runs no sequences" in p.empty.text()
    p.set_reset(RESET)
    assert p.empty.isHidden()
    p.shutdown()


# --- logs -------------------------------------------------------------------------------

def test_the_arrow_opens_the_log_and_it_asks_only_for_new_lines(panel):
    panel.set_loops({"auto_tof": LOOP})
    card = panel.loop_cards["auto_tof"]
    card.toggle.click()
    assert not card.log.isHidden() and card.toggle.text() == "▴"
    req = _reply(panel, {"status": "ok", "lines": ["── 16:00:00 1st run ──", "Run ID: 81000"],
                         "first": 1, "next": 3, "more": False})
    assert (req["type"], req["kind"], req["key"], req["after"]) == \
        ("output", "run_loop", "auto_tof", 0)
    assert card.log.toPlainText().splitlines() == ["── 16:00:00 1st run ──", "Run ID: 81000"]
    panel.fetch_output(card)
    n = len(panel._sender.requests)
    panel.fetch_output(card)                                # one request at a time
    assert len(panel._sender.requests) == n
    req = _reply(panel, {"status": "ok", "lines": ["shot 9/9"], "first": 10, "next": 11,
                         "more": False})
    assert req["after"] == 2
    lines = card.log.toPlainText().splitlines()
    assert lines[-2:] == ["(7 lines were not kept)", "shot 9/9"] and card.after == 10
    # more waiting: asked again at once
    panel.fetch_output(card)
    _reply(panel, {"status": "ok", "lines": ["a"], "first": 11, "next": 12, "more": True})
    assert panel._sender.requests[-1]["after"] == 11
    card.toggle.click()
    assert card.log.isHidden() and card.toggle.text() == "▾"


def test_a_restarted_server_starts_the_log_over(panel):
    panel.set_reset(RESET)
    card = panel.reset_card
    card.toggle.click()
    req = _reply(panel, {"status": "ok", "lines": ["x"] * 5, "first": 1, "next": 6,
                         "more": False})
    assert req["kind"] == "reset" and "key" not in req
    panel.fetch_output(card)
    _reply(panel, {"status": "ok", "lines": [], "first": 2, "next": 2, "more": False})
    assert card.after == 0 and card.log.toPlainText() == ""
    assert panel._sender.requests[-1]["after"] == 0         # asked again from the start


def test_an_older_server_says_it_keeps_no_output_and_shows_the_tail(panel):
    panel.set_reset(dict(RESET, state="failed", tail=["Traceback", "boom"]))
    card = panel.reset_card
    card.toggle.click()
    _reply(panel, {"status": "error", "msg": "unknown type output"})
    text = card.log.toPlainText()
    assert "keeps no output" in text and "boom" in text and card.unsupported
    n = len(panel._sender.requests)
    panel.fetch_output(card)
    assert len(panel._sender.requests) == n                 # not asked again


def test_an_unreachable_server_is_said_once(panel):
    panel.set_reset(RESET)
    card = panel.reset_card
    card.toggle.click()
    _reply(panel, {"status": "error", "msg": "monitor server unreachable"})
    panel.fetch_output(card)
    _reply(panel, {"status": "error", "msg": "monitor server unreachable"})
    assert card.log.toPlainText().count("monitor server unreachable") == 1


def test_logs_are_polled_only_while_the_tab_is_on_screen(panel, qapp):
    panel.set_reset(RESET)
    card = panel.reset_card
    card.toggle.click()
    _reply(panel, {"status": "ok", "lines": [], "first": 1, "next": 1, "more": False})
    n = len(panel._sender.requests)
    panel.poll_logs()                                       # never shown
    assert len(panel._sender.requests) == n
    panel.show()
    qapp.processEvents()
    panel.poll_logs()
    assert len(panel._sender.requests) == n + 1
    _reply(panel, {"status": "ok", "lines": [], "first": 1, "next": 1, "more": False})
    card.toggle.click()                                     # closed: not polled
    panel.poll_logs()
    assert len(panel._sender.requests) == n + 1
    panel.hide()


# --- in the Device Control GUI ----------------------------------------------------------

@pytest.fixture
def gui(qapp, monkeypatch):
    for name in ("_setup_update_sender", "_setup_state_listener", "_setup_state_worker",
                 "setup_status_checker", "setup_timer", "request_state"):
        monkeypatch.setattr(dc.DeviceStateGUI, name, lambda self, *a, **k: None)
    g = dc.DeviceStateGUI(composite_devices=[DEVICE], composite_connections=[AWG])
    g.sent_commands = []
    g._send_monitor_command = g.sent_commands.append
    yield g
    g.close()


def test_sequences_is_the_last_tab_and_follows_the_server(gui):
    tabs = [gui.tab_widget.tabText(i) for i in range(gui.tab_widget.count())]
    assert tabs[-2:] == ["Composite", "Sequences"]
    seq = gui.sequences_panel
    gui._on_status_detail({"state": STATES.READY, "sub_state": "running",
                           "run_loops": {"auto_tof": LOOP}})
    assert seq.reachable and "auto_tof" in seq.loop_cards
    gui._on_state_broadcast({"type": "run_loop", "loop": dict(LOOP, state="running")})
    assert seq.loop_cards["auto_tof"].pill.text() == "RUNNING"
    gui._set_reset(dict(RESET))
    assert not seq.reset_card.isHidden()
    gui.on_connection_failed()
    assert not seq.reachable


def test_the_reset_cards_run_goes_through_the_hosts_dialog(gui, monkeypatch):
    gui._set_monitor_state(STATES.READY)
    gui._set_reset(dict(RESET))
    sent = []
    monkeypatch.setattr(dc, "QMessageBox", _AcceptingBox)
    gui._send_request = lambda obj, callback: sent.append(obj)
    gui.sequences_panel.reset_card.start_button.click()
    assert sent and sent[-1]["type"] == "reset_state"


def test_the_connection_bar_sits_in_the_status_row(gui):
    bar = gui.composite_panel.connection_bar
    assert gui._connections_slot.indexOf(bar) == 0
    assert not gui.composite_panel.isAncestorOf(bar)
    assert bar.isVisibleTo(gui.centralWidget())
    assert "Tweezer AWG" in [pill.text() for pill in bar._pills.values()]


def test_the_notice_sits_in_the_middle_and_moves_nothing(gui, qapp):
    """The notice appears right of the monitor pill; the connection pills and
    Log stay where they were."""
    bar, log = gui.composite_panel.connection_bar, gui.changes_button
    assert log.text() == "Log"
    # Wide enough for the row even without system fonts (offscreen Qt then
    # draws every glyph as a wide box; with real fonts it needs ~650 px).
    gui.resize(2400, 700)
    gui.show()
    gui._on_status_detail({"state": STATES.READY})
    qapp.processEvents()
    assert gui.monitor_notice.isHidden()
    before = (bar.mapTo(gui, bar.rect().topLeft()), log.mapTo(gui, log.rect().topLeft()))
    gui._on_status_detail({"state": STATES.NOT_READY, "sub_state": "never_started",
                           "reason": ""})
    qapp.processEvents()
    notice = gui.monitor_notice
    assert not notice.isHidden()
    after = (bar.mapTo(gui, bar.rect().topLeft()), log.mapTo(gui, log.rect().topLeft()))
    assert after == before
    assert gui.status_pill.x() < notice.x() < bar.x()


def test_the_status_row_has_no_monitor_buttons_and_the_strip_no_monitor_banner(gui):
    for name in ("start_button", "restart_button", "stop_button", "reset_button"):
        assert not hasattr(gui, name)
    assert "monitor" not in gui.summary.banners
    texts = {b.text() for b in gui.centralWidget().findChildren(QPushButton)}
    assert not texts & {"Start monitor", "Restart", "Expand all"}


def test_the_notice_says_why_edits_are_not_applied(gui):
    notice = gui.monitor_notice
    gui._on_status_detail({"state": STATES.NOT_READY, "sub_state": "never_started",
                           "reason": ""})
    assert not notice.isHidden() and notice.label.text() == NOTICE_NOT_RUNNING
    assert not notice.button.isHidden() and notice.button.isEnabled()
    gui._on_status_detail({"state": STATES.NOT_READY, "sub_state": "interrupted_by_run",
                           "reason": ""})
    assert notice.label.text() == NOTICE_INTERRUPTED and notice.level == "info"
    gui._on_status_detail({"state": STATES.LOADING, "sub_state": "starting"})
    assert notice.isHidden()
    gui._on_status_detail({"state": STATES.READY, "sub_state": "running"})
    assert notice.isHidden()
    gui.on_connection_failed()
    assert notice.label.text() == NOTICE_UNREACHABLE and notice.button.isHidden()


def test_the_status_detail_does_not_repeat_the_notice(gui):
    gui._on_status_detail({"state": STATES.READY, "sub_state": "running",
                           "since": time.time() - 125})
    assert gui.status_detail_label.text() == "for 2 min"
    gui._on_status_detail({"state": STATES.NOT_READY, "sub_state": "interrupted_by_run",
                           "since": time.time() - 5})
    assert gui.status_detail_label.text() == "for 5 s"
    gui._on_status_detail({"state": STATES.NOT_READY, "sub_state": "failed",
                           "reason": "exit code 1: ModuleNotFoundError",
                           "since": time.time() - 5})
    assert gui.status_detail_label.text().startswith("exit code 1: ModuleNotFoundError")


def test_notice_start_asks_first_when_an_experiment_holds_the_core(gui, monkeypatch):
    gui._on_status_detail({"state": STATES.NOT_READY, "sub_state": "never_started"})
    gui.monitor_notice.button.click()
    assert gui.sent_commands == ["reset"]                   # nothing to cut off: no question
    answers = []

    def question(*a, **k):
        answers.append(a[2])
        return dc.QMessageBox.StandardButton.No
    monkeypatch.setattr(dc.QMessageBox, "question", staticmethod(question))
    gui._on_status_detail({"state": STATES.NOT_READY, "sub_state": "interrupted_by_run"})
    gui.monitor_notice.button.click()
    assert gui.sent_commands == ["reset"] and "cuts that experiment off" in answers[-1]
    monkeypatch.setattr(dc.QMessageBox, "question",
                        staticmethod(lambda *a, **k: dc.QMessageBox.StandardButton.Yes))
    gui.monitor_notice.button.click()
    assert gui.sent_commands == ["reset", "reset"]


def test_the_pill_menu_offers_what_the_state_allows(gui):
    gui._on_status_detail({"state": STATES.NOT_READY, "sub_state": "never_started"})
    assert gui._monitor_commands_allowed() == (True, False)
    gui._on_status_detail({"state": STATES.READY, "sub_state": "running"})
    assert gui._monitor_commands_allowed() == (False, True)
    gui.on_connection_failed()
    assert gui._monitor_commands_allowed() == (False, False)


# --- pick loop (file chosen at Start) -------------------------------------------------------

PICK = {"key": "expt_loop", "title": "Experiment loop", "expt": "", "path": "", "about": "",
        "pick": True, "root": "C:/code/k-exp/kexp/experiments", "rel": "",
        "state": "idle", "text": "not started", "runs": 0, "run_id": None, "last": None}


def test_server_path_maps_a_local_file_onto_the_servers_root(tmp_path):
    root = tmp_path / "kexp" / "experiments"
    (root / "JP").mkdir(parents=True)
    f = root / "JP" / "rabi.py"
    f.write_text("")
    assert sp.server_path(str(f), str(root)) == "JP/rabi.py"
    # another PC: a different checkout, same layout under kexp/experiments
    other = r"D:\lab\k-exp\KEXP\Experiments\JP\sub\rabi.py"
    assert sp.server_path(other, r"C:\code\k-exp\kexp\experiments") == "JP/sub/rabi.py"
    assert sp.server_path(r"D:\elsewhere\rabi.py", r"C:\code\k-exp\kexp\experiments") is None


def test_pick_card_chooses_a_file_describes_it_then_starts_it(panel, monkeypatch):
    panel.set_loops({"expt_loop": PICK})
    card = panel.loop_cards["expt_loop"]
    assert card.start_button.text() == "Start…" and "no file chosen" in card.title.text()
    panel.choose_file = lambda caption, start: r"Z:\x\kexp\experiments\JP\rabi.py"
    card.start_button.click()
    req = _reply(panel, {"status": "ok", "path": "C:/code/k-exp/kexp/experiments/JP/rabi.py",
                         "rel": "JP/rabi.py", "expt": "rabi", "about": "Rabi flop."})
    assert (req["action"], req["loop"], req["path"]) == ("describe", "expt_loop", "JP/rabi.py")
    title, text, verb = panel.confirm_answers[-1]
    assert title == "Experiment loop: rabi" and "Rabi flop." in text
    assert "on the monitor server" in text and "read again at every run" in text
    req = _reply(panel, {"status": "ok", "loop": dict(PICK, state="running", expt="rabi",
                                                      path="C:/.../rabi.py")})
    assert (req["action"], req["path"]) == ("start", "JP/rabi.py")
    assert card.title.text() == "Experiment loop: rabi" and card.pill.text() == "RUNNING"


def test_pick_card_shows_why_the_server_refused_the_file(panel):
    panel.set_loops({"expt_loop": PICK})
    card = panel.loop_cards["expt_loop"]
    n = len(panel.confirm_answers)
    panel.choose_file = lambda caption, start: r"Z:\x\kexp\experiments\tools\monitor.py"
    card.start_button.click()
    _reply(panel, {"status": "error", "msg": "monitor.py cannot be looped"})
    assert len(panel.confirm_answers) == n and "cannot be looped" in card.status.text()
    # cancelling the dialog sends nothing
    sent = len(panel._sender.requests)
    panel.choose_file = lambda caption, start: ""
    card.start_button.click()
    assert len(panel._sender.requests) == sent


# --- scan settings (⚙) ------------------------------------------------------------------

SCAN = {"xvar": "t_tof", "unit": "ms", "scale": 1e-3, "minimum": 0.0, "maximum": 25e-3,
        "max_points": 200, "max_repeats": 100,
        "settings": {"start": 1e-3, "stop": 4e-3, "n": 9, "repeats": 5},
        "text": "t_tof 1–4 ms, 9 points × 5 repeats (45 shots)"}


def test_the_gear_shows_only_for_a_loop_with_a_scan_and_sends_it(panel):
    panel.set_loops({"auto_tof": LOOP})
    card = panel.loop_cards["auto_tof"]
    assert card.settings_button.isHidden()
    panel.set_loops({"auto_tof": dict(LOOP, scan=SCAN)})
    assert not card.settings_button.isHidden() and card.settings_button.isEnabled()
    assert "scan: t_tof 1–4 ms, 9 points" in card.status.text()
    row = card.layout().itemAt(0).layout()
    assert row.indexOf(card.settings_button) == row.indexOf(card.toggle) - 1
    asked = []
    new = {"start": 2e-3, "stop": None, "n": 1, "repeats": 20}
    panel.ask_scan = lambda title, scan, running=False: asked.append((title, running)) or new
    card.settings_button.click()
    assert asked == [("BEC TOF loop", False)]
    req = _reply(panel, {"status": "ok", "loop": dict(LOOP, scan=dict(
        SCAN, settings=new, text="t_tof 2 ms × 20 repeats (20 shots)"))})
    assert (req["type"], req["action"], req["loop"], req["scan"]) == \
        ("run_loop", "configure", "auto_tof", new)
    assert "scan: t_tof 2 ms × 20 repeats" in card.status.text()
    card.settings_button.click()
    _reply(panel, {"status": "error", "msg": "the start value -1 ms is below the minimum 0 ms"})
    assert "✕ scan not set: the start value -1 ms" in card.status.text()
    panel.ask_scan = lambda *a, **k: None                   # Cancel: nothing sent
    n = len(panel._sender.requests)
    card.settings_button.click()
    assert len(panel._sender.requests) == n
    panel.set_reachable(False)
    assert not card.settings_button.isEnabled()


def test_the_dialog_grays_points_without_a_stop_value_and_checks_like_the_server(qapp):
    d = sp.ScanSettingsDialog("BEC TOF loop", SCAN, running=True)
    ok = d.buttons.button(sp.QDialogButtonBox.StandardButton.Ok)
    assert (d.start.text(), d.stop.text(), d.points.value(), d.repeats.value()) == \
        ("1", "4", 9, 5)
    assert d.points.isEnabled() and ok.isEnabled()
    assert d.settings() == {"start": 1e-3, "stop": 4e-3, "n": 9, "repeats": 5}
    assert "45 shots" in d.summary.text()
    d.stop.setText("")
    assert not d.points.isEnabled()
    d.repeats.setValue(20)
    assert d.settings() == {"start": 1e-3, "stop": None, "n": 1, "repeats": 20}
    assert d.summary.text() == "t_tof 1 ms × 20 repeats (20 shots)"
    d.start.setText("30")
    assert not ok.isEnabled() and "above the maximum 25 ms" in d.summary.text()
    assert d.settings() is None
    d.start.setText("abc")
    assert not ok.isEnabled() and "not a number" in d.summary.text()
    d.start.setText("0.5")
    d.stop.setText("2.5")
    d.points.setValue(5)
    assert ok.isEnabled() and d.settings() == pytest.approx(
        {"start": 0.5e-3, "stop": 2.5e-3, "n": 5, "repeats": 20})
    d.close()


# --- the run queue line, and a host's requester ------------------------------------------

QUEUE = {"enabled": True, "state": "running", "text": "job 2 (rabi) running, run 85600",
         "current": {"id": 2, "state": "running"}, "next": [3],
         "counts": {"queued": 2, "running": 1}, "alarm": None}


def test_the_queue_line_follows_status_json_and_broadcasts(gui):
    seq = gui.sequences_panel
    assert seq.queue_line.isHidden()
    gui._on_status_detail({"state": STATES.READY, "sub_state": "running", "run_queue": QUEUE})
    assert not seq.queue_line.isHidden()
    assert seq.queue_line.label.text() == ("Queue: running (2 queued) -- open the Monitor "
                                           "panel in the Server Dashboard")
    gui._on_state_broadcast({"type": "run_queue", "run_queue": dict(QUEUE, state="idle",
                                                                    counts={"queued": 0})})
    assert seq.queue_line.label.text().startswith("Queue: idle (0 queued)")
    gui._on_status_detail({"state": STATES.READY, "sub_state": "running"})   # older server
    assert seq.queue_line.isHidden()


def test_a_hosts_requester_carries_the_requests(qapp):
    """With a requester the panel never uses its own sender (no discovery)."""
    asked = []

    def requester(obj):
        asked.append(dict(obj))
        return {"status": "ok", "loop": dict(LOOP, state="running")}
    p = sp.SequencesPanel(requester=requester, synchronous_requests=True, show_hold=False,
                          show_queue=False)
    p.confirm = lambda title, text, verb="Send": True
    p.set_reachable(True)
    p.set_loops({"auto_tof": LOOP})
    p.set_hold({"active": False})
    p.set_queue(QUEUE)
    assert p.hold_row.isHidden() and p.queue_line.isHidden()
    p.loop_cards["auto_tof"].start_button.click()
    assert asked[-1]["type"] == "run_loop" and asked[-1]["owner"] == "person"
    assert p._sender.requests == []
    assert p.loop_cards["auto_tof"].pill.text() == "RUNNING"
    p.shutdown()
