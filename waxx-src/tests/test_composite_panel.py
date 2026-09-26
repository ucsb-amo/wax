"""The Composite tab (waxx.util.guis.composite_panel) and its wiring into the
Device Control GUI.  Offscreen Qt; the op sender is a fake (nothing goes on
the network), QSettings is faked (nothing written to the registry), and the
confirmation / refusal dialogs are replaced by recorders."""
import os
import time

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from waxx.util.comms_server.comm_server import STATES
from waxx.util.device_state import composite as cmp
from waxx.util.device_state.composite import (
    Arg, Buttons, ChannelToggle, Check, CompositeDevice, FieldRow, Info, Lamp, Menu, Op,
    Readout, Status, Table, TableRow,
)
from waxx.util.guis import composite_panel as cp
from waxx.util.guis import device_control_gui as dc


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class FakeSettings:
    store: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def value(self, key, default=None, type=None):
        return FakeSettings.store.get(key, default)

    def setValue(self, key, value):
        FakeSettings.store[key] = value


class FakeSender(QObject):
    replied = pyqtSignal(int, dict)
    status_replied = pyqtSignal(int, dict)
    requested = pyqtSignal(int, dict)
    client_name = "test-pc"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sent = []
        self.queries = []
        self.requests = []

    def submit(self, req, op, sig, args, payload, operator=""):
        self.sent.append({"req": req, "op": op, "sig": sig, "args": args, "payload": payload,
                          "operator": operator})

    def request(self, req, obj):
        self.requests.append({"req": req, **obj})

    def query(self, seq):
        self.queries.append(seq)

    def start(self):
        pass

    def stop(self):
        pass

    def wait(self, *a):
        return True


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeSettings.store = {}
    monkeypatch.setattr(cp, "QSettings", FakeSettings)
    monkeypatch.setattr(dc, "QSettings", FakeSettings)
    monkeypatch.setattr(cp, "_OpSender", FakeSender)


def _state(ctx):
    on = ctx.is_on("ttl", "sw")
    return Status("unknown", "") if on is None else Status("on" if on else "off",
                                                          "ON" if on else "off")


DEVICE = CompositeDevice(
    key="beam", title="Beam",
    fields=(
        Arg("v", "Setpoint", unit="V", decimals=2, minimum=0., maximum=5., warn_above=4.,
            default=1.5, readback=lambda ctx: ctx.dac_voltage("pid")),
        Arg("t", "Time", unit="ms", scale=1e3, decimals=1, minimum=1e-3, maximum=1., default=0.1),
    ),
    tables=(Table("rows", "Rows", columns=(Arg("f", "f", unit="MHz", scale=1e-6, step=0.5,
                                                   minimum=1e6, maximum=100e6),
                                               Arg("a", "a", minimum=0., maximum=1.)),
                  default=[[70e6, 0.2]],
                  check=lambda rows, ctx: Check.error("sum > 1")
                  if sum(r[1] for r in rows) > 1 else None),),
    ops=(
        Op("on", "On", args=("v",), code="expt.beam.on({v})"),
        Op("off", "Off", code="expt.beam.off()"),
        Op("set", "Set", args=("v",), code="expt.beam.set({v})"),
        Op("ramp", "Ramp", args=("v", "t"), code="expt.beam.ramp({v}, {t})"),
        Op("snap", "Snap", code="expt.beam.snap()", danger=True, confirm="Really?"),
        Op("load", "Load", payload=("rows",), host=lambda e, a, p: None,
           code="expt.beam.trig()"),
    ),
    lamps=(Lamp("switch", "ttl", "sw"),),
    readouts=(Readout("pid", lambda ctx: f"{ctx.dac_voltage('pid'):.2f} V"),),
    state=_state,
    layout=(
        Buttons(("on", "off"), main=True),
        FieldRow(("v",), ("set", "ramp")),
        FieldRow(("t",)),
        Info(lambda ctx: f"typed {ctx.field('v'):.2f}"),
        ChannelToggle("hold", "Hold"),
        TableRow("rows", ("load",)),
        Menu(("snap",)),
    ),
)

CONFIG = {"ttl": {"sw": {"ch": 1, "ttl_state": 1}, "hold": {"ch": 2, "ttl_state": 0}},
          "dac": {"pid": {"ch": 3, "voltage": 2.0}}, "dds": {}}


@pytest.fixture
def panel(qapp):
    channel_sends, lines = [], []
    p = cp.CompositePanel([DEVICE], channel_sender=lambda *a: channel_sends.append(a),
                          log_line=lines.append)
    p.confirm_answers = []
    p.confirm = lambda title, text, verb="Send", danger=False: (
        p.confirm_answers.append((title, text, danger, verb)) or p.answer)
    p.answer = True
    p.channel_sends, p.lines = channel_sends, lines
    p.set_config({k: {n: dict(c) for n, c in v.items()} for k, v in CONFIG.items()})
    p.set_monitor_state(STATES.READY, reachable=True)
    p.set_monitor_detail({"composite_ops": {"registered": True, "hash": p.table.hash,
                                            "count": len(p.table)}})
    p.refresh()             # set_config coalesces; tests want the result now
    yield p
    p.shutdown()


def _card(p):
    return p.cards[0]


def test_lamps_state_and_readback(panel):
    card = _card(panel)
    assert card.state_pill.text() == "ON"
    assert "switch" in card.lamps[0][1].text() and "on" in card.lamps[0][1].text()
    assert card.editors["v"].value() == pytest.approx(2.0)         # hardware readback
    assert card.readouts[0][1].text() == "pid 2.00 V"


def test_edited_field_is_not_overwritten_until_sent(panel):
    card = _card(panel)
    card.editors["v"].spin.setValue(3.0)                            # operator types
    assert card.editors["v"].dirty
    panel.config["dac"]["pid"]["voltage"] = 2.5                     # hardware moves
    panel.refresh()
    assert card.editors["v"].value() == pytest.approx(3.0)
    assert card.trigger("set")
    sent = panel._sender.sent[-1]
    assert sent["op"] == "beam.set" and sent["args"] == {"v": pytest.approx(3.0)}
    assert sent["sig"] == panel.table.get("beam.set").signature
    panel._sender.replied.emit(sent["req"], {"status": "ok", "seq": 7})
    assert card.editors["v"].dirty                                  # queued is not done
    panel.on_op_result({"seq": 7, "ok": True, "elapsed": 0.1, "op": "beam.set"})
    assert not card.editors["v"].dirty
    panel.config["dac"]["pid"]["voltage"] = 3.0
    panel.refresh()
    assert card.editors["v"].value() == pytest.approx(3.0)


def test_display_scale_is_undone_before_sending(panel):
    card = _card(panel)
    card.editors["t"].spin.setValue(250.)                           # 250 ms
    card.trigger("ramp")
    assert panel._sender.sent[-1]["args"]["t"] == pytest.approx(0.25)


def test_soft_limit_asks_and_cancel_sends_nothing(panel):
    card = _card(panel)
    card.editors["v"].spin.setValue(4.5)
    panel.answer = False
    assert not card.trigger("set")
    assert panel._sender.sent == []
    assert "above" in panel.confirm_answers[-1][1]
    panel.answer = True
    assert card.trigger("set")


def test_hard_limit_is_refused_without_asking(panel):
    card = _card(panel)
    table = card.tables["rows"]
    table.set_rows([[70e6, 0.7], [71e6, 0.7]], dirty=True)
    assert not card.trigger("load")
    assert panel._sender.sent == [] and panel.confirm_answers == []
    # refused on the card itself, no pop-up
    assert "not sent" in card.footer.text() and "sum > 1" in card.footer.text()


def test_danger_op_always_confirms(panel):
    card = _card(panel)
    assert card.trigger("snap")
    title, text, danger, verb = panel.confirm_answers[-1]
    assert danger and "Really?" in text and verb == "Snap"


def test_ops_disabled_and_refused_when_monitor_not_ready(panel):
    card = _card(panel)
    panel.set_monitor_state(STATES.NOT_READY, reachable=True)
    assert not card.op_buttons["on"][0].isEnabled()
    assert not card.trigger("on")
    assert panel._sender.sent == []
    assert "not running" in panel.path_label.text()
    panel.set_monitor_state(STATES.READY, reachable=True)
    assert card.op_buttons["on"][0].isEnabled()


def test_header_flags_a_monitor_built_from_other_definitions(panel):
    assert "ready" in panel.path_label.text()
    panel.set_monitor_detail({"composite_ops": {"registered": True, "hash": "0" * 16,
                                                "count": 3}})
    assert "different composite definitions" in panel.path_label.text()
    panel.set_monitor_detail({"composite_ops": {"registered": False}})
    assert not panel.ops_allowed()[0]


def test_result_flow_updates_footer_and_log(panel):
    card = _card(panel)
    card.trigger("off")
    req = panel._sender.sent[-1]["req"]
    assert "sending" in card.footer.text()
    assert not card.op_buttons["off"][0].isEnabled()                # pending
    panel._sender.replied.emit(req, {"status": "ok", "seq": 41})
    assert "queued (#41)" in card.footer.text()
    panel.on_op_result({"type": "op_result", "seq": 41, "ok": True, "elapsed": 0.25,
                        "op": "beam.off"})
    assert card.footer.text().startswith("✓ Off")
    assert "0.25 s" in card.footer.text()
    assert card.op_buttons["off"][0].isEnabled()
    assert panel.lines[-1].startswith("[op] beam.off #41: done")
    card.trigger("off")
    req = panel._sender.sent[-1]["req"]
    panel._sender.replied.emit(req, {"status": "ok", "seq": 42})
    panel.on_op_result({"seq": 42, "ok": False, "text": "RTIO underflow inside the op"})
    assert card.footer.text().startswith("✕ Off") and "underflow" in card.footer.text()


def test_server_refusal_is_shown(panel):
    card = _card(panel)
    card.trigger("on")
    req = panel._sender.sent[-1]["req"]
    panel._sender.replied.emit(req, {"status": "error", "msg": "restart the monitor"})
    assert "not queued: restart the monitor" in card.footer.text()
    assert card.op_buttons["on"][0].isEnabled()


def test_lost_broadcast_is_polled_then_times_out(panel):
    card = _card(panel)
    card.trigger("off")
    req = panel._sender.sent[-1]["req"]
    panel._sender.replied.emit(req, {"status": "ok", "seq": 9})
    c, e, t0, r, a, pl = panel._by_seq[9]
    panel._by_seq[9] = (c, e, time.monotonic() - cp.OP_STATUS_POLL_AFTER_S - 0.1, r, a, pl)
    panel._on_tick()
    assert panel._sender.queries == [9]
    panel._sender.status_replied.emit(9, {"status": "ok", "state": "done",
                                          "result": {"seq": 9, "ok": True, "elapsed": 1.0}})
    assert card.footer.text().startswith("✓")
    card.trigger("off")
    req = panel._sender.sent[-1]["req"]
    panel._sender.replied.emit(req, {"status": "ok", "seq": 10})
    c, e, t0, r, a, pl = panel._by_seq[10]
    panel._by_seq[10] = (c, e, time.monotonic() - cp.OP_RESULT_TIMEOUT_S - 1, r, a, pl)
    panel._on_tick()
    assert "outcome unknown" in card.footer.text()


def test_other_clients_results_are_logged(panel):
    panel.on_op_result({"type": "op_result", "seq": 99, "op": "beam.on", "ok": True,
                        "text": "done", "client": "kong"})
    assert panel.lines[-1] == "[op] beam.on #99: done (from kong)"


def test_ping(panel):
    panel.ping()
    sent = panel._sender.sent[-1]
    assert sent["op"] == cmp.PING_OP and sent["args"] == {}
    panel._sender.replied.emit(sent["req"], {"status": "ok", "seq": 5})
    panel.on_op_result({"seq": 5, "ok": True, "elapsed": 0.12})
    assert "Ping round trip 0.12 s" in panel.path_label.text()


def test_channel_toggle_uses_the_channel_path(panel):
    card = _card(panel)
    toggle, button = card.toggles[0]
    assert "off" in button.text()
    button.click()
    assert panel.channel_sends[-1] == ("ttl", "hold", {"ttl_state": 1})


def test_table_add_row_and_payload(panel):
    card = _card(panel)
    table = card.tables["rows"]
    assert table.rows() == [[pytest.approx(70e6), pytest.approx(0.2)]]
    table.add_button.click()
    rows = table.rows()
    assert rows[1][0] == pytest.approx(70.5e6)                      # one step on
    card.trigger("load")
    assert panel._sender.sent[-1]["payload"]["rows"] == rows


def test_default_button_fills_without_sending(panel):
    card = _card(panel)
    card.editors["v"].default_button.click()
    assert card.editors["v"].value() == pytest.approx(1.5)
    assert card.editors["v"].dirty
    assert panel._sender.sent == []


def test_invalid_definitions_show_an_error_not_a_crash(qapp):
    bad = CompositeDevice(key="x", title="x", ops=(Op("o", "o", code="{nope}"),))
    p = cp.CompositePanel([bad])
    assert p.definition_error and not p.cards
    assert not p.ops_allowed()[0]
    p.shutdown()


# --- wiring into the Device Control GUI -------------------------------------------

@pytest.fixture
def gui(qapp, monkeypatch):
    for name in ("_setup_update_sender", "_setup_state_listener", "_setup_state_worker",
                 "setup_status_checker", "setup_timer", "request_state"):
        monkeypatch.setattr(dc.DeviceStateGUI, name, lambda self, *a, **k: None)
    g = dc.DeviceStateGUI(composite_devices=[DEVICE])
    g.composite_panel.confirm = lambda *a, **k: True
    g.config_data = {k: {n: dict(c) for n, c in v.items()} for k, v in CONFIG.items()}
    g.update_device_widgets()
    g._refresh_composite()
    yield g
    g.close()


def test_composite_tab_is_added_last(gui):
    assert gui.tab_widget.tabText(gui.tab_widget.count() - 1) == "Composite"
    assert [gui.tab_widget.tabText(i) for i in range(3)] == ["DDS", "DAC", "TTL"]


def test_no_composite_tab_without_definitions(qapp, monkeypatch):
    for name in ("_setup_update_sender", "_setup_state_listener", "_setup_state_worker",
                 "setup_status_checker", "setup_timer", "request_state"):
        monkeypatch.setattr(dc.DeviceStateGUI, name, lambda self, *a, **k: None)
    g = dc.DeviceStateGUI()
    assert g.composite_panel is None and g.tab_widget.count() == 3
    g.close()


def test_broadcasts_reach_the_panel(gui):
    panel = gui.composite_panel
    gui._version = 10
    gui._on_state_broadcast({"type": "state_update", "version": 11, "device_type": "ttl",
                             "device_name": "sw", "changes": {"ttl_state": 0}})
    panel.refresh()                                                 # (coalesced)
    assert panel.cards[0].state_pill.text() == "off"
    gui._on_status_detail({"state": STATES.READY, "sub_state": "running",
                           "composite_ops": {"registered": True, "hash": panel.table.hash,
                                             "count": len(panel.table)}})
    assert panel.ops_allowed()[0]
    card = panel.cards[0]
    card.trigger("off")
    sent = panel._sender.sent[-1]
    panel._sender.replied.emit(sent["req"], {"status": "ok", "seq": 3})
    gui._on_state_broadcast({"type": "op_result", "seq": 3, "ok": True, "elapsed": 0.1,
                             "op": "beam.off"})
    assert card.footer.text().startswith("✓")
    assert any("[op] beam.off #3" in line for line in gui._changes)
    gui.on_connection_failed()
    assert not panel.ops_allowed()[0]


def test_panel_toggle_updates_the_ttl_tab_too(gui, monkeypatch):
    sent = []

    class Sender:        # closeEvent stops/waits it: an exception there aborts Qt
        def enqueue(self, *a):
            sent.append(a)

        def stop(self):
            pass

        def wait(self, *a):
            return True

    gui._update_sender = Sender()
    card = gui.composite_panel.cards[0]
    card.toggles[0][1].click()
    assert sent[-1] == ("ttl", "hold", {"ttl_state": 1})
    assert gui.config_data["ttl"]["hold"]["ttl_state"] == 1
    assert gui.device_widgets["ttl.hold"].state_button.isChecked()


# --- groups, collapse, search, editing ---------------------------------------------

class Params:
    def __init__(self):
        self.v_def = 1.5
        self.i_def = 10.0


def _coil_state(ctx):
    i = ctx.dac_voltage("coil_i") or 0.
    return Status("hazard" if i > 1. else "off", f"ON {i:.1f} A" if i > 1. else "off")


COIL = CompositeDevice(
    key="coil", title="Coil", group="Fields",
    fields=(Arg("i", "Current", unit="A", minimum=0., maximum=100., default=0.,
                readback=lambda ctx: ctx.dac_voltage("coil_i"), param="i_def"),
            Arg("t", "Ramp", unit="ms", scale=1e3, minimum=1e-3, maximum=1., default=0.05),
            Arg("i_meas", "Measured", minimum=-1., maximum=500., replay=-1.,
                default=lambda ctx: (ctx.measured("ks/i", 5.).value
                                     if ctx.measured("ks/i", 5.) else -1.))),
    ops=(Op("ramp", "Ramp to I", args=("i", "t"), code="expt.coil.ramp({i}, {t})"),
         Op("off", "Off", args=("t", "i_meas"), code="expt.coil.off({t}, {i_meas})")),
    measured=(cmp.Measured("measured", "ks/i", unit="A", decimals=1,
                           expect=lambda ctx: ctx.dac_voltage("coil_i"), tolerance=2.),),
    hazard=lambda ctx: _coil_state(ctx).text if _coil_state(ctx).level == "hazard" else None,
    state=_coil_state, safe_op="off", max_on_s=60.,
    layout=(FieldRow(("i",), ("ramp",)), FieldRow(("t",)), Buttons(("off",), main=True)),
)

SCENE = cmp.Scene(key="hold", title="Hold a field",
                  fields=(Arg("i", "Current", minimum=0., maximum=100., default=5.),
                          Arg("hold", "Hold", unit="s", minimum=0., maximum=100., default=2.)),
                  steps=(cmp.Step("coil.ramp", {"i": "{i}"}, "ramp"), cmp.Hold("{hold}")),
                  finally_=(cmp.Step("coil.off", {"i_meas": -1.}, "off"),))


@pytest.fixture
def panel2(qapp):
    lines = []
    p = cp.CompositePanel([DEVICE, COIL], params=Params(), log_line=lines.append,
                          scenes=[SCENE])
    p.confirm_answers = []
    p.confirm = lambda title, text, verb="Send", danger=False: (
        p.confirm_answers.append((title, text, danger, verb)) or True)
    p.lines = lines
    cfg = {k: {n: dict(c) for n, c in v.items()} for k, v in CONFIG.items()}
    cfg["dac"]["coil_i"] = {"ch": 9, "voltage": 0.}
    p.set_config(cfg)
    p.set_monitor_state(STATES.READY, reachable=True)
    p.set_monitor_detail({"composite_ops": {"registered": True, "hash": p.table.hash,
                                            "count": len(p.table)}})
    p.refresh()
    yield p
    p.shutdown()


def test_cards_are_grouped_and_scenes_have_their_own_section(panel2):
    assert [s.title for s in panel2.sections] == ["Devices", "Fields", "Scenes"]
    assert panel2.masonry.count() == 3


def test_everything_starts_collapsed_every_time(panel2, qapp):
    card = panel2.cards[1]
    assert all(c.is_collapsed() for c in panel2.cards) and panel2.scenes_card.is_collapsed()
    assert card.body.isHidden() and card.chevron.text() == "▸"
    assert panel2.collapse_button.text() == "Expand all"
    card.set_collapsed(False)
    again = cp.CompositePanel([DEVICE, COIL], params=Params(), scenes=[SCENE])
    assert all(c.is_collapsed() for c in again.cards) and again.scenes_card.is_collapsed()
    again._toggle_all()
    assert not any(c.is_collapsed() for c in again.cards)
    assert not again.scenes_card.is_collapsed()
    assert again.collapse_button.text() == "Collapse all"
    again.shutdown()


def test_opening_and_collapsing_never_moves_a_group_to_another_column(qapp):
    from PyQt6.QtCore import QRect
    groups = [dict(key=f"d{i}", title=f"D{i}", group=f"G{i}") for i in range(5)]
    devices = [CompositeDevice(ops=(Op("on", "On", code="pass"),),
                               layout=(Buttons(("on",), main=True),) +
                               tuple(Info(lambda ctx, j=j: "x" * 40) for j in range(3 + 4 * i)),
                               **g) for i, g in enumerate(groups)]
    p = cp.CompositePanel(devices)
    width = 1500

    def columns():
        p.masonry.setGeometry(QRect(0, 0, width, p.masonry.heightForWidth(width)))
        return [s.geometry().x() for s in p.sections]

    before = columns()
    assert len(set(before)) > 1                             # really several columns
    for card in p.cards[::2]:
        card.set_collapsed(False)                           # uneven heights now
        assert columns() == before
    p._toggle_all()
    assert columns() == before
    p._toggle_all()
    assert columns() == before
    p.shutdown()


def test_search_filters_cards_and_groups(panel2):
    panel2.apply_search("coil")
    assert panel2.cards[0].isHidden() and not panel2.cards[1].isHidden()
    assert panel2.sections[0].isHidden()
    panel2.apply_search("")
    assert not panel2.cards[0].isHidden()


def test_escape_reverts_to_the_hardware_value(panel2):
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QKeyEvent
    editor = panel2.cards[0].editors["v"]
    editor.spin.setValue(3.3)
    assert editor.dirty
    editor.eventFilter(editor.spin, QKeyEvent(QEvent.Type.KeyPress, cp.Qt.Key.Key_Escape,
                                              cp.Qt.KeyboardModifier.NoModifier))
    assert not editor.dirty and editor.value() == pytest.approx(2.0)


def test_wheel_is_ignored_without_focus(panel2):
    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtGui import QWheelEvent
    editor = panel2.cards[0].editors["v"]
    event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                        cp.Qt.MouseButton.NoButton, cp.Qt.KeyboardModifier.NoModifier,
                        cp.Qt.ScrollPhase.NoScrollPhase, False)
    assert cp._wheel_guard().eventFilter(editor.spin, event) is True
    assert editor.value() == pytest.approx(2.0)
    assert editor.default_button.focusPolicy() == cp.Qt.FocusPolicy.NoFocus


def test_field_stays_dirty_after_a_failure_or_a_new_edit(panel2):
    card = panel2.cards[0]
    card.editors["v"].spin.setValue(3.0)
    card.trigger("set")
    sent = panel2._sender.sent[-1]
    panel2._sender.replied.emit(sent["req"], {"status": "ok", "seq": 1})
    panel2.on_op_result({"seq": 1, "ok": False, "text": "underflow"})
    assert card.editors["v"].dirty
    card.trigger("set")
    sent = panel2._sender.sent[-1]
    panel2._sender.replied.emit(sent["req"], {"status": "ok", "seq": 2})
    card.editors["v"].spin.setValue(3.5)                  # typed again meanwhile
    panel2.on_op_result({"seq": 2, "ok": True, "elapsed": 0.1})
    assert card.editors["v"].dirty


def test_verb_confirm_names_the_values(panel2):
    card = panel2.cards[0]
    card.editors["v"].spin.setValue(4.5)                  # above warn_above
    card.trigger("set")
    assert panel2.confirm_answers[-1][3] == "Set (Setpoint 4.50 V)"


def test_measured_chip_ok_warn_and_stale(panel2):
    card = panel2.cards[1]
    panel2.config["dac"]["coil_i"]["voltage"] = 20.
    panel2.refresh()
    panel2.set_telemetry({"ks/i": cmp.Sample(20.5)})
    chip = card.measured[0][1]
    assert "20.5 A" in chip.text() and cp.WARN_TEXT not in chip.styleSheet()
    panel2.set_telemetry({"ks/i": cmp.Sample(10.)})
    assert cp.WARN_TEXT in chip.styleSheet()
    panel2.set_telemetry({"ks/i": cmp.Sample(20., age_s=30.)})
    assert "--" in chip.text()


def test_hazards_and_make_safe_send_the_safe_op_without_asking(panel2):
    seen = []
    panel2.hazards_changed.connect(lambda: seen.append(1))
    panel2.config["dac"]["coil_i"]["voltage"] = 20.
    panel2.set_telemetry({"ks/i": cmp.Sample(19.8)})
    panel2.refresh()
    assert seen and [h["key"] for h in panel2.hazards()] == ["coil"]
    assert panel2.cards[1].state_pill.text() == "ON 20.0 A"
    op_key, args, text = panel2.safe_plan("coil")
    assert op_key == "off" and args["i_meas"] == pytest.approx(19.8)
    n = len(panel2.confirm_answers)
    assert panel2.make_safe(["coil"]) == []
    assert panel2._sender.sent[-1]["op"] == "coil.off"
    assert len(panel2.confirm_answers) == n              # the dialog already asked


def test_watchdog_arm_never_replays_a_measurement(panel2):
    panel2.config["dac"]["coil_i"]["voltage"] = 20.
    panel2.set_telemetry({"ks/i": cmp.Sample(19.8)})
    panel2.refresh()
    card = panel2.cards[1]
    card.set_collapsed(False)
    card.watchdog_arm.click()
    req = panel2._sender.requests[-1]
    assert req["type"] == "arm_watchdog" and req["device"] == "coil"
    assert req["op"] == "coil.off" and req["args"]["i_meas"] == -1.
    assert req["sig"] == panel2.table.get("coil.off").signature
    panel2._sender.requested.emit(req["req"], {"status": "ok", "fires_in_s": 180.})
    assert card.watchdog_extend.isVisibleTo(card) and "armed" in card.watchdog_label.text()
    panel2.on_watchdog({"type": "watchdog", "device": "coil", "state": "warning",
                        "fires_in_s": 100.})
    assert "WATCHDOG" in card.watchdog_label.text()
    panel2.on_watchdog({"type": "watchdog", "device": "coil", "state": "fired",
                        "text": "sent coil.off (#4)"})
    assert card.watchdog is None and "fired" in card.footer.text()


def test_copy_as_code_and_adopt_diff(panel2, qapp):
    card = panel2.cards[1]
    card.editors["i"].spin.setValue(12.)
    text = card.copy_as_code("ramp")
    assert "self.coil.ramp(12.0, 0.05)" in text
    assert qapp.clipboard().text() == text
    diff = card.copy_adopt_diff()
    assert "-        self.i_def = 10.0" in diff
    assert "+        # self.i_def = 10.0" in diff
    assert "+        self.i_def = 12.0 # Device Control" in diff
    assert "typed, NOT applied" in diff and "NOT applied: paste by hand" in diff


def test_other_guis_results_show_on_the_card(panel2):
    panel2.on_op_result({"type": "op_result", "seq": 77, "op": "coil.off", "ok": True,
                         "text": "done", "client": "pc2", "operator": "ada",
                         "origin": "watchdog"})
    footer = panel2.cards[1].footer.text()
    assert footer.startswith("✓ Off") and "watchdog" in footer and "ada @ pc2" in footer


def test_run_pending_refuses_ops(panel2):
    panel2.set_run_pending({"run_id": 81000, "expt": "x"})
    assert not panel2.ops_allowed()[0]
    assert not panel2.cards[0].op_buttons["on"][0].isEnabled()
    assert not panel2.cards[0].trigger("on")
    assert "81000" in panel2.cards[0].footer.text()
    panel2.set_run_pending(None)
    assert panel2.cards[0].op_buttons["on"][0].isEnabled()


def test_scene_request_is_checked_resolved_and_sent(panel2):
    sc = panel2.scenes_card
    sc.editors[("hold", "i")].spin.setValue(7.)
    assert panel2.run_scene(SCENE)
    req = panel2._sender.requests[-1]
    assert req["type"] == "run_scene" and req["scene"] == "hold"
    assert req["steps"][0]["args"] == {"i": 7., "t": pytest.approx(0.05)}
    assert req["steps"][1] == {"hold": 2., "label": "hold"}
    assert req["finally"][0]["args"]["i_meas"] == -1.
    assert "Always afterwards" in panel2.confirm_answers[-1][1]
    panel2.on_scene({"type": "scene", "state": "running", "title": "Hold a field",
                     "phase": "steps", "step": 2, "n": 2, "label": "hold",
                     "hold_left_s": 1.5, "id": 1})
    assert "running" in sc.summary.text() and sc.summary.isVisibleTo(sc)   # collapsed
    sc.set_collapsed(False)
    assert "2/2" in sc.progress.text() and sc.cancel_button.isVisibleTo(sc)
    sc.cancel_button.click()
    last = panel2._sender.requests[-1]
    assert last["type"] == "cancel_scene" and last["id"] == 1
    panel2.on_scene({"type": "scene", "state": "cancelled", "title": "Hold a field",
                     "text": "cancelled at step 2/2; cleanup ran"})
    assert "cleanup ran" in sc.progress.text()


def test_scene_is_refused_when_a_step_would_be(panel2):
    sc = panel2.scenes_card
    n = len(panel2._sender.requests)
    bad = cmp.Scene(key="bad", title="Bad", steps=(cmp.Step("coil.ramp", {"i": 150.}),),
                    leaves_on="the coil")
    assert not panel2.run_scene(bad)
    assert len(panel2._sender.requests) == n
    assert "not started" in sc.progress.text()


# --- the strip above the tabs ------------------------------------------------------

def test_strip_trust_run_hazards_and_broadcasts(gui):
    strip = gui.summary
    gui._on_state_broadcast({"type": "trust", "trust": {"trusted": False,
                                                        "reason": "run 5 never reported"}})
    assert strip.banners["trust"].isVisibleTo(strip)
    assert "run 5 never reported" in strip.banners["trust"].label.text()
    gui._on_state_broadcast({"type": "run_pending", "run_pending": {"run_id": 6, "expt": "e"}})
    assert strip.banners["run"].isVisibleTo(strip)
    assert "Run 6" in strip.banners["run"].label.text()
    assert not gui.composite_panel.ops_allowed()[0]
    gui._on_state_broadcast({"type": "run_pending", "run_pending": None})
    gui._on_state_broadcast({"type": "trust", "trust": {"trusted": True, "reason": "ok"}})
    assert not strip.banners["trust"].isVisibleTo(strip)
    asked = []
    gui.request_state = lambda: asked.append(1)
    gui._on_state_broadcast({"type": "state_reset", "version": 40})
    assert asked == [1]
    gui._on_state_broadcast({"type": "busy", "seconds": 3.})
    assert strip.banners["busy"].isVisibleTo(strip)


class _AcceptingBox:
    """QMessageBox stand-in that clicks the accept-role button."""
    Icon = dc.QMessageBox.Icon
    ButtonRole = dc.QMessageBox.ButtonRole
    texts = []

    def __init__(self, *a):
        self._yes = None

    def setIcon(self, *a):
        pass

    def setWindowTitle(self, *a):
        pass

    def setText(self, text):
        _AcceptingBox.texts.append(text)

    def addButton(self, text, role):
        button = object()
        if role == self.ButtonRole.AcceptRole:
            self._yes = button
        return button

    def exec(self):
        return 0

    def clickedButton(self):
        return self._yes


def test_the_run_banner_offers_to_clear_a_fence(gui, monkeypatch):
    strip = gui.summary
    gui._on_state_broadcast({"type": "run_pending", "run_pending": {
        "run_id": 81000, "expt": "rabi", "token": "tok"}})
    banner = strip.banners["run"]
    assert banner.button.isVisibleTo(strip) and banner.button.text() == "Clear fence…"
    # liveOD also still shows the dead run as in progress: the button stays
    gui._telemetry_samples = {"live_od/run_in_progress": cmp.Sample(True),
                              "live_od/run_id": cmp.Sample(81000)}
    gui._refresh_summary()
    assert banner.button.isVisibleTo(strip)
    sent = []
    monkeypatch.setattr(dc, "QMessageBox", _AcceptingBox)
    gui._send_request = lambda obj, callback: sent.append((obj, callback))
    gui.composite_panel.operator.setText("ada")
    banner.button.click()
    obj, callback = sent[-1]
    assert obj["type"] == "clear_run_pending" and obj["token"] == "tok"
    assert obj["operator"] == "ada"
    assert "81000" in _AcceptingBox.texts[-1]
    callback({"status": "ok"})
    assert gui._run_pending is None and gui.composite_panel.run_pending is None
    assert "Run 81000" in banner.label.text() and not banner.button.isVisibleTo(strip)


def test_strip_interlock_and_live_od_from_telemetry(gui):
    gui._telemetry_samples = {"interlock/state": cmp.Sample("tripped"),
                              "live_od/run_in_progress": cmp.Sample(True),
                              "live_od/run_id": cmp.Sample(81234),
                              "live_od/expt_name": cmp.Sample("rabi"),
                              "live_od/n_shots": cmp.Sample(3),
                              "live_od/n_shots_expected": cmp.Sample(40)}
    gui._refresh_summary()
    assert "TRIPPED" in gui.summary.banners["interlock"].label.text()
    assert "81234" in gui.summary.banners["run"].label.text()
    assert "3/40" in gui.summary.banners["run"].label.text()


def test_changes_window_shows_the_server_journal(gui, qapp):
    gui._record_line("dac pid  1.000 V → 2.000 V")
    loads = []
    gui._load_journal = lambda: loads.append(1)
    gui.show_changes_log()
    win = gui._changes_window
    win.journal_button.setChecked(True)
    assert loads == [1]
    win.show_journal({"status": "ok", "path": "X:/j.jsonl", "entries": [
        {"t": "2026-09-26 10:00:01", "kind": "op_submit", "op": "coil.off", "seq": 3,
         "args": {"t_ramp": 0.05}, "operator": "ada", "client": "pc2", "origin": "gui"}]})
    assert win.list.count() == 1
    assert "coil.off(t_ramp=0.05) queued #3 by ada@pc2" in win.list.item(0).text()
    win.journal_button.setChecked(False)
    assert win.list.count() == 1 and "2.000 V" in win.list.item(0).text()
    win.close()
