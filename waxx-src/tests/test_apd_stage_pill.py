"""The APD stage pill (waxx.util.guis.apd_stage_pill): what it shows from the
PDXC server's snapshot, when it allows a move, and that a move calls
``move_to`` (with ``force`` when the server already has it there).

Offscreen Qt and a fake PDXC client: nothing reaches the network (this
machine may run the real PDXC server), and the confirmation is stubbed.
"""
import os
import time

import pytest

from PyQt6.QtWidgets import QApplication

from waxx.util.guis import apd_stage_pill as asp


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class FakeClient:
    def __init__(self, position="out", com="connected", fail=None):
        self.position = position
        self.com = com
        self.fail = fail
        self.moves = []

    def get_snapshot(self):
        if self.fail:
            raise RuntimeError(self.fail)
        return {"com": {"status": self.com, "held": False, "last_error": ""},
                "position": self.position, "move_in_progress": False}

    def move_to(self, state, force=False):
        self.moves.append((state, force))
        if not force and state == self.position:
            return f"already {state}"
        self.position = state
        return f"moved {state}"


def _wait(qapp, pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if pred():
            return True
        time.sleep(0.01)
    return False


def _pill(qapp, client, run_text=None, lines=None):
    pill = asp.ApdStagePill(run_text=run_text,
                            record=(lines.append if lines is not None else None),
                            client_factory=lambda: client, poll_ms=0)
    pill.confirm = lambda *a, **k: True
    pill.poll()
    assert _wait(qapp, lambda: pill.polled_at is not None)
    return pill


def test_shows_position(qapp):
    pill = _pill(qapp, FakeClient(position="in"))
    assert pill.text() == "APD stage · in"
    assert pill.move_allowed() == (True, "")


def test_unknown_position(qapp):
    pill = _pill(qapp, FakeClient(position="unknown"))
    assert pill.text() == "APD stage · ?"
    assert pill.move_allowed()[0]


def test_unreachable_server(qapp):
    pill = _pill(qapp, FakeClient(fail="no beacon"))
    assert pill.text() == "APD stage"
    allowed, why = pill.move_allowed()
    assert not allowed and "not reachable" in why
    assert "no beacon" in pill.toolTip()


def test_com_not_connected(qapp):
    pill = _pill(qapp, FakeClient(com="error"))
    assert pill.text() == "APD stage · no COM"
    assert not pill.move_allowed()[0]


def test_allowed_during_a_run_with_warning(qapp):
    client, lines, asked = FakeClient(position="out"), [], []
    pill = _pill(qapp, client, run_text=lambda: "run 91234 (hf_bec)", lines=lines)
    assert pill.move_allowed() == (True, "")
    menu = pill.build_menu()
    enabled = {a.objectName(): a.isEnabled() for a in menu.actions() if a.objectName()}
    assert enabled["apd_stage_move_in"] is True
    assert enabled["apd_stage_move_out"] is True
    pill.confirm = lambda title, text, verb: asked.append(text) or True
    assert pill.request_move("in")
    assert "run 91234 (hf_bec) is using the machine" in asked[0]
    assert _wait(qapp, lambda: pill._moving is None)
    assert client.moves == [("in", False)]
    assert lines[0] == "[apd stage] moving in during run 91234 (hf_bec)"


def test_move_in(qapp):
    client, lines = FakeClient(position="out"), []
    pill = _pill(qapp, client, lines=lines)
    assert pill.request_move("in")
    assert pill.text() == "APD stage · moving…"
    assert _wait(qapp, lambda: pill._moving is None and pill.position == "in")
    assert client.moves == [("in", False)]
    assert pill.text() == "APD stage · in"
    assert lines == ["[apd stage] moving in", "[apd stage] moved in"]


def test_move_to_current_position_forces(qapp):
    client = FakeClient(position="out")
    pill = _pill(qapp, client)
    assert pill.request_move("out")
    assert _wait(qapp, lambda: pill._moving is None)
    assert client.moves == [("out", True)]


def test_cancelled_confirmation_sends_nothing(qapp):
    client = FakeClient(position="out")
    pill = _pill(qapp, client)
    pill.confirm = lambda *a, **k: False
    assert not pill.request_move("in")
    assert client.moves == []


def test_device_control_gui_hosts_the_pill(qapp, monkeypatch):
    """apd_stage=True puts the pill in the Connected devices group; off by default."""
    from test_device_control_gui import FakeSettings
    from waxx.util.guis import device_control_gui as dc
    monkeypatch.setattr(dc, "QSettings", FakeSettings)
    for name in ("_setup_update_sender", "_setup_state_listener", "_setup_state_worker",
                 "setup_status_checker", "setup_timer", "request_state"):
        monkeypatch.setattr(dc.DeviceStateGUI, name, lambda self, *a, **k: None)
    from test_composite_panel import FakeSender
    from waxx.util.guis import sequences_panel
    monkeypatch.setattr(sequences_panel, "_OpSender", FakeSender)
    # the pill polls the fake, never the network
    monkeypatch.setattr(asp, "_default_client_factory", lambda: FakeClient(position="in"))

    plain = dc.DeviceStateGUI()
    assert plain.apd_stage_pill is None
    plain.close()

    g = dc.DeviceStateGUI(apd_stage=True)
    try:
        assert g.apd_stage_pill is not None
        assert g.connected_devices.isAncestorOf(g.apd_stage_pill)
        # no monitor status yet: the host cannot tell whether a run is going
        assert g.apd_stage_pill._run_text() is None
    finally:
        g.close()
