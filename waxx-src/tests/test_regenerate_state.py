"""Regenerate state file (2026-09-28): the monitor server's regenerate_state
request and the Device Control GUI's status pill menu item.

The state file is a temp file, the generator a fake, the broadcaster a
recorder, Qt offscreen; nothing reaches the network or the real state file.
"""
import json
import os
import time

import pytest

from waxx.util.comms_server.comm_server import STATES

OLD = {"dds": {"imaging": {"frequency": 100e6, "amplitude": 0.3, "v_pd": 1.0, "sw_state": 1,
                           "urukul_idx": 0, "ch": 0},
               "push": {"frequency": 80e6, "amplitude": 0.5, "v_pd": 0.0, "sw_state": 0,
                        "urukul_idx": 0, "ch": 1}},
       "ttl": {"shutter": {"ch": 3, "ttl_state": 1}},
       "dac": {"coil": {"ch": 0, "voltage": 2.5}},
       "metadata": {"state_trust": {"trusted": True, "reason": "end state of run 1",
                                    "since": 1.0}}}

DEFAULTS = {"dds": {"imaging": {"frequency": 110e6, "amplitude": 0.3, "v_pd": 1.0,
                                "sw_state": 0, "urukul_idx": 0, "ch": 0, "aom_order": 1},
                    "push": {"frequency": 80e6, "amplitude": 0.5, "v_pd": 0.0, "sw_state": 0,
                             "urukul_idx": 0, "ch": 1, "aom_order": 1}},
            "ttl": {"shutter": {"ch": 3, "ttl_state": 0},
                    "new_ttl": {"ch": 9, "ttl_state": 0}},
            "dac": {"coil": {"ch": 0, "voltage": 0.0}},
            "metadata": {"timestamp": "x"}}


class Recorder:
    def __init__(self, *a, **k):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _make_server(monkeypatch, tmp_path, generator):
    from waxx.util.guis import monitor_server_gui as msg
    monkeypatch.setattr(msg, "StateBroadcaster", Recorder)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    path = tmp_path / "state.json"
    path.write_text(json.dumps(OLD))
    s = msg.MonitorUDPServer(config_file_path=str(path), journal_dir=str(tmp_path / "journal"),
                             state_generator=generator)
    s.test_path = path
    return s


@pytest.fixture
def server(qapp, monkeypatch, tmp_path):
    calls = []

    def generator():
        calls.append(1)
        return json.loads(json.dumps(DEFAULTS))

    s = _make_server(monkeypatch, tmp_path, generator)
    s.test_calls = calls
    yield s
    s.sock.close()


def _ask(server, obj):
    return json.loads(server.generate_reply(json.dumps(obj)))


def _backups(tmp_path):
    folder = tmp_path / "journal" / "state_backups"
    return sorted(folder.iterdir()) if folder.is_dir() else []


def _kinds(server):
    return [e.get("kind") for e in server.journal.tail(50)]


def test_regenerates_backs_up_and_keeps_the_trust_flag(server, tmp_path):
    server.status.set_state(STATES.READY, "running")
    v0 = server._version
    reply = _ask(server, {"type": "regenerate_state", "client": "kong", "operator": "jp"})
    assert reply["status"] == "ok", reply
    assert reply["changed"] == ["dds.imaging", "ttl.new_ttl", "ttl.shutter", "dac.coil"] \
        or sorted(reply["changed"]) == sorted(["dds.imaging", "ttl.new_ttl", "ttl.shutter",
                                               "dac.coil"])
    assert "dds.push" not in reply["changed"]            # same values: not a change
    data = json.loads(server.test_path.read_text())
    for k in ("dds", "ttl", "dac"):
        assert data[k] == DEFAULTS[k]
    meta = data["metadata"]
    assert meta["state_trust"] == OLD["metadata"]["state_trust"]    # untouched
    assert meta["regenerated_by"] == "jp on kong"
    (backup,) = _backups(tmp_path)
    assert str(backup) == reply["backup"] == meta["regenerate_backup"]
    assert json.loads(backup.read_text()) == OLD                  # the file as it was
    assert server._version == v0 + 1 == reply["version"]
    assert {"type": "state_reset", "version": v0 + 1} in server._broadcaster.sent
    assert "state_regenerated" in _kinds(server)
    assert reply["monitor"] == "READY"


def test_allowed_while_the_monitor_is_simply_not_running(server, tmp_path):
    server.status.set_state(STATES.NOT_READY, "stopped_on_request")
    assert _ask(server, {"type": "regenerate_state"})["status"] == "ok"
    server.status.set_state(STATES.NOT_READY, "never_started")
    assert _ask(server, {"type": "regenerate_state"})["status"] == "ok"
    assert len(_backups(tmp_path)) == 2                         # one per regeneration


@pytest.mark.parametrize("setup, text", [
    (lambda s: s.status.set_state(STATES.LOADING, "starting"), "monitor is starting"),
    (lambda s: s.status.set_state(STATES.NOT_READY, "interrupted_by_run"), "taken the core"),
    (lambda s: setattr(s, "_run_pending", {"run_id": 7, "expt": "x", "t0": time.monotonic()}),
     "run 7 (x) is starting"),
    (lambda s: setattr(s.reset, "_current", {"state": "running"}), "state reset is running"),
])
def test_refused_while_anything_could_hold_or_want_the_state(server, tmp_path, setup, text):
    server.status.set_state(STATES.READY, "running")
    setup(server)
    reply = _ask(server, {"type": "regenerate_state"})
    assert reply["status"] == "error" and text in reply["msg"]
    assert json.loads(server.test_path.read_text()) == OLD
    assert _backups(tmp_path) == [] and server.test_calls == []
    assert "state_regenerate_refused" in _kinds(server)


def test_refused_without_a_generator_or_when_it_fails(qapp, monkeypatch, tmp_path):
    s = _make_server(monkeypatch, tmp_path, None)
    try:
        assert "no state generator" in _ask(s, {"type": "regenerate_state"})["msg"]
        assert json.loads(s.generate_reply("status_json"))["state_generator"] is False

        def broken():
            raise RuntimeError("frames did not build")

        s._state_generator = broken
        assert json.loads(s.generate_reply("status_json"))["state_generator"] is True
        assert "frames did not build" in _ask(s, {"type": "regenerate_state"})["msg"]
        s._state_generator = lambda: {"dds": {}}
        assert "no dds/ttl/dac" in _ask(s, {"type": "regenerate_state"})["msg"]
        assert json.loads(s.test_path.read_text()) == OLD and _backups(tmp_path) == []
    finally:
        s.sock.close()


def test_no_backup_no_regeneration(server, tmp_path, monkeypatch):
    import shutil
    server.status.set_state(STATES.READY, "running")

    def fail(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", fail)
    reply = _ask(server, {"type": "regenerate_state"})
    assert reply["status"] == "error" and "could not back up" in reply["msg"]
    assert "disk full" in reply["msg"] and "nothing changed" in reply["msg"]
    assert json.loads(server.test_path.read_text()) == OLD


def test_an_unreadable_file_is_backed_up_as_it_is_and_rebuilt(server, tmp_path):
    server.status.set_state(STATES.READY, "running")
    server.test_path.write_text("{not json")
    server._state_cache = None
    server._state_mtime = None
    reply = _ask(server, {"type": "regenerate_state"})
    assert reply["status"] == "ok"
    (backup,) = _backups(tmp_path)
    assert backup.read_text() == "{not json"
    data = json.loads(server.test_path.read_text())
    assert data["dds"] == DEFAULTS["dds"] and data["ttl"] == DEFAULTS["ttl"]
    assert data["metadata"]["state_trust"] == server._trust
    assert data["metadata"]["regenerate_backup"] == str(backup)


def test_journal_lines():
    from waxx.util.device_state.op_journal import describe_entry
    base = {"t": "2026-09-28T12:00:00"}
    line = describe_entry({**base, "kind": "state_regenerated", "by": "jp on kong",
                           "n_changed": 4, "backup": "B"})
    assert "REGENERATED" in line and "4 channel(s)" in line and "backup B" in line
    assert "REFUSED: busy" in describe_entry({**base, "kind": "state_regenerate_refused",
                                              "by": "kong", "msg": "busy"})


# --- the Device Control GUI -------------------------------------------------------------

@pytest.fixture
def gui(qapp, monkeypatch):
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
                        lambda obj, cb, **kw: g._test_sent.append((obj, kw)))
    monkeypatch.setattr(g, "_live_od_status", lambda: {})
    yield g
    g.close()


def _regen_action(g):
    menu, actions = g._build_status_menu()
    regen = next(a for a in actions if a.objectName() == "regenerate_state")
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    return regen, texts, actions


def test_menu_item_needs_a_server_that_offers_it(gui):
    regen, texts, _ = _regen_action(gui)
    assert "Regenerate state file…" in texts and not regen.isEnabled()
    gui._on_status_detail({"state": int(STATES.READY), "sub_state": "running"})
    regen, texts, _ = _regen_action(gui)
    assert not regen.isEnabled() and any("does not offer it" in t for t in texts)
    gui._on_status_detail({"state": int(STATES.READY), "sub_state": "running",
                           "state_generator": True})
    regen, _, _ = _regen_action(gui)
    assert regen.isEnabled()
    gui._on_status_detail({"state": int(STATES.NOT_READY), "sub_state": "interrupted_by_run",
                           "state_generator": True})
    regen, texts, _ = _regen_action(gui)
    assert not regen.isEnabled() and any("taken the core" in t for t in texts)
    gui._on_status_detail({"state": int(STATES.READY), "sub_state": "running",
                           "state_generator": True, "run_pending": {"run_id": 5}})
    regen, texts, _ = _regen_action(gui)
    assert not regen.isEnabled() and any("run 5 is starting" in t for t in texts)
    gui.on_connection_failed()
    regen, _, _ = _regen_action(gui)
    assert not regen.isEnabled()


def test_confirm_then_one_untimed_out_attempt(gui, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    gui._on_status_detail({"state": int(STATES.READY), "sub_state": "running",
                           "state_generator": True})
    texts = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: texts.append(self.text()) or 0)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: next(b for b in self.buttons() if b.text() == "Cancel"))
    _, _, actions = _regen_action(gui)
    actions[next(a for a in actions if a.objectName() == "regenerate_state")]()
    assert gui._test_sent == [] and "applies the defaults to the HARDWARE at once" in texts[0]
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: next(b for b in self.buttons() if b.text() == "Regenerate"))
    gui._regenerate_state_file()
    obj, kw = gui._test_sent[-1]
    assert obj["type"] == "regenerate_state" and kw == {"timeout": 30.0, "attempts": 1}


def test_reply_is_logged_and_shown(gui, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    gui._on_regenerate_reply({"status": "ok", "changed": ["dds.imaging", "dac.coil"],
                              "backup": "C:/b.json"})
    assert "2 channel(s) changed" in shown[-1] and "C:/b.json" in shown[-1]
    assert any("[state] regenerated" in line and "backup C:/b.json" in line
               for line in gui._changes)
    assert any("dds.imaging, dac.coil" in line for line in gui._changes)
    gui._on_regenerate_reply({"status": "error", "msg": "a run is starting"})
    assert "a run is starting" in shown[-1]


def test_request_worker_passes_timeouts_only_when_asked(qapp, monkeypatch):
    from waxx.util.guis import device_control_gui as dc
    seen = []

    class OldFake:                      # the signature older fakes and callers use
        def __init__(self, *a, **k):
            pass

        def request(self, obj):
            seen.append(("old", obj))
            return {"status": "ok"}

    class NewFake(OldFake):
        def request(self, obj, timeout=5.0, attempts=2):
            seen.append(("new", timeout, attempts))
            return {"status": "ok"}

    monkeypatch.setattr(dc, "MonitorClient", OldFake)
    dc._RequestWorker({"type": "x"}).run()
    monkeypatch.setattr(dc, "MonitorClient", NewFake)
    dc._RequestWorker({"type": "x"}, timeout=30.0, attempts=1).run()
    assert seen == [("old", {"type": "x"}), ("new", 30.0, 1)]
