"""Monitor panel changes of 2026-10-21 (user): warm start a stored setting
(default on) with a toggle on the panel; no SLM / tweezer AWG display; the run
loops under a "Sequences" dropdown; a Start liveOD button.

Offscreen Qt.  No socket is opened by the panel tests (socket.socket fails the
test); the run queue is built on a temporary folder with fakes and never
launches anything; liveOD's launcher is never run (``start`` is a fake, and
:func:`start_bat` is tested against a fake Popen); the beacon registry is a
fake module (importing the real one would bind UDP 50099 here)."""
import json
import os
import socket
import sys
import time
import types

import pytest
from PyQt6.QtWidgets import QApplication

from waxx.util.device_state import run_queue as rq
from waxx.util.device_state.run_queue import RunQueue
from waxx.util.guis import live_od_launch as lol
from waxx.util.guis import monitor_state_panel as msp

NOW = time.time()


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


# --- warm start: stored, default on ----------------------------------------------------------

class Journal:
    def __init__(self):
        self.entries = []

    def record(self, kind, **fields):
        self.entries.append(dict(fields, kind=kind))

    def of(self, kind):
        return [e for e in self.entries if e["kind"] == kind]


def _no_spawn(*a, **k):
    raise AssertionError("the queue launched something")


def queue(folder, journal=None, **kw):
    q = RunQueue(str(folder), journal=journal, spawn=_no_spawn,
                 adopt=lambda pid, started: None, kill=_no_spawn, client_name="kong",
                 set_priority=lambda pid, started: True, **kw)
    q.flush_journal()
    return q


def test_warm_start_is_on_when_nothing_is_stored(tmp_path):
    q = queue(tmp_path / "rq", warm_ahead=None)
    assert q.warm_ahead is True and q.warm_ahead_source == "default"
    info = q.info()
    assert info["warm_ahead"] is True and info["warm_ahead_source"] == "default"
    assert info["warm_ahead_env"] is None
    assert not (tmp_path / "rq" / rq.SETTINGS_FILE).exists()       # nothing written by reading
    # the class default for library use and tests stays "the value given"
    assert queue(tmp_path / "rq2").warm_ahead is False


def test_a_switch_is_stored_journalled_and_read_back_at_the_next_start(tmp_path):
    folder, j = tmp_path / "rq", Journal()
    q = queue(folder, j, warm_ahead=None)
    r = q.set_warm_ahead({"on": False, "by": "jp@kong", "owner": "person"})
    assert r == {"status": "ok", "warm_ahead": False, "was": True, "persisted": True,
                 "env_override": None}
    q.flush_journal()
    stored = json.loads((folder / rq.SETTINGS_FILE).read_text())["warm_ahead"]
    assert stored["on"] is False and stored["by"] == "jp@kong" and stored["owner"] == "person"
    assert isinstance(stored["at"], float) and stored["t"]
    # journalled: who, when, value -- in the ops journal and the queue's own journal
    rec = j.of("run_queue_warm_ahead")
    assert len(rec) == 1 and rec[0]["on"] is False and rec[0]["was"] is True
    assert (rec[0]["by"], rec[0]["owner"], rec[0]["stored"]) == ("jp@kong", "person", False)
    lines = [json.loads(ln) for ln in (folder / "journal.jsonl").read_text().splitlines()]
    mine = [ln for ln in lines if ln["kind"] == "run_queue_warm_ahead"]
    assert len(mine) == 1 and mine[0]["by"] == "jp@kong" and mine[0]["t"] and mine[0]["ts"]
    # no change, no record; the server restarts: the stored value comes back
    q.set_warm_ahead({"on": False, "by": "jp@kong", "owner": "person"})
    q.flush_journal()
    assert len(j.of("run_queue_warm_ahead")) == 1
    q2 = queue(folder, warm_ahead=None)
    assert q2.warm_ahead is False and q2.warm_ahead_source == "stored"
    # an agent may switch it too (the journal names the owner)
    r = q2.set_warm_ahead({"on": True, "by": "claude@kong", "owner": "agent"})
    assert r["persisted"] and queue(folder, warm_ahead=None).warm_ahead is True
    assert q2.set_warm_ahead({"on": True, "by": "x"})["status"] == "error"   # owner required


def test_the_env_var_overrides_the_stored_value_and_says_so(tmp_path, monkeypatch):
    folder, j = tmp_path / "rq", Journal()
    queue(folder, warm_ahead=None).set_warm_ahead({"on": False, "by": "jp", "owner": "person"})
    monkeypatch.setenv(rq.WARM_AHEAD_ENV, "1")
    q = queue(folder, j, warm_ahead=None)
    assert q.warm_ahead is True and q.warm_ahead_source == "env"
    start = j.of("run_queue_warm_ahead")
    assert len(start) == 1 and start[0]["owner"] == "env" and start[0]["on"] is True
    assert start[0]["stored"] is False
    assert q.info()["warm_ahead_env"] == "WAXX_QUEUE_WARM_AHEAD=1"
    report = q.set_warm_ahead({"owner": "person"})
    assert report["source"] == "env" and report["env_override"] == "WAXX_QUEUE_WARM_AHEAD=1"
    r = q.set_warm_ahead({"on": True, "by": "jp", "owner": "person"})
    # unchanged at run time, but now stored as on: that is a change, journalled
    assert r["persisted"] and r["env_override"] == "WAXX_QUEUE_WARM_AHEAD=1" and r["was"] is True
    q.flush_journal()
    assert len(j.of("run_queue_warm_ahead")) == 2
    monkeypatch.setenv(rq.WARM_AHEAD_ENV, "0")
    assert queue(folder, warm_ahead=None).warm_ahead is False
    monkeypatch.setenv(rq.WARM_AHEAD_ENV, "maybe")                     # not understood: ignored
    assert queue(folder, warm_ahead=None).warm_ahead_source == "stored"
    monkeypatch.delenv(rq.WARM_AHEAD_ENV)
    assert queue(folder, warm_ahead=None).warm_ahead is True           # what was stored


def test_a_failed_store_still_switches_and_says_not_stored(tmp_path):
    folder, j = tmp_path / "rq", Journal()
    q = queue(folder, j, warm_ahead=None)
    (folder / rq.SETTINGS_FILE).mkdir(parents=True)                 # cannot be replaced
    r = q.set_warm_ahead({"on": False, "by": "jp", "owner": "person"})
    assert r["warm_ahead"] is False and r["persisted"] is False and r["persist_error"]
    assert q.warm_ahead is False
    q.flush_journal()
    assert j.of("run_queue_warm_ahead")[0]["persist_error"]


def test_kq_warm_says_stored_or_overridden(tmp_path, monkeypatch):
    from waxx.util.device_state import kq

    class Client:
        def __init__(self, reply):
            self.reply = reply

        def warm_ahead(self, on, owner=None):
            return self.reply

    class Ctx:
        def __init__(self, reply):
            self.client, self.lines = Client(reply), []

        def say(self, text):
            self.lines.append(text)

    args = types.SimpleNamespace(state="on", agent=False)
    ctx = Ctx({"warm_ahead": True, "was": False, "persisted": True,
               "env_override": "WAXX_QUEUE_WARM_AHEAD=0"})
    kq._cmd_warm(ctx, args)
    assert "switched on (stored; WAXX_QUEUE_WARM_AHEAD=0 is set" in ctx.lines[0]
    assert "until the monitor server restarts" not in ctx.lines[0]
    ctx = Ctx({"warm_ahead": True, "was": False})                    # an older server
    kq._cmd_warm(ctx, args)
    assert ctx.lines[0].endswith("(until the monitor server restarts)")
    ctx = Ctx({"warm_ahead": True, "was": False, "persisted": False, "persist_error": "disk"})
    kq._cmd_warm(ctx, args)
    assert "NOT stored (disk)" in ctx.lines[0]
    ctx = Ctx({"warm_ahead": False, "source": "default"})
    kq._cmd_warm(ctx, types.SimpleNamespace(state=None, agent=False))
    assert ctx.lines == ["[kq] warm start is off (the default: nothing stored)"]


# --- the panel ---------------------------------------------------------------------------------

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

    def sent(self, kind):
        return [r for r in self.requests if r.get("type") == kind]


TOF = {"key": "auto_tof", "title": "BEC TOF loop", "expt": "auto_tof",
       "path": "C:/code/auto_tof.py", "about": "BEC TOF loop.", "state": "running",
       "text": "run 85601 in progress", "runs": 1, "run_id": 85601, "last": None,
       "started": NOW - 300, "owner": "person", "category": "yields"}
PICK = {"key": "expt_loop", "title": "Experiment loop", "expt": "", "path": "", "about": "",
        "pick": True, "root": "C:/code/k-exp/kexp/experiments", "rel": "",
        "state": "idle", "text": "not started", "runs": 0, "run_id": None, "last": None}

STATUS = {"state": 0, "state_name": "READY", "sub_state": "running", "reason": "",
          "since": NOW - 30, "pid": 1234, "expt_path": "C:/code/monitor.py",
          "trust": {"trusted": True, "reason": "", "since": NOW},
          "run_pending": None,
          "connections": {"awg": {"label": "Tweezer AWG", "state": "failed",
                                  "detail": "card not found", "since": NOW - 30}},
          "slm_reinit": {"label": "SLM reinit", "state": "failed", "detail": "unreachable"},
          "run_loops": {"auto_tof": TOF, "expt_loop": PICK},
          "run_queue": {"state": "idle", "warm_ahead": True, "warm_ahead_source": "default",
                        "warm_ahead_env": None},
          "person_hold": {"active": False}}


@pytest.fixture
def server(qapp, monkeypatch):
    monkeypatch.setattr(socket, "socket", _NoSocket)
    s = FakeServer()
    s.answers["get_journal"] = {"status": "ok", "entries": []}
    s.answers["run_loop"] = lambda obj: {"status": "ok", "loop": dict(PICK)}
    s.answers["run_queue"] = lambda obj: {"status": "ok", "warm_ahead": obj["on"],
                                          "was": not obj["on"], "persisted": True,
                                          "env_override": None}
    return s


def make_panel(server, **kw):
    p = msp.MonitorStatePanel(server, synchronous=True, by="jp@test", **kw)
    p.sequences.confirm = lambda title, text, verb="Send": True
    p.said = []
    p.message.connect(p.said.append)
    return p


@pytest.fixture
def panel(server):
    p = make_panel(server)
    yield p
    p.shutdown()


def test_no_slm_or_awg_anywhere_in_the_bar_or_details(panel):
    from PyQt6.QtWidgets import QAbstractButton, QLabel
    panel.set_state(dict(STATUS))
    panel.on_broadcast({"type": "slm_reinit", "slm_reinit": STATUS["slm_reinit"]})
    panel.on_broadcast({"type": "connections", "connections": STATUS["connections"]})
    widgets = (panel.bar.findChildren(QLabel) + panel.bar.findChildren(QAbstractButton)
               + panel.details.findChildren(QLabel))
    text = "\n".join(w.text() + "\n" + w.toolTip() for w in widgets)
    assert "SLM" not in text and "AWG" not in text and "reinit" not in text
    assert "Connections" not in text and "Run loops" not in text


def test_both_loops_are_under_the_sequences_dropdown_collapsed_at_start(panel, qapp):
    panel.set_state(dict(STATUS))
    panel.show()
    qapp.processEvents()
    assert panel.sequences_toggle.text() == "Sequences"
    assert not panel.sequences_expanded and not panel.sequences_area.isVisible()
    cards = panel.sequences.loop_cards
    assert set(cards) == {"auto_tof", "expt_loop"}
    for card in cards.values():
        assert panel.sequences_area.isAncestorOf(card)
        assert not panel.details.isAncestorOf(card) and not panel.bar.isAncestorOf(card)
    # the summary beside the toggle: the one active loop, both in its tooltip
    assert panel.sequences_chip.text() == "BEC TOF loop: running · yields"
    tip = panel.sequences_chip.toolTip()
    assert "BEC TOF loop: running · yields" in tip and "Experiment loop: idle" in tip
    panel.sequences_toggle.click()
    qapp.processEvents()
    assert panel.sequences_expanded and panel.sequences_area.isVisible()
    panel.set_sequences_expanded(False)
    assert not panel.sequences_area.isVisible()
    panel.hide()


def test_sequences_controls_send_as_before(server):
    p = make_panel(server)
    try:
        p.set_state(dict(STATUS, run_loops={"auto_tof": dict(TOF, state="idle"),
                                            "expt_loop": PICK}))
        p.set_sequences_expanded(True)
        card = p.sequences.loop_cards["auto_tof"]
        server.answers["run_loop"] = lambda obj: {"status": "ok", "loop": dict(TOF)}
        card.start_button.click()
        req = server.sent("run_loop")[-1]
        assert (req["action"], req["loop"], req["owner"]) == ("start", "auto_tof", "person")
        assert card.pill.text() == "RUNNING"                     # the server's reply
        two = {"auto_tof": dict(TOF), "expt_loop": dict(PICK, state="running",
                                                        text="run 1 of rabi")}
        p.set_state(dict(STATUS, run_loops=two))
        assert p.sequences_chip.text() == "2 active"
        p.set_state(dict(STATUS, run_loops={"auto_tof": dict(TOF, state="failed")}))
        assert p.sequences_chip.text() == "BEC TOF loop: failed"
    finally:
        p.shutdown()


def test_the_warm_toggle_shows_the_server_value_and_sends_a_persons_switch(panel, server):
    panel.set_state(dict(STATUS))
    box = panel.warm_box
    assert box.isChecked() and box.isEnabled()
    assert "Warm start is ON (the default: nothing stored)" in box.toolTip()
    box.click()                                                     # -> off
    req = server.sent("run_queue")[-1]
    assert req == {"type": "run_queue", "action": "warm_ahead", "on": False,
                   "owner": "person", "by": "jp@test"}
    assert not box.isChecked() and box.isEnabled()
    assert panel.said[-1] == "warm start off; stored"
    assert panel.state["run_queue"]["warm_ahead"] is False
    # a broadcast moves it (another client switched it)
    panel.on_broadcast({"type": "run_queue", "run_queue": dict(STATUS["run_queue"],
                                                               warm_ahead=True,
                                                               warm_ahead_source="stored")})
    assert box.isChecked() and "(stored)" in box.toolTip()
    # a refusal: the box goes back to the server's value
    server.answers["run_queue"] = {"status": "error", "msg": "server is restarting"}
    box.click()
    assert box.isChecked() and panel.said[-1] == "warm start: server is restarting"
    # an env override is named
    panel.set_state(dict(STATUS, run_queue=dict(STATUS["run_queue"], warm_ahead_source="env",
                                                warm_ahead_env="WAXX_QUEUE_WARM_AHEAD=1")))
    assert "WAXX_QUEUE_WARM_AHEAD=1" in box.toolTip()


def test_the_warm_toggle_is_off_without_a_value(panel, server):
    panel.set_state(dict(STATUS, run_queue={"state": "idle"}))       # an older server
    assert not panel.warm_box.isEnabled() and "older code" in panel.warm_box.toolTip()
    assert panel.set_warm_ahead(True) is False and server.sent("run_queue") == []
    panel.set_state(None)
    assert not panel.warm_box.isEnabled() and "not answering" in panel.warm_box.toolTip()


# --- Start liveOD -----------------------------------------------------------------------------

class Probe:
    def __init__(self, state="absent"):
        self.state = state

    def __call__(self):
        return self.state, {"running": "liveOD is running: its beacon came from 192.168.1.76",
                            "absent": "no liveOD beacon heard in the last 3 s",
                            "unknown": "listening for liveOD's beacon..."}[self.state]


class Starter:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def __call__(self, bat):
        self.calls.append(bat)
        if self.fail:
            raise self.fail
        return 4242


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_no_button_without_a_launcher(panel):
    assert panel.live_od_button is None


def test_the_button_is_off_while_live_od_beacons(server):
    probe, start = Probe("running"), Starter()
    p = make_panel(server, live_od_bat="C:/k/live_od.bat", live_od_start=start,
                   live_od_probe=probe)
    try:
        b = p.live_od_button
        assert p.bar.isAncestorOf(b) and b.text() == "Start liveOD"
        assert not b.isEnabled() and "liveOD is running" in b.toolTip()
        assert b.press() is False and start.calls == []
        assert p.said[-1].startswith("liveOD not started: liveOD is running")
        probe.state = "unknown"                                    # not listened long enough
        b.refresh()
        assert not b.isEnabled() and "listening" in b.toolTip()
        probe.state = "absent"
        b.refresh()
        assert b.isEnabled() and "C:/k/live_od.bat" in b.toolTip()
    finally:
        p.shutdown()


def test_one_press_one_launch_guarded_and_journalled(server):
    probe, start, clock = Probe("absent"), Starter(), Clock()
    p = make_panel(server, live_od_bat="C:/k/live_od.bat", live_od_start=start,
                   live_od_probe=probe)
    try:
        b = p.live_od_button
        b._clock = clock
        b.click()
        b.click()                                                    # a double click
        assert start.calls == ["C:/k/live_od.bat"]
        assert not b.isEnabled() and "waiting for its beacon" in b.toolTip()
        rec = server.sent("live_od_launch")
        assert len(rec) == 1
        assert (rec[0]["by"], rec[0]["owner"], rec[0]["bat"], rec[0]["ok"], rec[0]["pid"]) == \
            ("jp@test", "person", "C:/k/live_od.bat", True, 4242)
        assert rec[0]["host"] == socket.gethostname()
        assert "liveOD launcher started" in p.said[0]
        clock.t += 10
        b.refresh()
        assert not b.isEnabled()                                     # still guarded
        clock.t += 21                                                 # 31 s, no beacon
        b.refresh()
        assert b.isEnabled() and "has not beaconed" in b.toolTip()
        b._launched = (clock.t, time.time())                           # launched again ...
        probe.state = "running"                                        # ... and it came up
        b.refresh()
        assert not b.isEnabled() and b._launched is None
    finally:
        p.shutdown()


def test_a_refused_pc_and_a_failed_launch(server):
    start = Starter(fail=FileNotFoundError("no launcher at C:/k/live_od.bat"))
    p = make_panel(server, live_od_bat="C:/k/live_od.bat", live_od_start=start,
                   live_od_probe=Probe("absent"),
                   live_od_refusal=lambda: "liveOD runs on kong; this PC is 192.168.1.9")
    try:
        b = p.live_od_button
        assert not b.isEnabled() and "this PC is 192.168.1.9" in b.toolTip()
        assert b.press() is False and start.calls == []
        b._refusal = None
        b.refresh()
        assert b.press() is False and len(start.calls) == 1
        assert b.isEnabled()                                          # failed: no guard
        rec = server.sent("live_od_launch")[-1]
        assert rec["ok"] is False and "FileNotFoundError" in rec["error"]
        assert "liveOD launcher failed" in p.said[-1]
    finally:
        p.shutdown()


def test_an_unjournalled_press_says_so(server):
    server.answers["live_od_launch"] = {"status": "error", "msg": "unknown type live_od_launch"}
    p = make_panel(server, live_od_bat="C:/k/live_od.bat", live_od_start=Starter(),
                   live_od_probe=Probe("absent"))
    try:
        p.live_od_button.press()
        assert p.said[-1] == "liveOD start NOT journalled: unknown type live_od_launch"
    finally:
        p.shutdown()


def test_start_bat_starts_the_bat_in_its_own_console(tmp_path):
    bat = tmp_path / "live od" / "live_od.bat"
    bat.parent.mkdir()
    bat.write_text("echo never run\n")
    calls = []

    class P:
        def __init__(self, argv, **kw):
            calls.append((argv, kw))
            self.pid = 77

    assert lol.start_bat(str(bat), popen=P) == 77
    argv, kw = calls[0]
    assert argv == ["cmd", "/c", "start", "", "/D", str(bat.parent), str(bat)]
    assert kw["cwd"] == str(bat.parent)
    if os.name == "nt":
        assert kw["creationflags"] & lol._CREATE_BREAKAWAY_FROM_JOB
        assert kw["creationflags"] & lol._CREATE_NO_WINDOW

        def deny(argv, **kw):
            if kw["creationflags"] & lol._CREATE_BREAKAWAY_FROM_JOB:
                raise PermissionError("no breakaway")
            return P(argv, **kw)
        assert lol.start_bat(str(bat), popen=deny) == 77
        assert not calls[-1][1]["creationflags"] & lol._CREATE_BREAKAWAY_FROM_JOB
    with pytest.raises(FileNotFoundError):
        lol.start_bat(str(tmp_path / "missing.bat"), popen=P)


def test_beacon_probe_reads_the_registry(monkeypatch):
    seen = {}
    listened = {"s": 5.0}
    fake = types.ModuleType("beacon.discovery.client")
    fake.discover_entries = lambda prefix="", max_age=None, collect_for=0.0: {
        sid: e for sid, e in seen.items() if sid.startswith(prefix)}
    fake.listening_for = lambda: listened["s"]
    if "beacon.discovery" not in sys.modules:
        pkg = types.ModuleType("beacon.discovery")
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "beacon.discovery", pkg)
    monkeypatch.setitem(sys.modules, "beacon.discovery.client", fake)
    assert lol.beacon_probe()[0] == "absent"
    listened["s"] = 0.5
    assert lol.beacon_probe()[0] == "unknown"
    seen["live_od_broadcast"] = types.SimpleNamespace(ip="192.168.1.76", age=0.2)
    assert lol.beacon_probe()[0] == "unknown"                       # not liveOD's own id
    seen["live_od"] = types.SimpleNamespace(ip="192.168.1.76", age=0.2)
    state, text = lol.beacon_probe()
    assert state == "running" and "192.168.1.76" in text


# --- the server: start value, the toggle through the window, the journal ---------------------

def test_the_server_starts_warm_on_and_journals_a_launch_record(tmp_path, monkeypatch, qapp):
    from waxx.util.guis import monitor_server_gui as msg

    class Broadcasts:
        def __init__(self):
            self.sent = []

        def send(self, payload):
            self.sent.append(payload)

        def close(self):
            pass

    monkeypatch.setattr(msg, "StateBroadcaster", Broadcasts)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    monkeypatch.setattr(msg, "discover", lambda *a, **k: None)
    monkeypatch.setattr(msg.MonitorUDPServer, "run", lambda self: None)
    monkeypatch.setattr(msg.MonitorUDPServer, "_start_beacon",
                        lambda self: pytest.fail("a test started a beacon"))
    folder = tmp_path / "rq"
    server = msg.MonitorUDPServer(config_file_path=str(tmp_path / "state.json"),
                                  journal_dir=str(tmp_path / "ops"), run_queue_dir=str(folder))
    server.run_queue._spawn = _no_spawn                 # never the real detached launcher
    server.run_queue._adopt = lambda pid, started: None
    assert server.run_queue.warm_ahead is True
    status = json.loads(server.generate_reply("status_json"))
    assert status["run_queue"]["warm_ahead"] is True
    assert status["run_queue"]["warm_ahead_source"] == "default"
    reply = json.loads(server.generate_reply(json.dumps(
        {"type": "run_queue", "action": "warm_ahead", "on": False, "owner": "person",
         "by": "jp@test"})))
    assert reply["status"] == "ok" and reply["persisted"] is True
    assert json.loads((folder / rq.SETTINGS_FILE).read_text())["warm_ahead"]["on"] is False
    kinds = [e["kind"] for e in server.journal.tail(50)]
    assert "run_queue_warm_ahead" in kinds
    reply = json.loads(server.generate_reply(json.dumps(
        {"type": "live_od_launch", "by": "jp@test", "owner": "person", "host": "kong",
         "bat": "C:/k/live_od.bat", "ok": True, "pid": 4242, "probe": "absent",
         "extra": "dropped"})))
    assert reply == {"status": "ok"}
    rec = [e for e in server.journal.tail(50) if e["kind"] == "live_od_launch"][-1]
    assert (rec["by"], rec["host"], rec["pid"], rec["ok"]) == ("jp@test", "kong", 4242, True)
    assert "extra" not in rec
    from waxx.util.device_state.op_journal import describe_entry
    assert "Start liveOD pressed by jp@test on kong" in describe_entry(rec)
    warm = [e for e in server.journal.tail(50) if e["kind"] == "run_queue_warm_ahead"][-1]
    assert "warm start OFF by jp@test (person)" in describe_entry(warm)
    # a restarted server reads it back
    again = msg.MonitorUDPServer(config_file_path=str(tmp_path / "state.json"),
                                 journal_dir=str(tmp_path / "ops"), run_queue_dir=str(folder))
    assert again.run_queue.warm_ahead is False
    assert again.run_queue.warm_ahead_source == "stored"
