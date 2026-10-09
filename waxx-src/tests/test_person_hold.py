"""A person's hold on the machine (waxx.util.device_state.person_hold), its
requests on the monitor server, the gates that report it, and its row on the
Sequences tab.  liveOD's POLL replies are dicts; nothing goes on the network;
files live under tmp_path."""
import json
import os

import pytest

from waxx.util.device_state import run_gate
from waxx.util.device_state.person_hold import PersonHold, describe


class Journal:
    def __init__(self):
        self.entries = []

    def record(self, kind, **fields):
        self.entries.append((kind, fields))

    @property
    def kinds(self):
        return [k for k, _ in self.entries]


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _poll(reset=False, run_id=None, in_progress=None):
    return {"ok": True, "run_in_progress": bool(run_id) if in_progress is None else in_progress,
            "run_id": run_id, "reset_requested": reset}


# --- the hold itself --------------------------------------------------------------------

def test_hold_and_release_are_journaled_and_kept_across_a_restart(tmp_path):
    path = tmp_path / "run_queue" / "person_hold.json"
    journal, seen = Journal(), []
    hold = PersonHold(str(path), journal=journal, on_change=seen.append)
    assert not hold.active and hold.text() == ""
    reply = hold.hold("aligning the tweezer", "jp@kong")
    assert reply["status"] == "ok" and reply["person_hold"]["active"]
    assert hold.text().startswith("person hold since ") and "by jp@kong: aligning" in hold.text()
    again = hold.hold("something else", "agent")
    assert again["already"] and again["person_hold"]["reason"] == "aligning the tweezer"
    # a new server reads it back: it never lapses by itself
    reborn = PersonHold(str(path))
    assert reborn.active and reborn.info()["by"] == "jp@kong"
    assert reborn.release("jp")["released"]["reason"] == "aligning the tweezer"
    assert not PersonHold(str(path)).active
    assert hold.release("x")["status"] == "ok"           # its own copy was still on
    assert journal.kinds == ["run_queue_hold", "run_queue_release"]
    assert [s["active"] for s in seen] == [True, False]


def test_release_without_a_hold_is_refused():
    reply = PersonHold().release("jp")
    assert reply["status"] == "error" and "no person hold" in reply["msg"]


def test_a_reminder_every_half_hour_while_held(caplog):
    clock = Clock()
    hold = PersonHold(clock=clock)
    hold.hold("fixing the MOT", "jp")
    with caplog.at_level("WARNING", logger="waxx.util.device_state.person_hold"):
        clock.t += 1700
        hold.tick()
        assert not [r for r in caplog.records if "Reminder" in r.getMessage()]
        clock.t += 200
        hold.tick()
        hold.tick()
        clock.t += 1800
        hold.tick()
    reminders = [r for r in caplog.records if "Reminder" in r.getMessage()]
    assert len(reminders) == 2 and "fixing the MOT" in reminders[0].getMessage()


# --- set by liveOD's Reset -----------------------------------------------------------------

def test_a_reset_on_a_persons_run_puts_the_hold_on():
    hold = PersonHold(clock=Clock())
    assert not hold.observe_poll(_poll(False, 900))        # baseline
    assert hold.observe_poll(_poll(True, 900))
    info = hold.info()
    assert info["active"] and info["by"] == "liveOD" and info["source"] == "live_od_reset"
    assert info["reason"].startswith("Reset in liveOD at ") and info["run_id"] == 900


def test_a_reset_with_no_run_puts_the_hold_on():
    hold = PersonHold(clock=Clock())
    hold.observe_poll(_poll(False))
    assert hold.observe_poll(_poll(True, 900, in_progress=False))
    assert hold.info()["run_id"] is None


@pytest.mark.parametrize("agent_ids, own_aborts", [({901}, set()), (set(), {901})])
def test_no_hold_for_an_agents_queued_run_or_the_queues_own_abort(agent_ids, own_aborts):
    hold = PersonHold(clock=Clock())
    hold.observe_poll(_poll(False, 901))
    assert not hold.observe_poll(_poll(True, 901), agent_ids, own_aborts)
    assert not hold.active


def test_only_the_edge_counts_and_the_first_poll_is_a_baseline():
    hold = PersonHold(clock=Clock())
    assert not hold.observe_poll(_poll(True, 900))         # pending at start: not new
    assert not hold.observe_poll(_poll(True, 900))         # still the same Reset
    assert not hold.observe_poll({"ok": False})            # liveOD down: no change
    assert not hold.observe_poll(_poll(False, 900))
    assert hold.observe_poll(_poll(True, 900))             # a new one
    hold.release("jp")
    assert not hold.observe_poll(_poll(True, 900))         # the same, still pending


# --- the gates -------------------------------------------------------------------------------

HELD = {"active": True, "since": 1_000_000.0, "by": "jp@kong", "reason": "aligning",
        "source": "request", "run_id": None}


def test_loops_verdict_and_assess_report_the_hold_first():
    status = {"state": 0, "run_pending": None, "run_loops": {}, "person_hold": HELD}
    v = run_gate.loops_verdict(status)
    assert v.state == "held" and v.blocks and not v.waivable
    assert v.reason == describe(HELD) and "by jp@kong: aligning" in v.reason
    assert run_gate.loops_verdict(dict(status, person_hold={"active": False})) is None

    class Live:
        def poll(self):
            return {"ok": True, "run_in_progress": True, "run_id": 5, "reset_requested": False,
                    "init_run_age_s": 1.0}

    class Mon:
        def get_status(self):
            return status
    assert run_gate.assess(Live(), Mon()).state == "held"


# --- on the monitor server ---------------------------------------------------------------

class Broadcasts:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def server(qapp, monkeypatch, tmp_path):
    from waxx.util.device_state.run_loop import LoopSpec
    from waxx.util.guis import monitor_server_gui as msg
    expt = tmp_path / "auto_tof.py"
    expt.write_text('"""BEC TOF loop."""\n')
    monkeypatch.setattr(msg, "StateBroadcaster", Broadcasts)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    s = msg.MonitorUDPServer(config_file_path=str(tmp_path / "state.json"),
                             journal_dir=str(tmp_path / "logs" / "ops_journal"),
                             run_loops=[LoopSpec("auto_tof", "BEC TOF loop", str(expt))])
    s.polls = [_poll(False)]
    s._live_od = lambda: s.polls[0]
    s.run_queue.poll_every_s = 0.0                     # every tick looks at liveOD
    s.loops["auto_tof"]._poll = lambda: _poll(False)
    s.ask = lambda obj: json.loads(s.generate_reply(json.dumps(obj)))
    yield s
    s.sock.close()


def test_server_hold_requests_status_and_broadcast(server, tmp_path):
    assert server.run_queue_dir == str(tmp_path / "run_queue_default")   # conftest's local default
    status = json.loads(server.generate_reply("status_json"))
    assert status["person_hold"]["active"] is False
    reply = server.ask({"type": "run_queue", "action": "hold", "reason": "aligning",
                        "by": "jp@kong"})
    assert reply["status"] == "ok" and reply["person_hold"]["by"] == "jp@kong"
    assert json.loads(server.generate_reply("status_json"))["person_hold"]["active"]
    assert (tmp_path / "run_queue_default" / "person_hold.json").is_file()
    assert {"type": "person_hold", "person_hold": reply["person_hold"]} in server._broadcaster.sent
    # the loops: Start refused, naming the hold
    refused = server.ask({"type": "run_loop", "action": "start", "loop": "auto_tof"})
    assert refused["status"] == "error" and refused["msg"].startswith("person hold since ")
    assert server.ask({"type": "run_queue", "action": "release", "by": "jp"})["status"] == "ok"
    assert server.ask({"type": "run_queue", "action": "release"})["status"] == "error"
    assert "unknown run_queue action" in server.ask({"type": "run_queue",
                                                     "action": "nonsense"})["msg"]
    kinds = [e["kind"] for e in server.journal.tail(50)]
    assert "run_queue_hold" in kinds and "run_queue_release" in kinds


def test_server_watch_sets_the_hold_on_a_reset(server):
    server.watch_tick()                                    # baseline
    server.polls[0] = _poll(True, 900)
    server.watch_tick()
    assert server.person_hold.active and server.person_hold.info()["run_id"] == 900


def test_server_watch_survives_live_od_being_down(server):
    def down():
        raise ConnectionError("no liveOD")
    server._live_od = down
    server.watch_tick()
    assert not server.person_hold.active


# --- the Sequences tab's row ------------------------------------------------------------------

def test_the_hold_row_puts_the_hold_on_and_releases_it(qapp, monkeypatch):
    from test_composite_panel import FakeSender
    from waxx.util.guis import sequences_panel as sp
    monkeypatch.setattr(sp, "_OpSender", FakeSender)
    panel = sp.SequencesPanel(log_line=lambda text: lines.append(text))
    lines = []
    try:
        panel.set_reachable(True)
        row = panel.hold_row
        assert row.isHidden()                                  # no person_hold reported yet
        panel.set_hold({"active": False})
        assert not row.isHidden() and row.button.text() == sp.HOLD_TEXT
        panel.ask_hold_reason = lambda: None                   # cancelled: nothing sent
        assert panel.toggle_hold() is False and panel._sender.requests == []
        panel.ask_hold_reason = lambda: "aligning"
        assert panel.toggle_hold()
        req = panel._sender.requests[-1]
        assert req["type"] == "run_queue" and req["action"] == "hold"
        assert req["reason"] == "aligning" and req["by"] == "Device Control GUI on test-pc"
        panel._sender.requested.emit(req["req"], {"status": "ok", "person_hold": HELD})
        assert row.button.text() == sp.RELEASE_TEXT and row.pill.text() == "HELD"
        assert "By jp@kong: aligning" not in row.status.text()
        assert "by jp@kong: aligning" in row.status.text()
        assert lines and lines[-1].startswith("[hold] person hold since")
        panel.toggle_hold()
        req = panel._sender.requests[-1]
        assert req["action"] == "release"
        panel._sender.requested.emit(req["req"], {"status": "error", "msg": "no person hold is on"})
        assert "✕ no person hold is on" in row.status.text()
        panel.set_reachable(False)
        assert not row.button.isEnabled()
    finally:
        panel.shutdown()


# --- the counter, not the level (review B2) ---------------------------------------------------

def _counted(person=0, queue=0, agent=0, last=None, reset=False, run_id=None):
    return {"ok": True, "run_in_progress": bool(run_id), "run_id": run_id,
            "reset_requested": reset, "reset_count": person + queue + agent,
            "reset_counts": {"person": person, "queue": queue, "agent": agent},
            "last_reset": last}


def test_a_quick_reset_between_two_polls_is_seen_by_the_count():
    hold = PersonHold(clock=Clock())
    assert not hold.observe_poll(_counted())                       # baseline
    # pressed and cleared (ABORT_RUN / INIT_RUN) between the polls: the level
    # never showed it
    last = {"at": 1_000_000.0, "count": 1, "run_id": 900, "source": "person"}
    assert hold.observe_poll(_counted(person=1, last=last, reset=False))
    info = hold.info()
    assert info["active"] and info["run_id"] == 900 and info["source"] == "live_od_reset"
    assert not hold.observe_poll(_counted(person=1, last=last))     # same count: nothing new


@pytest.mark.parametrize("source", ["queue", "agent"])
def test_the_queues_and_an_agents_aborts_never_hold(source):
    hold = PersonHold(clock=Clock())
    hold.observe_poll(_counted())
    last = {"at": 1.0, "count": 1, "run_id": 900, "source": source}
    assert not hold.observe_poll(_counted(**{source: 1}, last=last, reset=True))
    assert not hold.active


def test_a_persons_reset_hidden_behind_a_later_queue_abort_still_holds():
    hold = PersonHold(clock=Clock())
    hold.observe_poll(_counted())
    last = {"at": 1.0, "count": 2, "run_id": 901, "source": "queue"}   # the later one
    assert hold.observe_poll(_counted(person=1, queue=1, last=last))


def test_only_reset_count_uses_last_reset_source():
    hold = PersonHold(clock=Clock())
    poll = {"ok": True, "reset_count": 0, "last_reset": None}
    hold.observe_poll(poll)
    assert not hold.observe_poll({"ok": True, "reset_count": 1,
                                  "last_reset": {"source": "agent", "run_id": 5}})
    assert hold.observe_poll({"ok": True, "reset_count": 2,
                              "last_reset": {"source": "person", "run_id": 5}})


def test_a_live_od_restart_resets_the_counts():
    hold = PersonHold(clock=Clock())
    hold.observe_poll(_counted(person=4))
    assert not hold.observe_poll(_counted(person=0))           # restarted, nothing since
    assert hold.observe_poll(_counted(person=1, last={"source": "person", "run_id": 1}))


def test_a_claimed_reset_sets_no_hold():
    hold = PersonHold(clock=Clock())
    hold.observe_poll(_counted())
    claimed = []
    assert not hold.observe_poll(_counted(person=1, last={"source": "person", "run_id": None}),
                                 on_person_reset=lambda last: claimed.append(last) or True)
    assert claimed and not hold.active


def test_an_older_live_od_is_watched_by_the_level_with_one_warning(caplog):
    hold = PersonHold(clock=Clock())
    with caplog.at_level("WARNING", logger="waxx.util.device_state.person_hold"):
        hold.observe_poll(_poll(False, 900))
        hold.observe_poll(_poll(False, 900))
        assert hold.observe_poll(_poll(True, 900))
    assert len([r for r in caplog.records if "no reset_count" in r.getMessage()]) == 1
