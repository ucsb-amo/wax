"""A person's hold on the machine (waxx.util.device_state.person_hold), its
requests on the monitor server, the gates that report it, and its row on the
Sequences tab.  liveOD's POLL replies are dicts; nothing goes on the network;
files live under tmp_path."""
import datetime
import json
import os

import pytest

from waxx.util.device_state import run_gate
from waxx.util.device_state.person_hold import (
    ExpiryPolicy, PersonHold, describe, lapse_text, parse_expiry)


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


def test_an_agent_releases_a_persons_hold_only_the_one_a_person_told_it_to():
    import time as _time
    clock = Clock()
    journal = Journal()
    hold = PersonHold(clock=clock, journal=journal)
    hold.hold("aligning", "jp")
    since = _time.strftime("%H:%M:%S", _time.localtime(clock.t))
    refused = hold.release("bot", owner="agent")                         # nobody told it
    assert refused["status"] == "error" and "unless a person told it" in refused["msg"]
    refused = hold.release("bot", owner="agent", instructed_by="jp")     # which hold?
    assert refused["status"] == "error" and "ask again" in refused["msg"]
    assert hold.active
    ok = hold.release("bot", owner="agent", instructed_by="jp", hold_since=since)
    assert ok["status"] == "ok" and not hold.active
    assert journal.entries[-1] == ("run_queue_release",
                                   {"by": "bot", "owner": "agent", "held_since": clock.t,
                                    "held_by": "jp", "reason": "aligning",
                                    "instructed_by": "jp"})
    # the instruction was for that hold: a later hold is not covered by it
    clock.t += 60
    hold.hold("aligning again", "jp")
    again = hold.release("bot", owner="agent", instructed_by="jp", hold_since=since)
    assert again["status"] == "error" and hold.active
    # an earlier day's hold is named as kq status shows it, "MM-DD HH:MM"
    day = _time.strftime("%m-%d %H:%M", _time.localtime(clock.t))
    assert hold.release("bot", owner="agent", instructed_by="jp",
                        hold_since=day)["status"] == "ok"


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
    s.run_queue._spawn = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no job is launched in these tests"))
    s.run_queue._adopt = lambda pid, started: None     # never a real process
    s.loops["auto_tof"]._poll = lambda: _poll(False)
    s.loops["auto_tof"]._spawn = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no loop run is launched in these tests"))
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
    assert server.ask({"type": "run_queue", "action": "release", "by": "jp",
                       "owner": "person"})["status"] == "ok"
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
        panel.ask_hold_lapse = lambda: None                    # cancelled at the 2nd question
        assert panel.toggle_hold() is False and panel._sender.requests == []
        panel.ask_hold_lapse = lambda: {"mode": "policy"}
        assert panel.toggle_hold()
        req = panel._sender.requests[-1]
        assert req["type"] == "run_queue" and req["action"] == "hold"
        assert "expiry" not in req                             # the default: the server's rule
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


# --- a hold ends a loop as a Stop does (review N8) ---------------------------------------------

def test_a_hold_ends_a_loop_saying_so_and_starts_the_monitor(tmp_path):
    import threading
    from waxx.util.device_state.run_loop import LoopSpec, RunLoop
    expt = tmp_path / "auto_tof.py"
    expt.write_text("x = 1\n")
    hold = PersonHold(clock=Clock())
    started = []
    live = lambda: _poll(False)                               # noqa: E731
    gate = threading.Event()

    class Proc:
        pid = 1

        def __init__(self):
            self.stdout = iter(["Run ID: 101\n"])

        def wait(self):
            hold.hold("mine", "jp")                           # pressed during the run
            return 0
    loop = RunLoop(LoopSpec("auto_tof", "BEC TOF loop", str(expt)),
                   poll=lambda: dict(live(), last_outcome={"run_id": 101, "outcome": "saved"}),
                   spawn=lambda command, extra_env=None: Proc(), held=hold.text,
                   start_monitor=started.append, gap_s=0., poll_s=0.01)
    assert loop.start()["status"] == "ok"
    loop.join(5)
    info = loop.info()
    assert info["state"] == "latched" and info["text"].startswith("person hold since")
    assert "does not run while it is on" in info["text"]
    assert len(started) == 1                                  # as for a Stop
    assert loop.start()["msg"].startswith("person hold since")
    assert gate is not None


# --- the hold lapses by itself (2026-11-16) ------------------------------------------------------
#
# Times are local wall-clock times (datetime(...).timestamp()), so the tests
# hold in any timezone.  2026-11-18 is a Wednesday, 11-20 a Friday, 11-21 a
# Saturday.

def _t(day, hour, minute=0):
    return datetime.datetime(2026, 11, day, hour, minute).timestamp()


LAB = ExpiryPolicy(idle_s={"live_od_reset": 3600.0}, quiet_idle_s=3600.0,
                   quiet_start_h=19.0, quiet_end_h=7.0, quiet_weekends=True)


def test_quiet_hours_and_the_next_quiet_start():
    assert not LAB.is_quiet(_t(18, 10)) and LAB.is_quiet(_t(18, 19)) and LAB.is_quiet(_t(19, 6))
    assert not LAB.is_quiet(_t(19, 7)) and LAB.is_quiet(_t(21, 12))         # Saturday noon
    assert LAB.next_quiet(_t(18, 10)) == _t(18, 19)
    assert LAB.next_quiet(_t(18, 20)) == _t(18, 20)                         # already quiet
    weekends_only = ExpiryPolicy(quiet_idle_s=60.0, quiet_weekends=True)
    assert weekends_only.next_quiet(_t(18, 10)) == _t(21, 0)                # Saturday 00:00
    assert ExpiryPolicy(quiet_idle_s=60.0).next_quiet(_t(18, 10)) is None   # no quiet hours


@pytest.mark.parametrize("source, last, lapses", [
    ("request", _t(18, 10), _t(18, 19)),            # day: at the evening's start
    ("request", _t(18, 18, 30), _t(18, 19, 30)),    # active late: an hour after
    ("request", _t(18, 20), _t(18, 21)),            # in the evening: an hour after
    ("request", _t(21, 10), _t(21, 11)),            # Saturday: an hour after
    ("live_od_reset", _t(18, 10), _t(18, 11)),      # a Reset: an hour after, any time
    ("unreadable_file", _t(18, 20), None),          # never
])
def test_the_policys_lapse_time(source, last, lapses):
    assert LAB.lapses_at(source, last) == lapses


def test_without_a_policy_a_hold_never_lapses_and_reads_as_before():
    clock = Clock(_t(18, 10))
    hold = PersonHold(clock=clock)
    hold.hold("aligning", "jp")
    info = hold.info()
    assert info["expiry"] == {"mode": "policy"} and info["lapses_at"] is None
    assert lapse_text(info) == "" and " -- " not in hold.text()
    clock.t += 10 * 86400
    assert not hold.tick() and hold.active


def test_a_reset_hold_lapses_after_an_idle_hour_and_says_so(caplog):
    clock, journal, seen = Clock(_t(18, 10)), Journal(), []
    hold = PersonHold(clock=clock, policy=LAB, journal=journal, on_change=seen.append)
    hold.observe_poll(_counted())
    assert hold.observe_poll(_counted(person=1, last={"source": "person", "run_id": 7}))
    assert hold.info()["lapses_at"] == _t(18, 11)
    assert lapse_text(hold.info(), now=_t(18, 10)) == "lapses at 11:00 unless a person is active"
    clock.t = _t(18, 10, 59)
    assert not hold.tick() and hold.active
    clock.t = _t(18, 11) + 1
    with caplog.at_level("WARNING", logger="waxx.util.device_state.person_hold"):
        assert hold.tick()
    assert not hold.active and not hold.tick()
    kind, fields = journal.entries[-1]
    assert kind == "run_queue_hold_lapsed" and fields["source"] == "live_od_reset"
    assert fields["last_activity"] == _t(18, 10) and "no person activity since" in fields["rule"]
    assert seen[-1]["active"] is False
    assert any("PERSON HOLD LAPSED" in r.getMessage() for r in caplog.records)


def test_person_activity_moves_the_lapse_and_an_agents_does_not():
    clock = Clock(_t(18, 10))
    hold = PersonHold(clock=clock, policy=LAB)
    hold.hold("aligning", "jp")
    assert hold.info()["lapses_at"] == _t(18, 19)
    clock.t = _t(18, 18, 30)
    assert not hold.touch("an agent's queue request", owner="agent")
    assert hold.info()["lapses_at"] == _t(18, 19)
    assert hold.touch("update from Device Control GUI")
    info = hold.info()
    assert info["lapses_at"] == _t(18, 19, 30) and info["activity"].startswith("update from")
    clock.t = _t(18, 19, 1)
    assert not hold.tick()                         # 19:00 passed, but it moved
    clock.t = _t(18, 19, 31)
    assert hold.tick() and not hold.active
    assert not hold.touch("too late")              # no hold: nothing to move


def test_a_reset_while_held_counts_as_activity():
    clock = Clock(_t(18, 10))
    hold = PersonHold(clock=clock, policy=LAB)
    hold.observe_poll(_counted())
    hold.observe_poll(_counted(person=1, last={"source": "person"}))
    clock.t = _t(18, 10, 45)
    assert not hold.observe_poll(_counted(person=2, last={"source": "person"}))
    assert hold.info()["lapses_at"] == _t(18, 11, 45)
    assert hold.info()["activity"] == "a Reset in liveOD"


def test_never_and_a_set_time():
    clock, journal = Clock(_t(18, 10)), Journal()
    hold = PersonHold(clock=clock, policy=LAB, journal=journal)
    hold.hold("on the table", "jp", expiry="never")
    assert hold.info()["lapses_at"] is None and hold.text().endswith("never lapses by itself")
    clock.t += 5 * 86400
    assert not hold.tick() and hold.active
    hold.release("jp")
    clock.t = _t(18, 10)
    hold.hold("short job", "jp", expiry={"for_s": 7200})
    assert hold.info()["expiry"] == {"mode": "at", "at": _t(18, 12)}
    assert lapse_text(hold.info(), now=_t(18, 10)) == "lapses at 12:00"
    clock.t = _t(18, 11, 59)
    hold.touch("activity does not move a set time")
    assert hold.info()["lapses_at"] == _t(18, 12)
    clock.t = _t(18, 12)
    assert hold.tick()
    assert "the time it was set to end" in journal.entries[-1][1]["rule"]


def test_a_persons_hold_request_while_held_extends_or_changes_it_and_an_agents_does_not():
    clock, journal = Clock(_t(18, 10)), Journal()
    hold = PersonHold(clock=clock, policy=LAB, journal=journal)
    hold.hold("aligning", "jp")
    clock.t = _t(18, 19, 30)
    again = hold.hold("x", "agent-7", owner="agent", expiry="never")
    assert again["already"] and hold.info()["expiry"] == {"mode": "policy"}
    assert hold.info()["lapses_at"] == _t(18, 19)                  # not moved by an agent
    again = hold.hold("x", "jp")                                   # a person: activity
    assert again["already"] and again["person_hold"]["lapses_at"] == _t(18, 20, 30)
    assert again["person_hold"]["reason"] == "aligning"
    hold.hold("", "jp", expiry="never")
    assert hold.info()["expiry"] == {"mode": "never"} and hold.info()["lapses_at"] is None
    assert journal.kinds[-1] == "run_queue_hold_changed"
    assert hold.hold("", "jp", expiry={"for_s": -5})["status"] == "error"


def test_a_person_busy_on_the_machine_keeps_the_hold_and_a_failing_check_too(caplog):
    clock = Clock(_t(18, 20))
    busy = {"text": "a person's job 3 (rabi) is running"}
    hold = PersonHold(clock=clock, policy=LAB, busy=lambda: busy["text"])
    hold.hold("mine", "jp")
    clock.t = _t(18, 23)
    assert not hold.tick() and hold.active
    assert hold.info()["activity"] == busy["text"] and hold.info()["lapses_at"] == _t(19, 0)
    busy["text"] = ""
    clock.t = _t(19, 0, 1)
    assert hold.tick()                                 # an hour after the job ended
    hold.hold("again", "jp")

    def broken():
        raise RuntimeError("queue lock")
    hold.set_busy_check(broken)
    clock.t += 3 * 3600
    with caplog.at_level("ERROR", logger="waxx.util.device_state.person_hold"):
        assert not hold.tick() and not hold.tick()
    assert hold.active
    assert len([r for r in caplog.records if "busy check failed" in r.getMessage()]) == 1


def test_lapse_state_survives_a_restart_and_a_passed_time_lapses_at_once(tmp_path):
    path = str(tmp_path / "person_hold.json")
    clock, journal = Clock(_t(18, 20)), Journal()
    hold = PersonHold(path, clock=clock, policy=LAB)
    hold.hold("aligning", "jp")
    with open(path, encoding="utf-8") as f:
        stored = json.load(f)
    assert stored["expiry"] == {"mode": "policy"} and stored["last_activity"] == _t(18, 20)
    clock.t = _t(18, 23)                                         # server down meanwhile
    reborn = PersonHold(path, clock=clock, policy=LAB, journal=journal)
    assert reborn.active and reborn.info()["lapses_at"] == _t(18, 21)
    assert reborn.tick() and journal.kinds == ["run_queue_hold_lapsed"]
    assert not PersonHold(path, clock=clock, policy=LAB).active


def test_an_older_hold_file_counts_its_idle_time_from_the_restart(tmp_path):
    path = tmp_path / "person_hold.json"
    path.write_text(json.dumps({"active": True, "since": _t(16, 9), "by": "jp",
                                "reason": "old", "source": "request", "owner": "person"}))
    clock = Clock(_t(18, 20))
    hold = PersonHold(str(path), clock=clock, policy=LAB)
    info = hold.info()
    assert info["expiry"] == {"mode": "policy"} and info["last_activity"] == _t(18, 20)
    assert info["lapses_at"] == _t(18, 21) and "server started" in info["activity"]


def test_an_unreadable_hold_file_never_lapses(tmp_path):
    path = tmp_path / "person_hold.json"
    path.write_text("{not json")
    clock = Clock(_t(18, 20))
    hold = PersonHold(str(path), clock=clock, policy=LAB)
    assert hold.info()["source"] == "unreadable_file" and hold.info()["lapses_at"] is None
    clock.t += 3 * 86400
    assert not hold.tick() and hold.active


@pytest.mark.parametrize("value, why", [
    ({"at": _t(18, 9)}, "already passed"),
    ({"at": _t(18, 10) + 8 * 86400}, "more than 7 days"),
    ({"for_s": 0}, "above 0"),
    ({"mode": "soon"}, "mode must be"),
    (42, "must be 'policy'"),
])
def test_bad_expiry_is_refused(value, why):
    parsed, refusal = parse_expiry(value, _t(18, 10))
    assert parsed is None and why in refusal


def test_the_server_counts_a_persons_requests_and_not_the_monitors_or_an_agents(server):
    server.person_hold.policy = LAB
    reply = server.ask({"type": "run_queue", "action": "hold", "reason": "aligning",
                        "by": "jp@kong", "expiry": "never"})
    assert reply["person_hold"]["expiry"] == {"mode": "never"}
    assert reply["person_hold"]["activity"] == "the hold was put on"
    server.ask({"type": "update_batch", "origin": "monitor write-back", "updates": []})
    server.ask({"type": "run_queue", "action": "list"})
    server.ask({"type": "run_pending", "run_id": 5, "expt": "x", "token": "t",
                "owner": "agent"})
    assert server.person_hold.info()["activity"] == "the hold was put on"
    server.ask({"type": "trust_ack", "operator": "jp"})
    assert server.person_hold.info()["activity"] == "trust_ack from jp"
    assert json.loads(server.generate_reply("status_json"))["person_hold"]["activity"] \
        == "trust_ack from jp"
    assert server.ask({"type": "run_queue", "action": "hold", "by": "jp",
                       "expiry": {"for_s": 0}})["status"] == "error"


def test_the_queue_reports_a_persons_job_as_busy(server, tmp_path):
    rq = server.run_queue
    (tmp_path / "a.py").write_text('"""a."""\nclass A: pass\n')
    reply = server.ask({"type": "run_queue", "action": "submit", "path": str(tmp_path / "a.py"),
                        "by": "jp", "owner": "person", "run_host": "pc-under-test",
                        "sha256": "0" * 64, "expt_class": "A", "calibrates_declared": [],
                        "cold_why": ""})
    assert reply["status"] == "ok", reply
    job = rq._jobs[reply["ids"][0]]
    assert rq._person_busy() == ""                     # waiting for its PC: not about to run
    rq._current, job.state = job.id, "running"
    assert rq._person_busy() == f"a person's {job.name} is running"
    job.owner = "agent"
    assert rq._person_busy() == ""
    rq._current = None


def test_kq_hold_options():
    from types import SimpleNamespace
    from waxx.util.device_state import kq
    assert kq.parse_duration("2h") == 7200 and kq.parse_duration("90m") == 5400
    assert kq.parse_duration("1h30m") == 5400 and kq.parse_duration("45") == 2700
    for bad in ("soon", "0", "-1h"):
        with pytest.raises(ValueError):
            kq.parse_duration(bad)

    def ns(**k):
        return SimpleNamespace(**{"never": False, "policy": False, "for_": None, "until": None,
                                  **k})
    assert kq.hold_expiry(ns()) is None
    assert kq.hold_expiry(ns(never=True)) == "never"
    assert kq.hold_expiry(ns(for_="2h")) == {"for_s": 7200.0}
    assert kq.hold_expiry(ns(until="12:00"), now=_t(18, 10)) == {"at": _t(18, 12)}
    held = {"active": True, "since": _t(18, 10), "by": "jp", "reason": "x",
            "expiry": {"mode": "never"}, "lapses_at": None}
    assert kq._hold_text(held).endswith(" -- never lapses by itself")
