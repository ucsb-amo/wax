"""The shared run gate (waxx.util.device_state.run_gate): the classifier's
states from dict inputs, the pid/host rule, the fence, the fetcher, and the
RUN_EXITED helper.  Nothing goes on the network: POLL replies, the monitor's
status and the liveOD client are fakes; the only real process checked is this
one and a child it started and waited for."""
import os
import socket
import subprocess
import sys

import pytest

from waxx.util.device_state import run_gate
from waxx.util.device_state.run_gate import GateState, assess, classify, tell_live_od_run_exited

HERE = socket.gethostname()
NOW = 1_800_000_000.0


def _poll(**kw):
    reply = {"ok": True, "run_in_progress": False, "run_id": 85527, "reset_requested": False,
             "n_shots": 0, "n_shots_expected": 0, "last_shot_age_s": None,
             "init_run_age_s": None, "run_state": "idle", "expt_name": "",
             "last_outcome": {}, "client_pid": None, "client_host": None, "launcher": ""}
    reply.update(kw)
    return reply


def _running(**kw):
    base = dict(run_in_progress=True, run_id=85528, expt_name="hf_bec", n_shots=10,
                n_shots_expected=40, init_run_age_s=320.0, last_shot_age_s=12.0,
                run_state="running", client_pid=4242, client_host=HERE, launcher="run_lock")
    base.update(kw)
    return _poll(**base)


def _dead(pid):
    return False


def _alive(pid):
    return True


def _never(pid):
    raise AssertionError("the pid must not be checked here")


# -- free / unknown ------------------------------------------------------------------

def test_free_when_idle_and_no_fence():
    st = classify(_poll(), None, now=NOW, pid_alive=_never)
    assert st.state == "free" and not st.waivable and not st.blocks


@pytest.mark.parametrize("poll", [None, {"ok": False, "error": "timeout"}, {"ok": True}])
def test_unknown_when_poll_failed_or_lacks_run_in_progress(poll):
    st = classify(poll, None, now=NOW, pid_alive=_never)
    assert st.state == "unknown" and not st.waivable and st.blocks
    assert "cannot confirm" in st.reason


def test_old_server_without_client_fields_is_never_dead():
    poll = _running()
    for k in ("client_pid", "client_host", "launcher"):
        poll.pop(k)
    poll["last_shot_age_s"] = 5000.0
    st = classify(poll, None, now=NOW, pid_alive=_never)
    assert st.state == "wedged" and not st.waivable
    assert "no client pid" in st.reason


# -- live / wedged ---------------------------------------------------------------------

def test_live_with_recent_shots_never_waivable_even_if_pid_unknown():
    st = classify(_running(client_host="other-pc"), None, now=NOW, pid_alive=_never)
    assert st.state == "live" and not st.waivable and st.run_id == 85528


def test_live_young_run_without_a_shot():
    st = classify(_running(n_shots=0, last_shot_age_s=None, init_run_age_s=30.0), None,
                  now=NOW, pid_alive=_alive)
    assert st.state == "live" and "no shot yet" in st.reason


def test_wedged_when_no_shot_since_init_for_long_and_pid_alive():
    st = classify(_running(n_shots=0, last_shot_age_s=None,
                           init_run_age_s=run_gate.YOUNG_RUN_S + 1), None,
                  now=NOW, pid_alive=_alive)
    assert st.state == "wedged" and not st.waivable and st.reason.startswith("WEDGED")
    assert "person must look" in st.reason


def test_live_limit_follows_the_shot_period():
    # 10 shots in 300 s -> 30 s period -> limit 150 s (5 periods, over the 120 s floor)
    live = classify(_running(init_run_age_s=440.0, last_shot_age_s=140.0), None,
                    now=NOW, pid_alive=_alive)
    assert live.state == "live" and live.detail["live_limit_s"] == pytest.approx(150.0)
    wedged = classify(_running(init_run_age_s=460.0, last_shot_age_s=160.0), None,
                      now=NOW, pid_alive=_alive)
    assert wedged.state == "wedged" and wedged.detail["live_limit_s"] == pytest.approx(150.0)


def test_live_limit_has_a_floor():
    # 10 shots in 10 s -> 1 s period: the limit is the 120 s floor
    st = classify(_running(init_run_age_s=100.0, last_shot_age_s=90.0), None,
                  now=NOW, pid_alive=_alive)
    assert st.state == "live" and st.detail["live_limit_s"] == run_gate.LIVE_MIN_S


def test_a_saving_run_is_live_whatever_its_shot_age():
    st = classify(_running(run_state="saving", last_shot_age_s=5000.0, init_run_age_s=9000.),
                  None, now=NOW, pid_alive=_alive)
    assert st.state == "live" and "saved" in st.reason


# -- a run being saved (review B1) --------------------------------------------------------

@pytest.mark.parametrize("poll", [
    # Reset pressed during the save: run_state "aborting", Abort pending
    _running(save_in_progress=True, reset_requested=True, run_state="aborting"),
    _running(save_in_progress=True, reset_requested=True, run_state="no_reply"),
    # its process has exited (as it does once END_RUN is sent): dead client + saving
    _running(save_in_progress=True, run_state="saving"),
    # an older liveOD: no save_in_progress, run_state "saving"
    _running(run_state="saving"),
])
def test_a_run_being_saved_is_live_before_any_client_check(poll):
    st = classify(poll, None, now=NOW, pid_alive=_never)
    assert st.state == "live" and not st.waivable and "being saved" in st.reason


# -- dead client / reset pending -------------------------------------------------------

def test_dead_client_is_waivable_only_when_the_pid_is_dead_on_this_host():
    st = classify(_running(), None, now=NOW, pid_alive=_dead)
    assert st.state == "dead_client" and st.waivable and not st.blocks
    assert "pid 4242" in st.reason and "run_lock" in st.reason
    assert st.detail["client"]["alive"] is False


@pytest.mark.parametrize("client", [{"client_pid": None, "client_host": None},
                                    {"client_host": "other-pc"}])
def test_a_run_liveod_heard_exit_is_a_dead_client_without_a_pid(client):
    # review N4: run_state "exited" while frames are still due
    st = classify(_running(run_state="exited", last_shot_age_s=1.0, **client), None,
                  now=NOW, pid_alive=_never)
    assert st.state == "dead_client" and st.waivable
    assert "run_state exited" in st.reason


def test_an_exited_run_whose_pid_is_alive_is_not_waived():
    st = classify(_running(run_state="exited", last_shot_age_s=1.0), None, now=NOW,
                  pid_alive=_alive)
    assert st.state == "live" and not st.waivable


def test_the_host_comparison_ignores_case():
    st = classify(_running(client_host=HERE.swapcase()), None, now=NOW, pid_alive=_dead)
    assert st.state == "dead_client"


def test_a_pid_on_another_host_is_never_checked():
    st = classify(_running(client_host=HERE + "-elsewhere", last_shot_age_s=9000.0), None,
                  now=NOW, pid_alive=_never)
    assert st.state == "wedged" and not st.waivable
    assert "not on" in st.detail["client"]["why"]


def test_a_failing_pid_check_counts_as_unknown():
    def broken(pid):
        raise OSError("nope")
    st = classify(_running(last_shot_age_s=9000.0), None, now=NOW, pid_alive=broken)
    assert st.state == "wedged" and st.detail["client"]["alive"] is None


def test_reset_pending_with_a_dead_client_is_waivable():
    # run 85528, 2026-10-09: killed with its client, Abort pending, no shot
    poll = _running(reset_requested=True, n_shots=0, last_shot_age_s=None,
                    init_run_age_s=3600.0, run_state="no_reply")
    st = classify(poll, None, now=NOW, pid_alive=_dead)
    assert st.state == "reset_pending" and st.waivable and st.run_id == 85528
    assert "next INIT_RUN" in st.reason and "discarded" in st.reason


@pytest.mark.parametrize("check, host", [(_alive, HERE), (_never, "other-pc")])
def test_reset_pending_with_a_live_or_unknown_client_is_not_waivable(check, host):
    poll = _running(reset_requested=True, client_host=host)
    st = classify(poll, None, now=NOW, pid_alive=check)
    assert st.state == "reset_pending" and not st.waivable


def test_reset_pending_without_a_run_after_a_failed_save_is_not_waivable():
    # review S3: an older liveOD's next INIT_RUN would delete the failed save's file
    poll = _poll(reset_requested=True, client_pid=4242, client_host=HERE,
                 last_outcome={"run_id": 85527, "outcome": "save_failed"})
    st = classify(poll, None, now=NOW, pid_alive=_dead)
    assert st.state == "reset_pending" and not st.waivable
    assert "save failed" in st.reason and "pid" not in st.reason


def test_reset_pending_without_a_run_follows_the_last_clients_pid():
    poll = _poll(reset_requested=True, client_pid=4242, client_host=HERE)
    assert classify(poll, None, now=NOW, pid_alive=_dead).waivable
    assert not classify(poll, None, now=NOW, pid_alive=_alive).waivable
    old = _poll(reset_requested=True)
    st = classify(old, None, now=NOW, pid_alive=_never)
    assert st.state == "reset_pending" and not st.waivable


# -- the monitor's run fence ----------------------------------------------------------------

def test_a_young_fence_makes_an_idle_machine_live():
    fence = {"run_id": 85600, "expt": "rabi", "since": NOW - 10.0, "token": "t"}
    st = classify(_poll(), fence, now=NOW, pid_alive=_never)
    assert st.state == "live" and not st.waivable and st.run_id == 85600
    assert "announced itself" in st.reason and "10 s ago" in st.reason


@pytest.mark.parametrize("ready", [0, "READY"])
def test_a_fence_older_than_its_ttl_is_not_counted_while_the_monitor_is_ready(ready):
    fence = {"run_id": 85600, "expt": "rabi", "since": NOW - run_gate.FENCE_TTL_S - 1}
    st = classify(_poll(), fence, now=NOW, pid_alive=_never, monitor_state=ready)
    assert st.state == "free" and "not counted" in st.reason
    assert st.detail["fence"]["active"] is False


@pytest.mark.parametrize("state", [1, 2, "NOT_READY", None])
def test_an_old_fence_stands_while_the_monitor_is_not_ready(state):
    # review S6: the server lapses a fence only while READY (it clears it when
    # the run takes the core); not READY or unknown: the fence stands
    fence = {"run_id": 85600, "expt": "rabi", "since": NOW - 10 * run_gate.FENCE_TTL_S}
    st = classify(_poll(), fence, now=NOW, pid_alive=_never, monitor_state=state)
    assert st.state == "live" and not st.waivable and st.run_id == 85600
    assert "fence held 1200 s, monitor not READY" in st.reason


@pytest.mark.parametrize("poll", [
    # liveOD closed run 85600 (exit grace, or run_lock's RUN_EXITED): it is
    # liveOD's last run, not in progress
    _poll(run_id=85600, run_state="exited",
          last_outcome={"run_id": 85600, "outcome": "exited"}),
    # liveOD has moved on (a later no-save run ended), the outcome names 85600
    _poll(run_id=0, last_outcome={"run_id": 85600, "outcome": "superseded"}),
    # not in progress and liveOD's run id is the fence's, whatever the record says
    _poll(run_id=85600, last_outcome={}),
])
@pytest.mark.parametrize("state", [2, None])
def test_a_killed_runs_fence_with_the_monitor_off_does_not_count_once_liveod_closed_it(
        poll, state):
    # second review: a run killed hard with the monitor off keeps its fence up
    # forever (the server lapses fences only while READY)
    fence = {"run_id": 85600, "expt": "hf_bec", "since": NOW - 3600.0}
    st = classify(poll, fence, now=NOW, pid_alive=_never, monitor_state=state)
    assert st.state == "free" and not st.blocks
    assert "liveOD shows that run ended -- not counted" in st.reason
    assert st.detail["fence"]["ended"] is True


@pytest.mark.parametrize("poll", [
    # announced after its INIT_RUN, in progress in liveOD, no record of an end
    _running(run_id=85600, n_shots=0, last_shot_age_s=None, init_run_age_s=30.0),
    # liveOD shows only the run before it (e.g. a liveOD restarted since)
    _poll(run_id=85599, last_outcome={"run_id": 85599, "outcome": "saved"}),
])
def test_an_announced_run_liveod_has_not_seen_end_is_still_live(poll):
    fence = {"run_id": 85600, "expt": "hf_bec", "since": NOW - 20.0}
    st = classify(poll, fence, now=NOW, pid_alive=_alive, monitor_state=2)
    assert st.state == "live" and not st.waivable
    assert st.detail["fence"]["ended"] is False


@pytest.mark.parametrize("fid", [0, None, "85600", True, -3])
def test_only_a_real_run_ids_fence_can_be_told_ended(fid):
    # final review nit: every save_data=False run is run id 0, so liveOD's run id
    # 0 (an earlier no-save run) says nothing about a fence of 0 -- a run that
    # never sent INIT_RUN would otherwise read as ended
    fence = {"run_id": fid, "expt": "nosave", "since": NOW - 20.0}
    poll = _poll(run_id=0 if fid in (0, None, True, -3) else 85600,
                 last_outcome={"run_id": 0, "outcome": "saved"})
    st = classify(poll, fence, now=NOW, pid_alive=_never, monitor_state=2)
    assert st.state == "live" and st.detail["fence"]["ended"] is False


def test_a_fence_without_a_date_counts():
    st = classify(_poll(), {"run_id": 85600, "expt": "rabi"}, now=NOW, pid_alive=_never)
    assert st.state == "live"


def test_a_foreign_fence_overrides_a_waiver():
    fence = {"run_id": 85600, "expt": "rabi", "since": NOW - 5.0}
    st = classify(_running(), fence, now=NOW, pid_alive=_dead)
    assert st.state == "live" and not st.waivable and "pid 4242" in st.reason


def test_the_runs_own_fence_changes_nothing():
    fence = {"run_id": 85528, "expt": "hf_bec", "since": NOW - 5.0}
    assert classify(_running(), fence, now=NOW, pid_alive=_dead).state == "dead_client"


# -- the fetcher ---------------------------------------------------------------------------

class FakeLiveOD:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.sent, self.closed = reply, error, [], False

    def poll(self):
        if self.error:
            raise self.error
        return dict(self.reply)

    def _send_recv(self, msg):
        self.sent.append(msg)
        return {"ok": True}

    def close(self):
        self.closed = True


class FakeMonitor:
    def __init__(self, status):
        self.status = status

    def get_status(self):
        return self.status


def test_assess_combines_poll_and_fence():
    st = assess(FakeLiveOD(_poll()), FakeMonitor({"state": 0, "run_pending": None}),
                now=NOW, pid_alive=_never)
    assert st.state == "free"
    fenced = assess(FakeLiveOD(_poll()),
                    FakeMonitor({"run_pending": {"run_id": 9, "expt": "x", "since": NOW}}),
                    now=NOW, pid_alive=_never)
    assert fenced.state == "live"


@pytest.mark.parametrize("loop_state, busy", [("running", True), ("stopping", True),
                                               ("stopped", False), ("latched", False),
                                               ("idle", False)])
def test_an_active_run_loop_is_busy(loop_state, busy):
    # review S5: an agent must not launch between the TOF loop's runs
    status = {"state": 0, "run_pending": None,
              "run_loops": {"auto_tof": {"title": "BEC TOF loop", "state": loop_state}}}
    st = assess(FakeLiveOD(_poll()), FakeMonitor(status), now=NOW, pid_alive=_never)
    assert (st.state == "live" and not st.waivable) is busy
    if busy:
        assert "BEC TOF loop active" in st.reason and "stop it first" in st.reason
    assert run_gate.active_loops(status) == (["BEC TOF loop"] if busy else [])


def test_the_active_loop_states_are_the_run_loops():
    from waxx.util.device_state.run_loop import ACTIVE
    assert tuple(run_gate.LOOP_ACTIVE_STATES) == tuple(ACTIVE)


def test_assess_failures_are_unknown():
    st = assess(FakeLiveOD(error=TimeoutError("no reply")), FakeMonitor({}), now=NOW)
    assert st.state == "unknown" and "no reply" in st.reason
    st = assess(FakeLiveOD(_poll()), FakeMonitor(None), now=NOW)
    assert st.state == "unknown" and "monitor server" in st.reason
    st = assess(FakeLiveOD({"ok": False}), FakeMonitor({}), now=NOW)
    assert st.state == "unknown"


def test_gate_state_round_trips_to_a_dict():
    d = GateState("free", 1, "x").to_dict()
    assert d == {"state": "free", "run_id": 1, "reason": "x", "waivable": False, "detail": {}}


# -- RUN_EXITED on a run's behalf ----------------------------------------------------------

def test_run_exited_is_sent_for_the_current_run():
    client = FakeLiveOD(_running())
    out = tell_live_od_run_exited(client, 85528, "launcher: child exited (code 1)",
                                  pid_alive=_dead)
    assert out["sent"] and out["ok"]
    assert client.sent == [{"tag": "RUN_EXITED", "run_id": 85528,
                            "reason": "launcher: child exited (code 1)"}]


def test_run_exited_is_sent_when_the_pid_cannot_be_checked():
    # an older client (no pid), or one on another host: the launcher's word counts
    for poll in (_running(client_pid=None), _running(client_host="other-pc")):
        client = FakeLiveOD(poll)
        assert tell_live_od_run_exited(client, 85528, "x", pid_alive=_never)["sent"]


@pytest.mark.parametrize("poll, why", [
    (_running(run_id=85529), "current run is 85529"),
    (_running(run_state="exited"), "already knows"),
    (_poll(run_id=85528), "no run in progress"),
    (_running(run_state="saving"), "saving"),
])
def test_run_exited_is_not_sent_otherwise(poll, why):
    client = FakeLiveOD(poll)
    out = tell_live_od_run_exited(client, 85528, "x", pid_alive=_dead)
    assert not out["sent"] and why in out["why"] and client.sent == []


@pytest.mark.parametrize("poll", [_running(client_pid=None),
                                  _running(client_host="other-pc")])
def test_require_known_dead_refuses_an_unknown_pid(poll):
    # run_lock after Ctrl-C killed only its shell: the experiment may live on
    client = FakeLiveOD(poll)
    out = tell_live_od_run_exited(client, 85528, "x", pid_alive=_never,
                                  require_known_dead=True)
    assert not out["sent"] and "not known to be gone" in out["why"] and client.sent == []


def test_require_known_dead_sends_for_a_dead_pid():
    client = FakeLiveOD(_running())
    assert tell_live_od_run_exited(client, 85528, "x", pid_alive=_dead,
                                   require_known_dead=True)["sent"]


@pytest.mark.parametrize("poll", [
    _running(save_in_progress=True, reset_requested=True, run_state="aborting"),
    _running(save_in_progress=True, run_state="running"),
    # an older liveOD that may be saving: never during these states
    _running(reset_requested=True, run_state="aborting"),
    _running(reset_requested=True, run_state="no_reply"),
    _running(run_state="saving"),
])
def test_run_exited_is_never_sent_during_a_save(poll):
    client = FakeLiveOD(poll)
    out = tell_live_od_run_exited(client, 85528, "x", pid_alive=_dead)
    assert not out["sent"] and "sav" in out["why"] and client.sent == []


def test_run_exited_for_an_abort_when_liveod_says_no_save_runs():
    client = FakeLiveOD(_running(save_in_progress=False, reset_requested=True,
                                 run_state="aborting"))
    assert tell_live_od_run_exited(client, 85528, "x", pid_alive=_dead)["sent"]


def test_run_exited_is_not_sent_while_the_runs_process_lives():
    # the launcher saw its shell exit; liveOD's client pid on this host is alive
    client = FakeLiveOD(_running())
    out = tell_live_od_run_exited(client, 85528, "x", pid_alive=_alive)
    assert not out["sent"] and "still alive" in out["why"] and client.sent == []


def test_a_failed_poll_is_not_read_as_no_run_in_progress():
    # review S7: {"ok": False} has no run_in_progress either
    out = tell_live_od_run_exited(FakeLiveOD(), 85528, "x",
                                  poll={"ok": False, "error": "busy"}, pid_alive=_dead)
    assert not out["sent"] and out["why"].startswith("POLL failed") and "busy" in out["why"]

    class NoReply:
        def poll(self):
            return None
    out = tell_live_od_run_exited(NoReply(), 85528, "x", pid_alive=_dead)
    assert not out["sent"] and out["why"].startswith("POLL failed")


def test_run_exited_without_a_run_id_sends_nothing():
    client = FakeLiveOD(_running())
    assert not tell_live_od_run_exited(client, None, "x")["sent"]
    assert client.sent == []


def test_run_exited_through_a_sender_of_its_own():
    got = []
    out = tell_live_od_run_exited(None, 85528, "x", poll=_running(), pid_alive=_dead,
                                  send=lambda rid, why: got.append((rid, why)) or {"ok": True})
    assert out["sent"] and out["ok"] and got == [(85528, "x")]


def test_run_exited_uses_a_poll_in_hand():
    client = FakeLiveOD(error=AssertionError("must not poll"))
    out = tell_live_od_run_exited(client, 85528, "x", poll=_running(), pid_alive=_dead)
    assert out["sent"]


# -- the real pid check ----------------------------------------------------------------------

def test_pid_alive_on_this_process():
    assert run_gate.pid_alive(os.getpid())


@pytest.mark.skipif(sys.platform != "win32", reason="relies on a held process handle")
def test_pid_alive_on_an_exited_child_whose_handle_we_hold():
    # Windows does not reuse a pid while a handle to the process is open, and
    # Popen keeps its handle until the object goes: the pid cannot have been
    # given to another process when it is checked (review N5).
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert child.returncode == 0
    assert run_gate.pid_alive(child.pid) is False      # exited: exit code, not STILL_ACTIVE
    assert child._handle                               # the handle was held throughout


def test_pid_alive_is_conservative_for_nonsense():
    assert run_gate.pid_alive(0) is True


# -- the monitor server's run queue (review B1) ------------------------------------------

def test_the_queues_job_in_its_slot_or_about_to_launch_is_busy():
    base = {"state": 2, "run_pending": None, "run_loops": {},
            "person_hold": {"active": False}}
    assert run_gate.loops_verdict(dict(base, run_queue={"current": None, "next": []})) is None
    assert run_gate.loops_verdict(base) is None                     # an older server
    v = run_gate.loops_verdict(dict(base, run_queue={
        "current": {"id": 12, "label": "rabi", "state": "running", "run_id": 85600},
        "next": [13]}))
    assert v.state == "live" and v.blocks and v.run_id == 85600
    assert "job 12 (rabi) running (run 85600)" in v.reason
    v = run_gate.loops_verdict(dict(base, run_queue={"current": None, "next": [13]}))
    assert v.state == "live" and "job 13 ready to start" in v.reason
    # the hold is said first
    held = dict(base, person_hold={"active": True, "since": 1.0, "by": "jp", "reason": "x"},
                run_queue={"current": None, "next": [13]})
    assert run_gate.loops_verdict(held).state == "held"
