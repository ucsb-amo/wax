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


# -- dead client / reset pending -------------------------------------------------------

def test_dead_client_is_waivable_only_when_the_pid_is_dead_on_this_host():
    st = classify(_running(), None, now=NOW, pid_alive=_dead)
    assert st.state == "dead_client" and st.waivable and not st.blocks
    assert "pid 4242" in st.reason and "run_lock" in st.reason
    assert st.detail["client"]["alive"] is False


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


def test_a_fence_older_than_its_ttl_is_not_counted():
    fence = {"run_id": 85600, "expt": "rabi", "since": NOW - run_gate.FENCE_TTL_S - 1}
    st = classify(_poll(), fence, now=NOW, pid_alive=_never)
    assert st.state == "free" and "not counted" in st.reason
    assert st.detail["fence"]["active"] is False


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
    out = tell_live_od_run_exited(client, 85528, "launcher: child exited (code 1)")
    assert out["sent"] and out["ok"]
    assert client.sent == [{"tag": "RUN_EXITED", "run_id": 85528,
                            "reason": "launcher: child exited (code 1)"}]


@pytest.mark.parametrize("poll, why", [
    (_running(run_id=85529), "current run is 85529"),
    (_running(run_state="exited"), "already knows"),
    (_poll(run_id=85528), "no run in progress"),
])
def test_run_exited_is_not_sent_otherwise(poll, why):
    client = FakeLiveOD(poll)
    out = tell_live_od_run_exited(client, 85528, "x")
    assert not out["sent"] and why in out["why"] and client.sent == []


def test_run_exited_without_a_run_id_sends_nothing():
    client = FakeLiveOD(_running())
    assert not tell_live_od_run_exited(client, None, "x")["sent"]
    assert client.sent == []


def test_run_exited_uses_a_poll_in_hand():
    client = FakeLiveOD(error=AssertionError("must not poll"))
    out = tell_live_od_run_exited(client, 85528, "x", poll=_running())
    assert out["sent"]


# -- the real pid check ----------------------------------------------------------------------

def test_pid_alive_on_this_process_and_on_an_exited_child():
    assert run_gate.pid_alive(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    pid = child.pid
    del child                                          # drop our handle to it
    assert run_gate.pid_alive(pid) is False


def test_pid_alive_is_conservative_for_nonsense():
    assert run_gate.pid_alive(0) is True
