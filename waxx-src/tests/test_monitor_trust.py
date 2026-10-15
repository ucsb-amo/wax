"""The monitor server's trust flag around a run's announcement (queue phase
2, M0 trust fix), and the run queue's ``warm`` action through the server.
The server is built offscreen with its broadcaster faked (the fixture of
test_run_queue_server.py); nothing is launched and nothing goes on the
network."""
import json
import time

from test_run_queue_server import expts, qapp, server  # noqa: F401  (fixtures)
from waxx.util.device_state import run_queue as rq
from waxx.util.device_state.ready_bit import STATES


def ask(s, obj):
    return json.loads(s.generate_reply(json.dumps(obj)))


def trust_kinds(s):
    return [e for e in s.journal.tail(200) if e["kind"] == "trust"]


def test_an_announced_run_makes_the_state_untrusted_until_its_end_state(server):
    assert server._trust["trusted"] is True
    ask(server, {"type": "run_pending", "run_id": 87022, "expt": "rabi", "token": "t1"})
    t = server._trust
    assert t["trusted"] is False
    assert t["reason"].startswith("run 87022 (rabi) announced at ")
    assert t["reason"].endswith("; untrusted until its end state")
    assert server._broadcaster.sent[-1]["type"] in ("run_pending", "trust")
    assert any(p.get("type") == "trust" and p["trust"]["trusted"] is False
               for p in server._broadcaster.sent)
    cfg = json.load(open(server.config_file_path))
    assert cfg["metadata"]["state_trust"]["trusted"] is False   # survives a restart
    reply = ask(server, {"type": "replace_state", "run_id": 87022, "expt": "rabi",
                         "config": {"dds": {}, "ttl": {}, "dac": {}}})
    assert reply["status"] == "ok"
    assert server._trust["trusted"] is True and server._trust["reason"] == \
        "end state of run 87022 (rabi)"
    assert server._trust_before_run is None


def test_a_withdrawn_run_gets_the_previous_trust_back(server):
    ask(server, {"type": "run_pending", "run_id": 5, "expt": "x", "token": "tok"})
    assert server._trust["trusted"] is False
    ask(server, {"type": "run_withdrawn", "token": "wrong"})        # not its token: nothing
    assert server._trust["trusted"] is False
    assert ask(server, {"type": "run_withdrawn", "token": "tok",
                        "took_core": False})["status"] == "ok"
    t = server._trust
    assert t["trusted"] is True
    assert "run 5 (x): the run exited without taking the core; trust as before" in t["reason"]


def test_a_withdraw_after_taking_the_core_keeps_it_untrusted(server):
    """A run that took the core and exited without an accepted end state (a
    kernel that raised outside scan()'s handler, end() failing before
    replace_state) withdraws with took_core=True: never trusted again."""
    ask(server, {"type": "run_pending", "run_id": 5, "expt": "x", "token": "tok"})
    assert ask(server, {"type": "run_withdrawn", "token": "tok",
                        "took_core": True})["status"] == "ok"
    t = server._trust
    assert t["trusted"] is False
    assert t["reason"].startswith("run 5 (x) exited without reporting its end state after "
                                  "taking the core")
    assert server._run_pending is None and server._trust_before_run is None
    cleared = [e for e in server.journal.tail(50) if e["kind"] == "run_pending_cleared"]
    assert "after taking the core" in cleared[-1]["why"]
    assert not any("without taking the core" in e.get("why", "") for e in cleared)


def test_a_withdraw_that_does_not_say_keeps_it_untrusted(server):
    """No took_core (an older client, a core class that cannot tell): the
    server cannot know the core was not taken -- untrusted."""
    for took in ({}, {"took_core": None}):
        ask(server, {"type": "run_pending", "run_id": 5, "expt": "x", "token": "tok"})
        ask(server, dict({"type": "run_withdrawn", "token": "tok"}, **took))
        assert server._trust["trusted"] is False
        assert "whether it took the core is unknown" in server._trust["reason"]


def test_a_withdrawn_run_restores_an_untrusted_state_as_untrusted(server):
    server._set_trust(False, "run 4 took the core and never reported")
    ask(server, {"type": "run_pending", "run_id": 5, "expt": "x", "token": "tok"})
    ask(server, {"type": "run_withdrawn", "token": "tok", "took_core": False})
    assert server._trust["trusted"] is False
    assert "run 4 took the core and never reported" in server._trust["reason"]


def test_an_operators_clear_leaves_it_untrusted(server):
    ask(server, {"type": "run_pending", "run_id": 5, "expt": "x", "token": "tok"})
    ask(server, {"type": "clear_run_pending", "token": "tok", "operator": "ada"})
    assert server._run_pending is None and server._trust["trusted"] is False


def test_the_never_took_the_core_ttl_restores_the_trust(server):
    server.status.state = STATES.READY
    ask(server, {"type": "run_pending", "run_id": 6, "expt": "x", "token": "tok"})
    assert server._trust["trusted"] is False
    server._run_pending["t0"] = time.monotonic() - server.RUN_PENDING_TTL_S - 1
    assert server._current_run_pending() is None
    assert server._trust["trusted"] is True
    assert "never took the core" in server._trust["reason"]


def test_taking_the_core_keeps_it_untrusted_and_forgets_the_old_trust(server):
    ask(server, {"type": "run_pending", "run_id": 7, "expt": "x", "token": "tok"})
    server.on_monitor_state(STATES.NOT_READY, "interrupted_by_run")
    assert server._trust["trusted"] is False and "took the core" in server._trust["reason"]
    assert server._trust_before_run is None
    ask(server, {"type": "run_withdrawn", "token": "tok"})          # (fence already lifted)
    assert server._trust["trusted"] is False


def test_an_abort_state_and_a_trust_ack_end_the_memory(server):
    ask(server, {"type": "run_pending", "run_id": 8, "expt": "x", "token": "tok"})
    ask(server, {"type": "abort_state", "run_id": 8, "expt": "x", "cause": "RTIOUnderflow",
                 "trusted": False, "config": {"dds": {}, "ttl": {}, "dac": {}}})
    assert server._trust["trusted"] is False and server._trust_before_run is None
    ask(server, {"type": "run_pending", "run_id": 9, "expt": "x", "token": "t9"})
    ask(server, {"type": "trust_ack", "operator": "ada"})
    assert server._trust["trusted"] is True and server._trust_before_run is None
    ask(server, {"type": "run_withdrawn", "token": "t9", "took_core": False})
    assert "acknowledged" in server._trust["reason"]               # not overwritten


def test_back_to_back_queue_runs_are_untrusted_while_they_run(server, expts):  # noqa: F811
    """The 2026-10-13 gap: a queue job runs right after another's end state,
    with the monitor never restarted -- it must not run labelled trusted."""
    ask(server, {"type": "replace_state", "run_id": 87021, "expt": "a",
                 "config": {"dds": {}, "ttl": {}, "dac": {}}})
    assert server._trust["trusted"] is True
    server.status.state = STATES.NOT_READY                         # no monitor between runs
    ask(server, {"type": "run_pending", "run_id": 87022, "expt": "b", "token": "t"})
    assert server._trust["trusted"] is False
    assert [e["trusted"] for e in trust_kinds(server)][-1] is False


# --- the warm action -----------------------------------------------------------------------

def test_the_warm_action_reaches_the_queue_and_is_answered_while_exiting(server, expts):  # noqa: F811
    reply = ask(server, {"type": "run_queue", "action": "warm", "stage": "poll",
                         "job": "1:abc", "warm": "x", "pid": 1, "started": 1.0})
    assert reply["status"] == "ok" and "not a job of this run queue" in reply["abort"]
    assert ask(server, {"type": "run_queue", "action": "warm", "stage": "?"})["status"] == "error"
    assert not server._reads_only("run_queue", {"action": "warm"})   # under the request lock
    server._exiting = "server is restarting"
    try:
        assert not server._starts_work("run_queue", {"action": "warm"})
        assert server._starts_work("run_queue", {"action": "submit"})
        reply = ask(server, {"type": "run_queue", "action": "warm", "stage": "poll",
                             "job": "1:abc", "warm": "x", "pid": 1, "started": 1.0})
        assert reply["status"] == "ok"
    finally:
        server._exiting = ""


def test_the_warm_request_never_calls_live_od(server, expts):  # noqa: F811
    def boom(*a, **kw):
        raise AssertionError("liveOD called from the warm request")
    server.run_queue._poll_fn = boom
    sub = ask(server, {"type": "run_queue", "action": "submit",
                       "path": str(expts / "rabi.py"), "owner": "agent"})
    jid, token = sub["ids"][0], sub["jobs"][0]["token"]
    job = server.run_queue._jobs[jid]
    job.warm = {"token": "w" * 16, "state": "warming", "attempt": 1, "spawned_at": time.time(),
                "spec": {}, "go": None}
    reply = ask(server, {"type": "run_queue", "action": "warm", "stage": "ready",
                         "job": f"{jid}:{token}", "warm": "w" * 16, "pid": 4242,
                         "started": 2.0, "manifest": [["a.py", 1, 2]]})
    assert reply["status"] == "ok" and reply["go"] is False
    assert job.warm["state"] == "ready" and job.warm["manifest"] == [["a.py", 1, 2]]
    assert isinstance(rq.WARM_LEAD_S, float)
