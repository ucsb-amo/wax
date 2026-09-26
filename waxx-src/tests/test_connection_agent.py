"""The connection agent (waxx.util.device_state.connection_agent), as real
child processes running a stand-in driver (fake_connection_driver.py): no
hardware.  Every test kills its agent in a finally, so none outlives it."""
import os
import threading
import time

import psutil
import pytest

from waxx.util.device_state.connection_agent import (
    AgentClient, AgentCommandError, AgentDied)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVER = "fake_connection_driver:FakeDriver"


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = TESTS_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _client(lines, **kwargs):
    return AgentClient(DRIVER, kwargs, name="test agent", log=lines.append, env=_env())


def _wait(pred, timeout=5.):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_round_trip_commands_errors_and_exit():
    lines = []
    agent = _client(lines)
    try:
        agent.start(timeout=20.)
        assert agent.state == {"open": False, "detail": ""}
        agent.call("open", timeout=5.)
        assert agent.state == {"open": True, "detail": "fake device · ok"}
        assert agent.call("echo", timeout=5., value=[1, 2]) == {"echo": [1, 2]}
        with pytest.raises(AgentCommandError, match="boom") as e:
            agent.call("boom", timeout=5.)
        assert e.value.state["open"] is True
        with pytest.raises(AgentCommandError, match="unknown command"):
            agent.call("format_disk", timeout=5.)
        assert agent.exit(timeout=5.)
        assert not agent.alive
        # prints went to the log, not into the protocol; exit closed the device
        assert "[test agent] fake: built" in lines
        assert "[test agent] fake: closed" in lines
    finally:
        agent.kill()


def test_the_agent_is_outside_the_callers_process_tree():
    lines = []
    agent = _client(lines)
    try:
        agent.start(timeout=20.)
        launcher = agent._proc
        assert _wait(lambda: launcher.poll() is not None)          # the launcher has gone
        assert agent.pid and agent.pid != launcher.pid
        me = psutil.Process()
        assert agent.pid not in [p.pid for p in me.children(recursive=True)]
        assert psutil.Process(agent.pid).ppid() != me.pid
    finally:
        agent.kill()


def test_when_the_server_goes_away_the_agent_closes_the_device_and_exits():
    lines = []
    agent = _client(lines)
    try:
        agent.start(timeout=20.)
        agent.call("open", timeout=5.)
        pid = agent.pid
        agent._proc.stdin.close()                     # as when the server process dies
        assert _wait(lambda: not agent.alive)
        assert _wait(lambda: not psutil.pid_exists(pid))
        assert any("went away" in line for line in lines)
        assert "[test agent] fake: closed" in lines
    finally:
        agent.kill()


def test_kill_ends_a_call_in_flight_at_once():
    lines = []
    agent = _client(lines, open_block_s=30.)
    outcome = {}

    def opener():
        try:
            agent.call("open", timeout=60.)
            outcome["result"] = "opened"
        except AgentDied as e:
            outcome["result"] = e

    try:
        agent.start(timeout=20.)
        pid = agent.pid
        t = threading.Thread(target=opener)
        t.start()
        time.sleep(0.3)
        t0 = time.monotonic()
        agent.kill()
        t.join(5.)
        assert isinstance(outcome.get("result"), AgentDied)
        assert time.monotonic() - t0 < 2.
        assert _wait(lambda: not psutil.pid_exists(pid))
    finally:
        agent.kill()


def test_the_service_drives_a_real_agent_end_to_end(monkeypatch):
    """ConnectionService with its default agent factory: a real agent process
    opens, is released synchronously for a run, and is gone afterwards."""
    from waxx.util.device_state import connections as conns
    monkeypatch.setenv("PYTHONPATH", _env()["PYTHONPATH"])
    lines = []
    svc = conns.ConnectionService(
        [conns.Connection(key="dev", label="Device", driver=DRIVER)], log=lines.append)
    svc.start()
    try:
        svc.run_over("the monitor is running", reopen=True)
        assert _wait(lambda: svc.snapshot()["dev"]["state"] == conns.CONNECTED, 20.)
        assert svc.snapshot()["dev"]["detail"] == "fake device · ok"
        assert svc.call("dev", "echo", {"value": 3}) == {"echo": 3}
        pid = svc._items["dev"].agent.pid
        svc.run_starting("run 1 (test)")
        assert svc.snapshot()["dev"]["state"] == conns.DISCONNECTED
        assert _wait(lambda: not psutil.pid_exists(pid))
        assert any("fake: closed" in line for line in lines)
    finally:
        svc.stop()


def test_a_driver_that_cannot_be_built_is_reported():
    lines = []
    agent = _client(lines, fail_build=True)
    try:
        with pytest.raises(AgentCommandError, match="no such device"):
            agent.start(timeout=20.)
        assert _wait(lambda: not agent.alive)
    finally:
        agent.kill()
