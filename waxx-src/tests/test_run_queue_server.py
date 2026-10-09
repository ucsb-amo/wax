"""The run queue in the monitor server: the ``run_queue`` requests (the
contract the kq / ar client will use), status_json, broadcasts, the watch
tick, and the queue's interplay with the server's loops, state reset and
monitor.  The server is built offscreen with its broadcaster faked; liveOD,
the job processes and the loops' runs are fakes; files live under
tmp_path.  Nothing is launched and nothing goes on the network."""
import json
import os
import threading

import pytest

from test_run_queue import Live, Spawner


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
def expts(tmp_path):
    folder = tmp_path / "experiments"
    folder.mkdir()
    for name in ("rabi", "auto_tof", "monitor"):
        (folder / f"{name}.py").write_text(f'"""{name}."""\nclass {name}: pass\n')
    return folder


@pytest.fixture
def server(qapp, monkeypatch, tmp_path, expts):
    from waxx.util.device_state.run_loop import LoopSpec
    from waxx.util.guis import monitor_server_gui as msg
    monkeypatch.setattr(msg, "StateBroadcaster", Broadcasts)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    s = msg.MonitorUDPServer(config_file_path=str(tmp_path / "state.json"),
                             journal_dir=str(tmp_path / "logs" / "ops_journal"),
                             run_loops=[LoopSpec("auto_tof", "BEC TOF loop",
                                                 str(expts / "auto_tof.py"))])
    s.status.expt_path = str(expts / "monitor.py")
    s.live = Live()
    s._live_od = s.live                               # the queue calls it late
    s.spawner = Spawner()
    s.run_queue._spawn = s.spawner
    s.run_queue.poll_every_s, s.run_queue._gap_s, s.run_queue._outcome_wait_s = 0., 0., 0.
    loop = s.loops["auto_tof"]
    loop._poll = s.live
    loop._gap_s, loop._poll_s = 0., 0.01
    s.monitor_starts = []
    s.start_monitor_signal.connect(s.monitor_starts.append)
    s.ask = lambda obj: json.loads(s.generate_reply(json.dumps(obj)))
    yield s
    loop.join(5)
    s.sock.close()


def _run(server, run_id, outcome="saved", code=0, lines=()):
    proc = server.spawner.procs[-1]
    proc.write(f"Run ID: {run_id}", *lines)
    server.live.start_run(run_id)
    server.watch_tick()
    server.live.end_run(run_id, outcome)
    proc.code = code
    server.watch_tick()


def test_the_request_contract(server, expts, tmp_path):
    path = str(expts / "rabi.py")
    reply = server.ask({"type": "run_queue", "action": "submit", "path": path, "by": "jp",
                        "label": "rabi", "priority": 2, "owner": "agent"})
    assert reply["status"] == "ok" and reply["ids"] == [1]
    j = reply["jobs"][0]
    assert {"id", "token", "path", "sha256", "label", "argv", "cwd", "owner", "priority",
            "due", "after", "chain", "stop_on_failure", "write_back", "allow_drift",
            "repeat_index", "repeat_of", "submitted_at", "submitted_by", "state", "reason",
            "pid", "pid_started", "client_pid", "run_id", "log_path", "exit_code",
            "outcome", "launched_at", "ended_at", "cancel", "adopted"} == set(j)
    assert j["state"] == "queued" and j["owner"] == "agent"
    status = json.loads(server.generate_reply("status_json"))
    rq = status["run_queue"]
    assert rq["enabled"] and rq["counts"]["queued"] == 1 and rq["next"] == [1]
    assert rq["directory"] == str(tmp_path / "run_queue_default")   # conftest's local default
    assert status["person_hold"] == rq["person_hold"]
    assert any(p.get("type") == "run_queue" for p in server._broadcaster.sent)
    listed = server.ask({"type": "run_queue", "action": "list"})
    assert [x["id"] for x in listed["jobs"]] == [1] and listed["next"] == [1]
    described = server.ask({"type": "run_queue", "action": "describe", "id": 1})
    assert described["job"]["id"] == 1 and described["waiting"] == "launching"
    assert server.ask({"type": "run_queue", "action": "pause", "scope": "agent",
                       "by": "jp"})["run_queue"]["paused"]["agent"]["by"] == "jp"
    assert server.ask({"type": "run_queue", "action": "resume", "scope": "agent",
                       "by": "jp", "owner": "person"})["status"] == "ok"
    no_owner = server.ask({"type": "run_queue", "action": "cancel", "id": 1, "by": "jp"})
    assert no_owner["status"] == "error" and "owner is required" in no_owner["msg"]
    assert server.ask({"type": "run_queue", "action": "cancel", "id": 1, "by": "jp",
                       "owner": "person"})["job"]["state"] == "cancelled"
    held = server.ask({"type": "run_queue", "action": "hold", "reason": "mine", "by": "jp"})
    assert held["person_hold"]["active"] and held["person_hold"]["owner"] == "person"
    assert server.ask({"type": "run_queue", "action": "release", "by": "jp",
                       "owner": "person"})["status"] == "ok"
    bad = server.ask({"type": "run_queue", "action": "explode"})
    assert bad["status"] == "error" and "known: submit" in bad["msg"]
    kinds = [e["kind"] for e in server.journal.tail(100)]
    for kind in ("run_queue_submit", "run_queue_pause", "run_queue_resume", "run_queue_end",
                 "run_queue_hold", "run_queue_release"):
        assert kind in kinds
    lines = (tmp_path / "run_queue_default" / "journal.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["kind"] == "run_queue_submit"


def test_the_monitors_own_experiment_is_refused(server, expts):
    reply = server.ask({"type": "run_queue", "action": "submit",
                        "path": str(expts / "monitor.py")})
    assert reply["status"] == "error" and "monitor's own experiment" in reply["msg"]


def test_the_watch_runs_the_job_and_asks_for_the_monitor_when_out_of_jobs(server, expts, qapp):
    from PyQt6.QtWidgets import QApplication
    server.ask({"type": "run_queue", "action": "submit", "path": str(expts / "rabi.py")})
    server.watch_tick()
    j = server.ask({"type": "run_queue", "action": "describe", "id": 1})["job"]
    assert j["state"] == "running" and server.spawner.calls[0]["env"]["WAXX_LAUNCHER"] == "kq"
    _run(server, 85600)
    server.watch_tick()
    QApplication.processEvents()
    j = server.ask({"type": "run_queue", "action": "describe", "id": 1})["job"]
    assert j["state"] == "saved" and j["run_id"] == 85600
    assert server.monitor_starts == ["the run queue has no job to run"]


def test_the_queue_waits_for_a_state_reset_and_a_starting_monitor(server, expts):
    from waxx.util.comms_server.comm_server import STATES
    server.ask({"type": "run_queue", "action": "submit", "path": str(expts / "rabi.py")})
    server.status.state = STATES.LOADING
    server.watch_tick()
    assert server.spawner.calls == []
    assert server.run_queue.info()["waiting"] == "the monitor is starting"
    server.status.state = STATES.READY
    server.watch_tick()
    assert len(server.spawner.calls) == 1


def test_a_reset_on_a_persons_run_holds_but_not_on_an_agents_queued_run(server, expts):
    server.ask({"type": "run_queue", "action": "submit", "path": str(expts / "rabi.py"),
                "owner": "agent"})
    server.watch_tick()
    server.spawner.procs[-1].write("Run ID: 85600")
    server.live.start_run(85600)
    server.watch_tick()
    server.live.state["reset_requested"] = True       # the agent resets its own run
    server.watch_tick()
    assert not server.person_hold.active
    server.live.end_run(85600, "discarded")
    server.spawner.procs[-1].code = 1
    server.watch_tick()
    server.live.state.update(run_in_progress=True, run_id=85601, reset_requested=False)
    server.watch_tick()
    server.live.state["reset_requested"] = True       # a person's run
    server.watch_tick()
    assert server.person_hold.active and server.person_hold.info()["run_id"] == 85601
    status = json.loads(server.generate_reply("status_json"))
    assert status["person_hold"]["reason"].startswith("Reset in liveOD at ")


# --- the queue and the server's real run loop ------------------------------------------------

class LoopProc:
    """One run of the loop's experiment: prints its run id, waits for
    ``release``, then liveOD records it saved; exit code 0."""

    def __init__(self, live, run_id, release: threading.Event):
        self.pid, self._live, self._run_id, self._release = 4242, live, run_id, release
        self.stdout = self._out()

    def _out(self):
        self._live.start_run(self._run_id)
        yield f"Run ID: {self._run_id}\n"
        self._release.wait(5)
        self._live.end_run(self._run_id)

    def wait(self):
        return 0


def _wait_for(predicate, timeout=5.0):
    import time
    t0 = time.monotonic()
    while not predicate() and time.monotonic() - t0 < timeout:
        time.sleep(0.01)
    return predicate()


def test_a_job_stops_the_loop_gracefully_and_the_loop_comes_back(server, expts, qapp):
    from PyQt6.QtWidgets import QApplication
    loop = server.loops["auto_tof"]
    release = threading.Event()
    run_ids = iter([81001, 81002])
    loop._spawn = lambda command, extra_env=None: LoopProc(server.live, next(run_ids), release)
    assert server.ask({"type": "run_loop", "action": "start", "loop": "auto_tof"})["status"] == "ok"
    assert _wait_for(lambda: loop.info().get("run_id") == 81001)
    server.ask({"type": "run_queue", "action": "submit", "path": str(expts / "rabi.py")})
    server.watch_tick()
    assert loop.info()["state"] == "stopping"            # its run in progress finishes
    assert server.spawner.calls == []
    assert "BEC TOF loop is finishing its run" in server.run_queue.info()["waiting"]
    assert server.run_queue.info()["resume_loop"]["key"] == "auto_tof"
    release.set()
    loop.join(5)
    QApplication.processEvents()
    assert loop.info()["state"] == "stopped" and loop.info()["runs"] == 1
    assert server.monitor_starts == []                   # no monitor between loop and job
    server.watch_tick()
    assert len(server.spawner.calls) == 1
    # a person's Start meanwhile is refused, naming the queue's work
    refused = server.ask({"type": "run_loop", "action": "start", "loop": "auto_tof"})
    assert refused["status"] == "error" and "the run queue has work (job 1" in refused["msg"]
    release.clear()                                      # the loop's next run will wait
    _run(server, 85600)                                  # its last tick starts the loop
    assert loop.info()["state"] == "running", loop.info()["text"]   # started again by the queue
    assert server.monitor_starts == []
    kinds = [e["kind"] for e in server.journal.tail(200)]
    assert "run_queue_loop_stop" in kinds and "run_queue_loop_resume" in kinds
    loop.stop()
    release.set()
    loop.join(5)


def test_a_persons_stop_request_cancels_the_queues_restart_and_starters_are_kept(
        server, expts, qapp):
    loop = server.loops["auto_tof"]
    release = threading.Event()
    loop._spawn = lambda command, extra_env=None: LoopProc(server.live, 81001, release)
    reply = server.ask({"type": "run_loop", "action": "start", "loop": "auto_tof",
                        "owner": "agent"})
    assert reply["status"] == "ok" and reply["loop"]["owner"] == "agent"
    assert _wait_for(lambda: loop.info().get("run_id") == 81001)
    server.ask({"type": "run_queue", "action": "submit", "path": str(expts / "rabi.py"),
                "owner": "agent"})
    server.watch_tick()
    assert loop.info()["state"] == "stopping"            # an agent's loop: an agent's job stops it
    assert server.run_queue.info()["resume_loop"]["key"] == "auto_tof"
    server.ask({"type": "run_loop", "action": "stop", "loop": "auto_tof", "operator": "jp"})
    assert server.run_queue.info()["resume_loop"] is None
    assert "run_queue_loop_resume_cancelled" in [e["kind"] for e in server.journal.tail(50)]
    release.set()
    loop.join(5)


def test_a_monitor_restart_waits_while_the_queue_has_work(server, expts, qapp):
    from PyQt6.QtWidgets import QApplication
    restarts = []
    server.reset_signal.connect(lambda: restarts.append("reset"))
    forwarded = []
    server.message_received.connect(forwarded.append)
    server.ask({"type": "run_queue", "action": "submit", "path": str(expts / "rabi.py")})
    server.on_message_received("reset")                 # eligible: about to launch
    server.watch_tick()
    server.on_message_received("reset")                 # in the slot
    server.on_message_received("run complete")
    QApplication.processEvents()
    assert restarts == [] and forwarded == []
    kinds = [e["kind"] for e in server.journal.tail(100)]
    assert kinds.count("run_queue_monitor_deferred") == 3
    _run(server, 85600)
    server.watch_tick()
    QApplication.processEvents()
    assert server.monitor_starts == ["the run queue has no job to run"]
    server.on_message_received("reset")                 # the queue is idle: as before
    QApplication.processEvents()
    assert restarts == ["reset"]


def test_a_state_reset_is_refused_while_a_queue_job_runs(server, expts):
    server.ask({"type": "run_queue", "action": "submit", "path": str(expts / "rabi.py")})
    server.watch_tick()
    reply = server.ask({"type": "reset_state", "operator": "jp"})
    assert reply["status"] == "error" and "the run queue's job 1 (rabi) is running" in reply["msg"]
    assert "run queue's job 1" in server._regenerate_blocker()
    from waxx.util.comms_server.comm_server import STATES
    server.status.state = STATES.READY                   # even with a monitor up
    assert "run queue's job 1" in server._slm_reinit_blocker()
