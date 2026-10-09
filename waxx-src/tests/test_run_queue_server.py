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
    assert rq["directory"] == str(tmp_path / "logs" / "run_queue")
    assert status["person_hold"] == rq["person_hold"]
    assert any(p.get("type") == "run_queue" for p in server._broadcaster.sent)
    listed = server.ask({"type": "run_queue", "action": "list"})
    assert [x["id"] for x in listed["jobs"]] == [1] and listed["next"] == [1]
    described = server.ask({"type": "run_queue", "action": "describe", "id": 1})
    assert described["job"]["id"] == 1 and described["waiting"] == "launching"
    assert server.ask({"type": "run_queue", "action": "pause", "scope": "agent",
                       "by": "jp"})["run_queue"]["paused"]["agent"]["by"] == "jp"
    assert server.ask({"type": "run_queue", "action": "resume", "scope": "agent",
                       "by": "jp"})["status"] == "ok"
    assert server.ask({"type": "run_queue", "action": "cancel", "id": 1,
                       "by": "jp"})["job"]["state"] == "cancelled"
    held = server.ask({"type": "run_queue", "action": "hold", "reason": "mine", "by": "jp"})
    assert held["person_hold"]["active"]
    assert server.ask({"type": "run_queue", "action": "release", "by": "jp"})["status"] == "ok"
    bad = server.ask({"type": "run_queue", "action": "explode"})
    assert bad["status"] == "error" and "known: submit" in bad["msg"]
    kinds = [e["kind"] for e in server.journal.tail(100)]
    for kind in ("run_queue_submit", "run_queue_pause", "run_queue_resume", "run_queue_end",
                 "run_queue_hold", "run_queue_release"):
        assert kind in kinds
    lines = (tmp_path / "logs" / "run_queue" / "journal.jsonl").read_text().splitlines()
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
