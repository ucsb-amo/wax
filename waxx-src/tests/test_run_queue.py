"""The run queue (waxx.util.device_state.run_queue): scheduling, launch,
outcome, cancel, pause / hold, persistence and re-adoption, the alarm, and
the loop interplay.  Pure Python: liveOD is a dict-backed fake, the job
processes are fakes that write to their log file under tmp_path, the clock is
injected.  Nothing is launched and nothing goes on the network (except the
detached-launcher test at the end, which runs `echo` through cmd)."""
import json
import os
import sys
import time

import pytest

from waxx.util.device_state import run_queue as rq
from waxx.util.device_state.person_hold import PersonHold
from waxx.util.device_state.run_queue import RunQueue


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class Live:
    """liveOD as POLL shows it, plus the two messages the queue may send."""

    def __init__(self):
        self.state = {"ok": True, "run_in_progress": False, "run_id": None,
                      "reset_requested": False, "last_outcome": {}}
        self.resets, self.exited, self.polls = [], [], 0
        self.down = False

    def __call__(self):
        if self.down:
            raise ConnectionError("no liveOD")
        self.polls += 1
        return json.loads(json.dumps(self.state))

    def run_exited(self, run_id, reason):
        self.exited.append((run_id, reason))
        self.state.update(run_in_progress=False, run_state="exited")
        return {"ok": True}

    def reset(self):
        self.resets.append(self.state["run_id"])
        self.state["reset_requested"] = True
        return {"ok": True}

    def start_run(self, run_id, **extra):
        self.state.update(dict(dict(run_in_progress=True, run_id=run_id,
                                    reset_requested=False, run_state="running",
                                    init_run_age_s=1.0), **extra))

    def end_run(self, run_id, outcome="saved", detail=""):
        self.state.update(run_in_progress=False, run_state="idle",
                          last_outcome={"run_id": run_id, "outcome": outcome, "detail": detail})


class Proc:
    def __init__(self, log_path, pid):
        self.pid, self.started, self.code, self.log_path = pid, 1234.5, None, log_path

    def write(self, *lines):
        with open(self.log_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def poll(self):
        return self.code


class Spawner:
    def __init__(self):
        self.procs, self.calls = [], []
        self.fail = None

    def __call__(self, command, cwd, env, log_path):
        if self.fail:
            raise OSError(self.fail)
        self.calls.append({"command": command, "cwd": cwd, "env": env, "log_path": log_path})
        proc = Proc(log_path, 5000 + len(self.procs))
        self.procs.append(proc)
        return proc


class Journal:
    def __init__(self):
        self.entries = []

    def record(self, kind, **fields):
        self.entries.append(dict(fields, kind=kind))

    @property
    def kinds(self):
        return [e["kind"] for e in self.entries]


class FakeLoop:
    """What the queue uses of a RunLoop."""

    class Spec:
        def __init__(self, key, title, pick=False):
            self.key, self.title, self.pick = key, title, pick

    def __init__(self, key="auto_tof", title="BEC TOF loop", pick=False, path="x.py"):
        self.spec = self.Spec(key, title, pick)
        self.path = path
        self.state, self.text = "running", "started by jp"
        self.stops, self.starts = [], []

    @property
    def active(self):
        return self.state in ("running", "stopping")

    def info(self):
        return {"state": self.state, "text": self.text}

    def stop(self, operator="", client="", start_monitor=True):
        self.stops.append((operator, start_monitor))
        self.state, self.text = "stopping", f"Stop pressed by {operator}@{client}"
        return {"status": "ok"}

    def finish_stop(self):
        self.state, self.text = "stopped", "stopped by run queue@kong after 3 saved runs"

    def start(self, operator="", client="", path=None):
        self.starts.append((operator, path))
        self.state = "running"
        return {"status": "ok"}


@pytest.fixture
def expts(tmp_path):
    folder = tmp_path / "experiments"
    folder.mkdir()
    for name in ("rabi", "tof", "cal", "monitor"):
        (folder / f"{name}.py").write_text(f'"""{name}."""\nclass {name}: pass\n')
    return folder


@pytest.fixture
def q(tmp_path, expts):
    return make_queue(tmp_path, expts)


def make_queue(tmp_path, expts, **kw):
    live, spawner, clock, journal, starts = Live(), Spawner(), Clock(), Journal(), []
    args = dict(poll=live, run_exited=live.run_exited, live_od_reset=live.reset,
                fence=lambda: None, monitor_state=lambda: 2, server_busy=lambda: "",
                loops={}, start_monitor=starts.append, journal=journal, spawn=spawner,
                adopt=lambda pid, started: None, clock=clock, poll_every_s=0.0, gap_s=0.0,
                outcome_wait_s=0.0, exclude=(str(expts / "monitor.py"),), client_name="kong")
    args.update(kw)
    queue = RunQueue(str(tmp_path / "logs" / "run_queue"), **args)
    queue.live, queue.spawner, queue.clock = live, spawner, clock
    queue.journal, queue.monitor_starts = journal, starts
    return queue


def submit(q, expts, name="rabi", **kw):
    reply = q.submit(dict({"path": str(expts / f"{name}.py"), "by": "test"}, **kw))
    assert reply["status"] == "ok", reply
    return reply["ids"] if len(reply["ids"]) > 1 else reply["ids"][0]


def job(q, job_id):
    return q.describe({"id": job_id})["job"]


def run_through(q, run_id, outcome="saved", code=0, lines=()):
    """The job in the slot prints its run id, runs, ends with ``outcome``."""
    proc = q.spawner.procs[-1]
    proc.write(f"Run ID: {run_id}", *lines)
    q.live.start_run(run_id, client_pid=7000 + run_id % 100, client_host="kong")
    q.tick()
    q.live.end_run(run_id, outcome)
    proc.code = code
    q.tick()


# --- scheduling ---------------------------------------------------------------------------

def test_order_is_priority_then_due_then_id(q, expts):
    a = submit(q, expts, "rabi")
    b = submit(q, expts, "tof", priority=5)
    c = submit(q, expts, "cal", priority=5, due=q.clock.t + 60)
    d = submit(q, expts, "cal", priority=5)
    assert q.list()["next"] == [b, d, a]                      # c is not due yet
    q.clock.t += 61
    assert q.list()["next"] == [b, d, c, a]
    q.tick()
    assert job(q, b)["state"] == "running" and len(q.spawner.calls) == 1
    run_through(q, 101)
    assert job(q, b)["state"] == "saved" and job(q, b)["run_id"] == 101
    q.tick()
    assert job(q, d)["state"] == "running"


def test_one_slot_and_the_next_waits_for_the_process_to_exit(q, expts):
    a, b = submit(q, expts), submit(q, expts, "tof")
    q.tick()
    q.tick()
    assert [job(q, a)["state"], job(q, b)["state"]] == ["running", "queued"]
    assert "job %d is in the slot" % a in q.describe({"id": b})["waiting"]
    run_through(q, 101)
    q.tick()
    assert job(q, b)["state"] == "running"


def test_after_waits_for_saved_and_skips_on_a_failure(q, expts):
    a = submit(q, expts)
    b = submit(q, expts, "tof", after=[a])
    q.tick()
    assert "waiting for job %d" % a in q.describe({"id": b})["waiting"]
    run_through(q, 101, outcome="saved_incomplete")
    q.tick()
    assert job(q, a)["state"] == "failed" and "saved_incomplete" in job(q, a)["reason"]
    assert job(q, b)["state"] == "skipped"
    assert job(q, b)["reason"].startswith("job %d ended failed" % a)
    assert len(q.spawner.calls) == 1


def test_a_chain_stops_on_a_failure(q, expts):
    ids = submit(q, expts, repeat=3)
    assert len(ids) == 3
    jobs = [job(q, i) for i in ids]
    assert {j["chain"] for j in jobs} == {f"repeat-{ids[0]}"}
    assert all(j["stop_on_failure"] for j in jobs)
    assert [j["repeat_index"] for j in jobs] == [1, 2, 3]
    other = submit(q, expts, "tof")                     # not in the chain
    q.tick()
    run_through(q, 101, code=1, outcome="", lines=["RuntimeError: boom"])
    assert job(q, ids[0])["state"] == "failed"
    for i in ids[1:]:
        assert job(q, i)["state"] == "cancelled"
        assert job(q, i)["reason"].startswith(f"chain repeat-{ids[0]} stopped: job {ids[0]}")
    q.tick()
    assert job(q, other)["state"] == "running"


def test_a_chain_can_go_on_after_a_failure_when_asked(q, expts):
    ids = submit(q, expts, chain="night", repeat=2, stop_on_failure=False)
    q.tick()
    run_through(q, 101, code=1, outcome="")
    q.tick()
    assert job(q, ids[1])["state"] == "running"


def test_a_changed_file_is_skipped_unless_drift_is_allowed(q, expts):
    a = submit(q, expts)
    b = submit(q, expts, "tof", allow_drift=True)
    (expts / "rabi.py").write_text("# edited\n")
    (expts / "tof.py").write_text("# edited too\n")
    q.tick()
    assert job(q, a)["state"] == "skipped" and job(q, a)["reason"] == "source changed since submit"
    assert job(q, b)["state"] == "running"
    with open(job(q, b)["log_path"], encoding="utf-8") as f:
        assert "drift allowed" in f.read()


def test_pause_scopes_and_the_person_hold(q, expts):
    agent = submit(q, expts, owner="agent", priority=9)
    person = submit(q, expts, "tof")
    q.pause({"scope": "all", "by": "jp", "reason": "realigning"})
    q.tick()
    assert q.spawner.calls == []
    assert "all jobs paused by jp" in q.describe({"id": person})["waiting"]
    q.resume({"scope": "all", "by": "jp"})
    q.pause({"scope": "agent", "by": "jp"})
    q.tick()
    assert job(q, person)["state"] == "running" and job(q, agent)["state"] == "queued"
    run_through(q, 101)
    q.resume({"scope": "agent", "by": "jp"})
    q.hold_request({"reason": "aligning", "by": "jp@kong"})
    q.tick()
    assert job(q, agent)["state"] == "queued"
    assert q.describe({"id": agent})["waiting"].startswith("person hold since")
    assert q.info()["state"] == "held"
    q.release_request({"by": "jp"})
    q.tick()
    assert job(q, agent)["state"] == "running"
    assert q.resume({"scope": "agent"})["status"] == "error"
    assert q.pause({"scope": "everyone"})["status"] == "error"


# --- launch -----------------------------------------------------------------------------

def test_the_launch_command_environment_and_log(q, expts, tmp_path):
    a = submit(q, expts, label="rabi scan!", argv=["-a", "x=1"], owner="agent",
               write_back=False)
    b = submit(q, expts, "tof")
    q.tick()
    call = q.spawner.calls[0]
    path = str((expts / "rabi.py").resolve())
    assert call["command"] == f'%kpy% & artiq_run --device-db "%db%" {path} -a x=1'
    env = call["env"]
    assert env["WAXX_LAUNCHER"] == "kq" and env["WAXX_QUEUE_JOB"] == str(a)
    assert env["WAXX_OWNER"] == "agent" and env["PYTHONUNBUFFERED"] == "1"
    assert env["WAXX_CAL_NO_WRITE_BACK"] == "1"
    assert call["cwd"] == str(expts.resolve())
    assert call["log_path"] == str(tmp_path / "logs" / "run_queue" / "logs" / f"{a}_rabi_scan.out")
    j = job(q, a)
    assert j["pid"] == 5000 and j["pid_started"] == 1234.5 and j["log_path"] == call["log_path"]
    run_through(q, 101)
    q.tick()
    assert "WAXX_CAL_NO_WRITE_BACK" not in q.spawner.calls[1]["env"]
    assert q.spawner.calls[1]["env"]["WAXX_OWNER"] == "person"
    assert job(q, a)["client_pid"] == 7001
    assert b


def test_a_path_with_spaces_is_quoted(q, tmp_path):
    folder = tmp_path / "M testing"
    folder.mkdir()
    (folder / "x.py").write_text("class x: pass\n")
    q.submit({"path": str(folder / "x.py")})
    q.tick()
    assert q.spawner.calls[0]["command"].endswith(f' "{(folder / "x.py").resolve()}"')


@pytest.mark.parametrize("obj, words", [
    ({"path": "rabi.py"}, "must be absolute"),
    ({"path": "C:/no/such/file.py"}, "no such file"),
    ({"path": "{expts}/rabi.py", "argv": ["a&b"]}, "not allowed"),
    ({"path": "{expts}/rabi.py", "write_back": True}, "may only veto"),
    ({"path": "{expts}/rabi.py", "owner": "robot"}, "owner must be"),
    ({"path": "{expts}/rabi.py", "after": [999]}, "unknown job"),
    ({"path": "{expts}/rabi.py", "repeat": 0}, "repeat must be"),
    ({"path": "{expts}/monitor.py"}, "monitor's own experiment"),
    ({"path": "{expts}/a&b.py"}, "not allowed"),
])
def test_submit_refusals(q, expts, obj, words):
    obj = {k: (v.format(expts=expts) if isinstance(v, str) else v) for k, v in obj.items()}
    reply = q.submit(obj)
    assert reply["status"] == "error" and words in reply["msg"], reply
    assert "run_queue_refused" in q.journal.kinds


def test_a_spawn_failure_fails_the_job(q, expts):
    a = submit(q, expts)
    q.spawner.fail = "no shell"
    q.tick()
    assert job(q, a)["state"] == "failed" and "could not start" in job(q, a)["reason"]


def test_no_folder_no_queue(tmp_path, expts):
    queue = RunQueue(None, poll=Live(), spawn=Spawner(), adopt=lambda p, s: None)
    reply = queue.submit({"path": str(expts / "rabi.py")})
    assert reply["status"] == "error" and "no run queue folder" in reply["msg"]


# --- the gate -----------------------------------------------------------------------------

def test_the_gate_waits_for_someone_elses_run_and_server_busy(q, expts):
    a = submit(q, expts)
    q.live.start_run(555, n_shots=1, last_shot_age_s=1.0, init_run_age_s=10.0)
    q.tick()
    assert q.spawner.calls == [] and "555" in q.describe({"id": a})["waiting"]
    q.live.end_run(555)
    q._server_busy = lambda: "a state reset is running"
    q.tick()
    assert q.spawner.calls == [] and q.info()["waiting"] == "a state reset is running"
    q._server_busy = lambda: ""
    q.live.down = True
    q.tick()
    assert "liveOD is not reachable" in q.info()["waiting"]
    q.live.down = False
    q.tick()
    assert job(q, a)["state"] == "running"


def test_a_foreign_fence_waits_but_the_queues_own_ended_runs_fence_does_not(q, expts):
    submit(q, expts)
    b = submit(q, expts, "tof")
    q.tick()
    run_through(q, 101)
    q._fence = lambda: {"run_id": 101, "expt": "rabi", "since": q.clock.t}
    q.tick()
    assert job(q, b)["state"] == "running"
    run_through(q, 102)
    c = submit(q, expts)
    q._fence = lambda: {"run_id": 900, "expt": "other", "since": q.clock.t}
    q.tick()
    assert job(q, c)["state"] == "queued" and "run 900" in q.info()["waiting"]


# --- outcomes -----------------------------------------------------------------------------

def test_outcomes(q, expts):
    ids = [submit(q, expts) for _ in range(3)]
    q.tick()
    run_through(q, 101)
    q.tick()
    run_through(q, 102, code=1, outcome="", lines=[
        "RuntimeError: Acquisition for run 102 aborted."])
    q.tick()
    run_through(q, 103, code=0, outcome="saved_incomplete")
    saved, aborted, incomplete = (job(q, i) for i in ids)
    assert saved["state"] == "saved" and saved["outcome"]["outcome"] == "saved"
    assert saved["exit_code"] == 0 and saved["reason"] == ""
    assert aborted["state"] == "failed" and aborted["outcome"]["aborted"]
    assert "aborted in liveOD" in aborted["reason"]
    assert incomplete["state"] == "failed" and "saved_incomplete" in incomplete["reason"]
    ends = [e for e in q.journal.entries if e["kind"] == "run_queue_end"]
    assert [e["state"] for e in ends] == ["saved", "failed", "failed"]


def test_a_job_killed_hard_is_reported_to_live_od(q, expts):
    a = submit(q, expts)
    q.tick()
    proc = q.spawner.procs[-1]
    proc.write("Run ID: 101")
    q.live.start_run(101)
    q.tick()
    proc.code = 3221225477                           # crashed: liveOD never heard
    q.tick()
    assert [rid for rid, _ in q.live.exited] == [101]
    assert "run queue" in q.live.exited[0][1]
    assert job(q, a)["state"] == "failed"
    assert job(q, a)["outcome"]["told_live_od"] == "sent"
    assert "run_queue_told_live_od_exited" in q.journal.kinds


# --- cancel -------------------------------------------------------------------------------

def test_cancel_a_queued_job(q, expts):
    a, b = submit(q, expts), submit(q, expts)
    assert q.cancel({"id": b, "by": "jp"})["job"]["state"] == "cancelled"
    assert q.cancel({"id": b})["status"] == "error"
    assert q.cancel({"id": 999})["status"] == "error"
    assert q.cancel({"id": a, "token": "nope"})["status"] == "error"
    q.tick()
    assert job(q, a)["state"] == "running" and len(q.spawner.calls) == 1


def test_cancel_a_running_job_sends_live_ods_abort_for_its_run_only(q, expts):
    a = submit(q, expts, owner="agent")
    q.tick()
    proc = q.spawner.procs[-1]
    polls = q.live.polls
    reply = q.cancel({"id": a, "by": "agent-7", "owner": "agent"})
    assert reply["status"] == "ok" and reply["pending"]
    assert q.live.resets == [] and q.live.polls == polls  # the request only records (S4)
    q.tick()
    assert q.live.resets == []                        # no run id yet: nothing to abort
    assert "no run id yet" in job(q, a)["cancel"]["abort_note"]
    proc.write("Run ID: 101")
    q.live.start_run(555)                             # liveOD's run is someone else's
    q.tick()
    assert q.live.resets == [] and "not 101" in job(q, a)["cancel"]["abort_note"]
    q.live.start_run(101)
    q.tick()
    assert q.live.resets == [101] and job(q, a)["cancel"]["abort_sent"]
    q.tick()
    assert q.live.resets == [101]                     # once
    assert job(q, a)["state"] == "running"            # never killed: it ends by itself
    proc.write("RuntimeError: Acquisition for run 101 aborted.")
    q.live.end_run(101, "discarded")
    proc.code = 1
    q.tick()
    j = job(q, a)
    assert j["state"] == "cancelled" and j["reason"].startswith("cancelled by agent-7 while running")
    assert 101 in q.own_abort_ids()
    assert "run_queue_abort_sent" in q.journal.kinds


def test_no_abort_while_live_od_saves_and_a_late_cancel_still_saves(q, expts):
    a = submit(q, expts)
    q.tick()
    proc = q.spawner.procs[-1]
    proc.write("Run ID: 101")
    q.live.start_run(101, save_in_progress=True)
    q.tick()
    q.cancel({"id": a, "by": "jp"})
    q.tick()
    assert q.live.resets == [] and "saving" in job(q, a)["cancel"]["abort_note"]
    q.live.end_run(101)
    proc.code = 0
    q.tick()
    assert job(q, a)["state"] == "saved" and "before the cancel" in job(q, a)["reason"]


def test_an_agent_may_not_abort_a_persons_running_job(q, expts):
    a = submit(q, expts)
    q.tick()
    reply = q.cancel({"id": a, "by": "agent-7", "owner": "agent"})
    assert reply["status"] == "error" and "person's run" in reply["msg"]
    assert job(q, a)["cancel"] is None


# --- persistence and restart ---------------------------------------------------------------

def test_the_queue_is_kept_and_read_back(tmp_path, expts, q):
    a = submit(q, expts, priority=3)
    q.pause({"scope": "agent", "by": "jp"})
    data = json.loads((tmp_path / "logs" / "run_queue" / "queue.json").read_text())
    assert data["version"] == 1 and data["jobs"][0]["id"] == a and data["paused"]["agent"]
    lines = (tmp_path / "logs" / "run_queue" / "journal.jsonl").read_text().splitlines()
    assert [json.loads(x)["kind"] for x in lines] == ["run_queue_submit", "run_queue_pause"]
    again = make_queue(tmp_path, expts)
    assert job(again, a)["priority"] == 3 and again.info()["paused"]["agent"]["by"] == "jp"
    b = submit(again, expts)
    assert b == a + 1                                  # ids keep counting


def test_a_running_job_is_adopted_after_a_restart(tmp_path, expts, q):
    a = submit(q, expts)
    q.tick()
    q.spawner.procs[-1].write("Run ID: 101", "shot 1/9")
    q.tick()
    adopted = Proc(job(q, a)["log_path"], 5000)
    seen = []

    def adopt(pid, started):
        seen.append((pid, started))
        return adopted
    again = make_queue(tmp_path, expts, adopt=adopt)
    assert seen == [(5000, 1234.5)]
    j = job(again, a)
    assert j["state"] == "running" and j["adopted"] and again.info()["current"]["id"] == a
    assert "run_queue_adopted" in again.journal.kinds
    again.tick()
    assert list(again.describe({"id": a})["tail"])[-1] == "shot 1/9"
    again.live.end_run(101)
    adopted.code = 0
    again.tick()
    assert job(again, a)["state"] == "saved"


@pytest.mark.parametrize("outcome, state, words", [
    ("saved", "saved", "ended while the monitor server was down"),
    (None, "failed", "server restarted; process gone"),
])
def test_a_job_whose_process_went_while_the_server_was_down(tmp_path, expts, q, outcome,
                                                             state, words):
    a = submit(q, expts)
    q.tick()
    q.spawner.procs[-1].write("Run ID: 101")
    q.tick()
    again = make_queue(tmp_path, expts)                # adopt -> None: gone
    assert "run_queue_process_gone" in again.journal.kinds
    if outcome:
        again.live.end_run(101, outcome)
    again.tick()
    j = job(again, a)
    assert j["state"] == state and words in j["reason"] and j["exit_code"] is None


# --- local folder, and the ops journal written outside the lock (review R2) --------------------

def test_the_default_folder_is_local(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.delenv(rq.DIR_ENV, raising=False)
    assert rq.default_dir() == str(Path.home() / ".waxx" / "run_queue")
    monkeypatch.setenv(rq.DIR_ENV, str(tmp_path / "q"))
    assert rq.default_dir() == str(tmp_path / "q")


def test_the_ops_journal_copy_is_never_written_under_the_lock(tmp_path, expts):
    seen = []

    class SlowShareJournal(Journal):
        def record(self, kind, **fields):
            seen.append((kind, queue._lock._is_owned()))
            super().record(kind, **fields)

    queue = None
    queue = make_queue(tmp_path, expts, journal=SlowShareJournal())
    submit(queue, expts)
    assert queue.submit({"path": str(expts / "rabi.py"), "after": [999]})["status"] == "error"
    queue.tick()
    run_through(queue, 101)
    queue.cancel({"id": 99})
    assert [k for k, _ in seen][:2] == ["run_queue_submit", "run_queue_refused"]
    assert "run_queue_end" in [k for k, _ in seen]
    assert not any(owned for _, owned in seen)
    lines = (tmp_path / "logs" / "run_queue" / "journal.jsonl").read_text().splitlines()
    assert len(lines) == len(seen)                       # nothing lost, nothing doubled


# --- saves never go back in time (review S5) ---------------------------------------------------

def test_an_older_snapshot_is_never_written_after_a_newer_one(q, expts, tmp_path):
    submit(q, expts)
    path = tmp_path / "logs" / "run_queue" / "queue.json"
    written = json.loads(path.read_text())["seq"]
    # a snapshot numbered before the one on disk (another thread's, slower to
    # reach the file) is dropped
    q._save_seq = written - 1
    submit(q, expts, "tof")                          # numbers its snapshot `written`
    assert json.loads(path.read_text())["seq"] == written
    q._save_seq = written + 5
    q._save()
    assert json.loads(path.read_text())["seq"] == written + 6


def test_the_holds_file_never_goes_back_either(tmp_path):
    hold = PersonHold(str(tmp_path / "h.json"))
    hold.hold("mine", "jp")
    hold._saved_seq = 99                             # a newer write is on disk
    hold.release("jp")                               # older-numbered: not written
    assert json.loads((tmp_path / "h.json").read_text())["active"] is True
    hold._save_seq = 200
    hold._save()
    assert json.loads((tmp_path / "h.json").read_text())["active"] is False


# --- the job's run from liveOD's queue_job (review S3) -----------------------------------------

def test_the_jobs_run_is_known_from_live_od_without_its_output(q, expts):
    a = submit(q, expts)
    q.tick()
    proc = q.spawner.procs[-1]
    proc.write("compiling", "shot 1/9")                  # WAX_VERBOSITY=0: no Run ID line
    q.live.start_run(555, launcher="run_loop", queue_job=str(a), client_pid=1)
    q.tick()
    assert job(q, a)["run_id"] is None                   # not the queue's launch: ignored
    q.live.start_run(101, launcher="kq", queue_job=str(a), client_pid=7101)
    q.tick()
    j = job(q, a)
    assert j["run_id"] == 101 and j["client_pid"] == 7101
    rec = [e for e in q.journal.entries if e["kind"] == "run_queue_run_id"][-1]
    assert rec["via"] == "liveOD"
    q.live.end_run(101)
    q.live.state["last_outcome"].update(launcher="kq", queue_job=str(a))
    proc.code = 0
    q.tick()
    assert job(q, a)["state"] == "saved"


def test_a_run_known_only_from_its_outcome(q, expts):
    a = submit(q, expts)
    q.tick()
    q.live.end_run(101)                                  # it ran and ended between polls
    q.live.state["last_outcome"].update(launcher="kq", queue_job=str(a))
    q.spawner.procs[-1].code = 0
    q.tick()
    assert job(q, a)["state"] == "saved" and job(q, a)["run_id"] == 101


# --- launching: saved before the spawn, never launched twice (review B3, S4, N1) ---------------

def test_the_job_is_saved_launching_before_its_process_starts(q, expts, tmp_path):
    seen = []
    plain = q.spawner

    def spawn(command, cwd, env, log_path):
        data = json.loads((tmp_path / "logs" / "run_queue" / "queue.json").read_text())
        seen.append([(j["id"], j["state"], j["launched_at"]) for j in data["jobs"]])
        return plain(command, cwd, env, log_path)
    q._spawn = spawn
    a = submit(q, expts)
    q.tick()
    assert seen == [[(a, "launching", q.clock.t)]]
    assert job(q, a)["state"] == "running"
    kinds = q.journal.kinds
    assert kinds.index("run_queue_launching") < kinds.index("run_queue_launch")


def _restart_with_a_launching_job(tmp_path, expts, q):
    a = submit(q, expts)
    gate = __import__("threading").Event()

    def hung(command, cwd, env, log_path):
        gate.wait(5)
        raise OSError("never mind")
    q._spawn, q._spawn_join_s = hung, 0.0
    q.tick()                                          # "launching" saved, spawn hanging
    assert job(q, a)["state"] == "launching"
    again = make_queue(tmp_path, expts)               # the server restarted meanwhile
    gate.set()
    return a, again


def test_a_launching_job_is_never_launched_again_and_is_adopted_from_live_od(
        tmp_path, expts, q):
    a, again = _restart_with_a_launching_job(tmp_path, expts, q)
    assert "run_queue_launching_at_restart" in again.journal.kinds
    adopted = Proc(job(again, a)["log_path"], 7101)
    again._adopt = lambda pid, started: adopted if pid == 7101 else None
    again.tick()
    assert again.spawner.calls == [] and job(again, a)["state"] == "launching"
    again.live.start_run(101, launcher="kq", queue_job=str(a), client_pid=7101)
    again.tick()
    j = job(again, a)
    assert j["state"] == "running" and j["run_id"] == 101 and j["adopted"]
    again.live.end_run(101)
    adopted.code = 0
    again.tick()
    assert job(again, a)["state"] == "saved" and again.spawner.calls == []


def test_a_launching_job_judged_from_its_outcome_or_failed_after_the_wait(
        tmp_path, expts, q):
    a, again = _restart_with_a_launching_job(tmp_path, expts, q)
    b = submit(again, expts)
    again.clock.t += rq.ORPHAN_WAIT_S - 1
    again.tick()
    assert job(again, a)["state"] == "launching"      # still looked for
    assert job(again, b)["state"] == "queued"         # the slot stays taken
    again.clock.t += 2
    again.tick()
    j = job(again, a)
    assert j["state"] == "failed" and j["reason"].startswith("server stopped while launching")
    again.tick()
    assert job(again, b)["state"] == "running"        # b goes; a was never launched again
    assert len(again.spawner.calls) == 1


def test_a_hung_launcher_stalls_nothing_and_its_job_is_found_in_live_od(q, expts):
    from waxx.util.device_state.detached import LaunchUnknown
    a = submit(q, expts)

    def unknown(command, cwd, env, log_path):
        raise LaunchUnknown("the launcher reported nothing within 30 s")
    q._spawn = unknown
    q.tick()
    assert job(q, a)["state"] == "launching" and "run_queue_launch_unknown" in q.journal.kinds
    assert q.list()["status"] == "ok"                 # requests answer meanwhile
    q.live.end_run(101)
    q.live.state["last_outcome"].update(launcher="kq", queue_job=str(a))
    q.tick()
    assert job(q, a)["state"] == "saved" and job(q, a)["run_id"] == 101


def test_after_a_restart_the_live_process_is_followed_first_and_never_called_gone(
        tmp_path, expts, q):
    a, b = submit(q, expts), submit(q, expts)
    q.tick()
    q.spawner.procs[-1].write("Run ID: 101")
    q.tick()
    # (a state no single server leaves: two jobs in the slot's states)
    data_path = tmp_path / "logs" / "run_queue" / "queue.json"
    data = json.loads(data_path.read_text())
    for j in data["jobs"]:
        if j["id"] == b:
            j.update(state="running", pid=5999, pid_started=1.0, run_id=102)
    data_path.write_text(json.dumps(data))
    alive = Proc(job(q, b)["log_path"] or job(q, a)["log_path"], 5999)
    again = make_queue(tmp_path, expts,
                       adopt=lambda pid, started: alive if pid == 5999 else None)
    assert again.info()["current"]["id"] == b          # the live one first
    again.tick()
    assert job(again, b)["state"] == "running"         # not called gone
    again.live.end_run(102)
    alive.code = 0
    again.tick()
    assert job(again, b)["state"] == "saved"
    again.tick()                                       # then a, whose process is gone
    assert job(again, a)["state"] == "failed"
    assert "process gone" in job(again, a)["reason"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process handles")
def test_a_process_that_cannot_be_opened_is_unknown_not_gone(monkeypatch):
    from waxx.util.device_state import detached
    watch = detached.ProcessWatch(os.getpid())         # no handle: liveness only
    assert watch.poll() is None and not watch.exit_code_known
    assert detached.ProcessWatch.open(0x7FFFFFF0) is None      # no such process


# --- unreadable files fail closed (review S6) --------------------------------------------------

def test_an_unreadable_queue_file_is_moved_aside_and_everything_paused(tmp_path, expts, q):
    a = submit(q, expts)
    folder = tmp_path / "logs" / "run_queue"
    (folder / "queue.json").write_text("{ this is not json")
    again = make_queue(tmp_path, expts)
    moved = [p.name for p in folder.iterdir() if p.name.startswith("queue.json.unreadable-")]
    assert len(moved) == 1
    assert (folder / moved[0]).read_text() == "{ this is not json"     # kept as it was
    info = again.info()
    assert info["paused"]["all"] and info["paused"]["agent"]
    assert "could not be read" in info["paused"]["all"]["reason"]
    assert submit(again, expts) == a + 1                 # ids go on from the journal
    again.tick()
    assert again.spawner.calls == []                     # nothing runs until a person resumes
    assert "run_queue_load_failed" in again.journal.kinds


def test_an_unreadable_queue_file_that_cannot_be_moved_is_never_overwritten(
        tmp_path, expts, monkeypatch):
    from waxx.util.device_state import person_hold
    folder = tmp_path / "logs" / "run_queue"
    folder.mkdir(parents=True)
    (folder / "queue.json").write_text("garbage")
    monkeypatch.setattr(person_hold.os, "rename", lambda a, b: (_ for _ in ()).throw(
        PermissionError("in use")))
    again = make_queue(tmp_path, expts)
    submit(again, expts)
    assert (folder / "queue.json").read_text() == "garbage"
    assert again.info()["paused"]["all"]


def test_an_unreadable_hold_file_starts_held(tmp_path):
    path = tmp_path / "person_hold.json"
    path.write_text("not json at all")
    hold = PersonHold(str(path))
    info = hold.info()
    assert info["active"] and info["by"] == "monitor server"
    assert info["reason"].startswith("could not read the hold file")
    moved = [p for p in tmp_path.iterdir() if p.name.startswith("person_hold.json.unreadable-")]
    assert len(moved) == 1 and moved[0].read_text() == "not json at all"
    assert json.loads(path.read_text())["active"] is True      # the hold is kept on
    assert PersonHold(str(path)).active


# --- the alarm ----------------------------------------------------------------------------

def test_the_alarm(q, expts, caplog):
    a = submit(q, expts)
    q.live.start_run(555, n_shots=1, last_shot_age_s=1.0, init_run_age_s=10.0)
    with caplog.at_level("WARNING", logger="waxx.util.device_state.run_queue"):
        q.tick()
        q.clock.t += 599
        q.tick()
        assert q.info()["alarm"] is None
        q.clock.t += 2
        q.tick()
        alarm = q.info()["alarm"]
        assert alarm["job"] == a and "555" in alarm["why"]
        q.clock.t += 300
        q.tick()
        q.clock.t += 301
        q.tick()
    warnings = [r for r in caplog.records if "RUN QUEUE ALARM" in r.getMessage()]
    assert len(warnings) == 2
    assert q.journal.kinds.count("run_queue_alarm") == 2
    q.live.end_run(555)
    q.tick()
    assert q.info()["alarm"] is None and job(q, a)["state"] == "running"
    assert "run_queue_alarm_cleared" in q.journal.kinds


# --- loops and the monitor ------------------------------------------------------------------

def test_a_loop_is_stopped_for_jobs_and_started_again_when_they_are_done(tmp_path, expts):
    loop = FakeLoop(key="expt_loop", title="Experiment loop", pick=True, path="C:/e/rabi.py")
    q = make_queue(tmp_path, expts, loops={"expt_loop": loop})
    a = submit(q, expts)
    q.tick()
    assert loop.stops == [("run queue", False)]       # graceful, no monitor start
    assert job(q, a)["state"] == "queued" and "finishing its run" in q.info()["waiting"]
    q.tick()
    assert len(loop.stops) == 1
    loop.finish_stop()
    q.tick()
    assert job(q, a)["state"] == "running"
    run_through(q, 101)
    q.tick()
    assert loop.starts == [("run queue", "C:/e/rabi.py")]
    assert q.monitor_starts == []                      # the loop owns the monitor again
    assert "run_queue_loop_resume" in q.journal.kinds


def test_a_latched_loop_is_not_started_again_and_the_monitor_is(tmp_path, expts):
    loop = FakeLoop()
    q = make_queue(tmp_path, expts, loops={"auto_tof": loop})
    submit(q, expts)
    q.tick()
    loop.state, loop.text = "latched", "run 99 was aborted in liveOD (Abort)"
    q.tick()
    run_through(q, 101)
    q.tick()
    assert loop.starts == [] and "run_queue_loop_not_resumed" in q.journal.kinds
    assert q.monitor_starts == ["the run queue has no job to run"]


def test_no_loop_restart_under_a_hold_but_the_monitor_starts(tmp_path, expts):
    loop = FakeLoop()
    q = make_queue(tmp_path, expts, loops={"auto_tof": loop})
    submit(q, expts)
    q.tick()
    loop.finish_stop()
    q.tick()
    q.hold_request({"reason": "mine now", "by": "jp"})
    run_through(q, 101)
    q.tick()
    assert loop.starts == [] and len(q.monitor_starts) == 1
    q.release_request({"by": "jp"})
    q.tick()
    assert loop.starts == [("run queue", None)]


def test_a_loop_a_person_is_stopping_is_left_to_them(tmp_path, expts):
    loop = FakeLoop()
    loop.state = "stopping"
    q = make_queue(tmp_path, expts, loops={"auto_tof": loop})
    submit(q, expts)
    q.tick()
    assert loop.stops == [] and q.info()["resume_loop"] is None


def test_the_monitor_is_asked_for_once_when_the_queue_runs_out(q, expts):
    submit(q, expts)
    submit(q, expts)
    q.tick()
    run_through(q, 101)
    q.tick()
    assert q.monitor_starts == []                      # one more to go
    run_through(q, 102)
    q.tick()
    q.tick()
    assert q.monitor_starts == ["the run queue has no job to run"]


# --- the person hold's watch through the queue -------------------------------------------------

def test_a_reset_on_an_agents_queued_run_sets_no_hold_but_a_persons_does(q, expts):
    submit(q, expts, owner="agent")
    q.tick()
    q.spawner.procs[-1].write("Run ID: 101")
    q.live.start_run(101)
    q.tick()
    q.live.state["reset_requested"] = True             # the agent resets its own run
    q.tick()
    assert not q.hold.active
    q.live.end_run(101, "discarded")
    q.spawner.procs[-1].code = 1
    q.tick()
    q.live.state.update(run_in_progress=True, run_id=555, reset_requested=False)
    q.tick()
    q.live.state["reset_requested"] = True             # a person's run
    q.tick()
    assert q.hold.active and q.hold.info()["run_id"] == 555


# --- the experiment side: a queued run leaves the monitor to the server ------------------------

def test_a_queued_run_does_not_restart_the_monitor(monkeypatch, capsys):
    import inspect
    from waxx.base import expt
    monkeypatch.delenv("WAXX_LAUNCHER", raising=False)
    assert expt._queue_restart_monitor(True) is True
    monkeypatch.setenv("WAXX_LAUNCHER", "run_loop")
    assert expt._queue_restart_monitor(True) is True
    assert expt._queue_restart_monitor(False) is False
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("WAXX_LAUNCHER", rq.LAUNCHER)
    monkeypatch.setenv("WAXX_QUEUE_JOB", "12")
    assert expt._queue_restart_monitor(True) is False
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "run queue (job 12)" in out[0] and "not restarted" in out[0]
    # end_wax applies it before anything else
    src = inspect.getsource(expt.Expt.end_wax)
    assert src.index("_queue_restart_monitor(restart_monitor)") < src.index("if restart_monitor")


# --- the real detached launcher (Windows; runs `echo` only) -------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="the detached launcher is Windows-only")
def test_the_detached_launcher_runs_a_command_outside_and_reads_its_exit_code(tmp_path):
    from waxx.util.device_state.detached import ProcessWatch, launch
    log_path = str(tmp_path / "out.log")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    watch = launch("echo hello from the queue & exit 3", cwd=str(tmp_path), env=env,
                   log_path=log_path)
    t0 = time.monotonic()
    while watch.poll() is None and time.monotonic() - t0 < 20:
        time.sleep(0.05)
    assert watch.poll() == 3
    assert watch.started is not None and abs(watch.started - time.time()) < 60
    with open(log_path, encoding="utf-8", errors="replace") as f:
        assert "hello from the queue" in f.read()
    # an exited process is not adopted; nor a pid with another creation time
    assert ProcessWatch.adopt(watch.pid, watch.started) is None
    assert ProcessWatch.adopt(os.getpid(), 1.0) is None
    me = ProcessWatch.open(os.getpid())
    assert ProcessWatch.adopt(os.getpid(), me.started) is not None
    watch.close()
    me.close()
