"""Each run queue job runs on the PC that submitted it (2026-11-15): the queue
with its launcher bridge (waxx.util.device_state.launcher_bridge) and the
real job launcher (waxx.util.device_state.launcher, ``kq launch``) talking
in-process.  liveOD is test_run_queue's fake; the PC's processes are fakes
(nothing is started); the queue's clock is injected; the launcher's link to
the queue is a direct call.  Nothing goes on the network, no real launcher or
process is ever started (conftest refuses both)."""
import json
import os
import time

import pytest

from test_run_queue import Clock, Journal, Live
from waxx.util.device_state import job_source
from waxx.util.device_state import launcher as L
from waxx.util.device_state import run_queue as rq
from waxx.util.device_state.run_queue import RunQueue

#: the real launcher start (conftest replaces L.start_detached in every test)
_START_DETACHED = L.start_detached

HOST = "pc-a"
ENV = {"kpy": r"C:\venv\activate.bat", "db": r"C:\db\device_db.py"}


class Child:
    """A process on the launcher's PC (what detached.launch returns)."""

    def __init__(self, pid, log_path):
        self.pid, self.started, self.code, self.log_path = pid, 4000.0 + pid, None, log_path

    def write(self, *lines):
        with open(self.log_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def poll(self):
        return self.code

    def close(self):
        pass


class PC:
    """The processes of one PC, as a launcher sees them."""

    def __init__(self):
        self.children, self.calls, self.kills, self.priorities = [], [], [], []

    def spawn(self, command, *, cwd, env, log_path, below_normal=False):
        child = Child(6000 + len(self.children), log_path)
        self.children.append(child)
        self.calls.append({"command": command, "cwd": cwd, "env": dict(env),
                           "log_path": log_path, "below_normal": below_normal})
        return child

    def adopt(self, pid, started):
        return next((c for c in self.children if c.pid == pid and c.code is None), None)

    def kill(self, pid, started):
        self.kills.append(pid)
        for c in self.children:
            if c.pid == pid and c.code is None:
                c.code = 1
                return {"pid": pid, "ok": True, "gone": False, "note": "SUCCESS"}
        return {"pid": pid, "ok": False, "gone": True, "note": "no such process"}

    def set_priority(self, pid, started):
        self.priorities.append((pid, started))
        return True


class Link:
    """The launcher's transport: the queue's request, called in-process (as
    JSON both ways, as on the wire)."""

    def __init__(self, q):
        self.q, self.down, self.sent = q, False, []

    def request(self, obj, timeout=None, attempts=1):
        if self.down:
            return None
        obj = json.loads(json.dumps(obj))
        self.sent.append(obj)
        return json.loads(json.dumps(self.q.launcher_request(obj)))


@pytest.fixture
def files(tmp_path):
    folder = tmp_path / "pc_a_code"
    folder.mkdir()
    for name in ("rabi", "tof"):
        (folder / f"{name}.py").write_text(f'"""{name}."""\nclass {name}: pass\n')
    (folder / "calib.py").write_text(
        "class Calibrate(object):\n    def prepare(self):\n"
        "        self.calibrates('t_pi', analysis='x')\n")
    return folder


def make_bq(tmp_path, **kw):
    """A queue with launchers, as the monitor server builds it."""
    live, clock, journal, starts = Live(), Clock(), Journal(), []
    args = dict(poll=live, run_exited=live.run_exited, live_od_reset=live.reset,
                fence=lambda: None, monitor_state=lambda: 2, server_busy=lambda: "",
                loops={}, start_monitor=lambda why: None, journal=journal, clock=clock,
                poll_every_s=0.0, gap_s=0.0, outcome_wait_s=0.0, client_name="kong",
                launchers=True, start_launcher=starts.append)
    args.update(kw)
    q = RunQueue(str(tmp_path / "server" / "run_queue"), **args)
    q.launchers.grace_s = 0.0
    q.launchers.spawn_wait_s = 2.0
    q._spawn_join_s = 0.01
    q.live, q.clock, q.journal, q.self_starts = live, clock, journal, starts
    return q


def submit(q, path, host=HOST, as_path=None, **kw):
    """A submit as kq sends it from ``host`` (``as_path``: the path it names,
    when not ``path`` itself)."""
    obj = dict({"path": str(as_path or path), "by": f"jp@{host}", "run_host": host,
                "host": host}, **job_source.source_info(str(path)))
    obj.update(kw)
    reply = q.submit(obj)
    assert reply["status"] == "ok", reply
    return reply


def launcher_for(reply_or_jobs, q, pc, tmp_path, *, host=HOST, env=None, name="pc_a"):
    jobs = reply_or_jobs["jobs"] if isinstance(reply_or_jobs, dict) else reply_or_jobs
    return L.Launcher([(j["id"], j["token"]) for j in jobs], transport=Link(q),
                      directory=str(tmp_path / name / "launcher"), host=host, spawn=pc.spawn,
                      adopt=pc.adopt, kill=pc.kill, set_priority=pc.set_priority,
                      command_for=lambda order: f"artiq_run {order['path']}",
                      env=ENV if env is None else env)


def cycle(q, *launchers, n=1):
    for _ in range(n):
        for la in launchers:
            if la.jobs:
                la.step()
        q.tick()
        time.sleep(0.005)


def until(cond, q, *launchers, n=60):
    for _ in range(n):
        if cond():
            return True
        cycle(q, *launchers)
    return cond()


def state(q, jid):
    return q.describe({"id": jid})["job"]["state"]


def start_running(q, la, jid, pc, run_id=101):
    """Launch ``jid`` through its launcher; liveOD shows its run."""
    assert until(lambda: state(q, jid) == "running", q, la), q.describe({"id": jid})
    child = pc.children[-1]
    child.write(f"Run ID: {run_id}")
    q.live.start_run(run_id, launcher="kq", queue_job=f"{jid}:{q._jobs[jid].token}")
    cycle(q, la, n=2)
    return child


# --- submit ---------------------------------------------------------------------------

def test_a_submit_from_an_older_kq_is_refused_and_the_server_never_opens_the_file(
        tmp_path, files, monkeypatch):
    q = make_bq(tmp_path)

    def never(*a, **k):
        raise AssertionError("the server opened a job's file")
    for name in ("file_sha256", "describe_source", "cold_reason"):
        monkeypatch.setattr(rq, name, never)
    r = q.submit({"path": str(files / "rabi.py"), "by": "jp"})       # an older kq
    assert r["status"] == "error" and "older than the run queue" in r["msg"]
    assert "uv sync" in r["msg"]
    # a path that exists only on the submitting PC: accepted from its text alone
    reply = submit(q, files / "calib.py", as_path=r"D:\elsewhere\calib.py")
    j = reply["jobs"][0]
    assert j["run_host"] == HOST and j["path"] == r"D:\elsewhere\calib.py"
    assert j["calibrates_declared"] == ["t_pi"] and j["cwd"] == r"D:\elsewhere"
    assert j["label"] == "calib"
    r = q.submit(dict(job_source.source_info(str(files / "rabi.py")), path="rabi.py",
                      run_host=HOST))
    assert r["status"] == "error" and "must be absolute" in r["msg"]


def test_a_queued_job_waits_for_its_launcher_and_after_a_restart_for_the_grace(
        tmp_path, files):
    q = make_bq(tmp_path)
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    q.tick()
    assert state(q, jid) == "queued"
    assert q.describe({"id": jid})["waiting"] == f"waiting for {HOST}'s launcher"
    q.launchers.grace_s = 15.0
    pc = PC()
    la = launcher_for(reply, q, pc, tmp_path)
    cycle(q, la, n=3)
    assert state(q, jid) == "queued" and pc.calls == []
    assert "just started" in q.info()["text"] or "attach first" in q._waiting
    q.clock.t += 16.0
    assert until(lambda: state(q, jid) == "running", q, la)


# --- a run, start to end --------------------------------------------------------------

def test_the_job_runs_on_its_own_pc_from_its_own_copy(tmp_path, files, monkeypatch):
    q = make_bq(tmp_path)
    for name in ("file_sha256", "describe_source", "cold_reason"):
        monkeypatch.setattr(rq, name, lambda *a, **k: pytest.fail("the server read the file"))
    pc = PC()
    reply = submit(q, files / "rabi.py", argv=["n=3"])
    jid, token = reply["ids"][0], reply["jobs"][0]["token"]
    la = launcher_for(reply, q, pc, tmp_path)
    child = start_running(q, la, jid, pc)
    call = pc.calls[0]
    assert call["command"] == f"artiq_run {files / 'rabi.py'}"
    assert call["cwd"] == str(files) and call["below_normal"] is False
    env = call["env"]
    assert env["WAXX_QUEUE_JOB"] == f"{jid}:{token}" and env["WAXX_LAUNCHER"] == "kq"
    assert env["WAXX_OWNER"] == "person" and "WAXX_QUEUE_WARM" not in env
    assert env["kpy"] == ENV["kpy"]                           # the PC's own environment
    view = q.describe({"id": jid})["job"]
    assert view["run_id"] == 101 and view["pid"] == child.pid
    assert view["launcher"]["host"] == HOST and view["launcher"]["attached"]
    # its output reaches the server's copy of the log (kq tail works anywhere)
    child.write("shot 1/2", "shot 2/2", "the last line")
    q.live.end_run(101, "saved")
    child.code = 0
    assert until(lambda: state(q, jid) == "saved", q, la)
    lines = q.tail({"id": jid, "token": token, "offset": 0})["lines"]
    assert "Run ID: 101" in lines and lines[-1] == "the last line"
    assert q.describe({"id": jid})["tail"][-1] == "the last line"
    cycle(q, la, n=2)
    assert la.jobs == {}                                      # done here: forgotten
    assert len(pc.calls) == 1                                 # started once


def test_a_file_changed_since_submit_is_skipped_by_its_launcher(tmp_path, files):
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    (files / "rabi.py").write_text("class rabi: edited = True\n")
    la = launcher_for(reply, q, pc, tmp_path)
    assert until(lambda: state(q, jid) == "skipped", q, la)
    assert "source changed since submit" in q.describe({"id": jid})["job"]["reason"]
    assert pc.calls == []


def test_a_pc_that_cannot_run_experiments_fails_the_job_with_the_reason(tmp_path, files):
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    la = launcher_for(reply, q, pc, tmp_path, env={"db": "x"})       # no %kpy%
    assert until(lambda: state(q, jid) == "failed", q, la)
    reason = q.describe({"id": jid})["job"]["reason"]
    assert f"could not start on {HOST}" in reason and "%kpy%" in reason
    assert pc.calls == []


def test_upper_case_env_names_from_windows_os_environ_count(tmp_path, files, monkeypatch):
    # dict(os.environ) on Windows has "KPY", "DB" (job 7186, 2026-11-15, was
    # refused on kong for "no %kpy%, %db%" with both set)
    monkeypatch.setattr(L, "_ENV_NAMES_IGNORE_CASE", True)
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    la = launcher_for(reply, q, pc, tmp_path, env={k.upper(): v for k, v in ENV.items()})
    assert until(lambda: state(q, jid) == "running", q, la), q.describe({"id": jid})
    assert len(pc.calls) == 1


# --- silence, kill, gone ----------------------------------------------------------------

def test_a_silent_launcher_holds_the_slot_until_a_person_says_it_is_gone(tmp_path, files):
    q = make_bq(tmp_path)
    pc_a, pc_b = PC(), PC()
    ra = submit(q, files / "rabi.py")
    a = ra["ids"][0]
    la = launcher_for(ra, q, pc_a, tmp_path)
    start_running(q, la, a, pc_a)
    rb = submit(q, files / "tof.py", host="pc-b")
    b = rb["ids"][0]
    lb = launcher_for(rb, q, pc_b, tmp_path, host="pc-b", name="pc_b")
    # --gone while its launcher answers is refused
    r = q.kill({"id": a, "owner": "person", "by": "jp", "gone": True})
    assert r["status"] == "error" and "answers" in r["msg"]
    # pc-a goes silent (its network, its sleep): the run may still be going
    q.clock.t += 40.0
    cycle(q, lb, n=5)
    assert state(q, a) == "running" and state(q, b) == "queued" and pc_b.calls == []
    alarm = q.info()["alarm"]
    assert alarm and "silent" in alarm["why"] and "--gone" in alarm["why"]
    assert q.live.exited == []                                 # liveOD is not told it ended
    r = q.kill({"id": a, "owner": "agent", "by": "agent"})
    assert r["status"] == "error"                              # a person's job
    r = q.kill({"id": a, "owner": "person", "by": "jp", "gone": True})
    assert r["status"] == "ok" and r["gone"] and not r["pending"]
    assert until(lambda: state(q, a) == "cancelled", q, lb)
    assert "declared gone by jp" in q.describe({"id": a})["job"]["reason"]
    assert until(lambda: state(q, b) == "running", q, lb)      # the slot is free again
    assert pc_a.kills == []                                    # nothing was killed


def test_kill_is_an_order_that_the_jobs_launcher_carries_out(tmp_path, files):
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    la = launcher_for(reply, q, pc, tmp_path)
    child = start_running(q, la, jid, pc)
    r = q.kill({"id": jid, "owner": "person", "by": "jp"})
    assert r["status"] == "ok" and r["pending"] and f"launcher on {HOST}" in r["note"]
    assert q.describe({"id": jid})["job"]["kill"]["pending"] is True
    assert until(lambda: state(q, jid) == "cancelled", q, la)
    assert pc.kills == [child.pid]
    k = q.describe({"id": jid})["job"]["kill"]
    assert k["pending"] is False and k["note"] == f"killed on {HOST}"
    assert k["results"][0]["host"] == HOST
    assert "force-killed by jp" in q.describe({"id": jid})["job"]["reason"]


def test_a_process_alive_after_its_job_was_declared_gone_blocks_every_launch(tmp_path, files):
    q = make_bq(tmp_path)
    pc = PC()
    ra = submit(q, files / "rabi.py")
    a = ra["ids"][0]
    la = launcher_for(ra, q, pc, tmp_path)
    child = start_running(q, la, a, pc)
    q.clock.t += 40.0
    q.kill({"id": a, "owner": "person", "by": "jp", "gone": True})
    until(lambda: state(q, a) == "cancelled", q)
    rb = submit(q, files / "tof.py")
    b = rb["ids"][0]
    lb = launcher_for(rb, q, pc, tmp_path, name="pc_a2")
    # pc-a comes back: job a's process was alive all along
    cycle(q, la, lb, n=5)
    assert state(q, b) == "queued"
    assert "although the job ended" in q._waiting and "nothing launches" in q._waiting
    child.code = 0                                            # it ends
    assert until(lambda: state(q, b) == "running", q, la, lb)


# --- restarts ------------------------------------------------------------------------------

def test_a_server_restart_follows_the_running_job_and_never_starts_it_again(tmp_path, files):
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    jid, token = reply["ids"][0], reply["jobs"][0]["token"]
    la = launcher_for(reply, q, pc, tmp_path)
    child = start_running(q, la, jid, pc)
    child.write("before the restart")
    cycle(q, la, n=2)
    # the monitor server restarts; the launcher's link finds the new one
    q2 = make_bq(tmp_path)
    q2.live.state = dict(q.live.state)
    la.transport = Link(q2)
    assert state(q2, jid) == "running"
    cycle(q2, la, n=3)
    child.write("after the restart")
    q2.live.end_run(101, "saved")
    child.code = 0
    assert until(lambda: state(q2, jid) == "saved", q2, la)
    assert len(pc.calls) == 1                                 # never started again
    lines = q2.tail({"id": jid, "token": token, "offset": 0})["lines"]
    assert lines.count("before the restart") == 1 and lines.count("after the restart") == 1


def test_a_silent_server_never_stops_the_run(tmp_path, files):
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    la = launcher_for(reply, q, pc, tmp_path)
    child = start_running(q, la, jid, pc)
    la.transport.down = True
    for _ in range(5):
        assert la.step() == L.RETRY_S
    assert child.code is None and pc.kills == [] and la.jobs   # the run goes on


def test_a_restarted_launcher_takes_its_child_back_and_never_starts_it_twice(tmp_path, files):
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    la = launcher_for(reply, q, pc, tmp_path)
    child = start_running(q, la, jid, pc)
    la.run = None                                             # (it is not run again)
    for lock in la._locks.values():
        lock.release()                                        # its process ended
    la2 = launcher_for(reply, q, pc, tmp_path)
    assert la2.jobs[jid]["children"]["cold"]["pid"] == child.pid
    q.clock.t += 40.0                                         # the first one is silent
    cycle(q, la2, n=3)
    assert len(pc.calls) == 1
    q.live.end_run(101, "saved")
    child.code = 0
    assert until(lambda: state(q, jid) == "saved", q, la2)


def test_a_launcher_that_lost_its_state_never_starts_a_job_that_ran(tmp_path, files):
    """The server's own guard: a job it knows ran (a pid, a run id) never gets
    a launch order again, even from a launcher with no record of it."""
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    la = launcher_for(reply, q, pc, tmp_path)
    start_running(q, la, jid, pc)
    for lock in la._locks.values():
        lock.release()
    q.clock.t += 40.0                                         # it went silent
    fresh = launcher_for(reply, q, pc, tmp_path, name="pc_a_wiped")   # no state file
    cycle(q, fresh, n=5)
    assert len(pc.calls) == 1                                 # not started a second time
    assert state(q, jid) == "running"                         # held: its end is unknown


def test_a_launcher_that_stopped_while_starting_holds_the_job_for_a_person(tmp_path, files):
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    jid, token = reply["ids"][0], reply["jobs"][0]["token"]
    folder = tmp_path / "pc_a" / "launcher"
    folder.mkdir(parents=True)
    # the last launcher wrote "starting", then died before it knew the pid
    (folder / f"{jid}_{token}.json").write_text(json.dumps(
        {"id": jid, "token": token, "children": {"cold": {"state": "starting"}}}))
    la = launcher_for(reply, q, pc, tmp_path)
    assert la.jobs[jid]["children"]["cold"]["state"] == "unknown"
    q._jobs[jid].state = "launching"                          # as the server left it
    q._current = jid
    cycle(q, la, n=10)
    assert pc.calls == []                                     # never started (again)
    assert state(q, jid) in ("launching", "running")          # held


def test_a_second_launcher_for_a_job_is_refused_while_the_first_answers(tmp_path, files):
    q = make_bq(tmp_path)
    pc_a, pc_a2 = PC(), PC()
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    la = launcher_for(reply, q, pc_a, tmp_path)
    la.step()
    other = launcher_for(reply, q, pc_a2, tmp_path, name="pc_a_other")
    other.step()
    assert other.jobs == {}                                   # it left the job alone
    assert until(lambda: state(q, jid) == "running", q, la)
    assert len(pc_a.calls) == 1 and pc_a2.calls == []


def test_another_pcs_launcher_is_never_given_the_job(tmp_path, files):
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    wrong = launcher_for(reply, q, pc, tmp_path, host="pc-z", name="pc_z")
    wrong.step()
    assert wrong.jobs == {} and pc.calls == []


def test_the_server_starts_a_launcher_only_for_its_own_hosts_jobs(tmp_path, files):
    q = make_bq(tmp_path)
    mine = submit(q, files / "rabi.py", host="kong")["jobs"][0]
    submit(q, files / "tof.py", host="pc-b")
    q.tick()
    assert q.self_starts == []                    # its submitter's own launcher is coming
    q.clock.t += 11.0
    q.tick()
    assert q.self_starts == [[(mine["id"], mine["token"])]]
    q.tick()
    assert len(q.self_starts) == 1                            # not again at once
    q.clock.t += 61.0
    q.tick()
    assert len(q.self_starts) == 2


# --- a run-again from another PC ------------------------------------------------------------

def test_a_run_again_for_another_pc_waits_for_that_pcs_launcher_to_read_the_file(tmp_path,
                                                                               files):
    q = make_bq(tmp_path)
    r = q.submit({"path": str(files / "calib.py"), "by": "jp@pc-b", "host": "pc-b",
                  "run_host": HOST, "again_of": 3, "owner": "person"})
    assert r["status"] == "ok", r
    jid = r["ids"][0]
    assert r["jobs"][0]["sha256"] == "" and r["jobs"][0]["run_host"] == HOST
    pc = PC()
    la = launcher_for(r, q, pc, tmp_path)
    la.step()                     # attaches; learns the path
    la.step()                     # sends the file's facts
    j = q.describe({"id": jid})["job"]
    assert j["sha256"] == job_source.file_sha256(files / "calib.py")
    assert j["calibrates_declared"] == ["t_pi"]
    assert until(lambda: state(q, jid) == "running", q, la)


# --- warm start through the launcher ---------------------------------------------------------

def test_warm_start_runs_through_the_jobs_launcher(tmp_path, files):
    q = make_bq(tmp_path, warm_ahead=True)
    q.live.state["features"] = {"reserve": True}
    q.live.count_resets()
    pc_a, pc_b = PC(), PC()
    ra = submit(q, files / "rabi.py")
    a = ra["ids"][0]
    la = launcher_for(ra, q, pc_a, tmp_path)
    assert until(lambda: state(q, a) == "running", q, la)
    q.live.start_run(101, launcher="kq", queue_job=f"{a}:{q._jobs[a].token}", n_shots=1,
                     n_shots_expected=20, client_pid=7001, last_shot_age_s=0.0)
    rb = submit(q, files / "tof.py", host="pc-b")
    b = rb["ids"][0]
    lb = launcher_for(rb, q, pc_b, tmp_path, host="pc-b", name="pc_b")
    assert until(lambda: bool(pc_b.calls), q, la, lb)
    call = pc_b.calls[0]
    token = call["env"]["WAXX_QUEUE_WARM"]
    assert call["below_normal"] is True and token == q._jobs[b].warm["token"]
    cycle(q, la, lb, n=2)
    assert q._jobs[b].warm["cmd_pid"] == pc_b.children[0].pid
    # the warm process (queue_gate) says hello and ready
    wp = {"job": f"{b}:{q._jobs[b].token}", "warm": token, "pid": 9001, "started": 4321.0}
    assert q.warm_request(dict(wp, stage="warming"))["status"] == "ok"
    assert q.warm_request(dict(wp, stage="ready", manifest=[]))["go"] is False
    # job a ends: b gets GO and runs as the same process -- never a cold one
    q.live.end_run(101, "saved")
    pc_a.children[0].code = 0
    assert until(lambda: state(q, b) in ("launching", "running"), q, la, lb)
    assert q.warm_request(dict(wp, stage="poll"))["go"] is True
    cycle(q, la, lb, n=3)
    assert state(q, b) == "running" and len(pc_b.calls) == 1
    assert (9001, 4321.0) in pc_b.priorities                 # GO's backstop, on its PC


def test_a_warm_process_that_stood_down_is_acknowledged_and_warmed_again(tmp_path, files):
    q = make_bq(tmp_path, warm_ahead=True)
    q.live.state["features"] = {"reserve": True}
    q.live.count_resets()
    pc_a, pc_b = PC(), PC()
    ra = submit(q, files / "rabi.py")
    a = ra["ids"][0]
    la = launcher_for(ra, q, pc_a, tmp_path)
    assert until(lambda: state(q, a) == "running", q, la)
    q.live.start_run(101, launcher="kq", queue_job=f"{a}:{q._jobs[a].token}", n_shots=1,
                     n_shots_expected=20, client_pid=7001, last_shot_age_s=0.0)
    rb = submit(q, files / "tof.py", host="pc-b")
    b = rb["ids"][0]
    lb = launcher_for(rb, q, pc_b, tmp_path, host="pc-b", name="pc_b")
    assert until(lambda: bool(pc_b.calls), q, la, lb)
    first = pc_b.calls[0]["env"]["WAXX_QUEUE_WARM"]
    pc_b.children[0].code = 78                                 # stood down before GO
    # its exit is taken (nothing of it ran), its record goes, and the queue
    # warms the job again (a new token: WARM_MAX_ATTEMPTS) -- never a cold child
    assert until(lambda: len(pc_b.calls) == 2, q, la, lb)
    second = pc_b.calls[1]["env"]["WAXX_QUEUE_WARM"]
    assert second != first and pc_b.calls[1]["below_normal"] is True
    assert state(q, b) == "queued" and q._jobs[b].warm["attempt"] == 2
    cycle(q, la, lb, n=2)
    reported = lb.report()["jobs"][0]["children"]
    assert first not in reported and second in reported        # the first one acknowledged
    assert "cold" not in reported


# --- faster hand-off (2026-11-16): held heartbeats, pokes, the server's address ---------

def test_a_held_heartbeat_gets_its_order_the_moment_the_job_launches(tmp_path, files):
    import threading
    from waxx.util.device_state import launcher_bridge as LB
    q = make_bq(tmp_path)
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    pc = PC()
    la = launcher_for(reply, q, pc, tmp_path)
    q._server_busy = lambda: "busy for the test"            # no launch yet
    la.step()                                               # attached
    beat = la.report()
    assert beat["wait_s"] == L.HOLD_S                       # nothing running: may be held
    first = q.launcher_request(beat)
    assert first["jobs"][str(jid)].get("_hold") is True and "order" not in first["jobs"][str(jid)]
    out = {}

    def held():
        t0 = time.monotonic()
        out["reply"] = q.launcher_hold(beat, first)
        out["s"] = time.monotonic() - t0
    t = threading.Thread(target=held)
    t.start()
    time.sleep(0.3)
    q._server_busy = lambda: ""
    q.tick()                                                # launching: the order is due now
    t.join(5)
    r = out["reply"]
    assert r["jobs"][str(jid)]["order"]["key"] == LB.COLD
    assert "_hold" not in r["jobs"][str(jid)]
    assert 0.25 <= out["s"] < 1.5 and r["next_beat_s"] == LB.FAST_BEAT_S


def test_a_held_heartbeat_with_no_order_ends_after_its_wait(tmp_path, files):
    from waxx.util.device_state import launcher_bridge as LB
    q = make_bq(tmp_path, server_busy=lambda: "busy for the test")
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    la = launcher_for(reply, q, PC(), tmp_path)
    la.step()
    beat = dict(la.report(), wait_s=0.3)
    first = q.launcher_request(beat)
    t0 = time.monotonic()
    r = q.launcher_hold(beat, first)
    assert 0.25 <= time.monotonic() - t0 < 1.0
    assert "order" not in r["jobs"][str(jid)] and r["next_beat_s"] == LB.AFTER_HOLD_BEAT_S
    # asked for more than the cap: held at most HOLD_MAX_S
    assert LB.HOLD_MAX_S < L.REQUEST_TIMEOUT_S


def test_a_job_not_near_the_front_is_never_held(tmp_path, files):
    q = make_bq(tmp_path, server_busy=lambda: "busy for the test")
    replies = [submit(q, files / "rabi.py") for _ in range(3)]
    third = replies[2]["ids"][0]
    la = launcher_for(replies[2], q, PC(), tmp_path)
    la.step()
    first = q.launcher_request(la.report())
    assert "_hold" not in first["jobs"][str(third)]          # third in line: beats as before


def test_a_launcher_asks_for_a_hold_only_with_nothing_running_and_reports_a_start_at_once(
        tmp_path, files):
    q = make_bq(tmp_path)
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    pc = PC()
    la = launcher_for(reply, q, pc, tmp_path)
    assert la.step() > L.BEAT_MIN_S                          # attached, nothing started
    q.tick()                                                  # launching
    assert state(q, jid) == "launching"
    assert la.step() == L.BEAT_MIN_S                          # started it: report it at once
    assert "wait_s" not in la.report()                        # a child runs: never held
    assert until(lambda: state(q, jid) == "running", q, la)


def test_a_launcher_attaching_or_a_child_changing_pokes_the_queue(tmp_path, files):
    q = make_bq(tmp_path)
    reply = submit(q, files / "rabi.py")
    pc = PC()
    la = launcher_for(reply, q, pc, tmp_path)
    q._poked.clear()
    la.step()
    assert q.wait_for_poke(0.0)                               # attached: tick now
    la.step()
    assert not q.wait_for_poke(0.0)                           # nothing new: no poke
    q.tick()
    la.step()                                                 # gets the order, starts it
    la.step()                                                 # reports it: a child changed
    assert q.wait_for_poke(0.0)


def test_the_launcher_takes_the_servers_address_from_kq(monkeypatch, tmp_path):
    from waxx.util.comms_server import comm_client
    assert L.parse_server("192.168.1.76:5123") == ("192.168.1.76", 5123)
    with pytest.raises(ValueError):
        L.parse_server("no-port")
    launched = []

    class Watch:
        pid = 1

        def close(self):
            pass
    # the real start_detached (conftest stands it in), with a fake process start
    _START_DETACHED([(5, "ab")], directory=str(tmp_path), server=("10.0.0.2", 77),
                    launch=lambda command, **kw: (launched.append(command), Watch())[1])
    assert launched[0].endswith("5:ab --server 10.0.0.2:77")
    # the first connection goes to that address, without discovery; later ones discover
    monkeypatch.setattr(comm_client.NetClient, "__init__",
                        lambda *a, **k: pytest.fail("discovery"))
    discovered = []
    monkeypatch.setattr(L, "_connect", lambda: discovered.append(1) or "by discovery")
    connect = L.connect_first_at(("10.0.0.2", 77))
    c = connect()
    assert isinstance(c, comm_client.MonitorClient) and c.server_address == ("10.0.0.2", 77)
    assert connect() == "by discovery" and discovered == [1]
    assert L.connect_first_at(None)() == "by discovery"




def test_kq_hands_its_server_address_to_the_launcher(monkeypatch):
    from waxx.util.device_state.run_queue_client import RunQueueClient
    seen = []
    monkeypatch.setattr(L, "start_detached", lambda pairs, **kw: seen.append(kw) or 42)

    class T:
        server_address = ("10.0.0.2", 77)
    client = RunQueueClient(T())
    r = client.start_launcher({"jobs": [{"id": 3, "token": "t", "run_host": "pc-a"}]})
    assert r == {"ok": True, "pid": 42} and seen == [{"server": ("10.0.0.2", 77)}]


# --- an ended child is reported at once (2026-11-17: Abort latency) -----------------------

class FakeTime:
    """The launcher's clock and sleep: a sleep moves the clock; ``on_sleep(n)``
    runs after the n-th sleep (1-based)."""

    def __init__(self, on_sleep=None):
        self.t, self.slept, self.on_sleep = 0.0, [], on_sleep

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s
        if self.on_sleep is not None:
            self.on_sleep(len(self.slept))


def test_a_launcher_wakes_for_a_child_that_exits_between_beats(tmp_path, files):
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    la = launcher_for(reply, q, pc, tmp_path)
    child = start_running(q, la, jid, pc)
    ft = FakeTime(on_sleep=lambda n: setattr(child, "code", 0) if n == 2 else None)
    la._clock, la._sleep = ft.clock, ft.sleep
    la._wait(5.0)
    assert ft.slept == [L.CHILD_POLL_S] * 2                   # woke at its exit, not after 5 s


def test_a_launcher_with_no_child_running_sleeps_the_whole_beat_at_once(tmp_path, files):
    q = make_bq(tmp_path)
    la = launcher_for(submit(q, files / "rabi.py"), q, PC(), tmp_path)
    ft = FakeTime()
    la._clock, la._sleep = ft.clock, ft.sleep
    la._wait(1.0)
    assert ft.slept == [1.0]


def test_a_child_whose_watch_fails_never_makes_the_launcher_beat_early(tmp_path, files):
    class Broken:
        pid = 1

        def poll(self):
            raise OSError("handle gone")
    q = make_bq(tmp_path)
    la = launcher_for(submit(q, files / "rabi.py"), q, PC(), tmp_path)
    la._watches[(1, "cold")] = Broken()
    ft = FakeTime()
    la._clock, la._sleep = ft.clock, ft.sleep
    la._wait(1.0)
    assert abs(sum(ft.slept) - 1.0) < 1e-9                    # the whole beat, in looks


def test_an_ended_child_beats_fast_until_the_server_took_its_output_and_exit(tmp_path, files):
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    jid, token = reply["ids"][0], reply["jobs"][0]["token"]
    la = launcher_for(reply, q, pc, tmp_path)
    child = start_running(q, la, jid, pc)
    ft = FakeTime()
    la._clock = ft.clock
    child.write("Acquisition for run 101 aborted.", "the last line")
    q.live.end_run(101, "aborted")
    child.code = 1
    beats = []
    for _ in range(20):
        if not la.jobs:
            break
        beats.append(la.step())
        q.tick()
    assert la.jobs == {}                                      # taken and forgotten
    assert beats[0] == L.BEAT_MIN_S                           # its output went at once
    assert state(q, jid) not in ("running", "launching")
    # the end was judged from the whole log: its last line is the run's last line
    assert q.tail({"id": jid, "token": token, "offset": 0})["lines"][-1] == "the last line"
    assert q.describe({"id": jid})["tail"][-1] == "the last line"


def test_fast_beats_after_an_end_stop_after_their_limit(tmp_path, files):
    q = make_bq(tmp_path)
    pc = PC()
    reply = submit(q, files / "rabi.py")
    jid = reply["ids"][0]
    la = launcher_for(reply, q, pc, tmp_path)
    child = start_running(q, la, jid, pc)
    ft = FakeTime()
    la._clock = ft.clock
    child.write("the last line")
    child.code = 0
    q.live.end_run(101, "saved")
    la.transport.down = True              # a server that never takes the end
    la.step()                              # records the exit; nothing reaches the server
    assert la._end_untaken()
    ft.t += L.FAST_AFTER_END_S + 0.1
    assert not la._end_untaken()           # back to the server's own beat
