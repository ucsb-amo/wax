"""The run queue's warm start (queue phase 2, waxx.util.device_state.run_queue):
when a warm launch starts, the GO handshake (saved first, idempotent, token /
pid / creation-time checks), every discard path, deaths at each stage,
restarts, the loop and monitor interplay, old queue.json files, the priority
flag and the environment variable.  Pure Python, as test_run_queue.py: liveOD
is a dict-backed fake, the processes are fakes, the clock is injected; the
warm process is played by calling ``RunQueue.warm_request`` the way
queue_gate does.  Nothing is launched and nothing goes on the network."""
import json
import os

import pytest

from test_run_queue import FakeLoop, job, make_queue, qj, submit
from waxx.util.device_state import detached
from waxx.util.device_state import run_queue as rq


@pytest.fixture
def expts(tmp_path):
    folder = tmp_path / "experiments"
    folder.mkdir()
    for name in ("rabi", "tof", "cal", "monitor", "fourth"):
        (folder / f"{name}.py").write_text(f'"""{name}."""\nclass {name}: pass\n')
    (folder / "calib.py").write_text(
        "class calib:\n    def prepare(self):\n        self.calibrates('t_pi', analysis='x')\n")
    (folder / "optout.py").write_text(
        "class optout(EnvExperiment, Base):\n    warm_launch = False\n"
        "    def scan_kernel(self): pass\n")
    (folder / "stages.py").write_text(
        "from kexp.control.misc import objective_stages\n"
        "class stages(EnvExperiment, Base):\n    def prepare(self):\n"
        "        self.os_control = objective_stages.controller()\n")
    return folder


@pytest.fixture
def lib(tmp_path):
    """A module file the warm process 'loaded' (its manifest)."""
    path = tmp_path / "lib.py"
    path.write_text("x = 1\n")
    return path


def manifest_of(*paths):
    out = []
    for p in paths:
        st = os.stat(p)
        out.append([str(p), st.st_mtime_ns, st.st_size])
    return out


def wq(tmp_path, expts, **kw):
    prio = []
    q = make_queue(tmp_path, expts, warm_ahead=True,
                   set_priority=lambda pid, started: prio.append((pid, started)) or True, **kw)
    q.priorities = prio
    q.live.state["features"] = {"reserve": True}
    q.live.count_resets()
    return q


def slot_running(q, expts, name="rabi", run_id=101, n_shots=1, **kw):
    """Submit a job, launch it (cold), and let liveOD show its run with
    ``n_shots`` taken."""
    ids = submit(q, expts, name, **kw)
    a = ids[0] if isinstance(ids, list) else ids
    q.tick()
    assert job(q, a)["state"] == "running"
    # init 1 s ago, last shot just now: an ETA from liveOD's shot timing
    q.live.start_run(run_id, launcher="kq", queue_job=qj(q, a), n_shots=n_shots,
                     n_shots_expected=20, client_pid=7001, last_shot_age_s=0.0)
    q.tick()
    return ids


def end_slot(q, run_id=101, outcome="saved", index=0, code=0):
    q.live.end_run(run_id, outcome)
    q.spawner.procs[index].code = code
    q.tick()


class WarmProc:
    """The warm process, as queue_gate drives the queue."""

    def __init__(self, q, jid, pid=9001, started=4321.0, call=-1):
        self.q, self.jid, self.pid, self.started = q, jid, pid, started
        self.call = q.spawner.calls[call]
        self.token = self.call["env"]["WAXX_QUEUE_WARM"]
        self.proc = q.spawner.procs[call]

    def ask(self, stage, manifest=None, **kw):
        obj = {"stage": stage, "job": qj(self.q, self.jid), "warm": self.token,
               "pid": self.pid, "started": self.started}
        if manifest is not None:
            obj["manifest"] = manifest
        obj.update(kw)
        return self.q.warm_request(obj)

    def ready(self, *paths):
        assert self.ask("warming")["status"] == "ok"
        return self.ask("ready", manifest=manifest_of(*paths))


def warm_rec(q, jid):
    """The raw warm record (with token and manifest)."""
    return q._jobs[jid].warm


def queue_json(q):
    with open(os.path.join(q.directory, "queue.json"), encoding="utf-8") as f:
        return {j["id"]: j for j in json.load(f)["jobs"]}


def ready_behind_slot(tmp_path, expts, lib, **kw):
    q = wq(tmp_path, expts, **kw)
    a = slot_running(q, expts)
    b = submit(q, expts, "tof")
    q.tick()
    wp = WarmProc(q, b)
    r = wp.ready(lib)
    assert r["status"] == "ok" and r["go"] is False
    return q, a, b, wp


# --- when a warm launch starts ------------------------------------------------------------

def test_warm_only_after_the_first_shot_with_reserve_and_one_at_a_time(tmp_path, expts):
    q = wq(tmp_path, expts)
    q.live.state.pop("features")
    a = slot_running(q, expts, n_shots=0)
    b, c = submit(q, expts, "tof"), submit(q, expts, "cal")
    q.tick()
    assert len(q.spawner.calls) == 1                  # no shot yet
    q.live.state["n_shots"] = 1
    q.tick()
    assert len(q.spawner.calls) == 1                  # liveOD does not advertise reserve
    q.live.state["features"] = {"reserve": False}
    q.tick()
    assert len(q.spawner.calls) == 1
    q.live.state["features"] = {"reserve": True}
    q.tick()
    assert len(q.spawner.calls) == 2
    call = q.spawner.calls[-1]
    assert call["below_normal"] is True and "below_normal" not in q.spawner.calls[0]
    assert call["env"]["WAXX_QUEUE_JOB"] == qj(q, b) and call["env"]["WAXX_LAUNCHER"] == "kq"
    assert len(call["env"]["WAXX_QUEUE_WARM"]) == 16
    assert "WAXX_QUEUE_WARM" not in q.spawner.calls[0]["env"]
    for _ in range(3):
        q.tick()
    assert len(q.spawner.calls) == 2                  # one warm process at a time, the head's
    jb = job(q, b)
    assert jb["state"] == "queued" and jb["warm"]["state"] == "warming"
    assert "token" not in jb["warm"] and "manifest" not in jb["warm"]
    assert job(q, c)["warm"] is None and job(q, a)["state"] == "running"
    rows = {r["id"]: r for r in q.list()["rows"]}
    assert rows[b]["warm"].startswith("warming") and rows[c]["warm"] == ""
    assert q.info()["warm"]["job"] == b
    assert "run_queue_warm_launch" in q.journal.kinds


def test_the_features_list_form_and_warm_ahead_off_by_default(tmp_path, expts):
    q = make_queue(tmp_path, expts)
    assert q.warm_ahead is False
    q.live.state["features"] = ["reserve"]
    slot_running(q, expts)
    submit(q, expts, "tof")
    q.tick()
    assert len(q.spawner.calls) == 1                  # off by default
    q.warm_ahead = True
    q.tick()
    assert len(q.spawner.calls) == 2                  # a list naming "reserve" counts


def test_never_an_after_on_the_running_job(tmp_path, expts):
    q = wq(tmp_path, expts)
    a = slot_running(q, expts)
    submit(q, expts, "tof", after=[a])
    submit(q, expts, "cal", at_end=True)              # behind it: it may not go first
    q.tick()
    assert len(q.spawner.calls) == 1


def test_never_while_the_slot_job_declares_a_calibration(tmp_path, expts):
    q = wq(tmp_path, expts)
    slot_running(q, expts, "calib")
    submit(q, expts, "tof")
    q.tick()
    assert len(q.spawner.calls) == 1


def test_not_while_the_slot_job_is_far_from_its_end(tmp_path, expts):
    q = wq(tmp_path, expts)
    slot_running(q, expts)
    q.live.state.update(n_shots=1, n_shots_expected=1000, init_run_age_s=10.0,
                        last_shot_age_s=0.0)              # 10 s a shot: ~2.8 h to go
    submit(q, expts, "tof")
    q.tick()
    assert len(q.spawner.calls) == 1
    q.live.state.update(n_shots=990)                      # (10-0)/990 s a shot: soon
    q.tick()
    assert len(q.spawner.calls) == 2


def test_an_unknown_eta_does_not_warm_until_it_is_known(tmp_path, expts):
    """N3: with no ETA (liveOD's shot timing incomplete, no duration
    history) a warm launch waits: warming right after the first shot of a
    long run would go stale and re-warm."""
    q = wq(tmp_path, expts)
    slot_running(q, expts)
    q.live.state.pop("last_shot_age_s")                   # shot timing not known
    submit(q, expts, "tof")
    q.tick()
    q.tick()
    assert len(q.spawner.calls) == 1
    q.live.state.update(last_shot_age_s=0.0)              # now it is: ~19 s to go
    q.tick()
    assert len(q.spawner.calls) == 2


def test_cold_jobs_never_warm(tmp_path, expts):
    q = wq(tmp_path, expts)
    slot_running(q, expts)
    b = submit(q, expts, "tof", cold=True)
    q.tick()
    assert len(q.spawner.calls) == 1 and job(q, b)["cold"] is True
    assert q.edit({"id": b, "owner": "person", "fields": {"cold": False}})["changed"] == ["cold"]
    q.tick()
    assert len(q.spawner.calls) == 2                      # warm once it is not cold
    bad = q.submit({"path": str(expts / "tof.py"), "cold": "yes"})
    assert bad["status"] == "error" and "cold must be" in bad["msg"]


@pytest.mark.parametrize("name, words", [("optout", "sets warm_launch = False"),
                                         ("stages", "objective_stages.controller")])
def test_an_opt_out_or_a_denylisted_prepare_makes_a_job_cold(tmp_path, expts, name, words):
    assert words in rq.cold_reason(expts / f"{name}.py")
    assert rq.cold_reason(expts / "tof.py") == ""
    assert "could not be read" in rq.cold_reason(expts / "missing.py")
    q = wq(tmp_path, expts)
    slot_running(q, expts)
    b = submit(q, expts, name)
    assert words in job(q, b)["cold_why"]
    q.tick()
    assert len(q.spawner.calls) == 1


def test_a_job_due_soon_warms_and_goes_only_once_due(tmp_path, expts, lib):
    q = wq(tmp_path, expts)
    slot_running(q, expts)
    b = submit(q, expts, "tof", due=q.clock.t + 120)
    q.tick()
    assert len(q.spawner.calls) == 1                      # due in 2 min: not yet
    q.clock.t += 61
    q.tick()
    assert len(q.spawner.calls) == 2                      # due within a minute: warms
    wp = WarmProc(q, b)
    wp.ready(lib)
    end_slot(q)
    assert job(q, b)["state"] == "queued" and wp.ask("poll")["go"] is False
    q.clock.t += 60
    q.tick()
    assert job(q, b)["state"] == "running" and wp.ask("poll")["go"] is True


# --- GO ---------------------------------------------------------------------------------

def test_go_is_saved_before_it_is_handed_out_and_repeats_are_the_same(tmp_path, expts, lib):
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib)
    r = wp.ask("poll")
    assert r["go"] is False and f"#{a} (rabi) is running, run 101" in r["why"]
    real = q._save
    q._save = lambda: False                               # queue.json cannot be written
    end_slot(q)
    assert job(q, a)["state"] == "saved"
    assert job(q, b)["state"] == "queued" and warm_rec(q, b)["go"] is None
    assert wp.ask("poll")["go"] is False
    assert "run_queue_launch_refused" in q.journal.kinds
    q._save = real
    q.tick()
    assert job(q, b)["state"] == "queued"                 # launches wait SAVE_RETRY_S
    q.clock.t += rq.SAVE_RETRY_S + 1
    q.tick()
    on_disk = queue_json(q)[b]
    assert on_disk["state"] in ("launching", "running")
    assert on_disk["warm"]["go"]["delivered"] is None     # decided and on disk, not handed out
    assert job(q, b)["state"] == "running" and job(q, b)["pid"] == wp.proc.pid
    r = wp.ask("poll")
    assert r == {"status": "ok", "go": True, "reset_count": 0}
    delivered = queue_json(q)[b]["warm"]["go"]["delivered"]
    assert delivered is not None
    q.clock.t += 5
    assert wp.ask("poll") == r                            # idempotent
    assert queue_json(q)[b]["warm"]["go"]["delivered"] == delivered
    assert q.priorities == [(9001, 4321.0)]               # the server's priority backstop
    kinds = q.journal.kinds
    assert kinds.index("run_queue_warm_go") < kinds.index("run_queue_warm_go_delivered")


def test_go_is_not_handed_out_when_its_hand_out_cannot_be_saved(tmp_path, expts, lib):
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib)
    end_slot(q)
    q._save = lambda: False
    r = wp.ask("poll")
    assert r["go"] is False and "could not be saved" in r["why"]
    assert warm_rec(q, b)["go"]["delivered"] is None


def test_a_decided_go_not_yet_on_disk_is_not_handed_out(tmp_path, expts, lib):
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib)
    warm_rec(q, b)["go"] = {"at": q.clock.t, "reset_count": 0, "delivered": None}
    r = wp.ask("poll")
    assert r["go"] is False and r["why"] == "GO is being saved"


def test_the_go_reply_carries_the_reset_count_last_seen(tmp_path, expts, lib):
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib)
    q.live.press("person")
    q.live.press("person")
    end_slot(q)
    assert wp.ask("poll")["reset_count"] == 2


def test_mismatched_tokens_pids_and_creation_times_are_told_to_abort(tmp_path, expts, lib):
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib)
    end_slot(q)
    assert "not the queue's current one" in wp.ask("poll", warm="0" * 16)["abort"]
    assert "not a job of this run queue" in wp.ask("poll", job=f"{b}:ffffff")["abort"]
    assert "not a job" in wp.ask("poll", job="nonsense")["abort"]
    assert "belongs to pid 9001" in wp.ask("poll", pid=9002)["abort"]
    assert "belongs to pid 9001" in wp.ask("poll", started=4322.0)["abort"]
    assert "no process creation time" in wp.ask("poll", started=None)["abort"]
    assert q.warm_request({"stage": "go!"})["status"] == "error"
    assert wp.ask("poll")["go"] is True                   # the real process is unaffected
    assert "run_queue_warm_abort_sent" in q.journal.kinds


def test_a_warm_process_without_a_creation_time_is_never_adopted(tmp_path, expts):
    """N1: Windows reuses pids -- a hello with a pid and no creation time is
    told to abort and nothing of it is recorded."""
    q = wq(tmp_path, expts)
    slot_running(q, expts)
    b = submit(q, expts, "tof")
    q.tick()
    wp = WarmProc(q, b, started=None)
    for stage in ("warming", "ready", "poll"):
        r = wp.ask(stage, manifest=[] if stage == "ready" else None)
        assert "no process creation time" in r["abort"], stage
    w = warm_rec(q, b)
    assert w["py_pid"] is None and w["py_started"] is None and w["state"] == "warming"


def test_the_priority_is_never_raised_on_a_pid_alone():
    """N1: set_normal_priority without a creation time does nothing (even
    for a live process it could reach)."""
    assert detached.set_normal_priority(os.getpid(), None) is False


def test_no_go_while_the_server_stops_launching(tmp_path, expts, lib):
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib)
    q.stop_launching("the monitor server is restarting")
    end_slot(q)
    assert job(q, b)["state"] == "queued" and warm_rec(q, b)["state"] == "ready"
    assert q.info()["waiting"] == "the monitor server is restarting"
    assert wp.ask("poll")["go"] is False
    q.stop_launching("")
    q.tick()
    assert wp.ask("poll")["go"] is True


def test_no_go_while_the_gate_is_busy(tmp_path, expts, lib):
    busy = [""]
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib, server_busy=lambda: busy[0])
    busy[0] = "a state reset"
    end_slot(q)
    assert job(q, b)["state"] == "queued" and q.info()["waiting"] == "a state reset"


def test_a_warming_head_is_waited_for_when_the_slot_frees(tmp_path, expts, lib):
    q = wq(tmp_path, expts)
    a = slot_running(q, expts)
    b = submit(q, expts, "tof")
    q.tick()
    wp = WarmProc(q, b)
    wp.ask("warming")
    end_slot(q)
    assert job(q, b)["state"] == "queued" and len(q.spawner.calls) == 2   # no cold second
    assert "warming" in q.info()["waiting"]
    r = wp.ask("poll")
    assert r["go"] is False and r.get("resend_ready") is True          # it must say ready
    wp.ask("ready", manifest=manifest_of(lib))
    q.tick()
    assert wp.ask("poll")["go"] is True


def test_a_warm_job_runs_to_its_end_and_its_estimate_basis_says_so(tmp_path, expts, lib):
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib)
    end_slot(q)
    assert wp.ask("poll")["go"] is True
    q.live.start_run(102, launcher="kq", queue_job=qj(q, b), n_shots=1, client_pid=9001)
    q.tick()
    assert job(q, b)["run_id"] == 102
    assert "GO to end, warm time excluded" in job(q, b)["estimate"]["basis"]
    q.live.end_run(102, "saved")
    wp.proc.code = 0
    q.tick()
    assert job(q, b)["state"] == "saved"
    assert q.monitor_starts == ["the run queue has no job to run"]


# --- discards ------------------------------------------------------------------------------

def _warm_agent_job(tmp_path, expts, lib):
    q = wq(tmp_path, expts)
    a = slot_running(q, expts)
    b = submit(q, expts, "tof", owner="agent")
    q.tick()
    wp = WarmProc(q, b)
    wp.ready(lib)
    return q, a, b, wp


def _discarded(q, b, wp, words):
    q.tick()
    w = warm_rec(q, b)
    assert w is not None and w["state"] == "aborting", w
    assert words in w["why"], w["why"]
    assert words in wp.ask("poll")["abort"]
    assert "run_queue_warm_discard" in q.journal.kinds


def test_a_person_submitting_ahead_discards_and_the_rewarm_waits_for_the_exit(tmp_path, expts,
                                                                              lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    p = submit(q, expts, "cal")                           # a person's job: ahead of the agent's
    _discarded(q, b, wp, f"no longer next (#{p} is)")
    for _ in range(3):
        q.tick()
    assert len(q.spawner.calls) == 2                      # never two warm processes
    wp.proc.code = rq.EXIT_WARM_ABORTED
    q.tick()
    assert warm_rec(q, b) is None and job(q, b)["state"] == "queued"
    assert len(q.spawner.calls) == 3                      # now the person's job warms
    assert q.spawner.calls[-1]["env"]["WAXX_QUEUE_JOB"] == qj(q, p)


def test_a_move_discards(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    c = submit(q, expts, "cal", owner="agent")
    q.move({"id": c, "to_index": 0, "owner": "agent"})
    _discarded(q, b, wp, "no longer next")


def test_a_pause_discards(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    q.pause({"scope": "agent", "by": "jp", "owner": "person"})
    _discarded(q, b, wp, "agent jobs paused")


def test_a_person_hold_discards_an_agents_warm_job_not_a_persons(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    q.hold_request({"reason": "mine", "by": "jp"})
    _discarded(q, b, wp, "held")
    q2 = wq(tmp_path / "two", expts)
    slot_running(q2, expts)
    pb = submit(q2, expts, "tof")
    q2.tick()
    WarmProc(q2, pb).ready(lib)
    q2.hold_request({"reason": "mine", "by": "jp"})
    q2.tick()
    assert warm_rec(q2, pb)["state"] == "ready"


@pytest.mark.parametrize("fields, words", [({"argv": ["x=1"]}, "edited (argv)"),
                                           ({"paused": True}, "edited (paused)"),
                                           ({"write_back": False}, "edited (write_back)"),
                                           ({"cold": True}, "made cold")])
def test_an_edit_discards(tmp_path, expts, lib, fields, words):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    assert q.edit({"id": b, "owner": "agent", "fields": fields})["status"] == "ok"
    _discarded(q, b, wp, words)


def test_an_edit_of_the_label_keeps_it(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    q.edit({"id": b, "owner": "agent", "fields": {"label": "x"}})
    q.tick()
    assert warm_rec(q, b)["state"] == "ready"


def test_a_cancel_tells_the_warm_process_to_exit(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    reply = q.cancel({"id": b, "owner": "agent", "by": "me"})
    assert reply["note"] == "not started (warm process told to exit)"
    assert job(q, b)["state"] == "cancelled" and warm_rec(q, b)["state"] == "aborting"
    assert "cancelled by me" in wp.ask("poll")["abort"]
    wp.proc.code = rq.EXIT_WARM_ABORTED
    q.tick()
    assert warm_rec(q, b) is None and job(q, b)["state"] == "cancelled"


def test_a_stopped_chain_discards(tmp_path, expts, lib):
    q = wq(tmp_path, expts)
    a, b = slot_running(q, expts, repeat=2)
    q.tick()
    wp = WarmProc(q, b)
    wp.ready(lib)
    end_slot(q, outcome="saved_incomplete")
    assert job(q, a)["state"] == "failed" and job(q, b)["state"] == "cancelled"
    assert warm_rec(q, b)["state"] == "aborting" and "abort" in wp.ask("poll")


def test_a_stale_ready_and_warm_start_switched_off_discard(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    q.clock.t += rq.WARM_READY_MAX_S + 1
    _discarded(q, b, wp, "stale")
    q2, a2, b2, wp2 = _warm_agent_job(tmp_path / "two", expts, lib)
    q2.warm_ahead = False
    _discarded(q2, b2, wp2, "switched off")


def test_a_warming_process_that_never_gets_ready_is_discarded(tmp_path, expts):
    q = wq(tmp_path, expts)
    slot_running(q, expts)
    b = submit(q, expts, "tof")
    q.tick()
    wp = WarmProc(q, b)
    wp.ask("warming")
    q.clock.t += rq.WARM_PREP_MAX_S + 1
    _discarded(q, b, wp, "not ready")


def test_a_manifest_change_rewarms_after_the_old_process_exits(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    lib.write_text("x = 22\n")
    _discarded(q, b, wp, "a file it loaded changed")
    q.tick()
    assert len(q.spawner.calls) == 2
    wp.proc.code = rq.EXIT_WARM_ABORTED
    q.tick()
    assert len(q.spawner.calls) == 3                      # re-warmed: attempt 2
    assert warm_rec(q, b)["attempt"] == 2 and job(q, b)["warm_attempts"] == 2
    wp2 = WarmProc(q, b, pid=9002)
    assert wp.ask("poll")["abort"]                        # the old token is dead
    wp2.ready(lib)
    end_slot(q)
    assert wp2.ask("poll")["go"] is True


def test_a_manifest_change_at_go_discards_and_the_job_launches_cold(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    q._warm_checked[b] = q.clock.t                       # checked just now
    lib.write_text("x = 333\n")
    end_slot(q)
    assert job(q, b)["state"] == "queued" and warm_rec(q, b)["state"] == "aborting"
    assert "exiting" in q.info()["waiting"] or "discarded" in q.info()["waiting"]
    wp.proc.code = rq.EXIT_WARM_ABORTED
    q.tick()
    q.tick()
    assert job(q, b)["state"] == "running"
    assert "below_normal" not in q.spawner.calls[-1]
    assert "WAXX_QUEUE_WARM" not in q.spawner.calls[-1]["env"]


def test_network_paths_are_never_kept_or_stat_ed_by_the_tick(tmp_path, expts, lib, monkeypatch):
    """S5: a manifest entry on a share (UNC or a mapped network drive) is
    dropped when "ready" is stored, and never stat-ed by the tick's
    freshness check (a hung share must not stall the queue's one thread)."""
    from waxx.util.device_state import queue_gate
    q = wq(tmp_path, expts)
    slot_running(q, expts)
    b = submit(q, expts, "tof")
    q.tick()
    wp = WarmProc(q, b)
    wp.ask("warming")
    unc = r"\\fileserver\share\roi.xlsx"
    r = wp.ask("ready", manifest=manifest_of(lib) + [[unc, 1, 2]])
    assert r["status"] == "ok"
    kept = [m[0] for m in warm_rec(q, b)["manifest"]]
    assert kept == [str(lib)]
    # an older record that still holds one: skipped by the check, not stat-ed
    warm_rec(q, b)["manifest"].append([unc, 1, 2])
    stats = []
    real_stat = os.stat
    monkeypatch.setattr(rq.os, "stat", lambda p, *a, **k: (stats.append(str(p)),
                                                             real_stat(p, *a, **k))[1])
    assert q._warm_stale(q._jobs[b]) == ""
    assert unc not in stats and str(lib) in stats
    assert queue_gate.is_network_path(unc)


def test_experiment_file_drift_discards_then_the_job_is_skipped_as_today(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    (expts / "tof.py").write_text('"""tof, edited."""\nclass tof: pass\n')
    _discarded(q, b, wp, "experiment file changed")
    wp.proc.code = rq.EXIT_WARM_ABORTED
    end_slot(q)
    q.tick()
    assert job(q, b)["state"] == "skipped" and job(q, b)["reason"] == "source changed since submit"
    assert len(q.spawner.calls) == 2


def test_a_discarded_process_that_does_not_exit_raises_the_alarm_and_is_never_killed(
        tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    q.cancel({"id": b, "owner": "agent", "by": "me"})
    q.tick()
    assert q.info()["warm_alarm"] is None
    q.clock.t += rq.WARM_EXIT_ALARM_S + 1
    q.tick()
    alarm = q.info()["warm_alarm"]
    assert alarm["job"] == b and "has not exited" in alarm["text"] and "nothing kills" in alarm["text"]
    assert q.journal.kinds.count("run_queue_warm_not_exiting") == 1
    q.tick()
    assert q.journal.kinds.count("run_queue_warm_not_exiting") == 1
    assert wp.proc.code is None and not hasattr(wp.proc, "killed")
    wp.proc.code = rq.EXIT_WARM_ABORTED
    q.tick()
    assert q.info()["warm_alarm"] is None


# --- deaths ---------------------------------------------------------------------------------

def test_death_while_warming_makes_the_job_cold_and_keeps_it_queued(tmp_path, expts):
    """S4: a warm process that dies before GO does not fail the job (the
    failure may exist only because the launch was warm): the job is made
    cold, stays queued, and launches cold when the slot frees."""
    q = wq(tmp_path, expts)
    slot_running(q, expts)
    b = submit(q, expts, "tof")
    q.tick()
    WarmProc(q, b).proc.code = 1
    q.tick()
    jb = job(q, b)
    assert jb["state"] == "queued" and jb["warm"] is None
    assert jb["cold_why"].startswith("warm attempt failed (exit 1): ")
    assert "while warming, before it took the machine" in jb["cold_why"]
    assert q.journal.kinds.count("run_queue_warm_made_cold") == 1
    rec = [e for e in q.journal.entries if e["kind"] == "run_queue_warm_exited"][-1]
    assert rec["failed"] is False and rec["made_cold"] is True
    q.tick()
    assert len(q.spawner.calls) == 2                      # never warmed again
    end_slot(q)
    q.tick()
    assert job(q, b)["state"] == "running"
    last = q.spawner.calls[-1]
    assert "WAXX_QUEUE_WARM" not in last["env"] and "below_normal" not in last


def test_death_while_ready_makes_the_job_cold(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    wp.proc.code = 1
    q.tick()
    jb = job(q, b)
    assert jb["state"] == "queued" and "while ready" in jb["cold_why"]
    assert jb["cold_why"].startswith("warm attempt failed (exit 1)")


def test_a_cold_launch_after_a_failed_warm_one_fails_as_before(tmp_path, expts):
    """A genuine bug: the warm attempt dies, then the cold launch dies too --
    that one fails the job (and its chain) as a cold run always did."""
    q = wq(tmp_path, expts)
    slot_running(q, expts)
    b = submit(q, expts, "tof")
    q.tick()
    WarmProc(q, b).proc.code = 1
    q.tick()
    end_slot(q)
    q.tick()
    assert job(q, b)["state"] == "running"
    q.spawner.procs[-1].code = 1
    q.tick()
    assert job(q, b)["state"] == "failed"


def test_a_stale_discard_does_not_count_towards_the_cold_cap(tmp_path, expts, lib):
    """N3: a warm launch discarded only for being stale (the slot job ran
    long) gives its attempt back; any number of stale re-warms never make
    the job cold."""
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib)
    for n in range(rq.WARM_MAX_ATTEMPTS + 1):
        q.clock.t += rq.WARM_READY_MAX_S + 1
        q.tick()
        w = warm_rec(q, b)
        assert w["state"] == "aborting" and w["why"].endswith(rq.WARM_STALE_MARK)
        q.spawner.procs[-1].code = rq.EXIT_WARM_ABORTED
        q.tick()                                          # exits; re-warms at once
        assert job(q, b)["warm_attempts"] == 1, n
        assert job(q, b)["cold_why"] == ""
        wp = WarmProc(q, b, pid=9100 + n)
        wp.ready(lib)
    rec = [e for e in q.journal.entries if e["kind"] == "run_queue_warm_exited"][-1]
    assert rec["attempt_refunded"] is True


def test_a_process_that_stood_down_by_itself_leaves_the_job_queued(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    wp.proc.code = rq.EXIT_WARM_ABORTED                   # e.g. lost the queue for 600 s
    q.tick()
    assert job(q, b)["state"] == "queued"
    rec = [e for e in q.journal.entries if e["kind"] == "run_queue_warm_exited"][-1]
    assert rec["failed"] is False and rec["code"] == 78
    assert warm_rec(q, b)["attempt"] == 2                 # warmed again (attempts are capped)
    for attempt in (3,):
        q.spawner.procs[-1].code = rq.EXIT_WARM_ABORTED
        q.tick()
        assert warm_rec(q, b)["attempt"] == attempt
    q.spawner.procs[-1].code = rq.EXIT_WARM_ABORTED
    q.tick()
    q.tick()
    assert warm_rec(q, b) is None and job(q, b)["warm_attempts"] == rq.WARM_MAX_ATTEMPTS


def test_death_after_go_saved_before_hand_out_requeues(tmp_path, expts, lib):
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib)
    end_slot(q)
    assert job(q, b)["state"] == "running"
    wp.proc.code = 1
    q.tick()
    jb = job(q, b)
    assert "back in the queue (nothing ran)" in jb["reason"]
    assert "run_queue_warm_requeued" in q.journal.kinds
    assert jb["state"] == "running" and "below_normal" not in q.spawner.calls[-1]   # cold now


def test_a_cancel_between_go_and_hand_out_ends_it_cancelled(tmp_path, expts, lib):
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib)
    end_slot(q)
    reply = q.cancel({"id": b, "owner": "person", "by": "jp"})
    assert reply["note"] == "not started (warm process told to exit)"
    assert "cancelled by jp" in wp.ask("poll")["abort"]
    wp.proc.code = rq.EXIT_WARM_ABORTED
    q.tick()
    jb = job(q, b)
    assert jb["state"] == "cancelled" and "before its warm process was given" in jb["reason"]


def test_death_after_hand_out_is_judged_as_any_run(tmp_path, expts, lib):
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib)
    end_slot(q)
    assert wp.ask("poll")["go"] is True
    wp.proc.code = 1
    q.tick()
    jb = job(q, b)
    assert jb["state"] == "failed" and "back in the queue" not in jb["reason"]


# --- restarts ---------------------------------------------------------------------------------

def _restarted(tmp_path, expts, alive):
    """A second queue on the same folder; ``alive``: {(pid, started): proc}."""
    q2 = wq(tmp_path, expts, adopt=lambda pid, started: alive.get((pid, started)))
    return q2


def test_restart_readopts_a_warm_process_by_pid_and_creation_time(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    slot = q.spawner.procs[0]
    alive = {(slot.pid, slot.started): slot, (wp.proc.pid, wp.proc.started): wp.proc}
    q2 = _restarted(tmp_path, expts, alive)
    assert q2._jobs[b].warm["state"] == "ready" and b in q2._warm_procs
    assert "run_queue_warm_adopted" in q2.journal.kinds
    wp.q = q2
    q2.tick()
    q2.live.end_run(101, "saved")
    slot.code = 0
    q2.tick()
    q2.tick()
    assert wp.ask("poll")["go"] is True
    # a pid now used by another process (other creation time) is not taken
    q3 = _restarted(tmp_path / "x", expts, {(wp.proc.pid, 1.0): wp.proc})
    assert q3._adopt_warm({"cmd_pid": wp.proc.pid, "cmd_started": wp.proc.started}) is None
    assert q3._adopt_warm({"cmd_pid": wp.proc.pid, "cmd_started": None}) is None


def test_restart_with_the_warm_process_gone_drops_the_record(tmp_path, expts, lib):
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    q2 = _restarted(tmp_path, expts, {})
    assert q2._jobs[b].warm is None and q2._jobs[b].state == "queued"
    assert "run_queue_warm_gone_at_restart" in q2.journal.kinds


def test_restart_before_any_pid_waits_for_the_hello(tmp_path, expts):
    q = wq(tmp_path, expts)
    slot_running(q, expts)
    b = submit(q, expts, "tof")
    q.tick()
    data = json.load(open(os.path.join(q.directory, "queue.json"), encoding="utf-8"))
    for j in data["jobs"]:
        if j["id"] == b:
            j["warm"]["cmd_pid"] = j["warm"]["cmd_started"] = None
    with open(os.path.join(q.directory, "queue.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    slot = q.spawner.procs[0]
    q2 = _restarted(tmp_path, expts, {(slot.pid, slot.started): slot})
    token = q.spawner.calls[-1]["env"]["WAXX_QUEUE_WARM"]
    q2.tick()
    assert q2._jobs[b].warm is not None
    r = q2.warm_request({"stage": "warming", "job": qj(q2, b), "warm": token, "pid": 9001,
                         "started": 4321.0})
    assert r["go"] is False and q2._jobs[b].warm["py_pid"] == 9001
    q2._adopt = lambda pid, started: slot if pid == slot.pid else None   # its python is gone
    q2.tick()
    assert q2._jobs[b].warm is None and q2._jobs[b].state == "queued"


def _go_then_crash(tmp_path, expts, lib, delivered=False):
    q, a, b, wp = ready_behind_slot(tmp_path, expts, lib)
    end_slot(q)
    if delivered:
        assert wp.ask("poll")["go"] is True
    path = os.path.join(q.directory, "queue.json")
    data = json.load(open(path, encoding="utf-8"))
    for j in data["jobs"]:
        if j["id"] == b:
            j["state"] = "launching"                      # crashed between the two saves
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return q, b, wp


def test_restart_with_a_saved_go_and_the_process_alive_hands_it_out(tmp_path, expts, lib):
    q, b, wp = _go_then_crash(tmp_path, expts, lib)
    q2 = _restarted(tmp_path, expts, {(wp.proc.pid, wp.proc.started): wp.proc})
    assert q2._jobs[b].state == "running" and q2._jobs[b].adopted
    wp.q = q2
    assert wp.ask("poll")["go"] is True


def test_restart_with_a_saved_go_not_handed_out_and_the_process_gone_requeues(tmp_path, expts,
                                                                              lib):
    q, b, wp = _go_then_crash(tmp_path, expts, lib)
    q2 = _restarted(tmp_path, expts, {})
    assert q2._jobs[b].state == "queued" and q2._jobs[b].warm is None
    assert "back in the queue" in q2._jobs[b].reason


def test_restart_with_a_handed_out_go_and_the_process_gone_is_an_orphan(tmp_path, expts, lib):
    q, b, wp = _go_then_crash(tmp_path, expts, lib, delivered=True)
    q2 = _restarted(tmp_path, expts, {})
    assert q2._jobs[b].state == "launching"
    q2.clock.t += rq.ORPHAN_WAIT_S + 1
    q2.tick()
    assert q2._jobs[b].state == "failed" and "server stopped while launching" in q2._jobs[b].reason


# --- loops, the monitor, old files, the environment ---------------------------------------------

def test_the_loop_the_queue_stopped_comes_back_after_a_warm_chain(tmp_path, expts, lib):
    loop = FakeLoop(owner="agent")
    q = wq(tmp_path, expts, loops={"auto_tof": loop})
    a = submit(q, expts, "rabi")
    q.tick()
    assert loop.stops and loop.state == "stopping"
    loop.finish_stop()
    q.tick()
    assert job(q, a)["state"] == "running"
    q.live.start_run(101, launcher="kq", queue_job=qj(q, a), n_shots=1, client_pid=7001,
                     n_shots_expected=20, last_shot_age_s=0.0)     # (an ETA is known)
    b = submit(q, expts, "tof")
    q.tick()
    wp = WarmProc(q, b)
    wp.ready(lib)
    end_slot(q)
    assert wp.ask("poll")["go"] is True
    q.live.start_run(102, launcher="kq", queue_job=qj(q, b), n_shots=1, client_pid=9001)
    q.tick()
    q.live.end_run(102, "saved")
    wp.proc.code = 0
    q.tick()
    q.tick()
    assert loop.starts and loop.state == "running" and q.monitor_starts == []


def test_a_persons_loop_means_no_slot_job_and_no_warm(tmp_path, expts):
    loop = FakeLoop(owner="person")
    q = wq(tmp_path, expts, loops={"auto_tof": loop})
    submit(q, expts, "tof", owner="agent")
    for _ in range(3):
        q.tick()
    assert q.spawner.calls == []


def test_an_old_queue_json_loads_and_runs(tmp_path, expts):
    folder = tmp_path / "logs" / "run_queue"
    folder.mkdir(parents=True)
    old_job = {"id": 3, "token": "abcdef", "path": str(expts / "tof.py"),
               "sha256": rq.file_sha256(expts / "tof.py"), "label": "tof", "state": "queued",
               "rank": 1.0, "submitted_at": 1.0}
    (folder / "queue.json").write_text(json.dumps({"version": 1, "next_id": 4,
                                                   "jobs": [old_job]}))
    q = wq(tmp_path, expts)
    j = q._jobs[3]
    assert j.warm is None and j.cold is False and j.cold_why == "" and j.warm_attempts == 0
    q.tick()
    assert job(q, 3)["state"] == "running"
    # a record with fields this code does not know (a newer server's) loads too
    assert rq.Job.from_dict(dict(old_job, some_future_field=1)).id == 3


def test_the_warm_variable_never_leaks_into_a_cold_launch(tmp_path, expts, monkeypatch):
    monkeypatch.setenv("WAXX_QUEUE_WARM", "left-over")
    q = wq(tmp_path, expts)
    submit(q, expts, "tof")
    q.tick()
    assert "WAXX_QUEUE_WARM" not in q.spawner.calls[0]["env"]


def test_the_launcher_adds_below_normal_only_for_a_warm_launch(tmp_path, monkeypatch):
    seen = []

    class FakePopen:
        def __init__(self, *a, **kw):
            seen.append(kw.get("creationflags", 0))

    monkeypatch.setattr(detached, "Popen", FakePopen)
    base = {"command": "echo x", "log_path": str(tmp_path / "l")}
    detached._start_job(dict(base))
    detached._start_job(dict(base, priority="below_normal"))
    assert not seen[0] & detached.BELOW_NORMAL_PRIORITY_CLASS
    assert seen[1] & detached.BELOW_NORMAL_PRIORITY_CLASS
    assert detached.set_normal_priority(0, None) is False          # never a bad pid
    assert detached.set_normal_priority(os.getpid(), 1.0) is False  # wrong creation time: untouched


def test_kq_shows_warm_jobs_and_the_cancel_note(tmp_path, expts, lib):
    from waxx.util.device_state import kq
    q, a, b, wp = _warm_agent_job(tmp_path, expts, lib)
    listed = q.list()
    view = next(j for j in listed["jobs"] if j["id"] == b)
    row = next(r for r in listed["rows"] if r["id"] == b)
    assert kq.warm_text(view, row) == "ready"
    assert "ready" in kq._note(view, row)
    assert kq._note(dict(view, waiting=""), dict(row, waiting="")) == "ready"
    lines = kq._warm_lines(q.describe({"id": b})["job"])
    assert lines[0].startswith("  warm: ready (attempt 1)") and "py_pid 9001" in lines[1]
    assert "next #%d ready" % b in kq.queue_line(q.info())
