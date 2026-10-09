"""The kq command line (waxx.util.device_state.kq): argument parsing, every
command against a fake monitor server in front of a real RunQueue, exit
codes, the Ctrl-C paths (KeyboardInterrupt raised from the fake), the refusal
when no queue is beaconing, and ASCII-only output: every test goes through
``kq()``, which checks that everything kq printed is ASCII.  Nothing is
launched and no socket is opened."""
import io
import json
import os
import time

import pytest

from test_run_queue import make_queue
from test_run_queue_client import FakeServer, Script, run_steps
from waxx.util.device_state import kq as kqmod
from waxx.util.device_state.run_queue_client import NoRunQueue, RunQueueClient


@pytest.fixture
def expts(tmp_path):
    folder = tmp_path / "experiments"
    folder.mkdir()
    for name in ("rabi", "tof", "monitor"):
        (folder / f"{name}.py").write_text(f'"""{name}."""\nclass {name}: pass\n')
    return folder


@pytest.fixture
def q(tmp_path, expts):
    return make_queue(tmp_path, expts)


@pytest.fixture
def server(q):
    return FakeServer(q)


@pytest.fixture(autouse=True)
def person(monkeypatch):
    monkeypatch.delenv("WAXX_OWNER", raising=False)


class Result:
    def __init__(self, code, out, err, asked):
        self.code, self.out, self.err, self.asked = code, out, err, asked


def kq(server, *argv, answer="n", tty=True):
    """Run kq; check that all it printed is ASCII."""
    out, err, asked = io.StringIO(), io.StringIO(), []

    def ask(prompt):
        asked.append(prompt)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def factory():
        if server is None:
            raise NoRunQueue("no run queue (monitor server) is beaconing (not found)")
        return RunQueueClient(server, by="jp@kong", sleep=lambda s: None, wait_poll_s=0.0)

    code = kqmod.main(list(argv), client_factory=factory, out=out, err=err, ask=ask,
                      isatty=lambda: tty)
    r = Result(code, out.getvalue(), err.getvalue(), asked)
    for text in [r.out, r.err] + asked:
        assert text.isascii(), text
    return r


def sent(server, action):
    return [o for o, _ in server.requests if o.get("action") == action]


# --- kq run ---------------------------------------------------------------------------------

def test_run_follows_the_job_and_exits_0_when_it_saved(server, q, expts):
    Script(server, run_steps(q))
    r = kq(server, "run", str(expts / "rabi.py"))
    assert r.code == 0, r.err
    lines = r.out.splitlines()
    assert lines[0] == "[kq] job 1 queued (position 1; launching)"
    assert lines[1].startswith("-- run queue ") and lines[2].startswith("$ ")
    assert lines[3:6] == ["Run ID: 101", "shot 1/2", "shot 2/2"]
    assert lines[-1] == "[kq] job 1 (rabi) saved (run 101)"
    s = sent(server, "submit")[0]
    assert s["owner"] == "person" and "priority" not in s     # the queue's default
    assert s["by"] == "jp@kong"


def test_run_exits_with_the_experiments_own_code(server, q, expts):
    Script(server, run_steps(q, outcome="discarded", code=2))
    r = kq(server, "run", str(expts / "rabi.py"))
    assert r.code == 2 and "[kq] job 1 (rabi) FAILED (run 101, exit code 2)" in r.err


def test_a_failure_with_exit_code_0_exits_1(server, q, expts):
    Script(server, run_steps(q, outcome="discarded", code=0))
    r = kq(server, "run", str(expts / "rabi.py"))
    assert r.code == 1 and "FAILED" in r.err


def test_a_job_skipped_at_launch_exits_3(server, q, expts):
    def edit_and_launch():
        (expts / "rabi.py").write_text("changed\n")
        q.tick()

    Script(server, [edit_and_launch])
    r = kq(server, "run", str(expts / "rabi.py"))
    assert r.code == 3 and "SKIPPED" in r.err and "source changed" in r.err


def test_run_options_reach_the_request(server, q, expts, monkeypatch):
    monkeypatch.setenv("WAXX_OWNER", "agent")
    r = kq(server, "submit", str(expts / "rabi.py"), "--label", "r1", "--priority", "4",
           "--repeat", "2", "--chain", "cal", "--no-stop-on-failure", "--no-write-back",
           "--allow-drift", "--at", str(time.time() + 3600), "--", "-a", "x=1")
    assert r.code == 0, r.err
    s = sent(server, "submit")[0]
    assert s["owner"] == "agent" and s["priority"] == 4 and s["repeat"] == 2
    assert s["chain"] == "cal" and s["stop_on_failure"] is False and s["write_back"] is False
    assert s["allow_drift"] is True and s["argv"] == ["-a", "x=1"] and s["due"] > time.time()
    assert s["label"] == "r1"
    assert "jobs 1-2 (chain cal) queued: r1, owner agent" in r.out
    assert "kq tail 1 -f" in r.out
    assert sent(server, "tail") == []                         # submit does not follow
    r = kq(server, "run", str(expts / "rabi.py"), "--detach", "--after", "1", "2")
    assert r.code == 0 and sent(server, "submit")[-1]["after"] == [1, 2]


def test_experiment_arguments_after_the_file_as_in_artiq_run(server, expts):
    r = kq(server, "submit", str(expts / "rabi.py"), "n=3", "m=x", "--label", "r", "--",
           "-c", "Rabi")
    assert r.code == 0, r.err
    assert sent(server, "submit")[-1]["argv"] == ["n=3", "m=x", "-c", "Rabi"]
    assert kq(server, "submit", str(expts / "rabi.py"), "-c", "Rabi").code == 2
    r = kq(server, "submit", str(expts / "rabi.py"), "a=1", "--label", "x", "b=2")
    assert r.code == 0 and sent(server, "submit")[-1]["argv"] == ["a=1", "b=2"]
    assert sent(server, "submit")[-1]["label"] == "x"
    r = kq(server, "submit", str(expts / "rabi.py"), "a=1", "--label", "x", "b=2", "-q")
    assert r.code == 2 and "experiment options go after --" in r.err
    assert kq(server, "list", "stray").code == 2
    r = kq(server, "submit", str(expts / "rabi.py"), "x=50%")
    assert r.code == 6 and "not allowed" in r.err


def test_agent_flag_and_environment(server, expts, monkeypatch):
    kq(server, "submit", str(expts / "rabi.py"), "--agent")
    assert sent(server, "submit")[-1]["owner"] == "agent"
    monkeypatch.setenv("WAXX_OWNER", "person")
    kq(server, "submit", str(expts / "rabi.py"))
    assert sent(server, "submit")[-1]["owner"] == "person"


def test_parse_at():
    now = time.mktime((2026, 10, 9, 14, 0, 0, 0, 0, -1))
    assert kqmod.parse_at("15:30", now) == now + 5400
    assert kqmod.parse_at("13:00", now) == pytest.approx(now + 23 * 3600, abs=3600)
    assert kqmod.parse_at("1800000000") == 1.8e9
    for bad in ("25:00", "noon", "1430", "86400"):
        with pytest.raises(ValueError):
            kqmod.parse_at(bad, now)


def test_parse_at_tomorrow_is_a_calendar_day_across_a_dst_change():
    import datetime as dt
    # the night before the US spring-forward (2027-03-14): now + 86400 s
    # would land on the 15th; the calendar says the 14th
    now = dt.datetime(2027, 3, 13, 23, 30).timestamp()
    assert kqmod.parse_at("23:00", now) == dt.datetime(2027, 3, 14, 23, 0).timestamp()
    now = dt.datetime(2026, 10, 31, 23, 30).timestamp()          # before fall-back
    assert kqmod.parse_at("23:00", now) == dt.datetime(2026, 11, 1, 23, 0).timestamp()


def test_a_bare_small_number_for_at_is_refused(server, expts):
    r = kq(server, "submit", str(expts / "rabi.py"), "--at", "1430")
    assert r.code == 2 and "write HH:MM" in r.err and sent(server, "submit") == []


def test_a_bad_at_is_a_usage_error(server, expts):
    r = kq(server, "run", str(expts / "rabi.py"), "--at", "noon")
    assert r.code == 2 and "--at takes" in r.err and sent(server, "submit") == []


def test_a_repeat_is_followed_job_by_job(server, q, expts):
    Script(server, run_steps(q, run_id=101) + run_steps(q, run_id=102))
    r = kq(server, "run", str(expts / "rabi.py"), "--repeat", "2")
    assert r.code == 0, r.err
    assert "[kq] job 1 (1 of 2)" in r.out and "[kq] job 2 (2 of 2)" in r.out
    assert r.out.count("Run ID: 10") == 2 and "[kq] job 2 (rabi) saved (run 102)" in r.out


def test_the_queue_refusing_exits_6(server, expts):
    r = kq(server, "run", str(expts / "nothing.py"))
    assert r.code == 6 and "no such file" in r.err and r.out == ""


def test_no_queue_beaconing_exits_4_and_runs_nothing(expts):
    r = kq(None, "run", str(expts / "rabi.py"))
    assert r.code == 4 and r.out == ""
    assert kqmod.NO_QUEUE_TEXT in r.err and "artiq_run --device-db %db%" in r.err


def test_an_older_monitor_server_exits_4(server, expts):
    server.known = False
    assert kq(server, "run", str(expts / "rabi.py")).code == 4
    assert kq(server, "status").code == 4


# --- Ctrl-C -----------------------------------------------------------------------------------

def _interrupt_on(server, action):
    def on_request(obj):
        if obj.get("action") == action:
            raise KeyboardInterrupt
    server.on_request = on_request


def test_ctrl_c_during_the_submit_says_the_job_may_be_queued(server, q, expts):
    _interrupt_on(server, "submit")
    r = kq(server, "run", str(expts / "rabi.py"))
    assert r.code == 130 and "the job may be queued -- check kq list" in r.err


def test_ctrl_c_right_after_the_submit_cancels_the_job_and_names_it(server, q, expts):
    _interrupt_on(server, "list")                 # the position lookup after the submit
    r = kq(server, "run", str(expts / "rabi.py"), "--repeat", "2")
    assert r.code == 130 and "cancelled jobs 1, 2 (not started)" in r.out
    assert [q.describe({"id": i})["job"]["state"] for i in (1, 2)] == ["cancelled"] * 2
    assert all(c.get("queued_only") for c in sent(server, "cancel"))


def test_ctrl_c_between_two_jobs_of_a_repeat(server, q, expts):
    q._gap_s = 1e9                                # job 2 stays queued after job 1 ends
    steps = run_steps(q, run_id=101)
    Script(server, steps)
    calls = {"n": 0}
    inner = server.on_request

    def on_request(obj):
        inner(obj)
        if obj.get("action") == "describe" and q.describe({"id": 1})["job"]["state"] == "saved":
            calls["n"] += 1
            if calls["n"] == 1:                   # the final describe of job 1
                raise KeyboardInterrupt

    server.on_request = on_request
    r = kq(server, "run", str(expts / "rabi.py"), "--repeat", "2", tty=False)
    assert r.code == 130
    assert q.describe({"id": 1})["job"]["state"] == "saved"
    assert q.describe({"id": 2})["job"]["state"] == "cancelled"
    assert "cancelled queued job 2" in r.out


def test_a_queue_error_inside_the_interrupt_still_exits_130(server, q, expts):
    def on_request(obj):
        if obj.get("action") == "list":
            server.silent = 10 ** 6               # the server goes quiet from the Ctrl-C on
            raise KeyboardInterrupt

    server.on_request = on_request
    r = kq(server, "run", str(expts / "rabi.py"))
    assert r.code == 130 and "did not answer" in r.err and "kq tail 1 -f" in r.out


def test_ctrl_c_while_queued_cancels_and_exits_130(server, q, expts):
    Script(server, [], interrupt_at={2})
    r = kq(server, "run", str(expts / "rabi.py"), "--repeat", "3")
    assert r.code == 130
    assert "cancelled jobs 1, 2, 3 (not started)" in r.out
    assert [q.describe({"id": i})["job"]["state"] for i in (1, 2, 3)] == ["cancelled"] * 3
    assert all(c.get("queued_only") for c in sent(server, "cancel"))
    assert r.asked == []


def test_ctrl_c_on_a_job_that_just_launched_never_aborts_without_asking(server, q, expts):
    def launch_then_interrupt():
        q.tick()                                  # it launches between the two requests
        raise KeyboardInterrupt

    steps = iter([launch_then_interrupt])
    server.on_request = lambda obj: next(steps)() if obj.get("action") == "tail" else None
    r = kq(server, "run", str(expts / "rabi.py"), tty=False)
    assert r.code == 130 and r.asked == []
    assert q.describe({"id": 1})["job"]["state"] == "running" and q.live.resets == []
    assert "left running" in r.out and "kq cancel 1" in r.out


def _interrupt_while_running(server, q):
    steps = run_steps(q)
    Script(server, steps[:2], interrupt_at={3})


def test_ctrl_c_while_running_answered_no_leaves_it_running(server, q, expts):
    _interrupt_while_running(server, q)
    r = kq(server, "run", str(expts / "rabi.py"), answer="n")
    assert r.code == 130 and len(r.asked) == 1
    assert "abort the run? it discards its data file [y/N]" in r.asked[0]
    assert q.describe({"id": 1})["job"]["state"] == "running" and q.live.resets == []
    assert sent(server, "cancel") == []
    assert "kq tail 1 -f" in r.out and "kq cancel 1" in r.out


def test_ctrl_c_at_the_prompt_is_no(server, q, expts):
    _interrupt_while_running(server, q)
    r = kq(server, "run", str(expts / "rabi.py"), answer=KeyboardInterrupt())
    assert r.code == 130 and sent(server, "cancel") == [] and q.live.resets == []


def test_ctrl_c_while_running_without_a_terminal_never_asks(server, q, expts):
    _interrupt_while_running(server, q)
    r = kq(server, "run", str(expts / "rabi.py"), tty=False)
    assert r.code == 130 and r.asked == [] and sent(server, "cancel") == []


def test_yes_at_the_prompt_sends_the_abort_and_cancels_the_rest(server, q, expts):
    def ends_aborted():
        proc = q.spawner.procs[-1]
        proc.write("RuntimeError: Acquisition for run 101 aborted.")
        q.live.end_run(101, "discarded")
        proc.code = 1
        q.tick()

    steps = run_steps(q)
    # the cancel only records the request; the queue's next tick sends the Abort
    Script(server, steps[:2] + [q.tick, ends_aborted], interrupt_at={3})
    r = kq(server, "run", str(expts / "rabi.py"), "--repeat", "2", answer="y")
    assert r.code == 3, (r.out, r.err)
    assert len(r.asked) == 1 and q.live.resets == [101]
    cancels = sent(server, "cancel")
    assert cancels[0]["id"] == 1 and not cancels[0].get("queued_only")
    assert cancels[0]["owner"] == "person"
    assert cancels[1]["id"] == 2 and cancels[1]["queued_only"]
    assert ("[kq] abort requested for job 1 (rabi) (run 101) by jp@kong; waiting for the run "
            "to end (Ctrl-C again leaves it)") in r.out
    assert r.out.count("abort requested") == 1
    assert "cancelled queued job 2" in r.out
    assert "CANCELLED" in r.err
    assert q.describe({"id": 2})["job"]["state"] == "cancelled"


# --- the other commands --------------------------------------------------------------------

def test_list_show_and_status(server, q, expts):
    kq(server, "submit", str(expts / "rabi.py"), "--label", "r1", "--priority", "1",
       "--no-write-back")
    kq(server, "submit", str(expts / "tof.py"), "--priority", "3")
    assert ("write-back veto requested (honoured once the calibration branch lands)"
            in kq(server, "show", "1").out)
    assert "write-back" not in kq(server, "show", "2").out
    r = kq(server, "list")
    assert r.code == 0
    header, *rows, state = r.out.splitlines()
    assert header.split()[:5] == ["id", "state", "owner", "label", "run_id"]
    assert [row.split()[0] for row in rows] == ["1", "2"]
    assert state.startswith("queue: waiting") and "next: 2, 1" in state
    j = json.loads(kq(server, "list", "--json").out)
    assert [x["id"] for x in j["jobs"]] == [1, 2] and j["next"] == [2, 1]
    assert kq(server, "list", "--state", "saved").out.startswith("(no jobs)")
    r = kq(server, "show", "1")
    assert r.code == 0 and r.out.startswith("job 1 (r1): queued")
    assert "waiting: job 2 goes first" in r.out
    assert json.loads(kq(server, "show", "1", "--json").out)["job"]["id"] == 1
    r = kq(server, "status")                                   # waiting jobs: not busy
    assert r.code == 0 and r.out.startswith("queue free | queue: waiting")
    j = json.loads(kq(server, "status", "--json").out)
    assert j["queue_busy"] is False and "not machine occupancy" in j["note"]
    q.tick()                                                   # a job in the slot
    r = kq(server, "status")
    assert r.code == 5 and r.out.startswith("queue busy | queue: running")


def test_status_is_queue_state_only(server, q, expts):
    r = kq(server, "status")
    assert r.code == 0 and r.out.startswith("queue free | queue: idle")
    assert "for occupancy use occupancy.py (agents) or the dashboard" in r.out
    # a run announced outside the queue, or a run loop, is not the queue's state
    server.run_pending = {"run_id": 900, "expt": "someone.py"}
    server.run_loops = {"auto_tof": {"state": "running", "text": "x"}}
    r = kq(server, "status")
    assert r.code == 0 and "900" not in r.out and "auto_tof" not in r.out
    kq(server, "submit", str(expts / "rabi.py"), "--at", str(time.time() + 86400))
    assert kq(server, "status").code == 0                      # due tomorrow: not busy
    kq(server, "hold", "aligning", "optics")
    r = kq(server, "status")
    assert r.code == 5 and "person hold since" in r.out and "aligning optics" in r.out


def test_list_shows_the_alarm_and_a_hold(server, q, expts):
    kq(server, "submit", str(expts / "rabi.py"), "--agent")
    r = kq(server, "hold", "mine")
    assert r.code == 0 and "agents' jobs wait" in r.out
    assert sent(server, "hold")[0]["owner"] == "person"
    r = kq(server, "hold", "again")
    assert "already on" in r.out
    q.tick()                                                   # agent job held
    q._alarm = {"since": 0, "warned": 0, "job": 1, "waited_s": 720.0,
                "why": "liveOD is not reachable — a run could not save"}
    r = kq(server, "list")
    assert "ALARM: job 1 ready to start for 12 min" in r.out and "-- a run could not" in r.out
    assert "person hold since" in r.out
    r = kq(server, "release")
    assert r.code == 0 and "released" in r.out
    assert kq(server, "release").code == 6                     # none on


def test_pause_and_resume(server, q):
    r = kq(server, "pause", "--all", "--reason", "lunch")
    assert r.code == 0 and "all jobs paused by jp@kong (lunch)" in r.out
    assert sent(server, "pause")[0]["scope"] == "all"
    assert sent(server, "pause")[0]["owner"] == "person"
    assert kq(server, "resume", "--all").code == 0
    assert kq(server, "resume").code == 6                      # agent jobs not paused
    kq(server, "pause", "--agent")
    assert sent(server, "pause")[-1]["owner"] == "agent" and sent(server, "pause")[-1]["scope"] == "agent"


def test_tail_and_tail_follow(server, q, expts):
    kq(server, "submit", str(expts / "rabi.py"))
    r = kq(server, "tail", "1")
    assert r.code == 0 and "job 1 is queued: no output yet" in r.out
    Script(server, run_steps(q))
    r = kq(server, "tail", "1", "-f")
    assert r.code == 0 and "Run ID: 101" in r.out and "[kq] job 1 (rabi) saved (run 101)" in r.out
    r = kq(server, "tail", "1")
    assert r.out.splitlines()[2:] == ["Run ID: 101", "shot 1/2", "shot 2/2"]


def test_tail_follow_ctrl_c_leaves_the_job(server, q, expts):
    kq(server, "submit", str(expts / "rabi.py"))
    Script(server, run_steps(q)[:1], interrupt_at={2})
    r = kq(server, "tail", "1", "-f")
    assert r.code == 130 and "left as it is" in r.out and sent(server, "cancel") == []


def test_cancel(server, q, expts):
    kq(server, "submit", str(expts / "rabi.py"), "--repeat", "2")
    r = kq(server, "cancel", "2")
    assert r.code == 0 and "cancelled (it had not started)" in r.out
    assert sent(server, "cancel")[-1]["queued_only"] is True
    assert kq(server, "cancel", "2").code == 6                 # already cancelled
    q.tick()
    q.spawner.procs[-1].write("Run ID: 101")
    q.live.start_run(101)
    q.tick()
    r = kq(server, "cancel", "1", tty=False)
    assert r.code == 6 and "Pass --yes" in r.err and q.live.resets == []
    r = kq(server, "cancel", "1", answer="n")
    assert r.code == 0 and "nothing cancelled" in r.out and q.live.resets == []
    r = kq(server, "cancel", "1", "--yes", tty=False)
    assert r.code == 0 and "abort requested for job 1 (rabi) (run 101)" in r.out
    assert "waiting for the run to end" in r.out
    assert q.live.resets == []                                 # sent by the tick, not the request
    q.tick()
    assert q.live.resets == [101]
    assert sent(server, "cancel")[-1]["owner"] == "person"
    r = kq(server, "list")
    assert "abort requested by jp@kong" in r.out


def test_an_agent_cannot_abort_a_persons_job(server, q, expts):
    kq(server, "submit", str(expts / "rabi.py"))
    q.tick()
    r = kq(server, "cancel", "1", "--yes", "--agent")
    assert r.code == 6 and "a person's job: an agent may not cancel it" in r.err
    q.tick()
    assert q.live.resets == []


def test_an_agent_cannot_release_a_persons_hold_or_resume_a_persons_pause(server, q,
                                                                           monkeypatch):
    kq(server, "hold", "mine")
    kq(server, "pause", "--all")
    monkeypatch.setenv("WAXX_OWNER", "agent")
    r = kq(server, "release")
    assert r.code == 6 and "an agent may not release it" in r.err
    r = kq(server, "resume", "--all")
    assert r.code == 6 and sent(server, "resume")[-1]["owner"] == "agent"
    assert q.hold.info()["active"]
    monkeypatch.delenv("WAXX_OWNER")
    assert kq(server, "resume", "--all").code == 0 and kq(server, "release").code == 0


def test_tail_follow_reports_an_abort_someone_else_asked_for(server, q, expts):
    kq(server, "submit", str(expts / "rabi.py"))
    steps = run_steps(q, outcome="discarded", code=1)

    def cancel_elsewhere():
        q.cancel({"id": 1, "by": "jp@other", "owner": "person"})
        q.tick()

    Script(server, steps[:2] + [cancel_elsewhere, lambda: None, steps[2]])
    r = kq(server, "tail", "1", "-f")
    assert r.code == 3 and q.live.resets == [101]
    assert ("[kq] abort requested for job 1 (rabi) (run 101) by jp@other; waiting for the run "
            "to end") in r.out
    assert r.out.count("abort requested") == 1


@pytest.mark.parametrize("extra, words", [
    (["--", "folder" + "\\"], "ends in a backslash"),
    (["--at", "inf"], "finite"),
])
def test_new_submit_refusals_are_shown_as_the_server_words_them(server, expts, extra, words):
    r = kq(server, "submit", str(expts / "rabi.py"), *extra)
    assert r.code == 6 and words in r.err and r.out == ""


def test_a_file_outside_the_queues_roots_is_refused(server, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_elsewhere"
    outside.mkdir()
    (outside / "x.py").write_text("class x: pass\n")
    r = kq(server, "submit", str(outside / "x.py"))
    assert r.code == 6 and "outside the folders" in r.err


def test_the_hold_shows_its_owner_and_source(server, q):
    kq(server, "hold", "aligning")
    r = kq(server, "status")
    assert "(owner person, source request): aligning" in r.out
    q.hold._s.update(source="unreadable_file", by="monitor server",
                     reason="could not read the hold file (bad json)")
    r = kq(server, "status")
    assert r.code == 5 and "source unreadable_file" in r.out
    assert "source unreadable_file" in kq(server, "list").out


def test_usage_errors_and_help(server):
    assert kq(server).code == 2
    assert kq(server, "run").code == 2
    assert kq(server, "list", "--", "x").code == 2
    assert kq(server, "frobnicate").code == 2
    r = kq(server, "--help")
    assert r.code == 0 and "exit codes:" in r.out and "launching" in r.out
    assert "artiq_run --device-db %db%" in r.out
    assert server.requests == []


def test_the_server_not_answering_exits_6(server, expts):
    server.silent = 1
    r = kq(server, "submit", str(expts / "rabi.py"))
    assert r.code == 6 and "may have acted" in r.err


def test_experiment_lines_the_terminal_cannot_show_are_replaced(server, q, expts):
    Script(server, run_steps(q, lines=("5 µs → ok", "done")))
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="ascii", newline="\n")
    code = kqmod.main(["run", str(expts / "rabi.py")], out=out, err=io.StringIO(),
                      client_factory=lambda: RunQueueClient(server, by="jp@kong",
                                                            sleep=lambda s: None),
                      ask=lambda p: "n", isatty=lambda: False)
    out.flush()
    assert code == 0 and "5 ?s ? ok" in raw.getvalue().decode("ascii")


def test_ascii_text():
    assert kqmod.ascii_text("a — b → c ±5 µs ≥ 中") == \
        "a -- b -> c +/-5 us >= ?"


def test_module_help_and_epilog_are_ascii():
    assert kqmod.__doc__.isascii() and kqmod.EPILOG.isascii()
    assert os.path.basename(kqmod.__file__) == "kq.py"
