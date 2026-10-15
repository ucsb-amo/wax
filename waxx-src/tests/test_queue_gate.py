"""waxx.util.device_state.queue_gate -- the experiment process's side of the
run queue's GO gate.  The queue is a scripted fake transport, or the real
RunQueue.warm_request called in-process (the handshake end to end); never a
beacon, never UDP 50099, never a real monitor server.  The clock and sleep are
replaced; nothing waits for real."""
import os
import sys

import pytest

from waxx.util.device_state import queue_gate as qg


class Script:
    """A transport answering from a list of replies (the last one repeats);
    ``None`` = no answer."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []

    def request(self, obj, timeout=None, attempts=None):
        self.sent.append(dict(obj))
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0] if self.replies else None


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    qg._reset_for_tests()
    t = {"now": 0.0}
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t["now"] += s

    monkeypatch.setattr(qg, "_monotonic", lambda: t["now"])
    monkeypatch.setattr(qg, "_sleep", sleep)
    monkeypatch.setattr(qg, "_raise_own_priority", lambda: True)
    monkeypatch.delenv(qg.WARM_ENV, raising=False)
    monkeypatch.setattr(qg, "_transport", lambda: qg._state["transport"])  # never discovery
    yield {"t": t, "sleeps": sleeps}
    qg._reset_for_tests()


def warm(monkeypatch, token="0123456789abcdef", job="7:abc123"):
    monkeypatch.setenv(qg.WARM_ENV, token)
    monkeypatch.setenv(qg.JOB_ENV, job)


def test_not_warm_everything_is_a_no_op(monkeypatch):
    script = Script({"status": "ok", "abort": "x"})
    qg.set_transport(script)
    assert qg.warm_token() is None and not qg.is_warm()
    assert qg.go_passed() and qg.go_info() == {}
    qg.hello("x")
    qg.ensure_go()
    called = []
    qg.register_claim(lambda: called.append(1))
    qg.run_claim()
    qg.run_claim()
    assert script.sent == [] and called == [1]


def test_the_variable_is_popped_at_first_read_and_children_never_see_it(monkeypatch):
    warm(monkeypatch)
    assert qg.warm_token() == "0123456789abcdef"
    assert qg.WARM_ENV not in os.environ                  # a child process inherits nothing
    assert qg.JOB_ENV in os.environ                       # the job id stays (INIT_RUN sends it)
    monkeypatch.setenv(qg.WARM_ENV, "another")
    assert qg.warm_token() == "0123456789abcdef"          # cached for the process
    import subprocess
    os.environ.pop(qg.WARM_ENV, None)
    out = subprocess.run([sys.executable, "-c",
                          f"import os; print(os.environ.get('{qg.WARM_ENV}'))"],
                         capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "None"


def test_hello_ready_poll_go(monkeypatch, fresh, tmp_path, capsys):
    warm(monkeypatch)
    monkeypatch.setenv(qg.ROOTS_ENV, str(tmp_path))
    data = tmp_path / "trap.json"
    data.write_text("{}")
    script = Script({"status": "ok", "go": False, "why": "hello"},
                    {"status": "ok", "go": False, "why": "job 6 is running"},
                    None, {"status": "error", "msg": "busy"},
                    {"status": "ok", "go": False, "why": "job 6 is running"},
                    {"status": "ok", "go": True, "reset_count": 4})
    qg.set_transport(script)
    qg.hello("rabi")
    with open(data) as f:                                 # opened for reading during prepare
        f.read()
    with open(tmp_path / "out.txt", "w") as f:            # a write is not in the manifest
        f.write("x")
    assert not qg.go_passed()
    qg.ensure_go()
    assert qg.go_passed()
    info = qg.go_info()
    assert info["job"] == "7:abc123" and info["reset_count"] == 4 and info["given_at"] > 0
    stages = [o["stage"] for o in script.sent]
    assert stages == ["warming", "ready", "poll", "poll", "poll", "poll"]
    first = script.sent[0]
    assert first["job"] == "7:abc123" and first["warm"] == "0123456789abcdef"
    assert first["pid"] == os.getpid() and first["expt"] == "rabi"
    manifest = script.sent[1]["manifest"]
    paths = [os.path.normcase(m[0]) for m in manifest]
    assert os.path.normcase(str(data)) in paths
    assert os.path.normcase(str(tmp_path / "out.txt")) not in paths
    assert all(isinstance(m[1], int) and isinstance(m[2], int) for m in manifest)
    assert fresh["sleeps"] and all(s == qg.POLL_S for s in fresh["sleeps"])
    out = capsys.readouterr().out
    assert "waiting for GO: job 6 is running" in out and "GO from the run queue" in out
    n = len(script.sent)
    qg.ensure_go()                                        # idempotent
    assert len(script.sent) == n


def test_an_abort_stands_down_with_exit_78(monkeypatch, capsys):
    warm(monkeypatch)
    qg.set_transport(Script({"status": "ok", "go": False}, {"status": "ok", "go": False},
                            {"status": "ok", "abort": "no longer next (#9 is)"}))
    qg.hello()
    with pytest.raises(SystemExit) as exc:
        qg.ensure_go()
    assert exc.value.code == qg.EXIT_WARM_ABORTED == 78
    assert "stood down before GO (nothing was claimed): no longer next" in capsys.readouterr().out
    assert not qg.go_passed()


def test_an_abort_at_hello_stands_down_at_once(monkeypatch):
    warm(monkeypatch)
    qg.set_transport(Script({"status": "ok", "abort": "discarded"}))
    with pytest.raises(SystemExit) as exc:
        qg.hello()
    assert exc.value.code == 78


def test_no_answer_for_warm_lost_s_exits_as_aborted(monkeypatch, fresh):
    warm(monkeypatch)
    script = Script(None)
    qg.set_transport(script)
    with pytest.raises(SystemExit) as exc:
        qg.ensure_go()
    assert exc.value.code == 78
    assert fresh["t"]["now"] > qg.WARM_LOST_S
    assert all(o["stage"] in ("warming", "ready") for o in script.sent)


def test_errors_keep_it_waiting_and_an_answer_restarts_the_lost_clock(monkeypatch, fresh):
    warm(monkeypatch)
    replies = [None] * 2000 + [{"status": "ok", "go": False}] + [None] * 2000 + \
        [{"status": "ok", "go": True, "reset_count": None}]
    qg.set_transport(Script({"status": "ok", "go": False}, *replies))
    qg.ensure_go()                                        # ~400 s + ~400 s: never 600 s silent
    assert qg.go_passed() and qg.go_info()["reset_count"] is None


def test_resend_ready_sends_the_manifest_again(monkeypatch):
    warm(monkeypatch)
    script = Script({"status": "ok", "go": False}, {"status": "ok", "go": False},
                    {"status": "ok", "go": False, "resend_ready": True},
                    {"status": "ok", "go": True, "reset_count": 0})
    qg.set_transport(script)
    qg.ensure_go()
    assert [o["stage"] for o in script.sent] == ["warming", "ready", "poll", "ready"]
    assert script.sent[1]["manifest"] == script.sent[3]["manifest"]


def test_the_claim_runs_once():
    calls = []
    qg.register_claim(lambda: calls.append("a"))
    qg.register_claim(lambda: calls.append("b"))          # the last one wins
    qg.run_claim()
    qg.run_claim()
    assert calls == ["b"]


def test_the_manifest_takes_modules_under_the_roots_and_skips_temp_files(tmp_path, monkeypatch):
    import tempfile
    mod_dir = tmp_path / "roots"
    mod_dir.mkdir()
    (mod_dir / "qg_probe_mod.py").write_text("X = 1\n")
    sys.path.insert(0, str(mod_dir))
    try:
        import qg_probe_mod  # noqa: F401
        tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".tmp")
        tmp.write("x")
        tmp.close()
        qg._opened.add(tmp.name)
        qg._opened.add(str(tmp_path / "gone.npy"))       # deleted since: left out
        manifest = qg.build_manifest([str(mod_dir)])
        paths = [os.path.normcase(m[0]) for m in manifest]
        assert os.path.normcase(str(mod_dir / "qg_probe_mod.py")) in paths
        assert os.path.normcase(tmp.name) not in paths
        assert not any(p.endswith("gone.npy") for p in paths)
        assert manifest == sorted(manifest, key=lambda m: os.path.normcase(
            os.path.abspath(m[0])))
        os.unlink(tmp.name)
    finally:
        sys.path.remove(str(mod_dir))
        sys.modules.pop("qg_probe_mod", None)


def test_queue_roots_are_the_queues(monkeypatch):
    from waxx.util.device_state import run_queue
    monkeypatch.setenv(qg.ROOTS_ENV, os.pathsep.join(["A", "B"]))
    assert qg.queue_roots() == run_queue.default_roots() == ["A", "B"]
    monkeypatch.delenv(qg.ROOTS_ENV)
    monkeypatch.setenv("code", "C:/somewhere")
    assert run_queue.default_roots() == ["C:/somewhere", r"C:\lab\skynet_log"]


# --- end to end against the real queue (in-process) -------------------------------------------

class QueueTransport:
    """The run queue's warm request, called in-process as the monitor server
    would answer it (no network)."""

    def __init__(self, q):
        self.q = q
        self.sent = []

    def request(self, obj, timeout=None, attempts=None):
        self.sent.append(obj)
        assert obj["type"] == "run_queue" and obj["action"] == "warm"
        return self.q.warm_request(obj)


def test_the_handshake_end_to_end(tmp_path, monkeypatch, fresh):
    from test_run_queue_warm import end_slot, slot_running, wq
    from test_run_queue import job, submit
    expts = tmp_path / "experiments"
    expts.mkdir()
    for name in ("rabi", "tof", "monitor"):
        (expts / f"{name}.py").write_text(f'"""{name}."""\nclass {name}: pass\n')
    q = wq(tmp_path, expts)
    slot_running(q, expts)
    b = submit(q, expts, "tof")
    q.tick()
    env = q.spawner.calls[-1]["env"]
    monkeypatch.setenv(qg.WARM_ENV, env["WAXX_QUEUE_WARM"])
    monkeypatch.setenv(qg.JOB_ENV, env["WAXX_QUEUE_JOB"])
    monkeypatch.setenv(qg.ROOTS_ENV, str(tmp_path))
    transport = QueueTransport(q)
    qg.set_transport(transport)
    qg.hello("tof")
    polls = {"n": 0}
    real_sleep = qg._sleep

    def sleep(s):                                         # the slot job ends after a few polls
        real_sleep(s)
        polls["n"] += 1
        if polls["n"] == 3:
            end_slot(q)
    monkeypatch.setattr(qg, "_sleep", sleep)
    qg.ensure_go()
    assert qg.go_passed() and qg.go_info()["reset_count"] == 0
    assert job(q, b)["state"] == "running"
    assert job(q, b)["warm"]["go"]["delivered"] is not None
    assert job(q, b)["warm"]["py_pid"] == os.getpid()
