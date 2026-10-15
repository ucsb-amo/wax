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


def _probe_module(folder, name, text="X = 1\n"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.py").write_text(text)
    sys.path.insert(0, str(folder))
    try:
        __import__(name)
    finally:
        sys.path.remove(str(folder))
    return folder / f"{name}.py"


def _norm_paths(manifest):
    return [os.path.normcase(m[0]) for m in manifest]


def test_the_manifest_takes_modules_under_the_roots_and_skips_temp_files(tmp_path, monkeypatch):
    import tempfile
    mod_dir = tmp_path / "roots"
    try:
        mod = _probe_module(mod_dir, "qg_probe_mod")
        tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".tmp")
        tmp.write("x")
        tmp.close()
        monkeypatch.setenv(qg.ROOTS_ENV, str(mod_dir))
        qg._start_recording()
        with open(tmp.name) as f:                         # temp, outside the roots
            f.read()
        gone = mod_dir / "gone.npy"
        gone.write_text("x")
        with open(gone) as f:
            f.read()
        gone.unlink()                                     # deleted since: kept at its load stat
        manifest = qg.build_manifest([str(mod_dir)])
        paths = _norm_paths(manifest)
        assert os.path.normcase(str(mod)) in paths
        assert os.path.normcase(tmp.name) not in paths
        assert manifest == sorted(manifest, key=lambda m: os.path.normcase(
            os.path.abspath(m[0])))
        assert qg.changed_since_load(manifest).endswith("gone.npy (gone)")
        os.unlink(tmp.name)
    finally:
        sys.modules.pop("qg_probe_mod", None)


def test_the_manifest_leaves_out_the_venv_site_packages_and_the_interpreter(tmp_path,
                                                                             monkeypatch):
    """S5: %code% holds .venv -- a manifest of a real warm process was 625 of
    879 files from it.  Only files under the roots and outside sys.prefix /
    sys.base_prefix / any site-packages / any .venv folder are kept."""
    code = tmp_path / "code"
    names = {"mine": code / "kexp" / "base",
             "venvmod": code / ".venv" / "Lib" / "site-packages" / "pkgx",
             "sitemod": code / "vendor" / "site-packages",
             "prefixmod": code / "pyprefix" / "Lib",
             "outside": tmp_path / "elsewhere"}
    monkeypatch.setattr(sys, "prefix", str(code / "pyprefix"))
    try:
        files = {k: _probe_module(folder, f"qg_s5_{k}") for k, folder in names.items()}
        manifest = qg.build_manifest([str(code)])
        paths = _norm_paths(manifest)
        assert os.path.normcase(str(files["mine"])) in paths
        for k in ("venvmod", "sitemod", "prefixmod", "outside"):
            assert os.path.normcase(str(files[k])) not in paths, k
        # opened files follow the same rule
        monkeypatch.setenv(qg.ROOTS_ENV, str(code))
        qg._start_recording()
        data = names["venvmod"] / "data.json"
        data.write_text("{}")
        mine = names["mine"] / "trap.json"
        mine.write_text("{}")
        for p in (data, mine):
            with open(p) as f:
                f.read()
        paths = _norm_paths(qg.build_manifest([str(code)]))
        assert os.path.normcase(str(mine)) in paths
        assert os.path.normcase(str(data)) not in paths
    finally:
        for k in names:
            sys.modules.pop(f"qg_s5_{k}", None)


def test_network_paths_are_never_in_a_manifest_and_never_stat_ed(tmp_path, monkeypatch):
    assert qg.is_network_path(r"\\fileserver\share\x.py")
    assert qg.is_network_path("//fileserver/share/x.py")
    qg._remote_drives["Q"] = True
    assert qg.is_network_path(r"Q:\data\roi.xlsx")
    drive = os.path.splitdrive(str(tmp_path))[0]
    if drive[1:] == ":":
        qg._remote_drives[drive[0].upper()] = False
        assert not qg.is_network_path(str(tmp_path))
        try:
            mod = _probe_module(tmp_path / "net", "qg_net_mod")
            stats = []
            real_stat = os.stat
            monkeypatch.setattr(qg.os, "stat", lambda p, *a, **k: (stats.append(str(p)),
                                                                     real_stat(p, *a, **k))[1])
            qg._remote_drives[drive[0].upper()] = True    # as if the folder were a share
            assert qg.build_manifest([str(tmp_path / "net")]) == []
            assert not any(os.path.normcase(s) == os.path.normcase(str(mod)) for s in stats)
        finally:
            sys.modules.pop("qg_net_mod", None)


def test_a_file_is_recorded_as_it_was_when_loaded(tmp_path, monkeypatch):
    """S3: the manifest holds each file's stat at its load, not at "ready":
    a file edited between its import and "ready" shows as changed."""
    monkeypatch.setenv(qg.ROOTS_ENV, str(tmp_path))
    qg._start_recording()                                 # (a warm process: at import)
    try:
        mod = _probe_module(tmp_path / "pkg", "qg_s3_mod", "X = 1\n")
        at_load = os.stat(mod)
        os.utime(mod, ns=(at_load.st_atime_ns, at_load.st_mtime_ns + 10**9))
        mod.write_text("X = 2  # edited after the import\n")
        manifest = qg.build_manifest()
        entry = [m for m in manifest if os.path.normcase(m[0]) == os.path.normcase(str(mod))]
        assert entry == [[os.path.abspath(str(mod)), at_load.st_mtime_ns, at_load.st_size]]
        assert os.path.normcase(qg.changed_since_load(manifest)) == os.path.normcase(
            os.path.abspath(str(mod)))
    finally:
        sys.modules.pop("qg_s3_mod", None)


def test_a_pyc_open_records_its_source(tmp_path, monkeypatch):
    """A module imported from its cached .pyc is opened as the .pyc only: the
    hook records the source's stat (the file that matters), not the .pyc."""
    monkeypatch.setenv(qg.ROOTS_ENV, str(tmp_path))
    qg._start_recording()
    src = tmp_path / "pkg" / "cached.py"
    pyc = tmp_path / "pkg" / "__pycache__" / "cached.cpython-313.pyc"
    pyc.parent.mkdir(parents=True)
    src.write_text("X = 1\n")
    pyc.write_bytes(b"\0" * 32)
    qg._audit("open", (str(pyc), "rb", 0))
    st = os.stat(src)
    assert qg._loaded[os.path.normcase(os.path.abspath(str(src)))][1:] == (st.st_mtime_ns,
                                                                           st.st_size)
    assert os.path.normcase(os.path.abspath(str(pyc))) not in qg._loaded


def test_ready_stands_down_when_a_loaded_file_changed(monkeypatch, tmp_path, capsys):
    """S3: a file changed between its load and "ready" -> exit 78 before
    "ready" is sent, naming the file (the queue warms the job again)."""
    warm(monkeypatch)
    monkeypatch.setenv(qg.ROOTS_ENV, str(tmp_path))
    script = Script({"status": "ok", "go": True, "reset_count": 0})
    qg.set_transport(script)
    qg.hello("rabi")
    data = tmp_path / "trap.json"
    data.write_text("{}")
    with open(data) as f:                                 # read during prepare
        f.read()
    data.write_text('{"edited": "after it was read"}')
    with pytest.raises(SystemExit) as e:
        qg.ensure_go()
    assert e.value.code == qg.EXIT_WARM_ABORTED
    assert [o["stage"] for o in script.sent] == ["warming"]     # never said "ready"
    out = capsys.readouterr().out
    assert "a file changed on disk after this process loaded it" in out
    assert "trap.json" in out
    assert not qg.go_passed()


def test_the_hook_goes_in_at_import_in_a_warm_process_and_the_variable_stays(tmp_path):
    """A fresh interpreter, warm: importing waxx installs the hook and stats
    the modules already loaded (the residual window); the variable is still
    there for warm_token() to pop.  Cold: no hook."""
    import subprocess
    import textwrap
    pre = tmp_path / "pre"
    pre.mkdir()
    (pre / "qg_pre_mod.py").write_text("X = 1\n")
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(pre)!r})
        import qg_pre_mod                      # loaded before waxx: the residual window
        import waxx
        qg = sys.modules.get("waxx.util.device_state.queue_gate")
        if qg is None:
            print("no-gate"); raise SystemExit
        n = os.path.normcase(os.path.abspath(qg_pre_mod.__file__))
        print(qg._state["hook"], qg._state["recording"], "WAXX_QUEUE_WARM" in os.environ,
              n in qg._loaded)
        print(qg.warm_token(), "WAXX_QUEUE_WARM" in os.environ)
    """)
    env = dict(os.environ, WAXX_RUN_QUEUE_ROOTS=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    env.pop("WAXX_QUEUE_WARM", None)
    cold = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=120)
    assert cold.stdout.split() == ["no-gate"], cold.stderr
    env["WAXX_QUEUE_WARM"] = "tok123"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.stdout.split() == ["True", "True", "True", "True", "tok123", "False"], out.stderr


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
