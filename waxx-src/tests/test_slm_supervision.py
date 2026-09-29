"""The SLM server under its supervisor (2026-09-28), offline.

* run_server.py (Meadowlark calls stubbed, 127.0.0.1): ``restart`` / ``shutdown``
  finish what is queued, answer, drop the rest and exit with the supervisor's
  codes; the pattern survives a restart through the state file; a failed
  start-up init exits the process; after a failed reinit no pattern is written;
  identity and supervision in ``status``; the heartbeat; one server per port;
* supervisor.py: the heartbeat hang check, the graceful stop, and an end-to-end
  run (a copy of run_server.py with a stub SDK, supervised on 127.0.0.1):
  started, its log read back (``log``), restarted with the pattern put back,
  shut down;
* the monitor server's service asks for a restart only while idle and only of a
  supervised server, and follows it to the new instance;
* the Device Control GUI's SLM pill offers it.

Nothing touches the lab network: every server listens on 127.0.0.1.
"""
import importlib.util
import itertools
import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import pytest

import waxx
from waxx.control.slm import slm_link
from waxx.control.slm.server.slm_protocol import control_line
from waxx.util.device_state import slm_reinit as sr
from waxx.util.device_state.slm_reinit import SlmReinitConfig, SlmReinitService
from waxx.util.supervise import kill_pid_tree

SLM_DIR = Path(waxx.__file__).parent / "control" / "slm" / "server"
_names = itertools.count()


def _wait(pred, timeout=5.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --- run_server.py with the SLM stubbed ------------------------------------------------

def _stub_module(host):
    class StubSLM:
        mask_type = "spot"

        def __init__(self):
            self.slm_initialized = False

        def initialize_slm(self):
            host.inits += 1
            if host.fail_init:
                raise RuntimeError(host.fail_init)
            self.slm_initialized = True

        def generate_mask(self, dimension, phase, center_x, center_y,
                          grating_spacing=10, angle_deg=0, mask=1):
            return (center_x, center_y, dimension)

        def fast_upload_to_slm(self, img):
            host.uploads.append(img)

        upload_to_slm = fast_upload_to_slm

        def release(self):
            host.releases += 1

    stub = types.ModuleType("slm_server")
    stub.SLM_server = StubSLM
    return stub


@pytest.fixture
def make_server(monkeypatch, tmp_path):
    """make_server(fail_init=None, supervised=True, start=True) -> host: a fresh
    run_server module on 127.0.0.1 with the SLM stubbed and its exit recorded."""
    made = []

    def make(fail_init=None, supervised=True, start=True, state_path=None):
        host = types.SimpleNamespace(uploads=[], inits=0, releases=0, fail_init=fail_init,
                                     exits=[], exited=threading.Event())
        monkeypatch.setitem(sys.modules, "slm_server", _stub_module(host))
        monkeypatch.syspath_prepend(str(SLM_DIR))
        if supervised:
            monkeypatch.setenv("SLM_SUPERVISED", "1")
            monkeypatch.setenv("SUPERVISOR_START_COUNT", "3")
            monkeypatch.setenv("SUPERVISOR_LAST_EXIT", "75 restart")
        else:
            monkeypatch.delenv("SLM_SUPERVISED", raising=False)
        spec = importlib.util.spec_from_file_location(f"run_server_sup_{next(_names)}",
                                                      SLM_DIR / "run_server.py")
        rs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rs)

        def fake_exit(code):
            host.exits.append(code)
            host.exited.set()

        rs._exit = fake_exit
        rs._state_path = str(state_path or (tmp_path / f"state_{len(made)}" / "last_pattern.json"))
        rs._heartbeat_path = str(tmp_path / f"heartbeat_{len(made)}.json")
        host.rs = rs
        if not start:
            made.append(host)
            return host
        rs._load_pattern()
        threading.Thread(target=rs.slm_worker, daemon=True).start()
        lsock = socket.socket()
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(4)
        host.lsock = lsock

        def accept_loop():
            while True:
                try:
                    conn, addr = lsock.accept()
                except OSError:
                    return
                rs.handle_client(conn, addr)

        threading.Thread(target=accept_loop, daemon=True).start()
        host.port = lsock.getsockname()[1]
        if not fail_init:
            assert _wait(lambda: len(host.uploads) == 1)
        made.append(host)
        return host

    yield make
    for h in made:
        if hasattr(h, "lsock"):
            h.lsock.close()
        h.rs.cmd_q.put(None)


def _ask(host, payload, until, control=False, **kw):
    kw.setdefault("total_s", 5.0)
    return slm_link.exchange("127.0.0.1", host.port, payload, until=until,
                             control=control, **kw)


def _status(host):
    return _ask(host, {"cmd": "status"}, ("ok", "error"), control=True)


def _apply(host, cx, cy, dimension=10):
    return _ask(host, {"mask": "spot", "center": [cx, cy], "dimension": dimension},
                ("applied", "error", "dropped"))


def _converse(port, lines, n_final, timeout=5.0):
    """Send raw lines on one connection; collect replies until ``n_final``
    replies with a final status (anything but "queued") have come."""
    replies = []
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.sendall("".join(lines).encode())
        s.settimeout(0.2)
        buf, t_end = b"", time.monotonic() + timeout
        while time.monotonic() < t_end:
            if sum(1 for r in replies if r.get("status") != "queued") >= n_final:
                break
            try:
                chunk = s.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                replies.append(json.loads(line))
    return replies


def test_restart_finishes_what_was_queued_then_exits_for_its_supervisor(make_server):
    host = make_server()
    lines = [json.dumps({"mask": "spot", "center": [11, 22], "dimension": 5, "seq": 1}) + "\n",
             control_line({"cmd": "restart", "seq": 2, "by": "test"}),
             json.dumps({"mask": "spot", "center": [33, 44], "dimension": 5, "seq": 3}) + "\n"]
    replies = _converse(host.port, lines, n_final=3)
    by_seq = {}
    for r in replies:
        by_seq.setdefault(r["seq"], []).append(r["status"])
    assert by_seq == {1: ["queued", "applied"], 2: ["queued", "restarting"],
                      3: ["queued", "dropped"]}
    assert host.exited.wait(2) and host.exits == [75]
    assert host.releases == 1
    assert host.uploads[-1] == (11, 22, 5)                     # (33, 44) was never written
    saved = json.load(open(host.rs._state_path, encoding="utf-8"))
    assert (saved["pattern"]["center_x"], saved["pattern"]["center_y"]) == (11, 22)


def test_restart_is_refused_without_a_supervisor(make_server):
    host = make_server(supervised=False)
    reply = _ask(host, {"cmd": "restart", "by": "test"}, ("queued", "error"), control=True)
    assert reply["status"] == "error" and "supervisor" in reply["error"]
    time.sleep(0.2)
    assert host.exits == []
    assert _status(host)["supervised"] is False


def test_shutdown_from_this_pc_exits_0(make_server):
    host = make_server()
    replies = _converse(host.port, [control_line({"cmd": "shutdown", "seq": 7})], n_final=1)
    assert [r["status"] for r in replies] == ["queued", "shutting_down"]
    assert host.exited.wait(2) and host.exits == [0]


def test_shutdown_from_another_pc_is_refused(make_server):
    host = make_server()
    sent = []
    replier = types.SimpleNamespace(send=sent.append)
    host.rs._handle_control({"cmd": "shutdown"}, 9, replier, local=False)
    assert sent[0]["status"] == "error" and "SLM PC itself" in sent[0]["error"]
    assert host.rs.cmd_q.qsize() == 0


def test_the_pattern_survives_a_restart(make_server, tmp_path):
    state = tmp_path / "shared" / "last_pattern.json"
    first = make_server(state_path=state)
    assert _apply(first, 100, 200, dimension=30)["status"] == "applied"
    second = make_server(state_path=state)                # the next process
    assert second.uploads[0] == (100, 200, 30)            # put back at start-up
    st = _status(second)
    assert st["pattern_source"].startswith("restored")
    assert (st["pattern"]["center_x"], st["pattern"]["center_y"]) == (100, 200)
    assert st["instance"] != _status(first)["instance"]


def test_an_unreadable_saved_pattern_is_not_used(make_server, tmp_path):
    state = tmp_path / "bad" / "last_pattern.json"
    state.parent.mkdir()
    state.write_text("{not json", encoding="utf-8")
    host = make_server(state_path=state)
    assert host.uploads[0] == (960, 600, 0)               # the default
    assert _status(host)["pattern_source"].startswith("default (saved pattern unreadable")


def test_a_failed_start_up_init_exits_the_process(make_server):
    host = make_server(fail_init="Load_lut returned 0")
    assert host.exited.wait(2) and host.exits == [71]
    assert host.uploads == []
    assert host.rs._worker_alive is False


def test_after_a_failed_reinit_no_pattern_is_written_until_one_succeeds(make_server):
    host = make_server()
    host.fail_init = "SDK said no"
    done = _ask(host, {"cmd": "reinit"}, ("reinit_done", "error"), control=True)
    assert done["status"] == "error"
    n = len(host.uploads)
    reply = _apply(host, 5, 6)
    assert reply["status"] == "error" and "not initialised" in reply["error"]
    assert len(host.uploads) == n                           # nothing written to a dead SDK
    st = _status(host)
    assert st["slm_ready"] is False and "SDK said no" in st["slm_not_ready"]
    host.fail_init = None
    assert _ask(host, {"cmd": "reinit"}, ("reinit_done", "error"), control=True)["status"] \
        == "reinit_done"
    assert _apply(host, 5, 6)["status"] == "applied"
    assert _status(host)["slm_ready"] is True


def test_status_says_who_the_server_is_and_that_it_is_supervised(make_server):
    host = make_server()
    _apply(host, 1, 2)
    st = _status(host)
    assert st["pid"] == os.getpid() and st["instance"] == host.rs.INSTANCE
    assert st["supervised"] is True and st["start_count"] == 3
    assert st["last_exit"] == "75 restart"
    assert {"restart", "shutdown", "reinit", "status", "seq"} <= set(st["capabilities"])
    assert st["slm_ready"] is True and st["applies"] == 1


def test_the_heartbeat_file(make_server):
    host = make_server()
    assert _wait(lambda: host.rs._current_task is None)   # start-up INIT finished
    host.rs._write_heartbeat()
    beat = json.load(open(host.rs._heartbeat_path, encoding="utf-8"))
    assert beat["pid"] == os.getpid() and beat["worker_alive"] is True
    assert beat["ppid"] == os.getppid()
    assert beat["task"] is None and beat["queue_len"] == 0 and beat["slm_ready"] is True
    assert abs(beat["t"] - time.time()) < 5


def test_one_server_per_port(make_server):
    host = make_server(start=False)
    rs = host.rs
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    try:
        with pytest.raises(rs.PortInUse):
            rs._open_listener("127.0.0.1", busy.getsockname()[1])
    finally:
        busy.close()
    free = rs._open_listener("127.0.0.1", _free_port())
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            assert rs._bind_mode == "exclusive"
    finally:
        free.close()


# --- supervisor.py -----------------------------------------------------------------------

@pytest.fixture
def supervisor_mod(monkeypatch):
    monkeypatch.syspath_prepend(str(SLM_DIR))
    spec = importlib.util.spec_from_file_location(f"slm_supervisor_{next(_names)}",
                                                  SLM_DIR / "supervisor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_heartbeat_hang_check(supervisor_mod, tmp_path):
    path = tmp_path / "hb.json"
    now = [1000.0]
    check = supervisor_mod.heartbeat_health(str(path), max_age_s=20, task_hang_s=120,
                                            init_hang_s=180, now=lambda: now[0])

    def beat(**kw):
        d = {"pid": 42, "t": 1000.0, "worker_alive": True, "task": None, "task_age_s": None}
        d.update(kw)
        path.write_text(json.dumps(d), encoding="utf-8")

    assert check(42)[0] is None                           # no file yet
    beat(pid=41)
    assert check(42)[0] is None                           # an earlier process's
    beat(pid=41, ppid=40)
    assert check(42)[0] is None
    beat()
    assert check(42) == (True, "")
    # Under a venv the supervisor starts a launcher, which starts the server:
    # the pid it holds is the server's parent.
    beat(pid=43, ppid=42)
    assert check(42) == (True, "")
    now[0] = 1030.0
    assert check(42)[0] is False                          # stale
    now[0] = 1000.0
    beat(worker_alive=False)
    assert check(42)[0] is False
    beat(task="APPLY", task_age_s=150.0)
    assert check(42)[0] is False
    beat(task="INIT", task_age_s=150.0)
    assert check(42)[0] is True                           # init gets longer
    beat(task="INIT", task_age_s=200.0)
    assert check(42)[0] is False


def test_the_graceful_stop_asks_for_a_shutdown(supervisor_mod):
    got = []
    lsock = socket.socket()
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(2)
    answers = iter(['{"seq": 1, "status": "queued"}\n', '{"seq": 1, "status": "error"}\n'])

    def serve():
        for _ in range(2):
            conn, _ = lsock.accept()
            with conn:
                got.append(conn.recv(4096).decode())
                conn.sendall(next(answers).encode())

    threading.Thread(target=serve, daemon=True).start()
    ask = supervisor_mod.graceful_shutdown("127.0.0.1", lsock.getsockname()[1], timeout_s=2.0)
    assert ask(1234) is True
    assert ask(1234) is False
    lsock.close()
    assert got[0].startswith("SLMCTL ") and json.loads(got[0][7:])["cmd"] == "shutdown"


STUB_SDK = '''
class SLM_server:
    """Stub Meadowlark SDK for the end-to-end test."""
    mask_type = "spot"
    def __init__(self):
        self.slm_initialized = False
    def initialize_slm(self):
        self.slm_initialized = True
    def generate_mask(self, dimension, phase, center_x, center_y, grating_spacing=10,
                      angle_deg=0, mask=1):
        return (center_x, center_y, dimension)
    def fast_upload_to_slm(self, img):
        pass
    upload_to_slm = fast_upload_to_slm
    def release(self):
        self.slm_initialized = False
'''


def _e2e_status(port):
    try:
        return slm_link.exchange("127.0.0.1", port, {"cmd": "status"}, until=("ok", "error"),
                                 control=True, connect_s=0.5, first_reply_s=2.0, total_s=2.0)
    except (OSError, slm_link.NoReply, slm_link.ReplyTimeout):
        return None


def test_the_supervisor_end_to_end(tmp_path, supervisor_mod):
    srv = tmp_path / "srv"
    srv.mkdir()
    for name in ("run_server.py", "slm_protocol.py"):
        shutil.copy(SLM_DIR / name, srv / name)
    (srv / "slm_server.py").write_text(textwrap.dedent(STUB_SDK), encoding="utf-8")
    state = tmp_path / "state"
    port = _free_port()
    log = open(tmp_path / "supervisor_console.txt", "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(SLM_DIR / "supervisor.py"), "--ip", "127.0.0.1",
         "--port", str(port), "--state-dir", str(state), "--server-dir", str(srv),
         "--health-grace-s", "2"],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        assert _wait(lambda: _e2e_status(port) is not None, 30), "no server came up"
        st1 = _e2e_status(port)
        assert st1["supervised"] is True and st1["start_count"] == 1
        # The hang check must take the server's heartbeat for its own child's.
        # Under a venv, sys.executable is a launcher and the server is the
        # launcher's child: on 2026-09-28 every server was killed as "hung"
        # 65 s after it started, because the pids never matched.
        check = supervisor_mod.heartbeat_health(str(state / "heartbeat.json"))
        server_pid = json.loads((state / "supervisor.json").read_text(encoding="utf-8"))[
            "server_pid"]
        assert _wait(lambda: check(server_pid)[0] is True, 10), check(server_pid)
        # The log: the supervisor's file, read back by the server; a log
        # request leaves no lines of its own in it.
        first = slm_link.exchange("127.0.0.1", port, {"cmd": "log", "cursor": None},
                                  until=("ok", "error"), control=True, total_s=5.0)
        assert first["status"] == "ok" and first["path"] == str(state / "logs")
        assert any("[supervisor] SLM server started" in line for line in first["lines"])
        again = slm_link.exchange("127.0.0.1", port, {"cmd": "log", "cursor": first["cursor"]},
                                  until=("ok", "error"), control=True, total_s=5.0)
        assert not any('"cmd":"log"' in line for line in first["lines"] + again["lines"])
        applied = slm_link.exchange("127.0.0.1", port,
                                    {"mask": "spot", "center": [123, 456], "dimension": 7},
                                    until=("applied", "error"), total_s=5.0)
        assert applied["status"] == "applied"

        queued = slm_link.exchange("127.0.0.1", port, {"cmd": "restart", "by": "test"},
                                   until=("queued", "error"), control=True, total_s=5.0)
        assert queued["status"] == "queued"

        def restarted():
            st = _e2e_status(port)
            return st is not None and st["instance"] != st1["instance"]

        assert _wait(restarted, 30), "no new server after the restart"
        st2 = _e2e_status(port)
        assert st2["start_count"] == 2 and st2["last_exit"] == "75 restart"
        assert st2["pattern_source"].startswith("restored")
        assert (st2["pattern"]["center_x"], st2["pattern"]["center_y"]) == (123, 456)
        assert st2["pid"] != st1["pid"]

        reply = slm_link.exchange("127.0.0.1", port, {"cmd": "shutdown", "by": "test"},
                                  until=("queued", "error"), control=True, total_s=5.0)
        assert reply["status"] == "queued"
        assert proc.wait(30) == 0                          # the supervisor stops too
        logs = list((state / "logs").glob("slm_server_*.log"))
        text = logs[0].read_text(encoding="utf-8")
        assert "restart requested" in text and "shut down on request" in text
        sup_status = json.loads((state / "supervisor.json").read_text(encoding="utf-8"))
        assert sup_status["state"] == "stopped" and sup_status["starts"] == 2
    finally:
        if proc.poll() is None:
            kill_pid_tree(proc.pid)
            proc.wait(10)
        log.close()


# --- the monitor server's service ------------------------------------------------------

class FakeSupervisedSlm:
    """The SLM server as the service sees it, with a supervisor behind it."""

    def __init__(self, supervised=True, capabilities=("seq", "status", "reinit", "restart",
                                                      "shutdown")):
        self.supervised = supervised
        self.capabilities = list(capabilities)
        self.instance = "aaaa0001"
        self.start_count = 1
        self.requests = []
        self.down_polls = 0             # polls unanswered after a restart
        self.stuck = False              # a restart that never brings a new server

    def __call__(self, host, port, payload, *, control, until, on_sent=None, **kw):
        cmd = payload["cmd"]
        self.requests.append(cmd)
        if cmd == "status" and self.down_polls > 0:
            self.down_polls -= 1
            if on_sent:
                on_sent()
            raise ConnectionRefusedError("refused")
        if cmd == "status" and self.stuck:
            if on_sent:
                on_sent()
            raise ConnectionRefusedError("refused")
        if on_sent:
            on_sent()
        if cmd == "status":
            return {"status": "ok", "reinit_due": False, "next_due_in_s": 1800.0,
                    "interval_s": 3600, "reinit_in_progress": False, "pattern_epoch": 0,
                    "pattern": {"center_x": 1}, "last_reinit_error": "",
                    "supervised": self.supervised, "capabilities": self.capabilities,
                    "instance": self.instance, "start_count": self.start_count,
                    "last_exit": None if self.start_count == 1 else "75 restart",
                    "slm_ready": True, "pattern_source": "restored (saved now)"}
        if cmd == "restart":
            self.instance = f"bbbb{self.start_count + 1:04d}"
            self.start_count += 1
            self.down_polls = 2
            return {"status": "queued"}
        raise AssertionError(cmd)


class Journal:
    def __init__(self):
        self.records = []

    def record(self, kind, **fields):
        self.records.append((kind, fields))

    def kinds(self):
        return [k for k, _ in self.records]


def _service(slm, blocker):
    clock = types.SimpleNamespace(t=0.0)
    journal = Journal()
    svc = SlmReinitService(SlmReinitConfig(host="127.0.0.1", port=1), blocker=blocker,
                           link=slm, clock=lambda: clock.t, journal=journal)
    return svc, clock, journal


def _run(svc, clock, until_t, step=0.5):
    while clock.t < until_t:
        svc.tick()
        clock.t += step


def test_the_service_restarts_a_supervised_server_and_follows_it():
    slm = FakeSupervisedSlm()
    svc, clock, journal = _service(slm, lambda: "")
    svc.tick()
    snap = svc.snapshot()
    assert snap["supervised"] and snap["can_restart"] and snap["server_instance"] == "aaaa0001"
    assert svc.request_restart("jp on kong") == ""
    _run(svc, clock, 1.0)
    assert slm.requests.count("restart") == 1
    assert svc.snapshot()["state"] == sr.RESTARTING
    _run(svc, clock, 8.0)
    snap = svc.snapshot()
    assert snap["state"] == sr.IDLE and snap["server_instance"] == "bbbb0002"
    assert snap["last_restart"]["result"] == "done" and snap["last_restart"]["by"] == "jp on kong"
    assert snap["start_count"] == 2
    assert "slm_restart_requested" in journal.kinds() and "slm_restart_done" in journal.kinds()


def test_the_service_never_restarts_while_the_machine_is_busy():
    slm = FakeSupervisedSlm()
    busy = ["run 83400 (feedback) is starting"]
    svc, clock, _ = _service(slm, lambda: busy[0])
    svc.tick()
    assert svc.request_restart("jp").startswith("run 83400")
    _run(svc, clock, 30.0)
    assert "restart" not in slm.requests


def test_the_service_refuses_to_restart_an_unsupervised_or_old_server():
    for slm, word in ((FakeSupervisedSlm(supervised=False), "supervisor"),
                      (FakeSupervisedSlm(capabilities=("seq", "status", "reinit")),
                       "no restart command")):
        svc, clock, _ = _service(slm, lambda: "")
        svc.tick()
        why = svc.request_restart("jp")
        assert word in why
        assert svc.snapshot()["can_restart"] is False
        _run(svc, clock, 10.0)
        assert "restart" not in slm.requests


def test_a_restart_that_brings_no_new_server_is_reported():
    slm = FakeSupervisedSlm()
    svc, clock, journal = _service(slm, lambda: "")
    svc.tick()
    assert svc.request_restart("jp") == ""
    _run(svc, clock, 1.0)
    slm.stuck = True
    _run(svc, clock, 70.0)
    snap = svc.snapshot()
    assert snap["state"] == sr.UNREACHABLE and "no new SLM server" in snap["detail"]
    assert "slm_restart_failed" in journal.kinds()


# --- the pill ------------------------------------------------------------------------------

@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _snap(**kw):
    d = {"state": "idle", "supervised": True, "can_restart": True, "blocked_by": "",
         "restart_pending": None, "start_count": 2, "last_exit": "75 restart",
         "last_restart": {"by": "jp", "at": time.time(), "t_s": 3.2, "result": "done"}}
    d.update(kw)
    return d


def test_the_pill_offers_a_restart_only_when_it_can_be_sent(qapp):
    from waxx.util.guis.slm_pill import SlmPill, describe_restart
    pill = SlmPill()
    pill.set_snapshot(_snap())
    assert pill.restart_allowed() == (True, "")
    menu = pill.build_menu()
    action = [a for a in menu.actions() if a.objectName() == "slm_restart_server"][0]
    assert action.isEnabled()
    assert "start 2" in describe_restart(pill.snapshot) and "done in 3.2 s" in \
        describe_restart(pill.snapshot)

    pill.set_snapshot(_snap(blocked_by="run 83400 is running"))
    assert pill.restart_allowed()[0] is False
    pill.set_snapshot(_snap(can_restart=False, supervised=False))
    ok, why = pill.restart_allowed()
    assert not ok and "supervisor" in why
    pill.set_snapshot({"state": "idle"})                  # a monitor server from before
    assert "predates" in pill.restart_allowed()[1]
