"""waxx.util.supervise, the Qt-free supervision core (2026-09-28).

Children are small Python scripts in pytest's tmp_path; nothing touches the
network, and every process started here is waited for or killed.
"""
import ctypes
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import waxx
from waxx.util.supervise import (EXIT_RESTART, KillOnCloseJob, ProcessSupervisor,
                                 RestartPolicy, kill_pid_tree)

PY = sys.executable
IS_WIN = sys.platform.startswith("win")


def _script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return [PY, "-u", str(p)]


def _fast(**kw):
    kw.setdefault("initial_s", 0.01)
    kw.setdefault("max_s", 0.05)
    kw.setdefault("window_s", 60.0)
    kw.setdefault("max_in_window", 3)
    kw.setdefault("slow_retry_s", None)
    return RestartPolicy(**kw)


def _wait(pred, timeout=10.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _alive(pid):
    """Without os.kill: on Windows os.kill(pid, 0) terminates the process."""
    if not IS_WIN:
        try:
            import os
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))       # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == 259                           # STILL_ACTIVE


def test_restart_policy_backs_off_then_retries_slowly():
    p = RestartPolicy(initial_s=1, max_s=5, window_s=100, max_in_window=3, slow_retry_s=50,
                      stable_s=10)
    assert [p.next_delay(t) for t in (0, 1, 2)] == [1, 2, 4]
    assert p.next_delay(3) == 50 and p.failing
    p.ran_for(11)                                  # a stable run clears the history
    assert p.history == [] and p.next_delay(200) == 1
    q = RestartPolicy(initial_s=1, window_s=100, max_in_window=1, slow_retry_s=None)
    q.next_delay(0)
    assert q.next_delay(1) is None                 # the dashboard's FAILED


def test_a_stop_code_ends_supervision_and_output_is_forwarded(tmp_path):
    cmd = _script(tmp_path, "c.py", "import sys; print('hello'); sys.exit(0)")
    lines = []
    sup = ProcessSupervisor("t", cmd, on_line=lines.append, kill_on_close=False)
    assert sup.run() == 0
    assert sup.starts == 1 and "hello" in lines
    assert [e.reason for e in sup.exits] == ["stopped"]


def test_the_restart_code_restarts_at_once_and_is_no_crash(tmp_path):
    log = tmp_path / "starts.txt"
    cmd = _script(tmp_path, "c.py", f"""
        import os, sys
        with open(r"{log}", "a") as fh:
            fh.write(os.environ["SUPERVISOR_START_COUNT"] + "|"
                     + os.environ["SUPERVISOR_LAST_EXIT"] + "\\n")
        sys.exit({EXIT_RESTART} if int(os.environ["SUPERVISOR_START_COUNT"]) < 3 else 0)
    """)
    events = []
    sup = ProcessSupervisor("t", cmd, policy=_fast(), kill_on_close=False,
                            on_event=lambda k, i: events.append((k, i)))
    assert sup.run() == 0
    assert log.read_text().splitlines() == ["1|", "2|75 restart", "3|75 restart"]
    assert [e.reason for e in sup.exits] == ["restart", "restart", "stopped"]
    assert sup.policy.history == []
    assert [i["delay_s"] for k, i in events if k == "restart_in"] == [0.0, 0.0]


def test_crashes_back_off_and_the_policy_can_give_up(tmp_path):
    cmd = _script(tmp_path, "c.py", "import sys; sys.exit(3)")
    events = []
    sup = ProcessSupervisor("t", cmd, policy=_fast(max_in_window=2), kill_on_close=False,
                            on_event=lambda k, i: events.append((k, i)))
    assert sup.run() == 3
    assert sup.starts == 3
    assert [i["delay_s"] for k, i in events if k == "restart_in"] == [0.01, 0.02]
    assert events[-1][0] == "give_up"


def test_a_failing_child_is_retried_slowly_rather_than_abandoned(tmp_path):
    cmd = _script(tmp_path, "c.py", "import sys; sys.exit(3)")
    events = []
    sup = ProcessSupervisor("t", cmd, kill_on_close=False,
                            policy=_fast(max_in_window=1, slow_retry_s=0.05),
                            on_event=lambda k, i: events.append((k, i)))
    th = threading.Thread(target=sup.run, daemon=True)
    th.start()
    assert _wait(lambda: sup.starts >= 4)
    sup.stop("test")
    th.join(10)
    assert not th.is_alive()
    delays = [i["delay_s"] for k, i in events if k == "restart_in"]
    assert delays[0] == 0.01 and all(d == 0.05 for d in delays[1:3])
    assert not any(k == "give_up" for k, _ in events)


def test_a_hung_child_is_killed_and_started_again(tmp_path):
    cmd = _script(tmp_path, "c.py", """
        import os, sys, time
        if os.environ["SUPERVISOR_START_COUNT"] == "1":
            time.sleep(60)          # hangs
        sys.exit(0)
    """)
    events = []
    sup = ProcessSupervisor("t", cmd, policy=_fast(), kill_on_close=False,
                            health=lambda pid: (False, "stuck"), health_period_s=0.1,
                            health_grace_s=0.3, health_failures=2,
                            on_event=lambda k, i: events.append((k, i)))
    t0 = time.monotonic()
    assert sup.run() == 0
    assert time.monotonic() - t0 < 20
    assert [e.reason for e in sup.exits] == ["hung", "stopped"]
    assert sup.exits[0].detail == "stuck"
    assert any(k == "hung" for k, _ in events)


def test_no_health_information_counts_only_after_the_grace_period(tmp_path):
    cmd = _script(tmp_path, "c.py", "import time; time.sleep(0.6)")
    calls = []

    def health(pid):
        calls.append(time.monotonic())
        return None, "no heartbeat yet"

    sup = ProcessSupervisor("t", cmd, kill_on_close=False, health=health,
                            health_period_s=0.05, health_grace_s=5.0)
    assert sup.run() == 0 and calls == []           # exited inside the grace period


def test_stop_asks_the_child_first(tmp_path):
    flag = tmp_path / "please_exit"
    cmd = _script(tmp_path, "c.py", f"""
        import os, sys, time
        while not os.path.exists(r"{flag}"):
            time.sleep(0.02)
        sys.exit(0)
    """)
    asked = []

    def graceful(pid):
        asked.append(pid)
        flag.write_text("x")
        return True

    sup = ProcessSupervisor("t", cmd, graceful_stop=graceful, stop_timeout_s=5.0,
                            kill_on_close=False)
    th = threading.Thread(target=sup.run, daemon=True)
    th.start()
    assert _wait(sup.child_alive)
    sup.stop("test")
    th.join(10)
    assert not th.is_alive() and asked
    assert sup.exits[-1].reason == "stop_requested" and sup.exits[-1].code == 0


def test_stop_kills_a_child_that_does_not_listen(tmp_path):
    cmd = _script(tmp_path, "c.py", "import time; time.sleep(60)")
    sup = ProcessSupervisor("t", cmd, graceful_stop=lambda pid: False, stop_timeout_s=0.3,
                            kill_on_close=False)
    th = threading.Thread(target=sup.run, daemon=True)
    th.start()
    assert _wait(sup.child_alive)
    pid = sup.pid
    sup.stop("test")
    th.join(15)
    assert not th.is_alive()
    assert sup.exits[-1].reason == "stop_requested"
    assert _wait(lambda: not _alive(pid), 5)


@pytest.mark.skipif(not IS_WIN, reason="job objects are Windows only")
def test_the_child_dies_with_its_supervisor(tmp_path):
    pidfile = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(f"import os, time\nopen(r'{pidfile}', 'w').write(str(os.getpid()))\n"
                     f"time.sleep(120)\n", encoding="utf-8")
    waxx_src = Path(waxx.__file__).resolve().parents[1]
    sup_py = tmp_path / "sup.py"
    sup_py.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, r"{waxx_src}")
        from waxx.util.supervise import ProcessSupervisor
        ProcessSupervisor("t", [sys.executable, r"{child}"]).run()
    """), encoding="utf-8")
    proc = subprocess.Popen([PY, str(sup_py)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    child_pid = None
    try:
        assert _wait(lambda: pidfile.exists() and pidfile.read_text().strip() != "", 20)
        child_pid = int(pidfile.read_text())
        assert _alive(child_pid)
        proc.kill()                                  # the supervisor only, abruptly
        proc.wait(10)
        assert _wait(lambda: not _alive(child_pid), 10), "the child outlived its supervisor"
    finally:
        if proc.poll() is None:
            proc.kill()
        if child_pid and _alive(child_pid):
            kill_pid_tree(child_pid)


@pytest.mark.skipif(not IS_WIN, reason="job objects are Windows only")
def test_a_job_can_be_made():
    job = KillOnCloseJob()
    assert job.error == "" and job._handle
