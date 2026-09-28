"""Keep the SLM server running on the SLM PC.

``server.bat`` starts this instead of ``run_server.py``; it runs the server as
its child (:class:`waxx.util.supervise.ProcessSupervisor`, the Qt-free core of
the dashboard's server supervision) and:

* starts it again at once when it asks to restart (exit 75: the ``restart``
  control command, e.g. from the Device Control GUI's SLM pill through the
  monitor server, which sends it only while no run is starting or running);
* starts it again after a crash, with a delay that grows while it keeps
  crashing (2 s doubling to 60 s; after 5 crashes in 10 min, every 5 min) --
  it never gives up, since nobody may be at the PC;
* kills and restarts it when it hangs: its heartbeat file (written every 2 s)
  goes stale, its SLM worker thread is dead, or one SLM task has run for
  minutes;
* stops when the server shuts down on purpose (exit 0), or on Ctrl+C here --
  it then asks the server to shut down (``shutdown``, which the server takes
  only from this PC) and kills it if it does not.

Because this process started the server, the server runs in the same Windows
session -- the console session server.bat hands itself to, where the SLM is
visible as a display. That is what makes a remote restart work at all: a
server started from a Remote Desktop session cannot see the SLM.

The server is in a job object that dies with this process, so closing this
window stops the server too (as closing server.bat's window always did), and
no orphaned server can hold port 5000 against the next one.

Logs: every line of the server and every supervisor event, with a time stamp,
in ``<state dir>\\logs\\slm_server_<date>.log`` (state dir: ``SLM_STATE_DIR``,
else ``%LOCALAPPDATA%\\slm_server``); ``supervisor.json`` there says what the
supervisor is doing.

Exit code 90: ``waxx.util.supervise`` could not be imported; server.bat then
runs the server unsupervised.
"""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXIT_NO_CORE = 90

try:
    from waxx.util.supervise import ProcessSupervisor, RestartPolicy, stamp
except ImportError:
    # Not installed in this environment: use the waxx package this file sits in
    # (waxx-src/waxx/control/slm/server -> waxx-src).
    sys.path.insert(0, str(HERE.parents[3]))
    try:
        from waxx.util.supervise import ProcessSupervisor, RestartPolicy, stamp
    except ImportError as _e:              # pragma: no cover
        print(f"[supervisor] cannot import waxx.util.supervise ({_e}); "
              f"server.bat runs the SLM server without supervision.")
        sys.exit(EXIT_NO_CORE)

sys.path.insert(0, str(HERE))
from slm_protocol import (control_line, default_state_dir, EXIT_MEANING, EXIT_RESTART,  # noqa: E402
                          EXIT_SHUTDOWN, DEFAULT_SERVER_IP, DEFAULT_SERVER_PORT)

#: The heartbeat may be this old before the server counts as hung.
HEARTBEAT_MAX_AGE_S = 20.0
#: One SLM task (a pattern, a reinit) may run this long before it counts as hung;
#: start-up initialisation gets longer.
TASK_HANG_S = 120.0
INIT_HANG_S = 180.0


def heartbeat_health(path, max_age_s=HEARTBEAT_MAX_AGE_S, task_hang_s=TASK_HANG_S,
                     init_hang_s=INIT_HANG_S, now=time.time):
    """``health(pid) -> (ok, why)`` from the server's heartbeat file; ok None:
    no heartbeat from this process yet.

    ``pid`` is the process the supervisor started. Under a venv that is the
    venv's ``python.exe`` launcher, and the server is the launcher's child, so
    the heartbeat's ``ppid`` matches instead of its ``pid`` (checking ``pid``
    alone had every server killed as hung 65 s after it started, 2026-09-28)."""
    def check(pid):
        try:
            with open(path, encoding="utf-8") as fh:
                beat = json.load(fh)
        except (OSError, ValueError):
            return None, "no heartbeat yet"
        if pid not in (beat.get("pid"), beat.get("ppid")):
            return None, "no heartbeat from this process yet"
        age = now() - float(beat.get("t", 0))
        if age > max_age_s:
            return False, f"heartbeat {age:.0f} s old (the server is not running its threads)"
        if not beat.get("worker_alive"):
            return False, "the SLM worker thread is not running"
        task, task_age = beat.get("task"), beat.get("task_age_s")
        limit = init_hang_s if task == "INIT" else task_hang_s
        if task and isinstance(task_age, (int, float)) and task_age > limit:
            return False, f"the SLM task {task} has been running {task_age:.0f} s"
        return True, ""
    return check


def graceful_shutdown(ip, port, timeout_s=2.0):
    """``graceful_stop(pid)``: ask the server to shut down; True if it said it will."""
    def ask(pid):
        with socket.create_connection((ip, port), timeout=timeout_s) as s:
            s.sendall(control_line({"cmd": "shutdown", "seq": 1, "by": "supervisor"}).encode())
            s.settimeout(timeout_s)
            buf = b""
            t_end = time.monotonic() + timeout_s
            while time.monotonic() < t_end:
                try:
                    chunk = s.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                buf += chunk
                for line in buf.split(b"\n")[:-1]:
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    if msg.get("status") in ("queued", "shutting_down"):
                        return True
                    if msg.get("status") == "error":
                        return False
        return False
    return ask


class Journal:
    """Time-stamped lines to the console and to a daily log file."""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def line(self, text):
        entry = f"{stamp()} | {text}"
        print(entry, flush=True)
        try:
            day = time.strftime("%Y-%m-%d")
            with open(self.log_dir / f"slm_server_{day}.log", "a", encoding="utf-8") as fh:
                fh.write(entry + "\n")
        except OSError:
            pass


def build(args):
    state_dir = Path(args.state_dir or default_state_dir())
    state_dir.mkdir(parents=True, exist_ok=True)
    journal = Journal(state_dir / "logs")
    heartbeat = state_dir / "heartbeat.json"
    status_path = state_dir / "supervisor.json"
    server_dir = Path(args.server_dir or HERE)
    ip, port = args.ip or DEFAULT_SERVER_IP, int(args.port or DEFAULT_SERVER_PORT)
    status = {"supervisor_pid": os.getpid(), "state": "starting", "starts": 0,
              "last_exit": None, "server_pid": None, "since": time.time(),
              "server": f"{ip}:{port}"}

    def write_status(**kw):
        status.update(kw, updated=time.time())
        try:
            tmp = status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(status, indent=1), encoding="utf-8")
            os.replace(tmp, status_path)
        except OSError:
            pass

    def on_event(kind, info):
        if kind == "start":
            journal.line(f"[supervisor] SLM server started (pid {info['pid']}, start "
                         f"{info['start']})")
            write_status(state="running", starts=info["start"], server_pid=info["pid"])
        elif kind == "exit":
            code = info["code"]
            meaning = EXIT_MEANING.get(code, "crashed" if info["reason"] == "crash" else "")
            detail = f": {info['detail']}" if info.get("detail") else ""
            journal.line(f"[supervisor] SLM server exited (code {code}, {info['reason']}"
                         f"{', ' + meaning if meaning else ''}) after {info['ran_s']:.0f} s{detail}")
            write_status(state="exited", server_pid=None,
                         last_exit={"code": code, "reason": info["reason"], "at": time.time()})
        elif kind == "restart_in":
            if info["delay_s"] > 0:
                slow = " (it keeps crashing: slow retries)" if info["failing"] else ""
                journal.line(f"[supervisor] starting it again in {info['delay_s']:.0f} s{slow}")
            write_status(state="restarting", restart_in_s=info["delay_s"])
        elif kind == "hung":
            journal.line(f"[supervisor] SLM server hung ({info['why']}): killing it")
        elif kind == "stopping":
            if "graceful_error" in info:
                journal.line(f"[supervisor] graceful shutdown request failed "
                             f"({info['graceful_error']}); killing the server")
            else:
                journal.line(f"[supervisor] stopping the SLM server ({info['reason']})")
            write_status(state="stopping")
        elif kind == "job":
            if not info["ok"]:
                journal.line(f"[supervisor] WARNING: the server is not tied to this window "
                             f"({info['error']}); closing this window may leave it running")
        elif kind == "give_up":
            journal.line("[supervisor] giving up")

    env = {"SLM_SUPERVISED": "1", "SLM_HEARTBEAT_PATH": str(heartbeat),
           "SLM_STATE_DIR": str(state_dir), "SLM_SERVER_IP": ip, "SLM_SERVER_PORT": str(port)}
    sup = ProcessSupervisor(
        "slm_server", [sys.executable, "-u", str(server_dir / "run_server.py")],
        cwd=str(server_dir), env=env,
        stop_codes=(EXIT_SHUTDOWN,), restart_codes=(EXIT_RESTART,),
        policy=RestartPolicy(initial_s=2.0, max_s=60.0, window_s=600.0, max_in_window=5,
                             slow_retry_s=300.0, stable_s=300.0),
        health=None if args.no_health else heartbeat_health(heartbeat),
        health_period_s=5.0, health_grace_s=float(args.health_grace_s), health_failures=2,
        graceful_stop=graceful_shutdown(ip, port), stop_timeout_s=10.0,
        on_line=lambda text: journal.line(text), on_event=on_event)
    return sup, journal, write_status


def main(argv=None):
    ap = argparse.ArgumentParser(description="Keep the SLM server running (see the module doc).")
    ap.add_argument("--ip", help=f"the server's address (default {DEFAULT_SERVER_IP})")
    ap.add_argument("--port", type=int, help=f"(default {DEFAULT_SERVER_PORT})")
    ap.add_argument("--state-dir", help="saved pattern, heartbeat, logs (default %%LOCALAPPDATA%%\\slm_server)")
    ap.add_argument("--server-dir", help="folder of run_server.py (default: this folder)")
    ap.add_argument("--health-grace-s", default=60.0, help="no hang check this long after a start")
    ap.add_argument("--no-health", action="store_true", help="no hang detection")
    args = ap.parse_args(argv)
    sup, journal, write_status = build(args)
    journal.line(f"[supervisor] pid {os.getpid()}: keeping the SLM server running. Ctrl+C "
                 f"stops it; closing this window stops it too.")
    code = sup.run()
    write_status(state="stopped", server_pid=None)
    journal.line(f"[supervisor] stopped (the server's last exit code: {code})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
