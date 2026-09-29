import socket
import json
import os
import sys
import time
import threading
import queue
import uuid
from slm_server import SLM_server
from slm_protocol import (split_commands, command_seq, control_command, Replier,
                          DEFAULT_SERVER_IP, DEFAULT_SERVER_PORT, CAPABILITIES,
                          EXIT_SHUTDOWN, EXIT_FATAL, EXIT_INIT_FAILED, EXIT_PORT_IN_USE,
                          EXIT_RESTART, default_state_dir, log_dir, read_log,
                          LOG_TAIL_LINES, LOG_REPLY_LINES)


# SLM_SERVER_IP / SLM_SERVER_PORT: for tests (127.0.0.1); the lab uses the defaults.
SERVER_IP = os.environ.get("SLM_SERVER_IP") or DEFAULT_SERVER_IP
SERVER_PORT = int(os.environ.get("SLM_SERVER_PORT") or DEFAULT_SERVER_PORT)
BUFFER_SIZE = 1024

REINIT_INTERVAL_SEC = 3600          # reinitialize period
MIN_IDLE_BEFORE_REINIT_SEC = 20     # idle time
CMD_QUEUE_MAXSIZE = 256
# False (default since 2026-09-28): the hourly timer only marks a reinit due
# (``status`` says so); the lab's monitor server asks for it (``reinit``) when no
# run is starting or running. The server-timed reinit used to land in the middle
# of runs that write the mask only at init and blank their mask (run 83344).
# True: the old behaviour -- reinit by itself when idle -- for a setup without a
# monitor server.
AUTO_REINIT = False
# A due reinit nobody asked for is reported (printed) after this long, then hourly.
REINIT_OVERDUE_WARN_SEC = 6 * 3600
# How often the heartbeat file (for the supervisor's hang check) is written.
HEARTBEAT_SEC = 2.0
# Control commands that leave no lines in the log: the monitor server reads the
# log with "log" every few seconds while someone watches it, and every request
# would otherwise add its own lines ("Connected by", "Received command") to the
# log it is reading.
QUIET_COMMANDS = ("log",)

slmtest = SLM_server()
cmd_q = queue.Queue(maxsize=CMD_QUEUE_MAXSIZE)

default_pattern = {
    "dimension": 0,
    "phase": 0.0,
    "center_x": 960,
    "center_y": 600,
    "grating_spacing": 10,
    "angle_deg": 0,
    "mask": 1  # 1=spot, 2=grating
}
last_pattern = {
    "dimension": 0,
    "phase": 0.0,
    "center_x": 960,
    "center_y": 600,
    "grating_spacing": 10,
    "angle_deg": 0,
    "mask": 1  # 1=spot, 2=grating
}

_last_activity_lock = threading.Lock()
_last_activity_monotonic = time.monotonic()

# Reinit bookkeeping, reported by the "status" command. pattern_epoch counts the
# reinits since this server started: a client that sees it change knows the SLM
# was re-initialised (and the last pattern put back) since it last looked.
_reinit_lock = threading.Lock()
_reinit = {"due_since": None,           # monotonic time the timer marked it due
           "next_due": None,            # monotonic time it marks it due next
           "in_progress": False,
           "last_done": None,           # monotonic time of the last reinit
           "last_error": "",
           "pattern_epoch": 0}

# This process: who it is, and whether the SLM can take a pattern. After a failed
# initialisation the SDK has been deleted (Load_lut failure path), so a pattern
# must not be written until a reinit succeeds.
INSTANCE = uuid.uuid4().hex[:8]
STARTED_AT = time.time()
_slm_ready = False
_slm_not_ready_why = "not initialised yet"
_applies = 0
_pattern_source = "default (nothing saved)"
_bind_mode = ""
# The task the worker is on, for the heartbeat: {"type", "since"} or None.
_current_task = None
_worker_alive = False
# Set by main() (from SLM_STATE_DIR / SLM_HEARTBEAT_PATH); None: not written.
_state_path = None
_heartbeat_path = None
# How the process ends itself. os._exit: the worker thread must be able to end
# the whole process (a restart, a shutdown, a fatal error). Tests replace it.
_exit = os._exit


def _touch_activity():
    global _last_activity_monotonic
    with _last_activity_lock:
        _last_activity_monotonic = time.monotonic()

def _seconds_since_last_activity():
    with _last_activity_lock:
        return time.monotonic() - _last_activity_monotonic


def _supervised():
    return os.environ.get("SLM_SUPERVISED") == "1"


def _end_process(code, why):
    """Leave now, with the exit code the supervisor acts on."""
    print(f"Exiting (code {code}): {why}", flush=True)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    _exit(code)


def _drop_queued(why):
    """Answer everything still queued: it will not be applied by this process."""
    while True:
        try:
            task = cmd_q.get_nowait()
        except queue.Empty:
            return
        if task is not None:
            _reply(task, status="dropped", error=why)
        cmd_q.task_done()


def slm_worker():
    global _slm_ready, _slm_not_ready_why, _worker_alive, _current_task
    _worker_alive = True
    try:
        _current_task = {"type": "INIT", "since": time.monotonic()}
        try:
            print("Initializing SLM...")
            slmtest.initialize_slm()
            _slm_ready, _slm_not_ready_why = True, ""
            _apply_pattern(last_pattern, fast=True)
            _save_pattern()
            _touch_activity()
        except Exception as e:
            # Without an initialised SLM this server is useless; its supervisor
            # starts it again (with a growing delay while it keeps failing).
            print(f"Error during initial init: {e}")
            _slm_ready, _slm_not_ready_why = False, f"start-up init failed: {e}"
            _current_task = None
            _worker_alive = False
            _end_process(EXIT_INIT_FAILED, f"the SLM could not be initialised: {e}")
            return
        _current_task = None

        while True:
            task = cmd_q.get()
            if task is None:
                break

            ttype = task.get("type")
            _current_task = {"type": ttype, "since": time.monotonic()}
            try:
                if ttype == "REINIT":
                    _do_reinit(task)

                elif ttype == "APPLY":
                    if not _slm_ready:
                        _reply(task, status="error",
                               error=f"the SLM is not initialised ({_slm_not_ready_why}); "
                                     f"ask for a reinit")
                        continue
                    last_pattern.update({
                        "dimension": task["dimension"],
                        "phase": task["phase"],
                        "center_x": task["center_x"],
                        "center_y": task["center_y"],
                        "grating_spacing": task["grating_spacing"],
                        "angle_deg": task["angle_deg"],
                        "mask": task["mask"]
                    })
                    t0 = time.monotonic()
                    _apply_pattern(last_pattern, fast=True)
                    _note_applied()
                    _touch_activity()
                    # Write_image has returned. The SLM still needs its next video
                    # frame and the liquid-crystal response time before the light
                    # sees the new pattern; the client waits that out itself.
                    _reply(task, status="applied",
                           center=[last_pattern["center_x"], last_pattern["center_y"]],
                           mask=last_pattern["mask"],
                           dimension=last_pattern["dimension"],
                           t_apply_s=round(time.monotonic() - t0, 4))

                elif ttype == "EXIT":
                    code = task["code"]
                    word = "restarting" if code == EXIT_RESTART else "shutting_down"
                    print(f"{'Restart' if code == EXIT_RESTART else 'Shutdown'} asked for by "
                          f"{task.get('by') or 'a client'}.")
                    _save_pattern()
                    try:
                        slmtest.release()
                    except Exception as e:
                        print(f"Releasing the SLM SDK failed (the process exits anyway): {e}")
                    _reply(task, status=word)
                    _drop_queued(f"the SLM server is {word.replace('_', ' ')}")
                    _end_process(code, word.replace("_", " "))
                    return

                else:
                    print(f"Unknown task type: {ttype}")
            except Exception as e:
                print(f"Error handling task {ttype}: {e}")
                _reply(task, status="error", error=str(e))
            finally:
                _current_task = None
                cmd_q.task_done()
    except BaseException as e:           # the worker must never die quietly
        _worker_alive = False
        _end_process(EXIT_FATAL, f"the SLM worker died: {type(e).__name__}: {e}")
        return
    _worker_alive = False


def _note_applied():
    global _applies
    _applies += 1
    _save_pattern()


def _do_reinit(task):
    """Re-initialise the SLM and put back the pattern it showed (last_pattern).

    It used to put back default_pattern (a blank mask): a run whose mask was
    written at its start then ran on blank (runs 83224-83344, 2026-09-27/28)."""
    global _slm_ready, _slm_not_ready_why
    with _reinit_lock:
        _reinit["in_progress"] = True
    t0 = time.monotonic()
    try:
        print("Reinitializing SLM...")
        slmtest.initialize_slm()
        _slm_ready, _slm_not_ready_why = True, ""
        print("Restoring the last pattern after reinitializing...\n")
        _apply_pattern(last_pattern, fast=True)
    except Exception as e:
        # Still due: the next request tries again. Until one succeeds the SDK may
        # be gone (Load_lut failure deletes it), so patterns are refused.
        print(f"Error during reinit: {e}")
        _slm_ready, _slm_not_ready_why = False, f"the last reinit failed: {e}"
        with _reinit_lock:
            _reinit["in_progress"] = False
            _reinit["last_error"] = str(e)
        _reply(task, status="error", error=f"reinit failed: {e}")
        return
    with _reinit_lock:
        done = time.monotonic()
        # the next one is due an interval after this one, however it was asked for
        _reinit.update(in_progress=False, due_since=None, last_done=done,
                       next_due=done + REINIT_INTERVAL_SEC,
                       last_error="", pattern_epoch=_reinit["pattern_epoch"] + 1)
        epoch = _reinit["pattern_epoch"]
    _reply(task, status="reinit_done", pattern=dict(last_pattern), pattern_epoch=epoch,
           t_reinit_s=round(time.monotonic() - t0, 3))


def _status():
    """What the "status" command answers."""
    now = time.monotonic()
    with _reinit_lock:
        r = dict(_reinit)
    try:
        start_count = int(os.environ.get("SUPERVISOR_START_COUNT", ""))
    except ValueError:
        start_count = None
    return {
        "status": "ok",
        "reinit_due": r["due_since"] is not None,
        "due_for_s": None if r["due_since"] is None else round(now - r["due_since"], 1),
        # when the timer next marks one due (None before the timer runs)
        "next_due_in_s": None if r["next_due"] is None else round(r["next_due"] - now, 1),
        "interval_s": REINIT_INTERVAL_SEC,
        "reinit_in_progress": r["in_progress"],
        "last_reinit_age_s": None if r["last_done"] is None else round(now - r["last_done"], 1),
        "last_reinit_error": r["last_error"],
        "pattern_epoch": r["pattern_epoch"],
        "pattern": dict(last_pattern),
        "idle_s": round(_seconds_since_last_activity(), 1),
        "queue_len": cmd_q.qsize(),
        "auto_reinit": AUTO_REINIT,
        "pid": os.getpid(),
        "instance": INSTANCE,
        "started_at": STARTED_AT,
        "supervised": _supervised(),
        "start_count": start_count,
        "last_exit": os.environ.get("SUPERVISOR_LAST_EXIT") or None,
        "slm_ready": _slm_ready,
        "slm_not_ready": "" if _slm_ready else _slm_not_ready_why,
        "applies": _applies,
        "pattern_source": _pattern_source,
        "bind": _bind_mode,
        "capabilities": list(CAPABILITIES),
    }


def _reply(task, **msg):
    """Answer the client that sent `task`, if it asked to be answered."""
    replier = task.get("replier")
    if replier is not None:
        replier.send({"seq": task["seq"], **msg})

def _apply_pattern(pat, fast=True): # Generate and upload a pattern
    img = slmtest.generate_mask(
        dimension=pat["dimension"],
        phase=pat["phase"],
        center_x=pat["center_x"],
        center_y=pat["center_y"],
        grating_spacing=pat["grating_spacing"],
        angle_deg=pat["angle_deg"],
        mask=pat["mask"]
    )
    if fast:
        slmtest.fast_upload_to_slm(img)
    else:
        slmtest.upload_to_slm(img)

    print(
        f'-> mask: {slmtest.mask_type}, '
        f'dimension={pat["dimension"]} um, '
        f'phase={pat["phase"]} pi, '
        f'center=({pat["center_x"]},{pat["center_y"]}), '
        f'spacing={pat["grating_spacing"]}, angle={pat["angle_deg"]}'
    )
    print('Waiting for next task...\n')


# --- the pattern survives a restart --------------------------------------------------

_PATTERN_KEYS = {"dimension": int, "phase": float, "center_x": int, "center_y": int,
                 "grating_spacing": int, "angle_deg": int, "mask": int}


def _save_pattern():
    """Write last_pattern to the state file (atomically), so the next process --
    after a restart or a crash -- puts the same pattern back."""
    if _state_path is None:
        return
    try:
        os.makedirs(os.path.dirname(_state_path), exist_ok=True)
        tmp = _state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"pattern": dict(last_pattern), "saved_at": time.time(),
                       "instance": INSTANCE}, fh)
        os.replace(tmp, _state_path)
    except OSError as e:
        print(f"Could not save the pattern to {_state_path}: {e}")


def _load_pattern():
    """last_pattern from the state file, if there is a valid one; says where from."""
    global _pattern_source
    if _state_path is None or not os.path.exists(_state_path):
        _pattern_source = "default (nothing saved)"
        return False
    try:
        with open(_state_path, encoding="utf-8") as fh:
            saved = json.load(fh)
        pat = {k: cast(saved["pattern"][k]) for k, cast in _PATTERN_KEYS.items()}
    except (OSError, ValueError, KeyError, TypeError) as e:
        _pattern_source = f"default (saved pattern unreadable: {e})"
        print(f"Saved pattern in {_state_path} not used: {e}")
        return False
    last_pattern.update(pat)
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(saved.get("saved_at", 0)))
    _pattern_source = f"restored (saved {when})"
    print(f"Restoring the pattern saved {when}: center=({pat['center_x']},{pat['center_y']}), "
          f"mask={pat['mask']}, dimension={pat['dimension']}")
    return True


# --- heartbeat, for the supervisor's hang check ----------------------------------------

def _write_heartbeat():
    if _heartbeat_path is None:
        return
    task = _current_task
    # ppid too: under a venv the supervisor starts a launcher, whose child this
    # process is, so the pid the supervisor holds is our parent's.
    beat = {"pid": os.getpid(), "ppid": os.getppid(), "instance": INSTANCE, "t": time.time(),
            "worker_alive": _worker_alive,
            "task": None if task is None else task["type"],
            "task_age_s": None if task is None else round(time.monotonic() - task["since"], 1),
            "queue_len": cmd_q.qsize(), "applies": _applies, "slm_ready": _slm_ready}
    try:
        tmp = _heartbeat_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(beat, fh)
        os.replace(tmp, _heartbeat_path)
    except OSError:
        pass


def heartbeat_loop(stop=None):
    while stop is None or not stop.is_set():
        _write_heartbeat()
        time.sleep(HEARTBEAT_SEC)


def periodic_reinit_scheduler(stop=None):
    """
    Every REINIT_INTERVAL_SEC: with AUTO_REINIT False (the default) mark a
    reinit due -- the monitor server asks for it ("reinit") when no run is
    starting or running, and "status" reports it until then. With AUTO_REINIT
    True, the old behaviour: enqueue REINIT if the server is idle long enough
    and the queue is empty. The interval restarts after every reinit (the
    reinit sets the next due time). `stop` (a threading.Event) ends it, for tests.
    """
    with _reinit_lock:
        if _reinit["next_due"] is None:
            _reinit["next_due"] = time.monotonic() + REINIT_INTERVAL_SEC
    next_overdue_warn = None
    while stop is None or not stop.is_set():
        time.sleep(0.5)
        now = time.monotonic()
        with _reinit_lock:
            due_since = _reinit["due_since"]
            next_due = _reinit["next_due"]
            if now >= next_due:
                # whole intervals: a loop that stalled past several fires it once
                _reinit["next_due"] = next_due + REINIT_INTERVAL_SEC * (
                    int((now - next_due) // REINIT_INTERVAL_SEC) + 1)
        if due_since is not None and now - due_since >= REINIT_OVERDUE_WARN_SEC:
            if next_overdue_warn is None or now >= next_overdue_warn:
                print(f"WARNING: a reinit has been due for {(now - due_since) / 3600:.1f} h "
                      f"and nobody asked for it (is the monitor server running?).")
                next_overdue_warn = now + 3600
        if now < next_due:
            continue

        if not AUTO_REINIT:
            with _reinit_lock:
                if _reinit["due_since"] is None:
                    _reinit["due_since"] = now
                    print("Reinit due: waiting for the monitor server to ask for it.")
            continue

        idle_secs = _seconds_since_last_activity()
        if idle_secs < MIN_IDLE_BEFORE_REINIT_SEC:
            print(f"Skipped: only idle {idle_secs:.1f}s (need {MIN_IDLE_BEFORE_REINIT_SEC}s).\n")
            continue

        if not cmd_q.empty():
            print("Skipped: command queue not empty.")
            continue

        try:
            cmd_q.put_nowait({"type": "REINIT"})
            print("Enqueued REINIT (idle & queue empty).")
        except queue.Full:
            print("Skipped: queue full.")


# --- listening ------------------------------------------------------------------------

class PortInUse(OSError):
    """Another server is listening on the port."""


def _port_has_listener(ip, port, timeout=0.5):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def _open_listener(ip, port):
    """The listening socket, never shared with another server.

    SO_REUSEADDR alone lets a second server bind the same port on Windows while
    the first one still listens -- two servers then take turns with the SLM.
    So: refuse if something already answers on the port; bind exclusively
    (SO_EXCLUSIVEADDRUSE) where the OS has it; fall back to SO_REUSEADDR only
    when the exclusive bind is refused with nobody listening (connections of
    the previous process lingering in TIME_WAIT after a restart)."""
    global _bind_mode
    if _port_has_listener(ip, port):
        raise PortInUse(f"something already listens on {ip}:{port}")
    excl = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
    if excl is not None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, excl, 1)
            s.bind((ip, port))
            s.listen(1)
            _bind_mode = "exclusive"
            return s
        except OSError as e:
            s.close()
            if _port_has_listener(ip, port):
                raise PortInUse(f"something already listens on {ip}:{port}") from e
            first_error = e
    else:
        first_error = None
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ip, port))
    s.listen(1)
    _bind_mode = ("reuseaddr" if first_error is None else
                  f"reuseaddr (exclusive bind refused: {first_error})")
    return s


def start_server(listener=None):
    server_socket = listener or _open_listener(SERVER_IP, SERVER_PORT)
    with server_socket:
        print(f'Server listening on {SERVER_IP}:{SERVER_PORT} ({_bind_mode})...')

        while True:
            conn, addr = server_socket.accept()
            handle_client(conn, addr)

def _is_local_peer(conn):
    """Whether a connection comes from this PC (loopback, or its own address)."""
    try:
        peer = conn.getpeername()[0]
        own = conn.getsockname()[0]
    except OSError:
        return False
    return peer == own or peer.startswith("127.") or peer == "::1"

def handle_client(conn, addr=None):
    replier = Replier(conn)
    local = _is_local_peer(conn)
    pending = ""
    # "Connected by" waits for the first command that is printed, so a
    # connection with only quiet commands (QUIET_COMMANDS) prints nothing.
    announced = []

    def announce():
        if not announced:
            announced.append(True)
            print(f'Connected by {addr}')

    with conn:
        while True:
            try:
                data = conn.recv(BUFFER_SIZE)
                at_eof = not data
                # One recv is not one command: TCP may join two commands or
                # split one. Cut complete commands out of everything received
                # so far and keep the rest for the next recv.
                pending += data.decode('utf-8', errors='replace')
                commands, pending = split_commands(pending, at_eof=at_eof)
                for command in commands:
                    _handle_command(command.strip(), replier, local=local, announce=announce)
                if at_eof:
                    if announced:
                        print("Client disconnected.")
                    break

            except ConnectionResetError:
                print("SLM_find_spot.py disconnected")
                break
            except Exception as e:
                print(f"Error while handling client: {e}")
                break

def _enqueue_exit(code, seq, replier, by):
    """Queue a restart / shutdown behind any pattern already queued."""
    task = {"type": "EXIT", "code": code, "by": by}
    if seq is not None:
        task["seq"] = seq
        task["replier"] = replier
        replier.send({"seq": seq, "status": "queued"})
    try:
        cmd_q.put_nowait(task)
    except queue.Full:
        _reply(task, status="error", error="command queue full")

def _log_lines(ctl):
    """What the "log" command answers: this PC's log files (written by the
    supervisor) from the client's cursor on; see slm_protocol.read_log."""
    if not _supervised():
        return {"status": "error",
                "error": "the SLM server is not running under its supervisor "
                         "(supervisor.py), which writes the log: its output is only in its "
                         "window on the SLM PC"}

    def count(key, default):
        value = ctl.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            return default
        return min(max(value, 0), LOG_REPLY_LINES)

    try:
        return read_log(log_dir(), ctl.get("cursor"), tail=count("tail", LOG_TAIL_LINES),
                        max_lines=max(count("max_lines", LOG_REPLY_LINES), 1))
    except Exception as e:
        return {"status": "error", "error": f"reading the log failed: {type(e).__name__}: {e}"}


def _handle_control(ctl, seq, replier, local=False):
    """A control command ({"cmd": ...}): "status" and "log" answer at once;
    "reinit" queues a reinit behind any pattern already queued and answers
    "queued", then "reinit_done" (or "error") when it is done; "restart" /
    "shutdown" queue the process's exit (see slm_protocol)."""
    cmd = ctl.get("cmd")
    if cmd == "status":
        if seq is not None:
            replier.send({"seq": seq, **_status()})
        return
    if cmd == "log":
        if seq is not None:
            replier.send({"seq": seq, **_log_lines(ctl)})
        return
    if cmd == "reinit":
        task = {"type": "REINIT"}
        if seq is not None:
            task["seq"] = seq
            task["replier"] = replier
            replier.send({"seq": seq, "status": "queued"})
        try:
            cmd_q.put_nowait(task)
            print(f"Enqueued REINIT (asked for by {ctl.get('by') or 'a client'}).")
        except queue.Full:
            _reply(task, status="error", error="command queue full")
        return
    if cmd == "restart":
        if not _supervised():
            why = ("not running under its supervisor (supervisor.py): nothing would start "
                   "it again")
            print(f"Restart asked for by {ctl.get('by') or 'a client'} REFUSED: {why}")
            if seq is not None:
                replier.send({"seq": seq, "status": "error", "error": why})
            return
        _enqueue_exit(EXIT_RESTART, seq, replier, ctl.get("by"))
        return
    if cmd == "shutdown":
        if not local:
            why = "shutdown is accepted only from the SLM PC itself"
            print(f"Shutdown asked for by {ctl.get('by') or 'a client'} REFUSED: {why}")
            if seq is not None:
                replier.send({"seq": seq, "status": "error", "error": why})
            return
        _enqueue_exit(EXIT_SHUTDOWN, seq, replier, ctl.get("by"))
        return
    if seq is not None:
        replier.send({"seq": seq, "status": "error", "error": f"unknown cmd {cmd!r}"})


def _handle_command(command, replier, local=False, announce=None):
    seq = command_seq(command)
    ctl = control_command(command)
    if ctl is not None and ctl.get("cmd") in QUIET_COMMANDS:
        _handle_control(ctl, seq, replier, local=local)
        return
    if announce is not None:
        announce()
    print(f"Received command: {command}")

    if ctl is not None:
        _handle_control(ctl, seq, replier, local=local)
        return

    dims = analyze_command(command)
    if dims is None:
        print("Ignoring malformed command.")
        if seq is not None:
            replier.send({"seq": seq, "status": "error", "error": "malformed command"})
        return

    dimension, phase, center_x, center_y, grating_spacing, angle_deg, mask = dims

    task = {
        "type": "APPLY",
        "dimension": dimension,
        "phase": phase,
        "center_x": center_x,
        "center_y": center_y,
        "grating_spacing": grating_spacing,
        "angle_deg": angle_deg,
        "mask": mask
    }
    if seq is not None:
        task["seq"] = seq
        task["replier"] = replier
        # before the put, so it cannot arrive after the worker's "applied"
        replier.send({"seq": seq, "status": "queued"})

    try:
        cmd_q.put_nowait(task)
    except queue.Full:
        try:
            dropped = cmd_q.get_nowait()
            cmd_q.task_done()
            _reply(dropped, status="dropped")
            cmd_q.put_nowait(task)
            print("Queue full: dropped one stale task to enqueue latest APPLY.")
        except Exception as e:
            print(f"Failed to enqueue APPLY: {e}")
            _reply(task, status="error", error=f"could not enqueue: {e}")

def analyze_command(command):
    dimension = 0
    phase = 0.0
    center_x = 1920 // 2
    center_y = 1200 // 2
    mask = 1  # 1=spot, 2=grating
    grating_spacing = 10
    angle_deg = 0

    try:
        d = json.loads(command)
        m = d.get("mask", "spot")

        cx, cy = d.get("center", [center_x, center_y])
        center_x, center_y = int(cx), int(cy)
        dimension = int(d.get("dimension", dimension))
        phase = float(d.get("phase", phase))
        grating_spacing = int(d.get("spacing", grating_spacing))
        angle_deg = int(d.get("angle", d.get("angle_deg", angle_deg)))

        if m == "spot":
            mask = 1
            print("Mask: spot")
        elif m == "grating":
            mask = 2
            print("Mask: grating")
        else:
            print("Unknown mask; set to default spot.")
            mask = 1

        return (dimension, phase, center_x, center_y, grating_spacing, angle_deg, mask)

    except json.JSONDecodeError:
        parts = command.split()
        if len(parts) == 3:
            try:
                dimension = int(parts[0])
                phase = float(parts[1])
                mask = int(parts[2])
                return (dimension, phase, center_x, center_y, grating_spacing, angle_deg, mask)
            except ValueError:
                print("Plaintext 3-arg parse failed.")
                return None

        elif len(parts) == 7:
            try:
                dimension = int(parts[0])
                phase = float(parts[1])
                center_x = int(parts[2])
                center_y = int(parts[3])
                grating_spacing = int(parts[4])
                angle_deg = int(parts[5])
                mask = int(parts[6])
                return (dimension, phase, center_x, center_y, grating_spacing, angle_deg, mask)
            except ValueError:
                print("Plaintext 7-arg parse failed.")
                return None

        else:
            print("Wrong plaintext format length.")
            return None


def main():
    global _state_path, _heartbeat_path
    _state_path = os.path.join(default_state_dir(), "last_pattern.json")
    _heartbeat_path = os.environ.get("SLM_HEARTBEAT_PATH") or None
    print(f"SLM server {INSTANCE} (pid {os.getpid()})"
          + (f", supervised, start {os.environ.get('SUPERVISOR_START_COUNT')}"
             if _supervised() else ", not supervised"))
    _load_pattern()
    # The port first: a second server must stop here, before it touches the SLM.
    try:
        listener = _open_listener(SERVER_IP, SERVER_PORT)
    except PortInUse as e:
        _end_process(EXIT_PORT_IN_USE, f"{e}: another SLM server is running")
        return
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    threading.Thread(target=slm_worker, daemon=True).start()
    threading.Thread(target=periodic_reinit_scheduler, daemon=True).start()
    start_server(listener)


if __name__ == '__main__':
    main()
