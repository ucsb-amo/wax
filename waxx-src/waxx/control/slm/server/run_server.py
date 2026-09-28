import socket
import json
import time
import threading
import queue
from slm_server import SLM_server
from slm_protocol import split_commands, command_seq, control_command, Replier


SERVER_IP = '192.168.1.102'
SERVER_PORT = 5000
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

def _touch_activity():
    global _last_activity_monotonic
    with _last_activity_lock:
        _last_activity_monotonic = time.monotonic()

def _seconds_since_last_activity():
    with _last_activity_lock:
        return time.monotonic() - _last_activity_monotonic

def slm_worker():
    try:
        print("Initializing SLM...")
        slmtest.initialize_slm()
        _apply_pattern(last_pattern, fast=True)
        _touch_activity()
    except Exception as e:
        print(f"Error during initial init: {e}")

    while True:
        task = cmd_q.get() 
        if task is None:
            break  

        ttype = task.get("type")
        try:
            if ttype == "REINIT":
                _do_reinit(task)

            elif ttype == "APPLY":
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
                _touch_activity()
                # Write_image has returned. The SLM still needs its next video
                # frame and the liquid-crystal response time before the light
                # sees the new pattern; the client waits that out itself.
                _reply(task, status="applied",
                       center=[last_pattern["center_x"], last_pattern["center_y"]],
                       mask=last_pattern["mask"],
                       dimension=last_pattern["dimension"],
                       t_apply_s=round(time.monotonic() - t0, 4))

            else:
                print(f"Unknown task type: {ttype}")
        except Exception as e:
            print(f"Error handling task {ttype}: {e}")
            _reply(task, status="error", error=str(e))
        finally:
            cmd_q.task_done()

def _do_reinit(task):
    """Re-initialise the SLM and put back the pattern it showed (last_pattern).

    It used to put back default_pattern (a blank mask): a run whose mask was
    written at its start then ran on blank (runs 83224-83344, 2026-09-27/28)."""
    with _reinit_lock:
        _reinit["in_progress"] = True
    t0 = time.monotonic()
    try:
        print("Reinitializing SLM...")
        slmtest.initialize_slm()
        print("Restoring the last pattern after reinitializing...\n")
        _apply_pattern(last_pattern, fast=True)
    except Exception as e:
        # Still due: the next request tries again.
        print(f"Error during reinit: {e}")
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

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((SERVER_IP, SERVER_PORT))
        server_socket.listen(1)
        print(f'Server listening on {SERVER_IP}:{SERVER_PORT}...')

        while True:
            conn, addr = server_socket.accept()
            print(f'Connected by {addr}')
            handle_client(conn)

def handle_client(conn):
    replier = Replier(conn)
    pending = ""
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
                    _handle_command(command.strip(), replier)
                if at_eof:
                    print("Client disconnected.")
                    break

            except ConnectionResetError:
                print("SLM_find_spot.py disconnected")
                break
            except Exception as e:
                print(f"Error while handling client: {e}")
                break

def _handle_control(ctl, seq, replier):
    """A control command ({"cmd": ...}): "status" answers at once; "reinit"
    queues a reinit behind any pattern already queued and answers "queued",
    then "reinit_done" (or "error") when it is done."""
    cmd = ctl.get("cmd")
    if cmd == "status":
        if seq is not None:
            replier.send({"seq": seq, **_status()})
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
    if seq is not None:
        replier.send({"seq": seq, "status": "error", "error": f"unknown cmd {cmd!r}"})


def _handle_command(command, replier):
    print(f"Received command: {command}")
    seq = command_seq(command)

    ctl = control_command(command)
    if ctl is not None:
        _handle_control(ctl, seq, replier)
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

if __name__ == '__main__':
    threading.Thread(target=slm_worker, daemon=True).start()

    threading.Thread(target=periodic_reinit_scheduler, daemon=True).start()

    start_server()
