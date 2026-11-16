"""``kq launch``: the PC that submits a run queue job runs it (user, 2026-11-15).

The monitor server's run queue (:mod:`~waxx.util.device_state.run_queue`)
decides the order, the gate and GO; it never starts a job's process.  A
*launcher* -- this module, a hidden detached process on the submitting PC,
one per submit call (all the jobs that call made) -- does:

* it attaches to its jobs and heartbeats the server (``{"type":
  "run_queue", "action": "launcher"}``; the server's side is
  :mod:`~waxx.util.device_state.launcher_bridge`) -- every second while one
  of its jobs is near the front, in the slot or warm, every few seconds
  otherwise, as the server asks;
* it starts ``artiq_run`` from THIS PC's copy of the file when the server
  gives an order -- *warm* (``WAXX_QUEUE_WARM``, BELOW_NORMAL priority) or
  *launch* (cold) -- after the checks the server used to make: the file is
  there and is the one submitted (else "skipped: source changed since
  submit", unless drift is allowed), the working folder exists, ``%kpy%`` and
  ``%db%`` are set;
* it follows the child, sends its output (a log on this PC,
  ``<dir>/logs/<id>_<token>.out``; the server keeps a copy, so ``kq tail``
  works from anywhere), reports its exit code until the server acknowledges
  it, kills it only on the server's order, and raises its priority at GO;
* it keeps a state file per job (``<dir>/<id>_<token>.json``) and writes
  "starting" there BEFORE it starts a child: a launcher started again for
  the same job (it died; the panel or the server started another) takes its
  child back by pid and creation time, never starts it twice, and reports a
  child it may or may not have started as "unknown" (held for a person);
* it exits once the server has no more use for any of its jobs.  It never
  kills a child because the server is silent: the run goes on, and the
  launcher keeps trying.

Start it with :func:`start_detached` (``kq run`` / ``submit`` / ``insert`` do,
right after the submit), or by hand: ``kq launch <id>:<token> ...``.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import sys
import time
from typing import Callable

log = logging.getLogger(__name__)

#: Environment variable that moves the launchers' folder (tests).
DIR_ENV = "WAXX_LAUNCHER_DIR"
#: Most bytes of a job's log one heartbeat carries.
LOG_CHUNK = 32 * 1024
#: Heartbeat interval before the server has said, and its bounds.
BEAT_S = 1.0
BEAT_MIN_S, BEAT_MAX_S = 0.2, 10.0
#: The wait between tries while the server does not answer.
RETRY_S = 2.0
#: A new connection to the monitor server after this many silent tries.
RECONNECT_AFTER = 15
REQUEST_TIMEOUT_S = 5.0
#: The environment variables this PC must have to run an experiment.
NEEDED_ENV = ("kpy", "db")
#: Characters never allowed in a path or argument (the command is a cmd line).
_SHELL_CHARS = set('&|<>^%"!')


def default_dir() -> str:
    return os.environ.get(DIR_ENV) or os.path.join(os.path.expanduser("~"), ".waxx", "run_queue",
                                                   "launcher")


def _quote(arg: str) -> str:
    return f'"{arg}"' if (" " in arg or "\t" in arg) else arg


def parse_pairs(words) -> list[tuple[int, str]]:
    """``["12:ab12cd", ...]`` -> ``[(12, "ab12cd"), ...]`` (ValueError when one
    is not ``<id>:<token>``)."""
    out = []
    for w in words:
        head, _, token = str(w).partition(":")
        if not token:
            raise ValueError(f"not <id>:<token>: {w!r}")
        out.append((int(head), token))
    return out


class _JobLock:
    """An OS lock on one job's state file for this process's life: two
    launchers on one PC never work on the same job."""

    def __init__(self, path: str):
        self.path = path
        self._f = None

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        f = open(self.path, "a+b")
        try:
            if sys.platform == "win32":
                import msvcrt  # noqa: PLC0415
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self._f = f
        return True

    def release(self) -> None:
        if self._f is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt  # noqa: PLC0415
                self._f.seek(0)
                msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        self._f.close()
        self._f = None


class Launcher:
    """The launcher for ``pairs`` (``[(id, token), ...]``).  ``transport``:
    an object with ``request(obj, timeout=, attempts=)`` -> reply dict or
    None (default: a MonitorClient found by discovery, made when first
    needed); the process callables default to
    :mod:`~waxx.util.device_state.detached`'s."""

    def __init__(self, pairs, *, transport=None, directory: str | None = None,
                 host: str | None = None, connect: Callable[[], object] | None = None,
                 spawn=None, adopt=None, kill=None, set_priority=None, command_for=None,
                 env: dict | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        from waxx.util.device_state import detached  # noqa: PLC0415
        self.directory = directory or default_dir()
        self.host = host or socket.gethostname()
        self.lid = secrets.token_hex(6)
        self.transport = transport
        self._connect = connect
        self._spawn = spawn or detached.launch
        self._adopt = adopt or detached.ProcessWatch.adopt
        self._kill = kill or detached.kill_tree
        self._set_priority = set_priority or detached.set_normal_priority
        self._command_for = command_for or _default_command
        self._env = dict(os.environ if env is None else env)
        self._clock = clock
        self._sleep = sleep
        self.pid = os.getpid()
        self.started = detached.process_started()
        self.jobs: dict[int, dict] = {}
        self._locks: dict[int, _JobLock] = {}
        self._watches: dict = {}
        self._sha_cache: dict = {}
        self.next_beat_s = BEAT_S
        self.silent_tries = 0
        for jid, token in pairs:
            self._take(int(jid), str(token))

    # -- the jobs' state files ------------------------------------------------------------

    def _state_path(self, jid: int, token: str) -> str:
        return os.path.join(self.directory, f"{jid}_{token}.json")

    def log_path(self, job: dict) -> str:
        return os.path.join(self.directory, "logs", f"{job['id']}_{job['token']}.out")

    def _take(self, jid: int, token: str) -> None:
        lock = _JobLock(self._state_path(jid, token) + ".lock")
        if not lock.acquire():
            log.warning("kq launch: job %s is another launcher's on this PC: left to it.", jid)
            return
        self._locks[jid] = lock
        job = {"id": jid, "token": token, "children": {}, "kills": {}, "log_sent": 0,
               "acked": [], "cancelled": [], "path": "", "sha256": None, "allow_drift": False}
        try:
            with open(self._state_path(jid, token), encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                job.update({k: v for k, v in saved.items() if k in job})
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            log.error("kq launch: job %s's state file could not be read (%s): its children "
                      "count as unknown.", jid, exc)
            job["children"] = {"?": {"state": "unknown",
                                     "why": f"the launcher's state file could not be read ({exc})"}}
        # a child the last launcher recorded as starting, with no pid: it may run
        for key, c in job["children"].items():
            if c.get("state") == "starting":
                c.update(state="unknown", why="a launcher stopped while starting it: it may or "
                                              "may not be running -- check this PC")
        self.jobs[jid] = job
        self._save(job)

    def _save(self, job: dict) -> None:
        path = self._state_path(job["id"], job["token"])
        tmp = path + ".tmp"
        try:
            os.makedirs(self.directory, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(job, f)
            os.replace(tmp, path)
        except OSError as exc:
            log.error("kq launch: could not write job %s's state file (%s).", job["id"], exc)

    def _forget(self, jid: int, why: str) -> None:
        job = self.jobs.pop(jid, None)
        if job is None:
            return
        log.info("kq launch: job %s done here: %s", jid, why)
        for key in list(self._watches):
            if key[0] == jid:
                w = self._watches.pop(key)
                try:
                    w.close()
                except Exception:                     # noqa: BLE001
                    pass
        try:
            os.remove(self._state_path(jid, job["token"]))
        except OSError:
            pass
        lock = self._locks.pop(jid, None)
        if lock is not None:
            lock.release()

    # -- children ---------------------------------------------------------------------------

    def _alive(self, job: dict) -> bool:
        return any(c.get("state") in ("starting", "running", "unknown")
                   for c in job["children"].values())

    def _poll_children(self) -> None:
        for jid, job in self.jobs.items():
            changed = False
            for key, c in job["children"].items():
                if c.get("state") != "running":
                    continue
                watch = self._watches.get((jid, key))
                if watch is None:
                    try:
                        watch = self._adopt(c.get("pid"), c.get("started"))
                    except Exception:                 # noqa: BLE001
                        watch = None
                    if watch is None:
                        c.update(state="exited", code=-1,
                                 why="gone when this launcher looked for it (its exit code is "
                                     "not known)")
                        changed = True
                        continue
                    self._watches[(jid, key)] = watch
                code = watch.poll()
                if code is not None:
                    c.update(state="exited", code=int(code), ended=time.time())
                    changed = True
                    try:
                        watch.close()
                    except Exception:                 # noqa: BLE001
                        pass
                    self._watches.pop((jid, key), None)
            if changed:
                self._save(job)

    def _check(self, order: dict) -> tuple[str, str]:
        """("", "") when the order may be carried out, else (state, why):
        "skipped" (the file changed since submit) or "refused"."""
        path, cwd = str(order.get("path") or ""), str(order.get("cwd") or "")
        argv = [str(a) for a in (order.get("argv") or [])]
        if not path or not os.path.isfile(path):
            return "refused", f"no such file on {self.host}: {path}"
        if not path.lower().endswith(".py"):
            return "refused", f"not a Python file: {path}"
        bad = sorted(_SHELL_CHARS & set(path + "".join(argv)))
        if bad:
            return "refused", f"the path or an argument contains {' '.join(bad)} (not allowed)"
        if cwd and not os.path.isdir(cwd):
            return "refused", f"no such working folder on {self.host}: {cwd}"
        missing = [v for v in NEEDED_ENV if not self._env.get(v)]
        if missing:
            return "refused", (f"this PC ({self.host}) has no " + ", ".join(f"%{v}%" for v in missing)
                               + ": it cannot run experiments (run its setup, then submit again)")
        want = order.get("sha256")
        if want and not order.get("allow_drift"):
            from waxx.util.device_state.job_source import sha256_or_none  # noqa: PLC0415
            have = sha256_or_none(path)
            if have is None:
                return "refused", f"the file cannot be read on {self.host}: {path}"
            if have != want:
                return "skipped", "source changed since submit"
        return "", ""

    def _start(self, job: dict, order: dict) -> None:
        key = str(order.get("key") or "")
        if not key or key in job["children"] or key in job["acked"]:
            return
        if key in job["cancelled"]:
            job["children"][key] = {"state": "cancelled", "why": "killed before it started"}
            self._save(job)
            return
        state, why = self._check(order)
        if state:
            job["children"][key] = {"state": state, "why": why}
            self._save(job)
            log.warning("kq launch: job %s not started (%s): %s", job["id"], state, why)
            return
        job["children"][key] = {"state": "starting", "at": time.time(),
                                "warm": key != "cold"}
        self._save(job)                               # BEFORE the child exists
        env = dict(self._env)
        for k, v in (order.get("env") or {}).items():
            if v is None:
                env.pop(str(k), None)
            else:
                env[str(k)] = str(v)
        command = self._command_for(order)
        logp = self.log_path(job)
        os.makedirs(os.path.dirname(logp), exist_ok=True)
        from waxx.util.device_state.detached import LaunchUnknown  # noqa: PLC0415
        try:
            watch = self._spawn(command, cwd=order.get("cwd") or os.path.dirname(order["path"]),
                                env=env, log_path=logp,
                                below_normal=bool(order.get("below_normal")))
        except LaunchUnknown as exc:
            job["children"][key] = {"state": "unknown", "why": str(exc)}
            self._save(job)
            log.error("kq launch: job %s: %s", job["id"], exc)
            return
        except Exception as exc:                      # noqa: BLE001
            job["children"][key] = {"state": "refused", "why": f"could not start: {exc!r}"}
            self._save(job)
            log.error("kq launch: job %s could not be started: %r", job["id"], exc)
            return
        job["children"][key] = {"state": "running", "pid": int(watch.pid),
                                "started": getattr(watch, "started", None), "at": time.time(),
                                "warm": key != "cold"}
        self._watches[(job["id"], key)] = watch
        self._save(job)
        log.info("kq launch: job %s started (%s, pid %s).", job["id"],
                 "warm" if key != "cold" else "cold", watch.pid)

    def _do_kill(self, job: dict, key: str, seq: int) -> None:
        done = job["kills"].get(key) or {}
        if done.get("seq") == seq:
            return
        c = job["children"].get(key) or {}
        if c.get("state") == "running":
            try:
                result = dict(self._kill(c.get("pid"), c.get("started")))
            except Exception as exc:                  # noqa: BLE001
                result = {"pid": c.get("pid"), "ok": False, "gone": False,
                          "note": f"kill failed: {exc!r}"}
        elif c.get("state") in ("starting", "unknown"):
            result = {"pid": c.get("pid"), "ok": False, "gone": False,
                      "note": "its start was not confirmed: nothing known to kill -- check "
                              f"{self.host}"}
        else:
            if key not in job["cancelled"]:
                job["cancelled"].append(key)
            if not c:
                job["children"][key] = {"state": "cancelled", "why": "killed before it started"}
            result = {"pid": None, "ok": False, "gone": True,
                      "note": "not running here (" + (c.get("state") or "never started") + ")"}
        result["role"] = "shell"
        job["kills"][key] = {"seq": seq, "result": result}
        self._save(job)
        log.warning("kq launch: kill of job %s (%s): %s", job["id"], key if key == "cold" else
                    "warm", result.get("note"))

    # -- the heartbeat ------------------------------------------------------------------------

    def _info(self, job: dict) -> dict | None:
        path = job.get("path")
        if not path:
            return None
        try:
            st = os.stat(path)
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            return {"sha256": None, "expt_class": "", "calibrates_declared": [],
                    "cold_why": "its file cannot be read on " + self.host}
        cached = self._sha_cache.get(path)
        if cached is None or cached[0] != stamp:
            from waxx.util.device_state.job_source import source_info  # noqa: PLC0415
            cached = (stamp, source_info(path))
            self._sha_cache[path] = cached
        return cached[1]

    def _log_entry(self, job: dict) -> dict | None:
        """The job's log from what the server has, at most :data:`LOG_CHUNK`
        bytes, whole lines -- the last unterminated line once no child runs."""
        logp = self.log_path(job)
        sent = int(job.get("log_sent") or 0)
        try:
            with open(logp, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size <= sent:
                    return {"offset": sent, "end": sent, "text": ""} if size == sent else None
                f.seek(sent)
                data = f.read(LOG_CHUNK)
        except OSError:
            return None
        if len(data) < size - sent or self._alive(job):
            cut = data.rfind(b"\n") + 1
            if cut == 0 and len(data) == LOG_CHUNK:
                cut = len(data)                       # one line longer than a chunk
            data = data[:cut]
        return {"offset": sent, "end": sent + len(data),
                "text": data.decode("utf-8", errors="replace")}

    def _log_behind(self, job: dict) -> bool:
        try:
            return os.path.getsize(self.log_path(job)) > int(job.get("log_sent") or 0)
        except OSError:
            return False

    def report(self) -> dict:
        jobs = []
        for jid, job in self.jobs.items():
            behind = self._log_behind(job)
            children = {}
            for key, c in job["children"].items():
                if key in job["acked"]:
                    continue
                c = dict(c)
                if c.get("state") == "exited" and behind:
                    # its end is reported once all its output is with the
                    # server (the queue judges the run from its last lines)
                    c = {"state": "running", "pid": c.get("pid"), "started": c.get("started"),
                         "warm": c.get("warm")}
                children[key] = c
            jobs.append({"id": jid, "token": job["token"], "children": children,
                         "kills": dict(job["kills"]), "log": self._log_entry(job),
                         "info": self._info(job)})
        return {"type": "run_queue", "action": "launcher",
                "launcher": {"id": self.lid, "host": self.host, "pid": self.pid,
                             "started": self.started}, "jobs": jobs}

    def apply(self, reply: dict) -> None:
        try:
            self.next_beat_s = min(BEAT_MAX_S, max(BEAT_MIN_S, float(reply.get("next_beat_s")
                                                                     or BEAT_S)))
        except (TypeError, ValueError):
            self.next_beat_s = BEAT_S
        for jid in list(self.jobs):
            r = (reply.get("jobs") or {}).get(str(jid))
            if not isinstance(r, dict):
                continue
            job = self.jobs[jid]
            if r.get("path"):
                job["path"] = str(r["path"])
            job["sha256"], job["allow_drift"] = r.get("sha256"), bool(r.get("allow_drift"))
            if r.get("log_ack") is not None:
                try:
                    job["log_sent"] = int(r["log_ack"])
                except (TypeError, ValueError):
                    pass
            for key in r.get("ack") or ():
                if key not in job["acked"]:
                    job["acked"].append(str(key))
            for key, seq in (r.get("kill") or {}).items():
                self._do_kill(job, str(key), int(seq))
            pr = r.get("priority")
            if isinstance(pr, dict) and pr.get("pid"):
                self._set_priority(pr.get("pid"), pr.get("started"))
            order = r.get("order")
            if isinstance(order, dict):
                self._start(job, order)
            self._save(job)
            why = r.get("drop") or r.get("refused")
            if why and not self._alive(job) and not self._log_behind(job):
                self._forget(jid, str(why))

    def _ask(self, obj: dict) -> dict | None:
        if self.transport is None:
            try:
                self.transport = (self._connect or _connect)()
            except Exception as exc:                  # noqa: BLE001
                log.info("kq launch: no monitor server found yet (%s).", exc)
                return None
        try:
            reply = self.transport.request(obj, timeout=REQUEST_TIMEOUT_S, attempts=1)
        except Exception as exc:                      # noqa: BLE001
            log.info("kq launch: the monitor server did not answer (%s).", exc)
            reply = None
        return reply if isinstance(reply, dict) else None

    def step(self) -> float:
        """One heartbeat -> seconds until the next."""
        self._poll_children()
        reply = self._ask(self.report())
        if reply is None:
            self.silent_tries += 1
            if self.silent_tries >= RECONNECT_AFTER and self._connect is not None:
                self.transport, self.silent_tries = None, 0
            return RETRY_S
        self.silent_tries = 0
        if reply.get("status") != "ok":
            msg = str(reply.get("msg") or reply)
            if "unknown run_queue action" in msg:
                # a monitor server from before 2026-11-15: it starts jobs itself
                for jid in list(self.jobs):
                    if not self._alive(self.jobs[jid]):
                        self._forget(jid, "the monitor server runs its jobs itself (older "
                                          "version): nothing to do here")
            else:
                log.warning("kq launch: the server refused the heartbeat: %s", msg)
            return RETRY_S
        self.apply(reply)
        return self.next_beat_s

    def run(self) -> int:
        """Heartbeat until no job is left."""
        try:
            while self.jobs:
                wait = self.step()
                if self.jobs:
                    self._sleep(wait)
        finally:
            for jid in list(self._locks):
                self._locks.pop(jid).release()
        return 0


def _connect():
    from waxx.util.device_state.run_queue_client import connect_monitor  # noqa: PLC0415
    return connect_monitor(3.0)


def _default_command(order: dict) -> str:
    from waxx.util.device_state.monitor_manager import ar_command  # noqa: PLC0415
    command = ar_command(_quote(str(order["path"])))
    argv = [str(a) for a in (order.get("argv") or [])]
    if argv:
        command += " " + " ".join(_quote(a) for a in argv)
    return command


def start_detached(pairs, *, directory: str | None = None, launch=None) -> int:
    """Start a launcher for ``pairs`` (``[(id, token), ...]``) on this PC,
    outside the caller's process tree (a closed terminal or a dashboard's
    ``taskkill /T`` leaves it running), with no window.  -> its pid."""
    from waxx.util.device_state import detached  # noqa: PLC0415
    pairs = [(int(j), str(t)) for j, t in pairs]
    if not pairs:
        raise ValueError("no job to launch")
    directory = directory or default_dir()
    os.makedirs(directory, exist_ok=True)
    exe = sys.executable
    if sys.platform == "win32" and exe.lower().endswith("pythonw.exe"):
        exe = exe[:-len("pythonw.exe")] + "python.exe"
    command = (f"{_quote(exe)} -m waxx.util.device_state.launcher "
               + " ".join(f"{j}:{t}" for j, t in pairs))
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    env.pop("WAXX_QUEUE_WARM", None)
    log_path = os.path.join(directory, f"launcher_{pairs[0][0]}.log")
    watch = (launch or detached.launch)(command, cwd=directory, env=env, log_path=log_path)
    pid = int(watch.pid)
    try:
        watch.close()
    except Exception:                                 # noqa: BLE001
        pass
    return pid


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    words = list(sys.argv[1:] if argv is None else argv)
    try:
        pairs = parse_pairs(words)
    except ValueError as exc:
        print(f"kq launch: {exc} (usage: kq launch <id>:<token> ...)", file=sys.stderr)
        return 2
    if not pairs:
        print("kq launch: no job given (usage: kq launch <id>:<token> ...)", file=sys.stderr)
        return 2
    launcher = Launcher(pairs, connect=_connect)
    if not launcher.jobs:
        print("kq launch: every job given has a launcher on this PC already", file=sys.stderr)
        return 0
    log.info("kq launch: pid %s on %s for job(s) %s.", launcher.pid, launcher.host,
             ", ".join(str(j) for j in launcher.jobs))
    return launcher.run()


if __name__ == "__main__":
    sys.exit(main())
