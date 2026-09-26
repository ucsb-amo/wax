"""A child process that holds one host-side connection for the monitor server.

The monitor server keeps the connections the machine needs between runs (the
tweezer AWG; see :mod:`waxx.util.device_state.connections`), but the driver
itself never runs in the server: each connection session gets its own agent
process.  The drivers behind these connections can hang -- ``spcm_vClose``
has joined a driver thread forever (run 83101) and a dead link has kept a
driver thread spinning -- and a hung agent is killed and replaced, where a
hung server would have to be restarted.  Spawned on open, told to close and
exit on release, killed if it does not.

**Detached.**  The server starts the agent through a launcher process that
starts the agent and exits at once, so the agent is not in the server's
process tree: the dashboard stops the monitor server with ``taskkill /T``,
which must not take the agent with it before it has closed its device.
However the server goes away -- a normal stop, a crash, that kill -- the
agent's stdin reaches EOF, and the agent closes the device (the driver's own
bounded close) and exits.

Protocol, one JSON object per line on stdin / stdout::

    -> {"id": n, "cmd": "open" | "close" | "status" | <driver command>, "kwargs": {...}}
    -> {"id": n, "cmd": "exit"}
    <- @@agent {"id": n, "ok": true,  "result": ..., "open": bool, "detail": str}
    <- @@agent {"id": n, "ok": false, "error": str,  "open": bool, "detail": str}

The first line is ``{"id": 0, "ready": true, "pid": ...}`` once the driver is
built (``ok`` false and an ``error`` if it could not be, then the agent
exits).  Only lines starting with the marker are protocol: everything the
agent or its libraries print goes to stderr, which the server logs.

A driver is ``module:factory``, called with the connection's
``driver_kwargs``; the object it returns has ``open()`` (raises on failure),
``close() -> bool`` (False: the close could not be confirmed), ``is_open()``,
``detail() -> str``, and ``COMMANDS``: the other method names the server may
call (e.g. ``write_traps``).
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import Callable

AGENT_MODULE = "waxx.util.device_state.connection_agent"
MARKER = "@@agent "

#: The commands every driver has; ``COMMANDS`` adds its own.
BASE_COMMANDS = ("open", "close", "status")

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


class AgentError(Exception):
    """The agent is gone, did not answer in time, or refused a command."""


class AgentDied(AgentError):
    pass


class AgentTimeout(AgentError):
    pass


class AgentCommandError(AgentError):
    """The driver raised; ``state`` is ``{"open", "detail"}`` after it."""

    def __init__(self, message: str, state: dict | None = None):
        super().__init__(message)
        self.state = dict(state or {})


def _error_text(e: BaseException) -> str:
    text = str(e).strip()
    return text if text else type(e).__name__


def load_factory(spec: str) -> Callable:
    module, _, name = str(spec).partition(":")
    if not module or not name:
        raise ValueError(f"driver {spec!r} is not 'module:factory'")
    obj = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


# --- the agent process -------------------------------------------------------------

def _driver_state(driver) -> dict:
    try:
        is_open = bool(driver.is_open())
    except Exception as e:
        return {"open": False, "detail": f"(is_open failed: {_error_text(e)})"}
    try:
        detail = str(driver.detail() or "") if is_open else ""
    except Exception as e:
        detail = f"(detail failed: {_error_text(e)})"
    return {"open": is_open, "detail": detail}


def agent_main(spec: str, kwargs: dict) -> None:
    """The agent: build the driver, serve commands until "exit" or EOF, close
    the driver if it is still open, and leave without waiting for driver
    threads (a stuck close thread must not keep the process alive)."""
    out = sys.stdout
    sys.stdout = sys.stderr          # stray prints must not reach the protocol

    def send(obj):
        out.write(MARKER + json.dumps(obj, default=str) + "\n")
        out.flush()

    try:
        driver = load_factory(spec)(**kwargs)
    except BaseException as e:
        traceback.print_exc()
        send({"id": 0, "ok": False, "ready": False, "pid": os.getpid(),
              "error": f"could not build the driver {spec}: {_error_text(e)}"})
        sys.stderr.flush()
        os._exit(2)
    allowed = set(BASE_COMMANDS) | set(getattr(driver, "COMMANDS", ()))
    send({"id": 0, "ok": True, "ready": True, "pid": os.getpid(), **_driver_state(driver)})

    why = "the monitor server went away (its end of the pipe closed)"
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            req = json.loads(line)
            rid = int(req.get("id"))
            cmd = str(req.get("cmd") or "")
            call_kwargs = dict(req.get("kwargs") or {})
        except Exception as e:
            print(f"agent: unreadable request {line!r} ({_error_text(e)})")
            continue
        if cmd == "exit":
            why = "asked to exit"
            send({"id": rid, "ok": True, "result": None, **_driver_state(driver)})
            break
        if cmd not in allowed:
            send({"id": rid, "ok": False, "error": f"unknown command {cmd!r}",
                  **_driver_state(driver)})
            continue
        try:
            result = None if cmd == "status" else getattr(driver, cmd)(**call_kwargs)
        except BaseException as e:
            traceback.print_exc()
            send({"id": rid, "ok": False, "error": _error_text(e), **_driver_state(driver)})
            continue
        send({"id": rid, "ok": True, "result": result, **_driver_state(driver)})

    try:
        if _driver_state(driver)["open"]:
            print(f"agent: {why}; closing the device before exiting.")
            closed = driver.close()
            print("agent: closed." if closed is not False else
                  "agent: the close did not finish; exiting anyway (the connection drops "
                  "with this process).")
    except BaseException:
        traceback.print_exc()
    sys.stderr.flush()
    os._exit(0)


def _launch_detached(argv) -> None:
    """The launcher: start the agent with this process's stdin/stdout/stderr
    (the server's pipes) and exit, so the agent's parent is gone and it is
    outside the server's process tree."""
    subprocess.Popen([sys.executable, "-m", AGENT_MODULE, *argv],
                     stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr,
                     creationflags=_NO_WINDOW | _NEW_GROUP, close_fds=True)
    os._exit(0)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    detach = "--detach" in argv
    if detach:
        argv.remove("--detach")
        _launch_detached(argv)
    spec = argv[argv.index("--driver") + 1]
    kwargs = json.loads(argv[argv.index("--kwargs") + 1]) if "--kwargs" in argv else {}
    agent_main(spec, kwargs)


# --- the server's side ---------------------------------------------------------------

class AgentClient:
    """One agent process, from the server.  Thread safe: calls from several
    threads are serialized by the caller (the connection's own lock), but
    :meth:`kill` may come from any thread at any time -- it makes a call in
    flight (or a start) fail at once with :class:`AgentDied`."""

    def __init__(self, driver: str, kwargs: dict | None = None, name: str = "agent",
                 log: Callable[[str], None] | None = None,
                 on_exit: Callable[["AgentClient"], None] | None = None,
                 env: dict | None = None, detach: bool = True):
        self.driver = driver
        self.kwargs = dict(kwargs or {})
        self.name = name
        self._log = log or (lambda text: None)
        self._on_exit = on_exit
        self._env = env
        self._detach = detach
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self.pid: int | None = None
        self._killed = False
        self._expected_exit = False
        self._dead = threading.Event()
        self._replies: dict[int, dict] = {}
        self._reply_events: dict[int, threading.Event] = {}
        self._next_id = 1
        self.state = {"open": False, "detail": ""}

    # -- lifecycle ---------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._proc is not None and not self._dead.is_set()

    def start(self, timeout: float) -> None:
        """Spawn the agent and wait for its driver to be built."""
        args = [sys.executable, "-m", AGENT_MODULE, "--driver", self.driver,
                "--kwargs", json.dumps(self.kwargs)]
        if self._detach:
            args.append("--detach")
        ready = threading.Event()
        self._reply_events[0] = ready
        env = dict(os.environ if self._env is None else self._env)
        env["PYTHONIOENCODING"] = "utf-8"       # driver messages are not all cp1252
        with self._lock:
            if self._killed:
                raise AgentDied(f"{self.name}: stopped before it started")
            self._proc = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=_NO_WINDOW, close_fds=True, env=env,
                encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._read_stdout, name=f"{self.name}-out", daemon=True).start()
        threading.Thread(target=self._read_stderr, name=f"{self.name}-err", daemon=True).start()
        if not self._wait(ready, 0, timeout):
            self.kill()
            raise AgentTimeout(f"{self.name}: its driver was not ready within {timeout:g} s")
        reply = self._replies.pop(0)
        self.pid = reply.get("pid")
        if not reply.get("ok"):
            self._expected_exit = True
            raise AgentCommandError(str(reply.get("error") or "the driver could not be built"))
        if self._killed:
            self.kill()
            raise AgentDied(f"{self.name}: stopped while it started")
        self.state = {"open": bool(reply.get("open")), "detail": str(reply.get("detail") or "")}

    def call(self, cmd: str, timeout: float, **kwargs):
        """Run a driver command; its result.  Raises AgentCommandError (the
        driver raised), AgentTimeout or AgentDied."""
        if not self.alive:
            raise AgentDied(f"{self.name} is not running")
        rid = self._next_id
        self._next_id += 1
        event = threading.Event()
        self._reply_events[rid] = event
        line = json.dumps({"id": rid, "cmd": cmd, "kwargs": kwargs}) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except (OSError, ValueError, AttributeError):
            self._reply_events.pop(rid, None)
            raise AgentDied(f"{self.name} is not running (its pipe is closed)")
        if not self._wait(event, rid, timeout):
            raise AgentTimeout(f"{self.name}: {cmd} did not answer within {timeout:g} s")
        reply = self._replies.pop(rid)
        self.state = {"open": bool(reply.get("open")), "detail": str(reply.get("detail") or "")}
        if not reply.get("ok"):
            raise AgentCommandError(str(reply.get("error") or f"{cmd} failed"), self.state)
        return reply.get("result")

    def exit(self, timeout: float) -> bool:
        """Ask the agent to exit (it closes a device still open) and wait for
        it to be gone.  False if it was still there after ``timeout``."""
        self._expected_exit = True
        if self.alive:
            try:
                self.call("exit", timeout=timeout)
            except AgentError:
                pass
        return self._dead.wait(timeout)

    def kill(self) -> None:
        """Stop the agent now: close its pipe and terminate the process.  The
        device is dropped with the process's connection, not closed."""
        with self._lock:
            self._killed = True
            self._expected_exit = True
            proc = self._proc
        if proc is None:
            return
        try:
            proc.stdin.close()
        except Exception:
            pass
        if self.pid and not self._dead.is_set():
            try:
                os.kill(int(self.pid), signal.SIGTERM)
            except Exception:
                pass
        if proc.poll() is None and not self._detach:
            try:
                proc.kill()
            except Exception:
                pass

    # -- internal ----------------------------------------------------------------------

    def _wait(self, event: threading.Event, rid: int, timeout: float) -> bool:
        deadline = time.monotonic() + max(float(timeout), 0.)
        while True:
            if event.wait(min(0.05, max(deadline - time.monotonic(), 0.))):
                self._reply_events.pop(rid, None)
                if rid not in self._replies:
                    raise AgentDied(f"{self.name} exited"
                                    + (" (stopped)" if self._killed else " unexpectedly"))
                return True
            if self._killed:
                # kill() came from another thread (a release): do not sit out
                # an agent that is still importing or opening.
                self._reply_events.pop(rid, None)
                raise AgentDied(f"{self.name} was stopped")
            if self._dead.is_set():
                self._reply_events.pop(rid, None)
                raise AgentDied(f"{self.name} exited"
                                + (" (stopped)" if self._killed else " unexpectedly"))
            if time.monotonic() >= deadline:
                self._reply_events.pop(rid, None)
                return False

    def _read_stdout(self) -> None:
        proc = self._proc
        try:
            for line in proc.stdout:
                if not line.startswith(MARKER):
                    text = line.rstrip()
                    if text:
                        self._log(f"[{self.name}] {text}")
                    continue
                try:
                    reply = json.loads(line[len(MARKER):])
                    rid = int(reply.get("id"))
                except Exception:
                    self._log(f"[{self.name}] unreadable reply {line.rstrip()!r}")
                    continue
                self._replies[rid] = reply
                event = self._reply_events.get(rid)
                if event is not None:
                    event.set()
        except Exception:
            pass
        self._dead.set()
        for event in list(self._reply_events.values()):
            event.set()
        if not self._detach:
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        if self._on_exit is not None:
            try:
                self._on_exit(self)
            except Exception:
                pass

    def _read_stderr(self) -> None:
        try:
            for line in self._proc.stderr:
                text = line.rstrip()
                if text:
                    self._log(f"[{self.name}] {text}")
        except Exception:
            pass

    @property
    def expected_exit(self) -> bool:
        return self._expected_exit


if __name__ == "__main__":
    main()
