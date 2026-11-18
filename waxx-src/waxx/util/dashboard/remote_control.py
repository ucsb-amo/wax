"""Close a dashboard from another PC: a small TCP control port, found by beacon.

Every dashboard window (server or client kind) that calls
``DashboardMainWindow.enable_remote_control`` opens a TCP port and advertises it
by UDP beacon as ``dashboard:<kind>:<host>:<pid>``.  ``kdash``
(:mod:`waxx.util.dashboard.kdash`) lists them and asks one, several or all of
them to close.

A remote close runs the dashboard's own close (the X button): the layout is
saved, every supervised server is asked to stop, COM servers get time to
release their ports, survivors are killed.  The differences from the X button:
the COM-shutdown dialog force-kills what is left after ``com_timeout_s``
(nobody may be at that PC to press its button; a person who is can still press
"Cancel close"), and every step is reported back to the requester.

Protocol (version :data:`PROTOCOL`): the client sends one JSON object on one
line; the server answers with one or more JSON lines and closes the socket.

``{"action": "status"}``
    one line: ``{"ok": true, "kind", "host", "pid", "title", "started",
    "uptime_s", "servers": {id: state}, "closing": bool, "protocol"}``.
``{"action": "close", "by", "owner", "force", "com_timeout_s"}``
    a refusal is one line ``{"ok": false, "refused": "<why>"}``.  Accepted:
    ``{"ok": true, "event": "accepted"}`` then progress lines
    (``{"event": "...", ...}``) and a final ``{"event": "done", "result":
    "closed" | "cancelled", ...}``.
``{"action": "kill", "by", "owner", "force"}``
    hard kill of the dashboard's whole process tree (its servers too), no
    graceful stop: for a dashboard whose window is frozen.  One line
    ``{"ok": true, "event": "killing"}``, then the process ends.

Rules (checked here, in the dashboard, not only by the client):

* an agent (``owner`` "agent") may ask for ``status`` only;
* ``busy_check`` (the server dashboard passes "is a run going?") refuses a
  close or kill while it returns a reason, unless ``force`` is set; a check
  that raises counts as busy;
* a second close while one is under way is refused.

Every request that changes something is logged at WARNING by the dashboard
(its log file and Log panel), with who asked and from which address.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import socket
import socketserver
import threading
import time
from typing import Callable, Optional

_LOG = logging.getLogger("waxx.dashboard.remote_control")

PROTOCOL = 1
BEACON_PREFIX = "dashboard:"
#: Longest request line accepted.
MAX_REQUEST_BYTES = 64 * 1024
#: How long a request may take to arrive once connected.
REQUEST_READ_TIMEOUT_S = 10.0
#: How long the control thread waits for the window (GUI thread) to take a close.
ACK_TIMEOUT_S = 5.0
#: Longest a close stream is kept open waiting for the window to finish.
CLOSE_STREAM_MAX_S = 900.0
#: COM servers' time to exit on a remote close before they are force-killed.
DEFAULT_COM_TIMEOUT_S = 30.0
#: Bounds on what a client may ask for.
COM_TIMEOUT_RANGE_S = (1.0, 600.0)


def beacon_id(kind: str, host: str, pid: int) -> str:
    return f"{BEACON_PREFIX}{kind}:{host}:{int(pid)}"


def parse_beacon_id(sid: str) -> Optional[tuple[str, str, int]]:
    """``(kind, host, pid)`` from a dashboard beacon id; None for any other id."""
    if not isinstance(sid, str) or not sid.startswith(BEACON_PREFIX):
        return None
    parts = sid[len(BEACON_PREFIX):].split(":")
    if len(parts) != 3 or not parts[0] or not parts[1]:
        return None
    try:
        return parts[0], parts[1], int(parts[2])
    except ValueError:
        return None


def clamp_com_timeout(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return DEFAULT_COM_TIMEOUT_S
    if v != v:  # NaN
        return DEFAULT_COM_TIMEOUT_S
    lo, hi = COM_TIMEOUT_RANGE_S
    return min(max(v, lo), hi)


class CloseProgress:
    """Thread-safe channel from the closing window to the waiting client.

    The window calls :meth:`report` for each step and :meth:`finish` once; the
    control thread forwards them.  ``finish`` waits (briefly) until the last
    line is sent, so the process does not exit with it still queued."""

    def __init__(self, request: dict):
        self.request = dict(request)
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._flushed = threading.Event()
        self._finished = False
        self.client_gone = False

    @property
    def com_timeout_s(self) -> float:
        return clamp_com_timeout(self.request.get("com_timeout_s", DEFAULT_COM_TIMEOUT_S))

    @property
    def by(self) -> str:
        return str(self.request.get("by") or "?")

    def report(self, event: str, **fields) -> None:
        if self._finished:
            return
        msg = {"event": str(event), "t": time.time()}
        msg.update(fields)
        self._q.put(msg)

    def finish(self, result: str, wait_s: float = 2.0, **fields) -> None:
        if self._finished:
            return
        self.report("done", result=str(result), **fields)
        self._finished = True
        if not self.client_gone:
            self._flushed.wait(timeout=max(0.0, float(wait_s)))

    # -- the control thread's side --------------------------------------------
    def next(self, timeout: float) -> Optional[dict]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def mark_flushed(self, client_gone: bool = False) -> None:
        if client_gone:
            self.client_gone = True
        self._flushed.set()


def _send(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj, default=repr) + "\n").encode("utf-8"))


def _read_request(sock: socket.socket) -> dict:
    sock.settimeout(REQUEST_READ_TIMEOUT_S)
    buf = bytearray()
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > MAX_REQUEST_BYTES:
            raise ValueError(f"request longer than {MAX_REQUEST_BYTES} bytes")
    line = bytes(buf).split(b"\n", 1)[0].strip()
    if not line:
        raise ValueError("empty request")
    obj = json.loads(line.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("request is not a JSON object")
    return obj


class _Handler(socketserver.BaseRequestHandler):
    server: "_TCPServer"

    def handle(self) -> None:
        ctl = self.server.controller
        sock = self.request
        peer = self.client_address[0] if self.client_address else "?"
        try:
            req = _read_request(sock)
        except Exception as exc:                          # noqa: BLE001
            try:
                _send(sock, {"ok": False, "error": f"bad request: {exc}"})
            except OSError:
                pass
            return
        req["_peer"] = peer
        try:
            ctl._dispatch(req, sock)
        except OSError:
            pass                                          # the client went away
        except Exception as exc:                          # noqa: BLE001
            _LOG.exception("remote control request failed: %r", req.get("action"))
            try:
                _send(sock, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass


class _TCPServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False
    controller: "DashboardControlServer"


class DashboardControlServer:
    """The control port of one dashboard.

    ``info()`` -> dict: the status snapshot (called from the control thread;
    must not touch Qt widgets).  ``on_close(progress)`` -> None (accepted) or a
    refusal string; called from the control thread, must hand the close to the
    GUI thread and return.  ``on_kill(request)``: hard-kill the process tree
    (called after the reply is sent).  ``busy_check()`` -> None or a reason the
    machine must not lose this dashboard now.  ``advertise=False``: no beacon
    (tests)."""

    def __init__(self, kind: str, *, info: Callable[[], dict],
                 on_close: Callable[[CloseProgress], Optional[str]],
                 on_kill: Callable[[dict], None],
                 busy_check: Optional[Callable[[], Optional[str]]] = None,
                 bind_host: str = "0.0.0.0", port: int = 0, advertise: bool = True,
                 host: Optional[str] = None, pid: Optional[int] = None):
        self.kind = str(kind)
        self.host = host or socket.gethostname()
        self.pid = int(pid if pid is not None else os.getpid())
        self._info = info
        self._on_close = on_close
        self._on_kill = on_kill
        self._busy_check = busy_check
        self._bind_host = bind_host
        self._port = int(port)
        self._advertise = bool(advertise)
        self._server: Optional[_TCPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._beacon = None
        self._lock = threading.Lock()
        self._busy = False     # a close/kill is under way

    @property
    def beacon_id(self) -> str:
        return beacon_id(self.kind, self.host, self.pid)

    @property
    def address(self) -> Optional[tuple[str, int]]:
        return self._server.server_address[:2] if self._server is not None else None

    def start(self) -> None:
        if self._server is not None:
            return
        try:
            srv = _TCPServer((self._bind_host, self._port), _Handler)
        except OSError as exc:
            if self._bind_host in ("0.0.0.0", ""):
                raise
            _LOG.warning("remote control: cannot bind %s (%s); listening on all addresses",
                         self._bind_host, exc)
            srv = _TCPServer(("0.0.0.0", self._port), _Handler)
        srv.controller = self
        self._server = srv
        self._thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.25},
                                        name=f"DashboardControl-{self.kind}", daemon=True)
        self._thread.start()
        port = srv.server_address[1]
        if self._advertise:
            from beacon.discovery.server import NetServer  # noqa: PLC0415
            beacon = NetServer()
            NetServer.__init__(beacon, self.beacon_id, port,
                               beacon_extra={"kind": self.kind, "host": self.host,
                                             "pid": self.pid, "proto": PROTOCOL})
            beacon._start_beacon()
            self._beacon = beacon
        _LOG.info("remote control on port %d as %s%s", port, self.beacon_id,
                  "" if self._advertise else " (not advertised)")

    def stop(self) -> None:
        beacon, self._beacon = self._beacon, None
        if beacon is not None:
            try:
                beacon._stop_beacon()
            except Exception:                             # noqa: BLE001
                pass
        srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:                             # noqa: BLE001
                pass

    def clear_closing(self) -> None:
        """The window stayed open (a person pressed Cancel close): accept a
        new close request again."""
        with self._lock:
            self._busy = False

    # -- requests -------------------------------------------------------------
    def _status(self) -> dict:
        try:
            info = dict(self._info() or {})
        except Exception as exc:                          # noqa: BLE001
            info = {"info_error": repr(exc)}
        info.update({"ok": True, "protocol": PROTOCOL, "kind": self.kind, "host": self.host,
                     "pid": self.pid})
        info.setdefault("closing", self._busy)
        return info

    def _refusal(self, req: dict) -> Optional[str]:
        if str(req.get("owner") or "").strip().lower() == "agent":
            return "an agent may not close or kill a dashboard; ask a person"
        if self._busy:
            return "this dashboard is already closing"
        if self._busy_check is not None and not req.get("force"):
            try:
                reason = self._busy_check()
            except Exception as exc:                      # noqa: BLE001
                reason = f"could not check whether the machine is busy ({exc})"
            if reason:
                return f"{reason} -- --force closes it anyway"
        return None

    def _dispatch(self, req: dict, sock: socket.socket) -> None:
        action = req.get("action")
        if action == "status":
            _send(sock, self._status())
            return
        if action not in ("close", "kill"):
            _send(sock, {"ok": False, "error": f"unknown action {action!r}"})
            return
        who = f"{req.get('by') or '?'} ({req.get('owner') or '?'}) from {req.get('_peer')}"
        with self._lock:
            refusal = self._refusal(req)
            if refusal is None:
                self._busy = True
        if refusal is not None:
            _LOG.warning("remote %s by %s REFUSED: %s", action, who, refusal)
            _send(sock, {"ok": False, "refused": refusal})
            return
        if action == "kill":
            _LOG.critical("remote KILL by %s (force=%s): killing this dashboard's process "
                          "tree, servers included, with no graceful stop",
                          who, bool(req.get("force")))
            try:
                _send(sock, {"ok": True, "event": "killing", "pid": self.pid})
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._on_kill(req)
            return
        _LOG.warning("remote close by %s (force=%s, COM timeout %.0f s)", who,
                     bool(req.get("force")), clamp_com_timeout(req.get("com_timeout_s",
                                                                       DEFAULT_COM_TIMEOUT_S)))
        progress = CloseProgress(req)
        try:
            refusal = self._on_close(progress)
        except Exception as exc:                          # noqa: BLE001
            refusal = f"the window could not start the close ({exc})"
        if refusal is not None:
            with self._lock:
                self._busy = False
            _LOG.warning("remote close by %s not started: %s", who, refusal)
            _send(sock, {"ok": False, "refused": refusal})
            return
        self._stream(progress, sock)

    def _stream(self, progress: CloseProgress, sock: socket.socket) -> None:
        try:
            _send(sock, {"ok": True, "event": "accepted"})
        except OSError:
            progress.mark_flushed(client_gone=True)
            return
        deadline = time.monotonic() + CLOSE_STREAM_MAX_S
        while time.monotonic() < deadline:
            msg = progress.next(timeout=0.5)
            if msg is None:
                continue
            try:
                _send(sock, msg)
            except OSError:
                progress.mark_flushed(client_gone=True)
                return
            if msg.get("event") == "done":
                if msg.get("result") != "closed":
                    with self._lock:
                        self._busy = False       # the window stayed open
                progress.mark_flushed()
                return
        progress.mark_flushed(client_gone=True)


__all__ = [
    "PROTOCOL", "BEACON_PREFIX", "DEFAULT_COM_TIMEOUT_S", "CloseProgress",
    "DashboardControlServer", "beacon_id", "parse_beacon_id", "clamp_com_timeout",
]
