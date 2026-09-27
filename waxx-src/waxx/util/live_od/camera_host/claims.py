"""Borrowing a camera from the beacon camera server that has it, and keeping it.

A Basler camera is listed by the beacon camera server of the PC it is plugged
into, which opens it whenever a viewer looks at it.  Before liveOD's camera
host opens one for a run, it asks that server to let go of it:

    RELINQUISH_CAMERA{camera_ids, holder, ttl_s}   the server closes the device
                                                   (verified) and keeps it
                                                   reserved for this holder
    RETURN_CAMERA{camera_ids, holder_id}           the holder gives it back

A reservation lapses unless renewed, so ``ReservationKeeper`` renews every
claim every ``ttl_s / 3`` (RELINQUISH is idempotent per holder and doubles as
the renewal).  If that server restarted in the meantime -- a new ``instance``
in its reply -- the same message grants the reservation anew, and the keeper
says so in the log; if it moved to another port, the keeper finds it again.

Only protocol v2 (``beacon.camera.protocol``): HELLO first, as a single
frame; a server that answers in the old pickle protocol is recognised by its
first byte and never unpickled -- such a server cannot relinquish, and is
skipped with a warning.

``resolver(camera_id) -> [ServerRef]`` says which servers list a camera.  The
default, ``DirectoryResolver``, reads this process's discovery cache (beacons
it has already heard; no wait, nothing is sent over UDP) and asks each server
for its list over TCP.  Tests pass their own.
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import zmq

from beacon.camera import protocol
from beacon.camera.reservations import Holder

logger = logging.getLogger("waxx.live_od.camera_host")

V2_PREFIX = "camera_server:"
LEGACY_PREFIX = "basler_server:"
CLIENT_LABEL = "liveOD camera host"
#: RELINQUISH ttl; renewed every ttl / 3
DEFAULT_TTL_S = 30.0


class ClaimError(RuntimeError):
    """A camera could not be borrowed from the server that has it."""


class ClaimRefused(ClaimError):
    """The server that has the camera refused to let it go (``holder`` says
    who has it, when the server said)."""

    def __init__(self, message: str, *, server_id: str = "", holder: Optional[dict] = None,
                 code: str = ""):
        super().__init__(message)
        self.server_id = server_id
        self.holder = dict(holder or {})
        self.code = code


@dataclass(frozen=True)
class ServerRef:
    server_id: str
    host: str
    port: int

    def describe(self) -> str:
        return f"{self.server_id} at {self.host}:{self.port}"


@dataclass
class Claim:
    camera_id: str
    server: ServerRef
    instance: str = ""
    since: float = field(default_factory=time.time)
    renewed: float = field(default_factory=time.monotonic)
    error: str = ""

    def to_dict(self) -> dict:
        return {"camera_id": self.camera_id, "server_id": self.server.server_id,
                "host": self.server.host, "port": self.server.port, "instance": self.instance,
                "since": self.since, "error": self.error}


def _client_info(label: str) -> dict:
    return {"label": str(label), "host": socket.gethostname(), "pid": os.getpid()}


def v2_request(ctx: zmq.Context, host: str, port: int, header: dict, timeout_s: float,
               label: str = CLIENT_LABEL) -> dict:
    """One v2 command on a fresh REQ socket, after a single-frame HELLO.

    Returns the reply header (with the HELLO reply under ``"_hello"``).  Raises
    ``TimeoutError`` when nothing comes within ``timeout_s`` (per message), and
    ``ClaimError`` for a server that speaks only the old protocol (its reply is
    recognised by its first byte, never unpickled)."""
    ms = max(1, int(float(timeout_s) * 1000))
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, ms)
    sock.setsockopt(zmq.SNDTIMEO, ms)
    sock.setsockopt(zmq.LINGER, 0)
    where = f"{host}:{port}"
    try:
        sock.connect(f"tcp://{host}:{port}")
        sock.send(protocol.encode_single({"cmd": "HELLO", "protocol": protocol.PROTOCOL_VERSION,
                                          "client": _client_info(label)}))
        first = sock.recv_multipart()
        if protocol.sniff(first[0]) != "v2":
            raise ClaimError(f"the camera server at {where} speaks only the old (pickle) "
                             f"protocol, which cannot relinquish a camera; update beacon there")
        hello, _ = protocol.decode(first)
        sock.send_multipart(protocol.encode(header))
        head, _ = protocol.decode(sock.recv_multipart())
        if not isinstance(head, dict):
            raise ClaimError(f"{header.get('cmd')} to {where}: the reply is not an object")
        head["_hello"] = hello
        return head
    except zmq.Again:
        raise TimeoutError(f"{header.get('cmd')} to {where}: no reply within "
                           f"{float(timeout_s):g} s") from None
    finally:
        sock.close(0)


class DirectoryResolver:
    """``resolver(camera_id) -> [ServerRef]``: the v2 camera servers that list
    ``camera_id``, except ours (``exclude_server_ids``).

    Server addresses come from this process's discovery cache
    (``beacon.discovery.client.discover_entries``: beacons heard within
    ``max_age_s``; nothing is sent over UDP), or from ``discover(prefix) ->
    {server_id: (host, port)}`` when given (tests)."""

    def __init__(self, exclude_server_ids=(), *, max_age_s: float = 5.0,
                 request_timeout_s: float = 0.5,
                 discover: Optional[Callable[[str], dict]] = None) -> None:
        self.exclude = set(exclude_server_ids or ())
        self.max_age_s = float(max_age_s)
        self.request_timeout_s = float(request_timeout_s)
        self._discover = discover
        self._ctx = zmq.Context()
        self._warned: set = set()

    def _addresses(self) -> dict:
        """{(host, port): [server_id, ...]}"""
        found: dict = {}
        for prefix in (V2_PREFIX, LEGACY_PREFIX):
            if self._discover is not None:
                got = {sid: (str(a[0]), int(a[1])) for sid, a in dict(self._discover(prefix) or {}).items()}
            else:
                from beacon.discovery import client as dc
                got = {sid: (str(e.ip), int(e.port))
                       for sid, e in dc.discover_entries(prefix, max_age=self.max_age_s).items()}
            for sid, addr in got.items():
                found.setdefault(addr, []).append(sid)
        return found

    def __call__(self, camera_id: str) -> list:
        out = []
        for (host, port), sids in self._addresses().items():
            if self.exclude & set(sids):
                continue
            sid = sorted(sids, key=lambda s: (not s.startswith(V2_PREFIX), s))[0]
            try:
                head = v2_request(self._ctx, host, port, {"cmd": "LIST_CAMERAS"},
                                  self.request_timeout_s)
            except Exception as exc:
                key = (sid, type(exc).__name__)
                if key not in self._warned:
                    self._warned.add(key)
                    logger.warning(f"camera host: {sid} at {host}:{port} not usable for "
                                   f"claiming cameras: {exc}")
                continue
            ids = {str(c.get("camera_id")) for c in head.get("cameras", ()) if isinstance(c, dict)}
            if camera_id in ids:
                out.append(ServerRef(sid, host, int(port)))
        return out

    def close(self) -> None:
        try:
            self._ctx.term()
        except Exception:
            pass


class ReservationKeeper:
    """Claims cameras from the servers that list them, renews the claims,
    gives them back.  Thread-safe; its renewal thread starts with the first
    claim and ends at ``shutdown``."""

    def __init__(self, resolver: Callable[[str], list], holder: Holder, *,
                 ttl_s: float = DEFAULT_TTL_S, request_timeout_s: float = 3.0) -> None:
        self._resolver = resolver
        self.holder = holder
        self.ttl_s = float(ttl_s)
        self.request_timeout_s = float(request_timeout_s)
        self._ctx = zmq.Context()
        self._lock = threading.RLock()
        self._claims: dict = {}            # camera_id -> [Claim]
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    # -- public ----------------------------------------------------------------

    def holds(self, camera_id: str) -> bool:
        with self._lock:
            return bool(self._claims.get(camera_id))

    def claims(self) -> dict:
        """{camera_id: [claim dicts]}"""
        with self._lock:
            return {cid: [c.to_dict() for c in cs] for cid, cs in self._claims.items() if cs}

    def claim(self, camera_id: str, timeout_s: float = 3.0) -> list:
        """Relinquish ``camera_id`` from every server that lists it (bounded by
        ``timeout_s`` per server, plus the lookup).  Returns the claims ([] when
        no server lists it: nothing to borrow).  Raises ``ClaimRefused`` naming
        who holds it, or ``ClaimError``/``TimeoutError``."""
        if self._closed:
            raise ClaimError("camera host is shutting down; nothing is claimed")
        with self._lock:
            if self._claims.get(camera_id):
                return list(self._claims[camera_id])
        servers = list(self._resolver(camera_id) or ())
        got = []
        for srv in servers:
            try:
                got.append(self._relinquish(camera_id, srv, timeout_s))
            except TimeoutError:
                # the server may have granted it without answering in time: undo
                self._return_quietly(camera_id, srv)
                for c in got:
                    self._return_quietly(camera_id, c.server)
                raise
            except Exception:
                for c in got:
                    self._return_quietly(camera_id, c.server)
                raise
        if got:
            with self._lock:
                self._claims[camera_id] = got
            self._ensure_thread()
            for c in got:
                logger.info(f"camera host: {camera_id} borrowed from {c.server.describe()} "
                            f"(reserved for liveOD, renewed every {self.ttl_s / 3:.0f} s)")
        return got

    def release(self, camera_id: str, timeout_s: float = 1.0) -> dict:
        """RETURN_CAMERA to every server ``camera_id`` was borrowed from.
        ``{server_id: "" | error}``; the claim is forgotten either way (an
        unanswered reservation lapses by itself within its ttl)."""
        with self._lock:
            claims = self._claims.pop(camera_id, [])
        out = {}
        for c in claims:
            try:
                head = v2_request(self._ctx, c.server.host, c.server.port,
                                  {"cmd": "RETURN_CAMERA", "camera_ids": [camera_id],
                                   "holder_id": self.holder.holder_id}, timeout_s)
                res = (head.get("cameras") or {}).get(camera_id) or {}
                if head.get("ok") and res.get("ok", True):
                    out[c.server.server_id] = ""
                    logger.info(f"camera host: {camera_id} given back to {c.server.describe()}")
                else:
                    out[c.server.server_id] = str(res.get("error") or head.get("error") or "refused")
                    logger.warning(f"camera host: RETURN of {camera_id} to {c.server.describe()} "
                                   f"refused: {out[c.server.server_id]}")
            except Exception as exc:
                out[c.server.server_id] = f"{type(exc).__name__}: {exc}"
                logger.warning(f"camera host: RETURN of {camera_id} to {c.server.describe()} "
                               f"failed ({exc}); its reservation lapses within {self.ttl_s:.0f} s")
        return out

    def shutdown(self, timeout_s: float = 1.0) -> dict:
        """Give every borrowed camera back (``timeout_s`` each), stop renewing."""
        self._closed = True
        self._stop.set()
        with self._lock:
            ids = list(self._claims)
        out = {cid: self.release(cid, timeout_s) for cid in ids}
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(max(0.1, timeout_s))
        for close in (getattr(self._resolver, "close", None), self._ctx.term):
            try:
                if callable(close):
                    close()
            except Exception:
                pass
        return out

    # -- internals ---------------------------------------------------------------

    def _relinquish(self, camera_id: str, srv: ServerRef, timeout_s: float) -> Claim:
        head = v2_request(self._ctx, srv.host, srv.port,
                          {"cmd": "RELINQUISH_CAMERA", "camera_ids": [camera_id],
                           "holder": self.holder.to_wire(), "ttl_s": self.ttl_s}, timeout_s)
        res = (head.get("cameras") or {}).get(camera_id) or {}
        if not (head.get("ok") and res.get("ok")):
            holder = res.get("holder") if isinstance(res.get("holder"), dict) else {}
            who = holder.get("label") or holder.get("server_id") or holder.get("holder_id") or ""
            where = holder.get("host") or ""
            by = f" (held by {who}{' on ' + where if where else ''})" if who else ""
            raise ClaimRefused(
                f"{srv.server_id} would not let go of {camera_id}{by}: "
                f"{res.get('error') or head.get('error') or 'refused'} "
                f"[{res.get('code') or head.get('code') or 'refused'}]",
                server_id=srv.server_id, holder=holder, code=str(res.get("code") or ""))
        inst = str((head.get("status") or {}).get("instance")
                   or (head.get("_hello") or {}).get("instance") or "")
        return Claim(camera_id, srv, instance=inst)

    def _return_quietly(self, camera_id: str, srv: ServerRef) -> None:
        try:
            v2_request(self._ctx, srv.host, srv.port,
                       {"cmd": "RETURN_CAMERA", "camera_ids": [camera_id],
                        "holder_id": self.holder.holder_id}, 1.0)
        except Exception:
            pass

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._renew_loop, daemon=True,
                                            name="CameraHost-claims")
            self._thread.start()

    def _renew_loop(self) -> None:
        while not self._stop.wait(max(0.5, self.ttl_s / 3.0)):
            with self._lock:
                items = [(cid, list(cs)) for cid, cs in self._claims.items()]
            for cid, claims in items:
                for c in claims:
                    if self._stop.is_set():
                        return
                    self._renew(cid, c)

    def _renew(self, camera_id: str, c: Claim) -> None:
        try:
            try:
                fresh = self._relinquish(camera_id, c.server, min(self.request_timeout_s, 2.0))
            except (TimeoutError, zmq.ZMQError, ClaimError) as exc:
                if isinstance(exc, ClaimRefused):
                    raise
                # the server may have moved (restarted on another port): look again
                moved = [s for s in (self._resolver(camera_id) or ())
                         if s.server_id == c.server.server_id and s != c.server]
                if not moved:
                    raise
                logger.warning(f"camera host: {c.server.server_id} moved to "
                               f"{moved[0].host}:{moved[0].port}; claiming {camera_id} there")
                c.server = moved[0]
                fresh = self._relinquish(camera_id, c.server, min(self.request_timeout_s, 2.0))
        except Exception as exc:
            if not c.error:
                logger.warning(f"camera host: could not renew the claim on {camera_id} at "
                               f"{c.server.describe()}: {exc} (retrying every "
                               f"{self.ttl_s / 3:.0f} s)")
            c.error = str(exc)
            return
        if c.instance and fresh.instance and fresh.instance != c.instance:
            logger.warning(f"camera host: {c.server.server_id} restarted (instance "
                           f"{c.instance} -> {fresh.instance}); {camera_id} relinquished again")
        elif c.error:
            logger.info(f"camera host: claim on {camera_id} at {c.server.describe()} renewed again")
        c.instance = fresh.instance or c.instance
        c.renewed = time.monotonic()
        c.error = ""


__all__ = ["ReservationKeeper", "DirectoryResolver", "ServerRef", "Claim", "ClaimError",
           "ClaimRefused", "v2_request", "DEFAULT_TTL_S"]
