"""Composite-op queue held by the monitor server.

The server is the meeting point between the GUIs that request composite ops
and the monitor experiment that runs them (see
:mod:`waxx.util.device_state.composite`).  This class is its bookkeeping,
free of sockets and Qt so it can be tested on its own:

* ``register`` -- the monitor experiment, at loop start, says which ops it
  compiled (name -> index, signature, argument order).  Nothing is accepted
  before that, and a request whose signature differs from the registered one
  is refused: the GUI and the monitor were built from different definitions.
* ``submit`` -- a GUI request.  Arguments are packed into the kernel's 8
  floats here, in the registered order.
* ``pop`` -- the monitor takes queued requests on its 10 Hz poll.
* ``done`` -- the monitor reports outcomes; they become results the server
  broadcasts and keeps for ``status`` queries.
* ``expire`` -- a request not taken within ``ttl_s`` fails as expired, so an
  op queued while the monitor was busy dying can never fire minutes later.
* ``unregister`` -- the monitor stopped: queued requests expire, taken but
  unreported ones are *lost* (they may or may not have run, and the result
  says exactly that).

All methods are thread safe: the TCP responder thread and the owner's Qt
thread both call in.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Callable, Mapping

from waxx.util.device_state.composite import (
    N_OP_ARGS, OP_EXPIRED, OP_LOST, OP_OK, OP_STATUS_TEXT, status_text,
)

#: A queued op not taken by the monitor within this long fails as expired.
#: The monitor polls at 10 Hz, so a healthy one takes ops within ~0.1 s.
OP_QUEUE_TTL_S = 3.0
#: Ops handed to the monitor per poll.
OP_POP_MAX = 16
#: Finished results kept for ``status`` queries.
OP_RESULTS_KEPT = 256
#: Request ids remembered for duplicate suppression (a client that retries
#: after a lost reply must not queue the op twice).
OP_RIDS_KEPT = 512


class OpQueue:
    def __init__(self, ttl_s: float = OP_QUEUE_TTL_S,
                 clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time):
        self.ttl_s = float(ttl_s)
        self._clock = clock
        self._wall = wall_clock
        self._lock = threading.Lock()
        self._registration: dict | None = None
        self._queue: deque = deque()
        self._dispatched: dict[int, dict] = {}
        self._results: OrderedDict[int, dict] = OrderedDict()
        self._rids: OrderedDict[str, int] = OrderedDict()
        # While the monitor is playing out a long op (a 10 s ramp) it cannot
        # poll; queued ops age from when it is free again, not from submission.
        self._busy_until = float("-inf")
        # Starts at the epoch second, like the device-state version, so a
        # server restart never reissues a sequence number a GUI still waits
        # on.  The monitor kernel carries it as an int32 (fine until 2038).
        self._seq = int(self._wall())
        #: Last host-side state each device reported (e.g. AWG connected),
        #: cleared when the monitor stops -- it described that process.
        self.device_state: dict[str, dict] = {}

    # --- registration ---------------------------------------------------------

    def register(self, registration: Mapping) -> dict:
        ops = registration.get("ops")
        if not isinstance(ops, Mapping) or not ops:
            return {"status": "error", "msg": "registration has no ops"}
        clean = {}
        for name, info in ops.items():
            try:
                clean[str(name)] = {
                    "index": int(info["index"]),
                    "sig": str(info["sig"]),
                    "args": [str(a) for a in info.get("args", [])],
                    "payload": [str(p) for p in info.get("payload", [])],
                }
            except (KeyError, TypeError, ValueError):
                return {"status": "error", "msg": f"bad registration entry for {name!r}"}
            if len(clean[str(name)]["args"]) > N_OP_ARGS:
                return {"status": "error", "msg": f"{name}: too many args"}
        with self._lock:
            self._registration = {
                "session": str(registration.get("session", "")),
                "hash": str(registration.get("hash", "")),
                "ops": clean,
                "since": self._wall(),
            }
            self.device_state.clear()
        return {"status": "ok", "count": len(clean)}

    def unregister(self, reason: str) -> list[dict]:
        """The monitor stopped.  Returns the results this produced (to broadcast)."""
        with self._lock:
            if self._registration is None and not self._queue and not self._dispatched:
                return []
            self._registration = None
            self.device_state.clear()
            finished = []
            while self._queue:
                req = self._queue.popleft()
                finished.append(self._finish(req, OP_EXPIRED, f"monitor stopped ({reason})"))
            for req in list(self._dispatched.values()):
                finished.append(self._finish(req, OP_LOST, f"monitor stopped ({reason})"))
            self._dispatched.clear()
            return finished

    @property
    def registered(self) -> bool:
        return self._registration is not None

    def info(self) -> dict:
        with self._lock:
            reg = self._registration
            return {
                "registered": reg is not None,
                "count": len(reg["ops"]) if reg else 0,
                "hash": reg["hash"] if reg else "",
                "session": reg["session"] if reg else "",
                "queued": len(self._queue),
                "running": len(self._dispatched),
                "busy_s": max(self._busy_until - self._clock(), 0.),
            }

    # --- requests -------------------------------------------------------------

    def submit(self, request: Mapping, client: str = "", origin: str = "gui") -> dict:
        name = request.get("op")
        rid = request.get("rid")
        with self._lock:
            if rid and str(rid) in self._rids:
                # A retry of a request already queued (its reply was lost).
                return {"status": "ok", "seq": self._rids[str(rid)], "duplicate": True}
            reg = self._registration
            if reg is None:
                return {"status": "error",
                        "msg": "the monitor has not registered any composite ops "
                               "(it is not running, or is still starting)"}
            info = reg["ops"].get(name)
            if info is None:
                return {"status": "error",
                        "msg": f"the running monitor has no op {name!r} -- restart the "
                               "monitor to load the current composite definitions"}
            sig = request.get("sig")
            if sig != info["sig"]:
                return {"status": "error",
                        "msg": f"{name}: the running monitor was built from a different "
                               "definition of this op than this GUI -- restart the monitor "
                               "(or update this GUI) so both use the same code"}
            args = request.get("args") or {}
            if not isinstance(args, Mapping):
                return {"status": "error", "msg": "args must be an object"}
            packed = []
            for arg in info["args"]:
                if arg not in args:
                    return {"status": "error", "msg": f"{name}: missing argument {arg!r}"}
                try:
                    value = float(args[arg])
                except (TypeError, ValueError):
                    return {"status": "error", "msg": f"{name}: {arg} is not a number"}
                if not math.isfinite(value):
                    return {"status": "error", "msg": f"{name}: {arg} is not finite"}
                packed.append(value)
            packed += [0.0] * (N_OP_ARGS - len(packed))
            payload = request.get("payload") or {}
            if not isinstance(payload, Mapping):
                return {"status": "error", "msg": "payload must be an object"}
            missing = [p for p in info["payload"] if p not in payload]
            if missing:
                return {"status": "error", "msg": f"{name}: missing payload {missing}"}
            self._seq += 1
            req = {
                "seq": self._seq,
                "op": name,
                "index": info["index"],
                "sig": info["sig"],
                "args": packed,
                "arg_values": {a: float(args[a]) for a in info["args"]},
                "payload": {p: payload[p] for p in info["payload"]},
                "client": str(client or request.get("client", "")),
                "operator": str(request.get("operator", "") or ""),
                "origin": str(origin or "gui"),
                "t_submit": self._clock(),
                "wall_submit": self._wall(),
            }
            self._queue.append(req)
            if rid:
                self._rids[str(rid)] = req["seq"]
                while len(self._rids) > OP_RIDS_KEPT:
                    self._rids.popitem(last=False)
            return {"status": "ok", "seq": req["seq"], "queued": len(self._queue)}

    def set_busy(self, seconds: float) -> None:
        """The monitor will not poll for ``seconds`` (it is playing out an op)."""
        with self._lock:
            self._busy_until = max(self._busy_until, self._clock() + max(float(seconds), 0.))

    def result(self, seq: int) -> dict | None:
        with self._lock:
            r = self._results.get(int(seq))
            return dict(r) if r is not None else None

    def pop(self, max_n: int = OP_POP_MAX) -> list[dict]:
        """Queued requests for the monitor, oldest first; marks them taken."""
        out = []
        with self._lock:
            if self._registration is None:
                return out
            while self._queue and len(out) < max_n:
                req = self._queue.popleft()
                req["t_dispatch"] = self._clock()
                self._dispatched[req["seq"]] = req
                out.append({k: req[k] for k in
                            ("seq", "op", "index", "sig", "args", "arg_values", "payload")})
        return out

    def done(self, reports) -> list[dict]:
        """Outcomes from the monitor: ``[{"seq", "status", "message", "state",
        "device"}]``.  Returns the results to broadcast."""
        finished = []
        with self._lock:
            for rep in reports or []:
                try:
                    seq = int(rep["seq"])
                    status = int(rep["status"])
                except (KeyError, TypeError, ValueError):
                    continue
                req = self._dispatched.pop(seq, None)
                if req is None:
                    # Rejected by the monitor before the kernel saw it, or a
                    # duplicate report: record what we can.
                    req = {"seq": seq, "op": rep.get("op", "?"), "client": "",
                           "t_submit": None, "wall_submit": None}
                state = rep.get("state")
                device = rep.get("device")
                if isinstance(state, Mapping) and device:
                    self.device_state[str(device)] = dict(state)
                finished.append(self._finish(req, status, str(rep.get("message", "") or ""),
                                             state=state, device=device))
        return finished

    def expire(self) -> list[dict]:
        """Fail queued requests older than the TTL.  Returns their results."""
        now = self._clock()
        finished = []
        with self._lock:
            while self._queue and \
                    now - max(self._queue[0]["t_submit"], self._busy_until) > self.ttl_s:
                req = self._queue.popleft()
                finished.append(self._finish(
                    req, OP_EXPIRED,
                    f"not taken by the monitor within {self.ttl_s:g} s"))
        return finished

    def status(self, seq: int) -> dict:
        with self._lock:
            if seq in self._results:
                return {"status": "ok", "state": "done", "result": dict(self._results[seq])}
            if seq in self._dispatched:
                return {"status": "ok", "state": "running"}
            if any(r["seq"] == seq for r in self._queue):
                return {"status": "ok", "state": "queued"}
        return {"status": "ok", "state": "unknown"}

    # --- internal ---------------------------------------------------------------

    def _finish(self, req: dict, status: int, message: str, state=None, device=None) -> dict:
        """Record a final result (lock held by the caller)."""
        now = self._clock()
        t_submit = req.get("t_submit")
        result = {
            "type": "op_result",
            "seq": req["seq"],
            "op": req.get("op", "?"),
            "status": int(status),
            "ok": int(status) == OP_OK,
            "status_text": OP_STATUS_TEXT.get(int(status), f"status {status}"),
            "message": message,
            "text": status_text(status, message) if status != OP_OK else "done",
            "client": req.get("client", ""),
            "operator": req.get("operator", ""),
            "origin": req.get("origin", ""),
            "arg_values": dict(req.get("arg_values") or {}),
            "elapsed": (now - t_submit) if t_submit is not None else None,
            "wall_submit": req.get("wall_submit"),
            "wall_done": self._wall(),
        }
        if device:
            result["device"] = str(device)
        if isinstance(state, Mapping):
            result["state"] = dict(state)
        self._results[req["seq"]] = result
        while len(self._results) > OP_RESULTS_KEPT:
            self._results.popitem(last=False)
        return result
