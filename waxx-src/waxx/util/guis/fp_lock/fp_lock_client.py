"""FPLockClient: discovery + one JSON request per TCP connection (see
:mod:`fp_lock_server`).  Short timeouts: a panel never blocks on it for long,
and calls it off the GUI thread anyway."""

from __future__ import annotations

import json
import logging
import socket
from typing import Any, Optional

from beacon.discovery.client import NetClient

__all__ = ["FPLockClient"]

_LOG = logging.getLogger("waxx.fp_lock.client")


class FPLockClient(NetClient):

    def __init__(self, server_id: str, discovery_timeout: float = 3.0, timeout: float = 1.5):
        super().__init__(server_id, discovery_timeout=discovery_timeout)
        self.server_id = server_id
        self.timeout = float(timeout)

    @classmethod
    def at(cls, host: str, port: int, server_id: str = "fp_lock", timeout: float = 1.5):
        """A client for a known address (no discovery; tests, scripts)."""
        self = cls.__new__(cls)
        self.server_id = server_id
        self._waxx_server_id = server_id
        self.reconnect_generation = 0
        self._fixed_address = True          # never rediscovers (no beacon listening)
        self.host, self.port = host, int(port)
        self.timeout = float(timeout)
        return self

    # -- wire ---------------------------------------------------------------

    #: Timeout for requests that wait on the board (sets, refresh, raw); polls
    #: use ``timeout``.
    ACTION_TIMEOUT_S = 15.0

    def request(self, cmd: str, *, timeout: Optional[float] = None, **fields: Any) -> dict:
        """Send one request; the reply dict.  Raises on a network failure."""
        payload = (json.dumps({"cmd": cmd, **fields}) + "\n").encode("utf-8")
        t = self.timeout if timeout is None else float(timeout)
        try:
            return self._roundtrip(payload, t)
        except OSError:
            # the server may have restarted on a new port
            if getattr(self, "_fixed_address", False) or not self._rediscover(timeout=1.0):
                raise
            return self._roundtrip(payload, t)

    def _roundtrip(self, payload: bytes, timeout: float) -> dict:
        with socket.create_connection((self.host, self.port), timeout=timeout) as s:
            s.sendall(payload)
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        raw = buf.decode("utf-8", errors="replace").strip()
        try:
            reply = json.loads(raw)
        except json.JSONDecodeError as exc:
            _LOG.warning("fp lock: non-JSON reply: %r", raw[:200])
            return {"status": "error", "message": f"bad reply: {exc}"}
        return reply if isinstance(reply, dict) else {"status": "error", "message": "bad reply"}

    # -- API ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        return self.request("get_snapshot")

    def get_history(self, since: int = 0, max_rows: Optional[int] = None) -> dict:
        fields = {"since": int(since)}
        if max_rows is not None:
            fields["max"] = int(max_rows)
        return self.request("get_history", **fields)

    def get_text(self, since: int = 0) -> dict:
        return self.request("get_text", since=int(since))

    def set_value(self, name: str, value) -> dict:
        return self.request("set", name=name, value=value, timeout=self.ACTION_TIMEOUT_S)

    def lock(self, on: bool) -> dict:
        return self.request("lock", on=bool(on), timeout=self.ACTION_TIMEOUT_S)

    def reset_setpoint(self) -> dict:
        return self.request("reset_setpoint", timeout=self.ACTION_TIMEOUT_S)

    def clear(self) -> dict:
        return self.request("clear", timeout=self.ACTION_TIMEOUT_S)

    def raw(self, line: str) -> dict:
        return self.request("raw", line=line, timeout=self.ACTION_TIMEOUT_S)

    def refresh_settings(self) -> dict:
        return self.request("refresh_settings", timeout=self.ACTION_TIMEOUT_S)

    def com_connect(self) -> tuple[bool, str]:
        return self._com_reply(self.request("com_connect"))

    def com_disconnect(self) -> tuple[bool, str]:
        return self._com_reply(self.request("com_disconnect"))

    def request_shutdown(self) -> bool:
        return self.request("shutdown").get("status") == "ok"

    @staticmethod
    def _com_reply(reply: dict) -> tuple[bool, str]:
        ok = reply.get("status") == "ok"
        return ok, str(reply.get("message") or ("" if ok else reply.get("status")))
