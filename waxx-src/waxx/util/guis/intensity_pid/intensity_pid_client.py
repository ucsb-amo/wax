"""IntensityPIDClient: discovery + one JSON request per TCP connection (see
:mod:`intensity_pid_server`).

From a script or an experiment's host code::

    from waxx.util.guis.intensity_pid.intensity_pid_client import IntensityPIDClient, require_ok
    c = IntensityPIDClient("raman_intensity")
    require_ok(c.hold(True))        # both outputs fixed at their current values
    ...
    require_ok(c.hold(False))       # release (bumpless)
    require_ok(c.clear())           # zero both integrators
    c.status()                      # {"meas1": ..., "set1": ..., "held": ..., ...}

Each call is a TCP round trip plus a serial one (milliseconds to tens of ms):
host code only, not inside a timed kernel sequence.
"""

from __future__ import annotations

import json
import logging
import socket
from typing import Any, Optional

from beacon.discovery.client import NetClient

__all__ = ["IntensityPIDClient", "IntensityPIDError", "require_ok"]

_LOG = logging.getLogger("waxx.intensity_pid.client")


class IntensityPIDError(RuntimeError):
    """The server refused or failed a request."""


def require_ok(reply: dict) -> dict:
    """``reply`` if its status is ok, else raise :class:`IntensityPIDError`."""
    if not isinstance(reply, dict) or reply.get("status") != "ok":
        msg = reply.get("message") if isinstance(reply, dict) else None
        raise IntensityPIDError(msg or f"request failed: {reply!r}")
    return reply


class IntensityPIDClient(NetClient):

    #: Timeout for requests that wait on the board; polls use ``timeout``.
    ACTION_TIMEOUT_S = 10.0

    def __init__(self, server_id: str, discovery_timeout: float = 3.0, timeout: float = 1.5):
        super().__init__(server_id, discovery_timeout=discovery_timeout)
        self.server_id = server_id
        self.timeout = float(timeout)

    @classmethod
    def at(cls, host: str, port: int, server_id: str = "intensity_pid", timeout: float = 1.5):
        """A client for a known address (no discovery; tests, scripts)."""
        self = cls.__new__(cls)
        self.server_id = server_id
        self._waxx_server_id = server_id
        self.reconnect_generation = 0
        self._fixed_address = True
        self.host, self.port = host, int(port)
        self.timeout = float(timeout)
        return self

    # -- wire ---------------------------------------------------------------

    def request(self, cmd: str, *, timeout: Optional[float] = None, **fields: Any) -> dict:
        """Send one request; the reply dict.  Raises OSError on a network failure."""
        payload = (json.dumps({"cmd": cmd, **fields}) + "\n").encode("utf-8")
        t = self.timeout if timeout is None else float(timeout)
        try:
            return self._roundtrip(payload, t)
        except OSError:
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
            _LOG.warning("intensity PID: non-JSON reply: %r", raw[:200])
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

    def status(self) -> dict:
        """The latest status row, flattened: meas1 set1 meas2 set2 out1 out2
        (V, one ~10 ms line), held, the gains, data_age_s, com, warnings."""
        snap = require_ok(self.get_snapshot())
        row = snap.get("latest") or {}
        out = {k: row.get(k) for k in ("meas1", "set1", "meas2", "set2", "out1", "out2")}
        out.update(held=snap.get("held"), data_age_s=snap.get("data_age_s"),
                   com=(snap.get("com") or {}).get("status"), warnings=snap.get("warnings") or [])
        out.update(snap.get("settings") or {})
        return out

    def hold(self, on: Optional[bool] = None) -> dict:
        """True: fix both outputs at their current values; False: release
        (bumpless); None: toggle.  The reply's ``on`` is the new state."""
        return self.request("hold", on=None if on is None else bool(on),
                            timeout=self.ACTION_TIMEOUT_S)

    def release(self) -> dict:
        return self.hold(False)

    def clear(self) -> dict:
        """Zero both integrators."""
        return self.request("clear", timeout=self.ACTION_TIMEOUT_S)

    def set_value(self, name: str, value) -> dict:
        """One setting.  A gain (p1 i1 p2 i2) is remembered and re-applied if
        the board resets."""
        return self.request("set", name=name, value=value, timeout=self.ACTION_TIMEOUT_S)

    def override(self, loop: int, on: bool, volt: Optional[float] = None) -> dict:
        """Loop 1 or 2: on = its output is ``volt`` (or the last ovolt) instead
        of the PID; off = the PID drives it again, starting from that voltage.
        Never re-applied after a board reset."""
        fields = {"loop": int(loop), "on": bool(on)}
        if volt is not None:
            fields["volt"] = float(volt)
        return self.request("override", timeout=self.ACTION_TIMEOUT_S, **fields)

    def gains(self) -> dict:
        """{p1, i1, p2, i2} as the board reads them back now."""
        return require_ok(self.request("refresh_settings", timeout=self.ACTION_TIMEOUT_S))["settings"]

    def refresh_settings(self) -> dict:
        return self.request("refresh_settings", timeout=self.ACTION_TIMEOUT_S)

    def raw(self, line: str) -> dict:
        return self.request("raw", line=line, timeout=self.ACTION_TIMEOUT_S)

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
