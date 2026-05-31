"""TCP client for the Keysight current-supply server."""
from __future__ import annotations

import json
import socket
from typing import Optional

from waxx.util.comms_server.waxx_client import WaxxClient

SERVER_ID = "keysight"


class KeysightClient(WaxxClient):
    """Discovers and communicates with a running ``KeysightServer``.

    All accessors return decoded JSON.  No direct VXI11 traffic — every
    GUI uses this to talk to the central server, which owns the supplies.
    """

    def __init__(self, timeout_s: float = 2.0, discovery_timeout: float = 3.0) -> None:
        super().__init__(SERVER_ID, discovery_timeout=discovery_timeout)
        self.timeout_s = float(timeout_s)

    # ------------------------------------------------------------------ #

    def _send(self, command: str) -> str:
        payload = f"{command}\n".encode("utf-8")
        for attempt in range(2):
            try:
                with socket.create_connection(
                    (self.host, self.port), timeout=self.timeout_s
                ) as sock:
                    sock.settimeout(self.timeout_s)
                    sock.sendall(payload)
                    chunks = []
                    while True:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        chunks.append(chunk)
                return b"".join(chunks).decode("utf-8", errors="replace").strip()
            except (ConnectionRefusedError, ConnectionResetError, OSError, socket.timeout):
                if attempt == 0 and self._rediscover(timeout=2.0):
                    continue
                raise

    # ------------------------------------------------------------------ #
    # Read-only accessors
    # ------------------------------------------------------------------ #

    def get_snapshot(self) -> list[dict]:
        """List one dict per supply with cached readings."""
        return json.loads(self._send("GET_SNAPSHOT"))

    def get_status(self) -> dict:
        return json.loads(self._send("GET_STATUS"))

    # ------------------------------------------------------------------ #
    # Targeted RPCs
    # ------------------------------------------------------------------ #

    def turn_on(self, ip: str) -> dict:
        return json.loads(self._send(f"TURN_ON {ip}"))

    def clear_protect(self, ip: str) -> dict:
        return json.loads(self._send(f"CLEAR_PROT {ip}"))

    def reconnect(self, ip: str) -> dict:
        return json.loads(self._send(f"RECONNECT {ip}"))


__all__ = ["KeysightClient", "SERVER_ID"]
