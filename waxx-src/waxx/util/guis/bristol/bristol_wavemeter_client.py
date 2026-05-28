"""TCP client for the Bristol wavemeter server."""
from __future__ import annotations

import json
import socket

from waxx.util.comms_server.waxx_client import WaxxClient

SERVER_ID = "bristol_wavemeter"


class BristolWavemeterGuiClient(WaxxClient):
    """Discovers and communicates with a running BristolWavemeterServer."""

    def __init__(self, timeout_s: float = 2.0, discovery_timeout: float = 3.0):
        super().__init__(SERVER_ID, discovery_timeout=discovery_timeout)
        self.timeout_s = timeout_s

    def _send_command(self, command: str) -> str:
        payload = f"{command}\n".encode("utf-8")
        for attempt in range(2):
            try:
                with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
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

    def get_reading(self) -> dict:
        """Return latest reading dict: wavelength_nm, frequency_thz, timestamp, connected."""
        return json.loads(self._send_command("GET_READING"))

    def get_status(self) -> dict:
        """Return server status: connected, host, error."""
        return json.loads(self._send_command("STATUS"))
