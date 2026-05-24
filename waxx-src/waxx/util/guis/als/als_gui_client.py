"""TCP client for the ALS hardware server.

The server owns the serial connection to the laser hardware and exposes a
simple line-based command API. This client is used by both remote automation
and the GUI frontend.
"""

from __future__ import annotations

import socket
import json
from typing import Optional

from waxx.util.comms_server.waxx_client import WaxxClient


class ALSGuiClient(WaxxClient):
    """TCP client for remote control and monitoring of the ALS server."""

    def __init__(self, timeout_s: float = 2.0, discovery_timeout: float = 3.0):
        super().__init__("als_laser", discovery_timeout=discovery_timeout)
        self.timeout_s = timeout_s

    def _send_command(self, command: str) -> str:
        """Send a single command and return a single-line server response.

        On a connection-level error, rediscovers the server via UDP broadcast
        and retries once — handles server restarts on a new ephemeral port.
        """
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

    def _send_ok_command(self, command: str) -> bool:
        response = self._send_command(command)
        return response.upper() == "OK"

    def _send_json_command(self, command: str) -> dict:
        response = self._send_command(command)
        return json.loads(response)

    def run_startup_sequence(self) -> bool:
        """Request the server to start the startup sequence."""
        return self._send_ok_command("START_STARTUP")

    def run_shutdown_sequence(self) -> bool:
        """Request the server to start the shutdown sequence."""
        return self._send_ok_command("START_SHUTDOWN")

    def interrupt_sequence(self) -> bool:
        """Request the server to interrupt the active sequence."""
        return self._send_ok_command("INTERRUPT")

    def connect_serial(self) -> bool:
        return self._send_ok_command("CONNECT_SERIAL")

    def disconnect_serial(self) -> bool:
        return self._send_ok_command("DISCONNECT_SERIAL")

    def set_power_percent(self, power_percent: float) -> bool:
        return self._send_ok_command(f"SET_POWER_PERCENT {float(power_percent):.6f}")

    def set_power_supply_on(self) -> bool:
        return self._send_ok_command("SET_POWER_SUPPLY_ON")

    def set_power_supply_off(self) -> bool:
        return self._send_ok_command("SET_POWER_SUPPLY_OFF")

    def set_interlock_on(self) -> bool:
        return self._send_ok_command("SET_INTERLOCK_ON")

    def set_interlock_off(self) -> bool:
        return self._send_ok_command("SET_INTERLOCK_OFF")

    def set_second_stage_on(self) -> bool:
        return self._send_ok_command("SET_SECOND_STAGE_ON")

    def set_second_stage_off(self) -> bool:
        return self._send_ok_command("SET_SECOND_STAGE_OFF")

    def get_snapshot(self) -> dict:
        """Fetch the latest status, sequence state, and log cursor from the server."""
        return self._send_json_command("GET_SNAPSHOT")

    def get_logs_since(self, start_index: int) -> dict:
        """Fetch log lines appended after the given absolute log index."""
        return self._send_json_command(f"GET_LOGS {int(start_index)}")

    def poll_power_setpoint_percent(self) -> Optional[float]:
        """Poll laser power setpoint percent from the ALS server.

        Expected response formats:
        - "POWER_SETPOINT: <float>"
        - "<float>"

        Returns:
        - float if a parsable value is returned
        - None if the server only returns "OK" (not implemented on server yet)
        """
        response = self._send_command("GET_POWER_SETPOINT")
        if not response:
            return None

        normalized = response.strip()
        if normalized.upper() == "OK":
            return None

        if ":" in normalized:
            _, value_text = normalized.split(":", 1)
            value_text = value_text.strip()
        else:
            value_text = normalized

        try:
            return float(value_text)
        except ValueError:
            return None


if __name__ == "__main__":
    client = ALSGuiClient()

    snapshot = client.get_snapshot()
    print(snapshot)
