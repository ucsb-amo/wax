"""TCP client for remote control and monitoring of the Precilaser server."""

from __future__ import annotations

import json
import socket


class PrecilaserGuiClient:
    def __init__(self, host: str = "192.168.1.76", port: int = 5560, timeout_s: float = 2.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    def _send_command(self, command: str) -> str:
        payload = f"{command}\n".encode("utf-8")
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

    def _send_ok_command(self, command: str) -> bool:
        response = self._send_command(command)
        return response.upper() == "OK"

    def _send_json_command(self, command: str) -> dict:
        response = self._send_command(command)
        return json.loads(response)

    def get_snapshot(self) -> dict:
        return self._send_json_command("GET_SNAPSHOT")

    def get_logs_since(self, start_index: int) -> dict:
        return self._send_json_command(f"GET_LOGS {int(start_index)}")

    def connect_serial(self) -> bool:
        return self._send_ok_command("CONNECT_SERIAL")

    def disconnect_serial(self) -> bool:
        return self._send_ok_command("DISCONNECT_SERIAL")

    def run_startup_sequence(self) -> bool:
        return self._send_ok_command("START_STARTUP")

    def run_shutdown_sequence(self) -> bool:
        return self._send_ok_command("START_SHUTDOWN")

    def interrupt_sequence(self) -> bool:
        return self._send_ok_command("INTERRUPT")

    def set_working_current(self, current_amps: float) -> bool:
        return self._send_ok_command(f"SET_WORKING_CURRENT {float(current_amps):.6f}")

    def set_laser_enable(self, enabled: bool) -> bool:
        return self._send_ok_command(f"SET_LASER_ENABLE {1 if enabled else 0}")

    def set_stability_mode(self, enabled: bool) -> bool:
        return self._send_ok_command(f"SET_STABILITY_MODE {1 if enabled else 0}")

    def set_startup_target_current(self, current_amps: float) -> bool:
        return self._send_ok_command(f"SET_STARTUP_TARGET_CURRENT {float(current_amps):.6f}")
