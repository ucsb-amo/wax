"""TCP clients for the Bristol wavemeter server.

``BristolWavemeterGuiClient`` is the GUI's client (retries, rediscovers).
``BristolAverageReader`` is the per-shot reader for experiments: one short
attempt per call behind a link latch, never raises, never blocks a run.
"""
from __future__ import annotations

import json
import socket
import time

from beacon.discovery.client import NetClient, discover
from waxx.util.link_latch import LinkLatch, T_LINK_RETRY

SERVER_ID = "bristol_wavemeter"

# Whole-request deadline for a per-shot read (connect + send + reply).
T_READ_TIMEOUT = 0.5


class BristolWavemeterGuiClient(NetClient):
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


def _request_line(addr, command: str, timeout_s: float) -> dict:
    """One line out, one JSON line back, all within ``timeout_s``."""
    deadline = time.monotonic() + timeout_s

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0.:
            raise socket.timeout(f"no reply within {timeout_s:g} s")
        return left

    with socket.create_connection(addr, timeout=remaining()) as sock:
        sock.sendall(f"{command}\n".encode("utf-8"))
        chunks = []
        while True:
            sock.settimeout(remaining())
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if chunk.endswith(b"\n"):
                break
    return json.loads(b"".join(chunks).decode("utf-8", errors="replace").strip())


class BristolAverageReader:
    """Per-shot read of the server's running average (``GET_AVERAGE``).

    Built for a call from a kernel between shots: one attempt per call with
    a ``timeout_s`` deadline, no retries, no sleeps, and it never raises.
    ``get_average`` returns the server's reply dict, or ``None`` when no
    reply was had (server not found, unreachable, hung, or too old to know
    ``GET_AVERAGE``).

    A failure trips a :class:`~waxx.util.link_latch.LinkLatch`: calls return
    ``None`` at once for ``retry_after`` seconds, then one attempt is made
    again. The server address is looked up in the beacon cache on every
    attempt (no wait), so a server restarted on a new port, or started
    after the run began, is picked up at the next retry.

    A reply with ``ok=False`` (server up, no fresh readings) is returned as
    is; it does not trip the latch.
    """

    def __init__(self, server_id: str = SERVER_ID,
                 discovery_timeout: float = 1.0,
                 timeout_s: float = T_READ_TIMEOUT,
                 retry_after: float = T_LINK_RETRY):
        self.server_id = server_id
        self.timeout_s = float(timeout_s)
        self.latch = LinkLatch(f"bristol wavemeter ({server_id})",
                               retry_after=retry_after)
        self._no_data = False
        # Wait once, here (prepare time), for the beacon; later lookups don't.
        if discover(server_id, timeout=discovery_timeout) is None:
            self.latch.trip(RuntimeError(
                f"server '{server_id}' not discovered within {discovery_timeout:g} s"))

    def get_average(self, n: int, max_age_s: float) -> dict | None:
        try:
            return self._get_average(n, max_age_s)
        except Exception as e:  # never raise into a run
            print(f"bristol wavemeter: read failed ({e!r})")
            return None

    def _get_average(self, n, max_age_s):
        if self.latch.should_skip():
            return None
        addr = discover(self.server_id, timeout=0.)
        if addr is None:
            self.latch.trip(RuntimeError(f"server '{self.server_id}' not discovered"))
            return None
        try:
            reply = _request_line(addr, f"GET_AVERAGE {int(n):d} {float(max_age_s):.6g}",
                                  self.timeout_s)
        except Exception as e:
            self.latch.trip(e)
            return None
        if not isinstance(reply, dict) or "n_used" not in reply:
            # An older server answers {"error": "unknown command: ..."}.
            err = reply.get("error", reply) if isinstance(reply, dict) else reply
            self.latch.trip(RuntimeError(
                f"server does not support GET_AVERAGE ({err!r}); "
                "restart the Bristol server from the Server Dashboard"))
            return None
        self.latch.clear()
        # Print once per edge when the server is up but has no fresh readings.
        no_data = not reply.get("ok")
        if no_data and not self._no_data:
            print(f"*** bristol wavemeter: no fresh readings ({reply.get('error', '')}; "
                  f"wavemeter connected={reply.get('connected')}). ***")
        elif self._no_data and not no_data:
            print("bristol wavemeter: readings back.")
        self._no_data = no_data
        return reply
