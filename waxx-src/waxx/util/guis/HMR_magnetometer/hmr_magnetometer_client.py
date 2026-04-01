"""Lightweight client for the HMR2300 magnetometer server.

Each call opens a fresh TCP connection, sends one command, reads the
newline-terminated JSON response, and closes.

Quick usage::

    from hmr_magnetometer_client import HMRClient

    client = HMRClient("localhost", 50000)
    field  = client.get_field()         # latest reading
    new    = client.get_since(1.74e9)   # all readings with t > timestamp
    ok     = client.ping()              # connectivity check
"""

from datetime import datetime

from artiq.language import portable, kernel, delay
import numpy as np

import json
import socket

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 50000

def _request(command: str, host: str, port: int, timeout: float) -> dict:
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall((command + "\n").encode("utf-8"))
        chunks = []
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if chunks[-1].endswith(b"\n"):
                break
    return json.loads(b"".join(chunks).decode("utf-8").strip())


class HMRClient:
    """Client for the HMR2300 magnetometer TCP server.

    Args:
        host: Server hostname or IP address.
        port: TCP port the server is listening on.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def _request(self, command: str, timeout: float) -> dict:
        return _request(command, self.host, self.port, timeout)

    def _ping(self, timeout: float = 2.0) -> dict:
        """Check server connectivity.

        Returns ``{"ok": True, "message": "pong"}`` or raises on failure.
        """
        return self._request("PING", timeout)

    def _get_field(self, timeout: float = 2.0) -> dict:
        """Fetch the most recent field reading.

        Returns::

            {"ok": True, "t": float, "Bx": float, "By": float,
                                     "Bz": float, "Btot": float}

        All field values are in Gauss.  *t* is Unix epoch seconds.
        """
        return self._request("GET_FIELD", timeout)

    def _get_since(self, timestamp_s: float, timeout: float = 5.0) -> dict:
        """Fetch all readings recorded after *timestamp_s*.

        Returns::

            {"ok": True, "readings": [{"t": float, "Bx": float, "By": float,
                                        "Bz": float, "Btot": float}, ...]}

        Readings are ordered oldest-first.  Pass 0.0 to retrieve the full
        server history buffer.
        """
        return self._request(f"GET_SINCE {timestamp_s:.6f}", timeout)

    def _set_reference(self, timeout: float = 2.0) -> dict:
        """Store the latest server field as a reference entry in the reference CSV."""
        return self._request("SET_REFERENCE", timeout)

    def _set_reference_values(
        self,
        bx: float,
        by: float,
        bz: float,
        btot: float,
        timeout: float = 2.0,
    ) -> dict:
        """Store explicit reference values in the server reference CSV."""
        return self._request(
            f"SET_REFERENCE_VALUES {bx:.9f} {by:.9f} {bz:.9f} {btot:.9f}",
            timeout,
        )

    def _get_reference_before(self, timestamp_s: float, timeout: float = 2.0) -> dict:
        """Fetch the latest reference with timestamp <= *timestamp_s*."""
        return self._request(f"GET_REFERENCE_BEFORE {timestamp_s:.6f}", timeout)

    def _get_serial_status(self, timeout: float = 2.0) -> dict:
        """Return server-side serial connection status."""
        return self._request("GET_SERIAL_STATUS", timeout)

    def _serial_disconnect(self, timeout: float = 2.0) -> dict:
        """Manually disconnect serial device on server."""
        return self._request("SERIAL_DISCONNECT", timeout)

    def _serial_reconnect(self, timeout: float = 3.0) -> dict:
        """Manually reconnect serial device on server."""
        return self._request("SERIAL_RECONNECT", timeout)

    def _restart_serial(self, timeout: float = 4.0) -> dict:
        """Force-restart (disconnect then reconnect) the serial device on server."""
        return self._request("RESTART_SERIAL", timeout)

    def get_field_magnitude(self, timeout: float = 5.) -> float:
        """Return the latest total field magnitude in Gauss.

        Raises ``RuntimeError`` if the server returns an error or has no data yet.
        """
        max_attempts = 5
        per_try_timeout = min(1.5, timeout / max_attempts)
        
        for attempt in range(max_attempts):
            try:
                result = self._get_field(timeout=per_try_timeout)
                if not result.get("ok"):
                    raise RuntimeError(result.get("error", "Server returned error"))
                return float(result["Btot"])
            except (socket.timeout, socket.error) as e:
                if attempt == max_attempts - 1:
                    print(f"Reading magnetometer failed after {max_attempts} attempts: {e}")
                    return float(0.)

    def get_reference_field_array(self, date=None, timeout: float = 2.0) -> np.ndarray:
        """Return [Bx, By, Bz, Btot] from the newest reference at or before *date*.

        Args:
            date: ``datetime`` (or ``None`` for current datetime).
            timeout: TCP request timeout in seconds.
        """
        field_vec, _meta = self.get_reference_field_array_with_metadata(
            date=date,
            timeout=timeout,
        )
        return field_vec

    def get_reference_field_array_with_metadata(self, date=None, timeout: float = 2.0):
        """Return reference vector and metadata for newest reference at or before *date*.

        Returns:
            tuple: ``(field_vec, metadata)`` where:
                - ``field_vec`` is ``np.ndarray([Bx, By, Bz, Btot], dtype=float)``
                - ``metadata`` is ``{"timestamp_s": float, "datetime_iso": str}``

        Args:
            date: ``datetime`` (or ``None`` for current datetime).
            timeout: TCP request timeout in seconds.
        """
        if date is None:
            date = datetime.now()
        if not isinstance(date, datetime):
            raise TypeError("date must be a datetime.datetime instance or None")

        result = self._get_reference_before(date.timestamp(), timeout=timeout)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Server returned error"))

        ref = result["reference"]
        field_vec = np.array(
            [
                float(ref["Bx"]),
                float(ref["By"]),
                float(ref["Bz"]),
                float(ref["Btot"]),
            ],
            dtype=float,
        )
        metadata = {
            "timestamp_s": float(ref["timestamp_s"]),
            "datetime_iso": str(ref.get("datetime_iso", "")),
        }
        return field_vec, metadata
