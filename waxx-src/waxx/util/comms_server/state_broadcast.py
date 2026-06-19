"""UDP push channel for device-state updates.

The monitor server is the single writer of the device-state JSON.  After it
applies a client delta it broadcasts the change (full delta + monotonic
version) on a dedicated UDP port so every connected GUI updates instantly,
without polling the shared drive.

* :class:`StateBroadcaster` — used by the server to emit ``state_update``
  datagrams to the lab broadcast address.
* :class:`StateListener` — a ``QThread`` used by clients; emits a Qt signal
  for every datagram received.

UDP is lossy by design; callers recover from dropped packets using the
``version`` field (a gap triggers a full ``get_state`` resync over TCP).
"""

from __future__ import annotations

import json
import socket

from PyQt6.QtCore import QThread, pyqtSignal

# Dedicated port for device-state push (distinct from the discovery beacon
# port 50099 in waxx_server).
STATE_BROADCAST_PORT: int = 50100
_BROADCAST_ADDR = "192.168.1.255"   # directed broadcast for the lab subnet


class StateBroadcaster:
    """Server-side UDP sender for ``state_update`` datagrams."""

    def __init__(self, port: int = STATE_BROADCAST_PORT):
        self.port = int(port)
        self._sock: socket.socket | None = None
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except Exception:
            self._sock = None

    def send(self, payload: dict) -> None:
        """Broadcast a JSON payload.  Never raises (push is best-effort)."""
        if self._sock is None:
            return
        try:
            data = json.dumps(payload).encode()
            self._sock.sendto(data, (_BROADCAST_ADDR, self.port))
        except Exception:
            # Best-effort: a missed broadcast is recovered via version-gap
            # resync, so swallow transient send errors.
            pass

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


class StateListener(QThread):
    """Client-side UDP listener.  Emits :attr:`state_received` per datagram.

    Multiple listeners on the same host can bind the same port concurrently
    (``SO_REUSEADDR``), so several GUIs on one machine all receive the push.
    """

    state_received = pyqtSignal(dict)

    def __init__(self, port: int = STATE_BROADCAST_PORT, parent=None):
        super().__init__(parent)
        self._port = int(port)
        self._running = True

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # SO_REUSEPORT is not available on Windows; ignore if missing.
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except Exception:
                pass
        try:
            sock.bind(("", self._port))
        except Exception:
            sock.close()
            return
        sock.settimeout(0.5)
        while self._running:
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                payload = json.loads(data.decode())
            except Exception:
                continue
            if isinstance(payload, dict):
                self.state_received.emit(payload)
        try:
            sock.close()
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
