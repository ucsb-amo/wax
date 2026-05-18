"""WaxxServer — mixin base for all waxx/kexp TCP servers.

Provides UDP broadcast service discovery: each server periodically advertises
its ``server_id``, IP address, and port to ``255.255.255.255:DISCOVERY_PORT``
so that clients can locate it without hard-coded IP addresses.

Usage (pure-Python server)::

    class MyServer(WaxxServer):
        def __init__(self, port=5562):
            WaxxServer.__init__(self, "my_server", port)

        def start(self):
            self._start_beacon()
            ...

        def stop(self):
            self._stop_beacon()
            ...

Usage (Qt / QThread server — explicit init required to avoid MRO conflict)::

    class MyQServer(QThread, WaxxServer):
        def __init__(self, port):
            super().__init__()                          # QThread.__init__
            WaxxServer.__init__(self, "my_server", port)  # explicit

        def run(self):
            self._start_beacon()
            ...
            self._stop_beacon()
"""

from __future__ import annotations

import json
import logging
import socket
import threading

logger = logging.getLogger(__name__)

DISCOVERY_PORT: int = 50099
_BROADCAST_ADDR = "255.255.255.255"


class WaxxServer:
    """Mixin that adds UDP broadcast service discovery to any server class.

    Does NOT call ``super().__init__()`` — safe for multiple inheritance with
    ``QObject`` / ``QThread`` bases.  Callers must invoke
    ``WaxxServer.__init__(self, server_id, port)`` explicitly.
    """

    def __init__(self, server_id: str = None, port: int = None,
                 beacon_interval: float = 2.0, **kwargs) -> None:
        if server_id is None or port is None:
            # Called via cooperative super() chain without args — skip init.
            # WaxxServer must be initialised explicitly: WaxxServer.__init__(self, id, port)
            return
        self._waxx_server_id: str = server_id
        self._waxx_port: int = int(port)
        self._waxx_beacon_interval: float = float(beacon_interval)
        self._waxx_beacon_thread: threading.Thread | None = None
        self._waxx_beacon_stop: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def _start_beacon(self) -> None:
        """Start the background beacon thread.  Safe to call multiple times."""
        if self._waxx_beacon_thread is not None and self._waxx_beacon_thread.is_alive():
            return
        self._waxx_beacon_stop.clear()
        self._waxx_beacon_thread = threading.Thread(
            target=self._beacon_loop,
            name=f"WaxxBeacon-{self._waxx_server_id}",
            daemon=True,
        )
        self._waxx_beacon_thread.start()
        logger.info("[WaxxServer] Beacon started: %s on port %d",
                    self._waxx_server_id, self._waxx_port)

    def _stop_beacon(self) -> None:
        """Signal the beacon thread to stop and wait briefly for it."""
        self._waxx_beacon_stop.set()
        if self._waxx_beacon_thread is not None:
            self._waxx_beacon_thread.join(timeout=self._waxx_beacon_interval + 1.0)
            self._waxx_beacon_thread = None
        logger.info("[WaxxServer] Beacon stopped: %s", self._waxx_server_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_subnet_ip(prefix: str = "192.168.1.") -> str | None:
        """Return the local IP on the ``prefix`` subnet, or ``None``.

        Uses the UDP connect trick — queries the OS routing table without
        transmitting any packets.
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("192.168.1.1", 1))
            ip = s.getsockname()[0]
            s.close()
            if ip.startswith(prefix):
                return ip
        except Exception:
            pass
        return None

    def _beacon_loop(self) -> None:
        """Daemon thread: broadcast a JSON beacon every ``beacon_interval`` seconds.

        Retries ``_get_subnet_ip()`` on every iteration so that a server started
        before the NIC is fully up will begin advertising once the interface appears.
        """
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except Exception as exc:
            logger.warning("[WaxxServer] Could not create beacon socket: %s", exc)
            return

        payload_template = json.dumps({
            "server_id": self._waxx_server_id,
            "port": self._waxx_port,
            "ip": None,  # replaced each iteration
        })

        while not self._waxx_beacon_stop.is_set():
            ip = self._get_subnet_ip()
            if ip is None:
                logger.debug("[WaxxServer] No 192.168.1.x NIC found, will retry.")
            else:
                payload = json.dumps({
                    "server_id": self._waxx_server_id,
                    "ip": ip,
                    "port": self._waxx_port,
                }).encode()
                try:
                    sock.sendto(payload, (_BROADCAST_ADDR, DISCOVERY_PORT))
                    logger.debug("[WaxxServer] Beacon: %s @ %s:%d",
                                 self._waxx_server_id, ip, self._waxx_port)
                except Exception as exc:
                    logger.warning("[WaxxServer] Beacon send failed: %s", exc)

            self._waxx_beacon_stop.wait(timeout=self._waxx_beacon_interval)

        try:
            sock.close()
        except Exception:
            pass
