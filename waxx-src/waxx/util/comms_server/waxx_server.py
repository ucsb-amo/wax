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
    def _get_local_ip() -> str | None:
        """Return the primary LAN IP for service-discovery beacons.

        Uses the routing-table trick — connects a UDP socket to a probe address
        and reads back the source IP the OS would assign.  No packet is ever
        transmitted; the OS just consults its routing table.

        Strategy (handles multi-homed machines such as lab Ethernet + campus WiFi):

        1. Probe the lab gateway (192.168.1.1) first.  Accept the result only if
           it lands on 192.168.1.x — this uniquely identifies the lab adapter
           even when an internet-facing adapter is also present.
        2. Fall back to generic probes for machines on other subnets (USB-Ethernet
           adapters, different lab network ranges, etc.).  Any non-loopback,
           non-APIPA address is accepted.
        """
        # Priority 1: lab subnet (192.168.1.x)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("192.168.1.1", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip.startswith("192.168.1.") and ip != "0.0.0.0":
                return ip
        except Exception:
            pass

        # Priority 2: any other private/routable adapter (USB-Ethernet, other subnets)
        for probe in ("10.0.0.1", "172.16.0.1", "8.8.8.8", "1.1.1.1"):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect((probe, 80))
                ip = s.getsockname()[0]
                s.close()
                if (ip and ip != "0.0.0.0"
                        and not ip.startswith("127.")
                        and not ip.startswith("169.254.")):
                    return ip
            except Exception:
                pass

        return None

    def _beacon_loop(self) -> None:
        """Daemon thread: broadcast a JSON beacon every ``beacon_interval`` seconds.

        Retries ``_get_local_ip()`` on every iteration so that a server started
        before the NIC is fully up will begin advertising once the interface appears.

        The beacon socket is bound to the specific local IP so that the UDP
        broadcast is transmitted on the correct adapter (e.g. the lab Ethernet
        rather than a WiFi default-route adapter).  The socket is recreated
        whenever the detected IP changes (e.g. adapter comes up late).
        """
        sock: socket.socket | None = None
        current_bound_ip: str | None = None

        def _close_sock() -> None:
            nonlocal sock, current_bound_ip
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
                sock = None
                current_bound_ip = None

        while not self._waxx_beacon_stop.is_set():
            ip = self._get_local_ip()

            if ip is None:
                logger.debug("[WaxxServer] Could not determine local LAN IP, will retry.")
                self._waxx_beacon_stop.wait(timeout=self._waxx_beacon_interval)
                continue

            # (Re)create socket when the local IP changes or on first use.
            # Binding to the specific IP forces the OS to send the broadcast
            # out on that adapter rather than the default-route interface.
            if ip != current_bound_ip:
                _close_sock()
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    sock.bind((ip, 0))
                    current_bound_ip = ip
                    logger.debug("[WaxxServer] Beacon socket bound to %s", ip)
                except Exception as exc:
                    logger.warning("[WaxxServer] Could not create beacon socket on %s: %s", ip, exc)
                    _close_sock()
                    self._waxx_beacon_stop.wait(timeout=self._waxx_beacon_interval)
                    continue

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
                _close_sock()

            self._waxx_beacon_stop.wait(timeout=self._waxx_beacon_interval)

        _close_sock()
