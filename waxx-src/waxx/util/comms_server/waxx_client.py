"""WaxxClient — base class for all waxx/kexp TCP clients.

Provides automatic server discovery via UDP broadcast.  A process-wide
``_ServiceDiscoveryRegistry`` singleton listens on ``DISCOVERY_PORT`` and
caches ``{server_id: (ip, port)}`` entries from server beacons.

``WaxxClient.__init__`` blocks up to ``discovery_timeout`` seconds waiting
for the target server's beacon to arrive, then sets ``self.host`` and
``self.port``.  Raises ``RuntimeError`` if discovery times out.

Module-level ``discover()`` is also exposed for call sites that only need the
IP without constructing a full client object (e.g. ``RemoteViewerWindow``).

Usage::

    class MyClient(WaxxClient):
        def __init__(self):
            super().__init__("my_server")
            # self.host and self.port are now set

    # Or standalone:
    result = discover("my_server", timeout=3.0)   # (ip, port) | None
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time

logger = logging.getLogger(__name__)

DISCOVERY_PORT: int = 50099


# ---------------------------------------------------------------------------
# Process-wide registry singleton
# ---------------------------------------------------------------------------

class _ServiceDiscoveryRegistry:
    """Listens for WaxxServer beacons and maintains a ``{server_id: (ip, port)}`` cache.

    Instantiated once at module import time.  All ``WaxxClient`` instances share
    the same cache — subsequent constructions for the same server are instant.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, int]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._start()

    def _start(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(0.5)
            sock.bind(("0.0.0.0", DISCOVERY_PORT))
            self._sock = sock
            self._running = True
            self._thread = threading.Thread(
                target=self._listen_loop,
                name="WaxxDiscoveryRegistry",
                daemon=True,
            )
            self._thread.start()
            logger.debug("[WaxxClient] Discovery registry listening on port %d", DISCOVERY_PORT)
        except OSError as exc:
            logger.warning(
                "[WaxxClient] Discovery registry could not bind port %d: %s. "
                "Service discovery unavailable; clients will raise RuntimeError on construction.",
                DISCOVERY_PORT, exc,
            )
            self._running = False

    def _listen_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode())
                server_id = str(msg["server_id"])
                ip = str(msg["ip"])
                port = int(msg["port"])
                with self._lock:
                    self._cache[server_id] = (ip, port)
            except Exception:
                pass

    def discover(self, server_id: str, timeout: float = 5.0) -> tuple[str, int] | None:
        """Block until ``server_id`` appears in cache or ``timeout`` expires.

        Returns ``(ip, port)`` or ``None`` — never raises.
        """
        if not self._running:
            with self._lock:
                return self._cache.get(server_id)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                entry = self._cache.get(server_id)
            if entry is not None:
                return entry
            time.sleep(0.025)

        # One final check after the sleep loop exits.
        with self._lock:
            return self._cache.get(server_id)

    def discover_prefix(self, prefix: str, collect_for: float = 3.0) -> dict[str, tuple[str, int]]:
        """Return all cached entries whose ``server_id`` starts with ``prefix``.

        Waits up to ``collect_for`` seconds so that any servers currently
        beaconing have time to appear in the cache, then returns a snapshot.
        Returns an empty dict if no matching servers are found.
        """
        if self._running:
            deadline = time.monotonic() + collect_for
            while time.monotonic() < deadline:
                time.sleep(0.1)
        with self._lock:
            return {sid: addr for sid, addr in self._cache.items()
                    if sid.startswith(prefix)}


# Module-level singleton — starts listening immediately on import.
_registry = _ServiceDiscoveryRegistry()


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def discover(server_id: str, timeout: float = 3.0) -> tuple[str, int] | None:
    """Discover ``server_id`` via UDP broadcast.

    Returns ``(ip, port)`` when found, ``None`` when the timeout expires.
    Never raises.  Useful for call sites that only need the server IP without
    constructing a full ``WaxxClient`` instance.

    Example::

        result = discover("live_od", timeout=3.0)
        if result is None:
            raise RuntimeError("liveOD server not found")
        ip, port = result
    """
    return _registry.discover(server_id, timeout=timeout)


def discover_prefix(prefix: str, collect_for: float = 3.0) -> dict[str, tuple[str, int]]:
    """Return all currently-beaconing servers whose ``server_id`` starts with ``prefix``.

    Waits ``collect_for`` seconds for beacons to accumulate, then returns a
    ``{server_id: (ip, port)}`` snapshot.  Returns an empty dict if none found.

    Example::

        servers = discover_prefix("basler_server:", collect_for=3.0)
        for sid, (ip, port) in servers.items():
            print(sid, ip, port)
    """
    return _registry.discover_prefix(prefix, collect_for=collect_for)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class WaxxClient:
    """Base class for all waxx/kexp network clients.

    Sets ``self.host`` and ``self.port`` via UDP broadcast service discovery.
    Raises ``RuntimeError`` if the server is not discovered within
    ``discovery_timeout`` seconds.

    Args:
        server_id: The ``server_id`` string advertised by the server beacon.
        discovery_timeout: Maximum seconds to wait for a beacon.
    """

    def __init__(self, server_id: str, discovery_timeout: float = 3.0) -> None:
        self._waxx_server_id = server_id
        result = _registry.discover(server_id, timeout=discovery_timeout)
        if result is None:
            raise RuntimeError(
                f"[WaxxClient] Server '{server_id}' not discovered within "
                f"{discovery_timeout:.1f} s. "
                f"Is the server running and on the 192.168.1.x subnet?"
            )
        self.host, self.port = result
        # Bumped every time ``_rediscover`` observes a *changed* (host, port),
        # i.e. the server restarted on a new address.  Callers can watch this
        # counter to detect a genuine reconnect (vs. a no-op rediscovery) and
        # rebuild sockets / re-establish server-side state accordingly.
        self.reconnect_generation: int = 0
        logger.debug("[WaxxClient] Discovered %s @ %s:%d", server_id, self.host, self.port)

    def _rediscover(self, timeout: float = 2.0) -> bool:
        """Refresh ``self.host`` and ``self.port`` from the latest beacon.

        Call this when a connection attempt fails so that if the server
        restarted on a new ephemeral port the client picks up the new address
        before retrying.  Returns ``True`` if a (possibly updated) entry was
        found, ``False`` if the server is currently unreachable.

        When the discovered ``(host, port)`` differs from the current one
        (the server moved / restarted), ``reconnect_generation`` is bumped so
        callers can detect the reconnect.
        """
        entry = _registry.discover(self._waxx_server_id, timeout=timeout)
        if entry is None:
            return False
        if entry != (self.host, self.port):
            self.reconnect_generation += 1
            logger.debug("[WaxxClient] Rediscovered %s @ %s:%d (reconnect #%d)",
                         self._waxx_server_id, entry[0], entry[1],
                         self.reconnect_generation)
        self.host, self.port = entry
        return True
