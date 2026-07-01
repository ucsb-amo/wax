"""TPI-1005-A client — discovers all ``tpi_server:*`` servers and provides
per-device handles for remote control and state subscription.

Usage::

    from waxx.util.guis.tpi.tpi_client import discover_all_tpi_devices, TpiStateSubscriber

    devices = discover_all_tpi_devices(collect_for=3.0)
    for d in devices:
        print(d.display_name, d.get_state())

    # Subscribe to live state broadcast from the first server found
    sub = TpiStateSubscriber(devices[0].connection)
    sub.start(callback=lambda msg: print(msg))
    ...
    sub.stop()

``TpiServerConnection`` is one ZMQ REQ connection to one server process.
``TpiDeviceClient`` is a handle to one device on one server.
``TpiStateSubscriber`` connects to a server's ZMQ PUB socket and delivers
state updates via callback or the ``latest_states`` dict.
"""
from __future__ import annotations

import logging
import pickle
import threading
from typing import Callable, Optional

import zmq

from waxx.util.comms_server.waxx_client import WaxxClient, discover_prefix
from waxx.util.guis.tpi.tpi_server import TPI_SERVER_PREFIX

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server connection
# ---------------------------------------------------------------------------

class TpiServerConnection(WaxxClient):
    """ZMQ REQ connection to one TpiDeviceServer.

    All requests are serialised by an internal lock so multiple
    ``TpiDeviceClient`` objects sharing this connection do not interleave
    messages on the socket.
    """

    DEFAULT_RCVTIMEO_MS = 4000
    DEFAULT_SNDTIMEO_MS = 4000

    def __init__(self, server_id: str, discovery_timeout: float = 3.0) -> None:
        super().__init__(server_id, discovery_timeout=discovery_timeout)
        self._ctx: Optional[zmq.Context] = None
        self._sock: Optional[zmq.Socket] = None
        self._lock = threading.Lock()
        self._pub_port: Optional[int] = None

    @property
    def ctx(self) -> zmq.Context:
        if self._ctx is None:
            self._ctx = zmq.Context()
        return self._ctx

    def _connect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:
                pass
        sock = self.ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self.DEFAULT_RCVTIMEO_MS)
        sock.setsockopt(zmq.SNDTIMEO, self.DEFAULT_SNDTIMEO_MS)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self.host}:{self.port}")
        self._sock = sock

    def _request(self, cmd: dict) -> dict:
        """Send *cmd* and return the reply dict.  Retries once with rediscovery."""
        with self._lock:
            if self._sock is None:
                self._connect()
            for attempt in range(2):
                try:
                    self._sock.send(pickle.dumps(cmd))
                    return pickle.loads(self._sock.recv())
                except zmq.Again:
                    if attempt == 0 and self._rediscover(timeout=2.0):
                        self._connect()
                        continue
                    raise ConnectionError(
                        f"[TpiClient] Timeout communicating with {self._waxx_server_id}"
                    )
                except Exception:
                    if attempt == 0:
                        self._rediscover(timeout=2.0)
                        self._connect()
                        continue
                    raise

    def list_devices(self) -> list[dict]:
        """Return device info dicts and cache the pub_port from the response."""
        r = self._request({"cmd": "LIST_DEVICES"})
        if self._pub_port is None:
            self._pub_port = r.get("pub_port")
        return r.get("devices", [])

    def rescan_devices(self) -> list[dict]:
        """Ask the server to re-enumerate USB devices and return the new list."""
        r = self._request({"cmd": "RESCAN_DEVICES"})
        self._pub_port = r.get("pub_port", self._pub_port)
        return r.get("devices", [])

    def get_pub_port(self) -> int:
        """Return the server's ZMQ PUB port, fetching it on first call."""
        if self._pub_port is None:
            r = self._request({"cmd": "GET_PUB_PORT"})
            self._pub_port = r["pub_port"]
        return self._pub_port

    @property
    def hostname(self) -> str:
        suffix = self._waxx_server_id[len(TPI_SERVER_PREFIX):]
        return suffix.split(":")[0]


# ---------------------------------------------------------------------------
# Per-device handle
# ---------------------------------------------------------------------------

class TpiDeviceClient:
    """Handle to a single TPI-1005-A on a single TpiDeviceServer.

    Constructed by ``discover_all_tpi_devices()`` or manually.
    """

    def __init__(
        self,
        connection: TpiServerConnection,
        serial: str,
        model: str,
        firmware: str = "",
    ) -> None:
        self.connection = connection
        self.serial = serial
        self.model = model
        self.firmware = firmware

    def _req(self, cmd: dict) -> dict:
        cmd["serial"] = self.serial
        return self.connection._request(cmd)

    def get_state(self) -> dict:
        """Return ``{"ok": True, "rf_on": bool, "freq_mhz": float, "level_dbm": int}``."""
        return self._req({"cmd": "GET_STATE"})

    def set_rf(self, on: bool) -> dict:
        return self._req({"cmd": "SET_RF", "on": bool(on)})

    def set_freq(self, mhz: float) -> dict:
        return self._req({"cmd": "SET_FREQ", "mhz": float(mhz)})

    def set_level(self, dbm: int) -> dict:
        return self._req({"cmd": "SET_LEVEL", "dbm": int(dbm)})

    @property
    def server_id(self) -> str:
        return self.connection._waxx_server_id

    @property
    def hostname(self) -> str:
        return self.connection.hostname

    @property
    def display_name(self) -> str:
        return f"{self.model} [{self.serial}] @ {self.hostname}"


# ---------------------------------------------------------------------------
# State subscriber
# ---------------------------------------------------------------------------

class TpiStateSubscriber:
    """Subscribes to a TpiDeviceServer's ZMQ PUB socket.

    Runs a background daemon thread that receives pickled state dicts from
    the server and delivers them via an optional callback and/or stores them
    in ``latest_states`` keyed by device serial.

    Usage::

        sub = TpiStateSubscriber(connection)
        sub.start(callback=lambda msg: print(msg["serial"], msg["rf_on"]))
        # ... do work ...
        sub.stop()
        # or read sub.latest_states["299"] at any time
    """

    def __init__(self, connection: TpiServerConnection) -> None:
        self._connection = connection
        self._ctx = zmq.Context()
        self._sock: Optional[zmq.Socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[dict], None]] = None
        self.latest_states: dict[str, dict] = {}
        self._states_lock = threading.Lock()

    def start(self, callback: Optional[Callable[[dict], None]] = None) -> None:
        """Connect to the PUB socket and start receiving state updates.

        Args:
            callback: Optional callable invoked on each received message.
                Called from the subscriber thread — keep it short.
        """
        self._callback = callback
        pub_port = self._connection.get_pub_port()
        sock = self._ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.connect(f"tcp://{self._connection.host}:{pub_port}")
        self._sock = sock
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"TpiSub-{self._connection._waxx_server_id}",
        )
        self._thread.start()
        logger.debug(
            "[TpiClient] Subscribed to %s PUB port %d",
            self._connection._waxx_server_id, pub_port,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None
        if self._sock:
            self._sock.close()
            self._sock = None

    def _loop(self) -> None:
        while self._running:
            try:
                raw = self._sock.recv()
            except zmq.Again:
                continue
            except Exception:
                break
            try:
                msg = pickle.loads(raw)
            except Exception:
                continue
            if not isinstance(msg, dict) or msg.get("type") != "state":
                continue
            serial = msg.get("serial", "")
            with self._states_lock:
                self.latest_states[serial] = msg
            if self._callback is not None:
                try:
                    self._callback(msg)
                except Exception as exc:
                    logger.warning("[TpiClient] Subscriber callback raised: %s", exc)


# ---------------------------------------------------------------------------
# Discovery helper
# ---------------------------------------------------------------------------

_LAST_DISCOVERY_FINGERPRINT: tuple | None = None
_LAST_NO_SERVERS_WARNED: bool = False
_LAST_CONTACT_FAILURES: set[str] = set()


def discover_all_tpi_devices(
    collect_for: float = 3.0,
    quiet: bool = False,
) -> list[TpiDeviceClient]:
    """Find all running ``tpi_server:*`` instances and list their devices.

    Waits ``collect_for`` seconds for UDP beacons, contacts each server,
    and returns one ``TpiDeviceClient`` per (server, device) pair.
    Unreachable servers are skipped with a warning.

    Set ``quiet=True`` for routine periodic polls to suppress repeated logs.
    """
    global _LAST_DISCOVERY_FINGERPRINT, _LAST_NO_SERVERS_WARNED, _LAST_CONTACT_FAILURES

    servers = discover_prefix(TPI_SERVER_PREFIX, collect_for=collect_for)
    if not servers:
        if not _LAST_NO_SERVERS_WARNED:
            logger.warning("[TpiClient] No '%s*' servers found after %.1f s scan.",
                           TPI_SERVER_PREFIX, collect_for)
            _LAST_NO_SERVERS_WARNED = True
        else:
            logger.debug("[TpiClient] Still no '%s*' servers found.", TPI_SERVER_PREFIX)
        _LAST_DISCOVERY_FINGERPRINT = ((), 0)
        return []
    _LAST_NO_SERVERS_WARNED = False

    clients: list[TpiDeviceClient] = []
    current_failures: set[str] = set()
    for sid in sorted(servers):
        try:
            conn = TpiServerConnection(sid, discovery_timeout=1.0)
            for dev_info in conn.list_devices():
                clients.append(TpiDeviceClient(
                    conn,
                    dev_info["serial"],
                    dev_info["model"],
                    dev_info.get("firmware", ""),
                ))
        except Exception as exc:
            current_failures.add(sid)
            if sid not in _LAST_CONTACT_FAILURES:
                logger.warning("[TpiClient] Could not contact server %s: %s", sid, exc)
            else:
                logger.debug("[TpiClient] Could not contact server %s: %s", sid, exc)
    _LAST_CONTACT_FAILURES = current_failures

    fingerprint = (tuple(sorted(servers)), len(clients))
    changed = fingerprint != _LAST_DISCOVERY_FINGERPRINT
    _LAST_DISCOVERY_FINGERPRINT = fingerprint
    if changed and not quiet:
        logger.info("[TpiClient] Discovered %d device(s) across %d server(s).", len(clients), len(servers))
    else:
        logger.debug("[TpiClient] Discovered %d device(s) across %d server(s).", len(clients), len(servers))
    return clients
