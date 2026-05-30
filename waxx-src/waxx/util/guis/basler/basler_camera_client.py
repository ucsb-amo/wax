"""Basler camera client — discovers all ``basler_server:*`` servers and
provides a per-camera handle for remote control and frame retrieval.

Usage::

    from waxx.util.guis.basler.basler_camera_client import discover_all_basler_cameras

    clients = discover_all_basler_cameras(collect_for=3.0)
    for c in clients:
        c.open()
        data = c.get_frame()   # {"ok": True, "frame": ndarray, ...}
        c.close()

``BaslerServerConnection`` is one ZMQ connection to one server process.
``BaslerCameraClient`` is a handle to one camera on one server.
Multiple ``BaslerCameraClient`` objects may share the same
``BaslerServerConnection`` when they live on the same host.
"""
from __future__ import annotations

import logging
import pickle
import threading
from typing import Optional

import zmq

from waxx.util.comms_server.waxx_client import WaxxClient, discover_prefix
from waxx.util.guis.basler.basler_camera_server import BASLER_SERVER_PREFIX

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server connection (one per discovered server process)
# ---------------------------------------------------------------------------

class BaslerServerConnection(WaxxClient):
    """ZMQ REQ connection to one BaslerCameraServer.

    The socket is created lazily on the first request and re-created on
    connection failure / rediscovery.  All requests are serialised by an
    internal lock so that multiple ``BaslerCameraClient`` objects sharing
    this connection do not interleave messages.
    """

    def __init__(self, server_id: str, discovery_timeout: float = 3.0) -> None:
        super().__init__(server_id, discovery_timeout=discovery_timeout)
        self._ctx: Optional[zmq.Context] = None
        self._sock: Optional[zmq.Socket] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #

    def _connect(self) -> None:
        if self._ctx is None:
            self._ctx = zmq.Context()
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:
                pass
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 4000)
        sock.setsockopt(zmq.SNDTIMEO, 4000)
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
                        f"[BaslerClient] Timeout communicating with {self._waxx_server_id}"
                    )
                except Exception:
                    if attempt == 0:
                        self._connect()
                        continue
                    raise

    # ------------------------------------------------------------------ #
    # Server-level commands
    # ------------------------------------------------------------------ #

    def list_cameras(self) -> list[dict]:
        """Return ``[{"serial", "model", "is_open"}, ...]`` for this server."""
        r = self._request({"cmd": "LIST_CAMERAS"})
        return r.get("cameras", [])

    def rescan_cameras(self) -> list[dict]:
        """Ask the server to re-enumerate USB devices and return the new list."""
        r = self._request({"cmd": "RESCAN_CAMERAS"})
        return r.get("cameras", [])

    @property
    def hostname(self) -> str:
        """Extract the hostname portion from ``"basler_server:<hostname>[:<n>]"``."""
        suffix = self._waxx_server_id[len(BASLER_SERVER_PREFIX):]
        return suffix.split(":")[0]


# ---------------------------------------------------------------------------
# Per-camera handle
# ---------------------------------------------------------------------------

class BaslerCameraClient:
    """Handle to a single camera on a single BaslerCameraServer.

    Constructed by ``discover_all_basler_cameras()`` or manually.
    All calls are forwarded to the owning ``BaslerServerConnection``.
    """

    def __init__(
        self,
        connection: BaslerServerConnection,
        serial: str,
        model: str,
        user_id: str = "",
    ) -> None:
        self.connection = connection
        self.serial = serial
        self.model = model
        self.user_id = user_id

    @property
    def display_name(self) -> str:
        """User-defined ID if set, otherwise model + serial."""
        return self.user_id if self.user_id else f"{self.model} [{self.serial}]"

    @property
    def server_id(self) -> str:
        return self.connection._waxx_server_id

    @property
    def hostname(self) -> str:
        return self.connection.hostname

    # ------------------------------------------------------------------ #

    def _req(self, cmd: dict) -> dict:
        cmd["serial"] = self.serial
        return self.connection._request(cmd)

    def open(self) -> dict:
        return self._req({"cmd": "OPEN_CAMERA"})

    def close(self) -> dict:
        return self._req({"cmd": "CLOSE_CAMERA"})

    def get_frame(self) -> dict:
        return self._req({"cmd": "GET_FRAME"})

    def get_gain(self) -> dict:
        return self._req({"cmd": "GET_GAIN"})

    def set_gain(self, value: float) -> dict:
        return self._req({"cmd": "SET_GAIN", "value": float(value)})

    def get_exposure(self) -> dict:
        return self._req({"cmd": "GET_EXPOSURE"})

    def set_exposure(self, value: float) -> dict:
        return self._req({"cmd": "SET_EXPOSURE", "value": float(value)})

    def get_gain_range(self) -> dict:
        return self._req({"cmd": "GET_GAIN_RANGE"})

    def get_exposure_range(self) -> dict:
        return self._req({"cmd": "GET_EXPOSURE_RANGE"})


# ---------------------------------------------------------------------------
# Discovery helper
# ---------------------------------------------------------------------------

def discover_all_basler_cameras(collect_for: float = 3.0) -> list[BaslerCameraClient]:
    """Find all running ``basler_server:*`` instances and list their cameras.

    Waits ``collect_for`` seconds for UDP beacons to arrive, then contacts
    every discovered server and asks for its camera list.  Returns one
    ``BaslerCameraClient`` per (server, camera) pair, in server-discovery order.

    Servers that cannot be reached within the ZMQ timeout are silently skipped
    (a warning is logged).
    """
    servers = discover_prefix(BASLER_SERVER_PREFIX, collect_for=collect_for)
    if not servers:
        logger.warning(
            "[BaslerClient] No '%s*' servers found after %.1f s scan.",
            BASLER_SERVER_PREFIX,
            collect_for,
        )
        return []

    clients: list[BaslerCameraClient] = []
    for sid in sorted(servers):
        try:
            conn = BaslerServerConnection(sid, discovery_timeout=1.0)
            for cam_info in conn.list_cameras():
                clients.append(
                    BaslerCameraClient(
                        conn,
                        cam_info["serial"],
                        cam_info["model"],
                        user_id=cam_info.get("user_id", ""),
                    )
                )
        except Exception as exc:
            logger.warning("[BaslerClient] Could not contact server %s: %s", sid, exc)

    logger.info(
        "[BaslerClient] Discovered %d camera(s) across %d server(s).",
        len(clients),
        len(servers),
    )
    return clients
