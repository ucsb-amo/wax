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
import uuid
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

    # Default ZMQ timeouts (ms).  Control RPCs (OPEN/CLOSE/SET_*) can be
    # slow; frame fetches should fail fast so a single missed trigger does
    # not stall the GUI for seconds.
    DEFAULT_RCVTIMEO_MS = 4000
    DEFAULT_SNDTIMEO_MS = 4000

    def __init__(self, server_id: str, discovery_timeout: float = 3.0) -> None:
        super().__init__(server_id, discovery_timeout=discovery_timeout)
        self._ctx: Optional[zmq.Context] = None
        self._sock: Optional[zmq.Socket] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #

    @property
    def ctx(self) -> zmq.Context:
        if self._ctx is None:
            self._ctx = zmq.Context()
        return self._ctx

    def make_socket(self, rcvtimeo_ms: int = DEFAULT_RCVTIMEO_MS,
                    sndtimeo_ms: int = DEFAULT_SNDTIMEO_MS) -> zmq.Socket:
        """Create a fresh REQ socket connected to this server.

        Used by ``BaslerCameraClient`` to obtain a dedicated per-camera
        frame socket so that ``GET_FRAME`` traffic does not serialise
        behind other cameras' RPCs on the shared control socket.
        """
        sock = self.ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, int(rcvtimeo_ms))
        sock.setsockopt(zmq.SNDTIMEO, int(sndtimeo_ms))
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self.host}:{self.port}")
        return sock

    def _connect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:
                pass
        self._sock = self.make_socket()

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
                        # The server may have restarted on a new ephemeral
                        # port; refresh the address before reconnecting so we
                        # don't keep dialing the dead one.
                        self._rediscover(timeout=2.0)
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
        # Stable per-handle identifier sent with OPEN/CLOSE so the server
        # can ref-count concurrent clients on the same physical camera.
        # The server keeps the device open as long as at least one client
        # holds it; only the last close() actually stops grabbing.
        self._client_id: str = uuid.uuid4().hex
        # Tracks whether this handle currently holds the camera open on the
        # server.  Set by ``open()`` / cleared by ``close()``.  Used to decide
        # whether to transparently re-send OPEN after a server restart.
        self._is_open: bool = False
        # The connection's ``reconnect_generation`` value last acted upon.
        # When the connection rediscovers the server at a new address this
        # diverges, signalling that we must re-establish server-side state.
        self._seen_reconnect_generation: int = connection.reconnect_generation
        # Dedicated frame socket — created lazily on first ``get_frame()``
        # call.  Using a per-camera socket keeps GET_FRAME traffic off the
        # shared control socket so multiple cameras on the same server do
        # not serialise behind each other, and a stalled trigger on one
        # camera does not delay control RPCs on another.
        self._frame_sock: Optional[zmq.Socket] = None
        self._frame_lock = threading.Lock()
        self._frame_rcvtimeo_ms: int = 400

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
        resp = self._req({"cmd": "OPEN_CAMERA", "client_id": self._client_id})
        if not isinstance(resp, dict) or resp.get("ok", True):
            # Treat a missing "ok" as success (legacy replies); only an
            # explicit failure leaves us in the closed state.
            self._is_open = True
        # Whatever the connection's address is now, consider it the baseline
        # so we don't immediately try to "re-open" on the next frame.
        self._seen_reconnect_generation = self.connection.reconnect_generation
        return resp

    def close(self) -> dict:
        self._is_open = False
        # Tear down the dedicated frame socket when the camera is closed
        # so we do not leak FDs and the next open() reconnects cleanly.
        with self._frame_lock:
            if self._frame_sock is not None:
                try:
                    self._frame_sock.close(linger=0)
                except Exception:
                    pass
                self._frame_sock = None
        return self._req({"cmd": "CLOSE_CAMERA", "client_id": self._client_id})

    def _maybe_reopen_after_reconnect(self) -> bool:
        """Re-establish camera state after the server restarted.

        A restarted ``BaslerCameraServer`` binds a new ephemeral port and
        loses all open-camera state.  Once the connection has rediscovered
        the new address (``reconnect_generation`` advanced), re-send OPEN so
        the fresh server starts grabbing again.  No-op unless this handle was
        previously opened.  Returns ``True`` if a reopen was attempted.
        """
        gen = self.connection.reconnect_generation
        if gen == self._seen_reconnect_generation:
            return False
        self._seen_reconnect_generation = gen
        if not self._is_open:
            return False
        # Drop the stale frame socket so the next fetch reconnects fresh.
        with self._frame_lock:
            if self._frame_sock is not None:
                try:
                    self._frame_sock.close(linger=0)
                except Exception:
                    pass
                self._frame_sock = None
        try:
            self._req({"cmd": "OPEN_CAMERA", "client_id": self._client_id})
        except Exception:
            # The server may not be reachable yet; leave _is_open set so a
            # later frame attempt retries the reopen.
            self._seen_reconnect_generation = gen - 1
            return False
        return True

    def set_frame_timeout_ms(self, timeout_ms: int) -> None:
        """Adjust the RCVTIMEO of the dedicated frame socket (re-created
        on next ``get_frame()`` call).
        """
        self._frame_rcvtimeo_ms = int(timeout_ms)
        with self._frame_lock:
            if self._frame_sock is not None:
                try:
                    self._frame_sock.close(linger=0)
                except Exception:
                    pass
                self._frame_sock = None

    def get_frame(self) -> dict:
        """Fetch one frame on a dedicated per-camera socket.

        Uses a short RCVTIMEO so a missed trigger costs ~``_frame_rcvtimeo_ms``
        rather than blocking for the control-socket timeout.  Does NOT touch
        the shared control socket / lock.
        """
        # If the server restarted, re-open the camera on the new address
        # before fetching so the fresh server is actually grabbing.
        self._maybe_reopen_after_reconnect()
        cmd = {"cmd": "GET_FRAME", "serial": self.serial}
        with self._frame_lock:
            if self._frame_sock is None:
                self._frame_sock = self.connection.make_socket(
                    rcvtimeo_ms=self._frame_rcvtimeo_ms,
                    sndtimeo_ms=1000,
                )
            for attempt in range(2):
                try:
                    self._frame_sock.send(pickle.dumps(cmd))
                    return pickle.loads(self._frame_sock.recv())
                except zmq.Again:
                    # Recreate the REQ socket (REQ state machine breaks on
                    # timeout) and either retry once or return a timeout error.
                    try:
                        self._frame_sock.close(linger=0)
                    except Exception:
                        pass
                    # A persistent miss may mean the server moved (restarted
                    # on a new port).  Refresh the address so the rebuilt
                    # socket dials the live server, not the dead one.
                    if attempt == 0:
                        self.connection._rediscover(timeout=1.0)
                    self._frame_sock = self.connection.make_socket(
                        rcvtimeo_ms=self._frame_rcvtimeo_ms,
                        sndtimeo_ms=1000,
                    )
                    if attempt == 0:
                        continue
                    return {"ok": False, "error": "frame timeout"}
                except Exception as exc:
                    try:
                        self._frame_sock.close(linger=0)
                    except Exception:
                        pass
                    self._frame_sock = None
                    if attempt == 0:
                        continue
                    return {"ok": False, "error": str(exc)}

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

    def get_trigger_mode(self) -> dict:
        return self._req({"cmd": "GET_TRIGGER_MODE"})

    def set_trigger_mode(self, value: str) -> dict:
        return self._req({"cmd": "SET_TRIGGER_MODE", "value": str(value)})

    def get_trigger_mode_options(self) -> dict:
        return self._req({"cmd": "GET_TRIGGER_MODE_OPTIONS"})

    def set_defaults(self, gain=None, exposure=None, trigger_mode=None,
                     roi=None, norm_reference=None) -> dict:
        """Persist this camera's open-time defaults on the server.

        Only non-None values are sent.  gain/exposure/trigger_mode are applied
        to the hardware on the next open; roi and norm_reference are opaque
        client display values the server stores and echoes back.
        """
        cmd: dict = {"cmd": "SET_DEFAULTS"}
        if gain is not None:
            cmd["gain"] = float(gain)
        if exposure is not None:
            cmd["exposure"] = float(exposure)
        if trigger_mode is not None:
            cmd["trigger_mode"] = str(trigger_mode)
        if roi is not None:
            cmd["roi"] = [int(v) for v in roi]
        if norm_reference is not None:
            cmd["norm_reference"] = float(norm_reference)
        return self._req(cmd)

    def get_defaults(self) -> dict:
        return self._req({"cmd": "GET_DEFAULTS"})


# ---------------------------------------------------------------------------
# Discovery helper
# ---------------------------------------------------------------------------

# Module-level cache used to deduplicate routine discovery log lines.
# ``_LAST_DISCOVERY_FINGERPRINT`` records the sorted server-id list from the
# previous call so we only emit an INFO line when the topology actually
# changes; periodic re-scans every 15 s would otherwise flood the dashboard
# log with identical "Discovered N camera(s)" lines.
_LAST_DISCOVERY_FINGERPRINT: tuple[tuple[str, ...], int] | None = None
_LAST_NO_SERVERS_WARNED: bool = False
_LAST_CONTACT_FAILURES: set[str] = set()


def discover_all_basler_cameras(
    collect_for: float = 3.0,
    quiet: bool = False,
) -> list[BaslerCameraClient]:
    """Find all running ``basler_server:*`` instances and list their cameras.

    Waits ``collect_for`` seconds for UDP beacons to arrive, then contacts
    every discovered server and asks for its camera list.  Returns one
    ``BaslerCameraClient`` per (server, camera) pair, in server-discovery order.

    Servers that cannot be reached within the ZMQ timeout are silently skipped
    (a warning is logged the first time a given server fails).

    Set ``quiet=True`` for routine periodic polls — status-quo results then
    log at DEBUG instead of INFO/WARNING.  Genuine changes (new server, lost
    server, new failure) are always surfaced at the higher level.
    """
    global _LAST_DISCOVERY_FINGERPRINT, _LAST_NO_SERVERS_WARNED, _LAST_CONTACT_FAILURES

    servers = discover_prefix(BASLER_SERVER_PREFIX, collect_for=collect_for)
    if not servers:
        # Only WARN the first time we transition into "no servers"; while
        # the condition persists, drop to DEBUG so periodic rescans don't
        # flood the log.
        if not _LAST_NO_SERVERS_WARNED:
            logger.warning(
                "[BaslerClient] No '%s*' servers found after %.1f s scan.",
                BASLER_SERVER_PREFIX,
                collect_for,
            )
            _LAST_NO_SERVERS_WARNED = True
        else:
            logger.debug(
                "[BaslerClient] Still no '%s*' servers found after %.1f s scan.",
                BASLER_SERVER_PREFIX,
                collect_for,
            )
        _LAST_DISCOVERY_FINGERPRINT = ((), 0)
        return []
    _LAST_NO_SERVERS_WARNED = False

    clients: list[BaslerCameraClient] = []
    current_failures: set[str] = set()
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
            current_failures.add(sid)
            # Only WARN on a server we haven't already complained about;
            # repeated failures from the same server drop to DEBUG.
            if sid not in _LAST_CONTACT_FAILURES:
                logger.warning("[BaslerClient] Could not contact server %s: %s", sid, exc)
            else:
                logger.debug("[BaslerClient] Could not contact server %s: %s", sid, exc)
    _LAST_CONTACT_FAILURES = current_failures

    fingerprint = (tuple(sorted(servers)), len(clients))
    changed = fingerprint != _LAST_DISCOVERY_FINGERPRINT
    _LAST_DISCOVERY_FINGERPRINT = fingerprint
    level = logging.INFO if (changed and not quiet) else logging.DEBUG
    logger.log(
        level,
        "[BaslerClient] Discovered %d camera(s) across %d server(s).",
        len(clients),
        len(servers),
    )
    return clients
