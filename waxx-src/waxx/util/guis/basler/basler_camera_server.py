"""BaslerCameraServer — WaxxServer that owns all Basler USB cameras on this machine.

Server-ID format:
    ``"basler_server:<hostname>"``          (instance_index == 0, the normal case)
    ``"basler_server:<hostname>:<n>"``      (instance_index > 0, rare multi-instance)

Clients discover any number of these servers via ``discover_prefix("basler_server:")``.

ZMQ REP protocol — all messages are pickle-serialised dicts.

Commands sent by client → replies from server:

    {"cmd": "LIST_CAMERAS"}
        → {"ok": True, "cameras": [{"serial": str, "model": str, "is_open": bool}, ...]}

    {"cmd": "RESCAN_CAMERAS"}
        → {"ok": True, "cameras": [...]}   (re-enumerates, does not open/close anything)

    {"cmd": "OPEN_CAMERA", "serial": "12345"}
        → {"ok": True} | {"ok": False, "error": str}

    {"cmd": "CLOSE_CAMERA", "serial": "12345"}
        → {"ok": True} | {"ok": False, "error": str}

    {"cmd": "GET_FRAME", "serial": "12345"}
        → {"ok": True, "frame": np.ndarray, "gain": float,
           "exposure": float, "max_pixel_value": int, "timestamp": float}
        | {"ok": False, "error": str}

    {"cmd": "GET_GAIN",          "serial": "12345"} → {"ok": True, "result": float}
    {"cmd": "SET_GAIN",          "serial": "12345", "value": 12.0} → {"ok": True}
    {"cmd": "GET_EXPOSURE",      "serial": "12345"} → {"ok": True, "result": float}
    {"cmd": "SET_EXPOSURE",      "serial": "12345", "value": 300.0} → {"ok": True}
    {"cmd": "GET_GAIN_RANGE",    "serial": "12345"} → {"ok": True, "result": [min, max]}
    {"cmd": "GET_EXPOSURE_RANGE","serial": "12345"} → {"ok": True, "result": [min, max]}

    {"cmd": "SET_DEFAULTS", "serial": "12345", "gain": 12.0, "exposure": 300.0,
            "trigger_mode": "Off", "roi": [x1, y1, x2, y2], "norm_reference": 1.0}
        → {"ok": True, "defaults": {...}}   (any subset of keys; persisted by serial)
    {"cmd": "GET_DEFAULTS", "serial": "12345"} → {"ok": True, "defaults": {...}}
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import socket
import threading
import time
from typing import Optional

import numpy as np
import zmq
from pypylon import pylon

from waxx.util.comms_server.waxx_server import WaxxServer

logger = logging.getLogger(__name__)

BASLER_SERVER_PREFIX = "basler_server:"

# Per-camera default settings are persisted here on the *server* host, keyed by
# camera serial, so they survive server restarts and apply on every
# OPEN_CAMERA regardless of which client connects.  Gain/exposure/trigger are
# applied to the hardware on open; ROI and normalization are opaque client-side
# display values that the server only stores and echoes back.
_DEFAULTS_FILE = os.path.join(
    os.path.expanduser("~"), ".waxx", "basler_server_defaults.json"
)


# ---------------------------------------------------------------------------
# Per-camera state machine
# ---------------------------------------------------------------------------

class _ManagedCamera:
    """Wraps one physical Basler camera by serial number.

    The camera device is NOT opened until ``open()`` is called.
    A background grab thread keeps ``_latest_frame`` current while open.
    """

    def __init__(self, serial: str, model: str, user_id: str = "") -> None:
        self.serial: str = serial
        self.model: str = model
        self.user_id: str = user_id

        # Open-time defaults for this serial: any of
        # {"gain", "exposure", "trigger_mode", "roi", "norm_reference"}.
        # gain/exposure/trigger_mode are applied to hardware on open(); roi and
        # norm_reference are opaque client display values stored & echoed back.
        self.defaults: dict = {}

        self._lock = threading.Lock()
        self._camera: Optional[pylon.InstantCamera] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_timestamp: float = 0.0
        # Cached hardware values so GET_FRAME never reads GenICam nodes over USB
        # on the hot path (those reads are slow and, done under the lock, also
        # stall the grab thread).  Refreshed on open() and on set_gain/exposure.
        self._cached_gain: float = 0.0
        self._cached_exposure: float = 0.0
        self._cached_max_pixel: int = 255
        self._is_open: bool = False
        # Set of client_ids currently holding this camera open.  The
        # physical device stays open as long as the set is non-empty;
        # close() only tears it down when the last client lets go (or
        # when called with force=True, e.g. server shutdown / operator
        # override on the local GUI).
        self._clients: set[str] = set()

        self._grab_stop = threading.Event()
        self._grab_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self, client_id: str) -> None:
        with self._lock:
            self._clients.add(client_id)
            if self._is_open:
                # Camera already streaming for another client; this new
                # client just shares it.  Nothing else to do.
                logger.info(
                    "[BaslerServer] Camera %s shared (now %d client(s))",
                    self.serial, len(self._clients),
                )
                return
            # Re-enumerate to get a fresh DeviceInfo (handles re-plug).
            tl = pylon.TlFactory.GetInstance()
            di = pylon.DeviceInfo()
            di.SetSerialNumber(self.serial)
            cam = pylon.InstantCamera(tl.CreateFirstDevice(di))
            cam.Open()
            # Free-run continuous mode for live viewing.
            cam.UserSetSelector = "Default"
            cam.UserSetLoad.Execute()
            # Apply this camera's persisted defaults, falling back to the
            # historical hardcoded values when no default is stored.
            try:
                cam.TriggerMode.SetValue(str(self.defaults.get("trigger_mode", "Off")))
            except Exception:
                try:
                    cam.TriggerMode.SetValue("Off")
                except Exception:
                    pass
            cam.AcquisitionMode.SetValue("Continuous")
            try:
                cam.Gain.SetValue(float(self.defaults.get("gain", 12.0)))
            except Exception:
                pass
            try:
                cam.ExposureTime.SetValue(float(self.defaults.get("exposure", 300.0)))
            except Exception:
                pass
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self._camera = cam
            self._is_open = True
            self._latest_frame = None
            # Prime the cached hardware values once, now, instead of reading
            # them back from the camera on every GET_FRAME.
            try:
                self._cached_gain = float(cam.Gain.GetValue())
            except Exception:
                self._cached_gain = float(self.defaults.get("gain", 12.0))
            try:
                self._cached_exposure = float(cam.ExposureTime.GetValue())
            except Exception:
                self._cached_exposure = float(self.defaults.get("exposure", 300.0))
            try:
                self._cached_max_pixel = int(cam.PixelDynamicRangeMax.GetValue())
            except Exception:
                self._cached_max_pixel = 255

        self._grab_stop.clear()
        self._grab_thread = threading.Thread(
            target=self._grab_loop,
            name=f"BaslerGrab-{self.serial}",
            daemon=True,
        )
        self._grab_thread.start()
        logger.info("[BaslerServer] Opened camera %s (%s)", self.serial, self.model)

    def close(self, client_id: str, force: bool = False) -> None:
        with self._lock:
            self._clients.discard(client_id)
            if not self._is_open:
                return
            if self._clients and not force:
                logger.info(
                    "[BaslerServer] Camera %s still held by %d client(s); leaving open",
                    self.serial, len(self._clients),
                )
                return
            if force:
                # Operator / shutdown override: drop any remaining holders
                # so a subsequent open() does a fresh device acquisition.
                self._clients.clear()
        # Tear down outside the lock so the grab thread can exit cleanly.
        self._grab_stop.set()
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=3.0)
            self._grab_thread = None
        with self._lock:
            if not self._is_open:
                return
            try:
                self._camera.StopGrabbing()
                self._camera.Close()
            except Exception:
                pass
            self._camera = None
            self._latest_frame = None
            self._is_open = False
        logger.info("[BaslerServer] Closed camera %s", self.serial)

    def _grab_loop(self) -> None:
        while not self._grab_stop.is_set():
            # Read camera reference without holding the lock during grab.
            with self._lock:
                cam = self._camera
                alive = self._is_open
            if cam is None or not alive or not cam.IsGrabbing():
                time.sleep(0.05)
                continue
            try:
                result = cam.RetrieveResult(500, pylon.TimeoutHandling_Return)
                if result is not None:
                    if result.GrabSucceeded():
                        frame = result.Array.copy()
                        with self._lock:
                            self._latest_frame = frame
                            self._frame_timestamp = time.monotonic()
                    result.Release()
            except Exception as exc:
                logger.debug("[BaslerServer] Grab error on %s: %s", self.serial, exc)
                time.sleep(0.05)

    # ------------------------------------------------------------------ #
    # Property accessors
    # ------------------------------------------------------------------ #

    def get_frame(self) -> dict:
        # Hot path: hold the lock only long enough to snapshot the latest frame
        # and the cached hardware values.  No GenICam / USB reads here, so many
        # cameras' GET_FRAMEs stay cheap and never stall the grab thread.
        with self._lock:
            if not self._is_open or self._camera is None:
                return {"ok": False, "error": "camera not open"}
            frame = self._latest_frame
            if frame is None:
                return {"ok": False, "error": "no frame yet"}
            gain = self._cached_gain
            exposure = self._cached_exposure
            max_pv = self._cached_max_pixel
            timestamp = self._frame_timestamp
        return {
            "ok": True,
            "frame": frame,
            "gain": gain,
            "exposure": exposure,
            "max_pixel_value": max_pv,
            "timestamp": timestamp,
        }

    def get_gain(self) -> dict:
        with self._lock:
            if not self._is_open or self._camera is None:
                return {"ok": False, "error": "camera not open"}
            try:
                return {"ok": True, "result": float(self._camera.Gain.GetValue())}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

    def set_gain(self, value: float) -> dict:
        with self._lock:
            if not self._is_open or self._camera is None:
                return {"ok": False, "error": "camera not open"}
            try:
                self._camera.Gain.SetValue(float(value))
                # Refresh the cached value GET_FRAME hands out.
                try:
                    self._cached_gain = float(self._camera.Gain.GetValue())
                except Exception:
                    self._cached_gain = float(value)
                return {"ok": True}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

    def get_exposure(self) -> dict:
        with self._lock:
            if not self._is_open or self._camera is None:
                return {"ok": False, "error": "camera not open"}
            try:
                return {"ok": True, "result": float(self._camera.ExposureTime.GetValue())}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

    def set_exposure(self, value: float) -> dict:
        with self._lock:
            if not self._is_open or self._camera is None:
                return {"ok": False, "error": "camera not open"}
            try:
                self._camera.ExposureTime.SetValue(float(value))
                try:
                    self._cached_exposure = float(self._camera.ExposureTime.GetValue())
                except Exception:
                    self._cached_exposure = float(value)
                return {"ok": True}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

    def get_gain_range(self) -> dict:
        with self._lock:
            if not self._is_open or self._camera is None:
                return {"ok": False, "error": "camera not open"}
            try:
                return {"ok": True, "result": [
                    float(self._camera.Gain.GetMin()),
                    float(self._camera.Gain.GetMax()),
                ]}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

    def get_exposure_range(self) -> dict:
        with self._lock:
            if not self._is_open or self._camera is None:
                return {"ok": False, "error": "camera not open"}
            try:
                return {"ok": True, "result": [
                    float(self._camera.ExposureTime.GetMin()),
                    float(self._camera.ExposureTime.GetMax()),
                ]}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

    def get_trigger_mode(self) -> dict:
        with self._lock:
            if not self._is_open or self._camera is None:
                return {"ok": False, "error": "camera not open"}
            try:
                return {"ok": True, "result": str(self._camera.TriggerMode.GetValue())}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

    def set_trigger_mode(self, value: str) -> dict:
        with self._lock:
            if not self._is_open or self._camera is None:
                return {"ok": False, "error": "camera not open"}
            try:
                # Stop grabbing while changing trigger configuration; pylon
                # requires this for the change to take effect cleanly.
                was_grabbing = bool(self._camera.IsGrabbing())
                if was_grabbing:
                    self._camera.StopGrabbing()
                self._camera.TriggerMode.SetValue(str(value))
                if was_grabbing:
                    self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                return {"ok": True}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

    def get_trigger_mode_options(self) -> dict:
        with self._lock:
            if not self._is_open or self._camera is None:
                return {"ok": False, "error": "camera not open"}
            try:
                node = self._camera.TriggerMode
                # GenApi enum: use GetSymbolics() for available values.
                try:
                    options = [str(s) for s in node.GetSymbolics()]
                except Exception:
                    options = ["Off", "On"]
                return {"ok": True, "result": options}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

    def set_defaults(self, gain=None, exposure=None, trigger_mode=None,
                     roi=None, norm_reference=None) -> dict:
        """Store this camera's open-time defaults.  Only provided keys update.

        gain/exposure/trigger_mode are applied to the hardware on the next
        ``open()``.  roi and norm_reference are opaque client display values
        the server only persists and echoes back via ``GET_DEFAULTS``.
        """
        with self._lock:
            if gain is not None:
                self.defaults["gain"] = float(gain)
            if exposure is not None:
                self.defaults["exposure"] = float(exposure)
            if trigger_mode is not None:
                self.defaults["trigger_mode"] = str(trigger_mode)
            if roi is not None:
                self.defaults["roi"] = [int(v) for v in roi]
            if norm_reference is not None:
                self.defaults["norm_reference"] = float(norm_reference)
            return {"ok": True, "defaults": dict(self.defaults)}

    def get_defaults(self) -> dict:
        with self._lock:
            return {"ok": True, "defaults": dict(self.defaults)}

    def info_dict(self) -> dict:
        return {"serial": self.serial, "model": self.model, "is_open": self._is_open, "user_id": self.user_id}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class BaslerCameraServer(WaxxServer):
    """Manages all Basler cameras attached to this machine over a ZMQ REP socket.

    Discovers cameras at startup but does not open them.  Clients send commands
    to open/close individual cameras and retrieve frames or adjust settings.

    Args:
        instance_index: Use 0 (default) for the sole server on this host.
            Increment for rare multi-instance deployments on one machine.
        port: ZMQ bind port.  0 lets the OS choose an ephemeral port, which is
            then broadcast via the WaxxServer beacon so clients discover it.
    """

    def __init__(self, instance_index: int = 0, port: int = 0,
                 n_workers: int = 0) -> None:
        hostname = socket.gethostname()
        sid = (
            f"{BASLER_SERVER_PREFIX}{hostname}"
            if instance_index == 0
            else f"{BASLER_SERVER_PREFIX}{hostname}:{instance_index}"
        )

        # Guard: refuse to start if another server with the same ID is already
        # beaconing on the network (same host, same instance_index).
        from waxx.util.comms_server.waxx_client import discover as _discover
        if _discover(sid, timeout=1.5) is not None:
            raise RuntimeError(
                f"A BaslerCameraServer with ID '{sid}' is already running on "
                f"the network. Stop the existing instance first, or start this "
                f"one with a different instance_index."
            )

        WaxxServer.__init__(self, sid, port)
        self._cameras: dict[str, _ManagedCamera] = {}
        self._running = False
        # Number of dispatch worker threads (0 = auto-size from camera count)
        # so each open stream is serviced concurrently instead of queueing
        # behind the others on a single request loop.
        self._n_workers = int(n_workers)
        self._defaults_store: dict[str, dict] = self._load_defaults()
        self._enumerate_cameras()

    # ------------------------------------------------------------------ #
    # Camera enumeration
    # ------------------------------------------------------------------ #

    def _enumerate_cameras(self) -> None:
        """Discover connected Basler devices without opening any."""
        new: dict[str, _ManagedCamera] = {}
        try:
            tl = pylon.TlFactory.GetInstance()
            for di in tl.EnumerateDevices():
                serial = di.GetSerialNumber()
                try:
                    model = di.GetModelName()
                except Exception:
                    model = "Unknown"
                try:
                    user_id = di.GetUserDefinedName()
                except Exception:
                    user_id = ""
                if serial in self._cameras:
                    # Preserve existing managed camera (may be open).
                    mc = self._cameras[serial]
                    mc.model = model
                    mc.user_id = user_id
                    new[serial] = mc
                else:
                    mc = _ManagedCamera(serial, model, user_id=user_id)
                    # Seed open-time defaults from the persisted store.
                    mc.defaults = dict(self._defaults_store.get(serial, {}))
                    new[serial] = mc
            self._cameras = new
            logger.info(
                "[BaslerServer] Enumerated %d camera(s): %s",
                len(self._cameras),
                list(self._cameras.keys()),
            )
        except Exception as exc:
            logger.error("[BaslerServer] Camera enumeration failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Defaults persistence
    # ------------------------------------------------------------------ #

    def _load_defaults(self) -> dict[str, dict]:
        """Load the per-serial defaults map from disk on the server host."""
        try:
            with open(_DEFAULTS_FILE) as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return {str(k): dict(v) for k, v in data.items()
                        if isinstance(v, dict)}
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("[BaslerServer] Could not read defaults file: %s", exc)
        return {}

    def _save_defaults(self) -> None:
        """Persist every camera's current defaults to disk, keyed by serial."""
        store: dict[str, dict] = dict(self._defaults_store)
        for serial, mc in self._cameras.items():
            if mc.defaults:
                store[serial] = dict(mc.defaults)
        self._defaults_store = store
        try:
            os.makedirs(os.path.dirname(_DEFAULTS_FILE), exist_ok=True)
            with open(_DEFAULTS_FILE, "w") as fh:
                json.dump(store, fh, indent=2)
        except Exception as exc:
            logger.warning("[BaslerServer] Could not write defaults file: %s", exc)

    # ------------------------------------------------------------------ #
    # Server lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Bind the ZMQ front end, start the beacon, and serve requests.

        Requests are dispatched by a pool of worker threads (ROUTER/DEALER
        pattern) so frame fetches and control RPCs for different cameras run
        concurrently instead of serialising behind one another on a single
        loop.  The main thread runs a lightweight pump that shuttles messages
        between the TCP front end and the in-process worker back end.

        Blocks until ``stop()`` is called or the process exits.
        """
        context = zmq.Context(io_threads=2)
        # TCP front end that clients' REQ sockets connect to.
        frontend = context.socket(zmq.ROUTER)
        actual_port = frontend.bind_to_random_port("tcp://0.0.0.0")
        # In-process back end that worker REP sockets pull from.
        backend = context.socket(zmq.DEALER)
        backend.bind("inproc://basler_workers")
        # Update the advertised port so the beacon carries the right value.
        self._waxx_port = actual_port
        self._running = True
        self._start_beacon()

        n_workers = self._n_workers or max(2, min(8, len(self._cameras) or 2))
        workers: list[threading.Thread] = []
        for i in range(n_workers):
            t = threading.Thread(
                target=self._worker_loop,
                args=(context,),
                name=f"BaslerWorker-{i}",
                daemon=True,
            )
            t.start()
            workers.append(t)

        logger.info(
            "[BaslerServer] Listening on ZMQ port %d  (server_id=%s, %d workers)",
            actual_port,
            self._waxx_server_id,
            n_workers,
        )

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(backend, zmq.POLLIN)
        try:
            # Pump only shuttles frames (no pickling / camera I/O), so the
            # expensive work stays parallel across the worker pool.  copy=False
            # avoids memcopying large frame replies through this thread.
            while self._running:
                socks = dict(poller.poll(timeout=500))
                if socks.get(frontend) == zmq.POLLIN:
                    backend.send_multipart(
                        frontend.recv_multipart(copy=False), copy=False
                    )
                if socks.get(backend) == zmq.POLLIN:
                    frontend.send_multipart(
                        backend.recv_multipart(copy=False), copy=False
                    )
        finally:
            self._stop_beacon()
            # Let workers observe _running == False and exit their poll loops.
            for t in workers:
                t.join(timeout=1.0)
            for mc in self._cameras.values():
                if mc.is_open:
                    try:
                        mc.close("__shutdown__", force=True)
                    except Exception:
                        pass
            frontend.close(0)
            backend.close(0)
            context.term()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    # Request dispatch
    # ------------------------------------------------------------------ #

    def _worker_loop(self, context: zmq.Context) -> None:
        """Dispatch worker: pull one request, handle it, send the reply.

        Each worker owns its own REP socket connected to the in-process back
        end, so N workers service N requests concurrently.  Per-camera locks in
        ``_ManagedCamera`` keep concurrent access to the same device safe, and
        pickling of large frame replies happens here (in parallel) rather than
        on one shared loop.
        """
        sock = context.socket(zmq.REP)
        sock.connect("inproc://basler_workers")
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            while self._running:
                socks = dict(poller.poll(timeout=500))
                if sock not in socks:
                    continue
                try:
                    cmd = pickle.loads(sock.recv())
                    reply = self._dispatch(cmd)
                except Exception as exc:
                    reply = {"ok": False, "error": f"dispatch error: {exc}"}
                sock.send(pickle.dumps(reply))
        finally:
            sock.close(0)

    def _dispatch(self, cmd: dict) -> dict:
        name = cmd.get("cmd", "")

        if name == "LIST_CAMERAS":
            return {"ok": True, "cameras": [mc.info_dict() for mc in self._cameras.values()]}

        if name == "RESCAN_CAMERAS":
            self._enumerate_cameras()
            return {"ok": True, "cameras": [mc.info_dict() for mc in self._cameras.values()]}

        serial = cmd.get("serial", "")
        if serial not in self._cameras:
            return {"ok": False, "error": f"Unknown serial: {serial!r}"}

        mc = self._cameras[serial]

        if name == "OPEN_CAMERA":
            client_id = cmd.get("client_id") or f"anon:{id(cmd)}"
            try:
                mc.open(client_id)
                return {"ok": True}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        if name == "CLOSE_CAMERA":
            client_id = cmd.get("client_id") or f"anon:{id(cmd)}"
            try:
                mc.close(client_id)
                return {"ok": True}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        handlers = {
            "GET_FRAME":               mc.get_frame,
            "GET_GAIN":                mc.get_gain,
            "GET_EXPOSURE":            mc.get_exposure,
            "GET_GAIN_RANGE":          mc.get_gain_range,
            "GET_EXPOSURE_RANGE":      mc.get_exposure_range,
            "GET_TRIGGER_MODE":        mc.get_trigger_mode,
            "GET_TRIGGER_MODE_OPTIONS": mc.get_trigger_mode_options,
        }

        if name == "SET_GAIN":
            return mc.set_gain(cmd.get("value", 0.0))
        if name == "SET_EXPOSURE":
            return mc.set_exposure(cmd.get("value", 0.0))
        if name == "SET_TRIGGER_MODE":
            return mc.set_trigger_mode(cmd.get("value", "Off"))

        if name == "SET_DEFAULTS":
            resp = mc.set_defaults(
                gain=cmd.get("gain"),
                exposure=cmd.get("exposure"),
                trigger_mode=cmd.get("trigger_mode"),
                roi=cmd.get("roi"),
                norm_reference=cmd.get("norm_reference"),
            )
            self._save_defaults()
            return resp
        if name == "GET_DEFAULTS":
            return mc.get_defaults()

        handler = handlers.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown command: {name!r}"}
        try:
            return handler()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
