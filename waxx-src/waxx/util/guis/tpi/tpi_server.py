"""TpiDeviceServer — WaxxServer that owns all TPI-1005-A signal generators on this machine.

Server-ID format:
    ``"tpi_server:<hostname>"``       (instance_index == 0, the normal case)
    ``"tpi_server:<hostname>:<n>"``   (instance_index > 0, rare multi-instance)

Clients discover any number of these servers via ``discover_prefix("tpi_server:")``.

ZMQ REP protocol — all messages are pickle-serialised dicts.

Commands sent by client → replies from server:

    {"cmd": "LIST_DEVICES"}
        → {"ok": True, "pub_port": int,
           "devices": [{"serial": str, "model": str, "firmware": str,
                        "port": str, "is_open": bool}, ...]}

    {"cmd": "RESCAN_DEVICES"}
        → {"ok": True, "pub_port": int, "devices": [...]}

    {"cmd": "GET_PUB_PORT"}
        → {"ok": True, "pub_port": int}

    {"cmd": "GET_STATE", "serial": "299"}
        → {"ok": True, "rf_on": bool, "freq_mhz": float, "level_dbm": int}
        | {"ok": False, "error": str}

    {"cmd": "SET_RF", "serial": "299", "on": True}
        → {"ok": True} | {"ok": False, "error": str}

    {"cmd": "SET_FREQ", "serial": "299", "mhz": 433.920}
        → {"ok": True, "freq_mhz": float} | {"ok": False, "error": str}

    {"cmd": "SET_LEVEL", "serial": "299", "dbm": -10}
        → {"ok": True, "level_dbm": int} | {"ok": False, "error": str}

ZMQ PUB socket — state broadcast, pickled dicts, at ~2 Hz per device:

    {"type": "state", "server_id": str, "serial": str,
     "rf_on": bool, "freq_mhz": float, "level_dbm": int, "timestamp": float}
"""
from __future__ import annotations

import logging
import pickle
import socket
import threading
import time
from typing import Optional

import zmq

from waxx.control.misc.tpi import TPI1005A, TPIError, find_all_devices
from waxx.util.comms_server.waxx_server import WaxxServer

logger = logging.getLogger(__name__)

TPI_SERVER_PREFIX = "tpi_server:"
POLL_INTERVAL = 0.5


# ---------------------------------------------------------------------------
# Per-device state machine
# ---------------------------------------------------------------------------

class _ManagedDevice:
    """Wraps one physical TPI-1005-A by serial number."""

    def __init__(self, port: str) -> None:
        self.port: str = port
        self.serial_number: str = ""
        self.model: str = ""
        self.firmware: str = ""
        self._device: Optional[TPI1005A] = None
        self._is_open: bool = False
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        dev = TPI1005A(self.port).open()
        with self._lock:
            self._device = dev
            self.model = dev.get_model()
            self.serial_number = dev.get_serial()
            self.firmware = dev.get_firmware()
            self._is_open = True
        logger.info("[TpiServer] Opened %s (%s) on %s", self.model, self.serial_number, self.port)

    def close(self) -> None:
        with self._lock:
            if not self._is_open:
                return
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
            self._is_open = False
        logger.info("[TpiServer] Closed device on %s", self.port)

    def get_state(self) -> dict:
        with self._lock:
            if not self._is_open or self._device is None:
                return {"ok": False, "error": "device not open"}
            try:
                return {
                    "ok": True,
                    "rf_on": self._device.get_rf(),
                    "freq_mhz": self._device.get_freq(),
                    "level_dbm": self._device.get_level(),
                }
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

    def set_rf(self, on: bool) -> dict:
        with self._lock:
            if not self._is_open or self._device is None:
                return {"ok": False, "error": "device not open"}
            try:
                self._device.set_rf(on)
                return {"ok": True}
            except TPIError as exc:
                return {"ok": False, "error": str(exc)}

    def set_freq(self, mhz: float) -> dict:
        with self._lock:
            if not self._is_open or self._device is None:
                return {"ok": False, "error": "device not open"}
            try:
                self._device.set_freq(mhz)
                actual = self._device.get_freq()
                return {"ok": True, "freq_mhz": actual}
            except (TPIError, ValueError) as exc:
                return {"ok": False, "error": str(exc)}

    def set_level(self, dbm: int) -> dict:
        with self._lock:
            if not self._is_open or self._device is None:
                return {"ok": False, "error": "device not open"}
            try:
                self._device.set_level(dbm)
                actual = self._device.get_level()
                return {"ok": True, "level_dbm": actual}
            except TPIError as exc:
                return {"ok": False, "error": str(exc)}

    def info_dict(self) -> dict:
        return {
            "serial": self.serial_number,
            "model": self.model,
            "firmware": self.firmware,
            "port": self.port,
            "is_open": self._is_open,
        }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class TpiDeviceServer(WaxxServer):
    """Manages all TPI-1005-A devices attached to this machine.

    Discovers and opens devices at startup.  Clients send ZMQ REQ commands to
    read or change device state.  State is also broadcast continuously on a
    ZMQ PUB socket so subscribers receive updates without polling.

    Args:
        instance_index: Use 0 (default) for the sole server on this host.
            Increment for rare multi-instance deployments on one machine.
        port: ZMQ REP bind port.  0 lets the OS choose an ephemeral port.
    """

    def __init__(self, instance_index: int = 0, port: int = 0) -> None:
        hostname = socket.gethostname()
        sid = (
            f"{TPI_SERVER_PREFIX}{hostname}"
            if instance_index == 0
            else f"{TPI_SERVER_PREFIX}{hostname}:{instance_index}"
        )

        from waxx.util.comms_server.waxx_client import discover as _discover
        if _discover(sid, timeout=1.5) is not None:
            raise RuntimeError(
                f"A TpiDeviceServer with ID '{sid}' is already running on the network. "
                f"Stop the existing instance first, or start this one with a different "
                f"instance_index."
            )

        WaxxServer.__init__(self, sid, port)
        self._devices: dict[str, _ManagedDevice] = {}
        self._running = False
        self._pub_port: int = 0
        self._enumerate_devices()

    # ------------------------------------------------------------------ #
    # Device enumeration
    # ------------------------------------------------------------------ #

    def _enumerate_devices(self) -> None:
        """Discover and open all connected TPI devices without disturbing already-open ones."""
        existing_by_port = {d.port: (serial, d) for serial, d in self._devices.items()}
        new: dict[str, _ManagedDevice] = {}

        for port in find_all_devices():
            if port in existing_by_port:
                serial, md = existing_by_port[port]
                new[serial] = md
            else:
                md = _ManagedDevice(port)
                try:
                    md.open()
                    if not md.serial_number:
                        md.close()
                        logger.warning("[TpiServer] Could not read serial from %s; skipping", port)
                    elif md.serial_number in new:
                        md.close()
                        logger.error(
                            "[TpiServer] Serial %r appears on both %s and %s — "
                            "duplicate firmware serial; skipping second device",
                            md.serial_number, new[md.serial_number].port, port,
                        )
                    else:
                        new[md.serial_number] = md
                except Exception as exc:
                    logger.error("[TpiServer] Failed to open %s: %s", port, exc)

        for serial, md in self._devices.items():
            if serial not in new:
                md.close()

        self._devices = new
        logger.info(
            "[TpiServer] %d device(s): %s",
            len(self._devices),
            list(self._devices.keys()),
        )

    # ------------------------------------------------------------------ #
    # Server lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Bind sockets, start beacon and poll thread, enter the REP request loop.

        Blocks until ``stop()`` is called or the process exits.
        """
        context = zmq.Context()

        rep_socket = context.socket(zmq.REP)
        rep_port = rep_socket.bind_to_random_port("tcp://0.0.0.0")
        self._waxx_port = rep_port

        pub_socket = context.socket(zmq.PUB)
        self._pub_port = pub_socket.bind_to_random_port("tcp://0.0.0.0")

        self._running = True
        self._start_beacon()

        poll_thread = threading.Thread(
            target=self._poll_loop,
            args=(pub_socket,),
            name=f"TpiPoll-{self._waxx_server_id}",
            daemon=True,
        )
        poll_thread.start()

        local_ip = self._get_local_ip() or "unknown"
        logger.info(
            "[TpiServer] REP port %d  PUB port %d  IP %s  (server_id=%s)",
            rep_port, self._pub_port, local_ip, self._waxx_server_id,
        )

        try:
            self._req_loop(rep_socket)
        finally:
            self._running = False
            self._stop_beacon()
            poll_thread.join(timeout=2.0)
            for md in self._devices.values():
                try:
                    md.close()
                except Exception:
                    pass
            pub_socket.close()
            rep_socket.close()
            context.term()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    # Request dispatch
    # ------------------------------------------------------------------ #

    def _req_loop(self, sock: zmq.Socket) -> None:
        sock.setsockopt(zmq.RCVTIMEO, 500)
        while self._running:
            try:
                raw = sock.recv()
            except zmq.Again:
                continue
            try:
                cmd = pickle.loads(raw)
                reply = self._dispatch(cmd)
            except Exception as exc:
                reply = {"ok": False, "error": f"dispatch error: {exc}"}
            sock.send(pickle.dumps(reply))

    def _dispatch(self, cmd: dict) -> dict:
        name = cmd.get("cmd", "")

        if name == "GET_PUB_PORT":
            return {"ok": True, "pub_port": self._pub_port}

        if name == "LIST_DEVICES":
            return {
                "ok": True,
                "pub_port": self._pub_port,
                "devices": [d.info_dict() for d in self._devices.values()],
            }

        if name == "RESCAN_DEVICES":
            self._enumerate_devices()
            return {
                "ok": True,
                "pub_port": self._pub_port,
                "devices": [d.info_dict() for d in self._devices.values()],
            }

        serial = cmd.get("serial", "")
        if serial not in self._devices:
            return {"ok": False, "error": f"Unknown serial: {serial!r}"}

        md = self._devices[serial]

        if name == "GET_STATE":
            return md.get_state()
        if name == "SET_RF":
            return md.set_rf(bool(cmd.get("on", False)))
        if name == "SET_FREQ":
            return md.set_freq(float(cmd.get("mhz", 0.0)))
        if name == "SET_LEVEL":
            return md.set_level(int(cmd.get("dbm", 0)))

        return {"ok": False, "error": f"Unknown command: {name!r}"}

    # ------------------------------------------------------------------ #
    # State broadcast
    # ------------------------------------------------------------------ #

    def _poll_loop(self, pub_socket: zmq.Socket) -> None:
        """Daemon thread: poll all devices and publish state on the PUB socket."""
        while self._running:
            for serial, md in list(self._devices.items()):
                state = md.get_state()
                if state.get("ok"):
                    msg = {
                        "type": "state",
                        "server_id": self._waxx_server_id,
                        "serial": serial,
                        "rf_on": state["rf_on"],
                        "freq_mhz": state["freq_mhz"],
                        "level_dbm": state["level_dbm"],
                        "timestamp": time.time(),
                    }
                    try:
                        pub_socket.send(pickle.dumps(msg), zmq.NOBLOCK)
                    except Exception:
                        pass
            time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="TPI-1005-A multi-device server")
    parser.add_argument("--instance", type=int, default=0,
                        help="Instance index for multi-server on one host (default 0)")
    args = parser.parse_args()

    try:
        srv = TpiDeviceServer(instance_index=args.instance)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        srv.start()
    except KeyboardInterrupt:
        srv.stop()
