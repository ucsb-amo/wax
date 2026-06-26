"""Headless TCP server for Keysight DC current supplies.

Owns the VXI11 connections to each supply, polls them periodically, and
serves the cached snapshot to GUI clients.  Clients should never talk to
the supplies directly — go through this server so the supplies are not
hammered by N concurrent dashboards.

Wire protocol (line-terminated, case-insensitive):

  GET_SNAPSHOT      -> JSON list, one entry per supply
                       [{"ip", "max_current", "connected", "output_on",
                         "current_a", "status", "error"}, ...]
  GET_STATUS        -> JSON {"connected": bool, "supplies": int}
  TURN_ON <ip>      -> JSON {"ok": bool, "error": str|None}
  CLEAR_PROT <ip>   -> JSON {"ok": bool, "error": str|None}
  RECONNECT <ip>    -> JSON {"ok": bool, "error": str|None}
"""
from __future__ import annotations

import atexit
import json
import logging
import signal
import socket
import threading
import time
from typing import Optional

import vxi11

from waxx.util.comms_server.waxx_server import WaxxServer

LOGGER = logging.getLogger("keysight_server")
LOGGER.setLevel(logging.INFO)

SERVER_ID = "keysight"
_VXI11_TIMEOUT_S = 2.0  # max seconds to wait for any VXI11 round-trip


# ---------------------------------------------------------------------------
# Per-supply hardware wrapper (server side only).
# ---------------------------------------------------------------------------


class _Supply:
    """Holds a VXI11 connection to one Keysight DC supply.

    All hardware I/O is serialised by ``self._lock`` so the poll loop and
    incoming RPCs (TURN_ON / CLEAR_PROT) cannot collide on the same socket.
    """

    def __init__(self, ip: str, max_current: int) -> None:
        self.ip = ip
        self.max_current = int(max_current)
        self._lock = threading.Lock()
        self._instr: Optional[vxi11.Instrument] = None
        # Cached snapshot (read by RPC threads under self._snap_lock).
        self._snap_lock = threading.Lock()
        self._snapshot: dict = {
            "ip": ip,
            "max_current": self.max_current,
            "connected": False,
            "output_on": None,
            "current_a": None,
            "status": 0,
            "error": None,
        }

    # ----- connection management ----------------------------------------- #

    def connect(self) -> None:
        with self._lock:
            try:
                instr = vxi11.Instrument(self.ip)
                instr.timeout = _VXI11_TIMEOUT_S
                # Configure: live inhibit pin mode.
                instr.write("OUTP:INH:MODE LIVE")
                self._instr = instr
            except Exception as exc:
                self._instr = None
                self._update_snapshot(connected=False, error=str(exc))
                raise

    def close(self) -> None:
        with self._lock:
            if self._instr is not None:
                try:
                    self._instr.close()
                except Exception:
                    pass
                self._instr = None

    # ----- poll (called from the poll thread) ---------------------------- #

    def poll(self) -> None:
        """Read current / output / status into the cached snapshot."""
        # Step 1: hold the lock only long enough to get (or lazily create) the
        # instrument reference.  All VXI11 network I/O happens outside the lock
        # so that close() (called from stop()) is never blocked behind a
        # potentially-hanging query.
        with self._lock:
            if self._instr is None:
                # Try to (re)connect lazily.
                try:
                    instr = vxi11.Instrument(self.ip)
                    instr.timeout = _VXI11_TIMEOUT_S
                    instr.write("OUTP:INH:MODE LIVE")
                    self._instr = instr
                except Exception as exc:
                    self._update_snapshot(connected=False, error=str(exc))
                    return
            instr = self._instr
        # Step 2: do all VXI11 I/O outside the lock.
        try:
            status = int(instr.ask("STAT:QUES:COND?"))
            output_on = bool(int(instr.ask("OUTP?")))
            current_a: Optional[float]
            if output_on and status == 0:
                current_a = float(instr.ask(":MEASure:CURRent:DC?"))
            else:
                current_a = None
        except Exception as exc:
            # Drop the broken connection so the next poll reconnects.
            with self._lock:
                if self._instr is instr:  # only clear if nobody replaced it
                    try:
                        instr.close()
                    except Exception:
                        pass
                    self._instr = None
            self._update_snapshot(connected=False, error=str(exc))
            return
        self._update_snapshot(
            connected=True,
            output_on=output_on,
            current_a=current_a,
            status=status,
            error=None,
        )

    # ----- RPC actions (called from accept-loop handler threads) --------- #

    def turn_on(self) -> dict:
        return self._safe_write("OUTP ON")

    def clear_protect(self) -> dict:
        result = self._safe_write("OUTP:PROT:CLE")
        if result["ok"]:
            self._update_snapshot(status=0)
        return result

    def reconnect(self) -> dict:
        self.close()
        try:
            self.connect()
            return {"ok": True, "error": None}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ----- internals ----------------------------------------------------- #

    def _safe_write(self, cmd: str) -> dict:
        with self._lock:
            instr = self._instr
        if instr is None:
            return {"ok": False, "error": "not connected"}
        try:
            instr.write(cmd)
            return {"ok": True, "error": None}
        except Exception as exc:
            with self._lock:
                if self._instr is instr:
                    try:
                        instr.close()
                    except Exception:
                        pass
                    self._instr = None
            self._update_snapshot(connected=False, error=str(exc))
            return {"ok": False, "error": str(exc)}

    def _update_snapshot(self, **kwargs) -> None:
        with self._snap_lock:
            self._snapshot.update(kwargs)

    def get_snapshot(self) -> dict:
        with self._snap_lock:
            return dict(self._snapshot)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class KeysightServer(WaxxServer):
    """Polls all configured Keysight supplies and serves snapshots over TCP."""

    def __init__(
        self,
        supplies: list[tuple[int, str]],
        host: str = "0.0.0.0",
        port: int = 0,
        poll_interval_s: float = 0.5,
    ) -> None:
        WaxxServer.__init__(self, SERVER_ID, port)
        self.host = host
        self.poll_interval_s = float(poll_interval_s)
        if not supplies:
            raise ValueError("KeysightServer requires at least one supply")
        cfg = list(supplies)
        # Index supplies by IP so RPCs (TURN_ON <ip>) can target one quickly.
        self._supplies: dict[str, _Supply] = {
            ip: _Supply(ip, max_current) for max_current, ip in cfg
        }
        self._order = [ip for _, ip in cfg]

        self.running = False
        self._server_socket: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stopped = False

    # ------------------------------------------------------------------ #
    # Public state accessors
    # ------------------------------------------------------------------ #

    def get_snapshot(self) -> list[dict]:
        return [self._supplies[ip].get_snapshot() for ip in self._order]

    def get_status(self) -> dict:
        connected = any(
            self._supplies[ip].get_snapshot()["connected"] for ip in self._order
        )
        return {"connected": connected, "supplies": len(self._order)}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self.running:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, 0))
        sock.listen(16)
        self._server_socket = sock
        self._waxx_port = sock.getsockname()[1]
        self._start_beacon()
        self.running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="KeysightAccept",
        )
        self._accept_thread.start()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="KeysightPoll",
        )
        self._poll_thread.start()
        LOGGER.info(
            "Server started on port %d, polling %d supply(ies): %s",
            self._waxx_port, len(self._order), ", ".join(self._order),
        )

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.running = False
        self._stop_beacon()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
        for s in self._supplies.values():
            s.close()
        LOGGER.info("Server stopped")

    # ------------------------------------------------------------------ #
    # Background threads
    # ------------------------------------------------------------------ #

    def _poll_loop(self) -> None:
        # Initial connect (best-effort; reconnect is also handled by poll()).
        for s in self._supplies.values():
            try:
                s.connect()
                LOGGER.info("Connected to %s", s.ip)
            except Exception as exc:
                LOGGER.warning("Initial connect to %s failed: %s", s.ip, exc)
        while self.running:
            for s in self._supplies.values():
                if not self.running:
                    break
                try:
                    s.poll()
                except Exception as exc:
                    LOGGER.debug("poll(%s) raised: %s", s.ip, exc)
            time.sleep(self.poll_interval_s)

    def _accept_loop(self) -> None:
        while self.running:
            try:
                conn, addr = self._server_socket.accept()
            except OSError:
                break
            threading.Thread(
                target=self._handle_client,
                args=(conn, addr),
                daemon=True,
            ).start()

    def _handle_client(self, conn: socket.socket, addr) -> None:
        try:
            conn.settimeout(5.0)
            with conn.makefile("rb") as f:
                line = f.readline().decode("utf-8", errors="replace").strip()
            response = self._dispatch(line)
            conn.sendall((response + "\n").encode("utf-8"))
        except Exception as exc:
            LOGGER.debug("Client handler error (%s): %s", addr, exc)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _dispatch(self, line: str) -> str:
        upper = line.upper()
        if upper == "GET_SNAPSHOT":
            return json.dumps(self.get_snapshot())
        if upper == "GET_STATUS":
            return json.dumps(self.get_status())
        # Targeted RPCs: "TURN_ON <ip>", "CLEAR_PROT <ip>", "RECONNECT <ip>"
        parts = line.split()
        if len(parts) == 2:
            cmd = parts[0].upper()
            ip = parts[1]
            supply = self._supplies.get(ip)
            if supply is None:
                return json.dumps({"ok": False, "error": f"unknown supply: {ip}"})
            if cmd == "TURN_ON":
                return json.dumps(supply.turn_on())
            if cmd == "CLEAR_PROT":
                return json.dumps(supply.clear_protect())
            if cmd == "RECONNECT":
                return json.dumps(supply.reconnect())
        return json.dumps({"error": f"unknown command: {line!r}"})


def main(supplies: list[tuple[int, str]]) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
    )
    server = KeysightServer(supplies=supplies)
    atexit.register(server.stop)

    def _sigterm(signum, frame):  # noqa: ARG001
        server.stop()

    signal.signal(signal.SIGTERM, _sigterm)
    server.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    # No defaults here — running this module directly requires the caller
    # to supply the IP list (the lab-specific config lives in kexp).
    raise SystemExit(
        "Run the experiment-specific entry point instead, e.g.\n"
        "    python -m kexp.util.guis.keysight_monitor.keysight_server_headless"
    )
