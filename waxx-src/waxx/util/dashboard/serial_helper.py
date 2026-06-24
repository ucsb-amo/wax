"""Hardened serial-port wrapper used by every COM-owning kexp server.

Replaces the ad-hoc ``serial.Serial(...)`` constructor calls scattered across
the codebase with a single, well-tested helper that:

* Opens lazily (never blocks ``__init__``).
* Logs every COM event at the level the dashboard logging contract specifies.
* Distinguishes "no data" (port is alive but PLC silent) from "port gone"
  (USB unplug / driver crash) via the OS report on read.
* Detects "ghost" connections (``is_open`` still True after cable yanked but
  reads return empty) via a configurable staleness threshold.
* Supports an opt-in bounded-backoff auto-reconnect loop on a background thread.
* Serializes all I/O through a lock so a dashboard "disconnect" RPC cannot
  race with an in-flight read.
* Provides a :meth:`snapshot` method whose return value plugs straight into
  the ``ComStatusButton`` UI widget without any reformatting.

Intentionally Qt-free so it can be unit tested in isolation and used by
non-Qt servers (interlock service runs as a Windows service).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import serial  # type: ignore
    import serial.tools.list_ports  # type: ignore
except Exception:  # pragma: no cover - serial is a hard runtime dep
    serial = None  # sentinel; .open() will raise clearly


# Connection states reported by .snapshot().status
STATUS_CONNECTED = "connected"
STATUS_CONNECTING = "connecting"
STATUS_DISCONNECTED = "disconnected"
STATUS_ERROR = "error"


@dataclass
class SerialSnapshot:
    port: str
    baud: int
    status: str
    last_error: Optional[str]
    last_rx_seconds_ago: Optional[float]
    reconnect_attempts: int
    config_valid: bool = True

    def as_dict(self) -> dict:
        return {
            "port": self.port,
            "baud": self.baud,
            "status": self.status,
            "last_error": self.last_error,
            "last_rx_seconds_ago": self.last_rx_seconds_ago,
            "reconnect_attempts": self.reconnect_attempts,
            "config_valid": self.config_valid,
        }


class SerialConnection:
    """Hardened wrapper around :class:`serial.Serial`.

    Parameters
    ----------
    port:
        Serial port name (``"COM5"`` on Windows, ``"/dev/ttyUSB0"`` on Linux).
    baudrate:
        Serial baud rate.
    timeout:
        Per-read timeout in seconds (``serial.Serial.timeout``).  Reads can
        return less data than requested when the timeout expires.
    on_safe_shutdown:
        Optional callable invoked just before the port is closed, intended to
        push the hardware to a known-safe state (e.g. "set output off").  If
        it raises, the exception is logged at ERROR but the close proceeds -
        leaking a serial port is worse than a failed shutdown line.
    reconnect:
        If True, on a ``SerialException`` the helper enters a bounded backoff
        loop and re-opens automatically.  Backoff sequence: 1, 2, 5, 10, then
        30 s capped.  Disabled by default so simple servers behave predictably.
    stale_rx_threshold_s:
        If non-None and no bytes have been received for this many seconds, the
        helper treats the connection as "ghost" and forces a reopen.  Set to
        ``None`` for query-only protocols where silence is normal.
    logger:
        Logger to use.  Defaults to ``"waxx.dashboard.serial"``.
    """

    _BACKOFF_S = (1.0, 2.0, 5.0, 10.0, 30.0)

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = 1.0,
        on_safe_shutdown: Optional[Callable[[object], None]] = None,
        reconnect: bool = False,
        stale_rx_threshold_s: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._port_name = port
        self._baud = int(baudrate)
        self._timeout = float(timeout)
        self._on_safe_shutdown = on_safe_shutdown
        self._reconnect_enabled = bool(reconnect)
        self._stale_threshold = stale_rx_threshold_s

        self._log = logger or logging.getLogger("waxx.dashboard.serial")
        self._lock = threading.RLock()
        self._serial: Optional["serial.Serial"] = None
        self._status: str = STATUS_DISCONNECTED
        self._last_error: Optional[str] = None
        self._last_rx_monotonic: Optional[float] = None
        self._reconnect_attempts: int = 0
        self._reconnect_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._config_valid: bool = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def validate_config(self) -> bool:
        """Check that the configured port exists in the OS port list.

        Logs a WARNING (not ERROR) if the port is missing - a USB-serial device
        that is currently unplugged is a recoverable condition.  Returns True
        if the port enumerates, False otherwise.  Result cached in
        :attr:`config_valid` for inclusion in snapshots.
        """
        if serial is None:
            self._config_valid = False
            self._last_error = "pyserial not installed"
            self._log.error("validate_config: pyserial not installed")
            return False
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
        except Exception as exc:
            self._log.error("validate_config: list_ports failed: %r", exc)
            self._config_valid = False
            return False
        ok = self._port_name in ports
        self._config_valid = ok
        if not ok:
            self._log.warning(
                "validate_config: %s not present (available: %s)",
                self._port_name, ", ".join(ports) or "<none>",
            )
        return ok

    def open(self) -> None:
        """Open the serial port. Blocks until open or raises.

        Must NOT be called on the Qt main thread; servers should call from
        their background "initial connect" thread to keep startup transparent.
        """
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        with self._lock:
            if self._serial is not None and self._serial.is_open:
                self._log.debug("open(%s): already open", self._port_name)
                return
            self._status = STATUS_CONNECTING
            self._log.info("open: port=%s baud=%d", self._port_name, self._baud)
            start = time.monotonic()
            try:
                self._serial = serial.Serial(
                    port=self._port_name,
                    baudrate=self._baud,
                    timeout=self._timeout,
                )
            except Exception as exc:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                self._status = STATUS_ERROR
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._log.error(
                    "open failed: port=%s elapsed_ms=%.1f error=%s",
                    self._port_name, elapsed_ms, self._last_error,
                )
                raise
            elapsed_ms = (time.monotonic() - start) * 1000.0
            self._status = STATUS_CONNECTED
            self._last_error = None
            self._reconnect_attempts = 0
            self._last_rx_monotonic = time.monotonic()
            self._log.info(
                "open success: port=%s elapsed_ms=%.1f", self._port_name, elapsed_ms,
            )

    def close(self) -> None:
        """Close the port, attempting safe-shutdown first.

        Idempotent.  Never raises.
        """
        with self._lock:
            if self._serial is None:
                self._status = STATUS_DISCONNECTED
                return
            ser = self._serial
            if self._on_safe_shutdown is not None and ser.is_open:
                try:
                    self._log.info("close: invoking safe-shutdown callable")
                    self._on_safe_shutdown(ser)
                except Exception as exc:
                    self._log.error("safe-shutdown failed: %r", exc)
            try:
                ser.close()
            except Exception as exc:
                self._log.error("serial.close raised: %r", exc)
            self._serial = None
            self._status = STATUS_DISCONNECTED
            self._last_error = None
            self._log.info("close: port=%s closed", self._port_name)

    def stop(self) -> None:
        """Stop any background reconnect thread and close the port."""
        self._stop_event.set()
        if self._reconnect_thread is not None:
            self._reconnect_thread.join(timeout=2.0)
        self.close()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._serial is not None and self._serial.is_open

    def _check_stale(self) -> None:
        """Force a reopen if no data has arrived in `stale_threshold` seconds."""
        if self._stale_threshold is None or self._last_rx_monotonic is None:
            return
        stale = time.monotonic() - self._last_rx_monotonic
        if stale > self._stale_threshold:
            self._log.warning(
                "read: rx stale for %.1fs (threshold=%.1fs) - forcing reopen",
                stale, self._stale_threshold,
            )
            try:
                self.close()
            finally:
                self._last_rx_monotonic = None
                if self._reconnect_enabled:
                    self._kick_reconnect()
            raise IOError(f"serial rx stale on {self._port_name}")

    def read(self, size: int = 1) -> bytes:
        """Read up to *size* bytes. Returns ``b""`` on timeout.

        Raises :class:`OSError` or :class:`serial.SerialException` on hard
        failure - caller is expected to react (or let the auto-reconnect take
        over if enabled).
        """
        with self._lock:
            if self._serial is None or not self._serial.is_open:
                raise IOError(f"serial port {self._port_name} is not open")
            try:
                data = self._serial.read(size)
            except Exception as exc:
                self._status = STATUS_ERROR
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._log.error(
                    "read failed: port=%s error=%s", self._port_name, self._last_error,
                )
                if self._reconnect_enabled:
                    self._kick_reconnect()
                raise
            if data:
                self._last_rx_monotonic = time.monotonic()
            else:
                self._check_stale()
            return data

    def write(self, payload: bytes, *, description: str = "") -> int:
        """Write *payload* to the port. Returns the number of bytes written."""
        with self._lock:
            if self._serial is None or not self._serial.is_open:
                raise IOError(f"serial port {self._port_name} is not open")
            try:
                truncated = payload[:64]
                self._log.info(
                    "write: port=%s desc=%s payload=%r%s",
                    self._port_name, description or "<unspecified>", truncated,
                    "..." if len(payload) > 64 else "",
                )
                n = self._serial.write(payload)
                self._serial.flush()
                return int(n or 0)
            except Exception as exc:
                self._status = STATUS_ERROR
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._log.error(
                    "write failed: port=%s payload=%r error=%s",
                    self._port_name, payload[:64], self._last_error,
                )
                if self._reconnect_enabled:
                    self._kick_reconnect()
                raise

    # ------------------------------------------------------------------
    # Auto-reconnect
    # ------------------------------------------------------------------

    def _kick_reconnect(self) -> None:
        """Start the bounded-backoff reopen thread if not already running."""
        if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
            return
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop, daemon=True, name=f"SerialReconnect-{self._port_name}",
        )
        self._reconnect_thread.start()

    def _reconnect_loop(self) -> None:
        attempt = 0
        while not self._stop_event.is_set():
            attempt += 1
            self._reconnect_attempts = attempt
            delay = self._BACKOFF_S[min(attempt - 1, len(self._BACKOFF_S) - 1)]
            self._log.warning(
                "reconnect attempt=%d next_delay_s=%.1f port=%s",
                attempt, delay, self._port_name,
            )
            try:
                self.open()
                self._log.info(
                    "reconnect success after attempts=%d port=%s",
                    attempt, self._port_name,
                )
                return
            except Exception:
                # Already logged inside open(); just back off and retry.
                if self._stop_event.wait(delay):
                    break
        self._log.error(
            "reconnect gave up: total_attempts=%d port=%s", attempt, self._port_name,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def snapshot(self) -> SerialSnapshot:
        with self._lock:
            age = (
                None if self._last_rx_monotonic is None
                else time.monotonic() - self._last_rx_monotonic
            )
            return SerialSnapshot(
                port=self._port_name,
                baud=self._baud,
                status=self._status,
                last_error=self._last_error,
                last_rx_seconds_ago=age,
                reconnect_attempts=self._reconnect_attempts,
                config_valid=self._config_valid,
            )


__all__ = [
    "SerialConnection",
    "SerialSnapshot",
    "STATUS_CONNECTED",
    "STATUS_CONNECTING",
    "STATUS_DISCONNECTED",
    "STATUS_ERROR",
]
