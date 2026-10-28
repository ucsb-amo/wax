"""ComLink: the serial (COM) device a server owns, opened after the server is up.

Every lab server that owns a COM port follows the same rules, and this class
is where they live:

* The server binds its socket and beacons first, then calls :meth:`ComLink.start`.
  The port is opened on the link's own thread and, while it fails, retried
  with backoff (``BACKOFF_S``: 1, 2, 5, 10, then every 30 s).  A missing or
  failing device never stops the server, and never blocks its requests.
* When the server's own I/O fails it calls :meth:`report_error`: the device is
  closed, and the link reopens it on the same backoff.
* :meth:`disconnect` (``hold=True``, a person's Disconnect) closes the device
  and holds it closed: no reconnect of any kind runs until :meth:`connect`
  releases the hold.  A subclass may refuse a disconnect (``ComLinkRefused``)
  when closing the port now would be unsafe.
* :meth:`snapshot` is the ``"com"`` dict of the server's snapshot (the
  dashboard's COM pill; ``SerialSnapshot`` shape plus ``held``), and every
  state change is advertised in the server's beacon too
  (``NetServer.set_com_state``: ``com`` and ``com_port``), so other programs
  can tell which port a server owns, and where it is with it, without
  opening anything.

States (``NetServer.COM_STATES``): ``connecting`` (an open is in progress),
``connected``, ``error`` (the last open failed or the device failed; a retry
is scheduled unless auto-reconnect is off), ``disconnected`` (not started,
stopped, or auto-reconnect off after a failure), ``held``.

A subclass fills in the device-specific part::

    class PDXCLink(ComLink):
        def _open(self):                 # open + handshake; raise on failure
            dev = PDXC(self.port)
            dev.initialize()
            return dev
        def _close(self, dev):
            dev.close()

and optionally ``_on_connected(dev)`` (after every successful open) and
``_disconnect_refusal()`` (the reason a disconnect is refused now, or None).

Locking: ``io_lock`` (an ``RLock``; pass the server's own device lock if it
has one, which must then be re-entrant) is held while the device is opened
and closed, so a close never lands in the middle of the server's I/O.  The
server reads the current handle with :attr:`device` (None while not
connected) or :meth:`require`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

__all__ = ["ComLink", "ComLinkRefused", "ComLinkNotConnected", "COM_STATES"]

COM_STATES = ("connecting", "connected", "error", "disconnected", "held")


class ComLinkRefused(RuntimeError):
    """A disconnect (or connect) the link's owner refuses right now."""


class ComLinkNotConnected(RuntimeError):
    """The device is not open (see the link's state)."""


class ComLink:
    """One serial device of a server (module docstring)."""

    #: Waits between failed opens (the last repeats).
    BACKOFF_S = (1.0, 2.0, 5.0, 10.0, 30.0)
    #: A device that fails within this long of being opened counts as a failed
    #: open (the backoff grows), so one that opens but fails at once is not
    #: reopened every second; after a longer good run it starts again at 1 s.
    STABLE_S = 5.0
    #: How long :meth:`stop` waits for the link's thread and the I/O lock.
    STOP_TIMEOUT_S = 3.0
    #: How long :meth:`disconnect` waits for the I/O lock (an open or a read in
    #: progress); after that the hold is in force anyway and the link's thread
    #: closes the device as soon as the lock is free.
    DISCONNECT_LOCK_WAIT_S = 2.0

    def __init__(
        self,
        port: str,
        *,
        baud: Optional[int] = None,
        server: Any = None,
        name: Optional[str] = None,
        auto_reconnect: bool = True,
        io_lock: Any = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.port = str(port)
        self.baud = baud
        self.name = name or self.port
        self.io_lock = io_lock if io_lock is not None else threading.RLock()
        self._server = server
        self._auto = bool(auto_reconnect)
        self._log = logger or logging.getLogger("waxx.com_link")

        self._lock = threading.Lock()          # guards the fields below
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._device: Any = None
        self._state = "disconnected"
        self._want = False                     # started, and not given up on
        self._held = False
        self._last_error: Optional[str] = None
        self._last_rx: Optional[float] = None
        self._failures = 0                     # failed opens (or quick failures) in a row
        self._reconnects = 0                   # successful opens after the first
        self._opened_once = False
        self._connected_at: Optional[float] = None
        self._next_try: Optional[float] = None  # time.monotonic() of the next open
        self._publish(self._state)

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #

    def _open(self) -> Any:
        """Open the device (and handshake); return its handle, raise on failure."""
        raise NotImplementedError

    def _close(self, device: Any) -> None:
        """Close the device handle ``_open`` returned."""
        close = getattr(device, "close", None)
        if callable(close):
            close()

    def _on_connected(self, device: Any) -> None:
        """After every successful open (re-initialise, resume work)."""

    def _disconnect_refusal(self) -> Optional[str]:
        """The reason a disconnect is refused now, or None to allow it."""
        return None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the link's thread; it opens the port at once, then keeps it
        open (unless held).  Returns immediately."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping.clear()
            self._want = True
            self._next_try = None
            self._thread = threading.Thread(target=self._run, name=f"ComLink-{self.name}",
                                            daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the thread and close the device (bounded: never waits longer
        than ``STOP_TIMEOUT_S`` for the thread or the I/O lock)."""
        self._stopping.set()
        with self._lock:
            self._want = False
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.STOP_TIMEOUT_S)
        got = self.io_lock.acquire(timeout=self.STOP_TIMEOUT_S)
        try:
            with self._lock:
                dev, self._device = self._device, None
            if dev is not None:
                self._safe_close(dev)
        finally:
            if got:
                self.io_lock.release()
        self._set_state("held" if self._held else "disconnected")

    # ------------------------------------------------------------------ #
    # Requests (any thread; none waits for an open)
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        """Release a hold (if any) and open the port as soon as possible:
        the attempt runs on the link's thread, now, even mid-backoff."""
        with self._lock:
            was_held = self._held
            self._held = False
            self._want = True
            self._next_try = None
        if was_held:
            self._log.info("[ComLink] %s: hold released; connecting", self.name)
        if self._device is None:
            self._set_state("connecting")
        if self._thread is None or not self._thread.is_alive():
            self.start()
        self._wake.set()

    def disconnect(self, hold: bool = True, force: bool = False) -> None:
        """Close the device now.  ``hold`` (a person's Disconnect): nothing
        reconnects it until :meth:`connect`.  Raises ``ComLinkRefused`` when
        the owner refuses (``_disconnect_refusal``) unless ``force``."""
        if not force:
            reason = self._disconnect_refusal()
            if reason:
                raise ComLinkRefused(reason)
        with self._lock:
            if hold:
                self._held = True
            else:
                self._want = False
            self._next_try = None
        if self.io_lock.acquire(timeout=self.DISCONNECT_LOCK_WAIT_S):
            try:
                self._close_now()
            finally:
                self.io_lock.release()
        else:
            self._log.warning("[ComLink] %s: the device is busy (an open or a read in progress); "
                              "it is closed as soon as that ends", self.name)
        self._log.info("[ComLink] %s: disconnected%s", self.name,
                       " and held (no reconnect until released)" if hold else "")
        self._set_state(self._idle_state())
        self._wake.set()

    def _close_now(self) -> None:
        """Close the device (caller holds io_lock)."""
        with self._lock:
            dev, self._device = self._device, None
        if dev is not None:
            self._safe_close(dev)

    def report_error(self, exc: BaseException, where: str = "") -> None:
        """The server's I/O on the device failed: close it; the link reopens
        it on its backoff (unless held, or auto-reconnect is off)."""
        with self.io_lock:
            with self._lock:
                dev, self._device = self._device, None
                if dev is None:
                    return                      # already closed / handled
                self._last_error = f"{type(exc).__name__}: {exc}"
                now = time.monotonic()
                up = self._connected_at
                if up is not None and now - up < self.STABLE_S:
                    self._failures += 1         # opened, then failed at once
                else:
                    self._failures = 1
                if self._auto and self._want and not self._held:
                    delay = self.BACKOFF_S[min(self._failures - 1, len(self.BACKOFF_S) - 1)]
                    self._next_try = now + delay
                else:
                    self._want = self._want and self._auto
                    self._next_try = None
            self._safe_close(dev)
        self._log.warning("[ComLink] %s: device failed%s (%s: %s); %s", self.name,
                          f" during {where}" if where else "", type(exc).__name__, exc,
                          "reconnecting" if self._auto and not self._held else "not reconnecting")
        self._set_state("error")
        self._wake.set()

    def note_rx(self) -> None:
        """The device answered (for ``last_rx_seconds_ago``)."""
        self._last_rx = time.monotonic()

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> Any:
        """The open device handle, or None."""
        return self._device

    def require(self) -> Any:
        """The open device handle; ``ComLinkNotConnected`` when there is none."""
        dev = self._device
        if dev is None:
            raise ComLinkNotConnected(f"{self.name} is not connected ({self.state})")
        return dev

    @property
    def state(self) -> str:
        return self._state

    @property
    def held(self) -> bool:
        return self._held

    @property
    def connected(self) -> bool:
        return self._device is not None

    def snapshot(self) -> dict:
        """The ``"com"`` dict for the server's snapshot (dashboard COM pill)."""
        with self._lock:
            rx = self._last_rx
            nxt = self._next_try
            return {
                "port": self.port,
                "baud": self.baud,
                "status": self._state,
                "held": self._held,
                "last_error": self._last_error,
                "last_rx_seconds_ago": None if rx is None else round(time.monotonic() - rx, 3),
                "reconnect_attempts": 0 if self._device is not None else self._failures,
                "reconnects": self._reconnects,
                "next_retry_s": (None if nxt is None or self._held or self._device is not None
                                 else round(max(0.0, nxt - time.monotonic()), 1)),
                "config_valid": True,
            }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _idle_state(self) -> str:
        if self._held:
            return "held"
        if not self._want or self._stopping.is_set():
            return "disconnected"
        return "error" if self._last_error else "connecting"

    def _set_state(self, state: str) -> None:
        with self._lock:
            changed = state != self._state
            self._state = state
        if changed:
            self._publish(state)

    def _publish(self, state: str) -> None:
        fn = getattr(self._server, "set_com_state", None)
        if callable(fn):
            try:
                fn(state, self.port)
            except Exception:                   # the beacon field is best-effort
                self._log.debug("[ComLink] %s: set_com_state failed", self.name, exc_info=True)

    def _safe_close(self, dev: Any) -> None:
        try:
            self._close(dev)
        except Exception as exc:
            self._log.warning("[ComLink] %s: close raised %s: %s", self.name,
                              type(exc).__name__, exc)

    def _run(self) -> None:
        while not self._stopping.is_set():
            with self._lock:
                now = time.monotonic()
                due = (self._want and not self._held and self._device is None
                       and (self._next_try is None or now >= self._next_try))
                late_close = (self._held or not self._want) and self._device is not None
                wait = (0.5 if self._next_try is None
                        else max(0.02, min(0.5, self._next_try - now)))
            if late_close:                      # a disconnect that could not wait for the lock
                with self.io_lock:
                    self._close_now()
                continue
            if due:
                self._attempt()
                continue
            self._wake.wait(wait)
            self._wake.clear()

    def _attempt(self) -> None:
        self._set_state("connecting")
        try:
            with self.io_lock:
                with self._lock:
                    abandon = self._held or not self._want or self._stopping.is_set()
                if abandon:
                    self._set_state(self._idle_state())
                    return
                dev = self._open()
                with self._lock:
                    keep = self._want and not self._held and not self._stopping.is_set()
                    if keep:
                        self._device = dev
                if not keep:                    # held / stopped while opening
                    self._safe_close(dev)
                    self._set_state(self._idle_state())
                    return
        except Exception as exc:
            with self._lock:
                self._failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                if self._auto:
                    delay = self.BACKOFF_S[min(self._failures - 1, len(self.BACKOFF_S) - 1)]
                    self._next_try = time.monotonic() + delay
                else:
                    delay = None
                    self._want = False
                    self._next_try = None
                failures = self._failures
            log = self._log.warning if failures <= len(self.BACKOFF_S) else self._log.debug
            log("[ComLink] %s: open failed (attempt %d: %s: %s)%s", self.name, failures,
                type(exc).__name__, exc,
                f"; retrying in {delay:g} s" if delay is not None else "; not retrying")
            self._set_state("error")
            return
        with self._lock:
            if self._opened_once:
                self._reconnects += 1
            self._opened_once = True
            # _failures is kept until the device has stayed up STABLE_S
            # (report_error), so a quick failure keeps growing the backoff.
            self._connected_at = time.monotonic()
            self._last_error = None
            self._last_rx = time.monotonic()
            self._next_try = None
        self._log.info("[ComLink] %s: connected", self.name)
        self._set_state("connected")
        try:
            self._on_connected(dev)
        except Exception as exc:
            self.report_error(exc, where="on_connected")
