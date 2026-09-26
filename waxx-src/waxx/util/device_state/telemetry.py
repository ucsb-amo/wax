"""Measured values for the Device Control GUI (read only).

The device-state JSON says what the hardware was *told*; telemetry says what
it *is*, where something measures it (a supply's output current, the
interlock's state, whether a run is in progress).  Cards show measurements
next to setpoints, never in place of them, and a measurement is only used
while it is fresh.

A :class:`TelemetryProvider` reads one source.  It must be read-only: it gets
the one read call it needs and nothing else (bind ``client.get_snapshot``, not
the client).  The :class:`TelemetryHub` polls each provider on its own thread
at the provider's interval, only while the GUI says it is visible
(:meth:`TelemetryHub.set_active`), so a closed or hidden panel costs nothing.

Samples are keyed ``"provider/key"``.  Their age grows while they sit here:
``samples()`` returns each with ``age_s`` = age at the source when it was
read + time since.  A failing provider keeps its last samples (ageing, so
consumers see them go stale) and reports its error in :meth:`TelemetryHub.status`.

No Qt here; the GUI pulls ``samples()`` on its own timer.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from typing import Callable, Iterable

from waxx.util.device_state.composite import Sample

_LOG = logging.getLogger("waxx.device_control.telemetry")


class TelemetryProvider:
    """One measured source.  Subclasses set ``name`` and ``interval_s`` and
    implement :meth:`poll` (blocking, called on the hub's thread; it may
    raise -- the hub records the error)."""

    name = "provider"
    interval_s = 1.0

    def poll(self) -> dict[str, Sample]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _Slot:
    def __init__(self, provider: TelemetryProvider):
        self.provider = provider
        self.samples: dict[str, tuple[Sample, float]] = {}   # key -> (sample, t_read)
        self.error = ""
        self.t_ok: float | None = None
        self.t_try: float | None = None
        self.thread: threading.Thread | None = None


class TelemetryHub:
    def __init__(self, providers: Iterable[TelemetryProvider] = (),
                 clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._slots = {p.name: _Slot(p) for p in providers}
        self._active = threading.Event()
        self._stop = threading.Event()
        self._started = False

    @property
    def providers(self) -> list[str]:
        return list(self._slots)

    # -- threads -------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for slot in self._slots.values():
            slot.thread = threading.Thread(target=self._run, args=(slot,), daemon=True,
                                           name=f"telemetry-{slot.provider.name}")
            slot.thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._active.set()          # wake waiting threads
        for slot in self._slots.values():
            if slot.thread is not None:
                slot.thread.join(timeout=2.)
            try:
                slot.provider.close()
            except Exception:
                pass

    def set_active(self, active: bool) -> None:
        """Poll only while something shows the values."""
        if active:
            self._active.set()
        else:
            self._active.clear()

    def _run(self, slot: _Slot) -> None:
        while not self._stop.is_set():
            self._active.wait()
            if self._stop.is_set():
                return
            self.poll_once(slot.provider.name)
            self._stop.wait(max(float(slot.provider.interval_s), 0.1))

    # -- polling -------------------------------------------------------------------

    def poll_once(self, name: str) -> None:
        """Read one provider now (the threads call this; so can tests)."""
        slot = self._slots[name]
        t = self._clock()
        try:
            got = slot.provider.poll() or {}
        except Exception as e:
            with self._lock:
                slot.t_try = t
                if slot.error != repr(e):
                    _LOG.info("telemetry %s: %r", name, e)
                slot.error = f"{type(e).__name__}: {e}"
            return
        with self._lock:
            slot.t_try = slot.t_ok = t
            slot.error = ""
            for key, sample in got.items():
                if isinstance(sample, Sample):
                    slot.samples[str(key)] = (sample, t)

    # -- reading -------------------------------------------------------------------

    def samples(self) -> dict[str, Sample]:
        now = self._clock()
        out = {}
        with self._lock:
            for name, slot in self._slots.items():
                for key, (sample, t_read) in slot.samples.items():
                    out[f"{name}/{key}"] = replace(sample, age_s=sample.age_s + (now - t_read))
        return out

    def status(self) -> dict[str, dict]:
        now = self._clock()
        with self._lock:
            return {name: {"ok": not slot.error and slot.t_ok is not None,
                           "error": slot.error,
                           "age_s": None if slot.t_ok is None else now - slot.t_ok}
                    for name, slot in self._slots.items()}


__all__ = ["Sample", "TelemetryHub", "TelemetryProvider"]
