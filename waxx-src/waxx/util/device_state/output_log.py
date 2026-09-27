"""The terminal output of a process the monitor server runs, kept for GUIs.

The server runs experiment files itself -- a run loop's runs
(:mod:`~waxx.util.device_state.run_loop`), the reset experiment
(:mod:`~waxx.util.device_state.state_reset`) -- and the Device Control GUI's
Sequences tab shows their output under each card.  :class:`OutputLog` keeps
the last lines, numbered from 1 since the server started, so a GUI asks only
for what it has not seen (``since(after)``) and can tell when lines it never
saw were dropped (``first > after + 1``) or the server restarted
(``next <= after``).

Only a view: the server log keeps every line as well.
"""

from __future__ import annotations

import threading
import time
from collections import deque

#: Lines kept per process (older ones are dropped).
OUTPUT_LINES = 2000
#: Most lines one ``since`` reply carries; a GUI asks again for the rest.
OUTPUT_REPLY_LINES = 500


class OutputLog:
    """Numbered lines, oldest dropped past ``maxlen``.  Thread-safe."""

    def __init__(self, maxlen: int = OUTPUT_LINES):
        self._lock = threading.Lock()
        self._lines: deque = deque(maxlen=int(maxlen))
        self._next = 1

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append((self._next, str(line)))
            self._next += 1

    def mark(self, text: str, clock=time.time) -> None:
        """A separator line of the server's own (a run starting, how it ended)."""
        self.append(f"── {time.strftime('%H:%M:%S', time.localtime(clock()))} {text} ──")

    def since(self, after: int = 0, limit: int = OUTPUT_REPLY_LINES) -> dict:
        """The lines numbered above ``after``, at most ``limit`` of them
        (the oldest first): ``{"lines", "first", "next", "more"}``, where
        ``first`` is the number of the first line returned (``next`` when none
        are), ``next`` the number after the last one returned, and ``more``
        says lines past ``limit`` are waiting."""
        try:
            after = max(int(after), 0)
        except (TypeError, ValueError):
            after = 0
        with self._lock:
            pending = [(n, text) for n, text in self._lines if n > after]
            nxt = self._next
        new = pending[:max(int(limit), 1)]
        return {"lines": [text for _, text in new],
                "first": new[0][0] if new else nxt,
                "next": new[-1][0] + 1 if new else nxt,
                "more": len(pending) > len(new)}
