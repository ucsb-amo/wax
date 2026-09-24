"""LinkLatch: remember that an instrument's network link is down.

Host-side instrument drivers that are called from kernels (via RPC) must not
stall a run when their LAN link is broken. Without a latch, every call pays
the full connect/reply timeout again -- one or two seconds per query, several
queries per shot -- for the whole run.

A ``LinkLatch`` trips on the first failure and tells callers to skip the
device (``should_skip()``) until ``retry_after`` seconds have passed; the
next attempt after that either clears the latch or re-arms it for another
cooldown. It prints once when the link goes down and once when it comes back,
so the terminal record shows the outage without a line per shot.

Drivers raise :class:`LinkDownError` when they skip a call; callers that
already treat comms failures as "value unknown" need no change.
"""

import time

# Seconds between retries once a link has been declared down.
T_LINK_RETRY = 30.


class LinkDownError(ConnectionError):
    """Raised when a call is skipped because the link latch is tripped."""


class LinkLatch:
    def __init__(self, name, retry_after=T_LINK_RETRY, clock=time.monotonic):
        self.name = name
        self.retry_after = retry_after
        self._clock = clock
        self._down = False
        self._retry_at = 0.
        self.last_error = None

    @property
    def down(self) -> bool:
        return self._down

    def should_skip(self) -> bool:
        """True while the link is down and the retry cooldown has not elapsed."""
        return self._down and self._clock() < self._retry_at

    def raise_if_skipping(self):
        if self.should_skip():
            raise LinkDownError(
                f"{self.name}: link down since "
                f"{self.last_error!r}; next retry in "
                f"{self._retry_at - self._clock():.0f} s")

    def trip(self, error):
        """Record a failure. Prints once per outage, then re-arms the cooldown."""
        if not self._down:
            print(f"*** {self.name}: link down ({error!r}). Calls will be "
                  f"skipped and retried every {self.retry_after:.0f} s. ***")
        self._down = True
        self.last_error = error
        self._retry_at = self._clock() + self.retry_after

    def clear(self):
        """Record a success. Prints once when a down link comes back."""
        if self._down:
            print(f"{self.name}: link back.")
        self._down = False
        self.last_error = None
