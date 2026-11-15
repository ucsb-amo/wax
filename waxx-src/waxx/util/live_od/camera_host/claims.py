"""liveOD's camera host borrowing cameras: ``beacon.camera.lending`` under
liveOD's names.

A Basler camera is listed by the beacon camera server of the PC it is plugged
into, which opens it whenever a viewer looks at it; the Andor is listed by the
SLM spot finder's own server while the spot finder has it open (it serves the
camera for exactly this).  Before liveOD's camera host opens one (at its
start, for a live stream, for a run), it asks that server to let go of it.
The mechanism -- RELINQUISH / RETURN, the renewal, following a server that
restarted or moved -- is beacon's (``beacon.camera.lending``, for any
borrower); here are only liveOD's label for the servers it asks and its log.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import zmq

from beacon.camera import lending as _lending
from beacon.camera.lending import (DEFAULT_TTL_S, LEGACY_PREFIX, V2_PREFIX, Claim, ClaimError,
                                   ClaimRefused, ServerRef)
from beacon.camera.reservations import Holder

logger = logging.getLogger("waxx.live_od.camera_host")

CLIENT_LABEL = "liveOD camera host"
_WHO = "camera host"            # what its log lines start with


def v2_request(ctx: zmq.Context, host: str, port: int, header: dict, timeout_s: float,
               label: str = CLIENT_LABEL) -> dict:
    """``beacon.camera.lending.v2_request`` with liveOD's label."""
    return _lending.v2_request(ctx, host, port, header, timeout_s, label=label)


class DirectoryResolver(_lending.DirectoryResolver):
    def __init__(self, exclude_server_ids=(), *, max_age_s: float = 5.0,
                 request_timeout_s: float = 0.5,
                 discover: Optional[Callable[[str], dict]] = None) -> None:
        super().__init__(exclude_server_ids, max_age_s=max_age_s,
                         request_timeout_s=request_timeout_s, discover=discover,
                         label=CLIENT_LABEL, who=_WHO, log=logger)


class ReservationKeeper(_lending.ReservationKeeper):
    def __init__(self, resolver: Callable[[str], list], holder: Holder, *,
                 ttl_s: float = DEFAULT_TTL_S, request_timeout_s: float = 3.0) -> None:
        super().__init__(resolver, holder, ttl_s=ttl_s, request_timeout_s=request_timeout_s,
                         label=CLIENT_LABEL, who=_WHO, log=logger)


__all__ = ["ReservationKeeper", "DirectoryResolver", "ServerRef", "Claim", "ClaimError",
           "ClaimRefused", "v2_request", "DEFAULT_TTL_S"]
