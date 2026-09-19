"""liveOD: the camera-owning acquisition window, the live OD viewer, and the client
the experiment process talks to. Lab-independent; a lab supplies a LiveODConfig
(config.py) from its launcher. Moved here from kexp.util.live_od on 2026-09-18;
the old import paths remain there as shims.

Keep this file free of module-level imports. The experiment process imports
``waxx.util.live_od.live_od_client`` and must not pay for cameras, Qt or the
analysis stack, so the names below are lazy (PEP 562).
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .camera_mother import CameraMother, CameraBaby, CameraNanny, DataHandler

_lazy_acquisition = ('CameraMother', 'CameraBaby', 'CameraNanny', 'DataHandler')


def __getattr__(name):
    if name in _lazy_acquisition:
        from . import camera_mother
        val = getattr(camera_mother, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'waxx.util.live_od' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_lazy_acquisition))
