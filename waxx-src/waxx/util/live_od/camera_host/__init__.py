"""liveOD's camera host: liveOD owns its cameras (one worker thread each) and
serves them to every other program (PLAN C3, k-jam/jpagett/camera_host).

Used only with ``LiveODConfig.use_camera_host``; with it off, liveOD opens its
cameras through CameraNanny as before.

    host.py          CameraHost (runs, live, persist, snapshots), HostServerCore
    sinks.py         RunSink: one run's frames, off the worker's taps
    legacy.py        HostNanny / HostCameraHandle: the camera thread's nanny and
                     camera in host mode (camera_mother unchanged)
    claims.py        ReservationKeeper: borrow a Basler from the beacon server
    bar.py           HostCameraBar / HostCameraButton: the old camera bar's surface
    qt_bridge.py     HostQtBridge: snapshots as a Qt signal
    local_stream.py  LocalHostStream: a camera as an in-process viewer source

The Qt and viewer names are imported lazily (PEP 562).
"""
from waxx.util.live_od.camera_host.host import (CameraHost, HostRefused, HostServerCore,
                                                PersistState, RunStart, RunSummary, legacy_state)
from waxx.util.live_od.camera_host.legacy import HostCameraHandle, HostNanny
from waxx.util.live_od.camera_host.sinks import RunSink

_lazy = {
    "HostQtBridge": "waxx.util.live_od.camera_host.qt_bridge",
    "LocalHostStream": "waxx.util.live_od.camera_host.local_stream",
    "HostCameraBar": "waxx.util.live_od.camera_host.bar",
    "HostCameraButton": "waxx.util.live_od.camera_host.bar",
    "ReservationKeeper": "waxx.util.live_od.camera_host.claims",
}


def __getattr__(name):
    if name in _lazy:
        import importlib
        val = getattr(importlib.import_module(_lazy[name]), name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'waxx.util.live_od.camera_host' has no attribute {name!r}")


__all__ = ["CameraHost", "HostRefused", "HostServerCore", "PersistState", "RunStart",
           "RunSummary", "legacy_state", "HostCameraHandle", "HostNanny", "RunSink", *_lazy]
