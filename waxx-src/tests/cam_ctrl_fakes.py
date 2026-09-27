"""A stand-in for liveOD's CameraHost, for the camera-control GUI tests
(test_cam_ctrl_*.py).  In-process and instant: no camera, no thread of its own,
no socket, no UDP.  It has exactly the host methods the GUI uses (HOST_METHODS)
plus what ``LocalHostStream`` needs, and records every call in ``calls``.
"""
import concurrent.futures
import dataclasses
import os
import threading
import time
import types
from typing import Optional

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

from beacon.camera.backend import ApplyRefused, Readback  # noqa: E402
from beacon.camera.schema import ANDOR_LIVE_EM_GAIN_CAP, get_category, to_wire  # noqa: E402
from beacon.camera.viewer.sources import CameraSource, ViewerFrame  # noqa: E402

#: the CameraHost methods the GUI calls (checked against the real class in a test)
HOST_METHODS = ("snapshot", "on_snapshot", "start_stream", "stop_stream", "set_live",
                "set_persist", "persist", "request", "describe")
#: and what LocalHostStream calls on it
STREAM_METHODS = ("spec", "wait_frame", "worker")

ANDOR_SETTINGS = {"exposure_time": 1e-3, "gain": 30, "trigger_mode": "int",
                  "sensor_roi": (0, 512, 0, 512, 1, 1), "hs_speed": 0, "preamp": 2,
                  "vs_speed": 1, "vs_amp": 3, "baseline_clamp": 1, "shutter": "open",
                  "temperature": -60.0, "cooler_status": "stabilized"}
BASLER_SETTINGS = {"exposure_time": 3e-4, "gain": 12.0, "trigger_source": "Line1",
                   "trigger_mode": "Off"}
ANDOR_DYNAMIC = {
    "trigger_mode": {"choices": ["int", "software", "ext"]},
    "hs_speed": {"choices": [0, 1, 2, 3], "labels": ["17 MHz", "10 MHz", "5 MHz", "1 MHz"]},
    "preamp": {"choices": [0, 1, 2], "labels": ["1x", "2x", "4.5x"]},
    "vs_speed": {"choices": [0, 1, 2, 3, 4],
                 "labels": ["0.3 us", "0.5 us", "0.9 us", "1.7 us", "3.3 us"]},
    "exposure_time": {"range": [1e-5, 10.0]},
}
BASLER_DYNAMIC = {"exposure_time": {"range": [1.9e-5, 1.0]}, "gain": {"range": [0.0, 36.0]}}


class HostRefused(RuntimeError):
    """What the real host raises when it will not do something now."""


@dataclasses.dataclass(frozen=True)
class FakePersistState:
    on: bool = False
    values: dict = dataclasses.field(default_factory=dict)
    since: Optional[float] = None
    whitelist: tuple = ()

    @property
    def since_iso(self):
        return None if self.since is None else time.strftime("%Y-%m-%dT%H:%M:%S",
                                                             time.localtime(self.since))


@dataclasses.dataclass(frozen=True, eq=False)
class FakeFrame:
    image: np.ndarray
    source: str = "live"
    run_tag: Optional[str] = None
    seq: int = 0
    settings: dict = dataclasses.field(default_factory=dict)
    t_host: float = 0.0
    acq_gen: int = 1


def make_stream(host, key):
    """A ``stream_factory`` for LiveViewWindow."""
    return FakeStream(host, key)


class FakeStream(CameraSource):
    """A viewer source on the FakeHost, as LocalHostStream is on the real host:
    open/close count a subscriber, frames come from ``host.wait_frame``."""
    protocol = "local"
    capabilities = frozenset({"settings"})

    def __init__(self, host, key):
        cam = host.cams[key]
        super().__init__(cam["camera_id"], category=cam["category"], serial=cam["serial"],
                         name=key, host=host.hostname, server_id=host.server_id)
        self._host, self._key, self._seq = host, key, None
        self.opened = self.closed = 0

    def open(self):
        self.opened += 1
        self._host.core.attach(self.camera_id, f"fake:{id(self):x}", "inproc")
        return {"ok": True}

    def close(self):
        self.closed += 1
        self._host.core.detach(self.camera_id, f"fake:{id(self):x}")
        return {"ok": True}

    def next_frame(self, timeout_s):
        f = self._host.wait_frame(self._key, after_seq=self._seq, timeout_s=timeout_s)
        if f is None:
            return None
        self._seq = f.seq
        return ViewerFrame(f.image, source=f.source, run_tag=f.run_tag, seq=f.seq,
                           t_host=f.t_host, settings=dict(f.settings))

    def status(self):
        c = self._host.cams[self._key]
        return {"state": c["host_state"], "list_state": c["list_state"],
                "run": {"active": bool(c["locked"]), "run_tag": c["run_tag"]}}

    def describe(self):
        d = self._host.describe(self._key)
        return {"ok": True, "schema": d["schema"], "dynamic": d["dynamic"],
                "settings": d["settings"], "settings_rev": d["settings_rev"]}

    def set_settings(self, values, confirmed=frozenset()):
        try:
            rb = self._host.set_live(self._key, dict(values))
        except Exception as exc:
            return {"ok": False, "code": "refused", "error": str(exc)}
        return {"ok": True, "readback": {k: {"value": v.value, "source": v.source}
                                         for k, v in rb.items()}}


def done_future(result=None, exc=None):
    fut = concurrent.futures.Future()
    if exc is not None:
        fut.set_exception(exc)
    else:
        fut.set_result(result)
    return fut


class _Core:
    """``host.core``: attach / detach count subscribers."""

    def __init__(self, host):
        self.host = host

    def attach(self, camera_id, key, kind, label=""):
        self.host.calls.append(("attach", camera_id, key))
        cam = self.host.key_of(camera_id)
        self.host.cams[cam]["n_subs"] += 1
        return done_future(True)

    def detach(self, camera_id, key):
        self.host.calls.append(("detach", camera_id, key))
        cam = self.host.key_of(camera_id)
        self.host.cams[cam]["n_subs"] = max(0, self.host.cams[cam]["n_subs"] - 1)
        return done_future(True)


class FakeHost:
    hostname = "testhost"
    server_id = "camera_server:testhost:liveod"

    def __init__(self, keys=("andor", "xy_basler")):
        self.calls = []
        self.cams = {}
        self.frames = {}
        self._seq = {}
        self._cv = threading.Condition()
        self._callbacks = []
        self.refuse_next = None         # an exception set_live raises once
        self.core = _Core(self)
        for key in keys:
            andor = "andor" in key
            cat = "andor_emccd" if andor else "basler_usb"
            self.cams[key] = {
                "key": key, "camera_id": f"{cat}:{key}-serial",
                "camera_type": "andor" if andor else "basler", "category": cat,
                "serial": f"{key}-serial", "state": "open", "host_state": "idle",
                "persist": False, "persisted": {}, "persist_since": None,
                "holder": {"label": "liveOD", "host": self.hostname}, "claims": [],
                "n_subs": 0, "error": None,
                "settings": dict(ANDOR_SETTINGS if andor else BASLER_SETTINGS),
                "settings_rev": 1, "locked": False, "run_id": None, "run_tag": None,
                "list_state": "open",
            }
            self.frames[key] = []
            self._seq[key] = 0

    # -- helpers for tests ----------------------------------------------------------

    def key_of(self, camera_id):
        return next(k for k, c in self.cams.items() if c["camera_id"] == camera_id)

    def category(self, key):
        return get_category(self.cams[key]["category"])

    def set_state(self, key, host_state, **kw):
        from waxx.util.live_od.gui.camera_control import LEGACY_STATE
        self.cams[key]["host_state"] = host_state
        self.cams[key]["state"] = LEGACY_STATE.get(host_state, "failed")
        self.cams[key]["locked"] = host_state in ("run_locked", "arming", "acquiring", "draining")
        self.cams[key].update(kw)

    def push_frame(self, key, image=None, source="live", run_tag=None):
        if image is None:
            image = np.full((8, 10), 500 + self._seq[key], np.uint16)
        with self._cv:
            self._seq[key] += 1
            frame = FakeFrame(image, source=source, run_tag=run_tag, seq=self._seq[key],
                              settings=dict(self.cams[key]["settings"]), t_host=time.time())
            self.frames[key].append(frame)
            self._cv.notify_all()
        return frame

    def emit(self):
        snap = self.snapshot()
        for cb in list(self._callbacks):
            cb(snap)

    def calls_of(self, name):
        return [c for c in self.calls if c[0] == name]

    # -- the host's API (HOST_METHODS) ---------------------------------------------

    def snapshot(self):
        return {"server_id": self.server_id, "instance": "fake", "serving": False,
                "started": True, "rev": 0, "t": time.time(),
                "cameras": {k: {**c, "settings": dict(c["settings"]),
                                "persisted": dict(c["persisted"])} for k, c in self.cams.items()}}

    def on_snapshot(self, cb):
        self._callbacks.append(cb)
        return lambda: self._callbacks.remove(cb)

    def start_stream(self, key):
        self.calls.append(("start_stream", key))
        self.set_state(key, "streaming")
        return done_future(True)

    def stop_stream(self, key):
        self.calls.append(("stop_stream", key))
        self.set_state(key, "idle")
        return done_future(True)

    def set_live(self, key, values, timeout_s=15.0):
        self.calls.append(("set_live", key, dict(values)))
        if self.refuse_next is not None:
            exc, self.refuse_next = self.refuse_next, None
            raise exc
        cam = self.cams[key]
        if cam["locked"]:
            raise HostRefused(f"{key}: run {cam['run_id']} holds the camera")
        values = dict(values)
        unlocked = bool(values.pop("em_gain_unlocked", False))
        if (cam["category"] == "andor_emccd" and "gain" in values
                and values["gain"] > ANDOR_LIVE_EM_GAIN_CAP and not unlocked):
            raise ApplyRefused("gain", f"gain={values['gain']!r} needs confirmation (live): "
                                       f"live EM gain above {ANDOR_LIVE_EM_GAIN_CAP} needs 'unlock'")
        cam["settings"].update(values)
        cam["settings_rev"] += 1
        return {k: Readback(v, "hw") for k, v in values.items()}

    def persist(self, key):
        cam = self.cams[key]
        return FakePersistState(cam["persist"], dict(cam["persisted"]), None,
                                self.category(key).persistable_keys())

    def set_persist(self, key, on, values=None):
        self.calls.append(("set_persist", key, bool(on)))
        cam = self.cams[key]
        if cam["locked"]:
            raise HostRefused(f"persist for {key} cannot change while run {cam['run_id']} "
                              f"holds the camera")
        whitelist = self.category(key).persistable_keys()
        if not on:
            cam.update(persist=False, persisted={}, persist_since=None)
            return FakePersistState(False, {}, None, whitelist)
        persisted = {k: cam["settings"][k] for k in whitelist if k in cam["settings"]}
        state = FakePersistState(True, persisted, time.time(), whitelist)
        cam.update(persist=True, persisted=dict(persisted), persist_since=state.since_iso)
        return state

    def request(self, key, action, origin=""):
        self.calls.append(("request", key, action, origin))
        return done_future({"ok": True})

    def describe(self, key, include_hidden=False):
        cam = self.cams[key]
        cat = self.category(key)
        andor = cat.name == "andor_emccd"
        profile = {k: v for k, v in cam["settings"].items()
                   if cat.has(k) and cat.setting(k).group != "status"}
        return {"ok": True, "key": key, "camera_id": cam["camera_id"], "category": cat.name,
                "schema": to_wire(cat, include_hidden=include_hidden),
                "dynamic": dict(ANDOR_DYNAMIC if andor else BASLER_DYNAMIC),
                "settings": dict(cam["settings"]), "settings_rev": cam["settings_rev"],
                "live_profile": profile, "live_defaults": {}, "constraints": [],
                "persist": cam["persist"], "persisted": dict(cam["persisted"]),
                "persist_since": cam["persist_since"],
                "persistable": list(cat.persistable_keys())}

    # -- what LocalHostStream uses (STREAM_METHODS, core, hostname, server_id) ---------

    def spec(self, key):
        cam = self.cams[key]
        return types.SimpleNamespace(key=key, camera_id=cam["camera_id"], serial=cam["serial"],
                                     category=self.category(key),
                                     camera_type=cam["camera_type"])

    def worker(self, key):
        return types.SimpleNamespace(settings_rev=self.cams[key]["settings_rev"])

    def wait_frame(self, key, after_seq=None, timeout_s=1.0, sources=("live", "snap", "run")):
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._cv:
            while True:
                for f in reversed(self.frames[key]):
                    if f.source in sources and (after_seq is None or f.seq > after_seq):
                        return f
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(min(left, 0.05))
