"""LocalHostStream: one of the camera host's cameras as a Camera Viewer source,
in-process (no ZMQ), for liveOD's own live window (PLAN T13).

It is a ``beacon.camera.viewer.sources.CameraSource`` (protocol "local"): the
viewer widget calls its methods from its worker threads.  Frames come
straight from the camera's latest-frame slot (``CameraHost.wait_frame``),
run frames included and tagged ``source="run"``.  Frames are handed out at
most every ``min_interval_s`` (10 Hz), and every ``run_interval_s`` (2 Hz)
while a run holds the camera: the GUI thread is the contended one, and the
run path must not wait on it (PLAN C3).  Settings go through
``CameraHost.set_live`` -- the full live profile, the live EM-gain cap -- so
this window can change what a remote viewer cannot.
"""
from __future__ import annotations

import time
from typing import Optional

from beacon.camera.core import error_reply, wire_readback
from beacon.camera.viewer.sources import CameraSource, ViewerFrame

#: frame pacing: ~10 Hz while live, ~2 Hz during a run
MIN_INTERVAL_S = 0.1
RUN_INTERVAL_S = 0.5


class LocalHostStream(CameraSource):
    protocol = "local"
    capabilities = frozenset({"settings"})

    def __init__(self, host, camera_key: str, *, min_interval_s: float = MIN_INTERVAL_S,
                 run_interval_s: float = RUN_INTERVAL_S) -> None:
        spec = host.spec(camera_key)
        cat = spec.category.name if spec.category else ""
        super().__init__(spec.camera_id or f"{cat}:{camera_key}", category=cat,
                         serial=spec.serial or camera_key, name=camera_key, model="",
                         host=host.hostname, server_id=host.server_id)
        self._host = host
        self._key = camera_key
        self._seq: Optional[int] = None
        self._last_t = 0.0
        self.min_interval_s = float(min_interval_s)
        self.run_interval_s = float(run_interval_s)
        self._attached = False
        self._attach_key = f"local:{id(self):x}"

    @property
    def camera_key(self) -> str:
        return self._key

    # -- lifecycle ----------------------------------------------------------------

    def open(self) -> dict:
        """Counts as a subscriber (``n_subs``); never starts acquisition --
        the live window's own button does that (``CameraHost.start_stream``)."""
        core = self._host.core
        if core is None:
            return {"ok": False, "code": "not_open", "error": "liveOD's camera host is not running"}
        if not self._attached:
            try:
                core.attach(self.camera_id, self._attach_key, "inproc", label="liveOD live view")
            except Exception as exc:
                return error_reply(exc)
            self._attached = True
        return {"ok": True}

    def close(self) -> dict:
        core = self._host.core
        if self._attached and core is not None:
            try:
                core.detach(self.camera_id, self._attach_key)
            except Exception as exc:
                return error_reply(exc)
        self._attached = False
        return {"ok": True}

    # -- frames -------------------------------------------------------------------------

    def next_frame(self, timeout_s: float) -> Optional[ViewerFrame]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        f = self._host.wait_frame(self._key, after_seq=self._seq, timeout_s=timeout_s)
        if f is None:
            return None
        pace = self.run_interval_s if f.source == "run" else self.min_interval_s
        wait = self._last_t + pace - time.monotonic()
        if wait > 0:
            if time.monotonic() + wait > deadline:
                return None           # the frame stays in the slot for the next call
            time.sleep(wait)
            newer = self._host.wait_frame(self._key, after_seq=f.seq, timeout_s=0.0)
            if newer is not None:
                f = newer
        self._seq = f.seq
        self._last_t = time.monotonic()
        settings = f.settings
        return ViewerFrame(f.image, source=f.source, run_tag=f.run_tag, seq=f.seq,
                           max_pixel_value=settings.get("max_pixel_value"), t_host=f.t_host,
                           settings=settings)

    # -- settings --------------------------------------------------------------------------

    def describe(self) -> dict:
        try:
            d = self._host.describe(self._key)
        except Exception as exc:
            return error_reply(exc)
        if not d.get("ok"):
            return d
        return {"ok": True, "schema": d["schema"], "dynamic": d["dynamic"],
                "settings": d["settings"], "settings_rev": d["settings_rev"]}

    def set_settings(self, values: dict, confirmed=frozenset()) -> dict:
        """``confirmed``: keys the operator confirmed past a "confirm" rule --
        "gain" is the live EM-gain unlock (CameraHost.set_live)."""
        try:
            rb = self._host.set_live(self._key, values, confirmed=confirmed)
        except Exception as exc:
            return error_reply(exc)
        try:
            rev = self._host.worker(self._key).settings_rev
        except Exception:
            rev = None
        return {"ok": True, "readback": wire_readback(rb), "settings_rev": rev}

    def status(self) -> dict:
        try:
            c = self._host.snapshot()["cameras"][self._key]
        except Exception:
            return {}
        return {"state": c["host_state"], "list_state": c["list_state"],
                "run": {"active": bool(c["locked"]), "run_tag": c["run_tag"]},
                "persist": c["persist"], "n_subs": c["n_subs"]}


__all__ = ["LocalHostStream", "MIN_INTERVAL_S", "RUN_INTERVAL_S"]
