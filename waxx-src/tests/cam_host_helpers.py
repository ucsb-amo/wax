"""Stand-ins for the camera host tests (test_cam_host_*.py).

Nothing here touches a real camera or the network beyond 127.0.0.1: cameras
are beacon's FakeBackend (built on their worker thread, as the real backends
are), no core beacons (``quiet_network``), the per-user state directory is a
temporary one, and a host's ReservationKeeper only ever talks to the servers a
test hands it.
"""
import functools
import threading
import time
import types

from beacon.camera.fake_backend import FakeBackend
from beacon.camera.reservations import Holder
# the process's discovery listener (a thread that never ends) exists before any
# test counts its threads; it only listens
import beacon.discovery.client  # noqa: F401

BASLER = types.SimpleNamespace(key="cam_b", camera_type="basler", serial_no="s1")
ANDOR = types.SimpleNamespace(key="cam_a", camera_type="andor")
APD = types.SimpleNamespace(key="apd", camera_type="apd")

ANDOR_CHOICES = {"trigger_mode": ("int", "software", "ext")}
ANDOR_RANGES = {"exposure_time": (2e-5, 10.0), "gain": (0, 300)}
FULL = (0, 512, 0, 512, 1, 1)


def basler_params(**kw):
    d = dict(key="cam_b", camera_type="basler", serial_no="s1",
             exposure_time=1e-3, gain=3.0, trigger_source="Line1")
    d.update(kw)
    return d


def andor_params(**kw):
    d = dict(key="cam_a", camera_type="andor", exposure_time=1e-3, gain=300,
             trigger_mode="ext", frame_transfer=0, sensor_roi=FULL, hs_speed=0, preamp=2,
             vs_speed=1, vs_amp=3, baseline_clamp=1)
    d.update(kw)
    return d


class Fakes(dict):
    """backend_factory for CameraHost: a FakeBackend per camera key, made on
    the camera's worker thread at its first open (kept here by key)."""

    def __init__(self, **per_key):
        super().__init__()
        self.per_key = per_key
        self.made = threading.Event()

    def __call__(self, spec):
        kw = dict(self.per_key.get(spec.key, {}))
        if spec.category.name == "andor_emccd":
            kw.setdefault("choices", dict(ANDOR_CHOICES))
            kw.setdefault("ranges", dict(ANDOR_RANGES))
            kw.setdefault("settings", {"trigger_mode": "int", "gain": 1, "exposure_time": 1e-3})
        kw.setdefault("shape", (4, 6))
        fb = FakeBackend(spec.camera_id, category=spec.category.name, **kw)
        self[spec.key] = fb
        self.made.set()
        return fb


def quiet_network(monkeypatch, tmp_path):
    """No beacon is ever sent, the duplicate-id guard does not listen, and the
    per-user camera state lives in tmp_path."""
    from beacon.discovery import server as dserver
    from beacon.camera.core import CameraServerCore
    monkeypatch.setattr(dserver.NetServer, "_start_beacon", lambda self: None)
    monkeypatch.setattr(CameraServerCore, "DUPLICATE_WINDOW_S", 0.0)
    monkeypatch.setenv("BEACON_STATE_DIR", str(tmp_path / "beacon_state"))


def keeper_for(resolver=None, ttl_s=30.0, holder_id="liveod:test"):
    from waxx.util.live_od.camera_host.claims import ReservationKeeper
    return ReservationKeeper(resolver or (lambda cid: []),
                             Holder(holder_id, server_id="camera_server:test:liveod",
                                    host="testhost", pid=1, label="liveOD"),
                             ttl_s=ttl_s)


def make_host(config, fakes, keeper=None, **kw):
    from waxx.util.live_od.camera_host import CameraHost, HostServerCore
    core_factory = functools.partial(HostServerCore, check_duplicate=False, bind_host="127.0.0.1")
    kw.setdefault("serve", False)
    kw.setdefault("local_addresses", {"127.0.0.1"})
    kw.setdefault("server_id", "camera_server:test:liveod")
    return CameraHost(config, backend_factory=fakes, core_factory=core_factory,
                      keeper=keeper or keeper_for(), **kw)


def wait_for(pred, timeout=5.0, step=0.005, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(step)
    raise AssertionError(f"{what} not met within {timeout:.1f} s")


def stop_host(host, timeout_s=5.0) -> list:
    """Shut the host down and join its camera threads; what is still running."""
    left = []
    host.shutdown(timeout_s)
    core = host.core
    if core is not None:
        for cid in core.camera_ids():
            w = core.worker(cid)
            w.join(timeout_s)
            if w.is_alive():
                left.append(w.name)
    return left


def join_new_threads(before, timeout_s=5.0) -> list:
    """Join every threading.Thread started since ``before``; what did not end."""
    left = []
    for t in threading.enumerate():
        if t in before or t is threading.current_thread() or isinstance(t, threading._DummyThread):
            continue
        t.join(timeout_s)
        if t.is_alive():
            left.append(t.name)
    return left
