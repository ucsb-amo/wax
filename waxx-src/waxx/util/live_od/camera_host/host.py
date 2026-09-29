"""CameraHost: liveOD owns its cameras; everything else subscribes (PLAN C3).

One ``CameraServerCore`` (``camera_server:<host>:liveod``, policy
"persistent", protocol v2 only) with one ``CameraWorker`` per camera liveOD
lists (``LiveODConfig.camera_params_list``): the Andor through waxx's
``EMCCDBackend``, each Basler through beacon's ``BaslerBackend``.  An entry
with no camera behind it (the APD) has no worker.  The worker is the only
thread that ever calls its camera; everything here asks it (``submit``).

Nothing is built or touched before ``start()``.  ``start()`` makes the core
and the workers, serves the core on the network (``serve=True``), and opens
the cameras in ``LiveODConfig.camera_host_claim_on_start``.

**Runs** (in-process only, never on the wire; the liveOD server calls them):

    begin_run(token, key, capture_images, camera_params)  INIT_RUN, before the
        run file is reserved: validate the run profile, borrow the camera from
        the server that has it (a Basler from the beacon server, the Andor from
        the SLM spot finder; bounded), lock the camera with the token
    note_run_id(token, run_id)                            once the id is known
    arm_run(token, camera_params, n_img) -> Future[ArmResult]
        the first WAIT_CAM_READY: apply the full run profile, start the
        triggered acquisition, check that nothing free-runs
    attach_run(token) -> HostCameraHandle                 the camera thread's "camera"
    end_run(token, reason) -> RunSummary                  END_RUN / ABORT_RUN /
        a superseding INIT_RUN: disarm and unlock; the camera stays idle at the
        run's settings (Q5) -- no stream resumes
    operator_release(key)                                 the operator ends a hold

A RESET keeps the lock (T10): the kernel may still be triggering.

**Live** (the liveOD GUI): ``start_stream``, ``stop_stream``, ``set_live``
(a full live profile each time; live EM gain capped unless
``em_gain_unlocked``), ``describe``, ``wait_frame``, ``request`` (open /
close / toggle), ``persist`` / ``set_persist``, ``snapshot`` /
``on_snapshot`` / ``poll_cameras``, ``shutdown``.

**Live requests**: the stream runs while anyone asks for it.  liveOD asks
through ``start_stream(key, requester="liveod")`` (the core's
``request_live``), a program on this PC through START_LIVE; each gives its
own request back (``stop_stream`` / STOP_LIVE, or detaching), and the stream
stops when the last one has.  So liveOD's live view stopping never ends a
stream the spot finder asked for, and the other way round.  A close, a run's
lock or a fault clears every request (after a run the camera stays idle).

**Persist** (INTERFACES D-a): per camera, off at start.  Turned on, it
captures the current live values of the whitelisted fields (Andor gain,
hs_speed, vs_speed, vs_amp, preamp, baseline_clamp; Basler gain -- never
exposure_time or a run-owned field) and puts them on top of every run's
camera_params until it is turned off.  Each run it changes is logged at
INIT_RUN and recorded as ``camera_overrides`` (origin "persist").  While it is
on, a remote write of a persisted field is refused.

**Remote clients** (T3, widened 2026-09-28): a program on any PC (the Camera
Viewer) may change live settings and ask for the live stream (START_LIVE, of
a camera liveOD has open: it never opens one) -- never while a run holds the
camera: the worker's run lock refuses both from INIT_RUN until the run ends.
Only programs on this PC may snap, rename or relinquish.  Owner-only fields
and persisted ones are refused to everyone, and so is ``em_gain_unlocked``:
the live EM-gain cap is lifted only in liveOD's own settings.  Attaching
starts nothing (policy "persistent": it never touches the device).
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from beacon.camera.backend import ApplyRefused, clamps as readback_clamps, plain_readback
from beacon.camera.core import CameraServerCore, list_state
from beacon.camera.reservations import Holder
from beacon.camera.schema import (ANDOR_EMCCD, ANDOR_LIVE_EM_GAIN_CAP, BASLER_USB, Category,
                                  RunProfileError, build_live_profile, build_run_profile,
                                  check_constraints, normalize_value, refusals, to_wire)
from beacon.camera.worker import LockedError

from waxx.util.live_od.camera_host.sinks import RunSink
from waxx.util.live_od.log import VERBOSE

logger = logging.getLogger("waxx.live_od.camera_host")

SERVER_ID_PREFIX = "camera_server:"
SERVER_ID_SUFFIX = ":liveod"
#: keys a live apply takes that are not settings
CONTROL_KEYS = ("em_gain_unlocked",)
#: begin_run: the camera must take the run lock within this
LOCK_TIMEOUT_S = 3.0
#: begin_run / open: a camera borrowed from another server within this (per server)
CLAIM_TIMEOUT_S = 3.0
#: ... the Andor within this: its verified close (stop, shutter closed, SDK
#: ShutDown) is slower than a Basler's, and the SLM spot finder's server waits
#: up to 15 s for it before answering (served_source.RELINQUISH_CLOSE_S)
ANDOR_CLAIM_TIMEOUT_S = 20.0
#: end_run: disarm and unlock within this, each
END_RUN_OP_S = 5.0
#: start_stream / stop_stream: liveOD's own live request (the core's request_live key)
LIVEOD_REQUESTER = "liveod"
#: the write_policy key of START_LIVE / STOP_LIVE (beacon.camera.core)
LIVE_KEY = "__live__"

#: host_state -> the legacy word POLL / CAMERA_STATE have always used (T14)
LEGACY_STATE = {
    "absent": "closed", "closed": "closed", "reserved": "closed", "held_elsewhere": "closed",
    "virtual": "closed",
    "claiming": "loading", "opening": "loading", "returning": "loading",
    "idle": "open", "streaming": "open",
    "run_locked": "grabbing", "arming": "grabbing", "acquiring": "grabbing", "draining": "grabbing",
    "error": "failed", "faulted": "failed", "run_fault": "failed", "hung": "failed",
}
RUN_PHASES = ("run_locked", "arming", "acquiring", "draining")


def legacy_state(host_state: str) -> str:
    return LEGACY_STATE.get(host_state, "failed")


def _s(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return "" if value is None else str(value)


def iso_time(t: Optional[float]) -> Optional[str]:
    return None if t is None else time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


def _ascii(text: str) -> str:
    text = str(text).replace(chr(0xB5), "u").replace(chr(0x3BC), "u")    # micro signs
    return text.encode("ascii", "replace").decode("ascii")


def _jsonable(value):
    """Tuples (a sensor_roi) as lists, all the way down; everything else as is."""
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def run_request_of(cat: Category, plan) -> dict:
    """What a run asked the camera for: its camera_params value of every
    run="param" setting of ``cat`` -- before Persist and clamps -- as plain
    JSON values (the Run tab's "Request" column)."""
    out = {}
    for key in cat.keys(run="param"):
        f = plan.overrides.get(key)
        if f is not None:
            out[key] = _jsonable(f["requested"])
        elif key in plan.profile:
            out[key] = _jsonable(plan.profile[key])
    return out


class HostRefused(RuntimeError):
    """The camera host will not do this now; the message says which rule and value."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraSpec:
    """One entry of liveOD's camera list, as the host sees it."""
    key: str
    camera_type: str
    category: Optional[Category]
    camera_id: Optional[str]
    serial: str
    params: Any = field(compare=False, repr=False, default=None)
    live_defaults: dict = field(compare=False, default_factory=dict)

    @property
    def has_camera(self) -> bool:
        return self.category is not None


@dataclass(frozen=True)
class PersistState:
    on: bool = False
    values: dict = field(default_factory=dict)
    since: Optional[float] = None           # time.time() when turned on
    whitelist: tuple = ()

    @property
    def since_iso(self) -> Optional[str]:
        return iso_time(self.since)


@dataclass(frozen=True)
class RunStart:
    """What ``begin_run`` found: the run profile, and what Persist changes in it."""
    camera_key: str
    locked: bool
    category: str = ""
    profile: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)      # {field: {requested, applied, origin}}
    refused: dict = field(default_factory=dict)        # persisted values NOT carried
    persist_on: bool = False
    persist_since: Optional[str] = None
    labels: dict = field(default_factory=dict)         # {field: {value: label}}

    def persist_warning(self, run_id: int) -> str:
        """The INIT_RUN WARNING (PLAN C6) for the fields Persist changed; "" if none."""
        if not self.overrides:
            return ""
        run = f"run {run_id}" if run_id else "this run (save_data=False)"
        width = max(len(k) for k in self.overrides)
        lines = [f"PERSISTED CAMERA SETTINGS: {run} on {self.camera_key} does NOT use the "
                 f"experiment's camera_params for {len(self.overrides)} field(s):"]
        for key, f in self.overrides.items():
            lines.append(f"    {key:<{width}}  {self._label(key, f['requested'])} -> "
                         f"{self._label(key, f['applied'])} (persisted)")
        lines.append(f"  Persist was turned on at {self.persist_since} in liveOD's "
                     f"{self.camera_key} settings. Turn it off there to go back to camera_params.")
        return _ascii("\n".join(lines))

    def _label(self, key, value) -> str:
        lab = (self.labels.get(key) or {}).get(value)
        return f"{value!r} ({lab})" if lab else repr(value)


@dataclass(frozen=True)
class RunSummary:
    """How a run's hold on its camera ended (``end_run``)."""
    token: str
    reason: str
    camera_key: str = ""
    camera_id: str = ""
    run_id: int = 0
    locked: bool = False
    armed: bool = False
    arm_error: str = ""
    n_frames: int = 0
    delivered: int = 0
    lost_idx: tuple = ()
    stopped_early: bool = False
    surplus: int = 0
    stale: int = 0
    foreign: int = 0
    overflow: int = 0
    run_fault: str = ""
    overrides: dict = field(default_factory=dict)
    persist_since: Optional[str] = None
    errors: tuple = ()

    @property
    def problems(self) -> list:
        """What makes the run's frames suspect, beyond a frame count the server
        already checks: frames the camera reported lost, a camera fault."""
        out = []
        if self.lost_idx:
            out.append(f"frame(s) {list(self.lost_idx)} of {self.n_frames} reported lost by "
                       f"the camera")
        if self.run_fault:
            out.append(f"camera error during the run: {self.run_fault}")
        if self.overflow:
            out.append(f"{self.overflow} frame(s) beyond the run's buffer were not kept")
        return out


@dataclass
class _Run:
    token: str
    key: str
    capture_images: bool
    locked: bool = False
    why_not: str = ""
    camera_id: str = ""
    category: Optional[Category] = None
    plan: Any = None
    persisted: dict = field(default_factory=dict)
    persist_since: Optional[str] = None
    run_id: int = 0
    run_tag: str = ""
    n_img: int = 0
    images_shape: Any = None
    arm_future: Optional[concurrent.futures.Future] = None
    arm_result: Any = None
    arm_error: str = ""
    clamps: dict = field(default_factory=dict)
    sink: Optional[RunSink] = None
    handle: Any = None


# ---------------------------------------------------------------------------
# The core, with the host's hooks
# ---------------------------------------------------------------------------

class HostServerCore(CameraServerCore):
    """``CameraServerCore`` plus the hooks the host needs:

    * ``event_hook(camera_id, kind)`` after every worker event (state, frame);
    * ``before_snap(camera_id)`` before a remote SNAP is queued (the host puts
      the live profile back first when the run's settings are still applied);
    * ``before_live(camera_id)`` before a remote START_LIVE is handled (the
      host puts the live profile back first on a camera that is not
      streaming, as ``start_stream`` does, so a stream never runs on a run's
      trigger or EM gain);
    * RELINQUISH_CAMERA is put to ``write_policy`` like every other write, so a
      program on another PC cannot take a camera away from liveOD.

    The before-hooks run only for a write the policy allows.
    """

    def __init__(self, server_id: str, **kw) -> None:
        super().__init__(server_id, **kw)
        self.event_hook: Optional[Callable[[str, str], None]] = None
        self.before_snap: Optional[Callable[[str], None]] = None
        self.before_live: Optional[Callable[[str], None]] = None

    def _on_worker_event(self, camera_id: str, kind: str) -> None:
        super()._on_worker_event(camera_id, kind)
        hook = self.event_hook
        if hook is not None:
            try:
                hook(camera_id, kind)
            except Exception:
                logger.exception("camera host: event hook failed")

    def _run_before(self, req, hook, key: str, what: str) -> None:
        cam = self._lookup(req.header.get("camera_id"))
        if cam is not None and hook is not None:
            ok, _ = self.check_write(req.client_addr, cam.camera_id, [key])
            if ok:
                try:
                    hook(cam.camera_id)
                except Exception:
                    logger.exception(f"camera host: {what} failed")

    def _h_snap(self, req):
        self._run_before(req, self.before_snap, "snap", "before_snap")
        return super()._h_snap(req)

    def _h_start_live(self, req):
        self._run_before(req, self.before_live, "__live__", "before_live")
        return super()._h_start_live(req)

    def _h_relinquish(self, req):
        ids = req.header.get("camera_ids")
        ids = [ids] if isinstance(ids, str) else (ids if isinstance(ids, list) else [])
        for q in ids:
            cam = self._lookup(q)
            if cam is None:
                continue
            ok, reason = self.check_write(req.client_addr, cam.camera_id, ["relinquish"])
            if not ok:
                req.camera_id = cam.camera_id
                return self._refused(cam.camera_id, ["relinquish"], reason)
        return super()._h_relinquish(req)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def refuse_combos_from(constraints) -> tuple:
    """``EMCCDBackend(refuse_combos=...)`` from schema Constraints whose every
    condition is an equality (e.g. the DU897's vs_speed == 0 and vs_amp == 0)."""
    out = []
    for c in constraints or ():
        if getattr(c, "level", "") != "refuse":
            continue
        conds = tuple(getattr(c, "when", ()) or ())
        if conds and all(len(w) == 3 and w[1] == "==" for w in conds):
            out.append((tuple((k, v) for k, _, v in conds), c.reason))
    return tuple(out)


def default_backend_factory(spec: CameraSpec, constraints=(), live_em_gain_cap=ANDOR_LIVE_EM_GAIN_CAP):
    """The backend for ``spec``, built on its worker thread at first open."""
    if spec.camera_type == "andor":
        from waxx.control.cameras.emccd_backend import EMCCDBackend
        return EMCCDBackend(serial=None, refuse_combos=refuse_combos_from(constraints),
                            live_em_gain_cap=live_em_gain_cap)
    if spec.camera_type == "basler":
        from beacon.camera.basler_backend import BaslerBackend
        return BaslerBackend(spec.serial)
    raise ValueError(f"{spec.key}: camera_type {spec.camera_type!r} has no camera backend")


def camera_specs(params_list) -> list:
    """A CameraSpec per entry of liveOD's camera list, in its order."""
    out = []
    for p in params_list or ():
        key = _s(getattr(p, "key", ""))
        ctype = _s(getattr(p, "camera_type", ""))
        if ctype == "basler":
            serial = _s(getattr(p, "serial_no", ""))
            cat, cid = BASLER_USB, f"{BASLER_USB.name}:{serial}"
        elif ctype == "andor":
            serial = _s(getattr(p, "serial_no", "") or getattr(p, "serial", "") or "")
            cat, cid = ANDOR_EMCCD, f"{ANDOR_EMCCD.name}:{serial or key}"
        else:
            serial, cat, cid = "", None, None      # the APD and other virtual entries
        live = build_live_profile(cat)[0] if cat is not None else {}
        out.append(CameraSpec(key, ctype, cat, cid, serial, p, live))
    return out


def _local_addresses() -> set:
    addrs = {"127.0.0.1", "::1", "localhost"}
    try:
        addrs.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return addrs


# ---------------------------------------------------------------------------
# The host
# ---------------------------------------------------------------------------

class CameraHost:
    """See the module docstring.

    ``backend_factory(spec) -> CameraBackend`` replaces the real backends
    (tests: FakeBackend); it is called on the camera's worker thread at its
    first open.  ``core_factory(server_id, **kw) -> CameraServerCore``
    replaces ``HostServerCore`` (tests: ``check_duplicate=False``,
    ``bind_host="127.0.0.1"``).  ``keeper``: a ``ReservationKeeper`` (tests
    give one with their own resolver); by default one is made at start that
    finds the beacon servers through this process's discovery cache.
    ``serve=False`` never binds or beacons (tests).  ``local_addresses``: the
    client addresses that count as this PC (default: loopback + this host's).
    """

    def __init__(self, config, *, backend_factory: Optional[Callable] = None,
                 core_factory: Optional[Callable] = None, keeper=None, serve: bool = True,
                 local_addresses=None, heartbeat_s: float = 1.0,
                 server_id: Optional[str] = None,
                 live_em_gain_cap: int = ANDOR_LIVE_EM_GAIN_CAP) -> None:
        self.config = config
        self.hostname = socket.gethostname()
        self.server_id = server_id or f"{SERVER_ID_PREFIX}{self.hostname}{SERVER_ID_SUFFIX}"
        self._constraints = {str(k): tuple(v or ()) for k, v in
                             dict(getattr(config, "camera_constraints", None) or {}).items()}
        self._live_cap = int(live_em_gain_cap)
        self._backend_factory = backend_factory
        self._core_factory = core_factory or HostServerCore
        self._keeper = keeper
        self._serve = bool(serve)
        self._local = set(local_addresses) if local_addresses is not None else None
        self._heartbeat_s = float(heartbeat_s)
        self._claim_on_start = tuple(_s(k) for k in
                                     (getattr(config, "camera_host_claim_on_start", ()) or ()))
        self.specs = camera_specs(getattr(config, "camera_params_list", ()))
        self._by_key = {s.key: s for s in self.specs}
        self._by_cid = {s.camera_id: s for s in self.specs if s.camera_id}

        self._core: Optional[CameraServerCore] = None
        self._started = False
        self._start_error: str = ""
        self._shut = False
        self._lock = threading.RLock()        # never held while waiting on a worker
        self._runs: dict = {}                 # token -> _Run
        self._run_by_key: dict = {}           # camera key -> token
        self._persist: dict = {s.key: PersistState(whitelist=s.category.persistable_keys())
                               for s in self.specs if s.has_camera}
        self._run_settings_rev: dict = {}     # camera key -> settings_rev of the last run apply
        self._run_request: dict = {}          # camera key -> the last run's request (run_request_of)
        self._busy: dict = {}                 # camera key -> "claiming" | "returning"
        self._held_elsewhere: dict = {}       # camera key -> why the last claim was refused
        self._frame_cv = {s.key: threading.Condition() for s in self.specs if s.has_camera}
        self._callbacks: list = []
        self._rev = 0
        self._rev_lock = threading.Lock()
        self._notify_evt = threading.Event()
        self._notify_stop = threading.Event()
        self._notifier: Optional[threading.Thread] = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    @property
    def started(self) -> bool:
        return self._started

    @property
    def core(self) -> Optional[CameraServerCore]:
        return self._core

    def start(self) -> None:
        """Build the core and a worker per camera, serve, and open the cameras
        in ``camera_host_claim_on_start``.  The first hardware touch."""
        if self._started or self._shut:
            return
        kw = dict(policy="persistent", allow_v1=False, write_policy=self._write_policy)
        try:
            core = self._core_factory(self.server_id, **kw)
        except Exception as exc:
            self._start_error = f"{type(exc).__name__}: {exc}"
            logger.error(f"camera host: could not start ({self._start_error}); runs that use a "
                         f"camera will be refused until liveOD is restarted")
            raise
        if hasattr(core, "event_hook"):
            core.event_hook = self._on_worker_event
            core.before_snap = self._before_network_snap
            core.before_live = self._before_network_live
        for spec in self.specs:
            if not spec.has_camera:
                continue
            core.add_camera(spec.camera_id, self._factory_for(spec), category=spec.category.name,
                            live_defaults=dict(spec.live_defaults), owner_key=spec.key,
                            info={"serial": spec.serial, "model": "", "user_id": spec.key})
        self._core = core
        if self._local is None:
            self._local = _local_addresses()
        if self._keeper is None:
            from waxx.util.live_od.camera_host.claims import DirectoryResolver, ReservationKeeper
            holder = Holder(holder_id=f"liveod:{self.hostname}:{os.getpid()}",
                            server_id=self.server_id, host=self.hostname, pid=os.getpid(),
                            label="liveOD")
            self._keeper = ReservationKeeper(DirectoryResolver({self.server_id}), holder)
        self._executor = concurrent.futures.ThreadPoolExecutor(2, thread_name_prefix="CameraHost")
        self._notifier = threading.Thread(target=self._notify_loop, daemon=True,
                                          name="CameraHost-snapshots")
        self._started = True
        self._notifier.start()
        for cat, cons in self._constraints.items():
            for c in cons:
                conds = " and ".join(f"{k} {op} {v!r}" for k, op, v in c.when)
                logger.debug(f"camera host: lab constraint on {cat} ({c.applies_to}): {conds} -> "
                             f"{c.level}: {c.reason}")
        if self._serve:
            core.start_in_thread()
            logger.info(f"camera host: serving {self.server_id} on port {core.port} "
                        f"(protocol v2; other PCs: live settings and the live stream, "
                        f"never during a run)")
        for key in self._claim_on_start:
            spec = self._by_key.get(key)
            if spec is None or not spec.has_camera:
                logger.warning(f"camera host: camera_host_claim_on_start names {key!r}, which "
                               f"has no camera here; ignored")
                continue
            self.request(key, "open", origin="claim at start")
        self._bump()

    def shutdown(self, timeout_s: float = 3.0) -> dict:
        """Close every camera (Andor: acquisition stopped, shutter closed, SDK
        closed) and give borrowed cameras back.  Idempotent; bounded:
        ``timeout_s`` for the cameras, then 1 s per borrowed camera."""
        with self._lock:
            if self._shut:
                return {"already": True}
            self._shut = True
            runs = list(self._runs.values())
        report = {"cameras_closed": True, "returned": {}, "errors": []}
        self._notify_stop.set()
        self._notify_evt.set()
        for run in runs:
            if run.sink is not None:
                run.sink.close()
            if run.handle is not None:
                run.handle._run_over("liveOD shut down", quiet=True)
        core = self._core
        if core is not None:
            done = threading.Event()

            def stop_core():
                try:
                    core.stop(timeout_s)
                except Exception as exc:
                    report["errors"].append(f"core stop: {exc}")
                finally:
                    done.set()
            t = threading.Thread(target=stop_core, daemon=True, name="CameraHost-stop")
            t.start()
            if not done.wait(timeout_s):
                report["cameras_closed"] = False
                busy = {cid: core.snapshot()["cameras"].get(cid, {}).get("busy_op")
                        for cid in core.camera_ids()}
                logger.warning(f"camera host: the cameras did not all close within "
                               f"{timeout_s:g} s (busy: {busy})")
        keeper = self._keeper
        if keeper is not None:
            try:
                report["returned"] = keeper.shutdown(1.0)
            except Exception as exc:
                report["errors"].append(f"return borrowed cameras: {exc}")
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        t = self._notifier
        if t is not None and t is not threading.current_thread():
            t.join(1.0)
        logger.info(f"camera host: shut down ({report})")
        return report

    def _factory_for(self, spec: CameraSpec):
        def make():
            if self._backend_factory is not None:
                return self._backend_factory(spec)
            return default_backend_factory(spec, self._constraints.get(spec.category.name, ()),
                                           self._live_cap)
        return make

    # ------------------------------------------------------------------ #
    # lookups
    # ------------------------------------------------------------------ #

    def spec(self, key: str) -> CameraSpec:
        try:
            return self._by_key[key]
        except KeyError:
            raise KeyError(f"liveOD has no camera {key!r}; it has {sorted(self._by_key)}") from None

    def keys(self) -> list:
        return [s.key for s in self.specs]

    def _require_started(self, what: str) -> CameraServerCore:
        core = self._core
        if not self._started or core is None:
            why = f" ({self._start_error})" if self._start_error else ""
            raise HostRefused(f"{what}: liveOD's camera host is not running{why}")
        if self._shut:
            raise HostRefused(f"{what}: liveOD's camera host is shutting down")
        return core

    def worker(self, key: str):
        spec = self.spec(key)
        if not spec.has_camera:
            raise HostRefused(f"{key!r} ({spec.camera_type or 'no camera type'}) has no camera "
                              f"in liveOD's camera host")
        return self._require_started(key).worker(spec.camera_id)

    def _constraints_for(self, spec: CameraSpec) -> tuple:
        return self._constraints.get(spec.category.name, ()) if spec.category else ()

    def _live_profile(self, key: str) -> dict:
        w = self.worker(key)
        prof = getattr(w, "_live_profile", None)       # replaced on every live apply, never mutated
        return dict(prof if prof else w.live_defaults)

    def _labels(self, key: str) -> dict:
        try:
            dyn = self.worker(key).dynamic
        except Exception:
            return {}
        out = {}
        for k, d in dyn.items():
            ch, lab = d.get("choices"), d.get("labels")
            if ch and lab and len(ch) == len(lab):
                out[k] = dict(zip(ch, lab))
        return out

    # ------------------------------------------------------------------ #
    # events, snapshots
    # ------------------------------------------------------------------ #

    def _bump(self) -> None:
        with self._rev_lock:
            self._rev += 1
        self._notify_evt.set()

    def _on_worker_event(self, camera_id: str, kind: str) -> None:
        spec = self._by_cid.get(camera_id)
        if spec is None:
            return
        if kind == "frame":
            cv = self._frame_cv.get(spec.key)
            if cv is not None:
                with cv:
                    cv.notify_all()
            return
        self._bump()

    def on_snapshot(self, cb: Callable[[dict], None]) -> Callable[[], None]:
        """``cb(snapshot)`` after every change and at least every heartbeat
        (1 s), on the host's own thread.  Returns the unsubscribe call."""
        with self._lock:
            self._callbacks.append(cb)
        self._notify_evt.set()
        return lambda: self.remove_snapshot_callback(cb)

    def remove_snapshot_callback(self, cb) -> None:
        with self._lock:
            self._callbacks = [c for c in self._callbacks if c is not cb]

    def _notify_loop(self) -> None:
        last_rev, last_t = -1, 0.0
        while not self._notify_stop.is_set():
            self._notify_evt.wait(self._heartbeat_s)
            self._notify_evt.clear()
            if self._notify_stop.is_set():
                return
            rev, now = self._rev, time.monotonic()
            if rev == last_rev and now - last_t < self._heartbeat_s * 0.9:
                continue
            try:
                snap = self.snapshot()
            except Exception:
                logger.exception("camera host: snapshot failed")
                continue
            last_rev, last_t = rev, now
            with self._lock:
                cbs = list(self._callbacks)
            for cb in cbs:
                try:
                    cb(snap)
                except Exception:
                    logger.exception("camera host: snapshot callback failed")

    def _host_state(self, spec: CameraSpec, ws: Optional[dict]) -> str:
        if not spec.has_camera:
            return "virtual"
        if ws is None:
            return "closed"
        st = ws["state"]
        busy = self._busy.get(spec.key)
        if busy:
            return busy
        with self._lock:
            token = self._run_by_key.get(spec.key)
            run = self._runs.get(token) if token else None
        if run is not None and run.locked and st in ("closed", "idle", "absent"):
            return "run_locked"
        if st in ("closed", "absent") and spec.key in self._held_elsewhere:
            return "held_elsewhere"
        return st

    def snapshot(self) -> dict:
        """The state of every camera (plain values; safe from any thread)."""
        core = self._core
        requesters = {}
        if core is not None and self._started:
            try:
                requesters = {cid: list(c.get("live_requesters") or ())
                              for cid, c in core.snapshot()["cameras"].items()}
            except Exception:
                requesters = {}
        cams = {}
        for spec in self.specs:
            ws = None
            if core is not None and spec.has_camera and self._started:
                try:
                    ws = dict(core.worker(spec.camera_id).snapshot())
                except KeyError:
                    ws = None
            hs = self._host_state(spec, ws)
            ps = self._persist.get(spec.key, PersistState())
            with self._lock:
                token = self._run_by_key.get(spec.key)
                run = self._runs.get(token) if token else None
            res = core.reservation(spec.camera_id) if (core is not None and spec.camera_id) else None
            if ws is not None and ws.get("is_open"):
                holder = {"label": "liveOD", "host": self.hostname, "pid": os.getpid()}
            elif res:
                holder = dict(res.get("holder") or {})
            else:
                holder = self._held_elsewhere.get(spec.key)
            claims = []
            keeper = self._keeper
            if keeper is not None and spec.camera_id:
                try:
                    claims = keeper.claims().get(spec.camera_id, [])
                except Exception:
                    claims = []
            cams[spec.key] = {
                "key": spec.key, "camera_id": spec.camera_id, "camera_type": spec.camera_type,
                "category": spec.category.name if spec.category else "", "serial": spec.serial,
                "state": legacy_state(hs), "host_state": hs,
                "persist": ps.on, "persisted": dict(ps.values), "persist_since": ps.since_iso,
                "holder": holder, "claims": claims,
                "n_subs": int(ws["n_subs"]) if ws else 0,
                "error": (ws or {}).get("error"),
                "settings": dict((ws or {}).get("settings") or {}),
                "settings_rev": int((ws or {}).get("settings_rev") or 0),
                "locked": bool((ws or {}).get("locked_by")) or bool(run is not None and run.locked),
                "run_id": run.run_id if run is not None else None,
                "run_tag": (ws or {}).get("run_tag"),
                "run_request": {k: _jsonable(v) for k, v in
                                (self._run_request.get(spec.key) or {}).items()},
                "live_requesters": requesters.get(spec.camera_id, []),
                "list_state": list_state(ws["state"], bool(res)) if ws else "closed",
            }
        return {"server_id": self.server_id, "instance": core.instance if core else None,
                "serving": bool(core is not None and getattr(core, "_serving", False)),
                "started": self._started, "rev": self._rev, "t": time.time(), "cameras": cams}

    def poll_cameras(self) -> dict:
        """For POLL: ``{key: {"state", "camera_type", "serial_no", ...}}`` -- the
        legacy words and keys, plus host_state, persist, holder, n_subs."""
        out = {}
        for key, c in self.snapshot()["cameras"].items():
            out[key] = {"state": c["state"], "camera_type": c["camera_type"],
                        "serial_no": c["serial"], "host_state": c["host_state"],
                        "persist": c["persist"], "holder": c["holder"], "n_subs": c["n_subs"]}
        return out

    def is_open(self, key: str) -> bool:
        try:
            return bool(self.worker(key).is_open)
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # remote writes (T3)
    # ------------------------------------------------------------------ #

    def _is_local(self, addr: str) -> bool:
        a = _s(addr).strip()
        if a.startswith("::ffff:"):
            a = a[len("::ffff:"):]
        if not a:
            return False
        return a in (self._local or set()) or a.startswith("127.")

    def _write_policy(self, client_addr: str, camera_id: str, keys) -> tuple:
        """The core's ``write_policy`` (module docstring, "Remote clients").
        A run's hold is not checked here: the worker's run lock refuses live
        settings and START_LIVE / STOP_LIVE for as long as a run holds the camera."""
        keys = [str(k) for k in keys]
        control = sorted(set(keys) & set(CONTROL_KEYS))
        if control:
            return False, (f"{control} can be set only in liveOD's own settings, never over "
                           f"the network")
        spec = self._by_cid.get(camera_id)
        cat = spec.category if spec is not None else None
        if not self._is_local(client_addr):
            other = sorted(k for k in keys
                           if k != LIVE_KEY and not (cat is not None and cat.has(k)))
            if other:
                return False, (f"{client_addr or 'an unknown address'} is not this PC: "
                               f"programs on other PCs may change the live settings of "
                               f"liveOD's cameras and ask for the live stream, but not "
                               f"{other} (T3)")
        if cat is None:
            return True, ""
        owner = sorted(k for k in keys if cat.has(k) and cat.setting(k).owner_only)
        if owner:
            return False, (f"{owner} can be changed only in liveOD's own {spec.key} settings "
                           f"(owner-only)")
        ps = self._persist.get(spec.key)
        if ps is not None and ps.on:
            held = sorted(set(keys) & set(ps.values))
            if held:
                return False, (f"persist is on for {spec.key} (since {ps.since_iso}): {held} "
                               f"{'is' if len(held) == 1 else 'are'} persisted into runs and "
                               f"cannot be changed remotely; turn Persist off in liveOD's "
                               f"{spec.key} settings first")
        return True, ""

    def _before_network_snap(self, camera_id: str) -> None:
        """A remote SNAP after a run: put the live profile back first, so the
        snap never runs on the run's trigger or EM gain."""
        spec = self._by_cid.get(camera_id)
        if spec is None:
            return
        w = self._core.worker(camera_id)
        if w.is_open and self._run_settings_rev.get(spec.key) == w.settings_rev \
                and not w.locked_by:
            w.submit("apply", values={}, purpose="live")

    def _before_network_live(self, camera_id: str) -> None:
        """A START_LIVE from a program on this PC: on an open camera that is
        not streaming, the live profile in full first (as ``start_stream``),
        queued ahead of the core's start_live.  A closed or locked camera is
        left alone: the core refuses those."""
        spec = self._by_cid.get(camera_id)
        if spec is None:
            return
        w = self._core.worker(camera_id)
        if w.is_open and not w.locked_by and w.state != "streaming":
            w.submit("apply", values={}, purpose="live")

    # ------------------------------------------------------------------ #
    # runs
    # ------------------------------------------------------------------ #

    def has_run(self, token: str) -> bool:
        with self._lock:
            return token in self._runs

    def arm_future(self, token: str) -> Optional[concurrent.futures.Future]:
        with self._lock:
            run = self._runs.get(token)
            return run.arm_future if run is not None else None

    def run_armed(self, token: str) -> bool:
        with self._lock:
            run = self._runs.get(token)
        if run is None or run.arm_result is None or not run.locked:
            return False
        try:
            return self.worker(run.key).locked_by == token
        except Exception:
            return False

    def _run_of(self, token: str) -> Optional[_Run]:
        with self._lock:
            return self._runs.get(token)

    def begin_run(self, token: str, camera_key: str, capture_images: bool,
                  camera_params=None, images_shape=None) -> RunStart:
        """INIT_RUN, before the run file is reserved.  Synchronous.

        A run without a camera (``capture_images`` False), or on an entry with
        no camera behind it (the APD), takes no lock.  Otherwise: the run
        profile is built and checked (camera_params, with Persist on top, the
        lab's constraints), the camera is borrowed from another server that
        has it, and it is locked with ``token``.  Raises ``HostRefused``
        naming the rule and the value; nothing is locked then."""
        token = str(token)
        with self._lock:
            old = self._run_by_key.get(camera_key)
        if old and old != token:
            self.end_run(old, "superseded by a new run")
        run = _Run(token=token, key=camera_key, capture_images=bool(capture_images),
                   images_shape=images_shape)
        spec = self._by_key.get(camera_key)
        if not capture_images or spec is None or not spec.has_camera:
            run.why_not = ("the run takes no frames" if not capture_images else
                           f"{camera_key!r} is not one of liveOD's cameras" if spec is None else
                           f"{camera_key!r} ({spec.camera_type or 'no camera type'}) has no "
                           f"camera in liveOD's camera host")
            with self._lock:
                self._runs[token] = run
            return RunStart(camera_key=camera_key, locked=False)
        core = self._require_started(f"INIT_RUN on {camera_key}")
        w = core.worker(spec.camera_id)
        cat = spec.category
        ps = self._persist.get(camera_key, PersistState())
        persisted = dict(ps.values) if ps.on else {}
        # the device's choices are checked here; its ranges are not: a value
        # outside one is clamped by the camera and recorded as such (D-c)
        choices = {k: {"choices": d["choices"]} for k, d in w.dynamic.items()
                   if isinstance(d, dict) and d.get("choices")}
        try:
            plan = build_run_profile(cat, camera_params or {}, persisted,
                                     extra=self._constraints_for(spec), dynamic=choices)
        except RunProfileError as exc:
            raise HostRefused(str(exc)) from None
        bad = refusals(plan.violations)
        if bad:
            raise HostRefused("; ".join(str(v) for v in bad))
        for key, why in plan.refused.items():
            logger.warning(f"camera host: persisted {camera_key} value not carried into the "
                           f"run: {why}")
        self._claim_if_needed(spec, w)
        try:
            w.call("lock", LOCK_TIMEOUT_S, token=token, reason="liveOD run (INIT_RUN)")
        except concurrent.futures.TimeoutError:
            busy = w.snapshot()["busy_op"]
            raise HostRefused(f"{camera_key}: the camera did not take the run lock within "
                              f"{LOCK_TIMEOUT_S:g} s (its thread is busy in {busy!r})") from None
        except LockedError as exc:
            raise HostRefused(f"{camera_key}: {exc}; if no run is using it, release it in "
                              f"liveOD (operator release)") from None
        run.locked = True
        run.camera_id = spec.camera_id
        run.category = cat
        run.plan = plan
        run.persisted = persisted
        run.persist_since = ps.since_iso if ps.on else None
        with self._lock:
            self._runs[token] = run
            self._run_by_key[camera_key] = token
            self._run_request[camera_key] = run_request_of(cat, plan)
        self._bump()
        overrides = {k: dict(v) for k, v in plan.overrides.items()}
        return RunStart(camera_key=camera_key, locked=True, category=cat.name,
                        profile=dict(plan.profile), overrides=overrides,
                        refused=dict(plan.refused), persist_on=ps.on,
                        persist_since=ps.since_iso if ps.on else None,
                        labels=self._labels(camera_key))

    def _claim_if_needed(self, spec: CameraSpec, w) -> None:
        """A camera that liveOD does not hold yet: borrow it from every other
        v2 camera server that lists it (bounded) -- a Basler from the beacon
        server of its PC, the Andor from the SLM spot finder while the spot
        finder has it open.  Nothing to borrow when no other server lists it.
        ``HostRefused`` naming the holder."""
        keeper = self._keeper
        if keeper is None or w.is_open or keeper.holds(spec.camera_id):
            return
        from waxx.util.live_od.camera_host.claims import ClaimRefused
        timeout_s = ANDOR_CLAIM_TIMEOUT_S if spec.camera_type == "andor" else CLAIM_TIMEOUT_S
        self._busy[spec.key] = "claiming"
        self._bump()
        try:
            keeper.claim(spec.camera_id, timeout_s=timeout_s)
            self._held_elsewhere.pop(spec.key, None)
        except ClaimRefused as exc:
            self._held_elsewhere[spec.key] = dict(exc.holder) or {"label": exc.server_id}
            raise HostRefused(f"{spec.key} ({spec.camera_id}) is held elsewhere: {exc}") from None
        except Exception as exc:
            raise HostRefused(f"{spec.key} ({spec.camera_id}): could not borrow it from the "
                              f"camera server that has it: {type(exc).__name__}: {exc}") from None
        finally:
            self._busy.pop(spec.key, None)
            self._bump()

    def note_run_id(self, token: str, run_id: int) -> None:
        """INIT_RUN reserved ``run_id``: it names the lock and the run's frames."""
        run = self._run_of(token)
        if run is None:
            return
        run.run_id = int(run_id or 0)
        if run.locked:
            try:
                self.worker(run.key).submit("lock", token=token, reason=f"run {run.run_id}")
            except Exception:
                pass
        self._bump()

    def arm_run(self, token: str, camera_params=None, n_img: Optional[int] = None
                ) -> concurrent.futures.Future:
        """Apply the full run profile and start the triggered acquisition of
        ``n_img`` frames; a Future of the worker's ``ArmResult``.  Idempotent
        per run: a second call returns the same Future."""
        fut: concurrent.futures.Future = concurrent.futures.Future()
        run = self._run_of(token)
        if run is None:
            fut.set_exception(HostRefused(f"arm: no run with token {str(token)[:8]}... (it "
                                          f"ended or a newer run took over)"))
            return fut
        with self._lock:
            if run.arm_future is not None:
                return run.arm_future
            run.arm_future = fut
        if not run.locked:
            fut.set_exception(HostRefused(f"arm: {run.why_not}; nothing to arm"))
            return fut
        spec = self._by_key[run.key]
        plan = run.plan
        if camera_params is not None:
            try:
                plan = build_run_profile(spec.category, camera_params, run.persisted,
                                         extra=self._constraints_for(spec))
                bad = refusals(plan.violations)
                if bad:
                    raise HostRefused("; ".join(str(v) for v in bad))
            except (RunProfileError, HostRefused) as exc:
                run.arm_error = str(exc)
                fut.set_exception(HostRefused(str(exc)))
                return fut
            with self._lock:
                if self._run_by_key.get(run.key) == token:
                    self._run_request[run.key] = run_request_of(spec.category, plan)
        try:
            n = int(n_img if n_img is not None else run.n_img)
            if n < 1:
                raise ValueError(f"n_img={n}, must be >= 1")
            w = self.worker(run.key)
        except Exception as exc:
            run.arm_error = str(exc)
            fut.set_exception(exc)
            return fut
        run.n_img = n
        run.run_tag = f"{run.run_id}:{token[:8]}"
        sink = RunSink(spec.camera_id, run.run_tag, n)
        run.sink = sink
        w.add_tap(sink)
        if w.snapshot()["state"] == "faulted":
            w.submit("open", token=token)            # a faulted camera is reopened, then armed
        wf = w.submit("arm", token=token, values=dict(plan.profile), n_frames=n,
                      run_tag=run.run_tag)

        def done(f, run=run, plan=plan, spec=spec):
            exc = f.exception()
            if exc is not None:
                run.arm_error = f"{type(exc).__name__}: {exc}"
                logger.error(f"camera host: run {run.run_id} on {spec.key}: arm failed: "
                             f"{run.arm_error}")
                fut.set_exception(exc)
                self._bump()
                return
            res = f.result()
            sink.set_acq_gen(res.acq_gen)
            run.clamps = readback_clamps(res.readback)
            run.plan = plan.with_clamps(run.clamps)
            run.arm_result = res
            self._run_settings_rev[spec.key] = w.settings_rev
            rb = plain_readback(res.readback)
            shown = ", ".join(f"{k}={rb[k]!r}" for k in sorted(rb))
            logger.debug(f"camera host: run {run.run_id} on {spec.key} armed for {n} "
                         f"frame(s) (acq_gen {res.acq_gen}, stray frames discarded "
                         f"{res.stale_discarded})")
            logger.log(VERBOSE, _ascii(f"camera host: run {run.run_id} on {spec.key} "
                                       f"read back: {shown}"))
            fut.set_result(res)
            self._bump()
        wf.add_done_callback(done)
        return fut

    def run_overrides(self, token: str) -> dict:
        """The run's overrides so far ({field: {requested, applied, origin}})."""
        run = self._run_of(token)
        if run is None or run.plan is None:
            return {}
        return {k: dict(v) for k, v in run.plan.overrides.items()}

    def attach_run(self, token: str):
        """The camera thread's handle on the run's frames (``HostCameraHandle``).
        A later attach takes over: the earlier handle ends."""
        from waxx.util.live_od.camera_host.legacy import HostCameraHandle
        run = self._run_of(token)
        if run is None:
            raise HostRefused(f"attach: no run with token {str(token)[:8]}... (it ended or a "
                              f"newer run took over)")
        handle = HostCameraHandle(self, run)
        with self._lock:
            prev, run.handle = run.handle, handle
        if prev is not None:
            prev._run_over("another camera thread attached to this run")
        return handle

    def end_run(self, token: str, reason: str = "END_RUN") -> RunSummary:
        """The run's hold on its camera ends: disarm (whatever is still in the
        camera is read), unlock.  The camera stays idle at the run's settings
        (no stream resumes).  An unknown token: an empty summary."""
        token = str(token)
        with self._lock:
            run = self._runs.pop(token, None)
            if run is not None and self._run_by_key.get(run.key) == token:
                del self._run_by_key[run.key]
        if run is None:
            return RunSummary(token=token, reason=reason)
        errors, disarm, run_fault = [], None, ""
        quiet = "superseded" in reason
        if quiet and run.handle is not None:
            # a run a newer one took over: its camera thread is about to be
            # stopped by the window, so it waits for that instead of failing
            # (a failure it reported could reach the newer run); said before
            # the lock goes, so the thread never sees it go unannounced
            run.handle._run_over(f"the run ended ({reason})", quiet=True)
        if run.locked and self._core is not None:
            w = self._core.worker(run.camera_id)
            try:
                disarm = w.call("disarm", END_RUN_OP_S, token=token)
            except Exception as exc:
                errors.append(f"disarm: {type(exc).__name__}: {exc}")
            ws = w.snapshot()
            if ws["state"] == "run_fault":
                # read after the disarm, which keeps the fault (and catches one in
                # the drain); the unlock reopens the camera
                run_fault = str(ws["error"] or "driver error")
            try:
                w.call("unlock", END_RUN_OP_S, token=token)
            except Exception as exc:
                errors.append(f"unlock: {type(exc).__name__}: {exc}")
        sink = run.sink
        if sink is not None:
            try:
                self._core.worker(run.camera_id).remove_tap(sink)
            except Exception:
                pass
            sink.close()
        if run.handle is not None and not quiet:
            run.handle._run_over(f"the run ended ({reason})")
        counts = sink.counts() if sink is not None else {}
        summary = RunSummary(
            token=token, reason=reason, camera_key=run.key, camera_id=run.camera_id,
            run_id=run.run_id, locked=run.locked, armed=run.arm_result is not None,
            arm_error=run.arm_error, n_frames=run.n_img,
            delivered=disarm.delivered if disarm else counts.get("accepted", 0),
            lost_idx=tuple(disarm.lost_idx) if disarm else (),
            stopped_early=bool(disarm.stopped_early) if disarm else False,
            surplus=int(getattr(disarm, "surplus", 0) or 0) if disarm else 0,
            stale=counts.get("stale", 0), foreign=counts.get("foreign", 0),
            overflow=counts.get("overflow", 0), run_fault=run_fault,
            overrides={k: dict(v) for k, v in (run.plan.overrides.items() if run.plan else ())},
            persist_since=run.persist_since, errors=tuple(errors))
        if run.locked:
            level = logging.WARNING if (summary.problems or errors or summary.surplus
                                        or summary.stale) else logging.DEBUG
            logger.log(level, f"camera host: run {run.run_id} on {run.key} released ({reason}): "
                              f"{summary.delivered}/{run.n_img} frame(s) delivered, lost "
                              f"{list(summary.lost_idx)}, surplus {summary.surplus}, stale "
                              f"{summary.stale}{', ' + '; '.join(errors) if errors else ''}; "
                              f"the camera stays idle at the run's settings")
        self._bump()
        return summary

    def abort_run(self, token: str) -> RunSummary:
        return self.end_run(token, "ABORT_RUN")

    def active_run(self, key: str) -> Optional[str]:
        with self._lock:
            return self._run_by_key.get(key)

    # ------------------------------------------------------------------ #
    # camera control (the window, CAMERA_CONTROL, the operator)
    # ------------------------------------------------------------------ #

    def _submit_task(self, fn, *args) -> concurrent.futures.Future:
        ex = self._executor
        if ex is None or self._shut:
            fut: concurrent.futures.Future = concurrent.futures.Future()
            fut.set_exception(HostRefused("liveOD's camera host is not running"))
            return fut
        return ex.submit(fn, *args)

    def request(self, key: str, action: str, origin: str = "") -> concurrent.futures.Future:
        """Open / close / toggle a camera, asynchronously.  Raises KeyError (an
        unknown key) and ValueError (an unknown action) at once; everything else
        comes back through the Future (and the log)."""
        spec = self.spec(key)
        if action not in ("open", "close", "toggle"):
            raise ValueError(f"camera action {action!r} is not open, close or toggle")
        if not spec.has_camera:
            fut: concurrent.futures.Future = concurrent.futures.Future()
            fut.set_exception(HostRefused(f"{key!r} has no camera in liveOD's camera host"))
            return fut

        def task():
            act = action
            if act == "toggle":
                act = "close" if self.is_open(key) else "open"
            logger.info(f"camera host: {key} -> {act}" + (f" ({origin})" if origin else ""))
            try:
                if act == "open":
                    return self._open(spec)
                return self._close(spec)
            except Exception as exc:
                logger.warning(f"camera host: {key} {act} failed: {exc}")
                raise
        return self._submit_task(task)

    def _open(self, spec: CameraSpec) -> dict:
        w = self.worker(spec.key)
        self._claim_if_needed(spec, w)
        return dict(w.call("open", 30.0))

    def _close(self, spec: CameraSpec, *, force: bool = False) -> dict:
        token = self.active_run(spec.key)
        if token and not force:
            run = self._run_of(token)
            raise HostRefused(f"close {spec.key}: run {run.run_id if run else '?'} holds the "
                              f"camera; release it as the operator to end that hold")
        w = self.worker(spec.key)
        report = w.call("close", 15.0)
        out = {"closed": True, "returned": {}}
        keeper = self._keeper
        if keeper is not None and spec.camera_id and keeper.holds(spec.camera_id):
            self._busy[spec.key] = "returning"
            self._bump()
            try:
                out["returned"] = keeper.release(spec.camera_id)
            finally:
                self._busy.pop(spec.key, None)
                self._bump()
        out["close_report"] = report
        return out

    def operator_release(self, key: str) -> concurrent.futures.Future:
        """The operator ends whatever holds ``key``: a run's lock (named in a
        WARNING), then closes it and gives a borrowed camera back."""
        spec = self.spec(key)

        def task():
            token = self.active_run(key)
            if token:
                run = self._run_of(token)
                logger.warning(f"camera host: operator release of {key}: ending run "
                               f"{run.run_id if run else '?'}'s hold on it; that run gets no "
                               f"more frames")
                self.end_run(token, "operator release")
            return self._close(spec, force=True)
        return self._submit_task(task)

    # ------------------------------------------------------------------ #
    # live (the liveOD GUI)
    # ------------------------------------------------------------------ #

    def start_stream(self, key: str,
                     requester: str = LIVEOD_REQUESTER) -> concurrent.futures.Future:
        """``requester`` asks for ``key``'s live stream: open if needed; on a
        camera not streaming yet, the live profile in full first (never the
        run's trigger or EM gain); then the core's ``request_live``.  Refused
        (in the Future) while a run holds the camera."""
        spec = self.spec(key)

        def task():
            core = self._require_started(f"start {key}'s live stream")
            w = self.worker(key)
            self._claim_if_needed(spec, w)
            w.call("open", 30.0)
            if w.state != "streaming":
                w.call("apply", 15.0, values={}, purpose="live")
            core.request_live(spec.camera_id, str(requester)).result(15.0)
            return True
        return self._submit_task(task)

    def stop_stream(self, key: str,
                    requester: str = LIVEOD_REQUESTER) -> concurrent.futures.Future:
        """``requester`` gives its request for ``key``'s live stream back; the
        stream stops only when nobody else (the spot finder, a viewer on this
        PC) still asks for it.  The Future is done once that is settled."""
        spec = self.spec(key)
        self.worker(key)                   # HostRefused at once: no camera, host not running
        return self._core.release_live(spec.camera_id, str(requester))

    def set_live(self, key: str, values: dict, confirmed=frozenset(),
                 timeout_s: float = 15.0) -> dict:
        """Change live settings: merged onto the live profile, the whole
        profile applied.  Refused (``ApplyRefused``, nothing sent) when a rule
        says no.  Returns ``{key: Readback}``.

        Live EM gain above the cap (a "confirm" rule) needs the unlock: either
        ``em_gain_unlocked=True`` in ``values`` (the cog's tick) or "gain" in
        ``confirmed`` (the operator confirmed this change, as the viewer asks).
        The unlock lasts while the gain stays above the cap; a change that
        leaves it at or below the cap locks it again."""
        spec = self.spec(key)
        w = self.worker(key)
        values = dict(values or {})
        confirmed = frozenset(confirmed or ())
        cat = spec.category
        if "em_gain_unlocked" in values and cat.name != ANDOR_EMCCD.name:
            raise ApplyRefused("em_gain_unlocked", f"only for {ANDOR_EMCCD.name} cameras, not "
                                                   f"{cat.name}")
        current = self._live_profile(key)
        merged = {**current, **values}
        if cat.name == ANDOR_EMCCD.name:
            if "em_gain_unlocked" in values:
                unlocked = bool(values["em_gain_unlocked"])
            elif "gain" in confirmed:
                unlocked = True
            else:
                try:
                    above = float(merged.get("gain", 0)) > self._live_cap
                except (TypeError, ValueError):
                    above = True
                unlocked = bool(current.get("em_gain_unlocked", False)) and above
            values["em_gain_unlocked"] = unlocked
            merged["em_gain_unlocked"] = unlocked
        check = {k: v for k, v in merged.items() if k not in CONTROL_KEYS}
        violations = check_constraints(cat, check, "live", extra=self._constraints_for(spec),
                                       dynamic=w.dynamic)
        bad = [v for v in violations
               if v.level == "refuse" or (v.level == "confirm"
                                          and not (merged.get("em_gain_unlocked")
                                                   or set(v.keys) <= confirmed))]
        if bad:
            raise ApplyRefused(",".join(bad[0].keys), str(bad[0]))
        return w.call("apply", timeout_s, values=values, purpose="live")

    def describe(self, key: str) -> dict:
        spec = self.spec(key)
        if not spec.has_camera:
            return {"ok": False, "error": f"{key!r} has no camera in liveOD's camera host"}
        w = self.worker(key)
        ps = self._persist.get(key, PersistState())
        return {"ok": True, "key": key, "camera_id": spec.camera_id, "category": spec.category.name,
                "schema": to_wire(spec.category), "dynamic": w.dynamic,
                "settings": dict(w.settings), "settings_rev": w.settings_rev,
                "live_profile": self._live_profile(key), "live_defaults": dict(spec.live_defaults),
                "constraints": [str(c.reason) for c in self._constraints_for(spec)],
                "persist": ps.on, "persisted": dict(ps.values), "persist_since": ps.since_iso,
                "persistable": list(spec.category.persistable_keys())}

    def wait_frame(self, key: str, after_seq: Optional[int] = None, timeout_s: float = 1.0,
                   sources=("live", "snap", "run")):
        """The newest frame of ``sources`` newer than ``after_seq``, waiting up
        to ``timeout_s``; None if none came.  Frames are read-only."""
        w = self.worker(key)
        cv = self._frame_cv[key]
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with cv:
            while True:
                f = w.slot.latest(sources=tuple(sources), after_seq=after_seq)
                if f is not None:
                    return f
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                cv.wait(min(left, 0.25))

    # ------------------------------------------------------------------ #
    # persist (D-a)
    # ------------------------------------------------------------------ #

    def persist(self, key: str) -> PersistState:
        spec = self.spec(key)
        return self._persist.get(key, PersistState(
            whitelist=spec.category.persistable_keys() if spec.category else ()))

    def set_persist(self, key: str, on: bool, values: Optional[dict] = None) -> PersistState:
        """Turn Persist on (capturing the current live values of the
        whitelisted fields, or ``values`` -- whitelisted keys only) or off.
        Refused while a run holds the camera."""
        spec = self.spec(key)
        if not spec.has_camera:
            raise HostRefused(f"{key!r} has no camera: nothing to persist")
        token = self.active_run(key)
        if token:
            run = self._run_of(token)
            raise HostRefused(f"persist for {key} cannot change while run "
                              f"{run.run_id if run else '?'} holds the camera")
        cat = spec.category
        whitelist = cat.persistable_keys()
        if not on:
            ps = PersistState(False, {}, None, whitelist)
            with self._lock:
                was = self._persist.get(key)
                self._persist[key] = ps
            if was is not None and was.on:
                logger.info(f"camera host: Persist OFF for {key}: runs use the experiment's "
                            f"camera_params again")
            self._bump()
            return ps
        if values is not None:
            bad = [k for k in values if k not in whitelist]
            if bad:
                raise ApplyRefused(bad[0], f"{bad[0]}={values[bad[0]]!r} does not persist into "
                                           f"runs: only {list(whitelist)} do (never exposure_time "
                                           f"or a run-owned field, D-a)")
            src = dict(values)
        else:
            src = self._live_profile(key) if self._started else dict(spec.live_defaults)
        persisted = {}
        for k in whitelist:
            if k in src and src[k] is not None:
                persisted[k] = normalize_value(cat.setting(k), src[k])
        ps = PersistState(True, persisted, time.time(), whitelist)
        with self._lock:
            self._persist[key] = ps
        shown = ", ".join(f"{k}={v!r}" for k, v in persisted.items())
        logger.warning(f"camera host: Persist ON for {key} at {ps.since_iso}: {shown} will "
                       f"replace the experiment's camera_params in every run on {key} until "
                       f"Persist is turned off (restarting liveOD turns it off)")
        self._bump()
        return ps


__all__ = ["CameraHost", "HostServerCore", "HostRefused", "CameraSpec", "PersistState",
           "RunStart", "RunSummary", "LEGACY_STATE", "legacy_state", "camera_specs",
           "default_backend_factory", "refuse_combos_from", "iso_time", "run_request_of",
           "LIVEOD_REQUESTER"]
