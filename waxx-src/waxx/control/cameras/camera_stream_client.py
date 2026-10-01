"""Per-shot auxiliary camera frames from a camera server (CameraStreamClient).

Snap one frame per shot from a camera that is NOT the run's acquisition
camera -- a camera hosted either by the beacon Basler server or by liveOD's
camera host -- at a sequence-defined moment, and save it into DataVault
containers. Built on ``beacon.camera.stream.CameraStream`` (v2 SNAP: an
on-demand acquisition matched by snap_id, not a conflatable live frame).

Timing model: the kernel never waits. ``grab(t_offset)`` sends one async RPC
carrying the kernel's current slack (``now_mu() - rtio_counter``) plus the
requested offset from the timeline cursor at the call. The host handler
timestamps arrival, computes the wall-clock target ``arrival + slack +
t_offset`` and enqueues it; a worker thread fires the snap at that target.
Accuracy is RPC-delivery-limited (~ms), the timeline and slack are untouched,
and the call is safe anywhere in the sequence. Several grabs may be scheduled
from the same point with different offsets::

    # prepare(), BEFORE finish_prepare():
    self.cam_mot = CameraStreamClient(self, '40277706', 'img_mot')
    # scan_kernel(), at the cursor the offsets are measured from:
    self.cam_mot.grab(self.p.t_mot_load - 100.e-3)

Settings: with no override, the camera's saved open-time defaults from the
beacon Basler server's per-serial store (``basler_server_defaults.json`` on
the camera's host, read over the network) are applied as the live settings
for the run; ``exposure_time`` / ``gain`` overrides replace the stored value.
The frame is cropped to the stored viewer ROI (``crop_to_saved_roi``). Prior
live settings are restored at the end of the run when nobody else changed
them meanwhile. Free-run snaps only: a camera whose trigger mode is not
"Off" is set to "Off" for the run (and restored).

Data: three host-only containers are registered on ``expt.data`` --
``<key>`` (the cropped frame, dtype from the probe snap), ``<key>_seq``
(int64 frame seq, -1 = no frame for that shot) and ``<key>_t`` /
``<key>_t_target`` (float64 server-monotonic frame time and the requested
target time; NaN = no frame). A grab that fails or lands late is recorded,
never raised: an async-RPC exception would kill the run with NO kernel
cleanup, so the RPC handler and the worker catch everything. finish() --
called from ``end_wax`` via ``expt.camera_streams`` -- drains in-flight
snaps, writes a provenance JSON into the run file
(``camera_stream_<key>``), prints a loud banner if any shot has no frame,
restores settings and closes the stream.
"""

import atexit
import json
import queue
import threading
import time

import numpy as np

from artiq.experiment import kernel, rpc
from artiq.language.core import now_mu

#: Worker heartbeat while idle: renews the v2 attachment on an on_demand
#: (beacon) server well inside its 30 s TTL, so a long pause between shots
#: cannot close the camera and silently reopen it at stored defaults.
HEARTBEAT_S = 8.0
#: A snap whose target is already this far in the past is skipped and
#: recorded as late (a backed-up worker must not snap ever-later frames).
T_LATE_TOLERANCE_S = 1.0
#: How many event records the provenance JSON keeps (it is an HDF5 attr:
#: keep it small; the per-shot numbers live in the _seq/_t containers).
MAX_EVENT_RECORDS = 100


_shared_dir_lock = threading.Lock()
_shared_dir = None


def shared_camera_directory():
    """One process-lived CameraDirectory for every CameraStreamClient: the
    camera list is collected once per process instead of once per camera
    (2 s of UDP collection each). Never closed -- the artiq_run process is
    short-lived."""
    global _shared_dir
    with _shared_dir_lock:
        if _shared_dir is None:
            from beacon.camera.directory import CameraDirectory
            _shared_dir = CameraDirectory()
        return _shared_dir


def crop_to_roi(image, roi):
    """``image[y1:y2, x1:x2]`` for a stored viewer ROI ``[x1, y1, x2, y2]``
    (x = column, y = row, end-exclusive), clamped to the frame. Returns the
    image unchanged for an empty/degenerate ROI."""
    if not roi or len(roi) != 4:
        return image
    H, W = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in roi)
    x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
    y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


#: liveOD's camera host advertises itself as ``camera_server:<host>:liveod``.
LIVEOD_SERVER_SUFFIX = ":liveod"


class _PinnedDirectory:
    """A CameraDirectory that resolves a camera on ONE server only.

    A CameraStream re-resolves its camera after a timeout. When liveOD holds
    the camera and stalls for a moment, the only listing left is the beacon
    server's "reserved" one, and a stream that follows it is refused there
    (run 84517) -- or, had the reservation lapsed, would open the USB device
    behind liveOD's back. Pinned, the re-resolution fails instead: that
    shot's snap is recorded as failed and the stream stays on liveOD."""

    def __init__(self, directory, server_id):
        self._directory = directory
        self._server_id = str(server_id)

    def find(self, query, *, refresh=False):
        from beacon.camera.directory import CameraNotFound
        entry = self._directory.find(query, refresh=refresh)
        if entry.server_id != self._server_id:
            raise CameraNotFound(
                f"{entry.camera_id} is served by {self._server_id}, which "
                f"did not answer; not following it to {entry.server_id} "
                f"(state {entry.state})")
        return entry

    def close(self):
        pass    # the shared directory is never closed


class CameraStreamClient:
    """One auxiliary camera, snapped per shot by async RPC (module docstring).

    Construct in ``prepare()`` BEFORE ``finish_prepare()`` (the containers
    must exist when ``DataVault.init()`` runs). ``camera`` is a serial, a
    camera_id, a name, or any object with ``.serial_no`` (a BaslerParams).
    ``exposure_time`` (s) / ``gain`` (dB) override the camera's saved
    defaults for this run. The kernel sees only ``grab`` /
    ``request_snap_mu`` / ``self.core``; every other attribute is host-only.
    """

    def __init__(self, expt, camera, key, *,
                 exposure_time=None, gain=None,
                 crop_to_saved_roi=True,
                 frame_timeout_s=5.0,
                 t_snap_lead=0.0,
                 t_late_tolerance=T_LATE_TOLERANCE_S,
                 label=None,
                 live_od_client=None,
                 directory=None,
                 _stream=None, _server_defaults=None):
        from beacon.camera.directory import query_string

        self._expt = expt
        self.core = expt.core
        self.key = str(key)
        self._exposure_override = exposure_time
        self._gain_override = gain
        self._crop_to_saved_roi = bool(crop_to_saved_roi)
        self._frame_timeout_s = float(frame_timeout_s)
        # A snap takes ~0.15-0.25 s from request to published frame (stop the
        # stream, arm a 1-frame grab, expose, read out, publish; measured on
        # kong 2026-09-30). Firing the snap t_snap_lead early puts the
        # EXPOSURE near the target instead of the publish; <key>_t always
        # records the true publish time, so the data shows what happened.
        self._t_snap_lead = float(t_snap_lead)
        self._t_late_tolerance = float(t_late_tolerance)
        self._live_od_client = live_od_client
        self.label = label or f"camera_stream:{self.key}"

        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._finished = False
        self._busy = False               # worker is serving a request
        self._events = []                # bounded event log for provenance
        self._shot_log = {"requested": 0, "ok": 0, "failed": 0, "late": 0,
                          "duplicate": 0, "rpc_errors": 0}
        self._req_count = {}             # idx -> times requested (warm-up guard)
        self._rev = None                 # settings_rev our snaps require
        self._priors = {}                # settings to restore at the end
        self._applied = {}               # what we set, with readback
        self._liveod_key = None
        self._roi = None

        self._check_key_available(expt, self.key)

        self.serial = query_string(camera)

        # --- camera resolution, settings, probe snap (network; may raise --
        # this is prepare(): failing the run here, with a clear message and
        # no run id spent, is the honest outcome) ---
        if _stream is not None:
            self._stream = _stream                     # test hook
            self._defaults = dict(_server_defaults or {})
        else:
            self._stream = self._resolve_and_connect(directory)
            self._defaults = self._read_server_defaults()
        try:
            frame0 = self._setup_settings_and_probe()
        except BaseException:
            try:
                self._stream.close()
            except Exception:
                pass
            raise

        self._frame_shape = tuple(frame0.shape)
        self._frame_dtype = frame0.dtype

        # --- containers (host-only; registered under the user's key) ---
        d = expt.data
        self.dc = d.add_host_data_container(self._frame_shape,
                                            dtype=self._frame_dtype,
                                            fill_value=0)
        self.dc_seq = d.add_host_data_container((1,), np.int64, fill_value=-1)
        self.dc_t = d.add_host_data_container((1,), np.float64,
                                              fill_value=np.nan)
        self.dc_t_target = d.add_host_data_container((1,), np.float64,
                                                     fill_value=np.nan)
        setattr(d, self.key, self.dc)
        setattr(d, self.key + "_seq", self.dc_seq)
        setattr(d, self.key + "_t", self.dc_t)
        setattr(d, self.key + "_t_target", self.dc_t_target)

        if not hasattr(expt, "camera_streams"):
            expt.camera_streams = []
        expt.camera_streams.append(self)

        self._worker = threading.Thread(target=self._work, daemon=True,
                                        name=f"CameraStreamClient:{self.key}")
        self._worker.start()
        atexit.register(self._atexit_cleanup)

        print(f"[{self.label}] {self.serial} via {self._stream.server_id}: "
              f"frame {self._frame_shape} {self._frame_dtype}"
              + (f", roi {self._roi}" if self._roi else "")
              + f", settings {self._applied_summary()}")

    # ------------------------------------------------------------------
    # kernel API
    # ------------------------------------------------------------------

    @kernel
    def grab(self, t_offset=0.):
        """Schedule this shot's snap ``t_offset`` seconds after the timeline
        cursor at this call. Emits no RTIO events, costs no slack."""
        self.request_snap_mu(now_mu() - self.core.get_rtio_counter_mu(),
                             t_offset)

    @rpc(flags={"async"})
    def request_snap_mu(self, slack_mu, t_offset):
        """Async RPC target. MUST NEVER RAISE: an exception here escapes the
        host serve loop and kills the run with no kernel cleanup."""
        try:
            t_arrival = time.monotonic()
            idx = tuple(int(x.counter) for x in self._expt.scan_xvars)
            try:
                slack_s = self.core.mu_to_seconds(slack_mu)
            except Exception:
                slack_s = float(slack_mu) * 1e-9
            target = t_arrival + slack_s + float(t_offset)
            with self._lock:
                n = self._req_count.get(idx, 0) + 1
                self._req_count[idx] = n
                self._shot_log["requested"] += 1
                if n > 1:
                    # warm-up shot or double grab: clear the slot so a stale
                    # frame can never pose as this shot's
                    self._shot_log["duplicate"] += 1
                    self._event("duplicate_request", idx=list(idx), count=n)
                    self._clear_slot_locked(idx)
            self._queue.put_nowait((idx, target, t_arrival))
        except Exception as e:
            try:
                with self._lock:
                    self._shot_log["rpc_errors"] += 1
                    self._event("rpc_error", error=repr(e))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # construction helpers (host)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_key_available(expt, key):
        if not key.isidentifier():
            raise ValueError(f"camera stream key {key!r} is not a valid "
                             f"python identifier")
        # after DataVault.init() every per-type kernel list is non-empty
        # (sentinels); before it they are all empty
        if expt.data.keys or getattr(expt.data, "_list_1d_f64", None):
            raise RuntimeError(
                f"camera stream '{key}': DataVault.init() already ran -- "
                f"construct CameraStreamClient before finish_prepare()")
        for k in (key, key + "_seq", key + "_t", key + "_t_target"):
            if hasattr(expt.data, k):
                raise ValueError(f"camera stream '{key}': expt.data.{k} "
                                 f"already exists")

    def _resolve_and_connect(self, directory):
        """Find the camera and open a CameraStream on the RIGHT server.

        A camera on liveOD's bar must be reached through liveOD's camera host
        (``camera_server:<host>:liveod``): resolving it to the beacon server
        while liveOD merely has it closed would open the USB device out from
        under liveOD. So: if liveOD lists the serial, take the lease from
        liveOD -- it borrows the camera from the server that has it and opens
        it (allowed only between runs) -- and insist the directory resolves it
        to the liveod server. A stream on liveOD's server stays there
        (_PinnedDirectory)."""
        from beacon.camera.stream import CameraStream

        directory = directory or shared_camera_directory()

        self._liveod_key, liveod_state = self._liveod_entry()
        if self._liveod_key is not None:
            if liveod_state not in ("open", "grabbing"):
                # refused while ANY run is active -- construct between runs
                try:
                    self._live_od_client.open_camera(self._liveod_key)
                except Exception as e:
                    raise RuntimeError(
                        f"[{self.label}] {self.serial} is on liveOD's bar as "
                        f"'{self._liveod_key}' (state {liveod_state}) and "
                        f"liveOD could not open it: {e}. A camera open is "
                        f"refused while any run is active -- retry between "
                        f"runs.") from e
            entry = directory.find(self.serial, refresh=True)
            if not entry.server_id.endswith(LIVEOD_SERVER_SUFFIX):
                raise RuntimeError(
                    f"[{self.label}] {self.serial} is on liveOD's bar as "
                    f"'{self._liveod_key}' but resolves to {entry.server_id} "
                    f"(state {entry.state}): liveOD's camera server did not "
                    f"answer -- refusing to open it behind liveOD's back")
        else:
            entry = directory.find(self.serial)
            holder = entry.holder or {}
            if (entry.state == "reserved" and str(
                    holder.get("server_id", "")).endswith(LIVEOD_SERVER_SUFFIX)):
                # lent to liveOD, and we could not ask liveOD about it (no
                # liveOD client, or its camera server did not answer)
                raise RuntimeError(
                    f"[{self.label}] {self.serial} is lent to liveOD "
                    f"({holder.get('server_id')}) by {entry.server_id}, but "
                    f"liveOD could not be asked for it"
                    + (" (this run has no liveOD client)"
                       if self._live_od_client is None else
                       " (liveOD did not answer)"))
        if entry.server_id.endswith(LIVEOD_SERVER_SUFFIX):
            directory = _PinnedDirectory(directory, entry.server_id)
        return CameraStream(entry, label=self.label, directory=directory)

    def _liveod_entry(self):
        """(liveOD camera key, state) for our serial, or (None, None)."""
        if self._live_od_client is None:
            return None, None
        try:
            cams = self._live_od_client.cameras()
        except Exception:
            return None, None
        for k, v in cams.items():
            if str(v.get("serial_no", "")) == self.serial:
                return k, v.get("state")
        return None, None

    def _read_server_defaults(self):
        """The per-serial saved defaults (SI) from the beacon Basler server on
        the camera's host -- the store the user edits with the viewer's "save
        defaults". Raises if no such server answers: the run was promised
        these settings."""
        from beacon.basler.camera_client import (BaslerCameraClient,
                                                 get_server_connection)
        from beacon.camera.directory import LEGACY_PREFIX
        from beacon.camera.viewer.sources import defaults_to_si
        from beacon.discovery import client as dc

        host = self._stream.entry.host
        dc.discover_prefix(LEGACY_PREFIX, collect_for=2.0)
        try:
            entries = dc.discover_entries(LEGACY_PREFIX, max_age=10.0)
        except Exception:
            entries = dc.discover_prefix(LEGACY_PREFIX, collect_for=0.0)
        match = None
        for sid, v in dict(entries or {}).items():
            ip = getattr(v, "ip", None) or (v.get("ip") if isinstance(v, dict)
                                            else None) or (v[0] if isinstance(
                                                v, (tuple, list)) else None)
            if str(ip) == str(host):
                match = str(sid)
                break
        if match is None:
            raise RuntimeError(
                f"[{self.label}] no beacon Basler server found on {host} to "
                f"read {self.serial}'s saved defaults from (is "
                f"basler_server running on that PC?)")
        conn = get_server_connection(match)
        r = BaslerCameraClient(conn, self.serial, model="").get_defaults()
        if not isinstance(r, dict) or not r.get("ok"):
            raise RuntimeError(f"[{self.label}] GET_DEFAULTS for "
                               f"{self.serial} on {match} failed: "
                               f"{(r or {}).get('error', 'no reply')}")
        return defaults_to_si(r.get("defaults"))

    def _setup_settings_and_probe(self):
        """Settle the camera, apply this run's settings, probe-snap.

        Returns the cropped probe frame (shape/dtype reference). The first
        snap also runs liveOD's post-run profile re-application, so settings
        are applied after it and pinned by settings_rev."""
        self._stream.snap(timeout=self._frame_timeout_s)   # settle profile

        current, rev = self._stream.get_settings()
        targets = {}
        exp = (self._exposure_override if self._exposure_override is not None
               else self._defaults.get("exposure_time"))
        gn = (self._gain_override if self._gain_override is not None
              else self._defaults.get("gain"))
        if exp is not None:
            targets["exposure_time"] = float(exp)
        if gn is not None:
            targets["gain"] = float(gn)
        if str(current.get("trigger_mode", "Off")) != "Off":
            targets["trigger_mode"] = "Off"   # free-run snaps only

        changed = {k: v for k, v in targets.items()
                   if self._differs(current.get(k), v)}
        if changed:
            self._priors = {k: current[k] for k in changed if k in current}
            rb = self._stream.set(**changed)
            if rb.get("errors"):
                raise RuntimeError(f"[{self.label}] settings refused for "
                                   f"{self.serial}: {rb['errors']}")
            rev = rb["settings_rev"]
            self._applied = rb.get("readback", {})
            for k, r in self._applied.items():
                if isinstance(r, dict) and r.get("origin") == "clamped":
                    print(f"[{self.label}] !! {self.serial} clamped {k}: "
                          f"requested {r.get('requested')}, applied "
                          f"{r.get('value')}")
        self._rev = rev

        if self._crop_to_saved_roi:
            self._roi = self._defaults.get("roi")

        frame0 = self._stream.snap(timeout=self._frame_timeout_s,
                                   require_rev=self._rev)
        return crop_to_roi(frame0.image, self._roi)

    @staticmethod
    def _differs(a, b, tol=1e-3):
        if a is None:
            return True
        if isinstance(b, str):
            return str(a) != b
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            return True
        scale = max(abs(a), abs(b), 1e-12)
        return abs(a - b) / scale > tol

    def _applied_summary(self):
        out = {}
        for k, r in (self._applied or {}).items():
            out[k] = r.get("value") if isinstance(r, dict) else r
        return out or "unchanged (server defaults already live)"

    # ------------------------------------------------------------------
    # worker (host thread)
    # ------------------------------------------------------------------

    def _work(self):
        while not self._stop.is_set():
            try:
                req = self._queue.get(timeout=HEARTBEAT_S)
            except queue.Empty:
                self._heartbeat()
                continue
            self._busy = True
            try:
                self._serve(*req)
            except Exception as e:
                with self._lock:
                    self._shot_log["failed"] += 1
                    self._event("worker_error", idx=list(req[0]),
                                error=repr(e))
            finally:
                self._busy = False
                self._queue.task_done()

    def _heartbeat(self):
        """Renew the v2 attachment so an on_demand server cannot close the
        camera (and reopen it later at stored defaults) during a pause."""
        if self._finished:
            return
        try:
            self._stream.get_settings()
        except Exception:
            pass

    def _serve(self, idx, target, t_arrival):
        # fire the snap t_snap_lead before the target so the exposure, not
        # the published frame, lands near the target
        t_fire = target - self._t_snap_lead
        # wait out the scheduled delay (interruptible)
        while not self._stop.is_set():
            dt = t_fire - time.monotonic()
            if dt <= 0:
                break
            time.sleep(min(dt, 0.2))
        if self._stop.is_set() and time.monotonic() < t_fire:
            with self._lock:
                self._shot_log["failed"] += 1
                self._event("aborted_before_target", idx=list(idx))
            return
        lateness = time.monotonic() - t_fire
        if lateness > self._t_late_tolerance:
            with self._lock:
                self._shot_log["late"] += 1
                self._event("late_skipped", idx=list(idx),
                            late_s=round(lateness, 3))
            return

        frame, failure = self._snap_once(idx)
        if frame is None:
            with self._lock:
                self._shot_log["failed"] += 1
                self._event("snap_failed", idx=list(idx), error=failure)
            return

        img = crop_to_roi(frame.image, self._roi)
        if tuple(img.shape) != self._frame_shape or img.dtype != self._frame_dtype:
            with self._lock:
                self._shot_log["failed"] += 1
                self._event("frame_shape_mismatch", idx=list(idx),
                            got=[list(img.shape), str(img.dtype)],
                            expected=[list(self._frame_shape),
                                      str(self._frame_dtype)])
            return

        with self._lock:
            if self._finished:
                self._event("frame_after_finish", idx=list(idx))
                return
            try:
                self.dc.put_shot_data_host(idx, img)
                self.dc_seq.put_shot_data_host(idx, frame.seq)
                self.dc_t.put_shot_data_host(idx, frame.t_mono)
                self.dc_t_target.put_shot_data_host(idx, target)
                self._shot_log["ok"] += 1
            except Exception as e:
                self._shot_log["failed"] += 1
                self._event("store_failed", idx=list(idx), error=repr(e))

    def _snap_once(self, idx):
        """One snap, with a single settings-reapply retry. Returns
        (frame, None) or (None, reason)."""
        from beacon.camera.stream import SettingsChanged
        try:
            return self._stream.snap(timeout=self._frame_timeout_s,
                                     require_rev=self._rev), None
        except SettingsChanged as e:
            # someone (a reopen at defaults, a viewer) changed the settings:
            # re-apply ours once and carry on at the new revision
            with self._lock:
                self._event("settings_changed_reapplied", idx=list(idx),
                            detail=str(e))
            try:
                reapply = dict(self._applied and {
                    k: (r.get("value") if isinstance(r, dict) else r)
                    for k, r in self._applied.items()})
                if reapply:
                    rb = self._stream.set(**reapply)
                    self._rev = rb["settings_rev"]
                else:
                    _, self._rev = self._stream.get_settings()
                return self._stream.snap(timeout=self._frame_timeout_s,
                                         require_rev=self._rev), None
            except Exception as e2:
                return None, repr(e2)
        except Exception as e:
            return None, repr(e)

    def _clear_slot_locked(self, idx):
        """Reset a shot's slots to the miss values (caller holds the lock)."""
        try:
            self.dc._run_data[tuple(idx)] = 0
            self.dc_seq.put_shot_data_host(idx, -1)
            self.dc_t.put_shot_data_host(idx, np.nan)
            self.dc_t_target.put_shot_data_host(idx, np.nan)
        except Exception:
            pass

    def _event(self, kind, **fields):
        if len(self._events) < MAX_EVENT_RECORDS:
            self._events.append({"event": kind, "t": round(time.time(), 3),
                                 **fields})

    # ------------------------------------------------------------------
    # end of run
    # ------------------------------------------------------------------

    def finish(self, timeout_s=None):
        """Drain in-flight snaps, write provenance, restore settings, close.
        Called from end_wax (expt.camera_streams). Never raises."""
        if self._finished:
            return
        try:
            timeout = (timeout_s if timeout_s is not None
                       else self._frame_timeout_s + 10.0)
            deadline = time.monotonic() + timeout
            while ((not self._queue.empty() or self._busy)
                   and time.monotonic() < deadline):
                time.sleep(0.05)
            drained = self._queue.empty() and not self._busy
            with self._lock:
                self._finished = True
                if not drained:
                    self._event("finish_timeout_inflight_lost",
                                queued=self._queue.qsize())
            self._stop.set()
            self._write_provenance()
            self._report()
        except Exception as e:
            print(f"[{self.label}] WARNING: finish bookkeeping failed: {e!r}")
        finally:
            self._restore_and_close()
            try:
                atexit.unregister(self._atexit_cleanup)
            except Exception:
                pass

    def _write_provenance(self):
        n_missing = int(self._shot_log["requested"] - self._shot_log["ok"]
                        - self._shot_log["duplicate"])
        record = {
            "key": self.key, "serial": self.serial,
            "camera_id": getattr(self._stream, "camera_id", "?"),
            "server_id": getattr(self._stream, "server_id", "?"),
            "liveod_key": self._liveod_key,
            "settings_applied": self._applied_summary()
            if self._applied else {},
            "settings_rev": self._rev,
            "server_defaults": {k: v for k, v in self._defaults.items()
                                if k != "norm_reference"},
            "overrides": {k: v for k, v in
                          (("exposure_time", self._exposure_override),
                           ("gain", self._gain_override)) if v is not None},
            "roi": self._roi,
            "frame_shape": list(self._frame_shape),
            "frame_dtype": str(self._frame_dtype),
            "shots": dict(self._shot_log),
            "n_shots_missing_frame": n_missing,
            "events": self._events,
        }
        try:
            self._expt._extra_file_texts[f"camera_stream_{self.key}"] = \
                json.dumps(record, default=repr)
        except Exception as e:
            print(f"[{self.label}] WARNING: could not store provenance "
                  f"({e!r})")

    def _report(self):
        log = self._shot_log
        n_missing = int(log["requested"] - log["ok"] - log["duplicate"])
        if n_missing > 0 or log["rpc_errors"]:
            print(f"[{self.label}] !! {n_missing} of {log['requested']} "
                  f"requested shots have NO frame in '{self.key}' "
                  f"(ok {log['ok']}, failed {log['failed']}, late "
                  f"{log['late']}, duplicates {log['duplicate']}, rpc errors "
                  f"{log['rpc_errors']}). Missing shots read "
                  f"{self.key}_seq == -1.")
        else:
            print(f"[{self.label}] {log['ok']}/{log['requested']} frames "
                  f"captured into '{self.key}'")

    def _restore_and_close(self):
        try:
            if self._priors:
                try:
                    _, rev = self._stream.get_settings()
                    if rev == self._rev:
                        self._stream.set(**self._priors)
                    else:
                        print(f"[{self.label}] note: {self.serial}'s settings "
                              f"were changed by someone else during the run "
                              f"(rev {rev} != {self._rev}); NOT restoring "
                              f"priors {self._priors}")
                except Exception as e:
                    print(f"[{self.label}] WARNING: could not restore "
                          f"{self.serial}'s settings ({e!r}); priors were "
                          f"{self._priors}")
        finally:
            try:
                self._stream.close()
            except Exception:
                pass

    def _atexit_cleanup(self):
        """Abort path (no end_wax): stop the worker, restore, close."""
        try:
            if not self._finished:
                with self._lock:
                    self._finished = True
                self._stop.set()
                self._restore_and_close()
        except Exception:
            pass

    def close(self):
        self.finish()


class DummyCameraStreamClient:
    """Stands in when the requested camera cannot be snapped this run (it is
    the run's own acquisition camera, or construction was told to degrade).
    Same kernel-visible surface as the real client, and the same four
    containers, so analysis reads every run alike -- but ``<key>`` is one
    uint8 per shot instead of a frame, and every shot carries the "no frame"
    marks (``<key>_seq`` -1, times NaN). Why is in the run's provenance."""

    def __init__(self, expt, key, reason=""):
        self._expt = expt
        self.core = expt.core
        self.key = str(key)
        self.label = f"camera_stream:{self.key}"
        self.reason = str(reason)
        CameraStreamClient._check_key_available(expt, self.key)
        d = expt.data
        self.dc = d.add_host_data_container((1,), np.uint8, fill_value=0)
        self.dc_seq = d.add_host_data_container((1,), np.int64, fill_value=-1)
        self.dc_t = d.add_host_data_container((1,), np.float64,
                                              fill_value=np.nan)
        self.dc_t_target = d.add_host_data_container((1,), np.float64,
                                                     fill_value=np.nan)
        setattr(d, self.key, self.dc)
        setattr(d, self.key + "_seq", self.dc_seq)
        setattr(d, self.key + "_t", self.dc_t)
        setattr(d, self.key + "_t_target", self.dc_t_target)
        if not hasattr(expt, "camera_streams"):
            expt.camera_streams = []
        expt.camera_streams.append(self)
        self._finished = False
        print(f"[{self.label}] note: no '{self.key}' frames this run "
              f"({self.reason})")

    @kernel
    def grab(self, t_offset=0.):
        pass

    def finish(self, timeout_s=None):
        if self._finished:
            return
        self._finished = True
        try:
            self._expt._extra_file_texts[f"camera_stream_{self.key}"] = \
                json.dumps({"key": self.key, "dummy": True,
                            "reason": self.reason,
                            "note": f"'{self.key}' is a placeholder (one "
                                    f"uint8 per shot), not a frame"})
        except Exception:
            pass

    def close(self):
        self.finish()
