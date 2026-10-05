"""Per-shot frames from auxiliary cameras, saved with the run.

A stream takes frames from a camera that is not the run's acquisition camera
(served by the beacon Basler server or by liveOD's camera host) and saves
them in DataVault containers: ``<key>`` holds the frame, ``<key>_meta`` one
record per shot (seq, times, camera counters, exposure, gain -- see
``waxa.data.camera_frames``; a NaN seq means no frame). Construct in
``prepare()`` before ``finish_prepare()``; ``finish()`` runs from end_wax.

Two kinds:

* ``CameraStreamClient`` -- free-run snaps. ``grab(t_offset)`` in the kernel
  sends an async RPC carrying the kernel's slack; a host thread snaps the
  camera at about that wall-clock time (RPC-limited, ~ms, plus the snap's
  own latency).
* ``TriggeredCameraStreamClient`` -- hardware-triggered frames. The camera
  waits for an edge on its TTL for the whole run; ``trigger(j)`` pulses it so
  frame ``j`` exposes at the timeline cursor, and the host thread fetches the
  frame the server publishes. Streams on a shared TTL are siblings: every
  edge is announced to all of them, and each keeps only its own frames.

Frames are not kept in the experiment process: the containers are
stream-only (``keep_run_data=False``), and each frame goes with its record
into the run's file during the run through the experiment's bounded shot
queue (``Expt.queue_shot_data``; waxx.base.shot_data_queue). A frame counts
as ``ok`` only once liveOD has taken it; one the queue could not deliver
(full, liveOD refusing, no PUT_DATA, left at the end) is counted as failed
with its reason, and its slot keeps the fill (zeros, NaN record). Nothing is
resent with END_RUN.

Neither ever raises into the run: a frame that fails or goes missing is
counted, kept in the provenance record (``camera_stream_<key>``) and printed
at the end. Settings come from the camera's saved beacon-server defaults,
with ``exposure_time`` / ``gain`` overrides; frames are cropped to the saved
viewer ROI. The camera is back in free-run after every run.
"""

import atexit
import bisect
import json
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np

from waxx.util import console

from artiq.experiment import kernel, rpc
from artiq.language.core import now_mu

from waxa.data.camera_frames import N_META, meta_row

# free-run: renew the attachment while idle (an on_demand server drops it after 30 s)
HEARTBEAT_S = 8.0
# free-run: a snap already this late is skipped rather than taken ever later
T_LATE_TOLERANCE_S = 1.0
# triggered: how long one frame poll waits; the fast poll when settings
# must be set between edges
TRIGGERED_POLL_S = 0.25
TRIGGERED_POLL_FAST_S = 0.05
# triggered: a frame's own settings are set at least this long before its edge
FRAME_SETTINGS_LEAD_S = 0.04
FRAME_SETTING_KEYS = ("exposure_time", "gain")
# settings a stream changes for itself and always puts back
TRIGGER_KEYS = ("trigger_mode", "trigger_source")
# finish() waits this long for the worker thread before closing the stream
WORKER_JOIN_S = 6.0
# finish() waits this long for the shot queue to deliver what is in it
# before writing the record (well inside Expt.T_STREAM_FINISH_S)
T_QUEUE_DRAIN_S = 30.0
# the provenance record keeps at most this many events, and per kind
MAX_EVENTS = 100
MAX_EVENTS_PER_KIND = 5

LIVEOD_SERVER_SUFFIX = ":liveod"      # liveOD's camera host: camera_server:<host>:liveod

_shared_dir = None
_shared_dir_lock = threading.Lock()
_beacons_collected = False


def shared_camera_directory():
    """One CameraDirectory per process (its camera listing costs 2 s)."""
    global _shared_dir
    with _shared_dir_lock:
        if _shared_dir is None:
            from beacon.camera.directory import CameraDirectory
            _shared_dir = CameraDirectory()
        return _shared_dir


def crop_to_roi(image, roi):
    """``image[y1:y2, x1:x2]`` for a viewer ROI ``[x1, y1, x2, y2]``; the
    image itself for no / degenerate ROI."""
    if not roi or len(roi) != 4:
        return image
    H, W = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in roi)
    x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
    y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def block_reduce(image, n):
    """``image`` downsampled n x n by block mean, same dtype."""
    n = int(n)
    if n <= 1:
        return image
    H, W = image.shape[:2]
    h, w = (H // n) * n, (W // n) * n
    if h == 0 or w == 0:
        return image
    out = image[:h, :w].reshape(h // n, n, w // n, n).mean(axis=(1, 3))
    if np.issubdtype(image.dtype, np.integer):
        out = np.rint(out)
    return out.astype(image.dtype)


def _as_keys(key):
    """One key or a sequence of keys as a tuple."""
    keys = (key,) if isinstance(key, str) else tuple(str(k) for k in key)
    if not keys:
        raise ValueError("a camera stream needs at least one key")
    if len(set(keys)) != len(keys):
        raise ValueError(f"camera stream keys repeat: {keys}")
    return keys


def _differs(a, b, tol=1e-3):
    if a is None:
        return True
    if isinstance(b, str):
        return str(a) != b
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return True
    return abs(a - b) > tol * max(abs(a), abs(b), 1e-12)


def _values(readback):
    """The plain values of a settings readback."""
    return {k: (r.get("value") if isinstance(r, dict) else r)
            for k, r in (readback or {}).items()}


def link_trigger_line(*streams):
    """Make triggered streams on one TTL siblings (module docstring)."""
    for s in streams:
        for o in streams:
            if o is not s and o not in s._siblings:
                s._siblings.append(o)


class _PinnedDirectory:
    """A camera directory that only ever resolves to one server. A stream on
    liveOD's host must not follow a momentary stall to the beacon server's
    listing of the same camera (it would be refused there, or worse, open
    the device behind liveOD's back)."""

    def __init__(self, directory, server_id):
        self._directory = directory
        self._server_id = str(server_id)

    def find(self, query, *, refresh=False):
        from beacon.camera.directory import CameraNotFound
        entry = self._directory.find(query, refresh=refresh)
        if entry.server_id != self._server_id:
            raise CameraNotFound(
                f"{entry.camera_id} is served by {self._server_id}, which did "
                f"not answer; not following it to {entry.server_id}")
        return entry

    def close(self):
        pass


class CameraStreamClient:
    """Free-run snaps of one camera, one per ``grab()`` (module docstring).

    ``camera`` is a serial, camera id, name or a params object with
    ``.serial_no``. ``exposure_time`` (s), ``gain`` (dB) and ``roi``
    (``[x1, y1, x2, y2]``) override the camera's saved defaults; with all
    three given the server's store is not read at all. ``t_snap_lead`` fires
    the snap early so the exposure, not the published frame, lands near the
    target (a snap takes ~0.15-0.4 s). A camera with no ROI stores its
    frames block-averaged ``no_roi_downsample`` x ``no_roi_downsample``
    (1 = full frames).
    """

    def __init__(self, expt, camera, key, *,
                 exposure_time=None, gain=None, roi=None, crop_to_saved_roi=True,
                 no_roi_downsample=1, frame_timeout_s=5.0, t_snap_lead=0.0,
                 t_late_tolerance=T_LATE_TOLERANCE_S, label=None,
                 live_od_client=None, directory=None,
                 _stream=None, _server_defaults=None):
        from beacon.camera.directory import query_string

        self._expt = expt
        self.core = expt.core
        self.key = str(key)
        self.label = label or f"camera_stream:{self.key}"
        self.serial = query_string(camera)
        self._exposure_override = exposure_time
        self._gain_override = gain
        self._roi_override = [int(v) for v in roi] if roi else None
        self._crop_to_saved_roi = bool(crop_to_saved_roi)
        self._no_roi_downsample = max(1, int(no_roi_downsample))
        self._frame_timeout_s = float(frame_timeout_s)
        self._t_snap_lead = float(t_snap_lead)
        self._t_late_tolerance = float(t_late_tolerance)
        self._live_od_client = live_od_client

        # reentrant: a shot-queue outcome can come back on the thread that
        # queued it (dropped at once), while that thread holds the lock
        self._lock = threading.RLock()
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._finished = False
        self._busy = False
        # ok = frames liveOD took (the slot's frame is in the run's file)
        self.counts = {"requested": 0, "ok": 0, "failed": 0, "late": 0,
                       "duplicate": 0, "rpc_errors": 0}
        self.events = []
        self._req_count = {}        # slot -> times requested (the slot's generation)
        self._slot_ok = {}          # slot -> its current request's frame is in the file
        self._provenance_written = False
        self._late_outcomes = 0     # queue outcomes after the record was written
        self._rev = None            # the settings revision our frames need
        self._priors = {}           # settings to put back at the end
        self._applied = {}          # readback of what we set
        self._liveod_key = None
        self._roi = None
        self._downsample = 1

        self._check_key_available(expt, self.key)

        # Network: resolve the camera, read its defaults, apply ours, take a
        # probe frame. This is prepare(): a failure here is loud and costs
        # no run id.
        if _stream is not None:                  # test hook
            self._stream = _stream
            self._defaults = dict(_server_defaults or {})
        else:
            self._stream = self._resolve_and_connect(directory)
            fully_given = (exposure_time is not None and gain is not None
                           and (self._roi_override or not self._crop_to_saved_roi))
            self._defaults = {} if fully_given else self._read_server_defaults()
        try:
            frame0 = self._setup_settings_and_probe()
        except BaseException:
            try:
                self._stream.close()
            except Exception:
                pass
            raise
        if not self._roi and self._no_roi_downsample > 1:
            self._downsample = self._no_roi_downsample
            frame0 = block_reduce(frame0, self._downsample)
            print(f"[{self.label}] !! {self.serial} has no saved ROI: its full "
                  f"{self._sensor_shape} frames are stored downsampled "
                  f"{self._downsample}x{self._downsample} -> {frame0.shape}. "
                  f"Save an ROI for it in the camera viewer.")
        self._frame_shape = tuple(frame0.shape)
        self._frame_dtype = frame0.dtype

        self._register_containers()
        if not hasattr(expt, "camera_streams"):
            expt.camera_streams = []
        expt.camera_streams.append(self)

        self._worker = threading.Thread(target=self._work, daemon=True,
                                        name=f"CameraStreamClient:{self.key}")
        self._worker.start()
        atexit.register(self._atexit_cleanup)

        console.info(f"[{self.label}] {self.serial} via {self._stream.server_id}: "
                     f"frame {self._frame_shape} {self._frame_dtype}"
                     + (f", roi {self._roi}" if self._roi else "")
                     + f", settings {_values(self._applied) or 'as saved'}",
                     console.VERBOSE)

    # ---- kernel side -----------------------------------------------------

    @kernel
    def grab(self, t_offset=0.):
        """Snap this shot ``t_offset`` seconds after the timeline cursor.
        No RTIO events, no slack used."""
        self.request_snap_mu(now_mu() - self.core.get_rtio_counter_mu(),
                             t_offset)

    @rpc(flags={"async"})
    def request_snap_mu(self, slack_mu, t_offset):
        """Async RPC target; must never raise (it would kill the run)."""
        try:
            idx = self._shot_index()
            target = time.monotonic() + self._slack_s(slack_mu) + float(t_offset)
            with self._lock:
                n = self._req_count.get(idx, 0) + 1
                self._req_count[idx] = n
                self.counts["requested"] += 1
                self._slot_ok[idx] = False
                if n > 1:        # a warm-up shot or a double grab: start over
                    self._clear_slot_locked(idx)
            if n > 1:
                self._note("duplicate", idx=list(idx), count=n)
            self._queue.put_nowait((idx, target, n))
        except Exception as e:
            self._note("rpc_errors", error=repr(e))

    @kernel
    def trigger(self, j=0):
        """Not for a free-run stream: counted and reported, no frame."""
        self.note_wrong_mode_call()

    @rpc(flags={"async"})
    def note_wrong_mode_call(self):
        self._note("wrong_mode_calls")

    def _shot_index(self):
        return tuple(int(x.counter) for x in self._expt.scan_xvars)

    def _slack_s(self, slack_mu):
        try:
            return self.core.mu_to_seconds(slack_mu)
        except Exception:
            return float(slack_mu) * 1e-9

    # ---- setup -----------------------------------------------------------

    @staticmethod
    def _check_key_available(expt, key):
        if not key.isidentifier():
            raise ValueError(f"camera stream key {key!r} is not an identifier")
        if expt.data.keys or getattr(expt.data, "_list_1d_f64", None):
            raise RuntimeError(f"camera stream '{key}': DataVault.init() already "
                               f"ran -- construct the stream before finish_prepare()")
        for k in (key, key + "_meta"):
            if hasattr(expt.data, k):
                raise ValueError(f"camera stream '{key}': expt.data.{k} exists")

    def _register_containers(self):
        d = self._expt.data
        # stream-only: no copy of the run's frames in this process
        self.dc = d.add_host_data_container(self._frame_shape,
                                            dtype=self._frame_dtype, fill_value=0,
                                            keep_run_data=False)
        self.dc_meta = d.add_host_data_container((N_META,), np.float64,
                                                 fill_value=np.nan,
                                                 keep_run_data=False)
        setattr(d, self.key, self.dc)
        setattr(d, self.key + "_meta", self.dc_meta)

    def _resolve_and_connect(self, directory):
        """A CameraStream on the right server: a camera on liveOD's bar is
        leased from liveOD (which borrows it from the beacon server) and the
        stream stays pinned there; any other camera from whoever serves it."""
        from beacon.camera.stream import CameraStream

        directory = directory or shared_camera_directory()
        self._liveod_key, state = self._liveod_entry()
        if self._liveod_key is not None:
            if state not in ("open", "grabbing"):
                try:
                    self._live_od_client.open_camera(self._liveod_key)
                except Exception as e:
                    raise RuntimeError(
                        f"[{self.label}] {self.serial} is on liveOD's bar as "
                        f"'{self._liveod_key}' (state {state}) and liveOD could "
                        f"not open it: {e} (opens are refused while a run is "
                        f"active -- retry between runs)") from e
            entry = directory.find(self.serial, refresh=True)
            if not entry.server_id.endswith(LIVEOD_SERVER_SUFFIX):
                raise RuntimeError(
                    f"[{self.label}] {self.serial} is on liveOD's bar but "
                    f"resolves to {entry.server_id}: liveOD's camera server did "
                    f"not answer -- refusing to open it behind liveOD's back")
        else:
            entry = directory.find(self.serial)
            holder = entry.holder or {}
            if (entry.state == "reserved"
                    and str(holder.get("server_id", "")).endswith(LIVEOD_SERVER_SUFFIX)):
                raise RuntimeError(
                    f"[{self.label}] {self.serial} is lent to liveOD "
                    f"({holder.get('server_id')}) but liveOD could not be asked "
                    f"for it" + (" (this run has no liveOD client)"
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
        """The camera's saved defaults (SI) from the beacon Basler server on
        its host -- the store the viewer's "save defaults" writes."""
        from beacon.basler.camera_client import BaslerCameraClient, get_server_connection
        from beacon.camera.directory import LEGACY_PREFIX
        from beacon.camera.viewer.sources import defaults_to_si
        from beacon.discovery import client as dc

        global _beacons_collected
        host = self._stream.entry.host

        def on_host(entries):
            for sid, v in dict(entries or {}).items():
                ip = (getattr(v, "ip", None) or (v.get("ip") if isinstance(v, dict) else None)
                      or (v[0] if isinstance(v, (tuple, list)) else None))
                if str(ip) == str(host):
                    return str(sid)
            return None

        if not _beacons_collected:          # 2 s, once per process
            dc.discover_prefix(LEGACY_PREFIX, collect_for=2.0)
            _beacons_collected = True
        try:
            server = on_host(dc.discover_entries(LEGACY_PREFIX, max_age=10.0))
        except Exception:
            server = None
        if server is None:
            server = on_host(dc.discover_prefix(LEGACY_PREFIX, collect_for=2.0))
        if server is None:
            raise RuntimeError(f"[{self.label}] no beacon Basler server on {host} "
                               f"to read {self.serial}'s saved defaults from")
        r = BaslerCameraClient(get_server_connection(server), self.serial,
                               model="").get_defaults()
        if not isinstance(r, dict) or not r.get("ok"):
            raise RuntimeError(f"[{self.label}] GET_DEFAULTS for {self.serial} on "
                               f"{server} failed: {(r or {}).get('error', 'no reply')}")
        return defaults_to_si(r.get("defaults"))

    def _setup_settings_and_probe(self):
        """Put the camera in free-run, apply this run's exposure / gain, take
        a probe frame (the stored shape and dtype). Returns the probe frame."""
        current, _ = self._stream.get_settings()
        if str(current.get("trigger_mode", "Off")) != "Off":
            # a stream that died without restoring left it waiting for edges
            self._priors["trigger_mode"] = current["trigger_mode"]
            self._stream.set(trigger_mode="Off")
        self._stream.snap(timeout=self._frame_timeout_s)      # settles the profile

        current, rev = self._stream.get_settings()
        wanted = {}
        exposure = (self._exposure_override if self._exposure_override is not None
                    else self._defaults.get("exposure_time"))
        gain = (self._gain_override if self._gain_override is not None
                else self._defaults.get("gain"))
        if exposure is not None:
            wanted["exposure_time"] = float(exposure)
        if gain is not None:
            wanted["gain"] = float(gain)
        changed = {k: v for k, v in wanted.items() if _differs(current.get(k), v)}
        if changed:
            for k in changed:
                if k in current:
                    self._priors.setdefault(k, current[k])
            rb = self._stream.set(**changed)
            if rb.get("errors"):
                raise RuntimeError(f"[{self.label}] settings refused for "
                                   f"{self.serial}: {rb['errors']}")
            rev = rb["settings_rev"]
            self._applied = rb.get("readback", {})
            for k, r in self._applied.items():
                if isinstance(r, dict) and r.get("origin") == "clamped":
                    print(f"[{self.label}] !! {self.serial} clamped {k}: asked "
                          f"{r.get('requested')}, got {r.get('value')}")
        self._rev = rev
        if self._roi_override:
            self._roi = self._roi_override
        elif self._crop_to_saved_roi:
            self._roi = self._defaults.get("roi")

        frame0 = self._stream.snap(timeout=self._frame_timeout_s, require_rev=self._rev)
        self._sensor_shape = tuple(frame0.image.shape)
        return crop_to_roi(frame0.image, self._roi)

    def _reduce(self, image):
        """A frame as stored: cropped, and downsampled if the camera has no ROI."""
        img = crop_to_roi(image, self._roi)
        return block_reduce(img, self._downsample) if self._downsample > 1 else img

    # ---- bookkeeping -----------------------------------------------------

    def _note(self, kind, **fields):
        """Count ``kind`` and keep the first few occurrences as events."""
        with self._lock:
            n = self.counts.get(kind, 0) + 1
            self.counts[kind] = n
            if n <= MAX_EVENTS_PER_KIND and len(self.events) < MAX_EVENTS:
                self.events.append({"event": kind, "t": round(time.time(), 3),
                                    **fields})

    def _fail(self, idx, reason, **fields):
        with self._lock:
            self.counts["failed"] += 1
        self._note(reason, idx=list(idx), **fields)

    def _store(self, dc, dc_meta, idx, img, frame, t_target, slot=None, gen=None,
               **more):
        """Queue one frame and its record for the run's file (one message);
        True when queued. ``slot`` / ``gen``: the request it answers -- a
        frame whose slot was requested again since (a warm-up shot) is not
        queued, so it cannot land after that slot's clear. Whether it reached
        the file comes later, to _on_pushed."""
        settings = getattr(frame, "settings", None) or {}
        row = meta_row(seq=frame.seq, t=frame.t_mono, t_target=t_target,
                       hw_ts=getattr(frame, "hw_ts", None),
                       hw_idx=getattr(frame, "hw_idx", None),
                       exposure=settings.get("exposure_time"),
                       gain=settings.get("gain"))
        slot = idx if slot is None else slot
        with self._lock:
            if self._finished:
                error = "finished"
            elif gen is not None and self._req_count.get(slot) != gen:
                error = "superseded"
            else:
                error = None
                # under the lock, so a re-request's clear cannot slip ahead
                # of it; returns at once (the network is the queue's thread)
                done = (lambda ok, reason, _s=slot, _g=gen, _i=idx, _m=more:
                        self._on_pushed(_s, _g, _i, ok, reason, _m))
                self._expt.queue_shot_data([(dc, img), (dc_meta, row)], idx=idx,
                                           on_done=done)
        if error == "finished":
            self._note("frame_after_finish", idx=list(idx), **more)
        elif error == "superseded":
            # its request was replaced by a later one for the same slot
            self._note("superseded_frames", idx=list(idx), **more)
        return error is None

    def _on_pushed(self, slot, gen, idx, ok, reason, more):
        """The shot queue's outcome for one frame (any thread)."""
        with self._lock:
            if ok:
                self.counts["ok"] += 1
                if gen is None or self._req_count.get(slot) == gen:
                    self._slot_ok[slot] = True
            if self._provenance_written:
                self._late_outcomes += 1
        if not ok:
            self._fail(idx, str(reason or "push_failed"), **more)

    def _clear_slot_locked(self, idx):
        """Start a scan point over as "no frame" (caller holds the lock):
        a clear goes to the run's file behind anything already queued for
        it, so a frame the earlier request sent does not stay there."""
        self._queue_clear(idx, [self.dc, self.dc_meta])

    def _queue_clear(self, idx, containers, **more):
        done = (lambda ok, reason, _i=idx, _m=more:
                self._on_cleared(_i, ok, reason, _m))
        try:
            self._expt.queue_shot_clear(containers, idx=idx, on_done=done)
        except Exception as e:
            self._on_cleared(idx, False, f"clear_error: {e!r}", more)

    def _on_cleared(self, idx, ok, reason, more):
        if ok:
            self._note("slot_clears_sent")
        elif reason == "no_put_data":
            # nothing of this run reaches the file: no earlier frame there
            self._note("slot_clears_not_needed")
        else:
            # the earlier request's frame may still be in this slot
            self._note("slot_clears_failed", idx=list(idx), reason=reason, **more)

    def _check_shape(self, idx, img, **more):
        if tuple(img.shape) == self._frame_shape and img.dtype == self._frame_dtype:
            return True
        self._fail(idx, "frame_shape_mismatch", got=[list(img.shape), str(img.dtype)],
                   expected=[list(self._frame_shape), str(self._frame_dtype)], **more)
        return False

    # ---- worker thread ---------------------------------------------------

    def _work(self):
        while not self._stop.is_set():
            try:
                req = self._queue.get(timeout=HEARTBEAT_S)
            except queue.Empty:
                self._heartbeat()
                continue
            if req is None:                     # the stop sentinel
                continue
            self._busy = True
            try:
                self._serve(*req)
            except Exception as e:
                self._fail(req[0], "worker_error", error=repr(e))
            finally:
                self._busy = False
                self._queue.task_done()

    def _heartbeat(self):
        if not self._finished:
            try:
                self._stream.get_settings()
            except Exception:
                pass

    def _serve(self, idx, target, gen=None):
        t_fire = target - self._t_snap_lead
        while not self._stop.is_set() and time.monotonic() < t_fire:
            time.sleep(min(t_fire - time.monotonic(), 0.2))
        if self._stop.is_set() and time.monotonic() < t_fire:
            self._fail(idx, "aborted_before_target")
            return
        lateness = time.monotonic() - t_fire
        if lateness > self._t_late_tolerance:
            self._note("late", idx=list(idx), late_s=round(lateness, 3))
            return
        frame, failure = self._snap_once()
        if frame is None:
            self._fail(idx, "snap_failed", error=failure)
            return
        img = self._reduce(frame.image)
        if self._check_shape(idx, img):
            self._store(self.dc, self.dc_meta, idx, img, frame, target,
                        slot=idx, gen=gen)

    def _snap_once(self):
        """One snap at our settings revision; if someone changed the settings
        meanwhile, re-apply ours once and snap again. (frame, None) or
        (None, reason)."""
        from beacon.camera.stream import SettingsChanged
        try:
            return self._stream.snap(timeout=self._frame_timeout_s,
                                     require_rev=self._rev), None
        except SettingsChanged as e:
            self._note("settings_changed_reapplied", detail=str(e))
            try:
                ours = _values(self._applied)
                if ours:
                    self._rev = self._stream.set(**ours)["settings_rev"]
                else:
                    _, self._rev = self._stream.get_settings()
                return self._stream.snap(timeout=self._frame_timeout_s,
                                         require_rev=self._rev), None
            except Exception as e2:
                return None, repr(e2)
        except Exception as e:
            return None, repr(e)

    # ---- end of run ------------------------------------------------------

    def finish(self, timeout_s=None):
        """Wait for frames still in flight, then for the shot queue to
        deliver them, write the provenance, print the tally, put the
        camera's settings back, close. Never raises."""
        if self._finished:
            return
        try:
            deadline = time.monotonic() + (timeout_s if timeout_s is not None
                                           else self._frame_timeout_s + 10.0)
            while self._inflight() and time.monotonic() < deadline:
                time.sleep(0.05)
            lost = self._inflight()
            with self._lock:
                self._finished = True
            if lost:
                self._note("finish_timeout_inflight_lost", queued=self._queue.qsize())
            self._stop_worker()
            # no frame can be queued once the worker is gone; then every
            # queued frame's outcome before the record is written
            self._join_worker()
            self._drain_shot_queue()
            with self._lock:
                # set first: an outcome racing the write is caught as late
                self._provenance_written = True
            self._write_provenance()
            self._report()
        except Exception as e:
            print(f"[{self.label}] WARNING: finish bookkeeping failed: {e!r}")
        finally:
            self._join_worker()
            self._restore_and_close()
            try:
                atexit.unregister(self._atexit_cleanup)
            except Exception:
                pass

    def close(self):
        self.finish()

    def _drain_shot_queue(self):
        """Wait (bounded) for the experiment's shot queue to be empty. The
        queue is shared by every stream, so this waits for theirs too; what
        is still there after T_QUEUE_DRAIN_S is settled by Expt when it
        closes the queue (refresh_after_queue_close)."""
        drain = getattr(self._expt, "drain_shot_data", None)
        if drain is None:
            return
        try:
            if not drain(T_QUEUE_DRAIN_S):
                self._note("queue_not_drained_at_finish", wait_s=T_QUEUE_DRAIN_S)
        except Exception as e:
            self._note("queue_not_drained_at_finish", error=repr(e))

    def refresh_after_queue_close(self):
        """Expt closed the shot queue: if any of our frames got its outcome
        after the record was written (left in the queue, or still being
        sent), write the record again and say so. Never raises."""
        try:
            with self._lock:
                late, self._late_outcomes = self._late_outcomes, 0
            if not late:
                return
            self._write_provenance()
            print(f"[{self.label}] {late} frame outcome(s) came after the record "
                  f"was first written (the shot queue closed); record updated:")
            self._report()
        except Exception as e:
            print(f"[{self.label}] WARNING: could not update the record ({e!r})")

    def _atexit_cleanup(self):
        """The abort path (no end_wax)."""
        try:
            if not self._finished:
                with self._lock:
                    self._finished = True
                self._stop_worker()
                self._join_worker()
                self._restore_and_close()
        except Exception:
            pass

    def _inflight(self):
        return (not self._queue.empty()) or self._busy

    def _stop_worker(self):
        self._stop.set()
        self._queue.put_nowait(None)            # wakes a worker waiting on the queue

    def _join_worker(self):
        """The stream must not be closed under a request in flight (run
        84562 hung in zmq term() on exactly that)."""
        w = self._worker
        if w is threading.current_thread() or not w.is_alive():
            return
        w.join(WORKER_JOIN_S)
        if w.is_alive():
            print(f"[{self.label}] note: worker still busy after {WORKER_JOIN_S:g} s")

    def _n_missing(self):
        """Requested slots whose frame is not in the run's file."""
        with self._lock:
            return sum(1 for ok in self._slot_ok.values() if not ok)

    def _n_slots(self):
        with self._lock:
            return len(self._slot_ok)

    # counts kinds that are a frame the shot queue did not deliver
    QUEUE_REASONS = ("queue_full", "push_failed", "not_sent_at_end",
                     "unconfirmed_at_end", "no_put_data", "queue_error")

    def _queue_losses(self):
        """{reason: n} of frames the shot queue did not deliver."""
        c = self.counts
        return {r: c[r] for r in self.QUEUE_REASONS if c.get(r)}

    def _provenance_extra(self):
        return {"mode": "free_run", "t_snap_lead": self._t_snap_lead}

    def _provenance_record(self):
        return {
            **self._provenance_extra(),
            "key": self.key,
            "serial": self.serial,
            "camera_id": getattr(self._stream, "camera_id", "?"),
            "server_id": getattr(self._stream, "server_id", "?"),
            "liveod_key": self._liveod_key,
            "settings_applied": _values(self._applied),
            "settings_rev": self._rev,
            "server_defaults": {k: v for k, v in self._defaults.items()
                                if k != "norm_reference"},
            "overrides": {k: v for k, v in (("exposure_time", self._exposure_override),
                                            ("gain", self._gain_override),
                                            ("roi", self._roi_override))
                          if v is not None},
            "roi": self._roi,
            "downsample": self._downsample,
            "sensor_frame_shape": list(getattr(self, "_sensor_shape", ())),
            "frame_shape": list(self._frame_shape),
            "frame_dtype": str(self._frame_dtype),
            # frames went into the run's file during the run (shot queue ->
            # PUT_DATA); shots.ok counts only those liveOD took, and a frame
            # the queue did not deliver is in shots.failed under its reason
            "storage": "stream_only",
            "shots": dict(self.counts),
            "frames_not_delivered": self._queue_losses(),
            "n_shots_missing_frame": self._n_missing(),
            "events": list(self.events),
        }

    def _write_provenance(self):
        try:
            self._expt._extra_file_texts[f"camera_stream_{self.key}"] = \
                json.dumps(self._provenance_record(), default=repr)
        except Exception as e:
            print(f"[{self.label}] WARNING: could not store provenance ({e!r})")

    def _report(self):
        c = self.counts
        if c.get("wrong_mode_calls"):
            print(f"[{self.label}] !! {c['wrong_mode_calls']} call(s) to the other "
                  f"mode's kernel method (grab vs trigger) were ignored")
        self._report_clears()
        missing = self._n_missing()
        if missing or c["rpc_errors"]:
            lost = self._queue_losses()
            print(f"[{self.label}] !! {missing} of {self._n_slots()} requested shots "
                  f"have NO frame in '{self.key}' (in the file {c['ok']}, failed "
                  f"{c['failed']}, late {c['late']}, duplicates {c['duplicate']}, "
                  f"rpc errors {c['rpc_errors']}"
                  + (f"; not delivered to liveOD: {lost}" if lost else "")
                  + f"); their {self.key}_meta seq is NaN")
        else:
            print(f"[{self.label}] {c['ok']}/{c['requested']} frames captured "
                  f"into '{self.key}'")

    def _report_clears(self):
        """A warm-up slot whose clear never reached the file may still hold
        the warm-up's frame: said out loud."""
        n = self.counts.get("slot_clears_failed", 0)
        if n:
            slots = [e.get("idx") for e in self.events
                     if e.get("event") == "slot_clears_failed"]
            print(f"[{self.label}] !! {n} slot clear(s) for a repeated request "
                  f"(warm-up shot) did NOT reach liveOD: an earlier frame may remain "
                  f"in those slots, e.g. {slots}")

    def _before_restore(self):
        pass

    def _restore_and_close(self):
        """Trigger mode off whatever it was before; the trigger line and --
        only if nobody else changed the camera during the run -- exposure
        and gain back to what we found. Then close the stream."""
        try:
            try:
                self._before_restore()
            except Exception:
                pass
            current, rev = self._stream.get_settings()
            restore = {}
            if str(current.get("trigger_mode", "Off")) != "Off":
                restore["trigger_mode"] = "Off"
            if "trigger_source" in self._priors:
                restore["trigger_source"] = self._priors["trigger_source"]
            theirs = {k: v for k, v in self._priors.items() if k not in TRIGGER_KEYS}
            if theirs and rev == self._rev:
                restore.update(theirs)
            elif theirs:
                print(f"[{self.label}] note: {self.serial} was changed by someone "
                      f"else during the run; leaving {theirs} as they are now")
            if restore:
                self._stream.set(**restore)
        except Exception as e:
            print(f"[{self.label}] WARNING: could not restore {self.serial}'s "
                  f"settings ({e!r}); they were {self._priors}")
        finally:
            try:
                self._stream.close()
            except Exception:
                pass


class TriggeredCameraStreamClient(CameraStreamClient):
    """Hardware-triggered frames of one camera (module docstring).

    ``key`` is one container key or a list of them, one per frame a shot
    takes with this camera; ``trigger(j)`` exposes frame ``j`` at the cursor
    by pulsing ``ttl`` (``exposure_delay`` early, like Image.trigger_camera).
    Keep a camera's edges >~50 ms apart: the server holds only its newest
    frame until it is fetched.

    Frames are paired to announced edges in order of expected time, and a
    frame is kept only when the camera's own frame counter advanced by one
    since the previous frame, it was taken in trigger mode at our settings
    revision, and it came within ``-t_early_tolerance .. +t_match_window``
    of its edge. A counter gap means a frame was lost on the way, so every
    edge due by then is recorded as missing; nothing is ever back-filled.

    ``frame_settings`` ({frame index or key: {exposure_time, gain}}) gives a
    frame its own settings: the worker sets them once the previous frame is
    in and at least FRAME_SETTINGS_LEAD_S before the edge, otherwise the
    frame is taken as the camera stands and flagged (its record has the
    real values either way). Which frame comes next is learned from the
    order the kernel announces them (frame 0 first, then the listed order
    until the run shows otherwise).
    """

    def __init__(self, expt, camera, key, *, ttl, trigger_source="Line1",
                 exposure_delay=0., t_trigger=2.e-6, t_match_window=2.0,
                 t_early_tolerance=0.25, t_stray_grace=0.3, frame_settings=None,
                 **kw):
        self.ttl = ttl                                  # kernel-visible
        self.exposure_delay = float(exposure_delay)
        self.t_trigger = float(t_trigger)
        self.keys = _as_keys(key)
        for k in self.keys:
            CameraStreamClient._check_key_available(expt, k)
        self._trigger_source = str(trigger_source)
        self._t_match_window = float(t_match_window)
        self._t_early_tolerance = float(t_early_tolerance)
        self._t_stray_grace = float(t_stray_grace)
        self._frame_settings = self._parse_frame_settings(frame_settings)
        self._base_settings = {}            # what frames without their own get
        self._cur_settings = None           # what the camera holds (None = unknown)
        self._settings_done = set()         # pending edges whose settings were handled
        self._last_frame = None             # index of the last frame received
        self._last_announced = None         # index of the last frame announced
        self._next_after = {}               # frame -> the frame announced after it
        self._siblings = []                 # streams on the same TTL
        self._pending = []                  # (t_expected, idx, j) sorted by time
        self._slot_ok = {}                  # (idx, j) -> frame stored
        self._slots = []                    # per-key containers
        self._after = 0                     # newest server seq seen
        self._instance = None               # server process we baselined on
        self._last_hw_idx = self._last_seq = self._last_acq_gen = None
        self._live_started = False
        kw.pop("t_snap_lead", None)
        if len(self.keys) > 1:
            kw.setdefault("label", f"camera_stream:{self.keys[0]}+{len(self.keys) - 1}")
        super().__init__(expt, camera, self.keys[0], **kw)
        if self._frame_settings:
            console.info(f"[{self.label}] per-frame settings: "
                         + ", ".join(f"{self.keys[j]}={v}"
                                     for j, v in sorted(self._frame_settings.items()))
                         + f"; others {self._base_settings}", console.VERBOSE)

    def _parse_frame_settings(self, frame_settings):
        out = {}
        for j, vals in dict(frame_settings or {}).items():
            jj = self.keys.index(j) if isinstance(j, str) else int(j)
            if not 0 <= jj < len(self.keys):
                raise ValueError(f"frame_settings for frame {j!r}: no such frame "
                                 f"(keys {self.keys})")
            bad = sorted(set(vals) - set(FRAME_SETTING_KEYS))
            if bad:
                raise ValueError(f"frame_settings for {self.keys[jj]!r}: {bad} "
                                 f"cannot change between frames (only "
                                 f"{FRAME_SETTING_KEYS})")
            out[jj] = {k: float(v) for k, v in vals.items()}
        return out

    # ---- kernel side -----------------------------------------------------

    @kernel
    def trigger(self, j=0):
        """Expose frame ``j`` at the timeline cursor; the cursor stays."""
        self.announce_trigger_mu(now_mu() - self.core.get_rtio_counter_mu(), j)
        delay(-self.exposure_delay)
        self.ttl.pulse(self.t_trigger)
        delay(self.exposure_delay - self.t_trigger)

    @kernel
    def grab(self, t_offset=0.):
        """Not for a triggered stream: counted and reported, no frame."""
        self.note_wrong_mode_call()

    @rpc(flags={"async"})
    def announce_trigger_mu(self, slack_mu, j):
        """Async RPC target; must never raise."""
        try:
            idx = self._shot_index()
            t_expected = time.monotonic() + self._slack_s(slack_mu)
            j = int(j)
            if not 0 <= j < len(self.keys):
                self._note("bad_frame_index", idx=list(idx), j=j)
                j = -1                      # the TTL pulsed: expect the frame, keep nothing
            self._announce(idx, j, t_expected)
            for sib in self._siblings:
                sib.announce_sibling_edge(idx, t_expected)
        except Exception as e:
            self._note("rpc_errors", error=repr(e))

    def announce_sibling_edge(self, idx, t_expected):
        """A sibling fired our line: the frame is expected, not kept."""
        try:
            self._note("sibling_edges")
            self._queue.put_nowait((float(t_expected), tuple(idx), -1, 0))
        except Exception:
            pass

    def _announce(self, idx, j, t_expected):
        """Queue an announced edge as (t_expected, idx, j, generation): the
        generation is the request count of slot (idx, j), so a frame that
        answers an earlier request of a re-requested slot is not stored."""
        duplicate = False
        n = 0
        with self._lock:
            if j >= 0:
                if self._last_announced is not None:
                    self._next_after[self._last_announced] = j
                self._last_announced = j
                n = self._req_count.get((idx, j), 0) + 1
                self._req_count[(idx, j)] = n
                self.counts["requested"] += 1
                self._slot_ok[(idx, j)] = False
                if n > 1:                   # a warm-up shot: start this frame over
                    self._clear_slot_locked(idx, j)
                    duplicate = True
        if duplicate:
            self._note("duplicate", idx=list(idx), frame=j, count=n)
        self._queue.put_nowait((float(t_expected), tuple(idx), int(j), n))

    # ---- setup -----------------------------------------------------------

    def _register_containers(self):
        d = self._expt.data
        for k in self.keys:
            # stream-only: no copy of the run's frames in this process
            slot = SimpleNamespace(
                key=k,
                dc=d.add_host_data_container(self._frame_shape,
                                             dtype=self._frame_dtype, fill_value=0,
                                             keep_run_data=False),
                dc_meta=d.add_host_data_container((N_META,), np.float64,
                                                  fill_value=np.nan,
                                                  keep_run_data=False))
            setattr(d, k, slot.dc)
            setattr(d, k + "_meta", slot.dc_meta)
            self._slots.append(slot)
        self.dc, self.dc_meta = self._slots[0].dc, self._slots[0].dc_meta

    def _setup_settings_and_probe(self):
        frame0 = super()._setup_settings_and_probe()
        try:
            self._enter_triggered()
        except BaseException:
            try:                            # leave the camera as we found it
                self._before_restore()
                if self._priors:
                    self._stream.set(**self._priors)
            except Exception:
                pass
            raise
        if self._frame_settings:            # what frames without their own get
            current, _ = self._stream.get_settings()
            self._base_settings = {k: current[k] for k in FRAME_SETTING_KEYS
                                   if current.get(k) is not None}
            self._cur_settings = dict(self._base_settings)
        return frame0

    def _enter_triggered(self):
        """Live stream on, then trigger mode on our line (in that order:
        liveOD's host puts its free-run profile back on a START_LIVE), and
        confirm both from the server."""
        self._stream.start_live()
        self._live_started = True
        current, _ = self._stream.get_settings()
        wanted = {"trigger_selector": "FrameStart", "trigger_source": self._trigger_source,
                  "line_mode": "Input", "trigger_mode": "On"}
        for k in TRIGGER_KEYS:
            if k in current and _differs(current[k], wanted[k]):
                self._priors.setdefault(k, current[k])
        rb = self._stream.set(**wanted)
        if rb.get("errors"):
            raise RuntimeError(f"[{self.label}] trigger settings refused for "
                               f"{self.serial}: {rb['errors']}")
        self._rev = rb["settings_rev"]
        self._applied = {**self._applied, **rb.get("readback", {})}

        settings, rev = self._stream.get_settings()
        got = (str(settings.get("trigger_mode")), str(settings.get("trigger_source")))
        if got != ("On", self._trigger_source) or rev != self._rev:
            raise RuntimeError(f"[{self.label}] {self.serial} is not waiting for a "
                               f"trigger on {self._trigger_source}: server says "
                               f"{got}, settings_rev {rev} (ours {self._rev})")
        deadline = time.monotonic() + 3.0
        while True:
            st = self._stream.status(refresh=True)
            if st.state == "streaming":
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"[{self.label}] {self.serial} did not start its "
                                   f"triggered stream: server state {st.state!r}")
            time.sleep(0.1)
        self._instance = st.instance or None
        self._after = int((st.raw or {}).get("seq") or 0)     # earlier frames are not ours
        self._last_hw_idx = self._last_seq = self._last_acq_gen = None

    # ---- worker thread ---------------------------------------------------

    def _work(self):
        poll = TRIGGERED_POLL_FAST_S if self._frame_settings else TRIGGERED_POLL_S
        while not self._stop.is_set():
            frame = None
            try:
                self._check_instance()
                self._drain_announcements()
                self._apply_frame_settings()
                frame = self._stream.latest(after_seq=self._after, timeout=poll,
                                            sources=("live",))
            except Exception as e:
                self._note("poll_errors", error=repr(e))
                self._stop.wait(0.2)
            t_recv = time.monotonic()
            try:
                self._drain_announcements()
                if frame is not None:
                    self._after = max(self._after, int(frame.seq))
                    self._on_frame(frame, t_recv)
                self._expire(time.monotonic())
            except Exception as e:
                self._note("worker_errors", error=repr(e))

    def _check_instance(self):
        """A restarted server numbers frames afresh and forgot our settings."""
        inst = getattr(self._stream.status(refresh=False), "instance", None)
        if not inst or self._instance is None or inst == self._instance:
            return
        self._note("server_restarted", old=self._instance, new=inst)
        self._instance = inst
        self._after = 0
        self._enter_triggered()

    def _drain_announcements(self):
        while True:
            try:
                entry = self._queue.get_nowait()
            except queue.Empty:
                return
            if entry is not None:           # None is the stop sentinel
                bisect.insort(self._pending, entry, key=lambda e: e[0])

    def _pop_pending(self, i=0):
        entry = self._pending.pop(i)
        if entry[2] >= 0:
            self._last_frame = entry[2]
        if (self._frame_settings and entry[2] >= 0
                and entry not in self._settings_done):
            target = self._frame_settings.get(entry[2], self._base_settings)
            if target and self._settings_differ(target):
                self._note("frame_settings_late", idx=list(entry[1]), frame=entry[2])
        self._settings_done.discard(entry)
        return entry

    def _expire(self, now):
        while self._pending and now - self._pending[0][0] > self._t_match_window:
            self._miss(self._pop_pending(), "no_frame")

    def _miss(self, entry, reason, **fields):
        """An announced edge got no usable frame."""
        t_expected, idx, j = entry[:3]
        if j < 0:                           # a sibling's edge: nothing of ours lost
            self._note("sibling_frame_" + reason, idx=list(idx), **fields)
        else:
            self._fail(idx, reason, frame=j, **fields)

    def _settings_differ(self, target):
        return self._cur_settings is None or any(
            _differs(self._cur_settings.get(k), v) for k, v in target.items())

    def _apply_frame_settings(self):
        """Keep the camera at the settings of the frame that comes next.

        With no edge pending, the likely next frame's settings go in now,
        while there is time: frame 0 before anything was received, then the
        frame that followed the last one the previous time round (the listed
        order until the run has shown its own). With an edge pending, that
        frame's own settings go in once it is the earliest pending one (so no
        earlier frame can carry the new revision) and FRAME_SETTINGS_LEAD_S
        remains; later than that the frame is taken as the camera stands and
        flagged."""
        if not self._frame_settings:
            return
        if not self._pending:
            last = self._last_frame
            j = 0 if last is None else self._next_after.get(last, (last + 1) % len(self.keys))
            target = self._frame_settings.get(j, self._base_settings)
            if target and self._settings_differ(target):
                self._set_now(target, frame=j)
            return
        entry = self._pending[0]
        t_expected, idx, j = entry[:3]
        if j < 0 or entry in self._settings_done:
            return
        target = self._frame_settings.get(j, self._base_settings)
        if target and self._settings_differ(target):
            late = time.monotonic() - (t_expected - FRAME_SETTINGS_LEAD_S)
            if late > 0:
                self._note("frame_settings_late", idx=list(idx), frame=j,
                           late_s=round(late, 3))
            else:
                self._set_now(target, frame=j, idx=list(idx))
        self._settings_done.add(entry)

    def _set_now(self, target, **fields):
        try:
            rb = self._stream.set(**target)
            if rb.get("errors"):
                raise RuntimeError(str(rb["errors"]))
            self._rev = rb["settings_rev"]
            applied = {k: v for k, v in _values(rb.get("readback")).items()
                       if k in FRAME_SETTING_KEYS}
            self._cur_settings = {**target, **applied}
            self._note("frame_settings_applied", **fields)
        except Exception as e:
            self._cur_settings = None
            self._note("frame_settings_failed", error=repr(e), **fields)

    def _gap(self, frame):
        """Frames the camera produced since the previous one that we never
        received (0 = contiguous). Uses the camera's own counter, the server
        seq when there is none."""
        hw_idx = getattr(frame, "hw_idx", None)
        acq_gen = getattr(frame, "acq_gen", None)
        gap = 0
        if self._last_acq_gen is not None and acq_gen != self._last_acq_gen:
            self._note("acquisition_restarted", acq_gen=acq_gen)
        elif hw_idx is not None and self._last_hw_idx is not None:
            gap = int(hw_idx) - self._last_hw_idx - 1
        elif hw_idx is None and self._last_seq is not None:
            gap = int(frame.seq) - self._last_seq - 1
        self._last_acq_gen = acq_gen
        self._last_hw_idx = None if hw_idx is None else int(hw_idx)
        self._last_seq = int(frame.seq)
        return gap

    def _candidate(self, t_recv):
        """The announced edge this frame answers, or None for a stray frame
        (after waiting t_stray_grace for an announcement on its way)."""
        deadline = time.monotonic() + self._t_stray_grace
        while True:
            self._drain_announcements()
            self._expire(t_recv)
            if self._pending:
                if self._pending[0][0] - t_recv <= self._t_early_tolerance:
                    return self._pop_pending()
                return None                 # the next edge is still ahead
            if time.monotonic() >= deadline or self._stop.is_set():
                return None
            try:
                entry = self._queue.get(timeout=0.02)
                if entry is not None:
                    bisect.insort(self._pending, entry, key=lambda e: e[0])
            except queue.Empty:
                pass

    def _on_frame(self, frame, t_recv):
        settings = getattr(frame, "settings", None) or {}
        rev = getattr(frame, "settings_rev", None)
        gap = self._gap(frame)              # every received frame moves the baseline
        if (str(settings.get("trigger_mode", "")) != "On"
                or (self._rev is not None and rev is not None and rev < self._rev)):
            self._note("stale_frames", seq=int(frame.seq))       # from before trigger mode
            return
        if gap != 0:
            self._note("gaps", missed=gap, seq=int(frame.seq))
            self._drain_announcements()
            while self._pending and self._pending[0][0] - t_recv <= self._t_early_tolerance:
                self._miss(self._pop_pending(), "frame_gap")
            return
        entry = self._candidate(t_recv)
        if entry is None:
            self._note("stray_frames", seq=int(frame.seq))
            return
        t_expected, idx, j, gen = entry
        if j < 0:
            self._note("sibling_frames_discarded")
            return
        if self._rev is not None and rev != self._rev:
            self._fail(idx, "settings_changed", frame=j, expected=self._rev, actual=rev)
            self._cur_settings = None
            try:
                self._enter_triggered()
            except Exception as e:
                self._note("reapply_failed", error=repr(e))
            return
        img = self._reduce(frame.image)
        if not self._check_shape(idx, img, frame=j):
            return
        slot = self._slots[j]
        # _slot_ok[(idx, j)] turns True when liveOD has the frame (_on_pushed)
        self._store(slot.dc, slot.dc_meta, idx, img, frame, t_expected,
                    slot=(idx, j), gen=gen, frame_index=j)

    def _clear_slot_locked(self, idx, j=None):
        """Start frame ``j`` (every frame when None) of a scan point over as
        "no frame" (caller holds the lock): a clear goes to the run's file
        behind anything already queued for it."""
        idx = tuple(idx)
        for jj in (range(len(self._slots)) if j is None else (j,)):
            self._queue_clear(idx, [self._slots[jj].dc, self._slots[jj].dc_meta],
                              frame=jj)

    # ---- end of run ------------------------------------------------------

    def _inflight(self):
        return (not self._queue.empty()) or bool(self._pending)

    def _frame_counts(self):
        """Per key: slots announced and slots whose frame is in the run's
        file (0 announced = that stage did not run this time)."""
        with self._lock:
            requested = {}
            for (idx, j) in self._req_count:
                requested[j] = requested.get(j, 0) + 1
            ok = {}
            for (idx, j), good in self._slot_ok.items():
                if good:
                    ok[j] = ok.get(j, 0) + 1
        return {k: {"requested": requested.get(j, 0), "ok": ok.get(j, 0)}
                for j, k in enumerate(self.keys)}

    def _provenance_extra(self):
        return {"mode": "triggered",
                "keys": list(self.keys),
                "frames": self._frame_counts(),
                "trigger_ttl": getattr(self.ttl, "key", "") or getattr(self.ttl, "name", ""),
                "trigger_ttl_ch": getattr(self.ttl, "ch", None),
                "trigger_source": self._trigger_source,
                "exposure_delay": self.exposure_delay,
                "t_trigger": self.t_trigger,
                "siblings_on_line": [list(s.keys) for s in self._siblings],
                "frame_settings": {self.keys[j]: v for j, v in sorted(self._frame_settings.items())},
                "base_settings": dict(self._base_settings),
                "t_match_window": self._t_match_window,
                "t_early_tolerance": self._t_early_tolerance}

    def _write_provenance(self):
        record = self._provenance_record()
        for i, k in enumerate(self.keys):
            try:
                self._expt._extra_file_texts[f"camera_stream_{k}"] = \
                    json.dumps({**record, "key": k, "frame": i}, default=repr)
            except Exception as e:
                print(f"[{self.label}] WARNING: could not store provenance for "
                      f"'{k}' ({e!r})")

    def _report(self):
        c = self.counts
        if c.get("wrong_mode_calls"):
            print(f"[{self.label}] !! {c['wrong_mode_calls']} call(s) to the other "
                  f"mode's kernel method (grab vs trigger) were ignored")
        keys = ", ".join(f"'{k}'" for k in self.keys)
        self._report_clears()
        missing = self._n_missing()
        if missing or c["rpc_errors"] or c.get("gaps"):
            lost = self._queue_losses()
            print(f"[{self.label}] !! {missing} of {self._n_slots()} requested frames "
                  f"are MISSING in {keys} (in the file {c['ok']}, failed {c['failed']}, "
                  f"counter gaps {c.get('gaps', 0)}, stray {c.get('stray_frames', 0)}, "
                  f"duplicates {c['duplicate']}, rpc errors {c['rpc_errors']}"
                  + (f"; not delivered to liveOD: {lost}" if lost else "")
                  + "); their <key>_meta seq is NaN")
        else:
            discarded = c.get("sibling_frames_discarded", 0)
            print(f"[{self.label}] {c['ok']}/{c['requested']} frames captured into {keys}"
                  + (f" ({discarded} sibling-edge frames discarded)" if discarded else ""))

    def _before_restore(self):
        if self._live_started:
            self._live_started = False
            self._stream.stop_live()


class InactiveCameraStream:
    """A stream that takes nothing (diagnostics off): kernel no-ops, no
    containers, nothing registered."""

    def __init__(self, expt, key, reason=""):
        self.core = expt.core
        self.keys = _as_keys(key)
        self.key = self.keys[0]
        self.label = f"camera_stream:{self.key}"
        self.reason = str(reason)

    @kernel
    def grab(self, t_offset=0.):
        pass

    @kernel
    def trigger(self, j=0):
        pass

    def finish(self, timeout_s=None):
        pass

    def close(self):
        pass


class DummyCameraStreamClient:
    """A placeholder for a camera that cannot be used this run (it is the
    run's own camera, shares its trigger line, or could not be set up): the
    same kernel surface and containers, but one byte per shot instead of a
    frame and a NaN record everywhere. The reason is in the provenance."""

    def __init__(self, expt, key, reason="", triggered=False):
        self._expt = expt
        self.core = expt.core
        self.keys = _as_keys(key)
        self.key = self.keys[0]
        self.label = f"camera_stream:{self.key}"
        self.reason = str(reason)
        self.triggered = bool(triggered)
        self._finished = False
        d = expt.data
        for k in self.keys:
            CameraStreamClient._check_key_available(expt, k)
        for k in self.keys:
            setattr(d, k, d.add_host_data_container((1,), np.uint8, fill_value=0))
            setattr(d, k + "_meta", d.add_host_data_container((N_META,), np.float64,
                                                              fill_value=np.nan))
        self.dc = getattr(d, self.key)
        self.dc_meta = getattr(d, self.key + "_meta")
        if not hasattr(expt, "camera_streams"):
            expt.camera_streams = []
        expt.camera_streams.append(self)
        keys = ", ".join(f"'{k}'" for k in self.keys)
        print(f"[{self.label}] note: no {keys} frames this run ({self.reason})")

    @kernel
    def grab(self, t_offset=0.):
        pass

    @kernel
    def trigger(self, j=0):
        pass

    def finish(self, timeout_s=None):
        if self._finished:
            return
        self._finished = True
        for k in self.keys:
            try:
                self._expt._extra_file_texts[f"camera_stream_{k}"] = json.dumps(
                    {"key": k, "keys": list(self.keys), "dummy": True,
                     "triggered": self.triggered, "reason": self.reason,
                     "note": f"'{k}' is a placeholder (one byte per shot), not a frame"})
            except Exception:
                pass

    def close(self):
        self.finish()
