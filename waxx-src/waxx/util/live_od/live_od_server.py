"""
ZMQ REP server running inside the liveOD GUI process.

Receives INIT_RUN / WAIT_CAM_READY / SHOT_COMPLETE / END_RUN messages
from the experiment client (possibly on a different machine) and
coordinates HDF5 file creation, camera spawning, and final data saving.

The server runs as a QThread to stay compatible with PyQt6 event dispatch
on the GUI machine. All HDF5 I/O stays on the server side; the experiment
client never needs the data drive mounted.
"""

import collections
import json
import logging
import pickle
import os
import threading
import time
import uuid

from waxx.config.timeouts import DATA_SAVER_TIMEOUT

import zmq
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from beacon.discovery.server import NetServer
from waxx.util.comms_server.hardware_id import scoped_server_id
from waxx.util.live_od.config import get_config
# Everything the server does to the run's data file (reserve, save, delete) is
# in live_od/data/run_file.py; this module keeps the protocol and the run state.
from waxx.util.live_od.data.image_writer import PutItem, writer_queue_stats
from waxx.util.live_od.data.run_file import RunFile, RunFileSaveError
from waxx.util.live_od.log import get_logger, get_log_buffer
from waxx.util.live_od.marker_store import MarkerStore
from waxx.util.live_od import frame_alignment

logger = get_logger("server")

# An Abort the experiment has not answered after this long is shown as "no_reply".
# The experiment checks for an abort at the end of each shot, so its answer is due
# within about one shot period (plus its cleanup): a few of the run's longest
# recent periods, never less than the minimum.
ABORT_REPLY_MIN_S = 30.0
ABORT_REPLY_SHOT_FACTOR = 3.0
# A run whose experiment exited with camera frames still due (RUN_EXITED) is closed
# once no frame has come for this long: triggers the core device had already queued
# can still bring frames after the process is gone, nothing later can.
EXITED_FRAME_GRACE_S = 5.0
# INIT_RUN waits this long for the previous run's asynchronous save before it
# refuses (the client's INIT_RUN timeout is 60 s)
SAVE_WAIT_BEFORE_INIT_S = 45.0
# A Reset pressed this long after an Abort the experiment has not answered closes
# the aborted run at once, as the next run start would (a second press, not the
# window's own echo of the first).
ABORT_AGAIN_MIN_S = 3.0


def _safe_repr(value) -> str:
    """``repr(value)``, or a stand-in when even that raises."""
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}: repr failed>"


def _client_of(msg: dict) -> dict:
    """The run's client from an INIT_RUN payload: ``client_pid`` (int or None),
    ``client_host``, ``launcher`` and ``queue_job`` (the run queue's job,
    "<id>:<token>" as the experiment's WAXX_QUEUE_JOB has it; "" when absent). A client that predates them sends none; a malformed
    value is dropped, never raised on."""
    pid = msg.get("client_pid")
    try:
        pid = int(pid) if pid is not None else None
    except (TypeError, ValueError):
        pid = None
    return {"client_pid": pid, "client_host": str(msg.get("client_host") or ""),
            "launcher": str(msg.get("launcher") or ""),
            "queue_job": str(msg.get("queue_job") or "")}


class LiveODServer(QThread, NetServer):
    """ZMQ REP server embedded in the liveOD process.

    Listens for messages from the experiment client and drives file I/O
    and camera management.

    Signals
    -------
    new_run_signal(filepath, camera_key, capture_images, save_data, imaging_type, n_img, n_shots, n_pwa_per_shot)
        Emitted after INIT_RUN.  ``filepath`` is ``""`` when
        ``save_data=False``.  ``capture_images`` indicates whether a
        CameraBaby should be spawned.  ``n_img``, ``n_shots``, and
        ``n_pwa_per_shot`` carry the image-count params from the payload so
        ``DataHandler`` can be initialised correctly even when
        ``save_data=False`` (and no HDF5 file is read).
    shot_progress_signal(shot_idx, N_shots_total, xvar_values_dict)
        Emitted for each SHOT_COMPLETE message.
    run_done_signal()
        Emitted after END_RUN is fully handled.
    run_state_signal(state, detail)
        Where the run is, for the GUI's status strip: ``waiting_camera``,
        ``waiting_grab_drain``, ``running``, ``saving``, ``saved``, ``done``
        (finished, nothing to save), ``aborting``, ``aborted``, ``error``.
        Emitted on change only.
    """

    new_run_signal = pyqtSignal(str, str, bool, bool, int, int, int, int, object, object, object)   # filepath, camera_key, capture_images, save_data, imaging_type, n_img, n_shots, n_pwa_per_shot, camera_params, params_payload, run_info_payload
    shot_progress_signal = pyqtSignal(int, int, object)       # shot_idx, N_total, xvar_values dict
    shot_conditions_signal = pyqtSignal(int, object)          # shot_idx, {key: float} recorded by the shot
    shot_timing_signal = pyqtSignal(float, str)               # delta_t_s, eta_str ("HH:MM" or "--:--")
    run_done_signal = pyqtSignal()
    run_started_signal = pyqtSignal(int, object)               # run_id, xvarnames (emitted after INIT_RUN, before new_run_signal)
    reset_signal = pyqtSignal()                               # triggered by remote RESET command
    camera_control_signal = pyqtSignal(str, str)              # camera_key, action ('open'|'close'|'toggle')
    adjust_specs_signal = pyqtSignal(list)                    # list of spec dicts, emitted after every INIT_RUN (empty list when no adjust params)
    shot_adjust_values_signal = pyqtSignal(dict)              # current adjust values dict, emitted per shot
    run_state_signal = pyqtSignal(str, str)                   # state, detail (see class docstring)
    markers_changed_signal = pyqtSignal(str, list)            # camera_key, markers (a remote viewer edited them)
    exited_run_signal = pyqtSignal(str)                       # how an exited run was closed: stop its camera thread, file untouched

    def __init__(self, server_talk, data_saver, port: int = 0, marker_path=None):
        super().__init__()  # QThread.__init__
        NetServer.__init__(self, scoped_server_id("live_od"), port)  # explicit — avoids MRO conflict
        self._server_talk = server_talk
        self._run_file = RunFile(data_saver)
        self._ip = "0.0.0.0"
        self._port = port
        self._cam_ready_event = threading.Event()
        self._running = False
        self._current_capture_images = False
        self._current_run_id = 0
        self._current_camera_key = ""
        # GUI-supplied callable -> {camera_key: {"state", "camera_type", "serial_no"}}
        # for POLL, so a remote tool can see which cameras liveOD holds (and
        # whether a release it asked for has happened).  Plain attribute reads
        # on the GUI's objects; it is called from this thread.
        self._camera_state_provider = None
        self._reset_requested = False   # set by RESET; cleared by next INIT_RUN
        # when the current run's Abort was requested (time.time()); None when no
        # abort is waiting for the experiment's answer (_check_abort_reply)
        self._abort_requested_at = None
        # A Reset pressed again on that unanswered abort: the server thread closes
        # the run (_check_abort_again)
        self._abort_again = False
        # The run's experiment exited (RUN_EXITED) with frames still due: {"why",
        # "t" (time.monotonic())} until _check_exited_run closes it; None otherwise
        self._exited_pending = None
        # why a Reset asked for that run to be closed now ("" when none did)
        self._exited_close_now = ""
        self._run_in_progress = False   # True between INIT_RUN and END_RUN/ABORT
        self._shot_timestamps: list = []  # Unix timestamps (s) recorded server-side on each SHOT_COMPLETE
        self._scalar_subscriber_count: dict = {}  # tier -> subscriber count
        self._scalar_lock = threading.Lock()
        self._adjust_specs: list = []    # list of spec dicts from INIT_RUN
        self._adjust_values: dict = {}   # key -> live value (written by GUI or remote viewers)
        self._adjust_lock = threading.Lock()
        # Basler-specific: track whether every grab loop older than the current
        # run's has fully exited.  Cleared on a Basler INIT_RUN that starts while
        # an earlier baby is still live, set again once only the current baby
        # remains (see on_basler_baby_done).  A plain "a baby is active" flag is
        # not enough: a stale baby's done_signal would clear it and open the gate
        # for a run whose own baby had not started grabbing yet.
        self._basler_prev_grab_done_event = threading.Event()
        self._basler_prev_grab_done_event.set()  # no previous grab initially
        self._basler_lock = threading.Lock()
        self._basler_babies_live = 0
        self._init_run_time: float = 0.0  # time.time() recorded at INIT_RUN
        self._shot_durations: list = []  # rolling list of last 5 shot durations (excluding first shot)
        self._run_state = "idle"
        self._current_expt_name = ""    # the run's experiment file (class name from an older client)
        # The run's client, from INIT_RUN: its process id and host, and who
        # launched it (WAXX_LAUNCHER: "run_lock", "run_loop", ...; "" for a
        # person). None/"" from a client that predates them. A launcher's gate
        # (waxx.util.device_state.run_gate) uses them to tell a run whose
        # process is gone from a live one.
        self._current_client = {"client_pid": None, "client_host": "", "launcher": "",
                                "queue_job": ""}
        # every Abort set (request_reset), for POLL's reset_count / last_reset
        self._reset_count = 0
        self._reset_counts = {"person": 0, "queue": 0, "agent": 0, "liveod": 0}
        self._last_reset = None
        # the run a Reset during its save was last warned about (one WARNING each)
        self._reset_during_save_warned = None
        self._current_n_shots = 0
        # Frames: how many the run asked for (N_img, 0 for a no-camera run) and
        # how many the DataHandler has taken off the camera queue so far
        # (on_image_received, from the DataHandler thread).  END_RUN compares
        # the two; a shortfall marks the file incomplete instead of complete.
        self._images_expected = 0
        self._images_received = 0
        self._images_lock = threading.Lock()
        self._grab_failure = ""         # the CameraBaby's reason, when its grab ended early
        self._frame_deficit_warned = 0  # the shortfall already warned about this run
        self._last_outcome = {}         # how the last run ended, for POLL / GET_LOG
        # The server's name for the current run: issued at INIT_RUN, sent back by
        # the client with every message about the run. A message carrying another
        # token belongs to a run that a later INIT_RUN superseded, and is ignored
        # (see _run_msg_ok). "" until the first INIT_RUN.
        self._run_token = ""
        # Every token this process has issued (bounded): a message whose token is
        # not among them is from a run before a liveOD restart, not a superseded one.
        self._issued_run_tokens = collections.deque(maxlen=512)
        # (report, token) of the camera-thread reports of replaced runs already
        # warned about (_report_is_current): one WARNING each, not one per frame
        self._stale_reports_warned = set()
        # Camera settings the current run got that differ from its camera_params
        # (clamps reported by the camera thread): {field: {requested, applied,
        # origin}}. Recorded in the run file as the root attribute camera_overrides.
        self._camera_overrides: dict = {}
        self._camera_overrides_key = ""
        self._camera_overrides_lock = threading.Lock()
        # For END_RUN's frame-alignment check, all on time.monotonic(): when each
        # frame came off the camera queue (under _images_lock), when each
        # SHOT_COMPLETE arrived, and when the experiment was first told the
        # camera was ready.
        self._frame_times: list = []
        self._shot_mono: list = []
        self._t_ready_mono = None
        # pins on the image, per camera; the file is not touched until first used
        self.markers = MarkerStore(marker_path)
        # The camera host (LiveODConfig.use_camera_host; the window hands it in with
        # set_camera_host). None: cameras through CameraNanny, as before, and none of
        # the host state below is used.
        self._camera_host = None
        self._host_run_token = ""       # the run whose camera lock the host holds
        self._host_run_start = None     # what begin_run found (profile, persist overrides)
        self._host_arm_future = None    # started by the run's first WAIT_CAM_READY
        self._host_camera_params = {}
        self._host_n_img = 0
        # the token of each camera run whose camera thread the window has not yet
        # spawned (new_run_signal is queued; a second INIT_RUN may come first).
        # Bounded: a server without a window (tests, tools) takes none of them.
        self._spawn_tokens = collections.deque(maxlen=64)
        # host mode: (token, RunStart, camera_params, n_img) of an INIT_RUN whose
        # camera is locked but which is not accepted yet (_host_commit_run)
        self._host_pending = None
        # when Persist was turned on, for the overrides record of the current run
        self._camera_overrides_persist_since = None
        # Arrays the experiment pushed during the run (PUT_DATA): the latest
        # record per key, for GET_AUX_DATA; reset at INIT_RUN.
        self._aux_latest = {}
        # Called on this thread with a small notice per PUT_DATA (what arrived,
        # no array); the window sets it to the broadcaster's thread-safe
        # broadcast_aux_notice (set_aux_notice_sink). None: no notices.
        self._aux_notice_sink = None

    def set_aux_notice_sink(self, sink):
        """``sink(notice)`` is called on the server thread for every PUT_DATA,
        with ``{run_id, t, items: [{key, index, offset, shape, dtype, nbytes}]}``
        -- no arrays: a viewer that wants them asks GET_AUX_DATA. It must be
        quick and thread-safe (LiveODBroadcaster.broadcast_aux_notice only
        queues the message)."""
        self._aux_notice_sink = sink

    def set_markers(self, camera_key: str, markers) -> list:
        """Store one camera's markers (from the GUI; remote viewers use SET_MARKERS)."""
        return self.markers.set(camera_key, markers)

    def _set_run_state(self, state: str, detail: str = ""):
        """Emit run_state_signal if the state changed. WAIT_CAM_READY arrives in
        short slices, so the same state is reported many times over."""
        if state == self._run_state and not detail:
            return
        self._run_state = state
        self.run_state_signal.emit(state, detail)

    # The server's old file-handling attributes, read-only, for anything that
    # still looks at them. The state itself is RunFile's.
    _current_filepath = property(lambda self: self._run_file.filepath)
    _current_save_data = property(lambda self: self._run_file.save_data)
    _data_handler_done_event = property(lambda self: self._run_file.writer_done)
    _data_saver = property(lambda self: self._run_file._data_saver)

    # ------------------------------------------------------------------
    # Public slots (safe to call from any thread)
    # ------------------------------------------------------------------

    def on_cam_ready(self, run_token=None):
        """Set camera-ready event.

        Connect this to ``CameraBaby.cam_status_signal`` filtered to
        status == 2 using ``Qt.ConnectionType.DirectConnection`` so the
        threading.Event is set from the CameraBaby thread immediately.
        ``run_token``: the run the camera thread works for (_report_is_current).
        """
        with self._images_lock:
            if self._report_is_current("camera ready (status 2)", run_token):
                self._cam_ready_event.set()

    def on_basler_baby_done(self):
        """Called when a Basler CameraBaby's thread has fully finished.

        Connect to ``CameraBaby.done_signal`` with
        ``Qt.ConnectionType.DirectConnection`` for Basler cameras.
        The gate is released only once the live-baby count is down to the
        current run's own baby, so a stale baby from an aborted run cannot
        report the camera free while it is still inside RetrieveResult().
        """
        with self._basler_lock:
            self._basler_babies_live = max(0, self._basler_babies_live - 1)
            remaining = self._basler_babies_live
            if remaining <= 1:
                self._basler_prev_grab_done_event.set()
        logger.info(f"Basler grab loop exited ({remaining} still live).")

    def _release_basler_slot(self):
        """Give back a Basler live-baby slot claimed by INIT_RUN when no
        CameraBaby ends up being spawned for that run."""
        with self._basler_lock:
            self._basler_babies_live = max(0, self._basler_babies_live - 1)
            if self._basler_babies_live <= 1:
                self._basler_prev_grab_done_event.set()

    def on_data_handler_done(self, run_token=None):
        """Set the data-handler-done event.

        Connect to ``DataHandler.done_writing_signal`` with
        ``Qt.ConnectionType.DirectConnection`` so the event is set from
        the DataHandler thread immediately after it closes the HDF5 file.
        ``run_token``: the run the image writer worked for; a replaced run's
        writer must not open the new run's END_RUN save (_report_is_current).
        """
        with self._images_lock:
            if self._report_is_current("an image writer's end", run_token):
                self._run_file.writer_finished()

    def wait_for_image_writer(self, timeout: float) -> bool:
        """True once the current run's writer has closed the run's file (at
        once when there is none). For the window's shutdown: the writer is
        told to finish first, with what it has (the run is over for liveOD)."""
        try:
            self._run_file.finish_writer()
        except Exception:
            logger.exception("shutdown: finishing the run's writer failed")
        return self._run_file.writer_done.wait(timeout)

    def on_image_received(self, *_, run_token=None):
        """One frame came off the camera queue. Connect to
        ``DataHandler.got_image_from_queue`` with a DirectConnection.
        ``run_token``: the run the image dispatcher works for (_report_is_current)."""
        with self._images_lock:
            if not self._report_is_current("a frame", run_token):
                return
            self._images_received += 1
            self._frame_times.append(time.monotonic())

    def on_grab_failed(self, reason: str, run_token=None):
        """The CameraBaby's grab ended early (timeout, camera error). Connect to
        ``CameraBaby.grab_failed_signal`` with a DirectConnection. END_RUN puts
        the reason in the file. ``run_token``: see _report_is_current."""
        with self._images_lock:
            if not self._report_is_current("a grab failure", run_token):
                return
            self._grab_failure = str(reason)
            get_log_buffer().update_run(grab_failure=self._grab_failure)

    def _images_received_now(self) -> int:
        with self._images_lock:
            return self._images_received

    # ------------------------------------------------------------------
    # Run identity
    # ------------------------------------------------------------------

    def _adopt_run_token(self, token: str):
        """The INIT_RUN is accepted: ``token`` names the current run from now on.
        Replaced under _images_lock, the lock the camera threads' reports are
        checked under (_report_is_current)."""
        with self._images_lock:
            self._run_token = token
            self._issued_run_tokens.append(token)
            self._stale_reports_warned.clear()

    def _report_is_current(self, what: str, run_token) -> bool:
        """Is a camera thread's report about the current run? ``run_token`` is
        the token of the run the thread was spawned for (LiveODWindow.spawn_baby
        stamps its threads with it); None is a caller that does not stamp, taken
        as before.

        Call with _images_lock held. INIT_RUN replaces the token under that lock
        and resets what these reports set only after it, so a report either
        lands before the new run's reset (which clears it) or is checked against
        the new token and dropped. Without the check, a thread of the previous
        run still got through between INIT_RUN (this thread) and the window's
        spawn_baby (the GUI thread; new_run_signal is queued): its status 2
        marked the new run's camera ready, its grab failure marked the new run
        incomplete, its frames counted as the new run's."""
        if run_token is None or str(run_token) == self._run_token:
            return True
        key = (what, str(run_token))
        if key not in self._stale_reports_warned:
            self._stale_reports_warned.add(key)
            logger.warning(f"Ignored {what} from a camera thread of an earlier run (token "
                           f"{str(run_token)[:8]}...): the current run's token is "
                           f"{self._run_token[:8]}...")
        return False

    def _run_msg_ok(self, msg: dict) -> bool:
        """Is this run message about the current run? A message without a
        token comes from experiment code that predates tokens and is taken as
        before; one with another run's token is not."""
        token = msg.get("run_token")
        return not token or str(token) == self._run_token

    def _stale_run_reply(self, tag: str, msg: dict) -> dict:
        """The reply to a message about a run that is not the current one: a run
        a later INIT_RUN superseded, or one this liveOD never issued a token for
        (``unknown_run``: liveOD was restarted after that run's INIT_RUN).
        Nothing is changed: no save, no delete, no run state."""
        token = str(msg.get("run_token"))
        if token in self._issued_run_tokens:
            logger.warning(f"{tag} carries the token of a superseded run ({token[:8]}...); the "
                           f"current run is {self._current_run_id} ({self._run_token[:8]}...). "
                           f"Ignored.")
            return {"ok": False, "stale_run": True,
                    "error": f"{tag} for superseded run (token {token}); ignored"}
        logger.warning(f"{tag} carries a run token unknown to this liveOD ({token[:8]}...): "
                       f"was liveOD restarted after that run's INIT_RUN? Ignored.")
        return {"ok": False, "stale_run": True, "unknown_run": True,
                "error": f"{tag} for a run unknown to this liveOD (restarted?) "
                         f"(token {token}); ignored"}

    # ------------------------------------------------------------------
    # Camera settings that differ from the request
    # ------------------------------------------------------------------

    @staticmethod
    def _plain(value):
        """A JSON-able copy of a driver value (numpy scalars, tuples)."""
        if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            return [LiveODServer._plain(v) for v in value]
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return value

    def on_camera_overrides(self, camera_key: str, clamps: dict, run_token=None):
        """The run's camera applied values other than those requested
        (``{field: (requested, applied)}``, from the driver's clamps). Connect to
        ``CameraBaby.camera_overrides_signal`` with a DirectConnection. Logged
        now, one WARNING per field; recorded in the file at END_RUN.
        ``run_token``: see _report_is_current."""
        fields = {}
        for key, pair in dict(clamps or {}).items():
            try:
                requested, applied = pair
            except (TypeError, ValueError):
                logger.warning(f"{camera_key}: unreadable clamp report for {key}: {pair!r}")
                continue
            fields[str(key)] = {"requested": self._plain(requested),
                                "applied": self._plain(applied), "origin": "clamped"}
        if not fields:
            return
        with self._images_lock:
            if not self._report_is_current("a camera settings report", run_token):
                return
            with self._camera_overrides_lock:
                self._camera_overrides.update(fields)
                self._camera_overrides_key = str(camera_key)
        for key, f in fields.items():
            logger.warning(f"CAMERA OVERRIDE: run {self._current_run_id} on {camera_key}: "
                           f"{key} requested {f['requested']!r} -> applied {f['applied']!r} "
                           f"({f['origin']}); camera_params in the file stay the request")

    def camera_overrides_record(self) -> dict:
        """The current run's overrides record (INTERFACES section 4), or {} if
        every applied value is the requested one. Plain JSON values only: a value
        JSON cannot hold (an array, a driver's object, an odd persisted value) is
        its repr, so the record can always be written and sent."""
        with self._camera_overrides_lock:
            if not self._camera_overrides:
                return {}
            record = {"schema": 1, "camera_key": self._camera_overrides_key,
                      "persist_since": self._camera_overrides_persist_since,
                      "fields": {k: dict(v) for k, v in self._camera_overrides.items()}}
        return self._json_safe_record(record)

    @staticmethod
    def _json_safe_record(record: dict) -> dict:
        """``record`` with every value JSON cannot hold replaced by its repr. If
        even that fails (a self-reference, a key of an odd type), every field's
        requested / applied value is its repr."""
        try:
            return json.loads(json.JSONEncoder(default=_safe_repr).encode(record))
        except Exception:
            fields = {}
            for key, f in dict(record.get("fields") or {}).items():
                f = f if isinstance(f, dict) else {"applied": f}
                fields[str(key)] = {"requested": _safe_repr(f.get("requested")),
                                    "applied": _safe_repr(f.get("applied")),
                                    "origin": str(f.get("origin", "?"))}
            since = record.get("persist_since")
            return {"schema": 1, "camera_key": str(record.get("camera_key", "")),
                    "persist_since": None if since is None else str(since), "fields": fields}

    def _with_camera_overrides(self, msg: dict) -> dict:
        """The END_RUN payload, plus the root attribute ``camera_overrides`` in its
        extra file texts when the run's camera applied anything other than the
        request. The payload itself is not modified. The record never blocks the
        save: if it cannot be added, the run is saved without it and an ERROR
        says what the camera ran with."""
        try:
            return self._add_camera_overrides(msg)
        except Exception as exc:
            with self._camera_overrides_lock:
                fields = dict(self._camera_overrides)
            logger.error(f"END_RUN: run {self._current_run_id}: the camera_overrides record "
                         f"could not be added ({type(exc).__name__}: {exc}); the run is saved "
                         f"WITHOUT it, so nothing in the file says its camera_params are only "
                         f"the request. The camera ran with: {_safe_repr(fields)}")
            return msg

    def _add_camera_overrides(self, msg: dict) -> dict:
        record = self.camera_overrides_record()
        if not record:
            return msg
        texts = dict(msg.get("extra_file_texts") or {})
        if "camera_overrides" in texts:
            logger.warning("END_RUN: the experiment sent its own 'camera_overrides' text; "
                           "liveOD's record replaces it.")
        texts["camera_overrides"] = json.dumps(record)
        out = dict(msg)
        out["extra_file_texts"] = texts
        return out

    # ------------------------------------------------------------------
    # Who may open / close a camera now
    # ------------------------------------------------------------------

    def camera_action_allowed(self, camera_key: str, action: str):
        """``(ok, reason)``: may ``camera_key`` be opened / closed / toggled now?
        The one rule for the remote CAMERA_CONTROL and the window's own camera
        button.

        During a run: closing the camera a CameraBaby is driving crashes its
        grab loop (dishonorable_death), and opening any camera blocks the GUI
        thread (Andor cooler init takes seconds) and stalls the queued
        SHOT_COMPLETE / RESET signals -- both refused.  Closing a camera the
        run is *not* using is allowed: that is how a tool borrows an idle
        Basler (frame grab through the beacon server) while a run on another
        camera goes on.  A no-camera run uses none of them.
        """
        if action not in ("open", "close", "toggle"):
            return False, f"Unknown camera action: {action}"
        if not camera_key:
            return False, "Missing camera_key"
        if not self._run_in_progress:
            return True, ""
        run_cam = self._current_camera_key if self._current_capture_images else ""
        if action == "close" and camera_key != run_cam:
            return True, ""
        return False, (f"Camera control rejected: run {self._current_run_id} in "
                       f"progress (uses {run_cam or 'no camera'}); only closing a "
                       f"camera the run does not use is allowed")

    # ------------------------------------------------------------------
    # Frame alignment
    # ------------------------------------------------------------------

    def _check_frame_alignment(self):
        """At END_RUN: do the frames sit in their shots' slots, judging by when
        they arrived? Proof that they do not (frame_alignment.assess: a frame
        that came before its shot could have been triggered) is a grab failure,
        so the run is saved incomplete. What the times cannot decide is only
        logged."""
        if self._reset_requested or not self._images_expected:
            return
        n_shots = int(self._current_n_shots or 0)
        if n_shots < 1 or self._images_expected % n_shots:
            logger.info(f"Frame alignment not checked: {self._images_expected} images "
                        f"do not divide into {n_shots} shots.")
            return
        per_shot = self._images_expected // n_shots
        with self._images_lock:
            frame_t = list(self._frame_times)
        # A camera whose readout outlasts the SHOT_COMPLETE RPC (the Andor) also has
        # each shot's last slot checked (frame_alignment.readout_outlasts_rpc). Its
        # own try: a camera table that cannot say only turns that part off.
        try:
            slow_readout = frame_alignment.readout_outlasts_rpc(
                get_config().resolve_camera_params(self._current_camera_key))
        except Exception as exc:
            logger.warning(f"Frame alignment: could not look up camera "
                           f"{self._current_camera_key!r} ({exc}); its last-slot check is off.")
            slow_readout = False
        try:
            result = frame_alignment.assess(frame_t, list(self._shot_mono), per_shot,
                                            t_start=self._t_ready_mono,
                                            dark_after_shot_complete=slow_readout)
        except Exception as exc:
            logger.warning(f"Frame alignment check failed ({exc}); not checked.")
            return
        for note in result.notes:
            logger.warning(f"Frame alignment (not conclusive): {note}")
        if result.issues:
            reason = "FRAME ALIGNMENT SUSPECT: " + "; ".join(result.issues)
            logger.error(f"Run {self._current_run_id}: {reason}")
            self._grab_failure = f"{self._grab_failure}; {reason}" if self._grab_failure else reason
            get_log_buffer().update_run(grab_failure=self._grab_failure)

    # ------------------------------------------------------------------
    # The camera host (LiveODConfig.use_camera_host; PLAN C3 / C5)
    # ------------------------------------------------------------------

    def set_camera_host(self, host):
        """Run cameras through ``host`` (a CameraHost) from now on: INIT_RUN locks
        the run's camera, the first WAIT_CAM_READY arms it, END_RUN / ABORT_RUN /
        a superseding INIT_RUN release it, CAMERA_CONTROL goes to the host."""
        self._camera_host = host

    @property
    def camera_host(self):
        return self._camera_host

    def _needs_grab_drain(self, camera_key: str) -> bool:
        """The Basler grab-drain gate. Not for a host camera: its worker is the
        only thread on the camera, and the gate would wait for a camera thread
        that itself waits for the arm."""
        return self._camera_host is None and get_config().camera_needs_grab_drain(camera_key)

    def take_spawn_token(self) -> str:
        """The run token for the camera thread the window is spawning now: the
        oldest camera run not yet spawned, in INIT_RUN order (the current run's
        when none is waiting: a spawn without an INIT_RUN)."""
        try:
            return self._spawn_tokens.popleft()
        except IndexError:
            return self._run_token

    def _refuse_init_run(self, error: str, state_detail: str) -> dict:
        """The reply to a refused INIT_RUN. It changes nothing of the run in
        progress, if there is one: its token (its messages still count), its
        camera, its state. The status strip shows the refusal only when no run
        is in progress."""
        if self._run_in_progress:
            logger.warning(f"The refused INIT_RUN changed nothing: run {self._current_run_id}, "
                           f"in progress, is left as it was.")
        else:
            self._set_run_state("error", state_detail)
        return {"ok": False, "error": error}

    def _host_begin_run(self, msg: dict, camera_key: str, capture_images: bool, camera_params,
                        token: str):
        """INIT_RUN in host mode, before the run file is reserved: this run's
        camera is locked with ``token``, which stays provisional until the
        INIT_RUN is accepted (_host_commit_run). Returns None, or the refusal
        reply: no run id is used, and the run in progress keeps its token and
        state (the host itself ends an older run on the same camera to lock it)."""
        host = self._camera_host
        if host is None:
            return None
        params = dict(camera_params or {})
        try:
            start = host.begin_run(token, camera_key, capture_images, camera_params=params,
                                   images_shape=msg.get('images_shape'))
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.error(f"INIT_RUN refused before a run id was reserved: camera "
                         f"{camera_key}: {reason}")
            return self._refuse_init_run(
                f"INIT_RUN refused (no run id used): camera {camera_key}: {reason}",
                f"Camera {camera_key}: {exc}")
        self._host_pending = (token, start, params, int(msg.get('params', {}).get('N_img', 1)))
        return None

    def _host_commit_run(self, token: str):
        """Host mode, the INIT_RUN that locked its camera with ``token`` is
        accepted: the previous run's hold on its camera ends (T10), and this run
        is the one the host state here is about."""
        pending, self._host_pending = self._host_pending, None
        if self._camera_host is None or pending is None or pending[0] != token:
            return
        if self._host_run_token and self._host_run_token != token:
            self._host_end_run("superseded by a new INIT_RUN", record=False)
        _, start, params, n_img = pending
        self._host_run_token = token
        self._host_run_start = start
        self._host_camera_params = params
        self._host_n_img = n_img
        self._host_arm_future = None

    def _host_abandon_run(self, token: str, reason: str):
        """Host mode, the INIT_RUN that locked its camera with ``token`` is refused
        after all (its data file was not created): that lock goes, and nothing
        else changes."""
        pending, self._host_pending = self._host_pending, None
        host = self._camera_host
        if host is None or pending is None or pending[0] != token:
            return
        try:
            host.end_run(token, reason)
        except Exception as exc:
            logger.warning(f"camera host: giving back the refused INIT_RUN's camera lock "
                           f"failed: {exc}")

    def _host_after_reserve(self, run_id: int):
        """INIT_RUN in host mode, once the run id is known: the lock is named
        after the run, and a run whose camera Persist changes is announced (one
        WARNING, on the run's log) and recorded (camera_overrides, origin
        "persist")."""
        host, start = self._camera_host, self._host_run_start
        if host is None:
            return
        self._camera_overrides_persist_since = None
        if start is None or not self._host_run_token:
            return
        host.note_run_id(self._host_run_token, run_id)
        for key, why in dict(start.refused or {}).items():
            logger.warning(f"Persisted {start.camera_key} value NOT used in run {run_id}: {why}")
        if not start.persist_on:
            return
        if not start.overrides:
            logger.info(f"Persist is on for {start.camera_key} (since {start.persist_since}), "
                        f"but every persisted value equals the run's camera_params: nothing "
                        f"overridden.")
            return
        logger.warning(start.persist_warning(run_id))
        with self._camera_overrides_lock:
            self._camera_overrides.update({k: dict(v) for k, v in start.overrides.items()})
            self._camera_overrides_key = str(start.camera_key)
            self._camera_overrides_persist_since = start.persist_since

    def _note_spawn_token(self, capture_images: bool):
        """A camera run's token, for the camera threads the window spawns for it
        (take_spawn_token): their reports carry it (_report_is_current), and in
        host mode their nanny locks with it."""
        if capture_images:
            self._spawn_tokens.append(self._run_token)

    def _host_start_arm(self):
        """WAIT_CAM_READY in host mode: the run's first one arms its camera.
        Returns the error text of an arm that failed, else None."""
        host = self._camera_host
        if host is None or not self._host_run_token:
            return None
        if self._host_arm_future is None:
            self._host_arm_future = host.arm_run(self._host_run_token, self._host_camera_params,
                                                 self._host_n_img)
        return self._host_arm_error()

    def _host_arm_error(self):
        fut = self._host_arm_future
        if fut is None or not fut.done():
            return None
        exc = fut.exception()
        if exc is None:
            return None
        return f"camera {self._current_camera_key} could not be armed: {type(exc).__name__}: {exc}"

    def _wait_cam_ready_event(self, deadline: float) -> bool:
        """Wait for the camera thread's "armed" (status 2) until ``deadline``; in
        host mode in short slices, so that an arm that failed ends the wait."""
        fut = self._host_arm_future if self._camera_host is not None else None
        if fut is None:
            return self._cam_ready_event.wait(timeout=max(0.0, deadline - time.time()))
        while True:
            left = deadline - time.time()
            if self._cam_ready_event.wait(timeout=max(0.0, min(0.02, left))):
                return True
            if fut.done() and fut.exception() is not None:
                return False
            if left <= 0:
                return False

    def _merge_camera_overrides(self, camera_key: str, fields: dict, persist_since=None):
        """Overrides the host recorded (persist, clamps) that are not in the run's
        record yet; each new one is logged."""
        new = {}
        with self._camera_overrides_lock:
            for key, f in dict(fields or {}).items():
                if key not in self._camera_overrides:
                    self._camera_overrides[key] = dict(f)
                    new[key] = f
            if self._camera_overrides:
                self._camera_overrides_key = self._camera_overrides_key or str(camera_key)
            if persist_since and not self._camera_overrides_persist_since:
                self._camera_overrides_persist_since = persist_since
        for key, f in new.items():
            logger.warning(f"CAMERA OVERRIDE: run {self._current_run_id} on {camera_key}: "
                           f"{key} requested {f.get('requested')!r} -> applied "
                           f"{f.get('applied')!r} ({f.get('origin')}); camera_params in the file "
                           f"stay the request")

    def _host_end_run(self, reason: str, record: bool = True):
        """The current run's hold on its camera ends (END_RUN, ABORT_RUN, a
        superseding INIT_RUN). With ``record``: the host's overrides go into the
        run's record, and what it saw that the frame count cannot show (frames
        the camera reported lost, a camera fault) into the grab failure."""
        host, token = self._camera_host, self._host_run_token
        if host is None or not token:
            return None
        self._host_run_token = ""
        self._host_arm_future = None
        self._host_run_start = None
        try:
            summary = host.end_run(token, reason)
        except Exception as exc:
            logger.exception(f"camera host: ending the run's hold on its camera failed: {exc}")
            return None
        if not record:
            return summary
        if summary.overrides:
            self._merge_camera_overrides(summary.camera_key, summary.overrides,
                                         summary.persist_since)
        problems = summary.problems
        if problems and not self._reset_requested:
            text = "camera host: " + "; ".join(problems)
            if text not in (self._grab_failure or ""):
                self._grab_failure = f"{self._grab_failure}; {text}" if self._grab_failure else text
                get_log_buffer().update_run(grab_failure=self._grab_failure)
        return summary

    def stop(self):
        """Request the server loop to stop on the next poll cycle."""
        self._stop_beacon()
        self._running = False

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        actual_port = socket.bind_to_random_port("tcp://0.0.0.0")
        self._port = actual_port
        self._waxx_port = actual_port   # sync beacon port before _start_beacon()
        socket.setsockopt(zmq.RCVTIMEO, 500)   # 500 ms poll so we can honour stop()
        self._running = True
        self._start_beacon()
        logger.info(f"liveOD server listening on tcp://0.0.0.0:{self._port}")
        try:
            while self._running:
                # every pass, not only on a poll timeout: viewers' POLLs can keep
                # the socket busy for as long as an unanswered abort waits
                try:
                    self._check_abort_reply()
                except Exception:
                    logger.exception("abort reply check failed")
                try:
                    self._check_abort_again()
                    self._check_exited_run()
                except Exception:
                    logger.exception("closing a run whose experiment is gone failed")
                try:
                    # a pickled dict, plus raw buffers for PUT_DATA (one frame
                    # per array, never copied through pickle)
                    frames = socket.recv_multipart()
                except zmq.Again:
                    continue          # poll timeout — loop to check _running

                reply = self._dispatch(frames)
                socket.send(pickle.dumps(reply))
        finally:
            # the run's writer must not outlive the server with the file open
            try:
                self._run_file.finish_writer()
            except Exception:
                logger.exception("closing the run's writer at shutdown failed")
            socket.close()
            context.term()

    def _dispatch(self, frames) -> dict:
        """One request to its handler: ``frames[0]`` is the pickled message,
        the rest are PUT_DATA's raw buffers. The reply dict; never raises."""
        tag = "<unknown>"
        try:
            msg = pickle.loads(frames[0])
            tag = msg.get("tag", "")
            if tag == "INIT_RUN":
                reply = self._handle_init_run(msg)
            elif tag == "WAIT_CAM_READY":
                reply = self._handle_wait_cam_ready(msg)
            elif tag == "SHOT_COMPLETE":
                reply = self._handle_shot_complete(msg)
            elif tag == "PUT_DATA":
                reply = self._handle_put_data(msg, frames[1:])
            elif tag == "END_RUN":
                reply = self._handle_end_run(msg)
            elif tag == "SAVE_STATUS":
                reply = self._handle_save_status(msg)
            elif tag == "GET_AUX_DATA":
                reply = self._handle_get_aux_data(msg)
            elif tag == "RESET":
                reply = self._handle_reset(msg)
            elif tag == "CAMERA_CONTROL":
                reply = self._handle_camera_control(msg)
            elif tag == "POLL":
                reply = self._handle_poll(msg)
            elif tag == "GET_LOG":
                reply = self._handle_get_log(msg)
            elif tag == "ABORT_RUN":
                reply = self._handle_abort_run(msg)
            elif tag == "RUN_EXITED":
                reply = self._handle_run_exited(msg)
            elif tag == "SUBSCRIBE_SCALARS":
                reply = self._handle_subscribe_scalars(msg)
            elif tag == "UNSUBSCRIBE_SCALARS":
                reply = self._handle_unsubscribe_scalars(msg)
            elif tag == "GET_ADJUST_VALUES":
                reply = self._handle_get_adjust_values(msg)
            elif tag == "SET_ADJUST_VALUE":
                reply = self._handle_set_adjust_value(msg)
            elif tag == "SET_ADJUST_SPEC":
                reply = self._handle_set_adjust_spec(msg)
            elif tag == "GET_MARKERS":
                reply = self._handle_get_markers(msg)
            elif tag == "SET_MARKERS":
                reply = self._handle_set_markers(msg)
            else:
                reply = {"ok": False, "error": f"Unknown tag: {tag}"}
        except Exception as exc:
            reply = {"ok": False, "error": str(exc)}
            logger.exception(f"Error handling {tag!r}: {exc}")
        return reply

    # ------------------------------------------------------------------
    # Data pushed during the run, and the asynchronous save
    # ------------------------------------------------------------------

    def _handle_put_data(self, msg: dict, buffers) -> dict:
        """Arrays the experiment pushes during the run (auxiliary camera frames,
        scope traces): one raw buffer per item, each for its slot of a dataset
        under ``data/`` (pre-allocated at INIT_RUN, or made on first use from
        ``full_shape``). In a run that saves they are queued for the run's
        writer, the only handle on the file; the reply says that much and no
        more -- what the writer could not write is in END_RUN's completeness
        report. In a run that saves nothing they are taken all the same
        (``written`` False). Either way the latest of each key is kept for
        GET_AUX_DATA, and a notice of what came (no arrays) goes out on the
        broadcast (AUX_NOTICE)."""
        if not self._run_msg_ok(msg):
            return self._stale_run_reply("PUT_DATA", msg)
        if not self._run_in_progress:
            return {"ok": False, "error": "no run in progress"}
        specs = list(msg.get("items", []))
        if len(specs) != len(buffers):
            return {"ok": False, "error": f"{len(specs)} items but {len(buffers)} buffers"}
        items = []
        for spec, buf in zip(specs, buffers):
            dtype = np.dtype(str(spec["dtype"]))
            arr = np.frombuffer(buf, dtype=dtype)
            shape = tuple(int(s) for s in spec.get("shape", (arr.size,)))
            arr = arr.reshape(shape)
            items.append(PutItem(spec["key"], spec.get("index"), arr,
                                 full_shape=spec.get("full_shape"), dtype=dtype,
                                 fill=spec.get("fill"), offset=spec.get("offset")))
        saving = self._run_file.pending
        written = self._run_file.put(items) if saving else False
        if saving and not written:
            return {"ok": False, "error": "the run's file is not taking data "
                                          "(its writer is closed)"}
        # the latest of each key, for live viewers and derived quantities,
        # whether the run saves or not
        self._note_aux_items(items)
        return {"ok": True, "queued": len(items) if written else 0, "written": written}

    def _note_aux_items(self, items):
        """Keep the latest array per key (GET_AUX_DATA) and send one notice of
        what came, without the arrays, to the broadcast (AUX_NOTICE). A slice
        of a whole array pushed at the end is not a shot's data and is left
        out of both.

        The arrays themselves are not broadcast: ~2.5 MB of diagnostic frames
        a shot, pickled whole into a PUB socket whose 8-message high-water mark
        they shared with OD_IMAGE / RUN_DONE / RUN_STATE, crowded those out of
        the viewers' pipes, and each one went through the GUI thread on its way
        (2026-10-05). Nothing subscribed to them."""
        now = time.time()
        notes = []
        for it in items:
            if it.index is None and it.offset is not None:
                continue
            index = None if it.index is None else list(it.index)
            self._aux_latest[it.key] = {"run_id": self._current_run_id, "key": it.key,
                                        "index": index, "array": it.array, "t": now}
            notes.append({"key": it.key, "index": index, "offset": it.offset,
                          "shape": list(it.array.shape), "dtype": it.array.dtype.str,
                          "nbytes": int(it.array.nbytes)})
        sink = self._aux_notice_sink
        if notes and sink is not None:
            try:
                sink({"run_id": self._current_run_id, "t": now, "items": notes})
            except Exception:
                logger.debug("aux notice failed", exc_info=True)

    def _handle_get_aux_data(self, msg: dict) -> dict:
        """The latest pushed array of each key (or of ``keys``), with the run
        it came from; for a viewer that wants a frame it missed (the broadcast
        only says that one came: AUX_NOTICE)."""
        latest = dict(self._aux_latest)
        keys = msg.get("keys")
        if keys:
            latest = {k: v for k, v in latest.items() if k in set(keys)}
        return {"ok": True, "run_id": self._current_run_id, "items": latest}

    def _handle_save_status(self, msg: dict) -> dict:
        """How far the asynchronous END_RUN save is (the client polls this
        instead of waiting on one long reply)."""
        if not self._run_msg_ok(msg):
            return self._stale_run_reply("SAVE_STATUS", msg)
        reply = self._run_file.status()
        reply["ok"] = True
        return reply

    def _on_async_save_done(self, result: dict):
        """The save thread is over (called on it): the run's outcome and state,
        as the synchronous END_RUN records them."""
        run_id = self._current_run_id
        state = result.get("state")
        if state == "failed":
            logger.error(f"END_RUN: save of run {run_id} failed: {result.get('error')}")
            self._record_outcome("save_failed", str(result.get("cause", "")))
            self._set_run_state("error", f"Save failed: {result.get('cause', '')}")
        elif state == "saved_incomplete":
            inc = result.get("incomplete") or {}
            logger.error(
                f"END_RUN: run_id={run_id} saved INCOMPLETE ({inc.get('reason', '')}). "
                f"The file is marked data_complete=False; its images are in arrival "
                f"order and do not line up with the shots."
            )
            self._record_outcome("saved_incomplete", str(inc.get("reason", "")))
            self._set_run_state("saved", f"INCOMPLETE: {inc.get('reason', '')}")
        else:
            logger.info(f"END_RUN: run_id={run_id} saved.")
            self._record_outcome("saved")
            self._set_run_state("saved")
        self._run_in_progress = False
        self._abort_requested_at = None
        self.run_done_signal.emit()

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _finalize_reset_run(self, notify_gui: bool = True):
        """Finalize a reset-aborted run.

        This is called either when the experiment explicitly sends ABORT_RUN,
        or opportunistically on the next INIT_RUN if the experiment process was
        killed and never sent that confirmation.

        ``notify_gui`` controls whether ``reset_signal`` is emitted to drive
        ``main_window.reset()``.  Set it to False when the GUI was already
        notified by the original ``_handle_reset`` call — re-emitting would
        race with the next ``INIT_RUN`` and cause ``main_window.reset()`` to
        set ``_reset_requested = True`` after the new run has already started,
        aborting the new run on its first poll.
        """
        # Emit reset_signal so the GUI's reset() handler interrupts the
        # DataHandler and CameraBaby.  This sets data_handler.interrupted=True
        # and calls data_handler.quit(), which causes the DataHandler to close
        # its HDF5 handle and emit done_writing_signal — which in turn sets
        # _data_handler_done_event.  Must happen BEFORE the wait below.
        # Only emit when a camera run is active: for no-camera runs there is
        # no DataHandler or CameraBaby to clean up, and reset() would
        # incorrectly call update_run_id() (no active baby → else branch).
        if notify_gui and self._current_capture_images:
            self.reset_signal.emit()
        # Delete the run's file if it still exists, once the image writer has
        # let go of it.
        self._run_file.discard()
        self._reset_requested = False
        self._abort_requested_at = None
        self._run_in_progress = False
        self._record_outcome("discarded", "reset")
        self._set_run_state("aborted")
        self.run_done_signal.emit()

    def _record_outcome(self, outcome: str, detail: str = ""):
        """How the current run ended, for the log buffer's run index and POLL."""
        self._last_outcome = {
            "run_id": self._current_run_id, "outcome": outcome, "detail": detail,
            "n_shots": len(self._shot_timestamps),
            "images_expected": self._images_expected,
            "images_received": self._images_received_now(),
            **self._current_client,
        }
        get_log_buffer().end_run(
            outcome, detail,
            n_shots=len(self._shot_timestamps),
            images_expected=self._images_expected,
            images_received=self._images_received_now(),
        )

    def _handle_init_run(self, msg: dict) -> dict:
        # The new run's name, provisional until this INIT_RUN is accepted: one
        # that is refused leaves the run in progress as it was (its token, so
        # its messages still count; its camera; its state).
        token = uuid.uuid4().hex
        # The previous run's save may still be running (an asynchronous
        # END_RUN): this run's file is not made under it.
        if self._run_file.saving:
            logger.info("INIT_RUN: waiting for the previous run's save to finish")
            if not self._run_file.wait_save(SAVE_WAIT_BEFORE_INIT_S):
                return self._refuse_init_run(
                    f"the previous run is still being saved after "
                    f"{SAVE_WAIT_BEFORE_INIT_S:.0f} s; try again",
                    "INIT_RUN refused: the previous run is still being saved")
        save_data = bool(msg.get("save_data", False))
        capture_images = bool(msg.get("capture_images", False))
        camera_key = str(msg.get("camera_key", ""))
        imaging_type = int(msg.get("imaging_type", 0))
        camera_params = msg.get('camera_params', {})

        # host mode: the run's camera is locked here, before a run id is used;
        # a refusal (profile, persist, camera held elsewhere) is the reply
        host_refusal = self._host_begin_run(msg, camera_key, capture_images, camera_params,
                                            token)
        if host_refusal is not None:
            return host_refusal

        run_id = 0
        filepath = ""

        if save_data:
            # Synchronous on purpose, and a small write: see RunFile.reserve. Only
            # the new file is made here: the run in progress keeps its own, and
            # the RunFile state for it, until this INIT_RUN is accepted below.
            _t_create = time.time()
            try:
                run_id, filepath = self._data_saver.reserve_run_id_and_path(msg)
            except Exception as exc:
                logger.exception(f"INIT_RUN: could not create data file: {exc}")
                # host mode: nor will this run use the camera it locked
                self._host_abandon_run(token, "INIT_RUN refused: the data file was not created")
                return self._refuse_init_run(f"Data file creation failed: {exc}",
                                             f"Data file creation failed: {exc}")
            _dt_create = time.time() - _t_create
            if _dt_create > 2.0:
                logger.warning(f"Data file creation took {_dt_create:.1f} s — is the data drive slow?")

        # The INIT_RUN is accepted: from here the new run takes over. Any message
        # that carries the previous run's token is ignored from now on --
        # including the rest of a run this INIT_RUN takes over from (which it
        # does, as before).
        if self._run_in_progress and not self._reset_requested:
            logger.warning(f"INIT_RUN while run {self._current_run_id} is still in progress: "
                           f"that run is superseded; its further messages will be ignored.")
            # Its outcome, before the new token: nothing else records one (its file
            # is closed with what it has by begin() below -- not saved, not
            # deleted). liveOD cannot see whether its process is gone; it names
            # the client it had.
            c = self._current_client
            who = (f"pid {c['client_pid']} on {c['client_host'] or '?'}"
                   + (f", launched by {c['launcher']}" if c["launcher"] else "")
                   if c["client_pid"] is not None else "client not recorded (older client)")
            self._record_outcome("superseded",
                                 f"superseded by a new INIT_RUN while in progress; its client "
                                 f"was {who}; file closed with what it had, not saved, "
                                 f"not deleted")
        # If the previous run was reset but the experiment process was killed
        # before sending ABORT_RUN, finalize that reset now. This keeps the
        # next run start non-blocking and stateless.  The GUI was already
        # notified by the original _handle_reset call, so don't re-emit
        # reset_signal — that would race with this INIT_RUN and cause the
        # new run to abort on its first poll. (Before the new token: until
        # then the reset run's camera threads still count as its own. Before
        # begin() below: the reset run's file is still the one RunFile knows.)
        # Only a run still in progress has anything to finalize: with none, the
        # Abort was pressed between runs and is just cleared (below). Finalizing
        # then recorded the previous run as "discarded/reset" and deleted its
        # file if its save had failed (RunFile keeps the path for a retry).
        if self._reset_requested and self._run_in_progress:
            self._finalize_reset_run(notify_gui=False)
        elif self._reset_requested:
            logger.info(f"INIT_RUN: an Abort pressed with no run in progress is cleared "
                        f"(run {self._current_run_id} had already ended; nothing discarded).")
        self._adopt_run_token(token)
        # host mode: the previous run's hold on its camera ends; this run's stays
        self._host_commit_run(token)
        self._current_camera_key = camera_key

        # For Basler cameras: if a previous baby is still running its grab
        # loop (e.g. after a reset), clear the event so WAIT_CAM_READY will
        # block until that grab loop fully exits and on_basler_baby_done() fires.
        # ("Basler": which cameras need this is the lab's call --
        # LiveODConfig.camera_needs_grab_drain; by default any key containing "basler".)
        if self._needs_grab_drain(camera_key) and capture_images:
            with self._basler_lock:
                self._basler_babies_live += 1
                stale = self._basler_babies_live - 1
                if stale > 0:
                    self._basler_prev_grab_done_event.clear()
            if stale > 0:
                logger.warning(f"INIT_RUN: {stale} Basler grab loop(s) not yet done — WAIT_CAM_READY will block until they exit.")

        self._cam_ready_event.clear()
        self._current_capture_images = capture_images
        # Gates the END_RUN save on the image writer finishing (a run without a
        # camera has none); the file reserved above is this run's from now on.
        self._run_file.begin(save_data, has_writer=capture_images)
        self._run_file.filepath = filepath

        self._current_run_id = run_id
        self._reset_requested = False
        self._abort_requested_at = None
        self._abort_again = False
        self._exited_pending = None
        self._exited_close_now = ""
        self._run_in_progress = True
        self._shot_timestamps = []       # reset per-run timestamp list
        self._aux_latest = {}            # the latest pushed array per key, this run
        self._init_run_time = time.time()
        self._shot_durations = []  # reset rolling average for new run

        n_img = int(msg.get('params', {}).get('N_img', 1))
        # the run's one handle on its file, open from here to END_RUN: the
        # camera's frames and the experiment's pushed arrays go through it
        self._run_file.start_writer(n_img)
        n_shots = int(msg.get('N_shots_with_repeats', 1))
        n_pwa = int(msg.get('N_pwa_per_shot', 1))
        self._current_n_shots = n_shots
        with self._images_lock:
            self._images_received = 0
            self._frame_times = []
        self._images_expected = n_img if capture_images else 0
        self._grab_failure = ""
        self._frame_deficit_warned = 0
        self._shot_mono = []
        self._t_ready_mono = None
        with self._camera_overrides_lock:
            self._camera_overrides = {}
            self._camera_overrides_key = ""

        params_payload = dict(msg.get('params', {}))
        run_info_payload = {
            'imaging_type': imaging_type,
            'save_data': int(msg.get('save_data_flag', int(save_data))),
            'run_date_str': str(msg.get('run_date_str', '')),
            'run_datetime_str': str(msg.get('run_datetime_str', '')),
            'expt_class': str(msg.get('expt_class', '')),
            'xvarnames': list(msg.get('xvarnames', [])),
        }

        # The run goes by its experiment file's name; an experiment process from
        # before that was sent only gives the class.
        self._current_expt_name = str(msg.get('expt_file') or msg.get('expt_class', ''))
        self._current_client = _client_of(msg)
        # {name: [min, max]} of the scan, for the units the viewer shows the xvars
        # in; an older experiment process does not send it
        self._current_xvar_ranges = dict(msg.get('xvar_ranges') or {})
        # From here every log record is this run's (GET_LOG).
        get_log_buffer().begin_run(
            run_id, self._current_expt_name,
            n_shots_expected=n_shots, images_expected=self._images_expected,
            camera_key=camera_key if capture_images else "", save_data=save_data,
            filepath=filepath,
        )

        # host mode: the lock takes the run's name; Persist is announced on its log
        self._host_after_reserve(run_id)

        adjust_specs = list(msg.get('adjust_specs', []))
        with self._adjust_lock:
            self._adjust_specs = adjust_specs
            self._adjust_values = {s['key']: s['current_val'] for s in adjust_specs}
        self.adjust_specs_signal.emit(adjust_specs)
        if adjust_specs and save_data:
            logger.warning(
                "Adjustable params are active with save_data=True. Values changed "
                "in the Adjust panel will NOT be reflected in saved data."
            )

        self._note_spawn_token(capture_images)
        self.run_started_signal.emit(run_id, list(run_info_payload.get('xvarnames', [])))
        self.new_run_signal.emit(filepath, camera_key, capture_images, save_data, imaging_type, n_img, n_shots, n_pwa, camera_params, params_payload, run_info_payload)
        self._run_state = "idle"    # so the new run's first state is always emitted
        self._set_run_state("waiting_camera" if capture_images else "running", camera_key)
        logger.info(
            f"INIT_RUN: run_id={run_id}, {self._current_expt_name}, "
            f"{n_shots} shots, save={save_data}, "
            f"camera={camera_key if capture_images else 'none'}"
        )
        reply = {"ok": True, "run_id": run_id, "filepath": filepath, "run_token": self._run_token,
                 # what this server can do beyond the original protocol: arrays
                 # pushed during the run, and a save the client polls for
                 "features": {"put_data": True, "async_save": True}}
        # camera settings this run will get that differ from its camera_params
        # (Persist, host mode); the client prints them as a banner
        overrides = self.camera_overrides_record()
        if overrides:
            reply["camera_overrides"] = overrides
        return reply

    def _handle_wait_cam_ready(self, msg: dict) -> dict:
        """Wait up to ``timeout`` s for the camera.

        The client asks in short slices rather than one long wait: this loop is
        single-threaded, so while it blocks here a RESET from the remote viewer
        cannot even be received, and a reset camera never becomes ready. Every
        reply carries ``reset_requested`` so the experiment can abort at once;
        ``timed_out`` marks "not ready yet" apart from a real failure.

        "Ready" is the camera thread's status 2, sent once the driver reports
        acquisition running (CameraBaby's on_armed). A camera whose grab failed
        before that will never be ready: that is reported at once, without
        ``timed_out``, so the experiment stops instead of waiting out its timeout.
        """
        if not self._run_msg_ok(msg):
            return self._stale_run_reply("WAIT_CAM_READY", msg)
        if self._reset_requested:
            return {"ok": True, "ready": False, "reset_requested": True}
        if self._grab_failure and not self._cam_ready_event.is_set():
            return {"ok": False, "ready": False, "reset_requested": False,
                    "error": f"camera failed before it was ready: {self._grab_failure}"}

        timeout = float(msg.get("timeout", 60.0))
        deadline = time.time() + timeout

        # For Basler cameras: wait for the previous grab loop to fully exit
        # before reporting camera-ready to the experiment.  This prevents the
        # experiment from arming the hardware while the old grab is still
        # blocking in RetrieveResult() and the camera is not yet free.
        # (not for a camera-host camera: its worker is the only thread on it)
        host = getattr(self, "_camera_host", None)
        if host is None and get_config().camera_needs_grab_drain(self._current_camera_key):
            if not self._basler_prev_grab_done_event.is_set():
                self._set_run_state("waiting_grab_drain")
            remaining = deadline - time.time()
            grab_done = self._basler_prev_grab_done_event.wait(timeout=max(0.0, remaining))
            if not grab_done:
                return {"ok": False, "ready": False, "timed_out": True,
                        "reset_requested": self._reset_requested,
                        "error": "Basler previous grab-loop exit timeout"}

        if not self._cam_ready_event.is_set():
            self._set_run_state("waiting_camera")
        if host is None:
            remaining = deadline - time.time()
            ready = self._cam_ready_event.wait(timeout=max(0.0, remaining))
        else:
            # host mode: the run's first WAIT_CAM_READY arms its camera; an arm
            # that failed is reported at once, without timed_out, so the
            # experiment stops
            arm_error = self._host_start_arm()
            ready = False if arm_error else self._wait_cam_ready_event(deadline)
            if not ready and not arm_error:
                arm_error = self._host_arm_error()
            if arm_error:
                return {"ok": False, "ready": False, "reset_requested": self._reset_requested,
                        "error": arm_error}
        if not ready:
            return {"ok": False, "ready": False, "timed_out": True,
                    "reset_requested": self._reset_requested,
                    "error": "Camera ready timeout"}
        if not self._reset_requested:
            self._set_run_state("running")
        if self._t_ready_mono is None:
            # no shot can trigger the camera before the experiment has this reply
            self._t_ready_mono = time.monotonic()
        reply = {"ok": True, "ready": True, "reset_requested": self._reset_requested}
        overrides = self.camera_overrides_record()
        if overrides:
            reply["camera_overrides"] = overrides
        return reply

    def _handle_shot_complete(self, msg: dict) -> dict:
        if not self._run_msg_ok(msg):
            return self._stale_run_reply("SHOT_COMPLETE", msg)
        now = time.time()
        # Taken first: the experiment schedules nothing of the next shot until
        # this reply, so no frame of it can come before this moment.
        self._shot_mono.append(time.monotonic())
        shot_idx = int(msg.get("shot_idx", 0))
        N_total = int(msg.get("N_shots_total", 1))
        xvar_values = msg.get("xvar_values", {})

        # Delta-t: time since last shot (or since INIT_RUN for the first shot).
        last_t = self._shot_timestamps[-1] if self._shot_timestamps else self._init_run_time
        delta_t = now - last_t if last_t > 0.0 else 0.0
        self._shot_timestamps.append(now)

        # ETA: 5-shot rolling average of inter-shot durations (excluding first shot).
        # Only show ETA after the second shot (shot_idx >= 1).
        if shot_idx > 0:  # Second shot onwards
            self._shot_durations.append(delta_t)
            if len(self._shot_durations) > 5:
                self._shot_durations.pop(0)
            
            avg_per_shot = sum(self._shot_durations) / len(self._shot_durations)
            remaining = max(0, N_total - shot_idx - 1)
            
            if avg_per_shot > 0.0 and remaining > 0:
                eta_epoch = now + avg_per_shot * remaining
                eta_str = time.strftime("%H:%M", time.localtime(eta_epoch))
            elif remaining == 0:
                eta_str = time.strftime("%H:%M", time.localtime(now))
            else:
                eta_str = "--:--"
        else:
            eta_str = "--:--"

        self.shot_progress_signal.emit(shot_idx, N_total, xvar_values)
        # What the shot recorded about itself ({key: float}; empty from an
        # experiment process that predates it). The Analyzer picks the absorption
        # cross section from it. Emitted after shot_progress_signal, which carries
        # this shot's xvar values to the Analyzer.
        self.shot_conditions_signal.emit(shot_idx, dict(msg.get("shot_conditions") or {}))
        self.shot_timing_signal.emit(float(delta_t), str(eta_str))

        with self._adjust_lock:
            adjust_values = dict(self._adjust_values)
        if adjust_values:
            self.shot_adjust_values_signal.emit(adjust_values)

        if not self._reset_requested:
            self._set_run_state("running")
        # Every shot goes to the terminal and the log file; the GUI's progress
        # bar carries the per-shot news, so its log only gets about one in twenty.
        line = f"shot {shot_idx + 1}/{N_total} (Δt={delta_t:.1f}s | ETA {eta_str})"
        sparse = shot_idx == 0 or shot_idx + 1 == N_total or (shot_idx + 1) % max(1, N_total // 20) == 0
        logger.log(logging.INFO if sparse else logging.DEBUG, line)
        self._check_frame_deficit(shot_idx, N_total)
        # Include reset flag so the experiment can abort at shot boundary
        # even if the POLL-based check misses it.
        reply = {"ok": True, "reset_requested": self._reset_requested, "adjust_values": adjust_values}
        # (getattr: k-exp's cross-section test runs this on a stand-in without it)
        grab_failure = getattr(self, "_grab_failure", "")
        if grab_failure:
            # The camera's grab has ended (a lost frame, a timeout): no frame of
            # this run is recorded from here on. The experiment is told (its client
            # warns once); stopping the run is left to the person running it.
            reply["grab_failure"] = grab_failure
        return reply

    def _check_frame_deficit(self, shot_idx: int, N_total: int):
        """Warn, during the run, when the camera has delivered fewer frames than
        the shots so far must have produced.

        SHOT_COMPLETE is an RPC from the kernel, which can run ahead of the
        hardware by a shot or so, and a frame is in flight for the readout
        time, so this only counts the shots *before* the one just reported and
        only speaks up once the shortfall is a whole shot's worth. It cannot
        abort: a legitimate run with a slow readout could look like this for a
        moment. END_RUN's count against ``N_img`` is the check that decides
        whether the file is marked complete.
        """
        if not self._images_expected or N_total <= 0:
            return
        per_shot = max(1, self._images_expected // N_total)
        must_have = per_shot * shot_idx           # frames from the shots before this one
        deficit = must_have - self._images_received_now()
        if deficit >= per_shot and deficit > self._frame_deficit_warned:
            self._frame_deficit_warned = deficit
            logger.warning(
                f"Camera is {deficit} frame(s) behind after shot {shot_idx + 1}/{N_total} "
                f"({self._images_received_now()} received, {must_have} due from the shots "
                f"before it). A trigger the camera was not ready for leaves no gap: "
                f"every later frame lands one slot early. If this does not clear, the "
                f"run will be saved as incomplete."
            )

    def _handle_end_run(self, msg: dict) -> dict:
        if not self._run_msg_ok(msg):
            return self._stale_run_reply("END_RUN", msg)
        # host mode: the run's hold on its camera ends before the save (the camera
        # stays idle at the run's settings); its overrides join the record below
        self._host_end_run("END_RUN")
        # camera settings that differed from the request go into the file too
        msg = self._with_camera_overrides(msg)
        # frames that provably are not in their shots' slots mark the run incomplete
        self._check_frame_alignment()
        if self._reset_requested:
            logger.warning(f"END_RUN: run {self._current_run_id} was reset — discarding data.")
            # As before the move: this path does not wait for the image writer.
            self._run_file.discard(wait_for_writer=False)
            self._reset_requested = False
            self._abort_requested_at = None
            self._run_in_progress = False
            self._record_outcome("discarded", "reset")
            self._set_run_state("aborted")
            self.run_done_signal.emit()
            return {"ok": True}
        incomplete = None
        if self._run_file.pending and msg.get("async_save"):
            # The save runs on a thread of its own and the client polls
            # SAVE_STATUS for it: a long save neither blocks every other
            # client nor looks like a dead server to the experiment.
            self._set_run_state("saving")
            self._run_file.save_async(
                msg, self._current_run_id, self._shot_timestamps,
                on_done=self._on_async_save_done,
                images_expected=self._images_expected,
                images_received=self._images_received_now(),
                grab_failure=self._grab_failure,
            )
            return {"ok": True, "saving": True}
        if self._run_file.pending:
            self._set_run_state("saving")
            try:
                # waits for the image writer, stashes the payload, then saves
                incomplete = self._run_file.save(
                    msg, self._current_run_id, self._shot_timestamps,
                    images_expected=self._images_expected,
                    images_received=self._images_received_now(),
                    grab_failure=self._grab_failure,
                )
            except RunFileSaveError as exc:
                error = str(exc)    # the saver's error, plus the way out if the payload was stashed
                # one record, so the error banner shows the failure and the way out together
                logger.error(f"END_RUN: save of run {self._current_run_id} failed: {error}")
                self._record_outcome("save_failed", exc.cause)
                self._set_run_state("error", f"Save failed: {exc.cause}")
                # The run is over either way.  Clear the in-progress state
                # before reporting the failure, otherwise the server rejects
                # all CAMERA_CONTROL for the rest of the session and the GUI
                # never resets.
                self._run_in_progress = False
                self.run_done_signal.emit()
                return {"ok": False, "error": error}
            if incomplete:
                # An ERROR, not a warning: the file exists but must not be
                # read as a complete run, and the banner should say so.
                logger.error(
                    f"END_RUN: run_id={self._current_run_id} saved INCOMPLETE "
                    f"({incomplete['reason']}). The file is marked data_complete=False; "
                    f"its images are in arrival order and do not line up with the shots."
                )
                self._record_outcome("saved_incomplete", incomplete["reason"])
                self._set_run_state("saved", f"INCOMPLETE: {incomplete['reason']}")
            else:
                logger.info(f"END_RUN: run_id={self._current_run_id} saved.")
                self._record_outcome("saved")
                self._set_run_state("saved")
        else:
            logger.info("END_RUN: save_data=False, nothing written.")
            self._record_outcome("nothing_written", "save_data=False")
            self._set_run_state("done", "save_data=False, nothing written")

        self._run_in_progress = False
        self._abort_requested_at = None
        self.run_done_signal.emit()
        reply = {"ok": True}
        if incomplete:
            reply["incomplete"] = dict(incomplete)
        return reply

    def note_reset_requested(self):
        """The GUI's own abort button was pressed (the remote path is _handle_reset).
        Starts the wait for the experiment's answer (_check_abort_reply); pressing
        Abort again neither restarts it nor turns "no_reply" back into "aborting"."""
        if self.reset_ignored_during_save():
            return False
        if self._run_in_progress:
            if self._abort_requested_at is None:
                self._abort_requested_at = time.time()
            if self._run_state != "no_reply":
                self._set_run_state("aborting")
        return True

    def reset_ignored_during_save(self) -> bool:
        """A Reset while the run's asynchronous save runs is ignored: the run is
        complete (END_RUN came) and only its write remains, so it is neither
        marked "aborting" nor left with an Abort pending (which a later
        RUN_EXITED / ABORT_RUN / INIT_RUN would turn into deleting its file).
        True when ignored. Call it once per press: an INFO line every press, and
        one WARNING per run. Any thread."""
        if not self._run_file.saving:
            return False
        run_id = self._current_run_id
        logger.info(self.reset_ignored_text())
        if self._reset_during_save_warned != run_id:
            self._reset_during_save_warned = run_id
            logger.warning(f"Reset pressed during the save of run {run_id}: ignored, "
                           f"the run is complete.")
        return True

    def reset_ignored_text(self) -> str:
        """What a press of Reset during the save is told (window, remote viewer)."""
        return f"Reset ignored: run {self._current_run_id} is being saved (it is complete)"

    def _abort_reply_limit(self) -> float:
        """How long an Abort may wait for the experiment's answer before it is shown
        as "no_reply": ABORT_REPLY_SHOT_FACTOR times the run's longest recent shot
        period (the first shot's, from INIT_RUN, before there are two), never less
        than ABORT_REPLY_MIN_S; the minimum before any shot."""
        if self._shot_durations:
            period = max(self._shot_durations)
        elif self._shot_timestamps and self._init_run_time:
            period = self._shot_timestamps[0] - self._init_run_time
        else:
            return ABORT_REPLY_MIN_S
        return max(ABORT_REPLY_MIN_S, ABORT_REPLY_SHOT_FACTOR * float(period))

    def _check_abort_reply(self, now=None):
        """An Abort nobody answered within _abort_reply_limit() becomes "no_reply".
        Nothing else changes: a live experiment is still told to stop (its next
        POLL or SHOT_COMPLETE answers as before), its file is untouched, and a late
        ABORT_RUN / END_RUN / RUN_EXITED, or the next INIT_RUN, closes the run out
        as for any abort. Run 83110 (2026-09-26): its process had died, and the
        pill sat on "Aborting" until the next run started."""
        t0 = self._abort_requested_at
        if t0 is None or not self._run_in_progress or not self._reset_requested:
            return
        if self._run_state == "no_reply":
            return                      # said once
        now = time.time() if now is None else float(now)
        limit = self._abort_reply_limit()
        waited = now - t0
        if waited <= limit:
            return
        detail = (f"Abort requested {waited:.0f} s ago and no answer from the experiment "
                  f"(limit {limit:.0f} s, from the shot period): its process may be gone "
                  f"or hung. It is still told to stop; the next run start discards its "
                  f"file, as for any abort.")
        logger.warning(f"Run {self._current_run_id}: {detail}")
        self._set_run_state("no_reply", detail)

    def _handle_run_exited(self, msg: dict) -> dict:
        """The experiment's process is exiting with its run still open -- no END_RUN,
        and no ABORT_RUN that reached liveOD (LiveODClient.notify_exit, an atexit
        handler). ``reason``: the uncaught exception, "" when none was reported.

        * During an abort: that is the abort's answer, taken exactly as ABORT_RUN.
        * With camera frames still due: the camera thread still owns the run's
          file, so the run stays open (as after a crash until now); only the state
          says what happened.
        * Otherwise the run is over: state and outcome "exited", and its file is
          left as it was -- not saved, not deleted, and forgotten, so no later
          reset can delete it.
        A superseded run's notice changes nothing; one after the run ended is ignored."""
        if not self._run_msg_ok(msg):
            return self._stale_run_reply("RUN_EXITED", msg)
        refused = self._refuse_while_saving("RUN_EXITED")
        if refused is not None:
            return refused
        # A notice sent on the process's behalf (the monitor server's run loop,
        # which has no token) names the run instead; another run's changes nothing.
        if msg.get("run_id") is not None and msg.get("run_id") != self._current_run_id:
            return {"ok": False, "stale_run": True,
                    "error": f"RUN_EXITED for run {msg.get('run_id')}, but the current run is "
                             f"{self._current_run_id}; ignored"}
        if not self._run_in_progress:
            return {"ok": True, "ignored": True}
        why = str(msg.get("reason") or "") or "no exception reported"
        run_id = self._current_run_id
        if self._reset_requested:
            logger.warning(f"RUN_EXITED: the experiment of run {run_id} exited during its "
                           f"abort ({why}); taken as the abort's acknowledgement.")
            return self._handle_abort_run(msg)
        received, expected = self._images_received_now(), self._images_expected
        if self._current_capture_images and received < expected:
            detail = (f"The experiment's process exited without END_RUN ({why}) with "
                      f"{received}/{expected} frames in. The run is closed once no frame has "
                      f"come for {EXITED_FRAME_GRACE_S:.0f} s (at once on Reset); its file is "
                      f"left as it was: not saved, not deleted.")
            logger.warning(f"RUN_EXITED: run {run_id}: {detail}")
            if self._exited_pending is None:
                self._exited_pending = {"why": why, "t": time.monotonic()}
            self._set_run_state("exited", detail)
            return {"ok": True}
        detail = (f"The experiment's process exited without END_RUN ({why}). Its file is "
                  f"left as it was: not saved, not deleted.")
        logger.warning(f"RUN_EXITED: run {run_id}: {detail}")
        self._host_end_run("RUN_EXITED", record=False)
        # the writer closes the file with what it has; forgotten: a reset
        # pressed later must not delete an exited run's file
        self._run_file.leave()
        self._run_in_progress = False
        self._abort_requested_at = None
        self._record_outcome("exited", why)
        self._set_run_state("exited", detail)
        self.run_done_signal.emit()
        return {"ok": True}

    # A run whose experiment is gone
    # ------------------------------------------------------------------

    def exited_run_pending(self) -> bool:
        """The current run's experiment exited with frames still due and the run is
        not closed yet (_check_exited_run)."""
        return self._exited_pending is not None and self._run_in_progress

    def close_exited_run_now(self, reason: str = "Reset pressed"):
        """Close that run on the next pass of the server loop instead of waiting for
        frames. Its file is kept as it is, as for any exited run. Any thread."""
        self._exited_close_now = str(reason) or "Reset pressed"

    def abort_unanswered_for(self):
        """Seconds since the current run's Abort, while the experiment has not
        answered it; None when no abort is waiting."""
        t0 = self._abort_requested_at
        if t0 is None or not self._run_in_progress or not self._reset_requested:
            return None
        return time.time() - t0

    def abort_again(self) -> bool:
        """Reset pressed again on an Abort nobody answered (at least
        ABORT_AGAIN_MIN_S after it): the experiment is taken to be gone and the run
        is closed on the next pass of the server loop, as the next run start would
        (its file discarded, as for any abort). False when that does not apply.
        Any thread."""
        waited = self.abort_unanswered_for()
        if waited is None or waited < ABORT_AGAIN_MIN_S:
            return False
        if self._run_file.saving:
            # the run is being saved: closing it as aborted would delete the
            # file under the save (it ends the run itself when it is done)
            logger.warning(f"Reset pressed again during run {self._current_run_id}'s save: "
                           f"the save is left to finish; nothing is closed.")
            return False
        self._abort_again = True
        return True

    def _refuse_while_saving(self, tag: str):
        """The reply refusing ``tag`` while the run's asynchronous save runs (None
        when no save runs): closing the run then would cut the save short, and
        closing it as aborted would delete the file being written. Nothing
        changes; the save ends the run itself."""
        if not self._run_file.saving:
            return None
        run_id = self._current_run_id
        logger.warning(f"{tag} for run {run_id} refused: the run is being saved; nothing "
                       f"changed (the save ends the run when it is done).")
        return {"ok": False, "saving": True,
                "error": f"run {run_id} is being saved; {tag} refused, nothing changed"}

    def _check_abort_again(self):
        """Server thread: carry out abort_again()."""
        if not self._abort_again:
            return
        self._abort_again = False
        waited = self.abort_unanswered_for()
        if waited is None:
            return                      # answered, or a new run, in the meantime
        run_id = self._current_run_id
        logger.warning(f"Reset pressed again: run {run_id}'s abort has had no answer for "
                       f"{waited:.0f} s. The run is closed now, as the next run start would "
                       f"close it: its file is discarded (it was aborted). A late message from "
                       f"its experiment is ignored.")
        self._host_end_run("ABORT_RUN", record=False)
        self._finalize_reset_run(notify_gui=False)
        # the aborted run's token is retired: a late message from its experiment
        # (a hung process, not a dead one) gets "superseded" instead of acting here
        self._adopt_run_token(uuid.uuid4().hex)

    def _check_exited_run(self, now=None):
        """Server thread: close the run whose experiment exited with frames still
        due (RUN_EXITED) once no more can come -- its camera thread has finished,
        the camera was never ready (so nothing triggered it), or no frame for
        EXITED_FRAME_GRACE_S -- or at once when a Reset asked for it. Until
        2026-09-29 such a run stayed "in progress" until the next run start (83706)."""
        pending = self._exited_pending
        if pending is None:
            return
        if not self._run_in_progress:
            self._exited_pending = None
            return
        now = time.monotonic() if now is None else float(now)
        if self._exited_close_now:
            how = self._exited_close_now
        elif self._run_file.images_done:
            how = "its camera thread has finished"
        elif not self._cam_ready_event.is_set():
            how = "the camera was never ready, so nothing triggered it"
        else:
            with self._images_lock:
                last = self._frame_times[-1] if self._frame_times else None
            quiet = now - max(pending["t"], last if last is not None else pending["t"])
            if quiet < EXITED_FRAME_GRACE_S:
                return
            how = f"no frame for {quiet:.0f} s after the experiment exited"
        self._close_exited_run(pending["why"], how)

    def _close_exited_run(self, why: str, how: str):
        self._exited_pending = None
        self._exited_close_now = ""
        run_id = self._current_run_id
        received, expected = self._images_received_now(), self._images_expected
        detail = (f"The experiment's process exited without END_RUN ({why}) with "
                  f"{received}/{expected} frames in; run closed ({how}). Its file is left "
                  f"as it was: not saved, not deleted.")
        logger.warning(f"Run {run_id}: {detail}")
        # the window stops the run's camera thread quietly (the file stays)
        self.exited_run_signal.emit(how)
        self._host_end_run("RUN_EXITED", record=False)
        # the writer closes the file with what it has; forgotten: a reset
        # pressed later must not delete an exited run's file
        self._run_file.leave()
        self._run_in_progress = False
        self._abort_requested_at = None
        self._record_outcome("exited", why)
        self._set_run_state("exited", detail)
        self.run_done_signal.emit()

    #: Who may send RESET (its optional ``source``; "person" when absent);
    #: "liveod": liveOD's own abort of a run whose data file is unusable.
    RESET_SOURCES = ("person", "queue", "agent", "liveod")

    def request_reset(self, source: str = "person") -> bool:
        """Set the pending Abort (``_reset_requested``) for the run in progress
        -- the Reset button's, the remote RESET's -- and count it: POLL's
        ``reset_count`` goes up by one and ``last_reset`` says when, for which
        run and from whom.  Already pending: nothing changes and it is not
        counted again (the remote RESET and the window's own reset() both get
        here for one press).  True when it was set now."""
        if self._reset_requested:
            self.note_reset_requested()
            return False
        source = source if source in self.RESET_SOURCES else "person"
        self._reset_count += 1
        self._reset_counts[source] += 1
        self._last_reset = {"at": time.time(), "count": self._reset_count,
                            "run_id": self._current_run_id if self._run_in_progress else None,
                            "source": source}
        self._reset_requested = True
        self.note_reset_requested()
        return True

    def _reset_refusal(self, msg: dict) -> dict | None:
        """A RESET naming a run (``run_id``, or ``run_token``) that is not the
        run in progress is refused, so a late or mistaken Abort never hits the
        next run; an unknown ``source`` is refused."""
        source = msg.get("source")
        if source is not None and source not in self.RESET_SOURCES:
            return {"ok": False, "refused": True,
                    "error": f"unknown RESET source {source!r} (known: "
                             f"{', '.join(self.RESET_SOURCES)})"}
        want = msg.get("run_id")
        if want is not None:
            try:
                want = int(want)
            except (TypeError, ValueError):
                return {"ok": False, "refused": True, "error": f"bad run_id {want!r}"}
            if not self._run_in_progress or want != self._current_run_id:
                current = self._current_run_id if self._run_in_progress else None
                logger.warning(f"RESET for run {want} refused: the run in progress is "
                               f"{current if current is not None else 'none'}.")
                return {"ok": False, "refused": True, "run_id": current,
                        "error": f"RESET names run {want}, but the run in progress is "
                                 f"{current if current is not None else 'none'}"}
        token = msg.get("run_token")
        if token and (not self._run_in_progress or str(token) != self._run_token):
            return {"ok": False, "refused": True,
                    "run_id": self._current_run_id if self._run_in_progress else None,
                    "error": "RESET carries the token of a run that is not in progress"}
        return None

    def _handle_reset(self, msg: dict) -> dict:
        refused = self._reset_refusal(msg)
        if refused is not None:
            return refused
        source = str(msg.get("source") or "person")
        if self.reset_ignored_during_save():
            # the run is complete and being written: nothing to abort, and the
            # window is not asked to reset (it would interrupt the writer)
            return {"ok": True, "ignored": True, "saving": True,
                    "run_id": self._current_run_id, "message": self.reset_ignored_text()}
        if self.exited_run_pending():
            logger.warning("RESET requested by remote viewer: the run's experiment has "
                           "exited; closing the run now (its file is kept).")
            self.close_exited_run_now("Reset pressed in a remote viewer")
            return {"ok": True, "closing_exited_run": True}
        if self.abort_again():
            logger.warning("RESET requested again by remote viewer on an unanswered abort.")
            return {"ok": True, "closing_aborted_run": True}
        logger.warning(f"RESET requested by remote viewer (source: {source}).")
        self.request_reset(source)
        self.reset_signal.emit()
        return {"ok": True, "reset_count": self._reset_count}

    def set_camera_state_provider(self, provider):
        """``provider()`` -> ``{camera_key: {"state", "camera_type", "serial_no"}}``,
        reported in every POLL reply.  The GUI installs its camera bar here."""
        self._camera_state_provider = provider

    def _camera_states(self) -> dict:
        if self._camera_state_provider is None:
            host = self._camera_host
            return host.poll_cameras() if host is not None else {}
        try:
            return dict(self._camera_state_provider())
        except Exception as exc:
            logger.debug(f"camera state provider failed: {exc}")
            return {}

    def _handle_camera_control(self, msg: dict) -> dict:
        camera_key = str(msg.get("camera_key", ""))
        action = str(msg.get("action", "toggle"))
        # the rule is camera_action_allowed's, shared with the window's own button
        ok, reason = self.camera_action_allowed(camera_key, action)
        if not ok:
            if action not in ("open", "close", "toggle") or not camera_key:
                return {"ok": False, "error": reason}
            run_cam = self._current_camera_key if self._current_capture_images else ""
            logger.warning(f"CAMERA_CONTROL rejected ({camera_key} -> {action}): "
                           f"run in progress (uses {run_cam or 'no camera'})")
            return {"ok": False, "error": reason,
                    "run_in_progress": True, "run_camera_key": run_cam}
        if self._run_in_progress:
            run_cam = self._current_camera_key if self._current_capture_images else ""
            logger.info(f"CAMERA_CONTROL: {camera_key} -> close during run "
                        f"{self._current_run_id} (which uses {run_cam or 'no camera'})")
        else:
            logger.info(f"CAMERA_CONTROL: {camera_key} -> {action}")
        host = self._camera_host
        if host is not None:
            # host mode: the host does it on its own threads; callers confirm via
            # POLL's "cameras", as before
            try:
                host.request(camera_key, action, origin="CAMERA_CONTROL")
            except (KeyError, ValueError) as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True}
        # Asynchronous: the GUI thread does the open/close.  Callers confirm via
        # POLL's "cameras" (LiveODClient.wait_camera_state).
        self.camera_control_signal.emit(camera_key, action)
        return {"ok": True}

    def _handle_abort_run(self, msg: dict) -> dict:
        """Experiment has acknowledged the abort — clean up and close out the run."""
        if not self._run_msg_ok(msg):
            # a superseded run's abort must not discard the current run's file
            return self._stale_run_reply("ABORT_RUN", msg)
        refused = self._refuse_while_saving("ABORT_RUN")
        if refused is not None:
            return refused
        logger.warning("Experiment acknowledged: run aborted.")
        # host mode: the aborted run's hold on its camera ends (a RESET alone keeps it)
        self._host_end_run("ABORT_RUN", record=False)
        # If _reset_requested is True, the viewer's _handle_reset already
        # emitted reset_signal and the GUI ran main_window.reset().  Re-emitting
        # reset_signal here would queue a second main_window.reset() that races
        # with the next INIT_RUN: when save_data=False, INIT_RUN is fast enough
        # to return (clearing _reset_requested) before that queued reset() runs,
        # and reset()'s unconditional `_reset_requested = True` then aborts the
        # new run on its first poll.  Only emit reset_signal if no prior RESET
        # set the flag (e.g. RTIOUnderflow path, where the experiment aborts
        # itself without going through _handle_reset).
        gui_already_notified = self._reset_requested
        self._finalize_reset_run(notify_gui=not gui_already_notified)
        self._reset_requested = False
        return {"ok": True}

    def _handle_poll(self, msg: dict) -> dict:
        """Lightweight poll — lets the experiment check for a pending reset, and
        lets tooling read run state without side effects.

        ``run_in_progress`` is the authoritative "is the machine busy" signal;
        ``run_id`` is the current run when in progress, else the last one.
        ``last_shot_age_s`` lets a caller tell a live run from a wedged one.
        Extra keys are ignored by older clients.
        """
        now = time.time()
        last_shot = self._shot_timestamps[-1] if self._shot_timestamps else None
        return {
            "ok": True,
            "reset_requested": self._reset_requested,
            # every Abort set (Reset button or RESET), counted: a watcher sees a
            # quick Reset that the level above hides (cleared by ABORT_RUN or
            # the next INIT_RUN between two polls); last_reset: {at, count,
            # run_id, source "person" | "queue" | "agent"} or None
            "reset_count": self._reset_count,
            "reset_counts": dict(self._reset_counts),      # by source
            "last_reset": dict(self._last_reset) if self._last_reset else None,
            "run_in_progress": self._run_in_progress,
            "run_id": self._current_run_id,
            "n_shots": len(self._shot_timestamps),
            "last_shot_age_s": (now - last_shot) if last_shot else None,
            "init_run_age_s": (now - self._init_run_time) if self._init_run_time else None,
            # newer keys; a client that predates them ignores them
            "run_state": self._run_state,
            "expt_name": self._current_expt_name,
            "n_shots_expected": self._current_n_shots,
            "images_expected": self._images_expected,
            "images_received": self._images_received_now(),
            "grab_failure": self._grab_failure,
            "last_outcome": dict(self._last_outcome),
            # the current (or last) run's client: client_pid, client_host,
            # launcher (None/"" from a client that predates them)
            **self._current_client,
            # an asynchronous END_RUN save is running (the run stays in progress
            # until it ends, whatever run_state says -- an Abort pressed during
            # it shows "aborting"); and the last save's state
            "save_in_progress": self._run_file.saving,
            "save_status": {k: v for k, v in self._run_file.status().items()
                            if k in ("state", "phase", "run_id", "elapsed_s")},
            # which cameras this liveOD holds, and the one the live run is using
            "cameras": self._camera_states(),
            "run_camera_key": (self._current_camera_key
                               if self._run_in_progress and self._current_capture_images
                               else ""),
            # the current (or last) run's camera settings that differ from its
            # camera_params; {} when none do
            "camera_overrides": self.camera_overrides_record(),
            # what the image writers hold that is not on disk yet:
            # {items, bytes, peak_bytes, warn_bytes} (image_writer.QueueGauge)
            "writer_queue": writer_queue_stats(),
        }

    def _handle_get_log(self, msg: dict) -> dict:
        """The server's log for one run, from the in-memory buffer, so a process
        that was not watching can still find out what became of a run.

        ``run_id`` (default: the current or last run; ``"all"`` for every
        record), ``seq`` (the buffer's own run counter, unique for unsaved runs
        that all have run_id 0), ``since`` (epoch seconds, later records only),
        ``min_level`` (a logging level number, default DEBUG), ``limit`` (the
        last N matching records, default 2000). With ``runs=True`` it returns
        the run index instead: one entry per run this process has seen, with
        how each one ended. Read-only.
        """
        buf = get_log_buffer()
        if msg.get("runs"):
            return {"ok": True, "runs": buf.runs(limit=int(msg.get("limit", 50)))}
        run_id = msg.get("run_id", None)
        seq = msg.get("seq", None)
        run = None if run_id == "all" else buf.find_run(run_id=run_id, seq=seq)
        if run_id != "all" and run is None:
            return {"ok": False, "error": f"No run with run_id={run_id!r} seq={seq!r} in this server's log "
                                          f"(the buffer starts when the server does)."}
        current = buf.find_run()
        if run is not None and current is not None and run["seq"] == current["seq"]:
            # the live run: the counters the run index only gets at the end
            run.update(n_shots=len(self._shot_timestamps),
                       images_received=self._images_received_now(),
                       run_state=self._run_state, run_in_progress=self._run_in_progress)
        records = buf.records(
            run_id=run_id, seq=seq, since=msg.get("since", None),
            min_level=int(msg.get("min_level", logging.DEBUG)),
            limit=int(msg.get("limit", 2000)),
        )
        return {"ok": True, "run": run, "records": records}

    def _handle_subscribe_scalars(self, msg: dict) -> dict:
        """Remote viewer subscribes to scalar compute tier."""
        tier = str(msg.get('tier', 'atom_number'))
        if tier not in ('atom_number', 'fits'):
            return {"ok": False, "error": f"Unknown tier: {tier}"}
        with self._scalar_lock:
            self._scalar_subscriber_count[tier] = self._scalar_subscriber_count.get(tier, 0) + 1
        return {"ok": True}

    def _handle_unsubscribe_scalars(self, msg: dict) -> dict:
        """Remote viewer unsubscribes from scalar compute tier."""
        tier = str(msg.get('tier', 'atom_number'))
        with self._scalar_lock:
            count = self._scalar_subscriber_count.get(tier, 0)
            self._scalar_subscriber_count[tier] = max(0, count - 1)
        return {"ok": True}

    def _handle_get_markers(self, msg: dict) -> dict:
        """Markers for ``camera_key`` (default: the current run's camera)."""
        camera_key = str(msg.get('camera_key') or self._current_camera_key)
        return {"ok": True, "camera_key": camera_key, "markers": self.markers.get(camera_key)}

    def _handle_set_markers(self, msg: dict) -> dict:
        """A remote viewer moved, added, reshaped or deleted a marker."""
        camera_key = str(msg.get('camera_key') or self._current_camera_key)
        if not camera_key:
            return {"ok": False, "error": "No camera to put markers on yet"}
        stored = self.markers.set(camera_key, msg.get('markers', []))
        self.markers_changed_signal.emit(camera_key, stored)
        return {"ok": True, "camera_key": camera_key, "markers": stored}

    def _handle_get_adjust_values(self, msg: dict) -> dict:
        """Return current adjust specs and values. Called by remote viewers on connect."""
        with self._adjust_lock:
            return {"ok": True, "specs": list(self._adjust_specs), "values": dict(self._adjust_values)}

    def _handle_set_adjust_value(self, msg: dict) -> dict:
        """Remote viewer writes a new value for an adjustable parameter."""
        key = str(msg.get('key', ''))
        value = msg.get('value')
        if value is None:
            return {"ok": False, "error": "Missing value"}
        with self._adjust_lock:
            if key not in self._adjust_values:
                return {"ok": False, "error": f"Unknown adjust key: {key!r}"}
            spec = next((s for s in self._adjust_specs if s['key'] == key), None)
            if spec:
                value = max(spec['min_val'], min(spec['max_val'], float(value)))
                if spec['dtype'] == 'int':
                    value = float(int(round(value)))
            self._adjust_values[key] = value
        return {"ok": True}

    def _handle_set_adjust_spec(self, msg: dict) -> dict:
        """Remote viewer or GUI updates the min/max/step bounds for an adjust spec."""
        key = str(msg.get('key', ''))
        with self._adjust_lock:
            spec = next((s for s in self._adjust_specs if s['key'] == key), None)
            if spec is None:
                return {"ok": False, "error": f"Unknown adjust key: {key!r}"}
            if 'min_val' in msg:
                spec['min_val'] = float(msg['min_val'])
            if 'max_val' in msg:
                spec['max_val'] = float(msg['max_val'])
            if 'step' in msg:
                spec['step'] = float(msg['step'])
            # Clamp current value to new bounds
            current = self._adjust_values.get(key, spec['min_val'])
            current = max(spec['min_val'], min(spec['max_val'], current))
            self._adjust_values[key] = current
        return {"ok": True}

    def update_adjust_spec(self, key: str, min_val: float, max_val: float, step: float):
        """Update an adjust spec's bounds from the GUI thread (same process as server)."""
        return self._handle_set_adjust_spec(
            {'key': key, 'min_val': min_val, 'max_val': max_val, 'step': step}
        )

    def update_adjust_value(self, key: str, value: float):
        """Update an adjust value from the GUI thread (same process as server)."""
        with self._adjust_lock:
            if key in self._adjust_values:
                self._adjust_values[key] = float(value)

    def get_requested_metrics(self) -> set:
        """Return set of tiers currently requested by any subscriber.

        Called from the Analyzer thread — reads under lock for safety.
        """
        with self._scalar_lock:
            return {
                tier
                for tier, count in self._scalar_subscriber_count.items()
                if count > 0
            }

    def register_scalar_subscription(self, tier: str):
        """In-process subscription (local scalar plot window)."""
        if tier not in ('atom_number', 'fits'):
            return
        with self._scalar_lock:
            self._scalar_subscriber_count[tier] = self._scalar_subscriber_count.get(tier, 0) + 1

    def unregister_scalar_subscription(self, tier: str):
        """In-process unsubscription (local scalar plot window)."""
        with self._scalar_lock:
            count = self._scalar_subscriber_count.get(tier, 0)
            self._scalar_subscriber_count[tier] = max(0, count - 1)
