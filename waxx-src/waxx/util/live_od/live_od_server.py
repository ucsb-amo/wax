"""
ZMQ REP server running inside the liveOD GUI process.

Receives INIT_RUN / WAIT_CAM_READY / SHOT_COMPLETE / END_RUN messages
from the experiment client (possibly on a different machine) and
coordinates HDF5 file creation, camera spawning, and final data saving.

The server runs as a QThread to stay compatible with PyQt6 event dispatch
on the GUI machine. All HDF5 I/O stays on the server side; the experiment
client never needs the data drive mounted.
"""

import logging
import pickle
import os
import threading
import time

from waxx.config.timeouts import DATA_SAVER_TIMEOUT

import zmq
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from beacon.discovery.server import NetServer
from waxx.util.comms_server.hardware_id import scoped_server_id
from waxx.util.live_od.config import get_config
# Everything the server does to the run's data file (reserve, save, delete) is
# in live_od/data/run_file.py; this module keeps the protocol and the run state.
from waxx.util.live_od.data.run_file import RunFile, RunFileSaveError
from waxx.util.live_od.log import get_logger, get_log_buffer
from waxx.util.live_od.marker_store import MarkerStore

logger = get_logger("server")


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
        # pins on the image, per camera; the file is not touched until first used
        self.markers = MarkerStore(marker_path)

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

    def on_cam_ready(self):
        """Set camera-ready event.

        Connect this to ``CameraBaby.cam_status_signal`` filtered to
        status == 2 using ``Qt.ConnectionType.DirectConnection`` so the
        threading.Event is set from the CameraBaby thread immediately.
        """
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

    def on_data_handler_done(self):
        """Set the data-handler-done event.

        Connect to ``DataHandler.done_writing_signal`` with
        ``Qt.ConnectionType.DirectConnection`` so the event is set from
        the DataHandler thread immediately after it closes the HDF5 file.
        """
        self._run_file.writer_finished()

    def on_image_received(self, *_):
        """One frame came off the camera queue. Connect to
        ``DataHandler.got_image_from_queue`` with a DirectConnection."""
        with self._images_lock:
            self._images_received += 1

    def on_grab_failed(self, reason: str):
        """The CameraBaby's grab ended early (timeout, camera error). Connect to
        ``CameraBaby.grab_failed_signal`` with a DirectConnection. END_RUN puts
        the reason in the file."""
        self._grab_failure = str(reason)
        get_log_buffer().update_run(grab_failure=self._grab_failure)

    def _images_received_now(self) -> int:
        with self._images_lock:
            return self._images_received

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
                try:
                    raw = socket.recv()
                except zmq.Again:
                    continue          # poll timeout — loop to check _running

                tag = "<unknown>"
                try:
                    msg = pickle.loads(raw)
                    tag = msg.get("tag", "")
                    if tag == "INIT_RUN":
                        reply = self._handle_init_run(msg)
                    elif tag == "WAIT_CAM_READY":
                        reply = self._handle_wait_cam_ready(msg)
                    elif tag == "SHOT_COMPLETE":
                        reply = self._handle_shot_complete(msg)
                    elif tag == "END_RUN":
                        reply = self._handle_end_run(msg)
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

                socket.send(pickle.dumps(reply))
        finally:
            socket.close()
            context.term()

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
        }
        get_log_buffer().end_run(
            outcome, detail,
            n_shots=len(self._shot_timestamps),
            images_expected=self._images_expected,
            images_received=self._images_received_now(),
        )

    def _handle_init_run(self, msg: dict) -> dict:
        # If the previous run was reset but the experiment process was killed
        # before sending ABORT_RUN, finalize that reset now. This keeps the
        # next run start non-blocking and stateless.  The GUI was already
        # notified by the original _handle_reset call, so don't re-emit
        # reset_signal — that would race with this INIT_RUN and cause the
        # new run to abort on its first poll.
        if self._reset_requested:
            self._finalize_reset_run(notify_gui=False)

        save_data = bool(msg.get("save_data", False))
        capture_images = bool(msg.get("capture_images", False))
        camera_key = str(msg.get("camera_key", ""))
        imaging_type = int(msg.get("imaging_type", 0))
        self._current_camera_key = camera_key

        # For Basler cameras: if a previous baby is still running its grab
        # loop (e.g. after a reset), clear the event so WAIT_CAM_READY will
        # block until that grab loop fully exits and on_basler_baby_done() fires.
        # ("Basler": which cameras need this is the lab's call --
        # LiveODConfig.camera_needs_grab_drain; by default any key containing "basler".)
        if get_config().camera_needs_grab_drain(camera_key) and capture_images:
            with self._basler_lock:
                self._basler_babies_live += 1
                stale = self._basler_babies_live - 1
                if stale > 0:
                    self._basler_prev_grab_done_event.clear()
            if stale > 0:
                logger.warning(f"INIT_RUN: {stale} Basler grab loop(s) not yet done — WAIT_CAM_READY will block until they exit.")

        camera_params = msg.get('camera_params', {})
        self._cam_ready_event.clear()
        self._current_capture_images = capture_images
        # Gates the END_RUN save on the image writer finishing (a run without a
        # camera has none).
        self._run_file.begin(save_data, has_writer=capture_images)

        run_id = 0
        filepath = ""

        if save_data:
            # Synchronous on purpose, and a small write: see RunFile.reserve.
            _t_create = time.time()
            try:
                run_id, filepath = self._run_file.reserve(msg)
            except Exception as exc:
                logger.exception(f"INIT_RUN: could not create data file: {exc}")
                # No CameraBaby will be spawned, so give back the Basler
                # slot claimed above — otherwise the count never returns to zero
                # and every later run blocks in WAIT_CAM_READY.
                self._release_basler_slot()
                self._run_in_progress = False
                self._set_run_state("error", f"Data file creation failed: {exc}")
                return {"ok": False, "error": f"Data file creation failed: {exc}"}
            _dt_create = time.time() - _t_create
            if _dt_create > 2.0:
                logger.warning(f"Data file creation took {_dt_create:.1f} s — is the data drive slow?")

        self._current_run_id = run_id
        self._reset_requested = False
        self._run_in_progress = True
        self._shot_timestamps = []       # reset per-run timestamp list
        self._init_run_time = time.time()
        self._shot_durations = []  # reset rolling average for new run

        n_img = int(msg.get('params', {}).get('N_img', 1))
        n_shots = int(msg.get('N_shots_with_repeats', 1))
        n_pwa = int(msg.get('N_pwa_per_shot', 1))
        self._current_n_shots = n_shots
        with self._images_lock:
            self._images_received = 0
        self._images_expected = n_img if capture_images else 0
        self._grab_failure = ""
        self._frame_deficit_warned = 0

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

        self.run_started_signal.emit(run_id, list(run_info_payload.get('xvarnames', [])))
        self.new_run_signal.emit(filepath, camera_key, capture_images, save_data, imaging_type, n_img, n_shots, n_pwa, camera_params, params_payload, run_info_payload)
        self._run_state = "idle"    # so the new run's first state is always emitted
        self._set_run_state("waiting_camera" if capture_images else "running", camera_key)
        logger.info(
            f"INIT_RUN: run_id={run_id}, {self._current_expt_name}, "
            f"{n_shots} shots, save={save_data}, "
            f"camera={camera_key if capture_images else 'none'}"
        )
        return {"ok": True, "run_id": run_id, "filepath": filepath}

    def _handle_wait_cam_ready(self, msg: dict) -> dict:
        """Wait up to ``timeout`` s for the camera.

        The client asks in short slices rather than one long wait: this loop is
        single-threaded, so while it blocks here a RESET from the remote viewer
        cannot even be received, and a reset camera never becomes ready. Every
        reply carries ``reset_requested`` so the experiment can abort at once;
        ``timed_out`` marks "not ready yet" apart from a real failure.
        """
        if self._reset_requested:
            return {"ok": True, "ready": False, "reset_requested": True}

        timeout = float(msg.get("timeout", 60.0))
        deadline = time.time() + timeout

        # For Basler cameras: wait for the previous grab loop to fully exit
        # before reporting camera-ready to the experiment.  This prevents the
        # experiment from arming the hardware while the old grab is still
        # blocking in RetrieveResult() and the camera is not yet free.
        if get_config().camera_needs_grab_drain(self._current_camera_key):
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
        remaining = deadline - time.time()
        ready = self._cam_ready_event.wait(timeout=max(0.0, remaining))
        if not ready:
            return {"ok": False, "ready": False, "timed_out": True,
                    "reset_requested": self._reset_requested,
                    "error": "Camera ready timeout"}
        if not self._reset_requested:
            self._set_run_state("running")
        return {"ok": True, "ready": True, "reset_requested": self._reset_requested}

    def _handle_shot_complete(self, msg: dict) -> dict:
        now = time.time()
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
        return {"ok": True, "reset_requested": self._reset_requested, "adjust_values": adjust_values}

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
        if self._reset_requested:
            logger.warning(f"END_RUN: run {self._current_run_id} was reset — discarding data.")
            # As before the move: this path does not wait for the image writer.
            self._run_file.discard(wait_for_writer=False)
            self._reset_requested = False
            self._run_in_progress = False
            self._record_outcome("discarded", "reset")
            self._set_run_state("aborted")
            self.run_done_signal.emit()
            return {"ok": True}
        incomplete = None
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
        self.run_done_signal.emit()
        reply = {"ok": True}
        if incomplete:
            reply["incomplete"] = dict(incomplete)
        return reply

    def note_reset_requested(self):
        """The GUI's own abort button was pressed (the remote path is _handle_reset)."""
        if self._run_in_progress:
            self._set_run_state("aborting")

    def _handle_reset(self, msg: dict) -> dict:
        logger.warning("RESET requested by remote viewer.")
        self._reset_requested = True
        self.note_reset_requested()
        self.reset_signal.emit()
        return {"ok": True}

    def set_camera_state_provider(self, provider):
        """``provider()`` -> ``{camera_key: {"state", "camera_type", "serial_no"}}``,
        reported in every POLL reply.  The GUI installs its camera bar here."""
        self._camera_state_provider = provider

    def _camera_states(self) -> dict:
        if self._camera_state_provider is None:
            return {}
        try:
            return dict(self._camera_state_provider())
        except Exception as exc:
            logger.debug(f"camera state provider failed: {exc}")
            return {}

    def _handle_camera_control(self, msg: dict) -> dict:
        camera_key = str(msg.get("camera_key", ""))
        action = str(msg.get("action", "toggle"))
        if action not in ("open", "close", "toggle"):
            return {"ok": False, "error": f"Unknown camera action: {action}"}
        if not camera_key:
            return {"ok": False, "error": "Missing camera_key"}
        # During a run: closing the camera a CameraBaby is driving crashes its
        # grab loop (dishonorable_death), and opening any camera blocks the GUI
        # thread (Andor cooler init takes seconds) and stalls the queued
        # SHOT_COMPLETE / RESET signals -- both refused.  Closing a camera the
        # run is *not* using is allowed: that is how a tool borrows an idle
        # Basler (frame grab through the beacon server) while a run on another
        # camera goes on.  A no-camera run uses none of them.
        if self._run_in_progress:
            run_cam = self._current_camera_key if self._current_capture_images else ""
            if action == "close" and camera_key != run_cam:
                logger.info(f"CAMERA_CONTROL: {camera_key} -> close during run "
                            f"{self._current_run_id} (which uses {run_cam or 'no camera'})")
            else:
                logger.warning(f"CAMERA_CONTROL rejected ({camera_key} -> {action}): "
                               f"run in progress (uses {run_cam or 'no camera'})")
                return {"ok": False,
                        "error": f"Camera control rejected: run {self._current_run_id} in "
                                 f"progress (uses {run_cam or 'no camera'}); only closing a "
                                 f"camera the run does not use is allowed",
                        "run_in_progress": True, "run_camera_key": run_cam}
        else:
            logger.info(f"CAMERA_CONTROL: {camera_key} -> {action}")
        # Asynchronous: the GUI thread does the open/close.  Callers confirm via
        # POLL's "cameras" (LiveODClient.wait_camera_state).
        self.camera_control_signal.emit(camera_key, action)
        return {"ok": True}

    def _handle_abort_run(self, msg: dict) -> dict:
        """Experiment has acknowledged the abort — clean up and close out the run."""
        logger.warning("Experiment acknowledged: run aborted.")
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
            # which cameras this liveOD holds, and the one the live run is using
            "cameras": self._camera_states(),
            "run_camera_key": (self._current_camera_key
                               if self._run_in_progress and self._current_capture_images
                               else ""),
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
