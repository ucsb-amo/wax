import inspect
import threading
import time
import numpy as np
import names
from PyQt6.QtCore import QThread, pyqtSignal

from waxx.control.cameras import DummyCamera

from waxx.config.timeouts import DATA_SAVER_TIMEOUT, CAMERA_GRAB_TIMEOUT_PER_WARMUP_SHOT
from waxx.util.live_od.camera_nanny import CameraNanny
from waxx.util.live_od.config import get_config
# Everything that touches the run's data file lives in live_od/data. SaveWorker
# is imported here only because it used to be defined here.
from waxx.util.live_od.data.image_writer import ImageWriter, SaveWorker
from waxx.util.live_od.log import get_logger
from waxa.data.server_talk import server_talk as st

logger = get_logger("camera")

from queue import Queue, Empty

# Removed in the move from kexp (2026-09-18): an unused import of the Basler SDK
# (loaded in every process that imported this module), and module-level
# DATA_DIR / RUN_ID_PATH, which were never used and raised TypeError on a PC
# without %data% set.

def nothing():
    pass

# A camera thread waiting for another thread's hold on its camera (the camera
# lock, CameraNanny.camera_lock) checks its own stop this often (s).
CAMERA_LOCK_POLL_S = 0.1

class CameraNotReadyError(ValueError):
    """The handshake got no open camera.  A ValueError because it used to be a
    bare ``ValueError("Camera not ready")``."""


def _takes_on_armed(start_grab) -> bool:
    """Does this driver's ``start_grab`` accept ``on_armed=`` (called once the
    acquisition is running)? Drivers from before 2026-09-26 do not."""
    try:
        params = inspect.signature(start_grab).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "on_armed" or p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params)


def _takes_first_frame_extra(start_grab) -> bool:
    """Does this driver's ``start_grab`` name ``first_frame_extra_s=`` (more time
    for the run's first frame)? Drivers from before 2026-09-28 do not; a bare
    ``**kwargs`` does not count, since it would drop the value unused."""
    try:
        return "first_frame_extra_s" in inspect.signature(start_grab).parameters
    except (TypeError, ValueError):
        return False

class CameraMother(QThread):
    """Legacy stub kept for import compatibility.

    File-watching has been removed.  Run spawning is now driven by
    ``LiveODServer.new_run_signal`` → ``LiveODWindow.spawn_baby``.
    """

    new_camera_baby = pyqtSignal(str, str)

    def __init__(self, output_queue: Queue = None, start_watching=False,
                 manage_babies=False, N_runs: int = None,
                 camera_nanny=None,
                 server_talk=None):
        super().__init__()
        # was a CameraNanny() default argument: one shared instance built at import
        if camera_nanny is None:
            camera_nanny = CameraNanny()

        if server_talk is None:
            self.server_talk = st()
        else:
            self.server_talk = server_talk
        self.server_talk: st

        self.camera_nanny = camera_nanny

        if not output_queue:
            self.output_queue = Queue()
        else:
            self.output_queue = output_queue

    def run(self):
        # No-op: file-watching has been removed.
        pass

class DataHandler(QThread):
    """Takes images off the camera's queue, shows them, and hands them to the
    run's ImageWriter (``self.writer``), which is the only thing here that
    touches the data file."""
    got_image_from_queue = pyqtSignal(np.ndarray)
    save_data_bool_signal = pyqtSignal(int)
    image_type_signal = pyqtSignal(bool)
    done_writing_signal = pyqtSignal()   # emitted after HDF5 handle is closed (or immediately when save_data=False)
    save_failed_signal = pyqtSignal(str) # re-emitted from SaveWorker — data file unusable, run should abort

    def __init__(self, queue: Queue, data_filepath: str,
                 save_data=None, imaging_type=None, camera_key="",
                 camera_params=None,
                 params_payload=None,
                 run_info_payload=None,
                 n_img=None, n_shots=None, n_pwa_per_shot=None):
        """Create a DataHandler.

        Parameters
        ----------
        queue:
            Image queue shared with CameraBaby.
        data_filepath:
            Path to the HDF5 file.  Pass ``""`` when ``save_data=False``.
        save_data:
            Pre-populate ``self.save_data`` so no HDF5 read is needed.
            If *None* the value will be read from the HDF5 in
            ``read_params()`` (legacy behaviour).
        imaging_type:
            Pre-populate ``run_info.imaging_type``.  *None* ⟹ read from HDF5.
        camera_key:
            Camera key string (e.g. ``'xy_basler'``) used to look up
            ``camera_params`` when no HDF5 file exists (``save_data=False``).
        """
        self.data_filepath = data_filepath
        super().__init__()
        self.queue = queue
        self.writer = ImageWriter(data_filepath)

        from waxa.dummy.camera_params import CameraParams
        from waxa.data import RunInfo
        # A holder for N_img / N_shots / N_pwa_per_shot plus whatever the run's
        # params payload sets; the lab may supply its own params class.
        self.params = get_config().params_factory()
        self.camera_params = CameraParams()
        self.run_info = RunInfo()
        self.interrupted = False
        # Set by the CameraBaby when its grab is over, however it ended.  The
        # dispatch loop then drains the queue and exits instead of waiting for
        # a frame index that will never arrive.
        self._grab_over = threading.Event()
        self.images_received = 0
        self._camera_key_hint = ""

        # Pre-populate from constructor arguments so HDF5 reads are
        # optional (required when save_data=False / no file exists).
        if save_data is not None:
            self.save_data = bool(save_data)
            self.run_info.save_data = int(bool(save_data))
        if imaging_type is not None:
            self.run_info.imaging_type = imaging_type
        if camera_key:
            self._camera_key_hint = camera_key

        self._camera_params_payload = None
        if camera_params:
            self._camera_params_payload = camera_params
        self._params_payload = params_payload or {}
        self._run_info_payload = run_info_payload or {}

        # Pre-populate image count params when provided (critical for
        # save_data=False runs where read_params() skips the HDF5 read).
        if n_img is not None:
            self.params.N_img = int(n_img)
        if n_shots is not None:
            self.params.N_shots = int(n_shots)
        if n_pwa_per_shot is not None:
            self.params.N_pwa_per_shot = int(n_pwa_per_shot)

    def get_save_data_bool(self, save_data_bool):
        self.save_data = save_data_bool

    def get_img_number(self, N_img, N_shots, N_pwa_per_shot):
        self.N_img = N_img
        self.N_shots = N_shots
        self.N_pwa_per_shot = N_pwa_per_shot

    def run(self):
        self.write_image_to_dataset()

    def grab_finished(self):
        """The CameraBaby's grab is over (all frames in, a camera timeout, an
        error). Once the queue is drained the dispatch loop exits and the
        writer closes the file. Safe from any thread."""
        self._grab_over.set()

    def read_params(self):
        """Populate camera/run-info attrs, then emit configuration signals.

        All three groups (camera_params, params, run_info) are taken directly
        from the in-memory payloads sent by the experiment client, bypassing
        the HDF5 file entirely.  This eliminates the race where the DataHandler
        opens the file before the background file-creation thread has finished
        writing a group.

        Falls back to reading from HDF5 only when no payload is available
        (legacy path).
        """
        have_payload = bool(
            getattr(self, '_camera_params_payload', None)
            or getattr(self, '_params_payload', None)
            or getattr(self, '_run_info_payload', None)
        )

        if not have_payload and getattr(self, 'save_data', True) and self.data_filepath:
            # Legacy path: no in-memory state — read everything from HDF5.
            self.writer.read_groups(self.camera_params, self.params, self.run_info)
        else:
            # Apply all three payloads from memory.
            if getattr(self, '_camera_params_payload', None):
                for key, val in self._camera_params_payload.items():
                    try:
                        setattr(self.camera_params, key, val)
                    except Exception as exc:
                        logger.warning(f"read_params: could not set camera_params.{key}={val!r}: {exc}")
            elif hasattr(self, '_camera_key_hint') and self._camera_key_hint:
                # No camera_params payload — ask the lab's camera table by key.
                found = get_config().resolve_camera_params(self._camera_key_hint)
                if found is not None:
                    self.camera_params = found

            for key, val in getattr(self, '_params_payload', {}).items():
                try:
                    setattr(self.params, key, val)
                except Exception as exc:
                    logger.warning(f"read_params: could not set params.{key}={val!r}: {exc}")

            for key, val in getattr(self, '_run_info_payload', {}).items():
                try:
                    setattr(self.run_info, key, val)
                except Exception as exc:
                    logger.warning(f"read_params: could not set run_info.{key}={val!r}: {exc}")

        self.image_type_signal.emit(self.run_info.imaging_type)
        self.save_data_bool_signal.emit(self.run_info.save_data)

    def write_image_to_dataset(self):
        # Start the writer immediately — it waits for the HDF5 file in its own
        # thread while the dispatch loop below runs unblocked, and reports
        # through this DataHandler's done / failed signals.
        try:
            if self.save_data:
                self.writer.start(
                    n_img=self.N_img,
                    check_interrupt_method=self.break_check,
                    done_signal=self.done_writing_signal,
                    failed_signal=self.save_failed_signal,
                )

            while True:
                if self.interrupted:
                    break
                try:
                    img, _, idx = self.queue.get(block=False)
                    img_t = time.time()
                    self.images_received += 1
                    self.got_image_from_queue.emit(img)   # immediate display / OD plot
                    if self.save_data:
                        self.writer.put(img, idx, img_t)   # non-blocking hand-off
                    if idx == (self.N_img - 1):
                        break
                except Empty:
                    if self._grab_over.is_set():
                        # The grab ended and the queue is drained: nothing more
                        # is coming.  Waiting on for the last index kept the
                        # writer's file handle open indefinitely after a camera
                        # timeout, and END_RUN's save then ran on top of it
                        # (run 80704, 2026-09-24).
                        break
                    self.msleep(1)
                except Exception as e:
                    logger.exception(f"DataHandler: unexpected error in image dispatch loop: {e}")
                    self.msleep(1)
        except Exception as e:
            logger.exception(f"DataHandler: write_image_to_dataset failed: {e}")

        if self.save_data and self.writer.started:
            # Propagate interruption so the writer drains without writing.  It
            # closes the file and emits done_writing_signal; don't emit it here.
            self.writer.finish(self.interrupted)
        else:
            self.done_writing_signal.emit()

    def break_check(self):
        return self.interrupted

    # DataHandler used to inherit these from Scribe; kept for old callers.
    def wait_for_data_available(self, *args, **kwargs):
        return self.writer.wait_for_data_available(*args, **kwargs)

    def remove_incomplete_data(self, delete_data_bool=True):
        return self.writer.discard(delete_data_bool)


class CameraBaby(QThread):
    image_captured = pyqtSignal(int)
    camera_connect = pyqtSignal(str)
    camera_grab_start = pyqtSignal(int,int,int)
    save_data_bool_signal = pyqtSignal(int)
    image_type_signal = pyqtSignal(bool)
    honorable_death_signal = pyqtSignal()
    dishonorable_death_signal = pyqtSignal()
    grab_failed_signal = pyqtSignal(str)    # the grab ended early; why (frames so far are kept)
    done_signal = pyqtSignal()
    break_signal = pyqtSignal()
    cam_status_signal = pyqtSignal(int)
    # camera_key, {field: (requested, applied)}: values the driver could not set
    # as the run asked (emitted once, after the run's settings were applied)
    camera_overrides_signal = pyqtSignal(str, object)

    def __init__(self,data_handler:DataHandler,
                 name,output_queue:Queue,
                 camera_nanny:CameraNanny):
        super().__init__()

        self.name = name
        self.camera_nanny = camera_nanny
        self.camera = DummyCamera()
        self.queue = output_queue
        self.death = self.dishonorable_death
        self.data_handler = data_handler
        self.interrupted = False
        self.dead = False
        # This baby's own stop, for when a newer run has replaced its run
        # (request_stop). Unlike an interrupt it discards nothing: the old run
        # may already be saved.
        self._stop = threading.Event()
        # what the thread says when it goes (request_stop)
        self._stop_reason = "its run was replaced by a newer one"
        self._ready_reported = False
        # the camera's lock while this thread applies its run's settings and
        # grabs (create_camera .. the end of grab_loop), else None
        self._camera_lock = None
        # why the run's settings could not be applied (the nanny's report)
        self._not_ready_reason = ""

    def request_stop(self, reason: str = "its run was replaced by a newer one"):
        """A newer run took over, or liveOD is shutting down: end the grab (or the
        wait for the camera) soon, and quietly -- no file is touched and nothing
        is reported. ``reason`` is what the thread says as it goes. Safe from any
        thread."""
        self._stop_reason = str(reason)
        self._stop.set()

    def run(self):
        # How the grab ended, other than by an interrupt or all frames in.  A
        # failure once frames may have arrived (timeout, driver error) keeps
        # the file: END_RUN saves what came and marks the run incomplete.  A
        # camera that never opened has nothing to keep, and the experiment
        # will not send END_RUN for it, so that path still deletes the file.
        failure = ""
        keep_file = False
        try:
            self.cam_status_signal.emit(0)
            logger.debug(f"{self.name}: I am born!")
            self.data_handler.read_params()
            try:
                self.handshake()
                self.grab_loop()
            finally:
                # held from applying the settings to the end of the grab
                self._release_camera_lock()
        except TimeoutError as e:
            # An expected failure (camera never triggered, experiment aborted
            # or stalled), so one line, no traceback.  Camera drivers raise the
            # builtin TimeoutError with the details in the message.
            failure, keep_file = f"camera timed out: {e}", True
            logger.warning(f"{self.name}: camera timed out. {e} Ending this run's grab. "
                           f"The frames that arrived are kept; the run will be saved as incomplete.")
        except CameraNotReadyError as e:
            # Also expected, and CameraNanny has already logged why: the run
            # was aborted while waiting for the camera, or opening/configuring
            # it failed.  One line, no traceback.
            failure = f"camera not ready: {e}"
            if self.interrupted or self._stop.is_set():
                logger.info(f"{self.name}: run aborted before the camera was ready.")
            else:
                logger.warning(f"{self.name}: {e}; not starting this run's grab. "
                               f"Check the camera connection and the messages above.")
                # This run's camera will never be ready: the server's
                # WAIT_CAM_READY now answers "camera failed before it was
                # ready: <this>" at once, instead of the experiment waiting
                # out its whole ready timeout (90 s) for nothing.
                self.grab_failed_signal.emit(failure)
        except Exception as e:
            failure, keep_file = f"{type(e).__name__}: {e}", True
            logger.exception(f"CameraBaby {self.name}: fatal error: {e}")
        # However it ended, the dispatch loop must not wait for frames that
        # will never come: it drains the queue, and the writer closes the file.
        self.data_handler.grab_finished()
        if self.interrupted and self.death is not self.honorable_death:
            logger.warning('Grab loop interrupted, shutting down.')
            self.death = self.dishonorable_death
        elif self._stop.is_set() and self.death is not self.honorable_death:
            # replaced by a newer run: its file is not this baby's to discard
            self.death = self.superseded_death
        elif failure and keep_file and self.death is not self.honorable_death:
            self.death = lambda: self.failed_death(failure)
        try:
            self.death()
        except Exception as e:
            logger.exception(f"CameraBaby {self.name}: error in death handler: {e}")
        finally:
            if self.interrupted:
                self.dead = True
            self.done_signal.emit()

    def handshake(self):
        """Connect the camera and apply the run's settings.

        Status codes (cam_status_signal)
        ------------
        0  baby born
        1  camera opened
        2  camera ready (triggers LiveODServer._cam_ready_event via
           a DirectConnection in LiveODWindow.spawn_baby)
        3  (legacy: ready-ack; kept for status-light compatibility)

        2 and 3 are no longer sent from here but from grab_loop, once the
        driver reports its acquisition running (``_report_ready``). Sent here,
        before the grab had started, a trigger in between was lost and every
        later frame landed one slot early.
        """
        self.create_camera()
        self.cam_status_signal.emit(1)
        if self.camera is None or not self.camera.is_opened():
            key = self.data_handler.camera_params.key
            if isinstance(key, bytes):
                key = key.decode()
            if self._not_ready_reason:
                raise CameraNotReadyError(f"Camera {key}: the run's settings were not "
                                          f"applied ({self._not_ready_reason})")
            raise CameraNotReadyError(f"Camera {key} is not open")

    def _report_ready(self):
        """Status 2 (the server's camera-ready event), then 3 (status lights).
        The driver's ``on_armed``: called once acquisition is running, before
        the first frame is waited for. Once per grab, however often it is called."""
        if self._ready_reported:
            return
        self._ready_reported = True
        self.cam_status_signal.emit(2)
        self.cam_status_signal.emit(3)

    def create_camera(self):
        # this baby's own break_check: the nanny's shared flag is cleared for
        # every new run, so it cannot stop a baby that a newer run replaced
        self.camera = self.camera_nanny.persistent_get_camera(self.data_handler.camera_params,
                                                              break_check=self.break_check)
        # From here to the end of the grab this thread holds the camera's lock
        # (released in run): a superseded run's thread still applying its
        # settings or grabbing is waited for, and a stopped thread applies
        # nothing -- its setters cannot interleave with the next run's.
        if not self._hold_camera_lock():
            self.camera = DummyCamera()       # stopped: nothing was applied
            return
        report = {}
        self.camera = self.camera_nanny.update_params(self.camera, self.data_handler.camera_params,
                                                      report=report)
        if self.break_check():
            # Superseded while applying: grab nothing. The next run's thread
            # applies its own settings once this one lets go of the lock.
            self.camera = DummyCamera()
            return
        self._not_ready_reason = str(report.get("error") or "")
        camera_select = self.data_handler.camera_params.key
        if type(camera_select) == bytes:
            camera_select = camera_select.decode()
        if report.get("clamps"):
            self.camera_overrides_signal.emit(camera_select, dict(report["clamps"]))
        self.camera_connect.emit(camera_select)

    def _hold_camera_lock(self) -> bool:
        """Take the camera's lock (the nanny's ``camera_lock``), waiting in
        CAMERA_LOCK_POLL_S slices while another thread holds it. False if this
        thread was stopped first (then nothing is held). Nothing to hold for a
        DummyCamera, or with a nanny that has no ``camera_lock`` (the camera
        host's: its worker is the only thread on the camera)."""
        if self.break_check():
            return False
        get_lock = getattr(self.camera_nanny, "camera_lock", None)
        lock = None
        if get_lock is not None and not isinstance(self.camera, DummyCamera):
            lock = get_lock(self.camera, self.data_handler.camera_params)
        if lock is None:
            return True
        said = False
        while not lock.acquire(timeout=CAMERA_LOCK_POLL_S):
            if self.break_check():
                return False
            if not said:
                said = True
                logger.info(f"{self.name}: waiting for another camera thread (a previous "
                            f"run's) to let go of the camera before applying this run's "
                            f"settings.")
        if self.break_check():
            lock.release()
            return False
        self._camera_lock = lock
        return True

    def _release_camera_lock(self):
        lock, self._camera_lock = self._camera_lock, None
        if lock is not None:
            try:
                lock.release()
            except RuntimeError as e:
                logger.warning(f"{self.name}: releasing the camera lock failed: {e}")

    def honorable_death(self):
        try:
            self.camera.stop_grab()
        except:
            pass
        logger.debug(f"{self.name}: All images captured.")
        time.sleep(0.1)
        self.honorable_death_signal.emit()
        self.cam_status_signal.emit(-1)
        return True
    
    def dishonorable_death(self,delete_data=True):
        try:
            self.camera.stop_grab()
        except:
            pass
        self.data_handler.writer.discard(delete_data)
        time.sleep(0.1)
        self.dishonorable_death_signal.emit()
        self.cam_status_signal.emit(-1)
        return True

    def superseded_death(self):
        """A newer run replaced this baby's run (request_stop): just go. Nothing
        is discarded -- the old run may be saved already -- and its signals to
        the server were disconnected when the new run started.

        No stop_grab: by the time this runs, the baby's grab (if it started one)
        has returned, and both drivers stop their own acquisition in start_grab's
        ``finally``, so a stop here would never have anything of this baby's to
        stop. The grab lock makes it a no-op while the new run's baby holds the
        camera (since 2026-09-26 from its update_params on, see camera_lock), but
        a stop would still be a second thread calling the camera's SDK."""
        logger.warning(f"{self.name}: {self._stop_reason}; grab ended.")
        self.cam_status_signal.emit(-1)
        return True

    def failed_death(self, reason: str):
        """The grab ended early with frames possibly already written: stop the
        camera, leave the file to END_RUN (which saves it marked incomplete), and
        tell the server why. Until 2026-09-24 this path tried to delete the file,
        failed because the writer still held it, and END_RUN then marked the
        run complete regardless."""
        try:
            self.camera.stop_grab()
        except:
            pass
        time.sleep(0.1)
        self.grab_failed_signal.emit(reason)
        self.dishonorable_death_signal.emit()
        self.cam_status_signal.emit(-1)
        return True

    def grab_loop(self):
        N_img = int(self.data_handler.params.N_img)
        N_shots = int(self.data_handler.params.N_shots)
        N_pwa_per_shot = int(self.data_handler.params.N_pwa_per_shot)
        self.camera_grab_start.emit(N_img,N_shots,N_pwa_per_shot)
        extra = self._first_frame_extra_kwargs()
        if _takes_on_armed(self.camera.start_grab):
            # ready is reported by the driver, once acquisition is running
            self.camera.start_grab(N_img,output_queue=self.queue,
                        check_interrupt_method=self.break_check,
                        on_armed=self._report_ready, **extra)
        else:
            logger.warning(f"{self.name}: this camera driver's start_grab() has no on_armed "
                           f"callback, so the run is told the camera is ready before its "
                           f"acquisition has started; a trigger in between is lost and "
                           f"shifts every later frame (update the driver).")
            self._report_ready()
            self.camera.start_grab(N_img,output_queue=self.queue,
                        check_interrupt_method=self.break_check, **extra)
        if not self.interrupted and not self._stop.is_set():
            self.death = self.honorable_death

    def _first_frame_extra_kwargs(self) -> dict:
        """``{"first_frame_extra_s": s}`` for a run with warm-up shots, else {}.

        The camera arms at INIT_RUN, but a run's N warm-up shots
        (params.N_warmup_shots) come before its first imaged shot, so the
        driver's first-frame timeout alone ends such runs before their first
        frame. Each warm-up adds CAMERA_GRAB_TIMEOUT_PER_WARMUP_SHOT."""
        try:
            n_warmup = int(getattr(self.data_handler.params, "N_warmup_shots", 0) or 0)
        except (TypeError, ValueError):
            n_warmup = 0
        if n_warmup <= 0:
            return {}
        extra_s = n_warmup * CAMERA_GRAB_TIMEOUT_PER_WARMUP_SHOT
        if not _takes_first_frame_extra(self.camera.start_grab):
            logger.warning(f"{self.name}: {n_warmup} warm-up shot(s), but this camera driver's "
                           f"start_grab() has no first_frame_extra_s, so the first frame gets "
                           f"only the driver's usual timeout (update the driver).")
            return {}
        logger.info(f"{self.name}: {n_warmup} warm-up shot(s): the first frame may take "
                    f"{extra_s:.0f} s longer than usual.")
        return {"first_frame_extra_s": extra_s}

    def break_check(self):
        return self.interrupted or self._stop.is_set()
