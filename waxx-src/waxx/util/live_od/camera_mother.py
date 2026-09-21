import time
import numpy as np
import names
from PyQt6.QtCore import QThread, pyqtSignal

from waxx.control.cameras import DummyCamera

from waxx.config.timeouts import DATA_SAVER_TIMEOUT
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

class CameraNotReadyError(ValueError):
    """The handshake got no open camera.  A ValueError because it used to be a
    bare ``ValueError("Camera not ready")``."""

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
                    self.got_image_from_queue.emit(img)   # immediate display / OD plot
                    if self.save_data:
                        self.writer.put(img, idx, img_t)   # non-blocking hand-off
                    if idx == (self.N_img - 1):
                        break
                except Empty:
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
    done_signal = pyqtSignal()
    break_signal = pyqtSignal()
    cam_status_signal = pyqtSignal(int)

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

    def run(self):
        try:
            self.cam_status_signal.emit(0)
            logger.debug(f"{self.name}: I am born!")
            self.data_handler.read_params()
            self.handshake()
            self.grab_loop()
        except TimeoutError as e:
            # An expected failure (camera never triggered, experiment aborted
            # or stalled), so one line, no traceback.  Camera drivers raise the
            # builtin TimeoutError with the details in the message.
            logger.warning(f"{self.name}: camera timed out. {e} "
                           f"Ending this run's grab; its incomplete data file is discarded.")
        except CameraNotReadyError as e:
            # Also expected, and CameraNanny has already logged why: the run
            # was aborted while waiting for the camera, or opening/configuring
            # it failed.  One line, no traceback.
            if self.interrupted:
                logger.info(f"{self.name}: run aborted before the camera was ready.")
            else:
                logger.warning(f"{self.name}: {e}; not starting this run's grab. "
                               f"Check the camera connection and the messages above.")
        except Exception as e:
            logger.exception(f"CameraBaby {self.name}: fatal error: {e}")
        if self.interrupted and self.death is not self.honorable_death:
            logger.warning('Grab loop interrupted, shutting down.')
            self.death = self.dishonorable_death
        try:
            self.death()
        except Exception as e:
            logger.exception(f"CameraBaby {self.name}: error in death handler: {e}")
        finally:
            if self.interrupted:
                self.dead = True
            self.done_signal.emit()

    def handshake(self):
        """Connect camera and signal readiness via cam_status_signal.

        Status codes
        ------------
        0  baby born
        1  camera opened
        2  camera ready (triggers LiveODServer._cam_ready_event via
           a DirectConnection in LiveODWindow.spawn_baby)
        3  (legacy: ready-ack; kept for status-light compatibility)
        """
        self.create_camera()
        self.cam_status_signal.emit(1)
        if self.camera is None or not self.camera.is_opened():
            key = self.data_handler.camera_params.key
            if isinstance(key, bytes):
                key = key.decode()
            raise CameraNotReadyError(f"Camera {key} is not open")
        # Status 2 → triggers server._cam_ready_event via DirectConnection
        self.cam_status_signal.emit(2)
        # Status 3 kept for the status-lights widget
        self.cam_status_signal.emit(3)

    def create_camera(self):
        self.camera = self.camera_nanny.persistent_get_camera(self.data_handler.camera_params)
        self.camera = self.camera_nanny.update_params(self.camera,self.data_handler.camera_params)
        camera_select = self.data_handler.camera_params.key
        if type(camera_select) == bytes: 
            camera_select = camera_select.decode()
        self.camera_connect.emit(camera_select)

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

    def grab_loop(self):
        N_img = int(self.data_handler.params.N_img)
        N_shots = int(self.data_handler.params.N_shots)
        N_pwa_per_shot = int(self.data_handler.params.N_pwa_per_shot)
        self.camera_grab_start.emit(N_img,N_shots,N_pwa_per_shot)
        self.camera.start_grab(N_img,output_queue=self.queue,
                    check_interrupt_method=self.break_check)
        if not self.interrupted:
            self.death = self.honorable_death

    def break_check(self):
        return self.interrupted
