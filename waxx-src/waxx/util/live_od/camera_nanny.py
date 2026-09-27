import numpy as np
import threading
import time

from waxx.control import AndorEMCCD, BaslerUSB, DummyCamera
from waxx.control.cameras.camera_param_classes import CameraParams
from waxx.util.live_od.log import get_logger

logger = get_logger("camera")

CHECK_EVERY = 0.2
CHECK_PERIOD = 2.0
N_NOTIFY = CHECK_PERIOD // CHECK_EVERY

def nothing():
    pass

# A run's camera fields and what an old payload (from before they existed) means:
# exactly what liveOD did before (AndorParams defaults, 2026-09-26).
ANDOR_RUN_FIELD_DEFAULTS = {"trigger_mode": "ext", "frame_transfer": 0,
                            "sensor_roi": (0, 512, 0, 512, 1, 1)}
BASLER_TRIGGER_SOURCE_DEFAULT = "Line1"

_RLOCK_TYPE = type(threading.RLock())


class CameraNanny():
    def __init__(self):
        self.interrupted = False
        # driver methods found missing, each warned about once (an old driver)
        self._missing_warned = set()
        # camera_lock's own locks, per camera key, for drivers without a
        # reentrant grab_lock()
        self._camera_locks = {}
        self._camera_locks_guard = threading.Lock()

    def break_check(self):
        return self.interrupted

    def camera_lock(self, camera, camera_params):
        """The lock a camera thread holds from applying its run's settings
        (update_params) to the end of its grab (CameraBaby), so that the thread
        of a superseded run, still applying its settings or in its last poll,
        cannot interleave its setters with the next run's: the new run could
        otherwise grab at the old exposure while camera_params/ records the new
        one.

        The driver's own per-device grab lock where it has a reentrant one
        (start_grab takes it again on the same thread; another thread's
        stop_grab stays a no-op while it is held), else a lock of this nanny's
        per camera key. None for a DummyCamera: nothing to guard."""
        if self._is_dummy_or_none(camera):
            return None
        grab_lock = getattr(camera, "grab_lock", None)
        if callable(grab_lock):
            try:
                lock = grab_lock()
            except Exception:
                lock = None
            if isinstance(lock, _RLOCK_TYPE):
                return lock
        camera_key = camera_params.key
        if isinstance(camera_key, bytes):
            camera_key = camera_key.decode()
        with self._camera_locks_guard:
            lock = self._camera_locks.get(camera_key)
            if lock is None:
                lock = self._camera_locks[camera_key] = threading.RLock()
            return lock

    def _warn_missing(self, camera_key, method, what):
        if (camera_key, method) in self._missing_warned:
            return
        self._missing_warned.add((camera_key, method))
        logger.warning(f"{camera_key}: the camera driver has no {method}(), so {what} "
                       f"is not re-applied per run (update the driver).")

    @staticmethod
    def _is_camera(obj):
        """A camera handle this nanny opened (not a DummyCamera, not a setting)."""
        return (obj is not None and not isinstance(obj, DummyCamera)
                and hasattr(obj, "is_opened") and hasattr(obj, "Close"))

    def _is_dummy_or_none(self, camera):
        return camera is None or isinstance(camera, DummyCamera)

    def _camera_is_opened(self, camera):
        if self._is_dummy_or_none(camera):
            return False
        try:
            return camera.is_opened()
        except Exception:
            return False

    def persistent_get_camera(self, camera_params, break_check=None) -> DummyCamera:
        """Open the camera, retrying every CHECK_PERIOD s until it opens or
        ``break_check()`` says stop (then a DummyCamera). ``break_check``: the
        calling camera thread's own interrupt. Without one, this nanny's shared
        flag, which the window clears for every new run -- so a thread from an
        earlier run waiting here would never be stopped by it."""
        check = break_check if break_check is not None else self.break_check
        got_camera = False
        count = 1
        while not got_camera:
            if check():
                break
            camera = self.get_camera(camera_params)
            if self._is_dummy_or_none(camera):
                count += 1
                # in CHECK_EVERY slices, so a stop is honoured within one
                t_retry = time.monotonic() + CHECK_PERIOD
                while time.monotonic() < t_retry and not check():
                    time.sleep(CHECK_EVERY)
                if np.mod(count,N_NOTIFY) == 0:
                    count = 1
                    logger.warning(f"Can't reach camera {camera_params.key}. Make it available to continue, or abort the run.")
            else:
                return camera

        return DummyCamera()

    def get_camera(self,camera_params:CameraParams) -> DummyCamera:
        camera_key = camera_params.key
        need_to_open = True
        camera = DummyCamera()
        if isinstance(camera_key, bytes):
            camera_key = camera_key.decode()
        if camera_key in self.__dict__.keys():
            camera = vars(self)[camera_key]
            need_to_open = not self._camera_is_opened(camera)
        if need_to_open:
            camera = self.open(camera_params)
            if not self._is_dummy_or_none(camera):
                vars(self)[camera_key] = camera
            else:
                camera = DummyCamera()
        return camera

    def update_params(self, camera, camera_params: CameraParams, report: dict = None):
        """Apply the run's camera_params to an open camera; returns the camera,
        or a DummyCamera if anything was refused or failed (the run then never
        becomes ready, and the log says why).

        Besides exposure, gain and the readout clocks, the trigger (and, for the
        Andor, frame transfer, sensor area and shutter) are re-asserted every
        run: the camera stays open between runs, and until 2026-09-26 these
        were set only when it was opened. ``report``: a dict to fill with
        ``"clamps"`` -- ``{field: (requested, applied)}`` for every value the
        driver could not set as asked -- for the run's overrides record, or,
        when the settings were refused or failed, ``"error"``: the exception's
        type and message (the camera thread reports it to the server, which
        then fails the run's WAIT_CAM_READY at once).

        The caller holds ``camera_lock`` around this and the grab (CameraBaby).
        """
        camera_type = camera_params.camera_type
        if isinstance(camera_type, bytes):
            camera_type = camera_type.decode()
        camera_key = camera_params.key
        if isinstance(camera_key, bytes):
            camera_key = camera_key.decode()

        if self._is_dummy_or_none(camera):
            return DummyCamera()

        clamps = {}
        try:
            if camera_type == "basler":
                # Guard: stop grabbing if the camera was left in grabbing state
                # (e.g. from an interrupted run whose StopGrabbing failed silently).
                # Go through stop_grab() rather than StopGrabbing() directly: it
                # is a no-op when another thread still owns the grab loop, so a
                # new baby's setup cannot tear down a grab in progress.
                if hasattr(camera, 'IsGrabbing') and camera.IsGrabbing():
                    try:
                        camera.stop_grab()
                    except Exception:
                        pass
                camera.set_exposure(camera_params.exposure_time)
                camera.set_gain(camera_params.gain)
                trigger_source = getattr(camera_params, "trigger_source", BASLER_TRIGGER_SOURCE_DEFAULT)
                if isinstance(trigger_source, bytes):
                    trigger_source = trigger_source.decode()
                if hasattr(camera, "configure_trigger"):
                    camera.configure_trigger(trigger_source)
                else:
                    self._warn_missing(camera_key, "configure_trigger", "the trigger (mode, source, line)")
                logger.info(f"{camera_params.key}: gain set to {camera_params.gain}")
            elif camera_type == "andor":
                camera.set_EMCCD_gain(camera_params.gain)
                camera.set_exposure(camera_params.exposure_time)
                if hasattr(camera, "set_amp_mode_checked"):
                    # One call for the whole amplifier mode, refused (nothing
                    # sent) if the camera does not offer it. As two calls, the
                    # preamp was set at the old hs_speed first, and pylablib
                    # silently substitutes a preamp that combination lacks.
                    camera.set_amp_mode_checked(channel=0, oamp=0,
                                                hsspeed=camera_params.hs_speed,
                                                preamp=camera_params.preamp)
                else:
                    self._warn_missing(camera_key, "set_amp_mode_checked",
                                       "the amplifier mode (hs_speed with preamp) as one checked setting")
                    camera.set_amp_mode(preamp=camera_params.preamp)
                    # keyword: set_hsspeed(typ, hs_speed) - the old positional call
                    # passed hs_speed as typ (harmless only because both are 0)
                    camera.set_hsspeed(hs_speed=camera_params.hs_speed)
                # 2026-09-24: the camera stays open between runs, so the
                # vertical clock was only ever applied at first open (server
                # start); per-run vs_speed / vs_amp were recorded in the HDF5
                # but never reached the camera (runs 80702-80706).
                camera.set_vsspeed(camera_params.vs_speed)
                camera.set_vsamplitude(camera_params.vs_amp)
                clamp = getattr(camera_params, 'baseline_clamp', 1)
                camera.set_baseline_clamp(clamp)
                # the run-owned fields; a payload from before they existed means
                # what liveOD always did (ext trigger, no FT, full frame)
                run_fields = {k: getattr(camera_params, k, v)
                              for k, v in ANDOR_RUN_FIELD_DEFAULTS.items()}
                if isinstance(run_fields["trigger_mode"], bytes):
                    run_fields["trigger_mode"] = run_fields["trigger_mode"].decode()
                if hasattr(camera, "apply_run_fields"):
                    # validates (refusing anything but ext / FT off / full frame)
                    # and sets trigger, FT, image area and shutter open
                    camera.apply_run_fields(run_fields["trigger_mode"],
                                            run_fields["frame_transfer"],
                                            run_fields["sensor_roi"])
                else:
                    self._warn_missing(camera_key, "apply_run_fields",
                                       "the trigger mode, frame transfer, sensor area and shutter")
                logger.info(f"{camera_params.key}: gain set to {camera_params.gain}")
            # what the driver could not set as asked (Basler exposure/gain limits)
            if hasattr(camera, "last_clamps"):
                clamps = dict(camera.last_clamps() or {})
        except Exception as e:
            logger.error(f"Could not apply the run's settings to camera {camera_params.key}: {e}")
            if report is not None:
                report["error"] = f"{type(e).__name__}: {e}"
            return DummyCamera()
        # (the server logs each clamp, with the run id, when the camera thread
        # passes them on)
        if report is not None:
            report["clamps"] = clamps
        return camera

    def open(self,camera_params:CameraParams):
        camera_type = camera_params.camera_type
        if isinstance(camera_type, bytes):
            camera_type = camera_type.decode()
        try:
            if camera_type == "basler":
                camera = BaslerUSB(BaslerSerialNumber=camera_params.serial_no,
                                    ExposureTime=camera_params.exposure_time,
                                    TriggerSource=camera_params.trigger_source,
                                    Gain=camera_params.gain)
            elif camera_type == "andor":
                camera = AndorEMCCD(ExposureTime=camera_params.exposure_time,
                                    gain = camera_params.gain,
                                    hs_speed=camera_params.hs_speed,
                                    vs_speed=camera_params.vs_speed,
                                    vs_amp=camera_params.vs_amp,
                                    preamp=camera_params.preamp,
                                    baseline_clamp=getattr(camera_params, 'baseline_clamp', 1))
            else:
                camera = DummyCamera()

            if camera is None:
                camera = DummyCamera()

        except Exception as e:
            camera = DummyCamera()
            # a warning, not an error: persistent_get_camera retries this every 2 s
            logger.warning(f"There was an issue opening the requested camera (key: {camera_params.key}): {e}")
        return camera

    def close_all(self) -> dict:
        """Close every camera this nanny opened, after stopping any grab, through
        the driver's safe close: ``close_safely()`` where the driver has it
        (Andor: acquisition stopped, shutter closed, cooler kept, SDK closed,
        never raising), else ``Close()`` (the lowercase ``close()`` skipped the
        Andor shutter). Called when liveOD shuts down. A camera that is closed
        afterwards is forgotten, so calling this again does nothing to it.
        Returns ``{camera_key: ""}`` for each camera closed cleanly, else the
        error text (a camera that closed with step errors is still forgotten;
        one that is still open is kept)."""
        results = {}
        for k in list(vars(self).keys()):
            obj = vars(self).get(k)
            if not self._is_camera(obj):
                continue
            errors = []
            try:
                if hasattr(obj, "stop_grab"):
                    obj.stop_grab()
            except Exception as e:
                errors.append(f"stop_grab(): {e}")
            try:
                closer = getattr(obj, "close_safely", None)
                if callable(closer):
                    errors += [str(e) for e in (closer() or [])]
                else:
                    errors += [str(e) for e in (obj.Close() or [])]
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
            try:
                still_open = bool(obj.is_opened())
            except Exception:
                still_open = False          # a handle that cannot say is not usable either
            results[k] = "; ".join(errors)
            if errors:
                logger.warning(f"Closing camera {k}: {results[k]}"
                               + (" -- it is still open." if still_open else ""))
            if not still_open:
                del vars(self)[k]
                logger.info(f"Closed camera {k}.")
        return results
