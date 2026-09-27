import logging
import numpy as np
import threading
import time

from pylablib.devices import Andor
from pylablib.devices.Andor.AndorSDK2 import _camfunc
from pylablib.devices.Andor.atmcd32d_lib import (wlib as lib, AndorSDK2LibError, DRV_STATUS,
                                                 AC_SETFUNC, AC_GETFUNC, AC_FEATURES, AC_ACQMODE)
from pylablib.core.utils import general as general_utils, py3
from pylablib.core.devio import interface

from queue import Queue

from beacon.camera.backend import Readback, ApplyRefused, ApplyMismatch

from waxx.config.timeouts import (CAMERA_GRAB_TIMEOUT_ANDOR as TIMEOUT)
from waxx.control.cameras.device_lock import DeviceLock
from waxx.control.cameras.errors import FrameLostError
from waxx.control.cameras.camera_param_classes import RunFieldRefused, check_andor_run_fields

logger = logging.getLogger(__name__)

# One process at a time may hold the SDK (see device_lock).  idx 0: the only
# camera this class opens.
DEVICE_LOCK_KEY = "andor_sdk2:0"

# SDK acquisition-mode code -> (setup method, _cpar key of its last parameters,
# whether those are stored as a tuple).  The vendored
# AndorSDK2Camera.set_acquisition_mode maps these one slot low
# (AndorSDK2.py:710-718): "single" runs the accum setup (SDK mode 2), "accum"
# the kinetic one, "kinetic" the fast-kinetic one, and "fast_kinetic" raises
# TypeError unpacking the scalar cont cycle time.  "single" (1) has no setup.
_ACQ_MODE_SETUP = {
    2: ("setup_accum_mode", "acq_params/accum", True),
    3: ("setup_kinetic_mode", "acq_params/kinetic", True),
    4: ("setup_fast_kinetic_mode", "acq_params/fast_kinetic", True),
    5: ("setup_cont_mode", "acq_params/cont", False),
}

# Grab loops on the camera are serialized by a lock per physical device (as in
# basler_usb): a CameraBaby left over from an aborted or superseded run can
# still be in its grab loop -- or in its death handler -- when the next run's
# baby starts, and its stop_grab() must not stop the new run's acquisition.
# Module-level and keyed by device, so it also holds across a replaced object.
_GRAB_LOCKS = {}
_GRAB_LOCKS_GUARD = threading.Lock()

def _grab_lock_for(key):
    with _GRAB_LOCKS_GUARD:
        lock = _GRAB_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _GRAB_LOCKS[key] = lock
        return lock

def nothing():
    return False

class AmpModeUnavailable(ValueError):
    """The requested (channel, output amp, hs_speed, preamp) is not a mode this
    camera offers.  pylablib's set_amp_mode would silently substitute the
    nearest one; this is raised instead."""

class AndorEMCCD(Andor.AndorSDK2Camera):
    def __init__(self,
                ExposureTime=0.,
                gain = 30,
                hs_speed:int=0,
                vs_speed:int=1,
                vs_amp:int=3,
                preamp = 2,
                baseline_clamp:int=1):
        # overwrite a broken method in the parent class
        self._initial_setup_temperature = self._initial_setup_temperature_fixed
        self._last_clamps = {}          # before the parent __init__, which sets the exposure
        # Take the OS lock before the SDK is touched (the parent __init__
        # calls Initialize): a second process gets DeviceBusy naming the
        # holder instead of an SDK error.  Released in close().
        self._device_lock = DeviceLock(DEVICE_LOCK_KEY)
        self._device_lock.acquire()
        try:
            # init the parent class
            super().__init__(temperature=-60,fan_mode="full")
            # run startup setting methods
            # self.activate_cameralink()
            # self.enable_frame_transfer_mode(enable=True)
            # self.set_emccd_advanced()
            self.set_EM_gain_mode(3)
            # advanced=False caps the gain at x300. Nothing in kexp asks for more,
            # and >x300 accelerates sensor ageing (SDK: SetEMAdvanced).
            self.set_EMCCD_gain(gain=gain,advanced=False)
            self.set_baseline_clamp(baseline_clamp)
            self.set_exposure(ExposureTime)
            self.set_trigger_mode("ext")
            self.setup_shutter_if_supported("open")
            self.set_vsspeed(vs_speed)
            self.set_vsamplitude(vs_amp)
            # One amplifier-mode call for output amp, hs_speed and preamp
            # (2026-09-26).  A direct SetHSSpeed bypassed pylablib's cache, and
            # the later set_amp_mode(preamp=...) wrote the cached (default)
            # hs index back, so a non-zero hs_speed never reached the camera.
            self.set_amp_mode_checked(channel=0, oamp=0, hsspeed=hs_speed, preamp=preamp)
            # Continuous mode, the one every grab runs in (was "single", which
            # the vendored set_acquisition_mode turned into accumulate).
            self.setup_cont_mode(0)
            # Frame transfer stays off: with an external trigger it makes the
            # exposure the time between triggers.
            self.enable_frame_transfer_mode(False)
            self.set_read_mode("image")
            self.set_cooler_mode(mode=1)
            self.activate_cameralink(1)

            # self.set_fast_trigger_mode(mode=1)
        except BaseException:
            self._close_after_failed_init()
            raise

        self._internal_output_queue = Queue()

    def _close_after_failed_init(self):
        # Leave nothing half-open: an SDK left initialized blocks the next
        # open (in this process and others) until the process exits.
        try:
            if getattr(self, "handle", None) is not None:
                try:
                    self.set_cooler_mode(1)
                except Exception:
                    pass
                self.close()
        except Exception as e:
            logger.warning(f"[AndorEMCCD] close after a failed init also failed: {e}")
        finally:
            self._release_device_lock()

    def _release_device_lock(self):
        lock = getattr(self, "_device_lock", None)
        if lock is not None:
            lock.release()

    def open(self):
        """Open the SDK connection -- once.  Reopening a closed handle is
        refused: it would restore pylablib's defaults (internal trigger,
        shutter closed, EM gain 0) without the settings this class applies,
        and without the device lock.  Construct a new AndorEMCCD instead."""
        if getattr(self, "_closed_once", False) and self.handle is None:
            raise self.Error("AndorEMCCD: reopening a closed handle is refused (it would "
                             "restore pylablib's defaults, internal trigger included). "
                             "Construct a new AndorEMCCD instead.")
        super().open()

    def close(self):
        """Close the SDK connection (ShutDown at the last handle), then release
        the device lock."""
        self._closed_once = True
        try:
            super().close()
        finally:
            self._release_device_lock()

    def Close(self):
        """Stop, close the shutter, keep the cooler on, then close the SDK.

        Each step before the SDK close has its own try, so a failure in one
        skips neither the others nor the close (a half-open SDK blocks the next
        open until the process exits).  The stop comes first: the SDK refuses
        SetShutter while acquiring.  Errors from the SDK close itself propagate.
        Returns the list of step failures (empty when all went through)."""
        if not self.is_opened():
            self._release_device_lock()
            return []
        errors = self._close_steps()
        self.close()
        return errors

    def close_safely(self):
        """Close() that never raises: returns every failure, the SDK close's
        included, as a list of strings (empty when all went through)."""
        if not self.is_opened():
            self._release_device_lock()
            return []
        errors = self._close_steps()
        try:
            self.close()
        except Exception as e:
            errors.append(f"close(): {e}")
            logger.warning(f"[AndorEMCCD] SDK close failed: {e}")
        return errors

    def _close_steps(self):
        errors = []
        for name, step in (("stop_acquisition()", self.stop_acquisition),
                           ("setup_shutter('closed')", lambda: self.setup_shutter_if_supported("closed")),
                           ("set_cooler_mode(1)", lambda: self.set_cooler_mode(1))):
            try:
                step()
            except Exception as e:
                errors.append(f"{name}: {e}")
                logger.warning(f"[AndorEMCCD] {name} failed: {e}. "
                               f"Proceeding with SDK close anyway.")
        return errors

    def Open(self):
        """Nothing to do if open; refused if closed (see open())."""
        if self.is_opened():
            return
        raise self.Error("AndorEMCCD.Open(): this handle was closed. Reopening it in place "
                         "would restore pylablib's defaults (internal trigger included); "
                         "construct a new AndorEMCCD instead.")

    @_camfunc(setpar="acq_mode")
    @interface.use_parameters(mode="acq_mode",_returns="acq_mode")
    def set_acquisition_mode(self, mode, setup_params=True):
        """
        Set the acquisition mode (overrides the vendored version, whose
        code map is one slot low -- see _ACQ_MODE_SETUP).

        Can be ``"single"``, ``"accum"``, ``"kinetic"``, ``"fast_kinetic"`` or ``"cont"`` (continuous).
        If ``setup_params==True``, the last parameters given to that mode's
        ``setup_*_mode`` are applied again (when there are any).
        """
        entry = _ACQ_MODE_SETUP.get(mode) if setup_params else None
        if entry is not None and entry[1] in self._cpar:
            method, cpar_key, as_tuple = entry
            params = self._cpar[cpar_key]
            if getattr(self, method)(*(params if as_tuple else (params,))) is None:
                return
        else:
            if not self._check_option("acq",self._acqmode_caps[mode]): return
            lib.SetAcquisitionMode(mode)
        return mode

    def start_grab(self, N_img, output_queue:Queue=None,
                   check_interrupt_method=None, on_armed=None):
        '''
        Acquire N_img externally triggered frames, putting (img, t, idx) on
        output_queue as each arrives.  Returns the frames in index order.

        idx is the frame's index within this acquisition as the SDK counts it
        (see read_frames_from), so a frame the ring buffer lost can never move a
        later frame into its slot: a lost frame raises FrameLostError (a
        TimeoutError) naming its index, after every frame that did arrive has
        been queued under its own index.

        on_armed() is called exactly once, after the acquisition is confirmed
        running and before the first frame is waited for: from then on no
        trigger can be missed.

        Holds this camera's grab lock throughout (see grab_lock), so a grab
        still running in another thread is waited for, and that thread's
        stop_grab() cannot stop this one.
        '''
        with self.grab_lock():
            return self._grab_loop(N_img, output_queue, check_interrupt_method, on_armed)

    def grab_lock(self):
        """The lock serializing grab loops on this physical camera."""
        return _grab_lock_for(DEVICE_LOCK_KEY)

    def _grab_loop(self, N_img, output_queue, check_interrupt_method, on_armed):
        N_img = int(N_img)
        if output_queue is None:
            output_queue = self._internal_output_queue
        check = check_interrupt_method or nothing
        restore_format = None
        if self.get_frame_format() != "list":
            # read_frames_from expects one 2D array per frame
            restore_format = self.get_frame_format()
            self.set_frame_format("list")
        frames = {}
        next_idx = 0
        surplus = 0
        self.start_acquisition(mode="cont")
        try:
            self._confirm_acquiring()
            if on_armed is not None:
                on_armed()
            while next_idx < N_img:
                if check():
                    print('Interrupt submitted, waiting for grab loop termination...')
                    break
                try:
                    running = self.wait_for_frame(timeout=TIMEOUT,check_interrupt_method=check)
                except self.TimeoutError:
                    # pylablib's AndorTimeoutError carries no message; re-raise as
                    # the builtin TimeoutError that liveOD reports without a traceback.
                    raise TimeoutError(
                        f"No Andor image within {TIMEOUT:.0f} s "
                        f"(got {len(frames)}/{N_img}). Camera not triggered?") from None
                new_frames, lost, first = self.read_frames_from(next_idx)
                if not new_frames and not lost:
                    if not running and not check():
                        raise TimeoutError(
                            f"Andor acquisition stopped with {len(frames)}/{N_img} images.")
                    continue
                lost = [i for i in lost if i < N_img]
                for k, frame in enumerate(new_frames):
                    idx = first + k
                    if idx >= N_img:
                        surplus += 1
                        continue
                    output_queue.put((frame, time.time(), idx))
                    frames[idx] = frame
                    print(f'gotem (img {idx+1}/{N_img})') # added this line to give print statements
                next_idx = first + len(new_frames)
                if lost:
                    raise FrameLostError(
                        f"Andor frame(s) {lost} of {N_img} were overwritten in the SDK ring "
                        f"buffer before they were read (got {len(frames)}); later frames "
                        f"kept their own index.", lost=lost)
            if surplus:
                logger.warning(f"[AndorEMCCD] {surplus} frame(s) beyond the {N_img} expected "
                               f"arrived before the acquisition stopped; not queued.")
            return [frames[i] for i in sorted(frames)]
        finally:
            self.stop_acquisition()
            if restore_format is not None:
                self.set_frame_format(restore_format)

    def read_frames_from(self, next_idx):
        """Read every frame from index next_idx on.

        Returns (frames, lost, first): frames[k] has hardware index first + k,
        and lost lists the indices from next_idx up to first that the SDK ring
        buffer overwrote before they were read.  Reads an explicit range: with
        rng=None pylablib drops the count of overwritten frames, and its
        missing_frame="none" mode fails on them in list format
        (_convert_frame_format calls frames[0].ndim on the None it prepends)."""
        frames, rng = self.read_multiple_images(rng=(next_idx, None), missing_frame="skip",
                                                return_rng=True)
        if frames is None or rng is None:
            return [], [], next_idx
        first = int(rng[0])
        return list(frames), list(range(next_idx, first)), first

    def _confirm_acquiring(self, timeout=1.0):
        deadline = time.monotonic() + timeout
        while not self.acquisition_in_progress():
            if time.monotonic() > deadline:
                raise self.Error("StartAcquisition returned but the camera does not report "
                                 "'acquiring'; not arming.")
            time.sleep(0.005)

    def apply_run_fields(self, trigger_mode="ext", frame_transfer=0,
                         sensor_roi=(0, 512, 0, 512, 1, 1)):
        """Re-assert the run-owned fields before a run: trigger, frame transfer
        off, image area, continuous acquisition and shutter open.

        Values are checked first (external trigger only, frame transfer 0, the
        full frame at bin 1); a refused one raises ApplyRefused naming it and
        nothing is sent.  Stops any acquisition.  Returns {field: Readback}.
        Raises ApplyMismatch if the driver ends up somewhere else -- pylablib
        silently truncates an image area it does not like (AndorSDK2.py:934-955).
        """
        try:
            trigger_mode, frame_transfer, sensor_roi = check_andor_run_fields(
                trigger_mode, frame_transfer, sensor_roi, purpose="run")
        except RunFieldRefused as e:
            raise ApplyRefused(e.field, f"{e.value!r} is refused: {e.reason}") from None
        self.stop_acquisition()
        self.set_trigger_mode(trigger_mode)
        ft = self.frame_transfer_off()
        roi_set = tuple(int(v) for v in self.setup_image_mode(*sensor_roi))
        self.set_acquisition_mode("cont")
        self.setup_shutter_if_supported("open")
        shutter = self.shutter_readback()
        readback = {
            "trigger_mode": Readback(self.get_trigger_mode(), "driver_cache"),
            "frame_transfer": ft,
            "sensor_roi": Readback(roi_set, "driver_cache"),
            "acq_mode": Readback(self.get_acquisition_mode(), "driver_cache"),
            "shutter": shutter,
        }
        mismatches = {}
        if readback["trigger_mode"].value != trigger_mode:
            mismatches["trigger_mode"] = (trigger_mode, readback["trigger_mode"].value)
        if ft.source != "unsupported" and ft.value != frame_transfer:
            mismatches["frame_transfer"] = (frame_transfer, ft.value)
        if roi_set != sensor_roi:
            mismatches["sensor_roi"] = (sensor_roi, roi_set)
        if readback["acq_mode"].value != "cont":
            mismatches["acq_mode"] = ("cont", readback["acq_mode"].value)
        if mismatches:
            raise ApplyMismatch(mismatches)
        return readback

    def set_exposure(self, exposure):
        """Set the exposure (s); returns what the camera reports it will use,
        and remembers the pair when that differs from the request (see
        last_clamps)."""
        applied = super().set_exposure(exposure)
        self._note_clamp("exposure_time", exposure, applied)
        return applied

    def set_EMCCD_gain(self, gain, advanced=None):
        """pylablib's set_EMCCD_gain, then the gain read back from the camera
        is compared with the request (see last_clamps)."""
        super().set_EMCCD_gain(gain, advanced=advanced)
        try:
            applied = self.get_EMCCD_gain()[0]
        except Exception:
            return
        self._note_clamp("gain", gain, applied)

    def _note_clamp(self, key, requested, applied):
        try:
            requested, applied = float(requested), float(applied)
        except (TypeError, ValueError):
            return
        if abs(applied - requested) > 1e-6 * max(abs(requested), 1e-9):   # beyond float32 rounding
            self._last_clamps[key] = (requested, applied)
        else:
            self._last_clamps.pop(key, None)

    def last_clamps(self):
        """{key: (requested, applied)} for the most recent set_exposure /
        set_EMCCD_gain whose value the camera did not take as asked
        (exposure_time in s, gain as the EM gain factor)."""
        return dict(self._last_clamps)

    def frame_transfer_off(self):
        if not self._has_option("acq", AC_ACQMODE.AC_ACQMODE_FRAMETRANSFER):
            return Readback(0, "unsupported")
        self.enable_frame_transfer_mode(False)
        return Readback(int(bool(self.is_frame_transfer_enabled())), "driver_cache")

    def setup_shutter_if_supported(self, mode):
        """setup_shutter(mode) if the camera has shutter control; returns
        whether anything was sent."""
        if not self._has_option("feat", AC_FEATURES.AC_FEATURES_SHUTTER):
            return False
        self.setup_shutter(mode=mode)
        return True

    def shutter_readback(self):
        """The shutter mode last sent (the SDK has no getter), or unsupported
        when the camera has no shutter control."""
        if not self._has_option("feat", AC_FEATURES.AC_FEATURES_SHUTTER):
            return Readback(None, "unsupported")
        return Readback(self.get_shutter(), "commanded")

    def _initial_setup_temperature_fixed(self):
        if self._start_temperature=="off":
            trng=self.get_temperature_range()
            self.set_temperature(trng[1] if trng else 0,enable_cooler=False)
        else:
            if self._start_temperature is None:
                trng=self.get_temperature_range()
                if trng:
                    self._start_temperature=trng[0]+int((trng[1]-trng[0])*0.2)
                else:
                    self._start_temperature=0
            self.set_temperature(self._start_temperature,enable_cooler=True)

    def stop_grab(self):
        # Non-blocking on purpose: if another thread owns the grab loop (a newer
        # CameraBaby that already claimed this camera), leave it alone -- its own
        # start_grab() finally will stop it.  Stopping it from a dying baby's
        # death handler is exactly what would break the next run's grab.
        lock = self.grab_lock()
        if not lock.acquire(blocking=False):
            print("Grab loop is owned by another thread; not stopping it here.")
            return
        try:
            self.stop_acquisition()
        except Exception:
            pass
        finally:
            lock.release()

    @interface.use_parameters(since="frame_wait_mode")
    def wait_for_frame(self, since="lastread", nframes=1, timeout=20., error_on_stopped=False,
                       check_interrupt_method=nothing):
        '''
        Wait for one or several new camera frames. (overloaded to accept interrupt)

        `since` specifies the reference point for waiting to acquire `nframes` frames;
        can be "lastread"`` (from the last read frame), ``"lastwait"`` (wait for the last successful :meth:`wait_for_frame` call),
        ``"now"`` (from the start of the current call), or ``"start"`` (from the acquisition start, i.e., wait until `nframes` frames have been acquired).
        `timeout` can be either a number, ``None`` (infinite timeout), or a tuple ``(timeout, frame_timeout)``,
        in which case the call times out if the total time exceeds ``timeout``, or a single frame wait exceeds ``frame_timeout``.
        If the call times out, raise ``TimeoutError``.
        If ``error_on_stopped==True`` and the acquisition is not running, raise ``Error``;
        otherwise, simply return ``False`` without waiting.
        '''
        wait_started=False
        if isinstance(timeout,tuple):
            timeout,frame_timeout=timeout
        else:
            frame_timeout=None
        ctd=general_utils.Countdown(timeout)
        frame_ctd=general_utils.Countdown(frame_timeout)
        if not self.acquisition_in_progress():
            if error_on_stopped:
                raise self.Error("waiting for a frame while acquisition is stopped")
            else:
                return False
        last_acquired_frames=None
        while True:
            if check_interrupt_method():
                break
            acquired_frames=self._get_acquired_frames()
            if acquired_frames is None:
                if error_on_stopped:
                    raise self.Error("waiting for a frame while acquisition is stopped")
                else:
                    return False
            if acquired_frames!=last_acquired_frames:
                frame_ctd.reset()
            last_acquired_frames=acquired_frames
            if not wait_started:
                self._frame_counter.wait_start(acquired_frames)
                wait_started=True
            if self._frame_counter.is_wait_done(acquired_frames,since=since,nframes=nframes):
                break
            to,fto=ctd.time_left(),frame_ctd.time_left()
            if fto is not None:
                to=fto if to is None else min(to,fto)
            if to is not None and to<=0:
                raise self.TimeoutError
            self._wait_for_next_frame(timeout=to,idx=acquired_frames)
        self._frame_counter.wait_done()
        return True

    def activate_cameralink(self,state=1):
        '''This function allows the user to enable or disable the Camera Link
        functionality for the camera. Enabling this functionality will start to
        stream all acquired data through the camera link interface.

        Args:
            state (int, optional): Enables/Disables Camera Link mode. 1 - Enable
            Camera Link 0 - Disable Camera Link. Defaults to 1.
        '''
        lib.SetCameraLinkMode(state)

    def set_emccd_advanced(self):
        '''
        This function turns on and off access to higher EM gain levels within
        the SDK. Typically, optimal signal to noise ratio and dynamic range is
        achieved between x1 to x300 EM Gain. Higher gains of > x300 are
        recommended for single photon counting only. Before using higher levels,
        you should ensure that light levels do not exceed the regime of tens of
        photons per pixel, otherwise accelerated ageing of the sensor can occur.
        '''
        lib.SetEMAdvanced(1)

    def set_fast_trigger_mode(self, mode:int = 1):
        '''
        This function will enable fast external triggering. When fast external
        triggering is enabled the system will NOT wait until a “Keep Clean”
        cycle has been completed before accepting the next trigger. This setting
        will only have an effect if the trigger mode has been set to External
        via SetTriggerMode.

        Args:
            mode (int, optional): 0 disabled. 1 enabled. Defaults to 1.
        '''
        lib.SetFastExtTrigger(mode)

    def set_cooler_mode(self, mode:int = 1):
        '''This function determines whether the cooler is switched off when the
        camera is shut down.

        Args:
            mode (int, optional): 1 – Temperature is maintained on ShutDown. 0 –
            Returns to ambient temperature on ShutDown. Defaults to 1.
        '''
        lib.SetCoolerMode(mode)

    def set_vsamplitude(self, vs_amp:int = 0):
        '''
        If you choose a high readout speed (a low readout time), then you should
        also consider increasing the amplitude of the Vertical Clock Voltage.
        There are five levels of amplitude available for you to choose from:
            * Normal
            * +1
            * +2
            * +3
            * +4
        Exercise caution when increasing the amplitude of the vertical clock voltage,
        since higher clocking voltages may result in increased clock-induced charge
        (noise) in your signal. In general, only the very highest vertical clocking
        speeds are likely to benefit from an increased vertical clock voltage amplitude.

        Args:
            vs_amp (int, optional): See docstring for amplitude settings.
            Defaults to 0.
        '''
        lib.SetVSAmplitude(vs_amp)

    def set_hsspeed(self, typ:int=0, hs_speed:int = 0):
        '''
        This function will set the speed at which the pixels are shifted into
        the output node during the readout phase of an acquisition. Typically
        your camera will be capable of operating at several horizontal shift
        speeds. To get the actual speed that an index corresponds to use the
        GetHSSpeed function.

        Goes through pylablib's set_amp_mode so its cached amplifier mode stays
        what the camera has (a direct SetHSSpeed left the cache stale, and the
        next set_amp_mode wrote the stale index back).  Raises
        AmpModeUnavailable, with nothing sent, if the combination with the
        current preamp is not one the camera offers.

        Args:
            hs_speed (int, optional): the horizontal speed to be used Valid
            values 0 to GetNumberHSSpeeds()-1. Defaults to 0.
            int type: the type of output amplifier on an EMCCD.
                * 0: Standard EMCCD gain register (default).
                * 1: Conventional CCD register.
        '''
        self.set_amp_mode_checked(oamp=typ, hsspeed=hs_speed)

    def is_amp_mode_available(self, channel, oamp, hsspeed, preamp):
        modes = {(m.channel, m.oamp, m.hsspeed, m.preamp) for m in self.get_all_amp_modes()}
        return (int(channel), int(oamp), int(hsspeed), int(preamp)) in modes

    def set_amp_mode_checked(self, channel=None, oamp=None, hsspeed=None, preamp=None):
        """set_amp_mode, refusing instead of letting pylablib substitute.

        None keeps the current value.  The full mode is checked against the
        camera's list before anything is sent, and the cache is compared with
        the request after."""
        current = (self._cpar.get("channel"), self._cpar.get("oamp"),
                   self._cpar.get("hsspeed"), self._cpar.get("preamp"))
        request = tuple(cur if new is None else int(new)
                        for cur, new in zip(current, (channel, oamp, hsspeed, preamp)))
        described = (f"(channel {request[0]}, output amp {request[1]}, "
                     f"hs_speed {request[2]}, preamp {request[3]})")
        if None not in request and not self.is_amp_mode_available(*request):
            offered = sorted({(m.hsspeed, m.preamp) for m in self.get_all_amp_modes()
                              if m.channel == request[0] and m.oamp == request[1]})
            raise AmpModeUnavailable(
                f"amplifier mode {described} is not available on this camera; nothing "
                f"was sent. (hs_speed, preamp) offered for output amp {request[1]}: {offered}")
        self.set_amp_mode(*request)
        got = (self.get_channel(), self.get_oamp(), self.get_hsspeed(), self.get_preamp())
        if tuple(got) != request:
            raise AmpModeUnavailable(
                f"amplifier mode {described} was requested but pylablib applied "
                f"(channel {got[0]}, output amp {got[1]}, hs_speed {got[2]}, preamp {got[3]})")

    def set_isolated_crop_mode_type(self, mode:int=1):
        '''This function determines the method by which data is transferred in
        isolated crop mode. The default method is High Speed where multiple
        frames may be stored in the storage area of the sensor before they
        are read out. In Low Latency mode, each cropped frame is read out as
        it happens.

        Args:
            mode (int): 0 – High Speed. 1 – Low Latency. Defaults to 1.
        '''
        lib.SetIsolatedCropModeType(mode)

    def set_EM_gain_mode(self, mode:int=3):
        lib.SetEMGainMode(mode)

    def set_baseline_clamp(self, state:int = 1):
        '''Turn the baseline clamp on or off (SDK: SetBaselineClamp). With
        the clamp on the bias level of every frame is held at the same value,
        so light-minus-dark subtraction does not pick up frame-to-frame offset
        drift. Skipped silently if the camera does not report the capability.

        Args:
            state (int, optional): 1 enable, 0 disable. Defaults to 1.
        '''
        if not self._has_option("set", AC_SETFUNC.AC_SETFUNCTION_BASELINECLAMP):
            print("[AndorEMCCD] baseline clamp not supported on this camera; skipped.")
            return
        lib.SetBaselineClamp(state)

    # -- read-only queries used by the camera backend's apply/readback -------
    @_camfunc
    def get_baseline_clamp(self):
        """Baseline clamp state read from the camera (GetBaselineClamp), or
        None if the camera does not report it."""
        if not self._has_option("get", AC_GETFUNC.AC_GETFUNCTION_BASELINECLAMP):
            return None
        return int(lib.GetBaselineClamp())

    @_camfunc
    def is_trigger_mode_available(self, mode):
        """IsTriggerModeAvailable for a pylablib trigger-mode name."""
        code = self._p_trigger_mode(mode)
        try:
            lib.IsTriggerModeAvailable(code)
        except AndorSDK2LibError as e:
            if e.code in (DRV_STATUS.DRV_INVALID_MODE, DRV_STATUS.DRV_NOT_SUPPORTED):
                return False
            raise
        return True

    @_camfunc
    def get_max_exposure(self):
        """Longest exposure the camera accepts (s)."""
        return float(lib.GetMaximumExposure())

    @_camfunc
    def get_vsamplitude_labels(self):
        """Labels of the vertical clock amplitudes, index order."""
        n = int(lib.GetNumberVSAmplitudes())
        labels = []
        for i in range(n):
            try:
                labels.append(py3.as_str(lib.GetVSAmplitudeString(i)))
            except AndorSDK2LibError:
                labels.append("Normal" if i == 0 else f"+{i}")
        return labels
