import logging
import threading
import time

from pypylon import pylon
import numpy as np

from queue import Queue
from PyQt6.QtCore import QThread, pyqtSignal

from waxx.config.timeouts import (CAMERA_GRAB_TIMEOUT_BASLER_INIT as TIMEOUT_INIT,
                                  CAMERA_GRAB_TIMEOUT_BASLER_RUN as TIMEOUT_RUN)
from waxx.control.cameras.device_lock import DeviceLock, DeviceBusy
from waxx.control.cameras.errors import FrameLostError

logger = logging.getLogger(__name__)

# RetrieveResult is polled in slices of this length (s) instead of blocking for
# the whole frame timeout, so an interrupted run releases the camera promptly.
GRAB_POLL_INTERVAL = 0.2

# A set value the camera read back as more than this away from the request is
# recorded as a clamp (last_clamps): out of range, or rounded to the camera's
# step by more than the tolerance. Exposure: 0.1% of the request (floor 1 ns),
# far above float noise from the us <-> s conversion, below any rounding that
# changes the signal noticeably. Gain: 0.01 dB (0.1% in amplitude).
EXPOSURE_CLAMP_REL = 1e-3
EXPOSURE_CLAMP_ABS_S = 1e-9
GAIN_CLAMP_DB = 0.01

# Frame counters on a grab result, compared frame to frame within one run grab:
# BlockID counts the frames the camera sent, ImageNumber the frames the Instant
# Camera received (a frame it skipped still counts). A jump of more than 1 is a
# frame lost without a failed grab result.
FRAME_COUNTERS = ("BlockID", "ImageNumber")

# Grab locks are keyed by serial number rather than kept on the BaslerUSB
# instance: pypylon's InstantCamera.__setattr__ routes any unknown attribute
# name into the GenICam node map, so plain attributes cannot be assigned here.
# Keying by serial also covers the case where the camera object was replaced
# (reopened) while an old one is still grabbing the same physical camera.
_GRAB_LOCKS = {}
_GRAB_LOCKS_GUARD = threading.Lock()

# Per serial, for the same reason: the process lock taken at open (released at
# Close), as (id of the owning BaslerUSB, DeviceLock), and the clamps of the
# most recent set_exposure / set_gain.
_DEVICE_LOCKS = {}
_LAST_CLAMPS = {}

def _grab_lock_for(serial):
    with _GRAB_LOCKS_GUARD:
        lock = _GRAB_LOCKS.get(serial)
        if lock is None:
            lock = threading.RLock()
            _GRAB_LOCKS[serial] = lock
        return lock

def nothing():
    return False

def _read_back(node, fallback):
    """The node's value as the camera reports it (float), else ``fallback``."""
    try:
        return float(node.GetValue())
    except Exception:
        return float(fallback)

def _frame_counter(grab, name):
    """A grab result's counter ``name`` (a FRAME_COUNTERS entry) as an int, or
    None if the result does not carry it."""
    try:
        value = getattr(grab, name)
        if callable(value):
            value = value()
    except Exception:
        try:
            value = getattr(grab, "Get" + name)()
        except Exception:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

class BaslerUSB(pylon.InstantCamera):
    '''
    BaslerUSB is an InstantCamera object which initializes the connected Basler camera.
    Excercise caution if multiple cameras are connected.

    Opening takes the process lock "basler:<serial>" (see device_lock): a
    second process gets DeviceBusy naming the holder.  Close() releases it.

    Args:
        ExposureTime (float): the exposure time in s. If below the minimum for the connected camera, sets to minimum value. (default: 0.)
        TriggerSource (str): picks the line that the camera triggers on. (default: 'Line1')
        TriggerMode (str): picks whether or not the camera waits for a trigger to capture frames. (default: 'On')
        BaslerSerialNumber (str): identifies which camera should be used via the serial number. (default: ExptParams.basler_serial_no_absorption)
    '''
    def __init__(self,ExposureTime=0.,Gain=0.,TriggerSource='Line1',TriggerMode='On',BaslerSerialNumber='40316451'):

        super().__init__()

        tl_factory = pylon.TlFactory.GetInstance()
        lock = None
        if BaslerSerialNumber == '':
            self.Attach(tl_factory.CreateFirstDevice())
            serial = self._serial()
            lock = self._take_device_lock(serial, destroy_on_busy=True)
        else:
            # lock first: a camera held elsewhere is never touched
            serial = str(BaslerSerialNumber)
            lock = self._take_device_lock(serial)
            try:
                di = pylon.DeviceInfo()
                di.SetSerialNumber(BaslerSerialNumber)
                self.Attach(tl_factory.CreateFirstDevice(di))
            except BaseException:
                self._drop_device_lock(serial, lock)
                raise

        try:
            self.Open()

            self.UserSetSelector = "Default"
            self.UserSetLoad.Execute()

            self.configure_trigger(TriggerSource, TriggerMode)

            self.set_exposure(ExposureTime)
            self.set_gain(Gain)
        except BaseException:
            try:
                pylon.InstantCamera.Close(self)
            except Exception:
                pass
            self._drop_device_lock(serial, lock)
            raise

    def _take_device_lock(self, serial, destroy_on_busy=False):
        lock = DeviceLock(f"basler:{serial}")
        try:
            lock.acquire()
        except DeviceBusy:
            if destroy_on_busy:
                try:
                    self.DestroyDevice()
                except Exception:
                    pass
            raise
        _DEVICE_LOCKS[serial] = (id(self), lock)
        return lock

    @staticmethod
    def _drop_device_lock(serial, lock):
        if lock is None:
            return
        lock.release()
        entry = _DEVICE_LOCKS.get(serial)
        if entry is not None and entry[1] is lock:
            del _DEVICE_LOCKS[serial]

    def _serial(self):
        try:
            return str(self.GetDeviceInfo().GetSerialNumber())
        except Exception:
            return ""

    def configure_trigger(self, trigger_source='Line1', trigger_mode='On'):
        """(Re)assert the hardware trigger: the line as an input, FrameStart
        triggered from it.  Called at open, and by liveOD before every run, so
        a trigger setting changed since open cannot carry into a run."""
        self.LineSelector.SetValue(trigger_source)
        self.LineMode.SetValue("Input")

        self.TriggerSelector.SetValue("FrameStart")
        self.TriggerMode.SetValue(trigger_mode)
        self.TriggerSource.SetValue(trigger_source)

    def last_clamps(self):
        """{key: (requested, applied)} for the most recent set_exposure /
        set_gain whose value the camera read back more than the tolerance away
        from the request -- out of range, or rounded to the camera's step
        (EXPOSURE_CLAMP_REL / _ABS_S, GAIN_CLAMP_DB); exposure_time in s, gain
        in dB. A key is absent when its last request was taken within tolerance."""
        return dict(_LAST_CLAMPS.get(self._serial(), {}))

    def _note_clamp(self, key, requested, applied, unit, clamped,
                    why="is outside the camera's range"):
        clamps = _LAST_CLAMPS.setdefault(self._serial(), {})
        if clamped:
            clamps[key] = (float(requested), float(applied))
            scale, shown = (1.e6, "us") if unit == "s" else (1., unit)
            logger.warning(f"Basler {self._serial()}: {key} {requested*scale:.4g} {shown} {why}; "
                           f"set to {applied*scale:.4g} {shown}")
        else:
            clamps.pop(key, None)

    def set_exposure(self,ExposureTime):
        ExposureTime_us = ExposureTime * 1.e6
        lo, hi = self.ExposureTime.GetMin(), self.ExposureTime.GetMax()
        value_us = min(max(ExposureTime_us, lo), hi)
        self.ExposureTime.SetValue(value_us)
        # what the camera took (it rounds to its own step), not what was sent
        applied = _read_back(self.ExposureTime, value_us) * 1.e-6
        tolerance = max(EXPOSURE_CLAMP_ABS_S, EXPOSURE_CLAMP_REL * abs(ExposureTime))
        clamped = abs(applied - ExposureTime) > tolerance
        why = ("is outside the camera's range" if value_us != ExposureTime_us
               else "was rounded by the camera")
        self._note_clamp("exposure_time", ExposureTime, applied, "s", clamped, why)

    def set_gain(self,Gain):
        lo, hi = self.Gain.GetMin(), self.Gain.GetMax()
        value = min(max(Gain, lo), hi)
        self.Gain.SetValue(value)
        applied = _read_back(self.Gain, value)
        clamped = abs(applied - Gain) > GAIN_CLAMP_DB
        why = "is outside the camera's range" if value != Gain else "was rounded by the camera"
        self._note_clamp("gain", Gain, applied, "dB", clamped, why)

    def Close(self):
        """Close the camera, then release the device lock."""
        serial = self._serial()
        try:
            pylon.InstantCamera.Close(self)
        finally:
            # only this object's lock: a second Close() on an old object must
            # not release the lock of a newer one on the same camera
            entry = _DEVICE_LOCKS.get(serial)
            if entry is not None and entry[0] == id(self):
                self._drop_device_lock(serial, entry[1])

    def close(self):
        self.Close()

    def open(self):
        self.Open()

    def is_opened(self):
        return self.IsOpen()

    def grab_lock(self):
        """The lock serializing grab loops on this physical camera."""
        return _grab_lock_for(self._serial())

    def start_grab(self,N_img,output_queue:Queue=None,
                   check_interrupt_method=None,on_armed=None,
                   first_frame_extra_s=0.):
        '''
        Grab N_img triggered frames, putting (img, t, idx) on output_queue.

        The first frame is waited for TIMEOUT_INIT + first_frame_extra_s (the
        run's warm-up shots, see CameraBaby), each later one TIMEOUT_RUN.

        on_armed() is called exactly once, after grabbing is confirmed started
        and before the first frame is waited for.  Frames are retrieved one by
        one in order (GrabStrategy_OneByOne); a grab the camera reports as
        failed raises FrameLostError naming its slot, so no later frame moves
        into it.  So does a frame lost without a failed result, seen as a jump
        of the result's BlockID or ImageNumber from one frame of this grab to
        the next (FRAME_COUNTERS): the frame after the gap is queued under the
        index the counter gives it, then FrameLostError names the slots in
        between.  A frame lost before the first one that arrives cannot be
        seen this way (the counters are only compared frame to frame).
        '''
        if output_queue is None:
            output_queue = Queue()
        check = check_interrupt_method or nothing
        # CameraNanny hands out one persistent camera object per key, so a
        # CameraBaby left over from an aborted run can still be inside its grab
        # loop when the next run's baby starts.  Without this lock both loops
        # drive the same InstantCamera, and the old loop's StopGrabbing() (in
        # the finally below) tears down the new run's grab -- which then times
        # out and kills the run after it, cascading until liveOD is restarted.
        with self.grab_lock():
            self._grab_loop(int(N_img), output_queue, check, on_armed,
                            first_frame_extra_s)

    def _grab_loop(self, Nimg, output_queue:Queue, check_interrupt_method, on_armed=None,
                   first_frame_extra_s=0.):
        extra_s = max(0., float(first_frame_extra_s or 0.))
        frame_timeout = TIMEOUT_INIT + extra_s # initial timeout
        # OneByOne (was LatestImages): every frame is retrieved in the order it
        # was grabbed; LatestImages may drop older frames when the host falls
        # behind, which would shift every later frame into the wrong slot.
        self.StartGrabbingMax(Nimg, pylon.GrabStrategy_OneByOne)
        count = 0
        counters = {}       # FRAME_COUNTERS name -> value on this grab's previous frame
        try:
            if not self.IsGrabbing():
                raise RuntimeError(f"Basler {self._serial()}: StartGrabbingMax({Nimg}) returned "
                                   f"but the camera is not grabbing; not arming.")
            if on_armed is not None:
                on_armed()
            deadline = time.monotonic() + frame_timeout
            while self.IsGrabbing():
                if check_interrupt_method():
                    break
                # Poll in short slices rather than blocking for the whole frame
                # timeout, so an interrupt is honoured within GRAB_POLL_INTERVAL
                # instead of holding the camera for up to TIMEOUT_INIT.
                grab = self.RetrieveResult(int(GRAB_POLL_INTERVAL*1000),
                                           pylon.TimeoutHandling_Return)
                try:
                    if grab is None or not grab.IsValid():
                        if time.monotonic() > deadline:
                            parts = (f" ({TIMEOUT_INIT:.0f} s + {extra_s:.0f} s for warm-up shots)"
                                     if count == 0 and extra_s else "")
                            raise TimeoutError(
                                f"No Basler image within {frame_timeout:.0f} s{parts} "
                                f"(got {count}/{Nimg}). Camera not triggered?")
                        continue
                    if not grab.GrabSucceeded():
                        raise FrameLostError(
                            f"Basler frame {count} of {Nimg} lost: {grab.GetErrorDescription()} "
                            f"(got {count}); later frames would have moved into its slot.",
                            lost=(count,))
                    img = np.uint8(grab.GetArray())
                    img_t = grab.TimeStamp
                    gap, jump = self._frames_skipped(grab, counters)
                finally:
                    if grab is not None and grab.IsValid():
                        grab.Release()
                if gap:
                    idx = count + gap
                    lost = tuple(range(count, min(idx, Nimg)))
                    if idx < Nimg:
                        output_queue.put((img, img_t, idx))
                    raise FrameLostError(
                        f"Basler {self._serial()}: frame(s) {list(lost)} of {Nimg} lost without a "
                        f"failed grab result ({jump}; got {count}); the frame after the gap "
                        + (f"was queued as frame {idx}, " if idx < Nimg else "is beyond the run, ")
                        + "nothing moved into the lost slots.", lost=lost)
                print(f'gotem (img {count+1}/{Nimg})')
                frame_timeout = TIMEOUT_RUN
                deadline = time.monotonic() + frame_timeout
                output_queue.put((img,img_t,count))
                count += 1
                if count >= Nimg:
                    break
        finally:
            self.StopGrabbing()

    def _frames_skipped(self, grab, counters):
        """How many frames went missing between the previous frame of this grab
        and ``grab``, by the result's own counters (FRAME_COUNTERS), and which
        counter jumped: ``(n, "BlockID went from 11 to 13")``, or ``(0, "")``.
        ``counters`` (name -> previous value; None: not usable) is updated. A
        result without a counter, or a counter that does not go up by at least
        1, turns that counter off for the rest of the grab (the latter logged)."""
        gap, jump = 0, ""
        for name in FRAME_COUNTERS:
            if name in counters and counters[name] is None:
                continue
            value = _frame_counter(grab, name)
            prev = counters.get(name)
            counters[name] = value
            if value is None or prev is None:
                continue
            step = value - prev
            if step < 1:
                counters[name] = None
                logger.warning(f"Basler {self._serial()}: {name} went from {prev} to {value}; "
                               f"lost frames are not checked by {name} for the rest of this grab")
            elif step - 1 > gap:
                gap, jump = step - 1, f"{name} went from {prev} to {value}"
        return gap, jump

    def stop_grab(self):
        # Non-blocking on purpose: if another thread owns the grab loop (a newer
        # CameraBaby that already claimed this camera), leave it alone -- its own
        # start_grab() finally will stop it.  Stopping it from a dying baby's
        # death handler is exactly what breaks the next run's grab.
        lock = self.grab_lock()
        if not lock.acquire(blocking=False):
            print("Grab loop is owned by another thread; not stopping it here.")
            return
        try:
            self.StopGrabbing()
        except Exception as e:
            print(f"Error stopping grab: {e}")
        finally:
            lock.release()
