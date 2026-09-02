import threading
import time

from pypylon import pylon
from artiq.experiment import *
import numpy as np

from queue import Queue
from PyQt6.QtCore import QThread, pyqtSignal

from waxx.config.timeouts import (CAMERA_GRAB_TIMEOUT_BASLER_INIT as TIMEOUT_INIT,
                                  CAMERA_GRAB_TIMEOUT_BASLER_RUN as TIMEOUT_RUN)

# RetrieveResult is polled in slices of this length (s) instead of blocking for
# the whole frame timeout, so an interrupted run releases the camera promptly.
GRAB_POLL_INTERVAL = 0.2

# Grab locks are keyed by serial number rather than kept on the BaslerUSB
# instance: pypylon's InstantCamera.__setattr__ routes any unknown attribute
# name into the GenICam node map, so plain attributes cannot be assigned here.
# Keying by serial also covers the case where the camera object was replaced
# (reopened) while an old one is still grabbing the same physical camera.
_GRAB_LOCKS = {}
_GRAB_LOCKS_GUARD = threading.Lock()

def _grab_lock_for(serial):
    with _GRAB_LOCKS_GUARD:
        lock = _GRAB_LOCKS.get(serial)
        if lock is None:
            lock = threading.RLock()
            _GRAB_LOCKS[serial] = lock
        return lock

def nothing():
    return False

class BaslerUSB(pylon.InstantCamera):
    '''
    BaslerUSB is an InstantCamera object which initializes the connected Basler camera.
    Excercise caution if multiple cameras are connected.

    Args:
        ExposureTime (float): the exposure time in s. If below the minimum for the connected camera, sets to minimum value. (default: 0.)
        TriggerSource (str): picks the line that the camera triggers on. (default: 'Line1')
        TriggerMode (str): picks whether or not the camera waits for a trigger to capture frames. (default: 'On')
        BaslerSerialNumber (str): identifies which camera should be used via the serial number. (default: ExptParams.basler_serial_no_absorption)
    '''
    def __init__(self,ExposureTime=0.,Gain=0.,TriggerSource='Line1',TriggerMode='On',BaslerSerialNumber='40316451'):

        super().__init__()

        tl_factory = pylon.TlFactory.GetInstance()
        if BaslerSerialNumber == '':
            self.Attach(tl_factory.CreateFirstDevice())
        else:
            di = pylon.DeviceInfo()
            di.SetSerialNumber(BaslerSerialNumber)
            self.Attach(tl_factory.CreateFirstDevice(di))

        self.Open()

        self.UserSetSelector = "Default"
        self.UserSetLoad.Execute()

        self.LineSelector = TriggerSource
        self.LineMode = "Input"

        self.TriggerSelector = "FrameStart"
        self.TriggerMode = TriggerMode
        self.TriggerSource = TriggerSource
        
        self.set_exposure(ExposureTime)
        self.set_gain(Gain)

    def set_exposure(self,ExposureTime):
        ExposureTime_us = ExposureTime * 1.e6
        if ExposureTime_us < self.ExposureTime.GetMin():
            ExposureTime_us = self.ExposureTime.GetMin()
            print(f"Exposure time requested is below camera minimum. Setting to minimum exposure : {ExposureTime_us:1.0f} us")
        if ExposureTime_us > self.ExposureTime.GetMax():
            ExposureTime_us = self.ExposureTime.GetMax()
            print(f"Exposure time requested is above camera maximum. Setting to maximum exposure : {ExposureTime_us:1.0f} us")
        self.ExposureTime.SetValue(ExposureTime_us)

    def set_gain(self,Gain):
        if Gain > self.Gain.GetMax():
            Gain = self.Gain.GetMax()
            print(f"Gain requested is above camera maximum. Setting to maximum gain : {Gain:1.0f} dB")
        if Gain > self.Gain.GetMax():
            Gain = self.Gain.GetMin()
            print(f"Gain requested is below camera minimum. Setting to minimum gain : {Gain:1.0f} dB")
        self.Gain.SetValue(Gain)

    def close(self):
        self.Close()

    def open(self):
        self.Open()
    
    def is_opened(self):
        return self.IsOpen()

    def grab_lock(self):
        """The lock serializing grab loops on this physical camera."""
        try:
            serial = str(self.GetDeviceInfo().GetSerialNumber())
        except Exception:
            serial = ""
        return _grab_lock_for(serial)

    def start_grab(self,N_img,output_queue:Queue,
                   check_interrupt_method=nothing):
        # CameraNanny hands out one persistent camera object per key, so a
        # CameraBaby left over from an aborted run can still be inside its grab
        # loop when the next run's baby starts.  Without this lock both loops
        # drive the same InstantCamera, and the old loop's StopGrabbing() (in
        # the finally below) tears down the new run's grab -- which then times
        # out and kills the run after it, cascading until liveOD is restarted.
        with self.grab_lock():
            self._grab_loop(int(N_img), output_queue, check_interrupt_method)

    def _grab_loop(self, Nimg, output_queue:Queue, check_interrupt_method):
        frame_timeout = TIMEOUT_INIT # initial timeout
        deadline = time.monotonic() + frame_timeout
        self.StartGrabbingMax(Nimg, pylon.GrabStrategy_LatestImages)
        count = 0
        try:
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
                            raise TimeoutError(
                                f"No image within {frame_timeout:.0f} s "
                                f"(got {count}/{Nimg}). Camera not triggered?")
                        continue
                    if not grab.GrabSucceeded():
                        print(f"Grab failed: {grab.GetErrorDescription()}")
                        continue
                    print(f'gotem (img {count+1}/{Nimg})')
                    img = np.uint8(grab.GetArray())
                    img_t = grab.TimeStamp
                finally:
                    if grab is not None and grab.IsValid():
                        grab.Release()
                frame_timeout = TIMEOUT_RUN
                deadline = time.monotonic() + frame_timeout
                output_queue.put((img,img_t,count))
                count += 1
                if count >= Nimg:
                    break
        finally:
            self.StopGrabbing()

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
