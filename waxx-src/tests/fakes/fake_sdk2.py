"""A fake Andor SDK2 library, swapped in for pylablib's ``wlib`` so the REAL
pylablib AndorSDK2Camera and waxx AndorEMCCD code run against it.

It is the object pylablib calls ``lib``: every SDK function it wraps
(``lib.SetTriggerMode(...)``, ``lib.GetAcquisitionTimings()``, ...) is a method
here with the same return convention as pylablib's ctypes wrapper (pointer
arguments come back as the return value; errors raise AndorSDK2LibError).  The
helper methods of the real ``AndorSDK2Lib`` (get_all_amp_modes, set_amp_mode,
get_EMCCD_gain, set_EMCCD_gain, SetRandomTracks) are inherited unchanged.

What it models:
  * hardware registers (``hw``) kept apart from pylablib's ``_cpar`` cache;
  * DRV_ACQUIRING from setters (and some getters) while acquiring, DRV_IDLE
    from AbortAcquisition when idle, P*INVALID from a bad SetImage;
  * a virtual clock: with the internal trigger a frame is produced every
    kinetic cycle as the clock advances (``advance(dt)``, or inside
    WaitForAcquisition); with the external trigger only on ``trigger(n)``;
    with the software trigger on SendSoftwareTrigger;
  * frames encode (acq_gen, idx): pixel [0, 0] = acq_gen, [0, 1] = idx
    (index within the acquisition), [0, 2] = 0xA5A5 marker;
  * a ring buffer of ``buffer_size`` frames: older ones are overwritten and
    GetImages16 refuses them with DRV_P1INVALID;
  * a call log (``log``: list of (name, args)) and one-shot fault injection
    (``fail_next(name, code)``).

Install with ``install(monkeypatch, fake)`` (patches pylablib's wlib, the
AndorSDK2 module's lib and libctl, and waxx's andor.lib).
"""
from __future__ import annotations

import collections
import time

import numpy as np

from pylablib.devices.Andor import AndorSDK2 as _sdk2_mod
from pylablib.devices.Andor import atmcd32d_lib as _lib_mod
from pylablib.devices.Andor.atmcd32d_defs import (AndorCapabilities, DRV_STATUS,
                                                  AC_FEATURES, AC_ACQMODE, AC_GETFUNC)
from pylablib.devices.Andor.atmcd32d_lib import AndorSDK2Lib, AndorSDK2LibError

D = DRV_STATUS
_CAPS_FIELDS = [name for name, _ in AndorCapabilities._fields_]
TCaps = collections.namedtuple("TCaps", _CAPS_FIELDS)

# SDK functions that are NOT refused while acquiring (everything else named
# Set*/Free*/Prepare*/Cooler* is).
_ALLOWED_WHILE_ACQUIRING = {
    "SetCurrentCamera", "SendSoftwareTrigger", "AbortAcquisition", "CancelWait",
}
# Getters the SDK refuses while acquiring (manual: GetAcquisitionTimings,
# GetKeepCleanTime, GetReadOutTime; GetTemperature on cameras without
# AC_FEATURES_TEMPERATUREDURINGACQUISITION).
_GETTERS_BLOCKED_WHILE_ACQUIRING = {
    "GetAcquisitionTimings", "GetKeepCleanTime", "GetReadOutTime",
}
_NEEDS_NO_INIT = {"GetAvailableCameras", "GetCameraHandle", "Initialize",
                  "GetCurrentCamera", "SetCurrentCamera", "ShutDown"}

TRIGGER_CODES = {"int": 0, "ext": 1, "ext_start": 6, "ext_exp": 7, "software": 10}


def _sdk(func):
    name = func.__name__

    def wrapped(self, *args):
        self.log.append((name, args))
        self._check_call(name)
        return func(self, *args)
    wrapped.__name__ = name
    wrapped._sdk = True
    return wrapped


class FakeSDK2Lib(AndorSDK2Lib):
    def __init__(self, *, detector=(512, 512), buffer_size=32, serial=6321,
                 head_model="DU897_BV", controller="CCI-24",
                 hs_speeds_mhz=(17.0, 10.0, 5.0, 1.0), conv_hs_speeds_mhz=(3.0, 1.0, 0.08),
                 preamp_gains=(1.0, 2.4, 5.1), unavailable_amp=(),
                 vs_speeds_us=(0.3, 0.5, 0.9, 1.7, 3.3), fastest_recommended_vs=1,
                 vs_amplitudes=("Normal", "+1", "+2", "+3", "+4"),
                 min_exposure=1.0e-6, exposure_quantum=None, exposure_scale=1.0,
                 readout_time=18.1e-3, keepclean_time=3.9e-3,
                 has_shutter=True, has_baseline_clamp=True,
                 temperature_during_acquisition=False,
                 trigger_modes_available=(0, 1, 6, 7, 10),
                 min_image_length=1):
        super().__init__()
        self.detector = tuple(detector)                # (width, height)
        self.buffer_size = int(buffer_size)
        self.serial = serial
        self.head_model = head_model
        self.controller = controller
        self.hs_speeds = {0: tuple(hs_speeds_mhz), 1: tuple(conv_hs_speeds_mhz)}
        self.preamp_gains = tuple(preamp_gains)
        self.unavailable_amp = {tuple(m) for m in unavailable_amp}   # (ch, oamp, hs, pa)
        self.vs_speeds = tuple(vs_speeds_us)
        self.fastest_recommended_vs = fastest_recommended_vs
        self.vs_amplitudes = tuple(vs_amplitudes)
        self.min_exposure = min_exposure
        self.exposure_quantum = exposure_quantum
        self.exposure_scale = exposure_scale
        self.readout_time = readout_time
        self.keepclean_time = keepclean_time
        self.trigger_modes_available = set(trigger_modes_available)
        self.min_image_length = int(min_image_length)
        self.blocked_getters = set(_GETTERS_BLOCKED_WHILE_ACQUIRING)
        if not temperature_during_acquisition:
            self.blocked_getters.add("GetTemperatureF")
        features = 0xFFFFFFFF
        if not has_shutter:
            features &= ~int(AC_FEATURES.AC_FEATURES_SHUTTER)
        if temperature_during_acquisition:
            features |= int(AC_FEATURES.AC_FEATURES_TEMPERATUREDURINGACQUISITION)
        else:
            features &= ~int(AC_FEATURES.AC_FEATURES_TEMPERATUREDURINGACQUISITION)
        get_funcs = 0xFFFFFFFF
        set_funcs = 0xFFFFFFFF
        if not has_baseline_clamp:
            get_funcs &= ~int(AC_GETFUNC.AC_GETFUNCTION_BASELINECLAMP)
        caps = {f: 0xFFFFFFFF for f in _CAPS_FIELDS}
        caps.update(ulFeatures=features, ulGetFunctions=get_funcs, ulSetFunctions=set_funcs,
                    ulCameraType=3, ulPixelMode=0, ulSize=0)
        self.caps = TCaps(**caps)

        self.log: list = []
        self._fail: dict = {}
        self.initialized = False
        self.shutdown_count = 0
        self.handle = 100
        self.current = None
        # hardware registers (what the camera has), apart from pylablib's _cpar
        self.hw = dict(
            acq_mode=1, read_mode=4, trigger=0, fast_ext=0, trigger_invert=0,
            trigger_term=0, trigger_level=None, ft=0,
            image=(1, 1, 1, self.detector[0], 1, self.detector[1]),
            adc=0, oamp=0, hs=0, preamp=0, vs=0, vs_amp=0,
            em_gain_mode=0, em_advanced=0, em_gain=0,
            exposure=0.0, kinetic_cycle=0.0, accum_cycle=0.0, n_kinetics=1, n_accum=1,
            baseline_clamp=0, shutter=None, cooler=0, cooler_mode=0, fan=0,
            temperature_setpoint=20, camlink=0, crop=None,
        )
        self.temperature = -60.0
        # acquisition
        self.clock = 0.0
        self.acquiring = False
        self.acq_gen = 0
        self.n_acquired = 0
        self.frames: dict = {}
        self._events = 0
        self._next_frame_t = None
        self.prepared = False

    # -- test controls --------------------------------------------------------
    def fail_next(self, name, code=int(D.DRV_ERROR_ACK)):
        """Make the next call of SDK function `name` raise with `code`."""
        self._fail[name] = int(code)

    def trigger(self, n=1):
        """n external trigger edges (frames only in ext mode while acquiring)."""
        for _ in range(int(n)):
            if self.acquiring and self.hw["trigger"] in (1, 6, 7):
                self._produce_frame()

    def advance(self, dt):
        """Advance the virtual clock; internal-trigger frames fall due."""
        self.clock += float(dt)
        self._produce_due_frames()

    def calls(self, name=None):
        return [c for c in self.log if name is None or c[0] == name]

    def names(self):
        return [c[0] for c in self.log]

    def index(self, name, start=0):
        for i, c in enumerate(self.log[start:], start):
            if c[0] == name:
                return i
        return -1

    def frame_rows_cols(self):
        hbin, vbin, hs, he, vs, ve = self.hw["image"]
        return (ve - vs + 1) // vbin, (he - hs + 1) // hbin

    @staticmethod
    def decode(frame):
        """(acq_gen, idx) a frame was made with."""
        return int(frame[0, 0]), int(frame[0, 1])

    # -- plumbing --------------------------------------------------------------
    def initlib(self):          # never loads a DLL
        self._initialized = True

    def _check_call(self, name):
        if name in self._fail:
            code = self._fail.pop(name)
            raise AndorSDK2LibError(name, code)
        if name not in _NEEDS_NO_INIT and not self.initialized:
            raise AndorSDK2LibError(name, int(D.DRV_NOT_INITIALIZED))
        if self.acquiring:
            mutating = name.startswith(("Set", "Free", "Prepare", "Cooler", "StartAcq")) \
                or name in ("EnableKeepCleans",)
            if (mutating and name not in _ALLOWED_WHILE_ACQUIRING) or name in self.blocked_getters:
                raise AndorSDK2LibError(name, int(D.DRV_ACQUIRING))

    def _err(self, name, code):
        raise AndorSDK2LibError(name, int(code))

    def _actual_exposure(self):
        exp = max(float(self.hw["exposure"]), self.min_exposure) * self.exposure_scale
        if self.exposure_quantum:
            q = self.exposure_quantum
            exp = np.ceil(exp / q - 1e-9) * q
        return exp

    def _cycle(self):
        exp = self._actual_exposure()
        floor = exp + self.readout_time + self.keepclean_time
        return max(float(self.hw["kinetic_cycle"]), floor)

    def _produce_frame(self):
        rows, cols = self.frame_rows_cols()
        idx = self.n_acquired
        img = np.full((rows, cols), (self.acq_gen * 1000 + idx) & 0xFFFF, dtype="<u2")
        img[0, 0] = self.acq_gen
        img[0, 1] = idx
        if cols > 2:
            img[0, 2] = 0xA5A5
        self.frames[idx] = img
        self.n_acquired += 1
        self._events += 1
        for old in [k for k in self.frames if k < self.n_acquired - self.buffer_size]:
            del self.frames[old]
        mode = self.hw["acq_mode"]
        if (mode == 1 and self.n_acquired >= 1) or (mode == 3 and self.n_acquired >= self.hw["n_kinetics"]):
            self.acquiring = False

    def _produce_due_frames(self):
        if not self.acquiring or self.hw["trigger"] != 0 or self._next_frame_t is None:
            return
        while self.acquiring and self._next_frame_t <= self.clock + 1e-15:
            self._produce_frame()
            self._next_frame_t += self._cycle()

    # -- SDK: library / camera selection --------------------------------------
    @_sdk
    def GetAvailableCameras(self):
        return 1

    @_sdk
    def GetCameraHandle(self, idx):
        if idx != 0:
            self._err("GetCameraHandle", D.DRV_P1INVALID)
        return self.handle

    @_sdk
    def GetCurrentCamera(self):
        return self.current if self.current is not None else self.handle

    @_sdk
    def SetCurrentCamera(self, handle):
        self.current = handle

    @_sdk
    def Initialize(self, path):
        self.initialized = True

    @_sdk
    def ShutDown(self):
        if not self.initialized:
            self._err("ShutDown", D.DRV_NOT_INITIALIZED)
        self.initialized = False
        self.acquiring = False
        self.shutdown_count += 1

    # -- SDK: info -------------------------------------------------------------
    @_sdk
    def GetCapabilities(self):
        return self.caps

    @_sdk
    def GetControllerCardModel(self):
        return self.controller.encode()

    @_sdk
    def GetHeadModel(self):
        return self.head_model.encode()

    @_sdk
    def GetCameraSerialNumber(self):
        return self.serial

    @_sdk
    def GetPixelSize(self):
        return (16.0, 16.0)

    @_sdk
    def GetDetector(self):
        return self.detector

    @_sdk
    def GetStatus(self):
        return int(D.DRV_ACQUIRING) if self.acquiring else int(D.DRV_IDLE)

    # -- SDK: temperature ------------------------------------------------------
    @_sdk
    def GetTemperatureRange(self):
        return (-100, 20)

    @_sdk
    def SetTemperature(self, t):
        self.hw["temperature_setpoint"] = int(t)

    @_sdk
    def GetTemperatureF(self):
        return (int(D.DRV_TEMPERATURE_STABILIZED), self.temperature)

    @_sdk
    def CoolerON(self):
        self.hw["cooler"] = 1

    @_sdk
    def CoolerOFF(self):
        self.hw["cooler"] = 0

    @_sdk
    def IsCoolerOn(self):
        return self.hw["cooler"]

    @_sdk
    def SetCoolerMode(self, mode):
        self.hw["cooler_mode"] = int(mode)

    @_sdk
    def SetFanMode(self, mode):
        self.hw["fan"] = int(mode)

    # -- SDK: amplifiers, shift speeds, gain ------------------------------------
    @_sdk
    def GetNumberADChannels(self):
        return 1

    @_sdk
    def GetBitDepth(self, ch):
        return 14

    @_sdk
    def GetNumberAmp(self):
        return 2

    @_sdk
    def GetAmpDesc(self, i):
        return b"Electron Multiplying" if i == 0 else b"Conventional"

    @_sdk
    def GetNumberHSSpeeds(self, ch, oamp):
        return len(self.hs_speeds[oamp])

    @_sdk
    def GetHSSpeed(self, ch, oamp, i):
        speeds = self.hs_speeds[oamp]
        if not 0 <= i < len(speeds):
            self._err("GetHSSpeed", D.DRV_P3INVALID)
        return speeds[i]

    @_sdk
    def GetNumberPreAmpGains(self):
        return len(self.preamp_gains)

    @_sdk
    def GetPreAmpGain(self, i):
        return self.preamp_gains[i]

    @_sdk
    def IsPreAmpGainAvailable(self, ch, oamp, hs, pa):
        return 0 if (ch, oamp, hs, pa) in self.unavailable_amp else 1

    @_sdk
    def SetADChannel(self, ch):
        self.hw["adc"] = int(ch)

    @_sdk
    def SetOutputAmplifier(self, oamp):
        self.hw["oamp"] = int(oamp)

    @_sdk
    def SetHSSpeed(self, oamp, i):
        if not 0 <= i < len(self.hs_speeds[oamp]):
            self._err("SetHSSpeed", D.DRV_P2INVALID)
        self.hw["hs"] = int(i)

    @_sdk
    def SetPreAmpGain(self, i):
        if not 0 <= i < len(self.preamp_gains):
            self._err("SetPreAmpGain", D.DRV_P1INVALID)
        self.hw["preamp"] = int(i)

    @_sdk
    def GetNumberVSSpeeds(self):
        return len(self.vs_speeds)

    @_sdk
    def GetVSSpeed(self, i):
        return self.vs_speeds[i]

    @_sdk
    def GetFastestRecommendedVSSpeed(self):
        return (self.fastest_recommended_vs, self.vs_speeds[self.fastest_recommended_vs])

    @_sdk
    def SetVSSpeed(self, i):
        if not 0 <= i < len(self.vs_speeds):
            self._err("SetVSSpeed", D.DRV_P1INVALID)
        self.hw["vs"] = int(i)

    @_sdk
    def GetNumberVSAmplitudes(self):
        return len(self.vs_amplitudes)

    @_sdk
    def GetVSAmplitudeString(self, i):
        return self.vs_amplitudes[i].encode()

    @_sdk
    def SetVSAmplitude(self, i):
        if not 0 <= i < len(self.vs_amplitudes):
            self._err("SetVSAmplitude", D.DRV_P1INVALID)
        self.hw["vs_amp"] = int(i)

    @_sdk
    def SetEMGainMode(self, mode):
        self.hw["em_gain_mode"] = int(mode)

    @_sdk
    def GetEMGainRange(self):
        return (1, 300) if not self.hw["em_advanced"] else (1, 1000)

    @_sdk
    def SetEMAdvanced(self, state):
        self.hw["em_advanced"] = int(state)

    @_sdk
    def GetEMAdvanced(self):
        return self.hw["em_advanced"]

    @_sdk
    def SetEMCCDGain(self, gain):
        top = 1000 if self.hw["em_advanced"] else 300
        if not 0 <= gain <= top:
            self._err("SetEMCCDGain", D.DRV_P1INVALID)
        self.hw["em_gain"] = int(gain)

    @_sdk
    def GetEMCCDGain(self):
        return self.hw["em_gain"]

    @_sdk
    def SetBaselineClamp(self, state):
        self.hw["baseline_clamp"] = int(state)

    @_sdk
    def GetBaselineClamp(self):
        return self.hw["baseline_clamp"]

    # -- SDK: shutter, trigger ----------------------------------------------------
    @_sdk
    def GetShutterMinTimes(self):
        return (27, 27)

    @_sdk
    def SetShutter(self, typ, mode, closing, opening):
        self.hw["shutter"] = (int(typ), int(mode), closing, opening)

    @_sdk
    def SetTriggerMode(self, mode):
        self.hw["trigger"] = int(mode)

    @_sdk
    def IsTriggerModeAvailable(self, mode):
        if mode not in self.trigger_modes_available:
            self._err("IsTriggerModeAvailable", D.DRV_INVALID_MODE)

    @_sdk
    def GetTriggerLevelRange(self):
        return (0.0, 5.0)

    @_sdk
    def SetTriggerLevel(self, level):
        self.hw["trigger_level"] = level

    @_sdk
    def SetTriggerInvert(self, invert):
        self.hw["trigger_invert"] = int(invert)

    @_sdk
    def SetExternalTriggerTermination(self, term):
        self.hw["trigger_term"] = int(term)

    @_sdk
    def SetFastExtTrigger(self, mode):
        self.hw["fast_ext"] = int(mode)

    @_sdk
    def SendSoftwareTrigger(self):
        if self.acquiring and self.hw["trigger"] == 10:
            self._produce_frame()

    @_sdk
    def SetCameraLinkMode(self, state):
        self.hw["camlink"] = int(state)

    @_sdk
    def SetIsolatedCropModeType(self, mode):
        self.hw["crop"] = int(mode)

    # -- SDK: acquisition mode & timing ---------------------------------------------
    @_sdk
    def SetAcquisitionMode(self, mode):
        if mode not in (1, 2, 3, 4, 5):
            self._err("SetAcquisitionMode", D.DRV_P1INVALID)
        self.hw["acq_mode"] = int(mode)

    @_sdk
    def SetNumberAccumulations(self, n):
        self.hw["n_accum"] = int(n)

    @_sdk
    def SetNumberKinetics(self, n):
        self.hw["n_kinetics"] = int(n)

    @_sdk
    def SetNumberPrescans(self, n):
        pass

    @_sdk
    def SetKineticCycleTime(self, t):
        self.hw["kinetic_cycle"] = float(t)

    @_sdk
    def SetAccumulationCycleTime(self, t):
        self.hw["accum_cycle"] = float(t)

    @_sdk
    def SetExposureTime(self, t):
        self.hw["exposure"] = float(t)

    @_sdk
    def GetAcquisitionTimings(self):
        f32 = lambda x: float(np.float32(x))           # the SDK returns c_float
        exp = self._actual_exposure()
        return (f32(exp), f32(max(self.hw["accum_cycle"], exp)), f32(self._cycle()))

    @_sdk
    def GetMaximumExposure(self):
        return 1.0e3

    @_sdk
    def SetFrameTransferMode(self, mode):
        self.hw["ft"] = int(mode)

    @_sdk
    def GetReadOutTime(self):
        return float(np.float32(self.readout_time))

    @_sdk
    def GetKeepCleanTime(self):
        return float(np.float32(self.keepclean_time))

    # -- SDK: read mode & image area --------------------------------------------
    @_sdk
    def SetReadMode(self, mode):
        self.hw["read_mode"] = int(mode)

    @_sdk
    def SetSingleTrack(self, centre, height):
        pass

    @_sdk
    def SetMultiTrack(self, number, height, offset):
        return (1, 0)

    @_sdk
    def SetRandomTracks_lib(self, ntracks, areas):
        pass

    @_sdk
    def SetImage(self, hbin, vbin, hstart, hend, vstart, vend):
        w, h = self.detector
        if not (1 <= hstart <= hend <= w):
            self._err("SetImage", D.DRV_P3INVALID if not 1 <= hstart <= w else D.DRV_P4INVALID)
        if not (1 <= vstart <= vend <= h):
            self._err("SetImage", D.DRV_P5INVALID if not 1 <= vstart <= h else D.DRV_P6INVALID)
        if hend - hstart + 1 < self.min_image_length or vend - vstart + 1 < self.min_image_length:
            self._err("SetImage", D.DRV_P4INVALID)
        if hbin < 1 or vbin < 1 or (hend - hstart + 1) % hbin or (vend - vstart + 1) % vbin:
            self._err("SetImage", D.DRV_P1INVALID if hbin < 1 or (hend - hstart + 1) % hbin
                      else D.DRV_P2INVALID)
        self.hw["image"] = (int(hbin), int(vbin), int(hstart), int(hend), int(vstart), int(vend))

    # -- SDK: acquisition ----------------------------------------------------------
    @_sdk
    def GetSizeOfCircularBuffer(self):
        return self.buffer_size

    @_sdk
    def PrepareAcquisition(self):
        self.prepared = True

    @_sdk
    def FreeInternalMemory(self):
        self.frames.clear()
        self.prepared = False

    @_sdk
    def StartAcquisition(self):
        self.acquiring = True
        self.acq_gen += 1
        self.n_acquired = 0
        self.frames.clear()
        self._events = 0
        self._next_frame_t = self.clock + self._cycle() if self.hw["trigger"] == 0 else None

    @_sdk
    def AbortAcquisition(self):
        if not self.acquiring:
            self._err("AbortAcquisition", D.DRV_IDLE)
        self.acquiring = False

    @_sdk
    def GetAcquisitionProgress(self):
        return (0, self.n_acquired)

    @_sdk
    def WaitForAcquisitionByHandleTimeOut(self, handle, timeout_ms):
        if self._events == 0 and self.acquiring and self.hw["trigger"] == 0 \
                and self._next_frame_t is not None:
            due = self._next_frame_t - self.clock
            if due <= timeout_ms * 1e-3:
                self.advance(max(due, 0.0))
        if self._events > 0:
            self._events -= 1
            return None
        self.clock += timeout_ms * 1e-3
        time.sleep(0.0005)          # do not spin a CPU in pylablib's wait loop
        self._err("WaitForAcquisitionByHandleTimeOut", D.DRV_NO_NEW_DATA)

    @_sdk
    def CancelWait(self):
        pass

    @_sdk
    def GetImages16(self, first, last, size):
        rows, cols = self.frame_rows_cols()
        idxs = range(first - 1, last)
        if first < 1 or last < first:
            self._err("GetImages16", D.DRV_P1INVALID)
        for i in idxs:
            if i not in self.frames:
                self._err("GetImages16", D.DRV_P1INVALID if i == first - 1 else D.DRV_P2INVALID)
        if size != rows * cols * len(idxs):
            self._err("GetImages16", D.DRV_P3INVALID)
        data = np.concatenate([self.frames[i].ravel() for i in idxs])
        return (data, first, last)


def install(monkeypatch, fake):
    """Route pylablib's AndorSDK2 and waxx's AndorEMCCD to `fake`."""
    import waxx.control.cameras.andor as andor_mod
    ctl = _sdk2_mod.LibraryController(fake)
    monkeypatch.setattr(_lib_mod, "wlib", fake)
    monkeypatch.setattr(_sdk2_mod, "lib", fake)
    monkeypatch.setattr(_sdk2_mod, "libctl", ctl)
    monkeypatch.setattr(andor_mod, "lib", fake)
    return ctl
