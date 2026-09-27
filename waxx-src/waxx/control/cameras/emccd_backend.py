"""The Andor EMCCD as a ``beacon.camera.backend.CameraBackend``.

One ``EMCCDBackend`` wraps one ``AndorEMCCD``.  Only its camera worker thread
calls it.  Every ``apply`` is a full state, never a change on top: it stops the
acquisition and sets everything in the order below, reads every setting back
before any acquisition starts (several SDK getters refuse while acquiring),
and raises ``ApplyMismatch`` when the camera does not have what was asked.

Keys are ``beacon.camera.schema.ANDOR_EMCCD``'s.  A profile must carry every
run parameter (exposure_time, gain, trigger_mode, frame_transfer, sensor_roi,
hs_speed, preamp, vs_speed, vs_amp, baseline_clamp); the fixed ones, when
present, must hold their fixed value; owner-only ones (temperature_setpoint,
cooler, fan) are applied when not None; status keys (temperature,
cooler_status) are ignored; ``em_gain_unlocked`` lifts the live EM-gain cap.
Anything else is refused.

Apply order (PLAN C4):
    validate -> stop + clear_acquisition -> setup_cont_mode(0)
    -> read mode image, fast ext trigger 0, ext trigger rising / high-Z
    -> trigger mode -> frame transfer off -> setup_image_mode(*sensor_roi)
    -> set_amp_mode(0, 0, hs, preamp) -> vs speed -> vs amplitude
    -> EM gain mode 3 -> EM gain (advanced=False) -> exposure -> baseline clamp
    -> shutter (open|closed, never auto) -> cooler/fan/setpoint, camera link,
       cooler mode 1 -> setup_acquisition("cont") -> trigger available?
    -> readback -> must-match check
"""
from __future__ import annotations

import logging
import time

import numpy as np

from beacon.camera.backend import (Readback, RawFrame, CloseReport, CameraError,
                                   ApplyRefused, ApplyMismatch)

from waxx.control.cameras.camera_param_classes import (AndorParams, RunFieldRefused,
                                                       check_andor_run_fields)

logger = logging.getLogger(__name__)

CATEGORY = "andor_emccd"
EM_GAIN_MAX = 300           # never above, never advanced mode
EXPOSURE_TOLERANCE = 0.05   # readback may exceed the request by this fraction
FLOAT32_REL = 1e-6          # the SDK reports times as c_float

RUN_PARAMS = ("exposure_time", "gain", "trigger_mode", "frame_transfer", "sensor_roi",
              "hs_speed", "preamp", "vs_speed", "vs_amp", "baseline_clamp")
FIXED = {
    "em_gain_mode": 3, "em_advanced": 0, "output_amp": 0, "read_mode": "image",
    "acquisition_mode": "cont", "fast_ext_trigger": 0, "ext_trigger_edge": "rising",
    "ext_trigger_termination": "high_z", "camera_link": 1, "cooler_mode": 1,
}
OWNER_ONLY = ("temperature_setpoint", "cooler", "fan")
STATUS_KEYS = ("temperature", "cooler_status")
CONTROL_KEYS = ("em_gain_unlocked",)
SHUTTER_MODES = ("open", "closed")
FAN_MODES = ("full", "low", "off")
MODES = ("live", "run", "snap")


def _plain(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _int(key, value):
    value = _plain(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ApplyRefused(key, f"{value!r} is not an integer")


def _float(key, value):
    value = _plain(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ApplyRefused(key, f"{value!r} is not a finite number")
    return float(value)


def _default_factory(andor_kwargs):
    def make():
        from waxx.control.cameras.andor import AndorEMCCD
        return AndorEMCCD(**andor_kwargs)
    return make


class EMCCDBackend:
    """CameraBackend for the Andor DU897 through ``AndorEMCCD``.

    ``camera_factory()`` returns an opened AndorEMCCD-like object (tests inject
    a fake); by default ``AndorEMCCD(**andor_kwargs)`` with EM gain 0 unless
    given.  ``serial``, if given, is the serial expected at open.
    ``refuse_combos`` is a sequence of ``(((key, value), ...), reason)``: a
    profile matching every pair is refused, live and run alike.
    """
    category = CATEGORY

    def __init__(self, camera_factory=None, *, serial=None, refuse_combos=(),
                 live_em_gain_cap=100, **andor_kwargs):
        andor_kwargs.setdefault("gain", 0)
        self._factory = camera_factory or _default_factory(andor_kwargs)
        self._expected_serial = None if serial is None else str(serial)
        self.camera_id = f"{CATEGORY}:{serial if serial is not None else 'unknown'}"
        self._refuse_combos = tuple((tuple((str(k), _plain(v)) for k, v in conds), str(reason))
                                    for conds, reason in refuse_combos)
        self._live_cap = int(live_em_gain_cap)
        self._cam = None
        self._info: dict = {}
        self._caps: dict = {}
        self._reset_state()

    def _reset_state(self):
        self._applied_purpose = None
        self._last_readback: dict = {}
        self._commanded: dict = {}
        self._last_status: dict = {}
        self._cycle_s = None
        self._mode = None
        self._n_frames = None
        self._next_idx = 0
        self._live_lost = 0

    # -- open / close ------------------------------------------------------------
    def open(self) -> dict:
        if self._cam is not None:
            return dict(self._info)
        cam = self._factory()   # DeviceBusy (a CameraUnavailable) when held elsewhere
        try:
            info = cam.get_device_info()
            serial = str(info.serial_number)
            if self._expected_serial is not None and serial != self._expected_serial:
                raise CameraError(f"expected Andor serial {self._expected_serial}, "
                                  f"found {serial}")
            cam.set_frame_format("list")        # one 2D array per frame
            caps = self._read_caps(cam)
        except BaseException:
            try:
                cam.Close()
            except Exception:
                pass
            raise
        self._cam = cam
        self._caps = caps
        self._reset_state()
        self.camera_id = f"{CATEGORY}:{serial}"
        self._info = {"camera_id": self.camera_id, "serial": serial,
                      "model": str(info.head_model), "controller": str(info.controller_model),
                      "detector": tuple(int(v) for v in cam.get_detector_size())}
        return dict(self._info)

    def _read_caps(self, cam) -> dict:
        """What the camera offers, read once at open.  A query the camera does
        not answer leaves its entry empty (and the check that uses it off);
        it does not stop the open."""
        def query(name, fn, default):
            try:
                return fn()
            except Exception as e:
                logger.warning(f"{self.camera_id}: {name} unavailable ({e}); "
                               f"not checked in apply")
                return default
        modes = query("amplifier modes", lambda: [m for m in cam.get_all_amp_modes()
                                                  if m.channel == 0 and m.oamp == 0], [])
        hs = sorted({(int(m.hsspeed), float(m.hsspeed_MHz)) for m in modes})
        preamps = sorted({(int(m.preamp), float(m.preamp_gain)) for m in modes})
        return {
            "amp_modes": {(int(m.hsspeed), int(m.preamp)) for m in modes},
            "hs": hs,
            "preamps": preamps,
            "preamps_by_hs": {h: sorted({int(m.preamp) for m in modes if m.hsspeed == h})
                              for h, _ in hs},
            "vs": query("vertical speeds", lambda: [float(v) for v in cam.get_all_vsspeeds()], []),
            "vs_amp_labels": query("vertical clock amplitudes",
                                   lambda: list(cam.get_vsamplitude_labels()), []),
            "max_exposure": query("maximum exposure", lambda: float(cam.get_max_exposure()), None),
            "temperature_range": query("temperature range",
                                       lambda: tuple(cam.get_temperature_range()), None),
            "triggers": query("trigger modes",
                              lambda: tuple(m for m in ("int", "software", "ext")
                                            if cam.is_trigger_mode_available(m)), ()),
        }

    def close(self) -> CloseReport:
        cam = self._cam
        if cam is None:
            return CloseReport(attached_after=False, accessible_after=None, errors=())
        errors = []
        try:
            logger.info(f"{self.camera_id}: closing at {cam.get_temperature():.1f} C")
        except Exception:
            pass
        try:
            # stop -> shutter closed -> cooler mode 1, each in its own try, then close
            errors.extend(cam.close_safely() or [])
        except Exception as e:                  # a camera object without close_safely
            errors.append(f"close: {e}")
        try:
            attached = bool(cam.is_opened())
        except Exception:
            attached = None
        lock = getattr(cam, "_device_lock", None)
        lock_free = None if lock is None else not lock.held
        self._cam = None
        self._reset_state()
        accessible = (attached is False and lock_free is not False) if not errors else None
        return CloseReport(attached_after=attached, accessible_after=accessible,
                           errors=tuple(errors))

    def _require_open(self):
        if self._cam is None:
            raise CameraError(f"{self.camera_id} is not open")
        return self._cam

    # -- description ----------------------------------------------------------------
    def describe_dynamic(self) -> dict:
        c = self._caps
        if not c:
            return {}
        out = {
            "hs_speed": {"choices": [h for h, _ in c["hs"]],
                         "labels": [f"{mhz:g} MHz" for _, mhz in c["hs"]]},
            "preamp": {"choices": [p for p, _ in c["preamps"]],
                       "labels": [f"x{g:g}" for _, g in c["preamps"]],
                       "by_hs_speed": {h: list(p) for h, p in c["preamps_by_hs"].items()}},
            "vs_speed": {"choices": list(range(len(c["vs"]))),
                         "labels": [f"{us:g} µs" for us in c["vs"]]},
            "vs_amp": {"choices": list(range(len(c["vs_amp_labels"]))),
                       "labels": list(c["vs_amp_labels"])},
            "gain": {"range": (0, EM_GAIN_MAX), "live_cap": self._live_cap},
            "trigger_mode": {"choices": list(c["triggers"])},
        }
        if c["max_exposure"] is not None:
            out["exposure_time"] = {"range": (0.0, c["max_exposure"])}
        if c["temperature_range"] is not None:
            out["temperature_setpoint"] = {"range": c["temperature_range"]}
        return out

    # -- apply ---------------------------------------------------------------------
    def _validate(self, values: dict, purpose: str) -> dict:
        if purpose not in ("live", "run"):
            raise ApplyRefused("purpose", f"{purpose!r} is not 'live' or 'run'")
        known = set(RUN_PARAMS) | set(FIXED) | set(OWNER_ONLY) | set(STATUS_KEYS) \
            | set(CONTROL_KEYS) | {"shutter"}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ApplyRefused(unknown[0], f"not an {CATEGORY} setting (unknown keys {unknown})")
        missing = [k for k in RUN_PARAMS if k not in values]
        if missing:
            raise ApplyRefused(missing[0], f"missing: apply takes a full profile, never a "
                                           f"change on top (missing {missing})")
        c = self._caps
        p = {}
        try:
            p["trigger_mode"], p["frame_transfer"], p["sensor_roi"] = check_andor_run_fields(
                values["trigger_mode"], values["frame_transfer"], values["sensor_roi"],
                purpose=purpose, detector_shape=AndorParams.DETECTOR_SHAPE)
        except RunFieldRefused as e:
            raise ApplyRefused(e.field, f"{e.value!r} is refused: {e.reason}") from None
        if c.get("triggers") and p["trigger_mode"] not in c["triggers"]:
            raise ApplyRefused("trigger_mode", f"{p['trigger_mode']!r} is not available on this "
                                               f"camera (available: {list(c['triggers'])})")

        exposure = _float("exposure_time", values["exposure_time"])
        if exposure < 0 or (purpose == "run" and exposure <= 0):
            raise ApplyRefused("exposure_time", f"{exposure!r} s: a {purpose} needs a "
                               f"{'positive' if purpose == 'run' else 'non-negative'} exposure")
        if c.get("max_exposure") and exposure > c["max_exposure"]:
            raise ApplyRefused("exposure_time", f"{exposure!r} s is above the camera's "
                                                f"maximum {c['max_exposure']!r} s")
        p["exposure_time"] = exposure

        unlocked = bool(_plain(values.get("em_gain_unlocked", False)))
        gain = _int("gain", values["gain"])
        if not 0 <= gain <= EM_GAIN_MAX:
            raise ApplyRefused("gain", f"EM gain {gain} is outside 0..{EM_GAIN_MAX} "
                                       f"(never above {EM_GAIN_MAX}, never advanced mode)")
        if purpose == "live" and gain > self._live_cap and not unlocked:
            raise ApplyRefused("gain", f"live EM gain {gain} is above the live cap "
                                       f"{self._live_cap}; tick 'unlock' to go up to {EM_GAIN_MAX}")
        p["gain"] = gain

        for key in ("hs_speed", "preamp", "vs_speed", "vs_amp", "baseline_clamp"):
            p[key] = _int(key, values[key])
            if p[key] < 0:
                raise ApplyRefused(key, f"{p[key]} is negative")
        if c.get("amp_modes") and (p["hs_speed"], p["preamp"]) not in c["amp_modes"]:
            raise ApplyRefused("hs_speed", f"hs_speed {p['hs_speed']} with preamp {p['preamp']} "
                                           f"is not an amplifier mode of this camera (output amp 0; "
                                           f"offered (hs_speed, preamp): {sorted(c['amp_modes'])})")
        if c.get("vs") and p["vs_speed"] >= len(c["vs"]):
            raise ApplyRefused("vs_speed", f"index {p['vs_speed']} is out of range "
                                           f"(0..{len(c['vs']) - 1})")
        if c.get("vs_amp_labels") and p["vs_amp"] >= len(c["vs_amp_labels"]):
            raise ApplyRefused("vs_amp", f"index {p['vs_amp']} is out of range "
                                         f"(0..{len(c['vs_amp_labels']) - 1})")
        if p["baseline_clamp"] not in (0, 1):
            raise ApplyRefused("baseline_clamp", f"{p['baseline_clamp']} is not 0 or 1")

        shutter = _plain(values.get("shutter", "open"))
        if shutter not in SHUTTER_MODES:
            raise ApplyRefused("shutter", f"{shutter!r} is refused: only 'open' or 'closed' "
                                          f"(never 'auto')")
        if purpose == "run" and shutter != "open":
            raise ApplyRefused("shutter", f"{shutter!r} is refused for a run: runs image "
                                          f"with the shutter open")
        p["shutter"] = shutter

        for key, fixed in FIXED.items():
            if key in values:
                got = _plain(values[key])
                if isinstance(fixed, int):
                    got = _int(key, got)
                if got != fixed:
                    raise ApplyRefused(key, f"{got!r} is refused: pinned to {fixed!r}")

        setpoint = _plain(values.get("temperature_setpoint"))
        if setpoint is not None:
            setpoint = _float("temperature_setpoint", setpoint)
            lo, hi = c.get("temperature_range") or (-np.inf, np.inf)
            if not lo <= setpoint <= hi:
                raise ApplyRefused("temperature_setpoint", f"{setpoint!r} C is outside the "
                                                           f"camera's range {lo}..{hi} C")
        p["temperature_setpoint"] = setpoint
        cooler = _plain(values.get("cooler"))
        p["cooler"] = None if cooler is None else bool(cooler)
        fan = _plain(values.get("fan"))
        if fan is not None and fan not in FAN_MODES:
            raise ApplyRefused("fan", f"{fan!r} is not one of {FAN_MODES}")
        p["fan"] = fan

        for conds, reason in self._refuse_combos:
            if all(p.get(k, _plain(values.get(k))) == v for k, v in conds):
                keys = ", ".join(k for k, _ in conds)
                what = " and ".join(f"{k}={v!r}" for k, v in conds)
                raise ApplyRefused(keys, f"{what} is refused: {reason}")
        return p

    def apply(self, values: dict, purpose: str) -> dict:
        cam = self._require_open()
        p = self._validate(dict(values), purpose)      # nothing sent before this passes
        self._applied_purpose = None
        self._mode = None

        cam.stop_acquisition()
        cam.clear_acquisition()
        cam.setup_cont_mode(0)
        cam.set_read_mode("image")
        cam.set_fast_trigger_mode(0)
        cam.setup_ext_trigger(invert=False, term_highZ=True)      # rising edge, high-Z
        cam.set_trigger_mode(p["trigger_mode"])
        ft_sent = cam.frame_transfer_off().source != "unsupported"
        cam.setup_image_mode(*p["sensor_roi"])
        cam.set_amp_mode_checked(channel=0, oamp=0, hsspeed=p["hs_speed"], preamp=p["preamp"])
        cam.set_vsspeed(p["vs_speed"])
        cam.set_vsamplitude(p["vs_amp"])
        cam.set_EM_gain_mode(FIXED["em_gain_mode"])               # before the gain
        cam.set_EMCCD_gain(p["gain"], advanced=False)
        cam.set_exposure(p["exposure_time"])
        cam.set_baseline_clamp(p["baseline_clamp"])
        cam.setup_shutter_if_supported(p["shutter"])
        if p["temperature_setpoint"] is not None:
            cam.set_temperature(p["temperature_setpoint"], enable_cooler=False)
        if p["cooler"] is not None:
            cam.set_cooler(p["cooler"])
        if p["fan"] is not None:
            cam.set_fan_mode(p["fan"])
        cam.activate_cameralink(FIXED["camera_link"])
        cam.set_cooler_mode(FIXED["cooler_mode"])
        cam.setup_acquisition("cont")
        self._commanded = {
            "vs_amp": p["vs_amp"],
            "fast_ext_trigger": FIXED["fast_ext_trigger"],
            "em_gain_mode": FIXED["em_gain_mode"],
            "camera_link": FIXED["camera_link"],
            "cooler_mode": FIXED["cooler_mode"],
            "_ft_sent": ft_sent,
        }
        trigger_available = cam.is_trigger_mode_available(p["trigger_mode"])

        readback = self._readback(p)                   # every getter before any start
        mismatches = self._mismatches(p, readback)
        if not trigger_available:
            mismatches["trigger_mode"] = (p["trigger_mode"],
                                          "not available with these settings (IsTriggerModeAvailable)")
        if mismatches:
            raise ApplyMismatch(mismatches)
        self._applied_purpose = purpose
        self._last_readback = readback
        self._cycle_s = readback["cycle_time"].value
        return readback

    def read_settings(self) -> dict:
        cam = self._require_open()
        if cam.acquisition_in_progress():
            # several getters refuse while acquiring: the last apply's readback
            return dict(self._last_readback)
        return self._readback(None)

    def _readback(self, p) -> dict:
        cam = self._cam
        cmd = self._commanded
        rb = {}
        exposure, _, kinetic = cam.get_cycle_timings()
        exposure = float(exposure)
        req = None if p is None else p["exposure_time"]
        if req is not None and abs(exposure - req) > FLOAT32_REL * max(req, 1e-9):
            rb["exposure_time"] = Readback(exposure, "hw", origin="clamped", requested=req)
        else:
            rb["exposure_time"] = Readback(exposure, "hw")
        rb["cycle_time"] = Readback(float(kinetic), "hw")
        readout = cam.get_readout_time()
        rb["readout_time"] = Readback(None if readout is None else float(readout), "hw")
        rb["keepclean_time"] = Readback(float(cam.get_keepclean_time()), "hw")
        gain, advanced = cam.get_EMCCD_gain()
        rb["gain"] = Readback(int(gain), "hw")
        rb["em_advanced"] = Readback(int(advanced), "hw")
        clamp = cam.get_baseline_clamp()
        rb["baseline_clamp"] = (Readback(int(clamp), "hw") if clamp is not None
                                else Readback(None, "unsupported"))
        rb["trigger_mode"] = Readback(cam.get_trigger_mode(), "driver_cache")
        ft = cam.is_frame_transfer_enabled()
        rb["frame_transfer"] = (Readback(int(bool(ft)), "driver_cache") if cmd.get("_ft_sent", True)
                                else Readback(0, "unsupported"))
        rb["sensor_roi"] = Readback(tuple(int(v) for v in cam.get_image_mode_parameters()),
                                    "driver_cache")
        rb["output_amp"] = Readback(cam.get_oamp(), "driver_cache")
        rb["hs_speed"] = Readback(cam.get_hsspeed(), "driver_cache")
        rb["preamp"] = Readback(cam.get_preamp(), "driver_cache")
        rb["vs_speed"] = Readback(cam.get_vsspeed(), "driver_cache")
        rb["read_mode"] = Readback(cam.get_read_mode(), "driver_cache")
        rb["acquisition_mode"] = Readback(cam.get_acquisition_mode(), "driver_cache")
        level, invert, high_z = cam.get_ext_trigger_parameters()
        rb["ext_trigger_edge"] = (Readback("falling" if invert else "rising", "driver_cache")
                                  if invert is not None else Readback(None, "unsupported"))
        rb["ext_trigger_termination"] = (Readback("high_z" if high_z else "50_ohm", "driver_cache")
                                         if high_z is not None else Readback(None, "unsupported"))
        for key in ("vs_amp", "fast_ext_trigger", "em_gain_mode", "camera_link", "cooler_mode"):
            rb[key] = Readback(cmd.get(key), "commanded")
        rb["shutter"] = cam.shutter_readback()
        rb["temperature"] = Readback(float(cam.get_temperature()), "hw")
        rb["cooler_status"] = Readback(cam.get_temperature_status(), "hw")
        rb["cooler"] = Readback(bool(cam.is_cooler_on()), "hw")
        rb["temperature_setpoint"] = Readback(cam.get_temperature_setpoint(), "driver_cache")
        rb["fan"] = Readback(cam.get_fan_mode(), "driver_cache")
        return rb

    @staticmethod
    def _mismatches(p, rb) -> dict:
        want = {
            "trigger_mode": p["trigger_mode"], "frame_transfer": p["frame_transfer"],
            "sensor_roi": p["sensor_roi"], "hs_speed": p["hs_speed"], "preamp": p["preamp"],
            "vs_speed": p["vs_speed"], "gain": p["gain"], "baseline_clamp": p["baseline_clamp"],
            "acquisition_mode": FIXED["acquisition_mode"], "output_amp": FIXED["output_amp"],
            "read_mode": FIXED["read_mode"], "em_advanced": FIXED["em_advanced"],
            "ext_trigger_edge": FIXED["ext_trigger_edge"],
            "ext_trigger_termination": FIXED["ext_trigger_termination"],
        }
        out = {}
        for key, value in want.items():
            got = rb[key]
            if got.source == "unsupported":
                continue
            if got.value != value:
                out[key] = (value, got.value)
        req, got = p["exposure_time"], rb["exposure_time"].value
        if req > 0 and not (req * (1 - FLOAT32_REL) <= got <= req * (1 + EXPOSURE_TOLERANCE) * (1 + FLOAT32_REL)):
            out["exposure_time"] = (req, got)
        return out

    # -- acquisition -----------------------------------------------------------------
    def start_acquisition(self, mode: str, n_frames: int | None = None) -> None:
        cam = self._require_open()
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} is not one of {MODES}")
        need = "run" if mode == "run" else "live"
        if self._applied_purpose != need:
            raise CameraError(f"start_acquisition({mode!r}) refused: the last successful apply "
                              f"was {self._applied_purpose!r}, not {need!r}; apply a {need} "
                              f"profile first")
        trigger = cam.get_trigger_mode()
        allowed = AndorParams.RUN_TRIGGER_MODES if mode == "run" else AndorParams.LIVE_TRIGGER_MODES
        if trigger not in allowed:
            raise CameraError(f"start_acquisition({mode!r}) refused: trigger_mode is "
                              f"{trigger!r}, {mode} needs one of {allowed}")
        if mode == "run":
            if n_frames is None or int(n_frames) < 1:
                raise ValueError(f"a run acquisition needs n_frames >= 1, got {n_frames!r}")
            n_frames = int(n_frames)
        cam.start_acquisition(mode="cont")
        self._mode = mode
        self._n_frames = n_frames
        self._next_idx = 0
        self._live_lost = 0
        if mode == "snap" and trigger == "software":
            cam.send_software_trigger()

    def stop_acquisition(self) -> None:
        self._require_open().stop_acquisition()

    def acquisition_state(self) -> dict:
        cam = self._require_open()
        acquiring = bool(cam.acquisition_in_progress())
        frames_done = int(cam.get_acquisition_progress().frames_done) if self._mode else 0
        state = {"acquiring": acquiring, "frames_done": frames_done, "cycle_s": self._cycle_s}
        if self._mode == "live":
            state["frames_lost"] = self._live_lost
        return state

    def _stop_run_if_complete(self, cam):
        # A run stops by itself after n_frames: nothing beyond them is acquired.
        if self._mode == "run" and cam.acquisition_in_progress():
            if cam.get_acquisition_progress().frames_done >= self._n_frames:
                cam.stop_acquisition()

    def _read_new(self, cam) -> list:
        # Every frame from the next expected hardware index on; the ones the
        # ring buffer overwrote come back as image=None under their own index
        # (see AndorEMCCD.read_frames_from).
        frames, lost, first = cam.read_frames_from(self._next_idx)
        t = time.time()
        out = []
        if lost:
            if self._mode == "live":
                self._live_lost += len(lost)        # live frames are conflated anyway
            else:
                out.extend(RawFrame(image=None, hw_idx=int(i), hw_ts=None, t_host=t)
                           for i in lost)
        for k, img in enumerate(frames):
            img = np.array(img, copy=True)          # own memory, not an SDK buffer view
            img.flags.writeable = False
            out.append(RawFrame(image=img, hw_idx=int(first + k), hw_ts=None, t_host=t))
        self._next_idx = first + len(frames)
        return out

    def retrieve(self, timeout_s: float) -> list:
        cam = self._require_open()
        self._stop_run_if_complete(cam)
        new = self._read_new(cam)
        if not new and timeout_s > 0 and cam.acquisition_in_progress():
            try:
                cam.wait_for_frame(since="lastread", nframes=1, timeout=timeout_s)
            except cam.TimeoutError:
                return []
            self._stop_run_if_complete(cam)
            new = self._read_new(cam)
        if self._mode == "snap" and new:
            cam.stop_acquisition()
            first = [f for f in new if f.hw_idx == 0]
            return first[:1] if first else new[:1]
        return new

    # -- telemetry ------------------------------------------------------------------
    def status(self) -> dict:
        cam = self._cam
        out = dict(self._last_status)
        if cam is None:
            out.update(open=False, stale=())
            return out
        stale = []
        for key, getter in (("temperature", cam.get_temperature),
                            ("cooler_status", cam.get_temperature_status),
                            ("cooler", cam.is_cooler_on)):
            try:
                value = getter()
                out[key] = self._last_status[key] = (bool(value) if key == "cooler" else value)
            except Exception:
                # e.g. DRV_ACQUIRING on a camera that cannot read its
                # temperature while acquiring: keep the last known value
                out.setdefault(key, None)
                stale.append(key)
        try:
            out["acquiring"] = bool(cam.acquisition_in_progress())
        except Exception:
            stale.append("acquiring")
        out["open"] = True
        out["stale"] = tuple(stale)
        return out
