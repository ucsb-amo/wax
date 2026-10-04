import numpy as np

from waxa.config.img_types import img_types
from waxa.dummy.camera_params import CameraParams


class RunFieldRefused(ValueError):
    """A run-owned Andor field (trigger, frame transfer, sensor_roi) holds a
    value that is not accepted.  ``field``, ``value`` and ``reason`` say which,
    what and why; str() reads as one sentence naming all three."""
    def __init__(self, field, value, reason):
        super().__init__(f"AndorParams.{field} = {value!r} is refused: {reason}")
        self.field = field
        self.value = value
        self.reason = reason


def _as_str(field, value):
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str):
        raise RunFieldRefused(field, value, "must be a string")
    return str(value)


def _as_int(field, value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    raise RunFieldRefused(field, value, "must be an integer (and never None)")


def _as_roi(value):
    if isinstance(value, (bytes, str)) or value is None:
        raise RunFieldRefused("sensor_roi", value,
                              "must be 6 integers (hstart, hend, vstart, vend, hbin, vbin)")
    try:
        items = list(np.asarray(value).ravel()) if isinstance(value, np.ndarray) else list(value)
    except TypeError:
        raise RunFieldRefused("sensor_roi", value,
                              "must be 6 integers (hstart, hend, vstart, vend, hbin, vbin)") from None
    if len(items) != 6:
        raise RunFieldRefused("sensor_roi", value,
                              "must be 6 integers (hstart, hend, vstart, vend, hbin, vbin)")
    return tuple(_as_int("sensor_roi", v) for v in items)


def _set_trigger_ttl(params, trigger_ttl):
    """Record which TTL line triggers this camera.

    ``trigger_ttl`` is a ttl_frame attribute name (str) or the TTL object
    itself (anything with ``.ch``; its ``.key`` is the attribute name the
    frame gave it). What is kept, and saved with every run as
    ``camera_params/``: ``trigger_ttl`` (the name) and ``trigger_ttl_ch``
    (the channel, -1 when only a name was given). The object goes under
    ``_trigger_ttl`` -- the underscore keeps it out of the run file -- for
    reference only: a camera table is module-level, so its TTL object is
    never the experiment's bound device; the experiment resolves the name /
    channel onto its own ttl frame (kexp.config.camera_id.camera_trigger_ttl).
    """
    if trigger_ttl is None or isinstance(trigger_ttl, str):
        params.trigger_ttl = str(trigger_ttl or "")
        params.trigger_ttl_ch = -1
        params._trigger_ttl = None
        return
    name = getattr(trigger_ttl, "key", "") or getattr(trigger_ttl, "name", "")
    ch = getattr(trigger_ttl, "ch", None)
    if ch is None:
        raise TypeError(f"trigger_ttl must be a ttl_frame attribute name or a "
                        f"TTL object (with .ch), got {trigger_ttl!r}")
    params.trigger_ttl = str(name)
    params.trigger_ttl_ch = int(ch)
    params._trigger_ttl = trigger_ttl


def full_frame_roi(detector_shape):
    """The sensor_roi of the whole sensor at bin 1; detector_shape is (rows, cols)."""
    rows, cols = detector_shape
    return (0, int(cols), 0, int(rows), 1, 1)


def roi_resolution(sensor_roi):
    """(rows, cols) of the image a sensor_roi reads out."""
    hstart, hend, vstart, vend, hbin, vbin = sensor_roi
    return ((vend - vstart) // vbin, (hend - hstart) // hbin)


def check_andor_run_fields(trigger_mode, frame_transfer, sensor_roi,
                           purpose="run", detector_shape=None):
    """Normalise and check the run-owned Andor fields.

    Returns (trigger_mode: str, frame_transfer: int, sensor_roi: tuple of 6 int).
    Raises RunFieldRefused (a ValueError) naming the field, the value and the
    rule.  ``purpose`` "run" accepts only an external trigger; "live" only
    internal or software.  Frame transfer is refused for both, and so is any
    sensor_roi other than the full frame at bin 1 (this build).
    """
    shape = AndorParams.DETECTOR_SHAPE if detector_shape is None else detector_shape
    trigger_mode = _as_str("trigger_mode", trigger_mode)
    frame_transfer = _as_int("frame_transfer", frame_transfer)
    sensor_roi = _as_roi(sensor_roi)

    if purpose == "run":
        if trigger_mode not in AndorParams.RUN_TRIGGER_MODES:
            raise RunFieldRefused("trigger_mode", trigger_mode,
                f"runs accept only {AndorParams.RUN_TRIGGER_MODES} (one frame per TTL edge); "
                f"ext_start free-runs after the first edge, ext_exp exposes while the TTL "
                f"is high, and the other modes are unsupported")
    elif purpose == "live":
        if trigger_mode not in AndorParams.LIVE_TRIGGER_MODES:
            raise RunFieldRefused("trigger_mode", trigger_mode,
                f"live streaming accepts only {AndorParams.LIVE_TRIGGER_MODES}")
    else:
        raise ValueError(f"purpose {purpose!r} is not 'run' or 'live'")

    if frame_transfer not in AndorParams.RUN_FRAME_TRANSFER:
        raise RunFieldRefused("frame_transfer", frame_transfer,
            "frame transfer is never used: with an external trigger the exposure "
            "becomes the time between triggers, so the recorded exposure_time "
            "would be false (Andor SDK2 manual p.55)")

    full = full_frame_roi(shape)
    hbin, vbin = sensor_roi[4], sensor_roi[5]
    if (hbin, vbin) != (1, 1):
        raise RunFieldRefused("sensor_roi", sensor_roi,
            f"binning {hbin}x{vbin} is not accepted, only bin 1 "
            f"(binning scales the atom number by the bin factor); "
            f"the only accepted value is {full}")
    if sensor_roi != full and not AndorParams.ALLOW_SENSOR_CROP:
        raise RunFieldRefused("sensor_roi", sensor_roi,
            f"only the full frame {full} is accepted in this build "
            f"(a crop moves every analysis ROI)")
    return trigger_mode, frame_transfer, sensor_roi


class BaslerParams(CameraParams):
    # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
    def __init__(self,serial_number='40320384',
                trigger_source='Line1',
                exposure_time_fluor = 1.e-3, amp_fluorescence=0.5, gain_fluor = 0.,
                exposure_time_abs = 19.e-6, amp_absorption = 0.248, gain_abs = 0.,
                exposure_time_dispersive = 100.e-6, amp_dispersive = 0.248, gain_dispersive = 0.,
                t_light_only_image_delay=25.e-3, t_dark_image_delay=20.e-3,
                resolution = (1200,1920,),
                magnification = 0.75,
                trigger_ttl = "",
                key = ""):
        super().__init__()
        self.key = key
        self.camera_type = "basler"
        self.serial_no = serial_number
        self.trigger_source = trigger_source
        # The TTL line wired to this camera's trigger input: a ttl_frame
        # attribute name or the TTL object (see _set_trigger_ttl).
        _set_trigger_ttl(self, trigger_ttl)
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.resolution = resolution
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.pixel_size_m = 3.45 * 1.e-6
        self.magnification = magnification
        self.exposure_delay = 17 * 1.e-6
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.__gain_fluor = gain_fluor
        self.__gain_abs = gain_abs
        self.__gain_dispersive = gain_dispersive
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.__exposure_time_fluor__ = exposure_time_fluor
        self.__exposure_time_abs__ = exposure_time_abs
        self.__exposure_time_dispersive__ = exposure_time_dispersive
        self.__amp_absorption__ = amp_absorption
        self.__amp_fluorescence__ = amp_fluorescence
        self.__amp_dispersive__ = amp_dispersive
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.t_light_only_image_delay = t_light_only_image_delay
        self.t_dark_image_delay = t_dark_image_delay

    def select_imaging_type(self,imaging_type):
        if imaging_type == img_types.ABSORPTION:
            self.amp_imaging = self.__amp_absorption__
            self.exposure_time = self.__exposure_time_abs__
            self.gain = self.__gain_abs
        elif imaging_type == img_types.FLUORESCENCE:
            self.amp_imaging = self.__amp_fluorescence__
            self.exposure_time = self.__exposure_time_fluor__
            self.gain = self.__gain_fluor
        elif imaging_type == img_types.DISPERSIVE:
            self.amp_imaging = self.__amp_dispersive__
            self.exposure_time = self.__exposure_time_dispersive__
            self.gain = self.__gain_dispersive

    def prepare_for_run(self):
        """Nothing to check for a Basler (see AndorParams.prepare_for_run)."""
        pass

class AndorParams(CameraParams):
    # Class constants, not instance attributes: the INIT_RUN payload (and so
    # camera_params/ in the file) is built from vars(camera_params), so these
    # are never recorded as if they were settings of the run.
    DETECTOR_SHAPE = (512, 512)            # (rows, cols), DU897
    RUN_TRIGGER_MODES = ("ext",)
    LIVE_TRIGGER_MODES = ("int", "software")
    RUN_FRAME_TRANSFER = (0,)
    ALLOW_SENSOR_CROP = False

    # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
    def __init__(self,
                 exposure_time_fluor = 25.e-6, amp_fluorescence=0.54, em_gain_fluor = 10.,
                 exposure_time_abs = 10.e-6, amp_absorption=0.1, em_gain_abs = 300.,
                 exposure_time_dispersive=100.e-6, amp_dispersive = 0.106, em_gain_dispersive = 300.,
                 t_light_only_image_delay=75.e-3, t_dark_image_delay=75.e-3,
                 hs_speed=0, vs_speed=1, vs_amp=3, preamp=2, baseline_clamp=1,
                 trigger_mode="ext", frame_transfer=0, sensor_roi=(0, 512, 0, 512, 1, 1),
                 resolution = (512,512,),
                 magnification = 50./3,
                 trigger_ttl = "",
                 key = ""):
        super().__init__()
        self.key = key
        self.camera_type = "andor"
        # The TTL line wired to this camera's trigger input: a ttl_frame
        # attribute name or the TTL object (see _set_trigger_ttl).
        _set_trigger_ttl(self, trigger_ttl)
        self.pixel_size_m = 16.e-6
        self.magnification = magnification
        self.exposure_delay = 0. # needs to be updated from docs
        self.connection_delay = 8.0
        self.t_camera_trigger = 200.e-9
        # Full-frame (512x512) readout at the fastest EM horizontal clock
        # (17 MHz), as reported by GetReadOutTime on the DU897_EXF, 2026-09-23.
        # Keep-clean adds ~3.9 ms before the camera accepts the next trigger.
        # (Was 512 * 3.3e-6 = 1.7 ms, which is the vertical-transfer time at
        # the slowest shift speed, not the readout time.)
        self.t_readout_time = 18.1e-3
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        # Readout clock indices, see the Andor SDK2 manual (SetHSSpeed,
        # SetVSSpeed, SetVSAmplitude, SetPreAmpGain, SetBaselineClamp). Applied
        # when liveOD opens the camera and reapplied per run by
        # CameraNanny.update_params (since 2026-09-24).
        self.hs_speed = hs_speed
        self.vs_speed = vs_speed
        self.vs_amp = vs_amp
        self.preamp = preamp
        self.baseline_clamp = baseline_clamp   # 1 on, 0 off (SDK SetBaselineClamp)
        # Run-owned fields (2026-09-26), recorded with every run and applied by
        # liveOD at run start.  Runs accept only these values for now:
        # external trigger, no frame transfer, full frame at bin 1.  sensor_roi
        # is (hstart, hend, vstart, vend, hbin, vbin), 0-based, end-exclusive,
        # in unbinned sensor pixels (pylablib's setup_image_mode convention).
        # Never None: h5py cannot store None and the value would silently be
        # missing from the file.  prepare_for_run() checks them.
        self.trigger_mode = trigger_mode
        self.frame_transfer = frame_transfer
        self.sensor_roi = sensor_roi
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.__em_gain_fluor = em_gain_fluor
        self.__em_gain_abs = em_gain_abs
        self.__em_gain_dispersive = em_gain_dispersive
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.resolution = resolution
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.__exposure_time_fluor__ = exposure_time_fluor
        self.__exposure_time_abs__ = exposure_time_abs
        self.__amp_absorption__ = amp_absorption
        self.__amp_fluorescence__ = amp_fluorescence
        self.__amp_dispersive__ = amp_dispersive
        self.__exposure_time_dispersive__ = exposure_time_dispersive
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.t_light_only_image_delay = t_light_only_image_delay
        self.t_dark_image_delay = t_dark_image_delay

    def select_imaging_type(self,imaging_type):
        if imaging_type == img_types.ABSORPTION:
            self.amp_imaging = self.__amp_absorption__
            self.exposure_time = self.__exposure_time_abs__
            self.gain = self.__em_gain_abs
        elif imaging_type == img_types.FLUORESCENCE:
            self.amp_imaging = self.__amp_fluorescence__
            self.exposure_time = self.__exposure_time_fluor__
            self.gain = self.__em_gain_fluor
        elif imaging_type == img_types.DISPERSIVE:
            self.amp_imaging = self.__amp_dispersive__
            self.exposure_time = self.__exposure_time_dispersive__
            self.gain = self.__em_gain_dispersive

    def prepare_for_run(self):
        """Normalise and check the run-owned fields, then derive resolution.

        Called first in Scanner.prepare_image_array, which runs inside
        finish_prepare before INIT_RUN, so a refused value raises here without
        reserving a run id.  Converts bytes to str and numpy scalars/arrays to
        int/tuple (values unpacked from a file come back as numpy); raises
        RunFieldRefused (a ValueError) naming the rule and the value for a
        trigger other than "ext", frame_transfer 1, a crop or binning, or a
        readout index that is not an integer.
        """
        self.trigger_mode, self.frame_transfer, self.sensor_roi = check_andor_run_fields(
            self.trigger_mode, self.frame_transfer, self.sensor_roi,
            purpose="run", detector_shape=self.DETECTOR_SHAPE)
        for name in ("hs_speed", "vs_speed", "vs_amp", "preamp"):
            setattr(self, name, _as_int(name, getattr(self, name)))
        clamp = _as_int("baseline_clamp", self.baseline_clamp)
        if clamp not in (0, 1):
            raise RunFieldRefused("baseline_clamp", clamp, "must be 0 (off) or 1 (on)")
        self.baseline_clamp = clamp
        self.resolution = roi_resolution(self.sensor_roi)

class APDParams(AndorParams):
    """The APD that shares the Andor imaging port.

    Not a camera: a beamsplitter on the PDXC stage picks light off ahead of the
    Andor and sends it to an avalanche photodiode, read out through the analog
    integrator on Sampler ch 7.  It is an entry in `cameras` because selecting
    it has to configure everything a camera selection configures -- trigger
    TTL, imaging shutters, SLM phase mask, imaging amplitude and detuning --
    plus the pickoff stage, and doing that by hand through four coupled
    Base.__init__ flags was error-prone.

    Subclasses AndorParams, so it takes the same parameters and defaults as
    the Andor.  kexp.config.camera_id sets its values; they may differ from
    the Andor's, e.g. where a setting only matters to a camera sensor (EM
    gain, magnification).  What differs structurally:

      camera_type = "apd"        -> CameraNanny opens a DummyCamera rather than
                                    a second AndorEMCCD handle on the same
                                    hardware (it caches by key, dispatches by
                                    camera_type).
      optical_path_key = "andor" -> inherits the Andor's shutter and SLM
                                    routing.  Getting this wrong reads as "the
                                    APD sees nothing".
      resolution = (1, 1)        -> no pixels; keeps prepare_image_array from
                                    allocating a full 512x512 frame per image
                                    on a run that takes no images.

    camera_type = "apd" is also what Base keys on: setup_camera=True with the
    APD means acquire through it -- pickoff stage in, no liveOD frames -- and
    the experiment's own kernel does the reading (integrated_imaging_pulse).
    See kexp.base.cameras.resolve_run_config.

    Imaging type is not among the differences: the APD works with absorption or dispersive
    imaging, and which one is a property of the measurement.  Pass imaging_type
    to Base.__init__ as with any other detector.

    select_imaging_type is inherited unchanged: AndorParams.__init__ sets the
    name-mangled _AndorParams__em_gain_* attributes on this instance, so the
    inherited lookup resolves to the APD's own amplitude, exposure and gain
    for whichever imaging type is chosen.
    """
    def __init__(self, **kwargs):
        super().__init__(resolution=(1,1,), **kwargs)
        self.camera_type = "apd"
        self.optical_path_key = "andor"

    def prepare_for_run(self):
        """No-op: the APD takes no frames, nothing applies the Andor's run
        fields for it, and resolution stays (1, 1)."""
        pass