import numpy as np

from waxa.config.img_types import img_types
from waxa.dummy.camera_params import CameraParams

# The Andor's run-field rules live with its driver in beacon; these names are
# kept here for everything that imports them from this module.
from beacon.drivers.andor import params as _andor
from beacon.drivers.andor.params import (RunFieldRefused, _as_str, _as_int, _as_roi,
                                         full_frame_roi, roi_resolution,
                                         check_andor_run_fields)


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
                pixel_size_m = 3.45e-6,
                exposure_delay = 17.e-6,
                exposure_delay_from_camera = True,
                pixel_format = "Mono8",
                trigger_ttl = "",
                key = ""):
        super().__init__()
        self.key = key
        self.camera_type = "basler"
        # "Mono8" (uint8 frames, 0..255) or "Mono12" / "Mono12p" (uint16, 0..4095:
        # about 16x the counts for the same light).  The run applies it (the
        # camera server's run profile) and the run's images are declared with
        # its dtype (image_dtype()).  Saved in camera_params/ of every run.
        self.pixel_format = pixel_format
        self.serial_no = serial_number
        self.trigger_source = trigger_source
        # The TTL line wired to this camera's trigger input: a ttl_frame
        # attribute name or the TTL object (see _set_trigger_ttl).
        _set_trigger_ttl(self, trigger_ttl)
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.resolution = resolution
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.pixel_size_m = pixel_size_m
        self.magnification = magnification
        # The trigger-to-exposure delay the kernel triggers ahead by
        # (Image.trigger_camera).  prepare_for_run() replaces it with the one
        # the camera reports (BslExposureStartDelay, via its camera server);
        # the configured value is used only when the camera does not report
        # one, and exposure_delay_source says which it was ("camera" or
        # "configured") in camera_params/ of every run.
        self.exposure_delay_configured = float(exposure_delay)
        self.exposure_delay_from_camera = bool(exposure_delay_from_camera)
        self.exposure_delay = float(exposure_delay)
        self.exposure_delay_source = "configured"
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

    def image_dtype(self):
        """The numpy dtype of this camera's frames in its pixel format
        (uint8 for Mono8, uint16 for Mono12 / Mono12p).  RunFieldRefused for a
        format no frame dtype is declared for."""
        from beacon.camera.schema import pixel_format_dtype
        try:
            return np.dtype(pixel_format_dtype(self.pixel_format))
        except ValueError as e:
            raise RunFieldRefused("pixel_format", self.pixel_format, str(e)) from None

    def check_pixel_format(self):
        """Check the pixel format (see AndorParams.prepare_for_run): called
        from prepare_for_run in finish_prepare before INIT_RUN, so a format
        that is refused costs no run id."""
        pf = getattr(self, "pixel_format", "Mono8")
        if isinstance(pf, (bytes, np.ndarray)):
            pf = _as_str("pixel_format", pf)
        self.pixel_format = pf
        self.image_dtype()

    def read_exposure_delay(self, directory=None):
        """``(seconds or None, why)``: the exposure start delay this camera's
        server reports (``CameraEntry.exposure_start_delay_s``), as read at
        the camera's last open / apply.  ``directory`` is a
        beacon CameraDirectory (default: the process's shared one, which the
        camera streams use too).  None, with the reason, when the camera
        cannot be found or does not report one -- never raises."""
        try:
            if directory is None:
                from waxx.control.cameras.camera_stream_client import shared_camera_directory
                directory = shared_camera_directory()
            entry = directory.find(str(self.serial_no))
        except Exception as e:
            return None, f"camera not found ({type(e).__name__}: {e})"
        value = getattr(entry, "exposure_start_delay_s", None)
        server = getattr(entry, "server_id", "?")
        if value is None:
            return None, (f"{server} does not report it (camera not opened yet, a model "
                          f"without the node, or a server older than 2026-10-20)")
        return float(value), f"reported by {server}"

    def prepare_for_run(self, directory=None):
        """Check the pixel format (check_pixel_format), then take the
        camera's own exposure start delay for this run (see
        read_exposure_delay), else the configured one with a printed warning.
        Called first in Scanner.prepare_image_array, inside finish_prepare,
        so the kernel is compiled with the value used and camera_params/
        records it with its source."""
        self.check_pixel_format()
        self.exposure_delay = self.exposure_delay_configured
        self.exposure_delay_source = "configured"
        if not self.exposure_delay_from_camera:
            return
        value, why = self.read_exposure_delay(directory)
        name = self.key or self.serial_no
        if value is None:
            print(f"[camera] !! {name}: exposure delay not read from the camera ({why}); "
                  f"using the configured {self.exposure_delay_configured * 1e6:.2f} us")
            return
        self.exposure_delay = value
        self.exposure_delay_source = "camera"
        if value != self.exposure_delay_configured:
            from waxx.util import console
            console.info(f"[camera] {name}: exposure delay {value * 1e6:.2f} us, {why} "
                         f"(configured {self.exposure_delay_configured * 1e6:.2f} us)")

class PCOParams(CameraParams):
    """A PCO sCMOS (pco.edge) served by the beacon PCO camera server
    (``beacon.pco``, category "pco_sc2").

    An auxiliary camera only: frames are taken with a camera stream
    (kexp.control.cameras.camera_stream, free-run or triggered) or as a
    diagnostic image.  It cannot be a run's acquisition camera -- liveOD has
    no backend for it -- and ``prepare_for_run`` refuses that before the run
    has an id.

    ``sensor_roi`` is (hstart, hend, vstart, vend, hbin, vbin): unbinned
    sensor pixels, 0-based, end-exclusive; the binning is done by the camera
    (it sums the binned pixels' counts), and frames are
    ((vend - vstart) / vbin, (hend - hstart) / hbin).  A stream's own ``roi``
    is a crop of those frames, in frame (binned) pixels.  Over Camera Link the
    server refuses frames wider than 640 pixels in global reset shutter mode
    (beacon ``CAMERA_LINK_MAX_OUTPUT_WIDTH``; wider ones arrive corrupted); in
    global shutter mode full-width frames are clean and the pco.edge does not
    bin (bin 1 only).

    ``readout_rate`` is "slow" or "fast" (the camera's lowest / highest pixel
    clock).  Frames are 16-bit ("Mono16", uint16); the camera has no gain
    stage (``gain`` 0).  ``trigger_source`` is its one trigger input, and
    ``trigger_edge`` ("rising" | "falling") the edge on it that starts an
    exposure.

    ``exposure_time``: the camera exposes in whole sensor lines (pco.edge 5.5
    at its slow clock: 27.5 us steps, at least one), and what it gave is in
    every frame's record.  A camera in GLOBAL RESET shutter mode (the lab's
    is, 2026-11-15) starts every row together but reads them out one after
    another, and a row collects light until it is read: with steady light the
    image hardly follows ``exposure_time`` (rows read later are brighter, by up
    to the ~30 ms readout).  It needs a light pulse inside the exposure.
    """
    PIXEL_FORMAT = "Mono16"
    TRIGGER_SOURCE = "ExposureTrigger"
    READOUT_RATES = ("slow", "fast")
    TRIGGER_EDGES = ("rising", "falling")

    # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
    def __init__(self, serial_number='',
                 exposure_time=1.e-3,
                 sensor_roi=(0, 2560, 0, 2160, 1, 1),
                 readout_rate="slow",
                 trigger_edge="rising",
                 magnification=1.,
                 pixel_size_m=6.5e-6,
                 exposure_delay=0.,
                 exposure_delay_from_camera=False,
                 trigger_ttl="",
                 key=""):
        super().__init__()
        self.key = key
        self.camera_type = "pco"
        self.serial_no = serial_number
        self.pixel_format = self.PIXEL_FORMAT
        self.trigger_source = self.TRIGGER_SOURCE
        # The TTL line wired to the camera's exposure-trigger input: a
        # ttl_frame attribute name or the TTL object (see _set_trigger_ttl).
        _set_trigger_ttl(self, trigger_ttl)
        if str(readout_rate) not in self.READOUT_RATES:
            raise RunFieldRefused("readout_rate", readout_rate,
                                  f"must be one of {self.READOUT_RATES}", owner="PCOParams")
        self.readout_rate = str(readout_rate)
        if str(trigger_edge) not in self.TRIGGER_EDGES:
            raise RunFieldRefused("trigger_edge", trigger_edge,
                                  f"must be one of {self.TRIGGER_EDGES}", owner="PCOParams")
        self.trigger_edge = str(trigger_edge)
        try:
            self.sensor_roi = _as_roi(sensor_roi)
        except RunFieldRefused as e:
            raise RunFieldRefused("sensor_roi", sensor_roi, e.reason,
                                  owner="PCOParams") from None
        if min(self.sensor_roi[4:]) < 1 or self.sensor_roi[1] <= self.sensor_roi[0] \
                or self.sensor_roi[3] <= self.sensor_roi[2]:
            raise RunFieldRefused("sensor_roi", sensor_roi, "empty, or a binning below 1",
                                  owner="PCOParams")
        self.resolution = roi_resolution(self.sensor_roi)
        self.exposure_time = float(exposure_time)
        self.gain = 0.
        self.pixel_size_m = pixel_size_m
        self.magnification = magnification
        # The lead a triggered stream gives the trigger edge.  The camera's
        # own figure is used when exposure_delay_from_camera is set and its
        # server reports one (``exposure_start_delay_s``).
        self.exposure_delay_configured = float(exposure_delay)
        self.exposure_delay_from_camera = bool(exposure_delay_from_camera)
        self.exposure_delay = float(exposure_delay)
        self.exposure_delay_source = "configured"

    def stream_settings(self):
        """The camera settings a stream sets beside exposure and gain."""
        return {"sensor_roi": tuple(self.sensor_roi), "readout_rate": self.readout_rate,
                "trigger_edge": self.trigger_edge}

    def image_dtype(self):
        from beacon.camera.schema import camera_pixel_format_dtype
        return np.dtype(camera_pixel_format_dtype(self.pixel_format))

    def prepare_for_run(self):
        raise RunFieldRefused(
            "camera_type", self.camera_type,
            f"'{self.key or self.serial_no}' cannot be a run's acquisition camera (liveOD has "
            f"no PCO backend); take its frames with a camera stream "
            f"(kexp.control.cameras.camera_stream) or as a diagnostic image",
            owner="PCOParams")


class AndorParams(CameraParams):
    # Class constants, not instance attributes: the INIT_RUN payload (and so
    # camera_params/ in the file) is built from vars(camera_params), so these
    # are never recorded as if they were settings of the run.
    # The rules themselves are beacon.drivers.andor.params (check_andor_run_fields).
    DETECTOR_SHAPE = _andor.DETECTOR_SHAPE            # (rows, cols), DU897
    RUN_TRIGGER_MODES = _andor.RUN_TRIGGER_MODES
    LIVE_TRIGGER_MODES = _andor.LIVE_TRIGGER_MODES
    RUN_FRAME_TRANSFER = _andor.RUN_FRAME_TRANSFER
    ALLOW_SENSOR_CROP = _andor.ALLOW_SENSOR_CROP

    # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
    def __init__(self,
                 serial_number='',
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
        # The serial the Andor SDK reports (GetCameraSerialNumber), as a
        # string.  It names the camera on the network (camera id
        # "andor_emccd:<serial>", the same in liveOD's camera host, the Andor
        # camera server and the SLM spot finder, so each can borrow it from
        # the other), and the backend refuses to open a camera that reports
        # another.  '' = not checked, camera id "andor_emccd:<key>".
        self.serial_no = str(serial_number or "")
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