from waxa.config.img_types import img_types
from waxa.dummy.camera_params import CameraParams

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
                key = ""):
        super().__init__()
        self.key = key
        self.camera_type = "basler"
        self.serial_no = serial_number
        self.trigger_source = trigger_source
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

class AndorParams(CameraParams):
    # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
    def __init__(self,
                 exposure_time_fluor = 25.e-6, amp_fluorescence=0.54, em_gain_fluor = 10.,
                 exposure_time_abs = 10.e-6, amp_absorption=0.1, em_gain_abs = 300.,
                 exposure_time_dispersive=100.e-6, amp_dispersive = 0.106, em_gain_dispersive = 300.,
                 t_light_only_image_delay=75.e-3, t_dark_image_delay=75.e-3,
                 resolution = (512,512,),
                 magnification = 50./3,
                 key = ""):
        super().__init__()
        self.key = key
        self.camera_type = "andor"
        self.pixel_size_m = 16.e-6
        self.magnification = magnification
        self.exposure_delay = 0. # needs to be updated from docs
        self.connection_delay = 8.0
        self.t_camera_trigger = 200.e-9
        self.t_readout_time = 512 * 3.3e-6
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.hs_speed = 0
        self.vs_speed = 1
        self.vs_amp = 3
        self.preamp = 2
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