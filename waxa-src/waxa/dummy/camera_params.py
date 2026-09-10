class CameraParams():
    # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
    def __init__(self):
        self.camera_type = ""
        self.key = ""
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.pixel_size_m = 0.
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.magnification = 13
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.exposure_delay = 0.
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.exposure_time = 0.
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.connection_delay = 0.0
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.t_camera_trigger = 2.e-6
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.gain = 0.
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.amp_imaging = 0.
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.resolution = (1,1,)
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.t_light_only_image_delay = 0.
        # DO NOT ASSIGN DEFAULT PARAMETERS HERE -- INSTEAD ASSIGN THEM IN kexp.config.camera_id!
        self.t_dark_image_delay = 0.

        # Which imaging port this entry sits on; defaults to `key`, filled in
        # by camera_frame._write_keys.  The APD carries "andor" so it inherits
        # that port's shutter and SLM routing.  Read in @kernel code, so it
        # must exist as a str on EVERY instance -- including the bare
        # CameraParams placeholders built before choose_camera runs.
        self.optical_path_key = ""

        # What selecting this detector implies when the experiment says
        # nothing.  Policy, not optical tuning, so unlike everything above it
        # does belong here.  Host-side only, never in a kernel; the leading
        # underscore keeps them out of the liveOD payload and the HDF5
        # camera_params group.
        #
        # imaging_type is deliberately NOT here: absorption vs dispersive is a
        # property of the measurement, not of the detector.  Every detector,
        # the APD included, works with either.
        self._default_setup_camera = True       # does selecting this grab frames?
        self._default_apd_stage = False         # retract; clear the camera

    def select_imaging_type(self,imaging_type):
        pass