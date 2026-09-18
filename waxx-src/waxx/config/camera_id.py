from waxx.control.cameras.camera_param_classes import CameraParams, BaslerParams, AndorParams, img_types

class camera_frame():
    def __init__(self):
        
        self.setup()

        self.cleanup()

    def setup(self):
        self.img_types = img_types

    def cleanup(self):
        self._write_keys()
    
    def _write_keys(self):
        """Adds the assigned keys to the CameraParams objects so that the
        user-defined names (key) are available with the CameraParams
        objects.

        Also defaults optical_path_key to key, for every entry that did not
        name a port of its own.  Only detectors that share another entry's
        optical path (the APD, which sits behind a pickoff on the Andor port)
        set it themselves."""
        for key in self.__dict__.keys():
            params = self.__dict__[key]
            if isinstance(params,CameraParams):
                params.key = key
                if not params.optical_path_key:
                    params.optical_path_key = key
        
cameras = camera_frame()