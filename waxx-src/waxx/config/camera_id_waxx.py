from waxx.control.cameras.camera_param_classes import CameraParams, BaslerParams, AndorParams, img_types

class camera_frame():
    def __init__(self):
        
        self.img_types = img_types
        self.write_keys()
    
    def write_keys(self):
        """Adds the assigned keys to the CameraParams objects so that the
        user-defined names (key) are available with the CameraParams
        objects."""
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key],CameraParams):
                self.__dict__[key].key = key
        
cameras = camera_frame()