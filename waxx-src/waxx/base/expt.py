import numpy as np
from pathlib import Path
import os

from artiq.experiment import *
from artiq.experiment import delay, delay_mu

from waxa import ExptParams
from waxa.data import DataSaver, RunInfo, counter, server_talk, DataVault
from waxa.base import Dealer, Scribe
from waxa.dummy.camera_params import CameraParams
from waxa import img_types

from artiq.language.core import kernel_from_string, now_mu

from waxx.base.scanner import Scanner
from waxx.control.misc.oscilloscopes import ScopeData
from waxx.util.artiq.async_print import aprint

RPC_DELAY = 10.e-3

class Expt(Dealer, Scanner, Scribe):
    def __init__(self,
                 setup_camera=True,
                 save_data=True,
                 absorption_image=None):
        
        if absorption_image != None:
            print("Warning: The argument 'absorption_image' is depreciated -- change it out for 'imaging_type'")
            print("Defaulting to absorption imaging.")

        Scanner.__init__(self)
        super().__init__()

        self.setup_camera = setup_camera
        self.run_info = RunInfo(self,save_data)
        self.scope_data = ScopeData()
        self._ridstr = " Run ID: "+ str(self.run_info.run_id)
        self._counter = counter()

        self.camera_params = CameraParams()

        self.params = ExptParams()
        self.p = self.params

        self.images = []
        self.image_timestamps = []

        self.xvarnames = []
        self.sort_idx = []
        self.sort_N = []

        self._setup_awg = False

        self.data = DataVault(expt=self)
        self.ds = DataSaver()

    def finish_prepare_wax(self,N_repeats=[],shuffle=True):
        """
        To be called at the end of prepare. 
        
        Automatically adds repeats either if specified in N_repeats argument or
        if previously specified in self.params.N_repeats. 
        
        Shuffles xvars if specified (defaults to True). Computes the number of
        images to be taken from the imaging method and the length of the xvar
        arrays.

        Computes derived parameters within ExptParams.

        Accepts an additional compute_derived method that is user defined in the
        experiment file. This is to allow for recomputation of derived
        parameters that the user created in the experiment file at each step in
        a scan. This must be an RPC -- no kernel decorator.
        """

        if hasattr(self,'monitor'):
            self.monitor.init_monitor()

        if self.run_info.imaging_type == img_types.ABSORPTION:
            if self.params.N_pwa_per_shot > 1:
                print("You indicated more than one PWA per shot, but the analysis is set to absorption imaging. Setting # PWA to 1.")
            self.params.N_pwa_per_shot = 1

        if not self.xvarnames:
            self.xvar("dummy",[0])
        if self.xvarnames and not self.scan_xvars:
            for key in self.xvarnames:
                self.xvar(key,vars(self.params)[key])
        self.plug_in_xvars()

        self.repeat_xvars(N_repeats=N_repeats)
        
        if shuffle:
            self.shuffle_xvars()
        
        self.params.N_img = self.get_N_img()
        self.prepare_image_array()

        self.params.compute_derived()
        self.compute_new_derived()

        self.xvardims = [len(xvar.values) for xvar in self.scan_xvars]
        self.scope_data.xvardims = self.xvardims

        self.data.write_keys()
        self.data.set_container_sizes()

        if self.setup_camera:
            self.data_filepath = self.ds.create_data_file(self)

        self.generate_assignment_kernels()
    
    def compute_new_derived(self):
        pass

    def prepare_image_array(self):
        if self.run_info.save_data:
            # print(self.camera_params.camera_type)
            if self.camera_params.camera_type == 'andor':
                dtype = np.uint16
            elif self.camera_params.camera_type == 'basler':
                dtype = np.uint8
            else:
                dtype = np.uint8
            self.images = np.zeros((self.params.N_img,)+self.camera_params.resolution,dtype=dtype)
            self.image_timestamps = np.zeros((self.params.N_img,))
        else:
            self.images = np.array([0])
            self.image_timestamps = np.array([0])
        
    def get_N_img(self):
        """
        Computes the number of images to be taken during the sequence from the
        length of the specified xvars, stores in self.params.N_img. For
        absorption imaging, 3 images per shot. For fluorescence imaging,
        variable pwa images (ExptParams.N_pwa_per_shot, default = 1), then 1
        each pwoa and dark images.
        """                
        N_img = 1
        msg = ""

        for xvar in self.scan_xvars:
            N_img = N_img * xvar.values.shape[0]
            msg += f" {xvar.values.shape[0]} values of {xvar.key}."
        self.params.N_shots_with_repeats = N_img

        msg += f" {N_img} total shots."

        ### I have no idea what this is for. ###
        if isinstance(self.params.N_repeats,list):
            if len(self.params.N_repeats) == 1:
                N_repeats = self.params.N_repeats[0]
            else:
                N_repeats = np.prod(self.params.N_repeats)
        else:
            N_repeats = 1
        self.params.N_shots = int(N_img / N_repeats)
        ###

        if self.run_info.imaging_type == img_types.ABSORPTION:
            images_per_shot = 3
        else:
            images_per_shot = self.params.N_pwa_per_shot + 2

        N_img = images_per_shot * N_img # 3 images per value of independent variable (xvar)

        msg += f" {N_img} total images expected."
        print(msg)
        return N_img
    
    def end(self, expt_filepath):

        self.scope_data.close()

        if self.setup_camera:
            if self.run_info.save_data:
                self.cleanup_scanned()
                self.write_data(expt_filepath)
            else:
                self.remove_incomplete_data()

        if hasattr(self,'monitor'):
            self.monitor.update_device_states()
            self.monitor.signal_end()
                
        # server_talk.play_random_sound()