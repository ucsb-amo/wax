import numpy as np
from pathlib import Path
import os

from artiq.experiment import *
from artiq.experiment import delay, delay_mu

from waxa.config.expt_params import ExptParams
from waxa.data import DataSaver, RunInfo, counter, server_talk
from waxa.base.dealer import Dealer
from waxa.base.scribe import Scribe
from waxa.dummy.camera_params import CameraParams
from waxa import img_types

from artiq.language.core import kernel_from_string, now_mu

from waxx.config.data_vault import DataVault
from waxx.base.scanner import Scanner
from waxx.control.misc.oscilloscopes import ScopeData
from waxx.util.artiq.async_print import aprint

from waxx.util.live_od.camera_client import CameraClient

RPC_DELAY = 10.e-3

class Expt(Dealer, Scanner, Scribe):
    def __init__(self,
                 setup_camera=True,
                 save_data=True,
                 absorption_image=None,
                 server_talk=None):
        
        if absorption_image != None:
            print("Warning: The argument 'absorption_image' is depreciated -- change it out for 'imaging_type'")
            print("Defaulting to absorption imaging.")

        Scanner.__init__(self)
        super().__init__()

        self.setup_camera = setup_camera
        self.run_info = RunInfo(self,save_data,server_talk=server_talk)
        self.scope_data = ScopeData()
        self._ridstr = " Run ID: "+ str(self.run_info.run_id)
        self._counter = counter()

        self.camera_params = CameraParams()

        self.live_od_client = CameraClient(None, None)

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

    def send_new_run(self):
        self.live_od_client.connect()
        ready = self.live_od_client.send_new_run(camera_params=self.camera_params,
                                            data_filepath=self.run_info.filepath,
                                            save_data = self.run_info.save_data,
                                            N_img = self.params.N_img,
                                            N_shots = self.params.N_shots,
                                            N_pwa_per_shot=self.p.N_pwa_per_shot,
                                            imaging_type=self.run_info.imaging_type,
                                            run_id=self.run_info.run_id)

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

        self.init_xvars(shuffle,N_repeats)

        self.data.init()

        if self.setup_camera:
            self.data_filepath = self.ds.create_data_file(self)
    
    def compute_new_derived(self):
        pass
    
    def end_wax(self, expt_filepath):

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

        self.live_od_client.send_run_complete()
                
        # server_talk.play_random_sound()