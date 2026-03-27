from artiq.experiment import *
from artiq.language import TBool, delay

from waxa.config.expt_params import ExptParams
from waxa.data import DataSaver, RunInfo, counter
from waxa.dummy.camera_params import CameraParams
from waxx.config.data_vault import DataVault
from waxx.control.misc.oscilloscopes import ScopeData
from waxx.util.live_od.camera_client import CameraClient

from waxx.base.scribe import Scribe
from waxx.base.scanner import Scanner
from waxa.base.dealer import Dealer

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

        self.xvarnames = []
        self.sort_idx = []
        self.sort_N = []

        self._setup_awg = False

        self.data = DataVault(expt=self)
        self.ds = DataSaver()
    
    @kernel
    def init_kernel_wax(self, notify_server=True):
        if notify_server:
            self.send_new_run()

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

    @kernel
    def cleanup_scan_kernel_wax(self):
        self.data.put_shot_data()

    def compute_new_derived(self):
        pass
    
    def end_wax(self, expt_filepath):

        self.scope_data.close()

        if self.run_info.save_data:
            self.cleanup_scanned()
            self._send_write_data_to_server(expt_filepath)

        if hasattr(self,'monitor'):
            self.monitor.update_device_states()
            self.monitor.signal_end()

        try:
            self.live_od_client.send_run_complete()
        except Exception as e:
            print(e)
                
        # server_talk.play_random_sound()

