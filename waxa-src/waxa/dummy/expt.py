import numpy as np
from waxa.dummy.run_info import RunInfo

class Expt():
    def __init__(self):
        self.xvardims = []
        self.xvarnames = []
        self.scan_xvars = []
        self.N_xvars = 1
        self.setup_camera = True
        self.save_data = True
        self.run_info = RunInfo()
        self.images = np.array([])
        self.image_timestamps = np.array([])
        self.params = []
        self.sort_idx = []
        self.sort_N = []
        self.data = []
        self.scope_data = []

    def _unshuffle_struct(self,struct,only_treat_first_Nvar_axes=False,reshuffle=False): pass
    def _unshuffle_ndarray(self,var,exclude_dims=0,reshuffle=False): pass
    def unscramble_images(self,reshuffle=False): pass
    def _unscramble_timestamps(self,reshuffle=False): pass