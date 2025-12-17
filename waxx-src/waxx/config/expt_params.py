import numpy as np
from waxa.config.expt_params import ExptParams as ExptParamsWaxa

class ExptParams(ExptParamsWaxa):
    def __init__(self):
        super().__init__()
        
        self.beatlock_sign = -1
        self.N_offset_lock_reference_multiplier = 8
        self.frequency_minimum_offset_beatlock = 250.e6
        