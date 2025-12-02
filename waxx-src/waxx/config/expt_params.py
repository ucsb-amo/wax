import numpy as np

class ExptParams():
    def __init__(self):
        self.beatlock_sign = -1
        self.N_offset_lock_reference_multiplier = 8
        self.frequency_minimum_offset_beatlock = 250.e6

    def compute_derived(self):
        '''loop through methods (except built in ones) and compute all derived quantities'''
        methods = [m for m in dir(self) if not m.startswith('__') and callable(getattr(self,m)) and not m == 'compute_derived']
        for m in methods:
            getattr(self,m)()
        