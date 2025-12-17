import numpy as np

class ExptParams():
    def __init__(self):
        self.N_repeats = 1
        self.N_pwa_per_shot = 1
        self.N_img = 1

        self._float_placeholder = 0.
        self._array_placeholder = np.array([0.])
        self._int32_placeholder = np.int32(1)
        self._int64_placeholder = np.int64(1)

    def compute_derived(self):
        '''loop through methods (except built in ones) and compute all derived quantities'''
        methods = [m for m in dir(self) if not m.startswith('__') and callable(getattr(self,m)) and not m == 'compute_derived']
        for m in methods:
            getattr(self,m)()
        