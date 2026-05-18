from artiq.experiment import *
import numpy as np
class DummyCore():
    @kernel
    def break_realtime(self):
        pass

    @kernel
    def wait_until_mu(self,t):
        pass

    @kernel
    def reset(self):
        pass

    @kernel
    def get_rtio_counter_mu(self):
        pass

    @portable
    def seconds_to_mu(self, seconds):
        return np.int64(0)
    
    @portable
    def mu_to_seconds(self, mu):
        return 0.
    