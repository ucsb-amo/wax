from artiq.experiment import *
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