from waxx.control.misc.sdg6000x import SDG6000X_CH
from artiq.coredevice.core import Core

class siglent_frame():
    def __init__(self, core=Core):
        self.setup(core=core)

        # assignement statements here

        self.cleanup()

    def setup(self, core):
        self.core = core

    def cleanup(self):
        pass

    def assign_sdg6000x_ch(self, ch, ip,
                           frequency,
                           amplitude_vpp,
                           max_amplitude_vpp,
                           default_state=1,
                           max_frequency=500.e6,
                           min_frequency=0.) -> SDG6000X_CH:
        siglent_ch = SDG6000X_CH(ch=ch,ip=ip,
                           frequency=frequency,
                           amplitude_vpp=amplitude_vpp,
                           max_amplitude_vpp=max_amplitude_vpp,
                           default_state = default_state,
                           max_frequency=max_frequency,
                           min_frequency=min_frequency,
                           core=self.core)
        return siglent_ch