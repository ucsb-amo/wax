from waxx.control.misc.sdg6000x import SDG6000X_CH, SDG6000X_FM_CH
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

    def assign_sdg6000x_fm_ch(self, ch, ip,
                              frequency_carrier,
                              max_amplitude_vpp,
                              fm_deviation=0.,
                              fm_rate=0.,
                              max_fm_deviation=10.e6,
                              max_fm_rate=1.e6) -> SDG6000X_FM_CH:
        return SDG6000X_FM_CH(ch=ch, ip=ip,
                              frequency_carrier=frequency_carrier,
                              max_amplitude_vpp=max_amplitude_vpp,
                              fm_deviation=fm_deviation,
                              fm_rate=fm_rate,
                              max_fm_deviation=max_fm_deviation,
                              max_fm_rate=max_fm_rate,
                              core=self.core)