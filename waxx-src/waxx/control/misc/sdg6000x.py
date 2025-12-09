import numpy as np

from artiq.coredevice.core import Core
from artiq.language import now_mu, kernel, delay

T_RPC_DELAY = 10.e-3

dv = -0.1

class SDG6000X():
    def __init__(self,ip):
        self.ip = ip
        import vxi11
        self.instr = vxi11.Instrument(self.ip)

    def _set_amp_command(self,ch,amp):
        self.instr.write(f"C{ch}:BSVP AMP,{amp}")

    def _set_freq_command(self,ch,freq):
        self.instr.write(f"C{ch}:BSVP FREQ,{freq}")

class SDG6000X_CH():
    def __init__(self,ch,ip,
                 core=Core,
                 max_amplitude_vpp=1.):

        self._instr = SDG6000X(ip)
        self.ch = ch

        self.frequency = 0.
        self.amplitude_vpp = 0.
        self.max_amplitude_vpp = max_amplitude_vpp

        self.core = core

    @kernel
    def set(self,ch,
            frequency=dv,
            amplitude=dv):

        freq_changed = (frequency >= 0.) and (frequency != self.frequency)
        amp_changed = (amplitude >= 0.) and (amplitude != self.amplitude_vpp)

        if freq_changed:
            self.frequency = frequency if frequency!=dv else self.frequency
        if amp_changed:
            if self.amplitude_vpp > self.max_amplitude_vpp:
                raise ValueError("Amplitdue requested for this channel is beyond configured maximum.")
            self.amplitude_vpp = amplitude if amplitude!=dv else self.amplitude_vpp

        self.core.wait_until_mu(now_mu())
        if freq_changed:
            self._instr._set_freq_command(ch,self.frequency)
        if amp_changed:
            self._instr._set_amp_command(ch,self.amplitude_vpp)
        delay(T_RPC_DELAY)