import numpy as np

from artiq.coredevice.core import Core
from artiq.language import now_mu, kernel, delay, portable

import vxi11

T_RPC_DELAY = 10.e-3

dv = -0.1

class SDG6000X(vxi11.Instrument):
    def __init__(self,ip):
        super().__init__(ip)
        self.ip = ip

    def _set_amp_command(self,ch,amp):
        self.write(f"C{ch}:BSWV AMP,{amp}")

    def _set_freq_command(self,ch,freq):
        self.write(f"C{ch}:BSWV FRQ,{freq}")

    def _sw_output(self,ch,state=0):
        if state == 1:
            s = "ON"
        else:
            s = "OFF"
        self.write(f"C{ch}:OUTP {s}")

class SDG6000X_CH():
    def __init__(self,ch,ip,
                 core=Core,
                 default_amplitude_vpp=0.,
                 max_amplitude_vpp=1.):

        self._instr = SDG6000X(ip)
        self.ch = ch

        self.frequency = 0.
        self.amplitude_vpp = default_amplitude_vpp
        self.max_amplitude_vpp = max_amplitude_vpp

        self.state = 1

        self.core = core

    @portable
    def _stash_defaults(self):
        self._frequency_default = self.frequency
        self._amplitude_vpp_default = self.amplitude_vpp

    @portable
    def _restore_defaults(self):
        self.frequency = self._frequency_default
        self.amplitude_vpp = self._amplitude_vpp_default

    @portable
    def set_output_rpc(self,state=1,init=False):
        if init:
            sw_changed = True
        else:
            sw_changed = bool(state) != (self.state == 1)

        if sw_changed:
            self.state = state if state >= 0. else self.state
            self._instr._sw_output(self.ch,self.state)

    @portable
    def fetch_state(self):

        reply = self._instr.ask(f"C{self.ch}:BSWV?")
        def parse_params(response):
            frq_start = response.find('FRQ,') + 4
            frq_end = response.find('HZ', frq_start)
            if frq_start > 3 and frq_end != -1:
                freq = float(response[frq_start:frq_end])
            
            amp_start = response.find('AMP,') + 4
            amp_end = response.find('V', amp_start)
            if amp_start > 3 and amp_end != -1:
                amp = float(response[amp_start:amp_end])
            return freq, amp
        freq, amp = parse_params(reply)

        reply = self._instr.ask(f"C{self.ch}:OUTP?")
        def parse_output_state(response):
            outp_start = response.find('OUTP ') + 5
            outp_end = response.find(',', outp_start)
            if outp_start > 4 and outp_end != -1:
                string = response[outp_start:outp_end]
                if string == 'ON':
                    state = 1
                elif string == 'OFF':
                    state = 0
                return state
            return None
        state = parse_output_state(reply)

        self.frequency = freq
        self.amplitude_vpp = amp
        self.state = state

    @portable
    def set_rpc(self,
            frequency=dv,
            amplitude=dv,
            init=False):

        if init:
            freq_changed = True
            amp_changed = True
        else:
            freq_changed = (frequency >= 0.) and (frequency != self.frequency)
            amp_changed = (amplitude >= 0.) and (amplitude != self.amplitude_vpp)
        if freq_changed:
            self.frequency = frequency if frequency!=dv else self.frequency
            self._instr._set_freq_command(self.ch,self.frequency)
        if amp_changed:
            if self.amplitude_vpp > self.max_amplitude_vpp:
                raise ValueError("Amplitdue requested for this channel is beyond configured maximum.")
            self.amplitude_vpp = amplitude if amplitude!=dv else self.amplitude_vpp
            self._instr._set_amp_command(self.ch,self.amplitude_vpp)

    @kernel
    def init(self):
        self._stash_defaults()
        self.set(init=True)
        self.set_output(init=True)
        
    @kernel
    def set(self,frequency=dv,amplitude=dv,init=False):
        self.core.wait_until_mu(now_mu())
        self.set_rpc(frequency,amplitude,init)
        delay(T_RPC_DELAY)

    @kernel
    def set_output(self,state=1,init=False):
        self.core.wait_until_mu(now_mu())
        self.set_output_rpc(state,init)
        delay(T_RPC_DELAY)