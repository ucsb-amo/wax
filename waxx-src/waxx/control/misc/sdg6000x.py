import numpy as np

from artiq.coredevice.core import Core
from artiq.language import now_mu, kernel, delay, portable

import vxi11

T_RPC_DELAY = 10.e-3

dv = -0.1

class SDG6000X_Params():
    def __init__(self,
                 frequency=0.,
                 amplitude_vpp=0.,
                 state=0,
                 max_amplitude_vpp=0.):
        self.frequency = frequency
        self.amplitude_vpp = amplitude_vpp
        self.max_amplitude_vpp = max_amplitude_vpp
        self.state = state

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
    def __init__(self,
                 ch,ip,
                 frequency,
                 amplitude_vpp,
                 default_state = 1,
                 max_amplitude_vpp=1.,
                 core=Core):

        self._instr = SDG6000X(ip)
        self.ch = ch
        
        self._p = SDG6000X_Params(frequency=frequency,
                                      amplitude_vpp=amplitude_vpp,
                                      state=default_state,
                                      max_amplitude_vpp=max_amplitude_vpp)
        
        self._frequency_default = 0.
        self._amplitude_vpp_default = 0.

        self.core = core

    @portable
    def _stash_defaults(self):
        self._frequency_default = self._p.frequency
        self._amplitude_vpp_default = self._p.amplitude_vpp

    @portable
    def _restore_defaults(self):
        self._p.frequency = self._frequency_default
        self._p.amplitude_vpp = self._amplitude_vpp_default

    @portable
    def set_output_rpc(self,state=1,init=False):
        if init:
            sw_changed = True
        else:
            sw_changed = bool(state) != (self._p.state == 1)

        if sw_changed:
            self._p.state = state if state >= 0. else self._p.state
            self._instr._sw_output(self.ch,self._p.state)

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

        self._p.frequency = freq
        self._p.amplitude_vpp = amp
        self._p.state = state

    @portable
    def set_rpc(self,
            frequency=dv,
            amplitude=dv,
            init=False):

        if init:
            freq_changed = True
            amp_changed = True
        else:
            freq_changed = (frequency >= 0.) and (frequency != self._p.frequency)
            amp_changed = (amplitude >= 0.) and (amplitude != self._p.amplitude_vpp)
        if freq_changed:
            self._p.frequency = frequency if frequency!=dv else self._p.frequency
            self._instr._set_freq_command(self.ch,self._p.frequency)
        if amp_changed:
            if self._p.amplitude_vpp > self._p.max_amplitude_vpp:
                raise ValueError("Amplitdue requested for this channel is beyond configured maximum.")
            self._p.amplitude_vpp = amplitude if amplitude!=dv else self._p.amplitude_vpp
            self._instr._set_amp_command(self.ch,self._p.amplitude_vpp)

    @kernel
    def init(self):
        self.core.wait_until_mu(now_mu())
        self._stash_defaults()
        self.set_rpc(init=True)
        self.set_output_rpc(init=True)
        delay(10*T_RPC_DELAY)
        
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