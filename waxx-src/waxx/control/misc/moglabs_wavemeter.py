from kamo import Potassium39
from waxx.control.misc.moglabs import MOGDevice
import numpy as np
import sys
import time

from waxx.util.artiq.async_print import aprint

class WavemeterController(MOGDevice):
    def __init__(self, addr, port=None, timeout=1, check=True):
        super().__init__(addr,port,timeout,check)
        self.set_units()

    def check_ch(self) -> int:
        try:
            ch = int(self.ask('OPTSW,SET'))
            return ch
        except:
            aprint('Failed to read wavemeter channel')
            return 0

    def set_channel(self, ch):
        try:
            last_ch = self.check_ch()
            if last_ch != ch:
                self._set_channel(ch)
        except:
            aprint('Failed to set channel')
        
    def select_channel(self, ch):
        try:
            self._last_ch = int(self.ask('OPTSW,SEL'))
            if self._last_ch != ch:
                self._select_channel(ch)
        except:
            aprint('Failed to communicate with wavemeter. Check connection and IP address.')

    def _set_channel(self, ch):
        self.ask(f'OPTSW,SET,{ch}')

    def _select_channel(self, ch):
        """Sets the channel for setting measurement parameters. Does not affect
        the channel being measured."""
        self.ask(f'OPTSW,SEL,{ch}')

    def get_frequency(self, ch) -> float:
        """Get frequency in Hz. Returns 0.0 if error occurs."""
        try:
            self.set_channel(ch)
            f = float(self.ask('MEAS,FREQ').split(' ')[0]) * 1.e12
        except:
            f = 0.0
            aprint('Error getting frequency from wavemeter')
        return f
    
    def get_saturation(self, ch) -> float:
        try:
            self.set_channel(ch)
            sat = float(self.ask('MEAS,SAT'))
            return sat
        except:
            aprint(f'Failed to read wavemeter saturation level.')
            return -1.
    
    def get_exposure(self, ch) -> float:
        try:
            s = self.ask(f'OPTSW,EXP,{ch}').split(' ')
            exp = float(s[0])
            if s[1] == 'ms':
                exp *= 1.e-3
            return exp
        except:
            aprint(f'Failed to read wavemeter exposure time.')
            return -1.
        
    def set_averaging_time(self, ch, t_s=0.):
        try:
            self.select_channel(ch)
            t_ms = 1e3 * t_s
            self.ask(f'MEAS,AVERAGE,{t_ms:1.0f}')
        except:
            aprint('failed to set averaging time')

    def set_units(self):
        try:
            self.ask('MEAS,UNITS,THz')
        except:
            aprint('failed to set wavemeter units')

class WavemeterClient():
    def __init__(self,
                ch,
                target_freq,
                wavemeter_device: WavemeterController,
                locked_tolerance):
        self.ch = ch
        self.target_freq = target_freq
        self.locked_tolerance = locked_tolerance

        self._f = 0.

        self.fzw = wavemeter_device
        self.key = ""

    def check_exposure(self) -> bool:
        exp = self.fzw.get_exposure(self.ch)
        if exp < 0:
            return False
        elif exp > 0.1:
            aprint(f'Wavemeter exposure time {exp*1.e3:1.0f} ms is too long for reliable lock detection.')
            return False
        else:
            return True
        
    def check_saturation(self) -> bool:
        sat = self.fzw.get_saturation(self.ch)
        if sat < 0:
            return False
        elif sat < 10:
            aprint(f'Wavemeter saturation level ({sat:1.0f}/100) is too low for reliable lock detection.')
            return False
        else:
            return True

    def get_frequency(self) -> float:
        """Get frequency in Hz. Returns 0.0 if error occurs."""
        return self.fzw.get_frequency(self.ch)

    def lock_status(self, robust=True) -> bool:
        if robust:
            ok = self.check_exposure()
            ok = ok & self.check_exposure()
        self._f = self.get_frequency()

        if abs(self._f - self.target_freq) < self.locked_tolerance:
            return True
        else:
            aprint(f'laser {self.key} unlocked')
            return False