from kamo import Potassium39
from waxx.control.misc.moglabs import MOGDevice
import numpy as np
import sys
import time

from waxx.util.artiq.async_print import aprint

class WavemeterController(MOGDevice):
    _instances: dict = {}

    def __new__(cls, addr, port=None, timeout=1, check=True):
        # Normalize the connection key the same way MOGDevice.__init__ does,
        # so that WavemeterController("192.168.x.y") always returns the same
        # object regardless of how many times it is called on one PC.
        if ":" not in addr:
            _port = port if port is not None else 7802
            key = f"{addr}:{_port}"
        else:
            key = addr
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, addr, port=None, timeout=1, check=True):
        if self._initialized:
            return
        super().__init__(addr, port, timeout, check)
        self.set_units()
        self._initialized = True

    def check_ch(self) -> int:
        try:
            ch = int(self.ask('OPTSW,SET'))
            return ch
        except:
            aprint('Failed to read wavemeter channel')
            return 0

    def set_channel(self, ch):
        try:
            self._set_channel(ch)
            for attempt in range(3):
                time.sleep(0.075)
                last_ch = self.check_ch()
                if last_ch == ch:
                    break
            else:
                aprint(f'Failed to set channel to {ch}, got {last_ch}')
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

    def lock_status(self, frequency_shift=0., robust=True) -> float:
        if robust:
            ok = self.check_exposure()
            ok = ok & self.check_exposure()
        self._f = self.get_frequency()

        f_target = self.target_freq + frequency_shift
        if abs(self._f - f_target) < self.locked_tolerance:
            return 1.
        else:
            aprint(f'laser {self.key} unlocked:')
            aprint(f'target = {f_target/1.e12:1.6f}, meas = {self._f/1.e12:1.6f}, diff = {(self._f - f_target)/1.e6:1.1f}')
            return 0.
        
class DummyWavemeterController():
    def check_ch(self) -> int:
        return 0
    
    def set_channel(self, ch):
        pass
    
    def select_channel(self, ch):
        pass
    
    def get_frequency(self, ch) -> float:
        return 0.0
    
    def get_saturation(self, ch) -> float:
        return 50.0
    
    def get_exposure(self, ch) -> float:
        return 0.05
    
    def set_averaging_time(self, ch, t_s=0.):
        pass
    
    def set_units(self):
        pass

class DummyWavemeterClient():
    def check_exposure(self) -> bool:
        return True
    
    def check_saturation(self) -> bool:
        return True
    
    def get_frequency(self) -> float:
        return 0.
    
    def lock_status(self) -> bool:
        return False