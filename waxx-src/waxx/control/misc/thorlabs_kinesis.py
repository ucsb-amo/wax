import serial
import time
import socket
import json
import threading
from typing import Union, Literal

import numpy as np
from waxa.helper.datasmith import normalize

from pylablib.devices.Thorlabs import KinesisMotor
from artiq.coredevice.core import Core
from artiq.language import TBool, TFloat, TInt32, now_mu, delay, kernel
from waxx.control.artiq import Sampler_CH

from waxx.util.artiq.async_print import aprint

POSITION_180 = -345535
POSITION_90 = POSITION_180 / 2
P_GAIN = 50

class ThorlabsKinesisMotor():
    def __init__(self, device_id = 27500961):
        self._device_id = device_id

    def init(self, force_home=True) -> TBool:
        self.motor = KinesisMotor(self._device_id)
        self.setup_velocity()
        self.motor._home(force=force_home)
        self.motor.move_to(0)
        while self.is_moving():
            time.sleep(0.01)
        self.motor._set_position_reference()
        return True

    def move_by(self,steps):
        self.motor._move_by(steps)

    def move_to(self,position):
        self.motor._move_to(position)

    def setup_velocity(self,acceleration=10000,max_velocity=100000):
        self.motor._setup_velocity(min_velocity=0,
                                acceleration=acceleration,
                                max_velocity=max_velocity)
        
    def is_moving(self) -> TBool:
        return self.motor._is_moving()
    
    def get_position(self) -> TInt32:
        return self.motor._get_position()

    def home(self) -> TBool:
        if not self.motor._is_homed():
            self.motor._home()
            while self.is_moving():
                time.sleep(0.1)
        return True

class WaveplateRotatorPhotodiodePID():
    def __init__(self, 
                 kinesis_device_id: int, 
                 sampler_ch: Sampler_CH,
                 core: Core):
        self._kinesis_devid = kinesis_device_id
        self.sampler_ch = sampler_ch
        self.core = core

        self.motor = ThorlabsKinesisMotor(self._kinesis_devid)

        self._v_temp = np.zeros(1000)
        self._p_temp = np.zeros(1000, dtype=np.int32)
        self.idx = 0

    @kernel
    def init(self, force_home=True) -> TBool:
        self.core.wait_until_mu(now_mu())
        self.motor.init(force_home=force_home)
        self.core.break_realtime()
        self.wait_until_stopped()
        return True
    
    @kernel
    def get_position(self) -> TInt32:
        self.core.wait_until_mu(now_mu())
        pos = self.motor.get_position()
        self.core.break_realtime()
        return pos
    
    @kernel
    def move_to(self, pos=0, fraction_power=-1):
        self.core.wait_until_mu(now_mu())
        if fraction_power != -1:
            pos = self.position_from_pfrac_calibration(fraction_power)
        self.motor.move_to(pos)
        self.core.break_realtime()

    @kernel
    def move_by(self, steps=0):
        self.core.wait_until_mu(now_mu())
        self.motor.move_by(steps)
        self.core.break_realtime()
    
    @kernel
    def is_moving(self) -> TBool:
        self.core.wait_until_mu(now_mu())
        b = self.motor.is_moving()
        self.core.break_realtime()
        return b
    
    @kernel
    def wait_until_stopped(self):
        while self.is_moving():
            pass

    @kernel
    def find_pd_range(self, n_calibration_steps=10):
        self.move_to(0)
        self.wait_until_stopped()

        self.idx = 0
        step_size = np.int32( POSITION_90 / n_calibration_steps )
        
        for _ in range(n_calibration_steps):
            self.move_by(step_size)
            self.wait_until_stopped()

            p = self.get_position()
            v = self.sampler_ch.sample()

            self._p_temp[self.idx] = p
            self._v_temp[self.idx] = v

            aprint(p,v)
            self.idx += 1

        self.build_lookup_table()
        
    def build_lookup_table(self):
        self._vpd = self._v_temp[0:self.idx]
        self._pos = self._p_temp[0:self.idx]
        
        # Smooth with 3-point moving average to find extrema
        smoothed = np.convolve(self._vpd, np.ones(3)/3, mode='same')
        
        # Find indices of max and min
        idx_max = np.argmax(smoothed)
        idx_min = np.argmin(smoothed)
        
        # Crop to monotonic region between min and max
        start_idx = min(idx_min, idx_max)
        end_idx = max(idx_min, idx_max)
        
        self._vpd = self._vpd[start_idx:end_idx+1]
        self._pos = self._pos[start_idx:end_idx+1]
        
        # Flip if necessary to make vpd monotonically increasing
        if self._vpd[0] > self._vpd[-1]:
            self._vpd = self._vpd[::-1]
            self._pos = self._pos[::-1]
        
        self._pfrac = normalize(self._vpd, map_minimum_to_zero=True)

    def position_from_pfrac_calibration(self, fraction_power) -> TInt32:
        if fraction_power < 0 or fraction_power > 1:
            raise ValueError("fraction_power must be between 0 and 1")
        pos = np.interp(fraction_power, self._pfrac, self._pos).astype(np.int32)
        return pos
    
    def get_set_point(self, fraction_power) -> TFloat:
        v_set = np.interp(fraction_power, self._pfrac, self._vpd).astype(np.int32)
        return v_set
    
    @kernel
    def set_fraction_power(self, fraction_power):
        v_set = self.get_set_point(fraction_power)

        # intial move
        self.move_to(fraction_power=fraction_power)
        while self.is_moving():
            pass

        # now PID
        n_iter = 0
        N_MAX_ITER = 100
        V_PD_THRESHOLD = 50.e-3
        while True:
            v_pd = self.sampler_ch.sample()
            err = v_set - v_pd
            if err < V_PD_THRESHOLD:
                break
            steps = np.int32(err * P_GAIN)
            self.move_by(steps)
            n_iter = n_iter + 1
            if n_iter > N_MAX_ITER:
                raise ValueError("Waveplate rotator lock failed to meet setpoint within threshold.")