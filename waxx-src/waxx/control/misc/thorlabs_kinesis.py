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

    def init(self,
             force_home=True,
             move_to_zero=True) -> TBool:
        aprint('initializing kinesis mount')
        self.motor = KinesisMotor(self._device_id)
        self.setup_velocity()
        aprint('start homing...')
        self.motor._home(force=force_home)
        aprint('moving...')
        if move_to_zero:
            self.move_to(0)
        while self.is_moving():
            time.sleep(0.01)
        self.motor._set_position_reference()
        return True

    def move_by(self,steps):
        self.motor._move_by(steps)

    def move_to(self,position):
        self.motor._move_to(position)

    def setup_velocity(self,acceleration=20000,max_velocity=100000):
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

        self._v_temp = np.zeros(100000)
        self._p_temp = np.zeros(100, dtype=np.int32)
        self.idx = 0

    def close(self):
        try:
            self.motor.motor.close()
        except:
            pass

    @kernel
    def init(self,
             force_home=True,
             move_to_zero=True) -> TBool:
        self.core.wait_until_mu(now_mu())
        self.motor.init(force_home=force_home,
                        move_to_zero=move_to_zero)
        self.wait_until_stopped()
        self.core.break_realtime()
        return True
    
    @kernel
    def get_position(self) -> TInt32:
        self.core.wait_until_mu(now_mu())
        pos = self.motor.get_position()
        delay(5.e-3)
        return pos
    
    @kernel
    def move_to(self, pos=0, fraction_power=-1):
        self.core.wait_until_mu(now_mu())
        if fraction_power != -1:
            pos = self.position_from_pfrac_calibration(fraction_power)
        self.motor.move_to(pos)
        delay(5.e-3)

    @kernel
    def move_by(self, steps=0):
        self.core.wait_until_mu(now_mu())
        self.motor.move_by(steps)
        delay(5.e-3)
    
    @kernel
    def is_moving(self) -> TBool:
        self.core.wait_until_mu(now_mu())
        b = self.motor.is_moving()
        delay(5.e-3)
        return b
    
    @kernel
    def wait_until_stopped(self):
        t = now_mu()
        while self.is_moving():
            pass
        tf = now_mu()
        aprint('move time: ',(tf - t)/1.e9)

    @kernel
    def find_pd_range(self,
                    n_calibration_steps=25,
                    N_samples_per_step=10):
        
        aprint('moving to 0 to start')

        self.move_to(0)
        self.wait_until_stopped()

        aprint('starting sample')

        Nr = N_samples_per_step

        self.idx = 0
        step_size = np.int32( POSITION_90 * 1.05 / n_calibration_steps )
        
        for _ in range(n_calibration_steps):
            if self.idx != 0:
                aprint('move to new spot....')

                self.move_by(step_size)
                self.wait_until_stopped()

                aprint('arrived at new spot. sampling...')

            p = self.get_position()
            self.core.break_realtime()
            t0 = now_mu()
            for j in range(N_samples_per_step):
                v = self.sampler_ch.sample()
                delay(8.e-6)
                self._v_temp[self.idx*Nr + j] = v
            tf = now_mu()
            aprint('done sampling for step', self.idx, '. sampling time ', (tf-t0)/1.e9)

            self._p_temp[self.idx] = p
            self.idx += 1

        self.build_lookup_table(self._p_temp[0:self.idx],
                                self._v_temp[0:int(Nr*self.idx)],
                                Nr=Nr)
        
    def build_lookup_table(self, p, v, Nr):
        self._pos = p
        self._vpd = np.reshape(v,(Nr,-1)).mean(axis=0)
        
        # Find indices of max and min
        idx_max = np.argmax(self._vpd)
        idx_min = np.argmin(self._vpd)
        
        # Crop to monotonic region between min and max
        start_idx = min(idx_min, idx_max)
        end_idx = max(idx_min, idx_max)
        
        self._vpd = self._vpd[start_idx:end_idx+1]
        self._pos = self._pos[start_idx:end_idx+1]
        
        # Flip if necessary to make vpd monotonically increasing
        if self._vpd[0] > self._vpd[-1]:
            self._vpd = self._vpd[::-1]
            self._pos = self._pos[::-1]
        
        try:
            self._pfrac = normalize(self._vpd, map_minimum_to_zero=True)
        except Exception as e:
            aprint('no dice on pfrac comp')
            print(e)

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