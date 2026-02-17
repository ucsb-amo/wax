import serial
import time
import socket
import json
import threading
from typing import Union, Literal

import numpy as np

from pylablib.devices.Thorlabs import KinesisMotor
from artiq.coredevice.core import Core
from artiq.language import TBool, TFloat, TInt32
from waxx.control.artiq import Sampler_CH

POSITION_180 = -345535
POSITION_90 = POSITION_180 / 2

class ThorlabsKinesisMotor():
    def __init__(self, device_id = 27500961):
        self._device_id = device_id

    def init(self):
        self.motor = KinesisMotor(self._device_id)
        self.setup_velocity()

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
    
    def home(self) -> TBool:
        self.motor._home()
        while self.is_moving():
            time.sleep(0.1)
        self.motor._set_position_reference()
        return True

class WaveplateRotatorPhotodiodePID():
    def __init__(self, 
                 kinesis_device_id: int, 
                 sampler_ch: Sampler_CH,
                 core: Core):
        self._kinesis_devid = kinesis_device_id
        self.sampler_ch = sampler_ch

        self._vpd = np.zeros(1000)

    def init(self) -> TBool:
        self.motor = ThorlabsKinesisMotor(self._kinesis_devid)
        self.motor.home()
        return True
    
    def find_pd_range(self):
        self.motor.move_to(0)
        # self.sampler_ch.sample()