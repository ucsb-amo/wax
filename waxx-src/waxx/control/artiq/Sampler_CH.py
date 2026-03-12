from artiq.language import delay, kernel, TFloat, TInt32
from artiq.coredevice.sampler import Sampler, adc_mu_to_volt
import numpy as np

di = -1

class Sampler_CH():
    def __init__(self,ch,
                 gain=0,
                 sample_array=np.zeros(8,dtype=float)):
        self.ch = ch
        self.gain = gain
        self.sampler_device = Sampler
        self.key = ""
        self.samples = sample_array

    @kernel
    def sample(self) -> TFloat:
        self.sampler_device.sample(self.samples)
        return self.samples[self.ch]
    
    @kernel
    def set_gain(self,gain=di):
        self.gain = gain if gain !=  di else self.gain
        self.sampler_device.set_gain_mu(self.ch,gain=self.gain)

class Sampler_Last_CH(Sampler_CH):
    def __init__(self,ch,gain=0,sample_array=np.zeros(8,dtype=float)):
        if ch != 6 and ch != 7:
            raise ValueError('Last channel readout only supported for channels 6 and 7!')
        super().__init__(ch=ch,gain=gain,sample_array=sample_array)

    @kernel
    def sample_single(self) -> TFloat:
        adc_data = [0]*2
        self.sample_single_mu(adc_data)
        cfs = self.sampler_device.corrected_fs
        for ch in [6,7]:
            channel = (ch-6) + 8 - len(self.samples)
            gain = (self.sampler_device.gains >> (channel)*2) & 0b11
            self.samples[ch] = adc_mu_to_volt(adc_data[ch-6], gain, cfs)
        return self.samples[self.ch]

    @kernel
    def sample_single_mu(self, data):
        self.sampler_device.cnv.pulse(30.e-9)
        delay(450.e-9)
        mask = 1 << 15
        self.sampler_device.bus_adc.write(0)
        val = self.sampler_device.bus_adc.read()
        data[1] = val >> 16
        val &= 0xffff
        data[0] = -(val & mask) + (val & ~mask)
            