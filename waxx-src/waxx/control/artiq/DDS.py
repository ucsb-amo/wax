from artiq.experiment import *
from artiq.experiment import delay_mu, delay, parallel
from artiq.language.core import now_mu, at_mu
import numpy as np
from numpy import int32, int64

from artiq.coredevice import ad9910, ad53xx, ttl
import artiq.coredevice.urukul as urukul
from artiq.coredevice import spi2 as spi

from waxx.util.artiq.async_print import aprint

T_AD9910_REGISTER_UPDATE_FROM_PHASE_ORIGIN_MU = np.int64(2030 - 688)
# T_AD9910_PIPELINE_LATENCY_MU = np.int64(107)
T_AD9910_PIPELINE_LATENCY_MU = np.int64(108)

T_TRACKING_PHASE_LAG_MU = 1960
DAC_CH_DEFAULT = -1
di2 = 2

PHASE_MODE_TRACKING = 1
PHASE_MODE_CONTINUOUS = 0

TWOPI = 2 * np.pi
TWOPI_NS = 2 * np.pi * 1.e-9

class DDS():

   def __init__(self, urukul_idx, ch, frequency=0., amplitude=0., v_pd=0., dac_device=[], device_db=None):
      self.urukul_idx = urukul_idx
      self.ch = ch
      self.frequency = frequency
      self.amplitude = amplitude
      self.phase_offset = 0.
      self.t_phase_origin_mu = np.int64(0)

      self.sw_state = 0
      self.aom_order = 0
      self.transition = 'None'
      self.double_pass = True
      self.v_pd = v_pd
      self.phase_mode = 0
      self.dac_ch = DAC_CH_DEFAULT
      self.key = ""

      self.dds_device = ad9910.AD9910
      self.name = f'urukul{self.urukul_idx}_ch{self.ch}'
      self.cpld_name = []
      self.cpld_device = urukul.CPLD
      self.bus_channel = []
      if device_db is not None:
         self.read_db(device_db)
      
      if dac_device:
         self.dac_device = dac_device
      else:
         self.dac_device = ad53xx.AD53xx
         
      self.dac_control_bool = self.dac_ch != DAC_CH_DEFAULT

      self._t_att_xfer_mu = np.int64(1592) # see https://docs.google.com/document/d/1V6nzPmvfU4wNXW1t9-mRdsaplHDKBebknPJM_UCvvwk/edit#heading=h.10qxjvv6p35q
      self._t_set_xfer_mu = np.int64(1248) # see https://docs.google.com/document/d/1V6nzPmvfU4wNXW1t9-mRdsaplHDKBebknPJM_UCvvwk/edit#heading=h.e1ucbs8kjf4z
      self._t_ref_period_mu = np.int64(8) # one clock cycle, 125 MHz --> T = 8 ns (mu)

      self._t_set_delay_mu = self._t_set_xfer_mu + self._t_ref_period_mu + 1
      self._t_att_delay_mu = self._t_att_xfer_mu + self._t_ref_period_mu + 1

      self._phase_at_t = 0 # phase at timestamp self._t_phase_mu
      self._t_phase_mu = np.int64(0) # timestamp at which the wave has phase self._phase_at_mu

      self._t_io_update_delay_mu = np.int64(0)
      self._last_ftw = 0
      self._ftw = 0
      self._pow = 0
      self._asf = 0
      self._phase_t_last_set = 0
      self._t_last_set_mu = np.int64(0)
      self._t_last_change_mu = np.int64(0)

      self.dds_device.sw: ttl.TTLOut

   @kernel
   def _store_io_update_delay(self):
      self._t_io_update_delay_mu = T_AD9910_PIPELINE_LATENCY_MU - np.int64(4) + self.dds_device.sync_data.io_update_delay

   @portable
   def _stash_defaults(self):
      self._frequency_default = self.frequency
      self._amplitude_default = self.amplitude

   @portable
   def _restore_defaults(self):
      self.frequency = self._frequency_default
      self.amplitude = self._amplitude_default

   @portable
   def update_dac_bool(self):
      self.dac_control_bool = (self.dac_ch != DAC_CH_DEFAULT)

   @portable(flags={"fast-math"})
   def detuning_to_frequency(self,linewidths_detuned) -> TFloat:
      '''
      Returns the DDS frequency value in MHz corresponding to detuning =
      linewidths_detuned * Gamma from the resonant D1, D2 transitions. Gamma = 2
      * pi * 6 MHz.

      D1 AOMs give detuning relative to |g> -> |F=2>.
      D2 AOMs give detuning relative to |g> -> unresolved D2 peak.

      Parameters
      ----------
      linewidths_detuned: float
         Detuning in units of linewidth Gamma = 2 * pi * 6 MHz.

      Returns
      -------
      float
         The corresponding AOM frequency setting in Hz.
      '''
      linewidths_detuned=float(linewidths_detuned)
      f_shift_to_resonance_MHz = 461.7 / 2 # half the crossover detuning. Value from T.G. Tiecke.
      linewidth_MHz = 6
      detuning_MHz = linewidths_detuned * linewidth_MHz
      freq = ( f_shift_to_resonance_MHz + self.aom_order * detuning_MHz ) / 2
      if not self.double_pass:
         freq = freq * 2
      return freq * 1.e6
   
   @portable(flags={"fast-math"})
   def frequency_to_detuning(self,frequency) -> TFloat:
      frequency = float(frequency) / 1e6
      f_shift_to_resonance = 461.7 / 2
      linewidth_MHz = 6
      if not self.double_pass:
         double_pass_multiplier = 1
      else:
         double_pass_multiplier = 2
      detuning = self.aom_order * (double_pass_multiplier * frequency - f_shift_to_resonance) / linewidth_MHz
      return detuning
   
   @kernel(flags={"fast-math"})
   def set_dds_gamma(self, delta=-1000., amplitude=-0.1, v_pd=-0.1, phase=0.,
               t_phase_origin_mu=np.int64(0)):
      '''
      Sets the DDS frequency and attenuation. Uses delta (detuning) in units of
      gamma, the linewidth of the D1 and D2 transition (Gamma = 2 * pi * 6 MHz).

      Parameters:
      -----------
      delta: float
         Detuning in units of linewidth Gamma = 2 * pi * 6 MHz. (default: use
         stored self.frequency)

      amplitude: float
      '''
      self.update_dac_bool()
      delta = float(delta)
      if delta == -1000.:
         frequency = -0.1
      else:
         frequency = self.detuning_to_frequency(linewidths_detuned=delta)

      self.set_dds(frequency=frequency, amplitude=amplitude, v_pd=v_pd,
                   t_phase_origin_mu=t_phase_origin_mu, phase=phase)

   @kernel(flags={"fast-math"})
   def set_dds(self, frequency=-0.1, amplitude=-0.1, v_pd=-0.1, phase=0.,
               t_phase_origin_mu=np.int64(-1),
               dt_phase_origin_shift_mu=T_AD9910_REGISTER_UPDATE_FROM_PHASE_ORIGIN_MU,
               init=False):
      '''
      Set the DDS (Direct Digital Synthesizer) frequency, amplitude, phase, and optionally DAC voltage.

      This method updates the DDS device with new frequency, amplitude, and phase values,
      and, if applicable, sets the associated DAC voltage. Only parameters with non-negative
      values different from the current state are updated. If init is True, all parameters
      are forced to update regardless of their values.

      Args:
         frequency (float, optional): 
            Frequency in Hz. If negative or unchanged, frequency is not updated.
         amplitude (float, optional): 
            Amplitude in V. If negative or unchanged, amplitude is not updated.
         v_pd (float, optional): 
            Voltage for the DAC. If negative or unchanged, voltage is not updated. 
            Only used if the DDS is controlled by a DAC.
         phase (float, optional): 
            Phase offset in radians (0 to 2π). If negative or unchanged, phase is not updated.
         t_phase_origin_mu (int, optional): 
            Phase origin timestamp in machine units. If zero or unchanged, not updated.
         init (bool, optional): 
            If True, force all parameters to update regardless of their values.

      Side Effects:
         Updates the internal state of the DDS object and applies the new settings to the hardware.
         If the DDS is associated with a DAC, also updates the DAC voltage.
      '''

      self.update_dac_bool()

      if init:
         # If init is True, force update
         freq_changed = True
         amp_changed = True
         vpd_changed = True
         phase_origin_changed = True
         phase_changed = True
      else:
         # Determine if frequency, amplitude, or v_pd should be updated
         freq_changed = (frequency >= 0.) and (frequency != self.frequency)
         amp_changed = (amplitude >= 0.) and (amplitude != self.amplitude)
         vpd_changed = (v_pd >= 0.) and (v_pd != self.v_pd)
         phase_origin_changed = t_phase_origin_mu >= 0. and (t_phase_origin_mu != self.t_phase_origin_mu)
         phase_changed = phase >= 0. and (phase != self.phase_offset)

      self._last_ftw = self.dds_device.frequency_to_ftw(self.frequency)
      # Update stored values
      if freq_changed:
         self.frequency = frequency if frequency >= 0. else self.frequency
         self._ftw = self.dds_device.frequency_to_ftw(self.frequency)
      if amp_changed:
         self.amplitude = amplitude if amplitude >= 0. else self.amplitude
         self._asf = self.dds_device.amplitude_to_asf(self.amplitude)
      if self.dac_control_bool and vpd_changed:
         self.v_pd = v_pd if v_pd >= 0. else self.v_pd
      if phase_origin_changed:
         self.t_phase_origin_mu = t_phase_origin_mu - dt_phase_origin_shift_mu if t_phase_origin_mu > 0 else self.t_phase_origin_mu
      if phase_changed:
         self.phase_offset = phase if phase >= 0. else self.phase_offset
         self._pow = self.dds_device.turns_to_pow(self.phase_offset/TWOPI)

      # Set DDS and DAC as needed
      if self.dac_control_bool and (vpd_changed or init):
         self.update_dac_setpoint(self.v_pd)
      if freq_changed or amp_changed or phase_origin_changed or phase_changed or init:
         
         self.dds_device.set(frequency=self.frequency,
                                    amplitude=self.amplitude, 
                                    phase=self.phase_offset/TWOPI,
                                    ref_time_mu=self.t_phase_origin_mu)
         # if self.phase_mode == PHASE_MODE_TRACKING:
         #    self._t_phase_mu = now_mu()
         #    self._phase_at_t = self.get_phase(self._t_phase_mu)

         # else:
         self.update_phase_at_set()

   @kernel
   def reset_phase(self):
      self._t_last_change_mu = now_mu()
      self._phase_t_last_set = 0
      
   @kernel
   def update_phase_at_set(self):
      t_now = now_mu()
      
      T = self._t_io_update_delay_mu

      self._phase_t_last_set += int32(self._last_ftw * T + self._ftw * (t_now - self._t_last_change_mu))

      self._t_last_set_mu = t_now
      self._t_last_change_mu = self._t_last_set_mu + T

   @kernel
   def get_phase(self):
      t_mu = now_mu()

      # if self.phase_mode == PHASE_MODE_TRACKING:
      #    t_mu_origin = t_mu_origin if t_mu_origin > 0 else self.t_phase_origin_mu
      #    phase = pow + ftw * (t_mu - (t_mu_origin - T_AD9910_REGISTER_UPDATE_FROM_PHASE_ORIGIN_MU))

      # else: # continuous, does not correctly account for manual phase stepping
      phase = int32(self._phase_t_last_set)
      phase += int32(self._last_ftw * T_AD9910_PIPELINE_LATENCY_MU \
                  + self._ftw * (t_mu - self._t_last_change_mu))
      return phase & int32(0xffff)
   
   @kernel
   def update_dac_setpoint(self, v_pd=-0.1, dac_load = True):

      self.v_pd = v_pd if v_pd >= 0. else self.v_pd
      v_pd = self.v_pd

      self.dac_device.write_dac(channel=self.dac_ch, voltage=v_pd)
      if dac_load:
         self.dac_device.load()

   def get_devices(self,expt):
      self.dds_device = expt.get_device(self.name)
      self.cpld_device = expt.get_device(self.cpld_name)

   @kernel
   def off(self, dac_update = True, dac_load = True):
      self.update_dac_bool()
      self.dds_device.sw.off()
      if self.dac_control_bool and dac_update:
         self.dac_device.write_dac(channel=self.dac_ch,voltage=0.)
         if dac_load:
            self.dac_device.load()
      self.sw_state = 0

   @kernel
   def on(self, dac_update = True, dac_load=True):
      self.update_dac_bool()
      if self.dac_control_bool and dac_update:
         self.dac_device.write_dac(channel=self.dac_ch,voltage=self.v_pd)
         if dac_load:
            self.dac_device.load()
      self.dds_device.sw.on()
      self.sw_state = 1

   @kernel
   def set_sw(self, state=-1):
      self.sw_state = state if state != -1 else self.sw_state

      if self.sw_state == 1:
         self.dds_device.sw.on()
      else:
         self.dds_device.sw.off()

   @kernel
   def set_phase_mode(self, mode=0):
      '''
      Sets the phase mode of the DDS. See ad9910.AD9910.set_phase_mode for
      details.

      Args:
          mode (int, optional): Phase mode to set. 0 for continuous phase mode,
          1 for tracking phase mode. Defaults to 0 (continuous phase mode).
      '''    
      if mode == 0:
         self.dds_device.set_phase_mode(ad9910.PHASE_MODE_CONTINUOUS)
         self.phase_mode = mode
      elif mode == 1:
         self.dds_device.set_phase_mode(ad9910.PHASE_MODE_TRACKING)
         self.phase_mode = mode

   @kernel
   def init(self, blind=False):
      self.cpld_device.init(blind=blind)
      delay(1*ms)
      self.dds_device.init(blind=blind)
      delay(1*ms)

   @kernel
   def write_frequency_register_mu(self, ftw):
      """
      Directly writes the frequency tuning word (FTW) to the DDS register. This
      is a low-level operation that bypasses the usual frequency setting
      methods, and should be used with caution. 
      
      Does not pulse the IO update pin, so the new frequency will not take
      effect until the next scheduled IO update. This method is intended for
      advanced users who need precise control over the timing of frequency
      changes and are familiar with the internal workings of the AD9910 DDS.
      """
      # ftw = self.ftw
      # self.dds_device.bus.set_config_mu(urukul.SPI_CONFIG, 8,
      #                          urukul.SPIT_DDS_WR, self.dds_device.chip_select)
      # self.dds_device.bus.write((ad9910._AD9910_REG_PROFILE0 + 7) << 24)
      # self.dds_device.bus.set_config_mu(urukul.SPI_CONFIG | spi.SPI_END, 32,
      #                          urukul.SPIT_DDS_WR, self.dds_device.chip_select)
      # self.dds_device.bus.write(ftw)
      # don't use rn, corrupts SPI transaction

   def read_db(self,ddb):
      '''read out info from ddb. ftw_per_hz comes from artiq.frontend.moninj, line 206-207'''
      v = ddb[self.name]
      self.cpld_name = v["arguments"]["cpld_device"]
      spi_dev = ddb[self.cpld_name]["arguments"]["spi_device"]
      self.bus_channel = ddb[spi_dev]["arguments"]["channel"]
   
   