import numpy as np
from numpy import int64

from artiq.coredevice.ad9910 import _AD9910_REG_PROFILE0
from artiq.experiment import TArray, TFloat, TTuple, TFloat, parallel
from artiq.language.core import now_mu, at_mu, kernel, portable, delay, parallel, delay_mu

from waxx.control.artiq.DDS import DDS, T_AD9910_REGISTER_UPDATE_FROM_PHASE_ORIGIN_MU
from waxx.util.artiq.async_print import aprint
from waxx.config.expt_params import ExptParams

dv = -0.1
dv_array = np.array([-0.1])
di = 0

TWOPI = 2*np.pi

DDS0_IDX = 0
DDS1_IDX = 1

class RamanBeamPair():
    kernel_invariants = {
        "_sysclk_per_mu"
    }
    def __init__(self,
                 dds0:DDS,
                 dds1:DDS,
                 dds_sw:DDS,
                 params=ExptParams(),
                 frequency_transition=0.,
                 fraction_power=0.):
        self.dds0 = dds0
        self.dds1 = dds1
        self.dds_sw = dds_sw
        self.params = params
        self.p = self.params

        self.frequency_transition = frequency_transition
        # self.amplitude = amplitude
        self.fraction_power = fraction_power

        self.global_phase = 0.
        self.relative_phase = 0.
        self.t_phase_origin_mu = np.int64(0)

        self.phase_mode = 0  # 0: independent, 1: synchronized

        # self._frequency_center_dds = 0.
        self._frequency_center_plus = 0.
        self._frequency_center_minus = 0.
        self._relative_sign_fcenter = 0
        self._frequency_array = np.array([0.,0.])
        self._amplitude_0 = 0.
        self._amplitude_1 = 0.

        self.t_timeline = np.zeros(5,dtype=np.int64)
        self.t_rtio = np.zeros(5,dtype=np.int64)
        self.t_idx = 0
        self._init()

        self._dummy = np.zeros(3).astype(float)

        self._sysclk_per_mu = self.dds0.dds_device.sysclk_per_mu
        self._f_to_ftw = self.dds0.dds_device.frequency_to_ftw
        self._turns_to_pow = self.dds0.dds_device.turns_to_pow
        self._amp_to_asf = self.dds0.dds_device.amplitude_to_asf
        self._pow_relphase = np.int32(0)
        self._asf0 = np.int32(0)
        self._asf1 = np.int32(0)

    @kernel
    def get_t(self):
        self.t_timeline[self.t_idx] = now_mu()
        self.t_rtio[self.t_idx] = now_mu()
        self.t_idx += 1

    def _init(self):
        self._frequency_center_0 = self.dds0.frequency
        self._frequency_center_1 = self.dds1.frequency
        self._amplitude_0 = self.dds0.amplitude
        self._amplitude_1 = self.dds1.amplitude

        self._frequency_ratio = self._frequency_center_1/self._frequency_center_0

        fc0 = self._frequency_center_0
        fc1 = self._frequency_center_1

        self._frequency_diff_sign = np.sign(fc0 - fc1)

    @kernel
    def init(self,
            frequency_transition,
            fraction_power,
            global_phase=0.,relative_phase=0.,
            t_phase_origin_mu=np.int64(-1),
            phase_mode=1):
        if t_phase_origin_mu < 0:
            t_phase_origin_mu = now_mu()
        self.set(frequency_transition=frequency_transition,
                 fraction_power_raman=fraction_power,
                 global_phase=global_phase,
                 relative_phase=relative_phase,
                 t_phase_origin_mu=t_phase_origin_mu,
                 phase_mode=phase_mode,
                 init=True)
        self.dds_sw._restore_defaults()
        self.dds_sw.set_dds(init=True)
        self.dds0.on()
        self.dds1.on()

    @portable(flags={"fast-math"})
    def state_splitting_to_ao_frequency(self,
                                        frequency_state_splitting):

        a0 = self.dds0.aom_order
        a1 = self.dds1.aom_order

        delta = frequency_state_splitting

        fc0 = self._frequency_center_0
        fc1 = self._frequency_center_1

        f = self._frequency_ratio

        sgn = float(self._frequency_diff_sign)

        if a0 * a1 > 0:
            df_0 = (delta/2 - sgn * (fc0 - fc1))/(1 + f)
            c0 = sgn
        else:
            df_0 = (delta/2 - (fc0 + fc1))/(1 + f)
            c0 = 1.

        df_1 = df_0 * f

        self._dummy[DDS0_IDX] = fc0 + c0 * df_0
        self._dummy[DDS1_IDX] = fc1 - c0 * a0 * a1 * df_1

    @kernel
    def set_transition_frequency(self,frequency_transition=dv):
        self.set(frequency_transition)

    @kernel
    def set_phase(self,relative_phase=dv,global_phase=dv,
                  t_phase_origin_mu=np.int64(-1),
                  pretrigger=True):
        """Shifts the phase of the Raman beams. If pretrigger is True, the phase
        is set 5 us before the current timeline cursor position and the function
        does not change the timeline cursor position. Otherwise, introduces a 5
        us timeline delay.

        Minimum time between pulses when pretriggering to avoid phase skips is 3
        us.

        Args:
            relative_phase (float, optional): Relative phase between the raman
            beams. If left unset, does not change the relative phase.
            global_phase (_type_, optional): Global phase of the raman beams
            relative to t_phase_origin_mu. If left unset, does not change the
            global phase.
            t_phase_origin_mu (_type_, optional): The timestamp used for phase=0
            for each beam. If this timestamp is T, the phase at time t for a
            beam of frequency f' is phi(t) = global_phase + f' * (t - T). If
            unset, does not change the phase origin.
            pretrigger (bool, optional): Whether or not to pretrigger the set
            command. If pretrigger is True, the set command runs 5 us before the
            current timeline cursor position and the function does not change
            the timeline cursor position. Otherwise, introduces a 5 us timeline
            delay.
        """        
        
        t = now_mu()
        if pretrigger:
            delay(-5e-6)
        self.set(phase_mode=1,
                 global_phase=global_phase,
                 relative_phase=relative_phase,
                 t_phase_origin_mu=t_phase_origin_mu)
        at_mu(t)
        if not pretrigger:
            delay(5.e-6)

    @kernel
    def on(self):
        self.dds_sw.on()

    @kernel
    def off(self):
        self.dds_sw.off()

    @kernel
    def set_up_fast_frequency_update(self):
        at_mu(now_mu() & ~7)
        with parallel:
            self.dds0.dds_device.set_cfr1(phase_autoclear=1)
            self.dds1.dds_device.set_cfr1(phase_autoclear=1)
        at_mu(now_mu() & ~7)

    @kernel
    def clean_up_fast_frequency_update(self):
        at_mu(now_mu() & ~7)
        with parallel:
            self.dds0.dds_device.set_cfr1()
            self.dds1.dds_device.set_cfr1()
        at_mu(now_mu() & ~7)

    @kernel
    def set_frequency_fast(self,
                 frequency_transition):
        
        self.frequency_transition = frequency_transition
        self.state_splitting_to_ao_frequency(frequency_transition)
        f0 = self._dummy[DDS0_IDX]
        f1 = self._dummy[DDS1_IDX]

        
        at_mu(now_mu() & ~7)
        # t0 = now_mu()

        dt = np.int32(now_mu()) - np.int32(self.t_phase_origin_mu - T_AD9910_REGISTER_UPDATE_FROM_PHASE_ORIGIN_MU)
        a = dt * self._sysclk_per_mu

        ftw0 = self._f_to_ftw(f0)
        ftw1 = self._f_to_ftw(f1)
        
        pow0 = a * ftw0
        pow1 = self._pow_relphase + a * ftw1

        dds0_asf_pow_data = (self._asf0 << 16) | (pow0 & 0xffff)
        dds1_asf_pow_data = (self._asf1 << 16) | (pow1 & 0xffff)

        with parallel:
            self.dds0.dds_device.write64(_AD9910_REG_PROFILE0,
                            dds0_asf_pow_data, ftw0)
            self.dds0.dds_device.cpld.io_update.pulse_mu(8)
            delay_mu(int64(self.dds0.dds_device.sync_data.io_update_delay))
        with parallel:
            self.dds1.dds_device.write64(_AD9910_REG_PROFILE0,
                            dds1_asf_pow_data, ftw1)
            self.dds0.dds_device.cpld.io_update.pulse_mu(8)
            delay_mu(int64(self.dds1.dds_device.sync_data.io_update_delay))

        at_mu(now_mu() & ~7)

    @kernel
    def set(self,
            frequency_transition=dv,
            fraction_power_raman=dv,
            global_phase=dv, relative_phase=dv,
            t_phase_origin_mu=np.int64(-1),
            phase_mode=-1,
            init=False) -> TTuple([TFloat,TFloat]):
        """
        Set the parameters of the Raman beam pair and update the DDS channels as needed.

        This method updates the frequency, amplitude, phase mode, phase origin, global phase,
        and relative phase of the Raman beams. Only parameters that are explicitly changed
        (i.e., not left at their default values) will be updated. If `init` is True, all
        parameters are forced to update regardless of their current values.

        Args:
            frequency_transition (float, optional): The two-photon transition frequency (Hz).
                If negative or unchanged, the frequency is not updated.
            
            fraction_power_raman (float, optional): The fractional power for the Raman beams.
                If negative or unchanged, the power is not updated.
            
            global_phase (float, optional): The global phase of the Raman beams (radians).
                If negative or unchanged, the global phase is not updated.
            
            relative_phase (float, optional): The relative phase between the Raman beams (radians).
                If negative or unchanged, the relative phase is not updated.
            
            t_phase_origin_mu (int, optional): The phase origin timestamp in machine units.
                If zero or unchanged, the phase origin is not updated.
            
            phase_mode (int, optional): Phase mode (0: independent, 1: synchronized).
                If unchanged, the phase mode is not updated.
            
            init (bool, optional): If True, force all parameters to update regardless of their values.

        Side Effects:
            Updates the internal state of the object and calls the appropriate methods on the
            DDS channels to apply the new settings.
        """

        # Determine if frequency, amplitude, or v_pd should be updated

        if init:
            freq_changed = True
            fraction_power_changed = True
            phase_mode_changed = True
            phase_origin_changed = True
            global_phase_changed = True
            relative_phase_changed = True
        else:
            freq_changed = (frequency_transition >= 0.) and (frequency_transition != self.frequency_transition)
            fraction_power_changed = (fraction_power_raman >= 0.) and (fraction_power_raman != self.fraction_power)
            phase_mode_changed = (phase_mode >= 0) and (phase_mode != self.phase_mode)
            phase_origin_changed = t_phase_origin_mu >= 0. and (t_phase_origin_mu != self.t_phase_origin_mu)
            global_phase_changed = global_phase >= 0. and (global_phase != self.global_phase)
            relative_phase_changed = relative_phase >= 0. and (relative_phase != self.relative_phase)

        # Update stored values
        if freq_changed:
            self.frequency_transition = frequency_transition if frequency_transition >= 0. else self.frequency_transition
        if fraction_power_changed:
            self.fraction_power = fraction_power_raman if fraction_power_raman >= 0. else self.fraction_power
            
        if phase_mode_changed:
            self.phase_mode = phase_mode if phase_mode >= 0 else self.phase_mode
        if phase_origin_changed:
            self.t_phase_origin_mu = t_phase_origin_mu - T_AD9910_REGISTER_UPDATE_FROM_PHASE_ORIGIN_MU if t_phase_origin_mu > 0 else self.t_phase_origin_mu
        if global_phase_changed:
            self.global_phase = global_phase if global_phase >= 0. else self.global_phase
        if relative_phase_changed:
            self.relative_phase = relative_phase if relative_phase >= 0. else self.relative_phase
            self._pow_relphase = self._turns_to_pow(self.relative_phase/(4*np.pi))
        if phase_mode_changed:
            self.dds0.set_phase_mode(self.phase_mode)
            self.dds1.set_phase_mode(self.phase_mode)

        p0 = 0.
        p1 = 0.

        # t0 = now_mu()
        if freq_changed or fraction_power_changed or phase_origin_changed or global_phase_changed or relative_phase_changed:
            self.state_splitting_to_ao_frequency(self.frequency_transition)

            amp0 = np.sqrt(self.fraction_power) * self._amplitude_0
            amp1 = np.sqrt(self.fraction_power) * self._amplitude_1
            self._asf0 = self._amp_to_asf(amp0)
            self._asf1 = self._amp_to_asf(amp1)

            self._frequency_array[DDS0_IDX] = self._dummy[DDS0_IDX]
            self._frequency_array[DDS1_IDX] = self._dummy[DDS1_IDX]

            with parallel:
                p0 = self.dds0.set_dds(self._frequency_array[DDS0_IDX],
                                    amp0,
                                    t_phase_origin_mu=self.t_phase_origin_mu,
                                    phase=self.global_phase/2)
                p1 = self.dds1.set_dds(self._frequency_array[DDS1_IDX],
                                    amp1,
                                    t_phase_origin_mu=self.t_phase_origin_mu,
                                    phase=(self.global_phase+self.relative_phase)/2)
            p0 = self.dds0.update_phase()
            p1 = self.dds1.update_phase()
            # aprint(now_mu()-t0)

        self.dds0.on()
        self.dds1.on()

        return p0, p1
    
    @kernel
    def get_phase(self,
               t_mu=np.int64(-1),
               t_mu_origin=np.int64(-1),
               frequency_transition=dv,
               relative_phase=dv) -> TFloat:
        """
        Get the current relative phase of the Raman beams at time t_mu relative
        to t_mu_origin. Defined as phi_0 - phi_1 (where phi_i is the phase of ddsi).
        
        Accounts for the fact that the AOMs are double-passed, such
        that the phase accumulated by each beam is twice that expected from the
        oscillator frequency.

        Args:
            t_mu (int, optional): The time at which to get the phase in machine units.
            t_mu_origin (int, optional): The time origin for the phase calculation in machine units.
            frequency_transition (float, optional): The two-photon transition frequency (Hz).
                If negative or unchanged, uses the current AO values.
            relative_phase (float, optional): The relative phase between the Raman beams (radians).
                If negative or unchanged, uses the current relative phase.

        Returns:
            A tuple containing the phase of each Raman beam (p0, p1) at time t_mu relative to t_mu_origin.
        """
        # aprint(self.frequency_transition - frequency_transition, self.frequency_transition, frequency_transition)
        if frequency_transition == dv or frequency_transition == self.frequency_transition:
            f0 = self._frequency_array[DDS0_IDX]
            f1 = self._frequency_array[DDS1_IDX]
        else:
            self.state_splitting_to_ao_frequency(frequency_transition)
            f0 = self._dummy[DDS0_IDX]
            f1 = self._dummy[DDS1_IDX]
        relative_phase = relative_phase if relative_phase >= 0. else self.relative_phase
    
        # aprint(frequency_transition,f0,f1,relative_phase)
        p0 = self.dds0.get_phase(t_mu,
                                t_mu_origin,
                                frequency=f0,
                                phase_offset=self.global_phase)
        p1 = self.dds1.get_phase(t_mu,
                                t_mu_origin,
                                frequency=f1,
                                phase_offset=self.global_phase+relative_phase)
        return (2*(p0-p1)) % TWOPI
        
    @kernel
    def pulse(self,t):
        """Pulses the raman beam. Does not set the DDS channels -- use
        init_raman_beams for this.

        Args:
            t (float): The pulse duration in seconds.
        """        
        # self.on()
        self.dds_sw.dds_device.sw.on()
        delay(t)
        self.dds_sw.dds_device.sw.off()
        # self.off()

    @kernel
    def sweep(self,t,
              frequency_center,
              frequency_sweep_fullwidth,
              n_steps=100):
        """Sweeps the transition frequency of the two-photon transition over the
        specified range.

        Args:
            t (float): The time (in seconds) for the sweep.
            frequency_center (float, optional): The center frequency (in Hz) for the sweep range.
            frequency_sweep_fullwidth (float, optional): The full width (in Hz) of the sweep range.
            n_steps (int, optional): The number of steps for the frequency sweep.
        """
        if self.phase_mode == 1:
            self.set(phase_mode=0)

        f0 = frequency_center - frequency_sweep_fullwidth / 2
        ff = frequency_center + frequency_sweep_fullwidth / 2
        df = (ff-f0)/(n_steps-1)
        dt = t / n_steps

        self.set(frequency_transition=f0)
        self.on()
        for i in range(n_steps):
            self.set(frequency_transition=f0+i*df)
            delay(dt)
        self.off()
