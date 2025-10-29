import numpy as np

from artiq.experiment import portable, kernel, rpc, \
                                TFloat, TArray
from artiq.language.core import delay, now_mu, at_mu

from waxx.control.artiq import DDS, TTL, DAC_CH
from waxx.config.expt_params_waxx import ExptParams
from waxx.util.artiq.async_print import aprint

dv = -10.e9

FREQUENCY_GS_HFS = 461.7 * 1.e6

class BeatLockImaging():
    def __init__(self,
                 dds_sw=DDS,
                 dds_beatref=DDS,
                 N_beatref_mult=8,
                 beatref_sign=-1,
                 frequency_minimum_beat=250.e6,
                 expt_params=ExptParams):
        
        self.dds_sw = dds_sw
        self.dds_beatref = dds_beatref

        self.params = expt_params
        self.p = self.params

        self._N_beatref_mult = N_beatref_mult
        # +1 for lock greater frequency than reference (Gain switch "+"), vice versa ("-")
        self._beat_sign = beatref_sign
        self._frequency_minimum_beat = frequency_minimum_beat

    @portable(flags={"fast-math"})
    def imaging_detuning_to_beat_ref(self, frequency_detuned) -> TFloat:
        """Converts a desired imaging detuning to the required beat lock reference.

        Makes reference to the beat lock sign, which DDS channel drives the AO
        to frequency shift the imaging light, and the reference multiplier
        setting on the beat lock controller.

        Args:
            frequency_detuned (float, optional): The desired imaging detuning
            from the brightest D2 resonance in Hz. Whether the detuning is
            relative to F=2 -> 4P3/2 or F=1 -> 4P3/2 depends on the parameter
            ExptParams.imaging_state (if == 1: F=1, if == 2: F=2)

        Returns:
            TFloat: the required beat lock reference frequency in Hz.
        """        

        f_shift_resonance = FREQUENCY_GS_HFS / 2
        f_ao_shift = self.dds_sw.frequency * self.dds_sw.aom_order * 2
        f_offset = self._beat_sign * (frequency_detuned - f_ao_shift - f_shift_resonance)

        f_beatlock_ref = f_offset / self._N_beatref_mult

        if f_offset < self._frequency_minimum_beat:
            aprint("The requested detuning results in an offset less than the minimum beat note frequency for the lock.")
        if f_beatlock_ref < 0.:
            aprint("The requested detuning would require a negative reference frequency. You'll need to flip the beat lock sign to reach this detuning.")
        if f_beatlock_ref > 400.e6:
            aprint("Invalid beatlock reference frequency for requested detuning (>400 MHz). Must be less than 400 MHz for ARTIQ DDS. Consider changing the beat lock reference multiplier.")

        return f_beatlock_ref
    
    @kernel(flags={"fast-math"})
    def set_imaging_detuning(self, frequency_detuned, amp):
        '''
        Sets the detuning of the beat-locked imaging laser (in Hz).

        Imaging detuning is controlled by two things -- the Vescent offset lock
        and a double pass (-1 order).

        The offset lock has a multiplier, N, that determines the offset lock
        frequency relative to the lock point of the D2 laser locked at the
        crossover feature for the D2 transition. Offset = N * reference freqeuency.
        
        The reference frequency is provided by a DDS channel (dds_frame.beatlock_ref).
        '''

        f_beatlock_ref = self.imaging_detuning_to_beat_ref(frequency_detuned=frequency_detuned)

        self.dds_sw.set_dds(amplitude=amp)

        f_offset = f_beatlock_ref * self._N_beatref_mult
        if f_offset < self._frequency_minimum_beat:
            raise ValueError("The beat lock is unhappy at a lock point below the minimum offset.")
        
        if f_beatlock_ref < 0.:
            raise ValueError("You tried to set the DDS to a negative frequency!")
        
        self.dds_beatref.set_dds(frequency=f_beatlock_ref)
        self.dds_beatref.on()

    @kernel
    def pulse(self,t):
        """Pulses the imaging beam.

        Args:
            t (float): The time of the imaging pulse.
        """        
        self.dds_sw.on()
        delay(t)
        self.dds_sw.off()

class PolModBeatLock(BeatLockImaging):
    def __init__(self,
                 dds_sw=DDS,
                 dds_polmod_v=DDS,
                 dds_polmod_h=DDS,
                 dds_beatref=DDS,
                 N_beatref_mult=8,
                 beatref_sign=-1,
                 frequency_minimum_beat=250.e6,
                 expt_params=ExptParams):
        super().__init__(dds_sw=dds_sw,
            dds_beatref=dds_beatref,
            N_beatref_mult=N_beatref_mult,
            beatref_sign=beatref_sign,
            frequency_minimum_beat=frequency_minimum_beat,
            expt_params=expt_params)
    
        self.dds_polmod_v = dds_polmod_v
        self.dds_polmod_h = dds_polmod_h

        self.frequency_polmod = 0.

        self._frequency_array = np.array([0.,0.])

        self._init()

    @portable(flags={"fast-math"})
    def imaging_detuning_to_beat_ref(self, frequency_detuned) -> TFloat:
        """Converts a desired imaging detuning to the required beat lock reference.

        Makes reference to the beat lock sign, which DDS channel drives the AO
        to frequency shift the imaging light, and the reference multiplier
        setting on the beat lock controller.

        Args:
            frequency_detuned (float, optional): The desired imaging detuning
            from the brightest D2 resonance in Hz. Whether the detuning is
            relative to F=2 -> 4P3/2 or F=1 -> 4P3/2 depends on the parameter
            ExptParams.imaging_state (if == 1: F=1, if == 2: F=2)

        Returns:
            TFloat: the required beat lock reference frequency in Hz.
        """        

        f_shift_resonance = FREQUENCY_GS_HFS / 2
        f_ao_shift = self.dds_sw.frequency * self.dds_sw.aom_order * 2
        if self.frequency_polmod > 0.:
            f_polmod_ao_shift = self.dds_polmod_v.aom_order * self.dds_polmod_v.frequency \
                                + self.dds_polmod_h.aom_order * self.dds_polmod_h.frequency
        else:
            f_polmod_ao_shift = self.dds_polmod_v.aom_order * self.dds_polmod_v.frequency
        f_offset = 1/self._beat_sign * (frequency_detuned - f_ao_shift - f_shift_resonance - f_polmod_ao_shift)

        f_beatlock_ref = f_offset / self._N_beatref_mult

        if f_offset < self._frequency_minimum_beat:
            aprint("The requested detuning results in an offset less than the minimum beat note frequency for the lock.")
        if f_beatlock_ref < 0.:
            aprint("The requested detuning would require a negative reference frequency. You'll need to flip the beat lock sign to reach this detuning.")
        if f_beatlock_ref > 400.e6:
            aprint("Invalid beatlock reference frequency for requested detuning (>400 MHz). Must be less than 400 MHz for ARTIQ DDS. Consider changing the beat lock reference multiplier.")

        return f_beatlock_ref
    
    @kernel(flags={"fast-math"})
    def set_imaging_detuning(self, frequency_detuned, amp,
                             frequency_polmod=0.):
        '''
        Sets the detuning of the beat-locked imaging laser (in Hz).

        Imaging detuning is controlled by two things -- the Vescent offset lock
        and a double pass (-1 order).

        The offset lock has a multiplier, N, that determines the offset lock
        frequency relative to the lock point of the D2 laser locked at the
        crossover feature for the D2 transition. Offset = N * reference freqeuency.
        
        The reference frequency is provided by a DDS channel (dds_frame.beatlock_ref).
        '''
        self.set_polmod(frequency_polmod=frequency_polmod)

        self.dds_sw.set_dds(amplitude=amp)

        f_beatlock_ref = self.imaging_detuning_to_beat_ref(frequency_detuned=frequency_detuned)

        f_offset = f_beatlock_ref * self._N_beatref_mult
        if f_offset < self._frequency_minimum_beat:
            raise ValueError("The beat lock is unhappy at a lock point below the minimum offset.")
        
        if f_beatlock_ref < 0.:
            raise ValueError("You tried to set the DDS to a negative frequency!")
        
        self.dds_beatref.set_dds(frequency=f_beatlock_ref)
        self.dds_beatref.on()

    def _init(self):
        self._frequency_center_dds = (self.dds_polmod_h.frequency + self.dds_polmod_v.frequency)/2
        if abs(self._frequency_center_dds - self.dds_polmod_h.frequency) != abs(self._frequency_center_dds - self.dds_polmod_v.frequency):
            raise ValueError("The - and + DDS frequencies should be equidistant from their mean for optimal efficiency.")

    @kernel(flags={"fast_math"})
    def polmod_frequency_to_ao_frequency(self, frequency_polmod)  -> TArray(TFloat):

        if frequency_polmod > 0.:
            order_p = self.dds_polmod_h.aom_order
            order_m = self.dds_polmod_v.aom_order

            frequency_polmod = frequency_polmod / 2 # bc the atoms respond same to polarization rotated by pi

            df = frequency_polmod / 4

            if order_p * order_m == -1:
                self._frequency_array[0] = df
                self._frequency_array[1] = df
            else:
                self._frequency_array[0] = self._frequency_center_dds + df
                self._frequency_array[1] = self._frequency_center_dds - df
        else:
            self._frequency_array[0] = self._frequency_center_dds
            self._frequency_array[1] = 0

        return self._frequency_array

    @kernel
    def set_polmod(self,
            frequency_polmod=dv,
            # amp=dv,
            global_phase=dv, relative_phase=dv,
            t_phase_origin_mu=np.int64(-1),
            phase_mode=0,
            init=False):
        """
        Set the parameters of the Raman beam pair and update the DDS channels as needed.

        This method updates the frequency, amplitude, phase mode, phase origin, global phase,
        and relative phase of the Raman beams. Only parameters that are explicitly changed
        (i.e., not left at their default values) will be updated. If `init` is True, all
        parameters are forced to update regardless of their current values.

        Args:
            frequency_polmod (float, optional): The two-photon transition frequency (Hz).
                If negative or unchanged, the frequency is not updated.
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
        freq_changed = (frequency_polmod >= 0.) and (frequency_polmod != self.frequency_polmod)
        # amp_changed = (amp >= 0.) and (amp != self.amplitude)
        phase_mode_changed = bool(phase_mode) != (self.phase_mode == 1)
        phase_origin_changed = t_phase_origin_mu >= 0. and (t_phase_origin_mu != self.t_phase_origin_mu)
        global_phase_changed = global_phase >= 0. and (global_phase != self.global_phase)
        relative_phase_changed = relative_phase >= 0. and (relative_phase != self.relative_phase)

        # Update stored values
        if freq_changed:
            self.frequency_polmod = frequency_polmod if frequency_polmod >= 0. else self.frequency_polmod
        # if amp_changed:
            # self.amplitude = amp_raman if amp_raman >= 0. else self.amplitude
        if phase_mode_changed:
            self.phase_mode = phase_mode
        if phase_origin_changed:
            self.t_phase_origin_mu = t_phase_origin_mu if t_phase_origin_mu > 0 else self.t_phase_origin_mu
        if global_phase_changed:
            self.global_phase = global_phase if global_phase >= 0. else self.global_phase
        if relative_phase_changed:
            self.relative_phase = relative_phase if relative_phase >= 0. else self.relative_phase

        if init:
            freq_changed = True
            # amp_changed = True
            phase_mode_changed = True
            phase_origin_changed = True
            global_phase_changed = True
            relative_phase_changed = True
        
        if phase_mode_changed:
            self.dds_polmod_h.set_phase_mode(self.phase_mode)
            self.dds_polmod_v.set_phase_mode(self.phase_mode)

        # if freq_changed or amp_changed or phase_origin_changed or global_phase_changed or relative_phase_changed:
        if freq_changed or phase_origin_changed or global_phase_changed or relative_phase_changed:
            self._frequency_array = self.polmod_frequency_to_ao_frequency(self.frequency_polmod)

            self.dds_polmod_h.set_dds(self._frequency_array[0],
                                # self.amplitude,
                                t_phase_origin_mu=self.t_phase_origin_mu,
                                phase=self.global_phase)
            
            self.dds_polmod_v.set_dds(self._frequency_array[1],
                                # self.amplitude,
                                t_phase_origin_mu=self.t_phase_origin_mu,
                                phase=self.global_phase+self.relative_phase)
            
        if self.frequency_polmod > 0.:
            self.dds_polmod_h.on()
            self.dds_polmod_v.on()
        else:
            self.dds_polmod_h.on()
            self.dds_polmod_v.off()
            
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
        self.set_polmod(phase_mode=1,
                 global_phase=global_phase,
                 relative_phase=relative_phase,
                 t_phase_origin_mu=t_phase_origin_mu)
        at_mu(t)
        if not pretrigger:
            delay(5.e-6)

    @kernel
    def init(self,frequency_polmod=0.,
            global_phase=0.,relative_phase=0.,
            t_phase_origin_mu=np.int64(-1),
            phase_mode=1):
        if t_phase_origin_mu < 0:
            t_phase_origin_mu = now_mu()
        self.set_polmod(frequency_polmod,
                        global_phase,relative_phase,
                        t_phase_origin_mu=t_phase_origin_mu,
                        phase_mode=phase_mode,
                        init=True)