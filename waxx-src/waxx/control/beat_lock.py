import numpy as np

from artiq.experiment import portable, kernel, rpc, \
                                TFloat, TInt32, TTuple, TArray
from artiq.language.core import delay, now_mu, at_mu
from artiq.coredevice.core import Core

from waxx.control.artiq import DDS, TTL_OUT, DAC_CH, DummyCore
from waxx.control.integrator import Integrator
from waxx.config.expt_params import ExptParams
from waxx.config.sampler_id import sampler_frame
from waxx.util.artiq.async_print import aprint

dv = -10.e9
di = -10000

FREQUENCY_GS_HFS = 461.7 * 1.e6

T_PID_RESET_PULSE = 1.e-6

# Sampler conversion is latched this many mu after CNV; the integrator clear is
# scheduled here so it never corrupts the in-flight conversion.
T_APD_SAMPLER_CONV_MU = 80

# Probe buffer for the imaging power map. Kernels cannot allocate, so the
# arrays are sized once here and only the first N_points slots are used.
N_MAX_POWER_MAP_POINTS = 32

POLMOD_H_IDX = 0
POLMOD_V_IDX = 1

class BeatLockImaging():
    def __init__(self,
                 dds_sw=DDS,
                 dds_beatref=DDS,
                 pid_override_ttl=TTL_OUT,
                 N_beatref_mult=di,
                 beatref_sign=di,
                 frequency_minimum_beat=dv,
                 expt_params=ExptParams()):
        
        self.dds_sw = dds_sw
        self.dds_beatref = dds_beatref

        self.ttl_pid_manual_override = pid_override_ttl

        self.params = expt_params
        self.p = self.params

        if N_beatref_mult == di:
            self._N_beatref_mult = self.p.N_offset_lock_reference_multiplier
        if beatref_sign == di:
            # +1 for lock greater frequency than reference (Gain switch "+"), vice versa ("-")
            self._beat_sign = self.p.beatlock_sign
        if frequency_minimum_beat == dv:
            self._frequency_minimum_beat = self.p.frequency_minimum_offset_beatlock

        self.phase_mode = 1

    @kernel
    def init(self):
        self.dds_beatref.on()
        self.dds_sw.dac_ch = -1 # disconnect the logic for dac control of the dds
        self.dds_sw.update_dac_bool()
        self.ttl_pid_manual_override.on()

    @kernel
    def set_power(self, power_control_parameter=dv, reset_pid=False):
        self.dds_sw.set_dds(amplitude=power_control_parameter)

    @kernel
    def pulse(self,t):
        """Pulses the imaging beam.

        Args:
            t (float): The time of the imaging pulse.
        """        
        self.dds_sw.on()
        delay(t)
        self.dds_sw.off()

    @kernel
    def on(self):
        self.dds_sw.on()
    
    @kernel
    def off(self):
        self.dds_sw.off()

    @kernel
    def pre_set_detuning(self):
        pass

    @portable(flags={"fast-math"})
    def get_ao_shift(self) -> TFloat:
        ao_shift = self.dds_sw.frequency * self.dds_sw.aom_order * 2
        return ao_shift
    
    @kernel(flags={"fast-math"})
    def set_imaging_detuning(self, frequency_detuned):
        '''
        Sets the detuning of the beat-locked imaging laser (in Hz).

        Imaging detuning is controlled by two things -- the Vescent offset lock
        and a double pass (-1 order).

        The offset lock has a multiplier, N, that determines the offset lock
        frequency relative to the lock point of the D2 laser locked at the
        crossover feature for the D2 transition. Offset = N * reference freqeuency.
        
        The reference frequency is provided by a DDS channel (dds_frame.beatlock_ref).
        '''
        self.pre_set_detuning()

        f_beatlock_ref = self.imaging_detuning_to_beat_ref(frequency_detuned=frequency_detuned)

        f_offset = f_beatlock_ref * self._N_beatref_mult
        if f_offset < self._frequency_minimum_beat:
            raise ValueError("The beat lock is unhappy at a lock point below the minimum offset.")
        
        if f_beatlock_ref < 0.:
            raise ValueError("You tried to set the DDS to a negative frequency!")
        
        self.dds_beatref.set_dds(frequency=f_beatlock_ref)
        self.dds_beatref.on()
    
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
            TFloat: the required beat lock reference frequency in Hz.dsa
        """        

        f_shift_resonance = FREQUENCY_GS_HFS / 2
        f_ao_shift = self.get_ao_shift()

        f_offset = 1/self._beat_sign * (frequency_detuned - f_ao_shift - f_shift_resonance)

        f_beatlock_ref = f_offset / self._N_beatref_mult

        if f_offset < self._frequency_minimum_beat:
            aprint("The requested detuning results in an offset less than the minimum beat note frequency for the lock.")
        if f_beatlock_ref < 0.:
            aprint("The requested detuning would require a negative reference frequency. You'll need to flip the beat lock sign to reach this detuning.")
        if f_beatlock_ref > 400.e6:
            aprint("Invalid beatlock reference frequency for requested detuning (>400 MHz). Must be less than 400 MHz for ARTIQ DDS. Consider changing the beat lock reference multiplier.")

        return f_beatlock_ref

class PolModBeatLock(BeatLockImaging):
    def __init__(self,
                 dds_sw=DDS,
                 dds_polmod_v=DDS,
                 dds_polmod_h=DDS,
                 dds_beatref=DDS,
                 pid_override_ttl=TTL_OUT,
                 N_beatref_mult=di,
                 beatref_sign=di,
                 frequency_minimum_beat=dv,
                 expt_params=ExptParams()):
        super().__init__(dds_sw=dds_sw,
            dds_beatref=dds_beatref,
            N_beatref_mult=N_beatref_mult,
            pid_override_ttl=pid_override_ttl,
            beatref_sign=beatref_sign,
            frequency_minimum_beat=frequency_minimum_beat,
            expt_params=expt_params)
    
        self.dds_polmod_v = dds_polmod_v
        self.dds_polmod_h = dds_polmod_h

        self.frequency_polmod = 0.
        self.global_phase = 0.
        self.relative_phase = 0.
        self.t_phase_origin_mu = np.int64(0)

        self.phase_mode = 0  # 0: independent, 1: synchronized

        self._frequency_center_dds = 0.
        self._frequency_array = np.array([0.,0.])
        self._init()

    @kernel
    def pre_set_detuning(self):
        self.set_polmod(self.frequency_polmod)

    def _init(self):
        self._frequency_center_dds = (self.dds_polmod_h.frequency + self.dds_polmod_v.frequency)/2
        if abs(self._frequency_center_dds - self.dds_polmod_h.frequency) != abs(self._frequency_center_dds - self.dds_polmod_v.frequency):
            raise ValueError("The - and + DDS frequencies should be equidistant from their mean for optimal efficiency.")
        self._frequency_array_defaults = [self.dds_polmod_h.frequency, self.dds_polmod_v.frequency]

    @kernel(flags={"fast_math"})
    def polmod_frequency_to_ao_frequency(self, frequency_polmod)  -> TArray(TFloat):

        if frequency_polmod > 0.:
            order_p = self.dds_polmod_h.aom_order
            order_m = self.dds_polmod_v.aom_order

            frequency_polmod = frequency_polmod # bc the atoms respond same to polarization rotated by pi

            df = frequency_polmod / 4

            if order_p * order_m == -1:
                self._frequency_array[POLMOD_H_IDX] = df
                self._frequency_array[POLMOD_V_IDX] = df
            else:
                self._frequency_array[POLMOD_H_IDX] = self._frequency_center_dds + df
                self._frequency_array[POLMOD_V_IDX] = self._frequency_center_dds - df
        else:
            self._frequency_array[POLMOD_H_IDX] = self._frequency_array_defaults[0]
            self._frequency_array[POLMOD_V_IDX] = 0.

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

            self.dds_polmod_h.set_dds(self._frequency_array[POLMOD_H_IDX],
                                t_phase_origin_mu=self.t_phase_origin_mu,
                                phase=self.global_phase)
            
            self.dds_polmod_v.set_dds(self._frequency_array[POLMOD_V_IDX],
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

    @portable(flags={"fast-math"})
    def get_ao_shift(self) -> TFloat:
        f_sw_ao_shift = self.dds_sw.frequency * self.dds_sw.aom_order * 2
        if self.frequency_polmod > 0.:
            f_polmod_ao_shift = self.dds_polmod_v.aom_order * self.dds_polmod_v.frequency * 2 \
                                + self.dds_polmod_h.aom_order * self.dds_polmod_h.frequency * 2
        else:
            f_polmod_ao_shift = self.dds_polmod_h.aom_order * self.dds_polmod_h.frequency * 2
        f_ao_shift = f_sw_ao_shift + f_polmod_ao_shift
        return f_ao_shift

    @kernel
    def init(self,frequency_polmod=0.,
            global_phase=0.,relative_phase=0.,
            t_phase_origin_mu=np.int64(-1),
            phase_mode=1,
            v_pd_imaging=0.):
        if t_phase_origin_mu < 0:
            t_phase_origin_mu = now_mu()
        self.dds_beatref.on()
        self.set_polmod(frequency_polmod,
                        global_phase,relative_phase,
                        t_phase_origin_mu=t_phase_origin_mu,
                        phase_mode=phase_mode,
                        init=True)
        self.dds_polmod_h._stash_defaults()
        self.dds_polmod_v._stash_defaults()

class BeatLockImagingPID(BeatLockImaging):
    def __init__(self,
                 dds_sw=DDS,
                 dds_pid=DDS,
                 pid_int_clear_ttl=TTL_OUT,
                 pid_override_ttl=TTL_OUT,
                 dac_pid_setpoint=DAC_CH,
                 dds_beatref=DDS,
                 integrator=Integrator,
                 sampler=sampler_frame,
                 core: Core = DummyCore(),
                 N_beatref_mult=di,
                 beatref_sign=di,
                 frequency_minimum_beat=dv,
                 expt_params=ExptParams()):

        self.dds_pid = dds_pid

        self.dac_pid = dac_pid_setpoint
        self.ttl_pid_int_clear = pid_int_clear_ttl

        # NOTE: the placeholders for integrator/sampler are the CLASSES, not
        # instances -- Integrator.__init__ raises unless sampler_ch is a
        # Sampler_Last_CH instance, and default args evaluate at import time.
        # Matches the dds_sw=DDS / dac_pid_setpoint=DAC_CH convention above.
        # These are the same objects the experiment holds as self.integrator /
        # self.sampler / self.core, so ARTIQ embeds each exactly once.
        self.integrator = integrator
        # the integrated measurement reads through integrator.sampler_ch (which
        # IS sampler.apd_integrator); this reference is here so other channels
        # of the same sampler are reachable from the imaging object.
        self.sampler = sampler
        self._core = core

        # Cached power map, fit by calibrate_power_map:
        #
        #     rate == v_signal / t_integration
        #          == _fit_c * v_pd**2 + _fit_a * v_pd + _fit_b
        #
        # The fit is against the RATE, not the raw integrated voltage, because
        # v_signal scales with the integration window while v/t does not (the
        # same invariant the light shift calibration is stored in). One map
        # therefore serves every commanded shift and every window.
        #
        # _fit_a <= 0. means "no usable map" -- it is the uncalibrated state and
        # also what a fit over a blocked beam produces, so one test covers both.
        self._fit_a = 0.        # (V/s) per volt of PID setpoint
        self._fit_b = 0.        # V/s extrapolated to v_pd = 0
        # Curvature. The map is compressive, so a straight line through a wide
        # probe band reads a chord slope -- too shallow near the top, too steep
        # near the bottom -- and that mismatch is most of why the open-loop jump
        # needed corrections at all. 0. collapses every formula downstream to
        # the affine case, so an unearned or ill-conditioned quadratic is safe.
        self._fit_c = 0.        # (V/s) per volt^2
        self._fit_rms = 0.      # rms fit residual, V/s
        self._fit_shots = np.int32(0)   # acquisitions since the map was last fit
        self._probe_v_pd = np.zeros(N_MAX_POWER_MAP_POINTS)
        self._probe_rate = np.zeros(N_MAX_POWER_MAP_POINTS)

        super().__init__(dds_sw=dds_sw,
                dds_beatref=dds_beatref,
                N_beatref_mult=N_beatref_mult,
                pid_override_ttl=pid_override_ttl,
                beatref_sign=beatref_sign,
                frequency_minimum_beat=frequency_minimum_beat,
                expt_params=expt_params)

    @kernel
    def init(self):
        self.dds_beatref.on()

        self.dds_pid.set_dds(init=True)
        self.dds_sw.set_dds(init=True)

        self.dds_pid.on()
        self.ttl_pid_manual_override.off()
        self.set_imaging_detuning(0.)
        self.set_power(0.25, reset_pid=True)

    @portable(flags={"fast-math"})
    def get_ao_shift(self) -> TFloat:
        ao_shift = (self.dds_sw.frequency * self.dds_sw.aom_order \
                    + self.dds_pid.aom_order * self.dds_pid.frequency) * 2
        return ao_shift

    @kernel
    def set_power(self, power_control_parameter=dv, reset_pid=False):
        self.dds_pid.set_dds(amplitude=self.dds_pid._amplitude_default,
                             v_pd=power_control_parameter)
        if reset_pid:
            self.ttl_pid_int_clear.pulse(T_PID_RESET_PULSE)
            delay(-T_PID_RESET_PULSE)

    @kernel
    def _integrated_pulse_v(self, t, dark) -> TFloat:
        """Core integrated-APD measurement: one imaging pulse of length t.

        Adds NO trailing slack, so the timeline is exactly what
        integrated_imaging_pulse has always produced. Use measure_integrated_v
        instead if you need to call this back-to-back.

        TIMELINE: Integrator.begin_integrate pretriggers 600 mu into the past
        (2100 mu if reset=True), so the caller MUST already have slack on the
        timeline. The sampler readback is a BLOCKING RTIO input that stalls the
        kernel until real time catches up, so on return slack is ~zero.

        Leaves the integrator held in clear, the precondition for the next call
        with reset=False.
        """
        self.integrator.begin_integrate(reset=False)
        if dark:
            delay(t)
        else:
            self.pulse(t)
        self.integrator.stop_and_settle()
        t0 = now_mu()
        # start the clear only after the integrator voltage is already latched
        # in the sampler
        at_mu(t0 + T_APD_SAMPLER_CONV_MU)
        self.integrator.clear(t=0)
        at_mu(t0)
        v = self.integrator.sample()
        return v

    @kernel
    def integrated_imaging_pulse(self, data_container, t, idx=0,
                                 dark=False):
        """Pulse the imaging beam and store the integrated APD voltage.

        Args:
            data_container: DataContainer to store the reading in.
            t (float): Length of the imaging pulse.
            idx (int, optional): Slot in the container to write. Defaults to 0.
            dark (bool, optional): If True, take a background reading -- the
                integration window runs with no imaging pulse. Defaults to False.
            reset (bool, optional): If True, fully reset the integrator before
                the window (costs 1500 mu more pretrigger). Defaults to False.

        Timeline is unchanged from when this lived on kexp.base.control.Control;
        kexp keeps a thin delegating wrapper so self.integrated_imaging_pulse
        still works from an experiment.
        """
        data_container.put_data(self._integrated_pulse_v(t, dark), idx)

    @kernel
    def measure_integrated_v(self, t, dark=False) -> TFloat:
        """Integrated APD voltage returned directly, with slack re-armed.

        Same measurement as integrated_imaging_pulse, but returns the value
        instead of storing it, and re-arms t_apd_slack afterwards so the
        blocking sampler readback cannot starve the next call's pretrigger.
        That trailing delay is why this is NOT the same as
        integrated_imaging_pulse -- adding it there would shift the timeline of
        every existing call site.
        """
        v = self._integrated_pulse_v(t, dark)
        delay(self.p.t_apd_slack)
        return v

    @portable
    def _clamp_v_pid(self, v) -> TFloat:
        """Clamp a PID setpoint into a strictly POSITIVE band.

        Lower rail: DDS.set_dds treats v_pd < 0 as "no change", so a negative
        setpoint is a SILENT no-op and the feedback loop would spin uselessly.

        Upper rail: DAC_CH.set ZEROES the channel if v > max_v -- it does not
        clamp -- which would turn the beam fully off AND poison dac_pid.v for the
        next ratio jump. Its error path is an async RPC, so the kernel never sees
        it. Clamping below max_v guarantees that branch can never be reached.
        """
        v_min = self.p.v_pid_imaging_min
        v_max = self.p.v_pid_imaging_max
        if v_max > self.dac_pid.max_v:
            v_max = self.dac_pid.max_v
        if v < v_min:
            v = v_min
        elif v > v_max:
            v = v_max
        return v

    @kernel
    def stabilize_power(self, v_target, t_integration=dv) -> TTuple([TInt32, TFloat]):
        """Servo the imaging PID setpoint until the integrated APD signal
        (light minus dark) equals v_target.

        check -> single multiplicative ratio jump -> P/I feedback.

        The dark/background is measured ONCE on entry and reused for every
        iteration, which halves the imaging light exposure relative to measuring
        it per iteration. All tuning is read from ExptParams (self.p.*).

        Costs roughly N_iter * (t_integration + t_apd_pid_settle) of
        imaging light -- this is a calibration routine, run it before atoms are
        present.

        Preconditions: dds_pid is on (see init()) and dds.stash_defaults() has
        run, since set_power reads _amplitude_default. This method deliberately
        does NOT turn the beam on -- that is the caller's job.

        Args:
            v_target (float): Target integrated APD voltage (light minus dark).
            t_integration (float): Integration window used for every check.
                Omit (or pass <= 0) to use p.t_apd_imaging_check. v_target scales
                with this window, so the two must have been computed together --
                see stabilize_lightshift, which sizes both at once.

        Returns:
            (n_iter, frac_err): iterations used and the fractional error of the
            setpoint left on the DAC. Convergence means frac_err <
            p.frac_err_threshold_imaging_pid. On failure to converge the servo
            is returned to its best-measured setpoint rather than left wherever
            the last iteration landed, so frac_err is the smallest error seen.
        """
        if v_target <= 0.:
            raise ValueError("stabilize_power needs a strictly positive target integrated APD voltage.")

        # hoist -- fewer attribute loads in the loop, and each type is pinned once
        t_check = t_integration
        if t_check <= 0.:
            t_check = self.p.t_apd_imaging_check
        t_settle = self.p.t_apd_pid_settle
        gain_p = self.p.gain_p_imaging_pid
        gain_i = self.p.gain_i_imaging_pid
        thresh = self.p.frac_err_threshold_imaging_pid
        N_max = self.p.N_max_iter_imaging_pid
        v_min = self.p.v_pid_imaging_min
        v_max = self.p.v_pid_imaging_max
        if v_max > self.dac_pid.max_v:
            v_max = self.dac_pid.max_v

        # the caller's slack is unknown, and the first measurement pretriggers
        # 2100 mu into the past
        self._core.break_realtime()

        # dark/background: measured ONCE, reused for every iteration. reset=True
        # self-arms the integrator regardless of prior state.
        v_dark = self.measure_integrated_v(t_check, dark=True)

        # a multiplicative jump can never move off a zero setpoint, and
        # DDS.off() zeroes dac_pid.v -- seed it so the ratio step has traction
        if self.dac_pid.v <= 0.:
            self.set_power(self.p.v_pid_imaging_seed)
            delay(t_settle)

        ### phase 1: check
        v_signal = self.measure_integrated_v(t_check) - v_dark

        ### phase 2: single multiplicative ratio jump
        # v_signal <= 0 means no measurable light (blocked beam, or dark drift).
        # Skip rather than dividing by zero or by a negative -- a negative ratio
        # would flip the setpoint sign into the silent-no-op region. The P/I loop
        # below still pushes in the right direction.
        if v_signal > 0.:
            v_new = self._clamp_v_pid(self.dac_pid.v * (v_target / v_signal))
            self.set_power(v_new)
            delay(t_settle)

        ### phase 3: P/I feedback
        n_iter = 0
        frac_err = 1.
        err_integral = 0.
        saturated = False
        railed = False

        # Best-so-far setpoint. The P/I loop is not monotonic -- it overshoots,
        # and on a run that never reaches thresh the setpoint the last iteration
        # happened to write is arbitrary. Track the setpoint that actually
        # measured the smallest error so it can be restored below.
        v_best = self.dac_pid.v
        frac_err_best = -1.  # sentinel: nothing measured yet

        while n_iter < N_max:
            v_signal = self.measure_integrated_v(t_check) - v_dark
            err = v_signal - v_target
            frac_err = abs(err / v_target)

            # v_signal was measured with the setpoint the DAC is holding NOW --
            # v_new below has not been written yet -- so dac_pid.v is the
            # setpoint that produced frac_err, not one derived from it.
            if frac_err_best < 0. or frac_err < frac_err_best:
                frac_err_best = frac_err
                v_best = self.dac_pid.v

            if frac_err < thresh:
                break

            v_new = self.dac_pid.v + gain_p * err + gain_i * (err_integral + err)

            if v_new > v_max:
                v_new = v_max
                saturated = True
            elif v_new < v_min:
                v_new = v_min
                saturated = True
            else:
                saturated = False
                err_integral += err  # anti-windup: only integrate when unsaturated

            n_iter += 1

            if saturated and (v_new == self.dac_pid.v):
                # already pinned at this rail and the correction only pushes
                # further into it -- more iterations cannot help. Abort instead
                # of burning the rest of N_max_iter of imaging light.
                railed = True
                break

            self.set_power(v_new)
            # servo settling, and slack for the next begin_integrate pretrigger
            delay(t_settle)

        if frac_err >= thresh:
            # Never converged. Rewind to the closest approach instead of
            # leaving the caller at whatever the last correction wrote -- with a
            # loop that overshoots, the final point can be worse than several it
            # already passed through.
            if frac_err_best >= 0.:
                if v_best != self.dac_pid.v:
                    self.set_power(v_best)
                    delay(t_settle)
                frac_err = frac_err_best

            # one aprint AFTER the loop, never inside it -- it is an async RPC,
            # so it does not stall the timeline, but marshalling the message
            # costs kernel CPU time and therefore eats slack
            if railed:
                aprint("stabilize_power: imaging PID setpoint pinned at rail",
                       v_best, "V -- cannot reach target. frac err =", frac_err)
            else:
                aprint("stabilize_power: no convergence in", n_iter,
                       "iterations. best setpoint", v_best,
                       "V, frac err =", frac_err)
            self._core.break_realtime()

        return n_iter, frac_err

    @kernel
    def _measure_v_mean(self, t, N_avg, dark) -> TFloat:
        """Mean of N_avg back-to-back integrated APD readings.

        Averaging is the whole point of the fit-based path: stabilize_power
        accepts on a SINGLE noisy sample, so it preferentially stops on the
        shots where noise happened to flatter it and its reported frac_err is
        biased low. Deciding on a mean of N shrinks that by sqrt(N).

        measure_integrated_v re-arms t_apd_slack itself, so these chain safely.
        """
        n = N_avg
        if n < 1:
            n = 1
        v_sum = 0.
        for i in range(n):
            v_sum += self.measure_integrated_v(t, dark)
        return v_sum / float(n)

    @kernel
    def _settle_after(self, t_start_mu, t_settle):
        """Hold the timeline until t_settle has elapsed since t_start_mu.

        Lets work that does not depend on the PID setpoint run DURING the
        settling time rather than before it -- the settle is the single largest
        cost in every path here, and it is otherwise dead time.

        Never rewinds: if the interleaved work outlasted t_settle then the loop
        is already settled and this is a no-op, which is why it is a comparison
        and not a bare at_mu.
        """
        t_end_mu = t_start_mu + self._core.seconds_to_mu(t_settle)
        if now_mu() < t_end_mu:
            at_mu(t_end_mu)

    @kernel
    def _verify_rate(self, t, v_dark, rate_target) -> TTuple([TFloat, TFloat, TInt32]):
        """Measure the signal rate, sampling until the accept/reject decision is
        statistically unambiguous instead of for a fixed number of shots.

        A fixed N_avg has to be chosen for the worst case and is then wasted on
        every shot that landed cleanly -- and, worse, it silently fails when the
        noise is larger than the threshold it is being compared against: the
        comparison becomes a coin flip, corrections chase noise, and the caller
        burns its whole correction budget on every shot forever. Sampling until
        the answer is unambiguous fixes both ends. A jump that landed well
        proves it in two or three samples; a marginal one keeps averaging.

        The three exits, with band = k_sigma * (standard error of the mean),
        expressed as a fraction of rate_target:

            frac + band < thresh        confidently inside  -> accept
            frac - band > thresh        confidently outside -> correct
            n == N_avg_imaging_verify_max                   -> caller decides

        On that last exit the caller compares frac against band: an offset it
        cannot resolve from zero is not worth correcting, however large the
        threshold says it is.

        Both tests are done on SQUARED quantities to keep a square root out of
        the per-sample inner loop. band is therefore returned squared as well --
        the caller's noise-limited test is frac_err*frac_err < band_sq.

        Returns:
            (rate_mean, band_sq, n): mean rate in V/s, the squared confidence
            band as a fraction of rate_target, and the samples it took.
        """
        thresh = self.p.frac_err_threshold_imaging_pid
        n_min = self.p.N_avg_imaging_verify_min
        n_max = self.p.N_avg_imaging_verify_max
        k_sigma = self.p.k_sigma_imaging_verify
        if n_min < 2:
            # one sample has no variance to estimate, so no decision is possible
            n_min = 2
        if n_max < n_min:
            n_max = n_min

        # Accumulate deviations from the first sample rather than the raw rates.
        # The variance is ~1e-6 of the mean squared here, so the textbook
        # sum(x^2) - sum(x)^2/n differences two nearly equal large numbers and
        # throws away most of the precision in the answer; differencing first
        # costs one subtraction and removes the hazard entirely.
        rate0 = 0.
        s1 = 0.
        s2 = 0.
        n = 0
        rate_mean = 0.
        band_sq = 0.

        while True:
            rate = (self.measure_integrated_v(t, False) - v_dark) / t
            if n == 0:
                rate0 = rate
            d = rate - rate0
            s1 += d
            s2 += d * d
            n += 1
            rate_mean = rate0 + s1 / float(n)

            if n < n_min:
                continue

            var = (s2 - s1 * s1 / float(n)) / float(n - 1)
            if var < 0.:
                # only reachable through rounding on a set of identical samples
                var = 0.
            band_sq = k_sigma * k_sigma * var / float(n) / (rate_target * rate_target)

            frac = rate_mean / rate_target - 1.
            if frac < 0.:
                frac = -frac

            if frac < thresh:
                margin = thresh - frac
                if band_sq < margin * margin:
                    break               # confidently inside
            else:
                margin = frac - thresh
                if band_sq < margin * margin:
                    break               # confidently outside
            if n >= n_max:
                break

        return rate_mean, band_sq, n

    @rpc
    def _fit_power_map(self, v_pd, rate, N, quadratic) -> TArray(TFloat):
        """Host-side least-squares fit of the probed power map over the first N
        points. Returns [a, b, rms_residual, rms_residual/mean_rate, c] for

            rate == c*v_pd**2 + a*v_pd + b

        with c == 0. whenever the quadratic term is not earned, so the caller
        never has to ask which form it got back.

        SYNCHRONOUS rpc (no async flag): the kernel needs the coefficients back
        before it can invert them, so it blocks here and the caller must
        break_realtime afterwards.

        The full preallocated buffers are passed and sliced here rather than
        sliced in the kernel -- they are 32 floats, so marshalling all of them
        costs nothing and it keeps the kernel side free of array views.

        The linear fit is AFFINE, not a pure ratio. stabilize_power's phase-2
        jump assumes v_signal is proportional to v_pd with no offset; b is
        exactly the term that assumption throws away, and it is not obviously
        zero (analog PID offset, AOM leakage through the dark subtraction).
        """
        x = np.asarray(v_pd, dtype=float)[:N]
        y = np.asarray(rate, dtype=float)[:N]

        A = np.vstack([x, np.ones_like(x)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a = float(coeffs[0])
        b = float(coeffs[1])
        c = 0.
        resid = y - (a * x + b)

        # The quadratic spends a degree of freedom, so it has to earn one back:
        # fit it only with points to spare, keep it only if chi2 PER DOF
        # improves, and reject it if its turning point lands inside the probed
        # band. That last test is the important one -- a c fit to noise puts a
        # spurious maximum somewhere in the operating range, and v_pd_for_rate
        # would then invert onto the wrong branch of a parabola that does not
        # exist.
        if quadratic and N >= 5:
            Aq = np.vstack([x**2, x, np.ones_like(x)]).T
            cq, _, _, _ = np.linalg.lstsq(Aq, y, rcond=None)
            resid_q = y - (cq[0] * x**2 + cq[1] * x + cq[2])
            better = np.sum(resid_q**2) / (N - 3) < np.sum(resid**2) / (N - 2)
            if cq[0] != 0.:
                v_turn = -cq[1] / (2. * cq[0])
                monotonic = not (x.min() < v_turn < x.max())
            else:
                monotonic = True
            if better and monotonic:
                c = float(cq[0])
                a = float(cq[1])
                b = float(cq[2])
                resid = resid_q

        rms = float(np.sqrt(np.mean(resid**2)))
        y_mean = float(np.mean(y))
        rms_frac = rms / y_mean if y_mean > 0. else 0.
        return np.array([a, b, rms, rms_frac, c])

    @kernel
    def calibrate_power_map(self, t_probe=dv, v_pd_lo=dv, v_pd_hi=dv,
                            N_points=di) -> TFloat:
        """Probe the imaging power map and cache an affine fit of it.

        Measures rate = (v_light - v_dark)/t at N_points settings of the PID
        setpoint, fits rate = a*v_pd + b on the host, and caches (a, b). After
        this, acquire_rate reaches any target in one open-loop jump instead of
        iterating a servo into it.

        Why this beats the servo at equal or lower cost: every measurement
        contributes to the slope estimate instead of being discarded through an
        integrator, the probe points are spread deliberately (a converging P/I
        loop clusters its late points, which is the worst possible leverage for
        estimating a slope), and the resulting jump is open-loop, so acceptance
        is not a race between a noisy sample and a threshold.

        Probe order is 0, N-1, 1, N-2, ... rather than a monotonic sweep. A
        monotonic sweep makes setpoint and elapsed time collinear, so any slow
        drift in imaging power or APD gain is absorbed straight into a -- the
        one number everything downstream depends on. Interleaving decorrelates
        them to first order.

        Costs N_points * N_avg_imaging_power_check imaging pulses plus one dark
        set. Run it with the beam unblocked and no atoms present.

        Args:
            t_probe (float): Integration window for the probe. Omit for
                p.t_apd_imaging_check. Only affects measurement noise -- the
                fitted rate is window-independent by construction -- but it must
                be short enough that the TOP probe point does not saturate the
                integrator.
            v_pd_lo, v_pd_hi (float): Probe band on the PID setpoint. Omit both
                to span p.frac_span_imaging_power_map fractionally about the
                current setpoint (or p.v_pid_imaging_seed if it is zero).
            N_points (int): Probe points. Omit for p.N_points_imaging_power_map.

        Returns:
            TFloat: rms fit residual as a fraction of the mean probed rate. This
            is the number that says whether the map is usable: it lumps together
            measurement noise and any real curvature in v_pd -> power. If it is
            not small compared to p.frac_err_threshold_imaging_pid, a single
            jump cannot land inside threshold and the corrections in
            acquire_rate are doing the actual work.
        """
        t = t_probe
        if t <= 0.:
            t = self.p.t_apd_imaging_check
        N = N_points
        if N <= 0:
            N = self.p.N_points_imaging_power_map
        if N < 3:
            raise ValueError("calibrate_power_map needs at least 3 probe points to fit a line and report a residual.")
        if N > N_MAX_POWER_MAP_POINTS:
            raise ValueError("calibrate_power_map: N_points exceeds the preallocated probe buffer.")

        v_lo = v_pd_lo
        v_hi = v_pd_hi
        if v_lo <= 0. or v_hi <= 0.:
            v_center = self.dac_pid.v
            if v_center <= 0.:
                v_center = self.p.v_pid_imaging_seed
            span = self.p.frac_span_imaging_power_map
            v_lo = self._clamp_v_pid(v_center * (1. - 0.5 * span))
            v_hi = self._clamp_v_pid(v_center * (1. + 0.5 * span))
        else:
            v_lo = self._clamp_v_pid(v_lo)
            v_hi = self._clamp_v_pid(v_hi)
        if v_hi <= v_lo:
            raise ValueError("calibrate_power_map: probe band has no width -- both ends clamped to the same rail.")

        t_settle = self.p.t_apd_pid_settle
        N_avg = self.p.N_avg_imaging_power_check
        N_avg_dark = self.p.N_avg_imaging_dark
        v_step = (v_hi - v_lo) / float(N - 1)

        # the caller's slack is unknown and the first measurement pretriggers
        # 2100 mu into the past
        self._core.break_realtime()

        # One dark set for the whole probe: it is a property of the detector,
        # not of the setpoint being probed, so it is also free to run DURING a
        # settle. The visit order below starts at i == 0, i.e. v_lo, so write
        # that setpoint first and take the dark while the loop settles onto it.
        # It is averaged harder than the probe points because it is subtracted
        # from every one of them -- its noise is common mode across the whole
        # fit, which is the one place averaging is paid for once and returned N
        # times.
        self.set_power(v_lo)
        t_first_mu = now_mu()
        v_dark = self._measure_v_mean(t, N_avg_dark, True)
        self._settle_after(t_first_mu, t_settle)

        for k in range(N):
            if k % 2 == 0:
                i = k // 2
            else:
                i = N - 1 - (k // 2)
            v_pd = v_lo + v_step * float(i)
            if k > 0:
                # k == 0 is v_lo, already written and already settled above
                self.set_power(v_pd)
                delay(t_settle)
            v_light = self._measure_v_mean(t, N_avg, False)
            self._probe_v_pd[i] = v_pd
            self._probe_rate[i] = (v_light - v_dark) / t

        coeffs = self._fit_power_map(self._probe_v_pd, self._probe_rate, N,
                                     self.p.imaging_power_map_quadratic)
        # synchronous RPC: real time ran ahead while the host fit
        self._core.break_realtime()

        self._fit_a = coeffs[0]
        self._fit_b = coeffs[1]
        self._fit_rms = coeffs[2]
        self._fit_c = coeffs[4]
        self._fit_shots = 0

        if self._fit_a <= 0.:
            # more power must give more signal; a non-positive slope means the
            # probe saw no light at all, or only noise. Zero it so the cache
            # reads as invalid rather than inverting to a nonsense setpoint.
            self._fit_a = 0.
            self._fit_c = 0.
            aprint("calibrate_power_map: fitted slope is not positive --",
                   "no usable power map. Is the imaging beam blocked?")
            self._core.break_realtime()

        return coeffs[3]

    @portable(flags={"fast-math"})
    def v_pd_for_rate(self, rate_target) -> TFloat:
        """PID setpoint the cached map says produces rate_target, in V/s of
        integrated APD signal. Clamped into the usable band.

        Inverts rate = c*v**2 + a*v + b on its RISING branch, written as

            v = 2*(rate - b) / (a + sqrt(a**2 - 4*c*(b - rate)))

        rather than the textbook (-a +- sqrt(...))/(2*c). Two reasons: this form
        has no 0/0 as c goes to zero, and at c == 0 exactly it collapses to
        (rate - b)/a, so the affine and quadratic maps share one expression and
        there is no branch to get wrong.
        """
        if self._fit_a <= 0.:
            raise ValueError("No imaging power map cached -- call calibrate_power_map first.")
        disc = self._fit_a * self._fit_a - 4. * self._fit_c * (self._fit_b - rate_target)
        if disc <= 0.:
            # Only reachable on a compressive map (c < 0) asked for a rate above
            # its peak, i.e. more light than the map says exists. Aim at the peak
            # -- the verification then reports the miss honestly and the caller
            # rails, rather than this returning a complex root as a real one.
            return self._clamp_v_pid(-self._fit_a / (2. * self._fit_c))
        return self._clamp_v_pid(2. * (rate_target - self._fit_b)
                                 / (self._fit_a + np.sqrt(disc)))

    @portable(flags={"fast-math"})
    def _slope_at(self, v_pd) -> TFloat:
        """Local d(rate)/d(v_pd) of the cached map, in (V/s) per volt.

        This, not _fit_a, is what a Newton step should divide by. On a
        compressive map _fit_a is a chord across the whole probe band, so it
        overestimates the response near the top and underestimates it near the
        bottom, and a step taken with it lands short in a direction that depends
        on where you are.
        """
        return self._fit_a + 2. * self._fit_c * v_pd

    @kernel
    def acquire_rate(self, rate_target, t_integration=dv) -> TTuple([TInt32, TFloat]):
        """Set the imaging power so the integrated APD signal rate (V/s) equals
        rate_target, by inverting the cached power map.

        invert the map -> jump -> averaged verify -> Newton corrections.

        This is the fit-based alternative to stabilize_power. The jump is
        open-loop from the cached fit, so a shot normally costs one settle and
        two measurements instead of a servo's worth of iterations, and the
        accept decision is made on a MEAN rather than on a single sample.

        The verification does not average a fixed number of times: _verify_rate
        samples until the accept/reject call is statistically unambiguous, so a
        jump that landed cleanly is confirmed in two or three samples while a
        marginal one keeps averaging. It also means a noise floor above
        p.frac_err_threshold_imaging_pid degrades into "accepted, and said so"
        rather than into burning every correction on every shot.

        Corrections, when needed, are Newton steps

            v_pd += (rate_target - rate_measured) / a_local

        against the LOCAL slope -- the map's derivative at the current setpoint,
        or the secant through the last two measured points once there are two.
        Not the P/I gains, and not the chord slope _fit_a, which on a compressive
        map is systematically wrong in a position-dependent direction. (Sanity
        check on a fit: p.gain_p_imaging_pid = -0.019 is a hand-tuned stand-in
        for -1/(a*t_check), so a fitted a should be consistent with that.)

        The map is fit lazily on first use and then reused across shots. Drift
        costs an extra correction but not accuracy, since the corrections are
        closed loop; only a map that has drifted enough for N_max corrections to
        miss triggers a re-probe, and then only once per call. Set
        p.N_shots_per_imaging_power_map_refit > 0 to also refit on a schedule.

        Args:
            rate_target (float): Target (v_light - v_dark)/t, in V/s.
            t_integration (float): Window for each verification measurement.
                Omit for p.t_apd_imaging_check. Unlike stabilize_power's window
                this does NOT have to agree with anything -- the target is a
                rate, so the window only sets measurement noise.

        Returns:
            (n_iter, frac_err): corrections applied (0 means the jump landed) and
            the fractional error of the setpoint left on the DAC. n_iter
            accumulates across BOTH attempts, so its ceiling is
            2*p.N_max_correction_imaging_power, and reaching that ceiling means
            a freshly probed map still could not land the target. As in
            stabilize_power, a call that never reaches threshold is rewound to
            its best-measured setpoint rather than left where the last
            correction put it.
        """
        if rate_target <= 0.:
            raise ValueError("acquire_rate needs a strictly positive target rate.")

        t_check = t_integration
        if t_check <= 0.:
            t_check = self.p.t_apd_imaging_check
        t_settle = self.p.t_apd_pid_settle
        thresh = self.p.frac_err_threshold_imaging_pid
        N_avg_dark = self.p.N_avg_imaging_dark
        N_corr = self.p.N_max_correction_imaging_power
        N_refit = self.p.N_shots_per_imaging_power_map_refit
        secant_max = self.p.frac_secant_slope_max

        # lazily fit on first use, so callers that never heard of the power map
        # (check_lightshift) still get the fast path
        if self._fit_a <= 0. or (N_refit > 0 and self._fit_shots >= N_refit):
            self.calibrate_power_map(t_check)

        n_iter = 0
        frac_err = 1.
        frac_err_best = -1.  # sentinel: nothing measured yet
        v_best = self.dac_pid.v
        railed = False
        noise_limited = False

        # attempt 0 uses the cached map; attempt 1 re-probes it first. Two is
        # the cap on purpose -- if a freshly measured map still cannot land the
        # target, the problem is not staleness and re-probing again would just
        # burn imaging light. With the secant correction below, attempt 1 should
        # be rare: the corrections themselves measure the local slope.
        for attempt in range(2):
            if attempt == 1:
                self.calibrate_power_map(t_check)

            self._core.break_realtime()

            # Setpoint first, dark second. A dark window is a plain delay with
            # the beam gated off (see _integrated_pulse_v), so it neither
            # depends on nor disturbs the setpoint, and running it inside the
            # settling time makes it cost nothing. If dark readings ever start
            # tracking the setpoint, electrical pickup from the slewing PID is
            # the reason and this overlap is where to look.
            self.set_power(self.v_pd_for_rate(rate_target))
            t_jump_mu = now_mu()
            v_dark = self._measure_v_mean(t_check, N_avg_dark, True)
            self._settle_after(t_jump_mu, t_settle)

            n_corr = 0
            v_prev = 0.
            rate_prev = 0.
            have_prev = False

            while True:
                rate, band_sq, n_samp = self._verify_rate(t_check, v_dark, rate_target)
                frac_err = rate / rate_target - 1.
                if frac_err < 0.:
                    frac_err = -frac_err

                # measured with the setpoint the DAC is holding NOW -- v_new
                # below has not been written yet
                if frac_err_best < 0. or frac_err < frac_err_best:
                    frac_err_best = frac_err
                    v_best = self.dac_pid.v

                if frac_err < thresh:
                    break
                # _verify_rate spent its whole sample budget and still cannot
                # resolve this offset from zero. Correcting would be chasing
                # noise, and the next verification would be no better -- this is
                # the state that otherwise burns every correction on every shot.
                if frac_err * frac_err < band_sq:
                    noise_limited = True
                    break
                if n_corr >= N_corr:
                    break

                # Local slope for the Newton step. Two measured points
                # straddling the operating point give a better derivative than
                # any fit across the whole probe band, and they are free -- the
                # corrections had to measure anyway. Guard it: a secant drawn
                # through two noise-dominated points can come out as anything,
                # including negative, so only believe it within a factor of
                # p.frac_secant_slope_max of what the map predicts.
                a_local = self._slope_at(self.dac_pid.v)
                if have_prev:
                    dv_meas = self.dac_pid.v - v_prev
                    if dv_meas != 0. and a_local > 0.:
                        a_secant = (rate - rate_prev) / dv_meas
                        if (a_secant > a_local / secant_max
                                and a_secant < a_local * secant_max):
                            a_local = a_secant
                if a_local <= 0.:
                    # curvature extrapolated past the map's turning point
                    a_local = self._fit_a

                v_prev = self.dac_pid.v
                rate_prev = rate
                have_prev = True

                v_new = self._clamp_v_pid(self.dac_pid.v + (rate_target - rate) / a_local)
                if v_new == self.dac_pid.v:
                    # the correction is pushing into a rail it is already
                    # pinned at. More steps cannot help.
                    railed = True
                    break

                self.set_power(v_new)
                delay(t_settle)
                n_corr += 1
                n_iter += 1

            if frac_err < thresh or railed or noise_limited:
                break

        if frac_err >= thresh:
            if frac_err_best >= 0.:
                if v_best != self.dac_pid.v:
                    self.set_power(v_best)
                    delay(t_settle)
                frac_err = frac_err_best

            # one aprint AFTER the loop -- async, so it does not stall the
            # timeline, but marshalling it costs kernel CPU and so eats slack
            if railed:
                aprint("acquire_rate: imaging PID setpoint pinned at rail",
                       v_best, "V -- cannot reach target. frac err =", frac_err)
            elif noise_limited:
                aprint("acquire_rate: measurement noise floor sits above",
                       "frac_err_threshold_imaging_pid -- accepted a setpoint",
                       "that cannot be told apart from target. frac err =",
                       frac_err, "-- raise N_avg_imaging_verify_max or the",
                       "integration window, or relax the threshold.")
            else:
                aprint("acquire_rate: no convergence in", n_iter,
                       "corrections. best setpoint", v_best,
                       "V, frac err =", frac_err, "-- check the map residual",
                       "from calibrate_power_map.")
            self._core.break_realtime()

        self._fit_shots += 1
        return n_iter, frac_err

    @kernel
    def acquire_power(self, v_target, t_integration=dv) -> TTuple([TInt32, TFloat]):
        """Fit-based drop-in for stabilize_power: same arguments, same returns.

        Converts the target voltage to the window-independent rate the power map
        is fit against and defers to acquire_rate, so v_target and t_integration
        must correspond exactly as they must for stabilize_power.
        """
        if v_target <= 0.:
            raise ValueError("acquire_power needs a strictly positive target integrated APD voltage.")
        t_check = t_integration
        if t_check <= 0.:
            t_check = self.p.t_apd_imaging_check
        return self.acquire_rate(v_target / t_check, t_check)

    @portable(flags={"fast-math"})
    def lightshift_to_t_integration(self, frequency_lightshift) -> TFloat:
        """Integration window that puts the target light shift at
        p.v_target_imaging_lightshift volts of integrated APD signal.

        The calibration fixes v/t at a given light shift:

            v_signal / t_integration = frequency_lightshift / slope

        so the voltage a measurement lands at is set entirely by how long you
        integrate. Solving for the window that hits a chosen voltage,

            t_integration = v_target * slope / frequency_lightshift

        which is the point of doing this at all: without it, a fixed window
        makes the measured voltage proportional to the commanded shift, so small
        shifts are read out of the bottom of the ADC range (where the
        dark-subtraction noise floor dominates the fractional error the servo is
        trying to drive below thresh) and large ones run the integrator toward
        saturation. Sizing the window instead holds the operating point fixed.

        Clamped to p.t_apd_imaging_check_max, since a long window costs imaging
        light on every one of the servo's iterations and the integrator has a
        finite hold time. A shift low enough to hit that cap simply lands below
        v_target -- correct, just noisier.

        Nothing is clamped at the short end: the caller warns instead (see
        stabilize_lightshift), because a window that short means the requested
        shift is high enough that the honest answer is to turn the imaging power
        down, not to silently measure something other than what was asked for.

        Args:
            frequency_lightshift (float): Target light shift, Hz.

        Returns:
            TFloat: Integration window, seconds.
        """
        slope = self.p.slope_imaging_lightshift_per_v_per_t
        if slope <= 0.:
            raise ValueError("slope_imaging_lightshift_per_v_per_t is unset -- run the light shift calibration notebook.")
        if frequency_lightshift <= 0.:
            raise ValueError("lightshift_to_t_integration needs a strictly positive target light shift.")

        t = self.p.v_target_imaging_lightshift * slope / frequency_lightshift
        t_max = self.p.t_apd_imaging_check_max
        if t > t_max:
            t = t_max
        return t

    @portable(flags={"fast-math"})
    def lightshift_to_v_target(self, frequency_lightshift, t_integration=100.e-6) -> TFloat:
        """Integrated APD voltage corresponding to a given imaging light shift.

        The light shift at the atoms and the APD photocurrent are both
        proportional to the imaging intensity, and the INTEGRATED voltage is
        additionally proportional to the integration window, so the quantity
        that is independent of how long you happen to integrate for is v/t:

            frequency_lightshift [Hz] = slope * (v_signal [V] / t_integration [s])

        One constant therefore covers every pulse length, which is why the
        calibration is stored as a slope in Hz per (V/s) rather than as a
        voltage. Inverting it is what lets the power servo be commanded in the
        units that matter at the atoms.

        Calibrated by
        # k-jam/analysis/artisinal/lightshift_vs_integrated_apd_voltage.ipynb,
        which eliminates amp_imaging between an imaging_apd_v_per_t_vs_amp run
        (v/t vs amp) and a check_lightshift run (Ramsey fringe phase vs amp).

        Args:
            frequency_lightshift (float): Target light shift, Hz. Signed as a
                magnitude that grows with imaging power, matching the
                calibration notebook and kexp.calibrations.imaging.
            t_integration (float): The integration window the returned voltage
                applies to. Must be the SAME window the measurement uses --
                v_signal scales with it, so a mismatch scales the target.

        Returns:
            TFloat: Target integrated APD voltage (light minus dark), in volts.
        """
        slope = self.p.slope_imaging_lightshift_per_v_per_t
        if slope <= 0.:
            # 0. is the uncalibrated default. Raising here rather than returning
            # 0./inf keeps a missing calibration from reaching stabilize_power as
            # a plausible-looking but meaningless setpoint.
            raise ValueError("slope_imaging_lightshift_per_v_per_t is unset -- run the light shift calibration notebook.")
        if frequency_lightshift <= 0.:
            raise ValueError("lightshift_to_v_target needs a strictly positive target light shift.")
        if t_integration <= 0.:
            raise ValueError("lightshift_to_v_target needs a strictly positive integration time.")
        return t_integration * frequency_lightshift / slope

    @portable(flags={"fast-math"})
    def v_to_lightshift(self, v_signal, t_integration) -> TFloat:
        """Inverse of lightshift_to_v_target: what light shift a measured
        integrated APD voltage corresponds to, in Hz.

        Use this to report what a servo actually landed on, e.g.
        v_to_lightshift(measure_integrated_v(t) - v_dark, t).
        """
        if t_integration <= 0.:
            raise ValueError("v_to_lightshift needs a strictly positive integration time.")
        return self.p.slope_imaging_lightshift_per_v_per_t * v_signal / t_integration

    @kernel
    def stabilize_lightshift(self, frequency_lightshift) -> TTuple([TInt32, TFloat, TFloat]):
        """Set the imaging PID setpoint so the imaging light shift at the atoms
        equals frequency_lightshift.

        Sizes the integration window, then dispatches: acquire_rate when
        p.imaging_power_use_fit (the default -- invert a cached map, jump, and
        accept on an averaged verification), otherwise the stabilize_power P/I
        servo. All the cost and preconditions are the chosen method's, so read
        its docstring before calling this. The two agree on their return
        contract, and switching between them changes nothing a caller sees
        except how long it takes and how accurate it is.

        The integration window is CHOSEN here rather than taken from
        p.t_apd_imaging_check, so that every commanded shift is measured at
        roughly the same integrated voltage (p.v_target_imaging_lightshift) and
        therefore at roughly the same fractional resolution. Window and target
        are computed from the one calibration slope in that order, so they
        always correspond -- computing one against a different window than the
        other converges to the wrong power silently, which is the failure this
        arrangement exists to make impossible.

        Args:
            frequency_lightshift (float): Target light shift, Hz.

        Returns:
            (n_iter, frac_err, t_integration): the first two as stabilize_power
            (or acquire_rate), plus the window it measured with. That window is
            needed to convert a readback back to Hz --
            v_to_lightshift(v_signal, t_integration) -- so it is returned rather
            than left for the caller to re-derive.
        """
        t_integration = self.lightshift_to_t_integration(frequency_lightshift)
        if t_integration < self.p.t_apd_imaging_check_min:
            # not an error: the measurement is still valid, it is just short
            # enough that the integrator's own settling and the sampler
            # conversion are no longer negligible against it. The fix is less
            # imaging power (or a smaller commanded shift), not a longer window,
            # which is why this warns instead of clamping.
            aprint("stabilize_lightshift: integration window", t_integration,
                   "s is below t_apd_imaging_check_min for a shift of",
                   frequency_lightshift, "Hz -- reduce the imaging power.")
            self._core.break_realtime()

        if self.p.imaging_power_use_fit:
            # The map is fit against v/t, and f = slope*(v/t), so the target
            # rate follows from the commanded shift ALONE -- the window never
            # enters the jump, only the noise on the verification. That is what
            # makes one cached map cover the whole range of commanded shifts
            # even though each one is measured at its own window.
            rate_target = frequency_lightshift / self.p.slope_imaging_lightshift_per_v_per_t
            n_iter, frac_err = self.acquire_rate(rate_target, t_integration)
        else:
            v_target = self.lightshift_to_v_target(frequency_lightshift, t_integration)
            n_iter, frac_err = self.stabilize_power(v_target, t_integration)
        return n_iter, frac_err, t_integration