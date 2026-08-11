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
            TFloat: the required beat lock reference frequency in Hz.
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
    def _integrated_pulse_v(self, t, dark, reset) -> TFloat:
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
        self.integrator.begin_integrate(reset=reset)
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
                                 dark=False, reset=False):
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
        data_container.put_data(self._integrated_pulse_v(t, dark, reset), idx)

    @kernel
    def measure_integrated_v(self, t, dark=False, reset=False) -> TFloat:
        """Integrated APD voltage returned directly, with slack re-armed.

        Same measurement as integrated_imaging_pulse, but returns the value
        instead of storing it, and re-arms t_apd_slack afterwards so the
        blocking sampler readback cannot starve the next call's pretrigger.
        That trailing delay is why this is NOT the same as
        integrated_imaging_pulse -- adding it there would shift the timeline of
        every existing call site.
        """
        v = self._integrated_pulse_v(t, dark, reset)
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
    def stabilize_power(self, v_target) -> TTuple([TInt32, TFloat]):
        """Servo the imaging PID setpoint until the integrated APD signal
        (light minus dark) equals v_target.

        check -> single multiplicative ratio jump -> P/I feedback.

        The dark/background is measured ONCE on entry and reused for every
        iteration, which halves the imaging light exposure relative to measuring
        it per iteration. All tuning is read from ExptParams (self.p.*).

        Costs roughly N_iter * (t_apd_imaging_check + t_apd_pid_settle) of
        imaging light -- this is a calibration routine, run it before atoms are
        present.

        Preconditions: dds_pid is on (see init()) and dds.stash_defaults() has
        run, since set_power reads _amplitude_default. This method deliberately
        does NOT turn the beam on -- that is the caller's job.

        Args:
            v_target (float): Target integrated APD voltage (light minus dark).

        Returns:
            (n_iter, frac_err): iterations used and the final fractional error.
            Convergence means frac_err < p.frac_err_threshold_imaging_pid.
        """
        if v_target <= 0.:
            raise ValueError("stabilize_power needs a strictly positive target integrated APD voltage.")

        # hoist -- fewer attribute loads in the loop, and each type is pinned once
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
        v_dark = self.measure_integrated_v(t_check, True, True)

        # a multiplicative jump can never move off a zero setpoint, and
        # DDS.off() zeroes dac_pid.v -- seed it so the ratio step has traction
        if self.dac_pid.v <= 0.:
            self.set_power(self.p.v_pid_imaging_seed)
            delay(t_settle)

        ### phase 1: check
        v_signal = self.measure_integrated_v(t_check, False, False) - v_dark

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

        while n_iter < N_max:
            v_signal = self.measure_integrated_v(t_check, False, False) - v_dark
            err = v_signal - v_target
            frac_err = abs(err / v_target)
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
            # one aprint AFTER the loop, never inside it -- it is an async RPC,
            # so it does not stall the timeline, but marshalling the message
            # costs kernel CPU time and therefore eats slack
            if railed:
                aprint("stabilize_power: imaging PID setpoint pinned at rail",
                       self.dac_pid.v, "V -- cannot reach target. frac err =", frac_err)
            else:
                aprint("stabilize_power: no convergence in", n_iter,
                       "iterations. frac err =", frac_err)
            self._core.break_realtime()

        return n_iter, frac_err