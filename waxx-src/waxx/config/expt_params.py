import numpy as np
from waxa.config.expt_params import ExptParams as ExptParamsWaxa

class ExptParams(ExptParamsWaxa):
    def __init__(self):
        super().__init__()
        
        self.beatlock_sign = -1
        self.N_offset_lock_reference_multiplier = 8
        self.frequency_minimum_offset_beatlock = 250.e6

        ### imaging power stabilization (BeatLockImagingPID.stabilize_power)
        self.t_apd_imaging_check = 50.e-6          # default integration window per APD power check
        self.t_apd_pid_settle = 1.e-3               # settling allowed after each setpoint write
        self.t_apd_slack = 12.e-6                   # slack re-armed after each blocking sampler read
        self.N_max_iter_imaging_pid = np.int32(50)  # iteration cap (np.int32 pins TInt32)
        self.frac_err_threshold_imaging_pid = 0.005 # convergence threshold on |v_signal/v_target - 1|
        self.gain_p_imaging_pid = -0.019            # proportional gain (tuned)
        self.gain_i_imaging_pid = -0.0175           # integral gain (tuned)
        self.v_pid_imaging_min = 0.05               # lower rail; MUST be > 0 (DDS.set_dds ignores v_pd < 0)
        self.v_pid_imaging_max = 9.9                # upper rail; MUST be <= DAC_CH.max_v (9.99), above which the DAC zeroes the channel
        self.v_pid_imaging_seed = 0.5               # restart setpoint when dac_pid.v == 0 (a ratio jump cannot move off zero)

        ### imaging light shift <-> integrated APD voltage
        # (BeatLockImagingPID.lightshift_to_v_target / stabilize_lightshift)
        # Light shift and APD photocurrent are both proportional to imaging
        # intensity, and the integrated voltage is additionally proportional to
        # the integration window, so the pulse-length-independent invariant is
        # v/t:  f_lightshift [Hz] = slope * (v_signal [V] / t_integration [s]).
        # Calibrated by
        # k-jam/analysis/artisinal/lightshift_vs_integrated_apd_voltage.ipynb.
        # 0. means UNCALIBRATED -- lightshift_to_v_target raises on it rather
        # than servoing to a meaningless power.
        self.slope_imaging_lightshift_per_v_per_t = 0.  # Hz per (V/s)

        # stabilize_lightshift sizes its own integration window instead of using
        # t_apd_imaging_check: since v = (f_lightshift/slope) * t, the window is
        # the only free knob that sets where in the ADC range the measurement
        # lands. Aiming every target at the same integrated voltage keeps the
        # servo's fractional resolution constant across commanded shifts --
        # otherwise a small shift is measured down in the digitization noise and
        # a large one saturates the integrator.
        self.v_target_imaging_lightshift = 1.       # integrated voltage the window is sized to produce
        self.t_apd_imaging_check_max = 150.e-6      # hard cap; low shifts hit this and land below v_target
        self.t_apd_imaging_check_min = 10.e-6       # below this the window is used but warned about

        ### fit-based imaging power acquisition
        # (BeatLockImagingPID.calibrate_power_map / acquire_rate)
        # Instead of servoing into the target, probe the setpoint -> power map
        # once, fit it, and invert it. Every probe point contributes to the
        # slope estimate (the P/I servo throws its measurements away through an
        # integrator), the points are spread rather than clustered where the
        # servo happened to converge, and the resulting jump is open-loop, so
        # acceptance is not a single noisy sample racing a threshold.
        self.imaging_power_use_fit = True           # False reverts stabilize_lightshift to the P/I servo
        self.N_points_imaging_power_map = np.int32(9)   # probe points per fit (>= 3, <= 32)
        self.frac_span_imaging_power_map = 0.6      # probe band, fractional full width about the current setpoint
        self.N_max_correction_imaging_power = np.int32(4)   # Newton corrections allowed after the jump
        # 0 = fit once and keep it. Drift costs an extra correction, not
        # accuracy, since the corrections are closed loop -- and a map stale
        # enough to miss triggers its own re-probe. Set > 0 only to refit on a
        # fixed schedule as well.
        self.N_shots_per_imaging_power_map_refit = np.int32(0)

        # Fit a quadratic rather than a line when the probe points can support
        # one. The map is compressive, so a chord across the probe band reads
        # too shallow at the top and too steep at the bottom, and that mismatch
        # is most of why the open-loop jump needs corrections at all. The fit
        # falls back to affine on its own if the quadratic is not earned.
        self.imaging_power_map_quadratic = True

        ### averaging, split by what each measurement is for
        # These deliberately differ. The dark is subtracted from every probe
        # point and every verification, so its noise is common mode across the
        # whole map -- averaging it is paid once and returned N times. The probe
        # points feed a 9-point fit that already averages by ~sqrt(N), so heavy
        # per-point averaging is largely redundant and the time is better spent
        # on more points over a wider lever arm.
        self.N_avg_imaging_dark = np.int32(8)       # readings per dark set
        self.N_avg_imaging_power_check = np.int32(2)    # readings per probe point

        # The verification does NOT use a fixed count: _verify_rate samples
        # until the accept/reject decision is unambiguous at k_sigma confidence,
        # between these bounds. A fixed count has to be sized for the worst case
        # and is wasted on every clean shot -- and if the noise exceeds
        # frac_err_threshold_imaging_pid it makes the comparison a coin flip, so
        # corrections chase noise and the whole budget burns on every shot.
        self.N_avg_imaging_verify_min = np.int32(3)     # >= 2; one sample has no variance
        self.N_avg_imaging_verify_max = np.int32(32)    # cap when the answer stays ambiguous
        self.k_sigma_imaging_verify = 2.5           # confidence multiplier on the standard error

        # A correction measures the local slope for free: the secant through the
        # last two points is a better derivative than any fit across the whole
        # band. Believe it only within this factor of what the map predicts --
        # a secant drawn through two noise-dominated points can be anything.
        self.frac_secant_slope_max = 4.
