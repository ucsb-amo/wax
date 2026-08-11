import numpy as np
from waxa.config.expt_params import ExptParams as ExptParamsWaxa

class ExptParams(ExptParamsWaxa):
    def __init__(self):
        super().__init__()
        
        self.beatlock_sign = -1
        self.N_offset_lock_reference_multiplier = 8
        self.frequency_minimum_offset_beatlock = 250.e6

        ### imaging power stabilization (BeatLockImagingPID.stabilize_power)
        self.t_apd_imaging_check = 150.e-6          # integration window per APD power check
        self.t_apd_pid_settle = 1.e-3               # settling allowed after each setpoint write
        self.t_apd_slack = 10.e-6                   # slack re-armed after each blocking sampler read
        self.N_max_iter_imaging_pid = np.int32(50)  # iteration cap (np.int32 pins TInt32)
        self.frac_err_threshold_imaging_pid = 0.003 # convergence threshold on |v_signal/v_target - 1|
        self.gain_p_imaging_pid = -0.019            # proportional gain (tuned)
        self.gain_i_imaging_pid = -0.0175           # integral gain (tuned)
        self.v_pid_imaging_min = 0.05               # lower rail; MUST be > 0 (DDS.set_dds ignores v_pd < 0)
        self.v_pid_imaging_max = 9.9                # upper rail; MUST be <= DAC_CH.max_v (9.99), above which the DAC zeroes the channel
        self.v_pid_imaging_seed = 0.5               # restart setpoint when dac_pid.v == 0 (a ratio jump cannot move off zero)
