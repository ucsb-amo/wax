import numpy as np
from artiq.experiment import TFloat, TArray, portable, rpc

# distance per MHz:
class tweezer_xmesh():
    def __init__(self,
                F_CE_MAX = 74.5e6,
                F_CE_MIN = 70.e6,
                F_NCE_MAX = 82.e6,
                F_NCE_MIN = 76.e6,
                X_TO_F_OFFSET_CE = 0.000423043,
                X_TO_F_OFFSET_NCE = -0.000442464,
                X_PER_F_CE = -5.7971e-12,
                X_PER_F_NCE = 5.7971e-12):
        """
        Defines the calibration of tweezer frequency to position. The
        positive direction is to the right as viewed on the Andor.

        To recalibrate:
        1. Run tweezer_xpf_calibration.py, making sure that frequency, amplitude
        lists produce a pair of trapped tweezers for both cateye (ce) and
        non-cateye (nce).
        2. Run analysis file:
        k-jam/analysis/measurements/tweezer_xgrid_calibration.ipynb
        3. Replace x_per_f_ce, x_per_f_nce, x_to_f_offset_ce, and
        x_to_f_offset_nce (output of last cell).
        """

        # calibration run 18543
        # calibration ROI saved in data

        self.f_ce_max = F_CE_MAX
        self.f_ce_min = F_CE_MIN

        self.f_nce_max = F_NCE_MAX
        self.f_nce_min = F_NCE_MIN

        self.x_to_f_offset_ce = X_TO_F_OFFSET_CE
        self.x_to_f_offset_nce = X_TO_F_OFFSET_NCE
        self.x_per_f_ce = X_PER_F_CE
        self.x_per_f_nce = X_PER_F_NCE

    def arrcast(self,v,dtype=float):
            if not (isinstance(v,np.ndarray) or isinstance(v,list)):
                v = [v]
            return np.array(v,dtype=dtype)
        
    @rpc
    def x_to_f(self, position, cateye) -> TArray(TFloat):
        """Converts a tweezer position into the corresponding AOD frequency.

        Args:
            position (float or list/ndarray): position (in m)
            cateye_bool (bool or list/ndarray): whether or not the tweezer is
            cat-eyed.
        """
        if isinstance(position,np.ndarray) or isinstance(position,list):
            if len(position) != len(cateye):
                raise ValueError("The length of the cateye list and position list are not the same.")
        else:
            position = [position]
        if not (isinstance(cateye,np.ndarray) or isinstance(cateye,list)):
            cateye = [cateye]

        position = np.asarray(position)
        cateye = np.asarray(cateye)

        f_out = []
        for i in range(len(position)):
            x = position[i]
            c = cateye[i]
            if c:
                x_per_f = self.x_per_f_ce
                x_offset = self.x_to_f_offset_ce
            else:
                x_per_f = self.x_per_f_nce
                x_offset = self.x_to_f_offset_nce
            f = (x - x_offset) / x_per_f
            f_out.append(f)
        return np.array(f_out)
        
    @rpc
    def f_to_x(self, frequency) -> TArray(TFloat):
        """Converts an AOD frequency (in Hz) into the corresponding real-space
        position.

        Args:
            frequency (float): AOD frequency (in Hz)
        """
        if not(isinstance(frequency,np.ndarray) or isinstance(frequency,list)):
            frequency = [frequency]
        frequency = np.asarray(frequency)
        cateye = frequency < self.f_ce_max
        
        x_out = []
        for i in range(len(frequency)):
            f = frequency[i]
            c = cateye[i]
            if c:
                x_per_f = self.x_per_f_ce
                x_offset = self.x_to_f_offset_ce
            else:
                x_per_f = self.x_per_f_nce
                x_offset = self.x_to_f_offset_nce
            x = x_per_f * f + x_offset
            x_out.append(x)
        return np.array(x_out)
    
    @rpc(flags={"async"})
    def check_valid_range(self, frequency, cateye):
        if cateye:
            if frequency > self.f_ce_max or frequency < self.f_ce_min:
                raise ValueError(f"Requested cateye frequency {frequency/1.e6:1.2f} out of safe range.")
        else:
            if frequency > self.f_nce_max or frequency < self.f_nce_min:
                raise ValueError(f"Requested non-cateye frequency {frequency/1.e6:1.2f} out of safe range.")