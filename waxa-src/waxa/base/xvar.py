import numpy as np

class xvar():
    def __init__(self,key:str,values:np.ndarray,position=0):
        """Defines an variable that will be scanned over in the scan_kernel.

        Args:
            key (str): The key of the ExptParams attribute to be scanned. Does
            not have to exist in ExptParams beforehand.
            values (np.ndarray): The values over which the attribute referenced
            by "key" should be scanned.
        """
        self.key = key
        if type(values) == float or type(values) == int:
            raise ValueError("xvar must be a list or ndarray")
        self.values = np.asarray(values)
        self.position = position
        self.counter = 0
        self.sort_idx = []