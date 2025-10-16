import numpy as np
from artiq.experiment import kernel
from artiq.coredevice.ttl import TTLOut, TTLInOut
from wax.control.artiq.TTL import TTL, TTL_IN, TTL_OUT

class ttl_frame():
    def __init__(self):

        self.ttl_list = []
        
        # add TTLs here

        self._write_ttl_keys()

        self.camera = TTL

    def assign_ttl_out(self,ch) -> TTL_OUT:
        this_ttl = TTL_OUT(ch)
        self.ttl_list.append(this_ttl)
        return this_ttl
    
    def assign_ttl_in(self,ch) -> TTL_IN:
        this_ttl = TTL_IN(ch)
        self.ttl_list.append(this_ttl)
        return this_ttl
    
    def ttl_by_ch(self,ch) -> TTL:
        ch_list = [ttl.ch for ttl in self.ttl_list]
        if ch in ch_list:
            ch_idx = ch_list.index(ch)
            return self.ttl_list[ch_idx]
        else:
            raise ValueError(f"TTL ch {ch} not assigned in ttl_id.")
    
    def _write_ttl_keys(self):
        '''Adds the assigned keys to the DDS objects so that the user-defined
        names (keys) are available with the DDS objects.'''
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key],TTL):
                self.__dict__[key].key = key
                