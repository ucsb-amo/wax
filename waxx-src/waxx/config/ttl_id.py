import numpy as np
from artiq.experiment import kernel
from artiq.coredevice.ttl import TTLOut, TTLInOut

from waxx.control.artiq.TTL import TTL, TTL_IN, TTL_OUT

N_TTL = 8

class ttl_frame():
    def __init__(self):

        self._db = None

        self.setup(N_TTL)

        # add TTLs here

        self.cleanup()
        
    def setup(self, N_TTL):
        self.populate_ttl_list(N_TTL)

    def cleanup(self):
        self._write_ttl_keys()
        self.camera = TTL_OUT
        self.populate_attrs()
        self.populate_typed_lists()

    def populate_attrs(self):
        for ttl in self.ttl_list:
            vars(self)[ttl.key] = ttl

    def populate_typed_lists(self):
        """Splits ttl_list into per-direction lists so kernels can loop over
        one kind of channel -- in particular clear_input_events() must drain
        every TTLInOut, whether or not it was assigned a name."""
        self.ttl_out_list = [ttl for ttl in self.ttl_list if isinstance(ttl, TTL_OUT)]
        self.ttl_in_list = [ttl for ttl in self.ttl_list if isinstance(ttl, TTL_IN)]

    @kernel
    def clear_input_events(self):
        """Drains the input-event FIFO of every TTLInOut channel. A stale edge
        left over from a previous shot's gate makes the next timestamp_mu
        return immediately with an old timestamp, which puts the timeline
        cursor in the past and underflows the next event."""
        for ttl_in in self.ttl_in_list:
            ttl_in.clear_input_events()

    def populate_ttl_list(self, N_TTL):
        self.ttl_list = [TTL(ch) for ch in range(N_TTL)]
        for ch in range(N_TTL):
            ttl = self.ttl_list[ch]
            if self._db is not None:
                cl = self._db[f"ttl{ch}"]["class"]
                if cl == "TTLOut":
                    ttl = TTL_OUT(ch)
                elif cl == "TTLInOut":
                    ttl = TTL_IN(ch)
            ttl.key = f"ttl{ch}"
            self.ttl_list[ch] = ttl

    def assign_ttl_out(self,ch) -> TTL_OUT:
        this_ttl = TTL_OUT(ch)
        self.ttl_list[ch] = this_ttl
        return this_ttl
    
    def assign_ttl_in(self,ch) -> TTL_IN:
        this_ttl = TTL_IN(ch)
        self.ttl_list[ch] = this_ttl
        return this_ttl
    
    # def ttl_by_ch(self,ch) -> TTL:
    #     ch_list = [ttl.ch for ttl in self.ttl_list]
    #     if ch in ch_list:
    #         ch_idx = ch_list.index(ch)
    #         return self.ttl_list[ch_idx]
    #     else:
    #         raise ValueError(f"TTL ch {ch} not assigned in ttl_id.")
    
    def _write_ttl_keys(self):
        '''Adds the assigned keys to the DDS objects so that the user-defined
        names (keys) are available with the DDS objects.'''
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key],TTL):
                self.__dict__[key].key = key
                