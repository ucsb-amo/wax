import numpy as np
from .oscilloscopes_base import Scope_Base, TektronixTBS1104B_Base, SiglentSDS2000X_Base
from artiq.language import TBool, now_mu
from artiq.experiment import kernel, rpc
from waxx.util.artiq.async_print import aprint

class ScopeData:
    def __init__(self):
        self.scopes = []
        self.xvardims = []
        self._scope_trace_taken = False

    def close(self):
        for scope in self.scopes:
            try:
                scope.close()
            except:
                pass

    def add_tektronix_scope(self,device_id="",label="",arm=True):
        scope = TektronixScope_TBS1104(device_id=device_id,
                                    label=label,
                                    arm=arm,
                                    scope_data=self)
        return scope
    
    def add_siglent_scope(self,device_id="",label="",arm=True):
        scope = SiglentScope_SDS2104X(device_id=device_id,
                                    label=label,
                                    arm=arm,
                                    scope_data=self)
        return scope
    
    def arm_rpc(self):
        for scope in self.scopes:
            try: 
                if scope._arm:
                    scope.scope.arm()
                else:
                    scope.scope.set_normal_trigger()
                    scope.scope.set_trigger_run()
            except: pass
        
    @kernel
    def arm(self):
        self.arm_rpc()

    def pad_to_n_shots(self, n_shots: int):
        """Pad all scopes to n_shots entries for save_on_underflow partial saves."""
        for scope in self.scopes:
            scope.pad_to_n_shots(n_shots)

class GenericWaxxScope():
    def __init__(self,device_id="",label="",arm=True,
                 scope_data=ScopeData()):
        """A scope object.

        Args:
            device_id (str): The USB VISA string that identifies the scope. If
            nothing is provided, will prompt user for an input. Default for no
            input is the first element (0 index) of
            pylablib.list_backend_resources.
            label (str): labels the scope. Defaults to "scope{idx}" where idx is
            how many scopes have been initialized for the given ScopeData
            object.
            scope_data (ScopeData): Should be the ScopeData object of the
            experiment ("self.scope_data").
        """        
        self._scopedata = scope_data
        self._arm = arm

        if label == "":
            idx = len(self._scopedata.scopes)
            label = f"scope{idx}"
        self.label = label
        self.device_id = self.handle_devid_input(device_id)
        self.scope_trace_taken_this_shot = False
        self._data = []
        self._channels = []
        self._reshaped = False
        
        self._scopedata.scopes.append(self)

        if not hasattr(self,'scope'):
            self.scope = Scope_Base()

    def clear_data(self):
        self._data = []
        self._reshaped = False

    def data(self):
        if self._scopedata.xvardims != []:
            self.reshape_data()
        return np.asarray(self._data)

    def close(self):
        self.scope.close()

    def reshape_data(self):
        if self._data == []:
            n_xvar_dims = len(self._scopedata.xvardims)
            print(f"[{self.label}] WARNING: reshape_data() called with no data — returning empty array.")
            return np.empty((0,) * (n_xvar_dims + 3))
        if not self._reshaped:
            self._data = np.asarray(self._data)
            Npts = np.array(self._data).shape[-1]
            self._data = self._data.reshape(*self._scopedata.xvardims,self._data.shape[-3],2,Npts)
            self._reshaped = True
        return self._data

    def handle_devid_input(self,device_id):
        default = (device_id == "")
        is_int = (isinstance(device_id,int))
        if default or is_int:
            from pylablib import list_backend_resources
            devs = list_backend_resources("visa")
            devs_usb = [dev for dev in devs if "USB" in dev]
            
            if default:
                if len(devs_usb) > 1:
                    print(*[dev+'\n' for dev in devs_usb])
                    idx = input("More than one USB device connected. Input the index of which device to use.")
                if idx == '':
                    idx = 0
                else:
                    try:
                        idx = int(idx)
                    except:
                        print('Input cannot be cast to int, using idx = 0.')
            if is_int:
                idx = device_id
            device_id = devs[idx]
        return device_id
    
    def arm(self):
        self.scope.arm()

    def pad_to_n_shots(self, n_shots: int):
        """Pad _data to n_shots entries with zeros for save_on_underflow partial saves.

        If no traces have been captured (empty _data), does nothing — the
        existing empty-_data guard in reshape_data() handles that case.
        If k captures exist where 0 < k < n_shots, appends zero-filled copies
        of _data[0] until len(_data) == n_shots.
        """
        n_captured = len(self._data)
        if n_captured == 0 or n_captured >= n_shots:
            return
        zero_entry = np.zeros_like(np.asarray(self._data[0]))
        while len(self._data) < n_shots:
            self._data.append(zero_entry.copy())

class SiglentScope_SDS2104X(GenericWaxxScope):
    def __init__(self,device_id="",label="",arm=True,
                 scope_data=ScopeData()):
        """A scope object.

        Args:
            device_id (str): The USB VISA or IP string that identifies the
            scope. If nothing is provided, will prompt user for an input.
            Default for no input is the first element (0 index) of
            pylablib.list_backend_resources.
            label (str): labels the scope. Defaults to "scope{idx}" where idx is
            how many scopes have been initialized for the given ScopeData
            object.
            scope_data (ScopeData): Should be the ScopeData object of the
            experiment ("self.scope_data").
        """        
        
        self.scope = SiglentSDS2000X_Base(device_id)
        super().__init__(device_id=device_id,label=label,arm=arm,scope_data=scope_data)
    
    def read_sweep(self,channels) -> bool:
        channels = np.atleast_1d(channels)
        self._scopedata._scope_trace_taken = True

        preamble = self.scope.get_waveform_preamble()
        Npts = preamble[0]
        data = []
        if np.any([ch not in range(4) for ch in channels]):
            raise ValueError('Invalid channel.')
        for ch in range(4):
            if self.scope.is_channel_visible(ch) and (ch in channels):
                try:
                    (t,v) = self.scope.read_sweep(ch)
                    data.append([t,v])
                except Exception as e:
                    print(f"[SiglentScope read_sweep] ch={ch} failed: {e}")
        self._data.append(np.array(data))
        return True

class TektronixScope_TBS1104(GenericWaxxScope):
    def __init__(self,device_id="",label="",arm=True,
                 scope_data=ScopeData()):
        """A scope object.

        Args:
            device_id (str): The USB VISA string that identifies the scope. If
            nothing is provided, will prompt user for an input. Default for no
            input is the first element (0 index) of
            pylablib.list_backend_resources.
            label (str): labels the scope. Defaults to "scope{idx}" where idx is
            how many scopes have been initialized for the given ScopeData
            object.
            scope_data (ScopeData): Should be the ScopeData object of the
            experiment ("self.scope_data").
        """  
        self.scope = TektronixTBS1104B_Base(self.device_id)
        super().__init__(device_id=device_id,label=label,arm=arm,scope_data=scope_data)

    def read_sweep(self,channels) -> TBool:
        """Read out the specified channels and records result to self.data.
        Channels not read in will be stored as all zeros.

        Args:
            channels (list/int): The channels to read out. 0-indexed.

        Returns:
            TBool: Returns true when read is complete.
        """        
        channels = np.atleast_1d(channels)
        self._scopedata._scope_trace_taken = True
        sweeps = self.scope.read_multiple_sweeps(list(np.array(channels) + 1))
        Npts = np.array(sweeps).shape[1]
        data = []
        d = np.zeros((2,Npts)) # data = np.zeros((4,2,Npts))
        j = 0
        for idx in range(4):
            if idx in channels:
                d[0] = sweeps[j][:,0] # data[idx][0] = sweeps[j][:,0]
                d[1] = sweeps[j][:,1] # data[idx][1] = sweeps[j][:,1]
                data.append(d)
                j += 1
        self._data.append(np.array(data))
        return True