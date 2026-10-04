import queue
import threading

import numpy as np
from .oscilloscopes_base import Scope_Base, SiglentSDS2000X_Base
from artiq.language import TBool, now_mu
from artiq.experiment import kernel, rpc
from waxx.util.artiq.async_print import aprint

# ScopeData.close() waits this long for the sender to push the last traces
SCOPE_SENDER_FLUSH_S = 30.


class ScopeTraces:
    """A run's traces from one scope, shot by shot, without a time axis per
    shot: every shot's voltages (``v``, one ``(channels, points)`` float32
    array each), the first shot's time axes (``t_ref``), and the axes of only
    those shots whose axes differed (``t_extra``). ``full()`` gives the old
    ``(shots, channels, 2, points)`` array back when END_RUN has to carry it.
    (Keeping t per shot doubled the memory: 8 MB/shot for a 1M-point trace.)"""

    def __init__(self):
        self.v = []
        self.t_ref = None
        self.t_extra = {}

    @property
    def n(self) -> int:
        return len(self.v)

    @property
    def t_varies(self) -> bool:
        """Some shot's time axes differ from the first shot's."""
        return bool(self.t_extra)

    def store(self, arr) -> int:
        """One shot's ``(channels, 2, points)`` array (t, v); its shot number."""
        arr = np.asarray(arr, dtype=np.float32)
        t, v = arr[:, 0, :], arr[:, 1, :]
        shot = len(self.v)
        self.v.append(np.array(v))
        if self.t_ref is None:
            self.t_ref = np.array(t)
        elif t.shape != self.t_ref.shape or not np.array_equal(t, self.t_ref):
            self.t_extra[shot] = np.array(t)
        return shot

    def pad(self, n_shots: int):
        """Zero shots up to ``n_shots`` (save_on_underflow partial saves)."""
        if self.n == 0:
            return
        while self.n < n_shots:
            self.t_extra[self.n] = np.zeros_like(self.t_ref)
            self.v.append(np.zeros_like(self.v[0]))

    def clear(self):
        self.v, self.t_ref, self.t_extra = [], None, {}

    def full(self):
        """``(shots, channels, 2, points)`` float32, t and v for every shot."""
        if not self.v:
            return np.empty((0, 0, 2, 0), np.float32)
        v = np.asarray(self.v, dtype=np.float32)
        out = np.empty((v.shape[0], v.shape[1], 2, v.shape[2]), np.float32)
        out[:, :, 1, :] = v
        out[:, :, 0, :] = self.t_ref
        for shot, t in self.t_extra.items():
            out[shot, :, 0, :] = t
        return out


class _ScopeSender(threading.Thread):
    """Pushes each shot's traces into the run's file as soon as read_sweep
    has them, on this thread: the kernel's RPC returns as before. The first
    shot also writes the time axis; a later shot with a different one marks
    the scope ``t_varies`` and END_RUN sends everything the old way."""

    def __init__(self, expt):
        super().__init__(name="scope-sender", daemon=True)
        self._expt = expt
        self._q = queue.Queue()
        self.start()

    def enqueue(self, scope, arr, idx):
        self._q.put((scope, arr, idx))

    def flush(self, timeout):
        self._q.put(None)
        self.join(timeout)
        return not self.is_alive()

    def run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            scope, arr, idx = item
            try:
                self._push_one(scope, arr, idx)
            except Exception as e:
                scope.push_failed = True
                print(f"[{scope.label}] WARNING: pushing a shot's traces failed: {e!r}")

    def _push_one(self, scope, arr, idx):
        expt = self._expt
        final = expt.final_shot_index(idx)
        n_ch, _, npts = arr.shape
        key = f"scope_data/{scope.label}"
        specs = []
        if not getattr(scope, "_t_pushed", False):
            # the first shot's time axis (channel 0, as the file stores it);
            # a later shot with another one is the scope's t_varies (store)
            scope._t_pushed = True
            t_ref = scope.traces.t_ref
            t = np.asarray(t_ref[0] if t_ref is not None else arr[0, 0, :], dtype=np.float32)
            specs.append({"key": key + "/t", "index": None, "array": t,
                          "full_shape": (npts,), "fill": None})
        specs.append({"key": key + "/v", "index": final, "array": arr[:, 1, :],
                      "full_shape": tuple(expt.xvardims) + (n_ch, npts), "fill": np.nan})
        if expt.push_raw(specs):
            scope.pushed_any = True
        else:
            scope.push_failed = True


class ScopeData:
    def __init__(self):
        self.scopes = []
        self.xvardims = []
        self._scope_trace_taken = False
        self._expt = None
        self._sender = None

    def attach_expt(self, expt):
        """The experiment whose shots the traces belong to (shot index, file)."""
        self._expt = expt

    def push(self, scope, arr):
        """One shot's traces, just read: into the run's file on the sender
        thread (nothing here waits). ``arr`` is ``(channels, 2, points)``."""
        expt = self._expt
        if expt is None or not getattr(expt, "push_data_enabled", False):
            return
        try:
            idx = expt.current_shot_index()
            if self._sender is None:
                self._sender = _ScopeSender(expt)
            self._sender.enqueue(scope, arr, idx)
        except Exception as e:
            scope.push_failed = True
            print(f"[{scope.label}] WARNING: could not queue a shot's traces: {e!r}")

    def close(self):
        if self._sender is not None:
            if not self._sender.flush(SCOPE_SENDER_FLUSH_S):
                for scope in self.scopes:
                    scope.push_failed = True        # END_RUN carries the traces
                print(f"[ScopeData] WARNING: the trace sender did not finish within "
                      f"{SCOPE_SENDER_FLUSH_S:.0f} s; the traces go with END_RUN.")
            self._sender = None
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
            except Exception as e:
                print(f"[ScopeData.arm_rpc] ERROR arming '{scope.label}': {e}")

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
        self.traces = ScopeTraces()     # this run's traces, one time axis
        self._channels = []
        self._reshaped = False
        self._full = None
        # traces pushed into the run's file shot by shot (ScopeData.push)
        self._t_pushed = False
        self.pushed_any = False
        self.push_failed = False
        
        self._scopedata.scopes.append(self)

        if not hasattr(self,'scope'):
            self.scope = Scope_Base()

    @property
    def t_varies(self) -> bool:
        """A shot's time axes differed from the first shot's: END_RUN sends
        every shot's traces, axes included."""
        return self.traces.t_varies

    @property
    def pushed_ok(self) -> bool:
        """Every shot's traces are in the run's file already (END_RUN need
        not carry them)."""
        return self.pushed_any and not self.push_failed and not self.t_varies

    def _store_sweep(self, arr):
        """One shot's ``(channels, 2, points)`` traces: kept (one time axis for
        the run) and pushed into the run's file."""
        self.traces.store(arr)
        self._reshaped = False
        if arr.ndim == 3:
            self._scopedata.push(self, arr)

    def clear_data(self):
        self.traces.clear()
        self._reshaped = False
        self._full = None

    def data(self):
        if self._scopedata.xvardims != []:
            return self.reshape_data()
        return self.traces.full()

    def close(self):
        self.scope.close()

    def reshape_data(self):
        """``(*xvardims, channels, 2, points)``: every shot's t and v."""
        if self.traces.n == 0:
            n_xvar_dims = len(self._scopedata.xvardims)
            print(f"[{self.label}] WARNING: reshape_data() called with no data — returning empty array.")
            return np.empty((0,) * (n_xvar_dims + 3))
        if not self._reshaped:
            full = self.traces.full()
            n_ch, npts = full.shape[1], full.shape[-1]
            self._full = full.reshape(*self._scopedata.xvardims, n_ch, 2, npts)
            self._reshaped = True
        return self._full

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
        self.traces.pad(n_shots)
        self._reshaped = False

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
        self._npts = {}             # channel -> length of its last good capture
        self.failed_captures = []   # (shot index, channel, error) this run

    def read_sweep(self,channels) -> bool:
        """Read the given channels (0-indexed) of this shot's acquisition.

        A requested channel that cannot be read (switched off, short or failed
        transfer, ...) is stored as NaN, t and v alike, and reported, so one
        bad shot neither shifts the channel order nor makes the run's traces
        ragged (a ragged run loses all its scope data at save time: run 83178).
        The NaN placeholder takes the length of that channel's last good
        capture, else the scope's current acquisition length.
        """
        channels = np.atleast_1d(channels)
        self._scopedata._scope_trace_taken = True
        if np.any([ch not in range(4) for ch in channels]):
            raise ValueError('Invalid channel.')
        shot = self.traces.n
        data = []
        for ch in range(4):
            if ch not in channels:
                continue
            try:
                if not self.scope.is_channel_on(ch):
                    raise RuntimeError(f"C{ch+1} is switched off")
                (t,v) = self.scope.read_sweep(ch)
                self._npts[ch] = len(v)
                data.append([t,v])
            except Exception as e:
                self.failed_captures.append((shot, ch, repr(e)))
                npts = self._npts.get(ch)
                if not npts:
                    try:
                        npts = int(float(self.scope.query(":ACQuire:POINts?").strip()))
                    except Exception:
                        npts = None
                if npts:
                    print(f"[SiglentScope read_sweep] ERROR shot {shot} ch={ch}: "
                          f"{e} -- stored as NaN ({npts} pts)")
                    data.append(np.full((2, npts), np.nan))
                else:
                    print(f"[SiglentScope read_sweep] ERROR shot {shot} ch={ch}: "
                          f"{e} -- no length to size a NaN placeholder; this "
                          f"run's traces will be ragged")
        # float32 from the start: what the file holds, at half the memory
        self._store_sweep(np.asarray(data, dtype=np.float32))
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
        # built on first access -- see oscilloscopes_base (pylablib is slow to import)
        from .oscilloscopes_base import TektronixTBS1104B_Base
        # the local: self.device_id only exists after super().__init__ below
        self.scope = TektronixTBS1104B_Base(device_id)
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
        self._store_sweep(np.asarray(data, dtype=np.float32))
        return True