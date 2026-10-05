import numpy as np
import copy
from artiq.language import delay, now_mu, kernel, TTuple, TBool

from waxa.config.data_vault import DataContainer as DataContainerWaxa

class DataContainer(DataContainerWaxa):
    """Abstract base -- NEVER instantiate or place in a kernel list directly.

    Holds only host-side logic (no @kernel methods). The concrete (ndim, dtype)
    subclasses below each define their OWN copies of the kernel methods
    (put_data / update_to_host / _put_shot_data). Those must NOT be factored up
    here and inherited: ARTIQ caches a quoted function by identity and gives its
    `self` a single TInstance, so a shared kernel method called on several
    subclass instances (as DataVault.put_shot_data does) would fail to unify
    either the `self` instance types or the differing `shot_data` array types.
    Distinct per-subclass functions get type-checked independently against their
    concrete attributes. The RPC targets below (update_from_kernel,
    _put_shot_data_to_run_data) stay shared because their bodies run in CPython
    and are never type-checked; only the per-subclass call sites are typed.
    """
    # Concrete subclasses override these. Kept here so the base is well-formed.
    _NDIM = 1
    _DTYPE = np.float64

    def __init__(self, per_shot_data_shape, dtype, external_data_bool, expt):
        self.key = ""
        # plain ints (not np scalars) so shape mismatch messages read cleanly
        self._per_shot_data_shape = tuple(int(n) for n in np.atleast_1d(per_shot_data_shape))
        self._dtype = dtype
        self._external_data_bool = external_data_bool
        self._expt = expt

        self._data_gotten = False
        # Sentinel = placeholder DataVault inserts to keep a per-type kernel list
        # non-empty (ARTIQ cannot infer the element type of an empty object list).
        # Its per-shot sync is a no-op (see each subclass's _put_shot_data), and
        # it is never added to self.keys / saved.
        self._is_sentinel = False

        self._run_data = np.zeros(per_shot_data_shape,dtype=dtype)
        # Shape of one shot's cell in _run_data. Equals _per_shot_data_shape
        # until set_container_size drops trailing size-1 axes (see squeeze_axes).
        self._cell_shape = self._per_shot_data_shape
        self.shot_data = np.zeros(per_shot_data_shape,dtype=dtype)
        self._reference_data = copy.deepcopy(self.shot_data)

    # ---- host-side only (RPC targets / setup); shared across subclasses ----

    def _put_shot_data_to_run_data(self):
        """Insert data into the array for the current shot.

        Args:
            value (_type_): _description_
        """
        if self._data_gotten:
            try:
                idx = tuple([x.counter for x in self._expt.scan_xvars])
                # Reshape to the cell shape rather than assigning shot_data
                # directly: squeeze_axes may have dropped trailing size-1 axes
                # from _run_data, and assigning a shape-(1,) array into a scalar
                # cell is deprecated in numpy (and will eventually raise). A
                # genuinely wrong-shaped shot_data raises here and is reported
                # against the declared shape below.
                self._run_data[idx] = self.shot_data.reshape(self._cell_shape)
            except Exception as e:
                if self.shot_data.shape != self._per_shot_data_shape:
                    print(f"Value is not correct shape for data container '{self.key}':\n"+
                    f"  expected shape {self._per_shot_data_shape} but value has shape {self.shot_data.shape}. Skipping.")
                else:
                    print(f"An error occurred with 'put_data' for data container '{self.key}':")
                    print(e)

    def set_container_size(self):
        """Takes the per-shot data array and patterns it to the appropriate shape.
        For xvardims = [n0,...,nN] and per-shot data of shape (p0,...,pM) (arb
        dimension), data array takes shape (n0,...,nN,p0,...,pM).
        """        
        xvd = self._expt.xvardims
        y = self._run_data
        for d in np.flip(xvd):
            y = [y]*d
        self._run_data = np.asarray(y)
        # drop trailing per-shot axes of length == 1
        self.squeeze_axes(xvd)

    def squeeze_axes(self, xvardims):
        """Drops trailing per-shot axes of size 1 to avoid unnecessary indexing.

        Example: per-shot data is a single float (not a list of floats), with a
        2D scan with xvardims = [4,3]. set_container_size produces a data array
        of shape (4,3,1), but this is annoying -- to get the value corresponding
        to the (i,j)th shot, you'd need to do array[i,j,0]. By dropping the
        trailing axis of size 1, we can index the (i,j)th value as array[i,j].

        Only *trailing* per-shot axes are dropped: (n0,...,nN,10,1) becomes
        (n0,...,nN,10), but (n0,...,nN,1,10) is left alone. The xvar axes are
        never touched, even when an xvar has a single value.

        Args:
            xvardims (list): The xvardims for the experiment.
        """
        n_xvar_axes = len(xvardims) # leading axes are the xvars; never dropped
        shape = list(self._run_data.shape)
        while len(shape) > n_xvar_axes and shape[-1] == 1:
            shape.pop()
        # reshape rather than squeeze(axis=...): no axis-index bookkeeping to get
        # backwards, and it is a no-op when there is nothing to drop
        self._run_data = self._run_data.reshape(shape)
        # _per_shot_data_shape deliberately keeps the shape the user declared --
        # shot_data is never squeezed, so the diagnostic in
        # _put_shot_data_to_run_data must compare against the declared shape.
        self._cell_shape = tuple(shape[n_xvar_axes:])

    def update_from_kernel(self, data):
        """Necessary to sync up host and kernel.
        """      
        self.shot_data = data
        if not self._data_gotten:
            self._data_gotten = not np.all(self.shot_data == self._reference_data)  


# --- Concrete (ndim, dtype) subclasses ---------------------------------------
# Each defines its OWN kernel methods (do not factor up into the base). The
# bodies are identical text, but ARTIQ type-checks each separately against the
# subclass's concrete shot_data type. Supported: 1D/2D of float64/int32/int64.

class DataContainer1D_f64(DataContainer):
    _NDIM, _DTYPE = 1, np.float64

    @kernel
    def put_data(self, value, i=0):
        self.shot_data[i] = value

    @kernel
    def put_data_1d(self, value):
        for j in range(len(value)):
            self.shot_data[j] = value[j]

    @kernel
    def update_to_host(self):
        self.update_from_kernel(self.shot_data)

    @kernel
    def _put_shot_data(self):
        if self._is_sentinel:
            return
        self.update_to_host()
        self._put_shot_data_to_run_data()

class DataContainer2D_f64(DataContainer):
    _NDIM, _DTYPE = 2, np.float64

    @kernel
    def put_data(self, value, i=0, j=0):
        self.shot_data[i, j] = value

    @kernel
    def put_data_1d(self, value, i=0):
        for j in range(len(value)):
            self.shot_data[i, j] = value[j]

    @kernel
    def put_data_2d(self, value):
        for i in range(len(value)):
            for j in range(len(value[i])):
                self.shot_data[i, j] = value[i,j]

    @kernel
    def update_to_host(self):
        self.update_from_kernel(self.shot_data)

    @kernel
    def _put_shot_data(self):
        if self._is_sentinel:
            return
        self.update_to_host()
        self._put_shot_data_to_run_data()

class DataContainer1D_i32(DataContainer):
    _NDIM, _DTYPE = 1, np.int32

    @kernel
    def put_data(self, value, i=0):
        self.shot_data[i] = value

    @kernel
    def put_data_1d(self, value):
        for j in range(len(value)):
            self.shot_data[j] = value[j]

    @kernel
    def update_to_host(self):
        self.update_from_kernel(self.shot_data)

    @kernel
    def _put_shot_data(self):
        if self._is_sentinel:
            return
        self.update_to_host()
        self._put_shot_data_to_run_data()

class DataContainer2D_i32(DataContainer):
    _NDIM, _DTYPE = 2, np.int32

    @kernel
    def put_data(self, value, i=0, j=0):
        self.shot_data[i, j] = value

    @kernel
    def put_data_1d(self, value, i=0):
        for j in range(len(value)):
            self.shot_data[i, j] = value[j]

    @kernel
    def put_data_2d(self, value):
        for i in range(len(value)):
            for j in range(len(value[i])):
                self.shot_data[i, j] = value[i,j]

    @kernel
    def update_to_host(self):
        self.update_from_kernel(self.shot_data)

    @kernel
    def _put_shot_data(self):
        if self._is_sentinel:
            return
        self.update_to_host()
        self._put_shot_data_to_run_data()

class DataContainer1D_i64(DataContainer):
    _NDIM, _DTYPE = 1, np.int64

    @kernel
    def put_data(self, value, i=0):
        self.shot_data[i] = value

    @kernel
    def put_data_1d(self, value):
        for j in range(len(value)):
            self.shot_data[j] = value[j]

    @kernel
    def update_to_host(self):
        self.update_from_kernel(self.shot_data)

    @kernel
    def _put_shot_data(self):
        if self._is_sentinel:
            return
        self.update_to_host()
        self._put_shot_data_to_run_data()

class DataContainer2D_i64(DataContainer):
    _NDIM, _DTYPE = 2, np.int64

    @kernel
    def put_data(self, value, i=0, j=0):
        self.shot_data[i, j] = value

    @kernel
    def put_data_1d(self, value, i=0):
        for j in range(len(value)):
            self.shot_data[i, j] = value[j]

    @kernel
    def put_data_2d(self, value):
        for i in range(len(value)):
            for j in range(len(value[i])):
                self.shot_data[i, j] = value[i,j]

    @kernel
    def update_to_host(self):
        self.update_from_kernel(self.shot_data)

    @kernel
    def _put_shot_data(self):
        if self._is_sentinel:
            return
        self.update_to_host()
        self._put_shot_data_to_run_data()


class HostDataContainer(DataContainer):
    """A container the HOST fills (no kernel involvement at all).

    For per-shot data a host thread produces during the shot -- e.g. an
    auxiliary camera frame snapped over the network (CameraStreamClient).
    Differences from the kernel containers:

    * any numpy dtype and 1D/2D/3D per-shot shape (uint8 frames included);
    * never routed to the per-type kernel lists, so ``put_shot_data`` never
      ships it through an RPC and the kernel never embeds its array;
    * ``_data_gotten`` is True from birth: END_RUN always saves ``_run_data``,
      so the fill value (e.g. -1 for "no frame") reaches the file instead of
      HDF5 zeros masquerading as data;
    * filled with ``put_shot_data_host(idx, value)`` from host code.  The
      caller is responsible for locking against END_RUN serialization.

    ``keep_run_data=False`` makes it *stream-only*: the experiment process
    keeps no copy of the run's values at all. Each shot goes into the run's
    file during the run (Expt.queue_shot_data -> liveOD PUT_DATA), liveOD
    pre-allocates the dataset with ``fill_value``, and END_RUN never carries
    it. ``_run_data`` is then a read-only zero-memory view of the fill value
    with the run's full shape and dtype (for INIT_RUN's shapes), never a
    real array; ``put_shot_data_host`` refuses. Camera streams use this:
    9 diagnostic frames a shot held for the whole run were ~2.5 GB per
    1000 shots in the experiment process.
    """

    def __init__(self, per_shot_data_shape, dtype, external_data_bool, expt,
                 fill_value=0, keep_run_data=True):
        super().__init__(per_shot_data_shape, dtype, external_data_bool, expt)
        self._data_gotten = True
        self._fill_value = fill_value
        self._keep_run_data = bool(keep_run_data)
        self._run_data = self._fill_array(self._per_shot_data_shape)
        # True once a shot of it went into the run's file during the run
        # (Expt.push_shot_data): END_RUN then sends no copy of the array
        self._pushed = False

    @property
    def stream_only(self) -> bool:
        """True: no in-memory copy; the run's file is the only store."""
        return not self._keep_run_data

    def _fill_array(self, shape):
        if self._keep_run_data:
            return np.full(shape, self._fill_value, dtype=self._dtype)
        # every element is the one 0-d fill value (strides 0): no memory
        # however big the shape, and writing to it raises
        return np.broadcast_to(np.array(self._fill_value, dtype=self._dtype), shape)

    def set_container_size(self):
        if self._keep_run_data:
            return super().set_container_size()
        # the shape set_container_size + squeeze_axes give, without building it
        xvd = [int(n) for n in self._expt.xvardims]
        shape = xvd + list(self._per_shot_data_shape)
        while len(shape) > len(xvd) and shape[-1] == 1:
            shape.pop()
        self._run_data = self._fill_array(tuple(shape))
        self._cell_shape = tuple(shape[len(xvd):])

    def put_shot_data_host(self, idx, value):
        """Write one shot's value at xvar-counter index ``idx`` (tuple).
        Raises on a shape/dtype mismatch -- the caller records the failure."""
        if not self._keep_run_data:
            raise RuntimeError(
                f"data container '{self.key}' is stream-only (keep_run_data=False): "
                f"it keeps no values in this process; send a shot with "
                f"Expt.queue_shot_data instead")
        self._run_data[tuple(idx)] = np.asarray(value).reshape(self._cell_shape)


class DataVault():

    def __init__(self, expt=None):
        self.keys = []
        self._container_list = []
        self._expt = expt

        # One homogeneous list per concrete (ndim, dtype). put_shot_data iterates
        # each separately (a single ARTIQ loop variable may not change type), and
        # each list is kept non-empty via a sentinel in init().
        self._list_1d_f64 = []
        self._list_2d_f64 = []
        self._list_1d_i32 = []
        self._list_2d_i32 = []
        self._list_1d_i64 = []
        self._list_2d_i64 = []

        # (ndim, dtype_scalar_type) -> (class, per-type list). Host-only; never
        # referenced in a kernel, so ARTIQ never tries to type it.
        self._registry = {
            (1, np.float64): (DataContainer1D_f64, self._list_1d_f64),
            (2, np.float64): (DataContainer2D_f64, self._list_2d_f64),
            (1, np.int32):   (DataContainer1D_i32, self._list_1d_i32),
            (2, np.int32):   (DataContainer2D_i32, self._list_2d_i32),
            (1, np.int64):   (DataContainer1D_i64, self._list_1d_i64),
            (2, np.int64):   (DataContainer2D_i64, self._list_2d_i64),
        }

    @staticmethod
    def _shape_ndim(per_shot_data_shape):
        return len(tuple(np.atleast_1d(per_shot_data_shape)))

    def add_data_container(self,
                            per_shot_data_shape=(1,),
                            dtype=np.float64,
                            external_data_bool=False) -> DataContainer:
        """Returns a data container object. This should be assigned to an
        attribute of the `DataVault` object, which will then write to the data
        container the key used for the assignment during `finish_prepare` of
        an experiment.

        Dispatches to the concrete container subclass matching (ndim, dtype),
        where ndim is inferred from `per_shot_data_shape` (a 1-element shape ->
        1D container, a 2-element shape -> 2D). Supported: 1D/2D of float64,
        int32, int64.

        Example in `prepare`: for an experiment with DataVault object `self.data`:
            self.data.my_data = self.data.add_data_container()                 # 1D float64
            self.data.counts  = self.data.add_data_container(8, np.int32)      # 1D int32, length 8
            self.data.grid    = self.data.add_data_container((4, 8), np.int64) # 2D int64

        Example in `kexp.config.data_vault`: add to `__init__`
            self.my_data = self.add_data_container()

        Both cases will result in the data being saved to hdf5 and loaded in
        atomdata with key 'my_data':
            in hdf5: f['data']['my_data']
            in atomdata: ad.data.my_data

        Args:
            per_shot_data_shape (tuple or array or int): Shape of the data per
                shot. A 1-element shape -> 1D container, a 2-element shape -> 2D.
                Defaults to (1,).
            dtype (_type_, optional): Data type for each value in the data
                array. One of np.float64, np.int32, np.int64. Defaults to
                np.float64.
            external_data_bool (bool, optional): Set to True if the data for
                this container will be populated directly into the hdf5 data file by
                a process external to the ARTIQ process. An example would be image
                data being stuck into the hdf5 file by LiveOD. Setting to True
                will cause the unshuffle code to load in the data from the hdf5
                at the end of the experiment for unshuffling (instead of
                overwriting the hdf5 contents with the placeholder arrays of
                zeros.) Defaults to False.

        Returns:
            DataContainer: a concrete (ndim, dtype) container subclass instance.
        """
        ndim = self._shape_ndim(per_shot_data_shape)
        dtype_type = np.dtype(dtype).type
        entry = self._registry.get((ndim, dtype_type))
        if entry is None:
            raise ValueError(
                f"Unsupported data container (ndim={ndim}, dtype={np.dtype(dtype)}). "
                f"Supported: 1D/2D of float64, int32, int64."
            )
        cls = entry[0]
        return cls(per_shot_data_shape,
                   dtype,
                   external_data_bool,
                   self._expt)
    
    def add_host_data_container(self,
                                per_shot_data_shape=(1,),
                                dtype=np.float64,
                                fill_value=0,
                                external_data_bool=False,
                                keep_run_data=True) -> HostDataContainer:
        """A container host code fills during the run (``put_shot_data_host``);
        the kernel never sees it, so any numpy dtype and per-shot shape go
        (e.g. a uint8 camera frame). ``fill_value`` is what a shot that never
        got data reads back as (pick something data cannot be, e.g. -1 / NaN).
        ``keep_run_data=False``: stream-only, no copy kept in this process
        (see HostDataContainer). Assign the result to an attribute of
        ``self.data`` before ``finish_prepare`` like any other container."""
        return HostDataContainer(per_shot_data_shape, dtype,
                                 external_data_bool, self._expt,
                                 fill_value=fill_value,
                                 keep_run_data=keep_run_data)

    def init(self):
        self.write_keys()
        self.set_container_sizes()
        self._ensure_type_lists_nonempty()

    def write_keys(self):
        for k in list(self.__dict__.keys()):
            obj = vars(self)[k]
            if isinstance(obj,DataContainer):
                obj.key = k
                self.keys.append(k)
                self._container_list.append(obj)
                self._route_to_type_list(obj)

    def _route_to_type_list(self, dc):
        """Append a real container to its concrete (ndim, dtype) kernel list."""
        if isinstance(dc, HostDataContainer):
            return   # host-filled: never shipped by put_shot_data / the kernel
        entry = self._registry.get((dc._NDIM, np.dtype(dc._DTYPE).type))
        if entry is None:
            raise ValueError(
                f"Data container '{dc.key}' has unsupported type "
                f"(ndim={dc._NDIM}, dtype={np.dtype(dc._DTYPE)})."
            )
        entry[1].append(dc)

    def _ensure_type_lists_nonempty(self):
        """For every type the experiment did not use, insert one no-op sentinel
        so its kernel list is non-empty (ARTIQ cannot infer the element type of
        an empty object list, and put_shot_data iterates all six lists). The
        sentinel matches the subclass ndim/dtype and is never saved."""
        for (ndim, dtype_type), (cls, lst) in self._registry.items():
            if len(lst) == 0:
                shape = (1,) if ndim == 1 else (1, 1)
                sentinel = cls(shape, dtype_type, False, self._expt)
                sentinel._is_sentinel = True
                lst.append(sentinel)

    def set_container_sizes(self):
        for key in self.keys:
            dc = vars(self)[key]
            if isinstance(dc,DataContainer):
                dc.set_container_size()

    @kernel
    def put_shot_data(self):
        # Iterate each per-type list with a distinctly named loop variable: an
        # ARTIQ variable may not change type between loops. Every list is
        # non-empty (sentinels), and each subclass has its own _put_shot_data,
        # so no cross-type unification occurs.
        self._expt.core.wait_until_mu(now_mu())
        for dc_1d_f64 in self._list_1d_f64:
            dc_1d_f64._put_shot_data()
        for dc_2d_f64 in self._list_2d_f64:
            dc_2d_f64._put_shot_data()
        for dc_1d_i32 in self._list_1d_i32:
            dc_1d_i32._put_shot_data()
        for dc_2d_i32 in self._list_2d_i32:
            dc_2d_i32._put_shot_data()
        for dc_1d_i64 in self._list_1d_i64:
            dc_1d_i64._put_shot_data()
        for dc_2d_i64 in self._list_2d_i64:
            dc_2d_i64._put_shot_data()
        self._expt.core.break_realtime()