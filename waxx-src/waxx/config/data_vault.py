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
        self._per_shot_data_shape = tuple(np.atleast_1d(per_shot_data_shape))
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
                self._run_data[idx] = self.shot_data
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
        # squeeze the data shape axes if they have length == 1
        # self.squeeze_axes(xvd)

    def squeeze_axes(self, xvardims):
        """Identifies if the per-shot data has any axes with dimension 1. If so,
        squeezes them to avoid unnecessary indexing.

        Example: per-shot data is a single float (not a list of floats), with a
        2D scan with xvardims = [4,3]. set_container_size produces a data array
        of shape (4,3,1), but this is annoying -- to get the value corresponding
        to the (i,j)th shot, you'd need to do array[i,j,0]. By squeezing out the
        axis of size 1, we can index the (i,j)th value as array[i,j]. 

        Args:
            xvardims (list): The xvardims for the experiment.
        """        
        n_axes_to_squeeze = self._run_data.ndim - len(xvardims) # how many axes are for the per-shot data
        sh_axes_to_squeeze = np.asarray(self._run_data.shape[-n_axes_to_squeeze:]) # their shape
        squeeze_mask = sh_axes_to_squeeze == 1 # check which dims have size == 1
        # make a mask to index these axes starting from the end (-1,-2,...),
        # since xvar axes come first
        ax_idx_to_squeeze = -(np.arange(0,n_axes_to_squeeze,dtype=int) + 1)[squeeze_mask]
        # do the squeeze (convert axis index list to tuple to make it work with np.ndarray.squeeze)
        self._run_data = self._run_data.squeeze(axis=tuple(ax_idx_to_squeeze))

        self._per_shot_data_shape = self._run_data.shape[len(xvardims):]

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
    def put_data(self, value, idx=0):
        self.shot_data[idx] = value
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
    def put_data(self, value, idx=0):
        # Element-wise row fill over the 2nd index (avoids multidim whole-row assignment).
        for j in range(len(value)):
            self.shot_data[idx, j] = value[j]
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
    def put_data(self, value, idx=0):
        self.shot_data[idx] = value
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
    def put_data(self, value, idx=0):
        for j in range(len(value)):
            self.shot_data[idx, j] = value[j]
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
    def put_data(self, value, idx=0):
        self.shot_data[idx] = value
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
    def put_data(self, value, idx=0):
        for j in range(len(value)):
            self.shot_data[idx, j] = value[j]
    @kernel
    def update_to_host(self):
        self.update_from_kernel(self.shot_data)
    @kernel
    def _put_shot_data(self):
        if self._is_sentinel:
            return
        self.update_to_host()
        self._put_shot_data_to_run_data()


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