import numpy as np
import copy
from artiq.language import delay, now_mu, kernel, TTuple, TBool

class DataContainer():
    _DTYPE_ORDER = ("i64", "u16", "u8", "i32", "f64")

    def __init__(self, per_shot_data_shape, dtype, external_data_bool, expt):
        self.key = ""
        self._per_shot_data_shape = tuple(np.atleast_1d(per_shot_data_shape))
        self._dtype = dtype
        self._external_data_bool = external_data_bool
        self._expt = expt

        self._dtype_id = self._normalize_dtype_id(dtype)
        self._dtype_idx = self._DTYPE_ORDER.index(self._dtype_id)
        self._num_dims = len(self._per_shot_data_shape)
        self._kernel_enabled = self._num_dims <= 2

        self._data_gotten = False

        self._run_data = np.zeros(per_shot_data_shape,dtype=dtype)
        self._init_shot_buffers()
        self._activate_shot_buffer()
        self._reference_data = copy.deepcopy(self.shot_data)

    @staticmethod
    def _normalize_dtype_id(dtype):
        normalized = np.dtype(dtype)
        if normalized == np.dtype(np.int64):
            return "i64"
        if normalized == np.dtype(np.uint16):
            return "u16"
        if normalized == np.dtype(np.uint8):
            return "u8"
        if normalized == np.dtype(np.int32):
            return "i32"
        if normalized == np.dtype(np.float64) or normalized == np.dtype(np.float32):
            return "f64"
        print(f"Unsupported DataContainer dtype '{dtype}', defaulting to float64 kernel path.")
        return "f64"

    def _init_shot_buffers(self):
        # Keep all inactive predeclared buffers at minimum size.
        self._shot_i64_r0 = np.zeros((), dtype=np.int64)
        self._shot_i64_r1 = np.zeros((1,), dtype=np.int64)
        self._shot_i64_r2 = np.zeros((1, 1), dtype=np.int64)

        self._shot_u16_r0 = np.zeros((), dtype=np.uint16)
        self._shot_u16_r1 = np.zeros((1,), dtype=np.uint16)
        self._shot_u16_r2 = np.zeros((1, 1), dtype=np.uint16)

        self._shot_u8_r0 = np.zeros((), dtype=np.uint8)
        self._shot_u8_r1 = np.zeros((1,), dtype=np.uint8)
        self._shot_u8_r2 = np.zeros((1, 1), dtype=np.uint8)

        self._shot_i32_r0 = np.zeros((), dtype=np.int32)
        self._shot_i32_r1 = np.zeros((1,), dtype=np.int32)
        self._shot_i32_r2 = np.zeros((1, 1), dtype=np.int32)

        self._shot_f64_r0 = np.zeros((), dtype=np.float64)
        self._shot_f64_r1 = np.zeros((1,), dtype=np.float64)
        self._shot_f64_r2 = np.zeros((1, 1), dtype=np.float64)

    def _activate_shot_buffer(self):
        # Allocate full-size buffer only for the active dtype/rank path.
        if not self._kernel_enabled:
            self.shot_data = np.zeros(self._per_shot_data_shape, dtype=self._dtype)
            return

        if self._dtype_id == "i64":
            if self._num_dims == 0:
                self._shot_i64_r0 = np.zeros((), dtype=np.int64)
                self.shot_data = self._shot_i64_r0
            elif self._num_dims == 1:
                self._shot_i64_r1 = np.zeros(self._per_shot_data_shape, dtype=np.int64)
                self.shot_data = self._shot_i64_r1
            else:
                self._shot_i64_r2 = np.zeros(self._per_shot_data_shape, dtype=np.int64)
                self.shot_data = self._shot_i64_r2
        elif self._dtype_id == "u16":
            if self._num_dims == 0:
                self._shot_u16_r0 = np.zeros((), dtype=np.uint16)
                self.shot_data = self._shot_u16_r0
            elif self._num_dims == 1:
                self._shot_u16_r1 = np.zeros(self._per_shot_data_shape, dtype=np.uint16)
                self.shot_data = self._shot_u16_r1
            else:
                self._shot_u16_r2 = np.zeros(self._per_shot_data_shape, dtype=np.uint16)
                self.shot_data = self._shot_u16_r2
        elif self._dtype_id == "u8":
            if self._num_dims == 0:
                self._shot_u8_r0 = np.zeros((), dtype=np.uint8)
                self.shot_data = self._shot_u8_r0
            elif self._num_dims == 1:
                self._shot_u8_r1 = np.zeros(self._per_shot_data_shape, dtype=np.uint8)
                self.shot_data = self._shot_u8_r1
            else:
                self._shot_u8_r2 = np.zeros(self._per_shot_data_shape, dtype=np.uint8)
                self.shot_data = self._shot_u8_r2
        elif self._dtype_id == "i32":
            if self._num_dims == 0:
                self._shot_i32_r0 = np.zeros((), dtype=np.int32)
                self.shot_data = self._shot_i32_r0
            elif self._num_dims == 1:
                self._shot_i32_r1 = np.zeros(self._per_shot_data_shape, dtype=np.int32)
                self.shot_data = self._shot_i32_r1
            else:
                self._shot_i32_r2 = np.zeros(self._per_shot_data_shape, dtype=np.int32)
                self.shot_data = self._shot_i32_r2
        else:
            if self._num_dims == 0:
                self._shot_f64_r0 = np.zeros((), dtype=np.float64)
                self.shot_data = self._shot_f64_r0
            elif self._num_dims == 1:
                self._shot_f64_r1 = np.zeros(self._per_shot_data_shape, dtype=np.float64)
                self.shot_data = self._shot_f64_r1
            else:
                self._shot_f64_r2 = np.zeros(self._per_shot_data_shape, dtype=np.float64)
                self.shot_data = self._shot_f64_r2

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

    @kernel
    def put_data(self,value,idx=0):
        if self._kernel_enabled:
            if self._num_dims == 0:
                self._put_data_r0(value)
            elif self._num_dims == 1:
                self._put_data_r1(value, idx)
            else:
                self._put_data_r2(value, idx)
        else:
            self.shot_data[idx] = value

    @kernel
    def _put_data_r0(self, value):
        if self._dtype_idx == 0:
            self._shot_i64_r0 = value
        elif self._dtype_idx == 1:
            self._shot_u16_r0 = value
        elif self._dtype_idx == 2:
            self._shot_u8_r0 = value
        elif self._dtype_idx == 3:
            self._shot_i32_r0 = value
        else:
            self._shot_f64_r0 = value

    @kernel
    def _put_data_r1(self, value, idx):
        if self._dtype_idx == 0:
            self._shot_i64_r1[idx] = value
        elif self._dtype_idx == 1:
            self._shot_u16_r1[idx] = value
        elif self._dtype_idx == 2:
            self._shot_u8_r1[idx] = value
        elif self._dtype_idx == 3:
            self._shot_i32_r1[idx] = value
        else:
            self._shot_f64_r1[idx] = value

    @kernel
    def _put_data_r2(self, value, idx):
        if self._dtype_idx == 0:
            self._shot_i64_r2[idx] = value
        elif self._dtype_idx == 1:
            self._shot_u16_r2[idx] = value
        elif self._dtype_idx == 2:
            self._shot_u8_r2[idx] = value
        elif self._dtype_idx == 3:
            self._shot_i32_r2[idx] = value
        else:
            self._shot_f64_r2[idx] = value

    @kernel
    def _put_shot_data(self):
        if self._kernel_enabled:
            self._update_active_to_host()
        else:
            self.update_to_host(self.shot_data)
        self._put_shot_data_to_run_data()

    @kernel
    def _update_active_to_host(self):
        if self._dtype_idx == 0:
            if self._num_dims == 0:
                self.update_to_host(self._shot_i64_r0)
            elif self._num_dims == 1:
                self.update_to_host(self._shot_i64_r1)
            else:
                self.update_to_host(self._shot_i64_r2)
        elif self._dtype_idx == 1:
            if self._num_dims == 0:
                self.update_to_host(self._shot_u16_r0)
            elif self._num_dims == 1:
                self.update_to_host(self._shot_u16_r1)
            else:
                self.update_to_host(self._shot_u16_r2)
        elif self._dtype_idx == 2:
            if self._num_dims == 0:
                self.update_to_host(self._shot_u8_r0)
            elif self._num_dims == 1:
                self.update_to_host(self._shot_u8_r1)
            else:
                self.update_to_host(self._shot_u8_r2)
        elif self._dtype_idx == 3:
            if self._num_dims == 0:
                self.update_to_host(self._shot_i32_r0)
            elif self._num_dims == 1:
                self.update_to_host(self._shot_i32_r1)
            else:
                self.update_to_host(self._shot_i32_r2)
        else:
            if self._num_dims == 0:
                self.update_to_host(self._shot_f64_r0)
            elif self._num_dims == 1:
                self.update_to_host(self._shot_f64_r1)
            else:
                self.update_to_host(self._shot_f64_r2)

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
        self.squeeze_axes(xvd)

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

    @kernel
    def update_to_host(self, data):
        """Necessary to sync up host and kernel.
        """   
        self.update_from_kernel(data)

    def get_run_data(self, copy_data=False):
        if copy_data:
            return self._run_data.copy()
        return self._run_data

class DataVault():
    
    def __init__(self, expt=None):
        self.keys = []
        self._container_list = []
        self._expt = expt

    def add_data_container(self,
                            per_shot_data_shape=(1,),
                            dtype=np.float64,
                            external_data_bool=False) -> DataContainer:
        """Returns a data container object. This should be assigned to an
        attribute of the `DataVault` object, which will then write to the data
        container the key used for the assignment during `finish_prepare` of
        an experiment.

        Example in `prepare`: for an experiment with DataVault object `self.data`:
            self.data.my_data = self.data.add_data_container()

        Example in `kexp.config.data_vault`: add to `__init__`
            self.my_data = self.add_data_container()

        Both cases will result in the data being saved to hdf5 and loaded in
        atomdata with key 'my_data':
            in hdf5: f['data']['my_data']
            in atomdata: ad.data.my_data

        Args:
            per_shot_data_shape (tuple or array or int): Shape of the data per
                shot. Defaults to (1,).
            dtype (_type_, optional): Data type for each value in the data
                array. Defaults to np.float64.
            external_data_bool (bool, optional): Set to True if the data for
                this container will be populated directly into the hdf5 data file by
                a process external to the ARTIQ process. An example would be image
                data being stuck into the hdf5 file by LiveOD. Setting to True
                will cause the unshuffle code to load in the data from the hdf5
                at the end of the experiment for unshuffling (instead of
                overwriting the hdf5 contents with the placeholder arrays of
                zeros.) Defaults to False.

        Returns:
            DataContainer: _description_
        """        
        return DataContainer(per_shot_data_shape,
                            dtype,
                            external_data_bool,
                            self._expt)
    
    def init(self):
        self.write_keys()
        self.set_container_sizes()

    def write_keys(self):
        for k in self.__dict__.keys():
            obj = vars(self)[k]
            if isinstance(obj,DataContainer):
                vars(self)[k].key = k
                self.keys.append(k)
                self._container_list.append(obj)

    def set_container_sizes(self):
        for key in self.keys:
            dc = vars(self)[key]
            if isinstance(dc,DataContainer):
                dc.set_container_size()

    def get_run_data(self, key, copy_data=False):
        dc = vars(self)[key]
        if not isinstance(dc, DataContainer):
            raise KeyError(f"Data container '{key}' not found.")
        return dc.get_run_data(copy_data=copy_data)

    def get_all_run_data(self, copy_data=False):
        data = {}
        for key in self.keys:
            dc = vars(self)[key]
            if isinstance(dc, DataContainer):
                data[key] = dc.get_run_data(copy_data=copy_data)
        return data

    @kernel
    def put_shot_data(self):
        # self._put_shot_data_rpc()
        self._expt.core.wait_until_mu(now_mu())
        for dc in self._container_list:
            dc._put_shot_data()
        self._expt.core.break_realtime()