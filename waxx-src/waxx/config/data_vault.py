import numpy as np
import copy
from artiq.language import delay, now_mu, kernel, TTuple, TBool, kernel_from_string, rpc

class DataContainer():
    def __init__(self, 
                 per_shot_data_shape,
                external_data_bool=False,
                expt=None):
        self.key = ""
        self._per_shot_data_shape = tuple(np.atleast_1d(per_shot_data_shape))
        if len(self._per_shot_data_shape) != 1:
            raise ValueError('per-shot data must be a 1D array')
        self._dtype = float
        self._external_data_bool = external_data_bool
        self._expt = expt

        self._data_gotten = False

        self._run_data = np.zeros(self._per_shot_data_shape,self._dtype)
        self.shot_data = np.zeros(self._per_shot_data_shape,self._dtype)
        self._reference_data = copy.deepcopy(self.shot_data)

    @rpc(flags={"async"})
    def _put_shot_data_to_run_data(self, data):
        """Insert data into the array for the current shot.

        Args:
            value (_type_): _description_
        """
        self.shot_data = data
        if not self._data_gotten:
            self._data_gotten = not np.all(self.shot_data == self._reference_data)

        if not self._data_gotten:
            return

        try:
            idx = tuple([x.counter for x in self._expt.scan_xvars])
            self._run_data[idx] = self.shot_data
        except Exception:
            # Ignore bad writes during acquisition; data saver/unshuffle handles
            # partial datasets more gracefully than a stalled experiment.
            pass

    @kernel
    def put_data_idx(self, value, idx=0):
        self.shot_data[idx] = value

    @kernel
    def _put_shot_data(self):
        # Single RPC per container: copy kernel shot_data to host and write it
        # into run_data.
        self._put_shot_data_to_run_data(self.shot_data)

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

class DataVault():
    
    def __init__(self, expt=None):
        self.keys = []
        self._container_list = []
        self._container_list_nonext = []
        self._keys_nonext= []
        self._expt = expt

        # self._xvar_writer_floats = []
        # self._xvar_writer_int32s = []
        # self._xvar_writer_int64s = []
        # self._xvar_writer_arrays = []

        # self._keylist_floats = []
        # self._keylist_int32s = []
        # self._keylist_int64s = []
        # self._keylist_arrays = []

        # Camera-server populated datasets. These are placeholders in the
        # experiment process and are filled externally by LiveOD.
        # Images/timestamps now live only on the server-side HDF5 path,
        # so no experiment-side DataContainer placeholders are created.

    def add_data_container(self,
                            per_shot_data_shape=(1,),
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
        return DataContainer(per_shot_data_shape=per_shot_data_shape,
                             external_data_bool=external_data_bool,
                             expt=self._expt)
    
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
                if not obj._external_data_bool:
                    self._keys_nonext.append(k)
                    self._container_list_nonext.append(obj)

    def set_container_sizes(self):
        for key in self.keys:
            dc = vars(self)[key]
            if isinstance(dc,DataContainer):
                dc.set_container_size()

    @kernel
    def put_shot_data(self):
        """Write all non-external containers' shot_data into run_data.
        Called as an implicit host RPC from cleanup_scan_kernel_wax at the
        end of every shot. Containers must have been updated via put_data()
        or put_data_idx() during the shot kernel.
        """
        for dc in self._container_list_nonext:
            dc._put_shot_data()

    # def generate_assignment_kernels(self):
    #     """Generates a list of kernel functions for each param datatype (int32,
    #     int64, ndarray, and float ) -- one for each ExptParam attribute. These
    #     can be called in the kernel to update the kernel experiment params with
    #     values from the host ExptParams returned by an RPC (the "fetch" functions).
    #     """

    #     for key in self._keys_nonext:
    #         bodycode = f"self.data.{key}._run_data = value"
    #         dtype = str(type(vars(self.data)[key]._run_data))

    #         if 'int' in dtype:
    #             if 'numpy.int64' in dtype:
    #                 self._keylist_int64s.append(key)
    #                 self._xvar_writer_int64s.append( kernel_from_string(["self","value"],bodycode) )
    #             else:
    #                 self._keylist_int32s.append(key)
    #                 self._xvar_writer_int32s.append( kernel_from_string(["self","value"],bodycode) )
    #         elif 'float' in dtype:
    #             self._keylist_floats.append(key)
    #             self._xvar_writer_floats.append( kernel_from_string(["self","value"],bodycode) )
    #         elif 'ndarray' in dtype:
    #             self._keylist_arrays.append(key)
    #             self._xvar_writer_arrays.append( kernel_from_string(["self","value"],bodycode) )

    