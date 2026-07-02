import numpy as np
from typing import TYPE_CHECKING
from waxa.base.xvar import xvar
from waxa.config.data_vault import DataContainer, DataVault
if TYPE_CHECKING:
    from waxx.config.data_vault import DataContainer, DataVault  # noqa: F811

# Selects how shots are ordered during a scan.
#   'nested' — legacy behaviour: xvar.values are permuted per-length and shots
#              step as nested for-loops (last xvar innermost). sort_idx/sort_N
#              record the per-length permutations for later unshuffling.
#   'global' — true-random: xvar.values stay in natural order and one single
#              permutation (execution_order) over all grid cells decides the
#              order shots are taken in. See generate_global_order().
SHUFFLE_MODE_NESTED = 'nested'
SHUFFLE_MODE_GLOBAL = 'global'


class Dealer():
    def __init__(self):
        self.sort_idx = []
        self.sort_N = []
        from waxa.config.expt_params import ExptParams
        self.params = ExptParams()
        self.xvarnames = []
        self.xvardims = []
        self.N_xvars = 0
        self.images = np.array([])
        self.image_timestamps = np.array([])

        # Global (true-random) ordering state. Populated by
        # generate_global_order() when shuffle_mode == 'global'; left empty for
        # the nested path so the presence of execution_order unambiguously
        # signals which ordering was used.
        self.shuffle_mode = SHUFFLE_MODE_NESTED
        self.execution_order = np.array([], dtype=int)
        self.grid_shape = []

        self.data = DataVault()

        from waxa.data.run_info import RunInfo
        self.run_info = RunInfo()

        self.scan_xvars = []
        self.Nvars = 0

    def plug_in_xvars(self):
        """Plugs in the first value of the xvar list in for the params before
        compilation, to avoid variable type clashes when scanning.
        """        
        for this_xvar in self.scan_xvars:
            this_xvar: xvar
            vars(self.params)[this_xvar.key] = this_xvar.values[0]

    def repeat_xvars(self,N_repeats=[]):
        """
        For each xvar in the scan_xvars list, replaces xvar.values with
        np.repeat(xvar.values,self.params.N_repeats).

        Parameters
        ----------
        N_repeats (int/list/ndarray, optional): The number of repeats to be
        implemented. Can be omitted to use the stored value of
        self.params.N_repeats. Must be either int or list/array of length one,
        or a list/array with one element per element of self.xvarnames.
        """        
        Nvars = self.Nvars

        # allow user to overwrite repeats number when repeat_xvars called
        if N_repeats != []:
            self.params.N_repeats = N_repeats

        error_msg = "self.params.repeats must have either have one element or length equal to the number of xvarnames"
        if isinstance(self.params.N_repeats,int):
            N_repeat = self.params.N_repeats
            self.params.N_repeats = [1 for _ in range(Nvars)]
            self.params.N_repeats[0] = N_repeat
        elif isinstance(self.params.N_repeats,list):
            if len(self.params.N_repeats) == 1:
                N_repeat = self.params.N_repeats[0]
                self.params.N_repeats = [1 for _ in range(Nvars)]
                self.params.N_repeats[0] = N_repeat
            elif len(self.params.N_repeats) != Nvars:
                raise ValueError(error_msg)
        elif isinstance(self.params.N_repeats,np.ndarray):
            if len(self.params.N_repeats) == 1:
                self.params.N_repeats = np.repeat(self.params.N_repeats,Nvars)
            elif len(self.params.N_repeats) != Nvars:
                raise ValueError(error_msg)

        for xvar in self.scan_xvars:
            xvar.values = np.repeat(xvar.values, self.params.N_repeats[xvar.position],axis=0)
        
        self.params.N_repeats = self.params.N_repeats[0]

    def shuffle_xvars(self,sort_preshuffle=True):
        """
        For each attribute of self.params with key specified in self.xvarnames,
        replaces the corresponding array with a shuffled version of that array.
        The shuffle orders are stored in self.sort_idx to be used in re-sorting
        derived arrays.

        Example: self.xvarnames = ['t_tof']. User specifies self.params.t_tof =
        [4.,6.,8.]. This function might rewrite-in-place self.params.t_tof =
        [8.,4.,6.], and record self.sort_idx = [3,1,2].

        Args:
            sort_preshuffle (bool, optional): If True, each xvar will be sorted
            so that its elements are sequential before being shuffled, such that
            when un-shuffled, the elements are in order. Defaults to True.
        """        
        rng = np.random.default_rng()
        sort_idx = []
        len_list = []

        # loop through xvars
        for xvar in self.scan_xvars:
            if sort_preshuffle:
                xvar.values = np.sort(xvar.values)

            # create list of scramble indices for each xvar
            # use same index list for xvars of same length
            ### Note: with new xvar class, this is not necessary. Update later.
            if xvar.values.shape[0] in len_list:
                match_idx = len_list.index(xvar.values.shape[0])
                sort_idx.append(sort_idx[match_idx])
            else:
                sort_idx.append( np.arange(xvar.values.shape[0]) )
                rng.shuffle(sort_idx[xvar.position])
                xvar.sort_idx = sort_idx[xvar.position]
            len_list.append(xvar.values.shape[0])
        
        # shuffle arrays with the scrambled indices
        for xvar in self.scan_xvars:
            scrambled_list = xvar.values.take(sort_idx[xvar.position],axis=0)
            xvar.values = scrambled_list

        # remove duplicates (shouldn't exist anyway), sort into lists
        sort_idx_w_duplicates = list(zip(len_list,sort_idx))
        self.sort_idx = []
        self.sort_N = []
        for elem in sort_idx_w_duplicates:
            if elem[0] not in self.sort_N:
                self.sort_N.append(elem[0])
                self.sort_idx.append(elem[1])

    def generate_global_order(self, sort_preshuffle=True):
        """Generate a single random permutation over ALL grid cells (true
        randomization), instead of the nested per-axis shuffle.

        Unlike shuffle_xvars(), this leaves every xvar.values array in its
        NATURAL (sorted) order. The randomness lives entirely in
        self.execution_order: a permutation of range(prod(grid_shape)), where
        grid_shape = [len(xvar.values) for xvar in scan_xvars] (after repeats).

        Shot k of the scan is taken at grid index
        np.unravel_index(execution_order[k], grid_shape). Because the scan sets
        each xvar.counter to that grid index, the DataVault hypercube and params
        are written in natural (grid) order on the host — no per-axis unshuffle
        is ever needed for them. Only arrays stored linearly in execution order
        (images, timestamps, scope traces) must be scattered back to grid order
        at save time (see scatter_exec_axis_to_grid).

        Call this once, in place of shuffle_xvars(), after repeat_xvars().
        """
        if sort_preshuffle:
            for this_xvar in self.scan_xvars:
                this_xvar.values = np.sort(this_xvar.values)
                this_xvar.sort_idx = []

        self.shuffle_mode = SHUFFLE_MODE_GLOBAL
        self.grid_shape = [xvar.values.shape[0] for xvar in self.scan_xvars]
        N_shots = int(np.prod(self.grid_shape)) if self.grid_shape else 1

        rng = np.random.default_rng()
        self.execution_order = rng.permutation(N_shots).astype(int)

        # The per-length permutation machinery is unused in global mode; keep it
        # empty so the ordering method is unambiguous everywhere downstream.
        self.sort_idx = []
        self.sort_N = []

    def _set_counters_from_step(self, k):
        """Set every xvar.counter to the grid index of the k-th shot in the
        global (true-random) order.

        Host-side (RPC) helper for the scan loop. In nested mode the counters
        are advanced by step_scan(); in global mode they are set here from
        execution_order.
        """
        k = int(k)
        flat = int(self.execution_order[k])
        midx = np.unravel_index(flat, tuple(self.grid_shape))
        for xvar in self.scan_xvars:
            xvar.counter = int(np.atleast_1d(midx)[xvar.position])

    def scatter_exec_axis_to_grid(self, arr, axis=0):
        """Reorder a linearly-stored (execution-order) array into natural grid
        order along the given axis.

        The array's `axis` must have length prod(grid_shape) (= N_shots). Shot k
        (stored at index k) belongs at grid-flat index execution_order[k], so
        the inverse permutation argsort(execution_order) maps grid index -> the
        stored execution index. Used at save time for images/timestamps/scope.
        """
        inv = np.argsort(self.execution_order)
        return np.take(arr, inv, axis=axis)

    def gather_grid_axis_to_exec(self, arr, axis=0):
        """Inverse of scatter_exec_axis_to_grid: reorder a grid-ordered array
        back into execution (time) order along the given axis."""
        return np.take(arr, self.execution_order, axis=axis)

    def unscramble_images(self,reshuffle=False):

        pwa, pwoa, dark = self.deal_data_ndarray(self.images)
        
        pwa = self._unshuffle_ndarray(pwa,exclude_dims=3,
                                        reshuffle=reshuffle)
        pwoa = self._unshuffle_ndarray(pwoa,exclude_dims=3,
                                        reshuffle=reshuffle)
        dark = self._unshuffle_ndarray(dark,exclude_dims=3,
                                        reshuffle=reshuffle)

        self.images = self.stack_linear_data_ndarray(pwa,pwoa,dark)

        return self.images

    def _unscramble_timestamps(self,reshuffle=False):

        t_pwa, t_pwoa, t_dark = self.deal_data_ndarray(self.image_timestamps)

        t_pwa = self._unshuffle_ndarray(t_pwa,reshuffle=reshuffle)
        t_pwoa = self._unshuffle_ndarray(t_pwoa,reshuffle=reshuffle)
        t_dark = self._unshuffle_ndarray(t_dark,reshuffle=reshuffle)

        self.image_timestamps = self.stack_linear_data_ndarray(t_pwa,t_pwoa,t_dark)

        return self.image_timestamps
    
    def stack_linear_data_ndarray(self,pwa,pwoa,dark):
        Ns = self.params.N_shots_with_repeats
        Nps = self.params.N_pwa_per_shot
        N_img = Ns*(Nps+2)

        ndarray = np.empty((Ns,Nps+2)+pwa.shape[(self.N_xvars+1):],
                            dtype=pwa.dtype)
        
        sh = pwa.shape
        
        pwa = pwa.reshape((Ns,Nps)+pwa.shape[(self.N_xvars+1):])
        pwoa = pwoa.reshape((Ns,Nps)+pwoa.shape[(self.N_xvars+1):])
        dark = dark.reshape((Ns,Nps)+dark.shape[(self.N_xvars+1):])

        for shot_idx in range(Ns):
            ndarray[shot_idx][:Nps] = pwa[shot_idx]
            ndarray[shot_idx][Nps] = pwoa[shot_idx][0]
            ndarray[shot_idx][Nps+1] = dark[shot_idx][0]

        ndarray = ndarray.reshape((N_img,)+sh[(self.N_xvars+1):])
        return ndarray

    def deal_data_ndarray(self,ndarray):
        Ns = self.params.N_shots_with_repeats
        Nps = self.params.N_pwa_per_shot
        ndarray = ndarray.reshape((Ns,Nps+2)+ndarray.shape[1:])

        pwa = ndarray[:,0:Nps]
        pwoa = np.expand_dims(ndarray[:,Nps],axis=1).repeat(Nps,axis=1)
        dark = np.expand_dims(ndarray[:,Nps+1],axis=1).repeat(Nps,axis=1)

        pwa = self._reshape_data_array_to_nxvar(pwa)
        pwoa = self._reshape_data_array_to_nxvar(pwoa)
        dark = self._reshape_data_array_to_nxvar(dark)

        return (pwa,pwoa,dark)
    
    def strip_shot_idx_axis(self,*args):
        out = []
        for arg in args:
            arg: np.ndarray
            if arg.shape[self.N_xvars] == 1:
                arg = arg.reshape(*self.xvardims,*arg.shape[(self.N_xvars+1):])
            out.append(arg)
        return out

    def _reshape_data_array_to_nxvar(self,ndarray):
        """Accepts an array of images of length equal to the number of shots in
        the order they were taken. Reshapes them to shape (n1,n2,...,nN,px,py),
        where ni is the length of the ith xvar.

        Args:
            ndarray (np.ndarray): an image array of shape (N,...), where N = the product of
        the lengths of all the xvars (= the number of shots), and px and py are
        the size of the image axes in pixels.
        """
        ndarray = ndarray.reshape(*self.xvardims,
                                  self.params.N_pwa_per_shot,
                                  *ndarray.shape[2:])
        return ndarray

    def _unshuffle_struct(self,
                          struct,
                          only_treat_first_Nvar_axes=False,
                          reshuffle=False):

        # only unshuffle if list has been shuffled
        if np.any(self.sort_idx):
            protected_keys = ['xvarnames','sort_idx','images',
                              'image_timestamps','sort_N','sort_idx',
                              'xvars','N_repeats','N_shots',
                              'N_shots_with_repeats','scan_xvars',
                              'xvardims','data']
            ks = struct.__dict__.keys()
            sort_ks = [k for k in ks if k not in protected_keys]
            for k in sort_ks:
                var = vars(struct)[k]
                if only_treat_first_Nvar_axes:
                    exclude_dims = np.ndim(var) - len(self.scan_xvars)
                else: exclude_dims = 0
                var = self._unshuffle_ndarray(var,
                                              exclude_dims=exclude_dims,
                                              reshuffle=reshuffle)
                vars(struct)[k] = var
    
    def _unshuffle_ndarray(self,var,
                           exclude_dims=0,
                           reshuffle=False):
        if isinstance(var,list):
            var = np.array(var)
        if isinstance(var,np.ndarray):
            sdims = self._dims_to_sort(var,exclude_dims)
            for dim in sdims:
                N = var.shape[dim]
                if N in self.sort_N:
                    i = np.where(np.array(self.sort_N) == N)[0][0]
                    shuf_idx = self.sort_idx[i]
                    shuf_idx = shuf_idx[shuf_idx >= 0].astype(int) # remove padding [-1]s
                    if not reshuffle:
                        unshuf_idx = np.zeros_like(shuf_idx)
                        unshuf_idx[shuf_idx] = np.arange(N)
                    else:
                        unshuf_idx = shuf_idx
                    var = var.take(unshuf_idx,dim)
        return var

    def _unshuffle_scopedata_dict(self,
                                  scope_data,
                                  reshuffle=False):
            for k in scope_data.keys():
                for ch in scope_data[k].keys():
                    for s in ['t','v']:
                        y = vars(scope_data[k][ch])[s]
                        exclude_dims = np.ndim(y) - len(self.scan_xvars)
                        y = self._unshuffle_ndarray(y,
                                                    exclude_dims=exclude_dims,
                                                    reshuffle=reshuffle)
                        vars(scope_data[k][ch])[s] = y

    def _dims_to_sort(self,var,exclude_dims=0):
        """For a given ndarray (var), determine which axes should be unshuffled.
        Can specify exclude_dims in order to prevent that many axes (counted
        from the deepest axis) from being unshuffled. For example, for an
        array of images, one would want to specify exclude_dims = 2, to
        prevent unshuffling of the pixels.

        Args:
            var (np.ndarray): The ndarray whose axes should be unshuffled.
            exclude_dims (int, optional): The number of axes (from the end) to
            exclude from unshuffling. Defaults to 0.

        Returns:
            np.array[int]: The indices of the axes that should be checked for
            unshuffling.
        """        
        ndims = var.ndim
        last_dim_to_sort = ndims - exclude_dims
        if last_dim_to_sort < 0: last_dim_to_sort = 0
        dims_to_sort = np.arange(0,last_dim_to_sort)
        return dims_to_sort
        
    # def shuffle_derived(self):
    #     '''
    #     Loop through all the attributes of params which are not in the list of
    #     protected keys. For each attribute which has a dimension of size equal
    #     to the length of one of the xvars specified in xvarnames, scramble that
    #     axis of the attribute in the same way that the xvar of matching length
    #     was scrambled.
    #     '''
    #     sort_N = self.sort_N
    #     if not isinstance(sort_N,np.ndarray):
    #         sort_N = np.array(sort_N)
    #     sort_idx = self.sort_idx

    #     protected_keys = ['xvarnames','sort_idx','images','image_timestamps','sort_N','sort_idx','xvars','N_repeats','N_shots_with_repeats']
    #     # get a list of the variable keys (that are not protected)
    #     ks = self.params.__dict__.keys()
    #     sort_ks = [k for k in ks if k not in protected_keys if k not in self.xvarnames]
    #     # loop over the keys
    #     for k in sort_ks:
    #         # get the value of the attribute with that key
    #         var = vars(self.params)[k]
    #         # cast arrays as np.ndarrays
    #         if isinstance(var,list):
    #             var = np.array(var)
    #         if isinstance(var,np.ndarray):
    #             # get a list of the dimensions to check for sorting, loop over them
    #             sdims = self._dims_to_sort(var)
    #             for dim in sdims:
    #                 N = var.shape[dim]
    #                 # check to see if this dimension is of a length which matches one of the xvars
    #                 # (sort_N is a list of the lengths of the xvars)
    #                 if N in sort_N:
    #                     # if this dim's length matches that of one of the xvars,
    #                     # grab the index of the match
    #                     i = np.where(sort_N == N)[0][0]
    #                     # get the indices used to shuffle the matching xvar
    #                     shuf_idx = sort_idx[i]
    #                     # remove padding [-1]s (added since the shuffling idx
    #                     # have to be the same length in the hdf5 later)
    #                     shuf_idx = shuf_idx[shuf_idx >= 0].astype(int)
    #                     # scramble the var along the this dimension according to
    #                     # the shuffling idx 
    #                     var = var.take(shuf_idx,dim)
    #                     # save the shuffled variable into params
    #                     vars(self.params)[k] = var