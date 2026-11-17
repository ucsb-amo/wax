"""Copies the host's params into the kernel before each shot.

Why this exists: ARTIQ bakes the value of every attribute a kernel reads into
the compiled kernel. A scanned param changes every shot, so before each shot
the kernel asks the host for the current value of every param (one RPC each)
and assigns it to its own copy.

How: at finish_prepare, ``generate_assignment_kernels`` sorts the params by
type and builds one tiny kernel per param (``self.params.<key> = value``).
Before each shot, ``write_host_params_to_kernel`` loops over them.

Only params whose value is an int, a float or a numpy array are copied. A
param of any other type (bool, str, list) keeps the value it had when the
kernel was compiled, for the whole run.
"""
from artiq.experiment import *
from artiq.language.core import kernel_from_string
import numpy as np


class ParamSync:

    def _init_param_sync(self):
        # one writer kernel per param, and the param's key at the same index
        self._xvar_writer_floats = []
        self._xvar_writer_int32s = []
        self._xvar_writer_int64s = []
        self._xvar_writer_arrays = []

        self._param_keylist_floats = []
        self._param_keylist_int32s = []
        self._param_keylist_int64s = []
        self._param_keylist_arrays = []

        # every array param travels through this one buffer (fetch_array)
        self._dummy_array = np.zeros(10000,dtype=float)
        self._N = 0

    def generate_assignment_kernels(self):
        """Generates a list of kernel functions for each param datatype (int32,
        int64, ndarray, and float ) -- one for each ExptParam attribute. These
        can be called in the kernel to update the kernel experiment params with
        values from the host ExptParams returned by an RPC (the "fetch" functions).
        """

        keylist = list(self.params.__dict__.keys())
        for key in keylist:
            bodycode = f"self.params.{key} = value"
            dtype = str(type(vars(self.params)[key]))

            if 'int' in dtype:
                if 'numpy.int64' in dtype:
                    self._param_keylist_int64s.append(key)
                    self._xvar_writer_int64s.append( kernel_from_string(["self","value"],bodycode) )
                else:
                    self._param_keylist_int32s.append(key)
                    self._xvar_writer_int32s.append( kernel_from_string(["self","value"],bodycode) )
            elif 'float' in dtype:
                self._param_keylist_floats.append(key)
                self._xvar_writer_floats.append( kernel_from_string(["self","value"],bodycode) )
            elif 'ndarray' in dtype:
                self._param_keylist_arrays.append(key)
                self._xvar_writer_arrays.append( kernel_from_string(["self","value"],bodycode) )

    @kernel
    def write_host_params_to_kernel(self):
        """Loops over all experiment params, and assigns the values of the
        kernel ExptParam attributes to those of the of the host ExptParam
        attributes.

        Must have run generate_assignment_kernels() in build first.
        """
        int32val = np.int32(1)
        int64val = np.int64(1)
        floatval = 0.1
        # self._dummy_array[:] = 0.
        arrval = np.array([1.])

        for idx in range(len(self._param_keylist_int32s)):
            int32val = self.fetch_int32(idx)
            self._xvar_writer_int32s[idx](self,int32val)

        for idx in range(len(self._param_keylist_int64s)):
            int64val = self.fetch_int64(idx)
            self._xvar_writer_int64s[idx](self,int64val)

        for idx in range(len(self._param_keylist_floats)):
            floatval = self.fetch_float(idx)
            self._xvar_writer_floats[idx](self,floatval)

        for idx in range(len(self._param_keylist_arrays)):
            N, self._dummy_array = self.fetch_array(idx)
            self._xvar_writer_arrays[idx](self,self._dummy_array[0:N])

    # The RPCs the kernel fetches values with: the value of the i-th param of
    # that type, in the order generate_assignment_kernels listed them.

    def fetch_float(self,i) -> TFloat:
        return vars(self.params)[self._param_keylist_floats[i]]

    def fetch_array(self,i) -> TTuple([TInt32,TArray(TFloat)]):
        """Returns (length, buffer): the array sits in the first ``length``
        cells of the shared buffer."""
        N = len(vars(self.params)[self._param_keylist_arrays[i]])
        self._dummy_array[0:N] = vars(self.params)[self._param_keylist_arrays[i]]
        return (N, self._dummy_array)

    def fetch_int64(self,i) -> TInt64:
        return int(round(vars(self.params)[self._param_keylist_int64s[i]]))

    def fetch_int32(self,i) -> TInt32:
        return int(round(vars(self.params)[self._param_keylist_int32s[i]]))
