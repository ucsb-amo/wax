from artiq.experiment import *
import numpy as np

from waxa.base import xvar
from waxa.data import RunInfo
from waxa.dummy.camera_params import CameraParams
from waxx.util.live_od import CameraClient

from artiq.language.core import kernel_from_string, now_mu
from artiq.experiment import delay

RPC_DELAY = 10.e-3

dv = -100.
dvlist = np.array([])

def nothing():
    pass

class Scanner():
    def __init__(self):

        from waxx.config.expt_params import ExptParams
        self.params = ExptParams()
        self.run_info = RunInfo()
        self.camera_params = CameraParams()

        self.live_od_client = CameraClient(None, None)

        self.xvarnames = []
        self.scan_xvars = []
        self.Nvars = 0
        
        self.update_nvars()
        self.compute_new_derived = nothing

        from waxx.control.artiq.dummy_core import DummyCore
        self.core = DummyCore()

        self._xvar_writer_floats = []
        self._xvar_writer_int32s = []
        self._xvar_writer_int64s = []
        self._xvar_writer_arrays = []

        self._param_keylist_floats = []
        self._param_keylist_int32s = []
        self._param_keylist_int64s = []
        self._param_keylist_arrays = []

        self._dummy_array = np.zeros(10000,dtype=float)
        self._N = 0

    def logspace(self,start,end,n):
        return np.logspace(np.log10(start),np.log10(end),int(n))

    def xvar(self,key,values):
        """Adds an xvar to the experiment.

        Args:
            key (str): The key of the ExptParams attribute to scan.
            values (ndarray): Values to scan over. Can be n-dimensional, scan will step over first index.
        """
        this_xvar = xvar(key,values,position=len(self.scan_xvars))
        if this_xvar.key in [x.key for x in self.scan_xvars]:
            raise ValueError(f"xvar of key {this_xvar.key} is assigned more than once.")
        self.scan_xvars.append(this_xvar)
        self.xvarnames.append(this_xvar.key)
        # check if params has this xvar key already -- if not, add it
        self.new_param_check(this_xvar)
        self.update_nvars()

    def new_param_check(self,xvar):
        self.param_police(xvar)
        
        params_keylist = list(self.params.__dict__.keys())
        if xvar.key not in params_keylist:
            # set value to a single value (vs list), it will be overwritten per shot in scan
            vars(self.params)[xvar.key] = xvar.values[0]

    def param_police(self,xvar):
        forbidden_chars = [":",",","."," ","-","+","(",")","@","#","$","%","^","&","*","=","!","[","]",";","/","\\","`","~"]
        if any([fc in xvar.key for fc in forbidden_chars]):
            raise ValueError("Key contains forbidden characters.")

    def update_nvars(self):
        """Updates the number of xvars to be scanned.
        """
        self.Nvars = len(self.scan_xvars)

    @kernel
    def scan_kernel(self):
        """The kernel function to be scanned in the experiment. Usually
        overloaded in kexp.Base.
        
        It should correspond to a single "shot" (single set of images to
        generate one OD).

        The scan kernel should accept no arguments. 
        
        Any parameters being scanned should be referenced in the scan kernel as
        an attribute of the experiment parameters attribute of the experiment
        class.
        """
        pass

    @kernel
    def pre_scan(self):
        """This method is run in scan before the scan loop.
        Usually overloaded in kexp.Base.
        """        
        pass

    @kernel
    def init_scan_kernel(self):
        """This method is run between each shot just before scan_kernel.
        Usually overloaded in kexp.Base.
        """
        pass

    @kernel
    def cleanup_scan_kernel(self):
        """This method is run just after each scan_kernel completes.
        Usually overloaded in kexp.Base.
        """
        pass

    @kernel
    def post_scan(self):
        """This method is run just after the whole scan completes.
        Usually overloaded in kexp.Base.
        """

    @kernel
    def scan(self):
        """
        Runs the scan_kernel function for each value of the xvars specified.
        
        The xvars are scanned as if looping over nested for loops, with the last
        xvar as the innermost loop.

        On each step of the scan, the host ExptParams is updated with the next
        values of the xvars and derived parameters are recomputed. Then, the
        updated host ExptParams values are written into the corresponding kernel
        ExptParams.
        """

        self.pre_scan()

        scanning = True

        while scanning:

            self._check_data_file_exists()
            
            self.core.wait_until_mu(now_mu())
            self.update_params_from_xvars()

            self.write_host_params_to_kernel()
            
            self.live_od_client.send_xvars(self.scan_xvars)
            self.core.break_realtime()

            # overloaded in kexp.Base
            self.init_scan_kernel()
            self.core.break_realtime()

            # overloaded by user per experiment
            self.scan_kernel()

            # overloaded in kexp.Base
            self.cleanup_scan_kernel()

            delay(self.params.t_recover)
            self.core.break_realtime()

            self.core.wait_until_mu(now_mu())
            scanning = self.step_scan()

            self.core.break_realtime()

        self.post_scan()

    def update_params_from_xvars(self):
        """Updates the host ExptParams attributes and recomputes derived
        parameters according to the current values of the scanned xvars.

        Does not update the values of the kernel ExptParams -- to do so, use
        write_host_params_to_kernel().
        """

        # update each xvar parameter in the host params
        for xvar in self.scan_xvars:
            vars(self.params)[xvar.key] = xvar.values[xvar.counter]
        # update derived params in the host params
        self.params.compute_derived()
        self.compute_new_derived()

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
            
    def fetch_float(self,i) -> TFloat:
        """Returns the value of the ith experiment parameter with datatype
        float.

        Args:
            i (int): index of the ith float experiment paramter in the list
            self._param_keylist_floats.

        Returns:
            TFloat: The value of the ith float ExptParam attribute.
        """        
        return vars(self.params)[self._param_keylist_floats[i]]
    
    def fetch_array(self,i) -> TTuple([TInt32,TArray(TFloat)]):
        """Returns the value of the ith experiment parameter with datatype
        ndarray.

        Args:
            i (int): index of the ith ndarray experiment paramter in the list
            self._param_keylist_arrays.

        Returns:
            TFloat: The value of the ith ndarray ExptParam attribute.
            TInt: length of the array
        """    
        N = len(vars(self.params)[self._param_keylist_arrays[i]])
        self._dummy_array[0:N] = vars(self.params)[self._param_keylist_arrays[i]]
        return (N, self._dummy_array)
    
    def fetch_int64(self,i) -> TInt64:
        """Returns the value of the ith experiment parameter with datatype
        int64.

        Args:
            i (int): index of the ith ndarray experiment paramter in the list
            self._param_keylist_int64s.

        Returns:
            TFloat: The value of the ith int64 ExptParam attribute.
        """      
        return vars(self.params)[self._param_keylist_int64s[i]]
    
    def fetch_int32(self,i) -> TInt32:
        """Returns the value of the ith experiment parameter with datatype
        int32.

        Args:
            i (int): index of the ith ndarray experiment paramter in the list
            self._param_keylist_int32s.

        Returns:
            TFloat: The value of the ith int32 ExptParam attribute.
        """     
        return vars(self.params)[self._param_keylist_int32s[i]]

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

    def step_scan(self,idx=0) -> TBool:
        '''
        Advances the counters of the xvars to the next step in the scan.

        Advances counters as if the xvars were looped over in nested for loops,
        with the last xvar being the innermost loop.
        '''
        out = True
        xvars = list(reversed(self.scan_xvars))
        last_xvar_idx = self.Nvars - 1
        last_xval_idx = xvars[idx].values.shape[0] - 1
        if idx < self.Nvars:
            if xvars[idx].counter == last_xval_idx:
                if idx != last_xvar_idx:
                    xvars[idx].counter = 0
                    out = self.step_scan(idx+1)
                else:
                    out = False
            else:
                xvars[idx].counter += 1
        return out

    def cleanup_scanned(self):
        """
        Sets the parameters in ExptParams to the lists that were used to take
        the data. 
        
        These are put in in the order the data was taken -- no unshuffling is
        done. 
        
        This is good for recordkeeping, and ensures backward compatability with
        analysis code.
        """
        for xvar in self.scan_xvars:
            vars(self.params)[xvar.key] = xvar.values
        try:
            self.params.compute_derived()
            self.compute_new_derived()
        except Exception as e:
            print(e)
            print('Derived parameters were not updated.')

    def cleanup_image_count(self):
        # dummy, overloaded by kexp.image.cleanup_image_count
        pass

    def init_xvars(self, shuffle=True, N_repeats=[]):
        from waxa import img_types
        if self.run_info.imaging_type == img_types.ABSORPTION:
            if self.params.N_pwa_per_shot > 1:
                print("You indicated more than one PWA per shot, but the analysis is set to absorption imaging. Setting # PWA to 1.")
            self.params.N_pwa_per_shot = 1

        if not self.xvarnames:
            self.xvar("dummy",[0])
        if self.xvarnames and not self.scan_xvars:
            for key in self.xvarnames:
                self.xvar(key,vars(self.params)[key])
        self.plug_in_xvars()

        self.repeat_xvars(N_repeats=N_repeats)
        
        if shuffle:
            self.shuffle_xvars()
        
        self.params.N_img = self.get_N_img()
        self.prepare_image_array()

        self.params.compute_derived()
        self.compute_new_derived()

        self.xvardims = [len(xvar.values) for xvar in self.scan_xvars]
        if hasattr(self,'scope_data'):
            self.scope_data.xvardims = self.xvardims

        self.generate_assignment_kernels()

    def prepare_image_array(self):
        if self.run_info.save_data:
            # print(self.camera_params.camera_type)
            if self.camera_params.camera_type == 'andor':
                dtype = np.uint16
            elif self.camera_params.camera_type == 'basler':
                dtype = np.uint8
            else:
                dtype = np.uint8
            self.images = np.zeros((self.params.N_img,)+self.camera_params.resolution,dtype=dtype)
            self.image_timestamps = np.zeros((self.params.N_img,))
        else:
            self.images = np.array([0])
            self.image_timestamps = np.array([0])

    def get_N_img(self):
        """
        Computes the number of images to be taken during the sequence from the
        length of the specified xvars, stores in self.params.N_img. For
        absorption imaging, 3 images per shot. For fluorescence imaging,
        variable pwa images (ExptParams.N_pwa_per_shot, default = 1), then 1
        each pwoa and dark images.
        """                
        N_img = 1
        msg = ""

        for xvar in self.scan_xvars:
            N_img = N_img * xvar.values.shape[0]
            msg += f" {xvar.values.shape[0]} values of {xvar.key}."
        self.params.N_shots_with_repeats = N_img

        msg += f" {N_img} total shots."

        ### I have no idea what this is for. ###
        if isinstance(self.params.N_repeats,list):
            if len(self.params.N_repeats) == 1:
                N_repeats = self.params.N_repeats[0]
            else:
                N_repeats = np.prod(self.params.N_repeats)
        else:
            N_repeats = 1
        self.params.N_shots = int(N_img / N_repeats)
        ###

        from waxa import img_types
        if self.run_info.imaging_type == img_types.ABSORPTION:
            images_per_shot = 3
        else:
            images_per_shot = self.params.N_pwa_per_shot + 2

        N_img = images_per_shot * N_img # 3 images per value of independent variable (xvar)

        msg += f" {N_img} total images expected."
        print(msg)
        return N_img