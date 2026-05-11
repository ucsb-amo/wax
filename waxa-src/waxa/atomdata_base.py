import numpy as np
import datetime
import h5py
import time

from waxa.image_processing.compute_ODs import compute_OD
from waxa.image_processing.compute_gaussian_cloud_params import fit_gaussian_sum_dist
from waxa.roi import ROI
from waxa.data.data_saver import DataSaver
from waxa.base import Dealer, xvar
from waxa.data.server_talk import server_talk as st
from waxa.helper.datasmith import *
from waxa.data.run_info import RunInfo
from waxa.config.expt_params import ExptParams
from waxa.dummy.camera_params import CameraParams
from waxa.config.img_types import img_types as img
    
class ScopeTraceArray():
    def __init__(self, scope_key, ch, t, v):
        self.scope_key = scope_key
        self.ch = ch
        self.t = t
        self.v = v

class _RepeatDataVault():
    def __init__(self):
        self.keys = []

class _RepeatSEMDataProxy():
    def __init__(self, source, sem_divisor):
        self._source = source
        self._sem_divisor = sem_divisor

    def __getattr__(self, key):
        value = getattr(self._source, key)
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            return value / self._sem_divisor
        return value

    @property
    def keys(self):
        return self._source.keys

class _RepeatZeroDataProxy():
    def __init__(self, source):
        self._source = source

    def __getattr__(self, key):
        value = getattr(self._source, key)
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            return np.zeros_like(value)
        if np.isscalar(value) and np.issubdtype(np.asarray(value).dtype, np.number):
            return np.zeros_like(value)
        return value

    @property
    def keys(self):
        return self._source.keys

class _RepeatZeroAtomdataProxy():
    def __init__(self, source):
        self._source = source

    def _zero_like_scope_data(self, scope_data):
        zero_scope = dict()
        for scope_key, channel_dict in scope_data.items():
            zero_scope[scope_key] = dict()
            for ch, trace in channel_dict.items():
                t = trace.t
                v = trace.v
                if isinstance(t, np.ndarray) and np.issubdtype(t.dtype, np.number):
                    t = np.zeros_like(t)
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                    v = np.zeros_like(v)
                zero_scope[scope_key][ch] = ScopeTraceArray(scope_key, ch, t, v)
        return zero_scope

    def __getattr__(self, name):
        if name == 'data':
            return _RepeatZeroDataProxy(self._source.data)
        if name == 'scope_data' and hasattr(self._source, 'scope_data'):
            return self._zero_like_scope_data(self._source.scope_data)

        value = getattr(self._source, name)
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            return np.zeros_like(value)
        if np.isscalar(value) and np.issubdtype(np.asarray(value).dtype, np.number):
            return np.zeros_like(value)
        return value

class _RepeatPassthroughAtomdataProxy():
    def __init__(self, source):
        self._source = source

    def __getattr__(self, name):
        return getattr(self._source, name)

class _RepeatLazyDataProxy():
    def __init__(self, source_data, parent_proxy):
        self._source_data = source_data
        self._parent_proxy = parent_proxy

    def __getattr__(self, key):
        return self._parent_proxy._get_data_field(key)

    @property
    def keys(self):
        return self._source_data.keys

class _RepeatLazyScopeChannelProxy():
    def __init__(self, scope_key, ch, source_trace, parent_proxy):
        self.scope_key = scope_key
        self.ch = ch
        self._source_trace = source_trace
        self._parent_proxy = parent_proxy

    def __getattr__(self, name):
        if name in ['t', 'v']:
            return self._parent_proxy._get_scope_field(self.scope_key, self.ch, name)
        return getattr(self._source_trace, name)

class _RepeatLazyScopeDataProxy():
    def __init__(self, source_scope_data, parent_proxy):
        self._source_scope_data = source_scope_data
        self._parent_proxy = parent_proxy
        self._scope_cache = {}

    def __getitem__(self, scope_key):
        if scope_key not in self._scope_cache:
            channel_dict = {}
            for ch, trace in self._source_scope_data[scope_key].items():
                channel_dict[ch] = _RepeatLazyScopeChannelProxy(scope_key, ch, trace, self._parent_proxy)
            self._scope_cache[scope_key] = channel_dict
        return self._scope_cache[scope_key]

    def keys(self):
        return self._source_scope_data.keys()

    def items(self):
        for scope_key in self._source_scope_data.keys():
            yield scope_key, self[scope_key]

    def __iter__(self):
        return iter(self._source_scope_data)

class _RepeatLazyParamsProxy():
    def __init__(self, source_params, overrides):
        self._source_params = source_params
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._source_params, name)

class _RepeatLazyStatAtomdataProxy():
    def __init__(self, source, xvar_idx, n_repeats, stat_kind, shared_cache):
        self._source = source
        self._xvar_idx = xvar_idx
        self._n_repeats = n_repeats
        self._stat_kind = stat_kind
        self._shared_cache = shared_cache

        self.xvarnames = source.xvarnames
        self.xvars = [np.asarray(x) for x in source.xvars]
        self.Nvars = len(self.xvars)
        self.xvardims = np.array([len(x) for x in self.xvars], dtype=int)

        self.xvars[xvar_idx] = source._get_unique_repeated_xvar(xvar_idx, n_repeats)
        self.xvardims[xvar_idx] = len(self.xvars[xvar_idx])

        param_overrides = {
            'N_repeats': np.ones(self.Nvars, dtype=int)
        }
        for idx, key in enumerate(self.xvarnames):
            param_overrides[key] = self.xvars[idx]
        self.params = _RepeatLazyParamsProxy(source.params, param_overrides)
        self.p = self.params

        self.sort_idx = np.array([])
        self.sort_N = np.array([])

        self.data = _RepeatLazyDataProxy(source.data, self)
        if hasattr(source, 'scope_data'):
            self.scope_data = _RepeatLazyScopeDataProxy(source.scope_data, self)

    def _select_stat(self, mean_val, std_val):
        if self._stat_kind == 'mean':
            return mean_val
        return std_val

    def _compute_and_cache(self, cache_key, value):
        if cache_key not in self._shared_cache:
            mean_val, std_val = self._source._reduce_repeat_ndarray_mean_std(
                value,
                self._xvar_idx,
                self._n_repeats,
            )
            self._shared_cache[cache_key] = (mean_val, std_val)
        return self._shared_cache[cache_key]

    def _get_data_field(self, key):
        value = getattr(self._source.data, key)
        if self._source._is_scan_shaped_numeric_array(value):
            cache_key = ('data', key)
            mean_val, std_val = self._compute_and_cache(cache_key, value)
            return self._select_stat(mean_val, std_val)
        return value

    def _get_scope_field(self, scope_key, ch, axis):
        trace = self._source.scope_data[scope_key][ch]
        value = getattr(trace, axis)
        if self._source._is_scan_shaped_numeric_array(value):
            cache_key = ('scope', scope_key, ch, axis)
            mean_val, std_val = self._compute_and_cache(cache_key, value)
            return self._select_stat(mean_val, std_val)
        return value

    def __getattr__(self, name):
        value = getattr(self._source, name)
        if self._source._is_scan_shaped_numeric_array(value):
            cache_key = ('attr', name)
            mean_val, std_val = self._compute_and_cache(cache_key, value)
            return self._select_stat(mean_val, std_val)
        return value

def format_scope_data(dataset, old_method=False):
    """Scope data formatter optimized for fast loading.
    
    Data MUST be materialized while h5py file is open (references become invalid after file close).
    Uses direct h5py indexing for efficient per-channel loading instead of np.take on full array.
    """
    scope_dict = dict()
    
    for scope_key in dataset.keys():
        this_scope_data = dict()
        if old_method:
            # Old format: single array with (channels, time/value, shots)
            data_array = dataset[scope_key][()]
            data_array: np.ndarray
            for ch in range(data_array.shape[-3]):
                ch_data = np.take(data_array, ch, -3)
                t = np.take(ch_data, 0, axis=-2)
                v = np.take(ch_data, 1, axis=-2)
                this_scope_data[ch] = ScopeTraceArray(scope_key, ch, t, v)
            scope_dict[scope_key] = this_scope_data
        if not old_method:
            # New format: separate 't' and 'v' datasets
            t = dataset[scope_key]['t'][()]  # Materialize now
            # Materialize once while the HDF5 file is open; channel axis is -2.
            v_data = dataset[scope_key]['v'][()]
            n_channels = v_data.shape[-2]

            for ch in range(n_channels):
                v = np.take(v_data, ch, axis=-2)
                this_scope_data[ch] = ScopeTraceArray(scope_key, ch, t, v)
            
            scope_dict[scope_key] = this_scope_data

    return scope_dict

def unpack_group(file,group_key,obj):
    """Looks in an open h5 file in the group specified by key, and iterates over
    every dataset in that h5 group, and for each dataset assigns an attribute of
    the object obj" with that dataset's key and value.

    Args:
        file (h5py.File, h5py dataset): An open h5py file object or dataset.
        group_key (str): The key of the group in the h5py file.
        obj (object): Any object to be populated with attributes by the fields
        in the provided dataset. 
    """            
    g = file[group_key]
    keys = list(g.keys())
    for k in keys:
        vars(obj)[k] = g[k][()]

class analysis_tags():
    """A simple container to hold analysis tags for analysis logic.
    """    
    def __init__(self,roi_id,imaging_type):
        self.roi_id = roi_id
        self.imaging_type = imaging_type
        self.xvars_shuffled = False
        self.transposed = False
        self.averaged = False
        self.repeats_reassigned = False

class atom_number_apd():
    def __init__(self, n_up, n_down):
        self.n_up = n_up
        self.n_down = n_down
        self.n_total = n_up + n_down
        self.frac_up = n_up / (self.n_total)
        self.frac_down = n_up / (self.n_total)

    def calibration(self):
        # for later
        pass

class expt_code():
    """A simple container to organize experiment text.
    """    
    def __init__(self,
                 experiment,
                 params,
                 cooling,
                 imaging,
                 control):
        self.experiment = experiment
        self.params = params
        self.cooling = cooling
        self.imaging = imaging
        self.control = control

class atomdata_base():
    '''
    Use to store and do basic analysis on data for every experiment.
    '''
    
    def __init__(self,
                idx=0,
                roi_id=None,
                path = "",
                lite = False,
                skip_saved_roi = False,
                transpose_idx = [],
                avg_repeats = False,
                server_talk = st()):
        '''
        Returns the atomdata stored in the `idx`th newest file at `path`.

        Parameters
        ----------
        idx: int
            If a positive value is specified, it is interpreted as a run_id (as
            stored in run_info.run_id), and that data is found and loaded. If zero
            or a negative number are given, data is loaded relative to the most
            recent dataset (idx=0).
        roi_id: None, int, or string
            Specifies which crop to use. If roi_id=None, defaults to the ROI saved in
            the data if it exists, otherwise prompts the user to select an ROI using
            the GUI. If an int, interpreted as an run ID, which will be checked for
            a saved ROI and that ROI will be used. If a string, interprets as a key
            in the roi.xlsx document in the PotassiumData folder.
        path: str
            The full path to the file to be loaded. If not specified, loads the file
            as dictated by `idx`.
        skip_saved_roi: bool
            If true, ignore saved ROI in the data file.

        Returns
        -------
        ad: atomdata
        '''

        self._lite = lite
        # When loading lite data, ignore any passed roi_id since lite files
        # are already pre-cropped to a specific ROI at creation time.
        if lite:
            roi_id = None

        # Lightweight profiling aid for load/analysis latency investigations.
        self._timing_enabled = True
        self._timing = {}

        self.avg = None
        self.std = None
        self.sem = None
        self._repeat_sem_source = None
        self._repeat_sem_divisor = None
        self._repeat_zero_proxy = None
        self._repeat_lazy_stat_context = None
        self._data_file_path = None
        self._saved_roi_from_file = False

        self.server_talk = server_talk

        t_init = time.perf_counter()
        t_stage = time.perf_counter()
        self._load_data(idx, path, lite=lite, roi_id=roi_id)
        self._timing['init_load_data_s'] = time.perf_counter() - t_stage

        ### Helper objects
        t_stage = time.perf_counter()
        self._ds = DataSaver()
        self._dealer = self._init_dealer()
        self._analysis_tags = analysis_tags(roi_id,self.run_info.imaging_type)
        self.roi = ROI(run_id = self.run_info.run_id,
                       roi_id = roi_id,
                       use_saved_roi = not skip_saved_roi,
                       lite = self._lite,
                       server_talk=self.server_talk,
                       current_file_path=self._data_file_path,
                       current_saved_roi=self._saved_roi_from_file)
        self._timing['init_setup_helpers_roi_s'] = time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        self._unshuffle_old_data()
        self._timing['init_unshuffle_old_data_s'] = time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        self._initial_analysis(transpose_idx,avg_repeats)
        self._timing['init_initial_analysis_s'] = time.perf_counter() - t_stage
        self._timing['init_total_s'] = time.perf_counter() - t_init

        if self._timing_enabled:
            print(
                (
                    "[atomdata timing] init total={:.3f}s | load_data={:.3f}s | "
                    "setup+roi={:.3f}s | unshuffle_old={:.3f}s | initial_analysis={:.3f}s"
                ).format(
                    self._timing['init_total_s'],
                    self._timing['init_load_data_s'],
                    self._timing['init_setup_helpers_roi_s'],
                    self._timing['init_unshuffle_old_data_s'],
                    self._timing['init_initial_analysis_s'],
                )
            )

    ###
    def recrop(self,roi_id=None,use_saved=False):
        """Selects a new ROI and re-runs the analysis. Uses the same logic as
        kexp.ROI.load_roi.

        For lite atomdata, rebuilds the lite file from the full original data
        using the new ROI, then reloads and re-analyzes.

        Args:
            roi_id (None, int, or str): Specifies which crop to use. If None,
            defaults to the ROI saved in the data if it exists, otherwise
            prompts the user to select an ROI using the GUI. If an int,
            interpreted as an run ID, which will be checked for a saved ROI and
            that ROI will be used. If a string, interprets as a key in the
            roi.xlsx document in the PotassiumData folder.

            use_saved (bool): If False, ignores saved ROI and forces creation of
            a new one. Default is False.
        """
        if self._lite:
            # Get new ROI (prompts GUI if roi_id is None and no saved ROI).
            self.roi.load_roi(roi_id, use_saved)
            # Save the new ROI into the regular (non-lite) h5 file so
            # create_lite_copy can read it back.
            self.roi.save_roi_h5(lite=False, printouts=False)
            # Rebuild the lite file from the full original images.
            self.server_talk.create_lite_copy(
                self.run_info.run_id,
                roi_id=self.run_info.run_id,
                use_saved_roi=True,
            )
            # Reload the freshly-written lite data.
            self._load_data(self.run_info.run_id, "", lite=True)
            self._dealer = self._init_dealer()
            self._sort_images()
            self.compute_raw_ods()
            self.analyze_ods()
            self._refresh_repeat_statistics()
        else:
            self.roi.load_roi(roi_id, use_saved)
            self.analyze_ods()
            self._refresh_repeat_statistics()

    ### ROI management
    def save_roi_excel(self,key=""):
        self.roi.save_roi_excel(key)

    def save_roi_h5(self,printouts=False):
        self.roi.save_roi_h5(lite=self._lite,printouts=printouts)

    def create_lite_copy(self, roi_id=None, use_saved_roi=True):
        """Creates a lite (ROI-cropped) copy of this run's data file.

        By default uses the ROI already loaded in this atomdata instance. The
        current ROI is saved to the h5 file first so that create_lite_copy
        can read it back automatically.

        Parameters
        ----------
        roi_id : None, int, or str
            Override the ROI source. If None (default), uses the ROI currently
            stored in this atomdata instance. If an int, interpreted as a run
            ID whose saved ROI will be used. If a str, looks up the key in
            roi.xlsx.
        use_saved_roi : bool
            Passed through to server_talk.create_lite_copy. Has no effect when
            roi_id is None, since the current ROI is written to the file first.
        """
        if roi_id is None:
            self.roi.save_roi_h5(lite=self._lite, printouts=False)
            use_saved_roi = True
        self.server_talk.create_lite_copy(
            self.run_info.run_id,
            roi_id=roi_id,
            use_saved_roi=use_saved_roi,
        )

    ### Analysis

    def _initial_analysis(self,transpose_idx,avg_repeats):
        t0 = time.perf_counter()

        t_stage = time.perf_counter()
        self._sort_images()
        t_sort = time.perf_counter() - t_stage

        if transpose_idx:
            self._analysis_tags.transposed = True
            t_stage = time.perf_counter()
            self.transpose_data(transpose_idx=False,reanalyze=False)
            t_transpose = time.perf_counter() - t_stage
        else:
            t_transpose = 0.0

        t_stage = time.perf_counter()
        self.compute_raw_ods()
        t_compute_raw = time.perf_counter() - t_stage

        if avg_repeats:
            t_stage = time.perf_counter()
            self.avg_repeats(reanalyze=False)
            t_avg_repeats = time.perf_counter() - t_stage
        else:
            t_avg_repeats = 0.0

        t_stage = time.perf_counter()
        self.analyze_ods()
        t_analyze_ods = time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        self._refresh_repeat_statistics()
        t_repeat_stats = time.perf_counter() - t_stage
        t_total = time.perf_counter() - t0

        self._timing['initial_analysis_sort_images_s'] = t_sort
        self._timing['initial_analysis_transpose_s'] = t_transpose
        self._timing['initial_analysis_compute_raw_ods_s'] = t_compute_raw
        self._timing['initial_analysis_avg_repeats_s'] = t_avg_repeats
        self._timing['initial_analysis_analyze_ods_s'] = t_analyze_ods
        self._timing['initial_analysis_refresh_repeat_stats_s'] = t_repeat_stats
        self._timing['initial_analysis_total_s'] = t_total

        if self._timing_enabled:
            print(
                (
                    "[atomdata timing] initial_analysis total={:.3f}s | sort_images={:.3f}s | "
                    "transpose={:.3f}s | compute_raw_ods={:.3f}s | avg_repeats={:.3f}s | "
                    "analyze_ods={:.3f}s | refresh_repeat_stats={:.3f}s"
                ).format(
                    t_total,
                    t_sort,
                    t_transpose,
                    t_compute_raw,
                    t_avg_repeats,
                    t_analyze_ods,
                    t_repeat_stats,
                )
            )

    def analyze(self):
        t0 = time.perf_counter()

        t_stage = time.perf_counter()
        self.compute_raw_ods()
        t_compute_raw = time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        self.analyze_ods()
        t_analyze_ods = time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        self._refresh_repeat_statistics()
        t_repeat_stats = time.perf_counter() - t_stage
        t_total = time.perf_counter() - t0

        self._timing['analyze_compute_raw_ods_s'] = t_compute_raw
        self._timing['analyze_analyze_ods_s'] = t_analyze_ods
        self._timing['analyze_refresh_repeat_stats_s'] = t_repeat_stats
        self._timing['analyze_total_s'] = t_total

        if self._timing_enabled:
            print(
                (
                    "[atomdata timing] analyze total={:.3f}s | compute_raw_ods={:.3f}s | "
                    "analyze_ods={:.3f}s | refresh_repeat_stats={:.3f}s"
                ).format(
                    t_total,
                    t_compute_raw,
                    t_analyze_ods,
                    t_repeat_stats,
                )
            )

    def compute_raw_ods(self):
        """Computes the ODs. If not absorption analysis, OD = (pwa - dark)/(pwoa - dark).
        """        
        self.od_raw = compute_OD(self.img_atoms,self.img_light,self.img_dark,
                                 imaging_type=self._analysis_tags.imaging_type)

    def analyze_ods(self):
        """Crops ODs, computes sum_ods, gaussian fits to sum_ods, and populates
        fit results.
        """
        # Lite files store images already cropped to an ROI during creation.
        # Avoid applying ROI cropping a second time on load.
        if self._lite:
            self.od = self.od_raw
        else:
            self.od = self.roi.crop(self.od_raw)
        self.sum_od_x = np.sum(self.od,self.od.ndim-2)
        self.sum_od_y = np.sum(self.od,self.od.ndim-1)

        self.axis_camera_px_x = np.arange(self.sum_od_x.shape[-1])
        self.axis_camera_px_y = np.arange(self.sum_od_y.shape[-1])

        self.axis_camera_x = self.camera_params.pixel_size_m * self.axis_camera_px_x
        self.axis_camera_y = self.camera_params.pixel_size_m * self.axis_camera_px_y

        self.axis_x = self.axis_camera_x / self.camera_params.magnification
        self.axis_y = self.axis_camera_y / self.camera_params.magnification

        self.cloudfit_x = fit_gaussian_sum_dist(self.sum_od_x,self.camera_params)
        self.cloudfit_y = fit_gaussian_sum_dist(self.sum_od_y,self.camera_params)

        self._remap_fit_results()

        self.compute_apd_atom_number()

        if self._analysis_tags.imaging_type == img.ABSORPTION:
            self.compute_atom_number()

        self.integrated_od = np.sum(np.sum(self.od,-2),-1)

    def compute_apd_atom_number(self):
        if 'post_shot_absorption' in self.data.keys:
            v = self.data.post_shot_absorption
            if np.all(v == 0.):
                return
            
            v_up = v[:,0]
            v_down = v[:,1]
            v_light = v[:,2]
            v_dark = v[:,3]

            light_only = v_light - v_dark
            up_only = v_up - v_dark
            down_only = v_down - v_dark

            # Only keep physically valid points for log argument > 0
            ratio_up = np.where((up_only > 0) & (light_only > 0), up_only / light_only, np.nan)
            ratio_down = np.where((down_only > 0) & (light_only > 0), down_only / light_only, np.nan)

            number_up = -np.log(ratio_up)
            number_down = -np.log(ratio_down)

            # calibrate later

            self.atom_number_apd = atom_number_apd(number_up, number_down)

    def _sort_images(self):
        imgs_tuple = self._dealer.deal_data_ndarray(self.images)
        self.img_atoms = imgs_tuple[0]
        self.img_light = imgs_tuple[1]
        self.img_dark = imgs_tuple[2]

        img_timestamp_tuple = self._dealer.deal_data_ndarray(self.image_timestamps)
        self.img_timestamp_atoms = img_timestamp_tuple[0]
        self.img_timestamp_light = img_timestamp_tuple[1]
        self.img_timestamp_dark = img_timestamp_tuple[2]
        
        if self.params.N_pwa_per_shot > 1:
            self.xvarnames = np.append(self.xvarnames,'idx_pwa')
            self.xvars.append(np.arange(self.params.N_pwa_per_shot))
            self.xvardims = np.append(self.xvardims,self.params.N_pwa_per_shot)
            self.Nvars += 1
            np.append(self.sort_idx,np.arange(self.params.N_pwa_per_shot))
            if not self.params.N_pwa_per_shot in self.sort_N:
                np.append(self.sort_N,self.params.N_pwa_per_shot)
        else:
            self.img_atoms = self._dealer.strip_shot_idx_axis(self.img_atoms)[0]
            self.img_light = self._dealer.strip_shot_idx_axis(self.img_light)[0]
            self.img_dark = self._dealer.strip_shot_idx_axis(self.img_dark)[0]

            self.img_timestamp_atoms = self._dealer.strip_shot_idx_axis(self.img_timestamp_atoms)[0]
            self.img_timestamp_light = self._dealer.strip_shot_idx_axis(self.img_timestamp_light)[0]
            self.img_timestamp_dark = self._dealer.strip_shot_idx_axis(self.img_timestamp_dark)[0]

    ### Physics
    def compute_atom_number(self):
        self.atom_cross_section = 5.878324268151581e-13 # from kamo.Potassium39.get_cross_section
        dx_pixel = self.camera_params.pixel_size_m / self.camera_params.magnification
        
        self.atom_number_fit_area_x = self.fit_area_x * dx_pixel / self.atom_cross_section
        self.atom_number_fit_area_y = self.fit_area_y * dx_pixel / self.atom_cross_section

        self.atom_number_density = self.od * dx_pixel**2 / self.atom_cross_section  
        self.atom_number = np.sum(np.sum(self.atom_number_density,-2),-1)

    def slice_atomdata(self, which_shot_idx=0, which_xvar_idx=0, ignore_repeats=False):
        """Slices along a given xvar index at a particular value (which_shot_idx) of
        that xvar, and returns an atomdata of reduced dimensionality as if that
        variable had been held constant.

        If the data has repeats on the axis being sliced:
          - Multi-axis scans (Nvars > 1): the repeats are automatically moved one
            axis deeper before slicing, so the result retains all repeats on the
            adjacent axis.
          - Single-axis scans (Nvars == 1): all shots whose xvar value equals the
            value at which_shot_idx are returned together (i.e. all repeats of
            that unique value), unless ignore_repeats=True.

        If the data has repeats on a different axis, they are left untouched and
        appear on the same axis in the returned atomdata.

        Args:
            which_shot_idx (int or list): The index (or indices) into the xvar
            specified by which_xvar_idx to select. When selecting a single index
            from a multi-axis scan this reduces the dimensionality by one.
            which_xvar_idx (int): The index of the xvar axis to slice along.
            ignore_repeats (bool): Only relevant for single-axis scans with
            repeats. When True, return only the single shot at which_shot_idx
            rather than all repeats of the corresponding unique value.

        Returns:
            atomdata: The sliced atomdata object.
        """

        ad = self.__class__(self.run_info.run_id,
                       avg_repeats=self._analysis_tags.averaged,
                       roi_id=self.run_info.run_id)

        # Normalize which_shot_idx early so the rest of the logic uses ndarrays.
        which_shot_idx = ensure_ndarray(which_shot_idx, enforce_1d=True)

        # Detect whether any axis carries repeated xvar values.
        _has_repeats = False
        _repeat_axis = -1
        _n_repeats = 1
        if not self._analysis_tags.averaged:
            try:
                _repeat_axis, _n_repeats = ad._get_repeat_axis_info()
                _has_repeats = True
            except ValueError:
                pass

        if _has_repeats and _repeat_axis == which_xvar_idx:
            if ad.Nvars > 1:
                # Move repeats one axis deeper so the slice axis becomes clean.
                # Normalize which_shot_idx in case it references a repeated
                # position in the original (longer) axis.
                which_shot_idx = np.unique(which_shot_idx // _n_repeats)
                _target = (which_xvar_idx + 1
                           if which_xvar_idx + 1 < ad.Nvars
                           else which_xvar_idx - 1)
                ad.reassign_repeats(_target)
            elif not ignore_repeats:
                # Single-axis scan: expand the selection to include all repeats
                # for every selected index value (supports list inputs).
                xvals = np.asarray(ad.xvars[0])
                selected_vals = xvals[which_shot_idx]
                if np.issubdtype(xvals.dtype, np.number):
                    mask = np.any(
                        np.isclose(
                            xvals[:, None].astype(float),
                            np.asarray(selected_vals, dtype=float)[None, :],
                        ),
                        axis=1,
                    )
                else:
                    mask = np.isin(xvals, selected_vals)
                which_shot_idx = np.where(mask)[0]

        # N_repeats on the returned atomdata: stays non-1 whenever repeats
        # survive the slice (i.e. in every case except a single-axis slice
        # with ignore_repeats=True that strips the only repeat axis).
        _repeats_survive = _has_repeats and not (
            _repeat_axis == which_xvar_idx
            and ad.Nvars == 1
            and ignore_repeats
        )
        ad.params.N_repeats = _n_repeats if _repeats_survive else 1

        # replace the param for the xvar being sliced with the slice value
        vars(ad.params)[ad.xvarnames[which_xvar_idx]] = ad.xvars[which_xvar_idx][which_shot_idx]

        def remap_sort_idx_to_sequential(x):
            """
            Relabels the elements of x to sequential integers in the same order
            as np.sort(x) starting from 0, ignoring any -1s (which are treated as padding).
            The same number of padding -1s are added to the end of the array after remapping.
            """
            x = np.asarray(x)
            # Find non-padding elements
            valid_mask = x != -1
            valid_vals = x[valid_mask]
            unique_sorted = np.sort(np.unique(valid_vals))
            mapping = {val: i for i, val in enumerate(unique_sorted)}
            remapped = np.array([mapping[val] for val in valid_vals])
            # Pad with -1s to match original length
            n_pad = len(x) - len(remapped)
            if n_pad > 0:
                remapped = np.concatenate([remapped, -1 * np.ones(n_pad, dtype=int)])
            return remapped

        def grab_these_sort_idx(indices,which_xvar_idx=0):
            """
            Grabs the sort_idx elements corresponding to the indices in
            which_shot_idx, and returns them as a new array padded with -1s
            so that the returned array has the same length as the original sort_idx array.
            """
            taken = ad.sort_idx[which_xvar_idx][indices]
            n_pad = len(ad.sort_idx[which_xvar_idx]) - len(taken)
            if n_pad > 0:
                taken = np.concatenate([taken, -1 * np.ones(n_pad, dtype=ad.sort_idx[which_xvar_idx].dtype)])
            return taken

        # remove the xvars, xvarnames, and xvardims entry for that xvar
        keys = ['xvars','xvarnames','xvardims']
        # only remove the sort_N and sort_idx for this xvar if is the only one of
        # its length (otherwise another xvar also uses that sort_idx list)
        sliced_xvardim = ad.xvardims[which_xvar_idx]

        if not self._analysis_tags.averaged and ad.sort_idx.size != 0:
            shuffled = True
            sort_N_idx = np.where(ad.sort_N == sliced_xvardim)[0][0]
        else:
            shuffled = False
            
        if len(which_shot_idx) == 1 and ad.Nvars > 1:
            # if you are slicing out a single shot, remove the xvar to reduce
            # the dimensionality of the dataset
            if shuffled:
                if np.sum(ad.xvardims == sliced_xvardim) == 1:
                    ad.sort_N = remove_element_by_index(ad.sort_N, sort_N_idx)
                    ad.sort_idx = remove_element_by_index(ad.sort_idx, sort_N_idx)

            for k in keys:
                vars(ad)[k] = remove_element_by_index(vars(ad)[k],
                                                        which_xvar_idx)
            # decrement the number of variables by one
            ad.Nvars -= 1
        else:
            # otherwise, just slice the xvar without reducing atomdata dimensionality
            if shuffled:
                ad.sort_N[sort_N_idx] = len(which_shot_idx)
                ad.sort_idx[sort_N_idx] = grab_these_sort_idx(which_shot_idx,which_xvar_idx)
                # you left out some sort_idx elements, so remap the sort_idx to
                # sequential integers starting from 0.
                ad.sort_idx[sort_N_idx] = remap_sort_idx_to_sequential(ad.sort_idx[sort_N_idx])

            ad.xvars[which_xvar_idx] = np.take(ad.xvars[which_xvar_idx],
                                               indices=which_shot_idx,
                                               axis=0)
            ad.xvardims[which_xvar_idx] = len(ad.xvars[which_xvar_idx])

        def slice_ndarray(array):
            sliced_array = np.take(array,
                                   indices=which_shot_idx,
                                   axis=which_xvar_idx)
            if ad.Nvars < self.Nvars and sliced_array.shape[which_xvar_idx] == 1:
                sliced_array = np.squeeze(sliced_array, axis=which_xvar_idx)
            return sliced_array
        nd_keys = ['img_atoms','img_light','img_dark',
                'img_timestamp_atoms','img_timestamp_light','img_timestamp_dark']
        for k in nd_keys:
            vars(ad)[k] = slice_ndarray(vars(ad)[k])
        for k in self.data.keys:
            vars(ad.data)[k] = slice_ndarray(vars(self.data)[k])
        if hasattr(self,'scope_data'):
            for k in ad.scope_data.keys():
                for ch in ad.scope_data[k].keys():
                    ad.scope_data[k][ch].t = slice_ndarray(self.scope_data[k][ch].t)
                    ad.scope_data[k][ch].v = slice_ndarray(self.scope_data[k][ch].v)

        ad.params.N_img = np.prod(ad.xvardims)
        ad.params.N_shots = int(ad.params.N_shots / sliced_xvardim)
        ad.params.N_shots_with_repeats = int(ad.params.N_shots_with_repeats / sliced_xvardim)

        ad.analyze()
        ad._refresh_repeat_statistics()

        return ad

    ### Averaging and transpose

    def _storage_key(self,key):
        return "_" + key + "_stored"

    def _sem_scale_value(self, value, sem_divisor):
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            return value / sem_divisor
        return value

    def _sem_scale_scope_data(self, scope_data, sem_divisor):
        scaled_scope = dict()
        for scope_key, channel_dict in scope_data.items():
            scaled_scope[scope_key] = dict()
            for ch, trace in channel_dict.items():
                t = trace.t
                v = trace.v
                if isinstance(t, np.ndarray) and np.issubdtype(t.dtype, np.number):
                    t = t / sem_divisor
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                    v = v / sem_divisor
                scaled_scope[scope_key][ch] = ScopeTraceArray(scope_key, ch, t, v)
        return scaled_scope

    def _copy_metadata_to_repeat_sibling(self, ad_out, xvar_idx, n_repeats):
        from copy import deepcopy

        ad_out._lite = self._lite
        ad_out.server_talk = self.server_talk
        ad_out._ds = self._ds
        ad_out._dealer = None
        ad_out.images = self.images
        ad_out.image_timestamps = self.image_timestamps
        ad_out.experiment_code = self.experiment_code

        ad_out.params = deepcopy(self.params)
        ad_out.p = ad_out.params
        ad_out.camera_params = deepcopy(self.camera_params)
        ad_out.run_info = deepcopy(self.run_info)
        ad_out.roi = deepcopy(self.roi)

        ad_out.xvarnames = deepcopy(self.xvarnames)
        ad_out.xvars = [np.array(x, copy=True) for x in self.xvars]
        ad_out.Nvars = len(ad_out.xvars)
        ad_out.xvardims = np.array([len(x) for x in ad_out.xvars], dtype=int)

        ad_out.xvars[xvar_idx] = self._get_unique_repeated_xvar(xvar_idx, n_repeats)
        ad_out.xvardims[xvar_idx] = len(ad_out.xvars[xvar_idx])
        for idx, key in enumerate(ad_out.xvarnames):
            vars(ad_out.params)[key] = ad_out.xvars[idx]
        ad_out.params.N_repeats = np.ones(ad_out.Nvars, dtype=int)

        ad_out.sort_idx = np.array([])
        ad_out.sort_N = np.array([])

        ad_out.data = _RepeatDataVault()
        ad_out.avg = None
        ad_out.std = None
        ad_out.sem = None
        ad_out._repeat_sem_source = None
        ad_out._repeat_sem_divisor = None

        ad_out._analysis_tags = analysis_tags(self._analysis_tags.roi_id, self._analysis_tags.imaging_type)
        ad_out._analysis_tags.xvars_shuffled = False
        ad_out._analysis_tags.transposed = self._analysis_tags.transposed
        ad_out._analysis_tags.averaged = False
        ad_out._analysis_tags.repeats_reassigned = self._analysis_tags.repeats_reassigned

    def _reduce_repeat_ndarray(self, arr: np.ndarray, xvar_idx: int, n_repeats: int, reducer: str):
        arr = np.asarray(arr)
        split_shape = (*arr.shape[0:xvar_idx], -1, n_repeats, *arr.shape[(xvar_idx+1):])
        reshaped = arr.reshape(split_shape)
        if reducer == 'mean':
            return np.mean(reshaped, axis=xvar_idx+1, dtype=np.float64)
        if reducer == 'std':
            return np.std(reshaped, axis=xvar_idx+1, dtype=np.float64)
        raise ValueError(f'Unknown repeat reducer: {reducer}')

    def _is_scan_shaped_numeric_array(self, value):
        if not isinstance(value, np.ndarray):
            return False
        if not np.issubdtype(value.dtype, np.number):
            return False
        if value.ndim < self.Nvars:
            return False
        return tuple(value.shape[:self.Nvars]) == tuple(self.xvardims)

    def _build_repeat_stat_atomdata(self, xvar_idx, n_repeats, reducer='mean'):
        ad_out = object.__new__(self.__class__)
        self._copy_metadata_to_repeat_sibling(ad_out, xvar_idx, n_repeats)

        reduce_op = lambda x: self._reduce_repeat_ndarray(x, xvar_idx, n_repeats, reducer)

        for key, value in vars(self).items():
            if key in ['avg', 'std', 'sem',
                       '_repeat_sem_source', '_repeat_sem_divisor',
                       'params', 'p', 'camera_params', 'run_info', 'roi',
                       'data', 'scope_data', '_analysis_tags', '_dealer',
                       '_ds', 'server_talk']:
                continue
            if self._is_scan_shaped_numeric_array(value):
                vars(ad_out)[key] = reduce_op(value)
            elif key not in vars(ad_out):
                vars(ad_out)[key] = value

        for key in self.data.keys:
            value = vars(self.data)[key]
            if self._is_scan_shaped_numeric_array(value):
                vars(ad_out.data)[key] = reduce_op(value)
            else:
                vars(ad_out.data)[key] = value
            ad_out.data.keys.append(key)

        if hasattr(self, 'scope_data'):
            ad_out.scope_data = dict()
            for scope_key in self.scope_data.keys():
                ad_out.scope_data[scope_key] = dict()
                for ch, trace in self.scope_data[scope_key].items():
                    t = trace.t
                    v = trace.v
                    if self._is_scan_shaped_numeric_array(t):
                        t = reduce_op(t)
                    if self._is_scan_shaped_numeric_array(v):
                        v = reduce_op(v)
                    ad_out.scope_data[scope_key][ch] = ScopeTraceArray(scope_key, ch, t, v)

        return ad_out

    def _reduce_repeat_ndarray_mean_std(self, arr: np.ndarray, xvar_idx: int, n_repeats: int):
        arr = np.asarray(arr)
        if n_repeats == 1:
            # Fast path for no-repeat runs: mean is identity, std is exactly zero.
            mean_val = arr.astype(np.float64, copy=False)
            std_val = np.zeros_like(mean_val)
            return mean_val, std_val
        split_shape = (*arr.shape[0:xvar_idx], -1, n_repeats, *arr.shape[(xvar_idx+1):])
        reshaped = arr.reshape(split_shape)
        mean_val = np.mean(reshaped, axis=xvar_idx+1, dtype=np.float64)
        std_val = np.std(reshaped, axis=xvar_idx+1, dtype=np.float64)
        return mean_val, std_val

    def _build_repeat_stat_atomdata_pair(self, xvar_idx, n_repeats):
        ad_avg = object.__new__(self.__class__)
        ad_std = object.__new__(self.__class__)
        self._copy_metadata_to_repeat_sibling(ad_avg, xvar_idx, n_repeats)
        self._copy_metadata_to_repeat_sibling(ad_std, xvar_idx, n_repeats)

        skip_keys = ['avg', 'std', 'sem',
                     '_repeat_sem_source', '_repeat_sem_divisor',
                     'params', 'p', 'camera_params', 'run_info', 'roi',
                     'data', 'scope_data', '_analysis_tags', '_dealer',
                     '_ds', 'server_talk']

        for key, value in vars(self).items():
            if key in skip_keys:
                continue
            if self._is_scan_shaped_numeric_array(value):
                mean_val, std_val = self._reduce_repeat_ndarray_mean_std(value, xvar_idx, n_repeats)
                vars(ad_avg)[key] = mean_val
                vars(ad_std)[key] = std_val
            else:
                if key not in vars(ad_avg):
                    vars(ad_avg)[key] = value
                if key not in vars(ad_std):
                    vars(ad_std)[key] = value

        for key in self.data.keys:
            value = vars(self.data)[key]
            if self._is_scan_shaped_numeric_array(value):
                mean_val, std_val = self._reduce_repeat_ndarray_mean_std(value, xvar_idx, n_repeats)
                vars(ad_avg.data)[key] = mean_val
                vars(ad_std.data)[key] = std_val
            else:
                vars(ad_avg.data)[key] = value
                vars(ad_std.data)[key] = value
            ad_avg.data.keys.append(key)
            ad_std.data.keys.append(key)

        if hasattr(self, 'scope_data'):
            ad_avg.scope_data = dict()
            ad_std.scope_data = dict()
            for scope_key in self.scope_data.keys():
                ad_avg.scope_data[scope_key] = dict()
                ad_std.scope_data[scope_key] = dict()
                for ch, trace in self.scope_data[scope_key].items():
                    t_mean = trace.t
                    v_mean = trace.v
                    t_std = trace.t
                    v_std = trace.v
                    if self._is_scan_shaped_numeric_array(trace.t):
                        t_mean, t_std = self._reduce_repeat_ndarray_mean_std(trace.t, xvar_idx, n_repeats)
                    if self._is_scan_shaped_numeric_array(trace.v):
                        v_mean, v_std = self._reduce_repeat_ndarray_mean_std(trace.v, xvar_idx, n_repeats)
                    ad_avg.scope_data[scope_key][ch] = ScopeTraceArray(scope_key, ch, t_mean, v_mean)
                    ad_std.scope_data[scope_key][ch] = ScopeTraceArray(scope_key, ch, t_std, v_std)

        return ad_avg, ad_std

    def _build_repeat_sem_atomdata(self, std, n_repeats):
        ad_sem = object.__new__(self.__class__)
        ad_sem._repeat_sem_source = std
        ad_sem._repeat_sem_divisor = np.sqrt(n_repeats)
        ad_sem.avg = None
        ad_sem.std = None
        ad_sem.sem = None
        return ad_sem

    def _clear_repeat_statistics(self):
        self.avg = None
        self.std = None
        self.sem = None
        self._repeat_lazy_stat_context = None

    def _materialize_repeat_std_sem(self):
        return

    def _refresh_repeat_statistics(self):
        try:
            xvar_idx, n_repeats = self._get_repeat_axis_info()
            _ = self._get_unique_repeated_xvar(xvar_idx, n_repeats)
        except Exception:
            self.avg = _RepeatPassthroughAtomdataProxy(self)
            zero_proxy = _RepeatZeroAtomdataProxy(self)
            self.std = zero_proxy
            self.sem = zero_proxy
            self._repeat_lazy_stat_context = None
            return

        self.avg, self.std = self._build_repeat_stat_atomdata_pair(xvar_idx, n_repeats)
        self.sem = self._build_repeat_sem_atomdata(self.std, n_repeats)
        self._repeat_lazy_stat_context = None

    def _get_repeat_axis_info(self):
        repeat_axes = []
        repeat_counts = []
        for axis_idx, values in enumerate(self.xvars):
            values = np.asarray(values)
            _, counts = np.unique(values, return_counts=True)
            repeated_counts = counts[counts > 1]
            if repeated_counts.size == 0:
                continue
            unique_counts = np.unique(repeated_counts)
            if unique_counts.size != 1:
                raise ValueError('Number of repeats per value of an xvar must be the same for all values.')
            repeat_axes.append(axis_idx)
            repeat_counts.append(int(unique_counts[0]))

        if len(repeat_axes) == 0:
            raise ValueError('No repeated xvar axis was found in this atomdata.')
        if len(repeat_axes) > 1:
            raise ValueError(f'Found repeated values on multiple xvar axes: {repeat_axes}.')
        return repeat_axes[0], repeat_counts[0]

    def _get_unique_repeated_xvar(self, xvar_idx, n_repeats):
        values = np.asarray(self.xvars[xvar_idx])
        if values.size % n_repeats != 0:
            raise ValueError('Repeated xvar length must be divisible by the repeat count.')
        reshaped = values.reshape(-1, n_repeats)
        unique_values = reshaped[:, 0]
        if not np.all(reshaped == unique_values[:, None]):
            raise ValueError('Repeated xvar values must be grouped consecutively to reassign repeats.')
        return unique_values

    def _reassign_repeat_ndarray(self, arr, source_xvar_idx, target_xvar_idx, n_repeats, old_xvardims):
        arr = np.asarray(arr)
        old_xvardims = tuple(np.asarray(old_xvardims, dtype=int))
        if arr.ndim < len(old_xvardims):
            raise ValueError('Array does not have enough dimensions to match the scan axes.')
        if tuple(arr.shape[:len(old_xvardims)]) != old_xvardims:
            raise ValueError('Array leading dimensions do not match the current xvar dimensions.')

        source_size = old_xvardims[source_xvar_idx]
        if source_size % n_repeats != 0:
            raise ValueError('Repeat axis length must be divisible by the repeat count.')

        expanded_shape = list(old_xvardims)
        expanded_shape[source_xvar_idx] = source_size // n_repeats
        expanded_shape.insert(source_xvar_idx + 1, n_repeats)
        arr = arr.reshape(*expanded_shape, *arr.shape[len(old_xvardims):])

        expanded_axes = []
        for axis_idx in range(len(old_xvardims)):
            expanded_axis = axis_idx
            if axis_idx > source_xvar_idx:
                expanded_axis += 1
            expanded_axes.append(expanded_axis)

        repeat_axis = source_xvar_idx + 1
        transpose_idx = []
        for axis_idx, expanded_axis in enumerate(expanded_axes):
            transpose_idx.append(expanded_axis)
            if axis_idx == target_xvar_idx:
                transpose_idx.append(repeat_axis)

        trailing_axes = list(range(len(old_xvardims) + 1, arr.ndim))
        arr = np.transpose(arr, transpose_idx + trailing_axes)

        new_xvardims = list(old_xvardims)
        new_xvardims[source_xvar_idx] //= n_repeats
        new_xvardims[target_xvar_idx] *= n_repeats
        arr = arr.reshape(*new_xvardims, *arr.shape[(len(old_xvardims) + 1):])
        return arr

    def reassign_repeats(self, xvar_idx):
        """Move repeated shots from their current xvar axis to another axis.

        This only supports data that is already unshuffled, with repeats created
        by consecutive np.repeat calls on a single xvar axis.

        Args:
            xvar_idx (int): The xvar index that should own the repeats after the
            reassignment.
        """
        if self._analysis_tags.xvars_shuffled:
            raise ValueError('Repeat reassignment only supports unshuffled atomdata. Call unshuffle() first.')

        if not isinstance(xvar_idx, (int, np.integer)):
            raise TypeError('xvar_idx must be an integer.')
        if xvar_idx < 0 or xvar_idx >= len(self.xvars):
            raise IndexError('xvar_idx is out of range for this atomdata.')

        source_xvar_idx, n_repeats = self._get_repeat_axis_info()
        if source_xvar_idx == xvar_idx:
            return

        old_xvardims = np.array(self.xvardims, dtype=int)
        keys = [
            'img_atoms', 'img_light', 'img_dark',
            'img_timestamp_atoms', 'img_timestamp_light', 'img_timestamp_dark'
        ]
        for key in keys:
            vars(self)[key] = self._reassign_repeat_ndarray(
                vars(self)[key],
                source_xvar_idx,
                xvar_idx,
                n_repeats,
                old_xvardims
            )

        for key in self.data.keys:
            vars(self.data)[key] = self._reassign_repeat_ndarray(vars(self.data)[key],
                                                                 source_xvar_idx,
                                                                 xvar_idx,
                                                                 n_repeats,
                                                                 old_xvardims)

        if hasattr(self,'scope_data'):
            for scope_key in self.scope_data.keys():
                for ch in self.scope_data[scope_key].keys():
                    for ax in ['t', 'v']:
                        vars(self.scope_data[scope_key][ch])[ax] = self._reassign_repeat_ndarray(
                            vars(self.scope_data[scope_key][ch])[ax],
                            source_xvar_idx,
                            xvar_idx,
                            n_repeats,
                            old_xvardims
                        )

        self.xvars[source_xvar_idx] = self._get_unique_repeated_xvar(source_xvar_idx, n_repeats)
        self.xvars[xvar_idx] = np.repeat(np.asarray(self.xvars[xvar_idx]), n_repeats)
        self.xvardims = np.array([len(x) for x in self.xvars], dtype=int)
        self.Nvars = len(self.xvars)
        for axis_idx, key in enumerate(self.xvarnames):
            vars(self.params)[key] = self.xvars[axis_idx]

        new_n_repeats = np.ones(self.Nvars, dtype=int)
        new_n_repeats[xvar_idx] = n_repeats
        self.params.N_repeats = new_n_repeats

        self.sort_idx = np.array([])
        self.sort_N = np.array([])
        self._dealer = self._init_dealer()
        self.images = self._dealer.stack_linear_data_ndarray(self.img_atoms,
                                                             self.img_light,
                                                             self.img_dark)
        self.image_timestamps = self._dealer.stack_linear_data_ndarray(self.img_timestamp_atoms,
                                                                       self.img_timestamp_light,
                                                                       self.img_timestamp_dark)
        self._dealer.images = self.images
        self._dealer.image_timestamps = self.image_timestamps

        self._sort_images()
        self.analyze()
        self._analysis_tags.repeats_reassigned = True

    def avg_repeats(self,xvars_to_avg=[],reanalyze=True):
        """
        Averages the images along the axes specified in xvars_to_avg. Uses
        absorption imaging analysis.

        Args:
            xvars_to_avg (list, optional): A list of xvar indices to average.
            reanalyze (bool, optional): _description_. Defaults to True.
        """
        if not self._analysis_tags.averaged:
            self._refresh_repeat_statistics()
            if not xvars_to_avg:
                xvars_to_avg = list(range(len(self.xvars)))
            if not isinstance(xvars_to_avg,list):
                xvars_to_avg = [xvars_to_avg]

            from copy import deepcopy

            self._xvars_stored = deepcopy(self.xvars)
            def store_values(struct,keylist):
                for key in keylist:
                    array = vars(struct)[key]
                    # save the old information
                    newkey = self._storage_key(key)
                    vars(struct)[newkey] = deepcopy(array)

            self._store_keys = ['xvars','xvardims','od_raw']
            store_values(self,self._store_keys)

            self._store_param_keys = ['N_repeats',*self.xvarnames]
            store_values(self.params,self._store_param_keys)

            self._store_data_keys = self.data.keys
            store_values(self.data,self._store_data_keys)

            if hasattr(self,'scope_data'):
                self._store_scope_keys = list(self.scope_data.keys())
                for k in self._store_scope_keys:
                    newkey = self._storage_key(k)
                    self.scope_data[newkey] = deepcopy(self.scope_data[k])

            def avg_attrs(struct,key_list):
                for key in key_list:
                    arr = vars(struct)[key]
                    arr = self._avg_repeated_ndarray(arr, xvar_idx)
                    vars(struct)[key] = arr

            def avg_scope_dict():
                if hasattr(self,'scope_data'):
                    for k in self._store_scope_keys:
                        sta:dict = self.scope_data[k]
                        for ch in sta.keys():
                            for ax in ['t','v']:
                                x = self._avg_repeated_ndarray(vars(sta[ch])[ax], xvar_idx)
                                vars(self.scope_data[k][ch])[ax] = x

            for xvar_idx in xvars_to_avg:
                avg_attrs(self, ['od_raw'])
                avg_attrs(self.data, self.data.keys)
                avg_scope_dict()
                # write in the unaveraged xvars
                self.xvars[xvar_idx] = np.unique(self.xvars[xvar_idx])
                vars(self.params)[self.xvarnames[xvar_idx]] = self.xvars[xvar_idx]
                self.xvardims[xvar_idx] = self.xvars[xvar_idx].shape[0]
            self.params.N_repeats = np.ones(len(self.xvars),dtype=int)
        
            if reanalyze:
                # don't unshuffle xvars again -- that will be confusing
                self.analyze_ods()

            self._analysis_tags.averaged = True
        else:
            print('Atomdata is already repeat averaged. To revert to original atomdata, use Atomdata.revert_repeats().')
                
    def _avg_repeated_ndarray(self,arr:np.ndarray,xvar_idx,N_repeats_for_this_xvar=-1):
        i = xvar_idx
        if N_repeats_for_this_xvar == - 1:
            # N = self.params.N_repeats[i]
            _, counts = np.unique(self.xvars[xvar_idx], return_counts=True)
            ucounts = np.unique(counts)
            if ucounts.size == 1:
                N = ucounts[0]
            else:
                raise ValueError('Number of repeats per value of an xvar must be the same for all values.')
        else:
            N = N_repeats_for_this_xvar
        arr = np.mean( arr.reshape(*arr.shape[0:i],-1,N,*arr.shape[(i+1):]), axis=i+1, dtype=np.float64)
        return arr
    
    def revert_repeats(self):
        if self._analysis_tags.averaged:
            def retrieve_values(struct,keylist):
                for key in keylist:
                    newkey = self._storage_key(key)
                    vars(struct)[key] = vars(struct)[newkey]
            def retrieve_scope_dict():
                if hasattr(self,'scope_data'):
                    for k in self._store_scope_keys:
                        newkey = self._storage_key(k)
                        self.scope_data[k] = self.scope_data[newkey]
            retrieve_values(self,self._store_keys)
            retrieve_values(self.params,self._store_param_keys)
            retrieve_values(self.data,self._store_data_keys)
            retrieve_scope_dict()

            self.analyze_ods()
            self._analysis_tags.averaged = False
            self._refresh_repeat_statistics()
        else:
            print("Atomdata is not repeat averaged. To average, use Atomdata.avg_repeats().")

    def transpose_data(self,new_xvar_idx=[], reanalyze=True):
        """Swaps xvar order, then reruns the analysis.

        Args:
            new_xvar_idx (list): The list of indices specifying the new order of
            the original xvars. 
            
            For example, in a run with four xvars, specifying [0,2,1,3] means
            that the second and third xvar will be swapped, while the first and
            fourth remain unchanged.

            No list needs to be provided for the case of one or two xvars. In
            the case of one xvar, does nothing.

            new_var_idx can also be set to True in the case of one or two xvars,
            for convenience.
        """        
        if self._analysis_tags.averaged:
            raise ValueError("This function was written poorly and doesn't work on repeat averaged data. You can revert_repeats, transpose, then re-average.")

        Nvars = len(self.xvars)

        if new_xvar_idx == [] or new_xvar_idx == True:
            if Nvars == 1:
                raise ValueError('There is only one variable -- no dimensions to permute.')
                
            elif Nvars == 2:
                new_xvar_idx = [1,0] # by default, flip for just two vars
            else:
                raise ValueError('For more than two variables, you must specify the new xvar order.')
        elif len(new_xvar_idx) != Nvars:
            raise ValueError('You must specify a list of axis indices that match the number of xvars.')

        # for things of a listlike nature which have one element per xvar, and so
        # should have the elements along the first dimension reorderd according
        # to the new_xvar_idx (instead of their axes swapped).
        def reorder_listlike(struct,keys):
            for key in keys:
                attr = vars(struct)[key]
                new_attr = [attr[i] for i in new_xvar_idx]
                if isinstance(attr,np.ndarray):
                    new_attr = np.array(new_attr)
                vars(struct)[key] = new_attr

        listlike_keys = ['xvars','xvarnames']
        reorder_listlike(self,listlike_keys)

        if isinstance(self.params, np.ndarray):
            param_keys = ['N_repeats']
            reorder_listlike(self.params,param_keys)

        # for things of an ndarraylike nature which have one axis per xvar, and
        # so should have the order of their axes switched.
        def transpose_ndattr(attr):
            ndim = np.ndim(attr)
            # figure out how many extra indices each has. add them to the new
            # axis index list without changing their order.
            dims_to_add = ndim - Nvars
            axes_idx_to_add = [Nvars+i for i in range(dims_to_add)]
            new_idx = np.concatenate( (new_xvar_idx, axes_idx_to_add) ).astype(int)
            attr = np.transpose(attr,new_idx)
            return attr

        def reorder_ndarraylike(struct,keylist):
            for key in keylist:
                attr = vars(struct)[key]
                vars(struct)[key] = transpose_ndattr(attr)

        def transpose_scopedata():
            if hasattr(self, 'scope_data'):
                key_list = list(self.scope_data.keys())
                for k in key_list:
                    for ch in self.scope_data[k]:
                        reorder_ndarraylike(self.scope_data[k][ch],['t','v'])

        ndarraylike_keys = ['img_atoms','img_light','img_dark']
        reorder_ndarraylike(self,ndarraylike_keys)
        reorder_ndarraylike(self.data,self.data.keys)
        transpose_scopedata()

        self._dealer = self._init_dealer()

        if reanalyze:
            self.analyze()

        self._analysis_tags.transposed = not self._analysis_tags.transposed

    ### Data handling

    def _remap_fit_results(self):
        try:
            fits_x = self.cloudfit_x
            self.fit_sd_x = self._extract_attr(fits_x,'sigma')
            self.fit_center_x = self._extract_attr(fits_x,'x_center')
            self.fit_amp_x = self._extract_attr(fits_x,'amplitude')
            self.fit_offset_x = self._extract_attr(fits_x,'y_offset')
            self.fit_area_x = self._extract_attr(fits_x,'area')

            fits_y = self.cloudfit_y
            self.fit_sd_y = self._extract_attr(fits_y,'sigma')
            self.fit_center_y = self._extract_attr(fits_y,'x_center')
            self.fit_amp_y = self._extract_attr(fits_y,'amplitude')
            self.fit_offset_y = self._extract_attr(fits_y,'y_offset')
            self.fit_area_y = self._extract_attr(fits_y,'area')
            
        except Exception as e:
            print(e)
            print("Unable to extract fit parameters. The gaussian fit must have failed")

    def _extract_attr(self,ndarray,attr):
        linarray = np.reshape(ndarray,np.size(ndarray))
        vals = [vars(y)[attr] for y in linarray]
        out = np.reshape(vals,ndarray.shape+(-1,))
        if out.ndim == 2 and out.shape[-1] == 1:
            out = out.flatten()
        return out

    def _map(self,ndarray,func):
        linarray = np.reshape(ndarray,np.size(ndarray))
        vals = [func(y) for y in linarray]
        return np.reshape(vals,ndarray.shape+(-1,))
    
    def _unpack_xvars(self):
        # fetch the arrays for each xvar from parameters

        if not isinstance(self.xvarnames,list) and not isinstance(self.xvarnames,np.ndarray):
            self.xvarnames = [self.xvarnames]

        xvarnames = self.xvarnames

        self.Nvars = len(xvarnames)
        xvars = []
        for i in range(self.Nvars):
            xvars.append(vars(self.params)[xvarnames[i]])
        
        # figure out dimensions of each xvar
        self.xvardims = np.zeros(self.Nvars,dtype=int)
        for i in range(self.Nvars):
            if type(xvars[i]) == np.int64:
                raise ValueError(f'Run {self.run_info.run_id} did not have a scanned parameter.')
            self.xvardims[i] = np.int32(len(xvars[i]))

        return xvars
    
    ## Unshuffling

    def _shuff(self, reshuffle_bool):
        self.images = self._dealer.unscramble_images(reshuffle=reshuffle_bool)
        self._dealer._unshuffle_struct(self, reshuffle=reshuffle_bool)
        self._dealer._unshuffle_struct(self.params, reshuffle=reshuffle_bool)
        self._dealer._unshuffle_struct(self.data,
                                       only_treat_first_Nvar_axes=True,
                                       reshuffle=reshuffle_bool)
        if hasattr(self,'scope_data'):
            self._dealer._unshuffle_scopedata_dict(self.scope_data)
        self.xvars = self._unpack_xvars()

    def reshuffle(self):
        if self._analysis_tags.repeats_reassigned:
            raise ValueError("Cannot reshuffle after repeats have been reassigned.")
        if self._analysis_tags.xvars_shuffled == False:
            self._shuff(reshuffle_bool=True)
            self._sort_images()
            self.analyze()
            self._analysis_tags.xvars_shuffled = True
        else:
            print("Data is already in shuffled order.")

    def unshuffle(self,reanalyze=True):
        if self._analysis_tags.xvars_shuffled == True:
            self._shuff(reshuffle_bool=False)
            if reanalyze:
                self._sort_images()
                self.analyze()
            self._analysis_tags.xvars_shuffled = False
        else:
            print("Data is already in unshuffled order.")

    def _unshuffle_old_data(self):
        """Unshuffles data that was taken before we started saving data in
        sorted order (before 2024/10/02).
        """        
        if datetime.datetime(*self.run_info.run_datetime[:3]) < datetime.datetime(2024,10,2):
            self._analysis_tags.xvars_shuffled = True
            self.unshuffle(reanalyze=False)

    ### Setup

    def _init_dealer(self) -> Dealer:
        dealer = Dealer()
        dealer.params = self.params
        dealer.run_info = self.run_info
        dealer.images = self.images
        dealer.image_timestamps = self.image_timestamps
        dealer.sort_idx = self.sort_idx
        dealer.sort_N = self.sort_N
        # reconstruct the xvar objects
        for idx in range(len(self.xvarnames)):
            this_xvar = xvar(self.xvarnames[idx],
                             self.xvars[idx],
                             position=idx)
            if np.any(self.sort_idx):
                sort_idx_idx = np.where(self.sort_N == len(this_xvar.values))[0][0]
                this_xvar.sort_idx = self.sort_idx[sort_idx_idx]
            else:
                this_xvar.sort_idx = []
            dealer.scan_xvars.append(this_xvar)
            dealer.xvardims.append(len(this_xvar.values))
        dealer.N_xvars = len(self.xvardims)
        return dealer

    def _load_data(self, idx=0, path = "", lite=False, roi_id=None, _allow_lite_autocreate=True):
        t_load_total = time.perf_counter()
        timing = {}

        try:
            t_stage = time.perf_counter()
            file, rid = self.server_talk.get_data_file(idx, path, lite)
            timing['get_data_file_initial_s'] = time.perf_counter() - t_stage
        except ValueError as e:
            timing['get_data_file_initial_s'] = time.perf_counter() - t_stage
            msg = str(e)
            lite_missing = ("was not found" in msg or "lite copy does not exist" in msg)
            if lite and _allow_lite_autocreate and lite_missing:
                # Missing lite file: load regular data, generate lite copy, then retry lite load.
                t_stage = time.perf_counter()
                self._load_data(idx, path, lite=False, roi_id=roi_id, _allow_lite_autocreate=False)
                timing['fallback_full_load_s'] = time.perf_counter() - t_stage

                t_stage = time.perf_counter()
                self.server_talk.create_lite_copy(
                    self.run_info.run_id,
                    roi_id=roi_id,
                    use_saved_roi=(roi_id is None),
                )
                timing['fallback_create_lite_copy_s'] = time.perf_counter() - t_stage

                t_stage = time.perf_counter()
                file, rid = self.server_talk.get_data_file(self.run_info.run_id, "", lite=True)
                timing['get_data_file_retry_lite_s'] = time.perf_counter() - t_stage
            else:
                raise

        self._data_file_path = file
        self._saved_roi_from_file = False
        self.scope_data = {}

        t_stage = time.perf_counter()
        with h5py.File(file,'r') as f:
            timing['h5_open_s'] = time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            self.params = ExptParams()
            self.p = self.params
            self.camera_params = CameraParams()
            self.run_info = RunInfo()

            unpack_group(f,'params',self.params)
            unpack_group(f,'camera_params',self.camera_params)
            unpack_group(f,'run_info',self.run_info)
            timing['h5_unpack_headers_s'] = time.perf_counter() - t_stage

            print(self.run_info.run_id)

            t_stage = time.perf_counter()
            self.images = f['data']['images'][()]
            self.image_timestamps = f['data']['image_timestamps'][()]
            self.xvarnames = f.attrs['xvarnames'][()]
            self.xvars = self._unpack_xvars()
            timing['h5_read_core_arrays_s'] = time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            if 'roix' in f.attrs and 'roiy' in f.attrs:
                self._saved_roi_from_file = [f.attrs['roix'], f.attrs['roiy']]
            timing['h5_saved_roi_attrs_s'] = time.perf_counter() - t_stage

            class DataVault():
                def __init__(self):
                    self.keys = []
            self.data = DataVault()

            t_stage = time.perf_counter()
            all_keys = list(f['data'].keys())
            filtered_keys = [k for k in all_keys if k not in ['images', 'image_timestamps', 'sort_N', 'sort_idx', 'scope_data']]

            for k in filtered_keys:
                data_k = f['data'][k][()]
                data_k: np.ndarray

                vars(self.data)[k] = data_k
                self.data.keys.append(k)
            timing['h5_read_datavault_s'] = time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            try:
                experiment_text = f.attrs['expt_file']
                params_text = f.attrs['params_file']
                cooling_text = f.attrs['cooling_file']
                imaging_text = f.attrs['imaging_file']
                control_text = f.attrs['control_file']
            except:
                experiment_text = ""
                params_text = ""
                cooling_text = ""
                imaging_text = ""
                control_text = ""
            self.experiment_code = expt_code(experiment_text,
                                                params_text,
                                                cooling_text,
                                                imaging_text,
                                                control_text)
            timing['h5_read_experiment_text_s'] = time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            if 'sort_idx' in all_keys and 'sort_N' in all_keys:
                self.sort_idx = f['data']['sort_idx'][()]
                self.sort_N = f['data']['sort_N'][()]
            else:
                self.sort_idx = np.array([])
                self.sort_N = np.array([])
            timing['h5_read_sort_metadata_s'] = time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            try:
                if 'scope_data' in all_keys:
                    d = f['data']['scope_data']
                    SCOPE_DATA_CHANGE_EPOCH = datetime.datetime(2026,1,16,0)
                    old_method_bool = datetime.datetime(*self.run_info.run_datetime[:4]) < SCOPE_DATA_CHANGE_EPOCH

                    self.scope_data = format_scope_data(d, old_method=old_method_bool)
            except Exception as e:
                print(e)
            timing['h5_read_scope_data_s'] = time.perf_counter() - t_stage

        timing['load_total_s'] = time.perf_counter() - t_load_total
        self._timing.update({f'load_{k}': v for k, v in timing.items()})

        if self._timing_enabled:
            print(
                (
                    "[atomdata timing] load total={:.3f}s | get_data_file(initial)={:.3f}s | "
                    "h5_open={:.3f}s | headers={:.3f}s | core_arrays={:.3f}s | "
                    "datavault={:.3f}s | scope_data={:.3f}s"
                ).format(
                    timing.get('load_total_s', 0.0),
                    timing.get('get_data_file_initial_s', 0.0),
                    timing.get('h5_open_s', 0.0),
                    timing.get('h5_unpack_headers_s', 0.0),
                    timing.get('h5_read_core_arrays_s', 0.0),
                    timing.get('h5_read_datavault_s', 0.0),
                    timing.get('h5_read_scope_data_s', 0.0),
                )
            )

            if 'get_data_file_retry_lite_s' in timing or 'fallback_create_lite_copy_s' in timing:
                print(
                    (
                        "[atomdata timing] load fallback | full_load={:.3f}s | "
                        "create_lite_copy={:.3f}s | get_data_file(retry_lite)={:.3f}s"
                    ).format(
                        timing.get('fallback_full_load_s', 0.0),
                        timing.get('fallback_create_lite_copy_s', 0.0),
                        timing.get('get_data_file_retry_lite_s', 0.0),
                    )
                )

    def __getattribute__(self, name):
        if name in ['_repeat_sem_source',
                    '_repeat_sem_divisor',
                    '_sem_scale_value',
                    '_sem_scale_scope_data',
                    '_repeat_lazy_stat_context',
                    '_materialize_repeat_std_sem',
                    '__dict__',
                    '__class__']:
            return object.__getattribute__(self, name)

        if name in ['std', 'sem']:
            local_dict = object.__getattribute__(self, '__dict__')
            if local_dict.get(name, None) is None and local_dict.get('_repeat_lazy_stat_context', None) is not None:
                object.__getattribute__(self, '_materialize_repeat_std_sem')()
            return object.__getattribute__(self, name)

        sem_source = object.__getattribute__(self, '__dict__').get('_repeat_sem_source', None)
        if sem_source is not None:
            if name == 'data':
                sem_divisor = object.__getattribute__(self, '_repeat_sem_divisor')
                return _RepeatSEMDataProxy(sem_source.data, sem_divisor)
            if name == 'scope_data':
                sem_divisor = object.__getattribute__(self, '_repeat_sem_divisor')
                if hasattr(sem_source, 'scope_data'):
                    return object.__getattribute__(self, '_sem_scale_scope_data')(sem_source.scope_data, sem_divisor)
                return {}

            local_dict = object.__getattribute__(self, '__dict__')
            if name in local_dict:
                return object.__getattribute__(self, name)

            source_value = getattr(sem_source, name)
            sem_divisor = object.__getattribute__(self, '_repeat_sem_divisor')
            return object.__getattribute__(self, '_sem_scale_value')(source_value, sem_divisor)

        return object.__getattribute__(self, name)

# class ConcatAtomdata(atomdata):
#     def __init__(self,rids=[],roi_id=None,lite=False):

#         self.params = ExptParams()
#         self.camera_params = CameraParams()
#         self.run_info = RunInfo()

#         file, rid = st.get_data_file(rids[0],lite=lite)
#         with h5py.File(file,'r') as f:
#             params = ExptParams()
#             unpack_group(f,'params',params)
#             self.xvarnames = f.attrs['xvarnames'][()]

#             images = f['data']['images'][()]
#             image_timestamps = f['data']['image_timestamps'][()]

#             self.images = np.zeros( np.shape(rids) + images.shape,
#                                     dtype=images.dtype )
#             self.image_timestamps = np.zeros( np.shape(rids) + image_timestamps.shape,
#                                               dtype=image_timestamps.dtype)

#         self.sort_idx = []
#         self.sort_N = []

#         for rid in rids:
#             file, rid = st.get_data_file(rid,lite=lite)
    
#             print(f"run id {rid}")
#             with h5py.File(file,'r') as f:
#                 params = ExptParams()
#                 camera_params = CameraParams()
#                 run_info = RunInfo()
#                 unpack_group(f,'params',params)
#                 unpack_group(f,'camera_params',camera_params)
#                 unpack_group(f,'run_info',run_info)
#                 self.images = f['data']['images'][()]
#                 self.image_timestamps = f['data']['image_timestamps'][()]
#                 self.xvarnames = f.attrs['xvarnames'][()]
#                 self.xvars = self._unpack_xvars()
#                 try:
#                     experiment_text = f.attrs['expt_file']
#                     params_text = f.attrs['params_file']
#                     cooling_text = f.attrs['cooling_file']
#                     imaging_text = f.attrs['imaging_file']
#                 except:
#                     experiment_text = ""
#                     params_text = ""
#                     cooling_text = ""
#                     imaging_text = ""
#                 self.experiment_code = expt_code(experiment_text,
#                                                 params_text,
#                                                 cooling_text,
#                                                 imaging_text)
                    

