import numpy as np
import os
import h5py

from waxa.data.server_talk import server_talk as st

# __DEFAULT_KEY = "no_one_will_ever_use_this_key000111"

from waxa.dummy.expt import Expt as DummyExpt

class DataSaver():
    def __init__(self,
                 data_dir="",
                 expt_repo_src_directory="",
                 expt_params_relative_filepath="",
                 cooling_relative_filepath="",
                 imaging_relative_filepath="",
                 control_relative_filepath="",
                 server_talk=None):
        
        self._data_dir = data_dir
        self._expt_repo_path = expt_repo_src_directory
        self._expt_params_path = os.path.join(expt_repo_src_directory,
                                              expt_params_relative_filepath)
        self._cooling_path = os.path.join(expt_repo_src_directory,
                                          cooling_relative_filepath)
        self._imaging_path = os.path.join(expt_repo_src_directory,
                                          imaging_relative_filepath)
        self._control_path = os.path.join(expt_repo_src_directory,
                                          control_relative_filepath)

        if server_talk == None:
            server_talk = st(data_dir=data_dir)
        else:
            server_talk = server_talk
        self.server_talk = server_talk

    def save_data(self,expt:DummyExpt,expt_filepath="",data_object=None):

        # from wax.base.sub.dealer import Dealer
        # expt: Dealer

        if expt.setup_camera:
            
            pwd = os.getcwd()
            os.chdir(self._data_dir)
            
            fpath, _ = self._data_path(expt.run_info)

            if data_object:
                f = data_object
            else:
                f = h5py.File(fpath,'r+')
            
                    
            if expt.sort_idx:
                # these were read in by liveOD, so we replace the expt empty arrays
                expt.images = np.array(f['data']['images'])
                expt.image_timestamps = np.array(f['data']['image_timestamps'])

                # I think these two lines are redundant, should already happen in prepare
                expt.xvardims = [len(xvar.values) for xvar in expt.scan_xvars]
                expt.N_xvars = len(expt.xvardims)

                expt._unshuffle_struct(expt) # this usually does nothing
                # now replace the data from the h5 with the unscrambled data
                f['data']['images'][...] = expt.unscramble_images()
                f['data']['image_timestamps'][...] = expt._unscramble_timestamps()
                expt._unshuffle_struct(expt.params)

            self._save_data_vault(f,expt)
            self._save_scope_data(f,expt)

            del f['params']
            params_dset = f.create_group('params')
            self._class_attr_to_dataset(params_dset,expt.params)

            if expt_filepath:
                f['run_info']['experiment_filepath'][...] = expt_filepath
                f.attrs['experiment_filepath'] = expt_filepath

            self._save_expt_files_text(f,expt_filepath)

            f.close()
            print("Parameters saved, data closed.")
            os.chdir(pwd)

    def get_xvardims(self,expt:DummyExpt):
        return [len(xvar.values) for xvar in expt.scan_xvars]
    
    def pad_sort_idx(self,expt:DummyExpt):
        maxN = np.max(expt.sort_N)
        for i in range(len(expt.sort_idx)):
            N_to_pad = maxN - len(expt.sort_idx[i])
            expt.sort_idx[i] = np.append(expt.sort_idx[i], [-1]*N_to_pad).astype(int)

    def create_data_file(self,expt:DummyExpt):

        pwd = os.getcwd()

        self.server_talk.check_for_mapped_data_dir()
        os.chdir(self._data_dir)

        fpath, folder = self._data_path(expt.run_info)

        if not os.path.exists(folder):
            os.mkdir(folder)

        expt.run_info.filepath = fpath
        expt.run_info.xvarnames = expt.xvarnames

        f = h5py.File(fpath,'w')
        data = f.create_group('data')

        f.attrs['camera_ready'] = 0
        f.attrs['camera_ready_ack'] = 0
        
        f.attrs['xvarnames'] = expt.xvarnames
        data.create_dataset('images',data=expt.images)
        data.create_dataset('image_timestamps',data=expt.image_timestamps)
        for key in expt.data.keys:
            this_data = vars(expt.data)[key]._run_data
            data.create_dataset(key, data=this_data)

        if expt.sort_idx:
            # pad with [-1]s to allow saving in hdf5 (avoid staggered array)
            self.pad_sort_idx(expt)
            data.create_dataset('sort_idx',data=expt.sort_idx)
            data.create_dataset('sort_N',data=expt.sort_N)
        
        # store run info as attrs
        self._class_attr_to_attr(f,expt.run_info)
        # also store run info as dataset
        runinfo_dset = f.create_group('run_info')
        self._class_attr_to_dataset(runinfo_dset,expt.run_info)
        params_dset = f.create_group('params')
        self._class_attr_to_dataset(params_dset,expt.params)
        cam_dset = f.create_group('camera_params')
        self._class_attr_to_dataset(cam_dset,expt.camera_params)
        
        f.close()

        os.chdir(pwd)

        return fpath
        
    def _save_data_vault(self,
                         h5File:h5py.File,
                         expt:DummyExpt):
        f = h5File
        for key in expt.data.keys:
            this_dc = vars(expt.data)[key]
            if this_dc._external_data_bool:
                # overwrite with data from hdf5 in case populated by a process outside expt
                this_data = f['data'][key][...]
            else:
                # otherwise, take the data that was stuck into the array during the expt
                this_data = this_dc._run_data
            if this_dc._external_data_bool or this_dc._data_gotten:
                if expt.sort_idx:
                    # unshuffle if shuffled
                    ndims_per_shot = len(this_data.shape) - len(expt.scan_xvars)
                    this_data = expt._unshuffle_ndarray(this_data,exclude_dims=ndims_per_shot)
                f['data'][key][...] = this_data

    def _save_scope_data(self,
                         h5File:h5py.File,
                         expt:DummyExpt):
        f = h5File
        if expt.scope_data._scope_trace_taken:
            scope_data = f['data'].create_group('scope_data')
            for scope in expt.scope_data.scopes:
                data = scope.reshape_data()
                # data comes out as shape (n0,...,nN,Nch,2,Npts)
                # ni = values for ith xvar
                # Nch = # scope channels the user captured from
                # 2 = axis for picking time or voltage axis
                # Npts = points per scan
                if expt.sort_idx:
                    data = expt._unshuffle_ndarray(data,exclude_dims=3).astype(np.float32)
                this_scope_data = scope_data.create_group(scope.label)
                # time/voltage axis always -2, take the first one for each capture
                # only take one time axis for all the channels on a given shot
                # resulting shape: (n0,...,nN,Npts)
                t = np.take(np.take(data,0,axis=-2),0,axis=-2)
                # take the voltage values
                # resulting shape: (n0,...,nN,Nch,Npts)
                v = np.take(data,1,-2)
                this_scope_data.create_dataset('t',data=t)
                this_scope_data.create_dataset('v',data=v)

    def _save_expt_files_text(self,
                              h5File:h5py.File,
                              expt_filepath):
        
        self._check_for_expt_files()

        f = h5File
        if expt_filepath:
                with open(expt_filepath) as expt_file:
                    expt_text = expt_file.read()
                f.attrs["expt_file"] = expt_text
        else:
            f.attrs["expt_file"] = ""

        if self._expt_params_path:
            with open(self._expt_params_path) as params_file:
                params_file = params_file.read()
            f.attrs["params_file"] = params_file
        
        if self._cooling_path:
            with open(self._cooling_path) as cooling_file:
                cooling_file = cooling_file.read()
            f.attrs["cooling_file"] = cooling_file

        if self._imaging_path:
            with open(self._imaging_path) as imaging_file:
                imaging_file = imaging_file.read()
            f.attrs["imaging_file"] = imaging_file

        if self._control_path:
            with open(self._control_path) as file:
                file = file.read()
            f.attrs["control_file"] = file

    def _check_for_expt_files(self):
        if not os.path.isfile(self._expt_params_path):
            print(f'expt_params file not found at {self._expt_params_path}, saving contents skipped')
            self._expt_params_path = ""
        if not os.path.isfile(self._cooling_path):
            print(f'cooling file not found at {self._cooling_path}, saving contents skipped')
            self._cooling_path = ""
        if not os.path.isfile(self._imaging_path):
            print(f'imaging file not found at {self._imaging_path}, saving contents skipped')
            self._imaging_path = ""
        if not os.path.isfile(self._control_path):
            print(f'control file not found at {self._control_path}, saving contents skipped')
            self._control_path = ""

    def _class_attr_to_dataset(self,dset,obj):
        try:
            keys = list(vars(obj)) 
            for key in keys:
                if not key.startswith("_"):
                    value = vars(obj)[key]
                    try:
                        dset.create_dataset(key, data=value)
                    except Exception as e:
                        print(f"Failed to save attribute \"{key}\" of {obj}.")
                        print(e)
        except Exception as e:
            print(e)

    def _class_attr_to_attr(self,dset,obj):
        try:
            keys = list(vars(obj))  
            for key in keys:
                value = vars(obj)[key]
                dset.attrs[key] = value
        except Exception as e:
            print(e)

    def _data_path(self,run_info,lite=False):
        this_data_dir = self._data_dir
        run_id_str = f"{str(run_info.run_id).zfill(7)}"
        expt_class = self._bytes_to_str(run_info.expt_class)
        datetime_str = self._bytes_to_str(run_info.run_datetime_str)
        if lite:
            run_id_str += "_lite"
            this_data_dir = os.path.join(self._data_dir,"_lite")
        filename = run_id_str + "_" + datetime_str + "_" + expt_class + ".hdf5"
        filepath_folder = os.path.join(this_data_dir,
                                       self._bytes_to_str(run_info.run_date_str))
        filepath = os.path.join(filepath_folder,filename)
        return filepath, filepath_folder

    def _update_run_id(self,run_info):
        self.server_talk.update_run_id(run_info)

    def _get_rid(self):
        return self.server_talk.get_run_id()
    
    def _bytes_to_str(self,attr):
        if isinstance(attr,bytes):
            attr = attr.decode("utf-8")
        return attr