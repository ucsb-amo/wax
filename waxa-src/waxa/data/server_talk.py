import os
import subprocess
from datetime import datetime, timedelta
import glob
import numpy as np

MAP_BAT_PATH = "\"G:\\Shared drives\\Weld Lab Shared Drive\\Infrastructure\\map_network_drives_PeterRecommended.bat\""

class server_talk():
    def __init__(self,
                 data_dir=os.getenv("data"),
                 run_id_relpath="run_id.py",
                 roi_spreadsheet_replath="roi.xlsx",
                 first_data_folder_date="",
                 on_data_dir_disconnected_bat_path=""):
        
        self.data_dir = data_dir
        self.run_id_path = os.path.join(data_dir,run_id_relpath)
        self.roi_csv_path = os.path.join(data_dir,roi_spreadsheet_replath)

        if first_data_folder_date == "":
            first_data_folder_date = datetime(2023,6,22)
        if on_data_dir_disconnected_bat_path == "":
            on_data_dir_disconnected_bat_path = MAP_BAT_PATH

        self._first_data_folder_date = first_data_folder_date
        self._bat_on_data_dir_disconnected = on_data_dir_disconnected_bat_path

        self._lite = False

        self.set_data_dir()

    def set_data_dir(self, lite=False):

        if self._lite == lite:
            pass
        elif not lite:
            self.data_dir = os.path.dirname(self.data_dir)
        else:
            self.data_dir = os.path.join(self.data_dir, "_lite")
        self._lite = lite

    def get_data_file(self, idx=0, path="", lite=False):
        '''
        Returns the data file path corresponding to index idx. For idx > 0, idx is
        intepreted as a run ID. For idx = 0, gets the latest data file path. For idx
        < 0, increments chronologically backward from the latest file.

        If path is instead specified, loads the file at the specified path.

        Parameters
        ----------
        idx: int
            If a positive value is specified, it is interpreted as a run_id (as
            stored in run_info.run_id), and that data is found and loaded. If zero
            or a negative number are given, data is loaded relative to the most
            recent dataset (idx=0).
        path: str
            The full path to the file to be loaded. If not specified, loads the file
            as dictated by `idx`.

        Returns
        -------
        str: the full path to the specified data file
        int: the run ID for the specified data file
        '''
        if path == "":
            latest_file = self.get_latest_data_file(lite)
            latest_rid = self.run_id_from_filepath(latest_file,lite)
            if idx == 0:
                file = latest_file
            if idx <= 0:
                file = self.recurse_find_data_file(latest_rid+idx,lite)
            if idx > 0:
                if latest_rid - idx < 10000:
                    file = self.recurse_find_data_file(idx,lite)
                else:
                    file = self.all_glob_find_data_file(idx,lite)
        else:
            if path.endswith('.hdf5'):
                file = path
            else:
                raise ValueError("The provided path is not a hdf5 file.")
            
        rid = self.run_id_from_filepath(file,lite)
        return file, rid

    def check_for_mapped_data_dir(self):
        self.set_data_dir()
        if not os.path.exists(self.data_dir):
            print(f"Data dir ({self.data_dir}) not found. Attempting to re-map network drives.")
            cmd = self._bat_on_data_dir_disconnected         
            result = subprocess.run(cmd, creationflags=subprocess.CREATE_NO_WINDOW)
            if not os.path.exists(self.data_dir):
                print(f"Data dir still not found. Are you connected to the physics network?") 
                return False
            else:
                print("Network drives successfully mapped.")
                return True
        else:
            return True

    def get_latest_date_folder(self,lite=False,days_ago=0):
        self.set_data_dir(lite)
        date = datetime.today() - timedelta(days=days_ago)
        date_str = date.strftime('%Y-%m-%d')
        folderpath=os.path.join(self.data_dir,date_str)
        if not os.path.exists(folderpath) or not os.listdir(folderpath):
            folderpath = self.get_latest_date_folder(lite,days_ago+1)
        return folderpath
        
    def get_latest_data_file(self,lite=False):
        self.check_for_mapped_data_dir()
        folderpath = self.get_latest_date_folder(lite)
        pattern = os.path.join(folderpath,'*.hdf5')
        files = list(glob.iglob(pattern))
        # Filter out files that do not exist before sorting
        existing_files = [f for f in files if os.path.exists(f)]
        files_sorted = sorted(
            existing_files,
            key=os.path.getmtime,
            reverse=True)
        latest_file = None
        for file in files_sorted:
            try:
                # Try to access getmtime, which will raise FileNotFoundError if missing
                os.path.getmtime(file)
                latest_file = file
                break
            except FileNotFoundError:
                continue
        return latest_file

    def recurse_find_data_file(self,r_id,lite=False,days_ago=0):
        date = datetime.today() - timedelta(days=days_ago)

        if date < self._first_data_folder_date:
            raise ValueError(f"Data file with run ID {r_id:1.0f} was not found.")
        
        date_str = date.strftime('%Y-%m-%d')

        self.set_data_dir(lite)
        folderpath=os.path.join(self.data_dir,date_str)

        if os.path.exists(folderpath):
            pattern = os.path.join(folderpath,'*.hdf5')
            files = np.array(list(glob.iglob(pattern)))
            r_ids = np.array([self.run_id_from_filepath(file,lite) for file in files])

            files_mask = (r_id == r_ids)
            file_with_rid = files[files_mask]

            if len(file_with_rid) > 1:
                print(f"There are two data files with run ID {r_id:1.0f}. Choosing the more recent one.")
                file_with_rid = max(file_with_rid,key=os.path.getmtime)
            elif len(file_with_rid) == 1:
                file_with_rid = file_with_rid[0]
            
            if not file_with_rid:
                file_with_rid = self.recurse_find_data_file(r_id,lite,days_ago+1)
        else:
            file_with_rid = self.recurse_find_data_file(r_id,lite,days_ago+1)
        return file_with_rid

    def all_glob_find_data_file(self,run_id,lite=False):
        self.set_data_dir(lite)
        folderpath=os.path.join(self.data_dir,'*','*.hdf5')
        list_of_files = glob.glob(folderpath)
        rids = [self.run_id_from_filepath(file,lite) for file in list_of_files]
        rid_idx = rids.index(run_id)
        file = list_of_files[rid_idx]
        return file

    def run_id_from_filepath(self,filepath,lite=False):
        self.set_data_dir(lite)
        run_id = int(os.path.normpath(filepath).split(os.path.sep)[-1].split("_")[0])
        return run_id

    def get_run_id(self):
        self.set_data_dir()

        pwd = os.getcwd()
        os.chdir(self.data_dir)
        with open(self.run_id_path,'r') as f:
            rid = f.read()
        os.chdir(pwd)
        return int(rid)

    def update_run_id(self,run_info=None):
        self.set_data_dir()

        pwd = os.getcwd()
        os.chdir(self.data_dir)

        if run_info is not None:
            rid = run_info.run_id
        else:
            with open(self.run_id_path,'r') as f:
                try: 
                    rid = int(f.read())
                except:
                    print(f'run id file at {self.run_id_path} is empty -- extracting from latest data file')
                    rid = self.run_id_from_filepath(self.get_latest_data_file())
        rid += 1
        with open(self.run_id_path,'w') as f:
            f.write(f"{rid}")

        os.chdir(pwd)

    def create_lite_copy(self,run_idx,roi_id=None,use_saved_roi=True):
        from waxa.data import RunInfo, DataSaver
        from waxa.atomdata import unpack_group
        import h5py
        from waxa import ROI

        original_data_filepath, rid = self.get_data_file(run_idx)

        ri = RunInfo()
        with h5py.File(original_data_filepath) as file:
            unpack_group(file,'run_info',ri)

        ds = DataSaver(data_dir=self.data_dir, server_talk=self)
        lite_data_path, lite_data_folder = ds._data_path(ri,lite=True)

        os.makedirs(lite_data_folder, exist_ok=True)

        with h5py.File(lite_data_path,'w') as f_lite:
            with h5py.File(original_data_filepath,'r') as f_src:
                # copy over other datasets (not data)
                keys = f_src.keys()
                for key in keys:
                    if key != 'data':
                        f_src.copy(f_src[key],f_lite,key)

                # copy over non-image data
                dkeys = f_src['data'].keys()
                f_lite.create_group('data')
                for key in dkeys:
                    if key != 'images':
                        f_src.copy(f_src['data'][key],f_lite['data'],key)

                # copy over attributes
                akeys = f_src.attrs.keys()
                for key in akeys:
                    f_lite.attrs[key] = f_src.attrs[key]

                roi = ROI(rid,roi_id=roi_id,use_saved_roi=use_saved_roi,server_talk=self)
                
                N_img = f_src['data']['images'].shape[0]
                px = np.diff(roi.roix)[0]
                py = np.diff(roi.roiy)[0]

                f_lite.attrs['roix'] = [0,px]
                f_lite.attrs['roiy'] = [0,py]

                dtype = f_src['data']['images'][0].dtype
                images = np.zeros((N_img,py,px),dtype=dtype)

                for idx in range(N_img):
                    images[idx] = roi.crop(f_src['data']['images'][idx])
                f_lite['data']['images'] = images
        print(f'Lite version of run {rid} saved at {lite_data_path}.')