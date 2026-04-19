import numpy as np
import pandas as pd
import os
import cv2

from waxa.data.server_talk import server_talk as st
from waxa.image_processing.compute_ODs import compute_OD
from waxa.config.img_types import img_types

import h5py

_ROI_EXCEL_CACHE = {}

def _load_roi_excel_cached(path):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None

    cached = _ROI_EXCEL_CACHE.get(path)
    if cached is not None and cached['mtime'] == mtime:
        return cached['df']

    df = pd.read_excel(path)
    _ROI_EXCEL_CACHE[path] = {
        'mtime': mtime,
        'df': df,
    }
    return df

class ROI():
    def __init__(self,
                 run_id=0,
                 roi_id=None,
                 key="",
                 use_saved_roi=True,
                 lite=False,
                 printouts=True,
                 server_talk=None,
                 current_file_path=None,
                 current_saved_roi=None):
        
        if server_talk == None:
            self.server_talk = st()
        else:
            self.server_talk = server_talk

        self.roix = [-1,-1]
        self.roiy = [-1,-1]
        self.key = key
        self.run_id = run_id
        self._current_file_path = current_file_path
        self._current_saved_roi = current_saved_roi
        self.load_roi(roi_id,
                      use_saved=use_saved_roi,
                      lite=lite,
                      printouts=printouts)

    def crop(self,OD):
        """Crops the given ndarray according the ROI.

        Args:
            OD (np.ndarray): The ndarray to be cropped.

        Returns:
            ndarray: The cropped ndarray.
        """        
        OD: np.ndarray
        idx_y = range(self.roiy[0],self.roiy[1])
        idx_x = range(self.roix[0],self.roix[1])
        cropOD = OD.take(idx_y,axis=OD.ndim-2).take(idx_x,axis=OD.ndim-1)
        return cropOD

    def load_roi(self,roi_id=None,use_saved=True,lite=False,
                 printouts=True):
        """Loads an ROI according to the provided roi_id.

        Args:
            roi_id (None, int, or str): Specifies which crop to use. If None,
            defaults to the ROI saved in the data if it exists, otherwise
            prompts the user to select an ROI using the GUI. If an int,
            interpreted as an run ID, which will be checked for a saved ROI and
            that ROI will be used. If a string, interprets as a key in the
            roi.xlsx document in the PotassiumData folder.
            printouts (bool): If True, prints out information about the
            ROI loading process.

            use_saved (bool): If False, ignores saved ROI and forces creation of
            a new one.
        """
        
        # Check for ROI saved in the current data file.
        saved_roi_bool = self.read_roi_from_h5(lite=lite,printouts=printouts)
        if roi_id == None:
            if saved_roi_bool and use_saved:
                if printouts: print("Using saved ROI.")
                pass
            elif saved_roi_bool:
                if printouts: print("Saved ROI was found, but is being overridden.")
                if printouts: print("Specify the new ROI.")
                self.select_roi()
            else:
                if printouts: print("Specify the new ROI.")
                self.select_roi()
        else:
            if saved_roi_bool:
                if printouts: print("Saved ROI was found, but is being overridden.")
                pass

        # Checks for ROI saved in the specified run ID.
        if isinstance(roi_id,int):
            if printouts: print("ROI specified by Run ID. Attempting to load ROI...")
            saved_roi_bool = self.read_roi_from_h5(roi_id,printouts=printouts)
            if saved_roi_bool:
                if printouts: print(f"Using ROI loaded from run {roi_id}.")
            else:
                if printouts: print(f"Specify the new ROI.")
                self.select_roi()

        if isinstance(roi_id,str):
            if printouts: print("ROI specified by string. Referencing roi.xslx spreadsheet (PotassiumData)...")
            roi_exists = self.read_roi_from_excel(roi_id,printouts=printouts)
            if not roi_exists:
                if printouts: print(f"Creating ROI for key {roi_id}.")
                self.select_roi()
                self._update_excel()

        if self.check_for_blank_roi():
            if printouts: print("ROI was not specified. Defaulting to whole image.")
            px, py = self.get_image_size()
            self.roix = [0,px]
            self.roiy = [0,py]

    def save_roi_h5(self, lite=False, printouts=False):
        fpath, _ = self.server_talk.get_data_file(self.run_id,lite=lite)
        with h5py.File(fpath,'r+') as f:
            f.attrs['roix'] = self.roix
            f.attrs['roiy'] = self.roiy
        if printouts: print(f"ROI saved to h5 file at {fpath}")

    def save_roi_excel(self,key=""):
        if self.key == "" and key == "":
            raise ValueError("You must specify a key to save the ROI to the spreadsheet.")
        if not isinstance(key,str):
            raise ValueError("The specified key must be a string.")
        
        if not key == "":
            self.key = key
        else:
            # saving will use the key already associated with self.roi.
            pass
        self._update_excel()

    def get_image_size(self):
        """Gets the size in pixels of the images (horizontal, vertical) in this run.

        Returns:
            px, py: The image dimensions (rows, columns).
        """        
        fpath, _ = self.server_talk.get_data_file(self.run_id)
        with h5py.File(fpath) as f:
            py, px = f['data']['images'].shape[-2:]
        return px, py

    def read_roi_from_h5(self, run_id=[], lite=False, printouts=True):
        """Looks in the hdf5 file with the corresponding run ID and attempts to
        read out a saved ROI. Returns True if successful and False otherwise.

        Args:
            run_id (int): The run ID of the run in which to look for a
            saved ROI.

        Returns:
            bool: Returns True if successful and False otherwise.
        """        
        if run_id == []:
            run_id = self.run_id
        try:
            # Fast path: atomdata already loaded this file and optionally read
            # roix/roiy, so avoid another resolver + file-open roundtrip.
            if run_id == self.run_id and self._current_saved_roi is not None:
                if self._current_saved_roi is False:
                    if printouts: print(f"No ROI saved in run {run_id} (cached).")
                    return False
                roix, roiy = self._current_saved_roi
                self.roix = roix
                self.roiy = roiy
                if printouts: print(f"ROI loaded from run {run_id} (cached).")
                return True

            if run_id == self.run_id and self._current_file_path is not None:
                fpath = self._current_file_path
            else:
                fpath, run_id = self.server_talk.get_data_file(run_id,lite=lite)

            with h5py.File(fpath) as f:
                roix = f.attrs['roix']
                roiy = f.attrs['roiy']
            self.roix = roix
            self.roiy = roiy
            if printouts: print(f"ROI loaded from run {run_id}.")
            return True
        except Exception as e:
            if printouts: print(f"No ROI saved in run {run_id}.")
            return False
        
    def read_roi_from_excel(self,key,printouts=True):
        """Reads in the ROI with the corresponding key from the roi spreadsheet
        (roi.xlsx in PotassiumData), if it exists. Returns True if successful
        and False otherwise.

        Args:
            key (str): The key of the ROI to retrieve from the spreadsheet.

        Returns:
            bool: Returns True if successful and False otherwise.
        """        
        self.key = key
        if key == "":
            raise ValueError("ROI key must be a non-empty string.")
        roicsv = _load_roi_excel_cached(self.server_talk.roi_csv_path)
        keymatch = roicsv.loc[ roicsv['key'] == self.key ]
        if np.any(keymatch):
            if printouts: print(f"ROI {key} found.")
            self.key = key
            self.roix = [ keymatch['roix0'].values[0], keymatch['roix1'].values[0] ]
            self.roiy = [ keymatch['roiy0'].values[0], keymatch['roiy1'].values[0] ]
            return True
        else:
            if printouts: print(f"ROI with key {key} does not exist.")
            return False

    def check_for_blank_roi(self):
        """The ROI selection GUI output is all -1s if no ROI is selected.
        Detects this result so that it can be handled.

        Returns:
            bool: whether or not the ROI selection is valid.
        """        
        return np.all(np.array([*self.roix,*self.roiy]) == -1)
    
    def select_roi(self, run_id=[]):
        """Brings up the GUI to select a new ROI rectangle. The user should
        click and drag (LMB) in order to select a rectangle, then hit Enter to
        submit their selection. RMB clears the rectangle, and Escape/the X
        button will close the window without selecting an ROI (in most cases,
        resulting in an ROI that spans the entire image).

        Args:
            run_id (int, optional): The run_id to use for displaying ODs during
            ROI selection.
        """        
        if run_id == []:
            run_id = self.run_id
        file_path = self._current_file_path if run_id == self.run_id else None
        update_bool, roix, roiy = roi_creator(
            run_id,
            self.key,
            self.server_talk,
            file_path=file_path,
        ).get_roi_rectangle()
        if update_bool:
            self.roix, self.roiy = roix, roiy
        else:
            print("ROI not selected, aborting.")

    def _update_excel(self):
        """Saves the ROI to the excel spreadsheet (roi.xlsx) in the
        PotassiumData folder. If the key already exists, updates the existing
        ROI. If not, creates a new line.
        """        
        key = self.key
        if key == "":
            raise ValueError("The key must be nonempty in order to save the ROI.")

        # Read the excel file
        df = pd.read_excel(self.server_talk.roi_csv_path)
        new_values = [*self.roix, *self.roiy]

        # Check if the label exists
        if key in df.iloc[:, 0].values:
            # Find the index of the row with the given label
            index = df[df.iloc[:, 0] == key].index[0]
            # Replace the row values with new values
            df.iloc[index, 1:] = new_values
        else:
            # Append a new row with the given label and new values
            new_row = pd.DataFrame([[key] + new_values], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)

        # Save the updated dataframe back to the excel file
        df.to_excel(self.server_talk.roi_csv_path, index=False)
        # Invalidate cached spreadsheet after write.
        if self.server_talk.roi_csv_path in _ROI_EXCEL_CACHE:
            del _ROI_EXCEL_CACHE[self.server_talk.roi_csv_path]
        print(f"Updated the spreadsheet ROI with key {key}.")

class roi_creator():
    window_name = 'recrop'

    def __init__(self, run_id, key, server_talk, file_path=None):

        self.key = key
        self.run_id = run_id
        self.server_talk = server_talk

        if file_path is None:
            filepath, _ = server_talk.get_data_file(run_id)
        else:
            filepath = file_path

        self.h5_file = h5py.File(filepath, 'r')
        self.images = self.h5_file['data']['images']
        self.N_img = self.images.shape[0] // 3
        try:
            self.analysis_type = self.h5_file['run_info']['imaging_type'][()]
        except Exception:
            self.analysis_type = img_types.ABSORPTION
        self._od_cache = {}

        self.image = self.get_od(0)

        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.end_x, self.end_y = -1, -1

    def _clip_point(self, x, y, shape):
        y_max, x_max = shape[:2]
        clipped_x = max(min(int(x), x_max - 1), 0)
        clipped_y = max(min(int(y), y_max - 1), 0)
        return clipped_x, clipped_y

    def _map_display_point_to_original(self, x, y, zoom_region, original_shape, display_shape):
        x, y = self._clip_point(x, y, display_shape)
        if zoom_region is None:
            return x, y

        display_h, display_w = display_shape[:2]
        x0, y0, x1, y1 = zoom_region
        zoom_w = max(x1 - x0, 1)
        zoom_h = max(y1 - y0, 1)

        mapped_x = x0 + int(x * zoom_w / display_w)
        mapped_y = y0 + int(y * zoom_h / display_h)
        return self._clip_point(mapped_x, mapped_y, original_shape)

    def _extract_display_image(self, original_image, zoom_region):
        if zoom_region is None:
            return original_image.copy()

        x0, y0, x1, y1 = zoom_region
        if x1 <= x0 or y1 <= y0:
            return original_image.copy()

        zoomed_region = original_image[y0:y1, x0:x1]
        if zoomed_region.size == 0:
            return original_image.copy()

        return cv2.resize(
            zoomed_region,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    def _colorize_image(self, image):
        max_pixel_value = float(np.max(image))
        if max_pixel_value <= 0.0:
            normalized_image = np.zeros_like(image, dtype=np.uint8)
        else:
            threshold = max(self.cmap_juice_factor * max_pixel_value, np.finfo(float).eps)
            normalized_image = np.clip(image, 0, threshold)
            normalized_image = cv2.normalize(normalized_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(normalized_image, cv2.COLORMAP_VIRIDIS)

    def _prepare_window(self, image_shape):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        max_width = 1400
        max_height = 900
        height, width = image_shape[:2]
        scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
        window_width = max(400, int(width * scale))
        window_height = max(300, int(height * scale))
        cv2.resizeWindow(self.window_name, window_width, window_height)
        try:
            import win32con
            import win32gui

            hwnd = win32gui.FindWindow(None, self.window_name)
            if hwnd:
                win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
                win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def get_od(self, idx):
        """Computes the idx'th OD for display in the ROI selection GUI.

        Args:
            idx (int): the index of the OD to display

        Returns:
            np.ndarray: the OD to display.
        """        
        if idx not in self._od_cache:
            pwa = self.images[3 * idx]
            pwoa = self.images[3 * idx + 1]
            dark = self.images[3 * idx + 2]
            self._od_cache[idx] = compute_OD(pwa, pwoa, dark, self.analysis_type)
        return self._od_cache[idx]
        
    def get_roi_rectangle(self):
        """Brings up the GUI to select an ROI over a display of the ODs from a
        given run.

        Controls:
            LMB + drag: Select an ROI rectangle.
            MMB + drag: Draw a dotted-line rectangle for zooming.
            MMB release: Zoom in to the selected region.
            RMB: Clear the drawn ROI.
            Mouse wheel scroll down: Zoom back out to full scale.
            L/R arrow keys: Scroll through ODs from the run while keeping zoom.
            Enter: Submit your selection.
            Escape / "X" button: Close the GUI without submitting selection.

        Returns:
            bool: Whether an ROI has been selected.
            tuple: roix, given as [roix0, roix1] (left and right bounds of the ROI).
            tuple: roiy, given as [roiy0, roiy1] (top and bottom bounds of the ROI).
        """       
        original_image = self.image.copy()
        display_image = original_image.copy()
        img_index = 0
        zooming = False
        zoom_region = None  # Store zoomed region coordinates
        # if self.analysis_type != img_types.ABSORPTION:
        #     self.cmap_juice_factor = 0.01
        # else:
        #     self.cmap_juice_factor = 0.8
        self.cmap_juice_factor = 1.

        def draw_rectangle(event, x, y, flags, param):
            nonlocal display_image, zoom_region, zooming

            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_x, self.start_y = self._clip_point(x, y, display_image.shape)
                
            elif event == cv2.EVENT_MBUTTONDOWN:
                zooming = True
                self.start_x, self.start_y = self._clip_point(x, y, display_image.shape)

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing or zooming:
                    self.end_x, self.end_y = self._clip_point(x, y, display_image.shape)
                    
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.end_x, self.end_y = self._clip_point(x, y, display_image.shape)

            elif event == cv2.EVENT_MBUTTONUP:
                zooming = False
                self.end_x, self.end_y = self._clip_point(x, y, display_image.shape)
                try:
                    if self.start_x != -1 and self.start_y != -1 and self.end_x != -1 and self.end_y != -1:
                        mapped_start = self._map_display_point_to_original(
                            self.start_x,
                            self.start_y,
                            zoom_region,
                            original_image.shape,
                            display_image.shape,
                        )
                        mapped_end = self._map_display_point_to_original(
                            self.end_x,
                            self.end_y,
                            zoom_region,
                            original_image.shape,
                            display_image.shape,
                        )
                        x0, x1 = sorted([mapped_start[0], mapped_end[0]])
                        y0, y1 = sorted([mapped_start[1], mapped_end[1]])
                        if x1 > x0 and y1 > y0:
                            zoom_region = (x0, y0, x1, y1)
                            display_image = self._extract_display_image(original_image, zoom_region)
                except Exception:
                    display_image = original_image.copy()
                    zoom_region = None
                self.start_x, self.start_y = -1, -1
                self.end_x, self.end_y = -1, -1

            elif event == cv2.EVENT_MOUSEWHEEL:
                zoom_region = None
                display_image = original_image.copy()
                self.start_x, self.start_y = -1, -1
                self.end_x, self.end_y = -1, -1

            elif event == cv2.EVENT_RBUTTONDOWN:
                self.drawing = False
                zooming = False
                self.start_x, self.start_y = -1, -1
                self.end_x, self.end_y = -1, -1
                

        def adjust_colormap_scale(key):
            """Adjusts the colormap scale factor based on arrow key input."""
            step_size = 0.00025 if self.analysis_type == img_types.DISPERSIVE else 0.1
            max_joos = 0.0005 if self.analysis_type == img_types.DISPERSIVE else 0.05
            if key == 0x260000:  # Up arrow key (increase fraction)
                self.cmap_juice_factor = max(self.cmap_juice_factor - step_size, max_joos)
            elif key == 0x280000:  # Down arrow key (decrease fraction)
                self.cmap_juice_factor = min(self.cmap_juice_factor + step_size, 1.0)

        self._prepare_window(display_image.shape)
        cv2.setMouseCallback(self.window_name, draw_rectangle)

        try:
            while True:
                overlay_image = self._colorize_image(display_image)

                if self.start_x != -1 and self.start_y != -1 and self.end_x != -1 and self.end_y != -1:
                    color = (255, 255, 255)
                    thickness = 2 if self.drawing else 1
                    line_type = cv2.LINE_8 if self.drawing else cv2.LINE_4
                    cv2.rectangle(
                        overlay_image,
                        (self.start_x, self.start_y),
                        (self.end_x, self.end_y),
                        color,
                        thickness,
                        line_type,
                    )

                cv2.imshow(self.window_name, overlay_image)

                key = cv2.waitKeyEx(15)
                if key == 13:
                    break

                if key in [0x260000, 0x280000]:
                    adjust_colormap_scale(key)

                elif key in [2555904, 2424832]:
                    if key == 2555904:
                        img_index = (img_index + 1) % self.N_img
                    else:
                        img_index = (img_index - 1) % self.N_img
                    original_image = self.get_od(img_index).copy()
                    display_image = self._extract_display_image(original_image, zoom_region)

                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if key == 27:
                    break
        finally:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                cv2.destroyAllWindows()
            self.h5_file.close()

        if zoom_region == None:
            mapped_start_x, mapped_start_y = self.start_x, self.start_y
            mapped_end_x, mapped_end_y = self.end_x, self.end_y
        else:
            mapped_start_x, mapped_start_y = self._map_display_point_to_original(
                self.start_x,
                self.start_y,
                zoom_region,
                original_image.shape,
                display_image.shape,
            )
            mapped_end_x, mapped_end_y = self._map_display_point_to_original(
                self.end_x,
                self.end_y,
                zoom_region,
                original_image.shape,
                display_image.shape,
            )

        out = np.array([mapped_start_x, mapped_start_y, mapped_end_x, mapped_end_y])
        update_bool = not np.all(out == -1)
        if mapped_start_x == mapped_end_x or mapped_start_y == mapped_end_y:
            update_bool = False
        return update_bool, np.sort([mapped_start_x, mapped_end_x]), np.sort([mapped_end_y, mapped_start_y])
