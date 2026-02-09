import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from waxa.image_processing import compute_OD, process_ODs
from waxa.fitting import GaussianFit
from queue import Queue

# Cross-section for K-39 D2 cycling transition (m^2)
_ATOM_CROSS_SECTION = 5.878324268151581e-13


class Analyzer(QThread):
    analyzed = pyqtSignal()
    shot_result = pyqtSignal(dict)  # derived quantities for each shot
    def __init__(self, plotting_queue: Queue, viewer=None):
        super().__init__()
        
        self.imgs = []
        self.plotting_queue = plotting_queue
        self.roi = []
        self.viewer = viewer

        # ---- per-run state ----
        self.camera_params = None   # set by viewer window on run_start
        self._shot_index = 0
        self._xvars: dict = {}      # latest xvar dict from server
        
        # Initialize run parameters with defaults (will be set by get_img_number)
        self.N_img = 0
        self.N_shots = 0
        self.N_pwa_per_shot = 0
        self.imaging_type = False

    def get_img_number(self, N_img, N_shots, N_pwa_per_shot):
        self.N_img = N_img
        self.N_shots = N_shots
        self.N_pwa_per_shot = N_pwa_per_shot
        self._shot_index = 0

    def get_analysis_type(self, imaging_type):
        self.imaging_type = imaging_type

    def set_xvars(self, xvars: dict):
        """Store the latest xvar values (called each scan iteration)."""
        self._xvars = dict(xvars)

    def got_img(self, img):
        # Skip until we have run parameters
        if self.N_pwa_per_shot == 0 and not self.imgs:
            return
        
        self.imgs.append(np.asarray(img))
        if len(self.imgs) == (self.N_pwa_per_shot + 2):
            self.analyze()
            self.imgs = []

    def analyze(self):
        try:
            self.img_atoms = self.imgs[0]
            self.img_light = self.imgs[self.N_pwa_per_shot]
            self.img_dark = self.imgs[self.N_pwa_per_shot + 1]
            self.od_raw = compute_OD(self.img_atoms, self.img_light, self.img_dark, imaging_type=self.imaging_type)
            self.od_raw = np.array([self.od_raw])
            self.od, self.sum_od_x, self.sum_od_y = process_ODs(self.od_raw, self.roi)
            self.od_raw = self.od_raw[0]
            self.od = self.od[0]
            
            # Crop the OD to the current view range for sum calculations
            cropped_od, x_slice, y_slice = self.crop_od_to_view_range(self.od)
            
            # Compute sums from the cropped OD
            cropped_sum_od_x = np.sum(cropped_od, axis=0)  # Sum along y-axis (rows)
            cropped_sum_od_y = np.sum(cropped_od, axis=1)  # Sum along x-axis (columns)
            
            # Use cropped sums instead of the original ones
            self.sum_od_x = cropped_sum_od_x
            self.sum_od_y = cropped_sum_od_y
            
        except Exception as e:
            print(f"[Analyzer] Error during OD computation: {e}")
            # Create dummy OD in case of error so we can still plot
            self.od = np.zeros_like(self.img_atoms)
            self.sum_od_x = np.zeros(self.img_atoms.shape[1])
            self.sum_od_y = np.zeros(self.img_atoms.shape[0])
        
        self.analyzed.emit()
        self.plotting_queue.put((self.img_atoms, self.img_light, self.img_dark, self.od, self.sum_od_x, self.sum_od_y))

        # ---- compute & emit derived quantities for pop-out plots ----
        self._emit_shot_result()

    # ------------------------------------------------------------------
    #  Derived-quantity computation
    # ------------------------------------------------------------------

    def _emit_shot_result(self):
        """Compute derived quantities from the current shot and emit."""
        result: dict = {
            "shot_index": self._shot_index,
            "timestamp": time.time(),
            "xvars": dict(self._xvars),
        }
        self._shot_index += 1

        od = self.od  # 2-D array for this shot

        # Integrated OD
        try:
            result["integrated_od"] = float(np.sum(od))
        except Exception:
            result["integrated_od"] = np.nan

        # Sum OD peaks
        try:
            result["sum_od_peak_x"] = float(np.max(self.sum_od_x))
            result["sum_od_peak_y"] = float(np.max(self.sum_od_y))
        except Exception:
            result["sum_od_peak_x"] = np.nan
            result["sum_od_peak_y"] = np.nan

        # Gaussian fits (requires camera_params)
        fit_x = None
        fit_y = None
        if self.camera_params is not None:
            cp = self.camera_params
            px = cp.pixel_size_m / cp.magnification
            xaxis_x = px * np.arange(self.sum_od_x.shape[-1])
            xaxis_y = px * np.arange(self.sum_od_y.shape[-1])
            try:
                fit_x = GaussianFit(xaxis_x, self.sum_od_x, print_errors=False)
            except Exception:
                fit_x = None
            try:
                fit_y = GaussianFit(xaxis_y, self.sum_od_y, print_errors=False)
            except Exception:
                fit_y = None

        def _safe(obj, attr):
            if obj is None:
                return np.nan
            v = getattr(obj, attr, np.nan)
            return float(v) if np.isfinite(v) else np.nan

        result["fit_sigma_x"]     = _safe(fit_x, "sigma")
        result["fit_sigma_y"]     = _safe(fit_y, "sigma")
        result["fit_center_x"]    = _safe(fit_x, "x_center")
        result["fit_center_y"]    = _safe(fit_y, "x_center")
        result["fit_amplitude_x"] = _safe(fit_x, "amplitude")
        result["fit_amplitude_y"] = _safe(fit_y, "amplitude")
        result["fit_offset_x"]    = _safe(fit_x, "y_offset")
        result["fit_offset_y"]    = _safe(fit_y, "y_offset")
        result["fit_area_x"]      = _safe(fit_x, "area")
        result["fit_area_y"]      = _safe(fit_y, "area")

        # Atom number from integrated OD
        if self.camera_params is not None:
            dx = self.camera_params.pixel_size_m / self.camera_params.magnification
            result["atom_number"] = result["integrated_od"] * dx**2 / _ATOM_CROSS_SECTION
            result["atom_number_fit_x"] = result["fit_area_x"] * dx / _ATOM_CROSS_SECTION
            result["atom_number_fit_y"] = result["fit_area_y"] * dx / _ATOM_CROSS_SECTION
        else:
            result["atom_number"] = np.nan
            result["atom_number_fit_x"] = np.nan
            result["atom_number_fit_y"] = np.nan

        try:
            self.shot_result.emit(result)
        except RuntimeError:
            pass
        
    def crop_od_to_view_range(self, od):
        """
        Crop the OD array to the current view range of the viewer's OD plot
        
        Args:
            od (numpy.ndarray): The OD array to crop
            
        Returns:
            tuple: (cropped_od, x_slice, y_slice) where slices indicate the cropping ranges
        """
        if self.viewer is None:
            return od, slice(None), slice(None)
        
        try:
            # Get the current view range from the viewer's OD plot
            x_range, y_range = self.viewer.get_od_view_range()
            
            # Convert view coordinates to array indices
            # Clamp to valid array bounds
            x_min = max(0, int(round(x_range[0])))
            x_max = min(od.shape[1], int(round(x_range[1])))
            y_min = max(0, int(round(y_range[0])))
            y_max = min(od.shape[0], int(round(y_range[1])))
            
            # Create slices for cropping
            y_slice = slice(y_min, y_max)
            x_slice = slice(x_min, x_max)
            
            # Crop the OD
            cropped_od = od[y_slice, x_slice]
            
            return cropped_od, x_slice, y_slice
            
        except Exception as e:
            # If anything goes wrong, return the original OD
            print(f"Warning: Could not crop OD to view range: {e}")
            return od, slice(None), slice(None)

    # def compute_atom_number(self,od):
    #     self.atom_cross_section = self.atom.get_cross_section()
    #     dx_pixel = self.camera_params.pixel_size_m / self.camera_params.magnification
        
    #     self.atom_number_fit_area_x = self.fit_area_x * dx_pixel / self.atom_cross_section
    #     self.atom_number_fit_area_y = self.fit_area_y * dx_pixel / self.atom_cross_section

    #     self.atom_number_density = self.od * dx_pixel**2 / self.atom_cross_section
    #     self.atom_number = np.sum(np.sum(self.atom_number_density,-2),-1)
