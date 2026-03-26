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

    def emit_xvar_only_shot_result(self):
        """Emit one shot result when no images are expected for this run."""
        result = self._build_base_result()
        self._append_xvar_fields(result)
        self._append_apd_fields(result)
        for key in [
            "integrated_od", "sum_od_peak_x", "sum_od_peak_y",
            "fit_sigma_x", "fit_sigma_y", "fit_center_x", "fit_center_y",
            "fit_amplitude_x", "fit_amplitude_y", "fit_offset_x", "fit_offset_y",
            "fit_area_x", "fit_area_y", "atom_number", "atom_number_fit_x", "atom_number_fit_y",
        ]:
            result[key] = np.nan
        try:
            self.shot_result.emit(result)
        except RuntimeError:
            pass

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
            self.od_raw, self.od, self.sum_od_x, self.sum_od_y = self._compute_od_and_sums(
                self.img_atoms,
                self.img_light,
                self.img_dark,
            )
            
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

    def _compute_od_and_sums(self, img_atoms, img_light, img_dark):
        od_raw = compute_OD(img_atoms, img_light, img_dark, imaging_type=self.imaging_type)
        od_stack = np.array([od_raw])
        od_processed, _, _ = process_ODs(od_stack, self.roi)
        od = od_processed[0]

        # Crop to current OD viewport before recomputing derived traces.
        cropped_od, _, _ = self.crop_od_to_view_range(od)
        sum_od_x = np.sum(cropped_od, axis=0)
        sum_od_y = np.sum(cropped_od, axis=1)
        return od_raw, od, sum_od_x, sum_od_y

    def compute_shot_result_from_images(self, img_atoms, img_light, img_dark, shot_index: int, xvars: dict):
        """Recompute one shot's derived quantities from image triplets.

        Uses the current ROI and OD-view crop to match interactive recalculation
        behavior requested from the viewer settings menu.
        """
        try:
            _, od, sum_od_x, sum_od_y = self._compute_od_and_sums(
                np.asarray(img_atoms), np.asarray(img_light), np.asarray(img_dark)
            )
            return self._compute_shot_result(od, sum_od_x, sum_od_y, shot_index, xvars)
        except Exception:
            od = np.zeros_like(np.asarray(img_atoms))
            sum_od_x = np.zeros(od.shape[1]) if od.ndim == 2 else np.array([])
            sum_od_y = np.zeros(od.shape[0]) if od.ndim == 2 else np.array([])
            return self._compute_shot_result(od, sum_od_x, sum_od_y, shot_index, xvars)

    # ------------------------------------------------------------------
    #  Derived-quantity computation
    # ------------------------------------------------------------------

    def _emit_shot_result(self):
        """Compute derived quantities from the current shot and emit."""
        shot_index = self._shot_index
        result = self._compute_shot_result(
            self.od,
            self.sum_od_x,
            self.sum_od_y,
            shot_index,
            dict(self._xvars),
        )
        self._shot_index += 1

        try:
            self.shot_result.emit(result)
        except RuntimeError:
            pass

    def _compute_shot_result(self, od, sum_od_x, sum_od_y, shot_index: int, xvars: dict):
        result: dict = {
            "shot_index": int(shot_index),
            "timestamp": time.time(),
            "xvars": dict(xvars),
        }

        # Integrated OD
        try:
            result["integrated_od"] = float(np.sum(od))
        except Exception:
            result["integrated_od"] = np.nan

        # Sum OD peaks
        try:
            result["sum_od_peak_x"] = float(np.max(sum_od_x))
            result["sum_od_peak_y"] = float(np.max(sum_od_y))
        except Exception:
            result["sum_od_peak_x"] = np.nan
            result["sum_od_peak_y"] = np.nan

        # Gaussian fits (requires camera_params)
        fit_x = None
        fit_y = None
        if self.camera_params is not None:
            cp = self.camera_params
            px = cp.pixel_size_m / cp.magnification
            xaxis_x = px * np.arange(sum_od_x.shape[-1])
            xaxis_y = px * np.arange(sum_od_y.shape[-1])
            try:
                fit_x = GaussianFit(xaxis_x, sum_od_x, print_errors=False)
            except Exception:
                fit_x = None
            try:
                fit_y = GaussianFit(xaxis_y, sum_od_y, print_errors=False)
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

        self._append_xvar_fields(result, xvars)
        self._append_apd_fields(result, xvars)
        return result

    def _build_base_result(self) -> dict:
        result: dict = {
            "shot_index": self._shot_index,
            "timestamp": time.time(),
            "xvars": dict(self._xvars),
        }
        self._shot_index += 1
        return result

    def _append_xvar_fields(self, result: dict, xvars: dict | None = None):
        source = self._xvars if xvars is None else xvars
        for key, value in source.items():
            result[f"xvar.{key}"] = self._to_plot_scalar(value)

    def _append_apd_fields(self, result: dict, xvars: dict | None = None):
        source = self._xvars if xvars is None else xvars
        raw = source.get("post_shot_absorption", None)
        if raw is None:
            return
        try:
            v = np.asarray(raw, dtype=float).reshape(-1)
            if v.size < 4:
                return
            v_up, v_down, v_light, v_dark = v[:4]
            up_only = v_up - v_dark
            down_only = v_down - v_dark
            light_only = v_light - v_dark
            if up_only <= 0 or down_only <= 0 or light_only <= 0:
                return
            n_up = -np.log(up_only / light_only)
            n_down = -np.log(down_only / light_only)
            result["atom_number_apd_up"] = float(n_up)
            result["atom_number_apd_down"] = float(n_down)
            result["atom_number_apd_total"] = float(n_up + n_down)
        except Exception:
            return

    def _to_plot_scalar(self, value):
        """Map any payload value to a scalar suitable for per-shot plotting."""
        try:
            if np.isscalar(value):
                return float(value)
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size == 0:
                return np.nan
            if arr.size == 1:
                return float(arr[0])
            return float(np.nanmean(arr))
        except Exception:
            return np.nan

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
