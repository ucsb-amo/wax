import numpy as np

from waxa.image_processing.compute_ODs import compute_OD
from waxa.image_processing.compute_gaussian_cloud_params import fit_gaussian_sum_dist
from waxa.config.img_types import img_types as img

from waxa.data.server_talk import server_talk as st
from waxa.atomdata_base import atomdata_base, atom_number_apd, unpack_group


class   atomdata(atomdata_base):
    """User-facing atomdata class with analysis-focused methods.

    Heavy data loading, repeat handling, shuffling, slicing, and transpose
    machinery lives in atomdata_base.
    """

    def __init__(
        self,
        idx=0,
        roi_id=None,
        path="",
        lite=False,
        skip_saved_roi=False,
        transpose_idx=[],
        avg_repeats=False,
        server_talk=st(),
    ):
        super().__init__(
            idx=idx,
            roi_id=roi_id,
            path=path,
            lite=lite,
            skip_saved_roi=skip_saved_roi,
            transpose_idx=transpose_idx,
            avg_repeats=avg_repeats,
            server_talk=server_talk,
        )

    # User-facing operations: thin wrappers over parent implementations.
    def recrop(self, roi_id=None, use_saved=False):
        return super().recrop(roi_id=roi_id, use_saved=use_saved)

    def save_roi_excel(self, key=""):
        return super().save_roi_excel(key=key)

    def save_roi_h5(self, printouts=False):
        return super().save_roi_h5(printouts=printouts)

    def unshuffle(self, reanalyze=True):
        return super().unshuffle(reanalyze=reanalyze)

    def reshuffle(self):
        return super().reshuffle()

    def slice_atomdata(self, which_shot_idx=0, which_xvar_idx=0, ignore_repeats=False):
        return super().slice_atomdata(
            which_shot_idx=which_shot_idx,
            which_xvar_idx=which_xvar_idx,
            ignore_repeats=ignore_repeats,
        )

    def reassign_repeats(self, xvar_idx):
        return super().reassign_repeats(xvar_idx)

    def avg_repeats(self, xvars_to_avg=[], reanalyze=True):
        return super().avg_repeats(xvars_to_avg=xvars_to_avg, reanalyze=reanalyze)

    def revert_repeats(self):
        return super().revert_repeats()

    def transpose_data(self, new_xvar_idx=[], reanalyze=True):
        return super().transpose_data(new_xvar_idx=new_xvar_idx, reanalyze=reanalyze)

    ### Analysis methods

    def _initial_analysis(self, transpose_idx, avg_repeats):
        self._sort_images()
        if transpose_idx:
            self._analysis_tags.transposed = True
            self.transpose_data(transpose_idx=False, reanalyze=False)
        self.compute_raw_ods()
        if avg_repeats:
            self.avg_repeats(reanalyze=False)
        self.analyze_ods()
        self._refresh_repeat_statistics()

    def analyze(self):
        self.compute_raw_ods()
        self.analyze_ods()
        self._refresh_repeat_statistics()

    def compute_raw_ods(self):
        """Computes OD (or normalized transmission for non-absorption imaging)."""
        self.od_raw = compute_OD(
            self.img_atoms,
            self.img_light,
            self.img_dark,
            imaging_type=self._analysis_tags.imaging_type,
        )

    def analyze_ods(self):
        """Crop ODs, build projections, fit Gaussians, and map fit results."""
        self.od = self.roi.crop(self.od_raw)
        self.sum_od_x = np.sum(self.od, self.od.ndim - 2)
        self.sum_od_y = np.sum(self.od, self.od.ndim - 1)

        self.axis_camera_px_x = np.arange(self.sum_od_x.shape[-1])
        self.axis_camera_px_y = np.arange(self.sum_od_y.shape[-1])

        self.axis_camera_x = self.camera_params.pixel_size_m * self.axis_camera_px_x
        self.axis_camera_y = self.camera_params.pixel_size_m * self.axis_camera_px_y

        self.axis_x = self.axis_camera_x / self.camera_params.magnification
        self.axis_y = self.axis_camera_y / self.camera_params.magnification

        self.cloudfit_x = fit_gaussian_sum_dist(self.sum_od_x, self.camera_params)
        self.cloudfit_y = fit_gaussian_sum_dist(self.sum_od_y, self.camera_params)

        self._remap_fit_results()

        self.compute_apd_atom_number()

        if self._analysis_tags.imaging_type == img.ABSORPTION:
            self.compute_atom_number()

        self.integrated_od = np.sum(np.sum(self.od, -2), -1)

    def compute_apd_atom_number(self):
        if 'post_shot_absorption' in self.data.keys:
            v = self.data.post_shot_absorption
            if np.all(v == 0.0):
                return

            v_up = v[:, 0]
            v_down = v[:, 1]
            v_light = v[:, 2]
            v_dark = v[:, 3]

            light_only = v_light - v_dark
            up_only = v_up - v_dark
            down_only = v_down - v_dark

            ratio_up = np.where((up_only > 0) & (light_only > 0), up_only / light_only, np.nan)
            ratio_down = np.where((down_only > 0) & (light_only > 0), down_only / light_only, np.nan)

            number_up = -np.log(ratio_up)
            number_down = -np.log(ratio_down)

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
            self.xvarnames = np.append(self.xvarnames, 'idx_pwa')
            self.xvars.append(np.arange(self.params.N_pwa_per_shot))
            self.xvardims = np.append(self.xvardims, self.params.N_pwa_per_shot)
            self.Nvars += 1
            np.append(self.sort_idx, np.arange(self.params.N_pwa_per_shot))
            if not self.params.N_pwa_per_shot in self.sort_N:
                np.append(self.sort_N, self.params.N_pwa_per_shot)
        else:
            self.img_atoms = self._dealer.strip_shot_idx_axis(self.img_atoms)[0]
            self.img_light = self._dealer.strip_shot_idx_axis(self.img_light)[0]
            self.img_dark = self._dealer.strip_shot_idx_axis(self.img_dark)[0]

            self.img_timestamp_atoms = self._dealer.strip_shot_idx_axis(self.img_timestamp_atoms)[0]
            self.img_timestamp_light = self._dealer.strip_shot_idx_axis(self.img_timestamp_light)[0]
            self.img_timestamp_dark = self._dealer.strip_shot_idx_axis(self.img_timestamp_dark)[0]

    def compute_atom_number(self):
        self.atom_cross_section = 5.878324268151581e-13
        dx_pixel = self.camera_params.pixel_size_m / self.camera_params.magnification

        self.atom_number_fit_area_x = self.fit_area_x * dx_pixel / self.atom_cross_section
        self.atom_number_fit_area_y = self.fit_area_y * dx_pixel / self.atom_cross_section

        self.atom_number_density = self.od * dx_pixel**2 / self.atom_cross_section
        self.atom_number = np.sum(np.sum(self.atom_number_density, -2), -1)

    def _remap_fit_results(self):
        try:
            fits_x = self.cloudfit_x
            self.fit_sd_x = self._extract_attr(fits_x, 'sigma')
            self.fit_center_x = self._extract_attr(fits_x, 'x_center')
            self.fit_amp_x = self._extract_attr(fits_x, 'amplitude')
            self.fit_offset_x = self._extract_attr(fits_x, 'y_offset')
            self.fit_area_x = self._extract_attr(fits_x, 'area')

            fits_y = self.cloudfit_y
            self.fit_sd_y = self._extract_attr(fits_y, 'sigma')
            self.fit_center_y = self._extract_attr(fits_y, 'x_center')
            self.fit_amp_y = self._extract_attr(fits_y, 'amplitude')
            self.fit_offset_y = self._extract_attr(fits_y, 'y_offset')
            self.fit_area_y = self._extract_attr(fits_y, 'area')

        except Exception as e:
            print(e)
            print("Unable to extract fit parameters. The gaussian fit must have failed")
