"""Counts the scan's shots and images, and steps the xvar counters.

Each xvar keeps a ``counter``: the index of the value the current shot uses.
The counters live on the host; ``step_scan`` advances them after each shot
like nested for loops (last xvar = innermost), and the kernel reads them back.
"""
from artiq.experiment import *
import numpy as np

from waxx.util import console


class ShotGrid:

    # ---- finish_prepare: how many shots and images, and the image array

    def get_N_img(self):
        """
        Computes the number of images to be taken during the sequence from the
        length of the specified xvars, stores in self.params.N_img. For
        absorption imaging, 3 images per shot. For fluorescence imaging,
        variable pwa images (ExptParams.N_pwa_per_shot, default = 1), then 1
        each pwoa and dark images.

        Also sets params.N_shots_with_repeats (every shot of the run) and
        params.N_shots (the scan points, repeats divided out), and prints the
        run's "Scan:" line.
        """
        N_img = 1
        parts = []

        for xvar in self.scan_xvars:
            N_img = N_img * xvar.values.shape[0]
            parts.append(f"{xvar.values.shape[0]} x {xvar.key}")
        self.params.N_shots_with_repeats = N_img

        # ASCII only: ExptBuilder pipes this through cp1252 on Windows.
        msg = f"Scan: {', '.join(parts)} -> {N_img} shots"

        # N_shots: the shots without their repeats. N_repeats is a list by now
        # (Dealer.repeat_xvars: one entry, or one per xvar).
        if isinstance(self.params.N_repeats,list):
            if len(self.params.N_repeats) == 1:
                N_repeats = self.params.N_repeats[0]
            else:
                N_repeats = np.prod(self.params.N_repeats)
        else:
            N_repeats = 1
        self.params.N_shots = int(N_img / N_repeats)

        from waxa import img_types
        if self.run_info.imaging_type == img_types.ABSORPTION:
            images_per_shot = 3
        else:
            images_per_shot = self.params.N_pwa_per_shot + 2

        N_img = images_per_shot * N_img # 3 images per value of independent variable (xvar)

        if self.setup_camera:
            msg += f" ({N_img} images)"
        notes = getattr(self, "_scan_line_notes", None) or {}
        if notes:
            msg += " | " + " | ".join(str(v) for v in notes.values())
        console.info(msg)
        return N_img

    def prepare_image_array(self):
        # Check and normalise the run-owned camera fields first (AndorParams:
        # trigger, frame transfer, sensor_roi -> resolution).  This runs inside
        # init_xvars, i.e. in finish_prepare before INIT_RUN, so a refused
        # value raises here without reserving a run id.  getattr: a bare
        # CameraParams placeholder may not have the method.
        prepare_for_run = getattr(self.camera_params, "prepare_for_run", None)
        if callable(prepare_for_run):
            prepare_for_run()
        if self.run_info.save_data:
            # The frames' dtype, declared to liveOD at INIT_RUN (images_dtype):
            # a camera that knows it says so (BaslerParams: uint8 for Mono8,
            # uint16 for Mono12 / Mono12p); otherwise by camera type.
            image_dtype = getattr(self.camera_params, "image_dtype", None)
            if callable(image_dtype):
                dtype = image_dtype()
            elif self.camera_params.camera_type == 'andor':
                dtype = np.uint16
            else:
                dtype = np.uint8
            # Only the shape and dtype are used (the INIT_RUN declaration);
            # liveOD writes the frames.  A zero-memory, read-only view of one
            # zero, not N_img full frames held for the whole run.
            shape = (self.params.N_img,) + tuple(self.camera_params.resolution)
            self.images = np.broadcast_to(np.zeros((), dtype=dtype), shape)
            self.image_timestamps = np.zeros((self.params.N_img,))
        else:
            self.images = np.array([0])
            self.image_timestamps = np.array([0])

    # ---- every shot: step to the next point of the scan

    def step_scan(self,idx=0) -> TBool:
        '''
        Advances the counters of the xvars to the next step in the scan.
        Returns False when the scan is over.

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

    @kernel
    def update_scan_xvar_counters(self):
        """The kernel's copy of each xvar counter, from the host's."""
        for xvar in self.scan_xvars:
            xvar.counter = self.return_scan_xvar_counter(xvar.position)

    def return_scan_xvar_counter(self, idx) -> TInt32:
        return self.scan_xvars[idx].counter

    # ---- end of the run

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
