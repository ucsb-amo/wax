"""Camera-stream trigger lead times, loaded into the kernel when it starts.

A triggered camera stream fires its TTL early by the camera's exposure start
delay. That delay is only known at the run's claim, which can come after the
kernel was compiled, so the kernel reads it from cells it fills itself at its
start. The whole mechanism is described in
``waxx.control.cameras.exposure_delay``; this is the kernel-side loader.
"""
from artiq.experiment import *
import numpy as np


class ExposureDelays:

    def _init_exposure_delays(self):
        # The list always holds a spare cell, so it is never empty for the
        # compiler.
        from waxx.control.cameras.exposure_delay import init_cells
        init_cells(self)
        self._exposure_delays_loaded = False

    @kernel
    def load_exposure_delays(self):
        """Write the exposure delays the run's claim decided for its triggered
        camera streams into the kernel's cells (waxx.control.cameras
        .exposure_delay): ONE synchronous RPC, then a store per cell. The
        trigger() kernels read the cells, so from here on every edge leads by
        the delay the camera reported at the claim, with this run's settings
        -- for a warm-launched run too, whose kernel was compiled before the
        claim. Called first thing by kexp's init_kernel and by scan(); does
        nothing once a kernel has loaded the claim's values (a call before
        the claim loads what there is and stays armed). Emits no RTIO event
        and moves no cursor; the RPC's wall time (~ms) comes out of the slack,
        so call it where none is needed: kexp's init_kernel calls it before
        its core.reset(); in scan() the next RTIO event comes after a
        break_realtime (pre_scan's warm-ups, or the shot loop's)."""
        if not self._exposure_delays_loaded:
            d = self._exposure_delays_at_claim()
            n = len(self._exposure_delay_cells)
            for i in range(n):
                self._exposure_delay_cells[i].value = d[i]
            self._exposure_delays_loaded = d[n] > 0.5

    def _exposure_delays_at_claim(self) -> TArray(TFloat):
        """RPC target of load_exposure_delays: one float per cell, then 1.0
        when they are all final (see exposure_delays_for_kernel). Never
        raises; then the run's records are refreshed (exposure_delays_loaded)."""
        from waxx.control.cameras.exposure_delay import exposure_delays_for_kernel, cells
        try:
            d = exposure_delays_for_kernel(self)
        except Exception as e:                      # belt and braces: never into the kernel
            c = cells(self)
            print(f"[exposure delay] !! could not collect the claim's exposure delays "
                  f"({e!r}); the kernel keeps its compiled values")
            d = np.array([float(x.value) for x in c] + [0.], dtype=np.float64)
        try:
            self.exposure_delays_loaded()
        except Exception as e:
            print(f"[exposure delay] WARNING: refreshing the run's records failed ({e!r})")
        return d

    def exposure_delays_loaded(self):
        """Machine hook: the kernel just took the claim's exposure delays
        (kexp.Base: rewrites the run's diagnostic_images record). Host side,
        inside a sync RPC: keep it short; must not raise."""
        pass
