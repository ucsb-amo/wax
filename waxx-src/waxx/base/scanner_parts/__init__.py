"""The parts of the Scanner nobody needs to read to write or follow an experiment.

``waxx.base.scanner.Scanner`` inherits every class here, so each method is
still ``self.<name>`` on an experiment, as before.

==================  ==========================================================
module              what it does
==================  ==========================================================
adjust              ``self.adjust(...)`` and the liveOD Adjust panel's specs
param_sync          copies the host's params into the kernel before each shot
shot_grid           counts shots and images, steps the xvar counters
abort_state         tells the host what a run that raised left the channels at
exposure_delays     camera-stream trigger lead times, loaded at kernel start
==================  ==========================================================
"""
from .adjust import AdjustSpec, AdjustPanel
from .param_sync import ParamSync
from .shot_grid import ShotGrid
from .abort_state import AbortState, WRITE_FAILURES, SNAPSHOT_PARAMS
from .exposure_delays import ExposureDelays
