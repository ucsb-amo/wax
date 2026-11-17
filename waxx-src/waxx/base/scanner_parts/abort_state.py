"""Tells the host what a run that raised left the channels at.

ARTIQ writes kernel attributes back to the host only when a kernel returns.
So when an exception ends ``scan()`` (the liveOD Abort button, an error in a
shot or a hook), nothing on the host knows what the hardware was left at.
``scan()``'s exception handlers therefore call ``_hand_over_abort_state``
before re-raising: it copies every channel's kernel-side state into arrays and
sends them to the host in one RPC. ``Expt._report_abort_state`` forwards them
to the monitor server as the run's end state.

Until ``Expt.finish_prepare_wax`` hands over the monitor's snapshot kernel and
arrays, the snapshot kernel is a no-op.
"""
from artiq.experiment import *
from artiq.language.core import kernel_from_string, now_mu
import numpy as np

# Exceptions a channel write can raise itself.  A run that ends on one may
# leave that one channel's cached value and the hardware apart (DAC_CH.set
# caches before it writes, the ramps only after the last point), so the state
# it reports is flagged as not trusted (see Expt._report_abort_state).
WRITE_FAILURES = ("RTIOUnderflow", "RTIODestinationUnreachable", "ValueError")

# Parameters of the channel-snapshot kernel (Monitor._build_snapshot_kernel).
SNAPSHOT_PARAMS = ["expt", "f", "a", "v", "s", "dv", "ts"]


class AbortState:

    def _init_abort_state(self):
        self._abort_snapshot_kernels = [kernel_from_string(SNAPSHOT_PARAMS, "pass")]
        self._abort_snap_dds_f = np.zeros(1)
        self._abort_snap_dds_a = np.zeros(1)
        self._abort_snap_dds_v = np.zeros(1)
        self._abort_snap_dds_sw = np.zeros(1, dtype=np.int32)
        self._abort_snap_dac_v = np.zeros(1)
        self._abort_snap_ttl_s = np.zeros(1, dtype=np.int32)

    @kernel
    def _hand_over_abort_state(self, what):
        # Events already on the timeline play out whether or not anyone
        # waits; waiting makes the snapshot what the hardware is at.
        self.core.wait_until_mu(now_mu())
        self._abort_snapshot_kernels[0](self, self._abort_snap_dds_f,
                                        self._abort_snap_dds_a, self._abort_snap_dds_v,
                                        self._abort_snap_dds_sw, self._abort_snap_dac_v,
                                        self._abort_snap_ttl_s)
        self._report_abort_state(what, self._abort_snap_dds_f, self._abort_snap_dds_a,
                                 self._abort_snap_dds_v, self._abort_snap_dds_sw,
                                 self._abort_snap_dac_v, self._abort_snap_ttl_s)

    def _report_abort_state(self, what, dds_f, dds_a, dds_v, dds_sw, dac_v, ttl_s):
        """RPC from scan()'s exception handler.  Overridden in Expt."""
        pass
