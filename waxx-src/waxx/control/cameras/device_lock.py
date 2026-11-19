"""Moved to ``beacon.camera.device_lock``; this name is the same module (same
lock directory and ``WAXX_*`` environment names, so old and new code take the
same lock file)."""
import sys

import beacon.camera.device_lock as _moved

sys.modules[__name__] = _moved
