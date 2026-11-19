"""Moved to ``beacon.drivers.andor.backend``; this name is the same module."""
import sys

import beacon.drivers.andor.backend as _moved

sys.modules[__name__] = _moved
