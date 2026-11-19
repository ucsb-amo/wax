"""Moved to ``beacon.drivers.andor.driver``; this name is the same module."""
import sys

import beacon.drivers.andor.driver as _moved

sys.modules[__name__] = _moved
