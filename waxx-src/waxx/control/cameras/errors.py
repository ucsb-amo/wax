"""Moved to ``beacon.camera.errors``; this name is the same module."""
import sys

import beacon.camera.errors as _moved

sys.modules[__name__] = _moved
