# Lazy camera exports.  This __init__ runs for EVERY `from waxx.control.<x>
# import ...` (raman_beams, beat_lock, slm, ...), i.e. in every experiment
# process.  Importing the camera drivers eagerly here pulled in pylablib (+numba)
# and pypylon -- about 1.2 s per run -- for processes that never open a camera.
# `from waxx.control import BaslerUSB` still works: module __getattr__ (PEP 562)
# imports the driver on first access.

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .cameras.basler_usb import BaslerUSB
    from .cameras.andor import AndorEMCCD
    from .cameras.dummy_cam import DummyCamera

_lazy = {
    'BaslerUSB':   '.cameras.basler_usb',
    'AndorEMCCD':  '.cameras.andor',
    'DummyCamera': '.cameras.dummy_cam',
}

def __getattr__(name):
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'waxx.control' has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | set(_lazy))
