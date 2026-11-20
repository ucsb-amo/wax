# Lazy top-level imports.  Eager imports here would be executed by every
# loky worker subprocess (which only needs waxa.fitting / waxa.image_processing)
# and would drag in server_talk (network I/O) and PyQt6 (GUI), causing
# BrokenProcessPool errors.  Python 3.7+ module __getattr__ defers these
# until the name is first accessed.

import sys
import types
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .atomdata import atomdata
    from .atomdata_vault import AtomdataVault
    from .data.load_atomdata import load_atomdata
    from .roi import ROI
    from .config.img_types import img_types
    from .config.expt_params import ExptParams
    from .climate.client import ClimateClient

_lazy = {
    'atomdata':     '.atomdata',
    'AtomdataVault': '.atomdata_vault',
    'load_atomdata': '.data.load_atomdata',
    'ROI':          '.roi',
    'img_types':    '.config.img_types',
    'ExptParams':   '.config.expt_params',
    'ClimateClient': '.climate.client',
}

def __getattr__(name):
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name], __name__)
        val = getattr(mod, name)
        # Cache so subsequent accesses are fast
        globals()[name] = val
        return val
    raise AttributeError(f"module 'waxa' has no attribute {name!r}")


class _Waxa(types.ModuleType):
    """``atomdata`` is both a submodule and the class this package exposes.

    Importing the submodule (AtomdataVault does) sets ``waxa.atomdata`` to the
    module, so ``__getattr__`` above never fires and ``from waxa import atomdata``
    returns the module. Keep the class. ``from waxa.atomdata import x`` is
    unaffected: it goes through ``sys.modules``.
    """

    def __setattr__(self, name, value):
        if name == 'atomdata' and isinstance(value, types.ModuleType):
            value = value.atomdata
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Waxa
