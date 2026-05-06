def __getattr__(name):
    if name == 'Expt':
        from .base.expt import Expt
        return Expt
    if name == 'img_types':
        from waxa import img_types
        return img_types
    raise AttributeError(f"module 'waxx' has no attribute {name!r}")

# from .base.expt import Expt
# from waxa.config.img_types import img_types