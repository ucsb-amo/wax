import os as _os

if _os.environ.get("WAXX_QUEUE_WARM"):
    # A process the run queue warm-launched: queue_gate records each file as
    # it is loaded (the freshness manifest), from as early as possible --
    # here, the first import of waxx. Presence only: queue_gate pops it later.
    from waxx.util.device_state import queue_gate as _queue_gate  # noqa: F401,E402


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