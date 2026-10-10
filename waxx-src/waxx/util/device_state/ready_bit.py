"""The monitor experiment's state constants (the ``status`` ReadyBit).

They live here, apart from :mod:`waxx.util.comms_server.comm_server` (which
re-exports them as before), so that a GUI module can use them without
importing the ``waxx.util.comms_server`` package -- whose ``__init__``
imports ``beacon.discovery``, and that starts the discovery listener (a UDP
socket bound to the discovery port) at import.
"""


class ReadyBit:
    READY = 0
    LOADING = 1
    NOT_READY = 2


STATES = ReadyBit()

__all__ = ["ReadyBit", "STATES"]
