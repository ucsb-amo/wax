"""seqview -- an interactive pulse-sequence viewer (PyQt6 + pyqtgraph).

Feed it a :class:`~waxx.util.seqview.bundle.Bundle` (see that module for
the schema) and it opens a window with stacked lanes, pulse objects that
know their source line, measurement cursors, a minimap, shot bands and a
code pane. The window is machine-agnostic: the producer (for the OPX,
``kexp.control.opx.viewer``) does all the lab-specific work.

    from waxx.util.seqview import show
    show(bundle)                 # separate process; the notebook never blocks
    show(bundle, inline=True)    # in-process (a Qt event loop must be running)

``show`` reuses an already open viewer process (the window keeps its zoom,
selection and lane layout and just swaps the data) and starts one when none
is listening. ``python -m waxx.util.seqview <bundle.npz>`` opens one by hand.
"""

from waxx.util.seqview.bundle import Bundle, VERSION
from waxx.util.seqview.launch import show, bundle_dir, send_to_viewer

__all__ = ['Bundle', 'VERSION', 'show', 'bundle_dir', 'send_to_viewer']
