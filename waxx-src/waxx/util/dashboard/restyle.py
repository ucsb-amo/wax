"""Change-only widget setters for per-tick status updates.

``QWidget.setStyleSheet`` has no "unchanged" shortcut: every call re-polishes
the widget and all of its children.  Status pills that are restyled on every
snapshot (1-10 Hz, in a dozen panels, for days) add up to a large share of the
dashboard's GUI-thread time.  These helpers compare with the widget's current
value first and only call the setter when something actually changed.
"""

from __future__ import annotations


def set_style(widget, css: str) -> bool:
    """``widget.setStyleSheet(css)`` unless it already has exactly *css*.

    Returns True when the stylesheet was changed.
    """
    if widget.styleSheet() == css:
        return False
    widget.setStyleSheet(css)
    return True


def set_tooltip(widget, tip: str) -> bool:
    """``widget.setToolTip(tip)`` unless the tooltip is already *tip*."""
    if widget.toolTip() == tip:
        return False
    widget.setToolTip(tip)
    return True


__all__ = ["set_style", "set_tooltip"]
