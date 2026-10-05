"""Small Qt habits for GUIs that run for days (the dashboards).

* :func:`set_style_if_changed` -- ``setStyleSheet`` re-parses the sheet and
  re-polishes the widget (and its children) on every call, even with the
  same string; refresh paths that run every second should only restyle on a
  change.
* :func:`delete_later` -- a ``QMenu(self)`` / ``QMessageBox(self)`` built per
  right-click or per confirm is parented to a long-lived widget and lives as
  long as it does unless deleted; call this once its result has been read
  (deletion is deferred to the event loop, so reading it afterwards in the
  same function is safe).  Objects without ``deleteLater`` (test stand-ins)
  are left alone.
"""

from __future__ import annotations

from waxx.util.dashboard.restyle import set_style


def set_style_if_changed(widget, css: str) -> bool:
    """``widget.setStyleSheet(css)`` unless it already has exactly ``css``.
    Returns True when it restyled. (The dashboard's ``restyle.set_style``.)"""
    return set_style(widget, css)


def delete_later(obj) -> None:
    """``obj.deleteLater()`` if it has one."""
    fn = getattr(obj, "deleteLater", None)
    if fn is not None:
        try:
            fn()
        except RuntimeError:          # already deleted
            pass


__all__ = ["delete_later", "set_style_if_changed"]
