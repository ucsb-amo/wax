"""Helpers for refactoring existing ``QMainWindow`` GUIs into embeddable panels.

Every kexp control GUI exposes a ``build_panel(*args, **kwargs) -> QWidget``
function.  The standalone ``QMainWindow`` wrapper just calls
``setCentralWidget(build_panel(...))`` so today's ``.bat`` launchers keep
working.

Embedded panels must satisfy a small contract:

* No ``QMainWindow`` features (statusBar, menuBar, toolBar) accessed via
  ``self.parent()`` - inside a dashboard ``self.parent()`` is a ``QDockWidget``.
* No ``Qt.WindowType.Window`` popups owned by the panel - popups are easy
  to lose when the dashboard is brought to front.
* No ``setMinimumSize`` / ``setFixedSize`` / ``setMaximumSize`` on the
  top-level body widget - the dashboard owns sizing.
* Any ``QTimer`` started in ``__init__`` must be stopped in ``cleanup()`` so
  closed panels don't keep polling dead servers.

This module provides:

* :class:`WidgetPanelBase` - base class with a ``cleanup()`` hook the framework
  wires to the dock's ``destroyed`` signal.
* :func:`replace_status_bar` - drop-in replacement that returns a ``QLabel``
  intended to sit at the bottom of the body layout (instead of ``statusBar()``).
* :func:`auto_cleanup_timers` - walks child ``QTimer``s and stops them.
* :func:`lint_panel` - non-fatal post-construction check that logs WARNING for
  any banned anti-pattern (popup windows, hardcoded min-size, etc.).
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6 import QtCore
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QLabel, QWidget


_LOG = logging.getLogger("waxx.dashboard.embed")


class WidgetPanelBase(QWidget):
    """Optional base class for embeddable panels.

    Provides a :meth:`cleanup` hook the dashboard guarantees to call when the
    panel is destroyed (timers, threads, sockets should be torn down here).
    Subclasses override :meth:`cleanup` - the default implementation calls
    :func:`auto_cleanup_timers`.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

    def cleanup(self) -> None:
        """Tear down timers / threads / sockets owned by this panel."""
        auto_cleanup_timers(self)


def replace_status_bar(initial_text: str = "") -> QLabel:
    """Return a small label suitable for the bottom of a panel layout.

    Drop-in replacement for the ``QMainWindow.statusBar().showMessage(...)``
    pattern that does not work inside a dock.  Caller is responsible for
    adding the returned widget to its layout.
    """
    label = QLabel(initial_text)
    label.setStyleSheet(
        "QLabel { color: #888; font-size: 11px; padding: 2px 4px; "
        "border-top: 1px solid #ddd; }"
    )
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return label


def auto_cleanup_timers(widget: QWidget) -> int:
    """Stop every ``QTimer`` child of *widget*.

    Returns the number of timers that were running and got stopped.  Safe to
    call multiple times; no-op for already-stopped timers.
    """
    n = 0
    for timer in widget.findChildren(QTimer):
        if timer.isActive():
            try:
                timer.stop()
                n += 1
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.warning("auto_cleanup_timers: failed to stop %r: %r", timer, exc)
    if n:
        _LOG.debug("auto_cleanup_timers: stopped %d timers in %s", n, widget.__class__.__name__)
    return n


def lint_panel(widget: QWidget, panel_id: str = "<unknown>") -> list[str]:
    """Check *widget* against the embedded-panel anti-patterns.

    Returns a list of warning strings (also emitted to ``waxx.dashboard.embed``
    at WARNING).  Never raises - lint is advisory.
    """
    warnings: list[str] = []

    # 1. Top-level widget MUST NOT carry a hard min/max size.
    min_sz = widget.minimumSize()
    if min_sz.width() > 0 or min_sz.height() > 0:
        warnings.append(
            f"panel {panel_id}: top-level setMinimumSize({min_sz.width()}, {min_sz.height()})"
            " - panel sizing should be owned by the dashboard via PanelSpec.preferred_min_size."
        )

    # 2. No child windows with Qt.WindowType.Window.
    for child in widget.findChildren(QWidget):
        if child is widget:
            continue
        try:
            wt = child.windowType()
        except Exception:
            continue
        if wt == Qt.WindowType.Window:
            warnings.append(
                f"panel {panel_id}: child widget {child.__class__.__name__} has"
                " Qt.WindowType.Window - popups owned by panels get lost inside a dashboard."
            )

    for msg in warnings:
        _LOG.warning(msg)
    return warnings


def embed_main_window(panel: QWidget, mw, *, take_menus: bool = True, compact: bool = False,
                      embed_as_window: bool = False) -> None:
    """Embed an existing ``QMainWindow`` inside *panel* by re-parenting its
    central widget (and optionally its menu/tool bars) into *panel*'s layout.

    The original QMainWindow is kept alive (parented to *panel*) so any
    signal/slot wiring it set up internally continues to work.  Anything
    that truly requires a top-level window (global shortcuts, etc.) will
    not work inside a dock - that's expected.

    Parameters
    ----------
    compact:
        If True, recursively zero ``minimumSize``/``minimumWidth``/
        ``minimumHeight`` on the central widget and all children so the
        dashboard user can shrink the dock to whatever pixel size they
        want.  Use sparingly for GUIs that hard-coded a large minimum.
    embed_as_window:
        If True, embed the entire ``QMainWindow`` widget (visible) instead
        of extracting its central widget.  Required for GUIs that host
        nested ``QDockWidget`` children inside their own ``QMainWindow``
        (e.g. Basler multi-camera dock host) - those docks live on the
        outer main window, not the central widget, and would be hidden if
        we extracted only the central widget.  ``take_menus`` is ignored
        in this mode (the embedded QMainWindow keeps its own menus).
    """
    from PyQt6.QtWidgets import QMenuBar, QToolBar, QVBoxLayout  # noqa: PLC0415

    layout = panel.layout()
    if layout is None:
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    if embed_as_window:
        # Strip the WindowFlags so it doesn't try to be a top-level window
        # when re-parented; embed the whole QMainWindow visibly.
        try:
            mw.setWindowFlags(Qt.WindowType.Widget)
        except Exception:
            pass
        mw.setParent(panel)
        layout.addWidget(mw, 1)
        mw.show()
        if compact:
            _strip_minimum_sizes(panel)
            _strip_minimum_sizes(mw)
        return

    if take_menus:
        mb = mw.menuBar() if hasattr(mw, "menuBar") else None
        if isinstance(mb, QMenuBar) and len(mb.actions()) > 0:
            mb.setParent(panel)
            layout.addWidget(mb)
        for tb in mw.findChildren(QToolBar):
            tb.setParent(panel)
            layout.addWidget(tb)

    central = mw.centralWidget() if hasattr(mw, "centralWidget") else None
    if central is not None:
        central.setParent(panel)
        layout.addWidget(central, 1)
    else:
        mw.setParent(panel)
        layout.addWidget(mw, 1)

    # Keep the QMainWindow alive (signal wiring lives on it).
    mw.setParent(panel)
    mw.hide()

    if compact:
        _strip_minimum_sizes(panel)
        if central is not None:
            _strip_minimum_sizes(central)


def _strip_minimum_sizes(root: QWidget) -> None:
    """Recursively clear minimumSize/minimumWidth/minimumHeight on *root* and children."""
    try:
        root.setMinimumSize(0, 0)
        root.setMinimumWidth(0)
        root.setMinimumHeight(0)
    except Exception:
        pass
    for child in root.findChildren(QWidget):
        try:
            child.setMinimumSize(0, 0)
            child.setMinimumWidth(0)
            child.setMinimumHeight(0)
        except Exception:
            pass


__all__ = ["WidgetPanelBase", "replace_status_bar", "auto_cleanup_timers", "lint_panel", "embed_main_window"]
