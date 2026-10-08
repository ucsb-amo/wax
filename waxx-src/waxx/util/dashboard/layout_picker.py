"""Toolbar dropdown for choosing, adding and managing named dashboard layouts.

``Layout [ Lab view • v ] [+] [-]``

* The dropdown lists the built-in *Default* placement, then every saved
  layout.  Clicking a name applies it.  Each saved row carries its own
  buttons: rename, save the current arrangement over it, delete.
* ``+`` saves the current arrangement as a new layout; ``-`` deletes the
  layout currently selected.
* A ``•`` after the name means the arrangement was changed since that layout
  was applied (the change is still remembered for the next launch).

The widget only emits requests; :class:`DashboardMainWindow` does the work
and calls :meth:`LayoutPicker.set_entries` to refresh it.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QWidget,
    QWidgetAction,
)

from waxx.util.dashboard import theme
from waxx.util.dashboard.layout_store import DEFAULT_NAME, UNSAVED_LABEL


def _paint_glyph(kind: str, color: str, size: int = 12) -> QPixmap:
    """Small line glyphs drawn with QPainter (no emoji-font fallback)."""
    dpr = 2.0
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(color)
    s = float(size)
    pen = QPen(c, s * 0.12)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    if kind == "plus":
        p.drawLine(QPointF(s * 0.5, s * 0.18), QPointF(s * 0.5, s * 0.82))
        p.drawLine(QPointF(s * 0.18, s * 0.5), QPointF(s * 0.82, s * 0.5))
    elif kind == "minus":
        p.drawLine(QPointF(s * 0.18, s * 0.5), QPointF(s * 0.82, s * 0.5))
    elif kind == "rename":   # pencil
        p.drawPolygon(QPolygonF([
            QPointF(s * 0.20, s * 0.80), QPointF(s * 0.22, s * 0.62), QPointF(s * 0.68, s * 0.16),
            QPointF(s * 0.84, s * 0.32), QPointF(s * 0.38, s * 0.78),
        ]))
        p.drawLine(QPointF(s * 0.58, s * 0.26), QPointF(s * 0.74, s * 0.42))
    elif kind == "overwrite":   # arrow down into a tray
        p.drawLine(QPointF(s * 0.5, s * 0.12), QPointF(s * 0.5, s * 0.62))
        p.drawPolyline(QPolygonF([QPointF(s * 0.30, s * 0.44), QPointF(s * 0.5, s * 0.64),
                                  QPointF(s * 0.70, s * 0.44)]))
        p.drawPolyline(QPolygonF([QPointF(s * 0.14, s * 0.62), QPointF(s * 0.14, s * 0.86),
                                  QPointF(s * 0.86, s * 0.86), QPointF(s * 0.86, s * 0.62)]))
    elif kind == "delete":   # bin
        p.drawLine(QPointF(s * 0.16, s * 0.26), QPointF(s * 0.84, s * 0.26))
        p.drawLine(QPointF(s * 0.40, s * 0.14), QPointF(s * 0.60, s * 0.14))
        p.drawPolyline(QPolygonF([QPointF(s * 0.26, s * 0.30), QPointF(s * 0.32, s * 0.88),
                                  QPointF(s * 0.68, s * 0.88), QPointF(s * 0.74, s * 0.30)]))
    elif kind == "check":
        p.drawPolyline(QPolygonF([QPointF(s * 0.16, s * 0.52), QPointF(s * 0.40, s * 0.76),
                                  QPointF(s * 0.84, s * 0.26)]))
    p.end()
    return pm


def _glyph_icon(kind: str) -> QIcon:
    icon = QIcon()
    icon.addPixmap(_paint_glyph(kind, theme.FG), QIcon.Mode.Normal)
    icon.addPixmap(_paint_glyph(kind, theme.FG_STRONG), QIcon.Mode.Active)
    icon.addPixmap(_paint_glyph(kind, theme.FG_DISABLED), QIcon.Mode.Disabled)
    return icon


def _icon_button(parent: QWidget, kind: str, tip: str) -> QToolButton:
    btn = QToolButton(parent)
    btn.setAutoRaise(True)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedSize(22, 22)
    btn.setIconSize(QSize(12, 12))
    btn.setIcon(_glyph_icon(kind))
    btn.setToolTip(tip)
    btn.setStyleSheet(
        "QToolButton { border: 0; border-radius: 3px; padding: 0; }"
        f"QToolButton:hover {{ background: {theme.BG_BUTTON_HOVER}; }}"
    )
    return btn


class _LayoutRow(QWidget):
    """One dropdown entry: check mark, name, and (for saved layouts) its buttons."""

    def __init__(self, picker: "LayoutPicker", name: str, *, active: bool, builtin: bool,
                 tip: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("layoutRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setStyleSheet(
            f"#layoutRow {{ background: transparent; border-radius: 3px; }}"
            f"#layoutRow:hover {{ background: {theme.HIGHLIGHT}; }}"
        )
        self.name = name
        h = QHBoxLayout(self)
        h.setContentsMargins(4, 1, 4, 1)
        h.setSpacing(2)

        mark = QLabel(self)
        mark.setFixedSize(16, 16)
        if active:
            mark.setPixmap(_paint_glyph("check", theme.ACCENT, 12))
        h.addWidget(mark)

        self.name_btn = QPushButton(name, self)
        self.name_btn.setFlat(True)
        self.name_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.name_btn.setMinimumWidth(150)
        self.name_btn.setToolTip(tip or f"Apply '{name}'")
        weight = "600" if active else "normal"
        self.name_btn.setStyleSheet(
            f"QPushButton {{ text-align: left; border: 0; padding: 3px 6px; color: {theme.FG}; "
            f"font-weight: {weight}; background: transparent; }}"
        )
        self.name_btn.clicked.connect(lambda: picker._emit_later(picker.apply_requested, name))
        h.addWidget(self.name_btn, 1)

        self.buttons: dict[str, QToolButton] = {}
        if not builtin:
            for kind, tip_, sig in (
                ("rename", "Rename", picker.rename_requested),
                ("overwrite", "Save the current arrangement over this layout", picker.overwrite_requested),
                ("delete", "Delete this layout", picker.delete_requested),
            ):
                b = _icon_button(self, kind, tip_)
                b.clicked.connect(lambda _c=False, s=sig: picker._emit_later(s, name))
                h.addWidget(b)
                self.buttons[kind] = b


class LayoutPicker(QWidget):
    """``Layout [name v] [+] [-]`` for the dashboard toolbar."""

    apply_requested = pyqtSignal(str)       # DEFAULT_NAME or a saved name
    add_requested = pyqtSignal()
    rename_requested = pyqtSignal(str)
    overwrite_requested = pyqtSignal(str)
    delete_requested = pyqtSignal(str)
    import_requested = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LayoutPicker")
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 0, 6, 0)
        h.setSpacing(2)

        lbl = QLabel("Layout", self)
        lbl.setStyleSheet(f"QLabel {{ color: {theme.FG_MUTED}; padding-right: 4px; }}")
        h.addWidget(lbl)

        self.dropdown = QToolButton(self)
        self.dropdown.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.dropdown.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.dropdown.setMinimumWidth(170)
        self.dropdown.setStyleSheet(
            f"QToolButton {{ background: {theme.BG_SUNKEN}; color: {theme.FG}; border: 1px solid {theme.BORDER};"
            f" border-radius: 3px; padding: 2px 22px 2px 8px; text-align: left; }}"
            f"QToolButton:hover {{ border-color: {theme.BORDER_STRONG}; }}"
            "QToolButton::menu-indicator { subcontrol-origin: padding; subcontrol-position: center right;"
            " right: 6px; }"
        )
        self.menu = QMenu(self.dropdown)
        self.dropdown.setMenu(self.menu)
        h.addWidget(self.dropdown)

        self.add_btn = _icon_button(self, "plus", "Save the current arrangement as a new layout")
        self.add_btn.clicked.connect(self.add_requested.emit)
        h.addWidget(self.add_btn)
        self.remove_btn = _icon_button(self, "minus", "Delete the selected layout")
        self.remove_btn.clicked.connect(lambda: self.delete_requested.emit(self._active))
        h.addWidget(self.remove_btn)

        self._rows: dict[str, _LayoutRow] = {}
        self._active = ""
        self.set_entries([], "", False)

    # ------------------------------------------------------------------

    def _emit_later(self, signal, *args) -> None:
        """Close the menu first, so a dialog the slot opens is not under it."""
        self.menu.close()
        QTimer.singleShot(0, lambda: signal.emit(*args))

    def set_entries(self, names: list[str], active: str, modified: bool,
                    saved_at: "dict[str, str] | None" = None) -> None:
        """Rebuild the dropdown: *names* are the saved layouts, *active* the one on screen."""
        saved_at = saved_at or {}
        self._active = active
        self.menu.clear()
        self._rows.clear()

        def add_row(name: str, builtin: bool, tip: str = "") -> None:
            row = _LayoutRow(self, name, active=(name == active), builtin=builtin, tip=tip)
            act = QWidgetAction(self.menu)
            act.setDefaultWidget(row)
            self.menu.addAction(act)
            self._rows[name] = row

        add_row(DEFAULT_NAME, True, "Apply the built-in default placement")
        if names:
            self.menu.addSeparator()
            for name in names:
                when = saved_at.get(name)
                add_row(name, False, f"Apply '{name}'" + (f"  (saved {when})" if when else ""))
        self.menu.addSeparator()
        for text, sig in (
            ("Save current as new layout…", self.add_requested),
            ("Import layout from file…", self.import_requested),
            ("Export current layout to file…", self.export_requested),
        ):
            a = QAction(text, self.menu)
            a.triggered.connect(lambda _c=False, s=sig: QTimer.singleShot(0, s.emit))
            self.menu.addAction(a)

        shown = active or UNSAVED_LABEL
        self.dropdown.setText(f"{shown}  •" if (modified and active) else shown)
        if not active:
            tip = "This arrangement is not saved as a named layout - + saves it"
        elif modified:
            tip = (f"Changed since '{active}' was applied (still remembered for the next launch).\n"
                   f"The save button on its row stores the change in '{active}'.")
        else:
            tip = f"'{active}' is applied"
        self.dropdown.setToolTip(tip)
        named = bool(active) and active != DEFAULT_NAME
        self.remove_btn.setEnabled(named)
        self.remove_btn.setToolTip(f"Delete '{active}'" if named else "Select a saved layout to delete it")

    def row(self, name: str) -> "_LayoutRow | None":
        return self._rows.get(name)


__all__ = ["LayoutPicker"]
