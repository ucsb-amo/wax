"""Markers (pins) on the liveOD image: the item drawn on the plot, the editor for
one marker (right-click a marker), and the panel listing all of them (the Markers
button).

A marker's size is a length on the image, stored in image pixels, so it zooms with
the image like the cloud next to it does. The editors show and take the size in
whatever the viewer is showing (px, or µm at the atoms). A hidden marker is kept
(and listed in the panel, with its eye closed) but not drawn.

None of this stores anything: the viewer owns the list, and the liveOD server keeps
it per camera (``marker_store``).
"""

import random

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PyQt6.QtWidgets import (QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QScrollArea, QToolButton, QVBoxLayout, QWidget)

from waxx.util.live_od.gui import theme
from waxx.util.live_od.marker_store import SHAPES, SIZE_LIMITS_PX, MAX_LABEL_LENGTH

MIN_HIT_RADIUS_PX = 7       # screen pixels: a tiny marker can still be grabbed


def random_marker_color(avoid=()) -> str:
    """A random bright, saturated colour for a new marker, easy to see on any
    colormap. Of a few random hues, the one furthest from the colours in ``avoid``
    (the markers already there), so two markers rarely come out alike."""
    taken = [QColor(c).hsvHueF() for c in avoid]
    taken = [h for h in taken if h >= 0]        # grey has no hue

    def distance(hue):
        return min((min(abs(hue - h), 1 - abs(hue - h)) for h in taken), default=1.0)
    hue = max((random.random() for _ in range(8)), key=distance)
    return QColor.fromHsvF(hue, 0.75, 1.0).name()


def _eye_icon(color: str) -> QIcon:
    """An open eye (checked: shown) and a struck-through one (unchecked: hidden),
    drawn rather than taken from a font: Windows draws the eye emoji in colour."""
    icon = QIcon()
    for shown in (True, False):
        pixmap = QPixmap(32, 32)            # drawn at 2x, shown at 16
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen_color = QColor(color)
        if not shown:
            pen_color.setAlpha(150)
        painter.setPen(QPen(pen_color, 2.4))
        outline = QPainterPath(QPointF(3, 16))
        outline.quadTo(QPointF(16, 3), QPointF(29, 16))
        outline.quadTo(QPointF(16, 29), QPointF(3, 16))
        painter.drawPath(outline)
        painter.setBrush(pen_color)
        painter.drawEllipse(QPointF(16, 16), 4.5, 4.5)
        if not shown:
            painter.drawLine(QPointF(6, 27), QPointF(26, 5))
        painter.end()
        icon.addPixmap(pixmap, QIcon.Mode.Normal, QIcon.State.On if shown else QIcon.State.Off)
    return icon


def _unit_path(shape: str) -> QPainterPath:
    """pyqtgraph's symbol, scaled so its larger side is exactly 1."""
    path = Symbols[shape]
    rect = path.boundingRect()
    extent = max(rect.width(), rect.height()) or 1.0
    return QTransform().scale(1.0 / extent, 1.0 / extent).map(path)


class MarkerItem(pg.GraphicsObject):
    """One marker, in image coordinates. Drag to move; right-click to edit."""

    sigMoved = pyqtSignal(object)           # the drag ended
    sigEditRequested = pyqtSignal(object, object)   # self, screen position

    def __init__(self, marker: dict):
        super().__init__()
        self.marker = dict(marker)
        self.moving = False
        self._hovered = False
        self._highlighted = False
        self._drag_offset = QPointF(0, 0)
        self._path = QTransform().scale(self.marker["size"], self.marker["size"]).map(
            _unit_path(self.marker["shape"]))
        self.setZValue(20)
        self.setPos(self.marker["x"], self.marker["y"])
        self.setVisible(not self.marker.get("hidden", False))
        self.setAcceptHoverEvents(True)
        self._label = pg.TextItem(anchor=(0, 1))    # screen-sized text, at the marker's corner
        self._label.setParentItem(self)
        self._label.setPos(self.marker["size"] / 2, self.marker["size"] / 2)
        self._render_label()

    # --- what the viewer reads back -----------------------------------

    def to_dict(self) -> dict:
        marker = dict(self.marker)
        marker["x"], marker["y"] = float(self.pos().x()), float(self.pos().y())
        return marker

    def set_highlighted(self, on: bool):
        """Lit up from the markers panel (hovering its row)."""
        if on != self._highlighted:
            self._highlighted = bool(on)
            self._render_label()
            self.update()

    # --- drawing -------------------------------------------------------

    def is_hovered(self) -> bool:
        return self._hovered and self.isVisible()

    def _lit(self) -> bool:
        return self._hovered or self._highlighted or self.moving

    def _render_label(self):
        color = "#ffffff" if self._lit() else self.marker["color"]
        self._label.setHtml(
            f'<span style="color:{color}; font-weight:bold; font-size:9pt">'
            f'{self.marker["label"]}</span>' if self.marker["label"] else "")

    def _hit_radius(self) -> float:
        """Half the marker's size, but never less than a few screen pixels."""
        radius = self.marker["size"] / 2
        try:
            px_w, px_h = self.pixelSize()
            radius = max(radius, MIN_HIT_RADIUS_PX * max(px_w or 0.0, px_h or 0.0))
        except Exception:
            pass
        return radius

    def boundingRect(self):
        r = self._hit_radius() * 1.15       # room for the pen
        return QRectF(-r, -r, 2 * r, 2 * r)

    def shape(self):
        path = QPainterPath()
        r = self._hit_radius()
        path.addEllipse(QPointF(0, 0), r, r)
        return path

    def viewTransformChanged(self):
        self.prepareGeometryChange()        # the hit radius is in screen pixels

    def paint(self, painter, *_):
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        color = QColor("#ffffff") if self._lit() else QColor(self.marker["color"])
        pen = pg.mkPen(color, width=3 if self._lit() else 2)
        pen.setCosmetic(True)               # the outline stays thin however far in you zoom
        fill = QColor(self.marker["color"])
        fill.setAlpha(110 if self._lit() else 45)
        painter.setPen(pen)
        painter.setBrush(fill)
        painter.drawPath(self._path)

    # --- mouse ---------------------------------------------------------

    def hoverEvent(self, ev):
        hovered = not ev.isExit()
        if hovered:
            ev.acceptDrags(Qt.MouseButton.LeftButton)
            ev.acceptClicks(Qt.MouseButton.RightButton)
        if hovered != self._hovered:
            self._hovered = hovered
            self._render_label()
            self.update()

    def mouseDragEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        ev.accept()
        if ev.isStart():
            self._drag_offset = self.pos() - self.mapToParent(ev.buttonDownPos())
            self.moving = True
        if not self.moving:
            return
        self.setPos(self._drag_offset + self.mapToParent(ev.pos()))
        if ev.isFinish():
            self.moving = False
            self.sigMoved.emit(self)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self.sigEditRequested.emit(self, ev.screenPos())


# ----------------------------------------------------------------------
# Editing
# ----------------------------------------------------------------------

class _ColorButton(QPushButton):
    color_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._color = "#ffffff"
        self.setFixedWidth(34)
        self.setToolTip("Marker color")
        self.clicked.connect(self._choose)

    def set_color(self, color: str):
        self._color = color
        self.setStyleSheet(f"background-color: {color}; border: 1px solid {theme.color('separator')};")

    def color(self) -> str:
        return self._color

    def _choose(self):
        chosen = QColorDialog.getColor(QColor(self._color), self, "Marker color")
        if chosen.isValid():
            self.set_color(chosen.name())
            self.color_changed.emit(chosen.name())


class MarkerFields(QWidget):
    """The widgets that edit one marker's label, shape, size, color and whether it
    is shown. ``changed`` fires on any edit; ``values()`` is what to merge into the
    marker."""

    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._scale = 1.0       # display units per image pixel
        self.label_edit = QLineEdit()
        self.label_edit.setMaxLength(MAX_LABEL_LENGTH)
        self.label_edit.setPlaceholderText("label")
        self.shape_combo = QComboBox()
        for symbol, name in SHAPES.items():
            self.shape_combo.addItem(name, symbol)
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setDecimals(1)
        self.size_spin.setToolTip("The marker's extent on the image; it zooms with the image")
        self.color_button = _ColorButton()
        self.visible_button = QToolButton()
        self.visible_button.setCheckable(True)
        self.visible_button.setChecked(True)
        self.visible_button.setAutoRaise(True)
        self.visible_button.setToolTip("Show or hide this marker (a hidden marker is kept)")
        self._restyle()
        theme.notifier().changed.connect(self._restyle)
        self.visible_button.toggled.connect(lambda _: self.changed.emit())
        self.label_edit.textEdited.connect(lambda _: self.changed.emit())
        self.shape_combo.activated.connect(lambda _: self.changed.emit())
        self.size_spin.valueChanged.connect(lambda _: self.changed.emit())
        self.color_button.color_changed.connect(lambda _: self.changed.emit())

    def _restyle(self, *_):
        self.visible_button.setIcon(_eye_icon(theme.color("muted")))

    def set_marker(self, marker: dict, unit: str, scale: float):
        """Show ``marker``; sizes are shown in ``unit`` (``scale`` of them per pixel).
        A field being typed in is left alone."""
        self._scale = float(scale)
        for widget in (self.label_edit, self.shape_combo, self.size_spin, self.visible_button):
            widget.blockSignals(True)
        self.visible_button.setChecked(not marker.get("hidden", False))
        if not self.label_edit.hasFocus():
            self.label_edit.setText(marker["label"])
        self.shape_combo.setCurrentIndex(max(0, self.shape_combo.findData(marker["shape"])))
        self.size_spin.setSuffix(f" {unit}")
        self.size_spin.setRange(SIZE_LIMITS_PX[0] * scale, SIZE_LIMITS_PX[1] * scale)
        self.size_spin.setSingleStep(max(0.1, round(2 * scale, 1)))
        if not self.size_spin.hasFocus():
            self.size_spin.setValue(marker["size"] * scale)
        self.color_button.set_color(marker["color"])
        for widget in (self.label_edit, self.shape_combo, self.size_spin, self.visible_button):
            widget.blockSignals(False)

    def values(self) -> dict:
        return {"label": self.label_edit.text(),
                "shape": self.shape_combo.currentData(),
                "size": self.size_spin.value() / self._scale,
                "color": self.color_button.color(),
                "hidden": not self.visible_button.isChecked()}


class MarkerDialog(QDialog):
    """Right-click a marker: its label, shape, size, color and visibility, applied as
    you change them, and a button to delete it."""

    def __init__(self, viewer, index: int):
        super().__init__(viewer)
        self.setWindowTitle("Marker")
        self._viewer = viewer
        self._index = index
        self.fields = MarkerFields()
        self.position_label = QLabel()
        form = QFormLayout()
        form.addRow("Label", self.fields.label_edit)
        form.addRow("Shape", self.fields.shape_combo)
        form.addRow("Size", self.fields.size_spin)
        form.addRow("Color", self.fields.color_button)
        form.addRow("Visible", self.fields.visible_button)
        form.addRow("Position", self.position_label)
        delete_button = QPushButton("Delete marker")
        delete_button.clicked.connect(self._delete)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        buttons.addButton(delete_button, QDialogButtonBox.ButtonRole.DestructiveRole)
        form.addRow(buttons)
        self.setLayout(form)
        self._refresh()
        self.fields.changed.connect(self._apply)

    def _refresh(self):
        markers = self._viewer.get_markers()
        if self._index >= len(markers):
            self.accept()
            return
        unit, scale = self._viewer._length_unit()
        marker = markers[self._index]
        self.fields.set_marker(marker, unit, scale)
        self.position_label.setText(f"{marker['x'] * scale:.0f}, {marker['y'] * scale:.0f} {unit}")

    def _apply(self):
        self._viewer.update_marker(self._index, **self.fields.values())

    def _delete(self):
        self._viewer.delete_marker(self._index)
        self.accept()


class _MarkerRow(QWidget):
    hovered = pyqtSignal(object)    # this row's index, or None on leaving it

    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

    def enterEvent(self, event):
        self.hovered.emit(self.index)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hovered.emit(None)
        super().leaveEvent(event)


class MarkerPanel(QWidget):
    """All the markers on the image, one row each. Hovering a row lights that marker
    up on the image."""

    def __init__(self, viewer):
        super().__init__(viewer, Qt.WindowType.Tool)
        self.setWindowTitle("Markers")
        self._viewer = viewer
        self._rows = []         # (row widget, MarkerFields, position label)

        self._grid_host = QWidget()
        self._grid = QVBoxLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(2)
        self._grid_host.setLayout(self._grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._grid_host)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        self._empty_label = QLabel("No markers on this camera. Right-click the image, press M with "
                                   "the cursor over it, or use Add marker.")
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet(f"color: {theme.color('muted')};")

        add_button = QPushButton("Add marker")
        add_button.setToolTip("A new marker in the middle of what the image is showing")
        add_button.clicked.connect(lambda: self._viewer.add_marker())
        clear_button = QPushButton("Remove all")
        clear_button.clicked.connect(self._viewer.clear_markers)
        buttons = QHBoxLayout()
        buttons.addWidget(add_button)
        buttons.addWidget(clear_button)
        buttons.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(self._empty_label)
        layout.addWidget(scroll, 1)
        layout.addLayout(buttons)
        self.setLayout(layout)
        self.resize(520, 260)

        viewer.markers_updated.connect(self.refresh)
        self.refresh()

    def refresh(self):
        """Match the rows to the viewer's markers. Rows are only rebuilt when the
        number of markers changes, so typing a label is not interrupted."""
        markers = self._viewer.get_markers()
        unit, scale = self._viewer._length_unit()
        self._empty_label.setVisible(not markers)
        if len(markers) != len(self._rows):
            self._build_rows(len(markers))
        for (_row, fields, position_label), marker in zip(self._rows, markers):
            fields.set_marker(marker, unit, scale)
            position_label.setText(f"{marker['x'] * scale:.0f}, {marker['y'] * scale:.0f} {unit}")

    def _build_rows(self, n: int):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._rows = []
        self._viewer.highlight_marker(None)
        for index in range(n):
            row = _MarkerRow(index)
            fields = MarkerFields()
            position_label = QLabel()
            position_label.setMinimumWidth(96)
            position_label.setStyleSheet(f"color: {theme.color('muted')};")
            delete_button = QPushButton("✕")
            delete_button.setFixedWidth(26)
            delete_button.setToolTip("Delete this marker")
            delete_button.clicked.connect(lambda _=False, i=index: self._viewer.delete_marker(i))
            fields.changed.connect(lambda i=index, f=fields: self._viewer.update_marker(i, **f.values()))
            row.hovered.connect(self._viewer.highlight_marker)
            line = QGridLayout()
            line.setContentsMargins(2, 1, 2, 1)
            for column, widget in enumerate((fields.visible_button, fields.color_button,
                                             fields.label_edit, fields.shape_combo,
                                             fields.size_spin, position_label, delete_button)):
                line.addWidget(widget, 0, column)
            line.setColumnStretch(2, 1)
            row.setLayout(line)
            self._grid.addWidget(row)
            self._rows.append((row, fields, position_label))
        self._grid.addStretch()

    def closeEvent(self, event):
        self._viewer.highlight_marker(None)
        super().closeEvent(event)
