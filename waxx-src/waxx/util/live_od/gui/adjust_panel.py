"""
Live-adjust panel for the liveOD GUI.

AdjustPanel hosts one AdjustParamRow per adjustable parameter registered with
``self.adjust()`` in an experiment's ``prepare()``.  Each row shows a checkbox,
the param name, a ⚙ button for min/max/step, a spinbox, and a unit dropdown;
the row's context menu (right click) also edits min/max/step and copies the
param's assignment line.

Values are SI everywhere -- in ExptParams, over the wire, and in every signal
this module emits.  The unit is display only: a ``t_tof`` of 2e-05 s is shown
as ``20 µs``, and switching the dropdown to ms only changes what is on screen.
Each param's chosen unit is remembered between runs and between sessions.

Only checked rows are included in "Copy params"; the check-all/uncheck-all
button toggles them in bulk, and clicking a param's name copies just its
assignment prefix ("self.p.key = ").  Rows whose value differs from the value
the experiment started with are highlighted, and ↺ reverts them.
"""

import re

from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from waxa.units import format_si, mult_for, unit_for_param, unit_options

_SCI_RE = re.compile(r'^-?[0-9]*\.?[0-9]*([eE][+-]?[0-9]*)?$')

# Widths that keep every row's columns lined up with each other.
_KEY_WIDTH   = 190
_SPIN_WIDTH  = 104
_UNIT_WIDTH  = 56
_BUTTON_SIZE = 22
_SPEC_BUTTON_SIZE = 16

_SETTINGS_ORG, _SETTINGS_APP = "waxx", "liveod"
_CHANGED_STYLE = "color: #b35c00; font-weight: bold;"


def _settings():
    """QSettings for the panel's remembered units and geometry.

    Looked up through the module global so a test can monkeypatch QSettings and
    keep the real registry out of it.
    """
    return QSettings(_SETTINGS_ORG, _SETTINGS_APP)


def _round_si(value):
    """Strips the float noise a unit round-trip leaves (20/1e6 -> 2e-05)."""
    return float(f"{float(value):.12g}")


class _NoScrollMixin:
    """Ignores the wheel unless focused.

    Without this, scrolling the panel's list changes whatever spinbox happens to
    be under the pointer -- which moves a live experiment parameter.
    """

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class ScientificDoubleSpinBox(_NoScrollMixin, QDoubleSpinBox):
    """QDoubleSpinBox that shows values at 12 significant figures.

    12 figures is enough that a displayed value reads back as the same number,
    so committing an edit (or just leaving the field) cannot quietly round the
    value; ``%g`` keeps it short for the values actually seen, and falls back to
    exponent form for a param with no unit to scale it.
    """

    def textFromValue(self, value: float) -> str:
        return f"{value:.12g}"

    def _strip_affixes(self, text: str) -> str:
        """The number alone -- Qt hands these methods the text with the unit suffix."""
        text = text.strip()
        prefix, suffix = self.prefix().strip(), self.suffix().strip()
        if prefix and text.startswith(prefix):
            text = text[len(prefix):]
        if suffix and text.endswith(suffix):
            text = text[:-len(suffix)]
        return text.strip()

    def valueFromText(self, text: str) -> float:
        try:
            return float(self._strip_affixes(text))
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        stripped = self._strip_affixes(text)
        try:
            float(stripped)
            return QValidator.State.Acceptable, text, pos
        except ValueError:
            if stripped in ('', '-') or _SCI_RE.match(stripped):
                return QValidator.State.Intermediate, text, pos
            return QValidator.State.Invalid, text, pos


class _NoScrollSpinBox(_NoScrollMixin, QSpinBox):
    """Integer spinbox with the same wheel guard."""


class _NoScrollComboBox(_NoScrollMixin, QComboBox):
    """Unit dropdown with the same wheel guard."""


class AdjustSpecDialog(QDialog):
    """Modal dialog for editing the min, max, and step of one adjust spec.

    Fields are shown in the row's display unit; :meth:`get_values` gives SI back.
    """

    def __init__(self, spec: dict, parent=None, unit=None):
        super().__init__(parent)
        self.spec = dict(spec)
        self._is_int = spec.get('dtype') == 'int'
        self._unit = '' if self._is_int else (spec.get('unit', '') if unit is None else unit)
        self._mult = mult_for(self._unit)
        self.setWindowTitle(f"Adjust spec: {spec['key']}")
        self.setModal(True)

        layout = QFormLayout(self)

        initial_step = float(spec.get('step', 1.0)) * self._mult
        step_sb_step = max(1, int(initial_step // 20)) if self._is_int else initial_step / 20

        def make_spin(value, min_v=None, max_v=None, single_step=None):
            if self._is_int:
                sb = _NoScrollSpinBox()
                sb.setRange(int(min_v) if min_v is not None else -2_000_000_000,
                            int(max_v) if max_v is not None else  2_000_000_000)
                sb.setValue(int(round(value)))
                if single_step is not None:
                    sb.setSingleStep(max(1, int(single_step)))
            else:
                sb = ScientificDoubleSpinBox()
                sb.setDecimals(12)
                sb.setRange(float(min_v) if min_v is not None else -1e18,
                            float(max_v) if max_v is not None else  1e18)
                sb.setSingleStep(float(single_step) if single_step is not None else initial_step)
                sb.setValue(float(value))
            return sb

        self._min_sb  = make_spin(spec['min_val'] * self._mult)
        self._max_sb  = make_spin(spec['max_val'] * self._mult)
        # step lower bound: 0 for float (functionally > 0 but hard to enforce), 1 for int
        # step spinbox increments at 1/20 of the initial step size
        self._step_sb = make_spin(spec['step'] * self._mult,
                                  min_v=1 if self._is_int else 0,
                                  single_step=step_sb_step)
        suffix = f" ({self._unit})" if self._unit else ""
        layout.addRow(f"Min{suffix}", self._min_sb)
        layout.addRow(f"Max{suffix}", self._max_sb)
        layout.addRow(f"Step{suffix}", self._step_sb)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_values(self) -> tuple:
        """Return (min_val, max_val, step) as SI floats."""
        return (
            _round_si(float(self._min_sb.value()) / self._mult),
            _round_si(float(self._max_sb.value()) / self._mult),
            _round_si(float(self._step_sb.value()) / self._mult),
        )


class ClickableLabel(QLabel):
    """QLabel that emits ``clicked`` on a left-button press."""

    clicked = pyqtSignal()

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class AdjustParamRow(QWidget):
    """One row: checkbox | key label | ⚙ | spinbox | unit | reset.

    The checkbox controls whether this row is included in "Copy params".
    Clicking the key label copies an assignment line prefix for that param.
    The ⚙ button opens the min/max/step dialog.
    The reset button (↺) reverts to the value present when adjust() was called,
    and is enabled only while the value differs from it.  Right-clicking the row
    opens min/max/step editing and the copy actions.

    Values held and emitted here are SI; ``unit`` only scales what is displayed.

    While the spinbox is being typed into, remote value updates (which arrive at
    the start of every shot) are ignored so an in-progress edit is not clobbered;
    the edited value is only published once the edit is finished (Enter or focus
    loss), so the experiment keeps using the last edited value until then.
    """

    value_changed  = pyqtSignal(str, float)               # key, new SI value
    spec_updated   = pyqtSignal(str, float, float, float)  # key, min, max, step (SI)
    name_clicked   = pyqtSignal(str)                      # key
    check_toggled  = pyqtSignal()
    unit_changed   = pyqtSignal(str, str)                 # key, unit
    copy_requested = pyqtSignal(str)                      # key

    def __init__(self, spec: dict, parent=None, unit=None):
        super().__init__(parent)
        self.spec = dict(spec)
        self.key  = spec['key']
        self._is_int = spec.get('dtype') == 'int'
        self._default_val = float(spec['current_val'])
        self._value_si = float(spec['current_val'])
        self._editing = False

        self._unit = '' if self._is_int else self._valid_unit(unit)
        self._mult = mult_for(self._unit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # --- Checkbox ---
        self._checkbox = QCheckBox()
        self._checkbox.setChecked(True)
        self._checkbox.setToolTip("Include in 'Copy params'")
        self._checkbox.toggled.connect(lambda _: self.check_toggled.emit())
        layout.addWidget(self._checkbox)

        self._label = ClickableLabel(self.key)
        self._label.setFixedWidth(_KEY_WIDTH)
        self._label.setToolTip(
            f"{self.key}\nClick to copy this param's assignment prefix to the clipboard"
        )
        self._label.clicked.connect(lambda: self.name_clicked.emit(self.key))
        self._elide_key()
        layout.addWidget(self._label)

        # --- Spec (min/max/step) button ---
        self._spec_btn = QToolButton()
        self._spec_btn.setText("⚙")
        self._spec_btn.setAutoRaise(True)
        self._spec_btn.setFixedSize(_SPEC_BUTTON_SIZE, _BUTTON_SIZE)
        self._spec_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._spec_btn.setToolTip("Edit min / max / step")
        self._spec_btn.clicked.connect(self.edit_spec)
        layout.addWidget(self._spec_btn)

        # --- Spinbox ---
        if self._is_int:
            self._spinbox = _NoScrollSpinBox()
            self._spinbox.setRange(int(spec['min_val']), int(spec['max_val']))
            self._spinbox.setSingleStep(max(1, int(spec['step'])))
            self._spinbox.setValue(int(round(spec['current_val'])))
        else:
            self._spinbox = ScientificDoubleSpinBox()
            self._spinbox.setDecimals(12)
            self._apply_unit_to_spinbox()
        self._spinbox.setFixedWidth(_SPIN_WIDTH)
        self._spinbox.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # only publish typed values once the edit is committed (Enter / focus out)
        self._spinbox.setKeyboardTracking(False)
        self._spinbox.valueChanged.connect(self._on_spinbox_changed)
        self._spinbox.editingFinished.connect(self._on_editing_finished)
        line_edit = self._spinbox.lineEdit()
        if line_edit is not None:
            line_edit.textEdited.connect(self._on_text_edited)
        layout.addWidget(self._spinbox)

        # --- Unit dropdown (a label when there is nothing to switch to) ---
        options = [] if self._is_int else unit_options(self._unit)
        if len(options) > 1:
            self._unit_box = _NoScrollComboBox()
            for label, _mult in options:
                self._unit_box.addItem(label)
            self._unit_box.setCurrentText(self._unit)
            self._unit_box.setFixedWidth(_UNIT_WIDTH)
            self._unit_box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self._unit_box.setToolTip("Display unit — the stored value stays SI")
            self._unit_box.currentTextChanged.connect(self._on_unit_selected)
            layout.addWidget(self._unit_box)
        else:
            self._unit_box = None
            unit_label = QLabel(self._unit)
            unit_label.setFixedWidth(_UNIT_WIDTH)
            unit_label.setStyleSheet("color: #666;")
            layout.addWidget(unit_label)

        # --- Reset button ---
        self._reset_btn = QToolButton()
        self._reset_btn.setText("↺")
        self._reset_btn.setAutoRaise(True)
        self._reset_btn.setFixedSize(_BUTTON_SIZE, _BUTTON_SIZE)
        self._reset_btn.setToolTip(f"Revert to default ({self._format_si(self._default_val)})")
        self._reset_btn.clicked.connect(self.reset)
        layout.addWidget(self._reset_btn)
        layout.addStretch(1)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self._refresh_changed()

    # ------------------------------------------------------------------
    # Units
    # ------------------------------------------------------------------

    def _spec_unit(self):
        """The unit the experiment sent, or a guess for a spec from before units.

        An older liveOD server (or an older experiment) sends specs with no
        'unit' key at all; guessing from the key and the value keeps those rows
        readable instead of dropping them back to raw SI.
        """
        if 'unit' in self.spec:
            return self.spec['unit'] or ''
        try:
            return unit_for_param(self.key, [self.spec['current_val'],
                                             self.spec['min_val'],
                                             self.spec['max_val']])
        except Exception:
            return ''

    def _valid_unit(self, unit):
        """A remembered unit is only used when it still fits the param's family."""
        spec_unit = self._spec_unit()
        if unit is None:
            return spec_unit
        if any(label == unit for label, _ in unit_options(spec_unit)):
            return unit
        return spec_unit

    def _apply_unit_to_spinbox(self):
        """Push SI range/step/value into the spinbox, scaled to the display unit."""
        mult = self._mult
        self._spinbox.blockSignals(True)
        self._spinbox.setRange(float(self.spec['min_val']) * mult,
                               float(self.spec['max_val']) * mult)
        self._spinbox.setSingleStep(float(self.spec['step']) * mult)
        self._spinbox.setValue(self._value_si * mult)
        self._spinbox.setSuffix(f" {self._unit}" if self._unit else "")
        self._spinbox.setToolTip(
            f"{self.key}\n"
            f"range {self._format_si(self.spec['min_val'])} … {self._format_si(self.spec['max_val'])}\n"
            f"default {self._format_si(self._default_val)}"
        )
        self._spinbox.blockSignals(False)

    def _on_unit_selected(self, unit: str):
        """Dropdown changed: rescale the display. The value itself is untouched."""
        if self._is_int or unit == self._unit:
            return
        self._unit = unit
        self._mult = mult_for(unit)
        self._apply_unit_to_spinbox()
        self._reset_btn.setToolTip(f"Revert to default ({self._format_si(self._default_val)})")
        self._refresh_changed()
        self.unit_changed.emit(self.key, unit)

    def _format_si(self, value):
        return format_si(value, self._unit, 'int' if self._is_int else float)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_spinbox_changed(self, v):
        self._editing = False
        self._value_si = int(v) if self._is_int else _round_si(float(v) / self._mult)
        self._refresh_changed()
        self.value_changed.emit(self.key, float(self._value_si))

    def _on_text_edited(self, _text):
        """User is typing into the spinbox -- hold off remote updates."""
        self._editing = True

    def _on_editing_finished(self):
        """Enter pressed or focus lost -- the edit is committed."""
        self._editing = False

    def reset(self):
        """Revert spinbox to the default value."""
        self._editing = False
        if self._is_int:
            self._spinbox.setValue(int(round(self._default_val)))
        else:
            self._spinbox.setValue(self._default_val * self._mult)
        # _on_spinbox_changed fires and emits value_changed

    def _refresh_changed(self):
        """Highlight the name while the value differs from the default."""
        changed = self.is_changed()
        self._label.setStyleSheet(_CHANGED_STYLE if changed else "")
        self._reset_btn.setEnabled(changed)

    def _elide_key(self):
        metrics = self._label.fontMetrics()
        self._label.setText(metrics.elidedText(self.key, Qt.TextElideMode.ElideMiddle,
                                               _KEY_WIDTH))

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        edit_action  = menu.addAction("Edit min / max / step…")
        reset_action = menu.addAction("Reset to default")
        reset_action.setEnabled(self.is_changed())
        menu.addSeparator()
        prefix_action = menu.addAction("Copy assignment prefix")
        line_action   = menu.addAction("Copy assignment line")
        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen is edit_action:
            self.edit_spec()
        elif chosen is reset_action:
            self.reset()
        elif chosen is prefix_action:
            self.name_clicked.emit(self.key)
        elif chosen is line_action:
            self.copy_requested.emit(self.key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_editing(self) -> bool:
        """True while the user is typing a new value into the spinbox."""
        return self._editing

    def is_checked(self) -> bool:
        return self._checkbox.isChecked()

    def set_checked(self, checked: bool):
        self._checkbox.setChecked(bool(checked))

    def is_changed(self) -> bool:
        """True when the value differs from the one the experiment started with."""
        return abs(self._value_si - self._default_val) > 1e-12 * max(1.0, abs(self._default_val))

    def value(self):
        """The SI value, whatever unit the row is displaying."""
        return self._value_si

    @property
    def unit(self) -> str:
        return self._unit

    def set_unit(self, unit: str):
        """Switch the display unit (no value change, no value_changed)."""
        if self._unit_box is not None:
            self._unit_box.setCurrentText(unit)      # fires _on_unit_selected
        else:
            self._on_unit_selected(unit)

    def formatted_value(self) -> str:
        """The SI value written the way it would be in source ('20.e-6')."""
        return self._format_si(self._value_si)

    def update_value(self, value: float):
        """Update spinbox without triggering value_changed (remote sync).

        Ignored while the user is mid-edit so a shot starting does not reset the
        spinbox out from under them. The value given is SI.
        """
        if self._editing:
            return
        self._value_si = int(round(value)) if self._is_int else _round_si(value)
        self._spinbox.blockSignals(True)
        if self._is_int:
            self._spinbox.setValue(int(round(value)))
        else:
            self._spinbox.setValue(float(value) * self._mult)
        self._spinbox.blockSignals(False)
        self._refresh_changed()

    def apply_spec(self, min_val: float, max_val: float, step: float):
        """Update SI range/step on the spinbox (called after the spec dialog)."""
        self.spec.update({'min_val': min_val, 'max_val': max_val, 'step': step})
        if self._is_int:
            self._spinbox.blockSignals(True)
            self._spinbox.setRange(int(min_val), int(max_val))
            self._spinbox.setSingleStep(max(1, int(step)))
            self._spinbox.blockSignals(False)
        else:
            self._apply_unit_to_spinbox()
        self._refresh_changed()

    def edit_spec(self):
        """Open the min/max/step dialog and publish the result."""
        dlg = AdjustSpecDialog(self.spec, self, unit=self._unit)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            min_v, max_v, step = dlg.get_values()
            self.apply_spec(min_v, max_v, step)
            self.spec_updated.emit(self.key, min_v, max_v, step)


class AdjustPanel(QWidget):
    """Panel with one AdjustParamRow per adjustable parameter.

    value_changed_signal is forwarded from individual rows (SI values).
    spec_updated_signal is forwarded from spec-dialog acceptances (SI values).
    """

    value_changed_signal = pyqtSignal(str, float)          # key, SI value
    spec_updated_signal  = pyqtSignal(str, float, float, float)  # key, min, max, step (SI)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: dict[str, AdjustParamRow] = {}
        self._checked_state: dict[str, bool] = {}
        self._unit_choice: dict[str, str] = {}

        # --- Toolbar ---
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(4)

        self._filter_box = QLineEdit()
        self._filter_box.setPlaceholderText("filter…")
        self._filter_box.setClearButtonEnabled(True)
        self._filter_box.textChanged.connect(self._apply_filter)
        bar.addWidget(self._filter_box, 1)

        self._expt_params_btn = QPushButton(".p")
        self._expt_params_btn.setCheckable(True)
        self._expt_params_btn.setChecked(True)
        self._expt_params_btn.setFixedWidth(30)
        self._expt_params_btn.setToolTip(
            "When checked, copies 'self.p.key = value'; "
            "when unchecked, copies 'self.key = value'"
        )
        bar.addWidget(self._expt_params_btn)

        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setToolTip(
            "Copy the checked params' current values as assignment lines"
        )
        self._copy_btn.clicked.connect(self._copy_params_to_clipboard)
        bar.addWidget(self._copy_btn)

        self._check_all_btn = QPushButton("Uncheck all")
        self._check_all_btn.setToolTip(
            "Uncheck (or check) every param's 'Copy params' checkbox"
        )
        self._check_all_btn.clicked.connect(self._on_check_all_clicked)
        bar.addWidget(self._check_all_btn)

        self._reset_all_btn = QToolButton()
        self._reset_all_btn.setText("↺")
        self._reset_all_btn.setAutoRaise(True)
        self._reset_all_btn.setFixedSize(_BUTTON_SIZE, _BUTTON_SIZE)
        self._reset_all_btn.setToolTip("Revert every changed param to its default")
        self._reset_all_btn.clicked.connect(self.reset_all_changed)
        bar.addWidget(self._reset_all_btn)

        # Scroll area so many params don't overflow the window
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        self._row_layout = QVBoxLayout(container)
        self._row_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.setSpacing(2)
        scroll.setWidget(container)

        self._empty_label = QLabel("No adjustable params in this run.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #888;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)
        outer.addLayout(bar)
        outer.addWidget(self._empty_label)
        outer.addWidget(scroll)
        self._scroll = scroll
        self.setMinimumWidth(320)
        self.setWindowTitle("Adjust Parameters")
        self._restore_geometry()
        self._update_empty_state()

    # ------------------------------------------------------------------
    # Populating
    # ------------------------------------------------------------------

    def populate(self, specs: list):
        """Rebuild rows from a list of spec dicts (called after INIT_RUN).

        Checkbox states and unit choices carry over for params that are still
        there, so a new run does not undo how the panel was set up.
        """
        # Disconnect old rows first
        for key, row in self._rows.items():
            self._checked_state[key] = row.is_checked()
            try:
                row.value_changed.disconnect()
                row.spec_updated.disconnect()
                row.name_clicked.disconnect()
                row.check_toggled.disconnect()
                row.unit_changed.disconnect()
                row.copy_requested.disconnect()
            except Exception:
                pass
            self._row_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

        for spec in specs:
            key = spec['key']
            row = AdjustParamRow(spec, self, unit=self._remembered_unit(key))
            row.value_changed.connect(self.value_changed_signal)
            row.spec_updated.connect(self.spec_updated_signal)
            row.name_clicked.connect(self._copy_key_prefix_to_clipboard)
            row.check_toggled.connect(self._update_check_all_button)
            row.unit_changed.connect(self._remember_unit)
            row.copy_requested.connect(self._copy_line_to_clipboard)
            row.set_checked(self._checked_state.get(key, True))
            self._rows[key] = row
            self._row_layout.addWidget(row)
        self._update_check_all_button()
        self._apply_filter(self._filter_box.text())
        self._update_empty_state()

    def update_values(self, values: dict):
        """Update displayed values from a broadcast without emitting signals."""
        for key, val in values.items():
            row = self._rows.get(key)
            if row is not None:
                row.update_value(float(val))

    def apply_spec_update(self, key: str, min_val: float, max_val: float, step: float):
        """Apply an externally-sourced spec change to a row (e.g. from remote viewer)."""
        row = self._rows.get(key)
        if row is not None:
            row.apply_spec(min_val, max_val, step)

    def reset_all_changed(self):
        """Revert every row whose value differs from its default."""
        for row in self._rows.values():
            if row.is_changed():
                row.reset()

    def param_count(self) -> int:
        return len(self._rows)

    # ------------------------------------------------------------------
    # Units and geometry, remembered between runs and sessions
    # ------------------------------------------------------------------

    def _remembered_unit(self, key: str):
        if key in self._unit_choice:
            return self._unit_choice[key]
        try:
            stored = _settings().value(f"adjust/units/{key}")
        except Exception:
            return None
        return str(stored) if stored else None

    def _remember_unit(self, key: str, unit: str):
        self._unit_choice[key] = unit
        try:
            _settings().setValue(f"adjust/units/{key}", unit)
        except Exception:
            pass

    def _restore_geometry(self):
        try:
            geometry = _settings().value("adjust/geometry")
        except Exception:
            return
        if geometry:
            try:
                self.restoreGeometry(geometry)
            except Exception:
                pass

    def closeEvent(self, event):
        try:
            _settings().setValue("adjust/geometry", self.saveGeometry())
        except Exception:
            pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Toolbar behaviour
    # ------------------------------------------------------------------

    def _apply_filter(self, text: str):
        needle = (text or "").strip().lower()
        for key, row in self._rows.items():
            row.setVisible(needle in key.lower())

    def _update_empty_state(self):
        empty = not self._rows
        self._empty_label.setVisible(empty)
        self._scroll.setVisible(not empty)
        for widget in (self._copy_btn, self._check_all_btn,
                       self._reset_all_btn, self._filter_box):
            widget.setEnabled(not empty)

    def _assignment_prefix(self, key: str) -> str:
        """'self.p.key = ' or 'self.key = ', per the .p toggle."""
        if self._expt_params_btn.isChecked():
            return f"self.p.{key} = "
        return f"self.{key} = "

    def _copy_key_prefix_to_clipboard(self, key: str):
        """Copy just the assignment prefix for one param (clicked key label)."""
        QApplication.clipboard().setText(self._assignment_prefix(key))

    def _copy_line_to_clipboard(self, key: str):
        """Copy one param's full assignment line (row context menu)."""
        row = self._rows.get(key)
        if row is not None:
            QApplication.clipboard().setText(
                f"{self._assignment_prefix(key)}{row.formatted_value()}")

    def _on_check_all_clicked(self):
        """Uncheck every row, or check them all if none are checked."""
        target = not any(row.is_checked() for row in self._rows.values())
        for row in self._rows.values():
            row.set_checked(target)
        self._update_check_all_button()

    def _update_check_all_button(self):
        """Label the button by what it will do next."""
        any_checked = any(row.is_checked() for row in self._rows.values())
        self._check_all_btn.setText("Uncheck all" if any_checked else "Check all")

    def _copy_params_to_clipboard(self):
        """Copy checked adjust values as assignment lines to the clipboard.

        Values are SI, written with the exponent of the row's unit, so a t_tof
        shown as 20 µs copies as '20.e-6' rather than '2e-05'.
        """
        indent = "        "  # two leading indents (8 spaces)
        lines = []
        for key, row in self._rows.items():
            if not row.is_checked():
                continue
            lines.append(f"{self._assignment_prefix(key)}{row.formatted_value()}")
        if lines:
            text = lines[0] + ("\n" + "\n".join(indent + l for l in lines[1:]) if len(lines) > 1 else "")
        else:
            text = ""
        QApplication.clipboard().setText(text)
