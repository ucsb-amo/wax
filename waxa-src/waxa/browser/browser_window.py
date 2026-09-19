import functools
import logging
import os
import re
import sys
import time
from datetime import date, timedelta

from PyQt6.QtCore import QDate, QItemSelectionModel, QPointF, QRectF, QSettings, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QIcon,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
    QTransform,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyle,
)

from .cache import MetadataCache
from .run_summary import RunSummary
from .scanner import (
    PARAM_SEARCH_MODES,
    AnnotationWriteWorker,
    FolderIndexWorker,
    LiteCreateWorker,
    ParamSearchLoader,
    RunScanner,
    ScanWorker,
    XvarDetailLoader,
    find_nearest_run_date_and_id,
    find_newer_completed_run_id,
)

# Days shown on either side of a run found via the run-ID jump.
RUN_ID_JUMP_CONTEXT_DAYS = 3
from ..data.server_talk import server_talk


LOGGER = logging.getLogger(__name__)

# Debounce intervals (ms).  Typing in a filter box or arrow-keying through the
# run list fires one event per keystroke; coalescing them keeps the table and
# the param-search loader from doing redundant work on every intermediate state.
FILTER_DEBOUNCE_MS = 120
PARAM_SYNC_DEBOUNCE_MS = 150

# Shared brushes for row styling; constructing a QColor per cell per row adds up
# when the table is rebuilt with thousands of runs.
_COLOR_LITE_ROW = QColor(220, 245, 223)
_COLOR_PLAIN_ROW = QColor(255, 255, 255)
_COLOR_MUTED_TEXT = QColor("#607380")
_COLOR_EXPERIMENT_TEXT = QColor("#17384b")
_COLOR_SCOPE_BG = QColor("#e1eef4")
_COLOR_SCOPE_FG = QColor("#24536b")
_COLOR_LITE_BG = QColor("#def2e4")
_COLOR_LITE_FG = QColor("#29603a")


def _configure_browser_logging():
    """Ensure browser diagnostics reach the launching terminal."""
    browser_logger = logging.getLogger("waxa.browser")
    browser_logger.setLevel(logging.DEBUG)

    for handler in browser_logger.handlers:
        if getattr(handler, "_waxa_browser_terminal_handler", False):
            return

    handler = logging.StreamHandler()
    handler._waxa_browser_terminal_handler = True
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(threadName)s | %(message)s",
        "%H:%M:%S",
    ))
    browser_logger.addHandler(handler)
    browser_logger.propagate = False
    LOGGER.debug("Browser logging configured for terminal output")


def make_cog_icon(color: str = "#ffffff", teeth: int = 8) -> QIcon:
    """Build a cog/gear icon; Qt ships no standard settings icon."""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.translate(size / 2.0, size / 2.0)

    body = QPainterPath()
    body.addEllipse(QPointF(0.0, 0.0), size * 0.30, size * 0.30)

    tooth_width = size * 0.15
    tooth_height = size * 0.42
    for index in range(teeth):
        tooth = QPainterPath()
        tooth.addRoundedRect(QRectF(-tooth_width / 2.0, -tooth_height, tooth_width, tooth_height), 3.0, 3.0)
        body.addPath(QTransform().rotate(index * (360.0 / teeth)).map(tooth))

    hub = QPainterPath()
    hub.addEllipse(QPointF(0.0, 0.0), size * 0.12, size * 0.12)

    painter.fillPath(body.simplified().subtracted(hub), QColor(color))
    painter.end()
    return QIcon(pixmap)


def parse_name_search_terms(query: str):
    normalized_query = (query or "").strip().lower()
    return [term.strip() for term in normalized_query.split("+") if term.strip()]


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")


@functools.lru_cache(maxsize=65536)
def normalize_match_text(value: str):
    # Memoised: the same xvar/experiment names recur across thousands of runs,
    # and this runs once per name per filter term on every filter pass.
    return _NON_ALNUM_RE.sub("", (value or "").lower())


def is_subsequence(needle: str, haystack: str):
    if not needle:
        return True
    index = 0
    for char in haystack:
        if char == needle[index]:
            index += 1
            if index == len(needle):
                return True
    return False


def name_matches_term(term: str, raw_name: str):
    raw_name = (raw_name or "").lower()
    normalized_name = normalize_match_text(raw_name)
    normalized_term = normalize_match_text(term)
    if term in raw_name:
        return True
    if normalized_term and normalized_term in normalized_name:
        return True
    if normalized_term and is_subsequence(normalized_term, normalized_name):
        return True
    return False


def name_matches_all_terms(raw_name: str, terms: list[str]):
    if not terms:
        return True
    return all(name_matches_term(term, raw_name) for term in terms)


def any_name_matches_all_terms(names: list[str], terms: list[str]):
    if not terms:
        return True
    lowered_names = [str(name).lower() for name in names]
    for term in terms:
        if not any(name_matches_term(term, name) for name in lowered_names):
            return False
    return True


class DateSeparatorDelegate(QStyledItemDelegate):
    """Draw a subtle separator line above rows where the run date changes."""

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window

    def paint(self, painter, option, index):
        super().paint(painter, option, index)

        table = self.parent_window.table

        # Draw a very subtle vertical divider at each column boundary.
        if index.column() < table.columnCount() - 1:
            painter.save()
            pen = QPen(QColor("#e7edf2"))
            pen.setWidth(1)
            painter.setPen(pen)
            rect = option.rect
            x = rect.right()
            painter.drawLine(x, rect.top() + 3, x, rect.bottom() - 3)
            painter.restore()

        # Draw separators once per row (first column only) to avoid per-cell
        # overhead on large tables.
        if index.column() != 0:
            return

        if index.row() <= 0:
            return

        current_run = self.parent_window._get_run_for_row(index.row())
        previous_run = self.parent_window._get_run_for_row(index.row() - 1)

        if current_run is None or previous_run is None:
            return

        if current_run.run_date_str == previous_run.run_date_str:
            return

        painter.save()
        pen = QPen(QColor("#bac6cf"))
        pen.setWidth(1)
        painter.setPen(pen)

        rect = option.rect
        left = 2
        right = table.viewport().width() - 2
        y = rect.top() + 1
        painter.drawLine(left, y, right, y)
        painter.restore()


class CopyPathLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ScrollableValueField(QScrollArea):
    """Single-line scrollable value field that doesn't force parent width expansion."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel("-", self)
        self._label.setWordWrap(False)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.setWidget(self._label)
        self.setWidgetResizable(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)

        field_height = self.fontMetrics().height() + 10
        self.setFixedHeight(field_height)

    def setText(self, text: str):
        value = "-" if text is None else str(text)
        self._label.setText(value)
        self._label.adjustSize()
        self._label.setToolTip(value)
        self.setToolTip(value)

    def text(self):
        return self._label.text()

    def setTextInteractionFlags(self, flags):
        self._label.setTextInteractionFlags(flags)


class _RunIdItem(QTableWidgetItem):
    """run_id cell that sorts numerically (text sort breaks at digit rollover)."""

    def __lt__(self, other):
        try:
            return int(self.data(Qt.ItemDataRole.UserRole)) < int(other.data(Qt.ItemDataRole.UserRole))
        except (TypeError, ValueError):
            return super().__lt__(other)


class ParamSearchDialog(QDialog):
    MODE_LABELS = {
        "params": "Params",
        "camera_params": "Camera Params",
        "data": "Data Containers",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._records_by_mode = {mode: [] for mode in PARAM_SEARCH_MODES}
        # Rows in results_table map 1:1 onto _table_records (the active mode's
        # full record list); filtering only hides rows, so typing in the search
        # box never rebuilds QTableWidgetItems.
        self._table_records = []
        self._table_mode = None
        self._filtered_records = []
        self._active_mode = "params"
        self._mode_buttons = {}
        self._selection_locked = False
        self._preferred_name_by_mode = {mode: None for mode in PARAM_SEARCH_MODES}

        self.setWindowTitle("Param Search")
        self.resize(860, 560)
        self.setMinimumSize(700, 420)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        toggle_group = QFrame(self)
        toggle_group.setObjectName("toolbarGroup")
        toggle_layout = QHBoxLayout(toggle_group)
        toggle_layout.setContentsMargins(6, 4, 6, 4)
        toggle_layout.setSpacing(4)

        for mode in PARAM_SEARCH_MODES:
            button = QToolButton(toggle_group)
            button.setText(self.MODE_LABELS.get(mode, mode))
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.clicked.connect(lambda checked=False, m=mode: self._on_mode_selected(m, checked))
            toggle_layout.addWidget(button)
            self._mode_buttons[mode] = button

        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("search params")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._apply_filter)

        self.run_badge = QLabel("run -", self)
        self.run_badge.setObjectName("paramRunBadge")
        self.run_badge.setProperty("state", "active")

        top_row.addWidget(toggle_group)
        top_row.addWidget(self.search_input, 1)
        top_row.addWidget(self.run_badge)

        self.status_label = QLabel("Ready", self)
        self.status_label.setObjectName("statusLabel")

        self.results_table = QTableWidget(self)
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["name", "dtype", "preview"])
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setWordWrap(False)
        self.results_table.setShowGrid(False)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.verticalHeader().setDefaultSectionSize(28)
        self.results_table.itemSelectionChanged.connect(self._update_detail_from_selection)
        header = self.results_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionsMovable(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.resizeSection(1, 120)

        self.detail_value = QPlainTextEdit(self)
        self.detail_value.setReadOnly(True)
        self.detail_value.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.detail_value.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.detail_value.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        if parent is not None and hasattr(parent, "_fixed_width_font"):
            self.detail_value.setFont(parent._fixed_width_font())

        layout.addLayout(top_row)
        layout.addWidget(self.results_table, 1)
        layout.addWidget(self.detail_value, 1)
        layout.addWidget(self.status_label)

        self._mode_buttons[self._active_mode].setChecked(True)
        self._set_mode(self._active_mode)
        self._apply_badge_style()

    def _apply_badge_style(self):
        state = "inactive" if self._selection_locked else "active"
        self.run_badge.setProperty("state", state)
        self.run_badge.style().unpolish(self.run_badge)
        self.run_badge.style().polish(self.run_badge)

    def _set_controls_enabled(self, enabled: bool):
        self.search_input.setEnabled(enabled)
        self.results_table.setEnabled(enabled)
        self.detail_value.setEnabled(enabled)
        for button in self._mode_buttons.values():
            button.setEnabled(enabled)

    def set_selection_lock_state(self, locked: bool):
        self._selection_locked = bool(locked)
        if self._selection_locked:
            self.run_badge.setText("multiple selected")
            self._set_controls_enabled(False)
            self.status_label.setText("Select a single run for param search")
            self.detail_value.setPlainText("Param search paused while multiple runs are selected.")
        else:
            self._set_controls_enabled(True)
        self._apply_badge_style()

    def set_run(self, run: RunSummary):
        self.setWindowTitle(f"Run {run.run_id} - Param Search")
        self.run_badge.setText(f"run {run.run_id}")
        self._apply_badge_style()

    def set_loading_state(self, run: RunSummary):
        self._remember_selection_for_active_mode()
        self.set_run(run)
        self._records_by_mode = {mode: [] for mode in PARAM_SEARCH_MODES}
        self.status_label.setText("Loading values…")
        # Keep the previous run's rows on screen until the new ones arrive:
        # clearing here and repopulating a few hundred ms later is what made
        # arrow-keying through runs flicker.  Rows are replaced in one shot by
        # the first append_records() for the active mode.
        self.detail_value.setPlainText("Loading values…")

    def append_records(self, mode: str, records: list):
        """Add records for one mode group; refreshes the table if that mode is active."""
        self._records_by_mode[mode] = list(records)
        n = len(records)
        self.status_label.setText(f"{mode}: {n} entries loaded")
        if mode == self._active_mode:
            self._rebuild_table()

    def set_error_message(self, message: str):
        self.status_label.setText("Load failed")
        self.results_table.setRowCount(0)
        self._table_records = []
        self._table_mode = None
        self._filtered_records = []
        self.detail_value.setPlainText(message)

    def set_stale_run_message(self, run_id: int):
        self.setWindowTitle(f"Run {run_id} - Param Search")
        self.run_badge.setText(f"run {run_id} outside range")
        self.status_label.setText("Showing stale values; run is outside the current load range")
        self._apply_badge_style()

    @staticmethod
    def _same_records(a: list, b: list):
        if len(a) != len(b):
            return False
        return all(x.get("name") == y.get("name") and x.get("preview") == y.get("preview") for x, y in zip(a, b))

    def set_records(self, records_by_mode: dict):
        """Final delivery of all modes.  Modes already delivered via
        append_records() are left alone so the active table is not rebuilt a
        second time with identical content."""
        active_changed = False
        for mode in PARAM_SEARCH_MODES:
            incoming = list(records_by_mode.get(mode, []))
            if self._same_records(self._records_by_mode.get(mode, []), incoming):
                continue
            self._records_by_mode[mode] = incoming
            if mode == self._active_mode:
                active_changed = True
        if active_changed or self._table_mode != self._active_mode:
            self._rebuild_table()
        else:
            self._apply_filter()

    def focus_search(self):
        self.search_input.setFocus()
        self.search_input.selectAll()

    def set_mode(self, mode: str):
        button = self._mode_buttons.get(mode)
        if button is not None:
            button.setChecked(True)
            self._set_mode(mode)

    def _on_mode_selected(self, mode: str, checked: bool):
        if checked:
            self._set_mode(mode)

    def _set_mode(self, mode: str):
        self._remember_selection_for_active_mode()
        self._active_mode = mode if mode in PARAM_SEARCH_MODES else "params"
        self._rebuild_table()

    def _selected_record_name(self):
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self._table_records):
            return None
        return self._table_records[row].get("name")

    def _remember_selection_for_active_mode(self):
        selected_name = self._selected_record_name()
        if selected_name:
            self._preferred_name_by_mode[self._active_mode] = selected_name

    def _rebuild_table(self):
        """Repopulate the table with the active mode's full record list, then
        apply the current text filter by hiding rows."""
        source_records = self._records_by_mode.get(self._active_mode, [])
        self._table_records = source_records
        self._table_mode = self._active_mode

        table = self.results_table
        table.setUpdatesEnabled(False)
        table.blockSignals(True)
        try:
            table.setRowCount(0)
            table.setRowCount(len(source_records))
            for row, record in enumerate(source_records):
                name_item = QTableWidgetItem(str(record.get("name", "-")))
                # Only the name cell carries the record payload; storing the
                # dict on all three cells tripled the QVariant conversions.
                name_item.setData(Qt.ItemDataRole.UserRole, record)
                table.setItem(row, 0, name_item)
                table.setItem(row, 1, QTableWidgetItem(str(record.get("dtype", "-"))))
                table.setItem(row, 2, QTableWidgetItem(str(record.get("preview", "-"))))
        finally:
            table.blockSignals(False)
            table.setUpdatesEnabled(True)

        self._apply_filter()

    def _apply_filter(self, preferred_name: str | None = None):
        if self._table_mode != self._active_mode:
            self._rebuild_table()
            return

        terms = parse_name_search_terms(self.search_input.text())
        source_records = self._table_records
        if preferred_name is None:
            preferred_name = self._preferred_name_by_mode.get(self._active_mode)

        table = self.results_table
        visible_rows = []
        table.setUpdatesEnabled(False)
        try:
            for row, record in enumerate(source_records):
                matches = name_matches_all_terms(record.get("name", ""), terms)
                table.setRowHidden(row, not matches)
                if matches:
                    visible_rows.append(row)
        finally:
            table.setUpdatesEnabled(True)

        self._filtered_records = [source_records[row] for row in visible_rows]

        total = len(source_records)
        count = len(visible_rows)
        self.status_label.setText(f"{count} matching entries" + ("" if count == total else f" of {total}"))

        if visible_rows:
            preferred_row = visible_rows[0]
            if preferred_name:
                for row in visible_rows:
                    if source_records[row].get("name") == preferred_name:
                        preferred_row = row
                        break
            # selectRow() emits itemSelectionChanged, which already refreshes
            # the detail pane; suppress it when the row is unchanged.
            if table.currentRow() == preferred_row and table.selectionModel().isRowSelected(preferred_row):
                self._update_detail_from_selection()
            else:
                table.selectRow(preferred_row)
            selected_name = source_records[preferred_row].get("name")
            if selected_name:
                self._preferred_name_by_mode[self._active_mode] = selected_name
        else:
            table.clearSelection()
            self.detail_value.setPlainText("No matching values.")

    def _update_detail_from_selection(self):
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self._table_records) or self.results_table.isRowHidden(row):
            self.detail_value.setPlainText("No matching values.")
            return

        record = self._table_records[row]
        selected_name = record.get("name")
        if selected_name:
            self._preferred_name_by_mode[self._active_mode] = selected_name
        detail_lines = [
            f"name: {record.get('name', '-')}",
            f"dtype: {record.get('dtype', '-')}",
            "",
            record.get("detail", record.get("preview", "-")),
        ]
        self.detail_value.setPlainText("\n".join(detail_lines))


class RunIdLookupWorker(QThread):
    lookup_ready = pyqtSignal(object, object, object)  # resolved_run_id, run_date, stats dict
    lookup_error = pyqtSignal(str)

    def __init__(self, server: server_talk, data_dir: str, requested_run_id: int, cache=None, parent=None):
        super().__init__(parent)
        self._server = server
        self._data_dir = data_dir
        self._requested_run_id = int(requested_run_id)
        self._cache = cache

    def run(self):
        stats = {}
        t0 = time.perf_counter()
        try:
            # Index-backed binary search over date folders: usually zero or one
            # directory listing and never an HDF5 open.
            resolved_run_id, run_date = find_nearest_run_date_and_id(
                self._data_dir, self._requested_run_id, cache=self._cache, stats=stats
            )
            if resolved_run_id is None:
                resolved_run_id, run_date = self._server.find_nearest_run_date_and_id(self._requested_run_id)
                stats["fallback"] = True
            stats["elapsed_ms"] = 1e3 * (time.perf_counter() - t0)
            self.lookup_ready.emit(resolved_run_id, run_date, stats)
        except Exception as exc:
            self.lookup_error.emit(str(exc))


class NewRunProbeWorker(QThread):
    """Auto-refresh probe: is there a completed run newer than what the table
    already shows?  Only files with a run id above ``current_latest`` are ever
    opened, so an idle tick costs a single directory listing."""

    latest_ready = pyqtSignal(object)
    latest_error = pyqtSignal(str)

    def __init__(self, server: server_talk, data_dir: str, current_latest: int, parent=None):
        super().__init__(parent)
        self._server = server
        self._data_dir = data_dir
        self._current_latest = int(current_latest)

    def run(self):
        try:
            newer = find_newer_completed_run_id(
                self._data_dir,
                self._current_latest,
                is_completed=self._server._is_completed_run,
            )
            self.latest_ready.emit(newer)
        except Exception as exc:
            self.latest_error.emit(str(exc))


class RunDetailPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        summary_group = QGroupBox("Run summary", self)
        summary_form = QFormLayout(summary_group)
        summary_form.setContentsMargins(10, 10, 10, 10)
        summary_form.setVerticalSpacing(4)

        self.run_id_value = QLabel("-", self)
        self.datetime_value = QLabel("-", self)
        self.experiment_value = QLabel("-", self)
        self.experiment_value.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.xvardims_value = QLabel("-", self)
        self.tags_value = QLabel("-", self)
        self.comment_value = ScrollableValueField(self)
        for label in [
            self.run_id_value,
            self.datetime_value,
            self.experiment_value,
            self.xvardims_value,
            self.tags_value,
            self.comment_value,
        ]:
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        summary_form.addRow("Run ID", self.run_id_value)
        summary_form.addRow("Datetime", self.datetime_value)
        summary_form.addRow("Experiment", self.experiment_value)
        summary_form.addRow("Xvar dims", self.xvardims_value)
        summary_form.addRow("Tags", self.tags_value)
        summary_form.addRow("Comment", self.comment_value)

        xvars_group = QGroupBox("Xvars", self)
        xvars_layout = QVBoxLayout(xvars_group)
        xvars_layout.setContentsMargins(10, 10, 10, 10)
        self.xvar_table = QTableWidget(self)
        self.xvar_table.setObjectName("xvarTable")
        self.xvar_table.setColumnCount(6)
        self.xvar_table.setHorizontalHeaderLabels(["xvarname", "min", "max", "unit", "preview", "N"])
        self.xvar_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.xvar_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.xvar_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.xvar_table.setAlternatingRowColors(True)
        self.xvar_table.setShowGrid(False)
        self.xvar_table.verticalHeader().setVisible(False)
        xvar_header = self.xvar_table.horizontalHeader()
        xvar_header.setStretchLastSection(False)
        xvar_header.setSectionsMovable(False)
        xvar_header.setMinimumSectionSize(30)
        xvar_header.setDefaultSectionSize(22)
        xvar_header.setMaximumHeight(22)
        for col in range(6):
            xvar_header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        # Col 0 (xvarname) stretches; fixed defaults for all other cols.
        xvar_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        xvar_header.resizeSection(1, 44)
        xvar_header.resizeSection(2, 44)
        xvar_header.resizeSection(3, 40)
        xvar_header.resizeSection(4, 90)
        xvar_header.resizeSection(5, 55)
        self.xvar_table.setMinimumHeight(80)
        xvars_layout.addWidget(self.xvar_table)

        containers_group = QGroupBox("Data containers", self)
        containers_layout = QVBoxLayout(containers_group)
        containers_layout.setContentsMargins(10, 10, 10, 10)
        self.data_list = QListWidget(self)
        self.data_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.data_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.data_list.setMinimumHeight(140)
        containers_layout.addWidget(self.data_list)

        paths_group = QFrame(self)
        paths_group.setObjectName("pathsRow")
        paths_layout = QHBoxLayout(paths_group)
        paths_layout.setContentsMargins(0, 0, 0, 0)
        paths_layout.setSpacing(6)
        self.experiment_path_value = CopyPathLabel("-", self)
        self.filepath_value = CopyPathLabel("-", self)
        self.experiment_path_value.setWordWrap(False)
        self.filepath_value.setWordWrap(False)
        self.experiment_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.filepath_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.experiment_path_value.setCursor(Qt.CursorShape.PointingHandCursor)
        self.filepath_value.setCursor(Qt.CursorShape.PointingHandCursor)
        self.experiment_path_value.setObjectName("pathValueLabel")
        self.filepath_value.setObjectName("pathValueLabel")
        self.experiment_path_value.setToolTip("-")
        self.filepath_value.setToolTip("-")
        self.experiment_path_value.clicked.connect(
            lambda: self._copy_path_text(self.experiment_path_value.text(), "Experiment")
        )
        self.filepath_value.clicked.connect(lambda: self._copy_path_text(self.filepath_value.text(), "HDF5"))

        self.exp_open_btn = QToolButton(self)
        self.exp_open_btn.setToolTip("Open experiment folder")
        self.exp_open_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.exp_open_btn.clicked.connect(
            lambda: self._open_path_directory(self.experiment_path_value.toolTip())
        )

        self.h5_open_btn = QToolButton(self)
        self.h5_open_btn.setToolTip("Open HDF5 folder")
        self.h5_open_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.h5_open_btn.clicked.connect(lambda: self._open_path_directory(self.filepath_value.toolTip()))

        # expt pill: folder button + path label
        exp_pill = QFrame(self)
        exp_pill.setObjectName("pathPill")
        exp_pill_layout = QHBoxLayout(exp_pill)
        exp_pill_layout.setContentsMargins(4, 2, 6, 2)
        exp_pill_layout.setSpacing(4)
        exp_pill_layout.addWidget(self.exp_open_btn)
        exp_pill_layout.addWidget(self.experiment_path_value, 1)

        # h5 pill: folder button + path label
        h5_pill = QFrame(self)
        h5_pill.setObjectName("pathPill")
        h5_pill_layout = QHBoxLayout(h5_pill)
        h5_pill_layout.setContentsMargins(4, 2, 6, 2)
        h5_pill_layout.setSpacing(4)
        h5_pill_layout.addWidget(self.h5_open_btn)
        h5_pill_layout.addWidget(self.filepath_value, 1)

        paths_layout.addWidget(exp_pill, 1)
        paths_layout.addWidget(h5_pill, 1)

        paths_group.setMaximumHeight(36)

        right_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        right_splitter.addWidget(xvars_group)
        right_splitter.addWidget(containers_group)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 2)
        right_splitter.setSizes([430, 300])

        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.addWidget(summary_group)
        main_splitter.addWidget(right_splitter)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 4)
        main_splitter.setSizes([220, 760])

        layout.addWidget(main_splitter, 1)
        layout.addWidget(paths_group, 0)

    @staticmethod
    def _shorten_path(path: str) -> str:
        """Replace everything up to and including the 'code' folder with '%code%'."""
        if not path or path == "-":
            return path
        # Normalise separators for matching, then find the 'code' segment
        norm = path.replace("\\", "/")
        lower = norm.lower()
        # Match e.g. "c:/users/bananas/code/" or ending exactly at "code"
        import re as _re
        m = _re.search(r'^.+?/code(?=/|$)', lower)
        if not m:
            return path
        remainder = norm[m.end():].lstrip("/")
        return "%code%/" + remainder if remainder else "%code%"

    def _copy_path_text(self, path_text: str, label: str):
        clean = (path_text or "").strip()
        if not clean or clean == "-":
            self.show_message(f"No {label.lower()} path to copy.")
            return
        QApplication.clipboard().setText(clean)
        self.show_message(f"Copied {label.lower()} path to clipboard.")

    def _open_path_directory(self, path_text: str):
        path_text = self._shorten_path(path_text)
        clean = (path_text or "").strip()
        if not clean or clean == "-":
            self.show_message("No path available.")
            return

        # Expand %VAR% placeholders (e.g. %code%) so shortened paths work too
        expanded = os.path.expandvars(clean)
        path = os.path.normpath(expanded)
        if os.path.isdir(path):
            folder = path
        else:
            folder = os.path.dirname(path)

        if not folder or not os.path.isdir(folder):
            self.show_message("Folder does not exist for this path.")
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        self.show_message("Opened folder in system file explorer.")

    def _open_file_in_default_program(self, path_text: str, label: str = "file"):
        clean = (path_text or "").strip()
        if not clean or clean == "-":
            self.show_message(f"No {label.lower()} available.")
            return

        expanded = os.path.expandvars(clean)
        path = os.path.normpath(expanded)
        if not os.path.isfile(path):
            self.show_message(f"{label} does not exist for this path.")
            return

        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        if opened:
            self.show_message(f"Opened {label.lower()} in default program.")
        else:
            self.show_message(f"Could not open {label.lower()} in default program.")

    def show_message(self, text: str):
        window = self.window()
        if hasattr(window, "status_label"):
            window.status_label.setText(text)

    def clear_details(self):
        self.run_id_value.setText("-")
        self.datetime_value.setText("-")
        self.experiment_value.setText("-")
        self.xvardims_value.setText("-")
        self.tags_value.setText("-")
        self.comment_value.setText("-")
        self.experiment_path_value.setText("-")
        self.filepath_value.setText("-")
        self.experiment_path_value.setToolTip("-")
        self.filepath_value.setToolTip("-")
        self.xvar_table.setRowCount(0)
        self.data_list.clear()
        self.show_message("Select a run to load details.")

    @staticmethod
    def _format_xvar_n_value(n_value, n_repeats: int, row_idx: int):
        try:
            n_int = int(n_value)
        except (TypeError, ValueError):
            return str(n_value)

        repeats = max(1, int(n_repeats or 1))
        if row_idx == 0 and repeats > 1 and n_int > 0 and n_int % repeats == 0:
            return f"{repeats} x {n_int // repeats}"
        return str(n_int)

    def set_run(self, run: RunSummary, datetime_text: str, xvardims_text: str):
        self.run_id_value.setText(str(run.run_id))
        self.datetime_value.setText(datetime_text or "-")
        _expt_name = run.experiment_name or "-"
        self.experiment_value.setText(_expt_name)
        self.experiment_value.setToolTip(_expt_name)
        self.xvardims_value.setText(xvardims_text or "()")
        self.tags_value.setText(", ".join(run.tags) if run.tags else "-")
        self.comment_value.setText(run.comment if run.comment else "-")
        self.experiment_path_value.setText(self._shorten_path(run.experiment_filepath or "-"))
        self.filepath_value.setText(self._shorten_path(run.filepath or "-"))
        self.experiment_path_value.setToolTip(run.experiment_filepath or "-")
        self.filepath_value.setToolTip(run.filepath or "-")

        rows = run.xvar_details or [{"name": "(no xvar details)", "min": "", "max": "", "n": ""}]
        self.xvar_table.setRowCount(len(rows))
        for row_idx, item in enumerate(rows):
            values = [
                item["name"],
                item["min"],
                item["max"],
                item.get("unit", ""),
                item.get("preview", "-"),
                self._format_xvar_n_value(item["n"], run.n_repeats, row_idx),
            ]
            for col_idx, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                self.xvar_table.setItem(row_idx, col_idx, table_item)

        self.data_list.clear()
        if run.data_container_keys:
            for key in run.data_container_keys:
                QListWidgetItem(key, self.data_list)
        else:
            QListWidgetItem("(none)", self.data_list)


class FixedLineScrollTable(QTableWidget):
    """Table that scrolls a fixed number of rows per wheel notch.

    Overrides the OS/Qt "lines per scroll" setting so the run list always moves
    a configurable number of rows per notch, regardless of system mouse settings.
    """

    DEFAULT_LINES_PER_WHEEL_NOTCH = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lines_per_wheel_notch = self.DEFAULT_LINES_PER_WHEEL_NOTCH
        self._wheel_remainder = 0.0

    def lines_per_wheel_notch(self):
        return self._lines_per_wheel_notch

    def set_lines_per_wheel_notch(self, lines: int):
        self._lines_per_wheel_notch = max(1, int(lines))
        self._wheel_remainder = 0.0

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0 or event.modifiers() != Qt.KeyboardModifier.NoModifier:
            # Horizontal wheels and modifier combos keep default behaviour.
            super().wheelEvent(event)
            return

        # Accumulate fractions so high-resolution wheels/trackpads still scroll.
        self._wheel_remainder += (delta / 120.0) * self._lines_per_wheel_notch
        steps = int(self._wheel_remainder)
        self._wheel_remainder -= steps

        if steps:
            bar = self.verticalScrollBar()
            bar.setValue(bar.value() - steps * bar.singleStep())
        event.accept()


class DataBrowserWindow(QMainWindow):
    COL_RUN_ID = 0
    COL_DATETIME = 1
    COL_EXPERIMENT = 2
    COL_XVARDIMS = 3
    COL_N_REPEATS = 4
    COL_XVARS = 5
    COL_DATA_KEYS = 6
    COL_SCOPE = 7
    COL_LITE = 8
    COL_TAGS = 9
    COLUMN_LABELS = {
        COL_RUN_ID: "run_id",
        COL_DATETIME: "datetime",
        COL_EXPERIMENT: "experiment",
        COL_XVARDIMS: "xvardims",
        COL_N_REPEATS: "N_r",
        COL_XVARS: "xvarnames",
        COL_DATA_KEYS: "data containers",
        COL_SCOPE: "scope",
        COL_LITE: "lite",
        COL_TAGS: "tags",
    }

    def __init__(self, data_dir: str):
        super().__init__()
        self.settings = QSettings("WeldLab", "WAXADataBrowser")
        # Prefer a user-overridden data dir from settings; fall back to what was passed
        saved_dir = self.settings.value("dataDir", "", type=str)
        self.data_dir = saved_dir if (saved_dir and os.path.isdir(saved_dir)) else data_dir
        self._scan_worker = None
        self._xvar_loader = None
        self._running_xvar_loaders: set = set()
        self._param_search_loader = None
        self._running_param_search_loaders: set = set()
        self._pending_scan_runs = []
        self._pending_scan_request_id = None
        self._scan_restore_state = None
        self._lite_worker = None
        self._run_id_lookup_worker = None
        self._latest_run_check_worker = None
        self._scan_request_id = 0
        self._detail_request_id = 0
        self._param_search_request_id = 0
        self._run_id_lookup_request_id = 0
        self._latest_run_check_request_id = 0

        self._runs_by_id = {}
        self._scan_loaded_count = 0
        self._active_filter_terms = []
        self._field_actions = {}
        self._busy_ops = 0
        self._pending_focus_run_id = None
        self._pending_requested_run_id = None
        self._detail_pane_run_id = None
        self._param_search_dialog = None
        self._param_search_current_run_id = None
        self._param_search_loading_run_id = None
        self._pending_param_sync_run_id = None
        self._annotation_workers: set = set()
        self._scan_after_annotations = False
        self._metadata_cache = None
        self._metadata_cache_dir = None
        self._folder_index_worker = None
        self._folder_index_complete_dir = None
        # (date_from, date_to) to widen to once a run-ID jump has focused its row.
        self._pending_widen_range = None
        self._pending_focus_message = None
        self._server_talk = server_talk(data_dir=self.data_dir)
        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.setSingleShot(False)
        self._auto_refresh_timer.timeout.connect(self._on_auto_refresh_timeout)
        self._day_rollover_timer = QTimer(self)
        self._day_rollover_timer.setSingleShot(False)
        self._day_rollover_timer.timeout.connect(self._check_date_rollover)
        # Coalesce filter keystrokes into one table pass.
        self._filter_debounce_timer = QTimer(self)
        self._filter_debounce_timer.setSingleShot(True)
        self._filter_debounce_timer.setInterval(FILTER_DEBOUNCE_MS)
        self._filter_debounce_timer.timeout.connect(self._apply_filter)
        # Coalesce rapid selection changes before (re)loading the param search.
        self._param_sync_timer = QTimer(self)
        self._param_sync_timer.setSingleShot(True)
        self._param_sync_timer.setInterval(PARAM_SYNC_DEBOUNCE_MS)
        self._param_sync_timer.timeout.connect(self._flush_pending_param_sync)

        self.setWindowTitle("Data Browser")
        self.setWindowIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.resize(980, 760)
        self.setMinimumSize(980, 620)

        self._setup_ui()
        self._start_scan()

    def _setup_ui(self):
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self.date_from = QDateEdit(self)
        self.date_from.setCalendarPopup(True)
        self.date_from.setDate(QDate.currentDate().addDays(-7))

        self.date_to = QDateEdit(self)
        self.date_to.setCalendarPopup(True)
        self.date_to.setDate(QDate.currentDate())
        self._date_to_tracks_today = True
        self.date_to.dateChanged.connect(self._on_date_to_changed)
        self._day_rollover_timer.start(60 * 1000)

        self.refresh_btn = QPushButton("Refresh", self)
        self.refresh_btn.clicked.connect(self._start_scan)

        self.fields_btn = QToolButton(self)
        self.fields_btn.setText("Fields")
        self.fields_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.fields_menu = QMenu(self.fields_btn)
        self.fields_btn.setMenu(self.fields_menu)

        self.status_label = QLabel("Ready", self)
        self.status_label.setObjectName("statusLabel")
        self.status_label.setMinimumHeight(22)
        self.status_label.setMaximumHeight(22)
        self.activity_indicator = QLabel("", self)
        self.activity_indicator.setObjectName("activityIndicator")
        self.activity_indicator.setFixedSize(14, 14)

        date_group = QFrame(self)
        date_group.setObjectName("toolbarGroup")
        date_group_layout = QHBoxLayout(date_group)
        date_group_layout.setContentsMargins(8, 4, 8, 4)
        date_group_layout.setSpacing(6)
        date_group_layout.addWidget(self.refresh_btn)
        date_group_layout.addWidget(QLabel("From", self))
        date_group_layout.addWidget(self.date_from)
        date_group_layout.addWidget(QLabel("To", self))
        date_group_layout.addWidget(self.date_to)
        self.loaded_chip = self._make_stat_chip("Loaded", "0")
        self.visible_chip = self._make_stat_chip("Visible", "0")
        self.selected_chip = self._make_stat_chip("Selected", "none")
        self.visible_chip.hide()
        self.selected_chip.hide()

        self.three_hour_btn = QToolButton(self)
        self.three_hour_btn.setText("3 hours")
        self.three_hour_btn.clicked.connect(lambda: self._apply_recent_hours_preset(3))
        self.today_btn = QToolButton(self)
        self.today_btn.setText("Today")
        self.today_btn.clicked.connect(lambda: self._apply_date_preset(0))
        self.three_day_btn = QToolButton(self)
        self.three_day_btn.setText("3 days")
        self.three_day_btn.clicked.connect(lambda: self._apply_date_preset(2))
        self.week_btn = QToolButton(self)
        self.week_btn.setText("1 week")
        self.week_btn.clicked.connect(lambda: self._apply_date_preset(6))
        self.month_btn = QToolButton(self)
        self.month_btn.setText("1 month")
        self.month_btn.clicked.connect(lambda: self._apply_date_preset(29))
        self.auto_refresh_btn = QToolButton(self)
        self.auto_refresh_btn.setText("Auto")
        self.auto_refresh_btn.setCheckable(True)
        self.auto_refresh_btn.setToolTip("Auto-refresh when new valid runs appear")
        self.auto_refresh_btn.setObjectName("autoRefreshBtn")
        self.auto_refresh_btn.toggled.connect(self._on_auto_refresh_toggled)
        preset_group = QFrame(self)
        preset_group.setObjectName("toolbarGroup")
        preset_group_layout = QHBoxLayout(preset_group)
        preset_group_layout.setContentsMargins(8, 4, 8, 4)
        preset_group_layout.setSpacing(6)
        for button in [self.three_hour_btn, self.today_btn, self.three_day_btn, self.week_btn, self.month_btn]:
            preset_group_layout.addWidget(button)

        toolbar.addWidget(date_group)
        toolbar.addWidget(preset_group)
        toolbar.addStretch(1)
        toolbar.addWidget(self.auto_refresh_btn)
        toolbar.addWidget(self.loaded_chip)
        toolbar.addWidget(self.activity_indicator)

        self.options_btn = QToolButton(self)
        self.options_btn.setToolTip("Options")
        self.options_btn.setIcon(make_cog_icon())
        self.options_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.options_menu = QMenu(self.options_btn)
        self.options_btn.setMenu(self.options_menu)

        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("xvarname search")
        self.search_input.textChanged.connect(self._on_filter_text_changed)
        self.search_input.setClearButtonEnabled(True)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
        self.run_id_jump_input = QLineEdit(self)
        self.run_id_jump_input.setPlaceholderText("by run ID")
        self.run_id_jump_input.setFixedWidth(110)
        self.run_id_jump_input.returnPressed.connect(self._jump_to_run_id)

        self.run_id_jump_btn = QPushButton("Go", self)
        self.run_id_jump_btn.setFixedWidth(44)
        self.run_id_jump_btn.clicked.connect(self._jump_to_run_id)

        self.experiment_filter_input = QLineEdit(self)
        self.experiment_filter_input.setPlaceholderText("experiment search")
        self.experiment_filter_input.textChanged.connect(self._on_filter_text_changed)

        self.tag_filter_input = QLineEdit(self)
        self.tag_filter_input.setPlaceholderText("tag search")
        self.tag_filter_input.setClearButtonEnabled(True)
        self.tag_filter_input.textChanged.connect(self._on_filter_text_changed)

        filter_row.addWidget(self.run_id_jump_input)
        filter_row.addWidget(self.run_id_jump_btn)
        filter_row.addWidget(self.experiment_filter_input, 2)
        filter_row.addWidget(self.search_input, 2)
        filter_row.addWidget(self.tag_filter_input, 1)
        filter_row.addWidget(self.fields_btn)
        filter_row.addWidget(self.options_btn)

        self.table = FixedLineScrollTable(self)
        self.table.set_lines_per_wheel_notch(self._wheel_scroll_lines())
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(
            [
                "run_id",
                "datetime",
                "experiment",
                "xvardims",
                "N_r",
                "xvarnames",
                "data containers",
                "scope",
                "lite",
                "tags",
            ]
        )
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setShowGrid(False)
        self.table.setCornerButtonEnabled(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemDoubleClicked.connect(self._on_row_double_clicked)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)

        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionsMovable(False)
        for col in range(self.table.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(self.COL_RUN_ID, 78)
        header.resizeSection(self.COL_DATETIME, 185)
        header.resizeSection(self.COL_EXPERIMENT, 190)
        header.resizeSection(self.COL_XVARDIMS, 90)
        header.resizeSection(self.COL_N_REPEATS, 82)
        header.resizeSection(self.COL_XVARS, 230)
        header.resizeSection(self.COL_DATA_KEYS, 250)
        header.resizeSection(self.COL_SCOPE, 70)
        header.resizeSection(self.COL_LITE, 58)
        header.resizeSection(self.COL_TAGS, 180)
        header.sortIndicatorChanged.connect(self._refresh_date_separators)

        self._date_separator_delegate = DateSeparatorDelegate(self)
        self.table.setItemDelegate(self._date_separator_delegate)
        self._initialize_field_selector()
        self._load_field_visibility_settings()
        self._initialize_options_menu()

        self.detail_pane = RunDetailPane(self)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(self.table)
        splitter.addWidget(self.detail_pane)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 150])

        self._splitter = splitter

        self._apply_styles()

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(0)
        status_row.addWidget(self.status_label, 1)
        self.lite_cancel_btn = QPushButton("Cancel lite", self)
        self.lite_cancel_btn.setToolTip("Stop lite creation; the run in progress is discarded")
        self.lite_cancel_btn.clicked.connect(self._cancel_lite_creation)
        self.lite_cancel_btn.setVisible(False)
        status_row.addWidget(self.lite_cancel_btn)

        layout.addLayout(toolbar)
        layout.addLayout(filter_row)
        layout.addWidget(splitter)
        layout.addLayout(status_row)

        # Keep filter-row navigation deterministic left-to-right.
        QWidget.setTabOrder(self.run_id_jump_input, self.run_id_jump_btn)
        QWidget.setTabOrder(self.run_id_jump_btn, self.search_input)
        QWidget.setTabOrder(self.search_input, self.experiment_filter_input)
        QWidget.setTabOrder(self.experiment_filter_input, self.tag_filter_input)

        self.setCentralWidget(root)
        self._install_shortcuts()
        self._set_activity_idle()
        auto_enabled = bool(self.settings.value("autoRefreshEnabled", False, type=bool))
        self.auto_refresh_btn.setChecked(auto_enabled)
        self._sync_auto_refresh_timer(show_status=False)

    def _initialize_field_selector(self):
        self.fields_menu.clear()
        self._field_actions = {}

        for col in range(self.table.columnCount()):
            label = self.COLUMN_LABELS.get(col, self.table.horizontalHeaderItem(col).text())
            action = QAction(label, self.fields_menu)
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(lambda checked, c=col: self._on_field_toggled(c, checked))
            self.fields_menu.addAction(action)
            self._field_actions[col] = action

    def _on_field_toggled(self, column: int, checked: bool):
        visible_cols = [
            col for col, action in self._field_actions.items() if action.isChecked() or col == column and checked
        ]
        if not visible_cols:
            # Keep at least one field visible to avoid a blank table.
            action = self._field_actions.get(column)
            if action is not None:
                action.blockSignals(True)
                action.setChecked(True)
                action.blockSignals(False)
            self.status_label.setText("At least one field must stay visible")
            return

        self.table.setColumnHidden(column, not checked)
        self._save_field_visibility_settings()

    def _save_field_visibility_settings(self):
        visible_columns = [
            int(col) for col, action in self._field_actions.items() if action.isChecked()
        ]
        self.settings.setValue("tableVisibleColumns", visible_columns)

    def _load_field_visibility_settings(self):
        raw = self.settings.value("tableVisibleColumns", [], type=list)
        if not isinstance(raw, list):
            raw = []

        try:
            visible = {int(item) for item in raw}
        except Exception:
            visible = set()

        if visible:
            visible.add(self.COL_N_REPEATS)

        if not visible:
            # Default: hide bulky/secondary columns; keep tags visible.
            visible = set(range(self.table.columnCount())) - {
                self.COL_DATA_KEYS,
                self.COL_SCOPE,
                self.COL_LITE,
            }

        for col in range(self.table.columnCount()):
            should_show = col in visible
            action = self._field_actions.get(col)
            if action is not None:
                action.blockSignals(True)
                action.setChecked(should_show)
                action.blockSignals(False)
            self.table.setColumnHidden(col, not should_show)

    def _make_stat_chip(self, label: str, value: str):
        chip = QFrame(self)
        chip.setObjectName("statChip")
        chip_layout = QVBoxLayout(chip)
        chip_layout.setContentsMargins(8, 5, 8, 5)
        chip_layout.setSpacing(0)

        label_widget = QLabel(label, chip)
        label_widget.setObjectName("statChipLabel")
        value_widget = QLabel(value, chip)
        value_widget.setObjectName("statChipValue")
        chip_layout.addWidget(label_widget)
        chip_layout.addWidget(value_widget)
        chip.value_widget = value_widget
        return chip

    def _set_stat_chip_value(self, chip: QFrame, value: str):
        chip.value_widget.setText(value)

    def _set_activity_busy(self, text=None):
        self._busy_ops += 1
        self.activity_indicator.setStyleSheet(
            "QLabel#activityIndicator {"
            "background: #f7b955; border: 1px solid #d59b3c;"
            "border-radius: 7px; min-width: 14px; max-width: 14px;"
            "min-height: 14px; max-height: 14px; }"
        )
        if text:
            self.status_label.setText(text)

    def _set_activity_idle(self, text=None):
        self._busy_ops = max(0, self._busy_ops - 1)
        if self._busy_ops > 0:
            return
        self.activity_indicator.setStyleSheet(
            "QLabel#activityIndicator {"
            "background: #63c889; border: 1px solid #49a66f;"
            "border-radius: 7px; min-width: 14px; max-width: 14px;"
            "min-height: 14px; max-height: 14px; }"
        )
        if text:
            self.status_label.setText(text)

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: #eef3f6;
            }
            QLabel {
                color: #22303a;
            }
            QFrame#statChip {
                background: #ffffff;
                border: 1px solid #d7dfe6;
                border-radius: 8px;
            }
            QFrame#toolbarGroup {
                background: #ffffff;
                border: 1px solid #d7dfe6;
                border-radius: 8px;
            }
            QLabel#statChipLabel {
                color: #71828c;
                font-size: 9px;
                font-weight: 700;
            }
            QLabel#statChipValue {
                color: #17384b;
                font-size: 13px;
                font-weight: 700;
            }
            QLineEdit, QDateEdit, QPushButton, QComboBox, QListWidget, QTableWidget {
                background: #ffffff;
                color: #1f2a33;
                border: 1px solid #d5dde4;
                border-radius: 8px;
                padding: 4px 7px;
                min-height: 22px;
            }
            QLineEdit:focus, QDateEdit:focus, QTableWidget:focus, QComboBox:focus, QListWidget:focus {
                border: 1px solid #6f93aa;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #1f2a33;
                selection-background-color: #dcebf3;
                selection-color: #13212b;
            }
            QPushButton, QToolButton {
                background: #1d506b;
                color: #ffffff;
                font-weight: 600;
                padding: 4px 10px;
                border: none;
                border-radius: 8px;
                min-height: 22px;
            }
            QPushButton:hover, QToolButton:hover {
                background: #276382;
            }
            QPushButton:checked, QToolButton:checked {
                background: #4c7f98;
            }
            QToolButton#autoRefreshBtn {
                background: #f6f7f9;
                color: #214056;
                border: 1px solid #b9c9d5;
                border-radius: 12px;
                padding: 4px 12px;
                min-height: 24px;
                font-weight: 700;
            }
            QToolButton#autoRefreshBtn:hover {
                background: #edf2f6;
                border-color: #95aebf;
            }
            QToolButton#autoRefreshBtn:checked {
                background: #2f8f5b;
                color: #ffffff;
                border: 1px solid #267247;
            }
            QGroupBox {
                color: #2a3f4b;
                font-weight: 700;
                border: 1px solid #d7dfe6;
                border-radius: 8px;
                margin-top: 6px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QLabel#statusLabel {
                color: #495d69;
                padding: 2px 4px 0 4px;
            }
            QLabel#pathBadge {
                color: #0f3a4f;
                background: #d9edf7;
                border: 1px solid #b8d7e6;
                border-radius: 5px;
                padding: 0px 4px;
                font-weight: 700;
                font-size: 10px;
            }
            QLabel#pathValueLabel {
                color: #0b4d6f;
                text-decoration: underline;
                padding: 0 2px;
            }
            QLabel#pathValueLabel:hover {
                color: #0a6c96;
            }
            QFrame#pathPill {
                background: #f0f6fa;
                border: 1px solid #c8dbe6;
                border-radius: 6px;
            }
            QHeaderView::section {
                background: #ebf1f4;
                color: #2a3f4b;
                border: none;
                border-right: 1px solid #d6dee5;
                border-bottom: 1px solid #d6dee5;
                padding: 2px 5px;
                font-weight: 700;
            }
            QTableWidget#xvarTable QHeaderView::section {
                padding: 1px 4px;
                font-weight: 700;
            }
            QTableWidget {
                alternate-background-color: #f8fbfc;
                selection-background-color: #dcebf3;
                selection-color: #13212b;
            }
            QMenu {
                background: #ffffff;
                color: #22303a;
                border: 1px solid #d5dde4;
            }
            QMenu::item:disabled {
                color: #a8b6bf;
            }
            QLabel#paramRunBadge {
                background: #d9edf7;
                color: #174760;
                border: 1px solid #b7d6e6;
                border-radius: 10px;
                padding: 2px 8px;
                font-weight: 700;
            }
            QLabel#paramRunBadge[state="inactive"] {
                background: #eceff2;
                color: #7a8790;
                border: 1px solid #d0d7de;
            }
            QSplitter::handle {
                background: #dbe4ea;
                height: 6px;
            }
            QSplitter::handle:hover {
                background: #c1ccd5;
            }
            """
        )

    def _fixed_width_font(self):
        families = QFontDatabase.families()
        preferred = ["Consolas", "Cascadia Mono", "Courier New"]
        for family in preferred:
            if family in families:
                return QFont(family, 10)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        return font

    def _current_selected_run_ids(self):
        run_ids = []
        for row in self._get_selected_rows():
            run = self._get_run_for_row(row)
            if run is not None:
                run_ids.append(int(run.run_id))
        return run_ids

    def _visible_row_count(self):
        return sum(1 for row in range(self.table.rowCount()) if not self.table.isRowHidden(row))

    def _capture_scan_restore_state(self, scan_request_id: int):
        current_run = self._get_run_for_row(self._get_selected_row())
        focus_widget = QApplication.focusWidget()
        state = {
            "scan_request_id": int(scan_request_id),
            "selected_run_ids": self._current_selected_run_ids(),
            "current_run_id": None if current_run is None else int(current_run.run_id),
            "detail_run_id": self._detail_pane_run_id,
            "param_current_run_id": self._param_search_current_run_id,
            "param_loading_run_id": self._param_search_loading_run_id,
            "param_visible": bool(self._param_search_dialog is not None and self._param_search_dialog.isVisible()),
            "focus_widget": focus_widget,
            "scroll_value": self.table.verticalScrollBar().value(),
            "old_row_count": self.table.rowCount(),
            "old_visible_count": self._visible_row_count(),
            "filters": {
                "run_id_jump": self.run_id_jump_input.text(),
                "experiment": self.experiment_filter_input.text(),
                "xvar": self.search_input.text(),
                "tag": self.tag_filter_input.text(),
            },
        }
        LOGGER.debug(
            "Scan state captured: request_id=%s selected=%s current=%s detail=%s param_current=%s "
            "param_loading=%s param_visible=%s rows=%s visible=%s filters=%s focus=%s scroll=%s",
            state["scan_request_id"],
            state["selected_run_ids"],
            state["current_run_id"],
            state["detail_run_id"],
            state["param_current_run_id"],
            state["param_loading_run_id"],
            state["param_visible"],
            state["old_row_count"],
            state["old_visible_count"],
            state["filters"],
            type(focus_widget).__name__ if focus_widget is not None else None,
            state["scroll_value"],
        )
        return state

    def _replace_table_with_runs(self, runs: list):
        LOGGER.debug("Replacing table contents: new_run_count=%s old_row_count=%s", len(runs), self.table.rowCount())
        previous_sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        self.table.setUpdatesEnabled(False)
        try:
            self.table.clearSelection()
            self.table.setRowCount(0)
            self._runs_by_id = {}
            self._scan_loaded_count = 0
            # Pre-size once instead of insertRow() per run: each insertRow on a
            # QTableWidget triggers model row-insert bookkeeping for every column.
            self.table.setRowCount(len(runs))
            filters = self._current_filter_queries()
            for row, run in enumerate(runs):
                self._fill_row(row, run, filters)
        finally:
            self.table.setUpdatesEnabled(True)
            self.table.blockSignals(False)
            self.table.setSortingEnabled(previous_sorting)
        self._update_summary_chips()

    def _restore_scan_state(self, state: dict | None):
        if not state:
            return

        selected_run_ids = [int(run_id) for run_id in state.get("selected_run_ids", [])]
        surviving_selected = [run_id for run_id in selected_run_ids if run_id in self._runs_by_id]
        dropped_selected = [run_id for run_id in selected_run_ids if run_id not in self._runs_by_id]
        current_run_id = state.get("current_run_id")
        detail_run_id = state.get("detail_run_id")

        LOGGER.debug(
            "Restoring scan state: request_id=%s surviving_selected=%s dropped_selected=%s current=%s detail=%s rows=%s",
            state.get("scan_request_id"),
            surviving_selected,
            dropped_selected,
            current_run_id,
            detail_run_id,
            self.table.rowCount(),
        )

        self.table.blockSignals(True)
        try:
            self.table.clearSelection()
            selection_model = self.table.selectionModel()
            # One pass over the table instead of a linear search per selected run.
            row_by_run_id = self._row_by_run_id_map() if surviving_selected else {}
            for run_id in surviving_selected:
                row = row_by_run_id.get(int(run_id))
                if row is None:
                    continue
                index = self.table.model().index(row, self.COL_RUN_ID)
                selection_model.select(
                    index,
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                )

            focus_run_id = current_run_id if current_run_id in surviving_selected else (surviving_selected[0] if surviving_selected else None)
            if focus_run_id is not None:
                row = row_by_run_id.get(int(focus_run_id))
                if row is not None:
                    self.table.setCurrentCell(row, self.COL_RUN_ID)
                    item = self.table.item(row, self.COL_RUN_ID)
                    if item is not None:
                        self.table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
            else:
                self.table.verticalScrollBar().setValue(int(state.get("scroll_value", 0)))
        finally:
            self.table.blockSignals(False)

        focus_widget = state.get("focus_widget")
        if focus_widget is not None:
            try:
                focus_widget.setFocus()
            except RuntimeError:
                LOGGER.debug("Could not restore focus widget; it was deleted during scan")

        if surviving_selected:
            self._on_selection_changed()
        else:
            self._set_stat_chip_value(self.selected_chip, "none")
            if detail_run_id is not None and int(detail_run_id) not in self._runs_by_id:
                self.status_label.setText(f"Run {detail_run_id} is outside the current range; showing stale details")
                LOGGER.info("Detail pane kept stale: run_id=%s outside current scan range", detail_run_id)
            elif detail_run_id is not None:
                run = self._runs_by_id.get(int(detail_run_id))
                if run is not None:
                    self._show_run_details(run)

        if dropped_selected:
            LOGGER.info("Selection partially dropped after scan: dropped_run_ids=%s", dropped_selected)

        if state.get("param_visible"):
            param_run_id = state.get("param_current_run_id") or state.get("param_loading_run_id")
            if param_run_id is not None and int(param_run_id) not in self._runs_by_id:
                dialog = self._param_search_dialog
                if dialog is not None:
                    dialog.set_stale_run_message(int(param_run_id))
                LOGGER.info("Param search kept stale: run_id=%s outside current scan range", param_run_id)

    @staticmethod
    def _worker_running(worker):
        """isRunning() that tolerates a worker whose C++ side was deleted."""
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except RuntimeError:
            return False

    def _release_worker(self, attr_name: str, worker):
        """finished-slot: drop the window's reference before deleteLater so a
        later isRunning() check never touches a deleted QThread."""
        if getattr(self, attr_name, None) is worker:
            setattr(self, attr_name, None)
        worker.deleteLater()

    def _get_metadata_cache(self):
        """One MetadataCache per data dir, kept across scans.  Re-parsing the
        JSON cache from the network drive on every refresh (auto-refresh can
        fire every 10 s) was pure overhead."""
        if self._metadata_cache is None or self._metadata_cache_dir != self.data_dir:
            self._metadata_cache = MetadataCache(self.data_dir)
            self._metadata_cache_dir = self.data_dir
        return self._metadata_cache

    def _start_scan(self):
        if not self.data_dir:
            self.status_label.setText("DATA_DIR is empty")
            return

        if self._annotation_workers:
            # A tag/comment write still holds a file open in append mode; the
            # scanner's read-open would fail and drop that run from the table.
            self._scan_after_annotations = True
            self.status_label.setText("Waiting for annotation save before refreshing…")
            return

        LOGGER.info(
            "Starting scan: data_dir=%s date_from=%s date_to=%s",
            self.data_dir,
            self.date_from.date().toString("yyyy-MM-dd"),
            self.date_to.date().toString("yyyy-MM-dd"),
        )

        self._set_activity_busy("Scanning data…")

        self._scan_request_id += 1
        current_scan_id = self._scan_request_id
        self._detail_request_id += 1
        self._scan_restore_state = self._capture_scan_restore_state(current_scan_id)
        self._pending_scan_runs = []
        self._pending_scan_request_id = current_scan_id

        if self._scan_worker is not None and self._scan_worker.isRunning():
            LOGGER.info("Superseding active scan: old_request_id=%s new_request_id=%s", self._pending_scan_request_id, current_scan_id)
            self._scan_worker.request_stop()
            # Do not wait here; stale worker outputs are ignored via request ids.
            # The superseded scan's completion is discarded, so retire its busy
            # mark now (the new scan took its own above).
            self._set_activity_idle()

        # The background folder indexer must not compete with HDF5 reads.
        if self._worker_running(self._folder_index_worker):
            self._folder_index_worker.request_stop()

        date_from = self.date_from.date().toPyDate()
        date_to = self.date_to.date().toPyDate()
        self._active_filter_terms = self._parse_search_terms(self.search_input.text())

        if date_from > date_to:
            LOGGER.warning("Invalid scan date range: date_from=%s date_to=%s request_id=%s", date_from, date_to, current_scan_id)
            self._pending_scan_runs = []
            self._pending_scan_request_id = None
            self._scan_restore_state = None
            self.status_label.setText("Invalid date range")
            self._set_activity_idle()
            return

        scanner = RunScanner(self.data_dir, date_from, date_to, cache=self._get_metadata_cache())
        self._scan_worker = ScanWorker(scanner, batch_size=256)
        self._scan_worker.run_batch_found.connect(
            lambda runs, rid=current_scan_id: self._append_run_batch_guarded(runs, rid)
        )
        # Backward-compatible path if the worker emits single rows.
        self._scan_worker.run_found.connect(
            lambda run, rid=current_scan_id: self._append_run_guarded(run, rid)
        )
        self._scan_worker.scan_done.connect(
            lambda count, rid=current_scan_id: self._on_scan_done_guarded(count, rid)
        )
        self._scan_worker.scan_error.connect(
            lambda msg, rid=current_scan_id: self._on_scan_error_guarded(msg, rid)
        )

        self.refresh_btn.setEnabled(False)
        self.status_label.setText("Scanning...")
        self._scan_worker.start()

    def _auto_refresh_interval_seconds(self):
        value = int(self.settings.value("autoRefreshIntervalSec", 10, type=int))
        return max(1, value)

    def _sync_auto_refresh_timer(self, show_status=True):
        if not hasattr(self, "auto_refresh_btn"):
            return

        enabled = bool(self.auto_refresh_btn.isChecked())
        interval_sec = self._auto_refresh_interval_seconds()
        self.settings.setValue("autoRefreshEnabled", enabled)

        if enabled:
            self._auto_refresh_timer.start(interval_sec * 1000)
            if show_status:
                self.status_label.setText(f"Auto-refresh on ({interval_sec}s)")
        else:
            self._auto_refresh_timer.stop()
            if show_status:
                self.status_label.setText("Auto-refresh off")

    def _on_auto_refresh_toggled(self, checked: bool):
        self._sync_auto_refresh_timer(show_status=True)

    def _on_auto_refresh_timeout(self):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            LOGGER.info("Auto-refresh tick skipped: scan already running")
            return

        if self._worker_running(self._latest_run_check_worker):
            LOGGER.info("Auto-refresh tick skipped: latest-run check already running")
            return

        self._latest_run_check_request_id += 1
        current_request_id = self._latest_run_check_request_id
        current_latest = max((int(run_id) for run_id in self._runs_by_id.keys()), default=-1)
        worker = NewRunProbeWorker(self._server_talk, self.data_dir, current_latest, self)
        worker.latest_ready.connect(
            lambda latest_run_id, req_id=current_request_id: self._on_latest_run_check_ready_guarded(latest_run_id, req_id)
        )
        worker.latest_error.connect(
            lambda message, req_id=current_request_id: self._on_latest_run_check_error_guarded(message, req_id)
        )
        # Parented QThreads are otherwise retained until the window closes;
        # one every 10 s adds up over a lab day.
        worker.finished.connect(lambda w=worker: self._release_worker("_latest_run_check_worker", w))
        self._latest_run_check_worker = worker
        worker.start()

    def _on_latest_run_check_ready_guarded(self, latest_run_id, request_id: int):
        if request_id != self._latest_run_check_request_id:
            return

        if latest_run_id is None:
            LOGGER.debug("Auto-refresh tick: no new completed run")
            return

        LOGGER.info("Auto-refresh tick: new run detected (run_id=%s)", latest_run_id)
        self.status_label.setText("New run detected, refreshing…")
        self._start_scan()

    def _on_latest_run_check_error_guarded(self, message: str, request_id: int):
        if request_id != self._latest_run_check_request_id:
            return
        LOGGER.warning("Auto-refresh latest-run check failed: %s", message)

    def _on_date_to_changed(self, _qdate):
        self._date_to_tracks_today = (self.date_to.date() == QDate.currentDate())

    def _check_date_rollover(self):
        if not self._date_to_tracks_today:
            return
        today = QDate.currentDate()
        if self.date_to.date() == today:
            return
        LOGGER.info(
            "Day changed; extending date_to to track today (%s -> %s)",
            self.date_to.date().toString("yyyy-MM-dd"),
            today.toString("yyyy-MM-dd"),
        )
        self.date_to.setDate(today)
        self.status_label.setText("Date range extended to include today")
        self._start_scan()

    def _append_run(self, run: RunSummary):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._fill_row(row, run, self._current_filter_queries())
        self._update_summary_chips()

    def _fill_row(self, row: int, run: RunSummary, filters: tuple):
        """Populate one already-allocated table row.  No per-row chip/status
        updates here: doing that inside the fill loop made a rebuild O(n^2)."""
        run_id_item = _RunIdItem(str(run.run_id))
        run_id_item.setData(Qt.ItemDataRole.UserRole, int(run.run_id))
        datetime_item = QTableWidgetItem(self._format_datetime_for_table(run.run_datetime_str))
        datetime_item.setData(Qt.ItemDataRole.UserRole, run.run_datetime_str)
        expt_item = QTableWidgetItem(run.experiment_name or "-")
        expt_item.setToolTip(run.experiment_filepath or run.experiment_name or "")
        xvardims_item = QTableWidgetItem(self._format_xvardims(run.xvardims))
        n_repeats_item = QTableWidgetItem(str(int(run.n_repeats)))
        xvars_item = QTableWidgetItem(self._format_name_list_for_table(run.xvarnames))
        xvars_item.setToolTip("\n".join(run.xvarnames))

        data_preview = self._format_data_container_preview(run.data_container_keys)
        data_item = QTableWidgetItem(data_preview)
        data_item.setToolTip("\n".join(run.data_container_keys))

        scope_item = QTableWidgetItem("yes" if run.has_scope_data else "-")
        lite_item = QTableWidgetItem("lite" if run.has_lite else "-")
        tags_item = QTableWidgetItem(", ".join(run.tags) if run.tags else "")

        self.table.setItem(row, self.COL_RUN_ID, run_id_item)
        self.table.setItem(row, self.COL_DATETIME, datetime_item)
        self.table.setItem(row, self.COL_EXPERIMENT, expt_item)
        self.table.setItem(row, self.COL_XVARDIMS, xvardims_item)
        self.table.setItem(row, self.COL_N_REPEATS, n_repeats_item)
        self.table.setItem(row, self.COL_XVARS, xvars_item)
        self.table.setItem(row, self.COL_DATA_KEYS, data_item)
        self.table.setItem(row, self.COL_SCOPE, scope_item)
        self.table.setItem(row, self.COL_LITE, lite_item)
        self.table.setItem(row, self.COL_TAGS, tags_item)

        if run.has_lite:
            self._set_lite_highlight(row, True)

        self._style_row_items(row, run)

        self._runs_by_id[run.run_id] = run
        self._scan_loaded_count += 1

        terms, experiment_query, tag_query = filters
        if terms or experiment_query or tag_query:
            self.table.setRowHidden(row, not self._run_matches_terms(run, terms, experiment_query, tag_query))

    def _append_run_batch(self, runs: list):
        if not runs:
            return

        self.table.setUpdatesEnabled(False)
        try:
            start = self.table.rowCount()
            self.table.setRowCount(start + len(runs))
            filters = self._current_filter_queries()
            for offset, run in enumerate(runs):
                self._fill_row(start + offset, run, filters)
        finally:
            self.table.setUpdatesEnabled(True)
        self._update_summary_chips()

    def _append_run_guarded(self, run: RunSummary, scan_request_id: int):
        if scan_request_id != self._scan_request_id:
            LOGGER.debug("Ignoring stale single run: request_id=%s active_request_id=%s", scan_request_id, self._scan_request_id)
            return
        if self._pending_scan_request_id == scan_request_id:
            self._pending_scan_runs.append(run)
            self.status_label.setText(f"Scanning... {len(self._pending_scan_runs)} runs found")
            return
        self._append_run(run)

    def _append_run_batch_guarded(self, runs: list, scan_request_id: int):
        if scan_request_id != self._scan_request_id:
            LOGGER.debug("Ignoring stale run batch: request_id=%s active_request_id=%s batch_size=%s", scan_request_id, self._scan_request_id, len(runs))
            return
        if self._pending_scan_request_id == scan_request_id:
            self._pending_scan_runs.extend(runs)
            self.status_label.setText(f"Scanning... {len(self._pending_scan_runs)} runs found")
            LOGGER.debug("Scan batch buffered: request_id=%s batch_size=%s buffered=%s", scan_request_id, len(runs), len(self._pending_scan_runs))
            return
        self._append_run_batch(runs)

    def _on_filter_text_changed(self, *_args):
        self._active_filter_terms = self._parse_search_terms(self.search_input.text())
        self._filter_debounce_timer.start()

    def _apply_date_preset(self, days_back: int):
        self.date_to.setDate(QDate.currentDate())
        self.date_from.setDate(QDate.currentDate().addDays(-days_back))
        self._start_scan()

    def _apply_recent_hours_preset(self, hours_back: int):
        del hours_back
        # Date filter granularity is day-level, so recent-hours maps to today.
        self.date_to.setDate(QDate.currentDate())
        self.date_from.setDate(QDate.currentDate())
        self._start_scan()

    def _jump_to_run_id(self):
        raw = self.run_id_jump_input.text().strip()
        if not raw:
            self.status_label.setText("Enter a run ID")
            return
        try:
            requested_run_id = int(raw)
        except ValueError:
            self.status_label.setText("Run ID must be an integer")
            return

        if self._worker_running(self._run_id_lookup_worker):
            self.status_label.setText("Run ID search already in progress")
            LOGGER.info("Run ID lookup request ignored: another lookup is already running")
            return

        # Fast path 1: the run is already loaded, so no directory walk or
        # rescan is needed -- clear filters that might hide it and focus the row.
        if requested_run_id in self._runs_by_id:
            self._clear_filters_silently()
            if self._focus_row_by_run_id(requested_run_id):
                LOGGER.info("Run ID lookup satisfied locally: run_id=%s", requested_run_id)
                self.status_label.setText(f"Focused run {requested_run_id}")
                return

        # Fast path 2: the id falls strictly inside the loaded id range, so the
        # nearest *valid* run is already in the table (any closer file would be
        # an incomplete run the browser would not show anyway).  Zero I/O.
        if self._runs_by_id:
            loaded_min = min(self._runs_by_id)
            loaded_max = max(self._runs_by_id)
            if loaded_min < requested_run_id < loaded_max:
                nearest = min(self._runs_by_id, key=lambda rid: (abs(rid - requested_run_id), rid))
                self._clear_filters_silently()
                if self._focus_row_by_run_id(nearest):
                    LOGGER.info("Run ID lookup satisfied locally: requested=%s nearest_loaded=%s", requested_run_id, nearest)
                    self.status_label.setText(f"Run {requested_run_id} not found; focused nearest loaded run {nearest}")
                    return

        LOGGER.info("Run ID lookup started: requested_run_id=%s", requested_run_id)
        self._set_activity_busy("Searching run ID…")
        self.run_id_jump_btn.setEnabled(False)

        self._run_id_lookup_request_id += 1
        current_request_id = self._run_id_lookup_request_id
        worker = RunIdLookupWorker(
            self._server_talk, self.data_dir, requested_run_id, cache=self._get_metadata_cache(), parent=self
        )
        worker.lookup_ready.connect(
            lambda resolved_run_id, run_date, stats, req_id=current_request_id, requested=requested_run_id:
            self._on_run_id_lookup_ready_guarded(requested, resolved_run_id, run_date, req_id, stats)
        )
        worker.lookup_error.connect(
            lambda message, req_id=current_request_id: self._on_run_id_lookup_error_guarded(message, req_id)
        )
        worker.finished.connect(lambda w=worker: self._release_worker("_run_id_lookup_worker", w))
        self._run_id_lookup_worker = worker
        worker.start()

    def _clear_filters_silently(self):
        """Clear the three filter boxes and apply the (now empty) filter once,
        without the debounce timer and without three separate table passes."""
        boxes = [self.search_input, self.experiment_filter_input, self.tag_filter_input]
        if not any(box.text() for box in boxes):
            return
        for box in boxes:
            box.blockSignals(True)
            box.clear()
            box.blockSignals(False)
        self._filter_debounce_timer.stop()
        self._active_filter_terms = []
        self._apply_filter()

    def _on_run_id_lookup_ready_guarded(self, requested_run_id: int, resolved_run_id, run_date, request_id: int, stats=None):
        if request_id != self._run_id_lookup_request_id:
            return

        self.run_id_jump_btn.setEnabled(True)
        if resolved_run_id is None or run_date is None:
            LOGGER.info("Run ID lookup complete: no runs found (requested=%s)", requested_run_id)
            self.status_label.setText("No HDF5 runs found in data directory")
            self._set_activity_idle()
            return

        LOGGER.info(
            "Run ID lookup complete: requested=%s resolved=%s date=%s stats=%s",
            requested_run_id,
            resolved_run_id,
            run_date,
            stats,
        )

        if int(resolved_run_id) in self._runs_by_id:
            # Nearest run is already on screen (typical when the requested id is
            # newer than anything on disk): focus it, no rescan.
            self._clear_filters_silently()
            if self._focus_row_by_run_id(int(resolved_run_id)):
                if int(resolved_run_id) != int(requested_run_id):
                    self.status_label.setText(f"Run {requested_run_id} not found; focused nearest run {resolved_run_id}")
                else:
                    self.status_label.setText(f"Focused run {resolved_run_id}")
                self._set_activity_idle()
                return

        center_qdate = QDate(run_date.year, run_date.month, run_date.day)
        widen_from = center_qdate.addDays(-RUN_ID_JUMP_CONTEXT_DAYS)
        widen_to = center_qdate.addDays(RUN_ID_JUMP_CONTEXT_DAYS)
        already_loaded = (
            self.date_from.date() <= center_qdate <= self.date_to.date()
            and self._scan_worker is not None
            and not self._scan_worker.isRunning()
        )
        if already_loaded:
            # The day is on screen already (id fell in a gap at the edge of the
            # loaded range); widen straight away in one scan.
            self.date_from.setDate(min(self.date_from.date(), widen_from))
            self.date_to.setDate(max(self.date_to.date(), widen_to))
            self._pending_widen_range = None
        else:
            # Two-phase: scan only the target day first so the row appears as
            # soon as possible, then widen to the surrounding days.
            self.date_from.setDate(center_qdate)
            self.date_to.setDate(center_qdate)
            self._pending_widen_range = (widen_from, widen_to)

        # Clear active filters so the target row stays visible.  The scan that
        # follows rebuilds the table anyway, so skip the immediate filter pass.
        for box in (self.search_input, self.experiment_filter_input, self.tag_filter_input):
            box.blockSignals(True)
            box.clear()
            box.blockSignals(False)
        self._filter_debounce_timer.stop()
        self._active_filter_terms = []

        self._pending_focus_run_id = int(resolved_run_id)
        self._pending_requested_run_id = int(requested_run_id)
        self._set_activity_idle()
        self._start_scan()

    def _on_run_id_lookup_error_guarded(self, message: str, request_id: int):
        if request_id != self._run_id_lookup_request_id:
            return
        LOGGER.error("Run ID lookup failed: %s", message)
        self.run_id_jump_btn.setEnabled(True)
        self._pending_widen_range = None
        self.status_label.setText("Run ID search failed")
        QMessageBox.warning(self, "Run ID Search Error", message)
        self._set_activity_idle()

    def _find_nearest_run_date_and_id(self, requested_run_id: int):
        try:
            return self._server_talk.find_nearest_run_date_and_id(requested_run_id)
        except Exception:
            return None, None

    def _focus_row_by_run_id(self, run_id: int):
        row = self._find_row_for_run_id(run_id)
        if row is None:
            return False
        self.table.selectRow(row)
        item = self.table.item(row, self.COL_RUN_ID)
        if item is not None:
            self.table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
        return True

    def _install_shortcuts(self):

        refresh_ctrl_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        refresh_ctrl_shortcut.activated.connect(self._start_scan)

        copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self)
        copy_shortcut.activated.connect(self._copy_selected_run_id)

        copy_lite_shortcut = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        copy_lite_shortcut.activated.connect(self._copy_selected_lite_arg)

        create_lite_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        create_lite_shortcut.activated.connect(self._create_lite_for_selected)

        open_data_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        open_data_shortcut.activated.connect(self._open_selected_data_location)

        open_experiment_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        open_experiment_shortcut.activated.connect(self._open_selected_experiment_location)

        edit_tags_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        edit_tags_shortcut.activated.connect(self._edit_tags_for_selected)

        param_search_shortcut = QShortcut(QKeySequence("Ctrl+P"), self)
        param_search_shortcut.activated.connect(self._open_param_search_for_selected)

        cycle_search_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        cycle_search_shortcut.activated.connect(self._focus_next_search_field)

    def _show_hotkeys_guide(self):
        entries = [
            ("Ctrl+L", "Create lite dataset(s)"),
            ("Ctrl+D", "Open data location"),
            ("Ctrl+E", "Open experiment location"),
            ("Ctrl+P", "Search params for selected run"),
            ("Ctrl+T", "Edit tags for selected run"),
            ("Ctrl+F", "Cycle search focus: experiment -> xvar -> tag"),
            ("Ctrl+R", "Refresh")
        ]
        text = "\n".join(f"{key:<8} {description}" for key, description in entries)
        QMessageBox.information(self, "Hotkeys", text)

    def _focus_next_search_field(self):
        fields = [self.experiment_filter_input, self.search_input, self.tag_filter_input]
        focused = self.focusWidget()
        if focused in fields:
            idx = fields.index(focused)
            target = fields[(idx + 1) % len(fields)]
        else:
            target = fields[0]
        target.setFocus()
        target.selectAll()

    def _open_selected_data_location(self):
        row = self._get_selected_row()
        run = self._get_run_for_row(row)
        if run is None:
            self.status_label.setText("No run selected")
            return
        self.detail_pane._open_path_directory(run.filepath)

    def _open_selected_experiment_location(self):
        row = self._get_selected_row()
        run = self._get_run_for_row(row)
        if run is None:
            self.status_label.setText("No run selected")
            return
        self.detail_pane._open_path_directory(run.experiment_filepath)

    def _edit_tags_for_selected(self):
        row = self._get_selected_row()
        run = self._get_run_for_row(row)
        if run is None:
            self.status_label.setText("No run selected")
            return
        self._edit_tags_for_run(run, row)

    def _load_saved_searches(self):
        pass  # removed — saved searches feature has been removed

    def _save_current_search(self):
        pass

    def _apply_saved_search(self, index: int):
        pass

    def _format_run_ids_for_copy(self, ids: list[int]) -> str:
        ids = sorted(int(i) for i in ids)
        if len(ids) <= 5:
            return "[" + ", ".join(str(i) for i in ids) + "]"

        groups = []
        start = prev = ids[0]
        for value in ids[1:]:
            if value == prev + 1:
                prev = value
                continue
            groups.append((start, prev))
            start = prev = value
        groups.append((start, prev))

        parts = [f"{lo}:{hi + 1}" if hi > lo else str(lo) for lo, hi in groups]
        return "np.r_[" + ", ".join(parts) + "]"

    def _copy_selected_run_id(self):
        row = self._get_selected_row()
        run = self._get_run_for_row(row)
        if run is None:
            return
        QApplication.clipboard().setText(str(run.run_id))
        self.status_label.setText(f"Copied: {run.run_id}")

    def _copy_selected_lite_arg(self):
        row = self._get_selected_row()
        run = self._get_run_for_row(row)
        if run is None or not run.has_lite:
            return
        text = f"{run.run_id}, lite=True"
        QApplication.clipboard().setText(text)
        self.status_label.setText(f"Copied: {text}")

    def _create_lite_for_selected(self):
        selected_rows = self._get_selected_rows()
        run_ids = []
        for row in selected_rows:
            run = self._get_run_for_row(row)
            if run is not None:
                run_ids.append(int(run.run_id))
        if not run_ids:
            return
        self._start_lite_creation_for_runs(run_ids)

    def _format_data_container_preview(self, keys: list[str]):
        return self._format_name_list_for_table(keys)

    def _format_xvardims(self, xvardims: tuple[int, ...]):
        if not xvardims:
            return "()"
        return str(tuple(int(value) for value in xvardims))

    def _format_name_list_for_table(self, names: list[str], max_items: int = 5):
        if not names:
            return "-"
        if len(names) <= max_items:
            return "  |  ".join(names)
        shown = "  |  ".join(names[:max_items])
        remaining = len(names) - max_items
        return f"{shown}  |  +{remaining} more"

    def _format_datetime_for_table(self, run_datetime_str: str):
        if not run_datetime_str:
            return ""

        def normalize_date(date_text: str):
            return date_text.replace("-", "/").strip()

        parts = run_datetime_str.split("_")
        if len(parts) != 2:
            return run_datetime_str.replace("_", " ").replace("-", "/").strip()
        date_str, time_str = parts
        date_str = normalize_date(date_str)
        time_bits = time_str.split("-")
        if len(time_bits) >= 2:
            hour_str, minute_str = time_bits[0], time_bits[1]
            use_ampm = bool(self.settings.value("timeFormatAMPM", True, type=bool))
            if use_ampm:
                try:
                    hour = int(hour_str)
                    suffix = "AM" if hour < 12 else "PM"
                    display_hour = hour % 12 or 12
                    return f"{date_str} {display_hour}:{minute_str} {suffix}"
                except ValueError:
                    pass
            return f"{date_str} {hour_str}:{minute_str}"
        return f"{date_str} {time_str.strip()}"

    def _set_lite_highlight(self, row: int, enabled: bool):
        color = _COLOR_LITE_ROW if enabled else _COLOR_PLAIN_ROW
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if item is not None:
                item.setBackground(color)

    def _style_row_items(self, row: int, run: RunSummary):
        xvardims_item = self.table.item(row, self.COL_XVARDIMS)
        if xvardims_item is not None:
            xvardims_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            xvardims_item.setForeground(_COLOR_MUTED_TEXT)

        n_repeats_item = self.table.item(row, self.COL_N_REPEATS)
        if n_repeats_item is not None:
            n_repeats_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            n_repeats_item.setForeground(_COLOR_MUTED_TEXT)

        experiment_item = self.table.item(row, self.COL_EXPERIMENT)
        if experiment_item is not None:
            experiment_item.setForeground(_COLOR_EXPERIMENT_TEXT)

        scope_item = self.table.item(row, self.COL_SCOPE)
        if scope_item is not None:
            scope_item.setText("scope" if run.has_scope_data else "-")
            scope_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if run.has_scope_data:
                scope_item.setBackground(_COLOR_SCOPE_BG)
                scope_item.setForeground(_COLOR_SCOPE_FG)

        lite_item = self.table.item(row, self.COL_LITE)
        if lite_item is not None:
            lite_item.setText("lite" if run.has_lite else "-")
            lite_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if run.has_lite:
                lite_item.setBackground(_COLOR_LITE_BG)
                lite_item.setForeground(_COLOR_LITE_FG)

    def _update_summary_chips(self):
        visible_count = 0
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                visible_count += 1
        self._set_stat_chip_value(self.loaded_chip, str(self.table.rowCount()))
        self._set_stat_chip_value(self.visible_chip, str(visible_count))

    def _current_filter_queries(self):
        """(xvar_terms, experiment_query, tag_query) read once per filter pass
        rather than once per row."""
        return (
            list(self._active_filter_terms),
            self.experiment_filter_input.text().strip().lower(),
            self.tag_filter_input.text().strip().lower(),
        )

    def _on_scan_done(self, count: int):
        pending_runs = list(self._pending_scan_runs) if self._pending_scan_request_id == self._scan_request_id else []
        restore_state = self._scan_restore_state
        LOGGER.info("Scan completed: count=%s buffered=%s request_id=%s", count, len(pending_runs), self._scan_request_id)
        self._replace_table_with_runs(pending_runs)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(self.COL_RUN_ID, Qt.SortOrder.DescendingOrder)
        self.refresh_btn.setEnabled(True)
        self.status_label.setText("Ready")
        self._refresh_date_separators()
        self._apply_filter()
        self._update_summary_chips()
        self._pending_scan_runs = []
        self._pending_scan_request_id = None
        self._scan_restore_state = None

        if self._pending_focus_run_id is not None:
            target = int(self._pending_focus_run_id)
            requested = self._pending_requested_run_id
            focused = self._focus_row_by_run_id(target)
            if focused:
                if requested is not None and int(requested) != target:
                    message = f"Run {requested} not found; focused nearest run {target}"
                else:
                    message = f"Focused run {target}"
            elif self._runs_by_id:
                # The nearest file on disk is not a completed run (still being
                # written, or broken), so it is not in the table.  Fall back to
                # the nearest run that is.
                anchor = target if requested is None else int(requested)
                fallback = min(self._runs_by_id, key=lambda rid: (abs(rid - anchor), rid))
                focused = self._focus_row_by_run_id(fallback)
                message = (
                    f"Run {anchor} not found; nearest file {target} is not a completed run; "
                    f"focused run {fallback}"
                )
            else:
                message = "Could not focus requested run"
            self.status_label.setText(message)
            self._pending_focus_message = message if focused else None
            self._pending_focus_run_id = None
            self._pending_requested_run_id = None
        else:
            self._restore_scan_state(restore_state)
            if self._pending_focus_message is not None:
                # Second phase of a run-ID jump: keep the focus message visible.
                self.status_label.setText(self._pending_focus_message)
                self._pending_focus_message = None

        self._set_activity_idle()

        if self._pending_widen_range is not None:
            # Phase two of a run-ID jump: the target day is on screen and its
            # row is focused; now widen to the surrounding days.  The restore
            # logic keeps the selection and re-centres the row.
            widen_from, widen_to = self._pending_widen_range
            self._pending_widen_range = None
            LOGGER.info("Run ID jump: widening range to %s..%s", widen_from.toString("yyyy-MM-dd"), widen_to.toString("yyyy-MM-dd"))
            self.date_from.setDate(widen_from)
            self.date_to.setDate(widen_to)
            self._start_scan()
            return

        self._maybe_start_folder_indexing()

    def _maybe_start_folder_indexing(self):
        """Kick off the one-time background run-id index build for this data
        dir (resumed if an earlier attempt was interrupted by a scan)."""
        if self._folder_index_complete_dir == self.data_dir:
            return
        if self._worker_running(self._folder_index_worker):
            return
        worker = FolderIndexWorker(self.data_dir, self._get_metadata_cache())
        worker.done.connect(lambda n, completed, d=self.data_dir: self._on_folder_index_done(d, n, completed))
        worker.finished.connect(lambda w=worker: self._release_worker("_folder_index_worker", w))
        self._folder_index_worker = worker
        worker.start()

    def _on_folder_index_done(self, data_dir: str, indexed: int, completed: bool):
        if completed and data_dir == self.data_dir:
            self._folder_index_complete_dir = data_dir
        LOGGER.info("Folder index pass finished: folders_indexed=%s completed=%s", indexed, completed)

    def _on_scan_done_guarded(self, count: int, scan_request_id: int):
        if scan_request_id != self._scan_request_id:
            return
        self._on_scan_done(count)

    def _refresh_date_separators(self):
        # Trigger table repaint so the delegate can draw date separators.
        self.table.viewport().update()

    def _on_scan_error(self, message: str):
        LOGGER.error("Scan failed: %s", message)
        self._pending_scan_runs = []
        self._pending_scan_request_id = None
        self._scan_restore_state = None
        self.refresh_btn.setEnabled(True)
        self.status_label.setText("Scan failed")
        QMessageBox.warning(self, "Scan Error", message)
        self._set_activity_idle()

    def _on_scan_error_guarded(self, message: str, scan_request_id: int):
        if scan_request_id != self._scan_request_id:
            return
        self._on_scan_error(message)

    def _apply_filter(self):
        self._filter_debounce_timer.stop()
        terms, experiment_query, tag_query = self._current_filter_queries()
        no_filter = not terms and not experiment_query and not tag_query

        table = self.table
        table.setUpdatesEnabled(False)
        try:
            for row in range(table.rowCount()):
                run = self._get_run_for_row(row)
                if run is None:
                    table.setRowHidden(row, True)
                    continue
                if no_filter:
                    table.setRowHidden(row, False)
                    continue
                table.setRowHidden(row, not self._run_matches_terms(run, terms, experiment_query, tag_query))
        finally:
            table.setUpdatesEnabled(True)
        self._update_summary_chips()

    def _parse_search_terms(self, query: str):
        return parse_name_search_terms(query)

    def _run_matches_terms(self, run: RunSummary, terms: list[str], experiment_query: str = None, tag_query: str = None):
        if experiment_query is None:
            experiment_query = self.experiment_filter_input.text().strip().lower()
        if tag_query is None:
            tag_query = self.tag_filter_input.text().strip().lower()

        if experiment_query and not name_matches_term(experiment_query, run.experiment_name or ""):
            return False

        if tag_query:
            if not any(name_matches_term(tag_query, t) for t in run.tags):
                return False

        if not terms:
            return True

        return any_name_matches_all_terms(run.xvarnames, terms)

    def _matches_xvar_term(self, term: str, normalized_term: str, raw_name: str, normalized_name: str):
        del normalized_term, normalized_name
        return name_matches_term(term, raw_name)

    def _is_subsequence(self, needle: str, haystack: str):
        return is_subsequence(needle, haystack)

    def _normalize_match_text(self, value: str):
        return normalize_match_text(value)

    def _get_selected_row(self):
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            return -1
        return indexes[0].row()

    def _get_selected_rows(self):
        indexes = self.table.selectionModel().selectedRows()
        rows = sorted({idx.row() for idx in indexes})
        return rows

    def _on_selection_changed(self):
        selected_rows = self._get_selected_rows()
        selected_run_ids = []
        for selected_row in selected_rows:
            selected_run = self._get_run_for_row(selected_row)
            if selected_run is not None:
                selected_run_ids.append(int(selected_run.run_id))
        LOGGER.debug(
            "Selection changed: rows=%s run_ids=%s detail_request_id=%s xvar_loader_running=%s "
            "param_current=%s param_loading=%s param_visible=%s focus=%s",
            selected_rows,
            selected_run_ids,
            self._detail_request_id,
            bool(self._xvar_loader is not None and self._xvar_loader.isRunning()),
            self._param_search_current_run_id,
            self._param_search_loading_run_id,
            bool(self._param_search_dialog is not None and self._param_search_dialog.isVisible()),
            type(QApplication.focusWidget()).__name__ if QApplication.focusWidget() is not None else None,
        )
        row = self._get_selected_row()
        run = self._get_run_for_row(row)
        multi = len(selected_rows) > 1

        if multi:
            self._set_stat_chip_value(self.selected_chip, "multi")
        if run is None:
            self._set_stat_chip_value(self.selected_chip, "none")
            self._detail_pane_run_id = None
            self.detail_pane.clear_details()
            self._sync_param_search_to_selected_run(None, selection_locked=False)
            return

        # One sync call per selection change (previously up to three).  The
        # actual param load is debounced so holding an arrow key does not
        # spawn a loader thread per row.
        self._sync_param_search_to_selected_run(None if multi else run, selection_locked=multi)

        self._detail_request_id += 1
        current_detail_id = self._detail_request_id
        self._set_stat_chip_value(self.selected_chip, str(run.run_id))

        if run.xvar_details:
            self._show_run_details(run)
            return

        if self._xvar_loader is not None and self._xvar_loader.isRunning():
            # Keep UI responsive on rapid row changes; stale loader results are
            # ignored via detail request ids.  Move the old loader into a
            # retention set so Python doesn't GC it while the C++ thread is
            # still running (PyQt6 segfault).  It is removed from the set via
            # its finished signal once the thread actually exits.
            old_loader = self._xvar_loader
            self._running_xvar_loaders.add(old_loader)
            old_loader.finished.connect(lambda l=old_loader: self._running_xvar_loaders.discard(l))
            LOGGER.debug(
                "Interrupting XvarDetailLoader: run_id=%s detail_request_id=%s retained_loaders=%s",
                run.run_id,
                current_detail_id,
                len(self._running_xvar_loaders),
            )
            old_loader.requestInterruption()
            # An interrupted loader never reports back, so retire its busy mark
            # here or the activity indicator stays amber for good.
            self._set_activity_idle()

        self.detail_pane.clear_details()
        self._set_activity_busy("Loading run details…")
        self._xvar_loader = XvarDetailLoader(run.filepath, run.xvarnames)
        self._xvar_loader.details_ready.connect(
            lambda details, run_id=run.run_id, req_id=current_detail_id: self._on_xvar_details_ready_guarded(run_id, details, req_id)
        )
        self._xvar_loader.details_error.connect(
            lambda msg, req_id=current_detail_id: self._on_xvar_details_error_guarded(msg, req_id)
        )
        self._xvar_loader.start()

    def _on_xvar_details_ready(self, run_id: int, details: list):
        run = self._runs_by_id.get(run_id)
        if run is None:
            self._set_activity_idle()
            return
        run.xvar_details = details
        self._show_run_details(run)
        self._set_activity_idle()

    def _on_xvar_details_ready_guarded(self, run_id: int, details: list, detail_request_id: int):
        if detail_request_id != self._detail_request_id:
            LOGGER.debug(
                "Dropping stale xvar details: run_id=%s request_id=%s active_request_id=%s details=%s",
                run_id,
                detail_request_id,
                self._detail_request_id,
                len(details),
            )
            return
        self._on_xvar_details_ready(run_id, details)

    def _on_xvar_details_error_guarded(self, message: str, detail_request_id: int):
        if detail_request_id != self._detail_request_id:
            LOGGER.debug(
                "Dropping stale xvar error: request_id=%s active_request_id=%s message=%s",
                detail_request_id,
                self._detail_request_id,
                message,
            )
            return
        LOGGER.error("Xvar detail load failed: %s", message)
        self.detail_pane.show_message(f"Failed to load xvar details: {message}")
        self._set_activity_idle()

    def _show_run_details(self, run: RunSummary):
        self._detail_pane_run_id = int(run.run_id)
        self.detail_pane.set_run(
            run,
            self._format_datetime_for_table(run.run_datetime_str),
            self._format_xvardims(run.xvardims),
        )

    def _on_row_double_clicked(self, item):
        self._copy_selected_run_id()

    def _on_context_menu(self, pos):
        row = self.table.rowAt(pos.y())
        run = self._get_run_for_row(row)
        if run is None:
            return

        # Right-clicking a non-selected row should target only that row.
        selected_rows = self._get_selected_rows()
        if row not in selected_rows:
            self.table.clearSelection()
            self.table.selectRow(row)
            selected_rows = [row]

        selected_run_rows = []
        for selected_row in selected_rows:
            selected_run = self._get_run_for_row(selected_row)
            if selected_run is not None:
                selected_run_rows.append((selected_run, selected_row))

        if not selected_run_rows:
            return

        is_multi_selection = len(selected_run_rows) > 1

        menu = QMenu(self)
        create_action = menu.addAction("Create Lite Dataset")
        copy_run_id_action = menu.addAction("Copy Run ID")
        copy_lite_arg_action = menu.addAction("Copy Lite Arg")
        open_h5_action = menu.addAction("Open H5 File")
        search_params_action = menu.addAction("Search Params…")

        if not run.has_lite:
            copy_lite_arg_action.setEnabled(False)

        if is_multi_selection:
            copy_lite_arg_action.setEnabled(False)
            open_h5_action.setEnabled(False)
            search_params_action.setEnabled(False)

        menu.addSeparator()
        add_tag_menu = menu.addMenu("Add Tag")
        common_tag_actions = {}
        common_tags = self._get_common_tags()
        if common_tags:
            for tag in common_tags:
                action = add_tag_menu.addAction(tag)
                action.setCheckable(True)
                action.setChecked(self._all_selected_runs_have_tag(selected_run_rows, tag))
                common_tag_actions[action] = tag
        else:
            add_tag_menu.addAction("(no common tags)").setEnabled(False)
        add_tag_menu.addSeparator()
        add_custom_tag_action = add_tag_menu.addAction("Custom Tag…")
        edit_tags_action = add_tag_menu.addAction("Edit Tags…")
        edit_common_tags_action = add_tag_menu.addAction("Edit Common Tags…")
        if is_multi_selection:
            edit_tags_action.setEnabled(False)

        edit_comment_action = menu.addAction("Edit Comment…")

        menu.addSeparator()
        roi_action = menu.addAction("Select ROI…")
        if is_multi_selection:
            roi_action.setEnabled(False)

        menu.addSeparator()
        view_files_submenu = self._build_view_files_submenu(menu, run)
        if is_multi_selection:
            view_files_submenu.menuAction().setEnabled(False)

        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is create_action:
            self._start_lite_creation_for_runs([r.run_id for r, _ in selected_run_rows])
        elif chosen is copy_run_id_action:
            if is_multi_selection:
                ids = [r.run_id for r, _ in selected_run_rows]
                text = self._format_run_ids_for_copy(ids)
            else:
                text = str(run.run_id)
            QApplication.clipboard().setText(text)
            self.status_label.setText(f"Copied: {text}")
        elif chosen is copy_lite_arg_action:
            if is_multi_selection:
                return
            text = f"{run.run_id}, lite=True"
            QApplication.clipboard().setText(text)
            self.status_label.setText(f"Copied: {text}")
        elif chosen is open_h5_action:
            self._open_file_in_default_program(run.filepath, "H5 file")
        elif chosen is search_params_action:
            self._open_param_search_for_selected()
        elif chosen in common_tag_actions:
            self._toggle_common_tag_for_runs(selected_run_rows, common_tag_actions[chosen])
        elif chosen is add_custom_tag_action:
            self._add_custom_tag_to_runs(selected_run_rows)
        elif chosen is edit_tags_action:
            self._edit_tags_for_run(run, row)
        elif chosen is edit_comment_action:
            self._edit_comment_for_runs([r for r, _ in selected_run_rows])
        elif chosen is edit_common_tags_action:
            self._edit_common_tags()
        elif chosen is roi_action:
            self._open_roi_select_for_run(run)

# ---------------------------------------------------------------------------
# Tags & comment editing
# ---------------------------------------------------------------------------

    def _edit_tags_for_run(self, run: RunSummary, row: int):
        current = ", ".join(run.tags)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit Tags — run {run.run_id}")
        dlg.resize(460, 190)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)
        note = QLabel("Enter comma-separated tags:", dlg)
        layout.addWidget(note)
        edit = QLineEdit(current, dlg)
        edit.setClearButtonEnabled(True)
        layout.addWidget(edit)

        common_row = QHBoxLayout()
        common_row.setSpacing(6)
        common_tags_btn = QToolButton(dlg)
        common_tags_btn.setText("Common Tags")
        common_tags_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        common_tags_menu = QMenu(common_tags_btn)
        common_tags_btn.setMenu(common_tags_menu)

        edit_common_btn = QPushButton("Edit Common Tags…", dlg)
        common_row.addWidget(common_tags_btn)
        common_row.addWidget(edit_common_btn)
        common_row.addStretch(1)
        layout.addLayout(common_row)

        def parse_edit_tags() -> list[str]:
            tags = []
            seen = set()
            for raw_tag in edit.text().split(","):
                tag = raw_tag.strip()
                if not tag:
                    continue
                key = tag.lower()
                if key in seen:
                    continue
                seen.add(key)
                tags.append(tag)
            return tags

        def set_edit_tags(tags: list[str]):
            edit.setText(", ".join(tags))

        def toggle_common_tag(tag: str):
            tags = parse_edit_tags()
            key = tag.lower()
            if any(existing.lower() == key for existing in tags):
                tags = [existing for existing in tags if existing.lower() != key]
            else:
                tags.append(tag)
            set_edit_tags(tags)

        def sync_common_checks():
            active = {tag.lower() for tag in parse_edit_tags()}
            for action in common_tags_menu.actions():
                data_tag = action.data()
                if not isinstance(data_tag, str):
                    continue
                action.blockSignals(True)
                action.setChecked(data_tag.lower() in active)
                action.blockSignals(False)

        def rebuild_common_tags_menu():
            common_tags_menu.clear()
            tags = self._get_common_tags()
            if not tags:
                common_tags_menu.addAction("(no common tags)").setEnabled(False)
                return
            for common_tag in tags:
                action = common_tags_menu.addAction(common_tag)
                action.setCheckable(True)
                action.setData(common_tag)
                action.triggered.connect(
                    lambda checked=False, t=common_tag: toggle_common_tag(t)
                )
            sync_common_checks()

        edit.textChanged.connect(sync_common_checks)
        edit_common_btn.clicked.connect(lambda: (self._edit_common_tags(), rebuild_common_tags_menu()))
        rebuild_common_tags_menu()

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        raw = edit.text()
        new_tags = [t.strip() for t in raw.split(",") if t.strip()]
        self._write_run_annotation(run, tags=new_tags)
        run.tags = new_tags
        tags_item = self.table.item(row, self.COL_TAGS)
        if tags_item is not None:
            tags_item.setText(", ".join(new_tags))
        self.status_label.setText(f"Tags saved for run {run.run_id}")
        # Reapply filter in case tag filter is active
        self._apply_filter()

    def _edit_comment_for_run(self, run: RunSummary):
        self._edit_comment_for_runs([run])

    def _edit_comment_for_runs(self, runs: list[RunSummary]):
        if not runs:
            return

        multiple = len(runs) > 1
        dlg = QDialog(self)
        if multiple:
            dlg.setWindowTitle(f"Edit Comment — {len(runs)} runs")
        else:
            dlg.setWindowTitle(f"Edit Comment — run {runs[0].run_id}")
        dlg.resize(480, 200)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)
        edit = QTextEdit(dlg)
        if not multiple:
            edit.setPlainText(runs[0].comment or "")
        layout.addWidget(edit)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_comment = edit.toPlainText()
        for run in runs:
            run.comment = new_comment
        self._write_run_annotations([(run, None, new_comment) for run in runs])

        if multiple:
            self.status_label.setText(f"Comment saved for {len(runs)} runs")
        else:
            self.status_label.setText(f"Comment saved for run {runs[0].run_id}")

        # Refresh detail pane if the currently shown run was edited.
        selected_row = self._get_selected_row()
        selected_run = self._get_run_for_row(selected_row)
        if selected_run is not None and any(int(r.run_id) == int(selected_run.run_id) for r in runs):
            self._show_run_details(selected_run)

    def _write_run_annotation(self, run: RunSummary, tags=None, comment=None):
        """Queue a browser_tags / browser_comment write for one run."""
        self._write_run_annotations([(run, tags, comment)])

    def _write_run_annotations(self, jobs: list):
        """Write annotations for several runs in one background thread.

        jobs: [(run, tags_or_None, comment_or_None)].  Opening a file in append
        mode over the network can take a few hundred ms; tagging a multi-select
        of 50 runs used to freeze the window for the whole batch.  The in-memory
        RunSummary and table cells are updated optimistically by the caller; a
        failed write is reported via a warning dialog.
        """
        payload = [
            (int(run.run_id), run.filepath, None if tags is None else list(tags), comment)
            for run, tags, comment in jobs
        ]
        if not payload:
            return

        self._set_activity_busy("Saving annotations…")
        worker = AnnotationWriteWorker(payload)
        self._annotation_workers.add(worker)
        worker.error.connect(self._on_annotation_write_error)
        worker.completed.connect(lambda ok, total, w=worker: self._on_annotation_writes_completed(w, ok, total))
        worker.start()

    def _on_annotation_write_error(self, run_id: int, message: str):
        LOGGER.error("Annotation write failed: run_id=%s error=%s", run_id, message)
        QMessageBox.warning(self, "Write Error", f"Could not save annotation for run {run_id}:\n{message}")

    def _on_annotation_writes_completed(self, worker, ok: int, total: int):
        self._annotation_workers.discard(worker)
        worker.deleteLater()
        self._set_activity_idle()
        if ok == total:
            LOGGER.info("Annotation writes completed: %s/%s", ok, total)
        if not self._annotation_workers and self._scan_after_annotations:
            self._scan_after_annotations = False
            self._start_scan()

    def _get_common_tags(self):
        raw = self.settings.value("commonTags", [], type=list)
        if not isinstance(raw, list):
            return []
        cleaned = []
        seen = set()
        for value in raw:
            tag = str(value).strip()
            if not tag:
                continue
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(tag)
        return cleaned

    def _set_common_tags(self, tags: list[str]):
        cleaned = []
        seen = set()
        for value in tags:
            tag = str(value).strip()
            if not tag:
                continue
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(tag)
        self.settings.setValue("commonTags", cleaned)

    def _run_has_tag(self, run: RunSummary, tag: str):
        key = (tag or "").strip().lower()
        if not key:
            return False
        return any(str(existing).strip().lower() == key for existing in (run.tags or []))

    def _all_selected_runs_have_tag(self, run_rows: list[tuple[RunSummary, int]], tag: str):
        if not run_rows:
            return False
        return all(self._run_has_tag(run, tag) for run, _ in run_rows)

    def _toggle_common_tag_for_runs(self, run_rows: list[tuple[RunSummary, int]], tag: str):
        tag = (tag or "").strip()
        if not tag or not run_rows:
            return

        remove_from_all = self._all_selected_runs_have_tag(run_rows, tag)
        changed = 0
        jobs = []
        for run, row in run_rows:
            tags = list(run.tags or [])
            if remove_from_all:
                new_tags = [t for t in tags if str(t).strip().lower() != tag.lower()]
            else:
                if any(str(t).strip().lower() == tag.lower() for t in tags):
                    new_tags = tags
                else:
                    new_tags = tags + [tag]

            if new_tags == tags:
                continue

            run.tags = new_tags
            jobs.append((run, new_tags, None))
            tags_item = self.table.item(row, self.COL_TAGS)
            if tags_item is not None:
                tags_item.setText(", ".join(new_tags))
            changed += 1
        self._write_run_annotations(jobs)

        selected_row = self._get_selected_row()
        selected_run = self._get_run_for_row(selected_row)
        if selected_run is not None:
            self._show_run_details(selected_run)

        self._apply_filter()

        total = len(run_rows)
        if remove_from_all:
            if total == 1:
                self.status_label.setText(f"Removed tag '{tag}' from run {run_rows[0][0].run_id}")
            else:
                self.status_label.setText(f"Removed tag '{tag}' from {changed}/{total} runs")
        else:
            if total == 1:
                self.status_label.setText(f"Added tag '{tag}' to run {run_rows[0][0].run_id}")
            else:
                self.status_label.setText(f"Applied tag '{tag}' to all {total} runs ({changed} changed)")

    def _edit_common_tags(self):
        current = self._get_common_tags()
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Common Tags")
        dlg.resize(420, 240)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)
        note = QLabel("Enter tags separated by commas or new lines:", dlg)
        layout.addWidget(note)
        edit = QTextEdit(dlg)
        edit.setPlainText("\n".join(current))
        layout.addWidget(edit)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        raw = edit.toPlainText().replace(",", "\n")
        parsed = [item.strip() for item in raw.splitlines() if item.strip()]
        self._set_common_tags(parsed)
        self.status_label.setText("Common tags updated")

    def _add_tag_to_run(self, run: RunSummary, row: int, tag: str):
        self._add_tag_to_runs([(run, row)], tag)

    def _add_tag_to_runs(self, run_rows: list[tuple[RunSummary, int]], tag: str):
        tag = (tag or "").strip()
        if not tag:
            return

        added_count = 0
        unchanged_count = 0
        jobs = []
        for run, row in run_rows:
            tags = list(run.tags or [])
            if any(existing.lower() == tag.lower() for existing in tags):
                unchanged_count += 1
                continue
            tags.append(tag)
            run.tags = tags
            jobs.append((run, tags, None))
            tags_item = self.table.item(row, self.COL_TAGS)
            if tags_item is not None:
                tags_item.setText(", ".join(tags))
            added_count += 1
        self._write_run_annotations(jobs)

        selected_row = self._get_selected_row()
        selected_run = self._get_run_for_row(selected_row)
        if selected_run is not None:
            self._show_run_details(selected_run)

        self._apply_filter()
        total = len(run_rows)
        if total == 1 and added_count == 1:
            self.status_label.setText(f"Added tag '{tag}' to run {run_rows[0][0].run_id}")
            return
        self.status_label.setText(
            f"Added tag '{tag}' to {added_count}/{total} runs"
            + ("" if unchanged_count == 0 else f" ({unchanged_count} unchanged)")
        )

    def _add_custom_tag_to_run(self, run: RunSummary, row: int):
        self._add_custom_tag_to_runs([(run, row)])

    def _add_custom_tag_to_runs(self, run_rows: list[tuple[RunSummary, int]]):
        if not run_rows:
            return

        run_count = len(run_rows)
        dlg = QDialog(self)
        if run_count == 1:
            dlg.setWindowTitle(f"Add Tag — run {run_rows[0][0].run_id}")
        else:
            dlg.setWindowTitle(f"Add Tag — {run_count} runs")
        dlg.resize(380, 110)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)
        edit = QLineEdit(dlg)
        edit.setPlaceholderText("new tag")
        edit.setClearButtonEnabled(True)
        layout.addWidget(edit)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        tag = edit.text().strip()
        if not tag:
            return
        self._add_tag_to_runs(run_rows, tag)

    def _start_lite_creation(self, run_id: int):
        self._start_lite_creation_for_runs([run_id])

    def _ensure_param_search_dialog(self):
        if self._param_search_dialog is None:
            self._param_search_dialog = ParamSearchDialog(self)
            saved_mode = self.settings.value("paramSearchMode", "params", type=str)
            self._param_search_dialog.set_mode(saved_mode)
            self._param_search_dialog.finished.connect(self._on_param_search_dialog_closed)
        return self._param_search_dialog

    def _on_param_search_dialog_closed(self):
        if self._param_search_dialog is not None:
            self.settings.setValue("paramSearchMode", self._param_search_dialog._active_mode)
        self._param_sync_timer.stop()
        self._pending_param_sync_run_id = None
        self._param_search_current_run_id = None
        self._param_search_loading_run_id = None
        if self._param_search_loader is not None and self._param_search_loader.isRunning():
            self._param_search_loader.requestInterruption()

    def _sync_param_search_to_selected_run(self, run: RunSummary | None, selection_locked: bool = False):
        dialog = self._param_search_dialog
        if dialog is None or not dialog.isVisible():
            return

        dialog.set_selection_lock_state(selection_locked)
        LOGGER.debug(
            "Param search sync requested: target_run=%s selection_locked=%s current=%s loading=%s visible=%s",
            None if run is None else int(run.run_id),
            selection_locked,
            self._param_search_current_run_id,
            self._param_search_loading_run_id,
            dialog.isVisible(),
        )
        if selection_locked:
            LOGGER.info("Param search sync paused: multiple runs selected")
            self._param_sync_timer.stop()
            self._pending_param_sync_run_id = None
            return

        if run is None:
            return

        target_run_id = int(run.run_id)
        if target_run_id in (self._param_search_current_run_id, self._param_search_loading_run_id):
            self._param_sync_timer.stop()
            self._pending_param_sync_run_id = None
            dialog.set_run(run)
            return

        # Show the target immediately so the badge tracks the selection, but
        # defer the HDF5 read until the selection has settled.
        dialog.set_run(run)
        self._pending_param_sync_run_id = target_run_id
        self._param_sync_timer.start()

    def _flush_pending_param_sync(self):
        run_id = self._pending_param_sync_run_id
        self._pending_param_sync_run_id = None
        if run_id is None:
            return
        dialog = self._param_search_dialog
        if dialog is None or not dialog.isVisible():
            return
        run = self._runs_by_id.get(int(run_id))
        if run is None:
            return
        # Only load if the run is still the single selected one.
        selected_rows = self._get_selected_rows()
        if len(selected_rows) != 1:
            return
        current = self._get_run_for_row(selected_rows[0])
        if current is None or int(current.run_id) != int(run_id):
            return
        self._load_param_search_for_run(run, show_dialog=False, focus_search=False, force=False)

    def _load_param_search_for_run(self, run: RunSummary, show_dialog: bool, focus_search: bool, force: bool):
        dialog = self._ensure_param_search_dialog()

        if show_dialog:
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        if focus_search:
            dialog.focus_search()

        target_run_id = int(run.run_id)
        if not force:
            if self._param_search_loading_run_id == target_run_id:
                LOGGER.debug("Param search load skipped: run_id=%s already loading", target_run_id)
                return
            if self._param_search_current_run_id == target_run_id:
                LOGGER.debug("Param search load skipped: run_id=%s already current", target_run_id)
                dialog.set_run(run)
                return

        LOGGER.info(
            "Param search load started: run_id=%s force=%s show_dialog=%s focus_search=%s current=%s loading=%s",
            target_run_id,
            force,
            show_dialog,
            focus_search,
            self._param_search_current_run_id,
            self._param_search_loading_run_id,
        )
        # Cancel any in-flight loader so it stops competing for I/O
        if self._param_search_loader is not None and self._param_search_loader.isRunning():
            old_loader = self._param_search_loader
            self._running_param_search_loaders.add(old_loader)
            old_loader.finished.connect(lambda l=old_loader: self._running_param_search_loaders.discard(l))
            LOGGER.debug(
                "Interrupting ParamSearchLoader: new_run_id=%s retained_loaders=%s request_id=%s",
                target_run_id,
                len(self._running_param_search_loaders),
                self._param_search_request_id,
            )
            old_loader.requestInterruption()
            # Its request id is now stale, so nothing else will decrement this.
            self._set_activity_idle()
        dialog.set_loading_state(run)
        self._set_activity_busy("Loading params…")
        self._param_search_request_id += 1
        current_request_id = self._param_search_request_id
        self._param_search_loading_run_id = target_run_id
        self._param_search_loader = ParamSearchLoader(run.filepath)
        self._param_search_loader.partial_records_ready.connect(
            lambda mode, records, req_id=current_request_id: self._on_param_search_partial_records_guarded(mode, records, req_id)
        )
        self._param_search_loader.records_ready.connect(
            lambda records, rid=run.run_id, req_id=current_request_id: self._on_param_search_records_ready_guarded(rid, records, req_id)
        )
        self._param_search_loader.load_error.connect(
            lambda message, req_id=current_request_id: self._on_param_search_error_guarded(message, req_id)
        )
        self._param_search_loader.start()

    def _open_param_search_for_selected(self):
        row = self._get_selected_row()
        run = self._get_run_for_row(row)
        if run is None:
            self.status_label.setText("No run selected")
            return

        selected_rows = self._get_selected_rows()
        if len(selected_rows) > 1:
            self.status_label.setText("Select a single run for param search")
            return

        self._param_sync_timer.stop()
        self._pending_param_sync_run_id = None
        self._load_param_search_for_run(run, show_dialog=True, focus_search=True, force=False)

    def _on_param_search_records_ready(self, run_id: int, records: dict):
        LOGGER.info("Param search load completed: run_id=%s", run_id)
        dialog = self._ensure_param_search_dialog()
        selected_run = self._runs_by_id.get(run_id)
        if selected_run is not None:
            dialog.set_run(selected_run)
        dialog.set_records(records)
        self._param_search_current_run_id = int(run_id)
        self._param_search_loading_run_id = None
        self.settings.setValue("paramSearchMode", dialog._active_mode)
        self.status_label.setText(f"Loaded params for run {run_id}")
        self._set_activity_idle()

    def _on_param_search_partial_records_guarded(self, mode: str, records: list, request_id: int):
        if request_id != self._param_search_request_id:
            LOGGER.debug(
                "Dropping stale param partial records: mode=%s request_id=%s active_request_id=%s records=%s",
                mode,
                request_id,
                self._param_search_request_id,
                len(records),
            )
            return
        dialog = self._ensure_param_search_dialog()
        dialog.append_records(mode, records)

    def _on_param_search_records_ready_guarded(self, run_id: int, records: dict, request_id: int):
        if request_id != self._param_search_request_id:
            LOGGER.debug(
                "Dropping stale param records: run_id=%s request_id=%s active_request_id=%s modes=%s",
                run_id,
                request_id,
                self._param_search_request_id,
                {mode: len(values) for mode, values in records.items()},
            )
            return
        self._on_param_search_records_ready(run_id, records)

    def _on_param_search_error_guarded(self, message: str, request_id: int):
        if request_id != self._param_search_request_id:
            LOGGER.debug(
                "Dropping stale param error: request_id=%s active_request_id=%s message=%s",
                request_id,
                self._param_search_request_id,
                message,
            )
            return
        LOGGER.error("Param search load failed: %s", message)
        dialog = self._ensure_param_search_dialog()
        dialog.set_error_message(message)
        self._param_search_loading_run_id = None
        self.status_label.setText("Param search failed")
        self._set_activity_idle()

    def _start_lite_creation_for_runs(self, run_ids: list[int]):
        unique_run_ids = sorted({int(rid) for rid in run_ids})
        if not unique_run_ids:
            return

        LOGGER.info("Lite creation requested: run_ids=%s", unique_run_ids)

        if self._lite_worker is not None and self._lite_worker.isRunning():
            QMessageBox.information(self, "Busy", "A lite creation job is already running.")
            return

        # The ROI is chosen here, on the GUI thread, before any background work
        # starts. The worker only ever gets explicit bounds: opening the ROI
        # dialog from the worker thread is what used to freeze the browser.
        roi = self._pick_lite_roi(unique_run_ids)
        if roi is None:
            self.status_label.setText("Lite creation cancelled (no ROI selected)")
            return
        roix, roiy = roi

        runs = []
        for run_id in unique_run_ids:
            run = self._runs_by_id.get(run_id)
            runs.append((run_id, run.filepath if run is not None else ""))

        LOGGER.info("Lite creation: runs=%s roix=%s roiy=%s", unique_run_ids, roix, roiy)
        self._lite_worker = LiteCreateWorker(self.data_dir, runs, roix, roiy)
        self._lite_worker.created.connect(self._on_lite_created)
        self._lite_worker.progress.connect(self._on_lite_progress)
        self._lite_worker.completed.connect(self._on_lite_batch_completed)
        self._lite_worker.error.connect(self._on_lite_error)
        self.lite_cancel_btn.setEnabled(True)
        self.lite_cancel_btn.setVisible(True)
        self._set_activity_busy(f"Creating lite datasets for {len(runs)} run(s)...")
        self._lite_worker.start()

    def _pick_lite_roi(self, run_ids: list[int]):
        """Opens the ROI dialog on the oldest selected run, pre-drawn with its
        saved ROI when it has one. Returns (roix, roiy) or None if closed."""
        from waxa.roi import pick_roi, read_saved_roi

        oldest = min(run_ids)
        run = self._runs_by_id.get(oldest)
        file_path = run.filepath if run is not None else None
        suffix = f" (applied to all {len(run_ids)} runs)" if len(run_ids) > 1 else ""
        self.status_label.setText(f"Select ROI on run {oldest}{suffix}...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        cursor_restored = False
        try:
            talk = server_talk(data_dir=self.data_dir)
            if file_path is None:
                file_path, _ = talk.get_data_file(oldest)
            initial_roi = read_saved_roi(file_path)
            # The dialog decodes its first OD while opening; the busy cursor
            # covers that read and is dropped once the window is up.
            QTimer.singleShot(0, QApplication.restoreOverrideCursor)
            cursor_restored = True
            return pick_roi(
                oldest,
                server_talk=talk,
                file_path=file_path,
                initial_roi=initial_roi,
                suggest_auto_roi=True,
                parent=self,
            )
        except Exception as exc:
            LOGGER.exception("ROI selection for lite creation failed")
            QMessageBox.warning(self, "Lite Creation Error", f"Could not open the ROI selector for run {oldest}:\n{exc}")
            return None
        finally:
            if not cursor_restored:
                QApplication.restoreOverrideCursor()

    def _cancel_lite_creation(self):
        if self._lite_worker is not None and self._lite_worker.isRunning():
            self._lite_worker.cancel()
            self.lite_cancel_btn.setEnabled(False)
            self.status_label.setText("Cancelling lite creation...")

    def _on_lite_progress(self, done: int, total: int, run_id: int):
        if run_id >= 0:
            self.status_label.setText(f"Creating lite datasets: {done}/{total} done, writing run {run_id}...")

    def _on_lite_batch_completed(self, created_count: int, total_count: int, cancelled: bool):
        LOGGER.info(
            "Lite creation finished: created=%s total=%s cancelled=%s", created_count, total_count, cancelled
        )
        self.lite_cancel_btn.setVisible(False)
        word = "cancelled" if cancelled else "complete"
        self._set_activity_idle()
        self.status_label.setText(f"Lite creation {word}: {created_count}/{total_count} runs")

    def _on_lite_created(self, run_id: int, lite_path: str):
        LOGGER.info("Lite created: run_id=%s path=%s", run_id, lite_path)
        row = self._find_row_for_run_id(run_id)
        run = self._runs_by_id.get(run_id)
        if row is None or run is None:
            self.status_label.setText(f"Lite created for run {run_id}")
            return

        run.has_lite = True

        lite_item = self.table.item(row, self.COL_LITE)
        if lite_item is None:
            lite_item = QTableWidgetItem("lite")
            self.table.setItem(row, self.COL_LITE, lite_item)
        else:
            lite_item.setText("lite")

        self._set_lite_highlight(row, True)
        self._style_row_items(row, run)
        self.status_label.setText(f"Lite created: {lite_path}")

    def _on_lite_error(self, message: str):
        # The worker follows every error with `completed`, which resets the
        # busy indicator and the cancel button.
        LOGGER.error("Lite creation failed: %s", message)
        QMessageBox.warning(self, "Lite Creation Error", message)

    def closeEvent(self, event):
        # A QThread destroyed while running aborts the process; stop the lite
        # job between frames (it discards the partial file) and let it finish.
        worker = self._lite_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(10000)
        super().closeEvent(event)

    def _get_run_for_row(self, row: int):
        if row < 0:
            return None
        run_id_item = self.table.item(row, self.COL_RUN_ID)
        if run_id_item is None:
            return None
        run_id = run_id_item.data(Qt.ItemDataRole.UserRole)
        if run_id is None:
            try:
                run_id = int(run_id_item.text())
            except ValueError:
                return None
        return self._runs_by_id.get(int(run_id))

    def _find_row_for_run_id(self, run_id: int):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_RUN_ID)
            if item is None:
                continue
            item_run_id = item.data(Qt.ItemDataRole.UserRole)
            if item_run_id is None:
                try:
                    item_run_id = int(item.text())
                except ValueError:
                    continue
            if int(item_run_id) == int(run_id):
                return row
        return None

    def _row_by_run_id_map(self):
        """run_id -> row for the table's current (possibly sorted) order.
        Use when several lookups are needed at once."""
        mapping = {}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.COL_RUN_ID)
            if item is None:
                continue
            item_run_id = item.data(Qt.ItemDataRole.UserRole)
            if item_run_id is None:
                try:
                    item_run_id = int(item.text())
                except ValueError:
                    continue
            mapping[int(item_run_id)] = row
        return mapping

    def _open_file_in_default_program(self, path_text: str, label: str = "file"):
        clean = (path_text or "").strip()
        if not clean or clean == "-":
            self.status_label.setText(f"No {label.lower()} available.")
            return

        expanded = os.path.expandvars(clean)
        path = os.path.normpath(expanded)
        if not os.path.isfile(path):
            self.status_label.setText(f"{label} does not exist for this path.")
            return

        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        if opened:
            self.status_label.setText(f"Opened {label.lower()} in default program.")
        else:
            self.status_label.setText(f"Could not open {label.lower()} in default program.")


# ---------------------------------------------------------------------------
# Options helpers
# ---------------------------------------------------------------------------

    def _initialize_options_menu(self):
        self.options_menu.clear()

        change_dir_action = QAction("Change Data Directory…", self.options_menu)
        change_dir_action.triggered.connect(self._change_data_dir)
        self.options_menu.addAction(change_dir_action)

        hotkeys_action = QAction("Hotkeys Guide…", self.options_menu)
        hotkeys_action.triggered.connect(self._show_hotkeys_guide)
        self.options_menu.addAction(hotkeys_action)

        edit_common_tags_action = QAction("Edit Common Tags…", self.options_menu)
        edit_common_tags_action.triggered.connect(self._edit_common_tags)
        self.options_menu.addAction(edit_common_tags_action)

        auto_refresh_interval_action = QAction("Set Auto-Refresh Interval…", self.options_menu)
        auto_refresh_interval_action.triggered.connect(self._set_auto_refresh_interval)
        self.options_menu.addAction(auto_refresh_interval_action)

        scroll_lines_action = QAction("Set Scroll Lines…", self.options_menu)
        scroll_lines_action.triggered.connect(self._set_scroll_lines)
        self.options_menu.addAction(scroll_lines_action)

        self.options_menu.addSeparator()

        self._dark_mode_action = QAction("Dark Mode", self.options_menu)
        self._dark_mode_action.setCheckable(True)
        self._dark_mode_action.setChecked(bool(self.settings.value("darkMode", False, type=bool)))
        self._dark_mode_action.toggled.connect(self._on_dark_mode_toggled)
        self.options_menu.addAction(self._dark_mode_action)

        self.options_menu.addSeparator()

        time_menu = self.options_menu.addMenu("Time Format")
        self._time_24h_action = QAction("24-hour (military)", time_menu)
        self._time_24h_action.setCheckable(True)
        self._time_ampm_action = QAction("12-hour (AM/PM)", time_menu)
        self._time_ampm_action.setCheckable(True)
        time_menu.addAction(self._time_24h_action)
        time_menu.addAction(self._time_ampm_action)

        use_ampm = bool(self.settings.value("timeFormatAMPM", True, type=bool))
        self._time_24h_action.setChecked(not use_ampm)
        self._time_ampm_action.setChecked(use_ampm)
        self._time_24h_action.triggered.connect(lambda: self._set_time_format(False))
        self._time_ampm_action.triggered.connect(lambda: self._set_time_format(True))

        # Apply dark mode if previously enabled
        if self._dark_mode_action.isChecked():
            self._apply_dark_mode(True)

    def _set_auto_refresh_interval(self):
        current_value = self._auto_refresh_interval_seconds()
        value, ok = QInputDialog.getInt(
            self,
            "Auto-Refresh Interval",
            "Seconds:",
            current_value,
            1,
            3600,
            1,
        )
        if not ok:
            return

        self.settings.setValue("autoRefreshIntervalSec", int(value))
        self._sync_auto_refresh_timer(show_status=False)
        if self.auto_refresh_btn.isChecked():
            self.status_label.setText(f"Auto-refresh interval set to {int(value)}s")
        else:
            self.status_label.setText(f"Auto-refresh interval saved ({int(value)}s)")

    def _wheel_scroll_lines(self):
        value = int(
            self.settings.value(
                "wheelScrollLines",
                FixedLineScrollTable.DEFAULT_LINES_PER_WHEEL_NOTCH,
                type=int,
            )
        )
        return max(1, value)

    def _set_scroll_lines(self):
        current_value = self._wheel_scroll_lines()
        value, ok = QInputDialog.getInt(
            self,
            "Scroll Lines",
            "Rows per mouse wheel notch:",
            current_value,
            1,
            50,
            1,
        )
        if not ok:
            return

        self.settings.setValue("wheelScrollLines", int(value))
        self.table.set_lines_per_wheel_notch(int(value))
        self.status_label.setText(f"Scroll set to {int(value)} rows per wheel notch")

    def _change_data_dir(self):
        new_dir = QFileDialog.getExistingDirectory(
            self, "Select Data Directory", self.data_dir or os.path.expanduser("~")
        )
        if new_dir and os.path.isdir(new_dir):
            self.data_dir = new_dir
            self._server_talk = server_talk(data_dir=self.data_dir)
            self.settings.setValue("dataDir", new_dir)
            self.status_label.setText(f"Data dir: {new_dir}")
            self._start_scan()

    def _on_dark_mode_toggled(self, checked: bool):
        self.settings.setValue("darkMode", checked)
        self._apply_dark_mode(checked)

    def _apply_dark_mode(self, dark: bool):
        if dark:
            extra = """
            QMainWindow, QWidget { background: #1e2832; color: #d0dce6; }
            QLabel { color: #d0dce6; }
            QFrame#statChip, QFrame#toolbarGroup {
                background: #263340; border-color: #3a4f60;
            }
            QLineEdit, QDateEdit, QPushButton, QListWidget, QTableWidget, QTextEdit {
                background: #263340; border-color: #3a4f60; color: #d0dce6;
            }
            QPushButton:checked, QToolButton:checked { background: #40627a; }
            QGroupBox { background: #263340; border-color: #3a4f60; color: #9bbdd0; }
            QHeaderView::section { background: #1e2832; color: #9bbdd0; border-right: 1px solid #3a4f60; border-bottom: 1px solid #3a4f60; }
            QTableWidget { alternate-background-color: #222d38; selection-background-color: #2d5470; color: #d0dce6; }
            QMenu { background: #263340; color: #d0dce6; }
            QMenu::item:disabled { color: #6f7f8b; }
            QDialog { background: #1e2832; color: #d0dce6; }
            QDialogButtonBox QPushButton { background: #3a4f60; color: #d0dce6; }
            QLabel#fileViewerPathLabel { color: #9bc4dc; }
            QToolButton#autoRefreshBtn {
                background: #2a3946;
                color: #cde1ee;
                border: 1px solid #4f6779;
            }
            QToolButton#autoRefreshBtn:hover {
                background: #324556;
                border-color: #6a8396;
            }
            QToolButton#autoRefreshBtn:checked {
                background: #2f8f5b;
                color: #ffffff;
                border: 1px solid #46ab74;
            }
            """
            self.setStyleSheet(self.styleSheet() + extra)
        else:
            self._apply_styles()

    def _set_time_format(self, use_ampm: bool):
        self.settings.setValue("timeFormatAMPM", use_ampm)
        self._time_24h_action.setChecked(not use_ampm)
        self._time_ampm_action.setChecked(use_ampm)
        # Reformat visible datetime cells
        for row in range(self.table.rowCount()):
            run = self._get_run_for_row(row)
            if run is None:
                continue
            item = self.table.item(row, self.COL_DATETIME)
            if item is not None:
                item.setText(self._format_datetime_for_table(run.run_datetime_str))

# ---------------------------------------------------------------------------
# ROI selection (right-click action)
# ---------------------------------------------------------------------------

    def _open_roi_select_for_run(self, run: RunSummary):
        """Open ROI selector once and save ROI to the run's h5 file."""
        self.status_label.setText(f"Opening ROI selector for run {run.run_id}…")
        self._set_activity_busy("Selecting ROI…")

        class _ROIWorker(QThread if False else __import__("PyQt6.QtCore", fromlist=["QThread"]).QThread):
            done = pyqtSignal(str)
            error = pyqtSignal(str)

            def __init__(self, run_id, filepath):
                super().__init__()
                self.run_id = run_id
                self.filepath = filepath

            def run(self):
                try:
                    from waxa.roi import ROI
                    roi = ROI(run_id=self.run_id, use_saved_roi=False, printouts=False)
                    roi.save_roi_h5(printouts=False)
                    self.done.emit(f"ROI saved for run {self.run_id}")
                except Exception as exc:
                    self.error.emit(str(exc))

        worker = _ROIWorker(run.run_id, run.filepath)
        worker.done.connect(lambda msg: (self.status_label.setText(msg), self._set_activity_idle()))
        worker.error.connect(lambda msg: (
            self.status_label.setText("ROI select failed"),
            QMessageBox.warning(self, "ROI Error", msg),
            self._set_activity_idle(),
        ))
        # Keep reference alive
        if not hasattr(self, "_roi_workers"):
            self._roi_workers = []
        self._roi_workers = [w for w in self._roi_workers if w.isRunning()]
        self._roi_workers.append(worker)
        worker.start()

# ---------------------------------------------------------------------------
# Experiment file viewer (right-click submenu)
# ---------------------------------------------------------------------------

    # Preferred ordering for the source-text attrs. `expt_file` (the
    # experiment itself) is the only one guaranteed to be present; every base
    # class module is saved under a `base_class_<module>` attr, and the legacy
    # keys are kept for older files. Any attr not listed here that still looks
    # like source text is appended afterwards so nothing is silently dropped.
    _EXPT_FILE_ATTR_KEYS = [
        "expt_file",
        "params_file",
        "control_file",
        "cooling_file",
        "sequence_file",
        "imaging_file",
    ]

    @staticmethod
    def _decode_attr_text(raw) -> str:
        if raw is None:
            return ""
        # numpy.bytes_ subclasses bytes, so this also covers h5py scalar strings.
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            return raw
        # numpy 0-d array holding a (byte)string.
        item = getattr(raw, "item", None)
        if callable(item) and getattr(raw, "shape", None) == ():
            scalar = raw.item()
            return scalar.decode("utf-8", errors="replace") if isinstance(scalar, bytes) else str(scalar)
        return str(raw)

    @staticmethod
    def _expt_file_label(attr_key: str) -> str:
        if attr_key.startswith("base_class_"):
            name = attr_key[len("base_class_"):]
        elif attr_key.endswith("_file"):
            name = attr_key[:-len("_file")]
        else:
            name = attr_key
        return name.replace("_", " ").title()

    @staticmethod
    def _expt_file_basename(attr_key: str) -> str:
        if attr_key.startswith("base_class_"):
            return f"{attr_key[len('base_class_'):]}.py"
        if attr_key.endswith("_file"):
            return f"{attr_key[:-len('_file')]}.py"
        return f"{attr_key}.py"

    def _build_view_files_submenu(self, menu: QMenu, run: RunSummary):
        """Add a lazily-populated submenu with one action per experiment source
        file stored in the HDF5 attrs (expt/params plus every base-class module).

        The HDF5 file is only opened when the submenu is actually hovered, so a
        right-click no longer pays a network round-trip before the context menu
        can appear.
        """
        submenu = menu.addMenu("View Experiment Files")
        placeholder = submenu.addAction("Loading…")
        placeholder.setEnabled(False)
        state = {"populated": False}

        def populate():
            if state["populated"]:
                return
            state["populated"] = True
            submenu.clear()
            try:
                import h5py
                with h5py.File(run.filepath, "r") as f:
                    attr_keys = set(f.attrs.keys())
                    # expt/params/legacy keys first (in preferred order), then every
                    # remaining base-class module alphabetically.
                    ordered_keys = [k for k in self._EXPT_FILE_ATTR_KEYS if k in attr_keys]
                    ordered_keys += sorted(
                        k for k in attr_keys
                        if k.startswith("base_class_") and k not in ordered_keys
                    )

                    found = False
                    for key in ordered_keys:
                        text = self._decode_attr_text(f.attrs.get(key))
                        if not text.strip():
                            continue
                        label = self._expt_file_label(key)
                        action = submenu.addAction(f"  {label}")
                        action.triggered.connect(
                            lambda checked=False, k=key, t=text, rid=run.run_id: self._open_file_viewer(rid, k, t)
                        )
                        found = True
                    if not found:
                        submenu.addAction("(no files in attributes)").setEnabled(False)
            except Exception as exc:
                submenu.addAction(f"Error: {exc}").setEnabled(False)

        submenu.aboutToShow.connect(populate)
        return submenu

    def _open_file_viewer(self, run_id: int, attr_key: str, content: str):
        """Open a resizable popup showing stored experiment source text."""
        win = QDialog(self)
        basename = self._expt_file_basename(attr_key)
        win.setWindowTitle(f"Run ID {run_id} - {basename}")
        win.setSizeGripEnabled(True)
        win.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, True)
        win.resize(600, 400)
        win.setMinimumSize(480, 300)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        layout = QVBoxLayout(win)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        editor = QPlainTextEdit(win)
        editor.setReadOnly(True)
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        editor.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        editor.setFont(font)
        dark_mode = bool(self.settings.value("darkMode", False, type=bool))
        if dark_mode:
            editor.setStyleSheet(
                "QPlainTextEdit { background: #1b2530; color: #d7e3ec; border: 1px solid #3a4f60; }"
            )
        else:
            editor.setStyleSheet(
                "QPlainTextEdit { background: #ffffff; color: #1f2a33; border: 1px solid #d5dde4; }"
            )

        # The source text is stored directly in the HDF5 attrs at save time.
        editor.setPlainText(content if content else "(empty)")

        layout.addWidget(editor)

        win.show()
        # Keep reference so the window isn't garbage-collected
        if not hasattr(self, "_file_viewer_windows"):
            self._file_viewer_windows = []
        self._file_viewer_windows.append(win)
        win.finished.connect(lambda: self._file_viewer_windows.remove(win) if win in self._file_viewer_windows else None)


def launch(data_dir: str):
    _configure_browser_logging()
    app = QApplication.instance()
    app_created = app is None
    if app_created:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("weldlab.kexp.gui.data_browser")
        app = QApplication(sys.argv)

    window = DataBrowserWindow(data_dir)
    window.show()

    if app_created:
        app.exec()

    return window


if __name__ == "__main__":
    launch(os.getenv("data", ""))
