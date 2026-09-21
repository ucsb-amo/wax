"""The log panel at the top of the liveOD viewer.

Timestamped, coloured by level, filterable, collapsible to its last line (by its
arrow only; the viewer's splitter never shrinks it to nothing), and with a banner that keeps the most recent error in view until it is dismissed.

It stands where a bare ``QPlainTextEdit`` used to, and still answers
``appendPlainText`` so existing callers (camera buttons, the remote viewer) work
unchanged.
"""

import html
import logging
import time
from collections import deque

from PyQt6.QtCore import Qt, QEvent, pyqtSignal
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QPlainTextEdit,
                             QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget)
from waxx.util.live_od.gui import theme

MAX_RECORDS = 5000

_FILTERS = (("All", logging.DEBUG), ("Info", logging.INFO),
            ("Warnings", logging.WARNING), ("Errors", logging.ERROR))


def _level_style(levelno: int) -> str:
    """Inline CSS for a record of this level, in the current theme's colours."""
    if levelno >= logging.ERROR:
        color = theme.color("error")
    elif levelno >= logging.WARNING:
        color = theme.color("warning")
    elif levelno >= logging.INFO:
        color = ""              # the widget's own text colour, whatever the theme
    else:
        color = theme.color("timestamp")
    style = f"color:{color};" if color else ""
    if levelno >= logging.ERROR:
        style += "font-weight:bold;"
    return style


def guess_level(text: str) -> int:
    """Level for a message that arrived as plain text (``appendPlainText``)."""
    low = text.lower()
    if any(word in low for word in ("error", "fail", "unusable")):
        return logging.ERROR
    if any(word in low for word in ("warning", "abort", "dishonorabl", "disconnected", "not found")):
        return logging.WARNING
    return logging.INFO


class ErrorBanner(QFrame):
    """The latest error, pinned until dismissed."""

    dismissed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._count = 0
        self._text = ""
        self._restyle()
        theme.notifier().changed.connect(self._restyle)
        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(lambda: QGuiApplication.clipboard().setText(self._text))
        dismiss_button = QPushButton("Dismiss")
        dismiss_button.clicked.connect(self.dismiss)
        layout = QHBoxLayout()
        layout.setContentsMargins(6, 3, 6, 3)
        layout.addWidget(self._label, 1)
        layout.addWidget(copy_button, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(dismiss_button, 0, Qt.AlignmentFlag.AlignTop)
        self.setLayout(layout)
        self.hide()

    def _restyle(self, *_):
        self.setStyleSheet(f"ErrorBanner {{ background-color: {theme.color('banner_bg')}; "
                           f"border: 1px solid {theme.color('banner_border')}; }}")

    def show_error(self, text: str, created: float):
        self._count += 1
        self._text = text
        stamp = time.strftime("%H:%M:%S", time.localtime(created))
        more = f"  (+{self._count - 1} earlier)" if self._count > 1 else ""
        self._label.setText(f"<b>{stamp}</b>{more}<br>" + html.escape(text).replace("\n", "<br>"))
        self.show()

    def dismiss(self):
        self._count = 0
        self._text = ""
        self.hide()
        self.dismissed.emit()


class LogPanel(QWidget):
    error_logged = pyqtSignal(str)      # an ERROR-or-worse record was appended

    def __init__(self):
        super().__init__()
        self._records = deque(maxlen=MAX_RECORDS)   # (levelno, text, created)
        self._min_level = logging.DEBUG
        self._collapsed = False

        self._toggle = QToolButton()
        self._toggle.setArrowType(Qt.ArrowType.DownArrow)
        self._toggle.setAutoRaise(True)
        self._toggle.setToolTip("Collapse the log to its last line")
        self._toggle.clicked.connect(lambda: self.set_collapsed(not self._collapsed))

        self._last_line = QLabel()
        self._last_line.setTextFormat(Qt.TextFormat.RichText)
        self._last_line.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        self._filter = QComboBox()
        for label, _ in _FILTERS:
            self._filter.addItem(label)
        self._filter.setToolTip("Lowest level shown")
        self._filter.currentIndexChanged.connect(self._on_filter_changed)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._toggle)
        header.addWidget(QLabel("Log"))
        header.addWidget(self._last_line, 1)
        header.addWidget(self._filter)
        self._header = QWidget()
        self._header.setLayout(header)

        self.banner = ErrorBanner()

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(MAX_RECORDS)
        self._text.setMinimumHeight(40)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self._text.setFont(font)

        # Copy and Clear float over the text's top-right corner, so they go away
        # with it when the log is collapsed. Children of the text box, not of its
        # viewport: the viewport scrolls its children along with the text.
        self._copy_button = QToolButton(self._text)
        self._copy_button.setText("Copy")
        self._copy_button.setToolTip("Copy the whole log to the clipboard")
        self._copy_button.clicked.connect(self.copy_all)
        self._clear_button = QToolButton(self._text)
        self._clear_button.setText("Clear")
        self._clear_button.setToolTip("Clear the log")
        self._clear_button.clicked.connect(self.clear)
        self._text.installEventFilter(self)
        self._text.viewport().installEventFilter(self)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._header)
        layout.addWidget(self.banner)
        layout.addWidget(self._text, 1)
        self.setLayout(layout)
        self._restyle_overlay_buttons()
        # level colours differ between the themes: redraw what is already shown
        theme.notifier().changed.connect(lambda _dark: self._on_filter_changed(self._filter.currentIndex()))
        theme.notifier().changed.connect(self._restyle_overlay_buttons)

    def _restyle_overlay_buttons(self, *_):
        # a plate that reads over any line of the log, in either theme
        style = (f"QToolButton {{ background-color: palette(button); "
                 f"border: 1px solid {theme.color('separator')}; border-radius: 3px; "
                 f"padding: 0px 5px; }} "
                 f"QToolButton:hover {{ border-color: {theme.color('muted')}; }}")
        for button in (self._copy_button, self._clear_button):
            button.setStyleSheet(style)
            button.adjustSize()
        self._place_overlay_buttons()

    def _place_overlay_buttons(self):
        """Top right of the text, left of its scroll bar."""
        viewport = self._text.viewport().geometry()
        margin = 4
        x = viewport.right() - margin
        for button in (self._clear_button, self._copy_button):
            x -= button.width()
            button.move(x, viewport.top() + margin)
            button.raise_()
            x -= 3

    def eventFilter(self, watched, event):
        # the viewport too: it narrows when the scroll bar appears
        if (watched in (self._text, self._text.viewport())
                and event.type() in (QEvent.Type.Resize, QEvent.Type.Show)):
            self._place_overlay_buttons()
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------
    # Appending
    # ------------------------------------------------------------------

    def append_record(self, levelno: int, text: str, created: float = None):
        """Slot for ``QtLogHandler.record_signal``."""
        if created is None:
            created = time.time()
        self._records.append((levelno, text, created))
        if levelno >= self._min_level:
            self._text.appendHtml(self._to_html(levelno, text, created))
        first_line = text.split("\n", 1)[0]
        self._last_line.setText(
            f'<span style="{_level_style(levelno)}">{html.escape(first_line)}</span>')
        if levelno >= logging.ERROR:
            self.banner.show_error(text, created)
            self.error_logged.emit(text)

    def append_message(self, text: str, levelno: int = logging.INFO):
        self.append_record(levelno, text)

    def append_separator(self, title: str):
        """A rule across the log marking the start of a run."""
        self.append_record(logging.INFO, f"──────── {title} ────────")

    # QPlainTextEdit's interface, for the callers that were handed one
    def appendPlainText(self, text: str):
        self.append_record(guess_level(str(text)), str(text))

    def setReadOnly(self, _read_only: bool):
        pass

    def toPlainText(self) -> str:
        return "\n".join(text for _, text, _ in self._records)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def set_collapsed(self, collapsed: bool):
        self._collapsed = bool(collapsed)
        self._text.setVisible(not self._collapsed)
        self._toggle.setArrowType(Qt.ArrowType.RightArrow if self._collapsed
                                  else Qt.ArrowType.DownArrow)
        if self._collapsed:
            self.setMaximumHeight(self._header.sizeHint().height()
                                  + (self.banner.sizeHint().height() if self.banner.isVisible() else 0)
                                  + 8)
        else:
            self.setMaximumHeight(16777215)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def copy_all(self):
        lines = [f"{time.strftime('%H:%M:%S', time.localtime(created))} "
                 f"{logging.getLevelName(levelno):<7} {text}"
                 for levelno, text, created in self._records]
        QGuiApplication.clipboard().setText("\n".join(lines))

    def clear(self):
        self._records.clear()
        self._text.clear()
        self._last_line.clear()

    def _on_filter_changed(self, index: int):
        self._min_level = _FILTERS[index][1]
        self._text.clear()
        for levelno, text, created in self._records:
            if levelno >= self._min_level:
                self._text.appendHtml(self._to_html(levelno, text, created))

    @staticmethod
    def _to_html(levelno: int, text: str, created: float) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(created))
        body = html.escape(text).replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")
        return (f'<span style="color:{theme.color("timestamp")}">{stamp}</span> '
                f'<span style="{_level_style(levelno)}">{body}</span>')
