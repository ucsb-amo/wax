import time

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPlainTextEdit
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt


LOG_LEVEL_ORDER = ["all", "normal", "xvar", "debug"]


def classify_log_level(message: str) -> str:
    text = str(message).strip().lower()
    if text.startswith("<< command from") or " command from " in text:
        return "debug"
    if text.startswith("xvars received:"):
        return "xvar"
    return "normal"


class FilteredLogPanel(QWidget):
    """Simple log panel with level filter dropdown and scrollable text area."""

    def __init__(
        self,
        *,
        title: str = "Log",
        show_timestamps: bool = True,
        max_entries: int = 1000,
        font_family: str = "Consolas",
        font_size: int = 9,
    ):
        super().__init__()
        self._show_timestamps = bool(show_timestamps)
        self._max_entries = int(max_entries) if max_entries is not None else None
        self._entries: list[dict] = []

        self.level_label = QLabel("Level")
        self.level_label.setStyleSheet("color: #9db6cc; font-weight: 600;")
        self.level_combo = QComboBox()
        self.level_combo.addItems(LOG_LEVEL_ORDER)
        self.level_combo.setCurrentText("normal")
        self.level_combo.currentTextChanged.connect(self._refresh_visible_text)

        self.text_box = QPlainTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setFont(QFont(font_family, font_size))
        self.text_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        top.addWidget(QLabel(title))
        top.addStretch(1)
        top.addWidget(self.level_label)
        top.addWidget(self.level_combo)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self.text_box)
        self.setLayout(layout)

    def set_placeholder_text(self, text: str):
        self.text_box.setPlaceholderText(str(text))

    def set_max_block_count(self, count: int):
        self.text_box.setMaximumBlockCount(int(count))

    def set_style_sheet(self, style_sheet: str):
        self.text_box.setStyleSheet(style_sheet)

    def clear(self):
        self._entries.clear()
        self.text_box.clear()

    def add_message(self, message: str, *, level: str | None = None, timestamp: float | None = None):
        if level is None:
            level = classify_log_level(message)
        lvl = str(level).lower()
        if lvl not in LOG_LEVEL_ORDER:
            lvl = "normal"
        ts = time.time() if timestamp is None else float(timestamp)
        self._entries.append({"timestamp": ts, "message": str(message), "level": lvl})
        if self._max_entries is not None and len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]
        self._refresh_visible_text()

    def set_entries(self, entries: list[dict]):
        self._entries = []
        for entry in entries:
            self.add_message(
                str(entry.get("message", "")),
                level=str(entry.get("level", "normal")),
                timestamp=float(entry.get("timestamp", 0.0)),
            )
        self._refresh_visible_text()

    def _format_line(self, entry: dict) -> str:
        msg = str(entry.get("message", ""))
        if not self._show_timestamps:
            return msg
        ts = float(entry.get("timestamp", 0.0))
        stamp = time.strftime("%H:%M:%S", time.localtime(ts)) if ts > 0 else "--:--:--"
        return f"[{stamp}] {msg}"

    def _refresh_visible_text(self):
        selected = self.level_combo.currentText().lower()
        visible_levels = {
            "all": {"normal", "xvar", "debug"},
            "normal": {"normal"},
            "xvar": {"normal", "xvar"},
            "debug": {"normal", "xvar", "debug"},
        }.get(selected, {"normal"})
        lines = []
        for entry in self._entries:
            level = str(entry.get("level", "normal")).lower()
            if level not in visible_levels:
                continue
            lines.append(self._format_line(entry))
        self.text_box.setPlainText("\n".join(lines))
        sb = self.text_box.verticalScrollBar()
        sb.setValue(sb.maximum())
