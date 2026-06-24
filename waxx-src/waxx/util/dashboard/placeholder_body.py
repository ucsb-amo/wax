"""Placeholder widget shown while a panel body is initializing.

Part of the transparency-first lifecycle: the dashboard window opens with
all panels showing this placeholder, then real body widgets swap in async
as factories complete.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderBody(QWidget):
    """Compact "Initializing..." widget with an animated dot ticker.

    Usage::

        body = PlaceholderBody("Connecting to ALS server...")
        # ... later when the real body is ready:
        body.replace_with(real_widget)
    """

    def __init__(self, message: str = "Initializing\u2026", parent: QWidget | None = None):
        super().__init__(parent)
        self._base_message = message
        self._tick = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        self._label = QLabel(message, self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "QLabel { color: #777; font-size: 12px; }"
        )
        layout.addStretch(1)
        layout.addWidget(self._label)
        layout.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    def set_message(self, message: str) -> None:
        self._base_message = message
        self._label.setText(message)

    def _on_tick(self) -> None:
        self._tick = (self._tick + 1) % 4
        dots = "." * self._tick
        # Trim any existing trailing ellipsis/dots from the base message so we
        # don't end up with "Connecting...\u2026"  type duplication.
        base = self._base_message.rstrip(".\u2026 ")
        self._label.setText(f"{base}{dots}")

    def stop(self) -> None:
        """Stop the animation. Called automatically when the widget is destroyed."""
        if self._timer.isActive():
            self._timer.stop()

    def closeEvent(self, event):  # noqa: N802 - Qt signature
        self.stop()
        super().closeEvent(event)
