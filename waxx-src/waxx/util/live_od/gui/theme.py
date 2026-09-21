"""Light / dark theme for the liveOD windows.

``apply_theme(dark)`` restyles the whole QApplication (Fusion style with a dark
palette, or the platform's own style back again) and tells the widgets that paint
their own colours -- the log panel, the status strip -- to repaint, through
``notifier().changed``. ``color(name)`` gives those widgets the colour for the
current theme.
"""

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

_COLORS = {
    #                 light       dark
    "muted":        ("#555555", "#a0a0a0"),     # secondary text
    "timestamp":    ("#888888", "#808080"),
    "warning":      ("#b36b00", "#ffb74d"),
    "error":        ("#c00000", "#ff6b6b"),
    "banner_bg":    ("#ffd6d6", "#4a1c1c"),
    "banner_border": ("#c00000", "#ff6b6b"),
    "separator":    ("#c0c0c0", "#555555"),
    "progress":     ("#aebfd0", "#46586a"),     # the progress bar's fill: deliberately dull
}


class _Notifier(QObject):
    changed = pyqtSignal(bool)      # dark?


_notifier = None
_dark = False
_original_style = None
_original_palette = None


def notifier() -> _Notifier:
    global _notifier
    if _notifier is None:
        _notifier = _Notifier()
    return _notifier


def is_dark() -> bool:
    return _dark


def color(name: str) -> str:
    return _COLORS[name][1 if _dark else 0]


def _dark_palette() -> QPalette:
    window, base, text = QColor("#2b2b2b"), QColor("#1e1e1e"), QColor("#e0e0e0")
    button, highlight, disabled = QColor("#3c3f41"), QColor("#2f65ca"), QColor("#777777")
    palette = QPalette()
    for role, value in ((QPalette.ColorRole.Window, window),
                        (QPalette.ColorRole.WindowText, text),
                        (QPalette.ColorRole.Base, base),
                        (QPalette.ColorRole.AlternateBase, window),
                        (QPalette.ColorRole.ToolTipBase, base),
                        (QPalette.ColorRole.ToolTipText, text),
                        (QPalette.ColorRole.Text, text),
                        (QPalette.ColorRole.Button, button),
                        (QPalette.ColorRole.ButtonText, text),
                        (QPalette.ColorRole.BrightText, QColor("#ffffff")),
                        (QPalette.ColorRole.Link, QColor("#6ea8fe")),
                        (QPalette.ColorRole.Highlight, highlight),
                        (QPalette.ColorRole.HighlightedText, QColor("#ffffff")),
                        (QPalette.ColorRole.PlaceholderText, disabled)):
        palette.setColor(role, value)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return palette


def apply_theme(dark: bool):
    """Restyle the running QApplication. Safe to call repeatedly."""
    global _dark, _original_style, _original_palette
    app = QApplication.instance()
    if app is None:
        return
    if _original_style is None:
        _original_style = app.style().objectName()
        _original_palette = QPalette(app.palette())
    _dark = bool(dark)
    if _dark:
        app.setStyle("Fusion")
        app.setPalette(_dark_palette())
        # tooltips take their colours from the style sheet under Fusion, not the palette
        app.setStyleSheet("QToolTip { color: #e0e0e0; background-color: #1e1e1e; "
                          "border: 1px solid #555555; }")
    else:
        app.setStyle(_original_style)
        app.setPalette(_original_palette)
        app.setStyleSheet("")
    notifier().changed.emit(_dark)
