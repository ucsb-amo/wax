"""Single source of colours and Qt style sheets for the dashboard and the GUIs
it embeds.

Every panel, header, tile and status widget used to carry its own colour
literals; the same "running" green existed as three different hex strings
and the Running Servers tiles disagreed with the panel headers.  Import the
tokens from here instead::

    from waxx.util.dashboard import theme
    label.setStyleSheet(f"color: {theme.OK};")
    led.set_color(theme.state_color(SupervisorState.RUNNING))

``app_stylesheet()`` is the one QSS string the dashboard installs on the
``QApplication``; ``apply_dark_theme(app)`` installs the matching palette.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette

# --- base surfaces -----------------------------------------------------------
BG            = "#2d2d2d"   # window background
BG_SUNKEN     = "#1e1e1e"   # text fields, tables
BG_RAISED     = "#2b2b2b"   # panel headers, menus, status bar
BG_BUTTON     = "#373737"
BG_BUTTON_HOVER = "#474747"
BG_ALT        = "#282828"   # alternate table rows
BG_CARD       = "#353535"   # device-control grid cells (one card per channel)

# --- text --------------------------------------------------------------------
FG            = "#dcdcdc"
FG_STRONG     = "#e8e8e8"
FG_MUTED      = "#9a9a9a"
FG_DISABLED   = "#6e6e6e"

# --- lines -------------------------------------------------------------------
BORDER        = "#4a4a4a"   # 1 px panel frame
BORDER_STRONG = "#6a6a6a"   # separators under headers
ACCENT        = "#4da6ff"   # focus / active panel / search highlight
HIGHLIGHT     = "#3c6eb4"   # selection

# --- semantic states ---------------------------------------------------------
OK            = "#2e8b57"   # running / connected / on
WARN          = "#d4a017"   # starting / stopping / loading
ERR           = "#b22222"   # crashed / failed / error
OFF           = "#777777"   # idle / disconnected / unknown
EXTERNAL      = "#2e8b57"   # running, but not ours
PENDING       = "#e08a1e"   # unsaved / staged edits (orange)
UNDO          = PENDING

# --- state maps ------------------------------------------------------------
# Keyed by SupervisorState *name* so this module stays import-light and the
# mapping works for enum members, their names, or lowercase strings.
_SUPERVISOR_STATE_COLOR = {
    "IDLE": OFF,
    "STARTING": WARN,
    "RUNNING": OK,
    "STOPPING": WARN,
    "CRASHED": ERR,
    "FAILED": ERR,
    "EXTERNAL": EXTERNAL,
}

_CONN_COLOR = {
    "connected": OK,
    "disconnected": OFF,
    "error": ERR,
    "connecting": WARN,
}


def state_color(state) -> str:
    """Colour for a ``SupervisorState`` (enum, its name, or a lowercase string)."""
    name = getattr(state, "name", None) or str(state)
    return _SUPERVISOR_STATE_COLOR.get(str(name).upper(), OFF)


def conn_color(status: str) -> str:
    return _CONN_COLOR.get(str(status).lower(), ERR)


def pill_css(bg: str, fg: str = "white") -> str:
    """Compact rounded status pill used by the header badges."""
    return (
        f"background: {bg}; color: {fg}; border-radius: 6px;"
        " padding: 1px 6px; font-size: 10px; font-weight: 600;"
    )


def header_button_css() -> str:
    return (
        f"QPushButton {{ color: {FG}; padding: 1px 8px; font-size: 11px;"
        f" border: 1px solid {BORDER}; border-radius: 3px;"
        f" background-color: {BG_BUTTON}; }}"
        f"QPushButton:hover {{ background-color: {BG_BUTTON_HOVER}; }}"
        f"QPushButton:disabled {{ color: {FG_DISABLED}; border-color: #333;"
        f" background-color: #2f2f2f; }}"
    )


def app_stylesheet() -> str:
    """The one QSS string installed on the QApplication."""
    return (
        f"QToolTip {{ color: {FG}; background-color: {BG_RAISED}; border: 1px solid {BORDER_STRONG}; }}"
        # Dock chrome.  Panel bodies/headers draw their own 1 px frame via
        # object-name rules in panel_container / panel_header so it survives
        # tabify / float / restoreState, which drop QDockWidget QSS.
        f" QDockWidget {{ border: 0; }}"
        f" QDockWidget::title {{ background: {BG_RAISED}; color: {FG}; padding: 2px;"
        f" border-bottom: 1px solid {BORDER_STRONG}; }}"
        f" QMainWindow::separator {{ background: {BORDER}; width: 3px; height: 3px; }}"
        f" QMainWindow::separator:hover {{ background: {ACCENT}; }}"
        f" QTabBar::tab {{ background: #353535; color: {FG}; padding: 4px 10px;"
        f" border: 1px solid #262626; border-bottom: 0; }}"
        f" QTabBar::tab:selected {{ background: #454545; color: {FG_STRONG}; }}"
        f" QTabBar::tab:hover {{ background: #3f3f3f; }}"
        f" QMenuBar {{ background: {BG_RAISED}; color: {FG}; }}"
        f" QMenuBar::item:selected {{ background: #444; }}"
        f" QMenu {{ background: {BG_RAISED}; color: {FG}; border: 1px solid {BORDER}; }}"
        f" QMenu::item:selected {{ background: #444; }}"
        f" QStatusBar {{ background: {BG_RAISED}; color: {FG_MUTED}; }}"
        f" QStatusBar::item {{ border: 0; }}"
        f" QToolBar {{ background: {BG_RAISED}; border: 0; spacing: 4px; padding: 2px; }}"
        f" QToolBar QToolButton {{ color: {FG}; padding: 3px 8px; border-radius: 3px; }}"
        f" QToolBar QToolButton:hover {{ background: {BG_BUTTON_HOVER}; }}"
        f" QScrollBar:vertical {{ background: {BG}; width: 10px; margin: 0; }}"
        f" QScrollBar::handle:vertical {{ background: #555; border-radius: 4px; min-height: 24px; }}"
        f" QScrollBar::handle:vertical:hover {{ background: #6a6a6a; }}"
        f" QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
        f" QScrollBar:horizontal {{ background: {BG}; height: 10px; margin: 0; }}"
        f" QScrollBar::handle:horizontal {{ background: #555; border-radius: 4px; min-width: 24px; }}"
        f" QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}"
    )


def apply_dark_theme(app) -> None:
    """Install the Fusion dark palette + :func:`app_stylesheet` on *app*."""
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(FG))
    palette.setColor(QPalette.ColorRole.Base, QColor(BG_SUNKEN))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_ALT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_RAISED))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(FG))
    palette.setColor(QPalette.ColorRole.Text, QColor(FG))
    palette.setColor(QPalette.ColorRole.Button, QColor(BG_BUTTON))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(FG))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    palette.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(HIGHLIGHT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(FG_DISABLED))
    app.setPalette(palette)
    app.setStyleSheet(app_stylesheet())


__all__ = [
    "BG", "BG_SUNKEN", "BG_RAISED", "BG_BUTTON", "BG_BUTTON_HOVER", "BG_ALT", "BG_CARD",
    "FG", "FG_STRONG", "FG_MUTED", "FG_DISABLED",
    "BORDER", "BORDER_STRONG", "ACCENT", "HIGHLIGHT",
    "OK", "WARN", "ERR", "OFF", "EXTERNAL", "PENDING", "UNDO",
    "state_color", "conn_color", "pill_css", "header_button_css",
    "app_stylesheet", "apply_dark_theme",
]
