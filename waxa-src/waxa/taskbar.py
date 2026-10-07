"""One Windows taskbar button for the consoles of every lab GUI.

Windows groups taskbar buttons by AppUserModelID.  Each GUI sets its own
(``SetCurrentProcessExplicitAppUserModelID``), so it gets its own button; the
console window it was launched from belongs to conhost.exe and would otherwise
sit loose on the taskbar.  :func:`group_console` tags that console window with
the shared :data:`CONSOLE_APP_ID`, so the terminals of all GUIs stack under one
button and the GUIs keep theirs.

Best effort and safe to call more than once: off Windows, without pywin32, or
without a console (pythonw) nothing is changed and it returns False.  The tag
stays when the GUI exits, so a launcher's ``pause`` keeps its console in the
group.

Lives in waxa (no dependencies) so waxx, kexp and the data browser can all use
it; beacon imports it optionally.
"""

from __future__ import annotations

import ctypes
import logging
import sys

CONSOLE_APP_ID = "wax.consoles"

_LOG = logging.getLogger("waxa.taskbar")


def set_hwnd_app_id(hwnd: int, app_id: str | None) -> bool:
    """Set (``app_id``) or clear (``None``) one window's AppUserModelID."""
    if not sys.platform.startswith("win") or not hwnd:
        return False
    try:
        import pythoncom  # noqa: PLC0415
        from win32com.propsys import propsys, pscon  # noqa: PLC0415

        store = propsys.SHGetPropertyStoreForWindow(int(hwnd), propsys.IID_IPropertyStore)
        if app_id is None:
            value = propsys.PROPVARIANTType(None, pythoncom.VT_EMPTY)
        else:
            value = propsys.PROPVARIANTType(app_id, pythoncom.VT_LPWSTR)
        store.SetValue(pscon.PKEY_AppUserModel_ID, value)
        store.Commit()
        return True
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("set_hwnd_app_id(%r, %r) failed: %r", hwnd, app_id, exc)
        return False


def set_terminal_icon(hwnd: int) -> bool:
    """Give a console window cmd.exe's icon (title bar and taskbar).

    A console opened by python.exe itself (no .bat in front, e.g. under the
    venv's pythonw launcher) otherwise shows the Python icon.
    """
    if not sys.platform.startswith("win") or not hwnd:
        return False
    try:
        import os  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        cmd = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "cmd.exe")
        large, small = wintypes.HICON(), wintypes.HICON()
        shell32, user32 = ctypes.windll.shell32, ctypes.windll.user32
        shell32.ExtractIconExW.argtypes = [wintypes.LPCWSTR, ctypes.c_int,
                                           ctypes.POINTER(wintypes.HICON),
                                           ctypes.POINTER(wintypes.HICON), wintypes.UINT]
        if shell32.ExtractIconExW(cmd, 0, ctypes.byref(large), ctypes.byref(small), 1) == 0:
            return False
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.SendMessageW.restype = wintypes.LPARAM
        WM_SETICON, ICON_SMALL, ICON_BIG = 0x0080, 0, 1
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, large.value or 0)
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small.value or large.value or 0)
        return True     # the icons live as long as the console; never destroyed here
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("set_terminal_icon(%r) failed: %r", hwnd, exc)
        return False


def group_console(app_id: str = CONSOLE_APP_ID, title: str | None = None) -> bool:
    """Put this process's console window on the shared consoles taskbar button.

    The console also gets the standard terminal (cmd.exe) icon, so the group's
    button looks the same whichever console opened first, and ``title`` (if
    given) as its window title.
    """
    if not sys.platform.startswith("win"):
        return False
    try:
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if title is not None:
            ctypes.windll.kernel32.SetConsoleTitleW(title)
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("group_console: no console: %r", exc)
        return False
    set_terminal_icon(hwnd)
    return set_hwnd_app_id(hwnd, app_id)
