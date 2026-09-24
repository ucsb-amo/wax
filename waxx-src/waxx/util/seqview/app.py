"""Process entry and in-process opening for the viewer."""

import faulthandler
import os
import sys
import traceback

from waxx.util.seqview.bundle import Bundle


def _app():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv[:1])
        created = True
    try:
        from waxx.util.live_od.gui import theme
        theme.apply_theme(True)
    except Exception:
        pass
    return app, created


def open_inline(bundle):
    """Build the window in this process (a Qt event loop must be running,
    e.g. ``%gui qt`` in Jupyter). Returns the window."""
    if 'PyQt5' in sys.modules:
        raise RuntimeError("seqview needs PyQt6; PyQt5 is already loaded in "
                           "this process (a Qt5 matplotlib backend?). Use "
                           "the subprocess viewer instead (inline=False).")
    from waxx.util.seqview.window import SeqViewWindow
    app, created = _app()
    win = SeqViewWindow(Bundle.coerce(bundle), serve=False)
    win.show()
    win.raise_window()
    if created:
        # no event loop running: run one (blocks until the window closes)
        app.exec()
    return win


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    faulthandler.enable()
    path = argv[0] if argv else None
    from waxx.util.seqview.window import SeqViewWindow
    app, _created = _app()
    win = SeqViewWindow(None, serve=True)
    if path:
        try:
            win.load_bundle(Bundle.load(path), keep_view=False, path=path)
        except Exception:
            traceback.print_exc()
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(None, 'seqview', f"could not load {path}:\n"
                                 f"{traceback.format_exc()}")
    win.show()
    win.raise_window()
    return app.exec()
