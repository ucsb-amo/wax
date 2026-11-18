"""Child process for test_dashboard_remote_close's process-exit tests.

A real DashboardMainWindow with its control port on 127.0.0.1, NOT advertised
(no beacon), QSettings in an INI under argv[1], no supervisors.  Prints
``PORT <n>`` and runs the app.  argv[2]:

* "modal": a QMessageBox is open (its own nested event loop) when the remote
  close arrives;
* "frozen_log": the GUI thread takes the root log handler's lock and never
  lets go (a GUI frozen inside a log call), for the kill watchdog.
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QSettings, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QMessageBox  # noqa: E402


def main() -> int:
    settings_dir, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
    from waxx.util.dashboard import dashboard_window as dw
    from waxx.util.dashboard.panel_container import ClientPanel
    real = QSettings
    ini = os.path.join(settings_dir, "dash.ini")
    dw.QSettings = lambda *_a, **_k: real(ini, real.Format.IniFormat)
    dw.DashboardMainWindow._probe_data_dir = lambda self: None

    app = QApplication([])
    panels = [dw.PanelPlacement(ClientPanel("alpha", "A", lambda: QLabel("a")), area="right")]
    win = dw.DashboardMainWindow("client", "child dashboard", panels, host_ip="test")
    win.show()
    assert win.enable_remote_control(bind_host="127.0.0.1", advertise=False)
    print(f"PORT {win._control.address[1]}", flush=True)
    if mode == "modal":
        def modal():
            print("MODAL", flush=True)
            QMessageBox.question(win, "left open", "nobody answers this")
            print("MODAL_RETURNED", flush=True)
        QTimer.singleShot(100, modal)
    if mode == "frozen_log":
        import logging
        import time
        handler = logging.StreamHandler(sys.stderr)
        logging.getLogger().addHandler(handler)

        def freeze():
            handler.acquire()
            print("FROZEN", flush=True)
            while True:
                time.sleep(1)
        QTimer.singleShot(100, freeze)
    rc = app.exec()
    print(f"EXIT {rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
