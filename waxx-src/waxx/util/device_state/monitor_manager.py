from pathlib import Path
import os
import threading
import numpy as np

from subprocess import PIPE, Popen
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread

class MonitorManager(QThread):
    msg = pyqtSignal(str)
    monitor_stopped = pyqtSignal(str)

    def __init__(self, monitor_expt_path):
        super().__init__()
        self.monitor_expt_path = monitor_expt_path
        # Child process running the monitor experiment ("ar <expt>").  Owned by
        # this thread so it can be killed from the outside without ever calling
        # QThread.terminate() (which on Windows force-kills the thread mid-Python
        # execution and corrupts the interpreter -> 0xC0000005 access violation).
        self._proc = None
        self._proc_lock = threading.Lock()
        self._stop_requested = False

    def run(self):
        self._stop_requested = False
        self.run_expt()

    def run_expt(self):
        try:
            expt_path = self.monitor_expt_path
            run_expt_command = r"%kpy% & ar " + str(expt_path)
            self.msg.emit("Starting monitor...")
            with self._proc_lock:
                if self._stop_requested:
                    return
                self._proc = Popen(run_expt_command, stdout=PIPE, stderr=PIPE,
                                   universal_newlines=True, shell=True)
            stdout, stderr = self._proc.communicate()
            combined_output = (stdout or "") + (stderr or "")
            if "WinError 10054" in combined_output:
                print("Monitor interrupted. Was another experiment submitted?")
            else:
                print(combined_output)
        except Exception as e:
            if "WinError 10054" in str(e):
                print("Monitor interrupted. Was another experiment submitted?")
            else:
                print(e)
        finally:
            with self._proc_lock:
                self._proc = None

    def stop(self, timeout_ms=1500):
        """Gracefully stop the monitor experiment.

        Kills the spawned child process tree (``ar`` + its descendants) so the
        blocking ``communicate()`` returns and ``run()`` exits on its own.  This
        replaces ``QThread.terminate()``, which force-kills the thread while it
        holds the GIL and crashes the whole dashboard subprocess.
        """
        self._stop_requested = True
        with self._proc_lock:
            proc = self._proc
        if proc is not None:
            pid = proc.pid
            killed = False
            try:
                from waxx.util.dashboard.server_supervisor import _kill_pid_tree  # noqa: PLC0415
                killed = _kill_pid_tree(pid)
            except Exception:
                killed = False
            if not killed:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.wait(timeout_ms)