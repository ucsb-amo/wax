from pathlib import Path
import os
import numpy as np

from subprocess import PIPE, run
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread

class MonitorManager(QThread):
    msg = pyqtSignal(str)
    monitor_stopped = pyqtSignal(str)

    def __init__(self, monitor_expt_path):
        super().__init__()
        self.monitor_expt_path = monitor_expt_path
        
    def run(self):
        self.run_expt()

    def run_expt(self):
        try:
            expt_path = self.monitor_expt_path
            run_expt_command = r"%kpy% & ar " + str(expt_path)
            self.msg.emit("Starting monitor...")
            result = run(run_expt_command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
            combined_output = (result.stdout or "") + (result.stderr or "")
            if "WinError 10054" in combined_output:
                print("Monitor interrupted. Was another experiment submitted?")
            else:
                print(result)
        except Exception as e:
            if "WinError 10054" in str(e):
                print("Monitor interrupted. Was another experiment submitted?")
            else:
                print(e)