"""Toolbar buttons that start the lab's tool windows (LiveODConfig.tool_windows).

Each tool runs as a process of its own: nothing it does reaches liveOD's GUI
thread, and closing liveOD does not close it. A second click while the tool is
still running starts no second copy; it says so instead. A tool that exits
within a few seconds of starting is reported with its exit code and the end
of what it printed.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QPushButton

_EARLY_EXIT_CHECK_MS = 4000


class ToolLauncher:
    """``msg(text, level)`` reports to the window's log."""

    def __init__(self, tools, msg=None):
        self._tools = list(tools or [])
        self._msg = msg or (lambda text, level=logging.INFO: print(text))
        self._procs = {}        # label -> (Popen, log path)
        self._buttons = []

    def buttons(self):
        if not self._buttons:
            for tool in self._tools:
                b = QPushButton(tool.label)
                b.setToolTip(tool.tooltip or f"Open {tool.label} (its own window)")
                b.clicked.connect(lambda _c=False, t=tool: self.launch(t))
                self._buttons.append(b)
        return list(self._buttons)

    def launch(self, tool):
        running = self._procs.get(tool.label)
        if running is not None and running[0].poll() is None:
            self._msg(f"{tool.label} is already open (pid {running[0].pid}).")
            return running[0]
        log = os.path.join(tempfile.gettempdir(), f"liveod_tool_{tool.label.replace(' ', '_')}.log")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        try:
            with open(log, "w", encoding="utf-8", errors="replace") as out:
                proc = subprocess.Popen(list(tool.command), stdout=out, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, creationflags=flags)
        except Exception as exc:        # noqa: BLE001 - reported in the window
            self._msg(f"{tool.label}: could not start ({exc})", logging.ERROR)
            return None
        self._procs[tool.label] = (proc, log)
        self._msg(f"{tool.label} started (pid {proc.pid}).")
        QTimer.singleShot(_EARLY_EXIT_CHECK_MS, lambda: self._check_early_exit(tool.label, proc))
        return proc

    def _check_early_exit(self, label, proc):
        code = proc.poll()
        if code is None or code == 0:
            return
        tail = ""
        try:
            with open(self._procs[label][1], encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-6:]).strip()
        except Exception:
            pass
        self._msg(f"{label} exited at start (code {code}). {tail}", logging.ERROR)


__all__ = ["ToolLauncher"]
