"""A supervised process that exits with EXIT_RESTART (3) is started again --
not counted as a crash, whatever restart_on_crash says; a real crash keeps
the crash policy.  QProcess.start is replaced by a no-op and QTimer.singleShot
is recorded: nothing is ever launched."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QApplication

from waxx.util.dashboard import exit_codes
from waxx.util.dashboard import server_supervisor as sv
from waxx.util.dashboard.server_supervisor import ServerSupervisor, SupervisorState


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def sup(qapp, monkeypatch):
    monkeypatch.setattr(QProcess, "start", lambda self, *a, **k: None)
    timers = []
    s = ServerSupervisor("monitor_unit", ["never-launched.exe"], requires_data_dir=False)

    class Timer(sv.QTimer):                               # instances as before
        @staticmethod
        def singleShot(ms, fn):
            timers.append((ms, fn))
    monkeypatch.setattr(sv, "QTimer", Timer)
    crashes = []
    s.crashed.connect(crashes.append)
    s.timers, s.crashes = timers, crashes
    yield s
    s.deleteLater()


def test_exit_restart_is_started_again_not_counted_as_a_crash(sup):
    assert exit_codes.EXIT_RESTART == 3
    assert sup.restart_on_crash is False                  # the monitor's spec, as today
    sup._restart_history = [1.0, 2.0]
    sup._on_finished(exit_codes.EXIT_RESTART, QProcess.ExitStatus.NormalExit)
    assert sup._state == SupervisorState.IDLE and sup.crashes == []
    assert sup.timers == [(int(sup.INITIAL_RESTART_DELAY_S * 1000), sup.start)]
    before = list(sup._restart_history)
    sup._spawn()                                          # the start it asked for
    assert sup._restart_history == [t for t in before
                                    if t >= __import__("time").monotonic() - sup.RESTART_WINDOW_S]


def test_a_real_crash_keeps_the_crash_policy(sup):
    sup._on_finished(1, QProcess.ExitStatus.NormalExit)
    assert sup._state == SupervisorState.CRASHED and sup.crashes == [1]
    assert sup.timers == []                               # restart_on_crash is False


def test_no_restart_when_stopping_or_shutting_down(sup):
    sup._stop_requested = True
    sup._on_finished(exit_codes.EXIT_RESTART, QProcess.ExitStatus.NormalExit)
    assert sup.timers == []
    sup._stop_requested = False
    sup._restart_suppressed = True
    sup._on_finished(exit_codes.EXIT_RESTART, QProcess.ExitStatus.NormalExit)
    assert sup.timers == [] and sup._state == SupervisorState.IDLE


def test_the_monitor_server_uses_the_shared_code():
    from waxx.util.guis import monitor_server_gui as msg
    assert msg.MonitorUDPServer.EXIT_RESTART == exit_codes.EXIT_RESTART
