"""A supervised process that exits with EXIT_RESTART (3) after running at least
RESTART_REQUEST_MIN_UPTIME_S is started again -- not counted as a crash,
whatever restart_on_crash says -- with its own storm guard (more than 3 within
5 min -> FAILED).  Sooner, exit 3 is a crash (abort(), qFatal, a platform-plugin
failure give 3 on Windows).  A real crash keeps the crash policy.
QProcess.start is replaced by a no-op and QTimer.singleShot is recorded:
nothing is ever launched."""

from __future__ import annotations

import time

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


def _ran(sup, seconds):
    """The child started ``seconds`` ago (as _on_started records it)."""
    sup._child_started_at = time.monotonic() - seconds


def _exit3(sup):
    sup._on_finished(exit_codes.EXIT_RESTART, QProcess.ExitStatus.NormalExit)


def test_exit_restart_is_started_again_not_counted_as_a_crash(sup):
    assert exit_codes.EXIT_RESTART == 3
    assert sup.restart_on_crash is False                  # the monitor's spec, as today
    sup._restart_history = [1.0, 2.0]
    _ran(sup, 60)
    _exit3(sup)
    assert sup._state == SupervisorState.IDLE and sup.crashes == []
    assert sup.timers == [(int(sup.INITIAL_RESTART_DELAY_S * 1000), sup.start)]
    before = list(sup._restart_history)
    sup._spawn()                                          # the start it asked for
    assert sup._restart_history == [t for t in before
                                    if t >= time.monotonic() - sup.RESTART_WINDOW_S]
    assert sup._child_started_at is None                  # a new child: not started yet


def test_exit_3_soon_after_start_is_a_crash_not_a_request(sup):
    assert sup.RESTART_REQUEST_MIN_UPTIME_S == 10.0
    _ran(sup, 2)                                          # e.g. qFatal at start-up
    _exit3(sup)
    assert sup._state == SupervisorState.CRASHED and sup.crashes == [3]
    assert sup.timers == []                               # restart_on_crash is False
    assert sup._requested_restart_history == []


def test_exit_3_before_the_child_ever_started_is_a_crash(sup):
    sup._child_started_at = None
    _exit3(sup)
    assert sup._state == SupervisorState.CRASHED and sup.timers == []


def test_exit_3_soon_after_start_follows_the_crash_restart_policy(sup):
    sup.restart_on_crash = True
    sup._restart_history = [time.monotonic()]
    _ran(sup, 1)
    _exit3(sup)
    assert sup._state == SupervisorState.CRASHED and sup.crashes == [3]
    assert len(sup.timers) == 1                           # the crash auto-restart, as today


def test_more_than_three_requested_restarts_in_five_minutes_fail(sup, caplog):
    assert (sup.MAX_REQUESTED_RESTARTS, sup.REQUESTED_RESTART_WINDOW_S) == (3, 300.0)
    lines = []
    sup.log_line.connect(lines.append)
    for _ in range(3):
        _ran(sup, 30)
        _exit3(sup)
        assert sup._state == SupervisorState.IDLE
    assert len(sup.timers) == 3
    _ran(sup, 30)
    with caplog.at_level("ERROR"):
        _exit3(sup)                                       # the fourth within 5 min
    assert sup._state == SupervisorState.FAILED and len(sup.timers) == 3
    assert any("more than 3 times" in r.getMessage() for r in caplog.records)
    assert any("FAILED" in line for line in lines)
    assert sup.crashes == []                              # never counted as crashes


def test_requested_restarts_outside_the_window_do_not_count(sup):
    old = time.monotonic() - sup.REQUESTED_RESTART_WINDOW_S - 1
    sup._requested_restart_history = [old, old, old]
    _ran(sup, 30)
    _exit3(sup)
    assert sup._state == SupervisorState.IDLE and len(sup.timers) == 1
    assert len(sup._requested_restart_history) == 1


def test_reset_and_start_clears_the_requested_restart_guard(sup, monkeypatch):
    monkeypatch.setattr(sup, "start", lambda: None)       # no discovery, no spawn
    sup._requested_restart_history = [time.monotonic()] * 3
    sup.reset_and_start()
    assert sup._requested_restart_history == []


def test_a_real_crash_keeps_the_crash_policy(sup):
    _ran(sup, 60)
    sup._on_finished(1, QProcess.ExitStatus.NormalExit)
    assert sup._state == SupervisorState.CRASHED and sup.crashes == [1]
    assert sup.timers == []                               # restart_on_crash is False


def test_no_restart_when_stopping_or_shutting_down(sup):
    _ran(sup, 60)
    sup._stop_requested = True
    _exit3(sup)
    assert sup.timers == []
    sup._stop_requested = False
    sup._restart_suppressed = True
    _ran(sup, 60)
    _exit3(sup)
    assert sup.timers == [] and sup._state == SupervisorState.IDLE


def test_the_monitor_server_uses_the_shared_code():
    from waxx.util.guis import monitor_server_gui as msg
    assert msg.MonitorUDPServer.EXIT_RESTART == exit_codes.EXIT_RESTART
