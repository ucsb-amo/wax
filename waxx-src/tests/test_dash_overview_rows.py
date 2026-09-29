"""Running Servers overview rows: play / stop toggle + restart per supervisor.
Offscreen Qt; the supervisor is a fake -- no process is started or killed."""
import os

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from waxx.util.dashboard.running_servers_panel import RunningServersPanel
from waxx.util.dashboard.server_supervisor import SupervisorState as S


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class FakeSupervisor(QObject):
    state_changed = pyqtSignal(object)

    def __init__(self, state=S.IDLE):
        super().__init__()
        self.state = state
        self.calls = []
        self.state_changed.connect(lambda st: setattr(self, "state", st))

    def start(self):
        self.calls.append("start")

    def reset_and_start(self):
        self.calls.append("reset_and_start")

    def stop(self):
        self.calls.append("stop")

    def restart(self):
        self.calls.append("restart")


@pytest.fixture
def row(qapp):
    sup = FakeSupervisor()
    asked, answer = [], [True]

    def confirm(parent, sid, action):
        asked.append((parent, sid, action))
        return answer[0]

    panel = RunningServersPanel([("mon", "Monitor", sup)], confirm=confirm)
    yield panel, panel._rows["mon"], sup, asked, answer
    panel.deleteLater()


def test_toggle_starts_when_down_and_stops_when_up(row):
    panel, r, sup, asked, _ = row
    assert r.toggle.isEnabled() and r.toggle.toolTip() == "Start"
    r.toggle.click()
    assert sup.calls == ["start"] and asked == []           # nothing to kill, no question
    sup.state_changed.emit(S.RUNNING)
    assert r.toggle.toolTip() == "Stop" and r.state.text() == "running"
    r.toggle.click()
    assert sup.calls == ["start", "stop"]
    assert asked == [(panel, "mon", "stop")]


def test_declined_confirmation_does_nothing(row):
    _, r, sup, asked, answer = row
    sup.state_changed.emit(S.RUNNING)
    answer[0] = False
    r.toggle.click()
    r.restart.click()
    assert sup.calls == []
    assert [a[2] for a in asked] == ["stop", "restart"]


def test_crashed_and_failed_reset_before_starting(row):
    _, r, sup, _, _ = row
    sup.state_changed.emit(S.CRASHED)
    r.toggle.click()
    sup.state_changed.emit(S.FAILED)
    r.toggle.click()
    assert sup.calls == ["reset_and_start", "reset_and_start"]
    assert not r.restart.isEnabled()                         # FAILED: play resets instead


def test_external_cannot_be_stopped_from_here(row):
    _, r, sup, _, _ = row
    sup.state_changed.emit(S.EXTERNAL)
    assert not r.toggle.isEnabled() and not r.restart.isEnabled()
    assert "outside this dashboard" in r.toggle.toolTip()
    assert "did not start it" in r.state.toolTip()
    sup.state_changed.emit(S.STOPPING)
    assert not r.toggle.isEnabled()


def test_restart_from_idle_needs_no_question(row):
    _, r, sup, asked, _ = row
    r.restart.click()
    assert sup.calls == ["restart"] and asked == []


def test_a_row_without_a_supervisor_is_inert(qapp):
    panel = RunningServersPanel([("x", "X", None)])
    r = panel._rows["x"]
    assert not r.toggle.isEnabled() and not r.restart.isEnabled()
    panel.deleteLater()
