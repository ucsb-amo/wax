"""liveOD's server loop sets an Andor EM gain above 300 back to 300 when the
run's experiment process is gone without a word (killed, crashed): only then,
at most every DEAD_CLIENT_EM_GAIN_CHECK_S, and without closing the run.  The
camera host is a stand-in; nothing opens a socket or a camera."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from live_od_data_fakes import FakeSaver


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class StubHost:
    def __init__(self, pending=True):
        self.pending = pending
        self.undone = []

    def extreme_gain_pending(self, token):
        return self.pending

    def undo_extreme_em_gain(self, token, reason):
        self.undone.append((token, reason))
        self.pending = False
        return True


@pytest.fixture
def server(app, tmp_path):
    from waxx.util.live_od.live_od_server import LiveODServer
    srv = LiveODServer(server_talk=None, data_saver=FakeSaver(tmp_path))   # never started
    srv._run_in_progress = True
    srv._current_run_id = 91000
    srv._host_run_token = "tok"
    return srv


def _with(server, monkeypatch, host, dead):
    server._camera_host = host
    monkeypatch.setattr(server, "client_known_dead", lambda: dead)
    return server


def test_a_dead_experiment_gets_its_gain_set_back_and_the_run_stays_open(server, monkeypatch):
    host = StubHost()
    srv = _with(server, monkeypatch, host, dead=True)
    srv._check_dead_client_em_gain(now=100.0)
    assert host.undone == [("tok", "the experiment's process is gone")]
    assert srv._run_in_progress and srv._host_run_token == "tok"          # nothing closed
    srv._check_dead_client_em_gain(now=200.0)
    assert len(host.undone) == 1                                           # once


@pytest.mark.parametrize("dead, pending", [(False, True), (True, False)])
def test_nothing_happens_otherwise(server, monkeypatch, dead, pending):
    host = StubHost(pending=pending)
    srv = _with(server, monkeypatch, host, dead=dead)
    srv._check_dead_client_em_gain(now=100.0)
    assert host.undone == []


def test_the_check_is_throttled_and_needs_a_run(server, monkeypatch):
    host = StubHost()
    calls = []
    srv = _with(server, monkeypatch, host, dead=False)
    monkeypatch.setattr(srv, "client_known_dead", lambda: calls.append(1) or False)
    srv._check_dead_client_em_gain(now=100.0)
    srv._check_dead_client_em_gain(now=100.5)                              # within 2 s
    assert len(calls) == 1
    srv._check_dead_client_em_gain(now=102.5)
    assert len(calls) == 2
    srv._run_in_progress = False
    srv._check_dead_client_em_gain(now=110.0)
    assert len(calls) == 2 and host.undone == []
