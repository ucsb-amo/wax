"""Supervisor bookkeeping that grows over a long session.

* stdout/stderr text without a newline is passed on in pieces, not buffered
  without bound;
* the previous (finished) QProcess is released when a new one is made;
* the restart history keeps only the restart window.

QProcess.start is replaced by a no-op: nothing is ever launched.
"""

from __future__ import annotations

import time

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QProcess
from PyQt6.QtWidgets import QApplication

from waxx.util.dashboard.server_supervisor import ServerSupervisor


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def sup(qapp, monkeypatch):
    def _never_start(self, *args, **kwargs):  # noqa: ANN001
        return None

    monkeypatch.setattr(QProcess, "start", _never_start)
    s = ServerSupervisor("lrs_unit", ["lrs-never-launched.exe"], requires_data_dir=False)
    lines: list[str] = []
    s.log_line.connect(lines.append)
    s.lines = lines
    yield s
    s.deleteLater()


def test_text_without_newline_is_passed_on_in_pieces(sup):
    cap = ServerSupervisor.MAX_LINE_CHARS
    sup._emit_lines("stdout", b"x" * (cap + 1000))
    assert len(sup.lines) == 1
    assert sup.lines[0].startswith("[OUT] " + "x" * 100)
    assert "line split" in sup.lines[0]
    assert len(sup._line_buffer["stdout"]) == 1000
    sup._emit_lines("stdout", b"yz\nnext")
    assert sup.lines[1] == "[OUT] " + "x" * 1000 + "yz"
    assert sup._line_buffer["stdout"] == "next"


def test_buffer_stays_bounded_under_a_long_newline_free_stream(sup):
    cap = ServerSupervisor.MAX_LINE_CHARS
    for _ in range(50):
        sup._emit_lines("stderr", b"." * 10_000)
    assert len(sup._line_buffer["stderr"]) <= cap
    total = sum(len(l) for l in sup.lines)
    assert total >= 50 * 10_000 - cap      # nothing silently dropped
    assert all(l.startswith("[ERR] ") for l in sup.lines)


def test_partial_last_line_is_flushed(sup):
    sup._emit_lines("stdout", b"done\nno newline at exit")
    sup._flush_partial_lines()
    assert sup.lines == ["[OUT] done", "[OUT] no newline at exit"]
    assert sup._line_buffer == {"stdout": "", "stderr": ""}


def test_previous_process_object_is_released_on_respawn(sup):
    sup._spawn()
    first = sup._proc
    assert first is not None and first.state() == QProcess.ProcessState.NotRunning
    sup._spawn()
    second = sup._proc
    assert second is not first
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    assert sip.isdeleted(first)
    assert not sip.isdeleted(second)


def test_restart_history_keeps_only_the_window(sup):
    old = time.monotonic() - 10 * ServerSupervisor.RESTART_WINDOW_S
    sup._restart_history = [old] * 500
    sup._spawn()
    assert len(sup._restart_history) == 1
