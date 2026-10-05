"""logging_setup: Qt messages and unhandled exceptions end up in the log.

A pythonw dashboard has no stderr, so Qt's warnings and the traceback of an
exception escaping a slot were lost (and under python.exe that exception
aborted the process).  Only warning / critical routing is exercised here; a
Qt fatal would abort the test process.
"""

from __future__ import annotations

import logging
import sys
import threading

import pytest
from PyQt6 import QtCore
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from waxx.util.dashboard import logging_setup


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def qt_handler(monkeypatch):
    """Install the handler fresh for one test and put the previous one back."""
    monkeypatch.setattr(logging_setup, "_QT_HANDLER_INSTALLED", False)
    monkeypatch.setattr(logging_setup, "_QT_PREVIOUS_HANDLER", None)
    assert logging_setup.install_qt_message_handler() is True
    try:
        yield
    finally:
        QtCore.qInstallMessageHandler(logging_setup._QT_PREVIOUS_HANDLER)


def test_qt_warning_and_critical_are_logged(qapp, qt_handler, caplog):
    with caplog.at_level(logging.DEBUG, logger="qt"):
        QtCore.qWarning("lrs qt warning one")
        QtCore.qCritical("lrs qt critical one")
    by_msg = {r.getMessage(): r for r in caplog.records if r.name == "qt"}
    warn = next(r for m, r in by_msg.items() if "lrs qt warning one" in m)
    crit = next(r for m, r in by_msg.items() if "lrs qt critical one" in m)
    assert warn.levelno == logging.WARNING
    assert crit.levelno == logging.ERROR


def test_repeated_qt_message_is_held_back_and_counted(qapp, qt_handler, caplog, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(logging_setup.time, "monotonic", lambda: clock[0])
    with caplog.at_level(logging.DEBUG, logger="qt"):
        for _ in range(5):
            QtCore.qWarning("lrs qt repeated")
        clock[0] += logging_setup.QT_REPEAT_WINDOW_S + 1
        QtCore.qWarning("lrs qt repeated")
    lines = [r.getMessage() for r in caplog.records
             if r.name == "qt" and "lrs qt repeated" in r.getMessage()]
    assert len(lines) == 2
    assert "repeated" in lines[1] and "4 more time(s)" in lines[1]


def test_install_is_idempotent(qapp, qt_handler):
    before = logging_setup._QT_PREVIOUS_HANDLER
    assert logging_setup.install_qt_message_handler() is True
    assert logging_setup._QT_PREVIOUS_HANDLER is before


@pytest.fixture
def default_hooks(monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    monkeypatch.setattr(threading, "excepthook", threading.__excepthook__)
    logging_setup.install_excepthooks()
    assert sys.excepthook is not sys.__excepthook__
    assert threading.excepthook is not threading.__excepthook__


def test_exception_in_a_slot_is_logged_and_the_process_carries_on(qapp, default_hooks, caplog):
    class _Emitter(QObject):
        fired = pyqtSignal()

    emitter = _Emitter()

    def _boom():
        raise ValueError("lrs slot boom")

    emitter.fired.connect(_boom)
    with caplog.at_level(logging.CRITICAL, logger="uncaught"):
        emitter.fired.emit()
    rec = [r for r in caplog.records if r.name == "uncaught"]
    assert rec and rec[-1].exc_info and "lrs slot boom" in str(rec[-1].exc_info[1])


def test_exception_in_a_thread_is_logged(default_hooks, caplog):
    def _boom():
        raise RuntimeError("lrs thread boom")

    with caplog.at_level(logging.CRITICAL, logger="uncaught"):
        t = threading.Thread(target=_boom, name="lrs-thread")
        t.start()
        t.join(5.0)
    rec = [r for r in caplog.records if r.name == "uncaught"]
    assert rec and "lrs-thread" in rec[-1].getMessage()
    assert "lrs thread boom" in str(rec[-1].exc_info[1])


def test_excepthooks_install_once_and_chain_an_earlier_hook(monkeypatch, caplog):
    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: seen.append(a[1]))
    monkeypatch.setattr(threading, "excepthook", threading.__excepthook__)
    logging_setup.install_excepthooks()
    hook = sys.excepthook
    logging_setup.install_excepthooks()
    assert sys.excepthook is hook
    err = KeyError("lrs chained")
    with caplog.at_level(logging.CRITICAL, logger="uncaught"):
        sys.excepthook(KeyError, err, None)
    assert seen == [err]
    assert any(r.name == "uncaught" for r in caplog.records)
