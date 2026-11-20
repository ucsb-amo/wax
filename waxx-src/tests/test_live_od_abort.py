"""The monitor panel's Abort (waxx.util.guis.live_od_abort): POLL first, RESET
only for the run in progress and naming it, no RESET without a run, honest
messages when liveOD does not answer, red while the status shows a run.

Offscreen Qt; the liveOD client is a fake (no discovery, no socket)."""
import os
import socket

import pytest
from PyQt6.QtWidgets import QApplication

from waxx.util.guis import live_od_abort as la


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


class _NoSocket:
    def __init__(self, *a, **k):
        raise AssertionError("the button opened a socket")


@pytest.fixture(autouse=True)
def no_sockets(monkeypatch):
    monkeypatch.setattr(socket, "socket", _NoSocket)


class FakeLiveOD:
    def __init__(self, poll=None, reset_reply=None, poll_raises=False, reset_raises=False):
        self.poll_reply = poll if poll is not None else {"ok": True, "run_in_progress": False}
        self.reset_reply = reset_reply if reset_reply is not None else {"ok": True}
        self.poll_raises, self.reset_raises = poll_raises, reset_raises
        self.sent, self.closed = [], False

    def poll(self):
        if self.poll_raises:
            raise TimeoutError("no reply")
        return dict(self.poll_reply)

    def _send_recv(self, msg):
        self.sent.append(dict(msg))
        if self.reset_raises:
            raise TimeoutError("no reply")
        return dict(self.reset_reply)

    def close(self):
        self.closed = True


def _button(fake):
    said = []
    made = []

    def factory():
        made.append(fake)
        return fake
    button = la.LiveODAbortButton(say=said.append, client_factory=factory, synchronous=True)
    return button, said, made


def test_no_run_sends_nothing(qapp):
    fake = FakeLiveOD()
    button, said, _ = _button(fake)
    button.click()
    assert fake.sent == []
    assert said == ["Abort: there is no run to abort."]


def test_run_in_progress_resets_that_run(qapp):
    fake = FakeLiveOD(poll={"ok": True, "run_in_progress": True, "run_id": 95001})
    button, said, _ = _button(fake)
    button.click()
    assert fake.sent == [{"tag": "RESET", "source": "person", "run_id": 95001}]
    assert "run 95001" in said[0]


def test_unsaved_run_resets_without_run_id(qapp):
    fake = FakeLiveOD(poll={"ok": True, "run_in_progress": True, "run_id": 0})
    button, said, _ = _button(fake)
    button.click()
    assert fake.sent == [{"tag": "RESET", "source": "person"}]
    assert "unsaved run" in said[0]


def test_saving_reply_is_shown(qapp):
    fake = FakeLiveOD(poll={"ok": True, "run_in_progress": True, "run_id": 7},
                      reset_reply={"ok": True, "ignored": True, "saving": True, "run_id": 7})
    button, said, _ = _button(fake)
    button.click()
    assert said == ["Abort ignored: run 7 is being saved (it is complete)"]


def test_poll_unanswered_says_nothing_sent_and_drops_client(qapp):
    fake = FakeLiveOD(poll_raises=True)
    button, said, made = _button(fake)
    button.click()
    assert fake.sent == [] and fake.closed
    assert "nothing was sent" in said[0]
    button.click()                          # rediscovers
    assert len(made) == 2


def test_reset_unanswered_does_not_claim_nothing_sent(qapp):
    fake = FakeLiveOD(poll={"ok": True, "run_in_progress": True, "run_id": 3},
                      reset_raises=True)
    button, said, _ = _button(fake)
    button.click()
    assert len(fake.sent) == 1
    assert "was sent" in said[0] and "nothing was sent" not in said[0]


def test_not_found(qapp):
    said = []

    def factory():
        raise RuntimeError("live_od not beaconing")
    button = la.LiveODAbortButton(say=said.append, client_factory=factory, synchronous=True)
    button.click()
    assert "not found" in said[0] and "nothing was sent" in said[0]


def test_red_while_a_run_holds_the_machine(qapp):
    button, _, made = _button(FakeLiveOD())
    assert made == []                       # no client at construction
    assert button.isEnabled() and button.styleSheet() == ""
    button.set_state({"run_pending": {"run_id": 1}})
    assert button.styleSheet() == la.ACTIVE_CSS
    button.set_state({"run_pending": None, "run_queue": {"current": None}})
    assert button.styleSheet() == ""
    button.set_state({"run_queue": {"current": {"id": 4, "state": "running"}}})
    assert button.styleSheet() == la.ACTIVE_CSS
    button.set_state(None)
    assert button.styleSheet() == "" and button.isEnabled()
