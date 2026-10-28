"""The dashboard's COM pill: the held state, and ServerLink sending its clicks
to the client's com_disconnect / com_connect on a worker thread (never the
GUI thread), with a refusal shown to the person."""
from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from waxx.util.dashboard import server_link as server_link_mod  # noqa: E402
from waxx.util.dashboard.server_link import ServerLink  # noqa: E402
from waxx.util.dashboard.widgets import ComStatusButton  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _pump(qapp, pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if pred():
            return True
        time.sleep(0.01)
    qapp.processEvents()
    return pred()


class FakeHeader:
    def __init__(self):
        self.btn = ComStatusButton("COM40")

    def com_button(self):
        return self.btn

    def set_conn(self, *a, **k):
        pass

    def set_com(self, com):
        pass


class FakeClient:
    timeout = 1.0

    def __init__(self, answer=(True, "")):
        self.answer = answer
        self.calls = []                       # (name, thread)

    def get_snapshot(self):
        return {"com": {"status": "connected", "port": "COM40"}}

    def com_disconnect(self):
        self.calls.append(("com_disconnect", threading.current_thread()))
        time.sleep(0.2)                       # a slow server must not freeze the GUI
        return self.answer

    def com_connect(self):
        self.calls.append(("com_connect", threading.current_thread()))
        return self.answer


def _link(qapp, client):
    header = FakeHeader()
    link = ServerLink("pdxc", lambda: client, header=header, supervisor=None)
    assert _pump(qapp, lambda: link.client is client)
    if link._poller is not None:
        link._poller.stop()
    return link, header


def test_held_has_its_own_colour_and_a_click_reconnects(qapp):
    btn = ComStatusButton("COM40")
    btn.set_status("held", port="COM40")
    assert btn.status() == "held"            # not shown as an error any more
    assert "nothing reconnects" in btn.toolTip()
    got = []
    btn.reconnect_requested.connect(lambda: got.append("reconnect"))
    btn.disconnect_requested.connect(lambda: got.append("disconnect"))
    btn._on_clicked()
    assert got == ["reconnect"]


def test_disconnect_click_runs_off_the_gui_thread(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(server_link_mod.QMessageBox, "warning",
                        lambda *a, **k: warnings.append(a))
    client = FakeClient()
    link, header = _link(qapp, client)
    done = []
    link.com_request_done.connect(lambda r, ok, m: done.append((r, ok, m)))
    t0 = time.monotonic()
    header.btn.disconnect_requested.emit()
    assert time.monotonic() - t0 < 0.1, "the click must not wait for the server"
    assert _pump(qapp, lambda: done)
    assert done == [("com_disconnect", True, "")]
    assert client.calls[0][1] is not threading.main_thread()
    assert warnings == []
    link.stop()


def test_a_refusal_is_shown_to_the_person(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(server_link_mod.QMessageBox, "warning",
                        lambda *a, **k: warnings.append(a))
    client = FakeClient(answer=(False, "magnets are enabled"))
    link, header = _link(qapp, client)
    header.btn.reconnect_requested.emit()
    assert _pump(qapp, lambda: warnings)
    assert "magnets are enabled" in warnings[0][2]
    link.stop()


def test_a_client_without_com_control_says_so(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(server_link_mod.QMessageBox, "warning",
                        lambda *a, **k: warnings.append(a))

    class Plain:
        timeout = 1.0

        def get_snapshot(self):
            return {}

    link, header = _link(qapp, Plain())
    header.btn.disconnect_requested.emit()
    assert _pump(qapp, lambda: warnings)
    assert "cannot connect or disconnect" in warnings[0][2]
    link.stop()
