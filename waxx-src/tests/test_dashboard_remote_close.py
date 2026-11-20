"""Remote close of a dashboard (remote_control + kdash + the window's close path).

Everything on 127.0.0.1 with advertise=False: no beacon is sent, nothing on the
lab network is reached.  Offscreen Qt; QSettings goes to an INI in tmp_path.
Fake supervisors only: no process is started or killed.
"""
import io
import json
import os
import socket
import threading
import time

import pytest
from PyQt6.QtCore import QObject, QSettings, pyqtSignal
from PyQt6.QtWidgets import QApplication, QLabel

from waxx.util.dashboard import kdash
from waxx.util.dashboard import remote_control as rc


# -- helpers ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _pump(qapp, cond=lambda: False, seconds=5.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class _Calls:
    def __init__(self):
        self.closes = []
        self.kills = []


def _server(*, busy=None, on_close=None, calls=None, kind="server"):
    calls = calls or _Calls()

    def default_close(progress):
        calls.closes.append(progress.request)

        def run():
            progress.report("stopping", servers=["a", "b"])
            progress.finish("closed", hard_killed=[], com_forced=[])
        threading.Thread(target=run, daemon=True).start()
        return None

    srv = rc.DashboardControlServer(
        kind, info=lambda: {"servers": {"a": "RUNNING"}, "uptime_s": 3.0},
        on_close=on_close or default_close, on_kill=lambda req: calls.kills.append(req),
        busy_check=busy, bind_host="127.0.0.1", advertise=False, host="testhost", pid=4242)
    srv.start()
    return srv, calls


def _ask(srv, obj, raw=None):
    ip, port = srv.address
    with socket.create_connection((ip, port), timeout=5) as s:
        s.sendall(raw if raw is not None else (json.dumps(obj) + "\n").encode())
        s.settimeout(10)
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def _dash(srv, kind="server", host="testhost"):
    ip, port = srv.address
    return kdash.Dashboard(rc.beacon_id(kind, host, 4242), kind, host, 4242, ip, port)


# -- ids ---------------------------------------------------------------------------

def test_beacon_id_round_trip_and_rejects():
    sid = rc.beacon_id("server", "kong", 123)
    assert sid == "dashboard:server:kong:123"
    assert rc.parse_beacon_id(sid) == ("server", "kong", 123)
    for bad in ("monitor", "dashboard:server:kong", "dashboard:server:kong:x",
                "dashboard::kong:1", None, 5):
        assert rc.parse_beacon_id(bad) is None


def test_com_timeout_clamped():
    assert rc.clamp_com_timeout("nan") == rc.DEFAULT_COM_TIMEOUT_S
    assert rc.clamp_com_timeout(None) == rc.DEFAULT_COM_TIMEOUT_S
    assert rc.clamp_com_timeout(-5) == rc.COM_TIMEOUT_RANGE_S[0]
    assert rc.clamp_com_timeout(1e9) == rc.COM_TIMEOUT_RANGE_S[1]
    assert rc.clamp_com_timeout(12) == 12.0


# -- control server ------------------------------------------------------------------

def test_status():
    srv, _ = _server()
    try:
        [st] = _ask(srv, {"action": "status"})
        assert st["ok"] and st["kind"] == "server" and st["host"] == "testhost"
        assert st["pid"] == 4242 and st["servers"] == {"a": "RUNNING"}
        assert st["closing"] is False and st["protocol"] == rc.PROTOCOL
    finally:
        srv.stop()


def test_close_streams_progress_then_refuses_a_second_close():
    srv, calls = _server()
    try:
        lines = _ask(srv, {"action": "close", "by": "me@pc", "owner": "person"})
        assert [m.get("event") for m in lines] == ["accepted", "stopping", "done"]
        assert lines[-1]["result"] == "closed"
        assert calls.closes[0]["by"] == "me@pc" and calls.closes[0]["_peer"] == "127.0.0.1"
        [again] = _ask(srv, {"action": "close", "owner": "person"})
        assert again["ok"] is False and "already closing" in again["refused"]
    finally:
        srv.stop()


def test_agent_refused_close_and_kill():
    srv, calls = _server()
    try:
        for action in ("close", "kill"):
            [r] = _ask(srv, {"action": action, "owner": "agent", "force": True})
            assert r["ok"] is False and "agent" in r["refused"]
        assert calls.closes == [] and calls.kills == []
        [st] = _ask(srv, {"action": "status", "owner": "agent"})
        assert st["ok"]
    finally:
        srv.stop()


def test_busy_check_refuses_unless_force_and_errors_count_as_busy():
    srv, calls = _server(busy=lambda: "run 123 in progress")
    try:
        [r] = _ask(srv, {"action": "close", "owner": "person"})
        assert r["ok"] is False and "run 123" in r["refused"] and "--force" in r["refused"]
        [r] = _ask(srv, {"action": "kill", "owner": "person"})
        assert r["ok"] is False and calls.kills == []
        lines = _ask(srv, {"action": "close", "owner": "person", "force": True})
        assert lines[-1]["result"] == "closed"
    finally:
        srv.stop()

    def boom():
        raise RuntimeError("liveOD gone")
    srv, calls = _server(busy=boom)
    try:
        [r] = _ask(srv, {"action": "close", "owner": "person"})
        assert r["ok"] is False and "liveOD gone" in r["refused"]
        assert calls.closes == []
    finally:
        srv.stop()


def test_window_refusal_frees_the_server_for_the_next_close():
    answers = iter(["not now", None])

    def on_close(progress):
        a = next(answers)
        if a is None:
            threading.Thread(target=lambda: progress.finish("closed"), daemon=True).start()
        return a
    srv, _ = _server(on_close=on_close)
    try:
        [r] = _ask(srv, {"action": "close", "owner": "person"})
        assert r["refused"] == "not now"
        assert _ask(srv, {"action": "close", "owner": "person"})[-1]["result"] == "closed"
    finally:
        srv.stop()


def test_only_the_window_frees_the_server_after_an_accepted_close():
    """A client hanging up, a cancelled close, the stream ending: none of them
    frees the server by itself; the window's clear_closing does."""
    holder = {}

    def on_close(progress):
        threading.Thread(target=lambda: progress.finish("cancelled", detail="pressed"),
                         daemon=True).start()
        return None
    srv, _ = _server(on_close=on_close)
    holder["srv"] = srv
    try:
        assert _ask(srv, {"action": "close", "owner": "person"})[-1]["result"] == "cancelled"
        [r] = _ask(srv, {"action": "close", "owner": "person"})
        assert "already closing" in r["refused"]
        srv.clear_closing()                  # what the window does when it stays open
        assert _ask(srv, {"action": "close", "owner": "person"})[-1]["result"] == "cancelled"
    finally:
        srv.stop()


def test_kill_allowed_while_a_close_is_stuck():
    def on_close(progress):
        return None                          # accepted, never finishes
    srv, calls = _server(on_close=on_close)
    try:
        ip, port = srv.address
        s = socket.create_connection((ip, port), timeout=5)
        s.sendall(b'{"action": "close", "owner": "person"}\n')
        s.recv(100)
        [r] = _ask(srv, {"action": "kill", "owner": "person"})
        assert r["event"] == "killing" and _wait(lambda: len(calls.kills) == 1)
        s.close()
    finally:
        srv.stop()


def test_heartbeats_while_closing_and_kdash_skips_them(monkeypatch):
    monkeypatch.delenv("WAXX_OWNER", raising=False)
    monkeypatch.setattr(rc, "HEARTBEAT_S", 0.2)

    def on_close(progress):
        threading.Thread(target=lambda: (time.sleep(1.2), progress.finish("closed")),
                         daemon=True).start()
        return None
    srv, _ = _server(on_close=on_close)
    try:
        lines = _ask(srv, {"action": "close", "owner": "person"})
        assert sum(m.get("event") == "heartbeat" for m in lines) >= 2
        srv.clear_closing()
        out = io.StringIO()
        # heartbeats, not lines with news, are what keep a short --wait alive
        assert kdash.main(["close", "testhost", "-y", "--wait", "5"], out=out,
                          discover=lambda: [_dash(srv)]) == 0, out.getvalue()
        assert "heartbeat" not in out.getvalue()
    finally:
        srv.stop()


def test_busy_check_runs_outside_the_lock():
    started = threading.Event()

    def slow():
        started.set()
        time.sleep(1.5)
        return None
    srv, _ = _server(busy=slow)
    try:
        t = threading.Thread(target=_ask, args=(srv, {"action": "close", "owner": "person"}),
                             daemon=True)
        t.start()
        assert started.wait(5)
        t0 = time.monotonic()
        srv.clear_closing()                  # the GUI thread's call: must not wait 1.5 s
        assert time.monotonic() - t0 < 0.2
        [st] = _ask(srv, {"action": "status"})
        assert st["ok"]
        t.join(10)
    finally:
        srv.stop()


def test_slow_trickle_client_cut_off_and_connection_cap(monkeypatch):
    monkeypatch.setattr(rc, "REQUEST_READ_TIMEOUT_S", 0.6)
    srv, _ = _server()
    try:
        ip, port = srv.address
        s = socket.create_connection((ip, port), timeout=5)
        t0 = time.monotonic()
        reply = b""
        s.settimeout(0.1)
        for _ in range(40):                  # one byte every 0.1 s, no newline
            try:
                s.sendall(b" ")
                reply = s.recv(1000)          # read before the next byte: once the
            except socket.timeout:            # server closes, a send would reset
                continue                      # the connection and drop its reply
            except OSError:
                break
            if reply:
                break
        s.close()
        assert b"within" in reply and time.monotonic() - t0 < 4
    finally:
        srv.stop()

    monkeypatch.setattr(rc, "MAX_CONNECTIONS", 2)
    monkeypatch.setattr(rc, "REQUEST_READ_TIMEOUT_S", 3.0)
    srv, _ = _server()
    try:
        ip, port = srv.address
        hogs = [socket.create_connection((ip, port), timeout=5) for _ in range(2)]
        time.sleep(0.3)
        [r] = _ask(srv, None, raw=b'{"action": "status"}\n')
        assert "too many" in r["error"]
        for h in hogs:
            h.close()
        assert _wait(lambda: _ask(srv, {"action": "status"})[0].get("ok") is True, 5)
    finally:
        srv.stop()


def test_bind_failure_does_not_fall_back_to_all_addresses():
    srv = rc.DashboardControlServer("client", info=dict, on_close=lambda p: None,
                                    on_kill=lambda r: None, bind_host="203.0.113.7",
                                    advertise=False)
    with pytest.raises(OSError):
        srv.start()
    assert srv.address is None


def test_beacon_failure_closes_the_port(monkeypatch):
    import beacon.discovery.server as bds

    def boom(self):
        raise RuntimeError("no beacon")
    # the beacon thread never starts: nothing is sent
    monkeypatch.setattr(bds.NetServer, "_start_beacon", boom)
    srv = rc.DashboardControlServer("client", info=dict, on_close=lambda p: None,
                                    on_kill=lambda r: None, bind_host="127.0.0.1",
                                    advertise=True)
    with pytest.raises(RuntimeError):
        srv.start()
    assert srv.address is None and srv._beacon is None


def test_kill_replies_then_kills():
    srv, calls = _server()
    try:
        [r] = _ask(srv, {"action": "kill", "owner": "person", "by": "me"})
        assert r == {"ok": True, "event": "killing", "pid": 4242}
        assert _wait(lambda: len(calls.kills) == 1)
    finally:
        srv.stop()


def _wait(cond, seconds=3.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_bad_requests_answered_and_server_keeps_serving():
    srv, calls = _server()
    try:
        assert "bad request" in _ask(srv, None, raw=b"not json\n")[0]["error"]
        assert "bad request" in _ask(srv, None, raw=b"[1, 2]\n")[0]["error"]
        assert "unknown action" in _ask(srv, {"action": "reboot"})[0]["error"]
        big = b'{"action": "status", "x": "' + b"a" * (rc.MAX_REQUEST_BYTES + 10)
        assert "longer than" in _ask(srv, None, raw=big)[0]["error"]
        assert _ask(srv, {"action": "status"})[0]["ok"]
        assert calls.closes == []
    finally:
        srv.stop()


def test_client_hanging_up_mid_close_does_not_stall_finish():
    finished = threading.Event()

    def on_close(progress):
        def run():
            time.sleep(0.5)                  # the client has gone by now
            for _ in range(5):
                progress.report("stopping", servers=["x" * 1000])
            t0 = time.monotonic()
            progress.finish("closed", wait_s=5.0)
            finished.set() if time.monotonic() - t0 < 4.0 else None
        threading.Thread(target=run, daemon=True).start()
        return None
    srv, _ = _server(on_close=on_close)
    try:
        ip, port = srv.address
        s = socket.create_connection((ip, port), timeout=5)
        s.sendall(b'{"action": "close", "owner": "person"}\n')
        s.recv(100)                              # "accepted"
        s.close()
        assert finished.wait(6.0)
    finally:
        srv.stop()


def test_stop_closes_the_port():
    srv, _ = _server()
    ip, port = srv.address
    srv.stop()
    with pytest.raises(OSError):
        socket.create_connection((ip, port), timeout=1).close()


# -- kdash ---------------------------------------------------------------------------

def test_kdash_list(monkeypatch):
    srv, _ = _server()
    try:
        out = io.StringIO()
        assert kdash.main(["list"], out=out, discover=lambda: [_dash(srv)]) == 0
        text = out.getvalue()
        assert "testhost" in text and "1/1 servers up" in text
        out = io.StringIO()
        assert kdash.main(["list", "--json"], out=out, discover=lambda: [_dash(srv)]) == 0
        assert json.loads(out.getvalue())[0]["status"]["kind"] == "server"
    finally:
        srv.stop()


def test_kdash_close_yes(monkeypatch):
    monkeypatch.delenv("WAXX_OWNER", raising=False)
    srv, calls = _server()
    try:
        out = io.StringIO()
        rc_ = kdash.main(["close", "TESTHOST", "--yes", "--com-timeout", "5"], out=out,
                         discover=lambda: [_dash(srv)])
        assert rc_ == 0, out.getvalue()
        assert "CLOSED" in out.getvalue()
        assert calls.closes[0]["com_timeout_s"] == 5.0 and calls.closes[0]["owner"] == "person"
    finally:
        srv.stop()


def test_kdash_refusal_exit_1(monkeypatch):
    monkeypatch.delenv("WAXX_OWNER", raising=False)
    srv, _ = _server(busy=lambda: "busy")
    try:
        out = io.StringIO()
        assert kdash.main(["close", "testhost", "-y"], out=out,
                          discover=lambda: [_dash(srv)]) == 1
        assert "REFUSED" in out.getvalue()
    finally:
        srv.stop()


def test_kdash_unmatched_target_does_nothing(monkeypatch):
    monkeypatch.delenv("WAXX_OWNER", raising=False)
    srv, calls = _server()
    try:
        out = io.StringIO()
        assert kdash.main(["close", "testhost", "nosuchpc", "-y"], out=out,
                          discover=lambda: [_dash(srv)]) == 2
        assert "nosuchpc" in out.getvalue() and calls.closes == []
    finally:
        srv.stop()


def test_kdash_needs_yes_off_a_terminal_and_honours_no(monkeypatch):
    monkeypatch.delenv("WAXX_OWNER", raising=False)
    srv, calls = _server()
    try:
        out = io.StringIO()
        assert kdash.main(["close", "--all"], out=out, inp=io.StringIO(""),
                          discover=lambda: [_dash(srv)]) == 3

        class Tty(io.StringIO):
            def isatty(self):
                return True
        out = io.StringIO()
        assert kdash.main(["close", "--all"], out=out, inp=Tty("n\n"),
                          discover=lambda: [_dash(srv)]) == 3
        assert calls.closes == []
        out = io.StringIO()
        assert kdash.main(["close", "--all"], out=out, inp=Tty("y\n"),
                          discover=lambda: [_dash(srv)]) == 0
        assert len(calls.closes) == 1
    finally:
        srv.stop()


def test_kdash_usage_and_agent_rules(monkeypatch):
    monkeypatch.delenv("WAXX_OWNER", raising=False)

    def no_discover():
        raise AssertionError("must not look for dashboards")
    out = io.StringIO()
    assert kdash.main(["close"], out=out, discover=no_discover) == 2
    assert kdash.main(["close", "kong", "--all"], out=out, discover=no_discover) == 2
    assert kdash.main(["kill"], out=out, discover=no_discover) == 2
    with pytest.raises(SystemExit):
        kdash.main(["kill", "--all"], out=out, discover=no_discover)
    monkeypatch.setenv("WAXX_OWNER", "agent")
    assert kdash.main(["close", "kong", "-y"], out=out, discover=no_discover) == 1
    assert kdash.main(["kill", "kong", "-y"], out=out, discover=no_discover) == 1


def test_kdash_kind_filter(monkeypatch):
    monkeypatch.delenv("WAXX_OWNER", raising=False)
    srv_s, calls_s = _server(kind="server")
    srv_c, calls_c = _server(kind="client")
    try:
        dashes = [_dash(srv_s, "server"), _dash(srv_c, "client")]
        out = io.StringIO()
        assert kdash.main(["close", "testhost", "--kind", "client", "-y"], out=out,
                          discover=lambda: dashes) == 0
        assert len(calls_c.closes) == 1 and calls_s.closes == []
    finally:
        srv_s.stop()
        srv_c.stop()


def test_kdash_dashboard_vanishing_mid_close_is_a_failure(monkeypatch):
    """The dashboard process dies after "accepted": no "done" -> exit 1."""
    monkeypatch.delenv("WAXX_OWNER", raising=False)
    lsock = socket.socket()
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(4)
    port = lsock.getsockname()[1]

    def serve():
        for _ in range(2):                         # the status probe, then the close
            conn, _a = lsock.accept()
            req = json.loads(conn.makefile().readline())
            if req["action"] == "status":
                conn.sendall(b'{"ok": true, "servers": {}}\n')
            else:
                conn.sendall(b'{"ok": true, "event": "accepted"}\n')
            conn.close()
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    d = kdash.Dashboard("dashboard:server:testhost:1", "server", "testhost", 1, "127.0.0.1", port)
    out = io.StringIO()
    try:
        assert kdash.main(["close", "testhost", "-y"], out=out, discover=lambda: [d]) == 1
        assert "ended before" in out.getvalue()
    finally:
        lsock.close()


# -- the COM dialog's auto force ---------------------------------------------------

class _FakeSup(QObject):
    log_line = pyqtSignal(str)
    state_changed = pyqtSignal(object)

    def __init__(self, sid, exits_on_request=True):
        super().__init__()
        self.server_id = sid
        self._alive = True
        self._exits = exits_on_request
        self.forced = False
        self.graceful_stop_timeout_s = 0.2
        self.shutdown_request = self._shutdown

        class _St:
            name = "RUNNING"
        self.state = _St()

    def _shutdown(self):
        return True

    def is_running(self):
        return self._alive

    def is_alive(self):
        return self._alive

    def suppress_restart(self):
        pass

    def request_terminate(self):
        if self._exits:
            self._alive = False

    def wait_for_finished(self, ms):
        return not self._alive

    def force_kill(self, wait_ms=200):
        self.forced = True
        self._alive = False

    def pid(self):
        return 0


def test_com_dialog_auto_forces_after_timeout(qapp):
    from waxx.util.dashboard.com_shutdown_dialog import RESULT_USER_FORCED, ComShutdownDialog
    stuck = _FakeSup("stuck", exits_on_request=False)
    dlg = ComShutdownDialog([("Stuck server", "COM9", stuck)], auto_force_after_s=0.3,
                            note="Remote close asked for by me")
    t0 = time.monotonic()
    dlg.exec()
    assert dlg.result_reason() == RESULT_USER_FORCED
    assert stuck.forced and dlg.forced_labels() == ["Stuck server"]
    assert time.monotonic() - t0 < 3.0
    dlg.deleteLater()


def test_com_dialog_without_timeout_has_no_countdown(qapp):
    from waxx.util.dashboard.com_shutdown_dialog import ComShutdownDialog
    dlg = ComShutdownDialog([("s", "COM1", _FakeSup("s"))])
    assert dlg._countdown.isHidden()
    dlg.deleteLater()


# -- the window --------------------------------------------------------------------

@pytest.fixture
def window_factory(qapp, tmp_path, monkeypatch):
    from waxx.util.dashboard import dashboard_window as dw
    from waxx.util.dashboard.panel_container import ClientPanel
    path = str(tmp_path / "dash.ini")
    real = QSettings
    monkeypatch.setattr(dw, "QSettings", lambda *_a, **_k: real(path, real.Format.IniFormat))
    monkeypatch.setattr(dw.DashboardMainWindow, "_probe_data_dir", lambda self: None)
    exits = []
    # a real app.exit would make the next test's dialog exec() return at once
    monkeypatch.setattr(dw.DashboardMainWindow, "_exit_app_after_remote_close",
                        lambda self: exits.append(self))
    made = []

    def make(sups=None, com_ids=()):
        panels = [dw.PanelPlacement(ClientPanel("alpha", "A", lambda: QLabel("a")), area="right")]
        win = dw.DashboardMainWindow("server", "Test dashboard", panels, host_ip="test")
        if sups:
            win.register_supervisors(sups, com_ids=set(com_ids))
        win.show()
        assert win.enable_remote_control(bind_host="127.0.0.1", advertise=False)
        made.append(win)
        return win

    yield make
    for win in made:
        ctl = win._control
        if ctl is not None:
            ctl.stop()
        win.hide()
        win.deleteLater()
    qapp.processEvents()


def _close_in_thread(win, req):
    out = {}

    def run():
        try:
            ip, port = win._control.address
            out["lines"] = list(kdash.request_lines(ip, port, req, first_timeout=20,
                                                    idle_timeout=20))
        except Exception as exc:              # noqa: BLE001
            out["error"] = exc
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def test_window_remote_close_end_to_end(qapp, window_factory):
    plain = _FakeSup("plain")
    stuck_com = _FakeSup("lock", exits_on_request=False)
    win = window_factory({"plain": plain, "lock": stuck_com}, com_ids={"lock"})
    [st] = _ask(win._control, {"action": "status"})
    assert st["servers"] == {"plain": "RUNNING", "lock": "RUNNING"} and not st["closing"]

    t, out = _close_in_thread(win, {"action": "close", "owner": "person", "by": "me@pc",
                                    "com_timeout_s": 1})
    assert _pump(qapp, lambda: not t.is_alive(), seconds=15), "close never finished"
    assert "error" not in out, out
    events = [m.get("event") for m in out["lines"]]
    assert events[0] == "accepted" and events[-1] == "done", events
    assert {"stopping", "com_wait", "com_forced", "servers_stopped"} <= set(events)
    assert out["lines"][-1]["result"] == "closed"
    assert out["lines"][-1]["com_forced"] == ["lock"]
    assert stuck_com.forced and not plain.forced
    assert not win.isVisible() and win._control is None


def test_window_frozen_gui_refuses_and_late_close_is_dropped(qapp, window_factory, monkeypatch):
    win = window_factory()
    monkeypatch.setattr(type(win), "REMOTE_ACK_TIMEOUT_S", 0.3)
    # do NOT pump: the GUI thread is "frozen" while the request waits
    lines = _ask(win._control, {"action": "close", "owner": "person"})
    assert lines[0]["ok"] is False and "did not answer" in lines[0]["refused"]
    _pump(qapp, seconds=0.5)                 # the late post arrives now
    assert win.isVisible() and win._remote_close is None
    [st] = _ask(win._control, {"action": "status"})
    assert st["closing"] is False


def _spawn_child(tmp_path, mode=""):
    import subprocess
    import sys
    child = os.path.join(os.path.dirname(__file__), "remote_close_child.py")
    proc = subprocess.Popen([sys.executable, child, str(tmp_path), mode],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    port = None
    end = time.monotonic() + 60
    while time.monotonic() < end and port is None:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line.rstrip())
        if line.startswith("PORT "):
            port = int(line.split()[1])
    if port is None:
        proc.kill()
        pytest.fail("child dashboard did not start:\n" + "\n".join(lines))
    return proc, port, lines


def _finish_child(proc, lines, timeout=30):
    try:
        out, _ = proc.communicate(timeout=timeout)
    except Exception:
        proc.kill()
        out, _ = proc.communicate()
        pytest.fail("child dashboard did not exit after the remote close:\n"
                    + "\n".join(lines) + out)
    return lines + out.splitlines()


@pytest.mark.parametrize("mode", ["", "modal"])
def test_remote_close_ends_the_process(tmp_path, mode):
    """A real process: after "done closed" it exits with code 0, even with a
    modal dialog's event loop open when the close arrived."""
    proc, port, lines = _spawn_child(tmp_path, mode)
    if mode == "modal":
        while "MODAL" not in lines:
            lines.append(proc.stdout.readline().rstrip())
        time.sleep(0.3)
    got = list(kdash.request_lines("127.0.0.1", port, {"action": "close", "owner": "person",
                                                       "by": "test"},
                                   first_timeout=20, idle_timeout=20))
    assert got[0]["event"] == "accepted" and got[-1]["result"] == "closed", got
    all_lines = _finish_child(proc, lines)
    assert proc.returncode == 0, all_lines
    assert "EXIT 0" in all_lines


def test_two_simultaneous_closes_of_a_process_one_accepted(tmp_path):
    """Two closes at once on a real process: one is accepted, the other refused."""
    proc, port, lines = _spawn_child(tmp_path)
    results = []

    def one():
        results.append(list(kdash.request_lines(
            "127.0.0.1", port, {"action": "close", "owner": "person"},
            first_timeout=20, idle_timeout=20)))
    ts = [threading.Thread(target=one) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(40)
    finals = [str(r[-1].get("result") or r[-1].get("refused")) for r in results]
    assert len(finals) == 2 and finals.count("closed") == 1, results
    assert any("already closing" in f for f in finals), results
    _finish_child(proc, lines)
    assert proc.returncode == 0


def test_window_cancel_at_pc_keeps_it_open(qapp, window_factory, monkeypatch):
    from waxx.util.dashboard import com_shutdown_dialog as csd
    stuck_com = _FakeSup("lock", exits_on_request=False)
    win = window_factory({"lock": stuck_com}, com_ids={"lock"})

    real_exec = csd.ComShutdownDialog.exec

    def cancel_exec(self):
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(50, self._on_cancel)
        return real_exec(self)
    monkeypatch.setattr(csd.ComShutdownDialog, "exec", cancel_exec)

    t, out = _close_in_thread(win, {"action": "close", "owner": "person", "com_timeout_s": 60})
    assert _pump(qapp, lambda: not t.is_alive(), seconds=15)
    last = out["lines"][-1]
    assert last["event"] == "done" and last["result"] == "cancelled"
    assert last["still_running"] == ["lock"]
    assert win.isVisible() and win._control is not None and win._remote_close is None
    # and a new close is accepted again
    monkeypatch.delattr(csd.ComShutdownDialog, "exec")   # back to QDialog.exec
    t, out = _close_in_thread(win, {"action": "close", "owner": "person", "com_timeout_s": 1})
    assert _pump(qapp, lambda: not t.is_alive(), seconds=15)
    assert out["lines"][-1]["result"] == "closed"


def test_window_close_that_raises_stays_open_and_can_close_again(qapp, window_factory,
                                                                  monkeypatch):
    from waxx.util.dashboard import dashboard_window as dw
    win = window_factory({"plain": _FakeSup("plain")})
    real = dw.shutdown_all
    calls = {"n": 0}

    def flaky(sups, grace_ms):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("taskkill missing")
        return real(sups, grace_ms=grace_ms)
    monkeypatch.setattr(dw, "shutdown_all", flaky)
    t, out = _close_in_thread(win, {"action": "close", "owner": "person"})
    assert _pump(qapp, lambda: not t.is_alive(), seconds=15)
    last = out["lines"][-1]
    assert last["result"] == "error" and "taskkill missing" in last["detail"]
    assert win.isVisible() and win._control is not None
    assert win._remote_close is None and win._closing_now is False
    t, out = _close_in_thread(win, {"action": "close", "owner": "person"})
    assert _pump(qapp, lambda: not t.is_alive(), seconds=15)
    assert out["lines"][-1]["result"] == "closed"


def test_window_refuses_remote_close_while_closing_locally(qapp, window_factory):
    win = window_factory()
    win._closing_now = True                  # the X was pressed; its COM dialog is up
    t, out = _close_in_thread(win, {"action": "close", "owner": "person"})
    assert _pump(qapp, lambda: not t.is_alive(), seconds=10)
    assert "own PC" in out["lines"][0]["refused"]
    assert win._remote_close is None
    win._closing_now = False
    [st] = _ask(win._control, {"action": "status"})
    assert st["closing"] is False


def test_window_reports_servers_that_survive_the_kills(qapp, window_factory):
    immortal = _FakeSup("immortal", exits_on_request=False)
    immortal.force_kill = lambda wait_ms=200: None       # the kill does not take
    win = window_factory({"immortal": immortal})
    t, out = _close_in_thread(win, {"action": "close", "owner": "person"})
    assert _pump(qapp, lambda: not t.is_alive(), seconds=20)
    last = out["lines"][-1]
    assert last["result"] == "closed_with_survivors" and last["still_running"] == ["immortal"]
    assert "STILL RUNNING" in kdash.describe_event(
        [m for m in out["lines"] if m.get("event") == "servers_stopped"][0])


@pytest.mark.parametrize("mode", ["", "frozen_log"])
def test_kdash_kill_ends_a_real_process(tmp_path, monkeypatch, mode):
    """kdash kill on a real window process; "frozen_log": its GUI thread holds
    the log handler's lock, so only the kill watchdog can end it."""
    monkeypatch.delenv("WAXX_OWNER", raising=False)
    proc, port, lines = _spawn_child(tmp_path, mode)
    if mode:
        while "FROZEN" not in lines:
            lines.append(proc.stdout.readline().rstrip())
    d = kdash.Dashboard("dashboard:client:childhost:1", "client", "childhost", 1,
                        "127.0.0.1", port)
    out = io.StringIO()
    t0 = time.monotonic()
    assert kdash.main(["kill", "childhost", "-y"], out=out, discover=lambda: [d]) == 0, \
        out.getvalue()
    assert "KILLED" in out.getvalue()
    _finish_child(proc, lines, timeout=20)
    assert proc.returncode == 1
    if mode:
        assert time.monotonic() - t0 < 20
