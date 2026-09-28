"""The SLM's reinit, offline (2026-09-28).

* the SLM server (``run_server.py``, Meadowlark calls stubbed, on 127.0.0.1):
  a reinit puts back the last pattern, the hourly timer only marks one due,
  ``SLMCTL`` control lines, and the old parser dropping them;
* ``slm_link.exchange`` and the experiment's ``SLM`` client: waits for
  "applied", raises when it cannot say the mask is up, falls back once for a
  server that never replies;
* ``SlmReinitService`` with a fake link and clock: asks only while idle, never
  once a run has announced itself, and the run's announcement waits for a
  request being sent;
* the monitor server's hooks (idle check, ``run_pending`` wait, ``status_json``).

Nothing touches the lab network: servers listen on 127.0.0.1 only, and the
monitor server's broadcaster is a recorder.
"""
import importlib.util
import inspect
import itertools
import json
import os
import socket
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

import waxx
from waxx.control.slm import slm_link
from waxx.control.slm.server.slm_protocol import (control_command, control_line,
                                                  command_seq, split_commands)
from waxx.util.device_state import slm_reinit as sr
from waxx.util.device_state.slm_reinit import SlmReinitConfig, SlmReinitService

SLM_SERVER_DIR = Path(waxx.__file__).parent / "control" / "slm" / "server"
_names = itertools.count()


def _wait(pred, timeout=3.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --- the SLM server, stubbed SLM --------------------------------------------------------

@pytest.fixture
def slm_host(monkeypatch):
    """A fresh run_server module (its own queue and state), its SLM worker and
    connection handling on 127.0.0.1, with the Meadowlark calls stubbed."""
    host = types.SimpleNamespace(uploads=[], inits=0, init_s=0.0, fail_init=None)

    class StubSLM:
        mask_type = "spot"

        def initialize_slm(self):
            host.inits += 1
            time.sleep(host.init_s)
            if host.fail_init:
                raise RuntimeError(host.fail_init)

        def generate_mask(self, dimension, phase, center_x, center_y,
                          grating_spacing=10, angle_deg=0, mask=1):
            return (center_x, center_y, dimension)

        def fast_upload_to_slm(self, img):
            time.sleep(0.01)
            host.uploads.append(img)

        upload_to_slm = fast_upload_to_slm

    stub = types.ModuleType("slm_server")
    stub.SLM_server = StubSLM
    monkeypatch.setitem(sys.modules, "slm_server", stub)
    monkeypatch.syspath_prepend(str(SLM_SERVER_DIR))
    spec = importlib.util.spec_from_file_location(f"run_server_under_test_{next(_names)}",
                                                  SLM_SERVER_DIR / "run_server.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    host.rs = rs
    threading.Thread(target=rs.slm_worker, daemon=True).start()

    lsock = socket.socket()
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(4)

    def accept_loop():      # run_server.start_server, minus the fixed address
        while True:
            try:
                conn, _ = lsock.accept()
            except OSError:
                return
            rs.handle_client(conn)

    threading.Thread(target=accept_loop, daemon=True).start()
    host.port = lsock.getsockname()[1]
    assert _wait(lambda: len(host.uploads) == 1)      # the start-up init + pattern
    yield host
    lsock.close()
    rs.cmd_q.put(None)


def _ask(host, payload, until, control=False, **kw):
    kw.setdefault("total_s", 5.0)
    return slm_link.exchange("127.0.0.1", host.port, payload, until=until,
                             control=control, **kw)


def _status(host):
    return _ask(host, {"cmd": "status"}, ("ok", "error"), control=True)


def _apply(host, cx, cy, dimension=10):
    return _ask(host, {"mask": "spot", "center": [cx, cy], "dimension": dimension},
                ("applied", "error", "dropped"))


def test_reinit_puts_back_the_last_pattern_not_the_blank_default(slm_host):
    assert _apply(slm_host, 321, 432)["status"] == "applied"
    assert slm_host.uploads[-1] == (321, 432, 10)
    before = _status(slm_host)
    assert before["pattern_epoch"] == 0 and before["reinit_due"] is False

    done = _ask(slm_host, {"cmd": "reinit", "by": "test"}, ("reinit_done", "error"),
                control=True)

    assert done["status"] == "reinit_done" and done["pattern_epoch"] == 1
    assert (done["pattern"]["center_x"], done["pattern"]["center_y"]) == (321, 432)
    assert slm_host.inits == 2                         # start-up + this one
    assert slm_host.uploads[-1] == (321, 432, 10)      # not (960, 600, 0)
    after = _status(slm_host)
    assert after["pattern_epoch"] == 1 and after["reinit_in_progress"] is False
    assert after["last_reinit_age_s"] is not None and after["last_reinit_error"] == ""


def test_a_failed_reinit_is_reported_once_and_stays_due(slm_host):
    slm_host.rs._reinit["due_since"] = time.monotonic()
    slm_host.fail_init = "SDK said no"
    got = []
    seq_replies = []
    with socket.create_connection(("127.0.0.1", slm_host.port)) as s:
        s.sendall(control_line({"cmd": "reinit", "seq": 5}).encode())
        s.settimeout(2.0)
        buf = b""
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end and len(seq_replies) < 3:
            try:
                chunk = s.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                seq_replies.append(json.loads(line))
            if any(r.get("status") == "error" for r in seq_replies):
                s.settimeout(0.3)      # anything more would be a duplicate
        got = [r["status"] for r in seq_replies]
    assert got == ["queued", "error"]
    assert "SDK said no" in seq_replies[1]["error"]
    st = _status(slm_host)
    assert st["reinit_due"] is True and st["pattern_epoch"] == 0
    assert "SDK said no" in st["last_reinit_error"]


def test_the_timer_only_marks_a_reinit_due(slm_host, monkeypatch):
    rs = slm_host.rs
    assert rs.AUTO_REINIT is False
    monkeypatch.setattr(rs, "REINIT_INTERVAL_SEC", 0.2)
    _apply(slm_host, 50, 60)
    n_uploads = len(slm_host.uploads)
    stop = threading.Event()
    t = threading.Thread(target=rs.periodic_reinit_scheduler, args=(stop,), daemon=True)
    t.start()
    try:
        assert _wait(lambda: _status(slm_host)["reinit_due"], timeout=3.0)
        time.sleep(0.8)                  # well past another tick
        st = _status(slm_host)
        assert st["reinit_due"] and st["due_for_s"] >= 0.5
        assert st["reinit_in_progress"] is False and st["pattern_epoch"] == 0
        assert slm_host.inits == 1 and len(slm_host.uploads) == n_uploads
    finally:
        stop.set()
        t.join(2.0)


def test_auto_reinit_still_works_when_switched_on(slm_host, monkeypatch):
    rs = slm_host.rs
    monkeypatch.setattr(rs, "AUTO_REINIT", True)
    monkeypatch.setattr(rs, "REINIT_INTERVAL_SEC", 0.2)
    monkeypatch.setattr(rs, "MIN_IDLE_BEFORE_REINIT_SEC", 0.0)
    _apply(slm_host, 70, 80)
    stop = threading.Event()
    t = threading.Thread(target=rs.periodic_reinit_scheduler, args=(stop,), daemon=True)
    t.start()
    try:
        assert _wait(lambda: slm_host.inits >= 2, timeout=3.0)
        assert _wait(lambda: slm_host.uploads[-1] == (70, 80, 10))
    finally:
        stop.set()
        t.join(2.0)


def test_unknown_control_command_is_an_error_and_applies_nothing(slm_host):
    n = len(slm_host.uploads)
    r = _ask(slm_host, {"cmd": "explode"}, ("ok", "error"), control=True)
    assert r["status"] == "error" and "explode" in r["error"]
    time.sleep(0.1)
    assert len(slm_host.uploads) == n


def test_control_lines_are_malformed_to_the_legacy_parser(slm_host):
    """Every server since b15b483 parses non-JSON text with this same
    plaintext branch: a control line must never come out as a pattern."""
    for obj in ({"cmd": "status", "seq": 1}, {"cmd": "reinit", "seq": 2, "by": "the monitor server"},
                {"cmd": "status"}):
        assert slm_host.rs.analyze_command(control_line(obj).strip()) is None


# --- wire format -----------------------------------------------------------------------

def test_control_line_round_trip_and_framing():
    line = control_line({"cmd": "reinit", "seq": 9, "by": "the monitor server"})
    assert line.startswith("SLMCTL {") and line.endswith("\n") and line.count("\n") == 1
    assert control_command(line) == {"cmd": "reinit", "seq": 9, "by": "the monitor server"}
    assert command_seq(line.strip()) == 9
    assert control_command('{"cmd": "status"}') is None          # bare JSON is a pattern
    assert control_command("SLMCTL not json") is None
    assert control_command("SLMCTL [1, 2]") is None
    # held until its newline arrives, however it is split
    assert split_commands(line[:3]) == ([], line[:3])
    assert split_commands(line[:12]) == ([], line[:12])
    assert split_commands(line) == ([line.strip()], "")
    assert split_commands(line[:12], at_eof=True) == ([line[:12]], "")
    # legacy plaintext and patterns are unchanged
    assert split_commands("10 0.5 1") == (["10 0.5 1"], "")
    a = '{"center": [1, 2]}'
    assert split_commands(line + a) == ([line.strip(), a], "")


# --- slm_link --------------------------------------------------------------------------

class ScriptedServer:
    """Accepts connections on 127.0.0.1; answers each command line with
    ``script(msg)`` -> a list of (delay_s, reply-dict-without-seq)."""

    def __init__(self, script):
        self.script = script
        self.received = []
        self.lsock = socket.socket()
        self.lsock.bind(("127.0.0.1", 0))
        self.lsock.listen(4)
        self.port = self.lsock.getsockname()[1]
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                conn, _ = self.lsock.accept()
            except OSError:
                return
            with conn:
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                text = buf.split(b"\n", 1)[0].decode()
                msg = control_command(text) or json.loads(text)
                self.received.append(msg)
                try:
                    for delay, reply in self.script(msg):
                        time.sleep(delay)
                        conn.sendall((json.dumps({"seq": msg["seq"], **reply}) + "\n").encode())
                    time.sleep(2.0)            # stay open, silent
                except OSError:
                    pass

    def close(self):
        self.lsock.close()


def test_exchange_no_reply_timeout_and_unreachable():
    silent = ScriptedServer(lambda msg: [])
    queued_only = ScriptedServer(lambda msg: [(0.0, {"status": "queued"})])
    try:
        t0 = time.monotonic()
        with pytest.raises(slm_link.NoReply):
            slm_link.exchange("127.0.0.1", silent.port, {"cmd": "status"}, control=True,
                              until=("ok",), first_reply_s=0.3, total_s=1.0)
        assert time.monotonic() - t0 < 1.0
        with pytest.raises(slm_link.ReplyTimeout):
            slm_link.exchange("127.0.0.1", queued_only.port, {"mask": "spot"},
                              until=("applied",), first_reply_s=0.3, total_s=0.6)
        sent = []
        with pytest.raises(OSError):
            slm_link.exchange("127.0.0.1", _free_port(), {"cmd": "status"}, control=True,
                              until=("ok",), connect_s=0.5, on_sent=lambda: sent.append(1))
        assert sent == [1]            # a failed send still releases a waiting run
    finally:
        silent.close()
        queued_only.close()


# --- the experiment's SLM client -------------------------------------------------------

def _slm(port):
    from waxx.control.slm.slm import SLM
    params = types.SimpleNamespace(dimension_slm_mask=10e-6, phase_slm_mask=np.pi / 2,
                                   px_slm_phase_mask_position_x=100,
                                   px_slm_phase_mask_position_y=200)
    return SLM(expt_params=params, core=None, server_ip="127.0.0.1", server_port=port)


def test_client_returns_once_the_mask_is_applied(slm_host):
    slm = _slm(slm_host.port)
    slm.write_phase_mask(dimension=20e-6, phase=np.pi, x_center=321, y_center=432,
                         verbose=False)
    assert slm_host.uploads[-1][:2] == (321, 432)          # already up, not "soon"
    assert slm._server_replies is True


def test_client_waits_behind_a_reinit_and_its_mask_lands_last(slm_host):
    _apply(slm_host, 11, 12)
    slm_host.init_s = 0.5
    slm = _slm(slm_host.port)
    with socket.create_connection(("127.0.0.1", slm_host.port)) as s:
        s.sendall(control_line({"cmd": "reinit"}).encode())
    t0 = time.monotonic()
    slm.write_phase_mask(x_center=13, y_center=14, verbose=False)
    assert time.monotonic() - t0 >= 0.4
    assert [u[:2] for u in slm_host.uploads[-2:]] == [(11, 12), (13, 14)]


def test_client_raises_when_the_server_refuses_or_is_gone(monkeypatch):
    from waxx.control.slm import slm as slm_mod
    refuses = ScriptedServer(lambda msg: [(0.0, {"status": "queued"}),
                                          (0.0, {"status": "error", "error": "bad mask"})])
    drops = ScriptedServer(lambda msg: [(0.0, {"status": "queued"}),
                                        (0.0, {"status": "dropped"})])
    stalls = ScriptedServer(lambda msg: [(0.0, {"status": "queued"})])
    monkeypatch.setattr(slm_mod, "SLM_APPLIED_TIMEOUT_S", 0.5)
    try:
        for server, text in ((refuses, "bad mask"), (drops, "dropped"), (stalls, "applied")):
            with pytest.raises(slm_mod.SLMWriteError, match=text):
                _slm(server.port).write_phase_mask(verbose=False)
        with pytest.raises(slm_mod.SLMWriteError):
            _slm(_free_port()).write_phase_mask(verbose=False)
    finally:
        for s in (refuses, drops, stalls):
            s.close()


def test_client_falls_back_once_for_a_server_that_never_replies(monkeypatch, capsys):
    from waxx.control.slm import slm as slm_mod
    monkeypatch.setattr(slm_mod, "SLM_FIRST_REPLY_S", 0.3)
    old = ScriptedServer(lambda msg: [])
    try:
        slm = _slm(old.port)
        t0 = time.monotonic()
        slm.write_phase_mask(x_center=1, y_center=2, verbose=False)
        t1 = time.monotonic()
        slm.write_phase_mask(x_center=3, y_center=4, verbose=False)
        t2 = time.monotonic()
        assert 0.25 <= t1 - t0 < 1.5 and t2 - t1 < 0.3
        assert slm._server_replies is False
        assert _wait(lambda: len(old.received) == 2)
        assert [m["center"] for m in old.received] == [[1, 2], [3, 4]]
        assert capsys.readouterr().out.count("does not confirm writes") == 1
    finally:
        old.close()


def test_kernel_wrapper_resyncs_the_timeline_after_the_rpc():
    """The RPC now blocks until the mask is up (seconds behind a reinit), so
    the kernel must break_realtime after it, before scheduling anything."""
    from waxx.control.slm.slm import SLM
    src = inspect.getsource(SLM.write_phase_mask_kernel)
    body = src[src.index('"""', src.index('"""') + 3) + 3:]
    i_rpc = body.index("self.write_phase_mask(")
    i_brk = body.index("self.core.break_realtime()")
    i_dly = body.index("delay(SLM_RPC_DELAY)")
    assert i_rpc < i_brk < i_dly


# --- SlmReinitService, fake link and clock ---------------------------------------------

class FakeSlm:
    """The SLM server as the service sees it through slm_link.exchange."""

    def __init__(self):
        self.due = False
        self.in_progress = False
        self.epoch = 0
        self.error = ""
        self.requests = []
        self.silent = False
        self.down = False
        self.hold_send = None          # threading.Event the reinit send waits on
        self.polls_to_finish = 2
        self._left = 0

    def __call__(self, host, port, payload, *, control, until, on_sent=None, **kw):
        cmd = payload["cmd"]
        self.requests.append(cmd)
        if self.down:
            if on_sent:
                on_sent()
            raise ConnectionRefusedError("refused")
        if cmd == "reinit" and self.hold_send is not None:
            self.hold_send.wait(5.0)
        if on_sent:
            on_sent()
        if self.silent:
            raise slm_link.NoReply("silent")
        if cmd == "status":
            if self.in_progress:
                self._left -= 1
                if self._left <= 0:
                    self.in_progress, self.due = False, False
                    self.epoch += 1
            return {"status": "ok", "reinit_due": self.due, "due_for_s": 60.0 if self.due else None,
                    "reinit_in_progress": self.in_progress, "pattern_epoch": self.epoch,
                    "pattern": {"center_x": 321}, "last_reinit_error": self.error}
        if cmd == "reinit":
            self.in_progress, self._left = True, self.polls_to_finish
            return {"status": "queued"}
        raise AssertionError(cmd)


class Journal:
    def __init__(self):
        self.records = []

    def record(self, kind, **fields):
        self.records.append((kind, fields))

    def kinds(self):
        return [k for k, _ in self.records]


def _service(slm, blocker, **cfg):
    clock = types.SimpleNamespace(t=0.0)
    journal = Journal()
    config = SlmReinitConfig(host="127.0.0.1", port=1, **cfg)
    svc = SlmReinitService(config, blocker=blocker, link=slm, clock=lambda: clock.t,
                           journal=journal)
    return svc, clock, journal


def _run(svc, clock, until_t, step=0.5):
    while clock.t < until_t:
        svc.tick()
        clock.t += step


def test_service_asks_when_idle_and_due_then_follows_it_through():
    slm = FakeSlm()
    slm.due = True
    svc, clock, journal = _service(slm, lambda: "")
    svc.tick()                                    # t=0: polled, due, not settled yet
    assert svc.snapshot()["state"] == sr.DUE and "reinit" not in slm.requests
    _run(svc, clock, 4.9)
    assert "reinit" not in slm.requests           # settle_s = 5
    _run(svc, clock, 5.1)
    assert slm.requests.count("reinit") == 1
    assert svc.snapshot()["state"] == sr.REINITIALISING
    _run(svc, clock, 9.0)
    snap = svc.snapshot()
    assert snap["state"] == sr.IDLE and snap["pattern_epoch"] == 1
    assert snap["last_reinit"]["epoch"] == 1 and snap["last_reinit"]["pattern"] == {"center_x": 321}
    assert "slm_reinit_requested" in journal.kinds() and "slm_reinit_done" in journal.kinds()
    _run(svc, clock, 200.0)
    assert slm.requests.count("reinit") == 1      # not due any more


def test_service_never_asks_while_anything_could_use_the_slm():
    slm = FakeSlm()
    slm.due = True
    svc, clock, _ = _service(slm, lambda: "run 83344 (feedback) is starting")
    _run(svc, clock, 300.0)
    assert "reinit" not in slm.requests
    snap = svc.snapshot()
    assert snap["state"] == sr.DUE and snap["blocked_by"].startswith("run 83344")
    assert slm.requests.count("status") == 15      # every poll_s = 20 s


def test_service_idle_clock_restarts_when_the_machine_is_busy_in_between():
    slm = FakeSlm()
    slm.due = True
    why = [""]
    svc, clock, _ = _service(slm, lambda: why[0])
    _run(svc, clock, 3.0)
    why[0] = "the monitor is not ready"
    _run(svc, clock, 4.0)
    why[0] = ""
    _run(svc, clock, 8.5)                          # idle again since t=4, < 5 s
    assert "reinit" not in slm.requests
    _run(svc, clock, 9.6)
    assert slm.requests.count("reinit") == 1


def test_a_run_announced_after_the_last_tick_stops_the_request():
    slm = FakeSlm()
    slm.due = True
    why = [""]

    def blocker():
        return why[0]

    svc, clock, _ = _service(slm, blocker)
    _run(svc, clock, 5.0)
    # the run announces itself between the idle check of this tick and the request
    real = svc._blocker
    calls = []

    def racing():
        calls.append(1)
        return "" if len(calls) == 1 else "run 1 (x) is starting"

    svc._blocker = racing
    svc.tick()
    assert "reinit" not in slm.requests
    svc._blocker = real


def test_run_starting_waits_for_a_request_being_sent_and_no_longer():
    slm = FakeSlm()
    slm.due = True
    slm.hold_send = threading.Event()
    svc, clock, _ = _service(slm, lambda: "")
    _run(svc, clock, 5.0)
    ticker = threading.Thread(target=svc.tick)
    ticker.start()                                 # blocks inside the send
    assert _wait(lambda: svc._sending)
    returned = threading.Event()
    waiter = threading.Thread(target=lambda: (svc.run_starting(timeout=5.0), returned.set()))
    waiter.start()
    time.sleep(0.2)
    assert not returned.is_set()                   # the request is not on its way yet
    slm.hold_send.set()
    assert returned.wait(2.0)
    ticker.join(2.0)
    waiter.join(2.0)
    # bounded when the send hangs
    svc._sending = True
    t0 = time.monotonic()
    svc.run_starting(timeout=0.2)
    assert 0.15 <= time.monotonic() - t0 < 1.0
    svc._sending = False


def test_an_old_server_is_left_alone_and_probed_rarely():
    slm = FakeSlm()
    slm.silent = True
    svc, clock, journal = _service(slm, lambda: "")
    _run(svc, clock, 599.0)
    assert slm.requests == ["status"]
    assert svc.snapshot()["state"] == sr.NO_CONTROL
    _run(svc, clock, 601.0)
    assert slm.requests == ["status", "status"]
    assert "reinit" not in slm.requests


def test_no_answer_from_a_server_that_answered_before_is_not_old():
    slm = FakeSlm()
    svc, clock, _ = _service(slm, lambda: "")
    svc.tick()
    assert svc.snapshot()["state"] == sr.IDLE
    slm.silent = True
    _run(svc, clock, 21.0)
    assert svc.snapshot()["state"] == sr.UNREACHABLE
    slm.silent = False
    slm.due = True
    _run(svc, clock, 70.0)
    assert slm.requests.count("reinit") == 1


def test_a_poll_failure_while_reinitialising_is_ridden_out():
    slm = FakeSlm()
    slm.due = True
    slm.polls_to_finish = 1
    svc, clock, journal = _service(slm, lambda: "")
    _run(svc, clock, 5.1)
    assert svc.snapshot()["state"] == sr.REINITIALISING
    slm.silent = True                              # a run's write holds the connection
    _run(svc, clock, 10.0)
    assert svc.snapshot()["state"] == sr.REINITIALISING
    slm.silent = False
    _run(svc, clock, 12.0)
    assert svc.snapshot()["state"] == sr.IDLE
    assert "slm_reinit_done" in journal.kinds()


def test_a_reinit_that_never_finishes_fails_and_is_retried_later():
    slm = FakeSlm()
    slm.due = True
    slm.polls_to_finish = 10 ** 9
    svc, clock, journal = _service(slm, lambda: "")
    _run(svc, clock, 70.0)
    snap = svc.snapshot()
    assert snap["state"] == sr.FAILED and "not done" in snap["detail"]
    assert slm.requests.count("reinit") == 1
    slm.in_progress = False                       # the server gave up; still due
    _run(svc, clock, 600.0)
    assert slm.requests.count("reinit") == 1
    _run(svc, clock, 700.0)
    assert slm.requests.count("reinit") == 2
    assert "slm_reinit_failed" in journal.kinds()


# --- the monitor server's hooks --------------------------------------------------------

class Recorder:
    def __init__(self, *a, **k):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def server(qapp, monkeypatch, tmp_path):
    from waxx.util.guis import monitor_server_gui as msg
    monkeypatch.setattr(msg, "StateBroadcaster", Recorder)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"dds": {}, "dac": {}, "ttl": {}}))
    s = msg.MonitorUDPServer(config_file_path=str(path),
                             slm_reinit=SlmReinitConfig(host="127.0.0.1", port=_free_port()))
    yield s
    s.sock.close()


def test_monitor_server_idle_check(server):
    from waxx.util.comms_server.comm_server import STATES
    assert server.slm_reinit is not None
    assert "monitor" in server._slm_reinit_blocker()             # never started
    server.status.set_state(STATES.READY, "running")
    assert server._slm_reinit_blocker() == ""
    server._run_pending = {"run_id": 83344, "expt": "feedback", "t0": time.monotonic()}
    assert "run 83344 (feedback) is starting" == server._slm_reinit_blocker()
    server._run_pending = None
    server.reset._current = {"state": "running"}
    assert server._slm_reinit_blocker() == "a state reset is running"
    server.reset._current = None
    server.status.set_state(STATES.NOT_READY, "interrupted_by_run")
    assert server._slm_reinit_blocker() == "the monitor is not ready"


def test_run_pending_waits_for_the_slm_before_replying_and_status_json_reports_it(server):
    order = []

    class Spy:
        def run_starting(self, timeout=sr.SEND_WAIT_S):
            order.append(("slm", server._run_pending is not None))

        def snapshot(self):
            return {"state": "idle"}

        def stop(self):
            pass

    server.slm_reinit = Spy()
    real = server.connections.run_starting
    server.connections.run_starting = lambda name, *a, **k: order.append(("connections", name))
    try:
        reply = json.loads(server.generate_reply(json.dumps(
            {"type": "run_pending", "run_id": 7, "expt": "x", "token": "t"})))
    finally:
        server.connections.run_starting = real
    assert reply["status"] == "ok"
    assert order == [("slm", True), ("connections", "run 7 (x)")]
    detail = json.loads(server.generate_reply("status_json"))
    assert detail["slm_reinit"] == {"state": "idle"}


def test_journal_lines_for_the_reinit():
    from waxx.util.device_state.op_journal import describe_entry
    base = {"t": "2026-09-28T12:00:00"}
    assert "asked for" in describe_entry({**base, "kind": "slm_reinit_requested", "due_for_s": 600})
    assert "pattern put back" in describe_entry({**base, "kind": "slm_reinit_done", "t_s": 2.1,
                                                 "pattern": {"center_x": 1}})
    assert "FAILED" in describe_entry({**base, "kind": "slm_reinit_failed", "text": "x"})
