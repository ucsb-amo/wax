"""The SLM's reinit, offline (2026-09-28).

* the SLM server (``run_server.py``, Meadowlark calls stubbed, on 127.0.0.1):
  a reinit puts back the last pattern, the hourly timer only marks one due,
  ``SLMCTL`` control lines, and the old parser dropping them;
* ``slm_link.exchange`` and the experiment's ``SLM`` client: waits for
  "applied", raises when it cannot say the mask is up, falls back once for a
  server that never replies; before a run starts it has a due reinit done and
  waits for it (``reinit_if_due``);
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


# --- the run-start reinit (SLM.reinit_if_due) --------------------------------------------

def _due(host):
    with host.rs._reinit_lock:
        host.rs._reinit["due_since"] = time.monotonic()


def test_run_start_with_no_reinit_due_asks_once_and_goes(slm_host):
    rec = _slm(slm_host.port).reinit_if_due(poll_s=0.02)
    assert rec["result"] == "not_due" and rec["error"] == ""
    assert rec["before"]["pattern_epoch"] == 0 and rec["after"] is None
    assert slm_host.inits == 1                              # only the start-up one


def test_run_start_asks_for_a_due_reinit_and_waits_until_it_is_done(slm_host):
    _apply(slm_host, 11, 12)
    _due(slm_host)
    slm_host.init_s = 0.3
    t0 = time.monotonic()
    rec = _slm(slm_host.port).reinit_if_due(by="run start (test)", poll_s=0.02)
    assert time.monotonic() - t0 >= 0.3                    # returned only once it was done
    assert rec["result"] == "reinit_done" and rec["t_s"] >= 0.3
    assert rec["before"]["reinit_due"] is True and rec["after"]["pattern_epoch"] == 1
    assert slm_host.inits == 2
    assert slm_host.uploads[-1] == (11, 12, 10)            # the pattern is back
    st = _status(slm_host)
    assert st["reinit_due"] is False and st["reinit_in_progress"] is False


def test_run_start_waits_for_a_reinit_already_running_and_asks_for_no_second(slm_host):
    _due(slm_host)
    slm_host.init_s = 0.5
    with socket.create_connection(("127.0.0.1", slm_host.port)) as s:
        s.sendall(control_line({"cmd": "reinit", "seq": 1}).encode())
    assert _wait(lambda: _status(slm_host)["reinit_in_progress"])
    rec = _slm(slm_host.port).reinit_if_due(poll_s=0.02)
    assert rec["result"] == "waited" and rec["after"]["pattern_epoch"] == 1
    assert slm_host.inits == 2                              # start-up + the one it waited for


def test_a_failed_run_start_reinit_stops_a_run_that_uses_the_slm_only(slm_host, capsys):
    from waxx.control.slm.slm import SLMWriteError
    _due(slm_host)
    slm_host.fail_init = "SDK said no"
    with pytest.raises(SLMWriteError, match="SDK said no"):
        _slm(slm_host.port).reinit_if_due(required=True, poll_s=0.02)
    assert "so it does not start" in capsys.readouterr().out
    # still due, with the last failure's text on the server: a second failure
    # is told from a reinit not yet started by the queue, not by the text
    rec = _slm(slm_host.port).reinit_if_due(required=False, poll_s=0.02)
    assert rec["result"] == "failed" and "SDK said no" in rec["error"]
    assert "starts anyway" in capsys.readouterr().out
    assert slm_host.inits == 3


def test_a_run_start_reinit_that_does_not_finish_in_time(slm_host):
    _due(slm_host)
    slm_host.init_s = 1.0
    rec = _slm(slm_host.port).reinit_if_due(timeout_s=0.3, poll_s=0.02)
    assert rec["result"] == "timeout" and rec["after"]["reinit_in_progress"] is True


def test_run_start_never_stopped_by_an_old_or_unreachable_server(monkeypatch, capsys):
    from waxx.control.slm import slm as slm_mod
    monkeypatch.setattr(slm_mod, "SLM_FIRST_REPLY_S", 0.3)
    old = ScriptedServer(lambda msg: [])
    try:
        rec = _slm(old.port).reinit_if_due(required=True)
        assert rec["result"] == "no_control"
        assert "BLANK mask" in capsys.readouterr().out
    finally:
        old.close()
    rec = _slm(_free_port()).reinit_if_due(required=True)
    assert rec["result"] == "unreachable"


def test_run_start_counts_a_new_server_process_as_reinitialised(monkeypatch):
    from waxx.control.slm import slm as slm_mod
    from waxx.control.slm.slm import SLMWriteError

    def fake_link(statuses):
        sent = []

        def exchange(host, port, payload, **kw):
            if payload["cmd"] == "status":
                return statuses.pop(0) if len(statuses) > 1 else statuses[0]
            sent.append(payload)
            return {"status": "queued"}
        return exchange, sent

    due = {"status": "ok", "reinit_due": True, "reinit_in_progress": False,
           "instance": "a", "pattern_epoch": 4, "queue_len": 0, "slm_ready": True}
    new = {"status": "ok", "reinit_due": False, "reinit_in_progress": False,
           "instance": "b", "pattern_epoch": 0, "queue_len": 0, "slm_ready": True,
           "pattern": {"mask": "spot", "dimension": 5, "center_x": 1, "center_y": 2}}
    exchange, sent = fake_link([dict(due), dict(new)])
    monkeypatch.setattr(slm_mod.slm_link, "exchange", exchange)
    rec = _slm(1).reinit_if_due(by="run start (x)", poll_s=0.01)
    assert rec["result"] == "restarted" and rec["after"]["instance"] == "b"
    assert sent == [{"cmd": "reinit", "by": "run start (x)"}]

    broken = dict(new, slm_ready=False, slm_not_ready="the LUT did not load")
    exchange, _ = fake_link([dict(due), broken])
    monkeypatch.setattr(slm_mod.slm_link, "exchange", exchange)
    with pytest.raises(SLMWriteError, match="LUT did not load"):
        _slm(1).reinit_if_due(required=True, poll_s=0.01)


def test_the_run_start_hook_comes_before_init_run():
    """The camera arms and the run id is handed out at INIT_RUN: the wait for
    a reinit must come before it (waxx Expt.finish_prepare_wax)."""
    from waxx.base.expt import Expt
    src = inspect.getsource(Expt.finish_prepare_wax)
    assert src.index("self.pre_init_run()") < src.index("_client.init_run(")


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
        self.next_due_in_s = 1800.0
        self.by = []                   # the "by" of each reinit request

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
                    "next_due_in_s": self.next_due_in_s, "interval_s": 3600,
                    "reinit_in_progress": self.in_progress, "pattern_epoch": self.epoch,
                    "pattern": {"center_x": 321}, "last_reinit_error": self.error}
        if cmd == "reinit":
            self.by.append(payload.get("by"))
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
    assert "due for 10 min" in describe_entry({**base, "kind": "slm_reinit_requested",
                                               "due_for_s": 600})
    assert "asked for by jp on kong" in describe_entry({**base, "kind": "slm_reinit_requested",
                                                        "by": "jp on kong", "due_for_s": None})
    assert "pattern put back" in describe_entry({**base, "kind": "slm_reinit_done", "t_s": 2.1,
                                                 "pattern": {"center_x": 1}})
    assert "FAILED" in describe_entry({**base, "kind": "slm_reinit_failed", "text": "x"})
    assert "asked for by kong" in describe_entry({**base, "kind": "slm_reinit_manual",
                                                  "by": "kong"})
    assert "REFUSED: busy" in describe_entry({**base, "kind": "slm_reinit_refused", "by": "kong",
                                              "msg": "busy"})


# --- the next due time and an operator's reinit (2026-09-28, the SLM pill) --------------

def test_server_reports_the_next_due_time_and_a_reinit_restarts_the_hour(slm_host, monkeypatch):
    rs = slm_host.rs
    monkeypatch.setattr(rs, "REINIT_INTERVAL_SEC", 100.0)
    assert _status(slm_host)["next_due_in_s"] is None           # timer not running yet
    stop = threading.Event()
    t = threading.Thread(target=rs.periodic_reinit_scheduler, args=(stop,), daemon=True)
    t.start()
    try:
        assert _wait(lambda: _status(slm_host)["next_due_in_s"] is not None)
        st = _status(slm_host)
        assert 95.0 <= st["next_due_in_s"] <= 100.0 and st["interval_s"] == 100.0
        rs._reinit["next_due"] = time.monotonic() + 3.0         # as if 97 s had passed
        assert _status(slm_host)["next_due_in_s"] <= 3.0
        _ask(slm_host, {"cmd": "reinit"}, ("reinit_done", "error"), control=True)
        st = _status(slm_host)
        assert 95.0 <= st["next_due_in_s"] <= 100.0              # the hour restarted
        assert st["reinit_due"] is False
    finally:
        stop.set()
        t.join(2.0)


def test_timer_catches_up_by_whole_intervals(slm_host, monkeypatch):
    rs = slm_host.rs
    monkeypatch.setattr(rs, "REINIT_INTERVAL_SEC", 10.0)
    rs._reinit["next_due"] = time.monotonic() - 35.0             # a stalled loop
    stop = threading.Event()
    t = threading.Thread(target=rs.periodic_reinit_scheduler, args=(stop,), daemon=True)
    t.start()
    try:
        assert _wait(lambda: _status(slm_host)["reinit_due"])
        st = _status(slm_host)
        assert 0.0 < st["next_due_in_s"] <= 10.0                 # one step, not four
        assert slm_host.inits == 1
    finally:
        stop.set()
        t.join(2.0)


def test_snapshot_has_the_next_due_wall_time():
    slm = FakeSlm()
    slm.next_due_in_s = 1200.0
    svc, clock, _ = _service(slm, lambda: "")
    t0 = time.time()
    svc.tick()
    snap = svc.snapshot()
    assert t0 + 1199.0 <= snap["next_due_at"] <= time.time() + 1200.0
    assert snap["interval_s"] == 3600


def test_request_now_is_sent_at_once_even_when_not_due_and_names_who_asked():
    slm = FakeSlm()                                  # not due
    svc, clock, journal = _service(slm, lambda: "")
    svc.tick()
    assert svc.snapshot()["state"] == sr.IDLE
    assert svc.request_now("jp on kong") == ""
    assert svc.snapshot()["manual_pending"] == "jp on kong"
    assert svc.request_now("someone else").startswith("a reinit asked for by jp on kong")
    svc.tick()                                       # same instant: no settle time
    assert slm.requests.count("reinit") == 1
    assert slm.by == ["jp on kong (through the monitor server)"]
    snap = svc.snapshot()
    assert snap["state"] == sr.REINITIALISING and snap["manual_pending"] is None
    assert snap["last_manual"]["result"] == "sent"
    assert ("slm_reinit_requested", "jp on kong") in [(k, f.get("by")) for k, f in journal.records]


def test_request_now_ignores_the_pause_after_a_failure():
    slm = FakeSlm()
    slm.due = True
    slm.polls_to_finish = 10 ** 9
    svc, clock, _ = _service(slm, lambda: "")
    _run(svc, clock, 70.0)
    assert svc.snapshot()["state"] == sr.FAILED
    slm.in_progress = False
    assert svc.request_now("kong") == ""
    svc.tick()
    assert slm.requests.count("reinit") == 2


@pytest.mark.parametrize("setup, text", [
    (lambda slm, why: why.__setitem__(0, "run 9 (x) is starting"), "run 9 (x) is starting"),
    (lambda slm, why: setattr(slm, "silent", True), "no control commands"),
])
def test_request_now_refused(setup, text):
    slm = FakeSlm()
    why = [""]
    svc, clock, _ = _service(slm, lambda: why[0])
    assert "not answered yet" in svc.request_now("kong")      # never polled
    setup(slm, why)
    svc.tick()
    assert text in svc.request_now("kong")
    svc.tick()
    assert "reinit" not in slm.requests


def test_request_now_refused_when_a_run_announces_itself_before_it_is_sent():
    slm = FakeSlm()
    why = [""]
    changes = []
    svc, clock, journal = _service(slm, lambda: why[0])
    svc._on_change = changes.append
    svc.tick()
    assert svc.request_now("kong") == ""
    why[0] = "run 11 (x) is starting"                 # between the request and the tick
    svc.tick()
    assert "reinit" not in slm.requests
    snap = svc.snapshot()
    assert snap["manual_pending"] is None
    assert snap["last_manual"]["result"] == "not sent: run 11 (x) is starting"
    assert ("slm_reinit_refused", "kong") in [(k, f.get("by")) for k, f in journal.records]
    assert changes and changes[-1]["last_manual"]["result"].startswith("not sent")


def test_monitor_server_slm_reinit_request(server):
    from waxx.util.comms_server.comm_server import STATES
    ask = lambda obj: json.loads(server.generate_reply(json.dumps(obj)))  # noqa: E731
    server.slm_reinit._state = sr.IDLE               # as after a poll
    reply = ask({"type": "slm_reinit", "action": "reinit", "client": "kong"})
    assert reply["status"] == "error" and "monitor is not ready" in reply["msg"]
    server.status.set_state(STATES.READY, "running")
    reply = ask({"type": "slm_reinit", "action": "reinit", "client": "kong", "operator": "jp"})
    assert reply == {"status": "ok"}
    assert server.slm_reinit.snapshot()["manual_pending"] == "jp on kong"
    assert ask({"type": "slm_reinit", "action": "status"})["slm_reinit"]["manual_pending"] \
        == "jp on kong"
    assert ask({"type": "slm_reinit", "action": "explode"})["status"] == "error"
    server.slm_reinit = None
    assert "no SLM reinit" in ask({"type": "slm_reinit", "action": "reinit"})["msg"]


# --- the SLM pill -----------------------------------------------------------------------

def _pill(qapp, can_launch=True):
    from waxx.util.guis.slm_pill import SlmPill
    return SlmPill(can_launch=can_launch)


def _menu_texts(pill):
    menu = pill.build_menu()
    return [(a.text(), a.isEnabled()) for a in menu.actions() if not a.isSeparator()], menu


def _action(menu, name):
    return next(a for a in menu.actions() if a.objectName() == name)


def test_pill_shows_the_state_and_the_next_reinit_time(qapp):
    from waxx.util.guis import slm_pill
    pill = _pill(qapp)
    now = time.time()
    pill.set_snapshot({"state": "idle", "next_due_at": now + 38 * 60, "blocked_by": "",
                       "last_reinit": {"at": now - 1200, "t_s": 2.1}})
    assert pill.text() == "SLM" and slm_pill.LOOK["idle"][0] in pill.styleSheet()
    at = time.strftime("%H:%M", time.localtime(now + 38 * 60))
    texts, menu = _menu_texts(pill)
    assert (f"Next reinit due {at} (in 38 min)", False) in texts
    assert any(t.startswith("Last reinit") and "2.1 s" in t for t, _ in texts)
    assert ("Re-initialise SLM now…", True) in texts
    assert ("Launch spot finder", True) in texts
    assert f"Next reinit due {at}" in pill.toolTip()


def test_pill_menu_actions_emit_and_disabled_reasons_show(qapp):
    pill = _pill(qapp)
    got = []
    pill.reinit_requested.connect(lambda: got.append("reinit"))
    pill.spot_finder_requested.connect(lambda: got.append("launch"))
    pill.set_snapshot({"state": "due", "reinit_due": True, "due_for_s": 120.0,
                       "blocked_by": "run 83344 (feedback) is starting"})
    assert pill.text() == "SLM · due"
    texts, menu = _menu_texts(pill)
    assert _action(menu, "slm_reinit_now").isEnabled() is False
    assert any("not now: the machine is not idle: run 83344" in t for t, _ in texts)
    assert any(t.startswith("Reinit due since") and "run 83344" in t for t, _ in texts)
    _action(menu, "slm_launch_spot_finder").trigger()
    pill.set_snapshot({"state": "due", "reinit_due": True, "blocked_by": ""})
    _, menu = _menu_texts(pill)
    _action(menu, "slm_reinit_now").trigger()
    assert got == ["launch", "reinit"]


def test_pill_before_the_monitor_server_reports_it(qapp):
    from PyQt6.QtWidgets import QWidget
    host = QWidget()                              # never shown: a pill in a status row
    pill = _pill(qapp, can_launch=True)
    pill.setParent(host)
    pill.set_snapshot(None, reported=False)
    assert not pill.isHidden()                    # shown with the row: it can launch
    texts, menu = _menu_texts(pill)
    assert ("SLM reinit: not reported by the monitor server", False) in texts
    assert _action(menu, "slm_reinit_now").isEnabled() is False
    assert _action(menu, "slm_launch_spot_finder").isEnabled() is True
    pill.set_reachable(False)
    assert "monitor server unreachable" in pill.toolTip()
    assert pill.reinit_allowed()[1] == "the monitor server is unreachable"
    hidden = _pill(qapp, can_launch=False)
    hidden.setParent(host)
    hidden.set_snapshot(None, reported=False)
    assert hidden.isHidden()                      # nothing to offer
    hidden.set_snapshot({"state": "idle"})
    assert not hidden.isHidden()
    lone = _pill(qapp)                            # parentless: never shows itself
    lone.set_snapshot({"state": "idle"})
    assert not lone.isVisible()


@pytest.mark.parametrize("state, allowed", [("idle", True), ("due", True), ("failed", True),
                                             ("unreachable", True), ("reinitialising", False),
                                             ("no_control", False), ("unknown", False)])
def test_pill_reinit_allowed_by_state(qapp, state, allowed):
    pill = _pill(qapp)
    pill.set_snapshot({"state": state, "blocked_by": ""})
    assert pill.reinit_allowed()[0] is allowed


# --- the Device Control GUI -------------------------------------------------------------

@pytest.fixture
def dcgui(qapp, monkeypatch):
    from waxx.util.guis import device_control_gui as dc
    from waxx.util.guis import sequences_panel
    from test_composite_panel import FakeSender
    from test_device_control_gui import FakeSettings
    monkeypatch.setattr(dc, "QSettings", FakeSettings)
    monkeypatch.setattr(sequences_panel, "_OpSender", FakeSender)
    for name in ("_setup_update_sender", "_setup_state_listener", "_setup_state_worker",
                 "setup_status_checker", "setup_timer", "request_state"):
        monkeypatch.setattr(dc.DeviceStateGUI, name, lambda self, *a, **k: None)
    launched = []

    def launcher():
        launched.append(1)
        return "console pid 1 on test-pc"

    g = dc.DeviceStateGUI(spot_finder_launcher=launcher)
    g._test_launched = launched
    g._test_sent = []
    monkeypatch.setattr(g, "_send_request", lambda obj, cb: (g._test_sent.append(obj),
                                                             cb({"status": "ok"})))
    yield g
    g.close()


def test_gui_feeds_the_pill_from_status_and_broadcasts(dcgui):
    pill = dcgui.slm_pill
    assert pill is not None and pill.can_launch
    dcgui._on_status_detail({"state": 0, "slm_reinit": {"state": "due", "reinit_due": True}})
    assert pill.reported and pill.text() == "SLM · due"
    dcgui._on_state_broadcast({"type": "slm_reinit", "slm_reinit": {"state": "reinitialising"}})
    assert pill.text() == "SLM · reinit…"
    dcgui._on_status_detail({"state": 0})                    # an older server
    assert pill.reported is False
    dcgui.on_connection_failed()
    assert pill.reachable is False


def test_gui_reinit_asks_then_sends(dcgui, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    dcgui._on_status_detail({"state": 0, "slm_reinit": {"state": "idle", "blocked_by": ""}})
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: next(b for b in self.buttons() if b.text() == "Cancel"))
    dcgui._request_slm_reinit()
    assert dcgui._test_sent == []                            # cancelled
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: next(b for b in self.buttons()
                                          if b.text() == "Re-initialise"))
    dcgui._request_slm_reinit()
    assert dcgui._test_sent[-1]["type"] == "slm_reinit"
    assert dcgui._test_sent[-1]["action"] == "reinit"
    assert any("[slm] reinit asked for" in line for line in dcgui._changes)


def test_gui_reinit_not_sent_when_not_allowed(dcgui, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    dcgui._on_status_detail({"state": 0, "slm_reinit": {"state": "idle",
                                                        "blocked_by": "a state reset is running"}})
    dcgui._request_slm_reinit()
    assert dcgui._test_sent == [] and "a state reset is running" in warned[0]


def test_gui_launches_the_spot_finder_and_reports_failures(dcgui, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    dcgui._launch_spot_finder()
    assert dcgui._test_launched == [1]
    assert any("spot finder started (console pid 1 on test-pc)" in line
               for line in dcgui._changes)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))

    def broken():
        raise FileNotFoundError("no spot finder at X")

    dcgui._spot_finder_launcher = broken
    dcgui._launch_spot_finder()
    assert "no spot finder at X" in warned[0]
    assert any("spot finder did not start" in line for line in dcgui._changes)
