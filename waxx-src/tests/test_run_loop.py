"""The monitor server's run loop (waxx.util.device_state.run_loop).  Pure
Python: the experiment processes and liveOD's POLL replies are fakes; nothing
is launched and nothing goes on the network."""
import threading
import time

import pytest

from waxx.util.device_state.run_loop import LoopSpec, RunLoop, loop_specs


class FakeLive:
    """liveOD as POLL shows it."""

    def __init__(self):
        self.run_in_progress = False
        self.run_id = None
        self.reset_requested = False
        self.last_outcome = {}
        self.polls = 0

    def __call__(self):
        self.polls += 1
        return {"ok": True, "run_in_progress": self.run_in_progress, "run_id": self.run_id,
                "expt_name": "someone_else", "reset_requested": self.reset_requested,
                "last_outcome": dict(self.last_outcome)}


class FakeProc:
    """One experiment process: prints ``lines`` (after ``before`` runs, if
    given), records ``outcome`` in liveOD for its run id, exits ``code``."""

    def __init__(self, live, run_id=None, lines=(), code=0, outcome="saved", before=None,
                 hold: threading.Event | None = None):
        self.pid = 1234
        self.code = code
        self._live, self._run_id, self._outcome = live, run_id, outcome
        self._lines, self._before, self._hold = list(lines), before, hold
        self.stdout = self._out()

    def _out(self):
        if self._before is not None:
            self._before()
        if self._run_id is not None:
            yield f"Run ID: {self._run_id}\n"
        for line in self._lines:
            yield line + "\n"
        if self._hold is not None:
            self._hold.wait(5)
        if self._run_id is not None and self._outcome:
            self._live.last_outcome = {"run_id": self._run_id, "outcome": self._outcome,
                                       "detail": "2 frames missing"
                                       if self._outcome != "saved" else ""}

    def wait(self):
        return self.code


class Journal:
    def __init__(self):
        self.kinds = []

    def record(self, kind, **fields):
        self.kinds.append(kind)


@pytest.fixture
def expt(tmp_path):
    path = tmp_path / "auto_tof.py"
    path.write_text('"""BEC TOF loop.\n\nt_tof 1-4 ms, 9 points."""\nclass auto_tof: pass\n')
    return path


def _loop(expt, live, procs, **kw):
    started, journal = [], Journal()
    queue = list(procs)

    def spawn(command):
        assert command.endswith(str(expt))
        item = queue.pop(0)
        return item() if callable(item) else item

    loop = RunLoop(LoopSpec("auto_tof", "BEC TOF loop", str(expt)), poll=live, spawn=spawn,
                   start_monitor=started.append, journal=journal, gap_s=0., poll_s=0.01,
                   **kw)
    loop.started_monitor, loop.journal = started, journal
    return loop


def test_runs_back_to_back_until_stop_then_starts_the_monitor(expt):
    live = FakeLive()
    loop = None

    def third():
        loop.stop(operator="jp", client="kong")           # pressed during run 3
        return FakeProc(live, 103)

    loop = _loop(expt, live, [FakeProc(live, 101), FakeProc(live, 102), third])
    assert loop.start(operator="jp", client="kong")["status"] == "ok"
    loop.join(5)
    info = loop.info()
    assert info["state"] == "stopped" and info["runs"] == 3
    assert "stopped by jp@kong after 3 saved runs" in info["text"]
    assert info["last"]["run_id"] == 103
    assert len(loop.started_monitor) == 1
    assert loop.journal.kinds.count("run_loop_run") == 3
    assert loop.journal.kinds[-1] == "run_loop_end"


def test_an_abort_in_live_od_latches_it_off(expt):
    live = FakeLive()
    aborted = FakeProc(live, 102, ["Run 102 reset -- aborting.",
                                   "RuntimeError: Acquisition for run 102 aborted."],
                       code=1, outcome="discarded")
    loop = _loop(expt, live, [FakeProc(live, 101), aborted])
    loop.start()
    loop.join(5)
    info = loop.info()
    assert info["state"] == "latched" and info["runs"] == 1
    assert "run 102 was aborted in liveOD" in info["text"]
    assert info["last"] == {"run_id": 102, "outcome": "failed", "exit_code": 1,
                            "ended": pytest.approx(time.time(), abs=10)}
    assert loop.started_monitor                   # "start unless running"


def test_a_run_saved_incomplete_latches_it_off(expt):
    live = FakeLive()
    loop = _loop(expt, live, [FakeProc(live, 101, outcome="saved_incomplete")])
    loop.start()
    loop.join(5)
    info = loop.info()
    assert info["state"] == "latched" and info["runs"] == 0
    assert "run 101 ended saved_incomplete (2 frames missing)" in info["text"]


def test_a_failure_keeps_the_last_lines_and_names_the_cause(expt):
    live = FakeLive()
    loop = _loop(expt, live, [FakeProc(live, None, ["Traceback (most recent call last):",
                                                    "ModuleNotFoundError: No module named x"],
                                       code=1)])
    loop.start()
    loop.join(5)
    info = loop.info()
    assert info["state"] == "latched"
    assert "exit code 1" in info["text"] and "likely cause" in info["text"]
    assert info["tail"][-1] == "ModuleNotFoundError: No module named x"


def test_losing_the_core_latches_without_starting_the_monitor(expt):
    live = FakeLive()
    loop = _loop(expt, live, [FakeProc(live, 101, ["ConnectionResetError: [WinError 10054] "
                                                   "An existing connection was forcibly "
                                                   "closed by the remote host"], code=1)])
    loop.start()
    loop.join(5)
    assert loop.info()["state"] == "latched"
    assert "lost the core device" in loop.info()["text"]
    assert loop.started_monitor == []


@pytest.mark.parametrize("setup, why", [
    (lambda live: setattr(live, "run_in_progress", True), "is in progress in liveOD"),
    (lambda live: setattr(live, "reset_requested", True), "Abort is pending"),
])
def test_start_is_refused_while_the_machine_is_not_free(expt, setup, why):
    live = FakeLive()
    setup(live)
    loop = _loop(expt, live, [])
    reply = loop.start()
    assert reply["status"] == "error" and why in reply["msg"]
    assert loop.info()["state"] == "idle"


def test_start_is_refused_for_a_fence_a_state_reset_or_a_missing_file(expt, tmp_path):
    live = FakeLive()
    fenced = _loop(expt, live, [], fence=lambda: {"run_id": 900, "expt": "rabi"})
    assert "run 900 (rabi) announced itself" in fenced.start()["msg"]
    busy = _loop(expt, live, [], busy=lambda: "a state reset is running")
    assert busy.start()["msg"] == "a state reset is running"
    gone = RunLoop(LoopSpec("x", "X", str(tmp_path / "nope.py")), poll=live)
    assert "does not exist" in gone.start()["msg"]


def test_someone_elses_run_between_runs_latches_without_the_monitor(expt):
    live = FakeLive()
    first = FakeProc(live, 101)
    output = first.stdout

    def then_a_person_starts_a_run():
        yield from output
        live.run_in_progress, live.run_id = True, 555

    first.stdout = then_a_person_starts_a_run()
    loop = _loop(expt, live, [first])
    loop.start()
    loop.join(5)
    info = loop.info()
    assert info["state"] == "latched" and info["runs"] == 1
    assert info["text"] == "run 555 (someone_else) is in progress in liveOD"
    assert loop.started_monitor == []                  # never take the core from it


def test_an_abort_pressed_while_a_run_starts_is_not_lost(expt):
    live = FakeLive()

    def starting():
        live.reset_requested = True                        # pressed before INIT_RUN
        time.sleep(0.1)                                    # (the loop polls meanwhile)
        live.reset_requested = False                       # INIT_RUN spends it

    loop = _loop(expt, live, [FakeProc(live, 101, before=starting), FakeProc(live, 102)])
    loop.start()
    loop.join(5)
    info = loop.info()
    assert info["state"] == "latched" and info["runs"] == 1
    assert "spent it on no run" in info["text"]


def test_the_monitor_being_started_ends_the_loop_latched(expt):
    live = FakeLive()
    hold = threading.Event()
    loop = _loop(expt, live, [FakeProc(live, 101, hold=hold), FakeProc(live, 102)])
    loop.start()
    time.sleep(0.05)
    loop.note_external("a client asked for the monitor")
    hold.set()
    loop.join(5)
    info = loop.info()
    assert info["state"] == "latched" and info["runs"] == 1
    assert info["text"] == "a client asked for the monitor"
    assert loop.started_monitor == []


def test_start_twice_and_stop_when_idle_are_refused(expt):
    live = FakeLive()
    hold = threading.Event()
    loop = _loop(expt, live, [FakeProc(live, 101, hold=hold)])
    assert loop.stop()["status"] == "error"
    loop.start()
    assert "already running" in loop.start()["msg"]
    assert loop.stop()["status"] == "ok"
    assert loop.info()["state"] == "stopping"
    hold.set()
    loop.join(5)
    assert loop.info()["state"] == "stopped"


# --- in the monitor server ---------------------------------------------------------

class Broadcasts:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        pass


@pytest.fixture(scope="module")
def qapp():
    import os
    from PyQt6.QtWidgets import QApplication
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])      # held: signals need it


@pytest.fixture
def server(qapp, monkeypatch, tmp_path, expt):
    import json
    from waxx.util.guis import monitor_server_gui as msg
    monkeypatch.setattr(msg, "StateBroadcaster", Broadcasts)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    s = msg.MonitorUDPServer(config_file_path=str(tmp_path / "state.json"),
                             run_loops=[LoopSpec("auto_tof", "BEC TOF loop", str(expt))])
    s.live = FakeLive()
    s.loops["auto_tof"]._poll = s.live
    s.loops["auto_tof"]._gap_s, s.loops["auto_tof"]._poll_s = 0., 0.01
    s.monitor_starts = []
    s.start_monitor_signal.connect(s.monitor_starts.append)
    s.ask = lambda obj: json.loads(s.generate_reply(json.dumps(obj)))
    yield s
    s.loops["auto_tof"].join(5)
    s.sock.close()


def test_server_offers_only_its_own_loops_and_reports_them(server):
    import json
    status = json.loads(server.generate_reply("status_json"))
    info = status["run_loops"]["auto_tof"]
    assert info["state"] == "idle" and info["title"] == "BEC TOF loop"
    assert info["about"].startswith("BEC TOF loop.")
    reply = server.ask({"type": "run_loop", "action": "start", "loop": "../../evil"})
    assert reply["status"] == "error" and "offered: auto_tof" in reply["msg"]
    server.live.run_in_progress, server.live.run_id = True, 7
    reply = server.ask({"type": "run_loop", "action": "start", "loop": "auto_tof"})
    assert reply["status"] == "error" and "run 7" in reply["msg"]


def test_server_loop_blocks_a_state_reset_and_ends_on_a_monitor_restart(server):
    hold = threading.Event()
    loop = server.loops["auto_tof"]
    loop._spawn = lambda command: FakeProc(server.live, 81001, hold=hold)
    reply = server.ask({"type": "run_loop", "action": "start", "loop": "auto_tof",
                        "operator": "jp", "client": "kong"})
    assert reply["status"] == "ok" and reply["loop"]["state"] == "running"
    assert "is running" in server.ask({"type": "reset_state"})["msg"]
    server.on_message_received("reset")                   # someone restarts the monitor
    hold.set()
    loop.join(5)
    from PyQt6.QtWidgets import QApplication
    QApplication.processEvents()
    assert loop.info()["state"] == "latched" and "(re)start the monitor" in loop.info()["text"]
    assert server.monitor_starts == []                    # it is being started already
    assert any(p.get("type") == "run_loop" for p in server._broadcaster.sent)


def test_server_stop_finishes_the_run_then_asks_for_the_monitor(server):
    hold = threading.Event()
    loop = server.loops["auto_tof"]
    loop._spawn = lambda command: FakeProc(server.live, 81001, hold=hold)
    server.ask({"type": "run_loop", "action": "start", "loop": "auto_tof"})
    reply = server.ask({"type": "run_loop", "action": "stop", "loop": "auto_tof",
                        "operator": "jp"})
    assert reply["loop"]["state"] == "stopping"
    hold.set()
    loop.join(5)
    from PyQt6.QtWidgets import QApplication
    QApplication.processEvents()            # emitted from the loop's thread: queued
    assert loop.info()["state"] == "stopped" and loop.info()["runs"] == 1
    assert len(server.monitor_starts) == 1 and "BEC TOF loop ended" in server.monitor_starts[0]


def test_info_describes_the_experiment_and_specs_skip_unset_paths(expt):
    loop = RunLoop(LoopSpec("auto_tof", "BEC TOF loop", str(expt)), poll=FakeLive())
    info = loop.info()
    assert info["about"] == "BEC TOF loop.\n\nt_tof 1-4 ms, 9 points."
    assert info["expt"] == "auto_tof" and info["state"] == "idle"
    assert loop_specs({"a": ("A", str(expt)), "b": ("B", None)}) == \
        [LoopSpec("a", "A", str(expt))]
