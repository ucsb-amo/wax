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


# --- a run whose process dies without telling liveOD --------------------------------

class DyingLive(FakeLive):
    """liveOD with run ``run_id`` of ``expt`` still in progress after its process died;
    RUN_EXITED sent on its behalf lands in ``notices`` and closes it."""

    def __init__(self, run_id, run_state="running", expt="auto_tof.py", age=1.0):
        super().__init__()
        self.run_in_progress, self.run_id = True, run_id
        self.run_state, self.expt, self.age = run_state, expt, age
        self.notices = []

    def __call__(self):
        reply = super().__call__()
        reply.update(run_state=self.run_state, expt_name=self.expt, init_run_age_s=self.age)
        return reply

    def run_exited(self, run_id, reason):
        self.notices.append((run_id, reason))
        self.run_in_progress, self.run_state = False, "exited"
        return {"ok": True}


def _dies(live, run_id, when_ready=None):
    """The process: liveOD is idle until it starts (the gate), then it dies hard."""
    live.run_in_progress = False

    def start():
        live.run_in_progress = True
    return FakeProc(live, run_id, ["Segmentation fault"], code=3221225477, outcome=None,
                    before=start)


def test_a_run_killed_hard_is_reported_to_live_od_by_the_loop(expt):
    live = DyingLive(101)
    loop = _loop(expt, live, [_dies(live, 101)])
    loop.start()
    loop.join(5)
    assert [n[0] for n in live.notices] == [101]
    assert "exit code 3221225477" in live.notices[0][1] and "Segmentation fault" in live.notices[0][1]
    assert loop.info()["state"] == "latched" and "exit code 3221225477" in loop.info()["text"]
    assert "run_loop_told_live_od_exited" in loop.journal.kinds
    assert any("without telling liveOD" in line for line in loop.output.since()["lines"])


@pytest.mark.parametrize("live_kw, run_id", [
    ({"run_state": "exited"}, 101),        # its own notice got through
    ({}, 102),                             # liveOD has another run
    ({"run_state": "saving"}, 101),        # liveOD is saving it: never cut that short
])
def test_no_notice_when_live_od_knows_or_has_another_run(expt, live_kw, run_id):
    live = DyingLive(101, **live_kw)
    loop = _loop(expt, live, [_dies(live, run_id)])
    loop.start()
    loop.join(5)
    assert live.notices == []


@pytest.mark.parametrize("expt_name, age, told", [
    ("auto_tof.py", 0.0, True),            # ours: our experiment, started after the launch
    ("rabi.py", 0.0, False),               # someone else's experiment
    ("auto_tof.py", 1e6, False),           # older than this launch
])
def test_a_run_killed_before_its_run_id_line_is_matched_by_name_and_age(expt, expt_name,
                                                                        age, told):
    live = DyingLive(101, expt=expt_name, age=age)
    loop = _loop(expt, live, [_dies(live, None)])
    loop.start()
    loop.join(5)
    assert [n[0] for n in live.notices] == ([101] if told else [])


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


def test_the_runs_output_is_kept_with_a_line_per_run_and_one_at_the_end(expt):
    live = FakeLive()
    loop = _loop(expt, live, [FakeProc(live, 101, ["compiling", "shot 1/9"]),
                              FakeProc(live, 102, ["RuntimeError: boom"], code=1, outcome="")])
    loop.start()
    loop.join(5)
    out = loop.output.since(0)
    lines = out["lines"]
    assert lines[0].startswith("── ") and "1st run of the loop: auto_tof" in lines[0]
    assert lines[1:4] == ["Run ID: 101", "compiling", "shot 1/9"]
    second = next(i for i, line in enumerate(lines) if "2nd run of the loop" in line)
    assert lines[second + 1:second + 3] == ["Run ID: 102", "RuntimeError: boom"]
    assert "LATCHED OFF after 1 saved run" in lines[-1]
    assert loop.output.since(out["next"] - 1)["lines"] == []


def test_server_serves_the_output_of_its_own_loops_only(server):
    server.loops["auto_tof"].output.append("hello")
    reply = server.ask({"type": "output", "kind": "run_loop", "key": "auto_tof", "after": 0})
    assert reply["status"] == "ok" and reply["lines"] == ["hello"] and reply["next"] == 2
    reply = server.ask({"type": "output", "kind": "run_loop", "key": "auto_tof", "after": 1})
    assert reply["lines"] == [] and reply["next"] == 2
    reply = server.ask({"type": "output", "kind": "run_loop", "key": "../../evil"})
    assert reply["status"] == "error" and "no run loop" in reply["msg"]
    reply = server.ask({"type": "output", "kind": "reset"})
    assert reply["status"] == "error" and "no reset experiment" in reply["msg"]
    assert server.ask({"type": "output", "kind": "shell"})["status"] == "error"


# --- pick loops (the file chosen at Start) ------------------------------------------

@pytest.fixture
def pick_root(tmp_path):
    root = tmp_path / "experiments"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "monitor.py").write_text("class monitor: pass\n")
    (root / "JP").mkdir()
    (root / "JP" / "rabi.py").write_text('"""Rabi flop, 17 points."""\nclass rabi: pass\n')
    (root / "JP" / "notes.txt").write_text("not an experiment\n")
    (tmp_path / "outside.py").write_text("class outside: pass\n")
    return root


def _pick_loop(root, live, procs):
    queue = list(procs)
    spawned = []

    def spawn(command):
        spawned.append(command)
        return queue.pop(0)

    spec = loop_specs({"expt_loop": {"title": "Experiment loop", "root": str(root),
                                     "exclude": (str(root / "tools" / "monitor.py"),)}})[0]
    loop = RunLoop(spec, poll=live, spawn=spawn, start_monitor=lambda why: None,
                   gap_s=0., poll_s=0.01)
    loop.spawned = spawned
    return loop


def test_loop_specs_reads_a_pick_entry_and_skips_one_without_a_root(pick_root):
    specs = loop_specs({"a": ("A", "x.py"), "p": {"title": "P", "root": str(pick_root)},
                        "none": {"title": "N", "root": None}})
    assert [s.key for s in specs] == ["a", "p"]
    assert specs[1].pick and not specs[0].pick and specs[1].expt_path == ""


def test_pick_loop_refuses_files_it_must_not_run(pick_root):
    loop = _pick_loop(pick_root, FakeLive(), [])
    info = loop.info()
    assert info["pick"] and info["expt"] == "" and info["path"] == ""
    for path, why in [(None, "no experiment file"), ("JP/notes.txt", "not a Python file"),
                      ("JP/missing.py", "no such file"), ("../outside.py", "only experiments inside"),
                      (str(pick_root.parent / "outside.py"), "only experiments inside"),
                      ("tools/monitor.py", "cannot be looped"), ("JP/a&b.py", "not allowed")]:
        reply = loop.start(path=path)
        assert reply["status"] == "error" and why in reply["msg"], (path, reply)
        assert loop.describe(path)["status"] == "error"
    assert loop.spawned == [] and loop.info()["state"] == "idle"


def test_pick_loop_describes_and_runs_the_chosen_file(pick_root):
    live = FakeLive()
    loop = _pick_loop(pick_root, live, [FakeProc(live, 201, code=1)])
    d = loop.describe("JP/rabi.py")
    assert d["status"] == "ok" and d["expt"] == "rabi" and d["rel"] == "JP/rabi.py"
    assert "Rabi flop" in d["about"]
    assert loop.start(operator="jp", client="kong", path=r"JP\rabi.py")["status"] == "ok"
    loop.join(5)
    info = loop.info()
    assert info["expt"] == "rabi" and info["rel"] == "JP/rabi.py" and "Rabi flop" in info["about"]
    assert loop.spawned[0].endswith(str((pick_root / "JP" / "rabi.py").resolve()))
    assert info["state"] == "latched"         # its one run failed (exit code 1)


def test_pick_loop_quotes_a_path_with_spaces(pick_root):
    (pick_root / "M testing").mkdir()
    (pick_root / "M testing" / "x.py").write_text("class x: pass\n")
    live = FakeLive()
    loop = _pick_loop(pick_root, live, [FakeProc(live, 301, code=1)])
    assert loop.start(path="M testing/x.py")["status"] == "ok"
    loop.join(5)
    assert loop.spawned[0].endswith('x.py"') and ' "' in loop.spawned[0]


# --- scan settings (the card's ⚙) --------------------------------------------------

TOF_SCAN = {"xvar": "t_tof", "unit": "ms", "scale": 1e-3, "minimum": 0.0, "maximum": 25e-3,
            "start": 1e-3, "stop": 4e-3, "n": 9, "repeats": 5}


def test_scan_settings_reach_each_run_and_a_change_applies_from_the_next(expt):
    import json
    from waxx.util.device_state.loop_scan import ENV_VAR
    live, envs, loop = FakeLive(), [], None
    (spec,) = loop_specs({"auto_tof": ("BEC TOF loop", str(expt), TOF_SCAN)})
    assert spec.scan.xvar == "t_tof"

    def first():
        # set during run 1: run 1 keeps what it was launched with
        assert loop.configure({"start": 2e-3, "stop": None, "n": 9, "repeats": 20},
                              operator="jp")["status"] == "ok"
        return FakeProc(live, 101)

    procs = [first, lambda: (loop.stop(), FakeProc(live, 102))[1]]

    def spawn(command, extra_env=None):
        envs.append(json.loads(extra_env[ENV_VAR]))
        return procs.pop(0)()

    loop = RunLoop(spec, poll=live, spawn=spawn, gap_s=0., poll_s=0.01, journal=Journal())
    info = loop.info()
    assert info["scan"]["settings"] == {"start": 1e-3, "stop": 4e-3, "n": 9, "repeats": 5}
    assert info["scan"]["text"] == "t_tof 1–4 ms, 9 points × 5 repeats (45 shots)"
    loop.start()
    loop.join(5)
    assert envs == [{"start": 1e-3, "stop": 4e-3, "n": 9, "repeats": 5},
                    {"start": 2e-3, "stop": None, "n": 1, "repeats": 20}]
    lines = loop.output.since(0)["lines"]
    assert any("scan set by jp: t_tof 2 ms × 20 repeats" in l and "next run" in l
               for l in lines)
    assert any("2nd run of the loop: auto_tof (t_tof 2 ms × 20 repeats" in l for l in lines)
    assert "run_loop_configure" in loop._journal.kinds


def test_bad_scan_settings_are_refused_and_a_loop_without_a_scan_has_none(expt):
    (spec,) = loop_specs({"auto_tof": ("BEC TOF loop", str(expt), TOF_SCAN)})
    loop = RunLoop(spec, poll=FakeLive())
    reply = loop.configure({"start": 1e-3, "stop": 30e-3, "n": 5, "repeats": 1})
    assert reply["status"] == "error" and "above the maximum 25 ms" in reply["msg"]
    assert loop.info()["scan"]["settings"]["stop"] == 4e-3            # unchanged
    plain = RunLoop(LoopSpec("auto_tof", "BEC TOF loop", str(expt)), poll=FakeLive())
    assert "scan" not in plain.info()
    assert "has no scan settings" in plain.configure({"start": 1e-3, "repeats": 1})["msg"]


def test_server_configures_a_loops_scan(qapp, monkeypatch, tmp_path, expt):
    import json
    from waxx.util.guis import monitor_server_gui as msg
    monkeypatch.setattr(msg, "StateBroadcaster", Broadcasts)
    monkeypatch.setattr(msg, "monitor_server_id", lambda: "monitor-under-test")
    s = msg.MonitorUDPServer(config_file_path=str(tmp_path / "state.json"),
                             run_loops=loop_specs({"auto_tof": ("BEC TOF loop", str(expt),
                                                                TOF_SCAN)}))
    try:
        ask = lambda obj: json.loads(s.generate_reply(json.dumps(obj)))
        reply = ask({"type": "run_loop", "action": "configure", "loop": "auto_tof",
                     "scan": {"start": 3e-3, "stop": 5e-3, "n": 3, "repeats": 2}})
        assert reply["status"] == "ok" and reply["loop"]["scan"]["settings"]["n"] == 3
        status = json.loads(s.generate_reply("status_json"))
        assert status["run_loops"]["auto_tof"]["scan"]["text"].startswith("t_tof 3–5 ms")
        assert any(p.get("type") == "run_loop" for p in s._broadcaster.sent)
        reply = ask({"type": "run_loop", "action": "configure", "loop": "auto_tof",
                     "scan": {"start": -1, "repeats": 1}})
        assert reply["status"] == "error" and "below the minimum" in reply["msg"]
    finally:
        s.sock.close()


# -- the real POLL client wrapper (_LiveOD) against a fake LiveODClient ---------------------

class _StaleThenFreshClient:
    """LiveODClient stand-in: the first client made points at a liveOD that has
    restarted (its POLL times out); every later one reaches the new liveOD."""
    made = []

    def __init__(self, timeout_ms=None, discovery_timeout=None):
        self.stale = not _StaleThenFreshClient.made
        self.closed = False
        self.sent = []
        _StaleThenFreshClient.made.append(self)

    def poll(self):
        if self.stale:
            raise ConnectionError("No response from liveOD server (old port)")
        return {"ok": True, "run_in_progress": False, "reset_requested": False}

    def _send_recv(self, payload):
        self.sent.append(payload)
        if self.stale:
            raise ConnectionError("No response from liveOD server (old port)")
        return {"ok": True}

    def close(self):
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch):
    import waxx.util.live_od.live_od_client as client_mod
    _StaleThenFreshClient.made = []
    monkeypatch.setattr(client_mod, "LiveODClient", _StaleThenFreshClient)
    return _StaleThenFreshClient


def test_poll_after_a_live_od_restart_retries_on_a_fresh_client(fake_client):
    """The first Start after a liveOD restart was refused ("liveOD is not
    reachable") while liveOD was up: the kept client still had the old port."""
    from waxx.util.device_state.run_loop import _LiveOD
    live = _LiveOD()
    assert live()["ok"] is True
    assert len(fake_client.made) == 2
    assert fake_client.made[0].closed and not fake_client.made[1].closed
    assert live()["ok"] is True                 # the fresh client is kept
    assert len(fake_client.made) == 2


def test_poll_still_fails_when_live_od_is_really_gone(fake_client, monkeypatch):
    from waxx.util.device_state.run_loop import _LiveOD
    monkeypatch.setattr(fake_client, "poll", lambda self: (_ for _ in ()).throw(
        ConnectionError("no liveOD")))
    with pytest.raises(ConnectionError):
        _LiveOD()()
    assert len(fake_client.made) == 2           # one retry, not more


def test_the_exit_notice_is_sent_once(fake_client):
    """RUN_EXITED is not repeated: a notice must never be delivered twice."""
    from waxx.util.device_state.run_loop import _LiveOD
    with pytest.raises(ConnectionError):
        _LiveOD().run_exited(84100, "test")
    assert len(fake_client.made) == 1 and len(fake_client.made[0].sent) == 1


# -- the gate's liveOD verdict comes from run_gate (2026-10-09) ---------------------------

class ClientLive(FakeLive):
    """FakeLive whose POLL also names the run's client (and any ``extra`` keys)."""

    def __init__(self, **client):
        super().__init__()
        self.client, self.extra = client, {}

    def __call__(self):
        reply = super().__call__()
        reply.update(self.client)
        reply.update(self.extra)
        return reply


def _stuck(reset, host=None, **extra):
    """liveOD holding run 85528 (as on 2026-10-09), its client pid 4242."""
    import socket
    live = ClientLive(client_pid=4242, client_host=host or socket.gethostname(),
                      launcher="run_lock")
    live.run_in_progress, live.run_id, live.reset_requested = True, 85528, reset
    live.extra = extra
    return live


@pytest.mark.parametrize("reset, state", [(True, "reset_pending"), (False, "dead_client")])
def test_a_run_whose_process_is_gone_is_waived_with_one_warning(expt, monkeypatch, caplog,
                                                                 reset, state):
    from waxx.util.device_state import run_gate
    monkeypatch.setattr(run_gate, "_default_pid_alive", lambda pid: False)
    live = _stuck(reset)
    loop = None

    def init_run():                       # the next INIT_RUN finalizes the dead run
        live.run_in_progress, live.reset_requested, live.run_id = False, False, 101
        loop.stop()

    loop = _loop(expt, live, [FakeProc(live, 101, before=init_run)])
    with caplog.at_level("WARNING", logger="waxx.util.device_state.run_loop"):
        assert loop.start()["status"] == "ok"
        loop.join(5)
    info = loop.info()
    assert info["state"] == "stopped" and info["runs"] == 1
    waived = [r for r in caplog.records if "waived" in r.getMessage()]
    assert len(waived) == 1
    assert "85528" in waived[0].getMessage() and state in waived[0].getMessage()
    assert "pid 4242" in waived[0].getMessage()
    assert loop.journal.kinds.count("run_loop_waived") == 1


def test_a_waived_runs_abort_is_not_taken_for_a_new_one(expt, monkeypatch):
    """Review S2: the waived run's reset_requested stays set while the next run
    compiles (until its INIT_RUN); the loop must not read it as an Abort pressed
    while the run was starting."""
    from waxx.util.device_state import run_gate
    monkeypatch.setattr(run_gate, "_default_pid_alive", lambda pid: False)
    live = _stuck(True)
    loop = None
    polls_before = []

    def compile_then_init_run():
        n0 = live.polls
        time.sleep(0.15)                  # compiling: the loop polls, reset still set
        polls_before.append(live.polls - n0)
        live.run_in_progress, live.reset_requested, live.run_id = False, False, 101
        loop.stop()

    loop = _loop(expt, live, [FakeProc(live, 101, before=compile_then_init_run)])
    assert loop.start()["status"] == "ok"
    loop.join(5)
    info = loop.info()
    assert polls_before and polls_before[0] >= 3           # it did look, several times
    assert info["state"] == "stopped" and info["runs"] == 1, info["text"]
    assert "spent it on no run" not in info["text"]


def test_an_abort_for_another_run_id_still_counts_after_a_waiver(expt, monkeypatch):
    from waxx.util.device_state import run_gate
    monkeypatch.setattr(run_gate, "_default_pid_alive", lambda pid: False)
    live = _stuck(True)
    loop = None

    def a_new_abort_then_init_run():
        # liveOD's run is no longer the waived one, and an Abort is pending
        live.run_in_progress, live.run_id, live.reset_requested = False, 85529, True
        time.sleep(0.1)
        live.reset_requested, live.run_id = False, 101

    loop = _loop(expt, live, [FakeProc(live, 101, before=a_new_abort_then_init_run)])
    loop.start()
    loop.join(5)
    assert loop.info()["state"] == "latched"
    assert "spent it on no run" in loop.info()["text"]


@pytest.mark.parametrize("reset, extra, alive, words", [
    (True, {}, True, "an Abort is pending in liveOD, waiting"),
    (False, {"n_shots": 3, "init_run_age_s": 6000.0, "last_shot_age_s": 5000.0}, True,
     "WEDGED"),
])
def test_a_run_whose_process_lives_still_refuses(expt, monkeypatch, reset, extra, alive, words):
    from waxx.util.device_state import run_gate
    monkeypatch.setattr(run_gate, "_default_pid_alive", lambda pid: alive)
    loop = _loop(expt, _stuck(reset, **extra), [])
    reply = loop.start()
    assert reply["status"] == "error"
    assert words in reply["msg"] and "85528" in reply["msg"]
    assert reply["msg"].count("run 85528") == 1                         # named once (N2)
    assert "is in progress in liveOD --" not in reply["msg"]
    assert "run_loop_waived" not in loop.journal.kinds


def test_a_client_on_another_host_is_never_waived(expt, monkeypatch):
    from waxx.util.device_state import run_gate

    def never(pid):
        raise AssertionError("a pid on another host must not be checked")
    monkeypatch.setattr(run_gate, "_default_pid_alive", never)
    loop = _loop(expt, _stuck(True, host="some-other-pc"), [])
    assert loop.start()["status"] == "error"


def test_the_loop_tells_its_runs_who_launched_them(monkeypatch):
    from waxx.util.device_state import run_loop
    seen = {}

    def popen(command, **kw):
        seen.update(kw["env"])
        return "proc"
    monkeypatch.setattr(run_loop, "Popen", popen)
    assert run_loop._spawn("ar x.py", {"WAXX_LOOP_SCAN": "{}"}) == "proc"
    assert seen["WAXX_LAUNCHER"] == "run_loop"
    assert seen["PYTHONUNBUFFERED"] == "1" and seen["WAXX_LOOP_SCAN"] == "{}"
