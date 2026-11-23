"""waxx.util.device_state.claim_overlap, and the Core's use of it.

A cold queue launch's claim on a thread beside its kernel compile. No hardware, no
sockets: the compile is a stub the test can hold open, the comm object records what it is
asked to do, and ``create_connection`` raises everywhere.
"""
import threading
import time

import pytest

from artiq.coredevice import comm_kernel

from waxx.util.artiq.core import Core
from waxx.util.device_state import claim_overlap as co
from waxx.util.device_state import queue_gate

TEST_HOST = "192.0.2.1"          # TEST-NET-1: never routed


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    def refuse(*args, **kwargs):
        raise ConnectionRefusedError(f"test: connection to {args} refused")

    monkeypatch.setattr(comm_kernel, "create_connection", refuse)
    for name in ("WAXX_QUEUE_WARM", "WAXX_LAUNCHER", co.ENV):
        monkeypatch.delenv(name, raising=False)
    queue_gate._reset_for_tests()
    co._reset_for_tests()
    yield
    co._reset_for_tests()
    queue_gate._reset_for_tests()


# ---- when it applies ----------------------------------------------------------------------

@pytest.mark.parametrize("launcher, env, warm, default_on, expected", [
    ("kq", "1", False, False, True),
    ("kq", None, False, False, False),         # unset: the default
    ("kq", None, False, True, True),
    ("kq", "0", False, True, False),           # 0 always turns it off
    ("kq", "1", True, True, False),            # a warm launch claims after GO, as ever
    (None, "1", False, True, False),           # direct artiq_run
    ("run_loop", "1", False, True, False),
    ("run_lock", "1", False, True, False),
])
def test_enabled_only_for_a_cold_queue_launch(monkeypatch, launcher, env, warm, default_on,
                                              expected):
    if launcher is not None:
        monkeypatch.setenv("WAXX_LAUNCHER", launcher)
    if env is not None:
        monkeypatch.setenv(co.ENV, env)
    if warm:
        monkeypatch.setenv("WAXX_QUEUE_WARM", "warm-token")
    queue_gate._reset_for_tests()
    monkeypatch.setattr(co, "DEFAULT_ON", default_on)
    assert co.enabled() is expected


def test_the_default_is_off():
    """Until the overlap has run on the machine (flip DEFAULT_ON then)."""
    assert co.DEFAULT_ON is False


def test_run_is_kernel():
    from artiq.language.core import kernel

    class K:
        @kernel
        def run(self):
            pass

    class H:
        def run(self):
            pass

    class N:
        pass

    assert co.run_is_kernel(K()) is True
    assert co.run_is_kernel(H()) is False
    assert co.run_is_kernel(N()) is False


# ---- background steps ---------------------------------------------------------------------

def test_wait_returns_the_result_and_raises_in_the_caller():
    assert co.wait(co.background("ok", lambda: 7)) == 7
    assert co.wait(None) is None

    def boom():
        raise ValueError("the stage did not move")

    task = co.background("bad", boom)
    with pytest.raises(ValueError, match="did not move"):
        co.wait(task)
    assert task.waited


def test_a_cancelled_wait_stops_waiting():
    release = threading.Event()
    task = co.background("slow", release.wait)
    co._cancel.set()
    with pytest.raises(co.ClaimCancelled):
        co.wait(task)
    assert not task.waited
    release.set()
    co._cancel.clear()
    assert co.wait(task) is True


def test_check_not_cancelled():
    co.check_not_cancelled()
    co._cancel.set()
    with pytest.raises(co.ClaimCancelled):
        co.check_not_cancelled()


def test_run_claim_now_runs_a_claim_no_kernel_started():
    ran = []
    assert co.run_claim_now() is False
    co.defer_claim(lambda: ran.append("claim"))
    assert co.claim_pending() and co.active()
    assert co.run_claim_now() is True
    assert ran == ["claim"] and not co.active()


# ---- the Core -----------------------------------------------------------------------------

class RecordingComm:
    def __init__(self, log):
        self.log = log

    def check_system_info(self):
        self.log.append("check_system_info")

    def load(self, kernel_library):
        self.log.append(("load", kernel_library))

    def run(self):
        self.log.append("run")

    def serve(self, embedding_map, symbolizer, demangler):
        self.log.append("serve")

    def close(self):
        pass


class NoSubkernels:
    def subkernels(self):
        return {}

    def subkernel_messages_unpaired(self):
        return []


def _core(log, compile_body=None):
    core = Core(None, TEST_HOST, 1e-9)
    assert core.queue_gate_armed is False
    core.comm = RecordingComm(log)

    def compile(function, args, kwargs, set_result=None, attribute_writeback=True, **rest):
        log.append("compile start")
        if compile_body is not None:
            compile_body()
        log.append("compile end")
        return NoSubkernels(), "elf", None, None, {}

    core.compile = compile
    return core


def _kernel(*args):
    pass


def test_nothing_deferred_is_upstream(monkeypatch):
    log = []
    core = _core(log)
    monkeypatch.setattr(Core, "_run_beside_claim",
                        lambda *a, **k: pytest.fail("the overlap path ran"))
    core.run(_kernel, (), {})
    assert log == ["compile start", "compile end", "check_system_info", ("load", "elf"),
                   "run", "serve"]


def test_the_claim_runs_beside_the_compile_and_the_upload_waits_for_it():
    log = []
    claim_started, compile_done = threading.Event(), threading.Event()

    def claim():
        log.append("claim start")
        claim_started.set()
        assert compile_done.wait(5)            # the claim outlasts the compile
        time.sleep(0.1)
        log.append("claim end")

    def compile_body():
        assert claim_started.wait(5)           # the claim is running during the compile

    core = _core(log, compile_body)
    real_compile = core.compile

    def compile(*a, **k):
        try:
            return real_compile(*a, **k)
        finally:
            compile_done.set()

    core.compile = compile
    co.defer_claim(claim)
    core.run(_kernel, (), {})
    assert log.index("claim start") < log.index("compile end")
    assert log.index("compile end") < log.index("claim end") < log.index("check_system_info")
    assert log[-4:] == ["check_system_info", ("load", "elf"), "run", "serve"]
    assert not co.active()
    rep = co.report()
    assert set(rep) == {"compile_s", "claim_s", "waited_after_compile_s"}
    assert rep["waited_after_compile_s"] >= 0.05

    log.clear()
    core.run(_kernel, (), {})                  # the next kernel: upstream, no second claim
    assert "claim start" not in log and log[-1] == "serve"


def test_a_claim_that_raises_stops_the_run_before_anything_is_sent():
    log = []

    def claim():
        raise RuntimeError("SLM reinit failed")

    core = _core(log)
    co.defer_claim(claim)
    with pytest.raises(RuntimeError, match="SLM reinit failed"):
        core.run(_kernel, (), {})
    assert "check_system_info" not in log and not any(isinstance(e, tuple) for e in log)


def test_a_compile_that_raises_stops_the_claim_before_the_liveod_start():
    log = []
    at_hardware, go_on = threading.Event(), threading.Event()

    def claim():
        log.append("claim hardware")
        at_hardware.set()
        assert go_on.wait(5)
        co.check_not_cancelled()               # as Expt.claim, before _start_liveod_run
        log.append("liveod start")

    def compile_body():
        assert at_hardware.wait(5)
        threading.Timer(0.1, go_on.set).start()
        raise KeyboardInterrupt()

    core = _core(log, compile_body)
    co.defer_claim(claim)
    with pytest.raises(KeyboardInterrupt):
        core.run(_kernel, (), {})
    assert "liveod start" not in log
    assert "check_system_info" not in log
    assert threading.active_count() >= 1 and not co.claim_pending()
    # the claim's thread has ended: the process does not leave half-way through it
    assert all(not t.name.startswith("claim-overlap:claim") or not t.is_alive()
               for t in threading.enumerate())


def test_a_compile_that_raises_after_the_liveod_start_waits_for_the_whole_claim():
    log = []
    started, go_on = threading.Event(), threading.Event()

    def claim():
        co.check_not_cancelled()
        log.append("liveod start")             # the run id is taken
        started.set()
        assert go_on.wait(5)
        log.append("announce")

    def compile_body():
        assert started.wait(5)
        threading.Timer(0.1, go_on.set).start()
        raise SyntaxError("kernel does not compile")

    core = _core(log, compile_body)
    co.defer_claim(claim)
    with pytest.raises(SyntaxError):
        core.run(_kernel, (), {})
    # as a cold run whose compile fails after its claim: the claim is complete
    assert [e for e in log if not e.startswith("compile")] == ["liveod start", "announce"]
    assert "check_system_info" not in log


def test_a_step_started_early_is_waited_for_with_no_claim_at_all():
    """An experiment that never calls finish_prepare: the stage's throw still ends
    before its first kernel is sent."""
    log = []
    release = threading.Event()

    def stage():
        release.wait(5)
        log.append("stage in place")

    co.background("APD stage", stage)
    assert co.active()
    core = _core(log, lambda: threading.Timer(0.1, release.set).start())
    core.run(_kernel, (), {})
    assert log.index("stage in place") < log.index("check_system_info")
    assert not co.active()


def test_a_stage_step_that_raises_with_no_claim_stops_the_run():
    log = []

    def stage():
        raise ConnectionError("PDXC server not found")

    co.background("APD stage", stage)
    core = _core(log)
    with pytest.raises(ConnectionError):
        core.run(_kernel, (), {})
    assert "check_system_info" not in log
