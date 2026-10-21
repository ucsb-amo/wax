"""The experiment's SLM writes without the per-shot cost (2026-10-21).

* write_phase_mask_kernel reuses a mask the SLM still shows -- the one this
  client wrote last, the server confirming (status: same process, same
  ``pattern_epoch``, same ``applies`` count, same pattern, ready) -- instead
  of writing it again and waiting SLM_RPC_DELAY; anything else is a write.
* write_phase_mask_async / wait_phase_mask_settled: the write in a host
  thread, and a check that raises unless it is on the SLM and settled by the
  time the shot needs it.

Host halves only (the kernels are compiled in k-exp
tests/test_shot_deadtime_compiles.py). The SLM server is the real
run_server.py on 127.0.0.1 with the SLM calls stubbed (test_slm_reinit's
fixture); nothing touches the lab network.
"""
import inspect
import types
import time

import numpy as np
import pytest

from test_slm_reinit import (ScriptedServer, _apply, _free_port, _slm, _status,  # noqa: F401
                             _wait, slm_host)
from waxx.control.slm import slm as slm_mod

ARGS = (10e-6, np.pi / 2, 100, 200, "spot", False, False)   # as the kernel passes them
OTHER = (10e-6, np.pi / 4, 100, 200, "spot", False, False)


@pytest.fixture
def fast_settle(monkeypatch):
    monkeypatch.setattr(slm_mod, "SLM_RPC_DELAY", 0.05)
    return 0.05


def _uploads(host):
    return len(host.uploads)


def _async(slm, args, skip=True, counter_mu=0):
    """The kernel's parallel write, as its async RPC arrives on the host."""
    slm._write_async(*args, skip, counter_mu)


def _client(host):
    slm = _slm(host.port)
    slm.core = types.SimpleNamespace(ref_period=1e-9)
    return slm


# --- reuse on the kernel path ------------------------------------------------------------

def test_first_write_is_real_then_an_unchanged_mask_is_reused(slm_host, fast_settle):
    slm = _client(slm_host)
    assert slm._settle_left_if_shown(*ARGS) == -1.          # nothing written yet
    n = _uploads(slm_host)
    slm._write_for_kernel(*ARGS)
    assert _uploads(slm_host) == n + 1
    assert slm_host.uploads[-1][:2] == (100, 200)
    left = slm._settle_left_if_shown(*ARGS)                  # still settling
    assert 0. < left <= fast_settle
    time.sleep(fast_settle + 0.01)
    assert slm._settle_left_if_shown(*ARGS) == 0.
    assert _uploads(slm_host) == n + 1                       # never rewritten


def test_parameter_defaults_resolve_to_the_same_mask(slm_host, fast_settle):
    """The kernel passes the sentinels; the params give the same command."""
    slm = _client(slm_host)
    d, di = slm_mod.dv, slm_mod.di
    slm._write_for_kernel(d, d, di, di, "spot", False, False)
    time.sleep(fast_settle + 0.01)
    assert slm._settle_left_if_shown(*ARGS) == 0.


def test_a_changed_mask_is_written(slm_host, fast_settle):
    slm = _client(slm_host)
    slm._write_for_kernel(*ARGS)
    assert slm._settle_left_if_shown(*OTHER) == -1.


def test_another_clients_write_forces_a_rewrite_even_of_the_same_mask(slm_host, fast_settle):
    slm = _client(slm_host)
    slm._write_for_kernel(*ARGS)
    _apply(slm_host, 7, 8)                                   # someone else's mask
    assert slm._settle_left_if_shown(*ARGS) == -1.
    slm._write_for_kernel(*ARGS)
    assert slm._settle_left_if_shown(*ARGS) >= 0.
    # the same pattern written by someone else still counts as a write since
    same, _ = slm._command(*ARGS[:6])
    slm_mod.slm_link.exchange("127.0.0.1", slm_host.port, same,
                              until=("applied",), total_s=5.0)
    assert slm._settle_left_if_shown(*ARGS) == -1.


def test_a_reinit_forces_a_rewrite(slm_host, fast_settle):
    slm = _client(slm_host)
    slm._write_for_kernel(*ARGS)
    done = slm_mod.slm_link.exchange("127.0.0.1", slm_host.port, {"cmd": "reinit"},
                                     control=True, until=("reinit_done", "error"),
                                     total_s=5.0)
    assert done["status"] == "reinit_done"
    assert slm._settle_left_if_shown(*ARGS) == -1.


def test_no_clear_status_means_a_write(slm_host, fast_settle, monkeypatch):
    slm = _client(slm_host)
    slm._write_for_kernel(*ARGS)

    def gone(self):
        raise OSError("unreachable")

    monkeypatch.setattr(slm_mod.SLM, "_status", gone)
    assert slm._settle_left_if_shown(*ARGS) == -1.


def test_a_failed_write_forgets_what_was_shown(slm_host, fast_settle, monkeypatch):
    slm = _client(slm_host)
    slm._write_for_kernel(*ARGS)

    def refuse(self, command):
        raise self._failed("refused")

    monkeypatch.setattr(slm_mod.SLM, "_write", refuse)
    with pytest.raises(slm_mod.SLMWriteError):
        slm._write_for_kernel(*ARGS)
    monkeypatch.undo()
    assert slm._shown is None
    assert slm._settle_left_if_shown(*ARGS) == -1.


def test_a_server_without_replies_is_never_reused(monkeypatch):
    monkeypatch.setattr(slm_mod, "SLM_FIRST_REPLY_S", 0.3)
    old = ScriptedServer(lambda msg: [])
    try:
        slm = _slm(old.port)
        slm._write_for_kernel(*ARGS)
        assert slm._server_replies is False
        assert slm._settle_left_if_shown(*ARGS) == -1.
    finally:
        old.close()


def test_the_first_reuse_is_said_once(slm_host, fast_settle, capsys):
    slm = _client(slm_host)
    args = ARGS[:-1] + (True,)                               # verbose
    slm._write_for_kernel(*args)
    for _ in range(3):
        slm._settle_left_if_shown(*args)
    assert capsys.readouterr().out.count("not rewritten") == 1


# --- the write in parallel with the sequence ---------------------------------------------

def test_async_write_then_check_passes(slm_host, fast_settle):
    slm = _client(slm_host)
    n = _uploads(slm_host)
    _async(slm, ARGS)
    slm._check_settled(int(1.0 / 1e-9))                      # 1 s of slack
    assert _uploads(slm_host) == n + 1
    assert slm_host.uploads[-1][:2] == (100, 200)
    assert slm._pending_checked is True


def test_async_reuses_an_unchanged_mask(slm_host, fast_settle):
    slm = _client(slm_host)
    slm._write_for_kernel(*ARGS)
    n = _uploads(slm_host)
    time.sleep(fast_settle + 0.01)
    _async(slm, ARGS)
    slm._check_settled(int(0.1 / 1e-9))
    assert _uploads(slm_host) == n


def test_check_raises_when_the_settle_would_be_late(slm_host, monkeypatch):
    monkeypatch.setattr(slm_mod, "SLM_RPC_DELAY", 0.5)
    slm = _client(slm_host)
    _async(slm, ARGS)
    with pytest.raises(slm_mod.SLMWriteError, match="would settle"):
        slm._check_settled(int(0.2 / 1e-9))


def test_check_raises_when_the_write_is_still_in_flight(fast_settle):
    slow = ScriptedServer(lambda msg: [(0.0, {"status": "queued"}), (0.8, {"status": "applied"})]
                          if "cmd" not in msg else [(0.0, {"status": "error", "error": "n/a"})])
    try:
        slm = _client(slow)
        _async(slm, ARGS)
        t0 = time.monotonic()
        with pytest.raises(slm_mod.SLMWriteError, match="still in flight"):
            slm._check_settled(int(0.15 / 1e-9))
        # it waited only while the slack allowed, keeping the return margin
        assert time.monotonic() - t0 < 0.15
    finally:
        slm._join_pending()
        slow.close()


def test_check_raises_when_the_write_failed():
    refuses = ScriptedServer(lambda msg: [(0.0, {"status": "queued"}),
                                          (0.0, {"status": "error", "error": "bad mask"})])
    try:
        slm = _client(refuses)
        _async(slm, ARGS)
        with pytest.raises(slm_mod.SLMWriteError, match="bad mask"):
            slm._check_settled(int(1.0 / 1e-9))
    finally:
        refuses.close()


def test_a_check_without_an_async_write_is_a_no_op(slm_host):
    _client(slm_host)._check_settled(0)


def test_a_sync_write_waits_for_the_async_one_in_flight(slm_host, fast_settle):
    slm_host.init_s = 0.0
    slm = _client(slm_host)
    _async(slm, OTHER)
    slm.write_phase_mask(x_center=300, y_center=400, verbose=False)
    assert [u[:2] for u in slm_host.uploads[-2:]] == [(100, 200), (300, 400)]


def test_a_failed_write_replaced_before_any_use_is_warned(capsys):
    refuses = ScriptedServer(lambda msg: [(0.0, {"status": "queued"}),
                                          (0.0, {"status": "error", "error": "bad mask"})])
    try:
        slm = _client(refuses)
        _async(slm, ARGS)
        assert _wait(lambda: slm._pending.done())
        _async(slm, OTHER)
        assert "an earlier mask write failed" in capsys.readouterr().out
    finally:
        slm._join_pending()
        refuses.close()


def test_skip_if_unchanged_false_always_writes(slm_host, fast_settle):
    slm = _client(slm_host)
    slm._write_for_kernel(*ARGS)
    n = _uploads(slm_host)
    time.sleep(fast_settle + 0.01)
    _async(slm, ARGS, skip=False)
    slm._check_settled(int(1.0 / 1e-9))
    assert _uploads(slm_host) == n + 1


# --- a first use too close to real time: checked at the end of the shot -------------------

def test_deferred_check_passes_when_the_use_came_after_the_settle(slm_host, fast_settle):
    slm = _client(slm_host)
    _async(slm, ARGS, counter_mu=1_000)
    assert _wait(lambda: slm._pending.done())
    # the use played out 1 s (timeline) after the write was issued
    slm._confirm_write(1_000 + int(1.0 / 1e-9))
    assert slm._pending_checked is True


def test_deferred_check_raises_when_the_use_came_before_the_settle(slm_host, monkeypatch):
    monkeypatch.setattr(slm_mod, "SLM_RPC_DELAY", 0.3)
    slm = _client(slm_host)
    _async(slm, ARGS, counter_mu=0)
    with pytest.raises(slm_mod.SLMWriteError, match="before the mask was settled"):
        slm._confirm_write(int(0.1 / 1e-9))                  # used 0.1 s after the write


def test_deferred_check_raises_on_a_failed_write():
    refuses = ScriptedServer(lambda msg: [(0.0, {"status": "queued"}),
                                          (0.0, {"status": "error", "error": "bad mask"})])
    try:
        slm = _client(refuses)
        _async(slm, ARGS)
        with pytest.raises(slm_mod.SLMWriteError, match="bad mask"):
            slm._confirm_write(int(5.0 / 1e-9))
    finally:
        refuses.close()


# --- the kernels' structure (compiled in k-exp tests/test_shot_deadtime_compiles.py) -------

def _body(fn):
    src = inspect.getsource(fn)
    return src[src.index('"""', src.index('"""') + 3) + 3:]


def test_parallel_write_waits_for_the_timeline_only_after_light_and_marks_the_check():
    body = _body(slm_mod.SLM.write_phase_mask_kernel)
    par = body[body.index("if parallel:"):body.index("        else:")]
    i_light = par.index("if self._light_this_shot:")
    i_wait = par.index("self.core.wait_until_mu(now_mu())")
    i_rpc = par.index("self._write_async(")
    i_pend = par.index("self._use_pending = True")
    assert i_light < i_wait < i_rpc < i_pend
    assert "delay(" not in par                                # nothing on the timeline


def test_first_use_checks_on_the_spot_only_with_slack():
    body = _body(slm_mod.SLM.ensure_mask_settled)
    assert "self._light_this_shot = True" in body
    i_slack = body.index("if slack > self.core.seconds_to_mu(CHECK_MIN_SLACK_S):")
    i_check = body.index("self._check_settled(slack)")
    i_note = body.index("self._t_use_mu = now_mu()")
    assert i_slack < i_check < i_note
    assert "delay(" not in body and "break_realtime" not in body


def test_the_imaging_beam_asks_its_guard_before_any_light():
    from waxx.control.beat_lock import BeatLockImaging, NoLightGuard
    for fn in (BeatLockImaging.on, BeatLockImaging.pulse):
        body = inspect.getsource(fn)        # on() has no docstring
        assert body.index("self.light_guard.ensure_mask_settled()") < body.index("self.dds_sw.on()")
    assert isinstance(BeatLockImaging().light_guard, NoLightGuard)


def test_bad_arguments_surface_at_the_check(slm_host):
    slm = _client(slm_host)
    _async(slm, (10e-6, 0., 1, 2, "hexagon", False, False))
    with pytest.raises(slm_mod.SLMWriteError, match="mask_type"):
        slm._check_settled(int(1.0 / 1e-9))


def test_the_kernel_check_changes_nothing_on_the_timeline():
    """wait_phase_mask_settled reports; it never delays the sequence."""
    src = inspect.getsource(slm_mod.SLM.wait_phase_mask_settled)
    body = src[src.index('"""', src.index('"""') + 3) + 3:]
    assert "delay(" not in body and "break_realtime" not in body and "at_mu" not in body
    assert "self._check_settled(now_mu() - self.core.get_rtio_counter_mu())" in body


# --- the scan loop tells the cleanup whether the shot returned normally ------------------

def test_shot_completed_is_set_only_by_a_normal_return():
    from waxx.base.scanner import Scanner
    src = inspect.getsource(Scanner._scan)
    i_false = src.index("self.shot_completed = False")
    i_call = src.index("self.scan_kernel()")
    i_true = src.index("self.shot_completed = True")
    i_exc = src.index("except RTIOUnderflow:")
    assert i_false < i_call < i_true < i_exc
    # nothing but whitespace between the call and the flag
    assert src[i_call + len("self.scan_kernel()"):i_true].strip() == ""
    assert src.count("self.shot_completed = True") == 1
