"""Shared test setup for waxx.

The development machine is the lab PC (kong), with a real monitor server
whose run queue keeps its files in ~/.waxx/run_queue: every test points the
queue's default folder at its own temporary folder instead, so no test ever
reads or writes the live queue's files."""
import pytest


@pytest.fixture(autouse=True)
def _run_queue_dir_is_temporary(monkeypatch, tmp_path):
    monkeypatch.setenv("WAXX_RUN_QUEUE_DIR", str(tmp_path / "run_queue_default"))
    # and the folders it runs files from: the test's own (the defaults are the
    # live code tree and the agents' log folder)
    monkeypatch.setenv("WAXX_RUN_QUEUE_ROOTS", str(tmp_path))


@pytest.fixture(autouse=True)
def _queue_gate_never_reaches_the_real_queue(monkeypatch):
    """A warm-launched process's queue_gate asks the monitor server for GO by
    discovery -- on kong that is the live server.  A test that gets that far
    without its own stand-in (``queue_gate.set_transport``) fails at once
    instead of polling the live server for WARM_LOST_S."""
    from waxx.util.device_state import queue_gate

    def no_real_transport():
        transport = queue_gate._state["transport"]
        if transport is None:
            pytest.fail("a test reached for the real run queue through queue_gate; "
                        "give it a stand-in with queue_gate.set_transport(...)")
        return transport
    monkeypatch.setattr(queue_gate, "_transport", no_real_transport)
