"""DeviceLock: one process at a time per device, with a message naming the holder."""
import os
import subprocess
import sys
import textwrap
import time

import pytest

from waxx.control.cameras import device_lock as dl
from waxx.control.cameras.device_lock import DeviceLock, DeviceBusy


@pytest.fixture(autouse=True)
def lock_dir(monkeypatch, tmp_path):
    d = tmp_path / "locks"
    monkeypatch.setenv(dl.ENV_LOCK_DIR, str(d))
    monkeypatch.setattr(dl, "_process_label", None)
    return d


def test_acquire_release_is_idempotent_and_a_context_manager(lock_dir):
    lock = DeviceLock("andor_sdk2:0")
    assert not lock.held
    lock.acquire()
    lock.acquire()                       # already held by this object: no-op
    assert lock.held
    assert (lock_dir / "andor_sdk2_0.json").exists()
    lock.release()
    lock.release()                       # idempotent
    assert not lock.held
    assert not (lock_dir / "andor_sdk2_0.json").exists()
    with DeviceLock("andor_sdk2:0") as again:
        assert again.held
    assert not again.held


def test_second_lock_in_this_process_is_busy_and_names_the_holder():
    dl.set_process_label("liveOD")
    first = DeviceLock("andor_sdk2:0").acquire()
    try:
        with pytest.raises(DeviceBusy) as err:
            DeviceLock("andor_sdk2:0").acquire()
        msg = str(err.value)
        assert msg.startswith(f"Andor SDK is held by pid {os.getpid()} (")
        assert "liveOD" in msg and "(this process)" in msg
        assert err.value.holder["pid"] == os.getpid()
        assert err.value.holder["exe"] == os.path.basename(sys.executable)
        assert "since" in err.value.holder
        # e.g. "Andor SDK is held by pid 1234 (python.exe, liveOD) since 14:02 (this process)"
        assert time.strftime("%H:") in msg
    finally:
        first.release()
    DeviceLock("andor_sdk2:0").acquire().release()


def test_busy_is_a_camera_unavailable():
    from beacon.camera.backend import CameraUnavailable
    assert issubclass(DeviceBusy, CameraUnavailable)


def test_other_keys_are_independent():
    a = DeviceLock("basler:1").acquire()
    b = DeviceLock("basler:2").acquire()
    assert "Basler camera 1" in DeviceLock("basler:1").description
    a.release()
    b.release()


def test_a_failed_acquire_leaves_the_holders_record_alone(lock_dir):
    holder = DeviceLock("basler:40316451").acquire()
    record = (lock_dir / "basler_40316451.json").read_text()
    for _ in range(3):
        with pytest.raises(DeviceBusy):
            DeviceLock("basler:40316451").acquire()
    assert (lock_dir / "basler_40316451.json").read_text() == record
    holder.release()


def test_acquire_with_timeout_waits_then_raises():
    holder = DeviceLock("basler:9").acquire()
    t0 = time.monotonic()
    with pytest.raises(DeviceBusy):
        DeviceLock("basler:9").acquire(timeout_s=0.3)
    assert time.monotonic() - t0 >= 0.28
    holder.release()


_CHILD = textwrap.dedent("""
    import os, sys
    from waxx.control.cameras.device_lock import DeviceLock, set_process_label
    set_process_label("child-holder")
    lock = DeviceLock("andor_sdk2:0").acquire()
    print("HELD", os.getpid(), flush=True)
    if sys.argv[1] == "wait":
        sys.stdin.readline()     # hold until killed
    # "exit": leave without releasing; the OS must drop the lock
""")


def _spawn(mode, lock_dir):
    env = dict(os.environ, **{dl.ENV_LOCK_DIR: str(lock_dir)})
    proc = subprocess.Popen([sys.executable, "-c", _CHILD, mode], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    line = proc.stdout.readline().split()
    assert line and line[0] == "HELD", proc.stderr.read()
    # the interpreter's own pid: a venv python.exe may be a launcher that runs
    # the interpreter as its child
    return proc, int(line[1])


def test_lock_held_by_another_process_names_it_and_dies_with_it(lock_dir):
    proc, pid = _spawn("wait", lock_dir)
    try:
        with pytest.raises(DeviceBusy) as err:
            DeviceLock("andor_sdk2:0").acquire()
        assert err.value.holder["pid"] == pid
        assert f"held by pid {pid} (" in str(err.value)
        assert "child-holder" in str(err.value)
        assert "(this process)" not in str(err.value)
    finally:
        proc.kill()
        proc.wait(timeout=10)
    lock = DeviceLock("andor_sdk2:0").acquire(timeout_s=5.0)   # released by the OS
    assert lock.held
    # the dead holder's stale record was overwritten with ours
    assert lock.holder()["pid"] == os.getpid()
    lock.release()


def test_lock_is_released_when_the_holder_exits_without_releasing(lock_dir):
    proc, _ = _spawn("exit", lock_dir)
    proc.wait(timeout=10)
    DeviceLock("andor_sdk2:0").acquire(timeout_s=5.0).release()
