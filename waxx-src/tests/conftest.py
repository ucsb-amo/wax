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


class _OfflineCameraDirectory:
    """No camera server on the network: what a test gets in place of the
    process's shared beacon CameraDirectory, so nothing asks kong's real
    camera servers (BaslerParams.prepare_for_run reads the exposure delay)."""
    def find(self, query, *, refresh=False):
        from beacon.camera.directory import CameraNotFound
        raise CameraNotFound(f"offline test: the camera servers are not asked ({query!r})")

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _no_shared_camera_directory(monkeypatch):
    import waxx.control.cameras.camera_stream_client as csc
    monkeypatch.setattr(csc, "shared_camera_directory", _OfflineCameraDirectory)
