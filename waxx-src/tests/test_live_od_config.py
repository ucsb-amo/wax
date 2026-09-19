"""waxx.util.live_od.config: the lab-facing config object for liveOD.
It sits on the experiment process's import path, so it must stay stdlib-only."""
import subprocess
import sys

from waxx.util.live_od.config import LiveODConfig, ImageCounts


def test_import_is_stdlib_only():
    code = ("import sys; import waxx.util.live_od, waxx.util.live_od.config; "
            "bad = [m for m in ('kexp', 'PyQt6', 'zmq', 'numpy', 'waxa') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_defaults():
    cfg = LiveODConfig(data_saver=object(), run_id_source=object())
    # the beacon ids are how every viewer and tool finds the server
    assert (cfg.server_id_base, cfg.broadcast_id_base) == ("live_od", "live_od_broadcast")
    assert cfg.camera_params_list == [] and cfg.cameras_closed_on_start == []
    assert cfg.resolve_camera_params("anything") is None
    assert cfg.default_roi_id_for("anything") is None
    assert cfg.cross_section_for_shot is None
    counts = cfg.params_factory()
    assert isinstance(counts, ImageCounts)
    assert (counts.N_img, counts.N_shots, counts.N_pwa_per_shot) == (1, 1, 1)


def test_mutable_defaults_are_not_shared():
    a = LiveODConfig(data_saver=None, run_id_source=None)
    b = LiveODConfig(data_saver=None, run_id_source=None)
    a.camera_params_list.append("x")
    assert b.camera_params_list == []
