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
    cfg = LiveODConfig()
    assert cfg.data_saver is None and cfg.run_id_source is None
    assert cfg.camera_params_list == [] and cfg.cameras_open_on_start == []
    assert cfg.camera_needs_grab_drain("xy_basler") and not cfg.camera_needs_grab_drain("other")
    assert cfg.resolve_camera_params("anything") is None
    assert cfg.default_roi_id_for("anything") is None
    assert cfg.cross_section_for_shot is None
    counts = cfg.params_factory()
    assert isinstance(counts, ImageCounts)
    assert (counts.N_img, counts.N_shots, counts.N_pwa_per_shot) == (1, 1, 1)


def test_mutable_defaults_are_not_shared():
    a, b = LiveODConfig(), LiveODConfig()
    a.camera_params_list.append("x")
    assert b.camera_params_list == []


def test_registry(monkeypatch):
    from waxx.util.live_od import config as module
    monkeypatch.setattr(module, "_active", None)
    default = module.get_config()
    assert isinstance(default, LiveODConfig) and module.get_config() is default
    mine = LiveODConfig(window_title="x")
    assert module.set_config(mine) is mine and module.get_config() is mine
