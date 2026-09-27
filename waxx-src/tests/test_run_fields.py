"""AndorParams run-owned fields (trigger_mode, frame_transfer, sensor_roi) and
the Scanner hook that checks them before a run id is reserved."""
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

from waxa.dummy.camera_params import CameraParams
from waxx.control.cameras.camera_param_classes import (AndorParams, APDParams, BaslerParams,
                                                       RunFieldRefused, check_andor_run_fields)

FULL = (0, 512, 0, 512, 1, 1)
REPO = Path(__file__).resolve().parents[3]


def payload_view(params):
    """What INIT_RUN (and so camera_params/ in the file) gets: the filter in
    Expt._serialize_init_payload."""
    return {k: v for k, v in vars(params).items() if not k.startswith('_')}


# -- defaults -----------------------------------------------------------------
def test_defaults_reproduce_today():
    p = AndorParams()
    assert p.trigger_mode == "ext"
    assert p.frame_transfer == 0
    assert p.sensor_roi == FULL
    assert p.baseline_clamp == 1
    assert p.resolution == (512, 512)
    p.prepare_for_run()
    assert p.resolution == (512, 512)
    assert (p.trigger_mode, p.frame_transfer, p.sensor_roi) == ("ext", 0, FULL)


def test_class_constants_are_not_recorded_as_settings():
    view = payload_view(AndorParams())
    for const in ("DETECTOR_SHAPE", "RUN_TRIGGER_MODES", "LIVE_TRIGGER_MODES",
                  "RUN_FRAME_TRANSFER", "ALLOW_SENSOR_CROP"):
        assert const not in view
    for field in ("trigger_mode", "frame_transfer", "sensor_roi", "baseline_clamp"):
        assert field in view


def test_every_recorded_value_is_storable():
    # h5py cannot store None; data_saver drops such keys silently (B2)
    p = AndorParams()
    p.select_imaging_type(0)
    p.prepare_for_run()
    for k, v in payload_view(p).items():
        assert v is not None, k
        assert np.asarray(v).dtype != object, k


# -- refusals -----------------------------------------------------------------
@pytest.mark.parametrize("field, value, words", [
    ("trigger_mode", "int", "runs accept only ('ext',)"),
    ("trigger_mode", "ext_start", "ext_start free-runs"),
    ("trigger_mode", b"ext_exp", "runs accept only"),
    ("trigger_mode", None, "must be a string"),
    ("frame_transfer", 1, "frame transfer is never used"),
    ("frame_transfer", True, "frame transfer is never used"),
    ("frame_transfer", None, "must be an integer"),
    ("frame_transfer", 0.5, "must be an integer"),
    ("sensor_roi", (0, 256, 0, 512, 1, 1), "only the full frame"),
    ("sensor_roi", (0, 512, 128, 384, 1, 1), "only the full frame"),
    ("sensor_roi", (0, 512, 0, 512, 2, 2), "binning 2x2 is not accepted"),
    ("sensor_roi", (0, 512, 0, 512, 1, 4), "binning 1x4 is not accepted"),
    ("sensor_roi", (0, 512, 0, 512), "must be 6 integers"),
    ("sensor_roi", None, "must be 6 integers"),
    ("sensor_roi", "full", "must be 6 integers"),
    ("sensor_roi", (0, 512.5, 0, 512, 1, 1), "must be an integer"),
])
def test_prepare_for_run_refuses_naming_rule_and_value(field, value, words):
    p = AndorParams(**{field: value})
    with pytest.raises(ValueError) as err:
        p.prepare_for_run()
    assert isinstance(err.value, RunFieldRefused)
    assert err.value.field == field
    msg = str(err.value)
    assert msg.startswith(f"AndorParams.{field} = ")
    assert words in msg


def test_readout_indices_and_clamp_are_checked():
    with pytest.raises(RunFieldRefused, match="baseline_clamp"):
        AndorParams(baseline_clamp=2).prepare_for_run()
    with pytest.raises(RunFieldRefused, match="hs_speed"):
        AndorParams(hs_speed=None).prepare_for_run()


def test_live_purpose_accepts_internal_triggers_only():
    assert check_andor_run_fields("int", 0, FULL, purpose="live")[0] == "int"
    assert check_andor_run_fields("software", 0, FULL, purpose="live")[0] == "software"
    with pytest.raises(RunFieldRefused, match="live streaming"):
        check_andor_run_fields("ext", 0, FULL, purpose="live")
    with pytest.raises(RunFieldRefused, match="frame transfer"):
        check_andor_run_fields("int", 1, FULL, purpose="live")


# -- normalisation --------------------------------------------------------------
def test_values_unpacked_from_a_file_are_normalised():
    p = AndorParams(trigger_mode=np.bytes_(b"ext"), frame_transfer=np.int64(0),
                    sensor_roi=np.array([0, 512, 0, 512, 1, 1], dtype=np.int32),
                    baseline_clamp=np.float64(1.0), hs_speed=np.int64(0), vs_speed=np.int32(1),
                    vs_amp=np.uint8(3), preamp=np.int16(2))
    p.prepare_for_run()
    assert p.trigger_mode == "ext" and type(p.trigger_mode) is str
    assert p.frame_transfer == 0 and type(p.frame_transfer) is int
    assert p.sensor_roi == FULL and type(p.sensor_roi) is tuple
    assert all(type(v) is int for v in p.sensor_roi)
    for name in ("baseline_clamp", "hs_speed", "vs_speed", "vs_amp", "preamp"):
        assert type(getattr(p, name)) is int, name
    assert p.resolution == (512, 512) and all(type(v) is int for v in p.resolution)


def test_list_roi_becomes_a_tuple():
    p = AndorParams(sensor_roi=[0, 512, 0, 512, 1, 1])
    p.prepare_for_run()
    assert p.sensor_roi == FULL


# -- APD and Basler --------------------------------------------------------------
def test_apd_keeps_its_one_pixel_resolution():
    apd = APDParams()
    assert apd.resolution == (1, 1)
    apd.prepare_for_run()
    assert apd.resolution == (1, 1)
    apd.trigger_mode = "int"            # nothing applies these for the APD
    apd.prepare_for_run()
    assert apd.resolution == (1, 1)


def test_basler_prepare_for_run_is_a_no_op():
    b = BaslerParams()
    before = dict(vars(b))
    b.prepare_for_run()
    assert vars(b) == before


# -- Scanner hook ---------------------------------------------------------------
def _scanner(camera_params, save_data=True, n_img=3):
    from waxx.base.scanner import Scanner
    sc = Scanner()
    sc.run_info.save_data = save_data
    sc.params.N_img = n_img
    sc.camera_params = camera_params
    return sc


def test_scanner_checks_camera_params_before_allocating_images():
    calls = []

    class Recording(CameraParams):
        def __init__(self):
            super().__init__()
            self.camera_type = "andor"

        def prepare_for_run(self):
            calls.append(hasattr(sc, "images"))
            self.resolution = (4, 6)

    sc = _scanner(Recording())
    sc.prepare_image_array()
    assert calls == [False]                      # before any image array existed
    assert sc.images.shape == (3, 4, 6)          # built from what prepare_for_run set


def test_scanner_refusal_raises_before_images_exist():
    sc = _scanner(AndorParams(frame_transfer=1))
    with pytest.raises(RunFieldRefused, match="frame_transfer"):
        sc.prepare_image_array()
    assert not hasattr(sc, "images")


def test_scanner_andor_images_follow_sensor_roi():
    sc = _scanner(AndorParams())
    sc.prepare_image_array()
    assert sc.images.shape == (3, 512, 512)
    assert sc.images.dtype == np.uint16


def test_scanner_accepts_a_bare_placeholder_and_no_save():
    # waxa's base CameraParams (the placeholder before choose_camera) may lack
    # the method; the hook skips it
    sc = _scanner(CameraParams(), save_data=False)
    sc.prepare_image_array()
    assert sc.images.shape == (1,)
    sc = _scanner(AndorParams(trigger_mode="int"), save_data=False)
    with pytest.raises(RunFieldRefused):        # checked whether or not data is saved
        sc.prepare_image_array()


def test_the_check_runs_before_init_run():
    # finish_prepare_wax: init_xvars (-> prepare_image_array) must come before
    # the INIT_RUN request, so a refusal reserves no run id.  Read from the
    # source (not imported): expt.py pulls in the monitor and device clients.
    import ast
    from waxx.base.scanner import Scanner
    path = REPO / "wax" / "waxx-src" / "waxx" / "base" / "expt.py"
    text = path.read_text(encoding="utf-8")
    func = next(n for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.FunctionDef) and n.name == "finish_prepare_wax")
    src = ast.get_source_segment(text, func)
    assert src.index("self.init_xvars(") < src.index(".init_run(")
    assert "self.prepare_image_array()" in inspect.getsource(Scanner.init_xvars)


# -- kexp's camera table -----------------------------------------------------------
def _load_kexp_camera_id():
    # by path, so the test does not import the kexp package (artiq, ...)
    path = REPO / "k-exp" / "kexp" / "config" / "camera_id.py"
    spec = importlib.util.spec_from_file_location("kexp_camera_id_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cameras


def test_kexp_andor_passes_the_run_fields_explicitly():
    cameras = _load_kexp_camera_id()
    andor = cameras.andor
    assert (andor.trigger_mode, andor.frame_transfer, andor.sensor_roi) == ("ext", 0, FULL)
    assert andor.baseline_clamp == 1
    andor.prepare_for_run()
    assert andor.resolution == (512, 512)
    src = (REPO / "k-exp" / "kexp" / "config" / "camera_id.py").read_text(encoding="utf-8")
    assert "2026-09-26: run-owned fields" in src


def test_kexp_apd_is_unchanged():
    apd = _load_kexp_camera_id().apd
    apd.prepare_for_run()
    assert apd.resolution == (1, 1)
    assert apd.camera_type == "apd"
