"""Basler pixel format on the experiment side: BaslerParams, the run's
declared image dtype (Scanner.prepare_image_array) and liveOD's per-run
apply on the legacy (non-host) path (CameraNanny.update_params)."""
import types

import numpy as np
import pytest

from waxx.control.cameras.camera_param_classes import BaslerParams, RunFieldRefused


def test_baslerparams_default_is_mono8_uint8():
    p = BaslerParams()
    assert p.pixel_format == "Mono8" and p.image_dtype() == np.uint8
    p.prepare_for_run()
    assert "pixel_format" in vars(p)              # saved in camera_params/


@pytest.mark.parametrize("pf", ["Mono12", "Mono12p"])
def test_baslerparams_12_bit_is_uint16(pf):
    p = BaslerParams(pixel_format=pf)
    p.prepare_for_run()
    assert p.image_dtype() == np.uint16


def test_baslerparams_refuses_an_unsupported_format_before_a_run_id():
    p = BaslerParams(pixel_format="Mono10p")
    with pytest.raises(RunFieldRefused, match="pixel_format"):
        p.prepare_for_run()


def test_baslerparams_from_a_file_value():
    p = BaslerParams()
    p.pixel_format = b"Mono12p"                     # as h5py hands it back
    p.prepare_for_run()
    assert p.pixel_format == "Mono12p"


def _scanner_self(camera_params, save_data=True, n_img=6):
    return types.SimpleNamespace(camera_params=camera_params,
                                 run_info=types.SimpleNamespace(save_data=save_data),
                                 params=types.SimpleNamespace(N_img=n_img))


@pytest.mark.parametrize("pf, dtype", [("Mono8", np.uint8), ("Mono12p", np.uint16)])
def test_the_run_declares_the_formats_dtype_and_holds_no_frames(pf, dtype):
    from waxx.base.scanner import Scanner
    s = _scanner_self(BaslerParams(pixel_format=pf, resolution=(1200, 1920)))
    Scanner.prepare_image_array(s)
    assert s.images.shape == (6, 1200, 1920) and s.images.dtype == dtype
    assert s.images.ndim > 1                        # what the INIT_RUN payload checks
    assert not s.images.flags.writeable             # a view of one zero, not 6 frames
    assert s.images.strides == (0, 0, 0)


def test_a_camera_without_image_dtype_keeps_the_type_rule():
    from waxx.base.scanner import Scanner
    cp = types.SimpleNamespace(camera_type="andor", resolution=(512, 512))
    s = _scanner_self(cp, n_img=3)
    Scanner.prepare_image_array(s)
    assert s.images.dtype == np.uint16 and s.images.shape == (3, 512, 512)


class _Cam:
    """BaslerUSB's per-run surface, recording calls."""

    def __init__(self, with_pixel_format=True):
        self.calls = []
        if with_pixel_format:
            self.set_pixel_format = lambda pf: self.calls.append(("set_pixel_format", pf))

    def IsGrabbing(self):
        return False

    def set_exposure(self, v):
        self.calls.append(("set_exposure", v))

    def set_gain(self, v):
        self.calls.append(("set_gain", v))

    def configure_trigger(self, source):
        self.calls.append(("configure_trigger", source))

    def last_clamps(self):
        return {}


def _params(**kw):
    p = types.SimpleNamespace(key="xy_basler", camera_type="basler", exposure_time=19e-6,
                              gain=6.0, trigger_source="Line2")
    vars(p).update(kw)
    return p


def test_each_run_sets_its_pixel_format():
    from waxx.util.live_od.camera_nanny import CameraNanny
    cam = _Cam()
    assert CameraNanny().update_params(cam, _params(pixel_format="Mono12p")) is cam
    assert ("set_pixel_format", "Mono12p") in cam.calls


def test_a_payload_without_pixel_format_runs_mono8():
    from waxx.util.live_od.camera_nanny import CameraNanny
    cam = _Cam()
    CameraNanny().update_params(cam, _params())
    assert ("set_pixel_format", "Mono8") in cam.calls


def test_a_driver_that_cannot_set_12_bit_fails_the_run_settings():
    from waxx.util.live_od.camera_nanny import CameraNanny
    cam, report = _Cam(with_pixel_format=False), {}
    out = CameraNanny().update_params(cam, _params(pixel_format="Mono12p"), report=report)
    assert out is not cam and "pixel_format" in report["error"]
