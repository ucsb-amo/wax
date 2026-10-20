"""AtomdataVault with runs whose Basler frames differ in bit depth (Mono8 uint8
vs Mono12 / Mono12p uint16): data keys are stacked in a dtype that holds every
run's values (never wrapped to the first run's), and the vault says the counts
are not on one scale."""
import types
import warnings

import numpy as np
import pytest


def _dv(**arrays):
    return types.SimpleNamespace(keys=list(arrays), **arrays)


def _chunk(run_id, n, data, pixel_format=None, images_dtype=np.uint8):
    cp = types.SimpleNamespace(key="xy_basler", camera_type="basler")
    if pixel_format is not None:
        cp.pixel_format = pixel_format
    return types.SimpleNamespace(
        camera_params=cp, run_info=types.SimpleNamespace(run_id=run_id), camera_overrides={},
        xvars=[np.arange(n)], data=data, _has_images=True,
        images=np.zeros((3 * n, 2, 2), dtype=images_dtype))


def test_a_12_bit_frame_after_an_8_bit_run_is_not_wrapped():
    from waxa.atomdata_vault import AtomdataVault
    vault = object.__new__(AtomdataVault)
    a = _chunk(1, 2, _dv(img_gm=np.full((2, 3, 3), 200, np.uint8)))
    b = _chunk(2, 2, _dv(img_gm=np.full((2, 3, 3), 4000, np.uint16)), pixel_format="Mono12p")
    with pytest.warns(UserWarning, match="not on one scale"):
        dv = vault._concat_data_vaults([a, b])
    out = dv.img_gm
    assert out.dtype == np.uint16
    assert int(out[0].max()) == 200 and int(out[2].max()) == 4000     # 4000 % 256 would be 160


def test_same_dtype_stays_and_is_quiet():
    from waxa.atomdata_vault import AtomdataVault
    vault = object.__new__(AtomdataVault)
    a = _chunk(1, 2, _dv(img_gm=np.full((2, 3, 3), 7, np.uint8)))
    b = _chunk(2, 2, _dv(img_gm=np.full((2, 3, 3), 9, np.uint8)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dv = vault._concat_data_vaults([a, b])
    assert dv.img_gm.dtype == np.uint8


def test_a_run_without_pixel_format_counts_as_mono8():
    from waxa.atomdata_vault import AtomdataVault
    vault = object.__new__(AtomdataVault)
    old = _chunk(1, 1, _dv())                                   # before the field existed
    new = _chunk(2, 1, _dv(), pixel_format="Mono12p", images_dtype=np.uint16)
    with pytest.warns(UserWarning) as caught:
        vault._note_camera_differences([old, new])
    text = " ".join(str(w.message) for w in caught)
    assert "pixel_format" in text and "not on one scale" not in text   # that one is for data keys
    assert vault.camera_param_disagreements["pixel_format"] == {1: "Mono8", 2: "Mono12p"}
    assert vault.image_dtype_by_run == {1: "uint8", 2: "uint16"}
    assert "raw counts are not" in text


def test_old_runs_only_are_quiet():
    from waxa.atomdata_vault import AtomdataVault
    vault = object.__new__(AtomdataVault)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vault._note_camera_differences([_chunk(1, 1, _dv()), _chunk(2, 1, _dv())])
    assert vault.camera_param_disagreements == {}
    assert vault.image_dtype_by_run == {1: "uint8", 2: "uint8"}
