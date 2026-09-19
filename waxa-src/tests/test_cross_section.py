"""Tests for the per-shot absorption cross section chosen from the recorded
outer-coil current (waxa.calibrations.cross_section) and its use in
atomdata_base.compute_atom_number.

The synthetic tests run anywhere. The kamo cross-check skips when kamo is not
installed.
"""

import numpy as np
import pytest

from waxa.calibrations import cross_section as cs
from waxa.atomdata_base import atomdata_base


class _Data:
    """Minimal stand-in for an atomdata DataVault: a keys list plus arrays."""

    def __init__(self, **arrays):
        self.keys = list(arrays)
        for k, v in arrays.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# cross_section_from_outer_current
# ---------------------------------------------------------------------------

def test_switch_on_threshold():
    i = np.array([0.0, 0.5, 0.999, 1.0, 1.5, 182.0, 194.0])
    sigma, source = cs.cross_section_from_outer_current(i)
    low = i < cs.I_OUTER_HF_THRESHOLD_A
    assert np.all(sigma[low] == cs.SIGMA_LF_M2)
    assert np.all(sigma[~low] == cs.SIGMA_HF_M2)
    assert np.all(source[low] == cs.TAG_LF)
    assert np.all(source[~low] == cs.TAG_HF)
    assert sigma.dtype == np.float64
    assert sigma.shape == source.shape == i.shape


def test_none_means_fallback():
    sigma, source = cs.cross_section_from_outer_current(None)
    assert sigma.shape == ()
    assert float(sigma) == cs.SIGMA_FALLBACK_M2
    assert str(source) == cs.TAG_FALLBACK


def test_nan_entries_fall_back_individually():
    # a vault chunk without the container is NaN-padded
    i = np.array([182.0, np.nan, 0.0])
    sigma, source = cs.cross_section_from_outer_current(i)
    assert sigma.tolist() == [cs.SIGMA_HF_M2, cs.SIGMA_FALLBACK_M2, cs.SIGMA_LF_M2]
    assert source.tolist() == [cs.TAG_HF, cs.TAG_FALLBACK, cs.TAG_LF]


def test_2d_scan_shape_preserved():
    i = np.full((3, 4), 182.0)
    i[1, 2] = 0.0
    sigma, source = cs.cross_section_from_outer_current(i)
    assert sigma.shape == (3, 4)
    assert sigma[1, 2] == cs.SIGMA_LF_M2
    assert np.sum(sigma == cs.SIGMA_HF_M2) == 11


# ---------------------------------------------------------------------------
# recorded_outer_current / cross_section_for_run: safe on old data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data", [
    None,
    object(),                                     # no keys attribute at all
    _Data(),                                      # empty keys
    _Data(b=np.zeros(5)),                         # other containers only
])
def test_missing_record_is_none(data):
    assert cs.recorded_outer_current(data) is None
    sigma, source = cs.cross_section_for_run(data, (5,))
    assert sigma.shape == (5,) and source.shape == (5,)
    assert np.all(sigma == cs.SIGMA_FALLBACK_M2)
    assert np.all(source == cs.TAG_FALLBACK)


def test_recorded_current_used():
    data = _Data(i_outer_imaging=np.array([182.0, 0.0, 182.0]))
    sigma, source = cs.cross_section_for_run(data, (3,))
    assert sigma.tolist() == [cs.SIGMA_HF_M2, cs.SIGMA_LF_M2, cs.SIGMA_HF_M2]


def test_trailing_axis_tolerated():
    data = _Data(i_outer_imaging=np.full((3, 1), 182.0))
    sigma, _ = cs.cross_section_for_run(data, (3,))
    assert sigma.shape == (3,)
    assert np.all(sigma == cs.SIGMA_HF_M2)


def test_shape_mismatch_falls_back():
    data = _Data(i_outer_imaging=np.full(7, 182.0))
    sigma, source = cs.cross_section_for_run(data, (3,))
    assert sigma.shape == (3,)
    assert np.all(source == cs.TAG_FALLBACK)


# ---------------------------------------------------------------------------
# compute_atom_number on a synthetic atomdata
# ---------------------------------------------------------------------------

class _Cam:
    pixel_size_m = 6.5e-6
    magnification = 2.0


def _synthetic_ad(i_outer, xvardims=(3,), od_value=0.5, ny=4, nx=6):
    ad = object.__new__(atomdata_base)
    ad.xvardims = list(xvardims)
    ad.camera_params = _Cam()
    ad.od = np.full((*xvardims, ny, nx), od_value)
    ad.fit_area_x = np.full(xvardims, 2.0)
    ad.fit_area_y = np.full(xvardims, 3.0)
    ad.data = None if i_outer is None else _Data(i_outer_imaging=np.asarray(i_outer, dtype=float))
    return ad


def test_compute_atom_number_scales_inversely_with_sigma():
    ad = _synthetic_ad(i_outer=[182.0, 0.0, 182.0])
    ad.compute_atom_number()
    dx = _Cam.pixel_size_m / _Cam.magnification
    expected_hf = 0.5 * 4 * 6 * dx**2 / cs.SIGMA_HF_M2
    expected_lf = 0.5 * 4 * 6 * dx**2 / cs.SIGMA_LF_M2
    assert ad.atom_number.shape == (3,)
    assert np.allclose(ad.atom_number, [expected_hf, expected_lf, expected_hf])
    assert np.allclose(ad.atom_number_fit_area_x,
                       2.0 * dx / np.array([cs.SIGMA_HF_M2, cs.SIGMA_LF_M2, cs.SIGMA_HF_M2]))
    assert ad.atom_number_density.shape == (3, 4, 6)
    assert ad.atom_cross_section.shape == (3,)
    assert ad.atom_cross_section_source.tolist() == [cs.TAG_HF, cs.TAG_LF, cs.TAG_HF]


def test_compute_atom_number_without_record_matches_high_field():
    ad_old = _synthetic_ad(i_outer=None)
    ad_new = _synthetic_ad(i_outer=[182.0, 182.0, 182.0])
    ad_old.compute_atom_number()
    ad_new.compute_atom_number()
    assert np.allclose(ad_old.atom_number, ad_new.atom_number)
    assert np.all(ad_old.atom_cross_section_source == cs.TAG_FALLBACK)


def test_compute_atom_number_2d_scan():
    i = np.full((2, 3), 182.0)
    ad = _synthetic_ad(i_outer=i, xvardims=(2, 3))
    ad.compute_atom_number()
    assert ad.atom_number.shape == (2, 3)
    assert ad.atom_cross_section.shape == (2, 3)


# ---------------------------------------------------------------------------
# provenance cross-check against kamo
# ---------------------------------------------------------------------------

def test_high_field_value_matches_kamo():
    kamo = pytest.importorskip("kamo")
    atom = kamo.Potassium39()
    # closed sigma- D2 line (4,0,1/2,-1/2,m_i) -> (4,1,3/2,-3/2,m_i) at the
    # high-field imaging point, 3 lambda^2 / 2 pi
    f_Hz = atom.get_transition_frequency((4, 0, 0.5, -0.5, -0.5),
                                         (4, 1, 1.5, -1.5, -0.5),
                                         B=520.583, relative_mode="absolute")
    f_Hz = float(np.asarray(f_Hz).ravel()[0])
    lam = 299792458.0 / f_Hz
    assert np.isclose(3 * lam**2 / (2 * np.pi), cs.SIGMA_HF_M2, rtol=1e-4)


# ---------------------------------------------------------------------------
# the numbers come from kamo; the built-in copy must never drift from it
# ---------------------------------------------------------------------------

def test_values_come_from_kamo():
    kamo_cs = pytest.importorskip("kamo.imaging.cross_sections")
    assert cs.SIGMA_HF_M2 == kamo_cs.K39_D2_CLOSED_HIGH_FIELD_M2
    assert cs.SIGMA_LF_M2 == kamo_cs.K39_LEGACY_LAMBDA_SQUARED_M2
    assert cs.SIGMA_LEGACY_D1_M2 == kamo_cs.K39_LEGACY_D1_M2
    assert cs.SIGMA_FALLBACK_M2 == cs.SIGMA_HF_M2


def test_fallback_copy_matches_kamo(monkeypatch):
    """An older kamo has no cross_sections module: waxa must warn and carry on
    with the same numbers."""
    import importlib, sys
    kamo_cs = pytest.importorskip("kamo.imaging.cross_sections")
    expected = (kamo_cs.K39_D2_CLOSED_HIGH_FIELD_M2,
                kamo_cs.K39_LEGACY_LAMBDA_SQUARED_M2,
                kamo_cs.K39_LEGACY_D1_M2)
    monkeypatch.setitem(sys.modules, "kamo.imaging.cross_sections", None)  # import -> ImportError
    try:
        with pytest.warns(UserWarning, match="update k-amo"):
            fallback = importlib.reload(cs)
        assert (fallback.SIGMA_HF_M2, fallback.SIGMA_LF_M2, fallback.SIGMA_LEGACY_D1_M2) == expected
    finally:
        monkeypatch.undo()
        importlib.reload(cs)
