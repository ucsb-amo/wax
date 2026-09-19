"""AtomdataVault.remap_xvar and data_container_to_xvar: relabelling an axis in place."""

import warnings

import numpy as np
import pytest

from test_vault_stacked import PHASES, _chunk, _vault

DETUNINGS = [-568e6, -559e6, -550e6]


def _two_run_vault():
    v = _vault([_chunk(7, 3.94, DETUNINGS), _chunk(5, 0.444, DETUNINGS)])
    # scan-shaped containers: one constant along the other axes, one per-shot noisy
    dims = tuple(int(d) for d in v.xvardims)
    v.data.f_meas = np.broadcast_to(np.asarray(DETUNINGS) + 1e3, dims).copy()
    v.data.p_meas = (np.array([1.0, 9.0])[:, None, None]
                     + 0.01 * np.arange(dims[1])[None, :, None] + np.zeros(dims))
    v.data.keys = list(v.data.keys) + ["f_meas", "p_meas"]
    return v


def test_func_generates_a_new_xvar_and_param_by_index_or_key():
    v = _two_run_vault()
    an = v.atom_number.copy()
    out = v.remap_xvar(0, lambda volts: 2.0e-3 * volts / 0.444, "p_tweezer")
    assert out is v
    assert v.xvarnames == ["p_tweezer", "relative_phase", "f_det"]
    assert np.allclose(v.xvars[0], [2.0e-3, 2.0e-3 * 3.94 / 0.444])
    assert v.params.p_tweezer is v.xvars[0]
    assert np.array_equal(v.params.v_squeeze, [0.444, 3.94])          # old param kept
    assert np.array_equal(v.atom_number, an, equal_nan=True)          # layout untouched
    assert v.xvar_remaps == [(0, "v_squeeze", "p_tweezer")]

    v.remap_xvar("f_det", lambda f: f + 515e6, "detune_mid")
    assert v.xvarnames[2] == "detune_mid" and np.allclose(v.xvars[2], [-53e6, -44e6, -35e6])
    assert v.avg.xvarnames[2] == "detune_mid"                         # siblings follow
    assert np.array_equal(v.avg.xvars[1], np.unique(PHASES))


def test_scalar_only_func_is_applied_elementwise():
    import math
    v = _two_run_vault()
    v.remap_xvar(-1, lambda f: math.floor(f / 1e6), "f_MHz")
    assert np.array_equal(v.xvars[2], [-568, -559, -550])


def test_existing_param_is_swapped_in_without_a_func():
    v = _two_run_vault()
    v.params.p_cal = np.array([1.5, 13.0])
    v.remap_xvar("v_squeeze", "p_cal")                                # shorthand form
    assert v.xvarnames[0] == "p_cal" and np.array_equal(v.xvars[0], [1.5, 13.0])
    v.remap_xvar(0, new_key="v_squeeze")                              # and back
    assert v.xvarnames[0] == "v_squeeze" and np.array_equal(v.xvars[0], [0.444, 3.94])


def test_bad_requests_raise():
    v = _two_run_vault()
    with pytest.raises(KeyError, match="existing param"):
        v.remap_xvar(0, new_key="not_a_param")
    with pytest.raises(ValueError, match="shape"):
        v.remap_xvar(0, new_key="f_det")                              # (3,) onto a (2,) axis
    with pytest.raises(TypeError, match="new_key"):
        v.remap_xvar(0, lambda x: x)
    with pytest.raises(KeyError, match="not an xvar"):
        v.remap_xvar("nope", lambda x: x, "k")
    with pytest.raises(IndexError):
        v.remap_xvar(3, lambda x: x, "k")
    with pytest.raises(ValueError, match="another axis"):
        v.remap_xvar(0, lambda x: x, "f_det")
    with pytest.raises(ValueError, match="NaN"):
        v.remap_xvar(0, lambda x: x * np.nan, "k")
    v.params.taken = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="overwrite"):
        v.remap_xvar(0, lambda x: x * 2, "taken")
    v.remap_xvar(0, lambda x: x * 2, "taken", overwrite=True)
    assert np.allclose(v.params.taken, [0.888, 7.88])
    assert v.xvarnames == ["taken", "relative_phase", "f_det"]        # failures left no trace


def test_many_to_one_map_warns():
    v = _two_run_vault()
    with pytest.warns(UserWarning, match="distinct"):
        v.remap_xvar(0, lambda x: np.zeros_like(x), "flat")


def test_data_container_to_xvar_constant_and_reduced():
    v = _two_run_vault()
    v.data_container_to_xvar("f_det", "f_meas")
    assert v.xvarnames[2] == "f_meas" and np.allclose(v.xvars[2], np.asarray(DETUNINGS) + 1e3)
    assert np.allclose(v.params.f_meas, v.xvars[2])

    with pytest.raises(ValueError, match="varies"):
        v.data_container_to_xvar(0, "p_meas")
    v.data_container_to_xvar(0, "p_meas", reduce="first")
    assert np.allclose(v.xvars[0], [1.0, 9.0])
    v.data_container_to_xvar(0, "p_meas", lambda p: 1e-3 * p, "p_W", reduce="mean")
    mean = 1.0 + 0.01 * np.arange(PHASES.size).mean()
    assert v.xvarnames[0] == "p_W" and np.allclose(v.xvars[0], [1e-3 * mean, 1e-3 * (mean + 8)])


def test_data_container_to_xvar_skips_stack_holes_and_rejects_bad_input():
    v = _vault([_chunk(7, 3.94, DETUNINGS), _chunk(5, 0.444, DETUNINGS[:2])])
    dims = tuple(int(d) for d in v.xvardims)
    rec = np.broadcast_to(np.asarray(DETUNINGS) * 2, dims).copy()
    rec[~v.stack_mask] = np.nan                                        # run 5 never took -550
    v.data.rec = rec
    v.data.keys = list(v.data.keys) + ["rec"]
    v.data_container_to_xvar(2, "rec")
    assert np.allclose(v.xvars[2], np.asarray(DETUNINGS) * 2)

    with pytest.raises(KeyError, match="not a data container"):
        v.data_container_to_xvar(0, "missing")
    with pytest.raises(ValueError, match="shape"):
        v.data_container_to_xvar(0, "apd")                            # trailing per-shot axis
    with pytest.raises(ValueError, match="reduce must be"):
        v.data_container_to_xvar(0, "rec", reduce="max")


def test_power_prefix_gets_a_unit():
    from waxa.plotting.units import guess_unit
    assert guess_unit("power_tweezer", [7.3e-5, 6.5e-4]) == ("µW", 1e6)
    assert guess_unit("power_tweezer", [0.2]) == ("mW", 1e3)
