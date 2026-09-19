"""AtomdataVault stacking of multi-axis runs along a promoted parameter."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from waxa.atomdata_vault import AtomdataVault

PHASES = np.repeat(np.linspace(0.0, 2 * np.pi, 4), 3)


def _chunk(run_id, squeeze, detunings):
    dims = (PHASES.size, len(detunings))
    # value encodes (run, phase index, detuning) so the scatter can be checked
    an = run_id + np.arange(dims[0])[:, None] * 0.01 + np.asarray(detunings)[None, :] * 1e-9
    data = SimpleNamespace(keys=["apd"], apd=an[..., None] * 2)
    return SimpleNamespace(
        Nvars=2, xvarnames=["relative_phase", "f_det"], xvars=[PHASES, np.asarray(detunings)],
        xvardims=np.array(dims), atom_number=an, od=np.ones((*dims, 3, 4)) * run_id,
        params=SimpleNamespace(v_squeeze=squeeze, N_repeats=3, N_shots=1, t_tof=run_id * 1e-6,
                               relative_phase=PHASES, f_det=np.asarray(detunings)),
        camera_params=SimpleNamespace(), data=data, roi=None, _has_images=False,
        run_info=SimpleNamespace(run_id=run_id, imaging_type=0),
    )


def _vault(chunks, promote="v_squeeze", structure="auto"):
    v = object.__new__(AtomdataVault)
    v._lite, v.server_talk, v._ignore_images = False, None, True
    v._drop_raw_images, v._merge_overlap, v._stacked = False, True, True
    v._timing_enabled, v._timing = False, {}
    v._validate_inputs_nd(chunks, ignore_images=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v._assemble_stacked(chunks, promote, structure, True, None)
    return v


def test_ragged_inner_axis_lands_on_the_union():
    a = _chunk(7, 3.94, [-568e6, -559e6, -550e6])
    b = _chunk(5, 0.444, [-559e6, -540e6])
    v = _vault([a, b])
    assert v.xvarnames == ["v_squeeze", "relative_phase", "f_det"]
    assert np.array_equal(v.xvars[0], [0.444, 3.94])                 # sorted: run 5 first
    assert np.array_equal(v.xvars[2], [-568e6, -559e6, -550e6, -540e6])
    assert v.run_info.run_id == [5, 7] and v.Nvars == 3
    assert v.atom_number.shape == (2, PHASES.size, 4)
    assert np.array_equal(v.atom_number[1][:, :3], a.atom_number)
    assert np.array_equal(v.atom_number[0][:, [1, 3]], b.atom_number)
    assert np.all(np.isnan(v.atom_number[0][:, [0, 2]]))
    assert np.array_equal(v.stack_mask, np.isfinite(v.atom_number))
    assert v.shots_from_run(5).sum() == PHASES.size * 2
    assert v.data.apd.shape == (2, PHASES.size, 4, 1) and v.od.shape[:3] == (2, PHASES.size, 4)


def test_repeat_statistics_reduce_the_shared_phase_axis():
    v = _vault([_chunk(7, 3.94, [-568e6, -559e6]), _chunk(5, 0.444, [-568e6, -559e6])])
    assert v.atom_number.dtype == _chunk(1, 1, [0.0]).atom_number.dtype   # no padding, no cast
    assert v.avg.atom_number.shape == (2, 4, 2)
    assert np.allclose(v.avg.atom_number[0, 0, 0], v.atom_number[0, :3, 0].mean())
    assert np.array_equal(v.avg.xvars[1], np.unique(PHASES))


def test_stack_key_must_be_named_when_ambiguous_and_unique_per_run():
    a, b = _chunk(7, 3.94, [0.0, 1.0]), _chunk(5, 0.444, [0.0, 1.0])
    with pytest.raises(ValueError, match="promote_xvar"):
        _vault([a, b], promote=None)              # v_squeeze and t_tof both differ
    with pytest.raises(NotImplementedError, match="share"):
        _vault([a, _chunk(5, 3.94, [0.0, 1.0])])


def test_flat_only_methods_refuse_a_stacked_vault():
    v = _vault([_chunk(7, 3.94, [0.0, 1.0]), _chunk(5, 0.444, [0.0, 1.0])])
    for call in (lambda: v.set_xvar("t_tof"), lambda: v.drop_runs([5]), v.collapse_to_unique):
        with pytest.raises(NotImplementedError, match="stacked"):
            call()


def test_ragged_axis_cannot_carry_the_repeats():
    a, b = _chunk(7, 3.94, [0.0, 0.0, 1.0]), _chunk(5, 0.444, [0.0, 1.0])
    with pytest.raises(NotImplementedError, match="repeats"):
        _vault([a, b])
