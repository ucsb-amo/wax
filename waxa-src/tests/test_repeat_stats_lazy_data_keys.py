"""Repeat statistics of the data containers: a key at or above
REPEAT_STAT_LAZY_BYTES (the diagnostic frame stacks) is reduced on first
access, once for the avg/std/sem siblings, instead of when the run loads;
small keys are reduced at load as before. Every file is a fresh HDF5 in
tmp_path."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from synthetic_run_file import make_loadable_run

XVAR = (1., 1., 2., 2.)                                   # two values, two repeats each
BIG = np.random.default_rng(11).integers(0, 255, size=(4, 8, 8)).astype(np.float64)   # 2048 B
SMALL = np.array([1., 3., 10., 20.])
GROUPED = BIG.reshape(2, 2, 8, 8)


def _load(tmp_path, monkeypatch):
    import waxa.atomdata_base as ab
    from waxa.atomdata import atomdata
    from waxa.data.server_talk import server_talk
    monkeypatch.setattr(ab, "REPEAT_STAT_LAZY_BYTES", 1024)
    path = make_loadable_run(tmp_path / "2026-10-05", {"big": BIG, "small": SMALL},
                             xvar=XVAR, params={"N_repeats": 2})
    return atomdata(path=path, ignore_images=True,
                    server_talk=server_talk(data_dir=str(tmp_path)))


def test_large_keys_wait_small_keys_do_not(tmp_path, monkeypatch):
    ad = _load(tmp_path, monkeypatch)
    np.testing.assert_array_equal(ad.data.big, BIG)
    assert "small" in vars(ad.avg.data) and "big" not in vars(ad.avg.data)
    assert "big" in ad.avg.data.keys                      # listed all the same
    np.testing.assert_array_equal(ad.avg.data.small, [2., 15.])
    np.testing.assert_array_equal(ad.std.data.small, [1., 5.])


def test_a_large_key_is_reduced_once_for_the_siblings_and_then_kept(tmp_path, monkeypatch):
    import waxa.atomdata_base as ab
    ad = _load(tmp_path, monkeypatch)
    calls = []
    real = ab.atomdata_base._reduce_repeat_ndarray_mean_std

    def counting(self, arr, *a, **k):
        calls.append(np.shape(arr))
        return real(self, arr, *a, **k)

    monkeypatch.setattr(ab.atomdata_base, "_reduce_repeat_ndarray_mean_std", counting)
    np.testing.assert_array_equal(ad.avg.data.big, GROUPED.mean(1))
    np.testing.assert_array_equal(ad.std.data.big, GROUPED.std(1))
    np.testing.assert_allclose(ad.sem.data.big, GROUPED.std(1) / np.sqrt(2))
    assert calls == [(4, 8, 8)]                           # one reduction serves all three
    assert "big" in vars(ad.avg.data) and "big" in vars(ad.std.data)
    # a key added to the parent afterwards still resolves (recomputed, as before)
    ad.data.later = SMALL * 2
    ad.data.keys.append("later")
    np.testing.assert_array_equal(ad.avg.data.later, [4., 30.])
    assert "later" not in vars(ad.avg.data)


def _flat_chunk(run_id, xvals, big, small):
    from types import SimpleNamespace
    xvals = np.asarray(xvals, dtype=float)
    n = xvals.size
    return SimpleNamespace(
        Nvars=1, xvarnames=["t_tof"], xvars=[xvals], xvardims=np.array([n]),
        params=SimpleNamespace(t_tof=xvals, N_repeats=1, N_shots=n,
                               N_shots_with_repeats=n, N_pwa_per_shot=1),
        camera_params=SimpleNamespace(), roi=None, _has_images=False,
        data=SimpleNamespace(keys=["big", "small"], big=big, small=small), scope_data={},
        run_info=SimpleNamespace(run_id=run_id, imaging_type=0),
        _analysis_tags=SimpleNamespace(xvars_shuffled=False),
        images=np.array([]), image_timestamps=np.array([]),
    )


def test_the_vault_groups_large_keys_on_first_access_too(monkeypatch):
    import waxa.atomdata_vault as av
    from waxa.atomdata_vault import AtomdataVault
    monkeypatch.setattr(av, "REPEAT_STAT_LAZY_BYTES", 1024)
    # two runs over the same two values -> four shots, grouped two by two
    a = _flat_chunk(7, [1., 2.], BIG[:2], SMALL[:2])
    b = _flat_chunk(5, [1., 2.], BIG[2:], SMALL[2:])
    monkeypatch.setattr(AtomdataVault, "_materialize_inputs", lambda self, *x, **k: [a, b])
    v = AtomdataVault([7, 5], ignore_images=True, auto_lite_threshold=None)
    assert v.data.big.nbytes == 2048 and v.xvars[0].tolist() == [1., 1., 2., 2.]
    assert "small" in vars(v.avg.data) and "big" not in vars(v.avg.data)
    assert "big" in v.avg.data.keys
    grouped = v.data.big.reshape(2, 2, 8, 8)              # shots sorted by value
    calls = []
    real = AtomdataVault._grouped_array_stats

    def counting(self, arr):
        calls.append(np.shape(arr))
        return real(self, arr)

    monkeypatch.setattr(AtomdataVault, "_grouped_array_stats", counting)
    np.testing.assert_array_equal(v.avg.data.big, grouped.mean(1))
    np.testing.assert_array_equal(v.std.data.big, grouped.std(1))
    np.testing.assert_allclose(v.sem.data.big, grouped.std(1) / np.sqrt(2))
    assert calls == [(4, 8, 8)]                           # grouped once for all three
    assert "big" in vars(v.std.data) and "big" in vars(v.sem.data)
    np.testing.assert_array_equal(v.avg.data.small, v.data.small.reshape(2, 2).mean(1))


def test_the_default_threshold_leaves_ordinary_keys_eager(tmp_path, monkeypatch):
    import waxa.atomdata_base as ab
    from waxa.atomdata import atomdata
    from waxa.data.server_talk import server_talk
    path = make_loadable_run(tmp_path / "2026-10-05", {"big": BIG, "small": SMALL},
                             xvar=XVAR, params={"N_repeats": 2})
    ad = atomdata(path=path, ignore_images=True, server_talk=server_talk(data_dir=str(tmp_path)))
    assert ab.REPEAT_STAT_LAZY_BYTES == 32 * 1024 ** 2
    assert "big" in vars(ad.avg.data) and "small" in vars(ad.avg.data)
