"""Repeat statistics when the repeats are not consecutive (#209): a
shuffle=False run whose xvar interleaves its repeats (np.tile(values, N)) is
saved in the order taken, so ad.avg / ad.std / ad.sem group the shots by value
instead of falling back to the unaveraged shots with zero spread. Every file
is a fresh HDF5 in tmp_path."""
import os
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from synthetic_run_file import make_loadable_run

XVAR = (3., 1., 2., 3., 1., 2.)                       # np.tile([3, 1, 2], 2)
SMALL = np.array([30., 10., 20., 32., 14., 26.])
BIG = np.random.default_rng(5).integers(0, 255, size=(6, 8, 8)).astype(np.float64)
ORDER = [0, 3, 1, 4, 2, 5]                            # shots of 3, then 1, then 2


def _load(tmp_path, xvar=XVAR, containers=None, n_repeats=1):
    from waxa.atomdata import atomdata
    from waxa.data.server_talk import server_talk
    containers = containers or {"small": SMALL, "big": BIG}
    path = make_loadable_run(tmp_path / "2026-10-05", containers, xvar=xvar,
                             params={"N_repeats": n_repeats})
    return atomdata(path=path, ignore_images=True,
                    server_talk=server_talk(data_dir=str(tmp_path)))


def test_interleaved_repeats_are_grouped_by_value_in_declared_order(tmp_path):
    ad = _load(tmp_path)
    np.testing.assert_array_equal(ad.xvars[0], XVAR)          # the source is untouched
    np.testing.assert_array_equal(ad.data.small, SMALL)
    np.testing.assert_array_equal(ad.avg.xvars[0], [3., 1., 2.])
    np.testing.assert_array_equal(ad.avg.data.small, [31., 12., 23.])
    np.testing.assert_array_equal(ad.std.data.small, [1., 2., 3.])
    np.testing.assert_allclose(ad.sem.data.small, np.array([1., 2., 3.]) / np.sqrt(2))
    grouped = BIG[ORDER].reshape(3, 2, 8, 8)
    np.testing.assert_array_equal(ad.avg.data.big, grouped.mean(1))
    np.testing.assert_array_equal(ad.std.data.big, grouped.std(1))


def test_avg_array_groups_interleaved_repeats_too(tmp_path):
    ad = _load(tmp_path)
    avg, std, sem = ad.avg_array(SMALL * 2)
    np.testing.assert_array_equal(avg, [62., 24., 46.])
    np.testing.assert_array_equal(std, [2., 4., 6.])
    np.testing.assert_allclose(sem, np.array([2., 4., 6.]) / np.sqrt(2))
    np.testing.assert_array_equal(ad.std_array(SMALL), [1., 2., 3.])


def test_consecutive_repeats_are_unchanged(tmp_path):
    ad = _load(tmp_path, xvar=(3., 3., 1., 1., 2., 2.))
    np.testing.assert_array_equal(ad.avg.xvars[0], [3., 1., 2.])
    np.testing.assert_array_equal(ad.avg.data.small, [20., 26., 20.])   # (30,10) (20,32) (14,26)


def test_ungroupable_repeats_warn_instead_of_returning_silent_zeros(tmp_path):
    with pytest.warns(UserWarning, match="not averaged"):
        ad = _load(tmp_path, xvar=(1., 1., 2., 1.), containers={"small": SMALL[:4]})
    np.testing.assert_array_equal(ad.avg.data.small, SMALL[:4])
    np.testing.assert_array_equal(ad.std.data.small, np.zeros(4))


def test_a_run_without_repeats_loads_without_a_warning(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ad = _load(tmp_path, xvar=(1., 2., 3., 4.), containers={"small": SMALL[:4]})
    np.testing.assert_array_equal(ad.avg.data.small, SMALL[:4])
