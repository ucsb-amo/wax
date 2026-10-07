"""How a run file is opened for reading: the whole-file modes ('core', one
read into memory; 'warm', one pass into the OS cache) against the
chunk-by-chunk default, chosen by file size and available memory, all
reading the same bytes. Every file is a fresh HDF5 in tmp_path."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

import waxa.atomdata_base as ab
from synthetic_run_file import make_loadable_run

GiB = 1024 ** 3


def _file(tmp_path, n=6, h=32, w=24):
    path = str(tmp_path / "run.hdf5")
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 4000, size=(n, h, w), dtype=np.uint16)
    stack = rng.integers(0, 255, size=(n, 8, 8), dtype=np.uint8)
    with h5py.File(path, "w") as f:
        g = f.create_group("data")
        g.create_dataset("images", data=frames, chunks=(1, h, w), compression="gzip", shuffle=True)
        g.create_dataset("img_x", data=stack, chunks=(1, 8, 8), compression="gzip")
        g["apd"] = np.linspace(0, 1, n)
        f.attrs["xvarnames"] = ["t"]
    return path, frames, stack


def test_auto_picks_by_size_against_the_caps():
    choose = ab.choose_h5_read_mode
    assert choose("x", mode="auto", size=100e6, available=96 * GiB) == "core"
    assert choose("x", mode="auto", size=4 * GiB, available=96 * GiB) == "core"
    # above the fixed cap: warm while it fits half the memory, else direct
    assert choose("x", mode="auto", size=5 * GiB, available=96 * GiB) == "warm"
    assert choose("x", mode="auto", size=5 * GiB, available=8 * GiB) == "direct"
    # under the cap but more than a quarter of what is free: warm, then direct
    assert choose("x", mode="auto", size=3 * GiB, available=8 * GiB) == "warm"
    assert choose("x", mode="auto", size=3 * GiB, available=4 * GiB) == "direct"


def test_unknown_memory_uses_the_fixed_cap_only(monkeypatch):
    monkeypatch.setattr(ab, "_available_memory_bytes", lambda: None)
    assert ab.choose_h5_read_mode("x", mode="auto", size=3 * GiB) == "core"
    assert ab.choose_h5_read_mode("x", mode="auto", size=5 * GiB) == "direct"


def test_a_named_mode_wins_and_a_missing_file_reads_direct(tmp_path, monkeypatch):
    for m in ("core", "warm", "direct"):
        assert ab.choose_h5_read_mode("x", mode=m, size=1, available=1) == m
    monkeypatch.setattr(ab, "H5_READ_MODE", "warm")           # the process-wide default
    assert ab.choose_h5_read_mode("x", size=1, available=10 ** 12) == "warm"
    with pytest.raises(ValueError, match="unknown h5 read mode"):
        ab.choose_h5_read_mode("x", mode="fast")
    assert ab.choose_h5_read_mode(str(tmp_path / "nope.hdf5"), mode="auto") == "direct"


@pytest.mark.parametrize("mode", ["core", "warm", "direct"])
def test_every_mode_reads_the_same_arrays(tmp_path, mode):
    path, frames, stack = _file(tmp_path)
    before = open(path, "rb").read()
    f, used = ab.open_run_file(path, mode)
    with f:
        assert used == mode
        assert (f.driver == "core") == (mode == "core")
        np.testing.assert_array_equal(f["data"]["images"][()], frames)
        np.testing.assert_array_equal(f["data"]["img_x"][()], stack)
        np.testing.assert_array_equal(f["data"]["apd"][()], np.linspace(0, 1, 6))
        assert list(f.attrs["xvarnames"]) == ["t"]
    assert open(path, "rb").read() == before                   # the file is untouched


def test_a_core_open_that_cannot_get_memory_falls_back(tmp_path, monkeypatch):
    path, frames, _ = _file(tmp_path)
    real = ab.h5py.File

    class NoMemory(real):
        def __init__(self, name, *a, **k):
            if k.get("driver") == "core":
                raise MemoryError("no room")
            super().__init__(name, *a, **k)

    monkeypatch.setattr(ab.h5py, "File", NoMemory)
    with pytest.warns(UserWarning, match="chunk by chunk"):
        f, used = ab.open_run_file(path, "core")
    with f:
        assert used == "direct"
        np.testing.assert_array_equal(f["data"]["images"][()], frames)


@pytest.mark.parametrize("mode", ["core", "warm", "direct"])
def test_atomdata_loads_the_same_run_through_every_mode(tmp_path, monkeypatch, mode):
    from waxa.atomdata import atomdata
    from waxa.data.server_talk import server_talk
    rng = np.random.default_rng(9)
    stack = rng.integers(0, 255, size=(4, 5, 6), dtype=np.uint8)
    apd = np.array([0.5, 0.25, 0.125, 1.0])
    path = make_loadable_run(tmp_path / "2026-10-05", {"img_mot": stack, "apd": apd})
    talk = server_talk(data_dir=str(tmp_path))
    monkeypatch.setattr(ab, "H5_READ_MODE", mode)
    ad = atomdata(path=path, ignore_images=True, server_talk=talk)
    assert ad._h5_read_mode == mode
    np.testing.assert_array_equal(ad.data.img_mot, stack)
    np.testing.assert_array_equal(ad.data.apd, apd)
    np.testing.assert_array_equal(ad.xvars[0], [1., 2., 3., 4.])
