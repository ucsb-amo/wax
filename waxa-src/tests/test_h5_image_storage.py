"""Image arrays are stored chunked per frame and gzip-compressed; everything
else stays plain; reading needs nothing special. Fresh files in tmp_path."""
import h5py
import numpy as np

from waxa.data.h5_image import (COMPRESSION_LEVEL, MIN_BYTES, create_image_dataset,
                                image_dataset_kwargs)


def test_only_large_integer_image_stacks_are_compressed():
    big = image_dataset_kwargs((25, 600, 600), np.uint8)
    assert big["chunks"] == (1, 600, 600) and big["compression"] == "gzip"
    assert big["compression_opts"] == COMPRESSION_LEVEL and big["shuffle"] is False
    wide = image_dataset_kwargs((3, 512, 512), np.uint16)
    assert wide["shuffle"] is True
    assert image_dataset_kwargs((4, 20, 30), np.uint8) == {}          # small
    assert image_dataset_kwargs((25, 7), np.float64) == {}            # records, not frames
    assert image_dataset_kwargs((10, 600, 600), np.float32) == {}     # floats stay plain
    assert image_dataset_kwargs((0,), np.uint16) == {}                # nothing yet
    assert MIN_BYTES == 256 * 1024


def test_round_trip_and_per_frame_writes(tmp_path):
    rng = np.random.default_rng(0)
    frames = np.zeros((6, 400, 500), dtype=np.uint8)           # dark with a bright blob
    frames[:, 150:250, 200:300] = rng.integers(0, 255, size=(6, 100, 100), dtype=np.uint8)
    path = tmp_path / "run.hdf5"
    with h5py.File(path, "w") as f:
        d = f.create_group("data")
        create_image_dataset(d, "img_a", data=frames)
        create_image_dataset(d, "images", shape=(3, 512, 512), dtype=np.uint16)
        d["images"][1] = np.full((512, 512), 7, np.uint16)    # written frame by frame
        create_image_dataset(d, "apd", data=np.arange(4.0))
    with h5py.File(path, "r") as f:
        d = f["data"]
        assert d["img_a"].compression == "gzip" and d["img_a"].chunks == (1, 400, 500)
        assert np.array_equal(d["img_a"][()], frames)
        assert d["images"].compression == "gzip" and d["images"].shuffle
        assert d["images"][1][0, 0] == 7 and d["images"][0].max() == 0
        assert d["apd"].compression is None
        assert np.array_equal(d["apd"][()], np.arange(4.0))
    # mostly-dark frames shrink a lot
    assert path.stat().st_size < frames.nbytes / 2
