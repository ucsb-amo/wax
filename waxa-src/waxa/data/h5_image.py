"""How image arrays are stored in a run file.

Frames are mostly dark and compress well, so an integer array of at least
MIN_BYTES with a trailing (H, W) is written chunked one frame at a time and
gzip-compressed (level COMPRESSION_LEVEL; byte-shuffled when wider than 8
bit, which helps gzip on 16-bit data). Everything else is stored plain.
Reading needs nothing special: h5py decompresses on access.

    create_image_dataset(group, 'images', shape=(n, H, W), dtype=np.uint16)
    create_image_dataset(group, 'img_mot', data=frames)
"""
import numpy as np

COMPRESSION_LEVEL = 2
MIN_BYTES = 256 * 1024
# The run's own camera frames (data/images, written by liveOD one frame at a
# time) too, not only the auxiliary-camera containers. Off = plain, as before.
COMPRESS_MAIN_IMAGES = True


def image_dataset_kwargs(shape, dtype):
    """The h5py ``create_dataset`` options for an array of this shape and
    dtype: chunked + gzip for a large integer image stack, ``{}`` otherwise."""
    shape = tuple(int(s) for s in shape)
    dtype = np.dtype(dtype)
    if len(shape) < 2 or not np.issubdtype(dtype, np.integer):
        return {}
    if int(np.prod(shape)) * dtype.itemsize < MIN_BYTES:
        return {}
    return {"chunks": (1,) * (len(shape) - 2) + shape[-2:],
            "compression": "gzip", "compression_opts": COMPRESSION_LEVEL,
            "shuffle": dtype.itemsize > 1}


def create_image_dataset(group, name, data=None, shape=None, dtype=None, **kw):
    """``group.create_dataset(name, ...)`` with the storage options above
    (from ``data``, or from ``shape`` / ``dtype`` for an empty dataset to be
    filled later)."""
    if data is not None:
        data = np.asarray(data)
        shape, dtype = data.shape, data.dtype
    opts = image_dataset_kwargs(shape, dtype)
    opts.update(kw)
    if data is not None:
        return group.create_dataset(name, data=data, **opts)
    return group.create_dataset(name, shape=shape, dtype=dtype, **opts)


__all__ = ["COMPRESSION_LEVEL", "MIN_BYTES", "image_dataset_kwargs", "create_image_dataset"]
