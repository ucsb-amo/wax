"""Per-shot auxiliary camera frames in a run file, and their records.

A camera stream (waxx ``CameraStreamClient`` / ``TriggeredCameraStreamClient``,
kexp ``Base.diag``) saves each frame under its key -- ``ad.data.img_mot`` is
the image, shaped ``(*xvardims, H, W)`` -- and one compact record per shot
under ``<key>_meta``, shaped ``(*xvardims, len(META_FIELDS))``::

    seq       the camera server's frame number (NaN = no frame for this shot)
    t         server-monotonic publish time (s)
    t_target  the time the frame was asked for / the TTL edge, this PC's
              monotonic clock (s)
    hw_ts     the camera's own timestamp counter (NaN = none)
    hw_idx    the camera's own frame counter (NaN = none)
    exposure  exposure time the frame was taken with (s)
    gain      gain the frame was taken with (dB)

Files from 2026-09-30 / 10-01 (before the record was folded into one
container) carry ``<key>_seq`` / ``_t`` / ``_t_target`` (and ``_hw_ts`` /
``_hw_idx`` / ``_exposure`` / ``_gain`` for triggered streams) instead;
``frame_meta`` reads both forms. A shot with no frame holds zeros in the
image container: always mask with ``frames_present`` before using one.
"""
import numpy as np

META_FIELDS = ("seq", "t", "t_target", "hw_ts", "hw_idx", "exposure", "gain")
N_META = len(META_FIELDS)

#: the legacy per-field containers, by META_FIELDS name
_LEGACY_SUFFIX = {"seq": "_seq", "t": "_t", "t_target": "_t_target",
                  "hw_ts": "_hw_ts", "hw_idx": "_hw_idx",
                  "exposure": "_exposure", "gain": "_gain"}


def meta_row(seq=np.nan, t=np.nan, t_target=np.nan, hw_ts=np.nan,
             hw_idx=np.nan, exposure=np.nan, gain=np.nan):
    """One shot's record as a float64 array in META_FIELDS order (NaN =
    unknown / no frame)."""
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    return np.array([_f(seq), _f(t), _f(t_target), _f(hw_ts), _f(hw_idx),
                     _f(exposure), _f(gain)], dtype=np.float64)


def frame_meta(ad, key):
    """``{field: array shaped (*xvardims,)}`` for the frames saved under
    ``key``; NaN where the shot has no frame or the file has no such field."""
    keys = list(ad.data.keys)
    if key + "_meta" in keys:
        m = np.asarray(getattr(ad.data, key + "_meta"), dtype=np.float64)
        return {f: m[..., i] for i, f in enumerate(META_FIELDS)}
    if key not in keys:
        raise KeyError(f"{key!r} is not a data container on this run "
                       f"(have: {keys})")
    img = np.asarray(getattr(ad.data, key))
    lead = img.shape[:-2] if img.ndim >= 2 else img.shape
    out = {}
    for f, suffix in _LEGACY_SUFFIX.items():
        if key + suffix in keys:
            v = np.asarray(getattr(ad.data, key + suffix), dtype=np.float64)
            if f in ("seq", "hw_ts", "hw_idx"):
                v = np.where(v < 0, np.nan, v)      # legacy "none" = -1
            out[f] = v.reshape(lead)
        else:
            out[f] = np.full(lead, np.nan)
    return out


def frames_present(ad, key):
    """Boolean mask shaped ``(*xvardims,)``: True where the shot has a frame
    under ``key``. All True for a file with no record at all (a container
    the kernel filled every shot)."""
    try:
        seq = frame_meta(ad, key)["seq"]
    except KeyError:
        raise
    if np.all(np.isnan(seq)) and key + "_meta" not in ad.data.keys \
            and key + "_seq" not in ad.data.keys:
        img = np.asarray(getattr(ad.data, key))
        return np.ones(img.shape[:-2] if img.ndim >= 2 else img.shape, bool)
    return np.isfinite(seq)


def stream_frame_keys(keys):
    """The data-container keys that hold camera-stream frames: a key with a
    ``<key>_meta`` record (or a legacy ``<key>_seq``) beside it. The records
    themselves are not included."""
    keys = list(keys)
    return [k for k in keys if (k + "_meta") in keys or (k + "_seq") in keys]


__all__ = ["META_FIELDS", "N_META", "meta_row", "frame_meta", "frames_present",
           "stream_frame_keys"]
