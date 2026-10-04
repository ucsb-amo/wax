"""Weighted image moments for fluorescence image stacks (MOT diagnostics).

Per-shot integrated counts, centroid and rms width of a bright cloud on a
dark background -- the quantities that proved useful as drift diagnostics for
the auxiliary MOT / 2D-MOT camera streams (``ad.data.img_mot`` etc.).

Weights are the frame minus its median (a robust background estimate for a
mostly-dark frame), clipped at zero; no ROI or threshold is tuned. Clipped
noise leaves a small positive pedestal across the frame, so the rms widths are
biased above the cloud's true size -- treat them as *relative* drift
diagnostics, not calibrated cloud sizes. A **fast
path** (``downsample=2``, the default) bins the frame 2x2 before the moment
sums; centroids and widths are always returned in *full-frame pixels* (the
coordinates are scaled back), so results from different ``downsample``
settings agree to well under a pixel for a many-pixel-wide cloud.
``downsample=1`` is the exact path.

Typical use::

    from waxa.image_processing.cloud_moments import mot_diagnostics
    d = mot_diagnostics(ad, "img_mot")
    d["sum"], d["centroid_x"], d["rms_y"]       # arrays (*xvardims,)
"""

from __future__ import annotations

import numpy as np

_KEYS = ("sum", "centroid_x", "centroid_y", "rms_x", "rms_y")


def image_moments(stack, downsample: int = 2) -> dict[str, np.ndarray]:
    """Per-frame weighted moments of a stack shaped ``(n, H, W)``.

    Returns ``{"sum", "centroid_x", "centroid_y", "rms_x", "rms_y"}``, each
    shaped ``(n,)``. ``sum`` is the total background-subtracted counts (of the
    downsampled frame, scaled by ``downsample**2`` so it estimates the
    full-frame value); centroids/widths are in full-frame pixels. Frames that
    contain non-finite values, or whose background-subtracted weight sums to
    zero, come back NaN.
    """
    stack = np.asarray(stack, float)
    if stack.ndim != 3:
        raise ValueError(f"expected (n, H, W), got shape {stack.shape}")
    ds = int(downsample)
    if ds < 1:
        raise ValueError("downsample must be >= 1")
    s = stack[:, ::ds, ::ds]
    n, H, W = s.shape
    # full-frame pixel coordinates of each downsampled sample
    ys = (np.arange(H) * ds)[None, :, None]
    xs = (np.arange(W) * ds)[None, None, :]

    good = np.isfinite(s).all(axis=(1, 2))
    med = np.median(np.where(np.isfinite(s), s, 0.0), axis=(1, 2), keepdims=True)
    w = np.clip(s - med, 0.0, None)
    tot = w.sum(axis=(1, 2))
    ok = good & (tot > 0)
    tot_safe = np.where(ok, tot, 1.0)

    cx = (w * xs).sum(axis=(1, 2)) / tot_safe
    cy = (w * ys).sum(axis=(1, 2)) / tot_safe
    rx = np.sqrt((w * (xs - cx[:, None, None]) ** 2).sum(axis=(1, 2)) / tot_safe)
    ry = np.sqrt((w * (ys - cy[:, None, None]) ** 2).sum(axis=(1, 2)) / tot_safe)

    nanfill = np.where(ok, 0.0, np.nan)
    return {"sum": tot * ds ** 2 + nanfill,
            "centroid_x": cx + nanfill, "centroid_y": cy + nanfill,
            "rms_x": rx + nanfill, "rms_y": ry + nanfill}


def mot_diagnostics(ad, key: str = "img_mot",
                    downsample: int = 2) -> dict[str, np.ndarray]:
    """:func:`image_moments` of a per-shot image container on an atomdata.

    *key* names a DataVault container (``img_mot``, ``img_2dmot``,
    ``img_mot_beams_xy``). Shots whose frame was missed (the stream's
    ``<key>_meta`` record, or the older ``<key>_seq == -1``; see
    ``waxa.data.camera_frames``) are NaN, never dropped. Leading axes are
    flattened to one shot axis; arrays come back shaped ``(n_shots,)``.
    """
    from waxa.data.camera_frames import frames_present
    if key not in ad.data.keys:
        raise KeyError(f"{key!r} is not a data container on this run "
                       f"(have: {list(ad.data.keys)})")
    stack = np.asarray(getattr(ad.data, key), float)
    stack = stack.reshape(-1, *stack.shape[-2:])
    missed = ~np.asarray(frames_present(ad, key)).ravel()
    stack[missed] = np.nan
    return image_moments(stack, downsample=downsample)
