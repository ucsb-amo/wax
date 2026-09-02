"""Automatic ROI suggestion.

Builds a per-pixel "where were the atoms ever" score map across every shot in a
run and returns the bounding box of the significant region. Intended to
pre-populate the manual ROI selector so a human confirms a box rather than
drawing one from scratch, and to give unattended callers a box plus a
confidence they can gate on.

The score is a SUM over shots of the above-noise excess::

    d      = atoms - light                     # signed; the dark frame cancels
    sigma  = noise floor of d, per shot
    score += max(|d| - k*sigma, 0)

A sum rather than a mean because shots with no atoms contribute ~0 and so
dilute nothing, and a cloud that moves between shots lights up the union of its
positions.

Detection deliberately uses the raw ``atoms - light`` difference rather than the
OD. OD divides by ``light``, which is ~0 outside the probe beam, so ``-log``
explodes at unilluminated frame edges and the bounding box lands there. The raw
difference is self-weighting by illumination: no light, no signal, no false
positive. Using ``abs()`` keeps this correct for every imaging type --
absorption darkens the atoms frame, fluorescence brightens it, and
dispersive/polmod can go either way.
"""

import numpy as np
from scipy import ndimage as ndi

__all__ = ["AutoRoiResult", "suggest_roi", "score_map_from_images", "split_images"]

# The Andor's last sensor rows carry a readout artifact hundreds of times
# brighter than a typical row; without trimming it dominates the score map and
# the box lands on the frame edge every time.
DEFAULT_BORDER_TRIM = 4

DEFAULT_K = 4.0                 # noise-floor multiplier for the relu
DEFAULT_DESPIKE = 3             # median_filter width, px -- annihilates isolated spikes
DEFAULT_SMOOTH = 5              # uniform_filter width, px
DEFAULT_THRESHOLD_FRAC = 0.3    # mask cut, as a fraction of the smoothed max
DEFAULT_COMPONENT_FRAC = 0.25   # keep components holding this share of the biggest
DEFAULT_MARGIN_FRAC = 0.5       # box padding, as a fraction of the box extent
DEFAULT_MARGIN_MIN = 6          # px, floor on the padding
DEFAULT_MIN_CONFIDENCE = 0.10   # below this the box is not trusted
DEFAULT_MAX_AREA_FRAC = 0.10    # a fragmented box larger than this is noise
DEFAULT_COMPACT_COMPONENTS = 2  # this few blobs is a real cloud, however large
DEFAULT_SIGMA_ROW_STRIDE = 4    # rows sampled for the noise estimate
DEFAULT_BYTE_BUDGET = 200.e6    # decimate above this many bytes of touched image


class AutoRoiResult():
    """Outcome of an ROI auto-detection.

    Attributes:
        roix (list): [x0, x1] pixel bounds, always populated (best effort).
        roiy (list): [y0, y1] pixel bounds.
        valid (bool): whether the box cleared the confidence and component
            gates. Callers running unattended should refuse to crop when False.
        confidence (float): fraction of the total score map mass falling inside
            the box. Compact, real clouds land well above 0.15; noise lands
            near 0.
        n_components (int): connected components in the threshold mask. A large
            count means the mask is fragmented, i.e. noise.
        n_kept (int): components significant enough to be included in the box.
            One is a single cloud; two is typically a cloud that moved. Many
            means the score map is scattered, i.e. noise.
        area_frac (float): box area as a fraction of the frame. Real clouds are
            compact; a scattered noise field unions into a large box.
        n_shots (int): shots that went into the score map.
        peak (float): maximum of the smoothed score map.
        reason (str): why valid is False, or "ok".
        score_map (np.ndarray): the (H, W) smoothed score map, placed back on
            the full frame so it can be displayed against the run's images.
    """

    def __init__(self, roix, roiy, valid, confidence, n_components,
                 n_shots, peak, reason, score_map, n_kept=0, area_frac=0.):
        self.roix = roix
        self.roiy = roiy
        self.valid = valid
        self.confidence = confidence
        self.n_components = n_components
        self.n_kept = n_kept
        self.area_frac = area_frac
        self.n_shots = n_shots
        self.peak = peak
        self.reason = reason
        self.score_map = score_map

    def __repr__(self):
        state = "valid" if self.valid else "INVALID (" + self.reason + ")"
        return (f"AutoRoiResult({state}, roix={self.roix}, roiy={self.roiy}, "
                f"confidence={self.confidence:.3f}, n_kept={self.n_kept}, "
                f"area_frac={self.area_frac:.4f}, n_components={self.n_components})")


def split_images(images, n_pwa_per_shot=1):
    """Split a flat image stack into atoms and light frames.

    The stack is laid out (N_shots, N_pwa_per_shot + 2, H, W) flattened over the
    first two axes -- the same layout waxa.base.dealer.deal_data_ndarray
    assumes. The reshape is a view, so this copies nothing in the common
    N_pwa_per_shot == 1 case.

    Args:
        images (np.ndarray): (N_img, H, W) stack as stored in the h5 file.
        n_pwa_per_shot (int): probe-with-atoms frames per shot.

    Returns:
        tuple: (atoms, light), both (N, H, W) and index-aligned.
    """
    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError(f"expected (N_img, H, W), got shape {images.shape}")

    nps = int(n_pwa_per_shot)
    per_shot = nps + 2
    n_img, height, width = images.shape
    if n_img % per_shot:
        raise ValueError(f"{n_img} images is not a multiple of {per_shot} "
                         f"frames per shot (N_pwa_per_shot={nps})")

    grouped = images.reshape(-1, per_shot, height, width)
    atoms = grouped[:, :nps]
    light = grouped[:, nps]
    if nps == 1:
        return atoms[:, 0], light
    # More than one probe-with-atoms frame per shot: flatten them onto the shot
    # axis and repeat the shared light frame to match.
    n_shots = grouped.shape[0]
    return (atoms.reshape(n_shots * nps, height, width),
            np.repeat(light, nps, axis=0))


def _bin_factor(n_shots, height, width, itemsize, byte_budget=DEFAULT_BYTE_BUDGET):
    """Smallest stride in {1,2,4,8} keeping the touched image volume in budget.

    Only the atoms and light frames are touched, hence the factor of 2. The
    Andor (512x512) stays at 1 for any realistic run; larger Basler frames
    decimate.
    """
    for b in (1, 2, 4, 8):
        if 2 * n_shots * (height // b) * (width // b) * itemsize <= byte_budget:
            return b
    return 8


def score_map_from_images(atoms, light, k=DEFAULT_K,
                          border_trim=DEFAULT_BORDER_TRIM,
                          sigma_row_stride=DEFAULT_SIGMA_ROW_STRIDE,
                          byte_budget=DEFAULT_BYTE_BUDGET):
    """Accumulate the above-noise excess over shots.

    Args:
        atoms (np.ndarray): (N, H, W) atoms frames.
        light (np.ndarray): (N, H, W) light (probe reference) frames.
        k (float): noise-floor multiplier.
        border_trim (int): pixels to drop from each edge before scoring.
        sigma_row_stride (int): row subsampling for the noise estimate.
        byte_budget (float): decimate when the touched volume exceeds this.

    Returns:
        tuple: (score, offset, step) -- the (h, w) accumulated score on the
        trimmed/decimated grid, the (y, x) origin of that grid in full-frame
        pixels, and the decimation stride.

    The loop over shots is deliberate. Vectorising over the whole stack
    allocates an (N, H, W) temporary; a reused (H, W) scratch buffer is both far
    smaller and measurably faster, since it stays in cache.
    """
    atoms = np.asarray(atoms)
    light = np.asarray(light)
    if atoms.shape != light.shape:
        raise ValueError(f"atoms shape {atoms.shape} != light shape {light.shape}")
    if atoms.ndim != 3:
        raise ValueError(f"expected (N, H, W) frames, got shape {atoms.shape}")

    t = int(max(border_trim, 0))
    if t:
        atoms = atoms[:, t:-t, t:-t]
        light = light[:, t:-t, t:-t]

    n_shots, height, width = atoms.shape
    step = _bin_factor(n_shots, height, width, atoms.dtype.itemsize, byte_budget)
    if step > 1:
        atoms = atoms[:, ::step, ::step]
        light = light[:, ::step, ::step]

    shape = atoms.shape[1:]
    score = np.zeros(shape, dtype=np.float32)
    scratch = np.empty(shape, dtype=np.int32)

    for i in range(n_shots):
        # Signed difference. The dark frame cancels exactly:
        # (atoms - dark) - (light - dark) == atoms - light.
        np.subtract(atoms[i], light[i], out=scratch, dtype=np.int32)

        # Noise floor from adjacent-pixel differences, which pass pixel-to-pixel
        # noise but suppress anything smooth -- the cloud, illumination
        # gradients, low-order fringes. Estimated on the SIGNED difference;
        # estimating on the rectified |d| would be a biased half-normal.
        rows = scratch[::sigma_row_stride]
        sigma = float(np.diff(rows, axis=-1).std()) / np.sqrt(2.)

        np.abs(scratch, out=scratch)
        np.subtract(scratch, int(k * sigma), out=scratch)
        np.maximum(scratch, 0, out=scratch)
        score += scratch

    return score, (t, t), step


def suggest_roi(atoms=None, light=None, images=None, n_pwa_per_shot=1,
                k=DEFAULT_K, border_trim=DEFAULT_BORDER_TRIM,
                despike=DEFAULT_DESPIKE, smooth=DEFAULT_SMOOTH,
                threshold_frac=DEFAULT_THRESHOLD_FRAC,
                component_frac=DEFAULT_COMPONENT_FRAC,
                margin_frac=DEFAULT_MARGIN_FRAC, margin_min=DEFAULT_MARGIN_MIN,
                min_confidence=DEFAULT_MIN_CONFIDENCE,
                max_area_frac=DEFAULT_MAX_AREA_FRAC,
                compact_components=DEFAULT_COMPACT_COMPONENTS,
                byte_budget=DEFAULT_BYTE_BUDGET):
    """Suggest an ROI bounding box for a run.

    Supply either atoms/light directly, or the flat images stack as it is stored
    in the h5 file (shape (N_shots * (N_pwa_per_shot + 2), H, W)).

    Args:
        atoms (np.ndarray): (N, H, W) atoms frames.
        light (np.ndarray): (N, H, W) light frames.
        images (np.ndarray): flat image stack, an alternative to atoms/light.
        n_pwa_per_shot (int): probe-with-atoms frames per shot, for splitting
            images.
        k (float): noise-floor multiplier for the relu.
        border_trim (int): pixels dropped from each edge before scoring.
        despike (int): median_filter width removing isolated spikes. A stuck
            pixel fires on every shot and so accumulates like real signal; a
            median filter annihilates it while sparing a real cloud.
        smooth (int): uniform_filter width applied to the score map.
        threshold_frac (float): mask cut as a fraction of the smoothed max.
        component_frac (float): keep components holding at least this share of
            the biggest component's score, so a cloud that moved between shots
            contributes every position it visited.
        margin_frac (float): padding as a fraction of the detected extent.
        margin_min (int): minimum padding in pixels.
        min_confidence (float): confidence below which valid is False.
        max_area_frac (float): box area fraction above which a multi-component
            detection is treated as noise.
        compact_components (int): a detection with at most this many components
            is a real cloud whatever its size, so the area gate does not apply.
        byte_budget (float): decimate when the touched volume exceeds this.

    Returns:
        AutoRoiResult: the box plus the QC scalars behind it. Always returns a
        result -- check .valid rather than catching exceptions.
    """
    if images is not None:
        atoms, light = split_images(images, n_pwa_per_shot)
    if atoms is None or light is None:
        raise ValueError("supply either atoms and light, or images")

    atoms = np.asarray(atoms)
    light = np.asarray(light)

    if atoms.ndim != 3 or atoms.shape[0] == 0:
        blank = np.zeros((1, 1), dtype=np.float32)
        return AutoRoiResult(roix=[-1, -1], roiy=[-1, -1], valid=False,
                             confidence=0., n_components=0, n_shots=0, peak=0.,
                             reason="no images", score_map=blank)

    full_shape = atoms.shape[-2:]
    n_shots = int(atoms.shape[0])

    def _failed(reason, score_map=None):
        if score_map is None:
            score_map = np.zeros(full_shape, dtype=np.float32)
        return AutoRoiResult(roix=[0, full_shape[1]], roiy=[0, full_shape[0]],
                             valid=False, confidence=0., n_components=0,
                             n_shots=n_shots, peak=0., reason=reason,
                             score_map=score_map, n_kept=0, area_frac=1.)

    if min(full_shape) <= 2 * border_trim:
        return _failed("frame too small to trim")

    score, (off_y, off_x), step = score_map_from_images(
        atoms, light, k=k, border_trim=border_trim, byte_budget=byte_budget)

    # Despike before smoothing. A hot pixel or a hot readout row exceeds the
    # noise floor on every single shot, so it accumulates exactly like a real
    # cloud; smoothing alone only spreads it. A median filter removes it.
    if despike and despike > 1:
        score = ndi.median_filter(score, size=int(despike))
    if smooth and smooth > 1:
        score = ndi.uniform_filter(score, size=int(smooth))

    # Place the (trimmed, possibly decimated) score map back on the full frame
    # so callers can display it against the run's own images.
    score_full = np.zeros(full_shape, dtype=np.float32)
    score_full[off_y:off_y + score.shape[0] * step:step,
               off_x:off_x + score.shape[1] * step:step] = score

    peak = float(score.max())
    total = float(score.sum())
    if peak <= 0. or total <= 0.:
        return _failed("no signal above the noise floor", score_full)

    mask = score > threshold_frac * peak
    labels, n_components = ndi.label(mask)
    if n_components == 0:
        return _failed("no signal above the noise floor", score_full)

    # Union every component that carries a real share of the score, not just
    # the largest -- a cloud that moved between shots leaves one blob per
    # position, and all of them belong inside the box.
    weights = ndi.sum(score, labels, index=np.arange(1, n_components + 1))
    keep = np.where(weights >= component_frac * weights.max())[0] + 1
    ys, xs = np.where(np.isin(labels, keep))
    n_kept = int(keep.size)

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # Confidence: how much of the score map's mass the box actually accounts
    # for. A compact real cloud captures a large share; scattered noise does not.
    confidence = float(score[y0:y1 + 1, x0:x1 + 1].sum() / total)
    area_frac = float(((y1 - y0 + 1) * (x1 - x0 + 1))
                      / float(score.shape[0] * score.shape[1]))

    # Back to full-frame pixels, then pad. The margin gives the downstream 1-D
    # Gaussian fits some background to fit their free y_offset against.
    y0, y1 = y0 * step + off_y, y1 * step + off_y
    x0, x1 = x0 * step + off_x, x1 * step + off_x

    my = max(margin_min, int(margin_frac * (y1 - y0 + 1)))
    mx = max(margin_min, int(margin_frac * (x1 - x0 + 1)))
    roiy = [max(0, y0 - my), min(full_shape[0], y1 + my + 1)]
    roix = [max(0, x0 - mx), min(full_shape[1], x1 + mx + 1)]

    # A detection made of one or two blobs is a cloud (possibly one that moved),
    # however large it is. More blobs than that, spread over a large box, is a
    # scattered noise field -- which is what an empty or unilluminated run gives.
    if confidence < min_confidence:
        reason = f"low confidence ({confidence:.3f} < {min_confidence})"
    elif n_kept > compact_components and area_frac > max_area_frac:
        reason = (f"score map is scattered ({n_kept} blobs over "
                  f"{100 * area_frac:.0f}% of the frame)")
    else:
        reason = "ok"

    return AutoRoiResult(roix=roix, roiy=roiy, valid=(reason == "ok"),
                         confidence=confidence, n_components=int(n_components),
                         n_shots=n_shots, peak=peak, reason=reason,
                         score_map=score_full, n_kept=n_kept,
                         area_frac=area_frac)
