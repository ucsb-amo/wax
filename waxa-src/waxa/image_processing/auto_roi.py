"""Automatic ROI suggestion.

Builds a per-pixel "where were the atoms ever" score map across every shot in a
run and returns the bounding box of the significant region. Intended to
pre-populate the manual ROI selector so a human confirms a box rather than
drawing one from scratch, and to give unattended callers a box plus a
confidence they can gate on.

Detection is a scale-space matched filter, run per shot::

    d       = atoms - light              # signed; the dark frame cancels
    z_L     = |smooth_L(d)| / sigma_L    # significance at pyramid level L
    z       = max over L of z_L
    score  += max(z - k, 0) / norm       # norm equalises the shots

Three things in that are load-bearing.

**Smoothing before thresholding.** A cloud is spread over many pixels, so
testing each pixel on its own throws its wings away: on a typical run the raw
per-pixel excess clears a 4-sigma floor by less than a factor of two even for a
dense cloud, while the same cloud smoothed over ~9 px sits at 20-40 sigma. A
diffuse cloud has a lower per-pixel amplitude still and vanishes entirely. The
pyramid tests every scale from a couple of pixels to a couple of hundred, so a
tweezer spot and a 20 ms time-of-flight cloud are both found, each at the scale
that suits it.

**A noise floor measured after smoothing.** sigma_L is a MAD of the smoothed
map itself, not a photon-noise estimate propagated through the kernel. Anything
smooth and shot-to-shot varying -- probe fringes above all -- lands in sigma_L
and so raises its own detection floor, which is what stops the pre-smoothing
from turning fringes into detections.

**Per-shot normalisation.** Each shot's excess is divided by its own peak, so a
shot contributes at most 1 wherever its cloud was. Without it a run holding
both dense and diffuse clouds -- any time-of-flight scan -- is scored almost
entirely by its dense shots, and the mask cut (a fraction of the map's peak)
then falls above the diffuse cloud and crops it out of the box. The divisor has
a floor, so a shot with no atoms still contributes ~0 rather than being
amplified to full weight.

Detection deliberately uses the raw ``atoms - light`` difference rather than the
OD. OD divides by ``light``, which is ~0 outside the probe beam, so ``-log``
explodes at unilluminated frame edges and the bounding box lands there. The raw
difference is self-weighting by illumination: no light, no signal, no false
positive. Using ``abs()`` keeps this correct for every imaging type --
absorption darkens the atoms frame, fluorescence brightens it, and
dispersive/polmod can go either way.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi

__all__ = ["AutoRoiResult", "suggest_roi", "score_map_from_images", "split_images"]

# The Andor's last sensor rows carry a readout artifact hundreds of times
# brighter than a typical row; without trimming it dominates the score map and
# the box lands on the frame edge every time.
DEFAULT_BORDER_TRIM = 4

DEFAULT_K = 4.0                 # significance floor for the relu, in sigma
DEFAULT_LEVELS = 4              # pyramid levels; level L smooths over ~3*2^L px
DEFAULT_NORM_FLOOR = 4.0        # floor on the per-shot normaliser, in sigma
DEFAULT_DESPIKE = 3             # median_filter width, px -- annihilates isolated spikes
DEFAULT_SMOOTH = 5              # uniform_filter width, px
DEFAULT_THRESHOLD_FRAC = 0.3    # mask cut, as a fraction of the smoothed max
DEFAULT_COMPONENT_FRAC = 0.3    # keep components peaking this high, vs the map max
DEFAULT_MARGIN_FRAC = 0.5       # box padding, as a fraction of the box extent
DEFAULT_MARGIN_MIN = 6          # px, floor on the padding
DEFAULT_MIN_CONFIDENCE = 0.02   # below this the box captured next to nothing
DEFAULT_MIN_DETECTED_FRAC = 0.05  # shots that must hold a cloud for a real run
DEFAULT_MAX_AREA_FRAC = 0.10    # a fragmented box larger than this is noise
DEFAULT_COMPACT_COMPONENTS = 2  # this few blobs is a real cloud, however large
MIN_LEVEL_SIDE = 16             # px, smallest pyramid level worth scoring
MIN_SIGMA_ROWS = 16             # rows the noise estimate always gets to see
DEFAULT_SIGMA_ROW_STRIDE = 4    # rows sampled for the noise estimate
DEFAULT_SIGMA_SAMPLE_CAP = 1 << 16  # cap on pixels used for the noise estimate
DEFAULT_DEFECT_SHOTS = 8        # shots sampled when hunting stuck pixels
DEFAULT_DEFECT_Z = 50.          # sigma above which a lone pixel is a defect
DEFAULT_DEFECT_ISOLATION = 0.2  # neighbourhood/peak ratio below which it is alone
DEFAULT_PIXEL_BUDGET = 30.e6    # decimate above this many touched pixels
DEFAULT_MAX_WORKERS = 4         # threads the shot loop is split over


class AutoRoiResult():
    """Outcome of an ROI auto-detection.

    Attributes:
        roix (list): [x0, x1] pixel bounds, always populated (best effort).
        roiy (list): [y0, y1] pixel bounds.
        valid (bool): whether the box cleared the confidence and shape gates.
            Callers running unattended should refuse to crop when False.
        confidence (float): fraction of the total score map mass falling inside
            the box. Informative, but only weakly gated on: it scales with the
            box's share of the frame, so a genuinely tiny ROI -- a tweezer spot
            on a Basler frame -- scores 0.06 while being exactly right.
        n_components (int): connected components in the threshold mask. A large
            count means the mask is fragmented, i.e. noise.
        n_kept (int): components significant enough to be included in the box.
            One is a single cloud, or a cloud whose positions overlap; a few is
            a cloud that jumped. Many means the score map is scattered.
        fill (float): masked area as a fraction of the box area. Reported, not
            gated on: a single cloud fills most of its own box (0.7 and up) and
            a noise field does not (0.03), but so does a legitimate scan whose
            cloud visited two well separated positions, and cropping those away
            is the failure this detector exists to avoid.
        peak_per_shot (float): the score map's peak divided by the shot count.
            Per-shot normalisation caps a shot's contribution at 1, so this is
            the fraction of shots that lit up the peak pixel. Real runs reach
            0.2 upwards even when the cloud visits many positions; noise stays
            near 0.01.
        area_frac (float): box area as a fraction of the frame.
        n_shots (int): shots that went into the score map.
        peak (float): maximum of the smoothed score map.
        reason (str): why valid is False, or "ok".
        score_map (np.ndarray): the (H, W) smoothed score map, placed back on
            the full frame so it can be displayed against the run's images.
    """

    def __init__(self, roix, roiy, valid, confidence, n_components,
                 n_shots, peak, reason, score_map, n_kept=0, area_frac=0.,
                 fill=0., peak_per_shot=0.):
        self.roix = roix
        self.roiy = roiy
        self.valid = valid
        self.confidence = confidence
        self.n_components = n_components
        self.n_kept = n_kept
        self.area_frac = area_frac
        self.fill = fill
        self.peak_per_shot = peak_per_shot
        self.n_shots = n_shots
        self.peak = peak
        self.reason = reason
        self.score_map = score_map

    def __repr__(self):
        state = "valid" if self.valid else "INVALID (" + self.reason + ")"
        return (f"AutoRoiResult({state}, roix={self.roix}, roiy={self.roiy}, "
                f"confidence={self.confidence:.3f}, fill={self.fill:.3f}, "
                f"peak_per_shot={self.peak_per_shot:.3f}, n_kept={self.n_kept}, "
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


def _bin_factor(n_shots, height, width, pixel_budget=DEFAULT_PIXEL_BUDGET):
    """Smallest stride in {1,2,4,8} keeping the touched pixel count in budget.

    Only the atoms and light frames are touched, hence the factor of 2. A count
    of pixels rather than of bytes: the cost here is a fixed number of passes
    per pixel and the loop is memory-bandwidth bound, so an 8-bit Basler frame
    and a 16-bit Andor frame of the same size cost the same to score even
    though one is half the size on disk. The Andor (512x512) stays at 1 for a
    run of a few hundred shots; the much larger Basler frames decimate.
    """
    for b in (1, 2, 4, 8):
        if 2 * n_shots * (height // b) * (width // b) <= pixel_budget:
            return b
    return 8


def _block_mean(a, factor):
    """Average a over non-overlapping factor x factor blocks, truncating.

    A mean rather than ``a[::factor, ::factor]``: this runs on the raw
    difference, which has had no low-pass applied yet, and plain subsampling
    would alias away precisely the smooth low-amplitude structure a diffuse
    cloud is made of.
    """
    if factor <= 1:
        return a
    h = (a.shape[0] // factor) * factor
    w = (a.shape[1] // factor) * factor
    return a[:h, :w].reshape(h // factor, factor,
                             w // factor, factor).mean(axis=(1, 3))


def _halve(a):
    """Decimate a smoothed map by 2, as a view.

    Subsampling is safe here, unlike in _block_mean, because `a` has just been
    through the 3x3 box filter and is already low-passed -- this is the ordinary
    smooth-then-drop-every-other-pixel image pyramid. Being a view rather than a
    reduction, it is free, which matters: on a Basler frame the block mean it
    replaces cost more than the filter itself.
    """
    return a[:(a.shape[0] // 2) * 2:2, :(a.shape[1] // 2) * 2:2]


def _box3(a, buf_rows, out):
    """3x3 box mean of `a` into `out`, treating outside the frame as zero.

    Identical to ``ndi.uniform_filter(a, size=3, mode='constant')`` -- the
    partial windows at the edges included, which the weights from
    _window_weights then correct for -- but around three times faster, because
    it is four passes of numpy adds over preallocated buffers instead of
    scipy's generic separable filter. The shot loop is memory-bandwidth bound
    on a large frame, so passes are the currency.

    Args:
        a (np.ndarray): (h, w) float32 input.
        buf_rows (np.ndarray): (h, w) float32 scratch, overwritten.
        out (np.ndarray): (h, w) float32 destination.

    Returns:
        np.ndarray: `out`.
    """
    if min(a.shape) < 3:
        return ndi.uniform_filter(a, size=3, mode='constant', output=out)

    # Vertical pass: sum of the three rows, with a zero row outside.
    np.add(a[:-2], a[1:-1], out=buf_rows[1:-1])
    np.add(buf_rows[1:-1], a[2:], out=buf_rows[1:-1])
    np.add(a[0], a[1], out=buf_rows[0])
    np.add(a[-2], a[-1], out=buf_rows[-1])

    # Horizontal pass over the row sums, then the 1/9 normalisation.
    np.add(buf_rows[:, :-2], buf_rows[:, 1:-1], out=out[:, 1:-1])
    np.add(out[:, 1:-1], buf_rows[:, 2:], out=out[:, 1:-1])
    np.add(buf_rows[:, 0], buf_rows[:, 1], out=out[:, 0])
    np.add(buf_rows[:, -2], buf_rows[:, -1], out=out[:, -1])
    out *= (1. / 9.)
    return out


def _pyramid_shapes(shape, levels, min_side=MIN_LEVEL_SIDE):
    """Grid shape at each pyramid level, stopping once a level gets too small.

    A level has to hold enough pixels for its own noise floor to be measurable.
    Below a couple of hundred the MAD is uncertain by enough that a routine
    3.5-sigma fluctuation scores above 4, and because the coarsest level sees
    the whole frame as a handful of pixels, that lands as one large confident
    blob of nothing. The floor gives up nothing real: a level that small is
    already smoothing over a good fraction of the frame.
    """
    shapes = []
    h, w = shape
    for _ in range(levels):
        if min(h, w) < min_side:
            break
        shapes.append((h, w))
        h, w = h // 2, w // 2
    return shapes or [shape]


def _window_weights(shapes):
    """Per-level fraction of the 3x3 window that lies inside the frame.

    ``uniform_filter(..., mode='constant')`` divides by the full window size
    whatever the padding, so a pixel one in from the edge is averaged over 6
    real values and 3 zeros. Dividing the significance by sqrt(weight) restores
    the correct noise scaling there -- exactly, not approximately -- which is
    what keeps a cloud sitting against the frame edge detectable instead of
    lost to a blind margin.
    """
    weights = []
    for shape in shapes:
        ones = np.ones(shape, dtype=np.float32)
        weights.append(np.sqrt(ndi.uniform_filter(ones, size=3, mode='constant')))
    return weights


def _noise_row_stride(height, width, sigma_row_stride, sigma_sample_cap):
    """Row stride keeping the noise estimate under `sigma_sample_cap` pixels.

    Only rows are thinned, never columns: the estimator wants an unbiased
    sample of the map, and at this cap the loss on a MAD is well under 1%. On
    the coarse pyramid levels the cap never binds and the floor below does,
    which is the point -- an underestimated sigma there is what turns noise
    into a detection.
    """
    stride = max(int(sigma_row_stride), 1)
    if sigma_sample_cap and sigma_sample_cap > 0 and width > 0:
        needed = int(np.ceil(height * width / float(sigma_sample_cap)))
        stride = max(stride, needed)
    return max(1, min(stride, height // MIN_SIGMA_ROWS))


def _mad_sigma(a, row_stride):
    """Robust standard deviation of `a`, from a strided row sample.

    A MAD, not a standard deviation: the map contains the cloud, and a cloud
    that occupies a percent of the frame drags an rms estimate up enough to
    matter. The MAD does not notice it.
    """
    sample = a[::row_stride]
    centre = np.median(sample)
    return 1.4826 * float(np.median(np.abs(sample - centre)))


def _defect_indices(atoms, light, n_sample=DEFAULT_DEFECT_SHOTS,
                    z_defect=DEFAULT_DEFECT_Z,
                    isolation=DEFAULT_DEFECT_ISOLATION,
                    row_stride=DEFAULT_SIGMA_ROW_STRIDE):
    """Flat indices of stuck pixels and readout lines, from a sample of shots.

    A defect has to be dealt with before the score is accumulated, not after.
    Per-shot normalisation caps every shot's contribution at 1, which is what
    lets a diffuse cloud compete with a dense one -- but it also lets a
    saturated pixel, which fires on every shot, accumulate exactly as fast as a
    real cloud. Smoothing then spreads it into a blob a median filter can no
    longer remove.

    Two conditions, both required. The pixel must be extreme in *every* sampled
    shot: a cloud that moves is not, and a cloud that does not move fails the
    second test anyway. And it must be alone -- its 3x3 neighbourhood far below
    its own value. Even a two-pixel tweezer cloud has bright neighbours and so
    survives comfortably; a stuck pixel or a one-pixel-wide readout line does
    not.

    Returns:
        np.ndarray or None: flat indices into a single frame, or None if the
        frame is clean.
    """
    n_shots = atoms.shape[0]
    if n_shots == 0:
        return None
    take = np.unique(np.linspace(0, n_shots - 1,
                                 min(n_shots, int(n_sample))).astype(int))

    worst = None
    for i in take:
        d = np.abs(np.subtract(atoms[i], light[i], dtype=np.float32))
        sigma = _mad_sigma(d, row_stride)
        if sigma <= 0:
            return None
        z = d / sigma
        worst = z if worst is None else np.minimum(worst, z, out=worst)

    candidate = worst > z_defect
    if not candidate.any():
        return None
    neighbourhood = ndi.median_filter(worst, size=3)
    bad = candidate & (neighbourhood < isolation * worst)
    if not bad.any():
        return None
    return np.flatnonzero(bad.ravel())


def _accumulate_shots(atoms, light, indices, k, step, shapes, weights,
                      row_strides, norm_floor, defects):
    """Sum the normalised multi-scale excess over `indices` into private buffers.

    Each caller (thread) owns its accumulators and scratch, so the shot loop
    parallelises with no locking -- every numpy call in here releases the GIL.

    Returns:
        tuple: (scores, n_detected) -- one (h, w) score array per pyramid level,
        and the number of these shots that held a cloud at all.
    """
    scores = [np.zeros(s, dtype=np.float32) for s in shapes]
    zs = [np.empty(s, dtype=np.float32) for s in shapes]
    rows = [np.empty(s, dtype=np.float32) for s in shapes]
    smooth = [np.empty(s, dtype=np.float32) for s in shapes]
    n_levels = len(shapes)
    n_detected = 0

    for i in indices:
        d = np.subtract(atoms[i], light[i], dtype=np.float32)
        if defects is not None:
            d.ravel()[defects] = 0.
        current = _block_mean(d, step)

        peak = 0.
        for level in range(n_levels):
            # Zero outside the frame, with the sqrt(weight) correction below, is
            # exact at the edge -- where reflecting or replicating would average
            # duplicated pixels and quietly inflate the significance, which is
            # how a detector ends up putting its box on a frame border.
            smoothed = _box3(current, rows[level], smooth[level])
            sigma = _mad_sigma(smoothed, row_strides[level])
            z = zs[level]
            if sigma <= 0:
                z.fill(0.)
            else:
                np.abs(smoothed, out=z)
                z /= sigma
                z /= weights[level]
                peak = max(peak, float(z.max()))
            if level + 1 < n_levels:
                current = _halve(smoothed)

        # Every shot is worth at most 1 at its own brightest point. The floor
        # keeps a shot with no atoms near 0 instead of scaling its noise up,
        # and a shot that never clears it is a shot with nothing in it.
        if peak - k >= norm_floor:
            n_detected += 1
        norm = max(peak - k, norm_floor)
        for level in range(n_levels):
            excess = np.subtract(zs[level], k, out=zs[level])
            np.maximum(excess, 0., out=excess)
            excess *= 1. / norm
            scores[level] += excess

    return scores, n_detected


def _combine_levels(maps, shape):
    """Collapse the pyramid onto the level-0 grid, taking the best scale."""
    combined = np.zeros(shape, dtype=np.float32)
    for level, m in enumerate(maps):
        if level:
            m = np.repeat(np.repeat(m, 1 << level, axis=0), 1 << level, axis=1)
        h = min(shape[0], m.shape[0])
        w = min(shape[1], m.shape[1])
        np.maximum(combined[:h, :w], m[:h, :w], out=combined[:h, :w])
    return combined


def _n_workers(max_workers, n_shots):
    """Threads to split the shot loop over."""
    if max_workers is None:
        max_workers = min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1)
    return max(1, min(int(max_workers), int(n_shots)))


def score_map_from_images(atoms, light, k=DEFAULT_K,
                          border_trim=DEFAULT_BORDER_TRIM,
                          levels=DEFAULT_LEVELS,
                          norm_floor=DEFAULT_NORM_FLOOR,
                          sigma_row_stride=DEFAULT_SIGMA_ROW_STRIDE,
                          sigma_sample_cap=DEFAULT_SIGMA_SAMPLE_CAP,
                          pixel_budget=DEFAULT_PIXEL_BUDGET,
                          max_workers=None):
    """Accumulate the normalised multi-scale excess over shots.

    Args:
        atoms (np.ndarray): (N, H, W) atoms frames.
        light (np.ndarray): (N, H, W) light (probe reference) frames.
        k (float): significance floor, in sigma.
        border_trim (int): pixels to drop from each edge before scoring.
        levels (int): pyramid levels. Level L detects structure ~3*2^L px
            across, so the default spans a couple of pixels to a couple of
            hundred.
        norm_floor (float): floor on the per-shot normaliser, in sigma, below
            which a shot is treated as having no atoms.
        sigma_row_stride (int): row subsampling for the noise estimate.
        sigma_sample_cap (int): cap on the pixels sampled per level for that
            estimate; large frames thin their rows further to respect it.
        pixel_budget (float): decimate when the touched pixel count exceeds
            this.
        max_workers (int or None): threads to split the shot loop over. None
            picks a small default; 1 runs inline.

    Returns:
        tuple: (score, n_detected, offset, step) -- the (h, w) accumulated
        score on the trimmed/decimated grid, the number of shots that held a
        cloud at all, the (y, x) origin of that grid in full-frame pixels, and
        the decimation factor.

    The per-shot loop is deliberate. Vectorising over the whole stack allocates
    an (N, H, W) temporary at every pyramid level; reused (H, W) scratch buffers
    are both far smaller and measurably faster, since they stay in cache. The
    shots are independent and the score is a plain sum, so the loop is split
    across threads over contiguous chunks of shots and the partial sums added at
    the end -- numpy drops the GIL for every op inside, so this is real
    parallelism.
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
    step = _bin_factor(n_shots, height, width, pixel_budget)

    defects = _defect_indices(atoms, light, row_stride=_noise_row_stride(
        height, width, sigma_row_stride, sigma_sample_cap))

    base = ((height // step), (width // step)) if step > 1 else (height, width)
    shapes = _pyramid_shapes(base, max(int(levels), 1))
    weights = _window_weights(shapes)
    row_strides = [_noise_row_stride(h, w, sigma_row_stride, sigma_sample_cap)
                   for h, w in shapes]

    workers = _n_workers(max_workers, n_shots)
    if workers <= 1:
        scores, n_detected = _accumulate_shots(
            atoms, light, range(n_shots), k, step, shapes, weights,
            row_strides, norm_floor, defects)
    else:
        chunks = [c for c in np.array_split(np.arange(n_shots), workers) if c.size]
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            partials = list(pool.map(
                lambda idx: _accumulate_shots(
                    atoms, light, idx, k, step, shapes, weights,
                    row_strides, norm_floor, defects),
                chunks))
        scores, n_detected = partials[0]
        for part_scores, part_detected in partials[1:]:
            n_detected += part_detected
            for level in range(len(shapes)):
                scores[level] += part_scores[level]

    return _combine_levels(scores, shapes[0]), n_detected, (t, t), step


def suggest_roi(atoms=None, light=None, images=None, n_pwa_per_shot=1,
                k=DEFAULT_K, border_trim=DEFAULT_BORDER_TRIM,
                levels=DEFAULT_LEVELS, norm_floor=DEFAULT_NORM_FLOOR,
                despike=DEFAULT_DESPIKE, smooth=DEFAULT_SMOOTH,
                threshold_frac=DEFAULT_THRESHOLD_FRAC,
                component_frac=DEFAULT_COMPONENT_FRAC,
                margin_frac=DEFAULT_MARGIN_FRAC, margin_min=DEFAULT_MARGIN_MIN,
                min_confidence=DEFAULT_MIN_CONFIDENCE,
                min_detected_frac=DEFAULT_MIN_DETECTED_FRAC,
                max_area_frac=DEFAULT_MAX_AREA_FRAC,
                compact_components=DEFAULT_COMPACT_COMPONENTS,
                pixel_budget=DEFAULT_PIXEL_BUDGET, max_workers=None):
    """Suggest an ROI bounding box for a run.

    Supply either atoms/light directly, or the flat images stack as it is stored
    in the h5 file (shape (N_shots * (N_pwa_per_shot + 2), H, W)).

    Args:
        atoms (np.ndarray): (N, H, W) atoms frames.
        light (np.ndarray): (N, H, W) light frames.
        images (np.ndarray): flat image stack, an alternative to atoms/light.
        n_pwa_per_shot (int): probe-with-atoms frames per shot, for splitting
            images.
        k (float): significance floor for the relu, in sigma.
        border_trim (int): pixels dropped from each edge before scoring.
        levels (int): pyramid levels; see score_map_from_images.
        norm_floor (float): floor on the per-shot normaliser, in sigma.
        despike (int): median_filter width applied to the score map, a backstop
            behind the per-shot defect rejection.
        smooth (int): uniform_filter width applied to the score map.
        threshold_frac (float): mask cut as a fraction of the smoothed max.
        component_frac (float): keep components whose own peak reaches at least
            this share of the map's peak, so a cloud that jumped between shots
            contributes every position it visited. Defaults to threshold_frac,
            i.e. every blob that entered the mask is kept; raise it to demand
            that secondary blobs be brighter.
        margin_frac (float): padding as a fraction of the largest component's
            extent.
        margin_min (int): minimum padding in pixels.
        min_confidence (float): confidence below which valid is False.
        min_detected_frac (float): fraction of shots that must hold a cloud.
            The primary signal/noise gate. A shot counts when its strongest
            point stands `norm_floor` above the significance floor, which
            photon noise essentially never reaches, so a run with atoms scores
            near 1 and a run without scores exactly 0.
        max_area_frac (float): box area fraction above which a multi-component
            detection is treated as noise.
        compact_components (int): a detection with at most this many components
            is a real cloud whatever its size, so the area gate does not apply.
        pixel_budget (float): decimate when the touched pixel count exceeds
            this.
        max_workers (int or None): threads to split the shot loop over. None
            picks a small default; 1 runs inline.

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

    score, n_detected, (off_y, off_x), step = score_map_from_images(
        atoms, light, k=k, border_trim=border_trim, levels=levels,
        norm_floor=norm_floor, pixel_budget=pixel_budget, max_workers=max_workers)

    # A backstop behind _defect_indices, for artifacts too mild to have been
    # caught there but still sharper than anything optical can be.
    if despike and despike > 1:
        score = ndi.median_filter(score, size=int(despike))
    if smooth and smooth > 1:
        score = ndi.uniform_filter(score, size=int(smooth))

    # Place the (trimmed, possibly decimated) score map back on the full frame
    # so callers can display it against the run's own images. Each score pixel
    # covers a step x step block, so fill the block rather than one pixel of it.
    score_full = np.zeros(full_shape, dtype=np.float32)
    spread = (np.repeat(np.repeat(score, step, axis=0), step, axis=1)
              if step > 1 else score)
    h = min(full_shape[0] - off_y, spread.shape[0])
    w = min(full_shape[1] - off_x, spread.shape[1])
    score_full[off_y:off_y + h, off_x:off_x + w] = spread[:h, :w]

    peak = float(score.max())
    total = float(score.sum())
    peak_per_shot = peak / n_shots
    detected_frac = n_detected / float(n_shots)
    if peak <= 0. or total <= 0.:
        return _failed("no signal above the noise floor", score_full)
    # Count the shots that held a cloud, rather than asking how many shots lit
    # one pixel: an SLM spot scanner or a wide time-of-flight scan puts its
    # cloud somewhere different on almost every shot, and there is nothing
    # wrong with that.
    if detected_frac < min_detected_frac:
        return _failed(f"no atoms detected ({n_detected} of {n_shots} shots "
                       f"held a cloud)", score_full)

    mask = score > threshold_frac * peak
    labels, n_components = ndi.label(mask)
    if n_components == 0:
        return _failed("no signal above the noise floor", score_full)

    # Union every component that stands high enough, not just the largest -- a
    # cloud that moved between shots leaves one blob per position, and all of
    # them belong inside the box.
    #
    # The test is on each component's own peak, not on its integrated score. A
    # component's integrated score is its area times the number of shots that
    # lit it, and a time-of-flight scan varies both: the long-TOF end of a scan
    # is a handful of shots holding a cloud whose amplitude has fallen, so it
    # carries a small fraction of the run's total score while being every bit as
    # real as the dense end. Its peak is the scale-free thing to test.
    index = np.arange(1, n_components + 1)
    weights = ndi.sum(score, labels, index=index)
    peaks = ndi.maximum(score, labels, index=index)
    keep = np.where(peaks >= component_frac * peak)[0] + 1
    kept = np.isin(labels, keep)
    ys, xs = np.where(kept)
    n_kept = int(keep.size)

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # Confidence: how much of the score map's mass the box actually accounts
    # for. Reported for the caller's benefit and gated on only loosely -- a
    # correct box around a tweezer spot on a Basler frame scores 0.06.
    confidence = float(score[y0:y1 + 1, x0:x1 + 1].sum() / total)
    # Fill: how much of that box the detection actually occupies. Diagnostic
    # only -- see the class docstring for why it is not a gate.
    fill = float(kept.sum()) / float((y1 - y0 + 1) * (x1 - x0 + 1))
    area_frac = float(((y1 - y0 + 1) * (x1 - x0 + 1))
                      / float(score.shape[0] * score.shape[1]))

    # Pad from the largest component's extent, not the union's. The margin is
    # there to give the downstream 1-D Gaussian fits some background to fit a
    # free y_offset against, and one cloud's worth of background is what that
    # needs -- scaling it by the span of a time-of-flight track would swamp the
    # box in empty frame.
    biggest = int(keep[int(np.argmax(weights[keep - 1]))])
    bys, bxs = np.where(labels == biggest)
    my = max(margin_min, int(margin_frac * (bys.max() - bys.min() + 1)))
    mx = max(margin_min, int(margin_frac * (bxs.max() - bxs.min() + 1)))

    # Back to full-frame pixels, then pad.
    y0, y1 = y0 * step + off_y, y1 * step + off_y
    x0, x1 = x0 * step + off_x, x1 * step + off_x
    my, mx = my * step, mx * step
    roiy = [max(0, y0 - my), min(full_shape[0], y1 + step + my)]
    roix = [max(0, x0 - mx), min(full_shape[1], x1 + step + mx)]

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
                         area_frac=area_frac, fill=fill,
                         peak_per_shot=peak_per_shot)
