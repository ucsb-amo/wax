"""Absorption-imaging cross section, chosen per shot from the recorded
outer-coil current.

The experiment records the outer-coil current flowing at the first camera
trigger of every shot in the DataVault container ``i_outer_imaging`` (A)
(``kexp.base.image.Image.record_imaging_conditions``, 2026-09-16). The
analysis switches the cross section on that current:

    i_outer_imaging >= I_OUTER_HF_THRESHOLD_A  ->  SIGMA_HF_M2   ('high-field')
    i_outer_imaging <  I_OUTER_HF_THRESHOLD_A  ->  SIGMA_LF_M2   ('low-field')
    no record (old data, NaN-padded vault)     ->  SIGMA_FALLBACK_M2 ('fallback-no-record')

Every shot carries its source tag in ``atomdata.atom_cross_section_source``
so a mixed vault or an old run is visible rather than silent.

Values and provenance
---------------------
SIGMA_HF_M2
    3 lambda^2 / 2 pi on the closed sigma- D2 line
    (4,0,1/2,-1/2,m_i) -> (4,1,3/2,-3/2,m_i) at the high-field imaging point
    B = 520.583 G (kamo ``Potassium39``, portal matrix elements, 2022-survey
    hyperfine constants; 2026-09-12). Replaces the value 2.8316243e-13 m^2 the
    analysis carried before 2026-09-16, which is 3 lambda^2 / 2 pi at the D1
    wavelength (0.9 % high). Kept below as SIGMA_LEGACY_D1_M2 for comparison.
    Treated as field-independent for any current above the threshold: the
    line shifts by ~1.4 MHz/G, i.e. 5e-6 in lambda^2 per gauss.
SIGMA_LF_M2
    Placeholder. The value the analysis carried (commented out) as the
    "0-field" cross section. It equals lambda_D2^2, not 3 lambda^2 / 2 pi
    (2.81e-13), so it is not a physical closed-line cross section; it is kept
    only so that switching to the low-field branch reproduces what the
    analysis code used to assume for low-field runs. The low-field cross
    section must be calibrated against field (open transition, optical
    pumping during the pulse; see the "Cross-section model" in
    k-jam/jpagett/imaging_field_record/PLAN.md). Replace this constant and its
    tag when that calibration lands.
SIGMA_FALLBACK_M2
    Used when a shot has no recorded current. Equal to the high-field value,
    which is what every run was assumed to be before the record existed.
I_OUTER_HF_THRESHOLD_A
    1 A. The high-field imaging points sit at 174-194 A and low-field imaging
    happens with the outer coil off (0 A), so anything in between is a
    misconfiguration worth noticing in the tag rather than a real regime.
    Distinct from kexp's I_LF_HF_THRESHOLD (45 A), which selects a detuning
    fit on the experiment side.
"""

import numpy as np

# -- tunables --
I_OUTER_HF_THRESHOLD_A = 1.0  # A; >= is high field, < is low field

SIGMA_HF_M2 = 2.80668e-13  # kamo closed sigma- D2 line at 520.583 G, 2026-09-12
SIGMA_LF_M2 = 5.878324268151581e-13  # legacy "0-field" placeholder (= lambda_D2^2); uncalibrated
SIGMA_LEGACY_D1_M2 = 2.8316243e-13  # 3 lambda^2/2pi at D1; the pre-2026-09-16 constant, not used
SIGMA_FALLBACK_M2 = SIGMA_HF_M2  # no recorded current: assume high field, as before

TAG_HF = 'high-field'
TAG_LF = 'low-field-uncalibrated'
TAG_FALLBACK = 'fallback-no-record'

I_OUTER_KEY = 'i_outer_imaging'


def cross_section_from_outer_current(i_outer_A):
    """Per-shot cross section from the recorded outer-coil current.

    Args:
        i_outer_A (array-like or None): outer-coil current at imaging (A), any
            shape. ``None`` means the run has no record at all. NaN entries
            (a vault chunk without the container) fall back individually.

    Returns:
        tuple: ``(sigma_m2, source)``; ``sigma_m2`` is a float64 array and
        ``source`` a string array, both with the shape of ``i_outer_A``
        (shape ``()`` for ``None``).
    """
    if i_outer_A is None:
        return (np.array(SIGMA_FALLBACK_M2, dtype=np.float64),
                np.array(TAG_FALLBACK))

    i = np.asarray(i_outer_A, dtype=np.float64)
    recorded = np.isfinite(i)
    high = recorded & (i >= I_OUTER_HF_THRESHOLD_A)
    low = recorded & ~high

    sigma = np.full(i.shape, SIGMA_FALLBACK_M2, dtype=np.float64)
    sigma[high] = SIGMA_HF_M2
    sigma[low] = SIGMA_LF_M2

    source = np.full(i.shape, TAG_FALLBACK, dtype='<U32')
    source[high] = TAG_HF
    source[low] = TAG_LF
    return sigma, source


def recorded_outer_current(data, key=I_OUTER_KEY):
    """The recorded outer-coil current from an atomdata ``data`` container, or
    ``None`` when the run predates the record.

    Safe on any object: a missing ``keys`` attribute, a key not in ``keys``,
    or a missing attribute all mean "not recorded".
    """
    if data is None:
        return None
    keys = getattr(data, 'keys', None)
    if keys is None or key not in list(keys):
        return None
    value = getattr(data, key, None)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def cross_section_for_run(data, scan_shape):
    """Per-shot cross section and source tag for a run.

    Args:
        data: the atomdata DataVault-like object (``ad.data``).
        scan_shape (tuple): ``ad.xvardims``; the returned arrays are broadcast
            to this shape so a run without the record still gets one value per
            shot.

    Returns:
        tuple: ``(sigma_m2, source)`` arrays of shape ``scan_shape``.
    """
    scan_shape = tuple(int(n) for n in np.atleast_1d(scan_shape))
    i_outer = recorded_outer_current(data)
    if i_outer is not None:
        # the container is scan-shaped (trailing size-1 axis already squeezed
        # by the saver); tolerate a stray trailing axis
        if i_outer.shape != scan_shape and i_outer.size == int(np.prod(scan_shape)):
            i_outer = i_outer.reshape(scan_shape)
        elif i_outer.shape != scan_shape:
            i_outer = None
    sigma, source = cross_section_from_outer_current(i_outer)
    sigma = np.broadcast_to(sigma, scan_shape).copy()
    source = np.broadcast_to(source, scan_shape).copy()
    return sigma, source
