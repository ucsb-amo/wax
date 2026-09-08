"""Tests for automatic ROI suggestion.

The synthetic tests run anywhere. The regression tests need the data drive
(%data%, BananaStand) and skip themselves when it is not mounted.
"""

import os

import numpy as np
import pytest

from waxa.image_processing.auto_roi import (AutoRoiResult, split_images,
                                            suggest_roi)

# A small synthetic frame. Big enough that the border trim and the smoothing
# kernels have room to work, small enough that a full test run is instant.
FRAME = 128
PROBE = 4000.0
DARK = 590.0


def _frames(n_shots=12, blob=None, rng_seed=0, hot_pixel=None, hot_row=None):
    """Build (atoms, light) frames with Poisson noise and an optional blob.

    Args:
        n_shots (int): number of shots.
        blob (tuple or None): (y, x, sigma, depth) absorption blob, where depth
            is the fractional absorption at the peak.
        rng_seed (int): seed for reproducibility.
        hot_pixel (tuple or None): (y, x) pixel forced to a huge value.
        hot_row (int or None): row index forced to a huge value.

    Returns:
        tuple: (atoms, light), both (n_shots, FRAME, FRAME) uint16.
    """
    rng = np.random.default_rng(rng_seed)
    light = rng.poisson(PROBE, size=(n_shots, FRAME, FRAME)) + DARK
    atoms = rng.poisson(PROBE, size=(n_shots, FRAME, FRAME)) + DARK

    if blob is not None:
        y, x, sigma, depth = blob
        yy, xx = np.mgrid[0:FRAME, 0:FRAME]
        profile = depth * np.exp(-((yy - y)**2 + (xx - x)**2) / (2 * sigma**2))
        atoms = atoms - (atoms - DARK) * profile[None]

    atoms = atoms.astype(np.uint16)
    light = light.astype(np.uint16)
    if hot_pixel is not None:
        atoms[:, hot_pixel[0], hot_pixel[1]] = 60000
    if hot_row is not None:
        atoms[:, hot_row, :] = 60000
    return atoms, light


def _contains(result, y, x):
    """Whether the suggested box contains the pixel (y, x)."""
    return (result.roiy[0] <= y < result.roiy[1]
            and result.roix[0] <= x < result.roix[1])


def test_finds_a_blob():
    atoms, light = _frames(blob=(70, 40, 4.0, 0.5))
    r = suggest_roi(atoms=atoms, light=light)
    assert r.valid, r.reason
    assert _contains(r, 70, 40)
    # A single small cloud must not spread the box over the whole frame.
    assert r.roiy[1] - r.roiy[0] < FRAME // 2
    assert r.roix[1] - r.roix[0] < FRAME // 2


def test_blob_present_in_only_some_shots():
    """A sum must not dilute: shots with no atoms contribute nothing."""
    with_atoms, light_a = _frames(n_shots=4, blob=(70, 40, 4.0, 0.5), rng_seed=1)
    without, light_b = _frames(n_shots=20, blob=None, rng_seed=2)
    atoms = np.concatenate([with_atoms, without])
    light = np.concatenate([light_a, light_b])
    r = suggest_roi(atoms=atoms, light=light)
    assert r.valid, r.reason
    assert _contains(r, 70, 40)


def test_moving_blob_gives_a_box_spanning_both_positions():
    a1, l1 = _frames(n_shots=8, blob=(50, 40, 4.0, 0.5), rng_seed=3)
    a2, l2 = _frames(n_shots=8, blob=(80, 40, 4.0, 0.5), rng_seed=4)
    r = suggest_roi(atoms=np.concatenate([a1, a2]),
                    light=np.concatenate([l1, l2]))
    assert r.valid, r.reason
    assert _contains(r, 50, 40) and _contains(r, 80, 40)


def test_diffuse_blob_survives_alongside_a_dense_one():
    """A TOF scan in one run: dense early, diffuse late.

    The diffuse cloud carries plenty of integrated signal but a low per-pixel
    amplitude, so a detector that scores per pixel and then cuts at a fraction
    of the global peak sees only the dense one and crops the long-TOF shots
    away.
    """
    dense, l1 = _frames(n_shots=20, blob=(45, 64, 3.0, 0.6), rng_seed=6)
    diffuse, l2 = _frames(n_shots=6, blob=(95, 64, 10.0, 0.10), rng_seed=7)
    r = suggest_roi(atoms=np.concatenate([dense, diffuse]),
                    light=np.concatenate([l1, l2]))
    assert r.valid, r.reason
    assert _contains(r, 45, 64), "lost the dense cloud"
    assert _contains(r, 95, 64), "lost the diffuse cloud"


def test_a_faint_diffuse_cloud_alone_is_found():
    """The same cloud with nothing bright to compete with."""
    atoms, light = _frames(n_shots=20, blob=(95, 64, 10.0, 0.10), rng_seed=8)
    r = suggest_roi(atoms=atoms, light=light)
    assert r.valid, r.reason
    assert _contains(r, 95, 64)


def test_small_blob_survives():
    """A few-pixel tweezer cloud must not be filtered away."""
    atoms, light = _frames(blob=(64, 64, 1.2, 0.7))
    r = suggest_roi(atoms=atoms, light=light)
    assert r.valid, r.reason
    assert _contains(r, 64, 64)


def test_no_atoms_is_rejected():
    atoms, light = _frames(blob=None)
    r = suggest_roi(atoms=atoms, light=light)
    assert not r.valid
    assert r.confidence < 0.15


def test_no_images_is_rejected_not_raised():
    r = suggest_roi(atoms=np.zeros((0, FRAME, FRAME), np.uint16),
                    light=np.zeros((0, FRAME, FRAME), np.uint16))
    assert not r.valid
    assert r.reason == "no images"


def test_hot_pixel_does_not_capture_the_box():
    """A single stuck pixel fires on every shot; smoothing must reject it."""
    atoms, light = _frames(blob=(70, 40, 4.0, 0.5), hot_pixel=(10, 110))
    r = suggest_roi(atoms=atoms, light=light)
    assert r.valid, r.reason
    assert _contains(r, 70, 40)
    assert not _contains(r, 10, 110)


def test_hot_border_row_does_not_capture_the_box():
    """The Andor's last rows carry a readout artifact; the trim must drop it."""
    atoms, light = _frames(blob=(70, 40, 4.0, 0.5), hot_row=FRAME - 1)
    r = suggest_roi(atoms=atoms, light=light)
    assert r.valid, r.reason
    assert _contains(r, 70, 40)


def test_blob_at_frame_edge_is_clipped_not_wrapped():
    atoms, light = _frames(blob=(6, 64, 3.0, 0.6))
    r = suggest_roi(atoms=atoms, light=light)
    assert r.roiy[0] >= 0 and r.roix[0] >= 0
    assert r.roiy[1] <= FRAME and r.roix[1] <= FRAME
    assert r.roiy[0] < r.roiy[1] and r.roix[0] < r.roix[1]


def test_fluorescence_sign_is_handled():
    """Fluorescence brightens the atoms frame; abs() must cover both signs."""
    atoms, light = _frames(blob=None, rng_seed=5)
    yy, xx = np.mgrid[0:FRAME, 0:FRAME]
    glow = 3000.0 * np.exp(-((yy - 70)**2 + (xx - 40)**2) / 32.0)
    atoms = np.clip(atoms.astype(np.float64) + glow, 0, 65535).astype(np.uint16)
    r = suggest_roi(atoms=atoms, light=light)
    assert r.valid, r.reason
    assert _contains(r, 70, 40)


def test_split_images_round_trips():
    n_shots = 5
    images = np.arange(n_shots * 3 * 4 * 4, dtype=np.uint16).reshape(n_shots * 3, 4, 4)
    atoms, light = split_images(images, n_pwa_per_shot=1)
    assert atoms.shape == (n_shots, 4, 4)
    assert light.shape == (n_shots, 4, 4)
    assert np.array_equal(atoms[0], images[0])
    assert np.array_equal(light[0], images[1])
    assert np.array_equal(atoms[1], images[3])


def test_split_images_handles_two_pwa_per_shot():
    n_shots = 3
    images = np.arange(n_shots * 4 * 2 * 2, dtype=np.uint16).reshape(n_shots * 4, 2, 2)
    atoms, light = split_images(images, n_pwa_per_shot=2)
    assert atoms.shape == (n_shots * 2, 2, 2)
    assert light.shape == (n_shots * 2, 2, 2)
    assert np.array_equal(atoms[1], images[1])
    assert np.array_equal(light[0], images[2])


def test_split_images_rejects_a_ragged_stack():
    with pytest.raises(ValueError):
        split_images(np.zeros((7, 4, 4), np.uint16), n_pwa_per_shot=1)


def test_suggest_roi_accepts_a_flat_stack():
    atoms, light = _frames(n_shots=6, blob=(70, 40, 4.0, 0.5))
    dark = np.full_like(atoms, int(DARK))
    images = np.empty((atoms.shape[0] * 3,) + atoms.shape[1:], atoms.dtype)
    images[0::3] = atoms
    images[1::3] = light
    images[2::3] = dark
    r = suggest_roi(images=images, n_pwa_per_shot=1)
    assert r.valid, r.reason
    assert _contains(r, 70, 40)


def test_dark_frame_is_irrelevant():
    """(atoms-dark) - (light-dark) == atoms-light, so dark must not matter."""
    atoms, light = _frames(blob=(70, 40, 4.0, 0.5))
    a = suggest_roi(atoms=atoms, light=light)
    b = suggest_roi(atoms=atoms + 0, light=light + 0)
    assert a.roix == b.roix and a.roiy == b.roiy


# Regression runs on the data drive: (run_id, relative path, expect_valid,
# cloud y, cloud x). The last one has no atoms and must be rejected.
REAL_RUNS = [
    (76047, r'2026-08-20\0076047_2026-08-20_18-11-45_hf_raman.hdf5', True, 269, 197),
    (76224, r'2026-08-24\0076224_2026-08-24_12-50-05_hf_raman.hdf5', True, 269, 197),
    (76036, r'2026-08-20\0076036_2026-08-20_16-51-12_hf_bec.hdf5', True, 269, 197),
    (76210, r'2026-08-24\0076210_2026-08-24_09-48-16_hf_raman.hdf5', False, None, None),
]

_DATA_DIR = os.getenv('data')
_has_data = bool(_DATA_DIR) and os.path.isdir(_DATA_DIR)
needs_data = pytest.mark.skipif(not _has_data,
                                reason="data drive (%data%) not reachable")


@needs_data
@pytest.mark.parametrize("run_id,rel,expect_valid,cy,cx", REAL_RUNS)
def test_real_runs(run_id, rel, expect_valid, cy, cx):
    import h5py
    path = os.path.join(_DATA_DIR, rel)
    if not os.path.isfile(path):
        pytest.skip(f"run {run_id} not on this data drive")
    with h5py.File(path, 'r') as h:
        images = h['data']['images'][()]
        nps = int(h['params']['N_pwa_per_shot'][()])
    r = suggest_roi(images=images, n_pwa_per_shot=nps)
    assert r.valid is expect_valid, f"run {run_id}: {r}"
    if expect_valid:
        assert _contains(r, cy, cx), \
            f"run {run_id} box {r.roiy},{r.roix} misses the cloud"


# A t_tof scan whose cloud both moves and spreads across the run. The box has
# to span every time of flight, not just the dense short-TOF end.
TOF_RUN = r'2026-09-01\0078237_2026-09-01_20-55-46_hf_bec.hdf5'
TOF_POSITIONS = [(267, 201), (213, 199), (165, 199)]


@needs_data
def test_tof_scan_box_spans_every_time_of_flight():
    import h5py
    path = os.path.join(_DATA_DIR, TOF_RUN)
    if not os.path.isfile(path):
        pytest.skip("TOF run not on this data drive")
    with h5py.File(path, 'r') as h:
        images = h['data']['images'][()]
        nps = int(h['params']['N_pwa_per_shot'][()])
    r = suggest_roi(images=images, n_pwa_per_shot=nps)
    assert r.valid, r.reason
    missed = [(y, x) for y, x in TOF_POSITIONS if not _contains(r, y, x)]
    assert not missed, f"box {r.roiy},{r.roix} misses {missed}"


@needs_data
def test_speed_is_a_small_fraction_of_load():
    """Detection must not noticeably slow down loading."""
    import h5py
    import time
    path = os.path.join(_DATA_DIR, REAL_RUNS[0][1])
    if not os.path.isfile(path):
        pytest.skip("reference run not on this data drive")
    with h5py.File(path, 'r') as h:
        images = h['data']['images'][()]
    suggest_roi(images=images)          # warm up
    t = time.perf_counter()
    suggest_roi(images=images)
    elapsed = time.perf_counter() - t
    assert elapsed < 0.5, f"detection took {elapsed * 1000.0:.0f} ms"
