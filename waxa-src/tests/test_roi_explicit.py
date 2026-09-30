"""roi_id given directly as a pixel pair ([x0, x1], [y0, y1])."""
import numpy as np
import pytest

from waxa.roi import ROI, explicit_roi


def _roi(roi_id, images=None, **kw):
    if images is None:
        images = np.zeros((3, 50, 80))   # (N, H, W): 80 px in x, 50 px in y
    return ROI(run_id=1, roi_id=roi_id, server_talk=object(), images=images,
               printouts=False, **kw)


@pytest.mark.parametrize("roi_id", [
    ([10, 20], [5, 15]),
    [[10, 20], [5, 15]],
    ((10, 20), (5, 15)),
    np.array([[10, 20], [5, 15]]),
    np.array([[10., 20.], [5., 15.]]),
    ([np.int64(10), np.int64(20)], [5, 15]),
])
def test_pixel_pair_forms(roi_id):
    roi = _roi(roi_id)
    assert roi.roix == [10, 20] and roi.roiy == [5, 15]
    assert all(type(v) is int for v in (*roi.roix, *roi.roiy))
    assert roi.crop(np.ones((2, 50, 80))).shape == (2, 10, 10)


@pytest.mark.parametrize("roi_id", [None, 0, 12345, np.int64(7), "auto", "key"])
def test_other_forms_are_not_pixel_pairs(roi_id):
    assert explicit_roi(roi_id) is None


@pytest.mark.parametrize("roi_id", [
    [10, 20],                      # one range only
    [10, 20, 5, 15],               # flat
    ([10, 20], [5, 15], [0, 1]),   # three ranges
    ([10.5, 20], [5, 15]),         # not whole pixels
    ([20, 10], [5, 15]),           # reversed
    ([10, 10], [5, 15]),           # empty
    ([-1, 10], [5, 15]),           # negative
    (["a", "b"], [5, 15]),
])
def test_bad_pixel_pairs_raise(roi_id):
    with pytest.raises(ValueError):
        explicit_roi(roi_id)


def test_pair_past_frame_raises():
    with pytest.raises(ValueError, match="camera frame"):
        _roi(([70, 81], [0, 10]))
    with pytest.raises(ValueError, match="camera frame"):
        _roi(([0, 10], [40, 51]))
    roi = _roi(([0, 80], [0, 50]))   # whole frame is fine
    assert roi.roix == [0, 80] and roi.roiy == [0, 50]


def test_pair_overrides_saved_roi():
    roi = ROI(run_id=1, roi_id=([1, 2], [3, 4]), server_talk=object(),
              images=np.zeros((1, 50, 80)), printouts=False,
              current_saved_roi=[[0, 80], [0, 50]])
    assert roi.roix == [1, 2] and roi.roiy == [3, 4]
