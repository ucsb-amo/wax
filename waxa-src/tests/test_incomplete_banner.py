"""The ``!!`` block atomdata prints for a run file with data_complete=False: the
"last slots are empty" text only when fewer images arrived than expected; a run
flagged for another reason with every image says so; unknown counts claim
neither. Pure text, no files.

Passes once the guarded proposal for waxa/atomdata_base.py (LIVEOD, incomplete
banner) is applied.
"""


def test_fewer_images_than_expected_keeps_the_shift_warning():
    from waxa.atomdata_base import incomplete_banner
    text = incomplete_banner(80713, "60/63 images received; camera timed out", 60, 63)
    lines = text.splitlines()
    assert lines[0] == lines[-1] == "!" * 72
    assert lines[1] == "!! RUN 80713 IS INCOMPLETE: 60/63 images received; camera timed out"
    assert "!! 60 of 63 images arrived. Images are stored in arrival order," in lines
    assert "the last slots are empty" in text and "All" not in text


def test_every_image_arrived_but_flagged():
    from waxa.atomdata_base import incomplete_banner
    reason = "FRAME ALIGNMENT SUSPECT: frame 0 is filed as shot 0's but arrived 0.500 s before ..."
    text = incomplete_banner(80714, reason, 63, 63)
    assert f"!! RUN 80714 IS INCOMPLETE: {reason}" in text
    assert "All 63 images arrived, but the run was flagged (reason above): do not" in text
    assert "trust per-shot image data from this run until the reason is understood." in text
    assert "last slots are empty" not in text and "of 63 images arrived" not in text


def test_counts_the_file_does_not_record_claim_neither():
    from waxa.atomdata_base import incomplete_banner
    for got, exp in ((None, None), (None, 63), ("?", "?")):
        text = incomplete_banner(1, "why", got, exp)
        assert "does not record how many images arrived" in text
        assert "last slots are empty" not in text and "All" not in text


def test_numpy_counts_from_the_file_work():
    import numpy as np
    from waxa.atomdata_base import incomplete_banner
    assert "All 3 images arrived" in incomplete_banner(1, "r", np.int64(3), np.int64(3))
    assert "2 of 3 images arrived" in incomplete_banner(1, "r", np.int64(2), np.int64(3))


def test_the_banner_is_ascii():
    from waxa.atomdata_base import incomplete_banner
    for got in (2, 3, None):
        assert incomplete_banner(1, "reason", got, 3).isascii()
