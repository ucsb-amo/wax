"""Host-side behaviour of AD9910FastInit. The kernels need a core device; what
is tested here is the bookkeeping that tells a person what happened."""
import json

import pytest

from waxx.control.ad9910_fast_init import AD9910FastInit, FAIL_REASONS, RECORD_KEY
from waxx.util import console


def make(n=24, key="k"):
    return AD9910FastInit(core=None, core_cache=None, dds_list=[0] * n, cache_key=key)


def test_cache_key_is_scoped_to_the_channel_count():
    # a cached list of another length can be neither used nor replaced within a
    # kernel, so each channel count gets its own key
    assert make(24).cache_key == "k_24"
    assert make(20).cache_key != make(24).cache_key


def test_report_says_why():
    init = make()
    cases = {(0, 48, False): "cache hit",
             (24, 0, False): "no cache",
             (24, 46, False): "cache has 46 values, expected 48",
             (24, 48, True): "forced"}
    for (n_full, n_cached, forced), expected in cases.items():
        init._record(n_full, 24, n_cached, forced, 0.0, 0.0)
        assert init.report["why"].startswith(expected), init.report
        assert init.report["n_full"] == n_full and init.report["n_channels"] == 24


def test_failure_is_recorded_and_announced(capsys):
    init = make()
    init._record_failure(2, 1, 4, -2147479552)
    assert init.failures == [dict(urukul=2, ch=1, reason=FAIL_REASONS[4], raw="0x80001000")]
    out = capsys.readouterr().out
    assert "WARNING" in out and "urukul 2 ch 1" in out


@pytest.fixture
def level():
    """Set the console verbosity for one test, then put it back."""
    before = console.get_level()
    yield console.set_level
    console.set_level(before)


def test_the_usual_all_skipped_run_is_quiet_unless_verbose(capsys, level):
    """Since the terminal-output cleanup (2026-09-23) a cache hit prints only at
    VERBOSE; a full init on any channel -- a slower run -- prints at NORMAL."""
    level(console.NORMAL)
    make()._record(0, 24, 48, False, 0.013, 0.003)
    assert capsys.readouterr().out == ""
    make()._record(24, 24, 0, False, 1.4, 0.0)
    out = capsys.readouterr().out
    assert len(out.splitlines()) == 1
    assert "full init on 24 of 24 channels" in out and "no cache" in out

    level(console.VERBOSE)
    make()._record(0, 24, 48, False, 0.013, 0.003)
    out = capsys.readouterr().out
    assert len(out.splitlines()) == 1
    assert "full init on 0 of 24 channels" in out and "24 skipped (cache hit)" in out
    assert "13 ms" in out and "WARNING" not in out


def test_every_run_keeps_its_outcome_with_its_data(capsys, level):
    """Quiet on the terminal, but never unrecorded: record_to (the
    experiment's _extra_file_texts) gets report + failures as JSON."""
    level(console.QUIET)
    texts = {}
    init = AD9910FastInit(core=None, core_cache=None, dds_list=[0] * 24, cache_key="k",
                          record_to=texts)
    init._record_failure(2, 1, 3, 0x12)
    assert json.loads(texts[RECORD_KEY])["failures"][0]["reason"] == FAIL_REASONS[3]
    init._record(1, 24, 48, False, 0.09, 0.003)
    stored = json.loads(texts[RECORD_KEY])
    assert stored["report"]["n_full"] == 1 and stored["report"]["why"] == "cache hit"
    assert stored["failures"] == [dict(urukul=2, ch=1, reason=FAIL_REASONS[3], raw="0x12")]
    assert "WARNING" in capsys.readouterr().out         # failures still always print
    make()._record(0, 24, 48, False, 0.013, 0.003)       # no record_to: nothing breaks
