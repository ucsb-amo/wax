"""A run loop's scan settings (waxx.util.device_state.loop_scan).  Pure Python."""
import numpy as np
import pytest

from waxx.util.device_state import loop_scan as ls

TOF = ls.ScanSpec(xvar="t_tof", unit="ms", scale=1e-3, minimum=0.0, maximum=25e-3,
                  start=1e-3, stop=4e-3, n=9, repeats=5)


def test_defaults_values_and_description():
    s = TOF.defaults()
    assert s == {"start": 1e-3, "stop": 4e-3, "n": 9, "repeats": 5}
    assert np.allclose(ls.values(s), np.linspace(1e-3, 4e-3, 9))
    assert ls.describe(TOF, s) == "t_tof 1–4 ms, 9 points × 5 repeats (45 shots)"


def test_no_stop_value_repeats_the_start_value():
    s = ls.normalize(TOF, {"start": 2e-3, "stop": None, "n": 9, "repeats": 20})
    assert s["n"] == 1 and list(ls.values(s)) == [2e-3]
    assert ls.describe(TOF, s) == "t_tof 2 ms × 20 repeats (20 shots)"
    assert ls.normalize(TOF, {"start": 2e-3, "stop": "", "repeats": 1})["stop"] is None


@pytest.mark.parametrize("settings, why", [
    ({"start": -1e-3, "stop": None, "repeats": 1}, "below the minimum 0 ms"),
    ({"start": 1e-3, "stop": 30e-3, "n": 5, "repeats": 1}, "above the maximum 25 ms"),
    ({"start": "x", "repeats": 1}, "not a number"),
    ({"start": float("nan"), "repeats": 1}, "not finite"),
    ({"start": 1e-3, "stop": 2e-3, "n": 1, "repeats": 1}, "number of points must be 2"),
    ({"start": 1e-3, "stop": 2e-3, "n": 2.5, "repeats": 1}, "whole number"),
    ({"start": 1e-3, "repeats": 0}, "repeats must be 1"),
    ({"start": 1e-3, "repeats": True}, "whole number"),
])
def test_bad_settings_are_refused_saying_why(settings, why):
    with pytest.raises(ls.ScanSettingsError, match=why):
        ls.normalize(TOF, settings)


def test_the_run_reads_what_the_loop_hands_it():
    s = TOF.defaults()
    env = {ls.ENV_VAR: ls.to_env(s)}
    assert ls.scan_from_env(env) == s
    assert ls.scan_from_env({}) is None
    with pytest.raises(RuntimeError, match="not readable"):
        ls.scan_from_env({ls.ENV_VAR: "{nope"})
    with pytest.raises(RuntimeError, match="no shots"):
        ls.scan_from_env({ls.ENV_VAR: '{"start": 0.001, "stop": null, "n": 1, "repeats": 0}'})


def test_spec_from_a_lab_mapping():
    spec = ls.ScanSpec.from_mapping({"xvar": "t_tof", "unit": "ms", "scale": 1e-3,
                                     "start": 1e-3, "stop": 4e-3, "n": 9, "repeats": 5})
    assert spec.defaults()["n"] == 9 and spec.info()["unit"] == "ms"
