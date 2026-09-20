"""Unit detection for params: the family, the size-picked unit, and the SI
formatting the Adjust panel copies with.

Nothing here touches hardware or data on disk.
"""
import numpy as np
import pytest

from waxa.units import (
    family_of,
    format_si,
    mult_for,
    unit_for_magnitude,
    unit_for_param,
    unit_options,
)


class FakeParams:
    """Stands in for ExptParams: unit detection reads the trailing comments."""

    def __init__(self):
        self.t_probe = 1.e-3          # s
        self.f_raman = 12.e6          # MHz
        self.v_bias = 3.3             # V
        self.no_comment_here = 1.0
        self.t_hold = 50.e-3          # s, sets how long the trap stays up
        self.i_feshbach = 182.        # 182 A on the outer coils
        self.pfrac_probe = 0.8
        # self.pfrac_probe = .736 # much higher power/volt than the repump


def test_only_a_comment_that_names_a_unit_is_believed():
    """A commented-out copy of a line, or prose, must not supply a unit."""
    params = FakeParams()
    # 'ms' used to be picked up out of the sentence on the commented-out line
    assert unit_for_param("pfrac_probe", [0.8], params_obj=params) == ""
    # a unit followed by prose is still a unit
    assert unit_for_param("t_hold", [50.e-3], params_obj=params) == "ms"
    # a number first means the comment is a note, not a unit: fall back to the name
    assert unit_for_param("i_feshbach", [182.], params_obj=params) == "A"


def test_the_comment_decides_the_family_and_the_value_decides_the_unit():
    params = FakeParams()
    # commented as seconds, but a 20 us default is shown in us
    assert unit_for_param("t_probe", [20.e-6], params_obj=params) == "µs"
    assert unit_for_param("t_probe", [1.5], params_obj=params) == "s"
    assert unit_for_param("f_raman", [12.e6], params_obj=params) == "MHz"
    assert unit_for_param("f_raman", [4.e3], params_obj=params) == "kHz"
    assert unit_for_param("v_bias", [3.3], params_obj=params) == "V"


def test_the_name_decides_when_there_is_no_comment():
    """The units the K machine's own adjust list needs."""
    cases = [
        ("t_tof", 20.e-6, "µs"),
        ("t_mot_load", 1.0, "s"),
        ("t_d1cmot", 10.e-3, "ms"),
        ("t_rtio", 8.e-9, "ns"),
        ("i_mot", 60.0, "A"),
        ("i_hf_tweezer_evap1_current", 195.0, "A"),
        ("v_xshim_current", 4.0, "V"),
        ("v_pd_lightsheet_rampup_end", 5.0, "V"),
        ("detune_d1_c_gm", 6.0, "Γ"),
        ("frequency_detuned_imaging", 24.e6, "MHz"),
        ("pfrac_d1_c_gm", 0.3, ""),
        ("amp_d2_c_imaging", 0.065, ""),
        ("power_tweezer", 7.3e-5, "µW"),
        ("something_unknown", 12.0, ""),
    ]
    for key, value, expected in cases:
        assert unit_for_param(key, [value]) == expected, key


def test_optical_power_is_a_family_like_the_others():
    """power_* was only ever guessed; it needs a family to be switchable."""
    assert [label for label, _ in unit_options("mW")] == ["nW", "µW", "mW", "W"]
    assert family_of("µW") == "power"
    assert mult_for("mW") == 1e3
    # sized like time and frequency: the value picks the unit
    assert unit_for_magnitude("power", [7.3e-5]) == "µW"
    assert unit_for_magnitude("power", [0.2]) == "mW"
    assert unit_for_magnitude("power", [2.0]) == "W"


def test_a_zero_default_falls_back_to_the_range_then_to_si():
    # t_lightsheet_rampdown3 starts at 0 and is adjusted over 0 .. 200 ms
    assert unit_for_param("t_lightsheet_rampdown3", [0.0, 0.0, 200.e-3]) == "ms"
    assert unit_for_param("t_nothing_known", [0.0, 0.0, 0.0]) == "s"


def test_a_unit_offers_its_family_and_nothing_else():
    assert [label for label, _ in unit_options("µs")] == ["ns", "µs", "ms", "s"]
    assert [label for label, _ in unit_options("MHz")] == ["Hz", "kHz", "MHz", "GHz"]
    assert unit_options("V") == [("V", 1.0)]
    assert unit_options("") == [("", 1.0)]
    assert family_of("ms") == "time" and family_of("V") is None


def test_multipliers_convert_si_to_the_unit():
    assert mult_for("µs") == 1e6
    assert mult_for("MHz") == 1e-6
    assert mult_for("V") == 1.0
    assert mult_for("nonsense") == 1.0
    assert 20.e-6 * mult_for("µs") == pytest.approx(20.0)


def test_unit_for_magnitude_keeps_the_number_readable():
    assert unit_for_magnitude("time", [2.e-5]) == "µs"
    assert unit_for_magnitude("time", [0.25]) == "ms"     # 250 ms, not 0.25 s
    assert unit_for_magnitude("time", [1.0]) == "s"
    assert unit_for_magnitude("time", [np.nan, 0.0, 5.e-9]) == "ns"
    assert unit_for_magnitude("frequency", [24.e6]) == "MHz"
    # angle is not sized by magnitude
    assert unit_for_magnitude("angle", [0.5]) == "rad"


def test_format_si_writes_values_the_way_the_source_does():
    assert format_si(2.e-5, "µs") == "20.e-6"
    assert format_si(1.05e-5, "µs") == "10.5e-6"
    assert format_si(24.e6, "MHz") == "24.e6"
    assert format_si(0.065, "") == "0.065"
    assert format_si(180.0, "A") == "180."
    assert format_si(0.0, "µs") == "0."
    assert format_si(-3.5e-3, "ms") == "-3.5e-3"
    assert format_si(7, "", dtype=int) == "7"
    assert format_si(7.0, "", dtype='int') == "7"
    # a unit with no power-of-ten exponent stays a plain number
    assert format_si(0.5, "π") == "0.5"


def test_formatted_values_read_back_as_the_same_number():
    for value, unit in [(2.e-5, "µs"), (1.05e-5, "µs"), (24.e6, "MHz"),
                        (0.065, ""), (180.0, "A"), (-3.5e-3, "ms")]:
        assert float(format_si(value, unit)) == pytest.approx(value, rel=1e-12)


def test_the_old_import_path_still_works_and_is_the_same_object():
    import waxa.units as new
    from waxa.plotting import units as old
    assert old.guess_unit is new.guess_unit
    assert old.detect_unit is new.detect_unit
    assert old._normalize_name is new._normalize_name
    assert old.UNIT_MAP_FROM_COMMENT is new.UNIT_MAP_FROM_COMMENT
