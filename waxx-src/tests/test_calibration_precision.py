"""waxx.calibration.precision: how many digits a write-back gets, and the format."""
import math
import random
from decimal import Decimal

import pytest

from waxx.calibration.precision import PrecisionError, format_value, parse_literal


@pytest.mark.parametrize("value, unc, old, new", [
    # the lab's own example (expt_params convention, 2026-08-20)
    (8.823712345e-06, 2.1e-08, "8.3588e-06", "8.8237e-06"),
    # the floor: the rule alone gives 3.8e-06; the line had 5 significant figures
    (3.80123456e-06, 5e-07, "3.7785e-06", "3.8012e-06"),
    # the rule adds digits beyond the line's
    (3.80123456e-06, 2.3e-10, "3.8e-06", "3.80123e-06"),
    # trailing zeros count: 150.0 is 4 figures
    (151.3712, 3.0, "150.0", "151.4"),
    (151.3712, 30.0, "150.0", "151.4"),
    (151.3712, 30.0, "150.", "151."),
    (151.3712, 0.04, "150.0", "151.371"),
    # scientific with a non-normalised mantissa keeps its exponent (the unit)
    (2.531712e-4, 1.2e-7, "250.e-6", "253.17e-6"),
    (2.531712e-4, 1.2e-7, "0.25e-3", "0.25317e-3"),
    # normalised mantissa, exponent padding, E and explicit + kept
    (9.87e-10, 1e-12, "1.0e-10", "9.87e-10"),
    (1.2345678e-7, 1e-11, "1.1e-07", "1.23457e-07"),
    (1.2345678e7, 1e3, "1.1E+07", "1.23457E+07"),
    (1.234e6, 1e3, "1e6", "1.234e6"),
    # negative values
    (-0.01912345, 0.0003, "-0.019", "-0.01912"),
    (-2.5317e-3, 2e-6, "-2.5e-3", "-2.5317e-3"),
    # the repr cap: never more digits than the float has
    (0.1 + 0.2, 0.0, "0.3", "0.30000000000000004"),
    (1.23e-3, 1e-300, "1.e-3", "1.23e-3"),
    # an absurd uncertainty: the line's figures still hold
    (1.23e-3, 1e10, "1.e-3", "1.e-3"),
    (151.3712, 1e9, "150.0", "151.4"),
    # zero
    (0.0, 0.0012, "0.005", "0.0000"),
    (0.0, 2.1e-8, "250.e-6", "0.000e-6"),
])
def test_rule_and_format(value, unc, old, new):
    got = format_value(value, unc, old)
    assert got == new
    assert float(got) == float(Decimal(got))      # a valid Python float literal


def test_raman_frequency_keeps_the_lines_digits():
    """A ~12-figure value with a large stated uncertainty: the rule alone would
    write 41.235e6 (and lose the frequency); the line's 13 figures are a floor,
    and the float's 12 digits are padded with an exact zero up to it."""
    old = "41.23456789123e6"
    v = 41234570.1234
    assert format_value(v, 5.0e4, old) == "41.23457012340e6"
    assert float(format_value(v, 5.0e4, old)) == v
    # same in plain decimal (14 figures on the line, 13 in the float)
    assert format_value(412345701.2345, 5.0e4, "412345678.91234") == "412345701.23450"


# ---- S11 (user ruling 2026-10-09, option b) ------------------------------------------

@pytest.mark.parametrize("value, unc, old, new", [
    # decimal literal, rounding place above the units: scientific, same figures
    (12345.6, 300.0, "1500.", "1.235e4"),
    (-12345.6, 300.0, "1500.", "-1.235e4"),
    (1523.7, 100.0, "150.", "1.52e3"),
    (1523.7, 100.0, "150.0", "1524."),            # 4-figure floor: units place, no padding
    (12345.6, 300.0, "float(1500.)", "float(1.235e4)"),
    # decimal literal, rounding exactly at the units place: stays decimal
    (12345.6, 30.0, "1500.", "12346."),
    # kept-exponent sci literal, rounding place above its last place: normalised
    (1.2345e-3, 2.0e-4, "250.e-6", "1.23e-3"),
    (1.2345e-3, 2.0e-4, "250.e-06", "1.23e-03"),
    # ...at its last place: no padding, exponent kept
    (1.2345e-3, 2.0e-5, "250.e-6", "1234.e-6"),
    (1.2345e-3, 2.0e-6, "250.e-6", "1234.5e-6"),
    # the repr cap never shrinks the floor: padded to the line's figures
    (6.6e-06, 1.0e-08, "6.6403e-06", "6.6000e-06"),
    (6.612e-06, 2.1e-08, "6.6403e-06", "6.6120e-06"),
    (151.0, 0.4, "150.0", "151.0"),
    (0.1, 0.0, "0.1000", "0.1000"),
    (2.5e-4, None, "250.0e-6", "250.0e-6"),
    # zero is not padded into false precision either way
    (0.0, 300.0, "1500.", "0."),
])
def test_s11_no_padding_that_reads_as_precision(value, unc, old, new):
    got = format_value(value, unc, old, allow_no_unc=True)
    assert got == new
    inner = got.split("(")[-1].rstrip(")")
    assert float(inner) == float(Decimal(inner))
    if unc:
        assert abs(float(inner) - value) <= 0.05 * unc + 1e-15 * abs(value)


def test_int_literals_take_integral_values_exactly():
    assert format_value(50.0, 3, "50") == "50"
    assert format_value(1234.0, 50.0, "1200") == "1234"          # never rounded
    assert format_value(-2.0, 1, "-1") == "-2"
    assert format_value(51, 1, "np.int32(50)") == "np.int32(51)"
    assert format_value(7, 1, "int64(5)") == "int64(7)"
    with pytest.raises(PrecisionError, match="not an integer"):
        format_value(49.7, 3, "50")
    with pytest.raises(PrecisionError, match="not an integer"):
        format_value(50.5, 0.1, "np.int32(50)")


def test_float_wrapper_is_kept():
    assert format_value(1.2345, 0.01, "float(1.0)") == "float(1.234)"
    assert format_value(1.2345, 0.01, "np.float64(1.0)") == "np.float64(1.234)"


def test_kind_override():
    assert format_value(2.531712e-4, 1.2e-7, "0.00025", kind="sci") == "2.5317e-4"
    assert format_value(2.531712e-4, 1.2e-7, "250.e-6", kind="decimal") == "0.00025317"
    with pytest.raises(PrecisionError):
        format_value(1.0, 0.1, "1.0", kind="hex")


def test_missing_uncertainty():
    with pytest.raises(PrecisionError, match="no uncertainty"):
        format_value(1.2345678, None, "1.2")
    assert format_value(1.2345678, None, "1.2", allow_no_unc=True) == "1.2345678"
    assert format_value(2.531712e-4, None, "250.e-6", allow_no_unc=True) == "253.1712e-6"


@pytest.mark.parametrize("value, unc", [
    (math.nan, 0.1), (math.inf, 0.1), (-math.inf, 0.1), (1.0, math.nan), (1.0, math.inf),
    (1.0, -0.1), ("abc", 0.1), (1.0, "x"),
])
def test_refuses_bad_numbers(value, unc):
    with pytest.raises(PrecisionError):
        format_value(value, unc, "1.0")


@pytest.mark.parametrize("literal", [
    "self.y * 2", "np.array([1.])", "'abc'", "np.float32(1.0)", "1_000", "2*np.pi",
    "np.int32(1.5)", "", "True", "x",
])
def test_refuses_non_literals(literal):
    with pytest.raises(PrecisionError):
        parse_literal(literal)


def test_literal_parsing():
    lit = parse_literal("150.0")
    assert (lit.kind, lit.sig_figs) == ("decimal", 4)
    assert parse_literal("3.7785e-06").sig_figs == 5
    assert parse_literal("250.e-6").sig_figs == 3
    assert parse_literal("0.0050").sig_figs == 2
    assert parse_literal("0.0").sig_figs == 1
    assert parse_literal("np.int32(50)").kind == "int"
    assert parse_literal("np.int32(50)").value() == 50
    assert parse_literal("-1").value() == -1
    assert parse_literal("1e6").kind == "sci"


def test_property_error_bounded_and_digits_capped():
    """Rounded at the uncertainty's 2nd digit or finer: |written - value| <=
    0.05 unc; and never more significant digits than the value's repr."""
    rng = random.Random(1234)
    lits = ["1.0", "3.7785e-06", "250.e-6", "150.0", "0.005", "1.2E+07", "-2.5e-3"]
    for _ in range(3000):
        v = rng.uniform(-1, 1) * 10 ** rng.uniform(-12, 9)
        u = abs(v) * 10 ** rng.uniform(-14, 2)
        lit = rng.choice(lits)
        got = format_value(v, u, lit)
        w = float(got)
        assert abs(w - v) <= 0.05 * u + 1e-15 * abs(v), (v, u, lit, got)
        repr_digits = len(Decimal(repr(v)).normalize().as_tuple().digits)
        written = parse_literal(got).sig_figs           # trailing zeros written count
        # never a digit below the repr's last one
        assert len(Decimal(got).normalize().as_tuple().digits) <= repr_digits, (v, u, lit, got)
        # the line's figures are a floor, always (padded with exact zeros if need be)
        assert written >= parse_literal(lit).sig_figs, (v, u, lit, got)
        # a decimal result only when the rounding place (computed here on its
        # own) is at or below the units: never zeros padded left of the point
        d = Decimal(repr(v))
        p_rule = max(Decimal(repr(u)).adjusted() - 1, d.normalize().as_tuple().exponent)
        p = min(p_rule, d.adjusted() - (parse_literal(lit).sig_figs - 1))
        if parse_literal(got).kind == "decimal":
            assert p <= 0, (v, u, lit, got)
