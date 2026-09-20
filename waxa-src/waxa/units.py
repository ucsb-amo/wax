"""Unit detection and unit families.

Everything in the codebase is SI. These helpers work out a display unit and
multiplier for a parameter so that plots can be labelled ``t_tof (µs)`` instead
of ``t_tof`` with values in seconds, and so that the liveOD Adjust panel can
show 20 µs instead of 2.000e-05.

Resolution order in :func:`detect_unit` (plots, by xvar):

1. An explicit ``xvarunit`` / ``xvarmult`` passed by the caller.
2. The unit comment on the parameter's definition line in ``ExptParams``
   (``self.t_tof = 1.e-3  # s``), via :func:`get_param`.
3. A guess from the parameter name and the magnitude of its values, via
   :func:`guess_unit`.

:func:`unit_for_param` is the same resolution for a single parameter, returning
just the display unit label; the Adjust panel uses it, then offers the other
units of the same family (:data:`UNIT_FAMILIES`) in a dropdown.

This module lives here, rather than in ``waxa.plotting``, so that importing it
does not drag in matplotlib. ``waxa.plotting.units`` re-exports it.
"""

import inspect
import re

import numpy as np

__all__ = [
    'UNIT_MAP_FROM_COMMENT',
    'UNIT_FAMILIES',
    'FIXED_UNITS',
    'get_param',
    'unit_from_comment',
    'guess_unit',
    'detect_unit',
    'family_of',
    'mult_for',
    'unit_options',
    'unit_for_magnitude',
    'unit_for_param',
    'format_si',
]


def _normalize_name(name):
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return name.strip().strip("\x00")


UNIT_MAP_FROM_COMMENT = {
    "ns":        ("ns", 1e9),
    "us":        ("µs", 1e6),
    "µs":        ("µs", 1e6),
    "ms":        ("ms", 1e3),
    "s":         ("s", 1.0),
    "MHz":       ("MHz", 1e-6),
    "kHz":       ("kHz", 1e-3),
    "Hz":        ("Hz", 1.0),
    "Gamma":     ("Γ", 1.0),
    "V":         ("V", 1.0),
    "A":         ("A", 1.0),
    "amplitude": ("", 1.0),
    "fraction":  ("", 1.0),
    "rad":       ("π", 1 / np.pi),
    "unitless":  ("", 1.0),
}


# Units that mean the same physical quantity, smallest first. The multiplier
# converts an SI value to that unit. Only the families listed in
# _SIZED_FAMILIES are re-picked from the size of a value.
UNIT_FAMILIES = {
    'time':      [("ns", 1e9), ("µs", 1e6), ("ms", 1e3), ("s", 1.0)],
    'frequency': [("Hz", 1.0), ("kHz", 1e-3), ("MHz", 1e-6), ("GHz", 1e-9)],
    'length':    [("nm", 1e9), ("µm", 1e6), ("mm", 1e3), ("m", 1.0)],
    'power':     [("nW", 1e9), ("µW", 1e6), ("mW", 1e3), ("W", 1.0)],
    'angle':     [("rad", 1.0), ("π", 1 / np.pi)],
}

_SIZED_FAMILIES = ('time', 'frequency', 'length', 'power')

# Units with no other unit to switch to.
FIXED_UNITS = {
    "":  1.0,
    "Γ": 1.0,
    "V": 1.0,
    "A": 1.0,
}


def get_param(params_obj, param_name):
    """Reads the unit comment off a parameter's definition line.

    Returns ``(unit_label, mult)``, or ``(None, 1.0)`` when the source is not
    available, the parameter has no trailing comment, or the comment does not
    name a unit in :data:`UNIT_MAP_FROM_COMMENT`.
    """
    param_name = _normalize_name(param_name)

    try:
        src = inspect.getsource(params_obj.__class__)
    except (OSError, TypeError):
        return None, 1.0
    pattern = rf"self\.{re.escape(param_name)}\s*=\s*.*?#\s*([^\n]+)"
    m = re.search(pattern, src)
    if not m:
        return None, 1.0

    raw_unit = m.group(1).strip()

    for key, (unit_label, mult) in UNIT_MAP_FROM_COMMENT.items():
        if key in raw_unit:
            return unit_label, mult

    return None, 1.0


def unit_from_comment(params_obj, param_name):
    """Like :func:`get_param`, but only trusts a comment that names a unit.

    Two things make :func:`get_param` say more than it knows: it matches
    commented-out copies of a definition (``ExptParams`` keeps old values that
    way), and it looks for the unit anywhere in the comment, so a prose note
    lends its letters to a unit that was never meant (``pfrac_d1_c_gm`` picked
    up 'ms' from the sentence on a commented-out line above it).

    Here the definition has to be live, and the comment has to *start* with the
    unit, which is how the unit comments are actually written.  Anything else
    gives ``(None, 1.0)``, and the caller falls back to the name.

    get_param is left as it was: plot labels have been reading it for a long
    time, and loosening or tightening them is a separate decision.
    """
    param_name = _normalize_name(param_name)
    try:
        src = inspect.getsource(params_obj.__class__)
    except (OSError, TypeError):
        return None, 1.0

    pattern = rf"^[ \t]*self\.{re.escape(param_name)}\s*=\s*[^#\n]*#\s*(\S+)"
    m = re.search(pattern, src, re.MULTILINE)
    if not m:
        return None, 1.0

    token = m.group(1).strip(".,;:!?()[]")
    if token in UNIT_MAP_FROM_COMMENT:
        return UNIT_MAP_FROM_COMMENT[token]
    return None, 1.0


def guess_unit(name, values):
    """Guesses a display unit from a parameter name and its values.

    Returns ``(unit_label, mult)``, or ``(None, 1.0)`` when nothing matches.
    """
    name = _normalize_name(name)
    lname = name.lower()

    try:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None, 1.0
        vmax = float(np.max(np.abs(vals)))
    except Exception:
        return None, 1.0

    # Time
    if lname.startswith("t_") or "time" in lname or lname.endswith("_t"):
        if vmax >= 1:
            return "s", 1.0
        elif vmax >= 1e-3:
            return "ms", 1e3
        elif vmax >= 1e-6:
            return "µs", 1e6
        elif vmax >= 1e-9:
            return "ns", 1e9
        else:
            return "s", 1.0

    # Frequency
    if ("freq" in lname or "frequency" in lname or "_detuning" in lname or lname.startswith("f_")):
        if vmax >= 1e9:
            return "GHz", 1e-9
        elif vmax >= 1e6:
            return "MHz", 1e-6
        elif vmax >= 1e3:
            return "kHz", 1e-3
        else:
            return "Hz", 1.0

    # detuning in units of Gamma Γ
    if lname.startswith("detune_") or "detun_" in lname:
        return "Γ", 1.0

    # Voltage
    if lname.startswith("v_") or "volt" in lname:
        return "V", 1.0

    # Current
    if lname.startswith("i_") or "current" in lname:
        return "A", 1.0

    # Amplitude / power fraction (dimensionless)
    if (lname.startswith("amp_") or
            lname.startswith("pfrac_") or "fraction" in lname):
        return "amp", 1.0

    # Optical power
    if lname.startswith("power_"):
        if vmax >= 1:
            return "W", 1.0
        elif vmax >= 1e-3:
            return "mW", 1e3
        elif vmax >= 1e-6:
            return "µW", 1e6
        else:
            return "nW", 1e9

    if (lname.startswith("phase_")):
        return "π", 1/np.pi

    if (lname.startswith("dimension_")):
        if vmax >= 1:
            return "m", 1.0
        elif vmax >= 1e-3:
            return "mm", 1e3
        elif vmax >= 1e-6:
            return "µm", 1e6
        elif vmax >= 1e-9:
            return "nm", 1e9
        else:
            return "m", 1.0

    # Default: unknown / unitless
    return None, 1.0


def detect_unit(
    ad: "atomdata | None" = None,
    xvar_idx=0,
    xvarunit="",
    xvarmult=1.0,
    xvarnames=None,
    xvar_values=None,
    params_obj=None,
    verbose=False,
):
    """Resolves the display unit for one xvar.

    Pass either an ``atomdata`` (``ad``) or an explicit ``xvarnames`` list,
    optionally with ``xvar_values`` and ``params_obj``.

    Returns ``(unit, mult, xvarname)``: the display unit string (``''`` when
    unknown), the multiplier that converts SI values to that unit, and the
    cleaned-up xvar name for labelling.
    """
    if ad is None and xvarnames is None:
        raise ValueError("detect_unit requires either `ad` or `xvarnames`.")

    if ad is not None:
        xvarname = _normalize_name(ad.xvarnames[xvar_idx])
        xvar_vals = ad.xvars[xvar_idx]
        params_obj = ad.params if params_obj is None else params_obj
    else:
        xvarname = _normalize_name(xvarnames[xvar_idx])
        xvar_vals = [] if xvar_values is None else xvar_values

    # not `unit_from_comment`: that is the name of the stricter helper above
    detected_unit = None
    detected_mult = 1.0
    if params_obj is not None:
        detected_unit, detected_mult = get_param(params_obj, xvarname)

    source = "comment"
    if detected_unit is None:
        detected_unit, detected_mult = guess_unit(xvarname, xvar_vals)
        source = "guess"

    final_unit = xvarunit if xvarunit != "" else (detected_unit or "")

    final_mult = xvarmult if xvarmult != 1.0 else (detected_mult or 1.0)

    if verbose:
        print(f"xvar = {xvarname}, source = {source}, "
              f"detected unit = {detected_unit}, multiplier = {detected_mult:.1e}")
        print(f"final xvarunit = {final_unit}, xvarmult = {final_mult:.1e}")

    return final_unit, final_mult, xvarname


# ----------------------------------------------------------------------
# Unit families — what a unit can be switched to, and how values scale
# ----------------------------------------------------------------------

def family_of(unit):
    """Name of the family *unit* belongs to, or None for a fixed/unknown unit."""
    for family, options in UNIT_FAMILIES.items():
        if any(label == unit for label, _ in options):
            return family
    return None


def mult_for(unit):
    """Multiplier that converts an SI value into *unit*. Unknown units give 1.0."""
    for options in UNIT_FAMILIES.values():
        for label, mult in options:
            if label == unit:
                return mult
    return FIXED_UNITS.get(unit, 1.0)


def unit_options(unit):
    """``[(label, mult), ...]`` a parameter shown in *unit* can be switched to.

    A unit with no family gives back just itself, so callers can always offer
    the list without special-casing.
    """
    family = family_of(unit)
    if family is None:
        return [(unit, mult_for(unit))]
    return list(UNIT_FAMILIES[family])


def unit_for_magnitude(family, values):
    """Largest unit of *family* that keeps the first usable value at >= 1.

    ``values`` is tried in order; the first finite, non-zero one decides. With
    nothing usable, the SI unit of the family is returned.
    """
    options = UNIT_FAMILIES[family]
    si_label = next((label for label, mult in options if mult == 1.0), options[-1][0])
    if family not in _SIZED_FAMILIES:
        return si_label

    magnitude = None
    for value in values:
        try:
            value = abs(float(value))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            magnitude = value
            break
    if magnitude is None:
        return si_label

    best = options[0][0]
    for label, mult in options:
        if magnitude * mult >= 1:
            best = label
        else:
            break
    return best


def unit_for_param(name, values=(), params_obj=None):
    """Display unit for one parameter.

    The ``ExptParams`` comment decides the family when *params_obj* is given;
    otherwise the name does. Within a sized family (time, frequency, length)
    the unit is then picked from the size of the first usable value, so a
    ``t_tof`` that starts at 20e-6 shows as µs even when it can be scanned into
    the ms range.

    Returns the unit label, ``''`` when the parameter is dimensionless or
    unrecognised.
    """
    name = _normalize_name(name)
    values = list(values)

    label = None
    if params_obj is not None:
        label, _ = unit_from_comment(params_obj, name)
    if label is None:
        label, _ = guess_unit(name, values if values else [0.0])
    if label is None or label == 'amp':
        return ''

    family = family_of(label)
    if family is None:
        return label if label in FIXED_UNITS else ''
    return unit_for_magnitude(family, values)


def format_si(value, unit='', dtype=float):
    """Formats an SI *value* the way parameters are written in source.

    The unit only sets the exponent, so a time shown in µs comes back as
    ``20.e-6`` rather than ``2e-05``; the value itself stays SI.
    """
    if dtype is int or dtype == 'int':
        return str(int(round(float(value))))

    value = float(value)
    mult = mult_for(unit)
    exponent = round(-np.log10(mult)) if mult > 0 else 0
    # only a power-of-ten unit gives a clean exponent (π, for one, does not)
    decade = mult > 0 and np.isclose(mult, 10.0 ** -exponent)

    if mult == 1.0 or value == 0 or not decade:
        text = f"{value:.6g}"
        return text + "." if re.fullmatch(r"-?\d+", text) else text

    mantissa = f"{value * mult:.6g}"
    if re.fullmatch(r"-?\d+", mantissa):
        mantissa += "."
    return f"{mantissa}e{int(exponent)}"
