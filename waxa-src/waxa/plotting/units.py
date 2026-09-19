"""Unit detection for scan variables.

Everything in the codebase is SI. These helpers work out a display unit and
multiplier for an xvar so that plots can be labelled ``t_tof (µs)`` instead of
``t_tof`` with values in seconds.

Resolution order in :func:`detect_unit`:

1. An explicit ``xvarunit`` / ``xvarmult`` passed by the caller.
2. The unit comment on the parameter's definition line in ``ExptParams``
   (``self.t_tof = 1.e-3  # s``), via :func:`get_param`.
3. A guess from the parameter name and the magnitude of its values, via
   :func:`guess_unit`.
"""

import inspect
import re

import numpy as np

__all__ = [
    'UNIT_MAP_FROM_COMMENT',
    'get_param',
    'guess_unit',
    'detect_unit',
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


def get_param(params_obj, param_name):
    """Reads the unit comment off a parameter's definition line.

    Returns ``(unit_label, mult)``, or ``(None, 1.0)`` when the source is not
    available, the parameter has no trailing comment, or the comment does not
    name a unit in :data:`UNIT_MAP_FROM_COMMENT`.
    """
    param_name = _normalize_name(param_name)

    try:
        src = inspect.getsource(params_obj.__class__)
    except OSError:
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

    unit_from_comment = None
    mult_from_comment = 1.0
    if params_obj is not None:
        unit_from_comment, mult_from_comment = get_param(params_obj, xvarname)

    source = "comment"
    if unit_from_comment is None:
        unit_from_comment, mult_from_comment = guess_unit(xvarname, xvar_vals)
        source = "guess"

    final_unit = xvarunit if xvarunit != "" else (unit_from_comment or "")

    final_mult = xvarmult if xvarmult != 1.0 else (mult_from_comment or 1.0)

    if verbose:
        print(f"xvar = {xvarname}, source = {source}, "
              f"detected unit = {unit_from_comment}, multiplier = {mult_from_comment:.1e}")
        print(f"final xvarunit = {final_unit}, xvarmult = {final_mult:.1e}")

    return final_unit, final_mult, xvarname
