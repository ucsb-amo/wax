"""Moved to :mod:`waxa.units` on 2026-09-20, so that unit detection can be used
without importing matplotlib (the liveOD Adjust panel does).

Every name that lived here is re-exported, including the private helper
``_normalize_name`` that ``plotting_1d`` re-exports in turn, and they are the
same objects, so patching one patches the other.
"""

from waxa.units import (  # noqa: F401
    UNIT_MAP_FROM_COMMENT,
    UNIT_FAMILIES,
    FIXED_UNITS,
    _normalize_name,
    detect_unit,
    family_of,
    format_si,
    get_param,
    guess_unit,
    mult_for,
    unit_for_magnitude,
    unit_for_param,
    unit_from_comment,
    unit_options,
)

__all__ = [
    'UNIT_MAP_FROM_COMMENT',
    'get_param',
    'guess_unit',
    'detect_unit',
]
