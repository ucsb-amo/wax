"""Per-key write-back policy: ``{key: {max_frac_change, min, max, min_shots, max_rel_unc, unit}}``.

The machine keeps its table in a module of its own (kexp:
``kexp.calibrations.writeback_policy.POLICY``). A key with no entry has no hard
checks (``record.evaluate`` then applies only the rules that need no policy).
An entry with a field this module does not know is refused, so a typo cannot
silently switch a check off.
"""

from __future__ import annotations

import importlib
from typing import Mapping, Optional, Union

FIELDS = ("max_frac_change", "min", "max", "min_shots", "max_rel_unc", "unit")


class PolicyError(ValueError):
    pass


def _check(policy: Mapping) -> dict:
    out = {}
    for key, entry in dict(policy).items():
        if not isinstance(key, str):
            raise PolicyError(f"policy key {key!r} is not a string")
        entry = dict(entry or {})
        unknown = sorted(set(entry) - set(FIELDS))
        if unknown:
            raise PolicyError(f"policy entry {key!r} has unknown field(s) {unknown}; "
                              f"known: {list(FIELDS)}")
        for f in ("max_frac_change", "min", "max", "max_rel_unc"):
            if entry.get(f) is not None:
                entry[f] = float(entry[f])
        for f in ("max_frac_change", "max_rel_unc"):
            if entry.get(f) is not None and entry[f] < 0:
                raise PolicyError(f"policy entry {key!r}: {f} must be >= 0")
        if entry.get("min_shots") is not None:
            entry["min_shots"] = int(entry["min_shots"])
        out[key] = entry
    return out


def load_policy(source: Union[None, Mapping, str] = None) -> dict:
    """The policy table from ``source``: a mapping, a module path (``pkg.mod``,
    whose ``POLICY`` is used, or ``pkg.mod:NAME``), or None (empty)."""
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return _check(source)
    if isinstance(source, str):
        mod_name, _, attr = source.partition(":")
        mod = importlib.import_module(mod_name)
        table = getattr(mod, attr or "POLICY")
        if not isinstance(table, Mapping):
            raise PolicyError(f"{source} is not a mapping")
        return _check(table)
    raise PolicyError(f"cannot load a policy from {source!r}")


def policy_for(policy: Optional[Mapping], key: str) -> dict:
    """The entry for ``key``; {} (no hard checks) when it has none."""
    return dict((policy or {}).get(key) or {})
