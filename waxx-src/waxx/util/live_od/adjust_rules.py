"""Rules for changing an Adjust-panel spec, shared by the liveOD server and the
panel (no Qt, no sockets).

A range change that would leave the param's current value outside it is
refused, never clamped: narrowing the range must not move a live param.
"""

LOCK_REASONS = {
    "save_data": "this run saves data: the Adjust panel is read-only",
    "derived": "recomputed every shot by compute_derived(): adjust its inputs instead",
}


def lock_reason_text(locked: str) -> str:
    """What a locked row's tooltip and the panel banner say."""
    return LOCK_REASONS.get(locked, f"read-only ({locked})")


def adjust_range_error(key, current, min_val, max_val):
    """None if [min_val, max_val] is a valid range for a param now at ``current``,
    else why not. ``current`` None (no value known) only checks min <= max."""
    min_val, max_val = float(min_val), float(max_val)
    if min_val > max_val:
        return f"{key}: min {min_val:g} > max {max_val:g}"
    if current is None:
        return None
    current = float(current)
    if current < min_val or current > max_val:
        return (f"{key}: range [{min_val:g}, {max_val:g}] excludes the current value "
                f"{current:g}; change the value first, or keep it inside the range")
    return None
