"""Central verbosity control for run-time terminal output.

One process-wide level decides how chatty a run is on the terminal:

    QUIET   = 0 : warnings and errors only
    NORMAL  = 1 : default -- one-line run milestones (run id, scan summary,
                  quarter-progress, completion)
    VERBOSE = 2 : everything (per-shot progress, hardware/OPX chatter)

Set it with ``Expt(verbosity=...)`` / ``Base(verbosity=...)`` or the
``WAX_VERBOSITY`` environment variable (the kwarg wins). Warnings, errors,
and anything bearing on data integrity must NOT go through :func:`info` --
print those unconditionally.

Kernel code cannot call this module; ``Expt.__init__`` copies the level to
``self._verbosity`` (int) for kernel-side gating.
"""

import os

QUIET = 0
NORMAL = 1
VERBOSE = 2


def _env_level():
    try:
        return int(os.environ.get("WAX_VERBOSITY", NORMAL))
    except (TypeError, ValueError):
        return NORMAL


_level = _env_level()


def set_level(level):
    global _level
    _level = int(level)


def get_level() -> int:
    return _level


def info(msg, level=NORMAL):
    """Print ``msg`` when the current verbosity is at least ``level``."""
    if _level >= level:
        print(msg)
