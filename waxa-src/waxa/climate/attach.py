"""Line climate data up with run data.

The anchor is the per-shot wall-clock time.  liveOD stamps every camera
frame with ``time.time()`` on the camera PC as it arrives, and
``atomdata`` exposes the atoms-frame stamp as ``ad.img_timestamp_atoms``
(shape ``(*xvardims,)``).  That is the best per-shot clock a run has, and
it survives shuffling, repeats and ``AtomdataVault`` stacking.

Fallbacks, in order, for runs without images: the per-shot
``timestamp_shot_end`` container, then the run start time parsed from the
file name (``<run_id>_<YYYY-MM-DD>_<HH-MM-SS>_<class>.hdf5``), repeated for
every shot.  :func:`shot_times` says which it used.

Clock caveat: the camera PC clock and the Zabbix server clock are both NTP
synced but are different machines; expect agreement to a second or so,
which is far below the one-minute sensor cadence.

Nothing here modifies the run: results come back as plain arrays (or are
set as new attributes on the ``atomdata`` object in memory only).
"""
from __future__ import annotations

import datetime as _dt
import os
import re
from typing import Iterable

import numpy as np

from waxa.climate.client import ClimateClient, ClimateSeries

_FNAME_RE = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")


def run_start_time(ad) -> float | None:
    """Run start as unix seconds, parsed from the data file name.

    Falls back to ``run_info.run_datetime_str`` (same format) and returns
    ``None`` if neither is available.  Deliberately does **not** use
    ``run_info.run_datetime``: on current-era files that is the load time,
    not the run time.
    """
    ri = getattr(ad, "run_info", None)
    cands = []
    fp = getattr(ri, "filepath", None)
    if isinstance(fp, (list, tuple, np.ndarray)):
        cands.extend(str(p) for p in fp)
    elif fp:
        cands.append(str(fp))
    for p in cands:
        m = _FNAME_RE.match(os.path.basename(p))
        if m:
            return _dt.datetime.strptime(f"{m.group(2)}_{m.group(3)}", "%Y-%m-%d_%H-%M-%S").timestamp()
    s = getattr(ri, "run_datetime_str", None)
    if isinstance(s, (bytes, np.bytes_)):
        s = s.decode()
    if isinstance(s, str):
        try:
            return _dt.datetime.strptime(s, "%Y-%m-%d_%H-%M-%S").timestamp()
        except ValueError:
            pass
    return None


def _usable_times(arr) -> np.ndarray | None:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return None
    ok = np.isfinite(a) & (a > 1e9)          # a real epoch stamp, not 0 / NaN padding
    if not ok.any():
        return None
    return np.where(ok, a, np.nan)


def shot_times(ad, return_source: bool = False):
    """Per-shot unix times, shape ``(*xvardims,)`` (or whatever the per-shot
    arrays are shaped on this object).

    Returns ``times`` or ``(times, source)`` where *source* is one of
    ``"img_timestamp_atoms"``, ``"timestamp_shot_end"``, ``"run_start"``.
    Missing / padded shots are NaN.  Raises if no clock is available.
    """
    for attr in ("img_timestamp_atoms", "timestamp_shot_end"):
        t = _usable_times(getattr(ad, attr, None))
        if t is not None:
            return (t, attr) if return_source else t
    t0 = run_start_time(ad)
    if t0 is None:
        raise ValueError("no per-shot timestamps and no parsable run start time on this object")
    shape = tuple(int(n) for n in np.atleast_1d(getattr(ad, "xvardims", [1])))
    t = np.full(shape, float(t0))
    return (t, "run_start") if return_source else t


def run_window(ad, pad_s: float = 600.0) -> tuple[float, float]:
    """``(t_from, t_to)`` unix seconds spanning the run's shots, padded."""
    t = shot_times(ad)
    return float(np.nanmin(t)) - pad_s, float(np.nanmax(t)) + pad_s


def climate_for_run(ad, names: Iterable[str] | None = None, client: ClimateClient | None = None,
                    host: str | None = None, method: str = "nearest",
                    max_gap_s: float | None = 180.0, pad_s: float = 600.0,
                    attach: bool = False) -> dict[str, np.ndarray]:
    """Climate readings for every shot of a run (or vault).

    Fetches the raw one-minute history of each sensor in *names* (default:
    every numeric sensor on *host*) over the run's time window and samples
    it at each shot time.  Returns ``{sensor name: array (*xvardims,)}``;
    the special key ``"_t_shot"`` holds the shot times used.  Shots more
    than ``max_gap_s`` from a sensor sample are NaN.

    With ``attach=True`` the arrays are also set on *ad* as
    ``ad.climate`` (a dict) - in memory only, nothing is written to disk.
    """
    cc = client if client is not None else ClimateClient(host=host or "K")
    t = shot_times(ad)
    a, b = float(np.nanmin(t)) - pad_s, float(np.nanmax(t)) + pad_s
    series = cc.room(a, b, host=host, names=names)
    out = {"_t_shot": t}
    for name, s in series.items():
        out[name] = s.at(t, method=method, max_gap_s=max_gap_s)
    if attach:
        ad.climate = out
        ad.climate_series = series
    return out


def series_for_run(ad, item, client: ClimateClient | None = None, host: str | None = None,
                   pad_s: float = 600.0) -> ClimateSeries:
    """The full one-minute series of one sensor across a run's window."""
    cc = client if client is not None else ClimateClient(host=host or "K")
    a, b = run_window(ad, pad_s)
    return cc.history(item, a, b, host=host)


__all__ = ["climate_for_run", "run_start_time", "run_window", "series_for_run", "shot_times"]
