"""Outdoor weather from NWS station observations (api.weather.gov).

Companion to :mod:`waxa.climate` (which reads the lab's own Zabbix sensors):
this module pulls *outdoor* observations from the nearest National Weather
Service station -- default ``KSBA`` (Santa Barbara airport, ~3 km from campus,
5-minute cadence) -- and samples them at a run's shot times.

Read-only HTTP GET against the public NWS API; no API key. Units are what the
API reports (SI-ish): temperature / dew point in deg C, relative humidity in %,
pressure in Pa, wind speed in km/h.

Typical use::

    from waxa.weather import weather_for_run
    wx = weather_for_run(ad)              # {field: array (*xvardims,)}
    wx["temperature"], wx["relativeHumidity"]

For many runs, fetch once and reuse::

    from waxa.weather import fetch_observations, at_times
    obs = fetch_observations(limit=250)   # ~most recent 21 h at KSBA
    temp = at_times(t_unix, obs)["temperature"]
"""

from __future__ import annotations

import datetime
import json
import urllib.request

import numpy as np

DEFAULT_STATION = "KSBA"
_API = "https://api.weather.gov/stations/{station}/observations?limit={limit}"
_USER_AGENT = "waxa.weather"

#: Observation fields extracted from each report (API name -> unit note).
FIELDS = {
    "temperature": "deg C",
    "dewpoint": "deg C",
    "relativeHumidity": "%",
    "barometricPressure": "Pa",
    "windSpeed": "km/h",
}

#: Shots further than this from the nearest observation come back NaN.
#: KSBA reports every ~5 min; the default tolerates one missed report.
DEFAULT_MAX_GAP_S = 900.0


def fetch_observations(station: str = DEFAULT_STATION, limit: int = 250,
                       timeout: float = 15.0) -> dict[str, np.ndarray]:
    """The station's most recent *limit* observations, oldest first.

    Returns ``{"t": unix seconds, <field>: values}`` with missing values as
    NaN. ``limit=250`` covers roughly the last 21 hours at KSBA's 5-minute
    cadence (the API caps a single page at 500).
    """
    url = _API.format(station=station, limit=int(limit))
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        feats = json.load(resp)["features"]
    rows = []
    for ft in feats:
        p = ft["properties"]
        ts = datetime.datetime.fromisoformat(p["timestamp"]).timestamp()
        vals = []
        for k in FIELDS:
            v = (p.get(k) or {}).get("value")
            vals.append(np.nan if v is None else float(v))
        rows.append((ts, *vals))
    if not rows:
        raise RuntimeError(f"no observations returned for station {station!r}")
    arr = np.array(sorted(rows), float)
    out = {"t": arr[:, 0]}
    for i, k in enumerate(FIELDS):
        out[k] = arr[:, 1 + i]
    return out


def at_times(t, obs: dict[str, np.ndarray] | None = None,
             station: str = DEFAULT_STATION, limit: int = 250,
             max_gap_s: float = DEFAULT_MAX_GAP_S) -> dict[str, np.ndarray]:
    """Each observation field linearly interpolated at unix times *t*.

    Times outside the fetched window, or further than *max_gap_s* from the
    nearest observation, come back NaN -- never extrapolated. Pass a dict from
    :func:`fetch_observations` as *obs* to avoid re-fetching per run.
    """
    t = np.asarray(t, float)
    if obs is None:
        obs = fetch_observations(station=station, limit=limit)
    ep = obs["t"]
    idx = np.searchsorted(ep, t).clip(1, len(ep) - 1)
    gap = np.minimum(np.abs(t - ep[idx - 1]), np.abs(ep[idx] - t))
    bad = ~np.isfinite(t) | (gap > max_gap_s)
    out = {}
    for k in FIELDS:
        m = np.isfinite(obs[k])
        if m.sum() < 2:
            out[k] = np.full(t.shape, np.nan)
            continue
        v = np.interp(t, ep[m], obs[k][m], left=np.nan, right=np.nan)
        v[bad] = np.nan
        out[k] = v
    return out


def weather_for_run(ad, obs: dict[str, np.ndarray] | None = None,
                    station: str = DEFAULT_STATION, limit: int = 250,
                    max_gap_s: float = DEFAULT_MAX_GAP_S) -> dict[str, np.ndarray]:
    """Outdoor observations sampled at every shot of a run (or vault).

    Shot times come from :func:`waxa.climate.shot_times`. Returns
    ``{field: array (*xvardims,)}`` plus ``"_t_shot"`` (the times used).
    Note the single-page API fetch only reaches ~21 h back at KSBA; older
    runs come back all-NaN rather than silently wrong.
    """
    from waxa.climate.attach import shot_times
    t = shot_times(ad)
    out = at_times(t, obs=obs, station=station, limit=limit, max_gap_s=max_gap_s)
    out["_t_shot"] = t
    return out
