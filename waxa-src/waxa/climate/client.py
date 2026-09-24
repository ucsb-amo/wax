"""Lab climate data (Vertiv Watchdog sensors) read from the Zabbix server.

The Watchdogs sample temperature, humidity, airflow and dew point once a
minute and push them into Zabbix over SNMP.  Zabbix keeps the raw
one-minute **history** for 90 days and hourly min/avg/max **trends** for a
year.  Everything here is read-only.

Hosts as named in Zabbix: ``"K"``, ``"Li"``, ``"Sr"``, ``"GL"`` (the general
lab watchdog) and ``"Banana Stand"`` (the data server's own temperature).

Units
-----
Zabbix stores the Watchdog temperatures in **°F**.  By default this module
converts temperature series to **°C** (``temperature_unit="C"``) and says
so: every :class:`ClimateSeries` carries both ``units`` (what the values
are in) and ``source_units`` (what Zabbix stored).  Pass
``temperature_unit="F"`` to get the stored values untouched.  Humidity is
``%RH``, airflow is the Watchdog's unitless 0-100 index, times are unix
seconds (UTC epoch) exactly as Zabbix records them.

Example
-------
>>> from waxa.climate import ClimateClient
>>> cc = ClimateClient()                       # guest login, host "K"
>>> s = cc.history("Machine Table", "2026-09-23", "2026-09-24")
>>> s.t[:3], s.v[:3], s.units
>>> room = cc.room("2026-09-24 08:00", "2026-09-24 18:00")   # every K sensor
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from waxa.climate.zabbix import DEFAULT_URL, ZabbixAPI

# Zabbix item value types -> which history table to read.
_VALUE_TYPE_FLOAT = 0
_VALUE_TYPE_UINT = 3
_NUMERIC_VALUE_TYPES = (_VALUE_TYPE_FLOAT, _VALUE_TYPE_UINT)

DEFAULT_HOST = "K"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def to_unix(t) -> float:
    """Coerce *t* to unix seconds.

    Accepts a number (already unix seconds), a ``datetime`` (naive = local
    time), a ``numpy.datetime64``, or a string ``"YYYY-MM-DD"`` /
    ``"YYYY-MM-DD HH:MM"`` / ``"YYYY-MM-DD HH:MM:SS"`` in local time.
    """
    if isinstance(t, (int, float, np.integer, np.floating)):
        return float(t)
    if isinstance(t, np.datetime64):
        return float(t.astype("datetime64[ns]").astype("int64")) * 1e-9
    if isinstance(t, _dt.datetime):
        return t.timestamp()
    if isinstance(t, _dt.date):
        return _dt.datetime(t.year, t.month, t.day).timestamp()
    if isinstance(t, str):
        s = t.strip().replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d_%H-%M-%S", "%Y-%m-%d"):
            try:
                return _dt.datetime.strptime(s, fmt).timestamp()
            except ValueError:
                continue
        raise ValueError(f"unrecognised time string {t!r}")
    raise TypeError(f"cannot convert {type(t).__name__} to unix seconds")


def to_datetime64(t_unix) -> np.ndarray:
    """Unix seconds -> ``datetime64[ns]`` in **UTC** (matplotlib plots these as
    given; use ``to_local`` for wall-clock labels)."""
    t = np.asarray(t_unix, dtype=float)
    return (t * 1e9).astype("int64").astype("datetime64[ns]")


def to_local(t_unix) -> list[_dt.datetime]:
    """Unix seconds -> naive local ``datetime`` objects (for axis labels)."""
    return [_dt.datetime.fromtimestamp(float(x)) for x in np.atleast_1d(t_unix)]


def f_to_c(v):
    return (np.asarray(v, dtype=float) - 32.0) * 5.0 / 9.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClimateItem:
    """One Zabbix item (= one sensor channel)."""
    itemid: str
    hostid: str
    host: str
    name: str
    key: str
    value_type: int
    units: str
    delay: str = ""

    @property
    def is_temperature(self) -> bool:
        return self.units in ("F", "C")

    @property
    def is_numeric(self) -> bool:
        return self.value_type in _NUMERIC_VALUE_TYPES


@dataclass
class ClimateSeries:
    """A sampled sensor channel: ``t`` unix seconds, ``v`` values.

    ``units`` is what ``v`` is in; ``source_units`` what Zabbix stored.  When
    the two differ, the conversion is the plain °F -> °C formula and nothing
    else has been done to the data.
    """
    item: ClimateItem
    t: np.ndarray
    v: np.ndarray
    units: str
    source_units: str
    kind: str = "history"          # "history" (raw 1-min) or "trend" (hourly avg)
    extra: dict = field(default_factory=dict)   # trend: {"min":..., "max":..., "num":...}

    @property
    def name(self) -> str:
        return self.item.name

    @property
    def host(self) -> str:
        return self.item.host

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def datetimes(self) -> np.ndarray:
        """``datetime64[ns]`` (UTC) for plotting."""
        return to_datetime64(self.t)

    @property
    def local_datetimes(self) -> list[_dt.datetime]:
        return to_local(self.t)

    def to_celsius(self) -> "ClimateSeries":
        if self.units == "C":
            return self
        if self.units != "F":
            raise ValueError(f"{self.name}: units {self.units!r} are not a temperature")
        return ClimateSeries(self.item, self.t, f_to_c(self.v), "C", self.source_units,
                             self.kind, {k: f_to_c(x) if k in ("min", "max") else x
                                         for k, x in self.extra.items()})

    def to_fahrenheit(self) -> "ClimateSeries":
        if self.units == "F":
            return self
        if self.units != "C":
            raise ValueError(f"{self.name}: units {self.units!r} are not a temperature")
        return ClimateSeries(self.item, self.t, self.v * 9.0 / 5.0 + 32.0, "F",
                             self.source_units, self.kind,
                             {k: (x * 9.0 / 5.0 + 32.0) if k in ("min", "max") else x
                              for k, x in self.extra.items()})

    def at(self, t_query, method: str = "nearest", max_gap_s: float | None = 180.0) -> np.ndarray:
        """Sample the series at unix times *t_query* (any shape).

        ``method="nearest"`` takes the closest sample; ``"interp"`` linearly
        interpolates between neighbours.  Any query further than
        ``max_gap_s`` from the nearest sample (a sensor outage, or a query
        outside the fetched window) comes back **NaN** rather than a stale
        value.  ``max_gap_s=None`` disables that guard.
        """
        tq = np.asarray(t_query, dtype=float)
        out = np.full(tq.shape, np.nan)
        if self.t.size == 0:
            return out
        flat = tq.ravel()
        idx = np.searchsorted(self.t, flat)
        lo = np.clip(idx - 1, 0, self.t.size - 1)
        hi = np.clip(idx, 0, self.t.size - 1)
        d_lo = np.abs(flat - self.t[lo])
        d_hi = np.abs(self.t[hi] - flat)
        nearest = np.where(d_lo <= d_hi, lo, hi)
        gap = np.minimum(d_lo, d_hi)
        if method == "nearest":
            vals = self.v[nearest]
        elif method == "interp":
            vals = np.interp(flat, self.t, self.v)
        else:
            raise ValueError(f"method must be 'nearest' or 'interp', got {method!r}")
        if max_gap_s is not None:
            vals = np.where(gap <= max_gap_s, vals, np.nan)
        out.ravel()[:] = vals
        return out

    def window(self, t_from, t_to) -> "ClimateSeries":
        """Sub-series with ``t_from <= t <= t_to`` (any :func:`to_unix` input)."""
        a, b = to_unix(t_from), to_unix(t_to)
        m = (self.t >= a) & (self.t <= b)
        return ClimateSeries(self.item, self.t[m], self.v[m], self.units, self.source_units,
                             self.kind, {k: np.asarray(x)[m] for k, x in self.extra.items()})

    def summary(self) -> dict:
        """mean / std / min / max / n over the series, NaNs ignored."""
        v = self.v[np.isfinite(self.v)]
        if v.size == 0:
            return {"n": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
        return {"n": int(v.size), "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max())}

    def __repr__(self) -> str:
        if self.t.size:
            span = f"{_dt.datetime.fromtimestamp(self.t[0]):%Y-%m-%d %H:%M} .. " \
                   f"{_dt.datetime.fromtimestamp(self.t[-1]):%Y-%m-%d %H:%M}"
        else:
            span = "empty"
        return (f"ClimateSeries({self.host}/{self.name!r}, {self.kind}, "
                f"n={self.t.size}, units={self.units!r}, {span})")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ClimateClient:
    """Read climate sensors from the lab Zabbix server.

    Parameters
    ----------
    host : str
        Default Zabbix host name (``"K"``).  Every method takes ``host=`` to
        override it.
    temperature_unit : ``"C"`` or ``"F"``
        Unit temperature series are returned in.  Zabbix stores °F.
    api : ZabbixAPI, optional
        Supply your own (other URL / credentials); default is guest at
        :data:`DEFAULT_URL`.
    """

    def __init__(self, host: str = DEFAULT_HOST, temperature_unit: str = "C",
                 api: ZabbixAPI | None = None, url: str = DEFAULT_URL, timeout: float = 15.0):
        if temperature_unit not in ("C", "F"):
            raise ValueError("temperature_unit must be 'C' or 'F'")
        self.host = host
        self.temperature_unit = temperature_unit
        self.api = api if api is not None else ZabbixAPI(url=url, timeout=timeout)
        self._hosts: dict[str, str] | None = None        # name -> hostid
        self._items: dict[str, list[ClimateItem]] = {}   # hostid -> items

    # -- discovery -------------------------------------------------------

    def hosts(self) -> dict[str, str]:
        """``{host name: hostid}`` for every host guest can see."""
        if self._hosts is None:
            rows = self.api.call("host.get", {"output": ["hostid", "name"]})
            self._hosts = {r["name"]: r["hostid"] for r in rows}
        return dict(self._hosts)

    def _hostid(self, host: str | None) -> tuple[str, str]:
        name = host or self.host
        hosts = self.hosts()
        if name in hosts:
            return name, hosts[name]
        # case-insensitive fallback
        for n, hid in hosts.items():
            if n.lower() == name.lower():
                return n, hid
        raise KeyError(f"no Zabbix host {name!r}; known: {sorted(hosts)}")

    def items(self, host: str | None = None, numeric_only: bool = True) -> list[ClimateItem]:
        """Every monitored item on *host* (sensor channels)."""
        name, hid = self._hostid(host)
        if hid not in self._items:
            rows = self.api.call("item.get", {
                "output": ["itemid", "hostid", "name", "key_", "value_type", "units", "delay"],
                "hostids": [hid], "monitored": True,
            })
            self._items[hid] = [
                ClimateItem(r["itemid"], r["hostid"], name, r["name"], r["key_"],
                            int(r["value_type"]), r.get("units", ""), r.get("delay", ""))
                for r in rows
            ]
        items = self._items[hid]
        return [it for it in items if it.is_numeric] if numeric_only else list(items)

    def item(self, name_or_key: str | ClimateItem, host: str | None = None) -> ClimateItem:
        """Look an item up by its display name or key (case-insensitive)."""
        if isinstance(name_or_key, ClimateItem):
            return name_or_key
        want = name_or_key.strip().lower()
        items = self.items(host, numeric_only=False)
        for it in items:
            if it.name.lower() == want or it.key.lower() == want:
                return it
        # unique substring match on the name as a convenience
        cands = [it for it in items if want in it.name.lower()]
        if len(cands) == 1:
            return cands[0]
        hostname = host or self.host
        names = [it.name for it in items]
        raise KeyError(f"no item {name_or_key!r} on host {hostname!r} "
                       f"({'ambiguous' if cands else 'no match'}); have {names}")

    # -- values ----------------------------------------------------------

    def snapshot(self, host: str | None = None) -> dict[str, tuple[float, float]]:
        """Latest value of every numeric item: ``{name: (value, unix_time)}``.

        Temperatures follow ``temperature_unit``.
        """
        name, hid = self._hostid(host)
        rows = self.api.call("item.get", {
            "output": ["itemid", "name", "units", "value_type", "lastvalue", "lastclock"],
            "hostids": [hid], "monitored": True,
        })
        out = {}
        for r in rows:
            if int(r["value_type"]) not in _NUMERIC_VALUE_TYPES:
                continue
            v = float(r["lastvalue"])
            if r.get("units") == "F" and self.temperature_unit == "C":
                v = float(f_to_c(v))
            out[r["name"]] = (v, float(r["lastclock"]))
        return out

    def history(self, item, t_from, t_to=None, host: str | None = None,
                chunk_s: float = 7 * 86400.0) -> ClimateSeries:
        """Raw one-minute samples of *item* between *t_from* and *t_to*.

        Zabbix keeps 90 days of raw history; ask for older data with
        :meth:`trends`.  Long windows are fetched in ``chunk_s`` pieces.
        """
        it = self.item(item, host)
        a = to_unix(t_from)
        b = to_unix(t_to) if t_to is not None else _dt.datetime.now().timestamp()
        if b < a:
            raise ValueError("t_to is before t_from")
        ts, vs = [], []
        lo = a
        while lo <= b:
            hi = min(lo + chunk_s, b)
            rows = self.api.call("history.get", {
                "output": "extend", "history": it.value_type, "itemids": [it.itemid],
                "time_from": int(np.floor(lo)), "time_till": int(np.ceil(hi)),
                "sortfield": "clock", "sortorder": "ASC",
            })
            ts.extend(float(r["clock"]) + float(r.get("ns", 0)) * 1e-9 for r in rows)
            vs.extend(float(r["value"]) for r in rows)
            lo = hi + 1e-3
            if hi >= b:
                break
        t = np.asarray(ts, dtype=float)
        v = np.asarray(vs, dtype=float)
        if t.size:
            order = np.argsort(t, kind="stable")
            t, v = t[order], v[order]
            keep = np.concatenate([[True], np.diff(t) > 0])   # drop exact duplicates at chunk seams
            t, v = t[keep], v[keep]
        return self._finish(it, t, v, "history", {})

    def trends(self, item, t_from, t_to=None, host: str | None = None) -> ClimateSeries:
        """Hourly trends of *item*: ``v`` is the hourly **mean**; ``extra``
        holds ``min``, ``max`` and ``num`` (samples per hour).  Kept a year.
        """
        it = self.item(item, host)
        a = to_unix(t_from)
        b = to_unix(t_to) if t_to is not None else _dt.datetime.now().timestamp()
        rows = self.api.call("trend.get", {
            "output": "extend", "itemids": [it.itemid],
            "time_from": int(np.floor(a)), "time_till": int(np.ceil(b)),
        })
        rows = sorted(rows, key=lambda r: int(r["clock"]))
        t = np.asarray([float(r["clock"]) for r in rows])
        v = np.asarray([float(r["value_avg"]) for r in rows])
        extra = {"min": np.asarray([float(r["value_min"]) for r in rows]),
                 "max": np.asarray([float(r["value_max"]) for r in rows]),
                 "num": np.asarray([int(r["num"]) for r in rows])}
        return self._finish(it, t, v, "trend", extra)

    def room(self, t_from, t_to=None, host: str | None = None,
             names: Iterable[str] | None = None, kind: str = "history") -> dict[str, ClimateSeries]:
        """Every numeric sensor on *host* (or just *names*) over a window,
        as ``{item name: ClimateSeries}``."""
        fetch = self.history if kind == "history" else self.trends
        items = [self.item(n, host) for n in names] if names is not None else self.items(host)
        return {it.name: fetch(it, t_from, t_to, host) for it in items}

    # -- internals -------------------------------------------------------

    def _finish(self, it: ClimateItem, t, v, kind, extra) -> ClimateSeries:
        s = ClimateSeries(it, t, v, it.units, it.units, kind, extra)
        if it.units == "F" and self.temperature_unit == "C":
            s = s.to_celsius()
        return s

    def close(self) -> None:
        self.api.logout()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
        return False


__all__ = [
    "DEFAULT_HOST", "DEFAULT_URL", "ClimateClient", "ClimateItem", "ClimateSeries",
    "f_to_c", "to_datetime64", "to_local", "to_unix",
]
