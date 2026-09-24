"""Lab climate data from the Zabbix server (Vertiv Watchdog sensors).

>>> from waxa.climate import ClimateClient, climate_for_run
>>> cc = ClimateClient()                         # guest, host "K", temps in degC
>>> cc.snapshot()                                # latest reading of every K sensor
>>> s = cc.history("Machine Table", "2026-09-23", "2026-09-24")
>>> clim = climate_for_run(ad)                   # per-shot readings for a run

Command line: ``python -m waxa.climate --help``.

Read-only.  Requires the Broida VPN.  See ``client.py`` for units.
"""
from waxa.climate.zabbix import DEFAULT_URL, ZabbixAPI, ZabbixError
from waxa.climate.client import (
    DEFAULT_HOST,
    ClimateClient,
    ClimateItem,
    ClimateSeries,
    f_to_c,
    to_datetime64,
    to_local,
    to_unix,
)
from waxa.climate.attach import (
    climate_for_run,
    run_start_time,
    run_window,
    series_for_run,
    shot_times,
)

__all__ = [
    "DEFAULT_HOST", "DEFAULT_URL", "ZabbixAPI", "ZabbixError",
    "ClimateClient", "ClimateItem", "ClimateSeries",
    "f_to_c", "to_datetime64", "to_local", "to_unix",
    "climate_for_run", "run_start_time", "run_window", "series_for_run", "shot_times",
]
