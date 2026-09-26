"""What the hardware was said to be at when a run started.

Before a run takes the core, :func:`pre_run_report` asks the monitor server
for the device state, whether it is trusted, and its journal since the last
run ended (every op and channel edit made in between, by whom), and
evaluates the lab's composite devices for hazards (a coil left at current).
The experiment prints the warnings and stores the whole report with its data
(``Expt._extra_file_texts``), so a run can be traced to the state it started
from -- as recorded by the server, which is not a measurement of the hardware
(the report says whether the state was trusted).

Never raises: a run is not stopped by its own bookkeeping.
"""

from __future__ import annotations

import time

from waxx.util.device_state.composite import Context, compact_state, hazards

#: Journal records kept in the stamp (oldest dropped first, and said so).
MAX_JOURNAL = 500


def pre_run_report(monitor, devices=(), params=None, frames=None) -> dict:
    report = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "server": False}
    try:
        state = monitor.server_state()
    except Exception as e:
        state = None
        report["error"] = repr(e)
    if state is None:
        report.setdefault("error", "the monitor server did not answer get_state")
        return report
    config = state.get("config") or {}
    trust = state.get("trust")
    ctx = Context(config, params, frames, host_state=state.get("composite_state") or {},
                  trust=trust)
    try:
        found = [{"device": d.key, "title": d.title, "text": text}
                 for d, text in hazards(devices, ctx)]
    except Exception as e:
        found = [{"device": "?", "title": "hazard check", "text": f"failed: {e!r}"}]
    try:
        journal = monitor.journal_since_last_run()
    except Exception:
        journal = None
    report.update({
        "server": True,
        "version": state.get("version"),
        "trust": trust,
        "hazards": found,
        "device_state": compact_state(config),
        "journal_since_last_run": (journal or [])[-MAX_JOURNAL:],
        "journal_available": journal is not None,
        "journal_truncated": bool(journal and len(journal) > MAX_JOURNAL),
    })
    return report


def report_warnings(report: dict) -> list[str]:
    """The lines a person must see before the run: hazards, untrusted state,
    and why the check could not be made."""
    lines = []
    if not report.get("server"):
        lines.append("could not read the device state from the monitor server "
                     f"({report.get('error')}); nothing was checked before this run.")
        return lines
    trust = report.get("trust") or {}
    if trust.get("trusted") is False:
        lines.append(f"the device state is UNTRUSTED ({trust.get('reason')}): what follows "
                     "is the state file, not necessarily the hardware.")
    for h in report.get("hazards") or []:
        lines.append(f"{h['title']} is {h['text']} (state file) as this run starts.")
    return lines
