"""Append-only journal of what the monitor server did to the hardware.

One JSON object per line, one file per day (``ops_journal_YYYY-MM-DD.jsonl``)
in the directory the lab configures.  Records: composite ops (submitted,
refused, finished -- with their argument values), channel updates (with who
sent them), end-of-run states, runs starting, monitor state changes, trust
changes, scenes and watchdogs.  The Device Control GUI's changes window and
the pre-run stamp read it back through the server.

Files are only ever appended to: never rewritten, rotated or deleted by this
code.  If the directory is unwritable the server keeps working and the
journal lives in memory only (the last :data:`MEMORY_KEPT` records), which it
says once in the log.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque

log = logging.getLogger(__name__)

MEMORY_KEPT = 4000


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


class OpJournal:
    def __init__(self, directory: str | None = None, clock=time.time):
        self.directory = directory
        self._clock = clock
        self._lock = threading.Lock()
        self._memory: deque = deque(maxlen=MEMORY_KEPT)
        self._warned = False

    def path_for(self, t: float | None = None) -> str | None:
        if not self.directory:
            return None
        day = time.strftime("%Y-%m-%d", time.localtime(self._clock() if t is None else t))
        return os.path.join(self.directory, f"ops_journal_{day}.jsonl")

    def record(self, kind: str, **fields) -> dict:
        ts = self._clock()
        entry = {"ts": ts, "t": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
                 "kind": kind}
        entry.update({k: _jsonable(v) for k, v in fields.items()})
        with self._lock:
            self._memory.append(entry)
            path = self.path_for(ts)
            if path is not None:
                try:
                    os.makedirs(self.directory, exist_ok=True)
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry) + "\n")
                except OSError as e:
                    if not self._warned:
                        self._warned = True
                        log.warning("Ops journal: cannot append to %s (%s); keeping the "
                                    "journal in memory only until the server restarts.",
                                    path, e)
        return entry

    def tail(self, n: int = 200, kinds=None) -> list[dict]:
        with self._lock:
            entries = list(self._memory)
        if kinds:
            entries = [e for e in entries if e.get("kind") in kinds]
        return entries[-int(n):]

    def since(self, marker_kinds=("run_end",)) -> list[dict]:
        """Records after the most recent marker (e.g. the last end-of-run
        state), oldest first; everything kept when there is no marker."""
        with self._lock:
            entries = list(self._memory)
        for i in range(len(entries) - 1, -1, -1):
            if entries[i].get("kind") in marker_kinds:
                return entries[i + 1:]
        return entries


def _who(e: dict) -> str:
    parts = [p for p in (e.get("operator"), e.get("client")) if p]
    return (" by " + "@".join(parts)) if parts else ""


def _args(args) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    return "(" + ", ".join(f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}"
                           for k, v in args.items()) + ")"


def describe_entry(e: dict) -> str:
    """One line for a journal record, for people."""
    kind = str(e.get("kind", "?"))
    t = str(e.get("t", ""))[11:] or "?"
    if kind == "op_submit":
        text = f"{e.get('op')}{_args(e.get('args'))} queued #{e.get('seq')}{_who(e)}"
        if e.get("origin") and e.get("origin") != "gui":
            text += f" [{e['origin']}]"
    elif kind == "op_refused":
        text = f"{e.get('op')}{_args(e.get('args'))} REFUSED{_who(e)}: {e.get('msg')}"
    elif kind == "op_result":
        text = f"{e.get('op')} #{e.get('seq')}: {e.get('text')}"
        if isinstance(e.get("elapsed"), (int, float)):
            text += f" ({e['elapsed']:.2f} s)"
    elif kind == "update":
        text = f"{e.get('device')} {_args(e.get('changes'))}"
        if e.get("origin"):
            text += f" [{e['origin']}]"
    elif kind == "run_pending":
        text = f"run {e.get('run_id')} ({e.get('expt') or '?'}) starting"
    elif kind == "run_end":
        text = f"run {e.get('run_id')} ({e.get('expt') or '?'}) "
        if e.get("aborted"):
            text += f"aborted ({e['aborted']}): state at the abort received"
            if e.get("trusted") is False:
                text += ", UNTRUSTED"
        else:
            text += "end state received"
    elif kind == "run_pending_cleared":
        text = f"run {e.get('run_id')} fence lifted: {e.get('why')}"
    elif kind == "trust":
        text = ("trusted" if e.get("trusted") else "UNTRUSTED") + f": {e.get('reason')}"
    elif kind == "monitor_state":
        text = f"monitor {e.get('state')} ({e.get('sub_state') or ''})"
    elif kind == "connection":
        text = f"{e.get('key')} {e.get('state')}"
        if e.get("detail"):
            text += f": {e['detail']}"
    elif kind == "connection_request":
        text = f"{e.get('key')} {e.get('action')} requested{_who(e)}"
    elif kind == "connection_refused":
        text = f"{e.get('key')} {e.get('action')} REFUSED{_who(e)}: {e.get('msg')}"
    elif kind.startswith("scene_"):
        text = f"{e.get('scene')} #{e.get('id')} {kind[6:]}"
        if e.get("text"):
            text += f": {e['text']}"
    elif kind.startswith("watchdog_"):
        text = f"{e.get('device')} {kind[9:]}"
        if e.get("text"):
            text += f": {e['text']}"
    elif kind == "slm_reinit_requested":
        text = "SLM reinit sent (machine idle)"
        if e.get("by"):
            text += f", asked for by {e['by']}"
        elif isinstance(e.get("due_for_s"), (int, float)):
            text += f", due for {e['due_for_s'] / 60:.0f} min"
    elif kind == "state_regenerated":
        n = e.get("n_changed")
        text = (f"state file REGENERATED from the device definitions by {e.get('by')}: "
                f"{n} channel(s) changed; backup {e.get('backup') or 'none'}")
    elif kind == "state_regenerate_refused":
        text = f"state file regeneration by {e.get('by')} REFUSED: {e.get('msg')}"
    elif kind == "slm_reinit_manual":
        text = f"SLM reinit asked for by {e.get('by')}"
    elif kind == "slm_reinit_refused":
        text = f"SLM reinit asked for by {e.get('by')} REFUSED: {e.get('msg')}"
    elif kind == "slm_reinit_done":
        text = f"SLM re-initialised in {e.get('t_s')} s, pattern put back: {e.get('pattern')}"
    elif kind == "slm_reinit_failed":
        text = f"SLM reinit FAILED: {e.get('text')}"
    elif kind == "slm_restart_manual":
        text = f"SLM server restart asked for by {e.get('by')}"
    elif kind == "slm_restart_requested":
        text = f"SLM server restart sent (machine idle), asked for by {e.get('by')}"
    elif kind == "slm_restart_refused":
        text = f"SLM server restart asked for by {e.get('by')} REFUSED: {e.get('msg')}"
    elif kind == "slm_restart_done":
        text = (f"SLM server restarted in {e.get('t_s')} s (instance {e.get('instance')}, "
                f"{e.get('pattern_source')})")
        if e.get("slm_ready") is False:
            text += " -- SLM NOT READY"
    elif kind == "slm_restart_failed":
        text = f"SLM server restart FAILED: {e.get('text')}"
    elif kind == "slm_reinit_state":
        text = f"SLM reinit {e.get('state')}"
        if e.get("detail"):
            text += f": {e['detail']}"
    elif kind.startswith("state_reset_"):
        text = f"{e.get('expt') or 'reset'} {kind[12:]}{_who(e)}"
        if e.get("text") or e.get("msg"):
            text += f": {e.get('text') or e.get('msg')}"
    else:
        rest = {k: v for k, v in e.items() if k not in ("ts", "t", "kind")}
        text = json.dumps(rest, default=repr)
    return f"{t}  {kind:<13} {text}"
