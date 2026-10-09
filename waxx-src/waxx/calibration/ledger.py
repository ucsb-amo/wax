"""The calibration ledger: every emitted result, and every apply / revert.

Layout under an injected ``ledger_dir`` (there is no default location):

- ``calibrations.jsonl`` -- append-only, one JSON object per line, each with an
  ``event``: ``"emit"`` (a CalResult), ``"apply"`` or ``"revert"`` (a
  WritebackReport plus key / run id). Appends happen under
  ``calibrations.jsonl.lock``.
- ``<key>/<run_id>.json`` -- the result of that run for that key, as last
  emitted, with the apply state kept up to date.
- ``<key>/<run_id>.png`` -- the analysis's figure, when it made one.

Nothing in the ledger is ever deleted or rewritten except the per-run JSON's
apply state; the jsonl is the history.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Iterator, Optional

from waxx.calibration._lock import file_lock, replace_bytes
from waxx.calibration.record import CalResult, _jsonable

_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _check_key(key):
    if not isinstance(key, str) or not _KEY.fullmatch(key):
        raise ValueError(f"calibration key {key!r} is not a params attribute name")
    return key


class Ledger:
    JSONL = "calibrations.jsonl"

    def __init__(self, ledger_dir):
        if not ledger_dir:
            raise ValueError("Ledger needs a directory (none is assumed)")
        self.dir = Path(ledger_dir)

    @property
    def jsonl(self) -> Path:
        return self.dir / self.JSONL

    def record_path(self, key, run_id) -> Path:
        return self.dir / _check_key(key) / f"{int(run_id)}.json"

    def figure_path(self, key, run_id) -> Path:
        return self.dir / _check_key(key) / f"{int(run_id)}.png"

    # ---- writing ---------------------------------------------------------------
    def append_event(self, event: dict):
        self.dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_jsonable(event), sort_keys=False)
        with file_lock(self.jsonl):
            with open(self.jsonl, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")

    def write_record(self, result: CalResult):
        """The emit: an ``emit`` line in the jsonl first (the history), then the
        per-run JSON (replaced if this run was emitted before)."""
        path = self.record_path(result.key, result.run_id)
        self.append_event({"event": "emit", **result.to_dict()})
        path.parent.mkdir(parents=True, exist_ok=True)
        replace_bytes(path, (result.to_json(indent=1) + "\n").encode("utf-8"))

    def note_writeback(self, report, run_id: Optional[int] = None, by: str = ""):
        """An ``apply`` / ``revert`` line; a successful apply also updates the
        run's per-run JSON."""
        ev = {"event": report.action, "key": report.key, "run_id": run_id, "by": by,
              "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
              **report.to_dict()}
        self.append_event(ev)
        if report.action == "apply" and report.ok and report.written and run_id is not None:
            path = self.record_path(report.key, run_id)
            if path.exists():
                r = CalResult.from_json(path.read_text(encoding="utf-8"))
                r.applied, r.applied_file, r.applied_line = True, report.file, report.line_no
                replace_bytes(path, (r.to_json(indent=1) + "\n").encode("utf-8"))

    # ---- reading ---------------------------------------------------------------
    def events(self, key: Optional[str] = None) -> Iterator[dict]:
        if not self.jsonl.exists():
            return
        with open(self.jsonl, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    ev = {"event": "unreadable", "line": n, "text": line[:200]}
                if key is None or ev.get("key") == key:
                    yield ev

    def load(self, key, run_id) -> CalResult:
        path = self.record_path(key, run_id)
        if path.exists():
            return CalResult.from_json(path.read_text(encoding="utf-8"))
        last = None
        for ev in self.events(key):
            if ev.get("event") == "emit" and ev.get("run_id") == int(run_id):
                last = ev
        if last is None:
            raise LookupError(f"no ledger record for {key} from run {run_id} in {self.dir}")
        return CalResult.from_dict(last)

    def history(self, key, n: Optional[int] = 10) -> list:
        """The last ``n`` emitted results for ``key`` (oldest first), with
        ``applied`` set from later apply events."""
        out, by_run = [], {}
        for ev in self.events(key):
            if ev.get("event") == "emit":
                r = CalResult.from_dict(ev)
                out.append(r)
                by_run[r.run_id] = r
            elif ev.get("event") == "apply" and ev.get("ok") and ev.get("written"):
                r = by_run.get(ev.get("run_id"))
                if r is not None:
                    r.applied, r.applied_file, r.applied_line = True, ev.get("file"), ev.get("line_no")
        return out if n is None else out[-int(n):]
