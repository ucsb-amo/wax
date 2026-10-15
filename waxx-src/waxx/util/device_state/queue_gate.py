"""PLACEHOLDER -- builder E's version replaces this file at merge (queue phase 2).

The experiment side (builder B, waxx Expt.claim) and the ARTIQ hook (builder
C) call the functions below; their signatures are the contract of BUILD_SPEC
section 3. This placeholder implements that contract with trivial behaviour
and NO network protocol: it reads WAXX_QUEUE_WARM, keeps the registered claim,
and treats GO as given at the first ``ensure_go()``. It exists only so the
experiment side imports and its tests run on its own branch.

Every function is a no-op / trivially true when the process was not
warm-launched, so cold runs, direct artiq_run, the monitor and the run loop
are unaffected.
"""

import os
import sys
import time

EXIT_WARM_ABORTED = 78
WARM_ENV = "WAXX_QUEUE_WARM"
WARM_LOST_S = 600.

_state = {"read": False, "token": None, "go": False, "go_info": {},
          "claim": None, "claimed": False, "hello": False, "opened": []}


def warm_token():
    """WAXX_QUEUE_WARM, read and popped from os.environ once (children never
    inherit it); cached for the process. None when not warm-launched."""
    if not _state["read"]:
        _state["read"] = True
        tok = os.environ.pop(WARM_ENV, None)
        _state["token"] = tok or None
    return _state["token"]


def is_warm() -> bool:
    return warm_token() is not None


def go_passed() -> bool:
    return (not is_warm()) or bool(_state["go"])


def go_info() -> dict:
    return dict(_state["go_info"]) if (is_warm() and _state["go"]) else {}


def hello(expt: str = "") -> None:
    """Placeholder: records that hello ran (E's version tells the server and
    installs the read-open audit hook)."""
    if is_warm():
        _state["hello"] = True


def register_claim(fn) -> None:
    """The experiment's claim (Expt.claim); last one wins."""
    _state["claim"] = fn


def ensure_go() -> None:
    """Placeholder: GO is given at once (E's version polls the monitor server
    and may raise SystemExit(EXIT_WARM_ABORTED)). Idempotent."""
    if not is_warm() or _state["go"]:
        return
    _state["go"] = True
    _state["go_info"] = {"job": os.environ.get("WAXX_QUEUE_JOB", ""),
                         "reset_count": None, "given_at": time.time()}


def run_claim() -> None:
    """Call the registered claim once (no-op if none / done)."""
    fn = _state["claim"]
    if fn is None or _state["claimed"]:
        return
    _state["claimed"] = True
    fn()


def build_manifest(roots=None) -> list:
    """[[path, mtime_ns, size], ...] of every sys.modules file (placeholder:
    no roots filter, no audit hook)."""
    out, seen = [], set()
    for m in list(sys.modules.values()):
        p = getattr(m, "__file__", None)
        if not p or p in seen:
            continue
        seen.add(p)
        try:
            st = os.stat(p)
            out.append([p, st.st_mtime_ns, st.st_size])
        except OSError:
            pass
    return out


def _reset_for_tests():
    """Tests only: forget everything (the placeholder's state is per process)."""
    _state.update({"read": False, "token": None, "go": False, "go_info": {},
                   "claim": None, "claimed": False, "hello": False, "opened": []})
