"""PLACEHOLDER -- builder E's version replaces this file at merge.

Only here so that ``waxx.util.artiq.core`` (builder C, queue phase 2) imports and its
tests run on the ``jep/qp2-core`` branch. The signatures are exactly BUILD_SPEC.md §3;
the behaviour is trivial: the token is read and popped from ``WAXX_QUEUE_WARM`` as
specified, but there is no server conversation -- ``ensure_go()`` gives GO at once.
Do not build on anything here beyond the signatures.
"""
import os

EXIT_WARM_ABORTED = 78

_UNREAD = object()
_token = _UNREAD
_go = None            # dict once GO was "received"
_claim = None
_claim_done = False


def warm_token():
    """Read and pop WAXX_QUEUE_WARM once; cached for the process."""
    global _token
    if _token is _UNREAD:
        _token = os.environ.pop("WAXX_QUEUE_WARM", None) or None
    return _token


def is_warm():
    return warm_token() is not None


def go_passed():
    return (not is_warm()) or _go is not None


def go_info():
    return dict(_go) if (is_warm() and _go is not None) else {}


def hello(expt=""):
    pass


def register_claim(fn):
    global _claim
    _claim = fn


def ensure_go():
    global _go
    if is_warm() and _go is None:
        import time
        _go = {"job": os.environ.get("WAXX_QUEUE_JOB", ""), "reset_count": None,
               "given_at": time.time()}


def run_claim():
    global _claim_done
    if _claim is None or _claim_done:
        return
    _claim_done = True
    _claim()


def build_manifest(roots=None):
    return []
