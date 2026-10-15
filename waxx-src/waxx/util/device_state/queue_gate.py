"""The experiment process's side of the run queue's GO gate (queue phase 2,
warm start).

A *warm* launch is a queued job whose process the run queue starts while the
job before it is still running: imports, ``prepare()`` and the kernel compile
happen early, then the process waits here until the queue says GO -- after
the previous job's process has exited and the machine is free -- and only
then claims anything (liveOD, the monitor fence, hardware, the core device).
The queue marks such a launch with ``WAXX_QUEUE_WARM=<warm token>`` in the
process's environment (next to ``WAXX_LAUNCHER=kq`` and
``WAXX_QUEUE_JOB=<id>:<token>``).

Every function here is a no-op -- or trivially true -- when the process was
NOT warm-launched, so direct ``artiq_run``, cold queue launches, the monitor
and the run loop are unaffected.

* :func:`warm_token` reads ``WAXX_QUEUE_WARM`` once and POPS it from
  ``os.environ``: a child process of the experiment never inherits it.
* The freshness manifest records each file AS IT WAS LOADED: in a warm
  process (``WAXX_QUEUE_WARM`` present -- checked, not popped) a
  ``sys.addaudithook`` on ``open`` is installed when this module is imported
  (``import waxx`` imports it first thing in a warm process), and stats
  (never reads) every file under the queue roots at its first open for
  reading -- a module's source when its ``.pyc`` is opened; the
  ``sys.modules`` files already loaded when the hook goes in are stat-ed
  then.  Residual window: a module imported before the hook (the experiment
  file's own first imports, ``artiq``, ...) is recorded at the hook's
  install, not at its load; one the hook never saw is stat-ed at "ready".
* :func:`hello` (``Expt.__init__``) tells the queue "warming" (pid, creation
  time, job, warm token) (and installs the hook if the import did not).
* :func:`ensure_go` (the Core hook, after compile) re-stats every manifest
  file at "ready": one that changed since it was loaded stands the process
  down (``EXIT_WARM_ABORTED``; the queue warms it again) naming the file --
  the process would otherwise run code older than what is on disk.  Else it
  sends "ready" once with the load-time manifest (:func:`build_manifest`),
  then polls every :data:`POLL_S` s.
  GO: the process raises its own priority back to NORMAL (the queue starts
  warm launches at BELOW_NORMAL), records :func:`go_info` and returns.
  Abort (the queue discarded this warm launch): it prints why and raises
  ``SystemExit(EXIT_WARM_ABORTED)`` -- a normal exit, atexit handlers run,
  nothing was claimed.  An error or no reply: it keeps waiting; after
  :data:`WARM_LOST_S` s with no answer from the queue it exits as aborted.
* :func:`register_claim` / :func:`run_claim`: the experiment's claim
  (``Expt.claim``), called once after GO.

The requests go to the monitor server (which runs the queue) as
``{"type": "run_queue", "action": "warm", "stage": "warming" | "ready" |
"poll", "job", "warm", "pid", "started", "manifest"?}`` through the
experiment's usual :class:`~waxx.util.comms_server.comm_client.MonitorClient`
(made lazily: the discovery listener starts only in a warm process).

Import-light: the standard library only at import time.  Machine-agnostic.
"""

from __future__ import annotations

import os
import sys
import threading
import time

#: The environment variable of a warm launch (set ONLY by the queue's warm
#: launches; read and popped at first read).
WARM_ENV = "WAXX_QUEUE_WARM"
#: The queue's job of this process, "<id>:<token>" (set on every queue launch;
#: not popped: the experiment sends it to liveOD at its run start).
JOB_ENV = "WAXX_QUEUE_JOB"
#: The folders the queue runs files from (os.pathsep-separated), as
#: ``run_queue.default_roots``.
ROOTS_ENV = "WAXX_RUN_QUEUE_ROOTS"
#: The exit code of a warm process told to stand down before GO (nothing was
#: claimed).  A normal ``sys.exit``: atexit handlers run.
EXIT_WARM_ABORTED = 78
#: Seconds between GO polls.
POLL_S = 0.2
#: With no answer from the queue for this long, the warm process exits as
#: aborted (the queue re-adopts a warm process after its own restart well
#: within this).
WARM_LOST_S = 600.0
#: Per-request socket timeout (s).
REQUEST_TIMEOUT_S = 2.0
#: At most this many files go in a manifest.
MANIFEST_MAX = 20000

_NORMAL_PRIORITY_CLASS = 0x00000020

#: Clock and sleep (tests replace them).
_monotonic = time.monotonic
_sleep = time.sleep

_lock = threading.RLock()
_state: dict = {
    "read": False,       # WAXX_QUEUE_WARM read (and popped) yet
    "token": None,       # the warm token, or None (not warm)
    "job": "",           # WAXX_QUEUE_JOB at first read
    "go": None,          # go_info once GO came
    "hello": False,
    "ready_sent": False,
    "claim": None,
    "claimed": False,
    "hook": False,       # the audit hook is installed (it cannot be removed)
    "recording": False,  # the hook records opened files
    "filter": None,      # _ManifestFilter of the recording
    "transport": None,   # the MonitorClient (or a test's stand-in)
    "expt": "",
}
#: normalised path -> (path, mtime_ns, size) at its first open for reading
#: (or at the hook's install, for a module loaded before it)
_loaded: dict = {}
#: normalised paths the filter left out (not stat-ed again)
_skipped: set = set()
#: drive letter -> DRIVE_REMOTE
_remote_drives: dict = {}
_in_hook = threading.local()


# -- identity -----------------------------------------------------------------------------

def warm_token() -> str | None:
    """The warm token from ``WAXX_QUEUE_WARM``, read once and popped from
    ``os.environ`` (children never inherit it); cached for the process.
    None when this process was not warm-launched."""
    with _lock:
        if not _state["read"]:
            _state["read"] = True
            token = os.environ.pop(WARM_ENV, None)
            _state["token"] = (token or "").strip() or None
            _state["job"] = os.environ.get(JOB_ENV, "") or ""
        return _state["token"]


def is_warm() -> bool:
    """This process was warm-launched by the run queue."""
    return warm_token() is not None


def go_passed() -> bool:
    """True when not warm, or once GO has been received."""
    return (not is_warm()) or _state["go"] is not None


def go_info() -> dict:
    """``{}`` when not warm or before GO; after GO ``{"job": "<id>:<token>",
    "reset_count": int | None, "given_at": time.time()}`` (``reset_count``:
    liveOD's Abort count the queue last saw, for the run start's check)."""
    go = _state["go"]
    return dict(go) if go else {}


def _started() -> float | None:
    """This process's creation time (as the queue's process watches read
    it), read once."""
    if "started" not in _state:
        try:
            from waxx.util.device_state.detached import process_started  # noqa: PLC0415
            _state["started"] = process_started()
        except Exception:                             # noqa: BLE001
            _state["started"] = None
    return _state["started"]


# -- the queue link -----------------------------------------------------------------------

def set_transport(transport) -> None:
    """Use ``transport`` (an object with ``request(obj, timeout=,
    attempts=)`` -> reply dict or None) instead of a MonitorClient made by
    discovery.  For tests and tools."""
    with _lock:
        _state["transport"] = transport


def _transport():
    t = _state["transport"]
    if t is not None:
        return t
    try:
        from waxx.util.comms_server.comm_client import MonitorClient  # noqa: PLC0415
        t = MonitorClient(discovery_timeout=3.0)
    except Exception:                                 # noqa: BLE001
        return None                                   # no monitor server found (yet)
    with _lock:
        _state["transport"] = t
    return t


def _ask(stage: str, manifest: list | None = None) -> dict | None:
    """One warm request; the reply dict, or None (no queue / no answer)."""
    t = _transport()
    if t is None:
        return None
    obj = {"type": "run_queue", "action": "warm", "stage": stage, "job": _state["job"],
           "warm": _state["token"], "pid": os.getpid(), "started": _started(),
           "expt": _state["expt"]}
    if manifest is not None:
        obj["manifest"] = manifest
    try:
        reply = t.request(obj, timeout=REQUEST_TIMEOUT_S, attempts=2)
    except Exception:                                 # noqa: BLE001
        return None
    return reply if isinstance(reply, dict) else None


def _say(text: str) -> None:
    try:
        print(f"[queue] {text}", flush=True)
    except Exception:                                 # noqa: BLE001
        pass


def _stand_down(why: str):
    _say(f"warm start stood down before GO (nothing was claimed): {why}")
    raise SystemExit(EXIT_WARM_ABORTED)


# -- the manifest -------------------------------------------------------------------------

def queue_roots() -> list[str]:
    """``$WAXX_RUN_QUEUE_ROOTS`` (os.pathsep-separated), else ``%code%`` and
    ``C:\\lab\\skynet_log`` -- the queue's default roots
    (``run_queue.default_roots`` reads this)."""
    env = os.environ.get(ROOTS_ENV)
    if env:
        return [r for r in env.split(os.pathsep) if r]
    roots = [os.environ.get("code") or "", r"C:\lab\skynet_log"]
    return [r for r in roots if r]


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(os.fsdecode(path)))


def _norm_forms(path) -> set:
    """A folder's normalised path as written and as resolved (an 8.3 short
    name such as the temp folder's and its long name are the same folder)."""
    out = set()
    try:
        out.add(_norm(path))
        out.add(os.path.normcase(os.path.realpath(os.fsdecode(path))))
    except Exception:                                 # noqa: BLE001
        pass
    return {p.rstrip("\\/") for p in out if p}


def is_network_path(path: str) -> bool:
    """A UNC path (``\\\\server\\share``) or a mapped network drive
    (``GetDriveTypeW == DRIVE_REMOTE``; one call per drive letter, cached).
    Such paths are never stat-ed for a manifest: a hung share must not stall
    the warm process's hook or the queue's tick."""
    p = os.fsdecode(path)
    if p[:2] in ("\\\\", "//"):
        return True
    if sys.platform != "win32" or len(p) < 2 or p[1] != ":" or not p[0].isalpha():
        return False
    d = p[0].upper()
    remote = _remote_drives.get(d)
    if remote is None:
        try:
            import ctypes  # noqa: PLC0415
            remote = ctypes.windll.kernel32.GetDriveTypeW(f"{d}:\\") == 4   # DRIVE_REMOTE
        except Exception:                             # noqa: BLE001
            remote = False
        _remote_drives[d] = remote
    return remote


class _ManifestFilter:
    """Which files a manifest holds: under one of the queue roots, and NOT
    under the interpreter's folders (sys.prefix, sys.base_prefix, ...), the
    temp folder, a ``site-packages`` or a ``.venv`` folder -- unless a root
    lies deeper than the excluding folder -- and never on a network path.
    (``%code%`` holds ``.venv``: without this a manifest is mostly the venv,
    re-stat-ed by the queue every few seconds, and a ``uv sync`` during a
    wait would discard warm jobs.)"""

    _DIR_NAMES = ("site-packages", ".venv")

    def __init__(self, roots=None):
        self.roots = set()
        for r in (queue_roots() if roots is None else roots):
            if r:
                self.roots |= _norm_forms(r)
        self.skip = set()
        for p in {sys.prefix, sys.base_prefix, sys.exec_prefix,
                  getattr(sys, "base_exec_prefix", "")}:
            if p:
                self.skip |= _norm_forms(p)
        try:
            import tempfile  # noqa: PLC0415
            self.skip |= _norm_forms(tempfile.gettempdir())
        except Exception:                             # noqa: BLE001
            pass

    @staticmethod
    def _deepest(n: str, prefixes) -> str | None:
        best = None
        for p in prefixes:
            if (n == p or n.startswith(p + os.sep)) and (best is None or len(p) > len(best)):
                best = p
        return best

    def keep(self, n: str) -> bool:
        """``n``: a normalised path."""
        root = self._deepest(n, self.roots)
        if root is None:
            return False
        skip = self._deepest(n, self.skip)
        if skip is not None and len(skip) >= len(root):
            return False
        below = n[len(root):].split(os.sep)
        if any(part in self._DIR_NAMES for part in below):
            return False
        return not is_network_path(n)


def _stat_pair(path: str):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return int(st.st_mtime_ns), int(st.st_size)


def _record_loaded(path: str) -> None:
    """Remember ``path``'s stat at its first open for reading (a ``.pyc``
    in ``__pycache__``: its source's, the file that matters; a sourceless
    ``.pyc``: its own).  Stat only, never a read."""
    n = _norm(path)
    if n in _loaded or n in _skipped:
        return
    flt = _state["filter"]
    if flt is None:
        return
    if n.endswith(".pyc"):
        folder, name = os.path.split(os.path.abspath(path))
        if os.path.basename(folder).lower() == "__pycache__":
            src = os.path.join(os.path.dirname(folder), name.split(".", 1)[0] + ".py")
            if os.path.isfile(src):
                path, n = src, _norm(src)
                if n in _loaded or n in _skipped:
                    return
    if not flt.keep(n):
        _skipped.add(n)
        return
    pair = _stat_pair(path)
    if pair is not None and os.path.isfile(path):
        _loaded.setdefault(n, (os.path.abspath(path),) + pair)


def _audit(event, args) -> None:
    """``sys.addaudithook`` callback: stat each file under the queue roots at
    its first open for reading, while recording.  Never raises (an audit
    hook that raises fails the open), never re-enters itself."""
    if event != "open" or not _state["recording"] or getattr(_in_hook, "on", False):
        return
    _in_hook.on = True
    try:
        path, mode, flags = args[0], args[1], args[2]
        if path is None or isinstance(path, int):
            return
        if isinstance(mode, str):
            if any(c in mode for c in "wax+"):
                return
        elif isinstance(flags, int):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT):
                return
        _record_loaded(os.fsdecode(path))
    except Exception:                                 # noqa: BLE001
        pass
    finally:
        _in_hook.on = False


def _snapshot_modules() -> None:
    """The ``sys.modules`` files loaded before the hook went in: recorded
    now (the residual window -- they may have changed between their load
    and this moment)."""
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f or not isinstance(f, str):
            continue
        try:
            _record_loaded(f)
        except Exception:                             # noqa: BLE001
            continue


def _start_recording() -> None:
    """Install the audit hook (once per process; it cannot be removed) and
    record from now on; on the first start, the modules already loaded are
    stat-ed."""
    with _lock:
        first = not _state["recording"]
        if first:
            _state["filter"] = _ManifestFilter()
            import ctypes  # noqa: F401,PLC0415  (loaded now, never from inside the hook)
        if not _state["hook"]:
            sys.addaudithook(_audit)
            _state["hook"] = True
        _state["recording"] = True
    if first:
        _snapshot_modules()


def build_manifest(roots=None) -> list[list]:
    """``[[path, mtime_ns, size], ...]``, sorted, de-duplicated (by normalised
    path): every ``sys.modules`` file and every file opened for reading that
    :class:`_ManifestFilter` keeps for ``roots`` (default
    :func:`queue_roots`), each with its stat AT LOAD (its first open while
    recording, or the hook's install for a module loaded before it; a module
    the hook never saw is stat-ed now).  Files that cannot be stat-ed are
    left out."""
    flt = _ManifestFilter(roots)
    entries: dict[str, list] = {}
    for n, (path, mtime, size) in list(_loaded.items()):
        if flt.keep(n):
            entries[n] = [path, mtime, size]
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f or not isinstance(f, str):
            continue
        try:
            n = _norm(f)
        except Exception:                             # noqa: BLE001
            continue
        if n in entries or not flt.keep(n):
            continue
        pair = _stat_pair(f)
        if pair is not None and os.path.isfile(f):
            entries[n] = [os.path.abspath(f), pair[0], pair[1]]
    out = [entries[n] for n in sorted(entries)]
    return out[:MANIFEST_MAX]


def changed_since_load(manifest) -> str:
    """The first manifest file whose stat now differs from its load-time
    stat (or that is gone), as text; "" when none."""
    for path, mtime, size in manifest:
        pair = _stat_pair(path)
        if pair is None:
            return f"{path} (gone)"
        if pair != (int(mtime), int(size)):
            return path
    return ""


# -- the gate -----------------------------------------------------------------------------

def hello(expt: str = "") -> None:
    """Best effort (warm processes only; once): make sure the files opened
    for reading are being recorded (the manifest; normally since this
    module's import), and tell the queue "warming".  An
    abort in the reply -- the queue no longer wants this warm launch --
    stands the process down at once (``SystemExit(EXIT_WARM_ABORTED)``;
    nothing has been claimed this early)."""
    if not is_warm():
        return
    with _lock:
        if _state["hello"]:
            return
        _state["hello"] = True
        _state["expt"] = str(expt or "")
    _start_recording()
    reply = _ask("warming")
    if reply and reply.get("status") == "ok" and reply.get("abort"):
        _stand_down(str(reply.get("abort")))


def register_claim(fn) -> None:
    """The experiment's claim (``Expt.claim``), called by :func:`run_claim`
    once after GO.  The last one registered wins."""
    with _lock:
        _state["claim"] = fn


def run_claim() -> None:
    """Call the registered claim once (no-op when none is registered or it
    has run)."""
    with _lock:
        fn = _state["claim"]
        if fn is None or _state["claimed"]:
            return
        _state["claimed"] = True
    fn()


def _raise_own_priority() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes  # noqa: PLC0415
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        return bool(k32.SetPriorityClass(k32.GetCurrentProcess(), _NORMAL_PRIORITY_CLASS))
    except Exception:                                 # noqa: BLE001
        return False


def ensure_go() -> None:
    """Warm and not passed: send "ready" with the manifest, then poll the
    queue every :data:`POLL_S` s until GO (return) or abort
    (``SystemExit(EXIT_WARM_ABORTED)``).  An error or no reply keeps it
    waiting; :data:`WARM_LOST_S` s with no answer at all exits as aborted.
    Not warm, or GO already received: returns at once.  Idempotent."""
    if go_passed():
        return
    if not _state["hello"]:
        hello()
    need_ready = not _state["ready_sent"]
    manifest = None
    last_answer = _monotonic()
    said_waiting = ""
    while True:
        if need_ready:
            if manifest is None:
                manifest = build_manifest()
                with _lock:
                    _state["recording"] = False       # prepare is over: the manifest is made
                changed = changed_since_load(manifest)
                if changed:
                    # the code or data on disk is newer than what this
                    # process loaded: it must not run; the queue warms anew
                    _stand_down(f"a file changed on disk after this process loaded it: "
                                f"{changed}")
            reply = _ask("ready", manifest)
        else:
            reply = _ask("poll")
        now = _monotonic()
        if not reply or reply.get("status") != "ok":
            if now - last_answer > WARM_LOST_S:
                _stand_down(f"no answer from the run queue for {WARM_LOST_S:.0f} s")
            _sleep(POLL_S)
            continue
        last_answer = now
        if need_ready:
            need_ready = False
            with _lock:
                _state["ready_sent"] = True
            _say(f"warm start ready ({len(manifest)} files in its manifest); waiting for GO")
        if reply.get("abort"):
            _stand_down(str(reply.get("abort")))
        if reply.get("go"):
            raised = _raise_own_priority()
            with _lock:
                _state["go"] = {"job": _state["job"], "reset_count": reply.get("reset_count"),
                                "given_at": time.time()}
            _say("GO from the run queue" + ("" if raised or sys.platform != "win32"
                                            else " (its priority could not be raised)"))
            return
        if reply.get("resend_ready"):
            need_ready = True
        why = str(reply.get("why") or "")
        if why and why != said_waiting:
            said_waiting = why
            _say(f"waiting for GO: {why}")
        _sleep(POLL_S)


def _reset_for_tests() -> None:
    """Forget everything (tests only; the audit hook stays installed but
    stops recording)."""
    with _lock:
        _state.update(read=False, token=None, job="", go=None, hello=False, ready_sent=False,
                      claim=None, claimed=False, recording=False, transport=None, expt="",
                      filter=None)
        _state.pop("started", None)
        _loaded.clear()
        _skipped.clear()
        _remote_drives.clear()


# A warm process records from its first import of this module (as early as
# possible: see the module docstring).  Only the presence is checked here --
# warm_token() pops the variable.
if os.environ.get(WARM_ENV):
    try:
        _start_recording()
    except Exception:                                 # noqa: BLE001
        pass
