"""Run a clean-up when the console liveOD runs in is closed or interrupted.

liveOD is started from a console (kexp: _bat/live_od.bat), not by the dashboard.
Closing that console window kills the process without running ``atexit``
handlers or Qt's ``aboutToQuit``, so nothing closes the cameras: the Andor keeps
its shutter state and SDK handle, a Basler stays claimed. On Windows a console
control handler (``SetConsoleCtrlHandler``) is told first, and the system waits
for it -- about 5 s for a close (CTRL_CLOSE_EVENT), longer for logoff/shutdown --
before it ends the process. ``install(callback)`` runs ``callback()`` there,
bounded by ``budget_s`` so the handler always returns inside that window.

Ctrl+C and Ctrl+Break do *not* run the callback: they are swallowed with a
WARNING and liveOD goes on acquiring. Ctrl+C is what someone presses to copy a
log line out of the console (with nothing selected, it is sent as an
interrupt), and a run must not end because of it. Before this module existed
Ctrl+C only printed a KeyboardInterrupt traceback; returning "handled" here
also keeps Python from raising KeyboardInterrupt into whatever Qt slot happens
to be running. To quit, close the liveOD window (or the console: that still
runs the callback). Off Windows this is a no-op.

Stdlib only.
"""

import ctypes
import logging
import sys
import threading

# the liveOD logger's child (waxx.util.live_od.log.get_logger("console")), named
# directly so that this module stays stdlib-only
logger = logging.getLogger("waxx.live_od.console")

CTRL_C_EVENT = 0
CTRL_BREAK_EVENT = 1
CTRL_CLOSE_EVENT = 2
CTRL_LOGOFF_EVENT = 5
CTRL_SHUTDOWN_EVENT = 6

EVENT_NAMES = {
    CTRL_C_EVENT: "Ctrl+C",
    CTRL_BREAK_EVENT: "Ctrl+Break",
    CTRL_CLOSE_EVENT: "console window closed",
    CTRL_LOGOFF_EVENT: "user logging off",
    CTRL_SHUTDOWN_EVENT: "system shutting down",
}

# Interrupts: swallowed, never a shutdown (see the module docstring).
INTERRUPT_EVENTS = (CTRL_C_EVENT, CTRL_BREAK_EVENT)

# Windows gives a CTRL_CLOSE_EVENT handler about 5 s; stay inside it.
DEFAULT_BUDGET_S = 4.0

# The installed ctypes callback. Kept referenced for the life of the process:
# if it were garbage-collected, Windows would call freed memory.
_installed = None


def _run_bounded(callback, event_name: str, budget_s: float) -> bool:
    """Run ``callback(event_name)`` on a helper thread, wait at most ``budget_s``.
    True if it finished in time."""
    done = threading.Event()

    def target():
        try:
            callback(event_name)
        except Exception as exc:
            logger.warning(f"console clean-up after '{event_name}' failed: {exc}")
        finally:
            done.set()

    threading.Thread(target=target, name="liveOD-console-cleanup", daemon=True).start()
    finished = done.wait(budget_s)
    if not finished:
        logger.warning(f"console clean-up after '{event_name}' did not finish within "
                       f"{budget_s:g} s; the process may end before it does")
    return finished


def _on_console_event(callback, ctrl_type: int, budget_s: float) -> int:
    """What the installed handler does with one console event; returns what it
    returns to Windows (1: handled). Ctrl+C / Ctrl+Break: a WARNING, nothing
    else. Close, logoff, shutdown: ``callback(event_name)``, bounded."""
    ctrl_type = int(ctrl_type)
    name = EVENT_NAMES.get(ctrl_type, f"console event {ctrl_type}")
    if ctrl_type in INTERRUPT_EVENTS:
        # handled, so no KeyboardInterrupt either
        logger.warning(f"{name} ignored -- close the liveOD window to quit")
        return 1
    _run_bounded(callback, name, budget_s)
    return 1        # handled: for a close, Windows ends the process once this returns


def install(callback, budget_s: float = DEFAULT_BUDGET_S) -> bool:
    """On Windows, call ``callback(event_name)`` (bounded by ``budget_s``) when the
    console is closed, or on logoff or shutdown. Ctrl+C / Ctrl+Break are ignored
    (logged; the callback is not called). Returns True if the handler was
    installed. Installing again replaces the callback."""
    global _installed
    if sys.platform != "win32":
        return False
    try:
        HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)

        def handler(ctrl_type):
            return _on_console_event(callback, ctrl_type, budget_s)

        c_handler = HandlerRoutine(handler)
        kernel32 = ctypes.windll.kernel32
        if _installed is not None:
            kernel32.SetConsoleCtrlHandler(_installed, 0)
        if not kernel32.SetConsoleCtrlHandler(c_handler, 1):
            logger.warning("could not install the console close handler "
                           f"(SetConsoleCtrlHandler failed, error {ctypes.GetLastError()}); "
                           "closing the console will not close the cameras")
            return False
        _installed = c_handler
        return True
    except Exception as exc:
        logger.warning(f"could not install the console close handler: {exc}; "
                       f"closing the console will not close the cameras")
        return False


def uninstall() -> None:
    """Remove the handler ``install`` put in (tests; a no-op if there is none)."""
    global _installed
    if _installed is None or sys.platform != "win32":
        _installed = None
        return
    try:
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_installed, 0)
    except Exception:
        pass
    _installed = None
