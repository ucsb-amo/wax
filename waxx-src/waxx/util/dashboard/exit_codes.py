"""Exit codes a supervised server and its supervisor agree on.

:data:`EXIT_RESTART`: the server exits on purpose and asks to be started again
(the monitor server's ``{"type": "server", "action": "restart"}`` request).
:class:`~waxx.util.dashboard.server_supervisor.ServerSupervisor` starts it again
after ``INITIAL_RESTART_DELAY_S`` whatever its ``restart_on_crash`` -- it is not a
crash: no CRASHED state, not counted toward the crash restart-storm guard --
provided the process ran for at least ``RESTART_REQUEST_MIN_UPTIME_S`` (10 s).
Exit code 3 is also what a C runtime ``abort()``, Qt's ``qFatal`` or a
platform-plugin failure gives on Windows; sooner than that it is taken as a
crash under the crash policy.  Requested restarts have their own guard: more
than ``MAX_REQUESTED_RESTARTS`` (3) within ``REQUESTED_RESTART_WINDOW_S``
(5 min) -> FAILED.  Any other non-zero exit is a crash and keeps the crash
policy.

A server run without the supervisor (the GUI launchers, a terminal) is not
restarted by anything: exit code 3 is then just an exit.
"""

#: The process asks its supervisor to start it again.
EXIT_RESTART = 3
