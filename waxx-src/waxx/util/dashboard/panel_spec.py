"""Dataclasses describing dashboard panels.

A panel is the unit of extensibility for the dashboard. The user adds a new
server or client by appending a single ``ServerSpec`` / ``ClientSpec`` literal
to the appropriate registry file.

There are three concepts in total - intentionally minimal:

* :class:`PanelSpec` - common base.
* :class:`ServerSpec` - panel that supervises a subprocess.
* :class:`ClientSpec` - panel that embeds an existing control GUI.

See ``README.md`` in this directory for copy-pasteable examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional


# Placement: "dock" floats into ``default_dock_area``; "tab" goes into the
# dashboard central QTabWidget grouped by ``tab_group``.
Placement = Literal["dock", "tab"]

# Dock areas accepted in committed config (mapped to ``Qt.DockWidgetArea`` at
# dashboard construction time so this module stays Qt-free for testing).
DockArea = Literal["left", "right", "top", "bottom"]


@dataclass
class PanelSpec:
    """Common base for every panel in a dashboard.

    Attributes
    ----------
    id:
        Stable unique identifier (used for layout persistence + host-config
        autostart lookup + log tagging).  Convention: snake_case, short.
    label:
        Human-readable title for the dock title bar.
    body_factory:
        Zero-argument callable that returns the panel body ``QWidget``.
        MUST not block on I/O - factories run on the Qt main thread and the
        dashboard's transparency-first lifecycle requires the window to be
        on screen before any factory is invoked.  If a factory needs server
        data, it returns a placeholder widget that fills itself async.
    default_dock_area:
        Where to dock when placement="dock".
    default_visible:
        If False, panel is created but hidden by default (user shows from menu).
    default_placement:
        "dock" or "tab".  Heavy multi-pane GUIs (DataBrowser) should default
        to "tab"; small status/control GUIs default to "dock".
    tab_group:
        When placement="tab", panels sharing a ``tab_group`` end up as siblings
        in the same QTabWidget.  Default "main".
    preferred_min_size:
        Soft minimum size the framework uses to size the dock.  Embedded
        widgets MUST NOT set their own setMinimumSize - the framework owns
        sizing so compact panels can stay compact.
    """

    id: str
    label: str
    body_factory: Optional[Callable[[], Any]] = None
    default_dock_area: DockArea = "right"
    default_visible: bool = True
    default_placement: Placement = "dock"
    tab_group: str = "main"
    preferred_min_size: tuple[int, int] = (320, 200)


@dataclass
class ServerSpec(PanelSpec):
    """Panel that supervises a server subprocess.

    Additional attributes
    ---------------------
    server_cmd:
        Command-line passed to ``QProcess`` (e.g. ``[sys.executable, "/.../als_server.py"]``).
    cwd:
        Optional working directory for the subprocess.
    env_extra:
        Optional dict merged into the subprocess environment.
    client_factory:
        Zero-argument callable returning a TCP client used by the snapshot
        poller (drives the conn badge + GenericServerStatusPanel).  Optional.
    com_label:
        If non-None, the panel header shows a :class:`ComStatusButton` with
        this label (typically the COM port name).  The server's snapshot
        must include a "com" key compatible with ``SerialSnapshot.as_dict()``.
    snapshot_host:
        Host the snapshot client should connect to.  Optional - if None, the
        client_factory is responsible for knowing where to connect.
    snapshot_port:
        Port for snapshot client (used by port-in-use detection).
    graceful_stop_timeout_s:
        How long to wait for the subprocess to exit after terminate() before
        kill().  Default 5 s.
    restart_on_crash:
        If True, the supervisor restarts the process when it exits non-zero
        with bounded exponential backoff (5 attempts within 60 s, then FAILED).
    """

    server_cmd: list[str] = field(default_factory=list)
    cwd: Optional[str] = None
    env_extra: dict[str, str] = field(default_factory=dict)
    client_factory: Optional[Callable[[], Any]] = None
    com_label: Optional[str] = None
    snapshot_host: Optional[str] = None
    snapshot_port: Optional[int] = None
    graceful_stop_timeout_s: float = 5.0
    restart_on_crash: bool = False
    hidden_panel: bool = False
    """If True, no dock panel is created — only the supervisor runs.

    Useful for headless servers whose state is already shown by another
    panel (e.g. the device-state monitor is surfaced inside the Device
    Control GUI, so a second dock tile is redundant)."""


@dataclass
class ClientSpec(PanelSpec):
    """Panel that embeds a client GUI (no subprocess supervision)."""

    pass


__all__ = ["PanelSpec", "ServerSpec", "ClientSpec", "Placement", "DockArea"]
