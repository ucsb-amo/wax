"""Dataclasses describing dashboard panels.

A panel is the unit of extensibility for the dashboard. The user adds a new
server or client by appending a single ``ServerSpec`` / ``ClientSpec`` literal
to the appropriate registry file.

There are three concepts in total - intentionally minimal:

* :class:`PanelSpec` - common base.
* :class:`ServerSpec` - panel that supervises a subprocess.
* :class:`ClientSpec` - panel that embeds an existing control GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional


# Placement: "dock" gets its own dock in ``default_dock_area``; "tab" is
# tabified with every other "tab" panel in the same area sharing ``tab_group``.
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
        "dock" gives the panel its own dock in the area, split from the
        others; "tab" stacks it with the other "tab" panels of the same
        ``tab_group`` in that area.  Heavy multi-pane GUIs should be "dock";
        small status GUIs that are glanced at occasionally can share a tab
        stack.
    tab_group:
        When placement="tab", panels sharing a ``tab_group`` end up as siblings
        in one tab stack.  Default "main".
    preferred_min_size:
        Soft minimum size the framework uses to size the dock.  Embedded
        widgets MUST NOT set their own setMinimumSize - the framework owns
        sizing so compact panels can stay compact.
    icon:
        Single-grapheme emoji shown before the panel title (and used as the
        window icon when the panel is popped out).  ``None`` renders no icon.
    warm_imports:
        Module names the dashboard imports on a background thread right
        after the window is shown, so the heavy part of the body factory
        (pyqtgraph, matplotlib, a camera SDK, ...) is already in
        ``sys.modules`` when the factory runs on the main thread.  Widget
        construction still happens on the main thread; only the import work
        moves.  List only modules whose import creates no Qt widgets.
    realize_eagerly:
        Build the body at startup even if the panel is hidden or behind a
        tab.  Default False: bodies are built when the panel first becomes
        visible.  Set True for in-process servers whose body *is* the
        server.
    """

    id: str
    label: str
    body_factory: Optional[Callable[[], Any]] = None
    default_dock_area: DockArea = "right"
    default_visible: bool = True
    default_placement: Placement = "dock"
    tab_group: str = "main"
    preferred_min_size: tuple[int, int] = (320, 200)
    icon: Optional[str] = None
    warm_imports: list[str] = field(default_factory=list)
    realize_eagerly: bool = False


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
        Zero-argument callable returning a TCP client for this server.  The
        dashboard constructs it on a background thread (discovery can take
        seconds) and uses it for three things: the snapshot poller that
        drives the header conn badge and COM pill (``client.get_snapshot()``
        returning a dict, with a ``"com"`` key shaped like
        ``SerialSnapshot.as_dict()`` when the server owns a serial port), the
        graceful shutdown request at dashboard close
        (``client.request_shutdown()``), and the Servers menu.  Optional.
    server_id:
        Discovery beacon id the server advertises (``NetServer`` /
        ``WaxxServer`` id).  When set, the supervisor checks the beacon
        registry before spawning and marks the panel EXTERNAL if another
        instance is already advertising - so a server someone launched from
        a ``.bat`` is never double-started.
    com_label:
        If non-None, the panel header shows a :class:`ComStatusButton` with
        this label (typically the COM port name), driven by the ``"com"``
        key of the snapshot.
    snapshot_host / snapshot_port:
        Legacy port-probe external check.  Prefer ``server_id``.
    graceful_stop_timeout_s:
        How long the close path waits for the server to exit after a
        ``request_shutdown()`` before hard-killing it.  Default 1.5 s.
    restart_on_crash:
        If True, the supervisor restarts the process when it exits non-zero
        with bounded exponential backoff (5 attempts within 60 s, then FAILED).
    hidden_panel:
        If True, no dock panel is created - only the supervisor runs.
    requires_data_dir:
        If True (default), the supervisor refuses to spawn this server when
        the shared data directory is unreachable.
    """

    server_cmd: list[str] = field(default_factory=list)
    cwd: Optional[str] = None
    env_extra: dict[str, str] = field(default_factory=dict)
    client_factory: Optional[Callable[[], Any]] = None
    server_id: Optional[str] = None
    com_label: Optional[str] = None
    snapshot_host: Optional[str] = None
    snapshot_port: Optional[int] = None
    graceful_stop_timeout_s: float = 1.5
    restart_on_crash: bool = False
    hidden_panel: bool = False
    requires_data_dir: bool = True


@dataclass
class ClientSpec(PanelSpec):
    """Panel that embeds a client GUI (no subprocess supervision)."""

    pass


__all__ = ["PanelSpec", "ServerSpec", "ClientSpec", "Placement", "DockArea"]
