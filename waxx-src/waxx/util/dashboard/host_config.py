"""Resolve which host this dashboard is running on, and what should autostart.

Used by both the server and client dashboards to:

* find the lab-subnet IP address of this PC (the one that matches the
  ``192.168.1.x`` range used by the lab network, not loopback / Wi-Fi)
* look up the autostart server list and per-host layout overrides in the
  *experiment-supplied* config modules registered via :func:`configure`
* support a ``--host-ip`` command-line override for testing on a laptop

This is the generic library implementation.  The experiment package wires it
up once (typically in its dashboard app ``main()``)::

    from waxx.util.dashboard import host_config
    host_config.configure(
        hosts_module='kexp.util.dashboard.dashboard_hosts',
        layout_module='kexp.util.dashboard.dashboard_layout',
        subnet_prefix='192.168.1.',
    )
"""

from __future__ import annotations

import importlib
import logging
import socket
from typing import Optional

_LOG = logging.getLogger("waxx.dashboard.host")

# Defaults; overridable via configure().
_LAB_SUBNET_PREFIX = "192.168.1."
_HOSTS_MODULE: Optional[str] = None
_LAYOUT_MODULE: Optional[str] = None


def configure(
    *,
    hosts_module: Optional[str] = None,
    layout_module: Optional[str] = None,
    subnet_prefix: Optional[str] = None,
) -> None:
    """One-time configuration; safe to call multiple times."""
    global _HOSTS_MODULE, _LAYOUT_MODULE, _LAB_SUBNET_PREFIX
    if hosts_module is not None:
        _HOSTS_MODULE = hosts_module
    if layout_module is not None:
        _LAYOUT_MODULE = layout_module
    if subnet_prefix is not None:
        _LAB_SUBNET_PREFIX = subnet_prefix


def resolve_host_ip(cli_override: Optional[str] = None) -> Optional[str]:
    """Return the lab-subnet IP of this PC, or ``None`` if undeterminable.

    ``cli_override`` (from ``--host-ip``) wins if provided.  Otherwise we try:

    1. enumerate ``socket.getaddrinfo(hostname)`` and pick a subnet-prefix-matching IP;
    2. as a fallback, the classic 'connect to a public IP, read the local
       endpoint' trick - but only to a lab address so we don't send any
       actual packets across the internet.
    """
    if cli_override:
        return cli_override.strip()

    # Strategy 1: getaddrinfo for our own hostname.
    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET)
        for _family, _type, _proto, _canon, sockaddr in infos:
            ip = sockaddr[0]
            if ip.startswith(_LAB_SUBNET_PREFIX):
                return ip
    except Exception as exc:
        _LOG.debug("getaddrinfo failed: %r", exc)

    # Strategy 2: UDP-connect trick (does not actually send for UDP-connect).
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.2)
            s.connect((_LAB_SUBNET_PREFIX + "1", 1))
            ip = s.getsockname()[0]
            if ip.startswith(_LAB_SUBNET_PREFIX):
                return ip
    except Exception as exc:
        _LOG.debug("UDP-connect trick failed: %r", exc)

    return None


def hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def load_autostart_set(host_ip: Optional[str]) -> set[str]:
    """Return the set of server ids that should autostart on this host.

    Looks up the host IP in ``<hosts_module>.HOST_AUTOSTART_SERVERS``
    (where *hosts_module* is set via :func:`configure`).  Missing host
    -> empty set (manual start only).  Missing/unimportable module is
    logged at WARNING but treated as empty.
    """
    if not host_ip or not _HOSTS_MODULE:
        return set()
    try:
        mod = importlib.import_module(_HOSTS_MODULE)
    except Exception as exc:
        _LOG.warning("hosts module %s not importable: %r", _HOSTS_MODULE, exc)
        return set()
    table = getattr(mod, "HOST_AUTOSTART_SERVERS", {})
    return set(table.get(host_ip, []))


def load_layout_overrides(host_ip: Optional[str]) -> dict:
    """Return per-host layout override dict, or ``{}`` if none."""
    if not host_ip or not _LAYOUT_MODULE:
        return {}
    try:
        mod = importlib.import_module(_LAYOUT_MODULE)
    except Exception as exc:
        _LOG.debug("layout module %s not importable: %r", _LAYOUT_MODULE, exc)
        return {}
    overrides = getattr(mod, "HOST_LAYOUT_OVERRIDES", {})
    return dict(overrides.get(host_ip, {}))


__all__ = [
    "configure",
    "resolve_host_ip",
    "hostname",
    "load_autostart_set",
    "load_layout_overrides",
]
