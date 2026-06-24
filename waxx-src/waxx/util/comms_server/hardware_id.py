"""Hardware-scoped identity for the monitor server/client.

The monitor server controls a specific piece of experiment hardware identified
by ``core_addr`` (the core device IP) in the ARTIQ device database.  The path to
that device database is given by the ``db`` environment variable (the same value
passed to ``artiq_master --device-db %db%``).

By scoping the discovery ``server_id`` to the host octet of ``core_addr`` (e.g.
``"192.168.1.86"`` -> ``"monitor:86"``) we get two properties:

1. A second monitor server started for the *same* hardware can be detected (its
   beacon already advertises the same id) and refused, avoiding two servers
   competing for signals / ports / broadcasts.
2. A client identifies the server matching *its own* db, so when two monitor
   servers (different hardware) run on the subnet, each client connects to the
   correct one.

This module is intentionally stdlib-only — ``waxx`` must not import ``kexp``.
"""

from __future__ import annotations

import logging
import os
import runpy

logger = logging.getLogger(__name__)

#: Base discovery id used when no hardware id can be resolved (back-compat).
MONITOR_BASE_ID: str = "monitor"


def get_core_addr() -> str | None:
    """Return ``core_addr`` from the device database pointed to by env var ``db``.

    Loads the ``device_db.py`` file at ``$db`` and reads its ``core_addr``
    variable, falling back to ``device_db["core"]["arguments"]["host"]``.
    Returns ``None`` if the env var is unset, the file is missing, or it cannot
    be parsed — callers treat ``None`` as "no hardware id available".
    """
    db_path = os.getenv("db")
    if not db_path:
        logger.debug("[hardware_id] env var 'db' not set; no core_addr available")
        return None
    if not os.path.isfile(db_path):
        logger.warning("[hardware_id] device db path '%s' does not exist", db_path)
        return None

    try:
        namespace = runpy.run_path(db_path)
    except Exception as exc:  # noqa: BLE001 — any failure -> no id, never raise
        logger.warning("[hardware_id] could not load device db '%s': %s", db_path, exc)
        return None

    core_addr = namespace.get("core_addr")
    if not core_addr:
        try:
            core_addr = namespace["device_db"]["core"]["arguments"]["host"]
        except Exception:  # noqa: BLE001
            core_addr = None

    if not isinstance(core_addr, str) or not core_addr:
        logger.warning("[hardware_id] no usable core_addr found in '%s'", db_path)
        return None
    return core_addr


def get_hardware_id() -> str | None:
    """Return the host octet of ``core_addr`` (the hardware's IP on the subnet).

    e.g. ``"192.168.1.86"`` -> ``"86"``.  Returns ``None`` when ``core_addr``
    cannot be resolved or the host octet is not a valid IPv4 octet (0-255).

    The octet is validated to be purely numeric and in range so it is always
    safe to embed in a filesystem path (``device_state_config_<id>.json``) and a
    network discovery id — it can never carry path-separator or traversal
    characters even if ``core_addr`` is malformed.
    """
    core_addr = get_core_addr()
    if not core_addr:
        return None
    last_octet = core_addr.rsplit(".", 1)[-1].strip()
    if not (last_octet.isdigit() and 0 <= int(last_octet) <= 255):
        logger.warning(
            "[hardware_id] core_addr '%s' has no valid numeric host octet; "
            "falling back to unscoped id", core_addr,
        )
        return None
    return last_octet


def scoped_server_id(base_id: str) -> str:
    """Return ``base_id`` scoped to this machine's hardware id.

    ``"<base_id>:<host-octet>"`` when the hardware id is resolvable (so each
    piece of hardware advertises a distinct id on the subnet), else the plain
    ``base_id`` (preserves single-hardware setups with no ``db`` set).
    """
    hw = get_hardware_id()
    return f"{base_id}:{hw}" if hw else base_id


def monitor_server_id() -> str:
    """Return the hardware-scoped monitor discovery id.

    ``"monitor:<host-octet>"`` when the hardware id is resolvable, else the plain
    base id ``"monitor"`` (preserves single-hardware setups with no ``db`` set).
    """
    return scoped_server_id(MONITOR_BASE_ID)


def _matches_base(server_id: str, base_id: str) -> bool:
    """True if ``server_id`` is exactly ``base_id`` or a scoped ``base_id:<hw>``.

    Excludes unrelated siblings that merely share a prefix (e.g. base
    ``"live_od"`` must not match ``"live_od_broadcast:86"``).
    """
    return server_id == base_id or server_id.startswith(base_id + ":")


def resolve_scoped_server_id(base_id: str, collect_for: float = 1.5) -> str:
    """Return the scoped ``server_id`` a client on this machine should connect to.

    When the hardware id is resolvable, returns ``"<base_id>:<host-octet>"`` so
    the client only ever talks to the server controlling *its own* hardware.

    When no hardware id is available (e.g. env var ``db`` unset on a viewer-only
    machine), falls back to discovering servers on the subnet and returns the
    unique match for ``base_id``.  Raises ``RuntimeError`` if zero or more than
    one matching server is found (ambiguous — cannot choose safely).
    """
    hw = get_hardware_id()
    if hw is not None:
        return f"{base_id}:{hw}"

    from waxx.util.comms_server.waxx_client import discover_prefix  # noqa: PLC0415
    servers = discover_prefix(base_id, collect_for=collect_for)
    matches = sorted(sid for sid in servers if _matches_base(sid, base_id))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(
            f"[hardware_id] No '{base_id}' server discovered on the subnet and "
            f"no hardware id available (env var 'db' unset). Start the server, or "
            f"set 'db' to this branch's device_db.py."
        )
    raise RuntimeError(
        f"[hardware_id] Multiple '{base_id}' servers found on the subnet "
        f"({', '.join(matches)}) but this machine has no hardware id to pick the "
        f"right one. Set env var 'db' to this branch's device_db.py."
    )


def discover_scoped(base_id: str, timeout: float = 3.0) -> tuple[str, int] | None:
    """Discover the hardware-scoped server for ``base_id``; return ``(ip, port)``.

    With a resolvable hardware id, discovers the exact ``"<base_id>:<host-octet>"``
    beacon.  Without one, returns the unique ``base_id`` match on the subnet.
    Returns ``None`` (never raises) when not found or when the match is
    ambiguous, so polling callers (e.g. the remote viewer) can simply retry.
    """
    from waxx.util.comms_server.waxx_client import discover, discover_prefix  # noqa: PLC0415
    hw = get_hardware_id()
    if hw is not None:
        return discover(f"{base_id}:{hw}", timeout=timeout)

    servers = discover_prefix(base_id, collect_for=min(timeout, 1.5))
    matches = {sid: addr for sid, addr in servers.items() if _matches_base(sid, base_id)}
    if len(matches) == 1:
        return next(iter(matches.values()))
    return None
