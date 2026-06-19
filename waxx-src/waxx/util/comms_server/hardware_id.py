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
    cannot be resolved.
    """
    core_addr = get_core_addr()
    if not core_addr:
        return None
    last_octet = core_addr.rsplit(".", 1)[-1].strip()
    if not last_octet:
        return None
    return last_octet


def monitor_server_id() -> str:
    """Return the hardware-scoped monitor discovery id.

    ``"monitor:<host-octet>"`` when the hardware id is resolvable, else the plain
    base id ``"monitor"`` (preserves single-hardware setups with no ``db`` set).
    """
    hw = get_hardware_id()
    return f"{MONITOR_BASE_ID}:{hw}" if hw else MONITOR_BASE_ID
