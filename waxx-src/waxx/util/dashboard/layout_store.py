"""Named dashboard layouts, kept in QSettings beside the working layout.

The dashboard keeps two kinds of layout per host:

* the **working layout** (``geometry`` / ``state`` / ``popped`` keys, written
  by :class:`DashboardMainWindow` about a second after any change), which is
  what the next launch opens to;
* any number of **named layouts** the user saved from the toolbar dropdown,
  stored here as one JSON list under ``<group>/named_layouts``.

Choosing a named layout applies it, and from then on it is also the working
layout.  ``active_layout`` remembers which entry the window last applied
(:data:`DEFAULT_NAME` for the built-in placement, ``""`` for none), and
``active_modified`` whether the arrangement was changed since.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

_LOG = logging.getLogger("waxx.dashboard.layouts")

#: The built-in placement spec; always listed first, cannot be renamed or deleted.
DEFAULT_NAME = "Default"
#: Shown in the dropdown when the arrangement is not a named layout.
UNSAVED_LABEL = "Unsaved layout"
MAX_NAME_LEN = 60


@dataclass
class LayoutSnapshot:
    """Everything needed to put the dashboard back the way it was."""

    geometry: bytes = b""
    state: bytes = b""
    popped: list[str] = field(default_factory=list)
    popped_geometry: dict[str, bytes] = field(default_factory=dict)
    saved_at: str = ""

    def to_json(self) -> dict:
        return {
            "geometry_hex": self.geometry.hex(),
            "state_hex": self.state.hex(),
            "popped": list(self.popped),
            "popped_geometry_hex": {k: v.hex() for k, v in self.popped_geometry.items()},
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_json(cls, d: dict) -> "LayoutSnapshot":
        return cls(
            geometry=bytes.fromhex(d.get("geometry_hex", "")),
            state=bytes.fromhex(d.get("state_hex", "")),
            popped=[str(p) for p in d.get("popped", [])],
            popped_geometry={str(k): bytes.fromhex(v) for k, v in (d.get("popped_geometry_hex") or {}).items()},
            saved_at=str(d.get("saved_at", "")),
        )


class LayoutStore:
    """CRUD over the named layouts of one dashboard (one QSettings group)."""

    def __init__(self, settings, group: str):
        self._settings = settings
        self._group = group.rstrip("/")

    def _key(self, which: str) -> str:
        return f"{self._group}/{which}"

    # --- the list -------------------------------------------------------

    def _load(self) -> list[dict]:
        raw = self._settings.value(self._key("named_layouts"))
        if not raw:
            return []
        try:
            items = json.loads(raw)
            return [it for it in items if isinstance(it, dict) and it.get("name")]
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("named layouts unreadable (%r); ignoring them", exc)
            return []

    def _store(self, items: list[dict]) -> None:
        self._settings.setValue(self._key("named_layouts"), json.dumps(items))
        self._settings.sync()

    def names(self) -> list[str]:
        return [it["name"] for it in self._load()]

    def _index(self, items: list[dict], name: str) -> int:
        for i, it in enumerate(items):
            if it["name"].casefold() == name.casefold():
                return i
        return -1

    def get(self, name: str) -> Optional[LayoutSnapshot]:
        items = self._load()
        i = self._index(items, name)
        if i < 0:
            return None
        try:
            return LayoutSnapshot.from_json(items[i])
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("named layout %r unreadable: %r", name, exc)
            return None

    def put(self, name: str, snap: LayoutSnapshot) -> None:
        """Add *name*, or overwrite it in place (keeps its position in the list)."""
        snap.saved_at = time.strftime("%Y-%m-%d %H:%M")
        entry = {"name": name, **snap.to_json()}
        items = self._load()
        i = self._index(items, name)
        if i < 0:
            items.append(entry)
        else:
            entry["name"] = items[i]["name"]
            items[i] = entry
        self._store(items)

    def rename(self, old: str, new: str) -> None:
        items = self._load()
        i = self._index(items, old)
        if i < 0:
            raise KeyError(old)
        items[i]["name"] = new
        self._store(items)
        if self.active.casefold() == old.casefold():
            self.active = new

    def delete(self, name: str) -> None:
        items = self._load()
        i = self._index(items, name)
        if i < 0:
            return
        del items[i]
        self._store(items)
        if self.active.casefold() == name.casefold():
            self.active = ""

    def saved_at(self, name: str) -> str:
        items = self._load()
        i = self._index(items, name)
        return str(items[i].get("saved_at", "")) if i >= 0 else ""

    def validate_name(self, name: str, *, renaming: Optional[str] = None) -> Optional[str]:
        """Return why *name* cannot be used, or None if it can."""
        name = name.strip()
        if not name:
            return "The name is empty."
        if len(name) > MAX_NAME_LEN:
            return f"Keep the name under {MAX_NAME_LEN} characters."
        if name.casefold() in (DEFAULT_NAME.casefold(), UNSAVED_LABEL.casefold()):
            return f"'{name}' is reserved."
        if renaming is not None and name.casefold() == renaming.casefold():
            return None
        if self._index(self._load(), name) >= 0:
            return f"A layout called '{name}' already exists."
        return None

    def unique_name(self, base: str) -> str:
        base = (base.strip() or "Layout")[:MAX_NAME_LEN - 5]
        if self.validate_name(base) is None:
            return base
        n = 2
        while self.validate_name(f"{base} ({n})") is not None:
            n += 1
        return f"{base} ({n})"

    # --- which one is on screen ------------------------------------------

    @property
    def active(self) -> str:
        v = self._settings.value(self._key("active_layout"))
        return str(v) if v else ""

    @active.setter
    def active(self, name: str) -> None:
        self._settings.setValue(self._key("active_layout"), name or "")
        self._settings.sync()

    @property
    def modified(self) -> bool:
        v = self._settings.value(self._key("active_modified"))
        return str(v).lower() in ("true", "1")

    @modified.setter
    def modified(self, value: bool) -> None:
        self._settings.setValue(self._key("active_modified"), "true" if value else "false")
        self._settings.sync()


__all__ = ["DEFAULT_NAME", "UNSAVED_LABEL", "LayoutSnapshot", "LayoutStore"]
