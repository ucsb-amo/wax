"""Push values into Zabbix trapper items (the ``zabbix_sender`` protocol).

Stdlib only (``socket``/``struct``).  The transport is one method,
:meth:`ZabbixSender._exchange`, which tests replace with a canned fake.

Zabbix only accepts a pushed value if the target host has an item of type
*Zabbix trapper* with that exact key; anything else is counted as
``failed`` in the reply (the connection itself still succeeds).  The guest
API login cannot create items, so a Zabbix admin has to add them once.

Wire format: ``b"ZBXD\\x01"`` + little-endian uint32 data length + uint32
reserved (0) + a JSON body ``{"request": "sender data", "data": [...]}``.
The server answers in the same framing with
``{"response": "success", "info": "processed: 1; failed: 0; total: 1; seconds spent: ..."}``.
"""
from __future__ import annotations

import json
import re
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

DEFAULT_SERVER = "weldlabaio1.physics.ucsb.edu"
DEFAULT_PORT = 10051

_HEADER = b"ZBXD\x01"
_BATCH = 250   # values per request, zabbix_sender's own default
_INFO = re.compile(r"processed: (\d+); failed: (\d+); total: (\d+)")


@dataclass(frozen=True)
class TrapperValue:
    """One value for one trapper item.  ``clock`` is unix seconds (``None`` = server time)."""
    host: str
    key: str
    value: object
    clock: Optional[float] = None


@dataclass
class SenderResult:
    """Totals over every request of one :meth:`ZabbixSender.send` call."""
    processed: int = 0
    failed: int = 0
    total: int = 0
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.processed == self.total


class ZabbixSender:
    """Send values to a Zabbix server's trapper port.

    Parameters
    ----------
    server : str
        Zabbix server host name or IP (not the frontend URL).
    port : int
        Trapper port, 10051 by default.
    timeout : float
        Per-request socket timeout in seconds.
    """

    def __init__(self, server: str = DEFAULT_SERVER, port: int = DEFAULT_PORT,
                 timeout: float = 10.0):
        self.server = server
        self.port = int(port)
        self.timeout = float(timeout)

    # -- transport -------------------------------------------------------

    def _exchange(self, packet: bytes) -> bytes:
        """One TCP round-trip.  Returns the raw framed reply."""
        try:
            with socket.create_connection((self.server, self.port), timeout=self.timeout) as s:
                s.sendall(packet)
                chunks = []
                while True:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
        except OSError as exc:
            raise ConnectionError(
                f"Zabbix trapper at {self.server}:{self.port} unreachable ({exc}). "
                "Are you on the Broida VPN?"
            ) from exc
        return b"".join(chunks)

    # -- framing ---------------------------------------------------------

    @staticmethod
    def pack(body: dict) -> bytes:
        data = json.dumps(body).encode("utf-8")
        return _HEADER + struct.pack("<II", len(data), 0) + data

    @staticmethod
    def unpack(reply: bytes) -> dict:
        if not reply.startswith(_HEADER) or len(reply) < 13:
            raise ValueError(f"not a Zabbix reply: {reply[:32]!r}")
        (length, _reserved) = struct.unpack("<II", reply[5:13])
        return json.loads(reply[13:13 + length].decode("utf-8"))

    @staticmethod
    def _entry(v: TrapperValue) -> dict:
        e = {"host": v.host, "key": v.key, "value": str(v.value)}
        if v.clock is not None:
            sec = int(v.clock)
            e["clock"] = sec
            e["ns"] = int(round((float(v.clock) - sec) * 1e9)) % 1_000_000_000
        return e

    # -- public ----------------------------------------------------------

    def send(self, values: Iterable[TrapperValue]) -> SenderResult:
        """Send ``values`` (in batches); raises ``ConnectionError`` if the server is unreachable."""
        values = list(values)
        result = SenderResult()
        for i in range(0, len(values), _BATCH):
            batch = values[i:i + _BATCH]
            body = {"request": "sender data", "data": [self._entry(v) for v in batch],
                    "clock": int(time.time())}
            reply = self.unpack(self._exchange(self.pack(body)))
            if reply.get("response") != "success":
                raise RuntimeError(f"Zabbix rejected sender data: {reply}")
            info = str(reply.get("info", ""))
            result.info.append(info)
            m = _INFO.search(info)
            if m:
                result.processed += int(m.group(1))
                result.failed += int(m.group(2))
                result.total += int(m.group(3))
            else:
                result.total += len(batch)
        return result

    def send_one(self, host: str, key: str, value, clock: Optional[float] = None) -> SenderResult:
        return self.send([TrapperValue(host, key, value, clock)])


__all__ = ["DEFAULT_PORT", "DEFAULT_SERVER", "SenderResult", "TrapperValue", "ZabbixSender"]
