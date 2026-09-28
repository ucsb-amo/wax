"""One command to the SLM server and the replies it asked for.

The wire format is in ``waxx/control/slm/server/slm_protocol.py``. Shared by
the experiment's :class:`~waxx.control.slm.slm.SLM` (a pattern, waiting until
it is on the display) and the monitor server's reinit service
(:mod:`waxx.util.device_state.slm_reinit`: ``status`` and ``reinit``).

One connection per command, closed once the wanted reply is in: the server
serves one connection at a time, so a connection held open blocks everyone
else. Imports nothing heavy -- the monitor server loads it.
"""

from __future__ import annotations

import itertools
import json
import socket
import time

from waxx.control.slm.server.slm_protocol import control_line

SLM_HOST = "192.168.1.102"
SLM_PORT = 5000


class NoReply(Exception):
    """The server took the command and did not answer: a server from before
    replies (``seq``, 2026-09-26) -- or, for a control command, from before
    control commands (2026-09-28). A pattern sent to such a server is applied
    all the same (it always was); a control command is dropped."""


class ReplyTimeout(TimeoutError):
    """The server answered (``queued``) but not with what was waited for in time."""


_seqs = itertools.count(1)


def exchange(host: str, port: int, payload: dict, *, until, control: bool = False,
             connect_s: float = 2.0, first_reply_s: float = 3.0, total_s: float = 15.0,
             on_sent=None) -> dict:
    """Send `payload` with a fresh ``seq``; return the first reply whose
    ``status`` is in `until`.

    `control`: send it as a control line (``SLMCTL ...``), else as a pattern
    (one JSON object). `on_sent` is called once the command is on its way
    (connected and sent) -- after that the server takes it before any
    command sent later -- or once sending it has failed.

    Raises :class:`NoReply` if the server says nothing within `first_reply_s`
    (or closes without a word), :class:`ReplyTimeout` if it answered but not
    with a status in `until` within `total_s`, and ``OSError`` if it cannot
    be reached.
    """
    seq = next(_seqs)
    msg = dict(payload, seq=seq)
    line = control_line(msg) if control else json.dumps(msg) + "\n"
    t0 = time.monotonic()
    sent = False
    try:
        with socket.create_connection((host, port), timeout=connect_s) as s:
            s.sendall(line.encode("utf-8"))
            sent = True
            if on_sent is not None:
                on_sent()
            buf, answered = b"", False
            while True:
                limit = t0 + (total_s if answered else min(first_reply_s, total_s))
                left = limit - time.monotonic()
                if left <= 0:
                    if not answered:
                        raise NoReply(f"no reply from the SLM server within {first_reply_s:g} s")
                    raise ReplyTimeout(f"the SLM server did not report {'/'.join(until)} "
                                       f"within {total_s:g} s")
                s.settimeout(left)
                try:
                    chunk = s.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    if not answered:
                        raise NoReply("the SLM server closed the connection without a reply")
                    raise ConnectionError("the SLM server closed the connection before "
                                          f"reporting {'/'.join(until)}")
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    try:
                        reply = json.loads(raw)
                    except ValueError:
                        continue
                    if not isinstance(reply, dict) or reply.get("seq") != seq:
                        continue
                    answered = True
                    if reply.get("status") in until:
                        return reply
    finally:
        if not sent and on_sent is not None:
            on_sent()
