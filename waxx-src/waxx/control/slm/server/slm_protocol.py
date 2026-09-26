"""Wire format of the SLM server: cutting commands out of the TCP stream, and
replying to a client that asks to be told when its pattern is on the SLM.

The server used to treat every ``recv()`` as exactly one command. TCP keeps no
message boundaries, so two commands sent close together could arrive in one
``recv`` -- both were then rejected as malformed JSON -- and one command could
be split across two. Either way the SLM silently stayed on the old pattern.

Clients send one JSON object per command: newline-terminated (the spot-finder
GUI) or unterminated and followed by closing the connection
(``waxx.control.slm.SLM``). Legacy clients send whitespace-separated plaintext,
one command per send.

A command that carries an integer ``"seq"`` asks for replies, one JSON object
per line on the same connection:

    {"seq": n, "status": "queued"}                      as soon as it is parsed
    {"seq": n, "status": "applied", "center": [x, y],   once Write_image returned
     "mask": ..., "dimension": ..., "t_apply_s": ...}
    {"seq": n, "status": "error", "error": "..."}       it could not be applied
    {"seq": n, "status": "dropped"}                     pushed out of a full queue

A command without ``"seq"`` gets no reply, so existing clients see no change.
"""

import json
import threading

# An unterminated JSON object that still does not parse after this much text
# is passed on as malformed rather than held forever.
MAX_PENDING_CHARS = 65536

_decoder = json.JSONDecoder()


def split_commands(buf: str, at_eof: bool = False):
    """Return ``(commands, remainder)`` for the text received so far.

    JSON objects are cut out by parsing them, so they may be back to back,
    newline separated, or unterminated. An object that does not parse yet is
    kept as the remainder until more text arrives -- unless its line is already
    complete, or the stream has ended, in which case it is passed on as it is
    and the caller reports it as malformed. Anything that does not start with
    ``{`` is a legacy plaintext command: one line, or the rest of the chunk
    when there is no newline, which is what the server always did.
    """
    commands = []
    i, n = 0, len(buf)
    while True:
        while i < n and buf[i].isspace():
            i += 1
        if i >= n:
            return commands, ""
        if buf[i] == "{":
            try:
                _, end = _decoder.raw_decode(buf, i)
            except json.JSONDecodeError:
                j = buf.find("\n", i)
                if j >= 0:
                    commands.append(buf[i:j])
                    i = j + 1
                    continue
                if at_eof or n - i > MAX_PENDING_CHARS:
                    commands.append(buf[i:])
                    return commands, ""
                return commands, buf[i:]
            commands.append(buf[i:end])
            i = end
        else:
            j = buf.find("\n", i)
            if j < 0:
                commands.append(buf[i:])
                return commands, ""
            commands.append(buf[i:j])
            i = j + 1


def command_seq(command: str):
    """The client's sequence number if it asked for replies, else None."""
    try:
        d = json.loads(command)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    seq = d.get("seq")
    # bool is an int subclass; a stray true/false is not a sequence number
    return seq if isinstance(seq, int) and not isinstance(seq, bool) else None


class Replier:
    """Send newline-terminated JSON replies on one client connection.

    The connection's reader thread says "queued" and the SLM worker thread
    says "applied", so sends are serialised by a lock.
    """

    def __init__(self, conn):
        self._conn = conn
        self._lock = threading.Lock()

    def send(self, msg: dict) -> bool:
        try:
            with self._lock:
                self._conn.sendall((json.dumps(msg) + "\n").encode("utf-8"))
            return True
        except OSError:
            # The client has gone. It is no longer waiting for this reply.
            return False
