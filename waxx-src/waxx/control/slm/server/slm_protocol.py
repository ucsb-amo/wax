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

Control commands (2026-09-28) are one line, ``SLMCTL <json>\\n`` (see
:func:`control_line`), never a bare JSON object: every earlier server takes any
JSON object for a pattern and fills in defaults -- a ``{"cmd": "status"}``
would put up a blank mask. Earlier servers drop an ``SLMCTL`` line as
malformed plaintext and do not answer, so a client that gets no reply knows the
server has no control commands.

    SLMCTL {"cmd": "status", "seq": n}
        -> {"seq": n, "status": "ok", "reinit_due": bool, "due_for_s",
            "next_due_in_s", "interval_s", "reinit_in_progress",
            "last_reinit_age_s", "last_reinit_error", "pattern_epoch", "pattern",
            "idle_s", "queue_len", "auto_reinit"}
    SLMCTL {"cmd": "reinit", "seq": n, "by": "..."}
        -> {"seq": n, "status": "queued"}, then
           {"seq": n, "status": "reinit_done", "pattern": {...},
            "pattern_epoch": m, "t_reinit_s": ...}   (or "error")

The server marks a reinit due an hour after the last one (or after it started)
and no longer does it by itself: the monitor server asks for it when no run is
starting or running. After a reinit the server puts back the pattern it
showed, and ``pattern_epoch`` goes up.

Supervision (2026-09-28): on the SLM PC the server runs under ``supervisor.py``
(:mod:`waxx.util.supervise`), which starts it again when it exits. Two more
control commands, both queued behind any pattern already queued:

    SLMCTL {"cmd": "restart", "seq": n, "by": "..."}
        -> {"seq": n, "status": "queued"}, then {"seq": n, "status": "restarting"};
           the process exits with EXIT_RESTART and its supervisor starts a new
           one, which puts the saved pattern back. Refused ("error") when the
           server is not supervised -- nothing would start it again.
    SLMCTL {"cmd": "shutdown", "seq": n, "by": "..."}
        -> {"seq": n, "status": "queued"}, then {"seq": n, "status": "shutting_down"};
           exits with EXIT_SHUTDOWN and the supervisor stops too. Accepted only
           from the SLM PC itself (the supervisor's graceful stop).

Commands still queued when the process exits are answered ``"dropped"``.
``status`` also reports ``pid``, ``instance`` (new for every process),
``started_at``, ``supervised``, ``start_count``, ``last_exit``, ``slm_ready``,
``applies``, ``pattern_source``, ``bind`` and ``capabilities``.

The log (2026-09-28): the supervisor writes every line of the server and every
event of its own to ``<state dir>\\logs\\slm_server_<date>.log`` on the SLM PC.
The server reads those files back for a client far from the SLM PC -- the
Device Control GUI, through the monitor server -- answered at once, never
queued behind the SLM:

    SLMCTL {"cmd": "log", "seq": n, "cursor": c, "tail": k}
        -> {"seq": n, "status": "ok", "lines": [...], "cursor": c', "more": bool,
            "restarted": bool, "skipped_bytes": b, "path": "<the log folder>"}

``cursor`` is the ``cursor`` of the previous reply (null for the first
request, which gets the last ``tail`` lines). ``more``: lines past this reply
are waiting -- ask again at once. ``restarted``: the cursor no longer fits the
files (one was removed or cut short), so this reply starts again from the last
lines. ``skipped_bytes``: this much unread text was jumped over (more than
:data:`LOG_SKIP_BYTES`: nobody followed it); the files keep it. Only a
supervised server has a log file; an unsupervised one answers ``error``. A
``log`` request leaves no lines in the log it reads (see ``QUIET_COMMANDS`` in
``run_server.py``).
"""

import json
import os
import threading

# An unterminated JSON object that still does not parse after this much text
# is passed on as malformed rather than held forever.
MAX_PENDING_CHARS = 65536

#: First word of a control command line.
CONTROL_PREFIX = "SLMCTL"

#: Where the SLM server listens (the SLM PC).
DEFAULT_SERVER_IP = "192.168.1.102"
DEFAULT_SERVER_PORT = 5000

#: Exit codes: the server's protocol with its supervisor (waxx.util.supervise).
EXIT_SHUTDOWN = 0          # asked to stop: the supervisor stops too
EXIT_FATAL = 70            # the SLM worker died unexpectedly
EXIT_INIT_FAILED = 71      # the SLM could not be initialised at start-up
EXIT_PORT_IN_USE = 72      # another SLM server is already listening on the port
EXIT_RESTART = 75          # asked to restart: the supervisor starts it again at once

EXIT_MEANING = {EXIT_SHUTDOWN: "shut down on request", EXIT_FATAL: "SLM worker died",
                EXIT_INIT_FAILED: "SLM initialisation failed at start-up",
                EXIT_PORT_IN_USE: "another SLM server holds the port",
                EXIT_RESTART: "restart requested"}

#: What this server understands, reported by ``status``.
CAPABILITIES = ("seq", "status", "reinit", "restart", "shutdown", "log")


def default_state_dir() -> str:
    """Where the saved pattern, the heartbeat and the supervisor's logs live on
    this PC: ``SLM_STATE_DIR``, else ``%LOCALAPPDATA%\\slm_server``."""
    import tempfile  # noqa: PLC0415
    return (os.environ.get("SLM_STATE_DIR")
            or os.path.join(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir(),
                            "slm_server"))


# --- the log files (written by supervisor.Journal, read back by "log") -------------------

LOG_DIRNAME = "logs"
LOG_PREFIX = "slm_server_"
LOG_SUFFIX = ".log"
#: Lines a first ``log`` reply (no cursor) starts with.
LOG_TAIL_LINES = 300
#: Most lines one ``log`` reply carries; the client asks again for the rest.
LOG_REPLY_LINES = 500
#: Longest line passed on; the rest is cut off, and the line says so.
LOG_LINE_CHARS = 2000
#: Unread text past this (about 3000 lines) is jumped over rather than sent
#: (see the module doc).
LOG_SKIP_BYTES = 300_000
#: Where the reading starts after a jump: this far before the end.
LOG_RESUME_BYTES = 64_000
#: Most bytes read from a file at once.
LOG_READ_BYTES = 256_000
#: Files a first reply's last lines may come from (a day that has just begun).
LOG_TAIL_FILES = 2


def log_dir(state_dir=None) -> str:
    """The folder of the supervisor's daily log files."""
    return os.path.join(state_dir or default_state_dir(), LOG_DIRNAME)


def _log_files(folder) -> list:
    """The daily log files, oldest first (their names sort by date)."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(n for n in names if n.startswith(LOG_PREFIX) and n.endswith(LOG_SUFFIX))


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _read(path, start, n) -> bytes:
    if n <= 0:
        return b""
    with open(path, "rb") as fh:
        fh.seek(start)
        return fh.read(n)


def _text(raw: bytes) -> str:
    line = raw.decode("utf-8", errors="replace").rstrip("\r")
    if len(line) > LOG_LINE_CHARS:
        line = f"{line[:LOG_LINE_CHARS]} … ({len(line) - LOG_LINE_CHARS} more characters)"
    return line


def _cut_lines(data: bytes, final: bool) -> list:
    """The lines in `data` as ``(bytes, bytes consumed)``: only lines ended by
    a newline, unless `final` (the file gets no more text, so its last line is
    whole without one)."""
    out, pos = [], 0
    while True:
        j = data.find(b"\n", pos)
        if j < 0:
            if final and pos < len(data):
                out.append((data[pos:], len(data) - pos))
            return out
        out.append((data[pos:j], j + 1 - pos))
        pos = j + 1


def _tail(folder, files, n):
    """The last `n` whole lines, oldest first, and the cursor after them."""
    lines, cursor = [], None
    for k, name in enumerate(reversed(files[-LOG_TAIL_FILES:])):
        path = os.path.join(folder, name)
        size = _size(path)
        if size is None:
            continue
        start = max(0, size - LOG_READ_BYTES)
        data = _read(path, start, size - start)
        newest = k == 0
        if newest:
            # the newest file may end in a line still being written
            end = data.rfind(b"\n") + 1
            data = data[:end]
            cursor = {"file": name, "offset": start + end}
        parts = _cut_lines(data, final=not newest)
        if start > 0 and parts:
            parts = parts[1:]           # began mid-line
        lines = [_text(raw) for raw, _ in parts] + lines
        if len(lines) >= n or start > 0:
            break                        # enough, or an older file would leave a gap
    return lines[-n:] if n > 0 else [], cursor


def read_log(folder, cursor=None, tail=LOG_TAIL_LINES, max_lines=LOG_REPLY_LINES) -> dict:
    """What the ``log`` command answers (see the module doc), from the log
    files in `folder`."""
    files = _log_files(folder)
    reply = {"status": "ok", "lines": [], "cursor": None, "more": False, "restarted": False,
             "skipped_bytes": 0, "path": str(folder)}
    if not files:
        return reply
    name = cursor.get("file") if isinstance(cursor, dict) else None
    offset = cursor.get("offset") if isinstance(cursor, dict) else None
    size = _size(os.path.join(folder, name)) if name in files else None
    if not isinstance(offset, int) or isinstance(offset, bool) or size is None \
            or not 0 <= offset <= size:
        reply["lines"], reply["cursor"] = _tail(folder, files, tail)
        reply["restarted"] = cursor is not None
        return reply

    i = files.index(name)
    later = [_size(os.path.join(folder, f)) or 0 for f in files[i + 1:]]
    unread = size - offset + sum(later)
    if unread > LOG_SKIP_BYTES:
        # Jump to near the end of the newest file, at a line start.
        last = os.path.join(folder, files[-1])
        last_size = _size(last) or 0
        start = max(0, last_size - LOG_RESUME_BYTES)
        if start > 0:
            nl = _read(last, start, LOG_RESUME_BYTES).find(b"\n")
            start = last_size if nl < 0 else start + nl + 1
        skipped = (size - offset) + sum(later[:-1]) + start if later else start - offset
        i, offset = len(files) - 1, start
        reply["skipped_bytes"] = int(skipped)

    lines = []
    while len(lines) < max_lines:
        path = os.path.join(folder, files[i])
        size = _size(path) or 0
        final = i < len(files) - 1      # an earlier day's file gets no more lines
        data = _read(path, offset, min(max(size - offset, 0), LOG_READ_BYTES))
        whole = offset + len(data) >= size
        parts = _cut_lines(data, final=final and whole)
        if not parts and not whole and data:
            parts = [(data, len(data))]  # one line longer than a read: pass it on cut
        for raw, n in parts[:max_lines - len(lines)]:
            lines.append(_text(raw))
            offset += n
        if len(lines) >= max_lines:
            break
        if offset >= size and final:
            i, offset = i + 1, 0         # on to the next day's file
            continue
        if not parts or whole:
            break
    remaining = (_size(os.path.join(folder, files[i])) or 0) - offset + sum(
        _size(os.path.join(folder, f)) or 0 for f in files[i + 1:])
    reply.update(lines=lines, cursor={"file": files[i], "offset": offset},
                 more=len(lines) >= max_lines and remaining > 0)
    return reply

_decoder = json.JSONDecoder()


def _maybe_control(rest: str) -> bool:
    """Whether unterminated text could be (the start of) a control line."""
    return CONTROL_PREFIX.startswith(rest[:len(CONTROL_PREFIX)])


def split_commands(buf: str, at_eof: bool = False):
    """Return ``(commands, remainder)`` for the text received so far.

    JSON objects are cut out by parsing them, so they may be back to back,
    newline separated, or unterminated. An object that does not parse yet is
    kept as the remainder until more text arrives -- unless its line is already
    complete, or the stream has ended, in which case it is passed on as it is
    and the caller reports it as malformed. Anything that does not start with
    ``{`` is a legacy plaintext command: one line, or the rest of the chunk
    when there is no newline, which is what the server always did. A control
    line (``SLMCTL ...``) is always newline-terminated, so one without its
    newline yet is kept until the rest arrives.
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
                if _maybe_control(buf[i:]) and not at_eof and n - i <= MAX_PENDING_CHARS:
                    return commands, buf[i:]
                commands.append(buf[i:])
                return commands, ""
            commands.append(buf[i:j])
            i = j + 1


def _seq_of(d: dict):
    seq = d.get("seq")
    # bool is an int subclass; a stray true/false is not a sequence number
    return seq if isinstance(seq, int) and not isinstance(seq, bool) else None


def command_seq(command: str):
    """The client's sequence number if it asked for replies, else None."""
    ctl = control_command(command)
    if ctl is not None:
        return _seq_of(ctl)
    try:
        d = json.loads(command)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    return _seq_of(d)


def control_line(obj: dict) -> str:
    """The wire form of a control command: ``SLMCTL <json>`` and a newline."""
    return f"{CONTROL_PREFIX} {json.dumps(obj, separators=(',', ':'))}\n"


def control_command(command: str):
    """The control command as a dict (with a string ``"cmd"``) if `command`
    is a control line, else None (a pattern, or malformed)."""
    if not isinstance(command, str):
        return None
    head, _, body = command.strip().partition(" ")
    if head != CONTROL_PREFIX:
        return None
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(d, dict) and isinstance(d.get("cmd"), str):
        return d
    return None


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
