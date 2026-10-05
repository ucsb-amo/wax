"""logging_setup: a failing log share must not block or flood the process.

The share is simulated in a temp dir.  The two failures reproduced here were
seen 2026-09-28 after an SMB reconnect: the open log handle went dead (every
tell/write raised Errno 22), and a new open of that one file hung ~70 s and
then raised PermissionError (Errno 13).  Before the fix, every record retried
the dead file and printed a "--- Logging error ---" traceback, and every
logging call in a server's request loop waited out the 70 s.
"""
import logging
import os
import time

from waxx.util.dashboard import logging_setup
from waxx.util.dashboard.logging_setup import _SafeRotatingFileHandler

_REAL_OPEN = _SafeRotatingFileHandler._open


def _record(msg):
    return logging.LogRecord("unit", logging.INFO, __file__, 0, msg, None, None)


def _handler(primary, fallback):
    h = _SafeRotatingFileHandler(str(primary), fallback=fallback, maxBytes=1 << 20,
                                 backupCount=1, encoding="utf-8", delay=True)
    h.setFormatter(logging.Formatter("%(message)s"))
    return h


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


class _DeadStream:
    """What the SMB reconnect left behind: every call fails with Errno 22."""
    closed = False

    def tell(self):
        raise OSError(22, "Invalid argument")

    def write(self, s):
        raise OSError(22, "Invalid argument")

    def flush(self):
        raise OSError(22, "Invalid argument")

    def close(self):
        raise OSError(22, "Invalid argument")


def test_dead_stream_moves_to_fallback(tmp_path, capsys):
    primary, fallback = tmp_path / "share" / "x.log", tmp_path / "local" / "x.log"
    primary.parent.mkdir()
    h = _handler(primary, fallback)
    h.emit(_record("before"))
    h.stream = _DeadStream()
    h.emit(_record("after 1"))
    h.emit(_record("after 2"))
    h.close()

    assert _lines(primary) == ["before"]
    lines = _lines(fallback)
    assert lines[0].startswith("logging_setup: cannot write")
    assert lines[1:] == ["after 1", "after 2"]
    err = capsys.readouterr().err
    assert err.count("cannot write") == 1
    assert "Logging error" not in err


def test_refused_primary_is_tried_once_not_per_record(tmp_path, monkeypatch, capsys):
    primary, fallback = tmp_path / "share" / "x.log", tmp_path / "local" / "x.log"
    primary.parent.mkdir()
    refused = []

    def refusing_open(self):
        if self.baseFilename == os.path.abspath(str(primary)):
            refused.append(1)
            raise PermissionError(13, "Permission denied")
        return _REAL_OPEN(self)

    monkeypatch.setattr(_SafeRotatingFileHandler, "_open", refusing_open)
    h = _handler(primary, fallback)
    for i in range(5):
        h.emit(_record(f"record {i}"))
    h.close()

    assert len(refused) == 1
    assert not primary.exists()
    assert _lines(fallback)[1:] == [f"record {i}" for i in range(5)]
    assert "Logging error" not in capsys.readouterr().err


def test_returns_to_primary_when_it_is_writable_again(tmp_path, monkeypatch):
    primary, fallback = tmp_path / "share" / "x.log", tmp_path / "local" / "x.log"
    primary.parent.mkdir()
    share_down = True

    def flaky_open(self):
        if share_down and self.baseFilename == os.path.abspath(str(primary)):
            raise PermissionError(13, "Permission denied")
        return _REAL_OPEN(self)

    monkeypatch.setattr(_SafeRotatingFileHandler, "_open", flaky_open)
    h = _handler(primary, fallback)
    h.emit(_record("while down"))
    share_down = False
    h.emit(_record("not yet retried"))   # RETRY_PRIMARY_S has not passed
    h.RETRY_PRIMARY_S = 0.0
    h.emit(_record("back up"))
    h.close()

    assert _lines(fallback)[1:] == ["while down", "not yet retried"]
    lines = _lines(primary)
    assert "is writable again" in lines[0]
    assert lines[1:] == ["back up"]


def test_logging_call_does_not_wait_for_a_hung_share(tmp_path, monkeypatch):
    share = tmp_path / "share"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    def hung_open(self):
        if self.baseFilename.startswith(os.path.abspath(str(share))):
            time.sleep(1.0)
            raise PermissionError(13, "Permission denied")
        return _REAL_OPEN(self)

    monkeypatch.setattr(_SafeRotatingFileHandler, "_open", hung_open)
    monkeypatch.setattr(logging_setup.faulthandler, "enable", lambda **kw: None)
    # Process-wide hooks: not this test's subject, and they would outlive it.
    monkeypatch.setattr(logging_setup, "install_excepthooks", lambda: None)
    monkeypatch.setattr(logging_setup, "install_qt_message_handler", lambda: False)
    for name in ("_APP_NAME", "_LOG_ROOT", "_LOGGER_NS", "_ACTIVE_LOG_DIR", "_FAULTHANDLER_FILE"):
        monkeypatch.setattr(logging_setup, name, getattr(logging_setup, name))
    monkeypatch.setattr(logging_setup, "_FILE_WRITERS", {})
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level

    logging_setup.configure(app_name="unit", log_root=str(share))
    try:
        t0 = time.monotonic()
        log_path = logging_setup.configure_server_logging("unit")
        logging.getLogger("unit.test").warning("not blocked")
        assert time.monotonic() - t0 < 0.5

        fallback = tmp_path / "local" / "unit" / "dashboard" / "_logs" / "server" / log_path.name
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if fallback.exists() and "not blocked" in fallback.read_text(encoding="utf-8"):
                break
            time.sleep(0.05)
        text = fallback.read_text(encoding="utf-8")
        assert "cannot write" in text
        assert "configure_server_logging: id=unit" in text
        assert "not blocked" in text
    finally:
        for writer in logging_setup._FILE_WRITERS.values():
            writer.stop()
            for h in writer.handlers:
                h.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        if logging_setup._FAULTHANDLER_FILE is not None:
            logging_setup._FAULTHANDLER_FILE.close()
