"""Opening the viewer: separate process (default) or in-process.

Subprocess mode writes the bundle to ``bundle_dir()`` and either hands the
path to a viewer process that is already listening (a TCP server on
127.0.0.1 whose port is in ``<bundle_dir>/viewer.port``) or starts one,
detached from the notebook kernel: no console window, its own process
group (a kernel interrupt does not reach it), broken away from any job
object the kernel runs in (a kernel restart does not kill it). Its stdout
and stderr go to ``<bundle_dir>/viewer.log``.

Nothing here imports Qt: the kernel side only needs the standard library
and numpy, so ``show()`` costs the bundle write and one socket connect.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time

from waxx.util.seqview.bundle import Bundle

PORT_FILE = 'viewer.port'
LOG_FILE = 'viewer.log'
_CONNECT_TIMEOUT = 0.5
_inline_windows = []     # keep in-process windows alive


def bundle_dir():
    d = os.environ.get('WAXX_SEQVIEW_DIR') or os.path.join(
        tempfile.gettempdir(), 'waxx-seqview')
    os.makedirs(d, exist_ok=True)
    return d


def _bundle_path(name=None):
    stamp = time.strftime('%Y%m%d-%H%M%S')
    base = f"{stamp}-{name}" if name else stamp
    base = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in base)
    path = os.path.join(bundle_dir(), base + '.npz')
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(bundle_dir(), f"{base}-{n}.npz")
    return path


def _prune(keep=12):
    """Old bundles are transient: keep the newest few."""
    d = bundle_dir()
    try:
        files = sorted((f for f in os.listdir(d) if f.endswith('.npz')),
                       key=lambda f: os.path.getmtime(os.path.join(d, f)))
    except OSError:
        return
    for f in files[:-keep]:
        try:
            os.remove(os.path.join(d, f))
        except OSError:
            pass


def code_stamp():
    """A stamp of the viewer's own source files: a running viewer whose
    stamp differs from the kernel's is running old code (the package was
    edited since it started) and is left alone -- a fresh one is spawned."""
    here = os.path.dirname(os.path.abspath(__file__))
    stamp = 0.
    try:
        for name in os.listdir(here):
            if name.endswith('.py'):
                stamp = max(stamp, os.path.getmtime(os.path.join(here, name)))
    except OSError:
        pass
    return round(stamp, 3)


def read_port():
    try:
        with open(os.path.join(bundle_dir(), PORT_FILE)) as f:
            info = json.load(f)
        return int(info['port']), int(info.get('pid', 0))
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


def read_stamp():
    try:
        with open(os.path.join(bundle_dir(), PORT_FILE)) as f:
            return json.load(f).get('stamp')
    except (OSError, ValueError, TypeError):
        return None


def write_port(port):
    with open(os.path.join(bundle_dir(), PORT_FILE), 'w') as f:
        json.dump({'port': int(port), 'pid': os.getpid(),
                   'stamp': code_stamp()}, f)


def clear_port():
    try:
        os.remove(os.path.join(bundle_dir(), PORT_FILE))
    except OSError:
        pass


def send_to_viewer(path):
    """Hand a bundle path to a running viewer. True if one took it."""
    port, _pid = read_port()
    if not port:
        return False
    if read_stamp() != code_stamp():
        return False     # stale viewer code: let the caller start a new one
    try:
        with socket.create_connection(('127.0.0.1', port),
                                      timeout=_CONNECT_TIMEOUT) as s:
            s.sendall((json.dumps({'open': os.path.abspath(path)}) + '\n')
                      .encode('utf-8'))
            s.settimeout(2.0)
            reply = s.recv(64)
        return reply.startswith(b'ok')
    except OSError:
        return False


def spawn_viewer(path):
    """Start a detached viewer process on the bundle. Returns the Popen."""
    log_path = os.path.join(bundle_dir(), LOG_FILE)
    log = open(log_path, 'ab')
    cmd = [sys.executable, '-m', 'waxx.util.seqview', os.path.abspath(path)]
    kw = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=log,
              close_fds=True, cwd=bundle_dir())
    if os.name == 'nt':
        flags = (subprocess.DETACHED_PROCESS
                 | subprocess.CREATE_NEW_PROCESS_GROUP)
        breakaway = 0x01000000    # CREATE_BREAKAWAY_FROM_JOB
        try:
            return subprocess.Popen(cmd, creationflags=flags | breakaway, **kw)
        except OSError:
            return subprocess.Popen(cmd, creationflags=flags, **kw)
    return subprocess.Popen(cmd, start_new_session=True, **kw)


def show(bundle, inline=False, reuse=True, name=None, path=None):
    """Open the viewer on `bundle` (a Bundle, a dict {meta, arrays} or a
    saved .npz path).

    inline=False (default): a separate process. reuse=True updates a viewer
    that is already open instead of starting another.
    inline=True: build the window in this process; requires a running Qt
    event loop (``%gui qt`` in Jupyter) -- the call returns the window.
    """
    if inline:
        from waxx.util.seqview.app import open_inline
        win = open_inline(Bundle.coerce(bundle))
        _inline_windows.append(win)
        return win

    if isinstance(bundle, (str, os.PathLike)) and path is None:
        path = os.fspath(bundle)
    else:
        b = Bundle.coerce(bundle)
        path = path or _bundle_path(name or b.meta.get('title', ''))
        b.save(path)
        _prune()
    if reuse and send_to_viewer(path):
        return path
    spawn_viewer(path)
    return path
