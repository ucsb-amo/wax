"""
ZMQ REQ client used by the experiment process to communicate with LiveODServer.

All public methods are synchronous (blocking) and must be called sequentially
— one outstanding request at a time, matching the REQ/REP pattern.

Typical per-run sequence:

    reply = client.init_run(payload)          # INIT_RUN
    if capture_images:
        client.wait_cam_ready()               # WAIT_CAM_READY
    for each shot:
        client.shot_complete(idx, N, xvars)   # SHOT_COMPLETE
    client.end_run(payload)                   # END_RUN

The INIT_RUN reply names the run (``run_token``); the client sends it back with
WAIT_CAM_READY, SHOT_COMPLETE, END_RUN and ABORT_RUN. If another INIT_RUN has
taken liveOD over in the meantime, the server ignores those messages and says so
(``stale_run``) instead of letting them save into, or delete, the newer run's file.

A process that exits with its run still open (no END_RUN, and no ABORT_RUN that
reached liveOD) tells liveOD so from an atexit handler (``notify_exit`` ->
RUN_EXITED); without it the run sat "in progress", or an abort on "Aborting",
until the next run started (run 83110, 2026-09-26).
"""

import atexit
import logging
import pickle
import sys
import threading
import time

import numpy as np
import zmq

from beacon.discovery.client import NetClient
from waxx.util.comms_server.hardware_id import resolve_scoped_server_id


# wait_cam_ready asks in slices this long, so a reset is noticed within one slice.
CAM_READY_SLICE_S = 0.5
# PUT_DATA: one array of a few MB on the lab network; the reply is immediate
PUT_DATA_TIMEOUT_MS = 10_000
# a request this big is said out loud (see _send_recv)
LARGE_MESSAGE_BYTES = 1 << 30
# The asynchronous END_RUN save: polled this often; given up when its phase
# has not changed for SAVE_STALL_S, or after SAVE_MAX_S in all.
SAVE_POLL_S = 0.5
SAVE_STALL_S = 600.0
SAVE_MAX_S = 3600.0


class LiveODDataSender:
    """PUT_DATA from the host threads that have a shot's data during the run
    (camera stream workers, the scope reader): a REQ socket of its own, one
    request at a time under a lock, each array sent as a raw frame (never
    pickled, never copied). The client's own socket stays the run's
    sequential channel."""

    def __init__(self, client):
        self._client = client
        self._lock = threading.Lock()
        self._ctx = None
        self._sock = None

    def _socket(self):
        if self._sock is None:
            if self._ctx is None:
                self._ctx = zmq.Context()
            sock = self._ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDTIMEO, PUT_DATA_TIMEOUT_MS)
            sock.setsockopt(zmq.RCVTIMEO, PUT_DATA_TIMEOUT_MS)
            sock.connect(f"tcp://{self._client._ip}:{self._client._port}")
            self._sock = sock
        return self._sock

    def _drop_socket(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    @staticmethod
    def _frames(specs):
        """``[header, array, array, ...]``: the header names each array's key,
        slot (``index``), or slice start (``offset``), shape and dtype, and
        what to make the dataset from if the file has none (``full_shape``,
        ``fill``)."""
        header = {"tag": "PUT_DATA", "items": []}
        frames = []
        for spec in specs:
            arr = np.ascontiguousarray(spec["array"])
            index = spec.get("index")
            full = spec.get("full_shape")
            header["items"].append({
                "key": str(spec["key"]),
                "index": None if index is None else [int(i) for i in index],
                "offset": None if spec.get("offset") is None else int(spec["offset"]),
                "shape": [int(s) for s in arr.shape],
                "dtype": arr.dtype.str,
                "full_shape": None if full is None else [int(s) for s in full],
                "fill": spec.get("fill"),
            })
            frames.append(arr)
        return header, frames

    def put(self, specs) -> dict:
        """Send the arrays; the server's reply (``ok``, ``queued``)."""
        header, frames = self._frames(specs)
        head = pickle.dumps(self._client._for_this_run(header))
        with self._lock:
            sock = self._socket()
            try:
                sock.send_multipart([head] + frames, copy=False)
                return pickle.loads(sock.recv())
            except zmq.Again:
                self._drop_socket()
                raise ConnectionError(
                    f"[LiveODClient] No reply to PUT_DATA from liveOD at "
                    f"tcp://{self._client._ip}:{self._client._port} within "
                    f"{PUT_DATA_TIMEOUT_MS / 1000:.0f} s")
            except Exception:
                # a REQ socket left between send and recv refuses every later
                # send: the shot queue's retry needs a fresh one
                self._drop_socket()
                raise

    def close(self):
        with self._lock:
            self._drop_socket()
            if self._ctx is not None:
                try:
                    self._ctx.term()
                except Exception:
                    pass
                self._ctx = None
# The exit notice's one request may take this long; the process is exiting.
EXIT_NOTICE_TIMEOUT_MS = 2000


def _ascii(text) -> str:
    """ExptBuilder pipes stdout through cp1252: print ASCII only."""
    return str(text).encode("ascii", "replace").decode("ascii")


class LiveODClient(NetClient):
    """REQ socket client for LiveODServer."""

    def __init__(self, timeout_ms: int = 5000, discovery_timeout: float = 10.0):
        super().__init__(resolve_scoped_server_id("live_od"), discovery_timeout=discovery_timeout)
        self._ip = self.host
        self._port = self.port
        self._timeout_ms = timeout_ms
        # Context and socket are created lazily on first _send_recv call so
        # that constructing this object in Base.__init__ / prepare() does NOT
        # open a network connection immediately.
        self._context = None
        self._socket = None
        self.last_adjust_values: dict = {}
        # Reset flag from the latest server reply that carried one.
        self.last_reset_requested: bool = False
        # The END_RUN reply; ``incomplete`` in it means frames were missing.
        self.last_end_run_reply: dict = {}
        # The server's name for this run (INIT_RUN reply), sent back on every
        # message about the run so a superseded run's messages cannot act on
        # the run that replaced it. "" against a server that issues none.
        self._run_token: str = ""
        # This run's "the camera stopped recording" warning has been printed.
        self._grab_failure_warned: bool = False
        # A run is open: INIT_RUN answered, and neither END_RUN nor ABORT_RUN has
        # reached liveOD since (notify_exit tells liveOD at exit).
        self._run_open: bool = False
        self._exit_notified: bool = False
        self._exit_hook_registered: bool = False
        # What the server said it can do (INIT_RUN reply ``features``): an
        # older server says nothing, and gets the original protocol.
        self.server_features: dict = {}
        self._sender = None

    def supports(self, feature: str) -> bool:
        """The server has ``feature`` (``put_data``, ``async_save``)."""
        return bool(self.server_features.get(feature))

    def put_data(self, specs) -> dict:
        """PUT_DATA: arrays into the run's file now (see LiveODDataSender).
        Thread-safe; raises ConnectionError on no reply."""
        if self._sender is None:
            self._sender = LiveODDataSender(self)
        return self._sender.put(specs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        """(Re)create the REQ socket and connect to the server."""
        if self._context is None:
            self._context = zmq.Context()
        if self._socket is not None:
            try:
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.close()
            except Exception:
                pass
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.connect(f"tcp://{self._ip}:{self._port}")

    def _rediscover(self) -> None:
        """Update ``_ip``/``_port`` from the latest beacon cache.

        Delegates to ``NetClient._rediscover()`` then syncs the ZMQ
        address fields from the updated ``self.host``/``self.port``.
        """
        super()._rediscover(timeout=2.0)
        self._ip = self.host
        self._port = self.port

    def _send_recv(self, payload: dict, rcvtimeo_ms: int = None) -> dict:
        """Send ``payload`` and return the decoded reply.

        If ``rcvtimeo_ms`` is given, the receive timeout is temporarily
        overridden (e.g. for WAIT_CAM_READY and END_RUN which can be slow).
        """
        if self._socket is None:
            self._connect()
        if rcvtimeo_ms is not None:
            self._socket.setsockopt(zmq.RCVTIMEO, rcvtimeo_ms)
        try:
            data = pickle.dumps(payload)
            if len(data) > LARGE_MESSAGE_BYTES:
                # runs 84979 / 84980 (2026-10-02): END_RUN messages of 4.3 GiB
                # never reached liveOD; the per-shot PUT_DATA path exists so
                # that no message need be this big
                print(f"[LiveODClient] WARNING: the {payload.get('tag')} message is "
                      f"{len(data) / 2 ** 30:.2f} GiB; messages this size have failed to "
                      f"arrive. Is liveOD taking data during the run (PUT_DATA)?")
            self._socket.send(data)
            return pickle.loads(self._socket.recv())
        except zmq.Again:
            # Re-discover in case liveOD restarted on a new port, then
            # recreate the socket to the (possibly updated) address.
            self._rediscover()
            self._connect()
            raise ConnectionError(
                f"[LiveODClient] No response from liveOD server at "
                f"tcp://{self._ip}:{self._port}. Is liveOD running?"
            )
        finally:
            if rcvtimeo_ms is not None:
                self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)

    def _send_once(self, payload: dict, timeout_ms: int) -> dict:
        """One request on a fresh socket that gives up after ``timeout_ms``: for
        the exit notice, which must neither reuse a socket an interrupted call
        left mid-request, nor rediscover, nor linger at interpreter exit."""
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        try:
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
            sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
            sock.connect(f"tcp://{self._ip}:{self._port}")
            sock.send(pickle.dumps(payload))
            return pickle.loads(sock.recv())
        finally:
            sock.close()
            ctx.term()

    @staticmethod
    def _uncaught_reason() -> str:
        """The exception the process is exiting on ("uncaught <type>: <message>"),
        "" when it is exiting without one."""
        exc = getattr(sys, "last_exc", None)
        if exc is None:
            exc = getattr(sys, "last_value", None)
        if exc is None:
            return ""
        return _ascii(f"uncaught {type(exc).__name__}: {exc}")

    def notify_exit(self) -> None:
        """atexit handler (registered by init_run): if this process is exiting with
        its run still open, tell liveOD (RUN_EXITED). Once; never raises.

        liveOD takes it as the answer to a pending abort, else marks the run
        "exited" and leaves its file as it is (see LiveODServer._handle_run_exited)."""
        if not getattr(self, "_run_open", False) or getattr(self, "_exit_notified", False):
            return
        self._exit_notified = True
        reason = self._uncaught_reason()
        payload = self._for_this_run({"tag": "RUN_EXITED", "reason": reason})
        try:
            reply = self._send_once(payload, EXIT_NOTICE_TIMEOUT_MS)
            if reply.get("ok"):
                outcome = "liveOD was told."
            elif reply.get("stale_run"):
                outcome = "liveOD had already moved on to another run."
            else:
                outcome = (f"liveOD did not take it ({reply.get('error')}; a liveOD older "
                           f"than RUN_EXITED needs a restart).")
            print(_ascii(f"[LiveODClient] exiting without END_RUN"
                         f"{' (' + reason + ')' if reason else ''}; {outcome}"))
        except Exception as exc:
            print(_ascii(f"[LiveODClient] exiting without END_RUN, and could not tell liveOD "
                         f"({type(exc).__name__}: {exc}); its run stays open until the next "
                         f"run starts or someone resets it."))

    def _for_this_run(self, payload: dict) -> dict:
        """``payload`` with this run's token added (when the server gave one)."""
        token = getattr(self, "_run_token", "")
        if token:
            payload["run_token"] = token
        return payload

    @staticmethod
    def _print_camera_overrides(reply: dict) -> None:
        """The server applied camera settings that differ from the run's
        camera_params (a clamp, or persisted settings): say so where the person
        who ran the experiment is looking. ASCII only: ExptBuilder pipes stdout
        through cp1252."""
        record = reply.get("camera_overrides") or {}
        fields = record.get("fields") or {}
        if not fields:
            return
        lines = [f"!! CAMERA SETTINGS DIFFER FROM camera_params ({record.get('camera_key', '?')}):"]
        for key, f in fields.items():
            lines.append(f"!!   {key}: requested {f.get('requested')!r} -> applied "
                         f"{f.get('applied')!r} ({f.get('origin', '?')})")
        lines.append("!! The run file records this in its root attribute camera_overrides;")
        lines.append("!! camera_params/ in the file is the request, not what the camera ran.")
        bar = "!" * 72
        print("\n" + bar + "\n" + "\n".join(lines) + "\n" + bar + "\n")

    @staticmethod
    def _print_grab_failure(reason) -> None:
        """liveOD's camera grab for this run has ended early (a lost frame, a
        camera timeout): nothing of the run is recorded from here on. Said once,
        where the person who ran the experiment is looking; ASCII only
        (ExptBuilder pipes stdout through cp1252)."""
        reason = str(reason).encode("ascii", "replace").decode("ascii")
        bar = "!" * 72
        print(f"\n{bar}\n"
              f"!! liveOD: THE CAMERA STOPPED RECORDING THIS RUN\n"
              f"!!   {reason}\n"
              f"!! The frames of the shots from here on are NOT recorded. The run is not\n"
              f"!! stopped by this: END_RUN will save what arrived, marked\n"
              f"!! data_complete=False. Abort the run if its images matter.\n"
              f"{bar}\n")

    @staticmethod
    def _wait_not_ready_yet(reply: dict) -> bool:
        """Is a WAIT_CAM_READY reply "not ready yet, ask again" rather than a
        failure to raise at once? A server that says ``timed_out`` decides it. A
        server that answers ``reset_requested`` or ``stale_run`` but no
        ``timed_out`` is one whose failures come without it: a failure, whatever
        its text (a camera error may well say "timeout"). Only a reply with none
        of these, from a server older than all of them, is judged by its text."""
        if "timed_out" in reply:
            return bool(reply["timed_out"])
        if "reset_requested" in reply or "stale_run" in reply:
            return False
        return "timeout" in str(reply.get("error", "")).lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_run(self, payload: dict) -> dict:
        """Send INIT_RUN.  Returns ``{"run_id": int, "filepath": str, "run_token": str}``
        (no ``run_token`` from an older server).

        The server creates and fully populates the HDF5 data file before
        replying, so this can take a while on a slow/mapped data drive — hence
        the raised receive timeout rather than the 5 s default.
        """
        payload["tag"] = "INIT_RUN"
        self.last_reset_requested = False    # the server clears its flag on INIT_RUN
        self._run_token = ""
        self._grab_failure_warned = False
        reply = self._send_recv(payload, rcvtimeo_ms=60_000)
        if not reply.get("ok"):
            raise RuntimeError(
                f"[LiveODClient] INIT_RUN failed: {reply.get('error')}"
            )
        self._run_token = str(reply.get("run_token") or "")
        self.server_features = dict(reply.get("features") or {})
        self._run_open = True
        self._exit_notified = False
        if not getattr(self, "_exit_hook_registered", False):
            atexit.register(self.notify_exit)
            self._exit_hook_registered = True
        self._print_camera_overrides(reply)
        return reply

    def wait_cam_ready(self, timeout: float = 60.0) -> bool:
        """Block until liveOD confirms the camera is ready.

        Returns True when the camera is ready and False when a reset was
        requested while waiting (``last_reset_requested`` is then set) -- the
        caller aborts the run. Raises ValueError on timeout or failure.

        The wait is asked for in slices of ``CAM_READY_SLICE_S``. The server
        handles one request at a time, so one long WAIT_CAM_READY would hold off
        a RESET from the remote viewer for the whole wait, and a camera that was
        reset never becomes ready: a reset used to cost the full ``timeout``.
        An older server (no ``timed_out`` / ``reset_requested`` in its replies)
        still works: its slice timeouts are recognised by their error text
        (_wait_not_ready_yet). A newer server's failure raises at once, whatever
        its text.
        """
        deadline = time.monotonic() + timeout
        last_error = None       # which wait the server was stuck in
        asked = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 and asked:
                raise ValueError(
                    f"[LiveODClient] Camera ready timed out after {timeout:.0f} s"
                    f" (server: {last_error})."
                )
            slice_s = min(CAM_READY_SLICE_S, max(remaining, 0.0))
            asked = True
            t_asked = time.monotonic()
            reply = self._send_recv(
                self._for_this_run({"tag": "WAIT_CAM_READY", "timeout": slice_s}),
                rcvtimeo_ms=int((slice_s + 5.0) * 1000),
            )
            if reply.get("reset_requested"):
                self.last_reset_requested = True
                return False
            if reply.get("ok") and reply.get("ready"):
                # settings the camera was clamped to are known once it is armed
                self._print_camera_overrides(reply)
                return True
            # A newer INIT_RUN took liveOD over (stale_run), or the camera failed:
            # not a slice timeout, so it falls through to the raise below.
            if not self._wait_not_ready_yet(reply):
                raise ValueError(
                    f"[LiveODClient] Camera ready failed: {reply.get('error')}"
                )
            last_error = reply.get("error")
            # A "not ready" that came straight back was not a slice the server
            # waited out (an old server failing fast with "timeout" in its text):
            # wait the slice out here instead of asking again at once.
            spent = time.monotonic() - t_asked
            if spent < 0.5 * slice_s:
                time.sleep(max(0.0, min(slice_s - spent, deadline - time.monotonic())))

    def shot_complete(
        self, shot_idx: int, N_shots_total: int, xvar_values: dict,
        shot_conditions: dict = None,
    ) -> bool:
        """Notify the server that a shot has completed.

        Returns True if the server has a pending reset request so the
        caller can abort the run at the shot boundary.

        ``shot_conditions`` ({key: float}, optional) is what the shot recorded
        about itself -- liveOD picks the absorption cross section from the
        outer-coil current in it. An older server ignores the extra key.

        Falls back to an explicit POLL if the server's SHOT_COMPLETE reply
        does not include ``reset_requested`` (older server builds that
        pre-date the field).
        """
        reply = self._send_recv(self._for_this_run(
            {
                "tag": "SHOT_COMPLETE",
                "shot_idx": shot_idx,
                "N_shots_total": N_shots_total,
                "xvar_values": xvar_values,
                "shot_conditions": dict(shot_conditions or {}),
            }
        ))
        if reply.get("stale_run"):
            # liveOD has started another run since this one's INIT_RUN (or was
            # restarted since, and does not know it), and no longer records this
            # one: stop it like a reset would.
            why = ("liveOD does not know this run (was it restarted?)" if reply.get("unknown_run")
                   else "liveOD is serving a newer run")
            print(f"[LiveODClient] {reply.get('error')} -- {why}; "
                  f"this run is stopped and nothing more of it is recorded.")
            self.last_reset_requested = True
            return True
        if reply.get("grab_failure") and not getattr(self, "_grab_failure_warned", False):
            # the camera's grab ended early: said once per run; the run goes on
            self._grab_failure_warned = True
            self._print_grab_failure(reply["grab_failure"])
        self.last_adjust_values = reply.get('adjust_values', {})
        if "reset_requested" in reply:
            # cached for Scribe._check_for_abort_signal (top of the scan loop)
            self.last_reset_requested = bool(reply["reset_requested"])
            return self.last_reset_requested
        # Old server: reset_requested field not present — fall back to POLL.
        print("[LiveODClient] shot_complete: reply missing 'reset_requested' field — "
              "falling back to poll_reset() (liveOD GUI may need a restart).")
        return self.poll_reset()

    def end_run(self, payload: dict) -> bool:
        """Send END_RUN with final params and DataVault data.

        The server may take several seconds to write the HDF5 file — and if
        the data drive drops out it retries the save with backoff — so the
        receive timeout is raised to 10 minutes.  This must stay comfortably
        above the server's worst case (DATA_SAVER_TIMEOUT plus DataSaver's
        retry budget), or the client gives up on a save that is still running.
        """
        payload["tag"] = "END_RUN"
        if self.supports("async_save"):
            # The server acknowledges at once and saves on a thread of its
            # own; this polls it, so a long save never looks like a dead
            # server, and the server's loop stays free meanwhile.
            payload["async_save"] = True
            reply = self._send_recv(self._for_this_run(payload), rcvtimeo_ms=60_000)
            if reply.get("ok") and reply.get("saving"):
                reply = self._wait_for_save()
        else:
            reply = self._send_recv(self._for_this_run(payload), rcvtimeo_ms=600_000)
        self._run_open = False               # liveOD answered: the run is closed there
        if not reply.get("ok"):
            raise RuntimeError(
                f"[LiveODClient] END_RUN failed: {reply.get('error')}"
            )
        self.last_end_run_reply = dict(reply)
        incomplete = reply.get("incomplete")
        if incomplete:
            # The file exists, with what arrived, marked data_complete=False.
            # Said here as well as in the file: this is the experiment's own
            # terminal, where the person (or agent) who ran it is looking.
            print(
                f"\n{'!' * 72}\n"
                f"!! RUN SAVED INCOMPLETE: {incomplete.get('reason', '')}\n"
                f"!! {incomplete.get('images_received', '?')} of "
                f"{incomplete.get('images_expected', '?')} images arrived. The file is marked\n"
                f"!! data_complete=False; its images are in arrival order and do not line up\n"
                f"!! with the shots. Do not analyze it as a complete run.\n"
                f"{'!' * 72}\n"
            )
        return True

    def _wait_for_save(self) -> dict:
        """Poll SAVE_STATUS until the asynchronous save is over; the END_RUN
        reply as the synchronous path would have given it (``ok``,
        ``incomplete``, ``error``), plus ``save_s``."""
        t0 = time.monotonic()
        last_phase, t_phase, failures = None, t0, 0
        while True:
            time.sleep(SAVE_POLL_S)
            try:
                st = self._send_recv(self._for_this_run({"tag": "SAVE_STATUS"}))
                failures = 0
            except ConnectionError:
                failures += 1
                if failures >= 3:
                    raise
                continue
            state = st.get("state")
            if state in ("saved", "saved_incomplete"):
                return {"ok": True, "incomplete": st.get("incomplete"),
                        "save_s": st.get("elapsed_s")}
            if state == "failed":
                return {"ok": False, "error": st.get("error"), "save_s": st.get("elapsed_s")}
            if state != "saving":
                raise RuntimeError(
                    f"[LiveODClient] END_RUN: liveOD reports no save in progress "
                    f"for this run ({'superseded' if st.get('stale_run') else state!r})")
            phase = st.get("phase")
            now = time.monotonic()
            if phase != last_phase:
                last_phase, t_phase = phase, now
            if now - t_phase > SAVE_STALL_S or now - t0 > SAVE_MAX_S:
                raise RuntimeError(
                    f"[LiveODClient] END_RUN: liveOD's save has been in phase "
                    f"{phase!r} for {now - t_phase:.0f} s ({now - t0:.0f} s in all); "
                    f"giving up on it")

    def poll(self) -> dict:
        """The server's POLL reply: run state, run id, shot and frame counts,
        how the last run ended. Read-only; raises on a network error."""
        return self._send_recv({"tag": "POLL"})

    def get_log(self, run_id=None, seq=None, since=None,
                min_level: int = logging.DEBUG, limit: int = 2000) -> dict:
        """The server's log records for one run, from its in-memory buffer.

        ``run_id``: a run id, ``None`` for the current or last run, or ``"all"``
        for every buffered record.  ``seq``: the server's own run counter (from
        ``list_runs``), which tells unsaved runs (all run id 0) apart.  ``since``:
        epoch seconds, later records only.  ``min_level``: a ``logging`` level.
        Returns ``{"run": {...}, "records": [{"t", "level", "levelname", "msg",
        "thread", ...}, ...]}``.  Raises ``LookupError`` when the server has no
        such run in its buffer (it starts empty when the server starts).
        """
        msg = {"tag": "GET_LOG", "min_level": int(min_level), "limit": int(limit)}
        if run_id is not None:
            msg["run_id"] = run_id
        if seq is not None:
            msg["seq"] = int(seq)
        if since is not None:
            msg["since"] = float(since)
        reply = self._send_recv(msg)
        if not reply.get("ok"):
            raise LookupError(f"[LiveODClient] GET_LOG: {reply.get('error')}")
        return {"run": reply.get("run"), "records": list(reply.get("records", []))}

    def list_runs(self, limit: int = 50) -> list:
        """The runs this server process has seen, oldest first, each with its
        ``seq``, ``run_id``, ``name``, start/end times and ``outcome`` (``saved``,
        ``saved_incomplete``, ``discarded``, ``save_failed``, ``nothing_written``,
        ``in_progress``)."""
        reply = self._send_recv({"tag": "GET_LOG", "runs": True, "limit": int(limit)})
        if not reply.get("ok"):
            raise LookupError(f"[LiveODClient] GET_LOG runs: {reply.get('error')}")
        return list(reply.get("runs", []))

    def abort_run(self) -> None:
        """Notify the server that the experiment has acknowledged the abort.

        Best-effort — all errors are suppressed because the experiment is
        already in the process of terminating and must not block. Carries the
        run's token, so an abort from a run that liveOD has since replaced is
        ignored instead of discarding the newer run's file.
        """
        try:
            self._send_recv(self._for_this_run({"tag": "ABORT_RUN"}))
            self._run_open = False           # it reached liveOD; else notify_exit still tells it
        except Exception:
            pass

    def poll_reset(self) -> bool:
        """Ask the server if a reset has been requested.

        Called as an RPC between shots. Returns True if the experiment
        should abort. Returns False on any network error so a transient
        glitch doesn't kill the run.
        """
        try:
            reply = self._send_recv({"tag": "POLL"})
            if not reply.get("ok", False):
                # Server returned an error — likely an old build that doesn't
                # recognise the POLL tag.  Warn once so the user knows.
                print(f"[LiveODClient] poll_reset: unexpected server reply: {reply}"
                      "\n  → liveOD GUI may need to be restarted to pick up new code.")
            return bool(reply.get("reset_requested", False))
        except Exception as exc:
            print(f"[LiveODClient] poll_reset: network error (returning False): {exc}")
            return False

    # ------------------------------------------------------------------
    # Cameras liveOD holds (remote open / release)
    # ------------------------------------------------------------------

    def cameras(self) -> dict:
        """``{camera_key: {"state", "camera_type", "serial_no"}}`` for every camera
        on liveOD's bar.  ``state`` is the bar's: ``closed``, ``loading``, ``open``,
        ``grabbing``, ``failed``.  Raises ``LookupError`` against a server that
        predates camera state in POLL."""
        reply = self.poll()
        if not reply.get("ok", False):
            raise RuntimeError(f"[LiveODClient] POLL: {reply.get('error')}")
        cams = reply.get("cameras")
        if cams is None:
            raise LookupError("[LiveODClient] this liveOD server does not report camera "
                              "state (restart it with current code)")
        return dict(cams)

    def camera_control(self, camera_key: str, action: str) -> dict:
        """Ask liveOD to ``open`` / ``close`` / ``toggle`` a camera.  The GUI does
        it asynchronously; confirm with ``wait_camera_state``.  Raises
        ``RuntimeError`` when the server refuses (a run is using that camera, or
        any open during a run)."""
        reply = self._send_recv({"tag": "CAMERA_CONTROL", "camera_key": camera_key,
                                 "action": action})
        if not reply.get("ok", False):
            raise RuntimeError(f"[LiveODClient] CAMERA_CONTROL {camera_key} -> {action}: "
                               f"{reply.get('error')}")
        return reply

    def wait_camera_state(self, camera_key: str, states, timeout: float = 10.0,
                          poll_s: float = 0.2) -> str:
        """Block until liveOD reports ``camera_key`` in one of ``states``; return
        that state.  ``TimeoutError`` otherwise, ``KeyError`` for an unknown key."""
        states = (states,) if isinstance(states, str) else tuple(states)
        deadline = time.monotonic() + timeout
        while True:
            cams = self.cameras()
            if camera_key not in cams:
                raise KeyError(f"[LiveODClient] liveOD has no camera {camera_key!r}; "
                               f"it has {sorted(cams)}")
            state = cams[camera_key].get("state")
            if state in states:
                return state
            if time.monotonic() >= deadline:
                raise TimeoutError(f"[LiveODClient] {camera_key} still {state!r} after "
                                   f"{timeout:g} s (wanted {states})")
            time.sleep(poll_s)

    def release_camera(self, camera_key: str, timeout: float = 10.0) -> dict:
        """Make liveOD close ``camera_key`` so another process (e.g. the beacon
        Basler server) can open the device.  Returns the camera's entry once
        closed.  The server refuses while a run is using that camera."""
        cams = self.cameras()
        if camera_key not in cams:
            raise KeyError(f"[LiveODClient] liveOD has no camera {camera_key!r}; "
                           f"it has {sorted(cams)}")
        if cams[camera_key].get("state") == "closed":
            return cams[camera_key]
        self.camera_control(camera_key, "close")
        state = self.wait_camera_state(camera_key, ("closed", "failed"), timeout)
        if state == "failed":
            raise RuntimeError(f"[LiveODClient] liveOD could not close {camera_key} "
                               f"(see its log)")
        return self.cameras()[camera_key]

    def open_camera(self, camera_key: str, timeout: float = 30.0) -> dict:
        """Make liveOD (re)open ``camera_key``.  Refused during a run."""
        cams = self.cameras()
        if camera_key not in cams:
            raise KeyError(f"[LiveODClient] liveOD has no camera {camera_key!r}; "
                           f"it has {sorted(cams)}")
        if cams[camera_key].get("state") in ("open", "grabbing"):
            return cams[camera_key]
        self.camera_control(camera_key, "open")
        state = self.wait_camera_state(camera_key, ("open", "grabbing", "failed"), timeout)
        if state == "failed":
            raise RuntimeError(f"[LiveODClient] liveOD could not open {camera_key}: is the "
                               f"device held by another process (beacon server, a viewer)?")
        return self.cameras()[camera_key]

    def close(self):
        """Release ZMQ resources."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
        try:
            self._context.term()
        except Exception:
            pass
