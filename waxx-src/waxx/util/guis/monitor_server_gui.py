import socket
import json
import logging
import time
from dataclasses import dataclass, field
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QFont, QIcon, QPixmap, QPainter

from waxx.util.device_state.monitor_manager import MonitorManager
from waxx.util.comms_server.comm_server import UdpServer, STATES, ReadyBit
from waxx.util.comms_server.state_broadcast import StateBroadcaster
from waxx.util.comms_server.hardware_id import monitor_server_id
from beacon.discovery.client import discover
import os
import threading

from waxx.util.device_state.state_file_io import (
    read_state, apply_delta, apply_deltas, replace_sections)
from waxx.util.device_state.op_queue import OpQueue
from waxx.util.device_state.op_journal import OpJournal
from waxx.util.device_state.op_runner import OpRunner
from waxx.util.device_state.state_reset import StateReset
from waxx.util.device_state.run_loop import RunLoop, active_loop
from waxx.util.device_state import connections as conns
from waxx.util.device_state.connections import ConnectionService
from waxx.util.device_state.slm_reinit import SlmReinitService

log = logging.getLogger(__name__)

_STATE_NAMES = {STATES.READY: "READY", STATES.LOADING: "LOADING",
                STATES.NOT_READY: "NOT_READY"}


def _state_name(state) -> str:
    return _STATE_NAMES.get(state, str(state))


def _run_name(run_id, expt) -> str:
    """"run 81234 (hf_bec)"; a run without an id (suppress_live_od: 0 or
    None) is named by its file alone."""
    expt = str(expt or "")
    if run_id:
        return f"run {run_id} ({expt or 'experiment'})"
    return expt or "an experiment"


@dataclass
class MonitorStatus:
    """What the monitor server reports about the monitor experiment.

    ``state`` is the :class:`ReadyBit` value clients poll with ``status``;
    the rest is the detail behind it, served by ``status_json``:

    * ``sub_state`` -- machine-readable reason for the state: ``"starting"``
      (LOADING), ``"running"`` (READY), and for NOT_READY ``"never_started"``,
      ``"stopped_on_request"``, or the classification
      :class:`~waxx.util.device_state.monitor_manager.MonitorManager` made of
      the last exit (``"interrupted_by_run"``, ``"exited"``, ``"failed"``,
      ``"preflight_failed"``).
    * ``reason`` -- human-readable detail, ``""`` when there is none.
    * ``since`` -- epoch seconds when the current (state, sub_state) began.
    * ``pid`` -- pid of the monitor experiment process, ``None`` when it is
      not running.
    * ``expt_path`` -- the monitor experiment file the server launches.

    The object is written by the owning server (main thread) and read by the
    TCP responder thread; plain attribute writes are atomic under the GIL and a
    reader seeing one field a tick stale is harmless.
    """

    state: int = STATES.NOT_READY
    sub_state: str = "never_started"
    reason: str = ""
    since: float = field(default_factory=time.time)
    pid: int | None = None
    expt_path: str = ""

    @property
    def state_name(self) -> str:
        return _state_name(self.state)

    def set_state(self, state, sub_state: str | None = None,
                  reason: str | None = None) -> bool:
        """Set the state; ``since`` moves only when state or sub_state change.

        ``sub_state=None`` keeps the current sub_state.  ``reason=None`` keeps
        the current reason unless the (state, sub_state) changed, in which case
        it is cleared so a stale reason never outlives its state.  Returns
        whether (state, sub_state) changed.
        """
        changed = (state != self.state) or (
            sub_state is not None and sub_state != self.sub_state)
        self.state = state
        if sub_state is not None:
            self.sub_state = sub_state
        if reason is not None:
            self.reason = reason
        elif changed:
            self.reason = ""
        if changed:
            self.since = time.time()
        return changed

    def to_dict(self) -> dict:
        """The ``status_json`` reply, keyed exactly as ``MonitorClient.get_status`` documents."""
        return {
            "state": int(self.state),
            "state_name": self.state_name,
            "sub_state": self.sub_state,
            "reason": self.reason,
            "since": self.since,
            "pid": self.pid,
            "expt_path": self.expt_path,
        }


# Older code constructs ``Status()`` / reads ``status.state``; keep the name.
Status = MonitorStatus


class MonitorUDPServer(UdpServer):
    """TCP responder + sole writer of the device-state JSON.

    Plain-text commands (newline framed, one reply line each):

    * ``status`` -- the bare ``ReadyBit`` integer as a string (legacy poll).
    * ``status_json`` -- the structured status, see :class:`MonitorStatus`.
      Neither status command is logged or forwarded to ``message_received``;
      the Device Control GUI polls at 2 Hz.
    * ``reset`` -- emit ``reset_signal`` (owner restarts the monitor).
    * ``stop`` -- emit ``stop_signal`` (owner stops the monitor and leaves it
      stopped until the next ``reset`` / ``run complete``); reply ``OK``.
    * ``run complete`` / ``monitor ready`` -- forwarded to the owner through
      ``message_received``; reply is the state integer.

    Structured JSON requests from clients:

    * ``{"type": "update", "device_type", "device_name", "changes"}`` — merge a
      delta into the JSON atomically, bump the version, broadcast the change.
    * ``{"type": "get_state"}`` — return the full snapshot + current version.
    * ``{"type": "get_version"}`` — return just the current version; the monitor
      experiment polls this at ~10 Hz and only re-reads the JSON from the share
      when it has moved.
    * ``{"type": "update_batch", "updates": [...], "origin"}`` — several deltas
      (the monitor writing back the channels a composite op changed) in ONE
      atomic write -- all land or none; each still gets its own version and
      broadcast.
    * ``{"type": "replace_state", "config", "run_id"}`` — an experiment's
      end-of-run state (``end()``); replaces the channel sections at once,
      marks the state trusted, broadcasts ``state_reset``.
    * ``{"type": "abort_state", "config", "run_id", "cause", "trusted",
      "caveat"}`` — an aborted run's state, from the kernel at the abort
      (scan()'s exception handler); as ``replace_state``, but trusted only
      when ``trusted`` is true.

    Composite ops (see :mod:`waxx.util.device_state.composite`; bookkeeping in
    :class:`~waxx.util.device_state.op_queue.OpQueue`):

    * ``register_ops`` — the monitor experiment's compiled op table.
    * ``op`` — a GUI request; refused unless the monitor is READY, has
      registered the op with the same signature, and no run is starting.
    * ``poll`` — the monitor's per-loop request: version + queued ops.
    * ``op_done`` — outcomes from the monitor, broadcast as ``op_result``.
    * ``op_status`` — one request's state (the GUI's fallback for a lost
      broadcast).
    * ``busy`` — the monitor is playing out a long op for N s.
    * ``run_pending`` — an experiment finished ``prepare()`` and is about to
      take the core: ops are refused from now until it ends (or 2 min pass
      without it taking the core).
    * ``trust_ack`` — an operator says the hardware matches the state file.
    * ``reset_state`` — run the reset experiment (``reset_expt_path``, see
      :class:`~waxx.util.device_state.state_reset.StateReset`): it takes the
      core, sets the hardware, and its end state marks the state trusted.
      Refused while a run is starting or a reset is running.  Progress is
      broadcast as ``reset_run``.
    * ``run_loop`` (``action`` ``start`` / ``stop``, ``loop``) — run one of
      the server's ``run_loops`` back to back (see
      :mod:`waxx.util.device_state.run_loop`); one loop at a time.  Changes
      are broadcast as ``run_loop``; ``status_json`` has ``run_loops``.  A
      ``reset`` (monitor restart) request ends a running loop; the loop's
      end asks for the monitor through ``start_monitor_signal``.
    * ``output`` (``kind`` ``run_loop`` with ``key``, or ``reset``;
      ``after``) — the terminal output of a loop's runs or of the reset
      experiment after line ``after`` (see
      :class:`~waxx.util.device_state.output_log.OutputLog`); the Sequences
      tab's log view polls it while open.  Not logged or journaled.
    * ``run_scene`` / ``cancel_scene``, ``arm_watchdog`` / ``extend_watchdog``
      / ``disarm_watchdog`` — see :mod:`waxx.util.device_state.op_runner`.
    * ``get_journal`` — recent journal records (``n``, or ``since``).

    Host-side connections this server holds between runs (the tweezer AWG;
    :class:`~waxx.util.device_state.connections.ConnectionService`, each in
    its own agent process): released before the reply to ``run_pending``, and
    when a run takes the core; opened again when the monitor is running.
    Their states are broadcast as ``connections`` and served in
    ``status_json`` / ``get_state``.

    * ``connection`` — a GUI's ``connect`` / ``disconnect`` of one (``key``,
      ``action``); a connect is refused while a run is starting or running.
    * ``connection_call`` — a driver command on an open connection (``key``,
      ``cmd``, ``kwargs``): the monitor's "Apply traps" writes the AWG tones
      this way, then pulses the AWG trigger from its kernel.

    The SLM's hourly reinit (``slm_reinit``, a
    :class:`~waxx.util.device_state.slm_reinit.SlmReinitConfig`): the SLM
    server marks it due; this server asks for it only while the monitor is
    running and no run is starting, and ``run_pending`` waits (≤ 1 s) for a
    request being sent.  Served in ``status_json`` as ``slm_reinit`` and
    broadcast as ``{"type": "slm_reinit"}``.

    Trust: when an experiment takes the core (the monitor is interrupted by a
    run) the state file stops describing the hardware until that run's
    ``end()`` sends its end state (or, aborted inside its scan, sends
    ``abort_state``).  A run that dies elsewhere never does, so from the
    interruption until ``replace_state`` (or ``trust_ack``) the state is
    *untrusted*: GUIs say so, and coil ops want a measured current.  It is
    kept in the file's metadata so a server restart does not forget it.
    ``reset_state`` ends it the way a normal run does -- through the reset
    experiment's own end state, never by itself.

    The version starts from the current epoch seconds so that a server restart
    always yields versions higher than any value a client still holds (forcing
    a clean resync rather than ignoring "older" updates).
    """

    reset_signal = pyqtSignal()
    stop_signal = pyqtSignal()
    #: A run loop ended: start the monitor unless it is running (the owner
    #: decides; the argument says why).
    start_monitor_signal = pyqtSignal(str)

    #: A run_pending that never became a run (its prepare succeeded, its
    #: run() never took the core) stops fencing ops after this long.
    RUN_PENDING_TTL_S = 120.0

    def __init__(self, config_file_path=None, journal_dir=None, reset_expt_path=None,
                 run_loops=(), connections=(), slm_reinit=None):
        super().__init__(host="0.0.0.0", port=0, server_id=monitor_server_id())

        self.status = MonitorStatus()
        self._print_connections_bool = False

        self.config_file_path = config_file_path
        self._version = int(time.time())
        self._broadcaster = StateBroadcaster()
        self.ops = OpQueue()
        self.journal = OpJournal(journal_dir)

        # The state file's content, kept in memory; re-read only when its
        # mtime says someone else wrote it (see _state()).
        self._state_lock = threading.RLock()
        self._state_cache: dict | None = None
        self._state_mtime = None

        self._run_pending: dict | None = None
        self._last_monitor_state = None
        self._trust = {"trusted": True, "reason": "", "since": time.time()}
        try:
            meta = self._state().get("metadata") or {}
            if isinstance(meta.get("state_trust"), dict):
                self._trust = dict(meta["state_trust"])
        except Exception:
            pass

        self._runner_lock = threading.RLock()
        self.runner = OpRunner(self._submit_internal, self.ops.result,
                               self._broadcaster_send, journal=self.journal)
        self._runner_stop = threading.Event()
        self._runner_thread = None

        self.reset = StateReset(reset_expt_path, on_change=self._on_reset_change,
                                journal=self.journal)

        # Experiments the GUIs may run back to back -- only these files.
        self.loops = {spec.key: RunLoop(spec, fence=self._current_run_pending,
                                        busy=self._loop_busy,
                                        start_monitor=self.start_monitor_signal.emit,
                                        on_change=self._on_loop_change, journal=self.journal)
                      for spec in run_loops}

        # Host-side connections (the tweezer AWG), held here between runs.
        # Bad definitions cost the connections, never the server.
        try:
            self.connections = ConnectionService(connections,
                                                 on_change=self._on_connections_change,
                                                 log=lambda text: log.info("%s", text))
        except ValueError as e:
            log.error("Connection definitions rejected (%s); running without them.", e)
            self.connections = ConnectionService((), log=lambda text: log.info("%s", text))

        # The SLM's hourly reinit, asked for only while the machine is idle
        # (a SlmReinitConfig, or None for a machine without an SLM server).
        self.slm_reinit = None
        if slm_reinit is not None:
            self.slm_reinit = SlmReinitService(slm_reinit, blocker=self._slm_reinit_blocker,
                                               on_change=self._on_slm_reinit_change,
                                               journal=self.journal,
                                               log=lambda text: log.info("%s", text))

    def on_message_received(self,message):
        m = message.strip()
        if m.startswith("{"):
            # Structured (JSON) requests are fully handled in generate_reply.
            return
        if m in ('status', 'status_json'):
            # Polled continuously; never logged, never forwarded.
            return
        if m == 'reset':
            loop = active_loop(self.loops.values())
            if loop is not None:
                loop.note_external("a client asked to (re)start the monitor, which takes "
                                   "the core")
            self.reset_signal.emit()
        if m == 'stop':
            log.info("Stop requested by a client: stopping the monitor experiment.")
            self.stop_signal.emit()
            return
        if "monitor ready" in m and self._run_pending is not None:
            # A monitor has started since: whatever run was pending is over.
            self._clear_run_pending("a monitor started")
        self.journal.record("message", text=m)
        self.message_received.emit(message)

    def generate_reply(self, message):
        m = message.strip()
        # Every request (the GUIs' 1 Hz status polls included) sweeps the op
        # queue, so an op the monitor never took fails on time even when the
        # monitor itself is gone and no longer polling.  (The runner thread
        # sweeps too, when the server is running.)
        self._broadcast_results(self.ops.expire())
        if m.startswith("{"):
            return self._handle_structured(m)
        if m == 'status_json':
            detail = self.status.to_dict()
            detail.update(self._extra_status())
            return json.dumps(detail)
        if m == 'stop':
            return "OK"
        return str(int(self.status.state))

    def _extra_status(self) -> dict:
        with self._runner_lock:
            runner = self.runner.info()
        return {"composite_ops": self.ops.info(), "trust": dict(self._trust),
                "run_pending": dict(self._run_pending) if self._run_pending else None,
                "runner": runner, "reset": self.reset.info(),
                "connections": self.connections.snapshot(),
                "slm_reinit": (self.slm_reinit.snapshot() if self.slm_reinit is not None
                               else None),
                "run_loops": {key: loop.info() for key, loop in self.loops.items()}}

    def _handle_structured(self, raw):
        try:
            obj = json.loads(raw)
        except Exception:
            return json.dumps({"status": "error", "msg": "invalid json"})
        mtype = obj.get("type")
        if mtype == "get_state":
            return self._reply_get_state()
        if mtype == "get_version":
            return json.dumps({"status": "ok", "version": self._version})
        if mtype == "update":
            return self._reply_update(obj)
        if mtype == "update_batch":
            return self._reply_update_batch(obj)
        if mtype == "poll":
            return json.dumps({"status": "ok", "version": self._version,
                               "ops": self.ops.pop(),
                               "registered": self.ops.registered})
        if mtype == "connection":
            return json.dumps(self._reply_connection_request(obj))
        if mtype == "connection_call":
            return json.dumps(self._reply_connection_call(obj))
        if mtype == "op":
            return json.dumps(self._submit_internal(obj, "gui"))
        if mtype == "op_done":
            self._broadcast_results(self.ops.done(obj.get("results")))
            return json.dumps({"status": "ok"})
        if mtype == "busy":
            try:
                seconds = float(obj.get("seconds", 0.))
            except (TypeError, ValueError):
                return json.dumps({"status": "error", "msg": "bad seconds"})
            self.ops.set_busy(seconds)
            self._broadcaster.send({"type": "busy", "seconds": seconds})
            return json.dumps({"status": "ok"})
        if mtype == "replace_state":
            return self._reply_replace_state(obj)
        if mtype == "abort_state":
            return self._reply_abort_state(obj)
        if mtype == "run_pending":
            return self._reply_run_pending(obj)
        if mtype in ("run_withdrawn", "clear_run_pending"):
            return self._reply_clear_run_pending(obj, by_operator=mtype == "clear_run_pending")
        if mtype == "trust_ack":
            who = str(obj.get("operator") or obj.get("client") or "?")
            self._set_trust(True, f"acknowledged on the Device Control GUI by {who}")
            return json.dumps({"status": "ok", "trust": dict(self._trust)})
        if mtype == "reset_state":
            return json.dumps(self._reply_reset_state(obj))
        if mtype == "run_loop":
            return json.dumps(self._reply_run_loop(obj))
        if mtype == "output":
            return json.dumps(self._reply_output(obj))
        if mtype == "get_journal":
            if obj.get("since"):
                entries = self.journal.since((str(obj["since"]),))
            else:
                entries = self.journal.tail(int(obj.get("n", 200)))
            return json.dumps({"status": "ok", "entries": entries,
                               "path": self.journal.path_for()})
        if mtype in ("run_scene", "cancel_scene", "arm_watchdog", "extend_watchdog",
                     "disarm_watchdog"):
            return json.dumps(self._runner_request(mtype, obj))
        if mtype == "op_status":
            try:
                seq = int(obj.get("seq"))
            except (TypeError, ValueError):
                return json.dumps({"status": "error", "msg": "bad seq"})
            return json.dumps(self.ops.status(seq))
        if mtype == "register_ops":
            reply = self.ops.register(obj)
            if reply.get("status") == "ok":
                log.info("Composite ops registered by the monitor: %d (definitions %s).",
                         reply.get("count", 0), obj.get("hash", "?"))
            else:
                log.warning("Composite op registration refused: %s", reply.get("msg"))
            return json.dumps(reply)
        return json.dumps({"status": "error", "msg": f"unknown type {mtype}"})

    # --- composite ops --------------------------------------------------------

    def _submit_internal(self, obj: dict, origin: str = "gui") -> dict:
        """The one gate every op passes -- from a GUI, a scene or a watchdog."""
        client = str(obj.get("client", ""))
        refusal = None
        if self.status.state != STATES.READY:
            state = self.status.state_name.lower().replace("_", " ")
            sub = str(self.status.sub_state).replace("_", " ")
            refusal = (f"the monitor is {state} ({sub}) -- composite ops run only "
                       "while it is ready")
        else:
            pending = self._current_run_pending()
            if pending is not None:
                refusal = (f"a run is starting (run {pending.get('run_id')}, "
                           f"{pending.get('expt') or 'experiment'}) -- composite ops are "
                           "refused until it ends")
        if refusal is not None:
            reply = {"status": "error", "msg": refusal}
        else:
            reply = self.ops.submit(obj, client=client, origin=origin)
        if reply.get("status") == "ok":
            if not reply.get("duplicate"):
                log.info("[OP] %s queued (#%s) from %s (%s)", obj.get("op"), reply["seq"],
                         client or "?", origin)
                self.journal.record("op_submit", seq=reply["seq"], op=obj.get("op"),
                                    args=obj.get("args"), client=client,
                                    operator=obj.get("operator", ""), origin=origin)
        else:
            log.warning("[OP] %s refused: %s", obj.get("op"), reply.get("msg"))
            self.journal.record("op_refused", op=obj.get("op"), args=obj.get("args"),
                                client=client, operator=obj.get("operator", ""),
                                origin=origin, msg=reply.get("msg"))
        return reply

    def _broadcaster_send(self, payload: dict) -> None:
        self._broadcaster.send(payload)

    def _broadcast_results(self, results) -> None:
        for result in results or []:
            if result.get("ok"):
                log.info("[OP] %s #%s done (%.2f s)", result["op"], result["seq"],
                         result.get("elapsed") or 0.0)
            else:
                log.warning("[OP] %s #%s: %s", result["op"], result["seq"], result["text"])
            self.journal.record("op_result", seq=result["seq"], op=result["op"],
                                status=result["status"], text=result["text"],
                                args=result.get("arg_values"), origin=result.get("origin"),
                                client=result.get("client"), operator=result.get("operator"),
                                elapsed=result.get("elapsed"))
            self._broadcaster.send(result)

    def _runner_request(self, mtype: str, obj: dict) -> dict:
        with self._runner_lock:
            if mtype == "run_scene":
                return self.runner.start_scene(obj, client=str(obj.get("client", "")),
                                               operator=str(obj.get("operator", "")))
            if mtype == "cancel_scene":
                return self.runner.cancel_scene(obj.get("id"))
            if mtype == "arm_watchdog":
                return self.runner.arm_watchdog(obj)
            if mtype == "extend_watchdog":
                return self.runner.extend_watchdog(str(obj.get("device", "")),
                                                   str(obj.get("operator", "")))
            return self.runner.disarm_watchdog(str(obj.get("device", "")))

    def runner_tick(self) -> None:
        """One step of the scene/watchdog runner and the expiry sweep."""
        self._broadcast_results(self.ops.expire())
        with self._runner_lock:
            self.runner.tick()
        self._current_run_pending()        # lets a stale fence lapse

    def _runner_loop(self) -> None:
        while not self._runner_stop.wait(0.2):
            try:
                self.runner_tick()
            except Exception:
                log.exception("Composite op runner tick failed")

    def run(self):
        self._runner_thread = threading.Thread(target=self._runner_loop, daemon=True,
                                               name="monitor-op-runner")
        self._runner_thread.start()
        self.connections.start()
        if self.slm_reinit is not None:
            self.slm_reinit.start()
        super().run()

    # --- run fence and trust ------------------------------------------------------

    def _reply_run_pending(self, obj: dict) -> str:
        self._run_pending = {"run_id": obj.get("run_id"), "expt": str(obj.get("expt", "")),
                             "client": str(obj.get("client", "")), "since": time.time(),
                             "token": str(obj.get("token") or ""), "t0": time.monotonic()}
        log.info("Run %s (%s) is starting: composite ops are fenced until it ends.",
                 obj.get("run_id"), obj.get("expt", ""))
        self.journal.record("run_pending", run_id=obj.get("run_id"), expt=obj.get("expt"),
                            client=obj.get("client"))
        self._broadcaster.send({"type": "run_pending", "run_pending": self._public_pending()})
        # No SLM reinit starts from now on; one being asked for is on its way
        # to the SLM server before the run's first mask (slm_reinit.SEND_WAIT_S).
        if self.slm_reinit is not None:
            self.slm_reinit.run_starting()
        # Before replying: the run opens the AWG right after it takes the
        # core, and the reply is what lets its finish_prepare return.
        # Bounded (connections.RELEASE_TIMEOUT_S), inside the client's timeout.
        self.connections.run_starting(_run_name(obj.get("run_id"), obj.get("expt")))
        return json.dumps({"status": "ok"})

    def _reply_clear_run_pending(self, obj: dict, by_operator: bool) -> str:
        """Lift the fence of one announced run, named by its token: the run
        withdrawing itself at exit (it never took the core), or an operator
        asserting from the GUI that it is dead.  A token that is not the
        current fence's -- a late message, a newer run -- changes nothing."""
        p = self._run_pending
        token = str(obj.get("token") or "")
        if p is None or not token or token != p.get("token"):
            return json.dumps({"status": "error",
                               "msg": "that run is no longer fencing composite ops"})
        if by_operator:
            who = str(obj.get("operator") or obj.get("client") or "?")
            why = f"cleared on the Device Control GUI by {who}"
        else:
            why = "the run exited without taking the core"
        log.info("Run %s: fence lifted -- %s.", p.get("run_id"), why)
        self._clear_run_pending(why)
        self.connections.run_over(why, reopen=True)
        return json.dumps({"status": "ok"})

    def _public_pending(self) -> dict | None:
        p = self._run_pending
        return None if p is None else {k: v for k, v in p.items() if k != "t0"}

    def _current_run_pending(self) -> dict | None:
        p = self._run_pending
        if p is not None and time.monotonic() - p["t0"] > self.RUN_PENDING_TTL_S \
                and self.status.state == STATES.READY:
            why = (f"run {p.get('run_id')} never took the core "
                   f"within {self.RUN_PENDING_TTL_S:.0f} s")
            self._clear_run_pending(why)
            self.connections.run_over(why, reopen=True)
            return None
        return p

    def _clear_run_pending(self, why: str) -> None:
        if self._run_pending is None:
            return
        self.journal.record("run_pending_cleared", run_id=self._run_pending.get("run_id"),
                            why=why)
        self._run_pending = None
        self._broadcaster.send({"type": "run_pending", "run_pending": None})

    def _reply_reset_state(self, obj: dict) -> dict:
        """Run the reset experiment.  Refused while a run is starting: the
        reset would take the core from under it (or it from the reset)."""
        operator = str(obj.get("operator") or "")
        client = str(obj.get("client") or "")
        loop = active_loop(self.loops.values())
        if loop is not None:
            msg = (f"{loop.spec.title} is running -- a reset would take the core from its run. "
                   "Stop it first")
            log.warning("State reset refused: %s", msg)
            self.journal.record("state_reset_refused", expt=self.reset.expt, operator=operator,
                                client=client, msg=msg)
            return {"status": "error", "msg": msg}
        pending = self._current_run_pending()
        if pending is not None and not self.reset.running:
            msg = (f"a run is starting (run {pending.get('run_id')}, "
                   f"{pending.get('expt') or 'experiment'}) -- a reset would take the core "
                   "from it. Wait for it to end, or clear its fence if it is dead")
            log.warning("State reset refused: %s", msg)
            self.journal.record("state_reset_refused", expt=self.reset.expt, operator=operator,
                                client=client, msg=msg)
            return {"status": "error", "msg": msg}
        return self.reset.start(operator=operator, client=client)

    def _on_reset_change(self, info) -> None:
        self._broadcaster.send({"type": "reset_run", "reset": info})

    # --- run loops ------------------------------------------------------------------

    def _reply_run_loop(self, obj: dict) -> dict:
        loop = self.loops.get(str(obj.get("loop") or ""))
        if loop is None:
            return {"status": "error",
                    "msg": f"no run loop {obj.get('loop')!r} on this monitor server "
                           f"(offered: {', '.join(self.loops) or 'none'})"}
        operator = str(obj.get("operator") or "")
        client = str(obj.get("client") or "")
        action = obj.get("action")
        if action == "stop":
            return loop.stop(operator=operator, client=client)
        if action != "start":
            return {"status": "error", "msg": f"unknown run loop action {action!r}"}
        other = active_loop(self.loops.values())
        if other is not None and other is not loop:
            return {"status": "error", "msg": f"{other.spec.title} is running -- one loop at "
                                              "a time"}
        return loop.start(operator=operator, client=client)

    def _reply_output(self, obj: dict) -> dict:
        """A loop's or the reset experiment's terminal output after line
        ``after`` (OutputLog.since)."""
        kind = obj.get("kind")
        if kind == "reset":
            if not self.reset.configured:
                return {"status": "error", "msg": "no reset experiment on this monitor server"}
            source = self.reset.output
        elif kind == "run_loop":
            loop = self.loops.get(str(obj.get("key") or ""))
            if loop is None:
                return {"status": "error",
                        "msg": f"no run loop {obj.get('key')!r} on this monitor server"}
            source = loop.output
        else:
            return {"status": "error", "msg": f"unknown output kind {kind!r}"}
        return dict(source.since(obj.get("after", 0)), status="ok")

    def _loop_busy(self) -> str:
        """Why something of this server's own holds the core, for the loops."""
        return "a state reset is running" if self.reset.running else ""

    def _on_loop_change(self, info) -> None:
        self._broadcaster.send({"type": "run_loop", "loop": info})

    def _set_trust(self, trusted: bool, reason: str) -> None:
        self._trust = {"trusted": bool(trusted), "reason": reason, "since": time.time()}
        log.info("Device state %s: %s", "trusted" if trusted else "UNTRUSTED", reason)
        self.journal.record("trust", trusted=bool(trusted), reason=reason)
        if self.config_file_path:
            try:
                with self._state_lock:
                    data = apply_deltas(self.config_file_path, [],
                                        metadata={"state_trust": dict(self._trust)})
                    self._remember_state(data)
            except Exception as e:
                log.warning("Could not store the trust flag in the state file: %s", e)
        self._broadcaster.send({"type": "trust", "trust": dict(self._trust)})

    def on_monitor_state(self, state, reason: str = "") -> None:
        """Called by the owner on every status update (8 Hz).  NOT_READY means
        the monitor process that registered the ops is gone: retire them.
        LOADING is left alone -- a starting monitor registers *before* it
        reports ready, while the owner still says LOADING.  A monitor
        interrupted by a run makes the state untrusted until that run's end
        state arrives."""
        previous, self._last_monitor_state = self._last_monitor_state, (state, reason)
        if state == STATES.NOT_READY:
            self.retire_ops(reason or _state_name(state))
            if reason == "interrupted_by_run" and previous != (state, reason):
                pending = self._run_pending
                who = _run_name(pending.get("run_id"), pending.get("expt")) if pending \
                    else "an experiment"
                self._set_trust(False, f"{who} took the core at "
                                       f"{time.strftime('%H:%M:%S')} and has not reported "
                                       f"its end state")
                # Connections stay released for the whole run (a run that did
                # not announce itself is released now).
                self.connections.run_running(who)
                self._clear_run_pending("the run took the core")
        elif state == STATES.READY and (previous is None or previous[0] != STATES.READY):
            # A running monitor proves no run has the core.
            self.connections.run_over("the monitor is running", reopen=True)
        if previous is None or previous[0] != state:
            self.journal.record("monitor_state", state=_state_name(state), sub_state=reason)

    def retire_ops(self, reason: str) -> None:
        """Drop the registration: queued ops expire, ops the monitor had taken
        but not reported are reported lost (they may or may not have run)."""
        info = self.ops.info()
        if info["registered"] or info["queued"] or info["running"]:
            self._broadcast_results(self.ops.unregister(reason))

    # --- the SLM's reinit ------------------------------------------------------------

    def _slm_reinit_blocker(self) -> str:
        """Why the SLM may be in use ("" when it is not): only a running
        monitor proves no run holds the core. Called under the service's
        lock -- reads attributes only."""
        if self.status.state != STATES.READY:
            return f"the monitor is {self.status.state_name.lower().replace('_', ' ')}"
        p = self._run_pending
        if p is not None:
            return f"{_run_name(p.get('run_id'), p.get('expt'))} is starting"
        if self.reset.running:
            return "a state reset is running"
        loop = active_loop(self.loops.values())
        if loop is not None:
            return f"{loop.spec.title} is running"
        return ""

    def _on_slm_reinit_change(self, snapshot: dict) -> None:
        self._broadcaster.send({"type": "slm_reinit", "slm_reinit": snapshot})

    # --- host-side connections ------------------------------------------------------

    def _on_connections_change(self, snapshot: dict, changed: list) -> None:
        """From ConnectionService, on any thread: log, journal, broadcast."""
        for key in changed:
            c = snapshot.get(key) or {}
            state, detail = c.get("state"), c.get("detail") or ""
            (log.warning if state == conns.FAILED else log.info)(
                "Connection %s: %s%s", c.get("label", key), state,
                f" -- {detail}" if detail else "")
            self.journal.record("connection", key=key, state=state, detail=detail)
        self._broadcaster.send({"type": "connections", "connections": snapshot})

    def _reply_connection_request(self, obj: dict) -> dict:
        """A GUI's connect / disconnect of one of this server's connections.
        The work happens on the connections' own thread; the outcome shows as
        a new ``connections`` state."""
        key = str(obj.get("key") or "")
        action = str(obj.get("action") or "")
        client = str(obj.get("client") or "")
        operator = str(obj.get("operator") or "")
        who = "@".join(p for p in (operator, client) if p)
        if action not in conns.ACTIONS:
            refusal = f"unknown action {action!r}"
        else:
            refusal = self.connections.request(key, action, who)
        label = (self.connections.snapshot().get(key) or {}).get("label", key)
        if refusal:
            log.warning("Connection %s %s refused: %s", label, action, refusal)
            self.journal.record("connection_refused", key=key, action=action, client=client,
                                operator=operator, msg=refusal)
            return {"status": "error", "msg": refusal}
        log.info("Connection %s: %s requested by %s", label, action, who or "?")
        self.journal.record("connection_request", key=key, action=action, client=client,
                            operator=operator)
        return {"status": "ok"}

    def _reply_connection_call(self, obj: dict) -> dict:
        """A driver command on an open connection, from the monitor's op
        host step (the AWG's write_traps).  Runs here, bounded by the
        connection's call timeout."""
        key = str(obj.get("key") or "")
        cmd = str(obj.get("cmd") or "")
        kwargs = obj.get("kwargs") or {}
        if not isinstance(kwargs, dict):
            return {"status": "error", "msg": "kwargs must be an object"}
        try:
            result = self.connections.call(key, cmd, kwargs)
        except (ConnectionRefusedError, RuntimeError) as e:
            log.warning("Connection %s %s failed: %s", key, cmd, e)
            return {"status": "error", "msg": str(e)}
        return {"status": "ok", "result": result}

    # --- device state -----------------------------------------------------------

    def _state(self) -> dict:
        """The state file's content from memory; re-read when its mtime says
        another writer touched it (the experiments' reconcile at prepare)."""
        with self._state_lock:
            try:
                mtime = os.stat(self.config_file_path).st_mtime_ns
            except (FileNotFoundError, TypeError):
                self._state_cache, self._state_mtime = {}, None
                return self._state_cache
            if self._state_cache is None or mtime != self._state_mtime:
                data = read_state(self.config_file_path)
                self._state_cache = data if isinstance(data, dict) else {}
                self._state_mtime = mtime
            return self._state_cache

    def _remember_state(self, data: dict) -> None:
        with self._state_lock:
            self._state_cache = data
            try:
                self._state_mtime = os.stat(self.config_file_path).st_mtime_ns
            except OSError:
                self._state_mtime = None

    def _reply_get_state(self):
        if not self.config_file_path:
            return json.dumps({"status": "error", "msg": "no config path"})
        try:
            cfg = self._state()
        except Exception as e:
            return json.dumps({"status": "error", "msg": str(e)})
        reply = {"status": "ok", "version": self._version, "config": cfg,
                 "composite_state": dict(self.ops.device_state)}
        reply.update(self._extra_status())
        return json.dumps(reply)

    def _reply_update(self, obj):
        if not self.config_file_path:
            return json.dumps({"status": "error", "msg": "no config path"})
        error = self._apply_updates([(obj.get("device_type"), obj.get("device_name"),
                                      obj.get("changes"))], origin=str(obj.get("origin", "")))
        if error:
            return json.dumps({"status": "error", "msg": error})
        return json.dumps({"status": "ok", "version": self._version})

    def _reply_update_batch(self, obj):
        if not self.config_file_path:
            return json.dumps({"status": "error", "msg": "no config path"})
        updates = obj.get("updates")
        if not isinstance(updates, list) or not all(isinstance(u, dict) for u in updates):
            return json.dumps({"status": "error", "msg": "bad update_batch"})
        first = self._version
        error = self._apply_updates([(u.get("device_type"), u.get("device_name"),
                                      u.get("changes")) for u in updates],
                                    origin=str(obj.get("origin", "")))
        if error:
            return json.dumps({"status": "error", "msg": error, "version": self._version})
        return json.dumps({"status": "ok", "version": self._version, "first_version": first})

    def _linked(self, cfg: dict, dtype: str, name: str, changes: dict) -> list:
        """Deltas that keep a DDS v_pd and its linked DAC voltage equal (the
        link is ``dac_ch_key`` in the DDS entry, written by the Generator);
        force_update_counter bumps travel with them."""
        out = []
        if dtype == "dds" and ("v_pd" in changes or "force_update_counter" in changes):
            dac_key = (cfg.get("dds", {}).get(name, {}) or {}).get("dac_ch_key", "")
            linked = {}
            if "v_pd" in changes:
                linked["voltage"] = changes["v_pd"]
            if "force_update_counter" in changes:
                linked["force_update_counter"] = changes["force_update_counter"]
            if dac_key and linked:
                out.append(("dac", dac_key, linked))
        elif dtype == "dac" and ("voltage" in changes or "force_update_counter" in changes):
            for dds_name, dds_cfg in (cfg.get("dds", {}) or {}).items():
                if (dds_cfg or {}).get("dac_ch_key", "") != name:
                    continue
                linked = {}
                if "voltage" in changes:
                    linked["v_pd"] = changes["voltage"]
                if "force_update_counter" in changes:
                    linked["force_update_counter"] = changes["force_update_counter"]
                if linked:
                    out.append(("dds", dds_name, linked))
        return out

    def _apply_updates(self, updates, origin: str = "") -> str | None:
        """Merge deltas -- plus the linked DDS/DAC mirrors they imply -- in one
        atomic write, then bump the version and broadcast once per delta.
        All land or none.  Returns an error message or None."""
        for dtype, name, changes in updates:
            if dtype not in ("dds", "dac", "ttl") or not name or not isinstance(changes, dict):
                return f"bad update {dtype}.{name}"
        with self._state_lock:
            try:
                cfg = self._state()
            except Exception as e:
                return f"could not read the state file: {e}"
            deltas, explicit = [], set()
            for dtype, name, changes in updates:
                deltas.append((dtype, name, dict(changes)))
                explicit.add((dtype, name))
            for dtype, name, changes in updates:
                for linked in self._linked(cfg, dtype, name, changes):
                    if (linked[0], linked[1]) not in explicit:
                        deltas.append(linked)
            try:
                data = apply_deltas(self.config_file_path, deltas)
            except Exception as e:
                return str(e)
            self._remember_state(data)
            for dtype, name, changes in deltas:
                self._log_update(dtype, name, changes, origin)
                self._version += 1
                self._broadcaster.send({
                    "type": "state_update",
                    "version": self._version,
                    "device_type": dtype,
                    "device_name": name,
                    "changes": changes,
                })
                self.journal.record("update", device=f"{dtype}.{name}", changes=changes,
                                    origin=origin, version=self._version)
        return None

    def _reply_replace_state(self, obj) -> str:
        """An experiment's end-of-run state, sent by its end() -- the only way
        a run reports what it left the hardware at."""
        if not self.config_file_path:
            return json.dumps({"status": "error", "msg": "no config path"})
        cfg = obj.get("config")
        if not isinstance(cfg, dict) or not all(isinstance(cfg.get(k), dict)
                                                for k in ("dds", "ttl", "dac")):
            return json.dumps({"status": "error", "msg": "replace_state needs dds/ttl/dac"})
        run_id = obj.get("run_id")
        expt = str(obj.get("expt") or "")
        reason = "end state of " + _run_name(run_id, expt)
        reset = self.reset.running_info(expt)
        if reset is not None:
            who = "@".join(p for p in (reset.get("operator"), reset.get("client")) if p) or "?"
            reason += f", the state reset requested by {who}"
        trust = {"trusted": True, "reason": reason, "since": time.time()}
        with self._state_lock:
            try:
                data = replace_sections(self.config_file_path,
                                        {k: cfg[k] for k in ("dds", "ttl", "dac")},
                                        metadata={"state_trust": trust,
                                                  "updated_from": "end of run",
                                                  "run_id": run_id,
                                                  "timestamp": time.strftime(
                                                      "%Y-%m-%dT%H:%M:%S")})
            except Exception as e:
                return json.dumps({"status": "error", "msg": str(e)})
            self._remember_state(data)
            self._version += 1
            version = self._version
        self._trust = trust
        self._clear_run_pending(f"{_run_name(run_id, expt)} ended")
        # The run let go of the AWG in its post_scan; the connections open
        # again when the monitor is running (a run loop's next run would
        # otherwise take them straight back).
        self.connections.run_over(f"{_run_name(run_id, expt)} ended", reopen=False)
        log.info("%s received; device state trusted.", reason[0].upper() + reason[1:])
        self.journal.record("run_end", run_id=run_id, expt=obj.get("expt"), version=version)
        self._broadcaster.send({"type": "state_reset", "version": version})
        self._broadcaster.send({"type": "trust", "trust": dict(self._trust)})
        if reset is not None:
            self.reset.end_state_received(expt)
        return json.dumps({"status": "ok", "version": version})

    def _reply_abort_state(self, obj) -> str:
        """An aborted run's state: every channel as the kernel had it when an
        exception ended the scan, sent by scan()'s exception handler (a kernel
        that raises never writes its attributes back, so end() cannot).
        Trusted only if the client says so (``trusted``); otherwise ``caveat``
        says which write may not have reached the hardware."""
        if not self.config_file_path:
            return json.dumps({"status": "error", "msg": "no config path"})
        cfg = obj.get("config")
        if not isinstance(cfg, dict) or not all(isinstance(cfg.get(k), dict)
                                                for k in ("dds", "ttl", "dac")):
            return json.dumps({"status": "error", "msg": "abort_state needs dds/ttl/dac"})
        run_id = obj.get("run_id")
        expt = str(obj.get("expt") or "")
        cause = str(obj.get("cause") or "an exception")
        trusted = obj.get("trusted") is True
        if trusted:
            reason = f"state of {_run_name(run_id, expt)} at its abort ({cause})"
        else:
            caveat = str(obj.get("caveat") or "")
            reason = (f"{_run_name(run_id, expt)} aborted ({cause}); the file holds its "
                      f"last commanded state" + (f", but {caveat}" if caveat else ""))
        trust = {"trusted": trusted, "reason": reason, "since": time.time()}
        with self._state_lock:
            try:
                data = replace_sections(self.config_file_path,
                                        {k: cfg[k] for k in ("dds", "ttl", "dac")},
                                        metadata={"state_trust": trust,
                                                  "updated_from": "abort of run",
                                                  "run_id": run_id,
                                                  "timestamp": time.strftime(
                                                      "%Y-%m-%dT%H:%M:%S")})
            except Exception as e:
                return json.dumps({"status": "error", "msg": str(e)})
            self._remember_state(data)
            self._version += 1
            version = self._version
        self._trust = trust
        self._clear_run_pending(f"{_run_name(run_id, expt)} aborted")
        self.connections.run_over(f"{_run_name(run_id, expt)} aborted", reopen=False)
        (log.info if trusted else log.warning)(
            "Device state %s: %s.", "trusted" if trusted else "UNTRUSTED", reason)
        self.journal.record("run_end", run_id=run_id, expt=obj.get("expt"), version=version,
                            aborted=cause, trusted=trusted)
        self._broadcaster.send({"type": "state_reset", "version": version})
        self._broadcaster.send({"type": "trust", "trust": dict(self._trust)})
        return json.dumps({"status": "ok", "version": version})

    def _log_update(self, dtype: str, name: str, changes: dict, origin: str = "") -> None:
        """Print a formatted confirmation of an accepted device-state update."""
        parts = []
        if dtype == "dds":
            if "frequency" in changes:
                parts.append(f"freq {changes['frequency'] / 1e6:.3f} MHz")
            if "amplitude" in changes:
                parts.append(f"amp {changes['amplitude']:.3f}")
            if "v_pd" in changes:
                parts.append(f"v_pd {changes['v_pd']:.3f} V")
            if "sw_state" in changes:
                parts.append("sw " + ("on" if changes["sw_state"] else "off"))
        elif dtype == "dac":
            if "voltage" in changes:
                parts.append(f"{changes['voltage']:.3f} V")
        elif dtype == "ttl":
            if "ttl_state" in changes:
                parts.append("on" if changes["ttl_state"] else "off")
        if parts:
            suffix = f"  ({origin})" if origin else ""
            log.info("[%s] %s -> %s%s", dtype.upper(), name, ", ".join(parts), suffix)

    def stop(self):
        self._runner_stop.set()
        if self.slm_reinit is not None:
            self.slm_reinit.stop()
        # Close the connections (bounded) while this process is still here;
        # their agents would close them anyway once it is gone.
        self.connections.stop()
        for loop in self.loops.values():
            loop.shutdown()
        try:
            self._broadcaster.close()
        except Exception:
            pass
        super().stop()


class MonitorServerGUI(QWidget):
    def __init__(self,
                monitor_expt_path,
                config_file_path=None,
                journal_dir=None,
                reset_expt_path=None,
                run_loops=(),
                connections=(),
                slm_reinit=None):
        super().__init__()

        self.config_file_path = config_file_path
        self.journal_dir = journal_dir
        self.reset_expt_path = reset_expt_path
        self.run_loops = tuple(run_loops or ())
        self.connection_defs = tuple(connections or ())
        self.slm_reinit_config = slm_reinit

        # Refuse to start a second monitor server for the same hardware.
        server_id = monitor_server_id()
        existing = discover(server_id, timeout=1.5)
        if existing is not None:
            ip, port = existing
            QMessageBox.critical(
                self,
                "Monitor server already running",
                f"A monitor server for '{server_id}' is already running at "
                f"{ip}:{port}.\n\nRefusing to start a second server for the same "
                "hardware.",
            )
            self._aborted = True
            QTimer.singleShot(0, self.close)
            return
        self._aborted = False

        self.setWindowTitle("Monitor Server")
        eye_icon = self._create_eye_icon()
        self.setWindowIcon(eye_icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(eye_icon)
        self.setGeometry(100, 100, 250, 80)

        # Everything the monitor reports goes through logging (stderr, line
        # buffered) rather than print, so it shows up promptly in the terminal
        # that launched this GUI as well as in the dashboard log.
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s %(levelname)s %(message)s")

        self.monitor_manager = MonitorManager(monitor_expt_path)
        self.monitor_manager.msg.connect(lambda m: log.info("monitor: %s", m))
        self.monitor_manager.monitor_stopped.connect(self._on_monitor_stopped)

        log.info("monitor experiment: %s", monitor_expt_path)
        log.info("device state file: %s", config_file_path)
        log.info("reset experiment: %s", reset_expt_path or "none (no reset button)")
        if config_file_path is None:
            log.error(
                "No device-state config path was passed: state reads/writes will "
                "fail with 'no config path'. The launcher could not resolve it "
                "(usually an unset env var or an unmapped drive)."
            )
        for problem in self.monitor_manager.preflight_problems():
            log.error("Monitor cannot be started as configured: %s", problem)

        self.setup_ui()
        self.setup_udp_server()
        # One status object, shared with the TCP responder so status_json
        # always serves what this window shows.
        self.status = self.udp_server.status
        self.status.expt_path = str(monitor_expt_path)

        # Initial status is "not ready".  (This used to pass False, which equals
        # STATES.READY == 0 -- so the server briefly advertised READY, and the
        # button flashed green, while nothing was running.)
        self.set_status(STATES.NOT_READY)

        self.monitor_check_timer = QTimer(self)
        self.monitor_check_timer.setInterval(125)
        self.monitor_check_timer.timeout.connect(self.check_monitor_status)
        self.monitor_check_timer.start()

    @staticmethod
    def _create_eye_icon(size=64):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        font = QFont("Segoe UI Emoji")
        font.setPixelSize(int(size * 0.8))
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "👁")
        painter.end()

        return QIcon(pixmap)

    def setup_ui(self):
        layout = QVBoxLayout()
        self.status_indicator = QPushButton("NOT READY")
        self.status_indicator.clicked.connect(self.on_button_clicked)
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.status_indicator.setFont(font)
        layout.addWidget(self.status_indicator)
        self.setLayout(layout)

    def setup_udp_server(self):
        self.server_thread = QThread()
        
        self.udp_server = MonitorUDPServer(config_file_path=self.config_file_path,
                                           journal_dir=self.journal_dir,
                                           reset_expt_path=self.reset_expt_path,
                                           run_loops=self.run_loops,
                                           connections=self.connection_defs,
                                           slm_reinit=self.slm_reinit_config)
        self.udp_server.moveToThread(self.server_thread)

        self.udp_server.reset_signal.connect(self.restart_monitor)
        self.udp_server.stop_signal.connect(self._stop_monitor)
        self.udp_server.start_monitor_signal.connect(self._start_monitor_unless_running)
        self.server_thread.started.connect(self.udp_server.run)
        self.udp_server.message_received.connect(self.handle_message)

        self.server_thread.start()

    def on_button_clicked(self):
        if self.status.state == STATES.READY:
            reply = QMessageBox.question(self, 'Restart Monitor',
                                         "Are you sure you'd like to restart the monitor experiment?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                log.info("Manual monitor restart triggered.")
                self.restart_monitor()
        elif self.status.state == STATES.NOT_READY:
            log.info("Manual monitor start triggered.")
            self.monitor_manager.start()

    def _on_monitor_stopped(self, reason: str) -> None:
        """Surface the monitor's own failure reason in the terminal and the UI."""
        log.warning("Monitor is not running: %s", reason)
        self.status_indicator.setToolTip(f"Monitor is not running: {reason}")
        # The signal is queued from the manager thread and may land a tick
        # before the thread has fully exited; the 8 Hz check picks the same
        # detail up from the manager as soon as isRunning() drops.
        if not self.monitor_manager.isRunning():
            self.set_status(STATES.NOT_READY,
                            self.monitor_manager.last_stop_kind or "failed", reason)

    def _not_ready_detail(self) -> tuple[str, str]:
        """(sub_state, reason) for NOT_READY when nothing more specific is known.

        A specific NOT_READY sub_state already recorded (by the stop handler or
        by ``_on_monitor_stopped``) is kept; otherwise it is the manager's
        classification of the last exit, or ``never_started``.
        """
        if self.status.state == STATES.NOT_READY and self.status.sub_state:
            return self.status.sub_state, self.status.reason
        kind = self.monitor_manager.last_stop_kind
        if kind is None:
            return "never_started", ""
        return kind, self.monitor_manager.last_stop_reason or ""

    def _stop_monitor(self):
        """``stop`` command: stop the monitor and leave it stopped."""
        if self.monitor_manager.isRunning():
            log.info("Stopping monitor experiment on request...")
            self.monitor_manager.stop()
        else:
            log.info("Stop requested; the monitor experiment is not running.")
        self.set_status(STATES.NOT_READY, "stopped_on_request",
                        "stopped by a client request")

    def _start_monitor_unless_running(self, why: str) -> None:
        """A run loop ended: bring the monitor back -- unless it already is
        (an aborted run's ``run complete`` has usually started it)."""
        if self.monitor_manager.isRunning():
            log.info("%s -- the monitor is already running.", why)
            return
        log.info("%s -- starting the monitor.", why)
        self.restart_monitor()

    def restart_monitor(self):
        if getattr(self, "_restarting", False):
            log.info("Monitor restart ignored: a restart is already in progress.")
            return
        self._restarting = True
        self.udp_server.retire_ops("monitor restarting")
        try:
            if self.monitor_manager.isRunning():
                log.info("Restarting monitor experiment...")
                self.monitor_manager.stop()
            else:
                log.info("Starting monitor experiment...")
            self.monitor_manager.start()
            if self.monitor_manager.isRunning():
                self.set_status(STATES.LOADING, "starting", "")
            else:
                # start() refused (pre-flight failure); it has already said why
                # and _on_monitor_stopped has recorded preflight_failed.
                self.set_status(STATES.NOT_READY, *self._not_ready_detail())
        finally:
            self._restarting = False

    def set_status(self, status, sub_state=None, reason=None):
        changed = self.status.set_state(status, sub_state, reason)
        self.status.pid = self.monitor_manager.pid
        self.udp_server.on_monitor_state(status, self.status.sub_state)
        # Logged only on change: check_monitor_status runs at 8 Hz.
        if changed or getattr(self, "_logged_state", None) is None:
            previous = getattr(self, "_logged_state", None)
            current = f"{_state_name(status)} ({self.status.sub_state})"
            if previous is None:
                log.info("state: %s", current)
            else:
                log.info("state: %s -> %s", previous, current)
            self._logged_state = current

        if status == STATES.READY:
            self.status_indicator.setText("READY")
            self.status_indicator.setStyleSheet("background-color: green; color: white;")
        elif status == STATES.NOT_READY:
            self.status_indicator.setText("NOT READY")
            self.status_indicator.setStyleSheet("background-color: #c46666; color: white;")
        else:
            self.status_indicator.setText("Loading...")
            self.status_indicator.setStyleSheet("background-color: orange; color: white;")

    def check_monitor_status(self):
        running = self.monitor_manager.isRunning()
        if running and self.status.state != STATES.READY:
            self.set_status(STATES.LOADING, "starting", "")
        elif not running:
            self.set_status(STATES.NOT_READY, *self._not_ready_detail())
        else:
            self.set_status(STATES.READY, "running", "")

    def handle_message(self, message):
        log.info("msg: %s", message)
        if "run complete" in message:
            log.info("Run complete message received. Restarting monitor.")
            self.restart_monitor()
        elif "monitor ready" in message:
            log.info("Monitor ready message received.")
            self.set_status(STATES.READY, "running", "")
        
    def closeEvent(self, event):
        # Guarded with getattr: when __init__ aborted early (another monitor
        # server was already running) none of these exist, and an
        # AttributeError traceback here would bury the message saying why.
        log.info("Closing monitor server GUI...")
        udp_server = getattr(self, "udp_server", None)
        if udp_server is not None:
            udp_server.stop()
        server_thread = getattr(self, "server_thread", None)
        if server_thread is not None:
            server_thread.quit()
            server_thread.wait()
        monitor_manager = getattr(self, "monitor_manager", None)
        if monitor_manager is not None:
            monitor_manager.stop()
        event.accept()