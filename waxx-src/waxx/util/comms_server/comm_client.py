import socket

from beacon.discovery.client import NetClient
from waxx.util.comms_server.hardware_id import MONITOR_BASE_ID, resolve_scoped_server_id

class CommClient(NetClient):
    """
    A TCP client that discovers its server via UDP broadcast.

    ``server_id`` is the discovery key (e.g. ``"monitor"``).  Raises
    ``RuntimeError`` if the server is not discovered within the timeout.
    """
    def __init__(self, server_id: str, discovery_timeout: float = 3.0):
        super().__init__(server_id, discovery_timeout=discovery_timeout)
        self.server_address = (self.host, self.port)
        
    def send_message(self, message, timeout: float = 5.0, attempts: int = 2):
        """
        Sends a newline-framed message to the server and returns the reply.

        Messages are framed with a trailing ``"\\n"`` and the reply is read
        until the first newline, so payloads larger than a single TCP segment
        (e.g. a full-state JSON snapshot) are handled correctly.  A socket
        timeout guarantees the call can never hang the caller indefinitely.

        :param message: The message to send (string).
        :param timeout: Socket timeout per attempt (s).
        :param attempts: Tries; a failed first try rediscovers the server.
        :returns: The decoded reply string, or ``None`` on failure.
        """
        for attempt in range(max(int(attempts), 1)):
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            try:
                self.sock.connect(self.server_address)
                self.sock.sendall((message + "\n").encode())
                buf = b""
                while b"\n" not in buf:
                    chunk = self.sock.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                return buf.split(b"\n", 1)[0].decode()
            except Exception:
                if attempt == 0 and attempts > 1:
                    # Rediscover server in case it restarted at a new IP/port.
                    if self._rediscover(timeout=2.0):
                        self.server_address = (self.host, self.port)
                    continue
                # Final failure: do not print/popup.  Callers detect the
                # failure via a ``None`` return and surface it as a red
                # status indicator in the GUI.
            finally:
                self.sock.close()
        
    def close(self):
        """
        Closes the socket.
        """
        self.sock.close()

class MonitorClient(CommClient):
    def __init__(self, discovery_timeout: float = 3.0):
        # Connect only to the monitor server controlling this machine's hardware
        # (matched via core_addr from env var 'db').  When no hardware id is
        # available, resolve_scoped_server_id falls back to the unique monitor
        # server on the subnet, or raises if the choice is ambiguous.
        super().__init__(
            resolve_scoped_server_id(MONITOR_BASE_ID),
            discovery_timeout=discovery_timeout,
        )

    def send_end(self):
        self.send_message("run complete")

    def send_ready(self):
        self.send_message("monitor ready")

    def check_status(self):
        """Legacy status poll: the bare ``ReadyBit`` integer as a string, or ``None``."""
        status = self.send_message("status")
        return status

    def get_status(self):
        """Structured status poll.

        Sends ``status_json`` and returns the parsed dict, or ``None`` on
        failure.  Keys the server guarantees::

            state        int   ReadyBit value (0 READY, 1 LOADING, 2 NOT_READY)
            state_name   str   "READY" | "LOADING" | "NOT_READY"
            sub_state    str   why it is in that state, machine-readable, e.g.
                               "running", "starting", "never_started",
                               "interrupted_by_run", "exited", "failed",
                               "preflight_failed", "stopped_on_request"
            reason       str   human-readable detail (may be "")
            since        float epoch seconds when the current state began
            pid          int|None  pid of the monitor experiment process
            expt_path    str   monitor experiment file the server launches
        """
        import json  # noqa: PLC0415
        reply = self.send_message("status_json")
        if reply is None:
            return None
        try:
            obj = json.loads(reply)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def send_reset(self):
        self.send_message("reset")

    def send_stop(self):
        """Ask the server to stop the monitor experiment and leave it stopped.

        Returns the raw reply string, or ``None`` on failure.
        """
        return self.send_message("stop")

    def send_update(self, device_type, device_name, changes):
        """Send a partial device-state delta to the server.

        The server merges ``changes`` into the JSON, bumps the version, and
        broadcasts the update.  Returns the parsed ack dict
        (``{"status": "ok", "version": N}``) or ``None`` on failure.
        """
        import json  # noqa: PLC0415
        msg = json.dumps({
            "type": "update",
            "device_type": device_type,
            "device_name": device_name,
            "changes": changes,
        })
        reply = self.send_message(msg)
        if reply is None:
            return None
        try:
            return json.loads(reply)
        except Exception:
            return None

    def get_state(self):
        """Request the full device-state snapshot from the server.

        Returns the parsed dict ``{"status": "ok", "version": N, "config": {...}}``
        or ``None`` on failure.  This is how clients obtain their initial state
        and resync after a missed broadcast — no shared-drive access required.
        """
        return self._request({"type": "get_state"})

    def _request(self, obj):
        """Send a structured request; the parsed reply dict, or ``None``."""
        import json  # noqa: PLC0415
        reply = self.send_message(json.dumps(obj))
        if reply is None:
            return None
        try:
            parsed = json.loads(reply)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    # --- composite ops (waxx.util.device_state.composite) ---------------------

    def send_op(self, op, sig, args, payload=None, client="", operator="", rid=None):
        """Request a composite op.  Reply ``{"status": "ok", "seq": N}`` or an
        error dict; ``None`` if the server is unreachable.  ``rid`` (made here
        if not given) lets the server drop the copy a retry after a lost reply
        would otherwise queue."""
        import uuid  # noqa: PLC0415
        return self._request({"type": "op", "op": op, "sig": sig, "args": args,
                              "payload": payload or {}, "client": client,
                              "operator": operator, "rid": rid or uuid.uuid4().hex})

    def request(self, obj):
        """Any structured request; the parsed reply or ``None``."""
        return self._request(obj)

    def replace_state(self, config, run_id=None, expt=""):
        """An experiment's end-of-run device state (its ``end()``)."""
        return self._request({"type": "replace_state", "config": config,
                              "run_id": run_id, "expt": expt})

    def announce_run(self, run_id=None, expt="", client="", token=""):
        """An experiment is about to take the core: fence composite ops.
        ``token`` names this announcement for :meth:`withdraw_run`."""
        return self._request({"type": "run_pending", "run_id": run_id, "expt": expt,
                              "client": client, "token": token})

    def withdraw_run(self, token, run_id=None, timeout: float = 2.0):
        """The announced run is exiting without having taken the core: lift
        its fence.  One short try (it is sent from a dying process)."""
        import json  # noqa: PLC0415
        reply = self.send_message(json.dumps({"type": "run_withdrawn", "token": token,
                                              "run_id": run_id}),
                                  timeout=timeout, attempts=1)
        if reply is None:
            return None
        try:
            parsed = json.loads(reply)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def get_journal(self, n=200, since=None):
        obj = {"type": "get_journal", "n": int(n)}
        if since:
            obj["since"] = since
        return self._request(obj)

    def op_status(self, seq):
        """``{"status": "ok", "state": "queued"|"running"|"done"|"unknown",
        "result": {...}}`` for one request, or ``None``."""
        return self._request({"type": "op_status", "seq": int(seq)})

    def register_ops(self, registration):
        """Monitor experiment: announce the compiled op table."""
        return self._request(registration)

    def poll(self):
        """Monitor experiment: version + queued ops, one round trip."""
        return self._request({"type": "poll"})

    def report_ops(self, results):
        """Monitor experiment: outcomes of ops it took."""
        return self._request({"type": "op_done", "results": results})

    def send_update_batch(self, updates, origin=""):
        """Several deltas in one round trip: ``updates`` is a list of
        ``(device_type, device_name, changes)``."""
        return self._request({
            "type": "update_batch", "origin": origin,
            "updates": [{"device_type": t, "device_name": n, "changes": c}
                        for t, n, c in updates],
        })

# if __name__ == '__main__':
#     # Example usage:
#     # This would be run on a machine that wants to send a message to the server.
#     # The server GUI should be running on the specified IP.
    
#     # Create a client to communicate with the server
#     # Replace with the actual server IP if different
#     client = CommClient('192.168.1.79') 

#     # Example of sending a "run complete" message
#     client.send_message("run complete")
    
#     # Example of sending a "monitor ready" message
#     # import time
#     # time.sleep(2)
#     # client.send_message("monitor ready")

#     client.close()