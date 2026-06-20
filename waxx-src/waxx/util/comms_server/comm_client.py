import socket

from waxx.util.comms_server.waxx_client import WaxxClient
from waxx.util.comms_server.hardware_id import MONITOR_BASE_ID, resolve_scoped_server_id

class CommClient(WaxxClient):
    """
    A TCP client that discovers its server via UDP broadcast.

    ``server_id`` is the discovery key (e.g. ``"monitor"``).  Raises
    ``RuntimeError`` if the server is not discovered within the timeout.
    """
    def __init__(self, server_id: str, discovery_timeout: float = 3.0):
        super().__init__(server_id, discovery_timeout=discovery_timeout)
        self.server_address = (self.host, self.port)
        
    def send_message(self, message):
        """
        Sends a newline-framed message to the server and returns the reply.

        Messages are framed with a trailing ``"\\n"`` and the reply is read
        until the first newline, so payloads larger than a single TCP segment
        (e.g. a full-state JSON snapshot) are handled correctly.  A socket
        timeout guarantees the call can never hang the caller indefinitely.

        :param message: The message to send (string).
        :returns: The decoded reply string, or ``None`` on failure.
        """
        for attempt in range(2):
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5.0)
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
                if attempt == 0:
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
        status = self.send_message("status")
        return status
    
    def send_reset(self):
        self.send_message("reset")

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
        import json  # noqa: PLC0415
        reply = self.send_message(json.dumps({"type": "get_state"}))
        if reply is None:
            return None
        try:
            return json.loads(reply)
        except Exception:
            return None

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