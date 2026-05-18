import socket

from waxx.util.comms_server.waxx_client import WaxxClient

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
        Sends a message to the server.

        :param message: The message to send (string).
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect(self.server_address)
            self.sock.sendall(message.encode())
            reply = self.sock.recv(1024)
            return reply.decode()
        except Exception as e:
            if "[WinError 10061]" in str(e):
                print("Connection refused [WinError 10061]: Unable to reach the server")
            else:
                print(e)
        finally:
            self.sock.close()
        
    def close(self):
        """
        Closes the socket.
        """
        self.sock.close()

class MonitorClient(CommClient):
    def __init__(self, discovery_timeout: float = 3.0):
        super().__init__("monitor", discovery_timeout=discovery_timeout)

    def send_end(self):
        self.send_message("run complete")

    def send_ready(self):
        self.send_message("monitor ready")

    def check_status(self):
        status = self.send_message("status")
        return status
    
    def send_reset(self):
        self.send_message("reset")

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