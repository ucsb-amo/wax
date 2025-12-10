import socket

class CommClient:
    """
    A client for sending UDP messages to a server.
    """
    def __init__(self, server_ip, server_port=6789):
        """
        Initializes the CommClient.

        :param server_ip: The IP address of the server.
        :param server_port: The port of the server. Defaults to 6789.
        """
        self.server_address = (server_ip, server_port)
        
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
            print(f'reply: {reply.decode()}')
        except Exception as e:
            print(e)
        finally:
            self.sock.close()
        
    def close(self):
        """
        Closes the socket.
        """
        self.sock.close()

class MonitorClient(CommClient):
    def __init__(self, server_ip, server_port=6789):
        super().__init__(server_ip,server_port)

    def send_end(self):
        self.send_message("run complete")

    def send_ready(self):
        self.send_message("monitor ready")

    def check_status(self):
        self.send_message("status")

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