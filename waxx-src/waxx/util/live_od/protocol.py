"""
Shared protocol utilities for camera server communication.

Message format:
    [4 bytes: message length (uint32 big-endian)][pickle-serialized payload]

Payload is a Python dict with at minimum a "cmd" key.
"""

import struct
import pickle
import socket

HEADER_SIZE = 4

def send_msg(sock, obj):
    """Send a length-prefixed pickled message over a socket.

    Args:
        sock: Connected socket.
        obj: Python object to send (must be picklable).
    """
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack('>I', len(data))
    sock.sendall(header + data)

def recv_msg(sock):
    """Receive a length-prefixed pickled message from a socket.

    Args:
        sock: Connected socket.

    Returns:
        Deserialized Python object, or None if the connection is closed.
    """
    raw_header = _recvall(sock, HEADER_SIZE)
    if raw_header is None:
        return None
    msg_len = struct.unpack('>I', raw_header)[0]
    data = _recvall(sock, msg_len)
    if data is None:
        return None
    return pickle.loads(data)

def _recvall(sock, n):
    """Receive exactly n bytes from a socket.

    Args:
        sock: Connected socket.
        n: Number of bytes to receive.

    Returns:
        bytes of length n, or None if the connection is closed before n bytes.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)
