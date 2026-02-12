"""
DC205 Precision DC Voltage Source Control Library
-------------------------------------------------

This module provides:

    • DC205        – Direct RS‑232/USB control class
    • DC205_Server – LAN server for remote control
    • DC205_Client – LAN client mirroring the DC205 API

All command strings are EXACTLY as defined in the DC205 manual.
"""

import serial
import socket
import json
import threading
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Container classes for syntax‑highlighting of settings
# ---------------------------------------------------------------------------

class DC205_Range:
    def __init__(self):
        self.RANGE1 = 0
        self.RANGE10 = 1
        self.RANGE100 = 2


class DC205_Sense:
    def __init__(self):
        self.TWOWIRE = 0
        self.FOURWIRE = 1


class DC205_Isolation:
    def __init__(self):
        self.GROUND = 0
        self.FLOAT = 1


class DC205_Output:
    def __init__(self):
        self.OFF = 0
        self.ON = 1


class DC205_ScanShape:
    def __init__(self):
        self.ONEDIR = 0
        self.UPDN = 1


class DC205_ScanCycle:
    def __init__(self):
        self.ONCE = 0
        self.REPEAT = 1


class DC205_ScanDisplay:
    def __init__(self):
        self.OFF = 0
        self.ON = 1


class DC205_ScanArm:
    def __init__(self):
        self.IDLE = 0
        self.ARMED = 1
        # SCANNING = 2 is query‑only


class DC205_Settings:
    def __init__(self):
        self.range = DC205_Range()
        self.sense = DC205_Sense()
        self.isolation = DC205_Isolation()
        self.output = DC205_Output()
        self.scan_shape = DC205_ScanShape()
        self.scan_cycle = DC205_ScanCycle()
        self.scan_display = DC205_ScanDisplay()
        self.scan_arm = DC205_ScanArm()


# ---------------------------------------------------------------------------
# DC205 Control Class
# ---------------------------------------------------------------------------

class DC205:
    """
    Direct control class for the DC205 Precision DC Voltage Source.
    """

    def __init__(self, port='COM10', baudrate=115200, timeout=1.0):
        self.port = port
        self.baud = baudrate
        self.timeout = timeout
        self.settings = DC205_Settings()

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            write_timeout=self.timeout
        )
        print(f"Connected to {self.port} at {self.baud} baud.")

    # ---------------- Low‑level helpers ----------------

    def close(self):
        if self.ser.is_open:
            self.ser.close()
            print("Serial port closed.")

    def _write(self, cmd: str):
        if not cmd.endswith("\n"):
            cmd += "\n"
        self.ser.write(cmd.encode("ascii"))

    def _query(self, cmd: str) -> str:
        if not cmd.endswith("\n"):
            cmd += "\n"
        self.ser.write(cmd.encode("ascii"))
        return self.ser.readline().decode("ascii").strip()

    # ---------------- Basic SCPI ----------------

    def identify(self):
        res = self._query("*IDN?")
        print(res)
        return res

    def reset(self):
        self._write("*RST")

    # ---------------- Configuration ----------------

    def set_range(self, r: int):
        self._write(f"RNGE {r}")

    def get_range(self):
        return self._query("RNGE?")

    def set_isolation(self, mode: int):
        self._write(f"ISOL {mode}")

    def get_isolation(self):
        return self._query("ISOL?")

    def set_sense(self, mode: int):
        self._write(f"SENS {mode}")

    def get_sense(self):
        return self._query("SENS?")

    def set_output(self, state: int):
        self._write(f"SOUT {state}")

    def get_output(self):
        return self._query("SOUT?")

    # ---------------- Voltage ----------------

    def set_voltage(self, volts: float):
        self._write(f"VOLT {volts:.6f}")

    def get_voltage(self):
        return float(self._query("VOLT?"))

    # ---------------- Scan Commands ----------------

    def set_scan_range(self, r: int):
        self._write(f"SCAR {r}")

    def set_scan_begin(self, v: float):
        self._write(f"SCAB {v:.6f}")

    def set_scan_end(self, v: float):
        self._write(f"SCAE {v:.6f}")

    def set_scan_time(self, t: float):
        self._write(f"SCAT {t:.1f}")

    def set_scan_shape(self, shape: int):
        self._write(f"SCAS {shape}")

    def set_scan_cycle(self, mode: int):
        self._write(f"SCAC {mode}")

    def set_scan_display(self, mode: int):
        self._write(f"SCAD {mode}")

    def arm_scan(self, state: int):
        self._write(f"SCAA {state}")

    def trigger_scan(self):
        self._write("*TRG")

    # ---------------- Status ----------------

    def get_interlock(self):
        return self._query("ILOC?")

    def get_overload(self):
        return self._query("OVLD?")

    def get_last_exec_error(self):
        return self._query("LEXE?")

    def get_last_cmd_error(self):
        return self._query("LCME?")

    def get_status_byte(self):
        return int(self._query("*STB?"))

    def clear_status(self):
        self._write("*CLS")


# ---------------------------------------------------------------------------
# DC205 Server
# ---------------------------------------------------------------------------

class DC205_Server:
    """
    LAN server that exposes a DC205 over TCP.
    """

    def __init__(self, port='COM10',
                baudrate=115200,
                server_ip='0.0.0.0',
                server_port=5555):
        self.device = DC205(port=port, baudrate=baudrate)
        self.server_ip = server_ip
        self.server_port = server_port
        self.running = False

    def _handle_client(self, sock, addr):
        print(f"Client connected: {addr}")
        try:
            while self.running:
                data = sock.recv(4096).decode()
                if not data:
                    break
                cmd = json.loads(data)
                method = cmd["method"]
                args = cmd.get("args", {})
                
                # Print received command
                print(f"Received command: {method}({', '.join(f'{k}={v}' for k, v in args.items())})")

                try:
                    result = getattr(self.device, method)(**args)
                    resp = {"status": "success", "result": result}
                except Exception as e:
                    resp = {"status": "error", "message": str(e)}
                    print(f"  Error: {e}")

                sock.sendall(json.dumps(resp).encode())
        finally:
            sock.close()
            print(f"Client disconnected: {addr}")

    def start(self):
        self.running = True
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.server_ip, self.server_port))
        server.listen(5)
        print(f"DC205 Server listening on {self.server_ip}:{self.server_port}")
        print("Press Ctrl+C to stop the server")

        try:
            while self.running:
                client, addr = server.accept()
                threading.Thread(target=self._handle_client, args=(client, addr), daemon=True).start()
        except KeyboardInterrupt:
            print("\n\nServer interrupted by user (Ctrl+C)")
        finally:
            self.stop()
            server.close()
            self.device.close()
            print("DC205 Server stopped")

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------------
# DC205 Client
# ---------------------------------------------------------------------------

class DC205_Client:
    """
    Remote client that mirrors the DC205 API over TCP.
    """

    def __init__(self, server_ip='localhost', server_port=5555, timeout=5.0):
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.settings = DC205_Settings()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.server_ip, self.server_port))
        print(f"Connected to DC205 Server at {server_ip}:{server_port}")

    def close(self):
        self.sock.close()

    def _send(self, method: str, **kwargs):
        msg = json.dumps({"method": method, "args": kwargs})
        self.sock.sendall(msg.encode())
        resp = json.loads(self.sock.recv(4096).decode())
        if resp["status"] == "error":
            raise RuntimeError(resp["message"])
        return resp.get("result")

    # Mirror all DC205 methods
    def identify(self): return self._send("identify")
    def reset(self): return self._send("reset")
    def set_range(self, r): return self._send("set_range", r=r)
    def set_voltage(self, v): return self._send("set_voltage", volts=v)
    def set_output(self, s): return self._send("set_output", state=s)
    def set_sense(self, s): return self._send("set_sense", mode=s)
    def set_isolation(self, m): return self._send("set_isolation", mode=m)
    def set_scan_range(self, r): return self._send("set_scan_range", r=r)
    def set_scan_begin(self, v): return self._send("set_scan_begin", v=v)
    def set_scan_end(self, v): return self._send("set_scan_end", v=v)
    def set_scan_time(self, t): return self._send("set_scan_time", t=t)
    def set_scan_shape(self, s): return self._send("set_scan_shape", shape=s)
    def set_scan_cycle(self, c): return self._send("set_scan_cycle", mode=c)
    def set_scan_display(self, d): return self._send("set_scan_display", mode=d)
    def arm_scan(self, a): return self._send("arm_scan", state=a)
    def trigger_scan(self): return self._send("trigger_scan")


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Server mode
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        com = sys.argv[2] if len(sys.argv) > 2 else "COM1"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 5555
        server = DC205_Server(port=com, server_port=port)
        server.start()

    # Client mode
    elif len(sys.argv) > 1 and sys.argv[1] == "client":
        ip = sys.argv[2] if len(sys.argv) > 2 else "localhost"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 5555
        c = DC205_Client(server_ip=ip, server_port=port)

        c.set_range(c.settings.range.RANGE10)
        c.set_voltage(1.234567)
        c.set_output(c.settings.output.ON)
        print("Voltage set remotely.")

    # Direct mode
    else:
        com = sys.argv[1] if len(sys.argv) > 1 else "COM1"
        d = DC205(port=com)

        d.reset()
        d.set_range(d.settings.range.RANGE1)
        d.set_voltage(0.500000)
        d.set_output(d.settings.output.ON)
        print("Direct DC205 control complete.")