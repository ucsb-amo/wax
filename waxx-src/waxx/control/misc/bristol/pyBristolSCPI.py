## @package pyBristolSCPI
# This module contains functions to call SCPI commands to collect data from the instrument.

import socket
from struct import unpack

try:
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:
    print("Modules not installed: {}".format(e))

_PORT = 23
_RECV_SIZE = 4096
_TIMEOUT = 5.0


class pyBristolSCPI:

    def __init__(self, host='10.199.199.1', port=_PORT, timeout=_TIMEOUT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect((host, port))
        self._sock_file = self._sock.makefile('rb')
        self.skipOpeningMessage()

    def readWL(self):
        out = self.getSimpleMsg(b':READ:WAV?')
        return float(out.strip().decode('ascii'))

    def getSimpleMsg(self, msg):
        self._sock.sendall(msg + b'\r\n')
        while True:
            out = self._sock_file.readline()
            out = out.strip()
            if out and out != b'1':
                return out

    def skipOpeningMessage(self):
        """Drain any banner text sent by the instrument on connect."""
        self._sock.settimeout(0.5)
        try:
            while True:
                chunk = self._sock.recv(_RECV_SIZE)
                if not chunk:
                    break
        except (socket.timeout, BlockingIOError):
            pass
        self._sock.settimeout(_TIMEOUT)

    def close(self):
        self._sock_file.close()
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _recv_exact(self, n: int) -> bytes:
        """Read exactly n bytes from the socket."""
        buf = b''
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed while reading")
            buf += chunk
        return buf

    def _recv_ieee_block(self) -> bytes:
        """Read an IEEE 488.2 definite-length binary block (#NXXXXXXX...)."""
        # Read '#'
        self._sock.recv(1)
        num_digits = int(self._sock.recv(1).decode('ascii'))
        length = int(self._sock.recv(num_digits).decode('ascii'))
        return self._recv_exact(length)

    def startBuffer(self):
        self._sock.sendall(b':MMEM:INIT\r\n')
        self._sock.sendall(b':MMEM:OPEN\r\n')

    def readBuffer(self, outfile, acq_time):
        self._sock.sendall(b':MMEM:CLOSE\r\n')
        self._sock.sendall(b':MMEM:DATA?\r\n')
        raw = self._recv_ieee_block()
        record_size = 20
        num_samples = len(raw) // record_size
        with open(outfile, 'w') as fs:
            for i in range(num_samples):
                chunk = raw[i * record_size:(i + 1) * record_size]
                wvl, pwr, status, scan_indx = unpack('<dfII', chunk)
                fs.write('{}, {}, {:f}, {:.4f} \n'.format(scan_indx, status, wvl, pwr))

    def getStartWL(self):
        out = self.getSimpleMsg(b':CALC2:WLIM:STAR?')
        return float(out.strip().decode('ascii'))

    def getEndWL(self):
        out = self.getSimpleMsg(b':CALC2:WLIM:STOP?')
        return float(out.strip().decode('ascii'))

    def getWLSpectrum(self, outfile):
        sample_size = 12
        self._sock.sendall(b':CALC3:DATA?\r\n')
        raw = self._recv_ieee_block()
        num_samples = len(raw) // sample_size
        with open(outfile, 'w') as fs:
            for i in range(num_samples):
                chunk = raw[i * sample_size:(i + 1) * sample_size]
                wvl, pwr = unpack('<df', chunk)
                fs.write('{:f}, {:.4f} \n'.format(wvl, pwr))

    def getSpectrum(self, outfile):
        self._sock.sendall(b':CALC2:DATA?\r\n')
        self._sock.sendall(b'*OPC?\r\n')
        spectrum = b''
        while True:
            out = self._sock_file.readline()
            spectrum += out
            if out.strip() == b'1':
                self._sock.sendall(b'*CLS\r\n')
                break
        values = spectrum[100:-1].replace(b'\r\n', b'').decode('ascii').split(',')
        with open(outfile, 'w') as fs:
            for s in values:
                fs.write(s + '\n')
        try:
            s = np.array(values).astype(float)
            startWL = self.getStartWL()
            endWL = self.getEndWL()
            w = np.linspace(startWL, endWL, 70001)
            plt.plot(w, s)
            plt.show()
        except Exception as e:
            print("error plotting: {}".format(e))
