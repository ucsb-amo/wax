"""
SRS560 Voltage Preamplifier Control Class

Control the SRS560 voltage preamplifier via RS232 serial commands.
Includes container classes for device settings with syntax highlighting support.
Provides server-client architecture for remote control over LAN.
"""

import serial
import time
import socket
import json
import threading
from typing import Union, Literal




class SRS560_Gain:
    """Container for gain settings with syntax highlighting support."""
    def __init__(self):
        # Gain values (0-14): 1, 2, 5, 10, 20, 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k, 50k
        self.GAIN_1 = 0
        self.GAIN_2 = 1
        self.GAIN_5 = 2
        self.GAIN_10 = 3
        self.GAIN_20 = 4
        self.GAIN_50 = 5
        self.GAIN_100 = 6
        self.GAIN_200 = 7
        self.GAIN_500 = 8
        self.GAIN_1k = 9
        self.GAIN_2k = 10
        self.GAIN_5k = 11
        self.GAIN_10k = 12
        self.GAIN_20k = 13
        self.GAIN_50k = 14


class SRS560_InputSource:
    """Container for input source settings."""
    def __init__(self):
        self.SOURCE_A = 0
        self.SOURCE_A_MINUS_B = 1
        self.SOURCE_B = 2


class SRS560_InputCoupling:
    """Container for input coupling settings."""
    def __init__(self):
        self.COUPLING_GROUND = 0
        self.COUPLING_DC = 1
        self.COUPLING_AC = 2


class SRS560_DynamicReserve:
    """Container for dynamic reserve settings."""
    def __init__(self):
        self.LOW_NOISE = 0
        self.HIGH_DR = 1
        self.CALIBRATION = 2


class SRS560_FilterMode:
    """Container for filter mode settings."""
    def __init__(self):
        self.BYPASS = 0
        self.ACTIVE = 1


class SRS560_FilterFrequency:
    """Container for filter frequency settings (both lowpass and highpass)."""
    def __init__(self):
        # Frequencies from 0.03 Hz to 1 MHz (0-15 for lowpass, 0-11 for highpass)
        self.FREQ_0p03Hz = 0
        self.FREQ_0p1Hz = 1
        self.FREQ_0p3Hz = 2
        self.FREQ_1Hz = 3
        self.FREQ_3Hz = 4
        self.FREQ_10Hz = 5
        self.FREQ_30Hz = 6
        self.FREQ_100Hz = 7
        self.FREQ_300Hz = 8
        self.FREQ_1kHz = 9
        self.FREQ_3kHz = 10
        self.FREQ_10kHz = 11
        self.FREQ_30kHz = 12      # Lowpass only
        self.FREQ_100kHz = 13     # Lowpass only
        self.FREQ_300kHz = 14     # Lowpass only
        self.FREQ_1MHz = 15       # Lowpass only


class SRS560_Settings:
    """Main container class for all SRS560 settings."""
    def __init__(self):
        self.gain = SRS560_Gain()
        self.source = SRS560_InputSource()
        self.coupling = SRS560_InputCoupling()
        self.dynamic_reserve = SRS560_DynamicReserve()
        self.filter_mode = SRS560_FilterMode()
        self.filter_freq = SRS560_FilterFrequency()



class SRS560:
    """
    Control class for the SRS560 Voltage Preamplifier via RS232.
    
    Provides methods to configure and control the SRS560 over serial connection.
    Use self.settings to access container classes for syntax highlighting.
    
    Example:
        srs = SRS560(port='COM3')
        srs.set_gain(srs.settings.gain.GAIN_100)
        srs.set_input_option(srs.settings.input_option.OPTION_A_MINUS_B)
        srs.set_lowpass_filter(srs.settings.filter.LOWPASS_10kHz)
    """
    
    def __init__(self, com_port: str = 'COM1', baudrate: int = 9600, timeout: float = 1.0):
        """
        Initialize the SRS560 controller.
        
        Args:
            port (str): Serial port (e.g., 'COM1', '/dev/ttyUSB0'). Default: 'COM1'
            baudrate (int): Serial connection baud rate. Default: 9600
            timeout (float): Read/write timeout in seconds. Default: 1.0
        """
        self.com_port = com_port
        self.baud = baudrate
        self.timeout = timeout
        self.settings = SRS560_Settings()
        
        # Initialize serial connection
        try:
            self.ser = serial.Serial(
                port=self.com_port,
                baudrate=self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout,
                write_timeout=timeout
            )
            if self.ser.is_open:
                print(f"Serial port {self.com_port} opened at {self.baud} baud.")
            else:
                print(f"Failed to open {self.com_port}.")
        except Exception as e:
            print(f"Error opening serial port {self.com_port}: {e}")
            raise
    
        self.listen_all()
    
    def close(self):
        """Close the serial connection."""
        if self.ser.is_open:
            self.ser.close()
            print(f"Serial port {self.com_port} closed.")
    
    def _send_command(self, command: str) -> bool:
        """
        Send a command to the SRS560 (listen-only mode, no response expected).
        Commands must end with carriage return and line feed (CR LF).
        
        Args:
            command (str): Command string (without line terminator, will be added automatically)
            
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        try:
            # Add CR LF terminator if not present
            if not command.endswith('\r\n'):
                command += '\r\n'
            
            self.ser.write(command.encode('utf-8'))
            print(f"Sent command: {command.strip()}")
            return True
        except Exception as e:
            print(f"Error sending command '{command.strip()}': {e}")
            return False
    
    def set_gain(self, gain: int) -> bool:
        """
        Set the preamplifier gain (GAIN command).
        
        Args:
            gain (int): Gain code from 0-14 (1, 2, 5, 10, 20, 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k, 50k)
                       Or use srs.settings.gain.GAIN_* constants
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if not (0 <= gain <= 14):
            print(f"Invalid gain value: {gain}. Valid range: 0-14")
            return False
        
        return self._send_command(f"GAIN {gain}")
    
    def set_input_coupling(self, coupling: int) -> bool:
        """
        Set the input coupling mode (CPLG command).
        
        Args:
            coupling (int): 0 = ground, 1 = DC, 2 = AC
                          Or use srs.settings.coupling.COUPLING_* constants
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if coupling not in [0, 1, 2]:
            print(f"Invalid coupling value: {coupling}. Valid values: 0 (ground), 1 (DC), 2 (AC)")
            return False
        
        return self._send_command(f"CPLG {coupling}")
    
    def set_input_source(self, source: int) -> bool:
        """
        Set the input source (SRCE command).
        
        Args:
            source (int): 0 = A, 1 = A-B, 2 = B
                         Or use srs.settings.source.SOURCE_* constants
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if source not in [0, 1, 2]:
            print(f"Invalid source value: {source}. Valid values: 0 (A), 1 (A-B), 2 (B)")
            return False
        
        return self._send_command(f"SRCE {source}")
    
    def set_dynamic_reserve(self, mode: int) -> bool:
        """
        Set the dynamic reserve mode (DYNR command).
        
        Args:
            mode (int): 0 = low noise, 1 = high DR, 2 = calibration gains (defaults)
                       Or use srs.settings.dynamic_reserve.* constants
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if mode not in [0, 1, 2]:
            print(f"Invalid dynamic reserve mode: {mode}. Valid values: 0 (low noise), 1 (high DR), 2 (calibration)")
            return False
        
        return self._send_command(f"DYNR {mode}")
    
    def set_filter_mode(self, mode: int) -> bool:
        """
        Set the filter mode (FLTM command).
        
        Args:
            mode (int): 0 = bypass, 1 = active
                       Or use srs.settings.filter_mode.* constants
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if mode not in [0, 1]:
            print(f"Invalid filter mode: {mode}. Valid values: 0 (bypass), 1 (active)")
            return False
        
        return self._send_command(f"FLTM {mode}")
    
    def set_lowpass_filter(self, frequency: int) -> bool:
        """
        Set the lowpass filter frequency (LFRQ command).
        
        Args:
            frequency (int): Filter code from 0-15
                            0-11: 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1k, 3k, 10k Hz
                            12-15: 30k, 100k, 300k, 1M Hz (lowpass only)
                           Or use srs.settings.filter_freq.FREQ_* constants
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if not (0 <= frequency <= 15):
            print(f"Invalid lowpass frequency code: {frequency}. Valid range: 0-15")
            return False
        
        return self._send_command(f"LFRQ {frequency}")
    
    def set_highpass_filter(self, frequency: int) -> bool:
        """
        Set the highpass filter frequency (HFRQ command).
        
        Args:
            frequency (int): Filter code from 0-11
                            0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1k, 3k, 10k Hz
                           Or use srs.settings.filter_freq.FREQ_* constants (0-11 only)
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if not (0 <= frequency <= 11):
            print(f"Invalid highpass frequency code: {frequency}. Valid range: 0-11")
            return False
        
        return self._send_command(f"HFRQ {frequency}")
    
    def set_blanking(self, blanked: int) -> bool:
        """
        Set the amplifier blanking mode (BLINK command).
        
        Args:
            blanked (int): 0 = not blanked, 1 = blanked
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if blanked not in [0, 1]:
            print(f"Invalid blanking value: {blanked}. Valid values: 0 (not blanked), 1 (blanked)")
            return False
        
        return self._send_command(f"BLINK {blanked}")
    
    def set_invert(self, inverted: int) -> bool:
        """
        Set the signal invert sense (INVT command).
        
        Args:
            inverted (int): 0 = non-inverted, 1 = inverted
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if inverted not in [0, 1]:
            print(f"Invalid invert value: {inverted}. Valid values: 0 (non-inverted), 1 (inverted)")
            return False
        
        return self._send_command(f"INVT {inverted}")
    
    def set_vernier_gain_status(self, status: int) -> bool:
        """
        Set the vernier gain status (UCAL command).
        
        Args:
            status (int): 0 = cal'd gain, 1 = vernier gain
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if status not in [0, 1]:
            print(f"Invalid vernier gain status: {status}. Valid values: 0 (cal'd), 1 (vernier)")
            return False
        
        return self._send_command(f"UCAL {status}")
    
    def set_vernier_gain(self, percent: int) -> bool:
        """
        Set the vernier gain to a percentage (UCGN command).
        
        Args:
            percent (int): Vernier gain percentage (0-100)
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if not (0 <= percent <= 100):
            print(f"Invalid vernier gain percentage: {percent}. Valid range: 0-100")
            return False
        
        return self._send_command(f"UCGN {percent}")
    
    def reset_overload(self) -> bool:
        """
        Reset overload for 1/2 second (ROLD command).
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        return self._send_command("ROLD")
    
    def listen(self, address: int) -> bool:
        """
        Make this SRS560 a listener (LISN command).
        
        Args:
            address (int): Device address (0, 1, 2, or 3)
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if address not in [0, 1, 2, 3]:
            print(f"Invalid address: {address}. Valid values: 0, 1, 2, 3")
            return False
        
        return self._send_command(f"LISN {address}")
    
    def listen_all(self) -> bool:
        """
        Make all attached SRS560's listeners (LALL command).
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        return self._send_command("LALL")
    
    def unlisten(self) -> bool:
        """
        Unlisten and unaddress all attached SRS560's (UNLS command).
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        return self._send_command("UNLS")
    
    def reset(self) -> bool:
        """
        Reset device to default settings (*RST command).
        
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        return self._send_command("*RST")


class SRS560_Server:
    """
    Network server for remote SRS560 control over LAN.
    
    Runs on the computer connected to the SRS560 via RS232.
    Listens for client connections and forwards commands to the device.
    
    Example:
        server = SRS560_Server(port='COM1', server_ip='192.168.1.100', server_port=5555)
        server.start()
    """
    
    def __init__(self, com_port: str = 'COM1', baudrate: int = 9600, 
                 server_ip: str = '0.0.0.0', server_port: int = 5555):
        """
        Initialize the SRS560 server.
        
        Args:
            com_port (str): Serial port for RS232 connection
            baudrate (int): Serial connection baud rate. Default: 9600
            server_ip (str): Server IP address to listen on. Default: '0.0.0.0' (all interfaces)
            server_port (int): Server port to listen on. Default: 5555
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.device = SRS560(com_port=com_port, baudrate=baudrate)
        self.running = False
        self.server_socket = None
        self.client_threads = []
        
        print(f"SRS560 Server initialized (listening on {self.server_ip}:{self.server_port})")
    
    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """
        Handle a client connection in a separate thread.
        
        Args:
            client_socket (socket.socket): Connected client socket
            client_address (tuple): Client address information
        """
        print(f"Client connected from {client_address}")
        
        try:
            while self.running:
                # Receive command from client
                data = client_socket.recv(4096).decode('utf-8')
                
                if not data:
                    break
                
                try:
                    command = json.loads(data)
                    response = self._execute_command(command)
                    client_socket.sendall(json.dumps(response).encode('utf-8'))
                except Exception as e:
                    error_response = {
                        'status': 'error',
                        'message': str(e)
                    }
                    client_socket.sendall(json.dumps(error_response).encode('utf-8'))
        
        except Exception as e:
            print(f"Error handling client {client_address}: {e}")
        
        finally:
            client_socket.close()
            print(f"Client disconnected from {client_address}")
    
    def _execute_command(self, command: dict) -> dict:
        """
        Execute a command on the SRS560 device.
        
        Args:
            command (dict): Command dictionary with 'method' and optional 'args'
            
        Returns:
            dict: Response dictionary with 'status' and 'result'
        """
        method = command.get('method')
        args = command.get('args', {})
        
        try:
            if method == 'set_gain':
                result = self.device.set_gain(args['gain'])
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'set_input_coupling':
                result = self.device.set_input_coupling(args['coupling'])
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'set_input_source':
                result = self.device.set_input_source(args['source'])
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'set_dynamic_reserve':
                result = self.device.set_dynamic_reserve(args['mode'])
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'set_filter_mode':
                result = self.device.set_filter_mode(args['mode'])
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'set_lowpass_filter':
                result = self.device.set_lowpass_filter(args['frequency'])
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'set_highpass_filter':
                result = self.device.set_highpass_filter(args['frequency'])
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'set_blanking':
                result = self.device.set_blanking(args['blanked'])
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'set_invert':
                result = self.device.set_invert(args['inverted'])
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'set_vernier_gain_status':
                result = self.device.set_vernier_gain_status(args['status'])
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'set_vernier_gain':
                result = self.device.set_vernier_gain(args['percent'])
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'reset_overload':
                result = self.device.reset_overload()
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'listen':
                result = self.device.listen(args['address'])
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'listen_all':
                result = self.device.listen_all()
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'unlisten':
                result = self.device.unlisten()
                return {'status': 'success', 'result': result, 'method': method}
            
            elif method == 'reset':
                result = self.device.reset()
                return {'status': 'success', 'result': result, 'method': method}
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'method': method}
    
    def start(self):
        """Start the server and listen for client connections."""
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.server_ip, self.server_port))
        self.server_socket.listen(5)
        
        print(f"SRS560 Server started on {self.server_ip}:{self.server_port}")
        print("Press Ctrl+C to stop the server")
        
        try:
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    thread.start()
                    self.client_threads.append(thread)
                except Exception as e:
                    if self.running:
                        print(f"Error accepting client connection: {e}")
        
        except KeyboardInterrupt:
            print("\n\nServer interrupted by user (Ctrl+C)")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the server and close all connections."""
        self.running = False
        
        if self.server_socket:
            self.server_socket.close()
        
        self.device.close()
        print("SRS560 Server stopped")


class SRS560_Client:
    """
    Network client for remote SRS560 control over LAN.
    
    Connects to an SRS560_Server running on another computer.
    Provides the same interface as SRS560 class.
    
    Example:
        client = SRS560_Client(server_ip='192.168.1.100', server_port=5555)
        client.set_gain(client.settings.gain.GAIN_100)
        client.close()
    """
    
    def __init__(self, server_ip: str = 'localhost', server_port: int = 5555, timeout: float = 5.0):
        """
        Initialize the SRS560 client.
        
        Args:
            server_ip (str): Server IP address. Default: 'localhost'
            server_port (int): Server port. Default: 5555
            timeout (float): Socket timeout in seconds. Default: 5.0
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.socket = None
        self.settings = SRS560_Settings()
        
        # Connect to server
        self._connect()
    
    def _connect(self):
        """Establish connection to the remote server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.server_ip, self.server_port))
            print(f"Connected to SRS560 Server at {self.server_ip}:{self.server_port}")
        except Exception as e:
            print(f"Error connecting to server: {e}")
            raise
    
    def close(self):
        """Close the connection to the server."""
        if self.socket:
            self.socket.close()
            print("Disconnected from SRS560 Server")
    
    def _send_command(self, method: str, **kwargs) -> dict:
        """
        Send a command to the remote server.
        
        Args:
            method (str): Method name to call on the server
            **kwargs: Arguments for the method
            
        Returns:
            dict: Response from the server
        """
        if not self.socket:
            raise ConnectionError("Not connected to server. Call _connect() first.")
        
        try:
            command = {
                'method': method,
                'args': kwargs
            }
            
            self.socket.sendall(json.dumps(command).encode('utf-8'))
            response_data = self.socket.recv(4096).decode('utf-8')
            response = json.loads(response_data)
            
            if response.get('status') == 'error':
                raise RuntimeError(f"Server error: {response.get('message')}")
            
            return response.get('result')
        
        except Exception as e:
            print(f"Error communicating with server: {e}")
            raise
    
    def set_gain(self, gain: int) -> bool:
        """Set the preamplifier gain remotely."""
        return self._send_command('set_gain', gain=gain)
    
    def set_input_coupling(self, coupling: int) -> bool:
        """Set the input coupling remotely."""
        return self._send_command('set_input_coupling', coupling=coupling)
    
    def set_input_source(self, source: int) -> bool:
        """Set the input source remotely."""
        return self._send_command('set_input_source', source=source)
    
    def set_dynamic_reserve(self, mode: int) -> bool:
        """Set the dynamic reserve mode remotely."""
        return self._send_command('set_dynamic_reserve', mode=mode)
    
    def set_filter_mode(self, mode: int) -> bool:
        """Set the filter mode remotely."""
        return self._send_command('set_filter_mode', mode=mode)
    
    def set_lowpass_filter(self, frequency: int) -> bool:
        """Set the lowpass filter remotely."""
        return self._send_command('set_lowpass_filter', frequency=frequency)
    
    def set_highpass_filter(self, frequency: int) -> bool:
        """Set the highpass filter remotely."""
        return self._send_command('set_highpass_filter', frequency=frequency)
    
    def set_blanking(self, blanked: int) -> bool:
        """Set blanking remotely."""
        return self._send_command('set_blanking', blanked=blanked)
    
    def set_invert(self, inverted: int) -> bool:
        """Set signal invert remotely."""
        return self._send_command('set_invert', inverted=inverted)
    
    def set_vernier_gain_status(self, status: int) -> bool:
        """Set vernier gain status remotely."""
        return self._send_command('set_vernier_gain_status', status=status)
    
    def set_vernier_gain(self, percent: int) -> bool:
        """Set vernier gain percentage remotely."""
        return self._send_command('set_vernier_gain', percent=percent)
    
    def reset_overload(self) -> bool:
        """Reset overload remotely."""
        return self._send_command('reset_overload')
    
    def listen(self, address: int) -> bool:
        """Make device a listener remotely."""
        return self._send_command('listen', address=address)
    
    def listen_all(self) -> bool:
        """Make all devices listeners remotely."""
        return self._send_command('listen_all')
    
    def unlisten(self) -> bool:
        """Unlisten all devices remotely."""
        return self._send_command('unlisten')
    
    def reset(self) -> bool:
        """Reset device remotely."""
        return self._send_command('reset')


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        # Server mode: Connect to SRS560 and listen for clients
        # python srs560.py server [port] [com_port]
        com_port = sys.argv[3] if len(sys.argv) > 3 else 'COM1'
        server_port = int(sys.argv[2]) if len(sys.argv) > 2 else 5555
        
        server = SRS560_Server(port=com_port, server_port=server_port)
        try:
            server.start()
        except KeyboardInterrupt:
            server.stop()
    
    else:
        # Direct mode or client mode
        if len(sys.argv) > 1 and sys.argv[1] == 'client':
            # Client mode: Connect to remote server
            # python srs560.py client [server_ip] [port]
            server_ip = sys.argv[2] if len(sys.argv) > 2 else 'localhost'
            server_port = int(sys.argv[3]) if len(sys.argv) > 3 else 5555
            
            client = SRS560_Client(server_ip=server_ip, server_port=server_port)
        else:
            # Direct mode: Connect directly to SRS560 via COM port
            # python srs560.py [com_port]
            com_port = sys.argv[1] if len(sys.argv) > 1 else 'COM1'
            client = SRS560(port=com_port)
        
        try:
            # Set various parameters using container classes for syntax highlighting
            client.set_gain(client.settings.gain.GAIN_100)
            time.sleep(0.1)
            
            client.set_gain_decade(client.settings.gain_decade.DECADE_10)
            time.sleep(0.1)
            
            client.set_input_option(client.settings.input_option.OPTION_A_MINUS_B)
            time.sleep(0.1)
            
            client.set_lowpass_filter(client.settings.filter.LOWPASS_10kHz)
            time.sleep(0.1)
            
            client.set_highpass_filter(client.settings.filter.HIGHPASS_1Hz)
            time.sleep(0.1)
            
            client.set_filter_slope(client.settings.filter.SLOPE_12dB)
            time.sleep(0.1)
            
            print("Settings sent successfully to SRS560")
            
        finally:
            client.close()