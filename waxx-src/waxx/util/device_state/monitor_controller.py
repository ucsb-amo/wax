"""
MonitorController - A programmatic interface for controlling devices via JSON configuration.

This module provides a clean API for accessing and modifying device parameters
(DDS, DAC, TTL) from Python scripts and Jupyter notebooks.

Example usage:
    controller = MonitorController(json_path="/path/to/config.json")
    
    # Access DDS devices
    controller.dds.MyDDS.set(frequency=100e6, amplitude=0.5, sw_state=1)
    
    # Access DAC devices
    controller.dac.MyDAC.set(voltage=5.0)
    
    # Access TTL devices
    controller.ttl.MyTTL.set(ttl_state=1)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List


class Device:
    """Base class for a single device (DDS, DAC, or TTL)"""
    
    def __init__(self, name: str, config: Dict[str, Any], parent_frame: 'DeviceFrame'):
        self.name = name
        self.config = config
        self.parent_frame = parent_frame
        
    def set(self, **kwargs) -> None:
        """
        Set device parameters.

        For DDS: frequency (Hz), amplitude (0-1), v_pd (V), sw_state (0/1)
        For DAC: voltage (V)
        For TTL: ttl_state (0/1)

        The delta is sent to the monitor server (the sole writer of the JSON);
        if nothing changed, no message is sent.  Raises on server error.
        """
        if not kwargs:
            return
        self.parent_frame.controller._send_update(
            self.parent_frame.device_type, self.name, kwargs)
        self.config.update(kwargs)
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.name}')"
    
    def __str__(self) -> str:
        return f"{self.name}: {self.config}"


class DDSDevice(Device):
    """Represents a single DDS device"""
    
    @property
    def frequency(self) -> float:
        """Get frequency in Hz"""
        return self.config.get("frequency", 0.0)
    
    @property
    def amplitude(self) -> float:
        """Get amplitude (0-1)"""
        return self.config.get("amplitude", 0.0)
    
    @property
    def v_pd(self) -> float:
        """Get photodiode voltage in V"""
        return self.config.get("v_pd", 0.0)
    
    @property
    def sw_state(self) -> int:
        """Get switch state (0=off, 1=on)"""
        return self.config.get("sw_state", 0)
    
    @property
    def urukul_idx(self) -> int:
        """Get Urukul index"""
        return self.config.get("urukul_idx", 0)
    
    @property
    def ch(self) -> int:
        """Get channel number"""
        return self.config.get("ch", 0)


class DACDevice(Device):
    """Represents a single DAC device"""
    
    @property
    def voltage(self) -> float:
        """Get voltage in V"""
        return self.config.get("voltage", 0.0)
    
    @property
    def ch(self) -> int:
        """Get channel number"""
        return self.config.get("ch", 0)


class TTLDevice(Device):
    """Represents a single TTL device"""
    
    @property
    def ttl_state(self) -> int:
        """Get TTL state (0=off, 1=on)"""
        return self.config.get("ttl_state", 0)
    
    @property
    def ch(self) -> int:
        """Get channel number"""
        return self.config.get("ch", 0)

class DeviceFrame:
    """Container for devices of a specific type (DDS, DAC, or TTL)"""
    
    def __init__(self, device_type: str, controller: 'MonitorController', device_class):
        self.device_type = device_type
        self.controller = controller
        self.device_class = device_class
        self._devices: Dict[str, Device] = {}
        
    def add_device(self, name: str, config: Dict[str, Any]) -> None:
        """Add a device to this frame"""
        device = self.device_class(name, config, self)
        self._devices[name] = device
        setattr(self, name, device)
        
    def get_devices(self) -> Dict[str, Device]:
        """Get all devices in this frame"""
        return self._devices.copy()
    
    def __repr__(self) -> str:
        devices = list(self._devices.keys())
        return f"{self.device_type.upper()}Frame({devices})"
    
    def __str__(self) -> str:
        return repr(self)

class MonitorController:
    """
    Main controller for managing device configurations via JSON file.
    
    Provides frame-like objects for DDS, DAC, and TTL devices that can be
    accessed and modified programmatically.
    """
    
    def __init__(self, json_path=None, discovery_timeout: float = 3.0):
        """
        Initialize the MonitorController.

        Connects to the monitor server (discovered over UDP) and pulls the
        current device state.  No shared-drive / file access is required, so
        this works from any lab machine.  The ``json_path`` argument is kept
        for backward compatibility with existing callers but is no longer used
        for I/O.

        Raises:
            RuntimeError: if the monitor server cannot be reached.
        """
        self.json_path = json_path
        self.config_data: Dict[str, Any] = {}
        self._version = None

        # Create device frames
        self.dds = DeviceFrame("dds", self, DDSDevice)
        self.dac = DeviceFrame("dac", self, DACDevice)
        self.ttl = DeviceFrame("ttl", self, TTLDevice)

        from waxx.util.comms_server.comm_client import MonitorClient  # noqa: PLC0415
        try:
            self._client = MonitorClient(discovery_timeout=discovery_timeout)
        except Exception as e:
            raise RuntimeError(
                "MonitorController could not reach the monitor server. "
                "Make sure the monitor server is running on the experiment PC."
            ) from e

        # Load initial configuration
        self.load_config()

    def load_config(self) -> None:
        """Pull the full device state from the monitor server."""
        state = self._client.get_state()
        if not state or state.get("status") != "ok":
            msg = (state or {}).get("msg", "no response")
            raise RuntimeError(f"Failed to load device state from server: {msg}")
        self.config_data = state.get("config", {}) or {}
        self._version = state.get("version")

        # Populate device frames
        self._populate_frames()

    def _send_update(self, device_type: str, device_name: str, changes: Dict[str, Any]) -> None:
        """Send a delta to the server and block on its ack."""
        ack = self._client.send_update(device_type, device_name, changes)
        if ack is None:
            raise RuntimeError(
                f"Monitor server unreachable; '{device_name}' was not updated."
            )
        if ack.get("status") != "ok":
            raise RuntimeError(
                f"Monitor server rejected update to '{device_name}': {ack.get('msg')}"
            )
        self._version = ack.get("version", self._version)
    
    def _populate_frames(self) -> None:
        """Populate device frames from config data"""
        # Clear existing devices
        self.dds._devices.clear()
        self.dac._devices.clear()
        self.ttl._devices.clear()
        
        # Remove old device attributes (but keep the frame objects)
        for attr in dir(self.dds):
            if not attr.startswith('_') and attr not in ['device_type', 'controller', 'device_class', 
                                                          'add_device', 'get_devices']:
                delattr(self.dds, attr)
        for attr in dir(self.dac):
            if not attr.startswith('_') and attr not in ['device_type', 'controller', 'device_class', 
                                                          'add_device', 'get_devices']:
                delattr(self.dac, attr)
        for attr in dir(self.ttl):
            if not attr.startswith('_') and attr not in ['device_type', 'controller', 'device_class', 
                                                          'add_device', 'get_devices']:
                delattr(self.ttl, attr)
        
        # Load DDS devices
        if "dds" in self.config_data:
            for device_name, device_config in self.config_data["dds"].items():
                self.dds.add_device(device_name, device_config)
        
        # Load DAC devices
        if "dac" in self.config_data:
            for device_name, device_config in self.config_data["dac"].items():
                self.dac.add_device(device_name, device_config)
        
        # Load TTL devices
        if "ttl" in self.config_data:
            for device_name, device_config in self.config_data["ttl"].items():
                self.ttl.add_device(device_name, device_config)
    
    def save_config(self) -> None:
        """Deprecated: the server is the sole writer of the JSON.

        Kept as a no-op for backward compatibility; per-device changes are
        persisted by the server when :meth:`Device.set` sends a delta.
        """
        pass

    def reload_config(self) -> None:
        """Reload configuration from the monitor server."""
        self.load_config()
    
    def get_device(self, device_type: str, device_name: str) -> Device:
        """
        Get a device by type and name.
        
        Args:
            device_type: "dds", "dac", or "ttl"
            device_name: Name of the device
            
        Returns:
            Device object
        """
        frame_map = {
            "dds": self.dds,
            "dac": self.dac,
            "ttl": self.ttl,
        }
        
        frame = frame_map.get(device_type.lower())
        if not frame:
            raise ValueError(f"Unknown device type: {device_type}")
        
        devices = frame.get_devices()
        if device_name not in devices:
            raise KeyError(f"Device '{device_name}' not found in {device_type}")
        
        return devices[device_name]
    
    def list_devices(self, device_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all devices, optionally filtered by type.
        
        Args:
            device_type: "dds", "dac", "ttl", or None for all types
            
        Returns:
            Dictionary mapping device types to lists of device names
        """
        result = {}
        
        if device_type is None or device_type.lower() == "dds":
            result["dds"] = list(self.dds.get_devices().keys())
        if device_type is None or device_type.lower() == "dac":
            result["dac"] = list(self.dac.get_devices().keys())
        if device_type is None or device_type.lower() == "ttl":
            result["ttl"] = list(self.ttl.get_devices().keys())
        
        return result
    
    def __repr__(self) -> str:
        dds_count = len(self.dds.get_devices())
        dac_count = len(self.dac.get_devices())
        ttl_count = len(self.ttl.get_devices())
        return (f"MonitorController(json_path={self.json_path}, "
                f"dds={dds_count}, dac={dac_count}, ttl={ttl_count})")
    
    def __str__(self) -> str:
        return repr(self)