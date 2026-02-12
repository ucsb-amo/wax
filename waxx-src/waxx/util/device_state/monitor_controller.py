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
        """
        self.config.update(kwargs)
        self.parent_frame.controller.save_config()
        
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
    
    def __init__(self, json_path):
        """
        Initialize the MonitorController.
        
        Args:
            json_path: Path to the device configuration JSON file.
                      If not provided, looks for a default location or prompts user.
        """
        self.json_path = Path(json_path)
        self.config_data: Dict[str, Any] = {}
        
        # Create device frames
        self.dds = DeviceFrame("dds", self, DDSDevice)
        self.dac = DeviceFrame("dac", self, DACDevice)
        self.ttl = DeviceFrame("ttl", self, TTLDevice)
        
        # Load initial configuration
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from JSON file and populate device frames"""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.json_path}")
        
        try:
            with open(self.json_path, 'r') as f:
                self.config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        
        # Populate device frames
        self._populate_frames()
    
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
        """Save current configuration back to JSON file"""
        try:
            with open(self.json_path, 'w') as f:
                json.dump(self.config_data, f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to save configuration to {self.json_path}: {e}")
    
    def reload_config(self) -> None:
        """Reload configuration from JSON file (useful if file was modified externally)"""
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