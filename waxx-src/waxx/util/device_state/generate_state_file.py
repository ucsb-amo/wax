#!/usr/bin/env python3
"""
Script to generate device state configuration files from _id files.

This script reads all *_id.py files in the config folder (except camera_id.py),
extracts devices that are assigned using the assign methods, and creates a
configuration file organized by device type (DDS, TTL, DAC) with current
state values for each device.

For DDS devices: frequency, amplitude, v_pd (voltage), sw_state, and state (on/off)
For TTL devices: state (on/off)  
For DAC devices: voltage
"""

import os
import sys

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# script_dir = Path(__file__).parent
kexp_root = Path(os.getenv('code')) / 'k-exp'
config_file_path_dir = Path(os.getenv('data'))
sys.path.insert(0, str(kexp_root))
    
class Generator():
    def __init__(self,dds_frame,ttl_frame,dac_frame,
                 state_file_path,
                 verbose = True):
        self.dds = dds_frame
        self.ttl = ttl_frame
        self.dac = dac_frame

        self._state_file = state_file_path
        self._verbose = verbose

        self.config_data: Dict = None

    def generate(self):
        self._generate_device_config()
        
        if self.config_data is None:
            print("Failed to generate configuration data.")
            return 1
        
        if self._verbose:
            self.print_summary()

        output_file = self.save_config_file()

        if self._verbose:
            if output_file:
                print(f"\nTotal devices found:")
                print(f"  DDS: {len(self.config_data['dds'])}")
                print(f"  TTL: {len(self.config_data['ttl'])}")
                print(f"  DAC: {len(self.config_data['dac'])}")
                return 0
            else:
                return 1
        else:
            print("Device states updated.")

    def extract_dds_devices(self) -> Dict[str, Dict[str, Any]]:
        """Extract DDS device states from a dds_frame object."""
        devices = {}

        from waxx.control.artiq.DDS import DDS
        
        # Iterate over _dds_list to extract DDS devices
        dds_list = getattr(self.dds, 'dds_list', [])
        for dds_device in dds_list:
            if isinstance(dds_device, DDS):
                devices[dds_device.key] = {
                    'frequency': getattr(dds_device, 'frequency', 0.0),
                    'amplitude': getattr(dds_device, 'amplitude', 0.0),
                    'v_pd': getattr(dds_device, 'v_pd', 0.0),  # voltage photodiode setpoint
                    'urukul_idx': getattr(dds_device, 'urukul_idx', 0),
                    'ch': getattr(dds_device, 'ch', 0),
                    'sw_state': getattr(dds_device, 'sw_state', 0),  # Hardware switch state (0/1)
                    'transition': getattr(dds_device, 'transition', 'None'),
                    'aom_order': getattr(dds_device, 'aom_order', 0),
                    'dac_ch': getattr(dds_device, 'dac_ch', -1)
                }
        
        return devices

    def extract_ttl_devices(self) -> Dict[str, Dict[str, Any]]:
        """Extract TTL device states from a ttl_frame object."""
        devices = {}

        from waxx.control.artiq.TTL import TTL_OUT
        
        # Iterate over _ttl_list to extract TTL devices
        ttl_list = getattr(self.ttl, 'ttl_list', [])
        for ttl_device in ttl_list:
            if isinstance(ttl_device, TTL_OUT):
                # Determine the specific TTL type
                ttl_type = 'out'
                
                # Get actual state from the TTL object
                ttl_state = getattr(ttl_device, 'state', 0)
                
                devices[ttl_device.key] = {
                    'ch': getattr(ttl_device, 'ch', 0),
                    'ttl_state': ttl_state,  # Raw state value (0/1)
                    'type': ttl_type
                }
        
        return devices

    def extract_dac_devices(self) -> Dict[str, Dict[str, Any]]:
        """Extract DAC device states from a dac_frame object."""
        devices = {}

        from waxx.control.artiq.DAC_CH import DAC_CH

        # Iterate over _dac_ch_list to extract DAC devices
        dac_ch_list = getattr(self.dac, 'dac_ch_list', [])
        for dac_device in dac_ch_list:
            if isinstance(dac_device, DAC_CH):
                devices[dac_device.key] = {
                    'ch': getattr(dac_device, 'ch', 0),
                    'voltage': getattr(dac_device, 'v', 0.0),
                    'max_voltage': getattr(dac_device, 'max_v', 9.99)
                }
        
        return devices

    def _generate_device_config(self):
        """Generate device configuration from all _id files."""
        
        device_config = {
            'dds': {},
            'ttl': {},
            'dac': {},
            'metadata': {
                'timestamp': None
            }
        }
        
        dds_devices = self.extract_dds_devices()
        device_config['dds'].update(dds_devices)
        

        ttl_devices = self.extract_ttl_devices()
        device_config['ttl'].update(ttl_devices)
        
                
        dac_devices = self.extract_dac_devices()
        device_config['dac'].update(dac_devices)

        if self._verbose:
            print(f"  Found {len(ttl_devices)} TTL devices")
            print(f"  Found {len(dds_devices)} DDS devices")
            print(f"  Found {len(dac_devices)} DAC devices")
        
        # Add timestamp
        from datetime import datetime
        device_config['metadata']['timestamp'] = datetime.now().isoformat()

        self.config_data = device_config

    def save_config_file(self):
        """Save the configuration data to a JSON file."""
        output_file = self._state_file
        
        # Helper function to convert numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        try:
            with open(output_file, 'w') as f:
                json.dump(self.config_data, f, indent=2, sort_keys=True, default=convert_numpy_types)
            if self._verbose:
                print(f"\nConfiguration saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving configuration file: {e}")
            return None

    def print_summary(self):
        """Print a summary of the generated configuration."""
        print("\n" + "="*60)
        print("DEVICE CONFIGURATION SUMMARY")
        print("="*60)
        
        for device_type in ['dds', 'ttl', 'dac']:
            devices = self.config_data.get(device_type, {})
            print(f"\n{device_type.upper()} DEVICES ({len(devices)} total):")
            print("-" * 40)
            
            for name, props in devices.items():
                if device_type == 'dds':
                    print(f"  {name:25} | Freq: {props['frequency']:12.1f} Hz | "
                        f"Amp: {props['amplitude']:5.3f} | V_pd: {props['v_pd']:6.3f} V | "
                        f"SW: {props['sw_state']}")
                elif device_type == 'ttl':
                    print(f"  {name:25} | Ch: {props['ch']:2d} | Type: {props['type']:3s} | "
                        f"TTL: {props['ttl_state']}")
                elif device_type == 'dac':
                    print(f"  {name:25} | Ch: {props['ch']:2d} | "
                        f"Voltage: {props['voltage']:6.3f} V | Max: {props['max_voltage']:6.3f} V")
