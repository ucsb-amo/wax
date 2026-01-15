from pathlib import Path
from typing import Optional, List, Tuple, Dict, Set
import os
import json
import time
import numpy as np

from artiq.language.core import kernel, kernel_from_string, delay, now_mu
from artiq.coredevice.core import Core

# from waxx.control.artiq import DDS, DAC_CH, TTL_OUT, TTL_IN

# from waxx.util.artiq.async_print import aprint

DEFAULT_UPDATE_2FLOAT = (-1, 0.0, 0.0)
DEFAULT_UPDATE_FLOAT = (-1, 0.0)
DEFAULT_UPDATE_BOOL = (-1, False)
DEFAULT_UPDATE_INT = (-1, 0)

T_MONITOR_UPDATE_INTERVAL = 0.1

from waxx.util.comms_server.comm_client import MonitorClient
from waxx.util.device_state.generate_state_file import Generator

class Monitor:
    """
    Detects changes in device state configuration and updates hardware devices.
    """
     
    def __init__(self, expt, monitor_server_ip, device_state_json_path):
        """
        Initialize the device state updater.
        
        Args:
            config_file: Path to device state config file. If None, uses default location.
        """
        self.config_file = device_state_json_path
        
        self.last_config_data = None

        # Preallocate kernel function lists
        self.dds_frequency_amplitude_kernels = []
        self.dds_vpd_kernels = []
        self.dds_sw_state_kernels = []
        self.ttl_kernels = []
        self.dac_kernels = []

        self.expt = expt

        self._monitor_client = MonitorClient(monitor_server_ip)

    def update_device_states(self):
        self.generator.generate()

    def signal_end(self):
        self._monitor_client.send_end()

    def signal_ready(self):
        self._monitor_client.send_ready()

    def init_monitor(self):

        self.clear_update_lists()

        self.core: Core = self.expt.core
        self.dds = self.expt.dds
        self.dac = self.expt.dac
        self.ttl = self.expt.ttl

        from waxx.config.dds_id import dds_frame
        from waxx.config.dac_id import dac_frame
        from waxx.config.ttl_id import ttl_frame
        self.dds: dds_frame = self.expt.dds
        self.dac: dac_frame = self.expt.dac
        self.ttl: ttl_frame = self.expt.ttl

        self.generator = Generator(self.dds,self.ttl,self.dac,
                                   self.config_file,
                                   verbose = False)

        self.build_device_lookup()

    def clear_update_lists(self):
        N = 500
        self.dds_frequency_amplitude_updates = [DEFAULT_UPDATE_2FLOAT] * N
        self.dds_vpd_updates = [DEFAULT_UPDATE_FLOAT] * N
        self.dds_sw_state_updates = [DEFAULT_UPDATE_INT] * N
        self.ttl_updates = [DEFAULT_UPDATE_INT] * N
        self.dac_updates = [DEFAULT_UPDATE_FLOAT] * N
    
    def build_device_lookup(self):
        """Build lookup dictionaries and preallocate kernel function lists."""
        self.dds_dict = {}
        self.ttl_dict = {}
        self.dac_dict = {}

        from waxx.control.artiq.DDS import DDS
        from waxx.control.artiq.DAC_CH import DAC_CH
        from waxx.control.artiq.TTL import TTL_OUT

        # Build DDS device kernels
        dds_idx = 0
        for attr_name in dir(self.dds):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.dds, attr_name)
                if isinstance(attr_value, DDS):
                    self.dds_dict[attr_name] = dds_idx
                    self.dds_frequency_amplitude_kernels.append(kernel_from_string(
                        ["expt","f", "a"],
                        f"expt.dds.{attr_name}.set_dds(frequency=f, amplitude=a)"
                    ))
                    self.dds_vpd_kernels.append(kernel_from_string(
                        ["expt","v_pd_val"],
                        f"expt.dds.{attr_name}.set_dds(v_pd=v_pd_val)"
                    ))
                    self.dds_sw_state_kernels.append(kernel_from_string(
                        ["expt","state"],
                        f"expt.dds.{attr_name}.set_sw(state);"
                    ))
                    dds_idx += 1

        ttl_idx = 0
        # Build TTL device kernels
        for attr_name in dir(self.ttl):
            if not attr_name.startswith('_') and attr_name not in ['ttl_list', 'camera']:
                attr_value = getattr(self.ttl, attr_name)
                if isinstance(attr_value, (TTL_OUT)):
                    self.ttl_dict[attr_name] = ttl_idx
                    self.ttl_kernels.append(kernel_from_string(
                        ["expt","state"],
                        f"expt.ttl.{attr_name}.set_state(state)"
                    ))
                    ttl_idx += 1
        

        dac_idx = 0
        # Build DAC device kernels
        for attr_name in dir(self.dac):
            if not attr_name.startswith('_') and attr_name not in ['dac_device', 'dac_ch_list']:
                attr_value = getattr(self.dac, attr_name)
                if isinstance(attr_value, DAC_CH):
                    self.dac_dict[attr_name] = dac_idx
                    self.dac_kernels.append(kernel_from_string(
                        ["expt","v"],
                        f"expt.dac.{attr_name}.set(v)"
                    ))
                    dac_idx += 1

    def check_config_key_alignment(self, verbose: bool = True) -> Tuple[bool, Dict[str, Dict[str, List[str]]]]:
        """Validate that lookup keys for DDS/TTL/DAC match what is stored in the JSON."""
        config_data = self.load_config_file()
        if config_data is None:
            if verbose:
                print("Unable to validate config keys because the config file could not be loaded.")
            empty_report = {k: [] for k in ["dds", "ttl", "dac"]}
            return False, {"missing_in_config": empty_report, "missing_in_lookup": empty_report}

        lookup_sets: Dict[str, Set[str]] = {
            "dds": set(self.dds_dict.keys()),
            "ttl": set(self.ttl_dict.keys()),
            "dac": set(self.dac_dict.keys()),
        }

        config_sets: Dict[str, Set[str]] = {
            "dds": set(config_data.get("dds", {}).keys()),
            "ttl": set(config_data.get("ttl", {}).keys()),
            "dac": set(config_data.get("dac", {}).keys()),
        }

        missing_in_config = {k: sorted(lookup_sets[k] - config_sets[k]) for k in lookup_sets}
        missing_in_lookup = {k: sorted(config_sets[k] - lookup_sets[k]) for k in lookup_sets}

        matches_bool = all(len(missing_in_config[k]) == 0 and len(missing_in_lookup[k]) == 0 for k in lookup_sets)

        if verbose:
            if matches_bool:
                print("Device keys in lookup dictionaries match the configuration file.")
            else:
                print("Device key mismatch detected between lookup dictionaries and configuration file.")
                for dev_type in ["dds", "ttl", "dac"]:
                    if missing_in_config[dev_type]:
                        print(f"  {dev_type.upper()} missing in config: {missing_in_config[dev_type]}")
                    if missing_in_lookup[dev_type]:
                        print(f"  {dev_type.upper()} missing in lookup: {missing_in_lookup[dev_type]}")

        # return matches, {"missing_in_config": missing_in_config, "missing_in_lookup": missing_in_lookup}
        return matches_bool

    def load_config_file(self) -> Optional[dict]:
        """Load configuration file and return data, retrying if file is in use."""
        max_attempts = 100
        wait_time = 0.05
        attempts = 0

        while attempts < max_attempts:
            try:
                if not os.path.isfile(self.config_file):
                    print(f"Config file {self.config_file} does not exist")
                    return None

                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                # Check for file-in-use error (Windows: PermissionError, OSError with errno 13)
                if isinstance(e, PermissionError) or (hasattr(e, 'errno') and e.errno == 13):
                    attempts += 1
                    if attempts % 20 == 0:
                        print(f"Warning: Config file {self.config_file} is in use by another process (attempt {attempts})")
                    time.sleep(wait_time)
                    continue
                print(f"Error loading config file: {e}")
                return None
        print(f"Failed to load config file {self.config_file} after {max_attempts} attempts due to file being in use.")
        return None

    def detect_changes(self, verbose: bool = True) -> Tuple[
        List[Tuple[np.int32, float, float]],
        List[Tuple[np.int32, float]],
        List[Tuple[np.int32, np.int32]],
        List[Tuple[np.int32, np.int32]],
        List[Tuple[np.int32, float]]]:
        """
        Detect changes in the configuration file and populate update lists.
        
        Args:
            verbose: If True, print information about detected changes.
            
        Returns:
            Tuple of all update lists.
        """
        current_config = self.load_config_file()
        if current_config is None:
            if verbose:
                print("No changes detected (config file could not be loaded).")
            return (self.dds_frequency_amplitude_updates, self.dds_vpd_updates, 
                    self.dds_sw_state_updates, self.ttl_updates, self.dac_updates)

        self.clear_update_lists()

        if self.last_config_data is None:
            self.last_config_data = current_config
            if verbose:
                print("No changes detected (initial load).")
            return (self.dds_frequency_amplitude_updates, self.dds_vpd_updates, 
                    self.dds_sw_state_updates, self.ttl_updates, self.dac_updates)

        changes_detected = False

        # Process DDS devices
        old_dds = self.last_config_data.get('dds', {})
        new_dds = current_config.get('dds', {})
        for device_name, new_config in new_dds.items():
            if device_name not in self.dds_dict:
                continue
            old_config = old_dds.get(device_name, {})
            kernel_index = self.dds_dict.get(device_name, -1)

            if old_config.get('frequency') != new_config.get('frequency') or \
            old_config.get('amplitude') != new_config.get('amplitude'):
                update_index = self.dds_frequency_amplitude_updates.index(DEFAULT_UPDATE_2FLOAT)
                self.dds_frequency_amplitude_updates[update_index] = (
                    kernel_index, new_config['frequency'], new_config['amplitude'])
                changes_detected = True
                if verbose:
                    print(f"DDS {device_name}: Frequency/Amplitude changed to {new_config['frequency']}/{new_config['amplitude']}")

            if old_config.get('v_pd') != new_config.get('v_pd'):
                update_index = self.dds_vpd_updates.index(DEFAULT_UPDATE_FLOAT)
                self.dds_vpd_updates[update_index] = (kernel_index, new_config['v_pd'])
                changes_detected = True
                if verbose:
                    print(f"DDS {device_name}: V_PD changed to {new_config['v_pd']}")

            if old_config.get('sw_state') != new_config.get('sw_state'):
                update_index = self.dds_sw_state_updates.index(DEFAULT_UPDATE_BOOL)
                self.dds_sw_state_updates[update_index] = (kernel_index, new_config['sw_state'])
                changes_detected = True
                if verbose:
                    print(f"DDS {device_name}: SW State changed to {new_config['sw_state']}")

        # Process TTL devices
        old_ttl = self.last_config_data.get('ttl', {})
        new_ttl = current_config.get('ttl', {})
        for device_name, new_config in new_ttl.items():
            if device_name not in self.ttl_dict:
                continue
            if old_ttl.get(device_name, {}).get('ttl_state') != new_config.get('ttl_state'):
                kernel_index = self.ttl_dict.get(device_name, -1)
                update_index = self.ttl_updates.index(DEFAULT_UPDATE_BOOL) # get next update from start of list
                self.ttl_updates[update_index] = (kernel_index, new_config['ttl_state'])
                changes_detected = True
                if verbose:
                    print(f"TTL {device_name}: State changed to {new_config['ttl_state']}")

        # Process DAC devices
        old_dac = self.last_config_data.get('dac', {})
        new_dac = current_config.get('dac', {})
        for device_name, new_config in new_dac.items():
            if device_name not in self.dac_dict:
                continue
            if abs(old_dac.get(device_name, {}).get('voltage', 0.0) - new_config.get('voltage', 0.0)) > 1e-6:
                kernel_index = self.dac_dict.get(device_name, -1)
                update_index = self.dac_updates.index(DEFAULT_UPDATE_FLOAT)
                self.dac_updates[update_index] = (kernel_index, new_config['voltage'])
                changes_detected = True
                if verbose:
                    print(f"DAC {device_name}: Voltage changed to {new_config['voltage']}")

        self.last_config_data = current_config

        if verbose and not changes_detected:
            print("No changes detected.")

        return (self.dds_frequency_amplitude_updates, self.dds_vpd_updates, 
                self.dds_sw_state_updates, self.ttl_updates, self.dac_updates)

    @kernel
    def sync_change_list(self,verbose=True):
        """
        Synchronize kernel variables with the non-kernel update lists.
        """
        (self.dds_frequency_amplitude_updates, self.dds_vpd_updates, \
          self.dds_sw_state_updates, self.ttl_updates, self.dac_updates) = self.detect_changes(verbose=verbose)

    @kernel
    def apply_updates(self):
        """
        Apply the detected updates to the hardware devices.
        """
        index = -1
        f = 0.
        a = 0.
        v_pd = 0.
        sw_state = 0
        ttl_state = 0
        v = 0.
        N = len(self.dds_frequency_amplitude_updates)
        t0 = 8.e-9

        for i in range(N):
            index, f, a = self.dds_frequency_amplitude_updates[i]
            if index == -1:
                break
            self.dds_frequency_amplitude_kernels[index](self.expt,f, a)
            delay(t0)

        for i in range(N):
            index, v_pd = self.dds_vpd_updates[i]
            if index == -1:
                break
            self.dds_vpd_kernels[index](self.expt,v_pd)
            delay(t0)

        for i in range(N):
            index, sw_state = self.dds_sw_state_updates[i]
            if index == -1:
                break
            self.dds_sw_state_kernels[index](self.expt,sw_state)
            delay(t0)

        for i in range(N):
            index, ttl_state = self.ttl_updates[i]
            if index == -1:
                break
            self.ttl_kernels[index](self.expt,ttl_state)
            delay(t0)

        for i in range(N):
            index, v = self.dac_updates[i]
            if index == -1:
                break
            self.dac_kernels[index](self.expt,v)
            delay(t0)
        
    def monitor_loop(self, verbose=False):
        self.signal_ready()
        while True:
            self.core.wait_until_mu(now_mu())
            if self.check_config_key_alignment():
                self.generator.generate()
                break
            self.sync_change_list(verbose=verbose)
            self.core.break_realtime()
            self.apply_updates()
            delay(T_MONITOR_UPDATE_INTERVAL)
        delay(1.)
        self.core.wait_until_mu(now_mu())
        self.signal_end()