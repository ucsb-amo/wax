from pathlib import Path
from typing import Optional, List, Tuple
import os
import json
import time
import numpy as np

from artiq.language.core import kernel, kernel_from_string, delay, now_mu
from artiq.language import TBool
from artiq.coredevice.core import Core

# from waxx.control.artiq import DDS, DAC_CH, TTL_OUT, TTL_IN

# from waxx.util.artiq.async_print import aprint

DEFAULT_UPDATE_2FLOAT = (-1, 0.0, 0.0)
DEFAULT_UPDATE_FLOAT = (-1, 0.0)
DEFAULT_UPDATE_BOOL = (-1, False)
DEFAULT_UPDATE_INT = (-1, 0)

T_MONITOR_UPDATE_INTERVAL = 0.1

# After a failed get_version request, do not ask the server again for this
# long (the file is read directly meanwhile).  A dead server would otherwise
# add the client's connect/rediscover timeouts to every loop iteration.
T_VERSION_PROBE_BACKOFF = 5.0

from waxx.util.comms_server.comm_client import MonitorClient
from waxx.util.device_state.generate_state_file import Generator

class Monitor:
    """
    Detects changes in device state configuration and updates hardware devices.
    """
     
    def __init__(self, expt, device_state_json_path):
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

        self._monitor_client = MonitorClient()

        # Device-state version last seen from the server (``get_version``).
        # ``detect_changes`` skips the JSON read while it is unchanged; ``None``
        # forces a read on the next call.
        self._last_seen_version: Optional[int] = None
        self._version_probe_retry_after = 0.0

        self._schema_changed = False

        # Tracks the last force_update_counter seen per device so that an
        # incremented counter unconditionally queues all params for that device.
        self._force_update_counters: dict = {}

    def update_device_states(self):
        self.generator.generate()

    def signal_end(self):
        self._monitor_client.send_end()

    def signal_ready(self):
        self._monitor_client.send_ready()

    def schema_changed(self) -> TBool:
        """Return True (and reset the flag) if JSON device keys changed since init."""
        result = self._schema_changed
        self._schema_changed = False
        return result

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

        self.reconcile_state_file()

    def reconcile_state_file(self):
        """Match the state file's key set to the device frames before looping.

        The _id files are the source of truth for which devices exist; the JSON
        holds their live values.  Edit an _id file and the file on disk still
        carries the old key set, so ``detect_changes`` sees keys it has no
        kernel for and asks for a restart -- which by itself fixes nothing and
        leaves the monitor restarting forever.  Rebuilding the schema here (keys
        from the frames, values from the file, migrated by channel across a
        rename) is what actually clears it.
        """
        try:
            report = self.generator.reconcile(known={'dds': self.dds_dict,
                                                     'ttl': self.ttl_dict,
                                                     'dac': self.dac_dict})
        except Exception as e:
            print(f"[Monitor] Could not reconcile the device state file: {e!r}")
            return

        if report['orphaned']:
            print(f"[Monitor] WARNING: {report['orphaned']} are listed by the device "
                  f"frames but are not attributes of them, so the monitor cannot "
                  f"drive them. Dropped from the device state file.")
        if not report['changed']:
            return
        for old_key, new_key in report['renamed']:
            print(f"[Monitor] Device state file: {old_key} renamed to {new_key} "
                  f"(same channel, state carried over).")
        if report['added']:
            print(f"[Monitor] Device state file: added {report['added']}.")
        if report['removed']:
            print(f"[Monitor] Device state file: removed {report['removed']} "
                  f"(no longer in the device frames).")
        print("[Monitor] Device state file rebuilt from the device frames.")

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
                        f"expt.dds.{attr_name}.set_dds(frequency=f, amplitude=a, init=True)"
                    ))
                    self.dds_vpd_kernels.append(kernel_from_string(
                        ["expt","v_pd_val"],
                        f"expt.dds.{attr_name}.set_dds(v_pd=v_pd_val, init=True)"
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

    def _fetch_server_version(self) -> Optional[int]:
        """Ask the monitor server for its device-state version.

        Returns the integer version, or ``None`` when the request failed or
        the reply did not parse -- callers then fall back to reading the file.
        A failure arms a backoff so a dead server is not asked at every loop
        iteration.
        """
        now = time.monotonic()
        if now < self._version_probe_retry_after:
            return None
        try:
            reply = self._monitor_client.send_message(json.dumps({"type": "get_version"}))
            if reply is None:
                raise ValueError("no reply")
            obj = json.loads(reply)
            if not isinstance(obj, dict) or obj.get("status") != "ok":
                raise ValueError(f"bad reply {reply!r}")
            return int(obj["version"])
        except Exception:
            self._version_probe_retry_after = now + T_VERSION_PROBE_BACKOFF
            return None

    def detect_changes(self, verbose: bool = True) -> Tuple[
        List[Tuple[np.int32, float, float]],
        List[Tuple[np.int32, float]],
        List[Tuple[np.int32, np.int32]],
        List[Tuple[np.int32, np.int32]],
        List[Tuple[np.int32, float]]]:
        """
        Detect changes in the configuration file and populate update lists.

        The JSON lives on a network share and this runs at ~10 Hz, so the file
        is only read when the server's device-state version has moved since
        the last read (``get_version``).  The first call always reads the file;
        so does any call for which the version request fails.  ``end()`` of a
        finished experiment regenerates the file without going through the
        server, but the monitor is restarted after every run and its first
        read picks that up.

        Args:
            verbose: If True, print information about detected changes.

        Returns:
            Tuple of all update lists.
        """
        # Version is fetched BEFORE the file read so an update landing in
        # between is seen as "version moved" next time, never missed.
        server_version = self._fetch_server_version()
        if (self.last_config_data is not None
                and server_version is not None
                and server_version == self._last_seen_version):
            self.clear_update_lists()
            return (self.dds_frequency_amplitude_updates, self.dds_vpd_updates,
                    self.dds_sw_state_updates, self.ttl_updates, self.dac_updates)

        current_config = self.load_config_file()
        if current_config is None:
            # Read again next time regardless of what the server says.
            self._last_seen_version = None
            if verbose:
                print("No changes detected (config file could not be loaded).")
            return (self.dds_frequency_amplitude_updates, self.dds_vpd_updates,
                    self.dds_sw_state_updates, self.ttl_updates, self.dac_updates)
        self._last_seen_version = server_version

        self.clear_update_lists()

        # Detect schema changes: new keys in JSON not known to device dicts
        # (happens when _id files are edited and JSON is regenerated)
        unknown_dds = set(current_config.get('dds', {}).keys()) - set(self.dds_dict.keys())
        unknown_dac = set(current_config.get('dac', {}).keys()) - set(self.dac_dict.keys())
        unknown_ttl = set(current_config.get('ttl', {}).keys()) - set(self.ttl_dict.keys())
        if unknown_dds or unknown_dac or unknown_ttl:
            print(f"[Monitor] JSON has new device keys not known to running monitor — "
                  f"DDS: {unknown_dds}, DAC: {unknown_dac}, TTL: {unknown_ttl}. "
                  f"Signaling restart — the restart reconciles the state file "
                  f"against the device frames.")
            self._schema_changed = True
            self.last_config_data = current_config
            return (self.dds_frequency_amplitude_updates, self.dds_vpd_updates,
                    self.dds_sw_state_updates, self.ttl_updates, self.dac_updates)

        if self.last_config_data is None:
            self.last_config_data = current_config
            for dtype in ('dds', 'dac', 'ttl'):
                for name, cfg in current_config.get(dtype, {}).items():
                    self._force_update_counters[(dtype, name)] = cfg.get('force_update_counter', 0)
            if verbose:
                print("No changes detected (initial load.)")
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

            new_counter = new_config.get('force_update_counter', 0)
            last_counter = self._force_update_counters.get(('dds', device_name), 0)
            force_this = new_counter != last_counter
            if force_this:
                self._force_update_counters[('dds', device_name)] = new_counter
                if verbose:
                    print(f"[FORCE_UPDATE] DDS {device_name}: force_update_counter {last_counter} → {new_counter}")

            if force_this or old_config.get('frequency') != new_config.get('frequency') or \
            old_config.get('amplitude') != new_config.get('amplitude'):
                update_index = self.dds_frequency_amplitude_updates.index(DEFAULT_UPDATE_2FLOAT)
                self.dds_frequency_amplitude_updates[update_index] = (
                    kernel_index, new_config['frequency'], new_config['amplitude'])
                changes_detected = True
                if verbose:
                    reason = "[FORCE_UPDATE]" if force_this else "[VALUE_CHANGE]"
                    print(f"DDS {device_name}: {reason} Frequency/Amplitude set to {new_config['frequency']}/{new_config['amplitude']}")

            if force_this or old_config.get('v_pd') != new_config.get('v_pd'):
                update_index = self.dds_vpd_updates.index(DEFAULT_UPDATE_FLOAT)
                self.dds_vpd_updates[update_index] = (kernel_index, new_config['v_pd'])
                changes_detected = True
                if verbose:
                    reason = "[FORCE_UPDATE]" if force_this else "[VALUE_CHANGE]"
                    print(f"DDS {device_name}: {reason} V_PD set to {new_config['v_pd']}")

            if force_this or old_config.get('sw_state') != new_config.get('sw_state'):
                update_index = self.dds_sw_state_updates.index(DEFAULT_UPDATE_INT)
                self.dds_sw_state_updates[update_index] = (kernel_index, new_config['sw_state'])
                changes_detected = True
                if verbose:
                    reason = "[FORCE_UPDATE]" if force_this else "[VALUE_CHANGE]"
                    print(f"DDS {device_name}: {reason} SW State set to {new_config['sw_state']}")

        # Process TTL devices
        old_ttl = self.last_config_data.get('ttl', {})
        new_ttl = current_config.get('ttl', {})
        for device_name, new_config in new_ttl.items():
            if device_name not in self.ttl_dict:
                continue

            new_counter = new_config.get('force_update_counter', 0)
            last_counter = self._force_update_counters.get(('ttl', device_name), 0)
            force_this = new_counter != last_counter
            if force_this:
                self._force_update_counters[('ttl', device_name)] = new_counter
                if verbose:
                    print(f"[FORCE_UPDATE] TTL {device_name}: force_update_counter {last_counter} → {new_counter}")

            if force_this or old_ttl.get(device_name, {}).get('ttl_state') != new_config.get('ttl_state'):
                kernel_index = self.ttl_dict.get(device_name, -1)
                update_index = self.ttl_updates.index(DEFAULT_UPDATE_INT) # get next update from start of list
                self.ttl_updates[update_index] = (kernel_index, new_config['ttl_state'])
                changes_detected = True
                if verbose:
                    reason = "[FORCE_UPDATE]" if force_this else "[VALUE_CHANGE]"
                    print(f"TTL {device_name}: {reason} State set to {new_config['ttl_state']}")

        # Process DAC devices
        old_dac = self.last_config_data.get('dac', {})
        new_dac = current_config.get('dac', {})
        for device_name, new_config in new_dac.items():
            if device_name not in self.dac_dict:
                continue

            new_counter = new_config.get('force_update_counter', 0)
            last_counter = self._force_update_counters.get(('dac', device_name), 0)
            force_this = new_counter != last_counter
            if force_this:
                self._force_update_counters[('dac', device_name)] = new_counter
                if verbose:
                    print(f"[FORCE_UPDATE] DAC {device_name}: force_update_counter {last_counter} → {new_counter}")

            if force_this or abs(old_dac.get(device_name, {}).get('voltage', 0.0) - new_config.get('voltage', 0.0)) > 1e-6:
                kernel_index = self.dac_dict.get(device_name, -1)
                update_index = self.dac_updates.index(DEFAULT_UPDATE_FLOAT)
                self.dac_updates[update_index] = (kernel_index, new_config['voltage'])
                changes_detected = True
                if verbose:
                    reason = "[FORCE_UPDATE]" if force_this else "[VALUE_CHANGE]"
                    print(f"DAC {device_name}: {reason} Voltage set to {new_config['voltage']}")

        self.last_config_data = current_config

        if verbose and not changes_detected:
            print("No changes detected.")

        return (self.dds_frequency_amplitude_updates, self.dds_vpd_updates, 
                self.dds_sw_state_updates, self.ttl_updates, self.dac_updates)

    @kernel
    def sync_change_list(self, verbose=True) -> TBool:
        """
        Synchronize kernel variables with the non-kernel update lists.
        Returns True if a schema change (new device keys) was detected.
        """
        (self.dds_frequency_amplitude_updates, self.dds_vpd_updates, \
          self.dds_sw_state_updates, self.ttl_updates, self.dac_updates) = self.detect_changes(verbose=verbose)
        return self.schema_changed()

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
        
    @kernel
    def monitor_loop(self, verbose=False):
        self.signal_ready()
        while True:
            self.core.wait_until_mu(now_mu())
            keys_changed = self.sync_change_list(verbose=verbose)
            self.core.break_realtime()
            if keys_changed:
                self.signal_end()
                break
            self.apply_updates()
            delay(T_MONITOR_UPDATE_INTERVAL)