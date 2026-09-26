from collections import deque
from pathlib import Path
from typing import Optional, List, Tuple
import os
import json
import time
import traceback
import numpy as np

from artiq.language.core import kernel, kernel_from_string, delay, now_mu, rpc
from artiq.language import TBool, TInt32
from artiq.coredevice.core import Core
from artiq.coredevice.exceptions import RTIOUnderflow

# from waxx.control.artiq import DDS, DAC_CH, TTL_OUT, TTL_IN

# from waxx.util.artiq.async_print import aprint

from waxx.util.device_state import composite as _composite
from waxx.util.device_state.composite import (
    OP_OK, OP_UNDERFLOW, OP_VALUE_ERROR, OP_RUNTIME_ERROR, OP_EXCEPTION,
    OP_HOST_ERROR, OP_REJECTED,
)

DEFAULT_UPDATE_2FLOAT = (-1, 0.0, 0.0)
DEFAULT_UPDATE_FLOAT = (-1, 0.0)
DEFAULT_UPDATE_BOOL = (-1, False)
DEFAULT_UPDATE_INT = (-1, 0)

T_MONITOR_UPDATE_INTERVAL = 0.1

# After a failed get_version request, do not ask the server again for this
# long (the file is read directly meanwhile).  A dead server would otherwise
# add the client's connect/rediscover timeouts to every loop iteration.
T_VERSION_PROBE_BACKOFF = 5.0

# Composite ops (waxx.util.device_state.composite).  One loop iteration runs
# at most N_OP_SLOTS of them; the rest wait for the next.  Each op starts with
# T_OP_SLACK of timeline slack (the op's first line, see composite.SLACK_PREFIX)
# -- ample for the kernel CPU to compute ramps ahead of their RTIO events, and
# invisible at the speed a person clicks.
N_OP_SLOTS = 16
DEFAULT_OP = (-1, 0) + (0.0,) * _composite.N_OP_ARGS
T_OP_SLACK = 10.e-3

# Timeline slack before a batch of channel updates.  break_realtime leaves
# ~125 us, which a handful of DDS rewrites (init=True) can outrun.
T_UPDATE_SLACK = 2.e-3

# What one poll found, as bits of the int sync_change_list returns.  The
# channel lists and op slots only cross the link when there is something in
# them; an idle iteration moves one int.
FLAG_CHANNELS = 1
FLAG_OPS = 2
FLAG_SCHEMA = 4

# Ops whose events are still queued this far ahead when they finish make the
# monitor tell the server it is busy (so queued requests do not expire while
# it waits for them to play out) before it reports them done.
T_BUSY_NOTICE = 0.5

# Retry cadence for a channel write-back the server did not accept, and for
# re-registering the op table with a server that has lost it.
T_WRITEBACK_RETRY = 2.0
T_REREGISTER_RETRY = 10.0

# Write-back tolerances: the kernel caches values quantized by the hardware
# (DDS frequency to its FTW, amplitude to its ASF), so an unchanged channel
# can differ from the JSON in the last digits.  Anything inside these is "the
# same"; a real change is always far outside them.
_TOL_FREQUENCY = 1.0        # Hz   (FTW step ~0.23 Hz)
_TOL_AMPLITUDE = 2.e-4      # ASF step ~6e-5
_TOL_VOLTAGE = 1.e-4        # V

from waxx.util.comms_server.comm_client import MonitorClient
from waxx.util.device_state.generate_state_file import Generator


class _PendingOp:
    """A composite op taken from the server, on its way through the kernel."""

    __slots__ = ("seq", "entry", "args", "payload", "packed", "message")

    def __init__(self, seq, entry, args, payload, packed):
        self.seq = seq
        self.entry = entry
        self.args = args
        self.payload = payload
        self.packed = packed
        self.message = ""


def _differs(old, new, tol) -> bool:
    try:
        return abs(float(old) - float(new)) > tol
    except (TypeError, ValueError):
        return True


def _jsonable(value):
    """numpy scalars/arrays -> plain Python, recursively."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

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

        # Composite ops.  Off unless the monitor experiment calls
        # init_composites(); every other experiment builds a Monitor too
        # (Clients), and none of this is compiled unless monitor_loop is.
        self.t_op_slack = T_OP_SLACK
        self._composites_enabled = False
        self._ops_supported = True
        self._op_table = None
        self._op_pending = deque()      # taken from the server, not yet in the kernel
        self._op_running = {}           # seq -> _PendingOp, handed to the kernel
        self.op_kernels = []
        self._op_host_before = []
        self._op_host_after = []
        self.op_updates = [DEFAULT_OP] * N_OP_SLOTS
        self._op_seq_buf = np.zeros(N_OP_SLOTS, dtype=np.int32)
        self._op_status_buf = np.zeros(N_OP_SLOTS, dtype=np.int32)

        # Channel write-back the server has not accepted yet: (type, name) ->
        # changes.  NOT folded into last_config_data until it is accepted --
        # folding first would make the next file read "see" the old values as
        # a change and put them back on the hardware.
        self._writeback_pending: dict = {}
        self._writeback_retry_after = 0.0
        self._writeback_warned = False
        self._reregister_after = 0.0

        # The run this process announced (token, run_id), until its end state
        # is accepted; withdrawn at exit otherwise (announce_run).
        self._announced = None
        self._withdraw_registered = False

    def update_device_states(self, run_id=None, expt=""):
        """Send the frames' current state (an experiment's end state) to the
        monitor server, which writes it atomically, bumps the version, marks
        the state trusted and tells every GUI to resync.  The server is the
        only writer of the state file; if it cannot be reached the file is
        left as it was and this says so loudly -- the Device Control GUIs will
        show the pre-run state until the next run ends normally."""
        try:
            self.generator.generate_device_config()
            config = self.generator.config_data
            payload = {k: config[k] for k in ('dds', 'ttl', 'dac')}
        except Exception as e:
            print(f"[Monitor] WARNING: could not collect the end-of-run device state: {e!r}")
            return False
        reply = self._monitor_client.replace_state(_jsonable(payload), run_id=run_id, expt=expt)
        if reply is None or reply.get("status") != "ok":
            print("[Monitor] WARNING: the monitor server did not accept this run's end state "
                  f"({None if reply is None else reply.get('msg')}); the device state file "
                  "still describes the hardware as it was BEFORE this run.")
            return False
        # The end state lifted the fence; nothing to withdraw at exit.
        self._announced = None
        return True

    def announce_run(self, run_id=None, expt=""):
        """Tell the monitor server an experiment is about to take the core, so
        composite ops are refused from now until it ends.

        A run that dies before its kernel takes the core (a compile error, an
        exception in prepare or host code) sends nothing the server would
        notice -- the monitor is never interrupted and ``end()`` never runs.
        So the announcement is named by a token, and an exit handler sends
        ``run_withdrawn`` with it unless this run's end state was accepted.
        The server lifts the fence only if the token is still the current
        one, so the message is harmless after the run took the core (that
        already lifted it) or once a newer run has announced itself.  A hard
        kill runs no exit handler; the server's own timeout covers that."""
        import atexit  # noqa: PLC0415
        import uuid  # noqa: PLC0415
        try:
            import socket  # noqa: PLC0415
            host = socket.gethostname()
        except Exception:
            host = ""
        token = uuid.uuid4().hex
        reply = self._monitor_client.announce_run(run_id=run_id, expt=expt, client=host,
                                                  token=token)
        if reply is None or reply.get("status") != "ok":
            print("[Monitor] note: could not tell the monitor server this run is starting "
                  "(server unreachable); composite ops are not fenced for it.")
            return
        self._announced = (token, run_id)
        if not self._withdraw_registered:
            atexit.register(self.withdraw_run_at_exit)
            self._withdraw_registered = True

    def withdraw_run_at_exit(self):
        """Exit handler (see :meth:`announce_run`).  Never raises."""
        announced, self._announced = self._announced, None
        if announced is None:
            return
        token, run_id = announced
        try:
            self._monitor_client.withdraw_run(token, run_id)
        except Exception:
            pass

    def server_state(self):
        """The monitor server's ``get_state`` reply, or None."""
        reply = self._monitor_client.get_state()
        if reply is None or reply.get("status") != "ok":
            return None
        return reply

    def journal_since_last_run(self):
        """Journal records since the last end-of-run state, or None."""
        reply = self._monitor_client.get_journal(since="run_end")
        if reply is None or reply.get("status") != "ok":
            return None
        return reply.get("entries") or []

    def signal_end(self):
        self._monitor_client.send_end()

    def signal_ready(self):
        if self._composites_enabled:
            self._register_ops()
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
        # Ping-only op table + channel snapshot kernel: cheap, and it keeps
        # monitor_loop compilable when no composite devices are loaded.
        self._build_op_kernels()
        self._build_snapshot_kernel()

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
        # One slot per channel plus the -1 terminator (each channel appears at
        # most once per list).  Before build_device_lookup the counts are
        # unknown and the lists are 1 long -- never empty, since the compiler
        # takes their element type from these host values.
        n_dds = len(getattr(self, 'dds_dict', ())) + 1
        n_ttl = len(getattr(self, 'ttl_dict', ())) + 1
        n_dac = len(getattr(self, 'dac_dict', ())) + 1
        self.dds_frequency_amplitude_updates = [DEFAULT_UPDATE_2FLOAT] * n_dds
        self.dds_vpd_updates = [DEFAULT_UPDATE_FLOAT] * n_dds
        self.dds_sw_state_updates = [DEFAULT_UPDATE_INT] * n_dds
        self.ttl_updates = [DEFAULT_UPDATE_INT] * n_ttl
        self.dac_updates = [DEFAULT_UPDATE_FLOAT] * n_dac

    def _have_channel_updates(self) -> bool:
        return any(lst[0][0] != -1 for lst in (
            self.dds_frequency_amplitude_updates, self.dds_vpd_updates,
            self.dds_sw_state_updates, self.ttl_updates, self.dac_updates))
    
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

        self._channel_names = {
            0: sorted(self.dds_dict, key=self.dds_dict.get),
            1: sorted(self.ttl_dict, key=self.ttl_dict.get),
            2: sorted(self.dac_dict, key=self.dac_dict.get),
        }
        self.clear_update_lists()

    # ------------------------------------------------------------------
    # Composite ops (waxx.util.device_state.composite)
    # ------------------------------------------------------------------

    def init_composites(self, devices, sync_cache=True):
        """Monitor experiment only, after finish_prepare(): compile one kernel
        per composite op, and seed the kernel's channel cache from the device
        state file.

        The seeding is what makes the write-back after an op truthful.  The
        monitor starts with the frames at their _id-file defaults, but the
        hardware is wherever the last experiment left it -- which is what the
        state file records.  An op that switches one channel would otherwise
        report every *other* channel at its default.  It also means the
        per-channel kernels (which rewrite frequency and amplitude together
        with init=True) rewrite the file's values rather than the defaults.
        """
        self._op_table = _composite.OpTable(devices)
        self._build_op_kernels()
        self._composites_enabled = True
        n_ops = len(self._op_table) - 1
        print(f"[Monitor] composite ops: {n_ops} op(s) on "
              f"{len(self._op_table.devices)} device(s), definitions {self._op_table.hash}.")
        if sync_cache:
            self.sync_cache_from_state_file()

    @property
    def composites_enabled(self) -> bool:
        return self._composites_enabled

    def disable_composites(self, reason=""):
        """Fall back to channel-only monitoring (ping-only op table, no
        registration, no polling for ops).  The monitor experiment does this
        when the composite definitions are rejected or do not compile, so a
        broken definition costs the Composite tab, never the monitor."""
        self._composites_enabled = False
        self._op_table = None
        self._op_pending.clear()
        self._op_running.clear()
        self._build_op_kernels()
        print(f"[Monitor] composite ops DISABLED for this monitor session"
              f"{': ' + reason if reason else ''}. The Composite tab will refuse ops; "
              f"channel control is unaffected.")

    def _build_op_kernels(self):
        table = self._op_table if self._op_table is not None else _composite.OpTable(())
        self.op_kernels = [kernel_from_string(list(_composite.KERNEL_PARAMS), e.body)
                           for e in table]
        self._op_host_before = [e.has_host and not e.host_after for e in table]
        self._op_host_after = [e.host_after for e in table]

    def _build_snapshot_kernel(self):
        """A kernel copying every channel's cached state into flat arrays, for
        the write-back after composite ops.  Kernel-side attribute changes are
        invisible to the host until a kernel returns, and the monitor kernel
        never returns -- so the state has to be handed over explicitly."""
        self._snap_dds_keys = sorted(self.dds_dict, key=self.dds_dict.get)
        self._snap_dac_keys = sorted(self.dac_dict, key=self.dac_dict.get)
        self._snap_ttl_keys = sorted(self.ttl_dict, key=self.ttl_dict.get)
        lines = []
        for i, name in enumerate(self._snap_dds_keys):
            lines += [f"f[{i}] = expt.dds.{name}.frequency",
                      f"a[{i}] = expt.dds.{name}.amplitude",
                      f"v[{i}] = expt.dds.{name}.v_pd",
                      f"s[{i}] = expt.dds.{name}.sw_state"]
        lines += [f"dv[{i}] = expt.dac.{name}.v" for i, name in enumerate(self._snap_dac_keys)]
        lines += [f"ts[{i}] = expt.ttl.{name}.state" for i, name in enumerate(self._snap_ttl_keys)]
        # One-element list: called like the other generated kernels.
        self._snapshot_kernels = [kernel_from_string(
            ["expt", "f", "a", "v", "s", "dv", "ts"], "\n".join(lines) or "pass")]
        n_dds = max(len(self._snap_dds_keys), 1)
        self._snap_dds_f = np.zeros(n_dds)
        self._snap_dds_a = np.zeros(n_dds)
        self._snap_dds_v = np.zeros(n_dds)
        self._snap_dds_sw = np.zeros(n_dds, dtype=np.int32)
        self._snap_dac_v = np.zeros(max(len(self._snap_dac_keys), 1))
        self._snap_ttl_s = np.zeros(max(len(self._snap_ttl_keys), 1), dtype=np.int32)

    def sync_cache_from_state_file(self):
        """Host side, before compile: set the frames' cached values (DDS
        frequency/amplitude/v_pd/switch, DAC voltage, TTL level) to the state
        file's, then run each composite device's own ``sync_cache`` hook.
        Nothing is written to hardware."""
        cfg = self.load_config_file()
        if not cfg:
            print("[Monitor] WARNING: could not read the device state file; the "
                  "kernel channel cache keeps the frame defaults, so the channel "
                  "write-back after a composite op may report untouched channels "
                  "at their defaults.")
            return
        counts = {"dds": 0, "dac": 0, "ttl": 0}
        failed = []
        for name, c in (cfg.get('dds') or {}).items():
            if name not in self.dds_dict:
                continue
            try:
                dds = getattr(self.dds, name)
                dev = dds.dds_device
                ftw = int(dev.frequency_to_ftw(float(c.get('frequency', dds.frequency))))
                asf = int(dev.amplitude_to_asf(float(c.get('amplitude', dds.amplitude))))
                dds._ftw = ftw
                dds.frequency = float(dev.ftw_to_frequency(ftw))
                dds._asf = asf
                dds.amplitude = float(dev.asf_to_amplitude(asf))
                dds.v_pd = float(c.get('v_pd', dds.v_pd))
                dds.sw_state = int(c.get('sw_state', dds.sw_state))
                counts["dds"] += 1
            except Exception as e:
                failed.append(f"dds.{name} ({e!r})")
        for name, c in (cfg.get('dac') or {}).items():
            if name not in self.dac_dict:
                continue
            try:
                getattr(self.dac, name).v = float(c.get('voltage', 0.0))
                counts["dac"] += 1
            except Exception as e:
                failed.append(f"dac.{name} ({e!r})")
        for name, c in (cfg.get('ttl') or {}).items():
            if name not in self.ttl_dict:
                continue
            try:
                getattr(self.ttl, name).state = int(c.get('ttl_state', 0))
                counts["ttl"] += 1
            except Exception as e:
                failed.append(f"ttl.{name} ({e!r})")
        for device in (self._op_table.devices if self._op_table is not None else ()):
            if device.sync_cache is None:
                continue
            try:
                device.sync_cache(self.expt, cfg)
            except Exception as e:
                failed.append(f"{device.key}.sync_cache ({e!r})")
        print(f"[Monitor] kernel channel cache seeded from the device state file "
              f"({counts['dds']} DDS, {counts['dac']} DAC, {counts['ttl']} TTL).")
        if failed:
            print(f"[Monitor] WARNING: could not seed {failed}; those keep their "
                  f"frame defaults.")

    def _register_ops(self):
        reg = self._op_table.registration(session=f"{os.getpid()}@{time.time():.0f}")
        reply = self._monitor_client.register_ops(reg)
        if reply is None or reply.get("status") != "ok":
            msg = None if reply is None else reply.get("msg")
            if msg and "unknown type" in msg:
                self._ops_supported = False
                print("[Monitor] WARNING: the monitor server does not know composite "
                      "ops (it runs older code) -- composite controls are off until it "
                      "is restarted.")
            else:
                print(f"[Monitor] WARNING: composite op registration failed ({msg or 'no reply'}); "
                      f"the GUI's Composite tab will be refused until the monitor restarts.")

    def _accept_polled_ops(self, ops):
        """Queue ops from a poll reply.  Each is re-checked here against the
        compiled table and the hard limits -- the server only knows names and
        signatures -- and refused ones are reported at once."""
        rejected = []
        for op in ops or []:
            try:
                seq = int(op["seq"])
                index = int(op["index"])
            except (KeyError, TypeError, ValueError):
                continue
            name = op.get("op", "?")
            entry = self._op_table.by_index(index) if self._op_table is not None else None
            if entry is None or entry.name != name or entry.signature != op.get("sig"):
                rejected.append({"seq": seq, "op": name, "status": OP_REJECTED,
                                 "message": "not in this monitor's compiled op table"})
                continue
            args = op.get("arg_values") or {}
            payload = op.get("payload") or {}
            problems = entry.hard_problems(args, payload)
            if problems:
                rejected.append({"seq": seq, "op": name, "status": OP_REJECTED,
                                 "message": "; ".join(problems)})
                continue
            self._op_pending.append(_PendingOp(seq, entry, dict(args), dict(payload),
                                               entry.pack(args)))
        if rejected:
            for r in rejected:
                print(f"[Monitor] composite op {r['op']} #{r['seq']} refused: {r['message']}")
            self._send_op_results(rejected)

    def fetch_ops(self) -> List[Tuple[np.int32, np.int32, float, float, float, float,
                                      float, float, float, float]]:
        """RPC: the next ops for the kernel, as (index, seq, a0..a7) slots.
        Never raises: an exception here would surface in the kernel and end
        the monitor loop."""
        slots = [DEFAULT_OP] * N_OP_SLOTS
        try:
            i = 0
            while self._op_pending and i < N_OP_SLOTS:
                p = self._op_pending.popleft()
                self._op_running[p.seq] = p
                slots[i] = (int(p.entry.index), int(p.seq)) + tuple(float(a) for a in p.packed)
                i += 1
        except Exception:
            print("[Monitor] WARNING: could not hand composite ops to the kernel:")
            traceback.print_exc()
        return slots

    def run_op_host_step(self, seq) -> TInt32:
        """RPC: the host part of an op (e.g. talking to the tweezer AWG).
        Failures come back as a status, never as an exception."""
        p = self._op_running.get(int(seq))
        if p is None:
            print(f"[Monitor] composite op #{seq}: no record of it on the host; "
                  f"its host step did not run.")
            return OP_HOST_ERROR
        if not p.entry.has_host:
            return OP_OK
        try:
            p.entry.op.host(self.expt, dict(p.args), dict(p.payload))
        except Exception as e:
            print(f"[Monitor] composite op {p.entry.name} #{p.seq}: host step failed:")
            traceback.print_exc()
            p.message = f"{type(e).__name__}: {e}"
            return OP_HOST_ERROR
        return OP_OK

    @rpc(flags={"async"})
    def report_ops(self, n, seqs, statuses, dds_f, dds_a, dds_v, dds_sw, dac_v, ttl_s):
        """RPC after an iteration that ran ops: write back the channels they
        changed, then report the outcomes (in that order, so a GUI that sees
        "done" already has the new channel states).  Async, and it must never
        raise: an async RPC's exception resurfaces in the kernel."""
        try:
            self._report_ops(n, seqs, statuses, dds_f, dds_a, dds_v, dds_sw, dac_v, ttl_s)
        except Exception:
            print("[Monitor] WARNING: reporting composite ops failed:")
            traceback.print_exc()

    def _report_ops(self, n, seqs, statuses, dds_f, dds_a, dds_v, dds_sw, dac_v, ttl_s):
        try:
            self._write_back_channels(dds_f, dds_a, dds_v, dds_sw, dac_v, ttl_s)
        except Exception:
            print("[Monitor] WARNING: channel write-back after composite ops failed:")
            traceback.print_exc()
        results = []
        for i in range(int(n)):
            seq, status = int(seqs[i]), int(statuses[i])
            p = self._op_running.pop(seq, None)
            result = {"seq": seq, "status": status,
                      "op": p.entry.name if p else "?",
                      "message": p.message if p else ""}
            device = p.entry.device if p else None
            if device is not None and device.host_state is not None:
                try:
                    result["state"] = dict(device.host_state(self.expt))
                    result["device"] = device.key
                except Exception as e:
                    print(f"[Monitor] {device.key}.host_state failed: {e!r}")
            print(f"[Monitor] composite op {result['op']} #{seq}: "
                  f"{_composite.status_text(status, result['message']) if status != OP_OK else 'done'}")
            results.append(result)
        self._send_op_results(results)

    def _send_op_results(self, results):
        reply = self._monitor_client.report_ops(results)
        if reply is None or reply.get("status") != "ok":
            print(f"[Monitor] WARNING: could not report {len(results)} composite op "
                  f"result(s) to the server; the GUI will show them as unknown.")

    def _write_back_channels(self, dds_f, dds_a, dds_v, dds_sw, dac_v, ttl_s):
        """Send the server every channel whose cached state no longer matches
        the state file.  The snapshot is the whole truth for every channel, so
        this set replaces whatever was still pending from an earlier failed
        write-back.  Values are folded into last_config_data only once the
        server has accepted them (see _flush_writeback)."""
        cfg = self.last_config_data
        if cfg is None:
            return
        pending = {}
        for dtype, name, changes in self._snapshot_differences(
                cfg, dds_f, dds_a, dds_v, dds_sw, dac_v, ttl_s):
            pending[(dtype, name)] = changes
        self._writeback_pending = pending
        self._writeback_retry_after = 0.0
        self._flush_writeback()

    def _flush_writeback(self):
        """Send the pending write-back; on acceptance fold it into
        last_config_data (with the DAC mirror of a linked v_pd, which the
        server writes too) so the next file read does not apply it again."""
        if not self._writeback_pending:
            return
        now = time.monotonic()
        if now < self._writeback_retry_after:
            return
        updates = [(t, n, dict(c)) for (t, n), c in self._writeback_pending.items()]
        reply = self._monitor_client.send_update_batch(updates, origin="monitor write-back")
        if reply is None or reply.get("status") != "ok":
            self._writeback_retry_after = now + T_WRITEBACK_RETRY
            if not self._writeback_warned:
                names = ", ".join(f"{t}.{n}" for t, n, _ in updates)
                print(f"[Monitor] WARNING: composite-op channel write-back was not accepted "
                      f"({None if reply is None else reply.get('msg')}); retrying every "
                      f"{T_WRITEBACK_RETRY:.0f} s. Until it lands the state file and the "
                      f"Device Control tabs show the old values for: {names}")
                self._writeback_warned = True
            return
        if self._writeback_warned:
            print("[Monitor] composite-op channel write-back accepted on retry.")
        self._writeback_warned = False
        self._writeback_pending = {}
        cfg = self.last_config_data
        if cfg is None:
            return
        for dtype, name, changes in updates:
            if name in cfg.get(dtype, {}):
                cfg[dtype][name].update(changes)
        for dtype, name, changes in updates:
            if dtype == 'dds' and 'v_pd' in changes:
                dac_key = cfg['dds'].get(name, {}).get('dac_ch_key', "")
                if dac_key and dac_key in cfg.get('dac', {}):
                    cfg['dac'][dac_key]['voltage'] = changes['v_pd']
        # Nothing else landed between our last read and this batch: the file
        # is now exactly last_config_data, so skip the re-read.
        try:
            if (self._last_seen_version is not None
                    and int(reply.get("first_version", -1)) == self._last_seen_version):
                self._last_seen_version = int(reply["version"])
        except (KeyError, TypeError, ValueError):
            pass

    def _drop_superseded_writeback(self, current_config):
        """A channel pending write-back that someone changed in the file since
        (a GUI edit, a force update): their value is about to be applied to
        the hardware, so the kernel's older value must not be written over it."""
        if not self._writeback_pending or self.last_config_data is None:
            return
        for key in list(self._writeback_pending):
            dtype, name = key
            old = (self.last_config_data.get(dtype) or {}).get(name)
            new = (current_config.get(dtype) or {}).get(name)
            if old != new:
                print(f"[Monitor] pending write-back of {dtype}.{name} dropped: it was "
                      f"changed in the state file meanwhile, and that value wins.")
                del self._writeback_pending[key]

    def _snapshot_differences(self, cfg, dds_f, dds_a, dds_v, dds_sw, dac_v, ttl_s):
        updates = []
        section = cfg.get('dds', {})
        for i, name in enumerate(self._snap_dds_keys):
            old = section.get(name)
            if old is None:
                continue
            changes = {}
            if _differs(old.get('frequency'), dds_f[i], _TOL_FREQUENCY):
                changes['frequency'] = float(dds_f[i])
            if _differs(old.get('amplitude'), dds_a[i], _TOL_AMPLITUDE):
                changes['amplitude'] = float(dds_a[i])
            if _differs(old.get('v_pd'), dds_v[i], _TOL_VOLTAGE):
                changes['v_pd'] = float(dds_v[i])
            if int(old.get('sw_state', 0)) != int(dds_sw[i]):
                changes['sw_state'] = int(dds_sw[i])
            if changes:
                updates.append(('dds', name, changes))
        section = cfg.get('dac', {})
        for i, name in enumerate(self._snap_dac_keys):
            old = section.get(name)
            if old is not None and _differs(old.get('voltage'), dac_v[i], _TOL_VOLTAGE):
                updates.append(('dac', name, {'voltage': float(dac_v[i])}))
        section = cfg.get('ttl', {})
        for i, name in enumerate(self._snap_ttl_keys):
            old = section.get(name)
            if old is not None and int(old.get('ttl_state', 0)) != int(ttl_s[i]):
                updates.append(('ttl', name, {'ttl_state': int(ttl_s[i])}))
        return updates

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
        if self._composites_enabled and self._ops_supported:
            # One round trip for both: the version and any queued ops.
            try:
                obj = self._monitor_client.poll()
                if obj is None:
                    raise ValueError("no reply")
                if obj.get("status") != "ok":
                    if "unknown type" in str(obj.get("msg", "")):
                        self._ops_supported = False
                        print("[Monitor] WARNING: the monitor server does not know "
                              "'poll' (older code) -- composite ops are off.")
                    raise ValueError(f"bad reply {obj!r}")
                version = int(obj["version"])
            except Exception:
                if not self._ops_supported:
                    return self._fetch_server_version()
                self._version_probe_retry_after = now + T_VERSION_PROBE_BACKOFF
                return None
            # The server has handed these over; losing them to an exception
            # here would leave the GUI waiting, so report instead of raising.
            try:
                self._accept_polled_ops(obj.get("ops"))
            except Exception:
                print("[Monitor] WARNING: could not queue polled composite ops:")
                traceback.print_exc()
            # A server restarted under a running monitor has no op table.
            if obj.get("registered") is False and now >= self._reregister_after:
                self._reregister_after = now + T_REREGISTER_RETRY
                print("[Monitor] the monitor server has no composite op registration "
                      "(restarted?) -- registering again.")
                self._register_ops()
            return version
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
        finished experiment sends its end state through the server
        (``replace_state``), and the monitor is restarted after every run, so
        its first read picks that up.

        Args:
            verbose: If True, print information about detected changes.

        Returns:
            Tuple of all update lists.
        """
        # Nothing from an earlier call may be applied again, whatever happens
        # below (a failed read used to return the previous call's lists).
        self.clear_update_lists()

        # Version is fetched BEFORE the file read so an update landing in
        # between is seen as "version moved" next time, never missed.
        server_version = self._fetch_server_version()
        if (self.last_config_data is not None
                and server_version is not None
                and server_version == self._last_seen_version):
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

        self._drop_superseded_writeback(current_config)

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

    def poll_changes(self, verbose: bool = False) -> TInt32:
        """RPC, once per loop: retry a pending write-back, look for channel
        changes and ops, and say what was found as FLAG_* bits.  The lists
        themselves are fetched only when there is something in them."""
        self._flush_writeback()
        self.detect_changes(verbose=verbose)
        flags = 0
        if self._have_channel_updates():
            flags |= FLAG_CHANNELS
        if self._composites_enabled and self._op_pending:
            flags |= FLAG_OPS
        if self.schema_changed():
            flags |= FLAG_SCHEMA
        return flags

    def channel_updates(self) -> Tuple[
        List[Tuple[np.int32, float, float]],
        List[Tuple[np.int32, float]],
        List[Tuple[np.int32, np.int32]],
        List[Tuple[np.int32, np.int32]],
        List[Tuple[np.int32, float]]]:
        """RPC: the update lists detect_changes filled this iteration."""
        return (self.dds_frequency_amplitude_updates, self.dds_vpd_updates,
                self.dds_sw_state_updates, self.ttl_updates, self.dac_updates)

    @rpc(flags={"async"})
    def channel_update_failed(self, kind, index):
        """A channel write raised on the core (kind 0 DDS freq/amp, 1 DDS v_pd,
        2 DDS switch, 3 TTL, 4 DAC).  The state file already holds the new
        value; the hardware may not."""
        try:
            what = {0: ("dds", "frequency/amplitude"), 1: ("dds", "v_pd"),
                    2: ("dds", "switch"), 3: ("ttl", "state"), 4: ("dac", "voltage")}[int(kind)]
            names = self._channel_names[{"dds": 0, "ttl": 1, "dac": 2}[what[0]]]
            name = names[int(index)] if 0 <= int(index) < len(names) else f"#{index}"
            print(f"[Monitor] WARNING: setting {what[0]}.{name} {what[1]} raised on the core "
                  f"(underflow or bad value). The state file shows the requested value; "
                  f"the hardware may not have it -- use Force update on that channel.")
        except Exception:
            print(f"[Monitor] WARNING: a channel update (kind {kind}, index {index}) "
                  f"raised on the core.")

    @rpc(flags={"async"})
    def notify_busy(self, seconds):
        """The ops just run keep the timeline busy for ``seconds``: tell the
        server, so ops queued meanwhile are not expired as never taken."""
        try:
            self._monitor_client.request({"type": "busy", "seconds": float(seconds)})
        except Exception:
            pass

    @kernel
    def sync_change_list(self, verbose=True) -> TInt32:
        """
        Synchronize kernel variables with the non-kernel update lists.
        Returns the FLAG_* bits of what was found.
        """
        flags = self.poll_changes(verbose)
        if flags & FLAG_CHANNELS:
            (self.dds_frequency_amplitude_updates, self.dds_vpd_updates, \
              self.dds_sw_state_updates, self.ttl_updates, self.dac_updates) = self.channel_updates()
        if flags & FLAG_OPS:
            self.op_updates = self.fetch_ops()
        return flags

    @kernel
    def apply_updates(self):
        """
        Apply the detected updates to the hardware devices.  A write that
        raises (an underflow, a value the channel refuses) is reported and
        skipped; the loop -- and every other channel -- carries on.
        """
        index = -1
        f = 0.
        a = 0.
        v_pd = 0.
        sw_state = 0
        ttl_state = 0
        v = 0.
        t0 = 8.e-9

        delay(T_UPDATE_SLACK)

        for i in range(len(self.dds_frequency_amplitude_updates)):
            index, f, a = self.dds_frequency_amplitude_updates[i]
            if index == -1:
                break
            try:
                self.dds_frequency_amplitude_kernels[index](self.expt,f, a)
            except:
                self.core.break_realtime()
                self.channel_update_failed(0, index)
            delay(t0)

        for i in range(len(self.dds_vpd_updates)):
            index, v_pd = self.dds_vpd_updates[i]
            if index == -1:
                break
            try:
                self.dds_vpd_kernels[index](self.expt,v_pd)
            except:
                self.core.break_realtime()
                self.channel_update_failed(1, index)
            delay(t0)

        for i in range(len(self.dds_sw_state_updates)):
            index, sw_state = self.dds_sw_state_updates[i]
            if index == -1:
                break
            try:
                self.dds_sw_state_kernels[index](self.expt,sw_state)
            except:
                self.core.break_realtime()
                self.channel_update_failed(2, index)
            delay(t0)

        for i in range(len(self.ttl_updates)):
            index, ttl_state = self.ttl_updates[i]
            if index == -1:
                break
            try:
                self.ttl_kernels[index](self.expt,ttl_state)
            except:
                self.core.break_realtime()
                self.channel_update_failed(3, index)
            delay(t0)

        for i in range(len(self.dac_updates)):
            index, v = self.dac_updates[i]
            if index == -1:
                break
            try:
                self.dac_kernels[index](self.expt,v)
            except:
                self.core.break_realtime()
                self.channel_update_failed(4, index)
            delay(t0)

    @kernel
    def apply_ops(self):
        """Run the composite ops fetched this iteration, in request order.

        A failing op is caught and reported rather than ending the monitor
        (which would leave the hardware in whatever state the op reached, with
        nothing holding it).  The handlers bind no names: the compiler gives a
        local one type, so they could not share one (see the scan loop).  After
        the ops, one snapshot of every channel goes to the host, which writes
        the changed ones back to the device-state file before reporting.
        """
        n = 0
        for i in range(len(self.op_updates)):
            index, seq, a0, a1, a2, a3, a4, a5, a6, a7 = self.op_updates[i]
            if index < 0:
                break
            status = OP_OK
            if self._op_host_before[index]:
                self.core.wait_until_mu(now_mu())
                status = self.run_op_host_step(seq)
                self.core.break_realtime()
            if status == OP_OK:
                try:
                    self.op_kernels[index](self.expt, a0, a1, a2, a3, a4, a5, a6, a7)
                except RTIOUnderflow:
                    status = OP_UNDERFLOW
                except ValueError:
                    status = OP_VALUE_ERROR
                except RuntimeError:
                    status = OP_RUNTIME_ERROR
                except:
                    status = OP_EXCEPTION
                if status != OP_OK:
                    self.core.break_realtime()
            if status == OP_OK and self._op_host_after[index]:
                # after the op's own events have played out
                self.core.wait_until_mu(now_mu())
                status = self.run_op_host_step(seq)
                self.core.break_realtime()
            self._op_seq_buf[n] = seq
            self._op_status_buf[n] = status
            n += 1
        if n > 0:
            # "done" means played out: a ramp still queued on the timeline has
            # not happened yet.  Long waits are announced first.
            remaining = self.core.mu_to_seconds(now_mu() - self.core.get_rtio_counter_mu())
            if remaining > T_BUSY_NOTICE:
                self.notify_busy(remaining)
            self.core.wait_until_mu(now_mu())
            self._snapshot_kernels[0](self.expt, self._snap_dds_f, self._snap_dds_a,
                                      self._snap_dds_v, self._snap_dds_sw,
                                      self._snap_dac_v, self._snap_ttl_s)
            self.report_ops(n, self._op_seq_buf, self._op_status_buf,
                            self._snap_dds_f, self._snap_dds_a, self._snap_dds_v,
                            self._snap_dds_sw, self._snap_dac_v, self._snap_ttl_s)

    @kernel
    def monitor_loop(self, verbose=False):
        self.signal_ready()
        while True:
            self.core.wait_until_mu(now_mu())
            flags = self.sync_change_list(verbose=verbose)
            self.core.break_realtime()
            if flags & FLAG_SCHEMA:
                self.signal_end()
                break
            if flags & FLAG_CHANNELS:
                self.apply_updates()
            if flags & FLAG_OPS:
                self.apply_ops()
            delay(T_MONITOR_UPDATE_INTERVAL)