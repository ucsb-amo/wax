"""Per-run AD9910 bring-up that skips the chips that are still set up.

``AD9910.init()`` takes ~60-100 ms a channel: an I2C EEPROM read (~40 ms), a fixed
50 ms wait, the PLL bring-up and ``tune_sync_delay``.  All of that state lives in
the DDS chips and survives ``core.reset()``, so on almost every run it is
rewritten with what is already there (measured on a 24-channel crate,
2026-09-18: ~1.4 s a run, down to ~13 ms with every channel skipped).

A channel is left alone when ALL of these hold, and gets the full ``init()``
otherwise:

  * the core device has not rebooted since a full init.  The core-device cache
    is cleared by a reboot, and a reboot restarts SYNC generation;
  * CFR3 reads back the PLL configuration ``init()`` writes.  A power-cycled chip
    has the PLL off and is not yet in the SPI mode that makes reads work, so this
    fails safe;
  * the SYNC receiver is still enabled (for channels that use SYNC);
  * the PLL-lock bit is set; and
  * SMP_ERR stays clear after being reset here -- the same sync sample-and-hold
    check ``tune_sync_delay`` uses to validate a delay.

Skipped channels still get CFR1/CFR2 put back to ``init()``'s defaults, so a
previous experiment cannot leave e.g. RAM mode behind.  Frequencies, amplitudes
and switches are not touched here.

The cache entry doubles as the "initialised since boot" flag and as a copy of
each channel's EEPROM sync data ``[seed, io_update_delay, ...]``: those are fixed
calibration constants, and re-reading them was most of what a skipped channel
used to cost.

Usage (host side, in prepare)::

    self.dds_initializer = AD9910FastInit(core=self.core,
                                          core_cache=self.get_device("core_cache"),
                                          dds_list=self.dds.dds_list,
                                          record_to=self._extra_file_texts)

and in the init kernel, after the Urukul CPLDs are initialised::

    self.dds_initializer.init(force)     # force=True: full init on every channel

Before relying on it for phase-coherent work, compare the relative phase of two
channels after a skipped run and after ``init(True)``.
"""

from artiq.experiment import kernel, rpc, delay, TBool, TInt32
from artiq.coredevice import urukul
from artiq.coredevice.ad9910 import _AD9910_REG_CFR3, _AD9910_REG_SYNC

import json

from waxx.util import console

DEFAULT_CACHE_KEY = "waxx_ad9910_sync_data"

#: Key under which the outcome is written into ``record_to`` (an experiment's
#: ``_extra_file_texts``, saved as an attribute of the run's HDF5 file).
RECORD_KEY = "dds_init"

# CFR3 fields AD9910.init() programs: DRV0, VCO select, charge-pump current,
# REFCLK divider bypass / resetb, PFD reset (must read 0), PLL enable, N.
_CFR3_CHECK_MASK = ((3 << 28) | (7 << 24) | (7 << 19) | (1 << 15) | (1 << 14)
                    | (1 << 10) | (1 << 8) | (0x7f << 1))

# why a channel was not skipped (raw = the register that failed the check)
FAIL_REASONS = {1: "CFR3 mismatch", 2: "SYNC receiver disabled",
                3: "PLL not locked", 4: "SMP_ERR set (sync sample error)"}


class AD9910FastInit:
    """See the module docstring.

    Args:
        core: the ARTIQ core device.
        core_cache: the ``CoreCache`` device ("core_cache" in the device db).
        dds_list: the waxx ``DDS`` wrappers to bring up (``dds_frame.dds_list``).
        cache_key (str): core-device cache key.  Give each crate / dds_list its
            own key if an experiment ever drives more than one.
        record_to (dict or None): where the outcome is kept with the run --
            the experiment's ``_extra_file_texts``; ``record_to["dds_init"]``
            becomes a JSON attribute of the run's HDF5 file.

    After a run, ``report`` (dict or None) and ``failures`` (list of dict) hold
    what happened, and with ``record_to`` both are stored with the run's data:
    whoever chases a DDS phase problem can see, for any run, whether its
    channels were fully initialised or skipped, and why.  On the terminal: one
    line at NORMAL verbosity when any channel got the full init (a slower run),
    the all-skipped line only at VERBOSE (it is the normal case), and a WARNING
    per channel that failed its check, always.  The `art` startup report shows
    the same.  Validated on hardware 2026-09-18 for the all-intact case only (0
    of 24 channels, 13 ms); the reboot and power-cycle paths and the
    phase-coherence comparison below are still owed.
    """

    kernel_invariants = {"core", "core_cache", "dds_list", "cache_key"}

    def __init__(self, core, core_cache, dds_list, cache_key=DEFAULT_CACHE_KEY,
                 record_to=None):
        self.core = core
        self.core_cache = core_cache
        self.dds_list = dds_list
        self.record_to = record_to
        # Scoped to the channel count: a cached list of another length (a channel
        # added or removed) can be neither used nor replaced -- a value read back
        # in a kernel is borrowed until the kernel ends -- so under one fixed key
        # every run would silently pay the full init until the next reboot.
        self.cache_key = f"{cache_key}_{len(dds_list)}"
        self.report = None
        self.failures = []

    @kernel
    def init(self, force=False):
        """Bring up every channel, skipping the intact ones.  Call after the
        Urukul CPLDs are initialised.  ``force``: full init() on every channel."""
        n_ch = len(self.dds_list)
        n_cached = 0
        if not force:
            n_cached = self._restore_sync_data()
        fast = n_ch > 0 and n_cached == 2 * n_ch

        t_start_mu = self.core.get_rtio_counter_mu()
        if fast:
            self.core.break_realtime()
            for dds in self.dds_list:
                dds.dds_device.set_cfr1()
                dds.dds_device.clear_smp_err()   # also rewrites CFR2, pulses IO_UPDATE
                delay(50.e-6)
            delay(1.e-3)    # let SMP_ERR integrate (tune_sync_delay waits 100 us)

        t_checks_mu = self.core.get_rtio_counter_mu()
        n_full = 0
        for dds in self.dds_list:
            intact = fast
            if intact:
                intact = self._is_intact(dds)
            if not intact:
                self.core.break_realtime()
                dds.dds_device.init()
                n_full += 1
            dds._store_io_update_delay()
            delay(10.e-6)

        # A non-empty cache value is borrowed for the rest of the kernel and cannot
        # be replaced, so only write when nothing was read back.
        if n_ch > 0 and (force or n_cached == 0):
            # every channel just ran init(), which read its EEPROM
            sync_values = [0 for _ in range(2 * n_ch)]
            i = 0
            for dds in self.dds_list:
                sync_values[2 * i] = dds.dds_device.sync_data.sync_delay_seed
                sync_values[2 * i + 1] = dds.dds_device.sync_data.io_update_delay
                i += 1
            self.core_cache.put(self.cache_key, sync_values)
        t_end_mu = self.core.get_rtio_counter_mu()
        self._record(n_full, n_ch, n_cached, force,
                     self.core.mu_to_seconds(t_end_mu - t_start_mu),
                     self.core.mu_to_seconds(t_checks_mu - t_start_mu))

    @kernel
    def _restore_sync_data(self) -> TInt32:
        """Copies the cached EEPROM sync data back onto the AD9910 drivers, if the
        cache holds one [seed, io_update_delay] pair per channel.  Returns the
        number of cached values (0: nothing cached since the core device booted)."""
        cached = self.core_cache.get(self.cache_key)
        n_cached = len(cached)
        if n_cached == 2 * len(self.dds_list):
            i = 0
            for dds in self.dds_list:
                dds.dds_device.sync_data.sync_delay_seed = cached[2 * i]
                dds.dds_device.sync_data.io_update_delay = cached[2 * i + 1]
                i += 1
        return n_cached

    @kernel
    def _is_intact(self, dds) -> TBool:
        """True if this AD9910 still holds what AD9910.init() set up.
        Expects clear_smp_err() to have run ~1 ms earlier."""
        dev = dds.dds_device
        if dev.chip_select < 4:
            return False    # not an individually addressed channel; don't guess
        bit = dev.chip_select - 4
        self.core.break_realtime()

        expected = (0x0807c000 | (dev.pll_vco << 24) | (dev.pll_cp << 19)
                    | (dev.pll_en << 8) | (dev.pll_n << 1))
        cfr3 = dev.read32(_AD9910_REG_CFR3)
        delay(100.e-6)
        if (cfr3 & _CFR3_CHECK_MASK) != (expected & _CFR3_CHECK_MASK):
            self._record_failure(dds.urukul_idx, dds.ch, 1, cfr3)
            return False

        uses_sync = dev.sync_data.sync_delay_seed >= 0
        if uses_sync:
            sync = dev.read32(_AD9910_REG_SYNC)
            delay(100.e-6)
            if (sync >> 27) & 1 == 0:       # SYNC receiver enable
                self._record_failure(dds.urukul_idx, dds.ch, 2, sync)
                return False

        sta = dev.cpld.sta_read()
        delay(100.e-6)
        # (the urukul_sta_* helpers leave the int width open; sta is an int32)
        pll_locked = (sta >> (urukul.STA_PLL_LOCK + bit)) & 1
        smp_err = (sta >> (urukul.STA_SMP_ERR + bit)) & 1
        if dev.pll_en == 1 and pll_locked == 0:
            self._record_failure(dds.urukul_idx, dds.ch, 3, sta)
            return False
        if uses_sync and smp_err != 0:
            self._record_failure(dds.urukul_idx, dds.ch, 4, sta)
            return False
        return True

    @rpc(flags={"async"})
    def _record(self, n_full, n_ch, n_cached, forced, t_total, t_check_pass):
        # `why` separates the three ways every channel can end up fully initialised:
        # asked for, nothing cached since the core device booted, or checks failed
        # (then `failures` says which).
        if forced:
            why = "forced"
        elif n_cached == 0:
            why = "no cache: first run since the core device booted"
        elif n_cached != 2 * n_ch:
            why = f"cache has {n_cached} values, expected {2 * n_ch}"
        else:
            why = "cache hit"
        self.report = dict(n_full=n_full, n_channels=n_ch, why=why, t_total_s=t_total,
                           t_check_pass_s=t_check_pass)
        self._store()
        # The all-skipped case is the normal one and worth no terminal line
        # (it is stored with the run); a full init on any channel means a
        # slower run, so say so.
        console.info(f"[dds init] full init on {n_full} of {n_ch} channels, "
                     f"{n_ch - n_full} skipped ({why}), {t_total * 1e3:.0f} ms",
                     level=console.NORMAL if n_full else console.VERBOSE)

    def _store(self):
        """The outcome into ``record_to`` (host side; never raises)."""
        if self.record_to is None:
            return
        try:
            self.record_to[RECORD_KEY] = json.dumps(
                {"report": self.report, "failures": list(self.failures)}, default=repr)
        except Exception as e:
            print(f"[dds init] WARNING: could not store the DDS init outcome with the "
                  f"run: {e!r}")

    @rpc(flags={"async"})
    def _record_failure(self, urukul_idx, ch, reason, raw):
        reason = FAIL_REASONS.get(reason, reason)
        raw = hex(raw & 0xffffffff)
        self.failures.append(dict(urukul=urukul_idx, ch=ch, reason=reason, raw=raw))
        self._store()       # now, not only with the summary: the kernel may not get there
        # On top of the one-line summary from _record: say which channel and why.
        print(f"[dds init] WARNING: urukul {urukul_idx} ch {ch} failed its check "
              f"({reason}, raw {raw}) -- running the full AD9910 init on it.")
