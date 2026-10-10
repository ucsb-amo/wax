"""An interleaved sub-scan: points taken between the main scan's shots.

A run is its main scan plus, optionally, one sub-scan of points the scan loop
takes between data shots on a schedule (every ``every_n_shots`` data shots,
and once before the first). The first use is a readout calibration: a pair of
"all up" / "all down" APD reads every 20 shots of a feedback run, so the
readout constants can be tracked without separate calibration runs.

The sub-scan is a scan in its own right:

* its own xvars (``xvar(key, values)``): *zipped* point lists of equal length,
  one point per index, repeated once per schedule event (the order reversed on
  every other event when ``alternate``), never shuffled -- acquisition order is
  declaration order. The first key declared is the sub-scan's one scan axis
  (``ad.cal.xvars[0]``); every other zipped key is recorded as a per-point
  container of the same name (``ad.cal.data.<key>``), so the sub-scan loads as
  an ordinary one-axis scan;
* its own containers (``add_data_container``) in its own ``DataVault`` whose
  slot is the sub-scan's counter, not the main scan's (data_vault.py
  ``_put_shot_data_to_run_data``), plus three host containers: ``time`` (when
  the point finished), ``main_index`` (flat acquisition index and flat final
  slot of the data shot taken before it, -1 before the first) and ``status``
  (0 not taken, 1 taken, 2 failed);
* its kernel body is the experiment's ``calibration_kernel`` (waxx Scanner),
  which reads the point's values as ``self.p.<key>`` -- the scan loop sets
  them on the host and ships every parameter to the kernel before the point,
  as it does before a data shot -- and ends with ``self.<name>.put_shot_data()``.

On disk the containers are ordinary ``data/<name>_<key>`` datasets (the
prefix is the sub-scan's name), written point by point during the run and
declared ``final_order`` at END_RUN so the saver never unshuffles them; a root
attribute ``cal_scan`` (JSON) describes the sub-scan and carries its log.
atomdata loads them as ``ad.cal``, a scan of its own.

Every host parameter the point sets is restored before the next data shot:
``begin_point`` snapshots ``vars(params)`` and ``end_point`` puts it back
whole, so derived parameters cannot leak either.

Refused: a run that captures camera frames (liveOD and the camera streams
count frames per shot, and an extra point would shift every later slot), a
sub-scan key that is also a main-scan xvar, and a main-scan xvar among the
parameters the calibration kernel reads (``reads_params``).
"""
import copy
import json
import time
import traceback

import numpy as np
from artiq.language.core import kernel, now_mu

from waxa.base.xvar import xvar
from waxx.config.data_vault import DataVault

FORMAT = 1


class SubScanRefused(ValueError):
    """The sub-scan cannot be taken in this run (the message says why)."""


class SubScan:

    def __init__(self, expt, name="cal", every_n_shots=20, at_start=True,
                 alternate=True, reads_params=()):
        self.expt = expt
        self.name = str(name)
        if not self.name.isidentifier():
            raise SubScanRefused(f"sub-scan name {name!r} must be an identifier")
        self.prefix = self.name + "_"
        self.every_n_shots = int(every_n_shots)
        if self.every_n_shots < 1:
            raise SubScanRefused("every_n_shots must be >= 1")
        self.at_start = bool(at_start)
        self.alternate = bool(alternate)
        self.reads_params = tuple(str(k) for k in reads_params)

        # what the kernel-side DataVault needs of an "experiment"
        self.core = expt.core
        self.params = expt.params
        self.data = DataVault(expt=self)

        self.xvarnames = []          # the scan axis (the first key declared)
        self.scan_xvars = []         # one xvar: its counter is the container slot
        self.point_xvars = []        # every zipped key, full tiled values
        self.point_keys = []
        self.xvardims = []
        self._point_values = {}
        self.n_points_per_event = 0
        self.n_cycle = 1
        self.n_events = 0
        self.N = 0
        self.event_starts = []
        self.counter = 0
        self.enabled = True
        self.disabled_reason = ""
        self.quiet = False
        self.finished = False
        self._snapshot = None
        self._point_keys = []
        self.n_ok = 0
        self.n_failed = 0
        self.pushes_failed = []
        self.log = {"points": [], "windows": [], "summary": {}, "notes": []}

    # ------------------------------------------------------------------ prepare

    def xvar(self, key, values):
        """One of the point's parameters: ``values`` per event, zipped with the
        other xvars (equal lengths). A 2-D ``values`` (n_cycle, n_points)
        cycles over its rows event by event (event e takes row e % n_cycle),
        e.g. a readout mode that alternates between events; every 2-D xvar
        must have the same n_cycle. The key must be a parameter the kernel
        reads (it is plugged into ``params`` when missing, so a kernel writer
        exists for it)."""
        if self.finished:
            raise SubScanRefused("xvar() after finish_prepare")
        key = str(key)
        if key in self._point_values:
            raise SubScanRefused(f"sub-scan xvar {key!r} given twice")
        self.expt.param_police(xvar(key, np.atleast_1d(values)))
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.size == 0:
            raise SubScanRefused(f"sub-scan xvar {key!r}: values must be a non-empty 1-D sequence "
                                 f"(per event) or 2-D (n_cycle, n_points)")
        # the kernel writer of an existing key is chosen by its value's type: a
        # float param given integer points would hand fetch_float a numpy int
        # (an RPC type error, outside the point's try), an integer param given
        # fractional points would be rounded silently -- coerce or refuse
        if key in vars(self.params):
            cur = vars(self.params)[key]
            if isinstance(cur, (bool, np.bool_)):
                pass
            elif isinstance(cur, (float, np.floating)):
                values = values.astype(np.float64)
            elif isinstance(cur, (int, np.integer)):
                if not np.all(np.equal(np.mod(values, 1), 0)):
                    raise SubScanRefused(f"sub-scan xvar {key!r}: the param is an integer but the "
                                         f"point values are not whole numbers")
                values = values.astype(np.int64)
        if key not in vars(self.params):
            vars(self.params)[key] = values[0, 0]
        self._point_values[key] = values
        self.point_keys.append(key)

    def add_data_container(self, per_shot_data_shape=(1,), dtype=np.float64):
        """A kernel container the calibration kernel fills (``put_data``);
        assign it to ``self.<name>.data.<key>`` before finish_prepare."""
        return self.data.add_data_container(per_shot_data_shape, dtype)

    def add_host_data_container(self, per_shot_data_shape=(1,), dtype=np.float64, fill_value=0):
        return self.data.add_host_data_container(per_shot_data_shape, dtype, fill_value=fill_value)

    def finish(self, n_data_shots, main_xvarnames=(), setup_camera=False, main_xvardims=()):
        """Called by Expt.finish_prepare_wax after the main vault's init():
        checks, the tiled point arrays, the containers (sized by this
        sub-scan) and their registration in the main vault under the prefix."""
        if self.finished:
            return
        self.finished = True
        if not self.enabled:
            # disabled before the run was prepared (the experiment keeps the
            # object so its kernel code type-checks): containers of zero
            # length, nothing registered, nothing taken
            self._init_empty()
            return
        main_xvarnames = [str(k) for k in main_xvarnames]
        if setup_camera:
            raise SubScanRefused(
                f"sub-scan '{self.name}': this run captures camera frames; liveOD and the camera "
                f"streams count frames per shot, so an interleaved point would shift every later "
                f"frame's slot. Interleaved points are for frame-free runs (APD readout) until "
                f"liveOD knows about them.")
        if not self._point_values:
            raise SubScanRefused(f"sub-scan '{self.name}': no xvar declared")
        lengths = {k: int(v.shape[1]) for k, v in self._point_values.items()}
        if len(set(lengths.values())) != 1:
            raise SubScanRefused(f"sub-scan '{self.name}': the xvars must have equal lengths "
                                 f"(zipped points), got {lengths}")
        cycles = {k: int(v.shape[0]) for k, v in self._point_values.items() if v.shape[0] > 1}
        if len(set(cycles.values())) > 1:
            raise SubScanRefused(f"sub-scan '{self.name}': every cycling xvar must have the same "
                                 f"number of rows, got {cycles}")
        self.n_cycle = int(next(iter(cycles.values()))) if cycles else 1
        clash = [k for k in self.point_keys if k in main_xvarnames]
        if clash:
            raise SubScanRefused(f"sub-scan '{self.name}': {clash} are main-scan xvars")
        clash = [k for k in self.reads_params if k in main_xvarnames]
        if clash:
            raise SubScanRefused(f"sub-scan '{self.name}': the calibration kernel reads {clash}, "
                                 f"which the main scan scans -- the points would mix settings")
        n_data_shots = int(n_data_shots)
        self.n_points_per_event = int(next(iter(lengths.values())))
        starts = [0] if self.at_start else []
        k = 1
        while k * self.every_n_shots < n_data_shots:
            starts.append(k * self.every_n_shots)
            k += 1
        self.event_starts = starts
        self.n_events = len(starts)
        self.N = self.n_events * self.n_points_per_event
        if self.N == 0:
            self.disable("no schedule event fits a run of "
                         f"{n_data_shots} data shots (every {self.every_n_shots})")
            self._init_empty()
            return
        # The containers are declared final_order and pushed point by point,
        # so the saver never unshuffles them -- unless every push failed and
        # they are carried whole at END_RUN, when an axis whose length equals
        # an xvar's would be unshuffled by axis length. Refuse the one
        # configuration where that could silently scramble them.
        clash = [int(d) for d in main_xvardims if int(d) == self.N]
        if clash:
            raise SubScanRefused(
                f"sub-scan '{self.name}': its point count {self.N} equals a main-scan xvar length "
                f"{clash[0]}; change every_n_shots / at_start / the points per event so they differ "
                f"(a container carried whole at END_RUN would otherwise be unshuffled by axis length)")
        # event e takes row e % n_cycle of each xvar; the point order is
        # reversed on every other cycle (not event), so a cycling value sees
        # both orders equally
        for i, key in enumerate(self.point_keys):
            rows = self._point_values[key]
            parts = []
            for e in range(self.n_events):
                row = rows[e % rows.shape[0]]
                reverse = self.alternate and ((e // self.n_cycle) % 2 == 1)
                parts.append(row[::-1] if reverse else row)
            full = np.concatenate(parts)
            self.point_xvars.append(xvar(key, full, position=i))
        # the scan axis is the first key; the containers' slot is its counter
        self.scan_xvars = [self.point_xvars[0]]
        self.xvarnames = [self.point_keys[0]]
        self.xvardims = [self.N]

        # every other zipped key: a per-point container holding its value
        for x in self.point_xvars[1:]:
            if x.key in vars(self.data):
                raise SubScanRefused(f"sub-scan '{self.name}': {x.key!r} is both a zipped key "
                                     f"and a container")
            vars(self.data)[x.key] = self.add_host_data_container((1,), np.float64, fill_value=np.nan)
        self.data.time = self.add_host_data_container((1,), np.float64, fill_value=np.nan)
        self.data.main_index = self.add_host_data_container((2,), np.int64, fill_value=-1)
        self.data.status = self.add_host_data_container((1,), np.int32, fill_value=0)
        self.data.init()
        for x in self.point_xvars[1:]:
            vars(self.data)[x.key]._run_data[:] = np.asarray(x.values, dtype=np.float64)
        main = self.expt.data
        for key in list(self.data.keys):
            dc = vars(self.data)[key]
            full = self.prefix + key
            if full in main.keys or full in vars(main):
                raise SubScanRefused(f"sub-scan '{self.name}': the main vault already has {full!r}")
            dc.key = full
            vars(main)[full] = dc
            main.keys.append(full)

    def _init_empty(self):
        """Size the containers to zero points so the kernel code that names
        them type-checks; they are never written, pushed or saved."""
        self.xvardims = [0]
        self.N = 0
        if not self.data.keys:
            self.data.init()

    def disable(self, reason, quiet=False):
        """No points this run. ``quiet``: an intended setting (the experiment
        turned the sub-scan off), not a failure -- nothing is printed loudly
        and nothing is recorded in the file."""
        self.enabled = False
        self.disabled_reason = str(reason)
        self.quiet = bool(quiet)
        self.log["notes"].append(f"disabled: {reason}")

    def describe(self):
        pts = ", ".join(f"{k}={np.asarray(v).tolist()}" for k, v in self._point_values.items())
        return (f"sub-scan '{self.name}': {self.n_points_per_event} point(s) [{pts}] per event, "
                f"{self.n_events} event(s) (before data shot(s) {self.event_starts}), "
                f"{self.N} points in all"
                + (", pair order alternating" if self.alternate and self.n_points_per_event > 1 else ""))

    # ------------------------------------------------------------------ run

    @property
    def keys(self):
        """The main-vault keys of this sub-scan's containers."""
        return [self.prefix + k for k in self.data.keys]

    def due(self, n_data_done):
        if not (self.enabled and self.finished and self.counter < self.N):
            return False
        event = self.counter // self.n_points_per_event
        return int(n_data_done) >= self.event_starts[event]

    def point_values(self, i=None):
        i = self.counter if i is None else int(i)
        return {x.key: x.values[i] for x in self.point_xvars}

    def begin_point(self):
        """Set the point's parameters on the host (the scan loop ships them to
        the kernel next). The whole params dict is snapshotted first."""
        if self._snapshot is not None:
            self._restore()
        self._snapshot = copy.deepcopy(vars(self.params))
        for x in self.point_xvars:
            x.counter = self.counter
            vars(self.params)[x.key] = x.values[self.counter]
        self.params.compute_derived()
        self.expt.compute_new_derived()

    def _restore(self):
        if self._snapshot is None:
            return
        d = vars(self.params)
        d.clear()
        d.update(self._snapshot)
        self._snapshot = None

    def end_point(self, ok, what=""):
        """After the point: restore the host params, record the point (time,
        the preceding data shot, status), push its containers' slot, log it.
        The counter advances whether the point succeeded or not."""
        self._restore()
        i = self.counter
        try:
            entry = {"slot": i, "event": i // self.n_points_per_event if self.n_points_per_event else 0,
                     "ok": bool(ok), "what": str(what), "t": time.time(),
                     "values": {k: _jsonable(v) for k, v in self.point_values(i).items()}}
            try:
                acq, final = getattr(self.expt, "_last_data_shot", (-1, -1))
            except Exception:
                acq, final = -1, -1
            entry["main_acq_index"] = int(acq)
            entry["main_final_slot"] = int(final)
            if ok:
                self.n_ok += 1
                self.data.time.put_shot_data_host((i,), entry["t"])
                self.data.main_index.put_shot_data_host((i,), np.array([acq, final], dtype=np.int64))
                self.data.status.put_shot_data_host((i,), 1)
                items = [(vars(self.data)[k], vars(self.data)[k]._run_data[i]) for k in self.data.keys]
                pushed = self.expt.push_shot_data(items, idx=(i,), final=True)
                entry["pushed"] = bool(pushed)
            else:
                self.n_failed += 1
                self.data.status.put_shot_data_host((i,), 2)
                items = [(self.data.status, self.data.status._run_data[i])]
                self.expt.push_shot_data(items, idx=(i,), final=True)
            self.log["points"].append(entry)
        finally:
            # whatever happened above, the point is done: a counter left behind
            # would make the scan loop take the same point again, forever
            self.counter += 1
        # the experiment's host hook (a calibration policy's entry point);
        # its errors are logged, never raised
        hook = getattr(self.expt, "on_sub_scan_point", None)
        if hook is not None:
            try:
                hook(self, i, bool(ok))
            except Exception as e:
                self.log["notes"].append(f"on_sub_scan_point({i}): {error_text(e)}")
                print(f"[sub-scan] !! host error in on_sub_scan_point (the run goes on): {e!r}")
        return entry

    def finish_run(self):
        """Before END_RUN: a container whose slot push failed is pushed whole,
        in slot order; then the description + log go to the file's attrs."""
        expt = self.expt
        failed = [k for k in self.keys if k in getattr(expt, "_push_failed", set())]
        if failed and getattr(expt, "push_data_enabled", False):
            for full in failed:
                dc = vars(expt.data)[full]
                spec = {"key": full, "index": None, "offset": 0, "array": dc._run_data,
                        "full_shape": dc._run_data.shape, "fill": getattr(dc, "_fill_value", None)}
                if expt._push([spec]):
                    dc._pushed = True
                    expt._push_failed.discard(full)
                else:
                    self.pushes_failed.append(full)
        elif failed:
            self.pushes_failed.extend(failed)
        # the experiment's end-of-run hook (a policy's evaluation into the log)
        hook = getattr(expt, "on_sub_scan_finish", None)
        if hook is not None:
            try:
                hook(self)
            except Exception as e:
                self.log["notes"].append(f"on_sub_scan_finish: {error_text(e)}")
                print(f"[sub-scan] !! host error in on_sub_scan_finish (the file still gets the "
                      f"points): {e!r}")
        self.log["summary"] = {
            "n_points": int(self.N), "n_taken": int(self.counter), "n_ok": int(self.n_ok),
            "n_failed": int(self.n_failed), "enabled": bool(self.enabled),
            "disabled_reason": self.disabled_reason,
            "pushes_failed_keys": list(self.pushes_failed),
        }
        expt._extra_file_texts["cal_scan"] = json.dumps(self.to_dict(), default=_jsonable)

    def to_dict(self):
        return {
            "format": FORMAT, "name": self.name, "prefix": self.prefix,
            "every_n_shots": self.every_n_shots, "at_start": self.at_start,
            "alternate": self.alternate, "reads_params": list(self.reads_params),
            "n_points_per_event": self.n_points_per_event, "n_cycle": self.n_cycle,
            "n_events": self.n_events,
            "N": self.N, "event_starts": list(self.event_starts),
            "xvarnames": list(self.xvarnames),
            "xvars": {x.key: np.asarray(x.values).tolist() for x in self.scan_xvars},
            "point_keys": list(self.point_keys),
            "point_xvars": {x.key: np.asarray(x.values).tolist() for x in self.point_xvars},
            "point_values": {k: np.asarray(v).tolist() for k, v in self._point_values.items()},
            "xvardims": list(self.xvardims), "keys": list(self.data.keys),
            "enabled": bool(self.enabled), "disabled_reason": self.disabled_reason,
            "log": self.log,
        }

    # ------------------------------------------------------------------ kernel

    @kernel
    def put_shot_data(self):
        """Sync this point's containers kernel -> host at the sub-scan's slot.
        The calibration kernel's last statement. A copy of
        DataVault.put_shot_data (one function per vault instance: ARTIQ types a
        kernel method against one ``self``)."""
        self.core.wait_until_mu(now_mu())
        for dc_1d_f64 in self.data._list_1d_f64:
            dc_1d_f64._put_shot_data()
        for dc_2d_f64 in self.data._list_2d_f64:
            dc_2d_f64._put_shot_data()
        for dc_1d_i32 in self.data._list_1d_i32:
            dc_1d_i32._put_shot_data()
        for dc_2d_i32 in self.data._list_2d_i32:
            dc_2d_i32._put_shot_data()
        for dc_1d_i64 in self.data._list_1d_i64:
            dc_1d_i64._put_shot_data()
        for dc_2d_i64 in self.data._list_2d_i64:
            dc_2d_i64._put_shot_data()
        self.core.break_realtime()


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)


def error_text(exc):
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-1500:]
