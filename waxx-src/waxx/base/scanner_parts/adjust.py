"""``self.adjust(...)``: params a person can change between shots from the
liveOD Adjust panel.

For the experiment writer there is one call, in prepare()::

    self.adjust('t_tof', min_val=20.e-6, max_val=20.e-3)

``adjust()`` only records the request. At finish_prepare the requests become
the run's ``AdjustSpec`` list (``_resolve_adjust_specs``), which liveOD shows
as the panel's rows. Before each shot ``Expt.apply_pending_adjust_values``
puts the values changed in the panel into the host params. The rules:

* a run nobody adjusts runs exactly as it would without its adjust() lines;
* a key that is also an xvar gets no row (the xvar wins);
* a key that is recomputed every shot (a derived param) gets a read-only row;
* with save_data=True every row is read-only and no panel value is applied.
"""
import numpy as np


class AdjustSpec:
    """Descriptor for a parameter that can be adjusted live between shots.

    ``current_val`` is the value the run starts with (the param's value at
    finish_prepare); ``locked`` is '' for a live row, else why the row is
    read-only (``LOCK_SAVE_DATA``, ``LOCK_DERIVED``): the host never applies a
    panel value to a locked key, and liveOD refuses to change one."""
    LOCK_SAVE_DATA = "save_data"
    LOCK_DERIVED = "derived"

    def __init__(self, key, min_val, max_val, step, dtype=float, current_val=None,
                 unit='', locked=''):
        self.key = key
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.step = float(step)
        self.dtype = dtype          # float or int
        self.unit = unit or ''      # display unit only; values stay SI everywhere
        self.locked = locked or ''
        raw_current = current_val if current_val is not None else min_val
        self.current_val = self.coerce(raw_current)

    def coerce(self, value, like=None):
        """Casts a value to this spec's dtype.

        The liveOD server ships all adjust values as floats over JSON. A param
        registered as an int must stay an int on the host, or the kernel-side
        RPC (fetch_int32/fetch_int64) fails with a type mismatch.

        Args:
            value: the value to cast.
            like: optional current param value whose numpy int type is preserved.
        """
        if self.dtype == int:
            out = int(round(float(value)))
            if isinstance(like, np.integer):
                return type(like)(out)
            return out
        return float(value)

    def to_dict(self) -> dict:
        return {
            'key':         self.key,
            'min_val':     self.min_val,
            'max_val':     self.max_val,
            'step':        self.step,
            'dtype':       'int' if self.dtype == int else 'float',
            'current_val': self.current_val,
            'unit':        self.unit,
            'locked':      self.locked,
        }


class AdjustPanel:

    def _init_adjust(self):
        self.__dict__.setdefault("_adjust_requests", [])  # adjust() calls, resolved at finish_prepare
        self._adjust_specs = []         # the resolved specs (_resolve_adjust_specs)
        self._pending_adjust_values = {}
        self._adjust_applied = set()    # keys a panel value currently overrides

    def adjust(self, param_key, min_val=None, max_val=None, step=None, dtype=None,
               default_val=None, unit=None):
        """Register a parameter for live adjustment between shots via the liveOD Adjust panel.

        Only records the request. Everything is settled at finish_prepare
        (``_resolve_adjust_specs``), after all of prepare() has run, so an
        override of the param anywhere in prepare() -- before or after this
        line -- is what the run starts with. Until a value is changed in the
        panel the param is never touched: a run nobody adjusts runs exactly as
        it would without its adjust() lines.

        Args:
            param_key (str): ExptParams attribute to adjust.
            min_val (float, optional): Minimum of the panel's range. Defaults to
                the start value - 50%. Lowered to the start value if the start
                value is below it (with a note); the param is never moved.
            max_val (float, optional): Maximum of the panel's range. Defaults to
                the start value + 50%; raised to the start value likewise.
                A start value of 0 needs both limits given.
            step (float, optional): Spinbox step size. Defaults to (max-min)/50.
                For dtype=int the minimum effective step is 1.
            dtype (type): float or int. Defaults to the type of the start value.
            default_val (optional): The start value for a param that does not
                exist in ExptParams. A param that exists (in ExptParams or set
                in prepare()) keeps its own value: default_val loses, with a
                note when they differ. Without either, min_val is used.
            unit (str, optional): Unit the Adjust panel shows this param in
                ('µs', 'MHz', 'V', ...). Defaults to the unit comment on the
                param's line in ExptParams, else a guess from its name and the
                size of its start value. Display only: the value stays SI
                here, over the wire, and in ExptParams.

        A key that is also an xvar is dropped (the xvar wins). A key that
        params.compute_derived() or compute_new_derived() recomputes every shot
        is shown read-only, since a panel value would be overwritten. With
        save_data=True every row is read-only and no panel value is applied.
        """
        requests = self.__dict__.setdefault('_adjust_requests', [])
        if any(r['key'] == param_key for r in requests):
            raise ValueError(f"adjust key {param_key!r} already registered.")
        requests.append(dict(key=param_key, min_val=min_val, max_val=max_val, step=step,
                             dtype=dtype, default_val=default_val, unit=unit))

    def _resolve_adjust_specs(self, save_data=False):
        """finish_prepare: turns the adjust() requests into the run's specs.

        Start value = the param's value now (prepare() is over), else
        default_val, else min_val. Limits that exclude it are moved out to it
        (printed); the param is not moved. An xvar of the same key wins. A key
        the per-shot derived recompute overwrites is locked ``derived``;
        ``save_data`` locks every spec."""
        self._adjust_specs = []
        self._adjust_applied = set()
        self._pending_adjust_values = {}
        p = vars(self.params)
        for r in self.__dict__.get('_adjust_requests', []):
            key = r['key']
            if key in self.xvarnames:
                print(f"[adjust] {key!r} is also an xvar: the xvar wins, no adjust row.")
                continue
            param_exists = key in p
            default_val = r['default_val']
            if param_exists:
                start = p[key]
                if default_val is not None and not np.isclose(float(default_val), float(start)):
                    print(f"[adjust] {key}: default_val={default_val!r} ignored, the param "
                          f"is already {start!r} (the param's value wins).")
            elif default_val is not None:
                start = default_val
            elif r['min_val'] is not None:
                start = r['min_val']
            else:
                raise ValueError(f"adjust param {key!r} does not exist: give default_val "
                                 f"(or min_val) for its start value.")
            dtype = r['dtype']
            if dtype is None:
                dtype = int if isinstance(start, (int, np.integer)) else float
            start_f = float(start)
            min_val, max_val = r['min_val'], r['max_val']
            if min_val is None or max_val is None:
                if start_f == 0.:
                    raise ValueError(f"adjust param {key!r} starts at 0: give both min_val "
                                     f"and max_val (the default range is the start value ±50%).")
                half = 0.5 * abs(start_f)
                if min_val is None:
                    min_val = start_f - half
                    if dtype == int:
                        min_val = np.floor(min_val)
                if max_val is None:
                    max_val = start_f + half
                    if dtype == int:
                        max_val = np.ceil(max_val)
            min_val, max_val = float(min_val), float(max_val)
            if min_val > max_val:
                raise ValueError(f"adjust param {key!r}: min_val {min_val} > max_val {max_val}.")
            if start_f < min_val:
                print(f"[adjust] {key} = {start!r} < min {min_val!r}; min lowered to {start!r}.")
                min_val = start_f
            if start_f > max_val:
                print(f"[adjust] {key} = {start!r} > max {max_val!r}; max raised to {start!r}.")
                max_val = start_f
            step = r['step']
            if step is None:
                raw = (max_val - min_val) / 50.0
                step = max(1, round(raw)) if dtype == int else raw
            elif dtype == int:
                step = max(1, int(step))
            unit = r['unit']
            if unit is None:
                unit = self._detect_adjust_unit(key, start_f, min_val, max_val, dtype)
            spec = AdjustSpec(key, min_val, max_val, step, dtype, start, unit)
            if not param_exists:
                # coerce keeps an int an int (the kernel writer is chosen by type)
                p[key] = spec.coerce(start)
            self._adjust_specs.append(spec)
        for spec in self._adjust_specs:
            if save_data:
                spec.locked = AdjustSpec.LOCK_SAVE_DATA
            elif self._adjust_key_is_derived(spec):
                spec.locked = AdjustSpec.LOCK_DERIVED
                print(f"[adjust] WARNING: p.{spec.key} is recomputed every shot by "
                      f"compute_derived() / compute_new_derived(); a value set in the Adjust "
                      f"panel would be overwritten, so its row is read-only. Adjust the "
                      f"params it is derived from instead.")
        if save_data and self._adjust_specs:
            print(f"[adjust] save_data=True: the Adjust panel is read-only for this run "
                  f"({len(self._adjust_specs)} param(s)); no panel value is applied.")

    def _adjust_key_is_derived(self, spec):
        """True when the per-shot recompute (params.compute_derived() then
        compute_new_derived()) overwrites ``spec.key``: set the key to another
        value, recompute, and see if it survives. Done on the real params
        (self.p and the device objects hold references to them, so a copy
        would not be what compute_new_derived writes), and every param is put
        back afterwards: arrays in place, so a reference to one stays valid.
        A failure to check returns False (the row stays live) with a note."""
        import copy
        d = vars(self.params)
        try:
            snapshot = copy.deepcopy(d)
        except Exception as exc:
            print(f"[adjust] could not check whether {spec.key!r} is a derived param "
                  f"(params not copyable: {exc!r}); its row stays adjustable.")
            return False
        try:
            v0 = d[spec.key]
            if spec.dtype == int:
                trial = spec.coerce(float(v0) + 7, like=v0)
            else:
                trial = spec.coerce(float(v0) * 1.37 + 1.e-3 * (1 + abs(float(v0))))
            d[spec.key] = trial
            self.params.compute_derived()
            self.compute_new_derived()
            after = d.get(spec.key)
            return not np.isclose(float(after), float(trial), rtol=1e-12, atol=0.)
        except Exception as exc:
            print(f"[adjust] could not check whether {spec.key!r} is a derived param "
                  f"({exc!r}); its row stays adjustable.")
            return False
        finally:
            for k in list(d):
                if k not in snapshot:
                    del d[k]
            for k, v in snapshot.items():
                cur = d.get(k)
                if (isinstance(cur, np.ndarray) and isinstance(v, np.ndarray)
                        and cur.shape == v.shape and cur.dtype == v.dtype):
                    cur[...] = v
                else:
                    d[k] = v
            try:
                # experiment state compute_new_derived keeps outside the params
                # goes back to what the real values give
                self.compute_new_derived()
            except Exception:
                pass

    def _detect_adjust_unit(self, param_key, current_val, min_val, max_val, dtype):
        """Display unit for an adjust param: the ExptParams comment, else the name.

        Sized by the default value rather than the range, so a t_tof that starts
        at 20 us shows in us even though it can be adjusted up into the ms.
        Ints are counts (shot numbers, indices) and stay unitless. A failure
        here must never stop an experiment from being built, so it falls back to
        no unit.
        """
        if dtype == int:
            return ''
        try:
            from waxa.units import unit_for_param
            return unit_for_param(param_key, [current_val, min_val, max_val],
                                  params_obj=self.params)
        except Exception as exc:
            print(f"[adjust] unit detection failed for {param_key!r}: {exc}")
            return ''

    def apply_pending_adjust_values(self):
        """Apply GUI-side adjust values to host params. Overridden in Expt."""
        pass
