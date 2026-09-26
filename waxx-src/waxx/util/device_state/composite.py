"""Composite devices: one control, many channels.

A composite device is anything whose simplest action touches several
channels -- the Raman pair ("on" = two AO frequencies, the switch AO and a
shutter), the lightsheet, the tweezer AWG, a coil.  The Device Control GUI's
Composite tab drives them through the monitor experiment:

    GUI --op request--> monitor server --poll--> monitor experiment
                          (queue, 3 s TTL)        runs the op's kernel code,
                                                  writes the channels it changed
                                                  back to the device-state JSON
    GUI <--op_result broadcast-- server <--report-- monitor

This module is the definition language.  It imports neither Qt nor ARTIQ so
all three processes can load it: the GUI renders cards from it, the monitor
experiment compiles one kernel per op from it, and the server only ever sees
op names and signatures.

Writing a device
----------------
A :class:`CompositeDevice` has

* ``fields`` -- the values an operator types (:class:`Arg`): units, display
  scale, hard limits (refused), soft limits (asked about), a default, and a
  ``readback`` that reads what the hardware is at now out of the channel
  states, so the GUI shows the machine and not the last thing typed.
* ``ops`` -- :class:`Op`: a kernel body run by the monitor on the experiment
  object ``expt`` (``expt.raman``, ``expt.ttl.raman_shutter``, ``expt.p``...),
  with ``{field}`` placeholders for the fields it sends.  An op may also have
  a ``host`` step, a Python function run in the monitor's host process (the
  tweezer AWG is a host-side device).
* ``lamps`` / ``readouts`` / ``state`` -- read-only indicators, computed from
  the device-state JSON the GUI already holds.
* ``layout`` -- rows (:class:`Buttons`, :class:`FieldRow`, :class:`Menu`,
  :class:`ChannelToggle`, :class:`TableRow`, :class:`Info`).

Kernel code rules (it is compiled with ``kernel_from_string``):

* Only ``expt`` and the placeholders are in scope, plus the ARTIQ builtins
  (``delay``, ``now_mu``, ``at_mu``, ``int``, ``float``, ``min``, ``max``,
  ``abs``, ``parallel``...).  No ``np``, no module constants -- format them in
  as literals, or read ``expt.p``.
* A placeholder becomes ``aN`` (float), ``int(aN)`` (int / choice) or
  ``(aN > 0.5)`` (bool), so write it where that expression fits.
* Every body is prefixed with ``delay(expt.monitor.t_op_slack)``: the op's
  timeline slack, and on purpose read from an attribute (see
  :data:`SLACK_PREFIX`).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

# --- kernel interface -------------------------------------------------------

#: Float arguments every op kernel takes after ``expt``.  Fixed arity is what
#: lets all op kernels share one list (and so one call site) in the monitor.
N_OP_ARGS = 8
OP_ARG_NAMES = tuple(f"a{i}" for i in range(N_OP_ARGS))
KERNEL_PARAMS = ("expt",) + OP_ARG_NAMES

#: First line of every op kernel.  It is the op's timeline slack, and it is
#: read from an attribute *on purpose*: ARTIQ's I/O-delay estimator cannot
#: evaluate an attribute, so every op kernel gets an indeterminate delay.  The
#: op kernels live in one list, whose element type unifies their function
#: types -- delay included -- and two ops with different fixed delays (one
#: with a 3 ms shutter wait, one without) would otherwise fail to compile
#: together ("delay ... is already constrained externally").
SLACK_PREFIX = "delay(expt.monitor.t_op_slack)"

#: Name of the built-in op at index 0: runs the slack delay only, touches no
#: hardware.  The GUI's "Ping" sends it to time the whole op path.
PING_OP = "monitor.ping"

# --- op status codes (kernel -> host -> server -> GUI) ----------------------

OP_OK = 0
OP_UNDERFLOW = 1
OP_VALUE_ERROR = 2
OP_RUNTIME_ERROR = 3
OP_EXCEPTION = 4
OP_HOST_ERROR = 5
OP_REJECTED = 6
OP_EXPIRED = 7
OP_LOST = 8

OP_STATUS_TEXT = {
    OP_OK: "done",
    OP_UNDERFLOW: ("RTIO underflow inside the op -- it ran out of timeline "
                   "slack part way through, so the hardware may be only "
                   "partly set"),
    OP_VALUE_ERROR: ("ValueError raised by the device code (one of its own "
                     "limit checks) -- nothing after the check ran"),
    OP_RUNTIME_ERROR: "RuntimeError raised by the device code",
    OP_EXCEPTION: "exception raised by the device code",
    OP_HOST_ERROR: "the host-side step failed",
    OP_REJECTED: "rejected by the monitor",
    OP_EXPIRED: "expired in the queue -- the monitor never picked it up",
    OP_LOST: ("the monitor stopped after taking the op and before reporting "
              "it -- it may or may not have run"),
}

OP_FINAL = frozenset(OP_STATUS_TEXT)

KIND_FLOAT = "float"
KIND_INT = "int"
KIND_BOOL = "bool"
KIND_CHOICE = "choice"
_KINDS = (KIND_FLOAT, KIND_INT, KIND_BOOL, KIND_CHOICE)

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DefinitionError(ValueError):
    """A composite device definition is inconsistent (checked at load)."""


# --- checks -----------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    """One finding about a value: ``"warn"`` asks first, ``"error"`` refuses."""

    level: str
    message: str

    @staticmethod
    def warn(message: str) -> "Check":
        return Check("warn", message)

    @staticmethod
    def error(message: str) -> "Check":
        return Check("error", message)

    @property
    def is_error(self) -> bool:
        return self.level == "error"


def _as_checks(result) -> list[Check]:
    if result is None:
        return []
    if isinstance(result, Check):
        return [result]
    return [c for c in result if c is not None]


def worst(checks: Iterable[Check]) -> str:
    """``"error"``, ``"warn"`` or ``"ok"`` for a set of findings."""
    levels = {c.level for c in checks}
    if "error" in levels:
        return "error"
    if "warn" in levels:
        return "warn"
    return "ok"


# --- context: what GUI-side callables see -----------------------------------

@dataclass(frozen=True)
class Sample:
    """One measured value from a telemetry source.  ``age_s`` is how old the
    measurement is by the source's own clock where it has one."""

    value: Any
    age_s: float = 0.0
    ok: bool = True
    error: str = ""


class Context:
    """Read access for GUI-side callables (readbacks, checks, defaults).

    ``config`` is the device-state JSON (``{"dds": {...}, "dac": {...},
    "ttl": {...}}``) as the GUI holds it; ``params`` the lab's ExptParams (or
    None); ``frames`` whatever the launcher passed (the lab's dds/dac frames);
    ``fields`` the card's current field values in SI; ``host_state`` the
    device's last host-side state reported by the monitor; ``telemetry``
    measured values by source key (:class:`Sample`); ``trust`` whether the
    device-state file is known to match the hardware (``{"trusted": bool,
    "reason": str}`` -- False after a run that never reported its end state).
    """

    def __init__(self, config: Mapping | None = None, params=None, frames=None,
                 fields: Mapping | None = None, host_state: Mapping | None = None,
                 telemetry: Mapping | None = None, trust: Mapping | None = None):
        self.config = config or {}
        self.params = params
        self.frames = frames
        self.fields = dict(fields or {})
        self.host_state = dict(host_state or {})
        self.telemetry = telemetry or {}
        self.trust = dict(trust or {"trusted": True, "reason": ""})

    @property
    def trusted(self) -> bool:
        return bool(self.trust.get("trusted", True))

    def measured(self, source: str, max_age_s: float | None = None) -> Sample | None:
        """The latest sample of a telemetry source, or None when there is none
        (or it is older than ``max_age_s``, or the source reported an error)."""
        sample = self.telemetry.get(source)
        if not isinstance(sample, Sample) or not sample.ok:
            return None
        if max_age_s is not None and sample.age_s > max_age_s:
            return None
        return sample

    def channel(self, dtype: str, name: str) -> dict | None:
        section = self.config.get(dtype) or {}
        entry = section.get(name)
        return entry if isinstance(entry, dict) else None

    def dds(self, name: str) -> dict | None:
        return self.channel("dds", name)

    def dac(self, name: str) -> dict | None:
        return self.channel("dac", name)

    def ttl(self, name: str) -> dict | None:
        return self.channel("ttl", name)

    def _number(self, dtype, name, key):
        entry = self.channel(dtype, name)
        if entry is None or entry.get(key) is None:
            return None
        try:
            return float(entry[key])
        except (TypeError, ValueError):
            return None

    def dds_frequency(self, name: str) -> float | None:
        return self._number("dds", name, "frequency")

    def dds_amplitude(self, name: str) -> float | None:
        return self._number("dds", name, "amplitude")

    def dds_v_pd(self, name: str) -> float | None:
        return self._number("dds", name, "v_pd")

    def dac_voltage(self, name: str) -> float | None:
        return self._number("dac", name, "voltage")

    def is_on(self, dtype: str, name: str) -> bool | None:
        """Switch state of a DDS (``sw_state``) or TTL (``ttl_state``)."""
        key = {"dds": "sw_state", "ttl": "ttl_state"}.get(dtype)
        value = self._number(dtype, name, key) if key else None
        return None if value is None else value > 0.5

    def field(self, name: str, default=None):
        value = self.fields.get(name)
        return default if value is None else value


# --- values -----------------------------------------------------------------

@dataclass(frozen=True)
class Arg:
    """A value an operator sets.  Everything is SI; ``scale`` is display only.

    ``default`` / ``readback`` / ``check`` may be callables taking a
    :class:`Context` (``check`` also takes the value first).  ``readback``
    returns the value the hardware is at now, or None when it cannot be read
    from the channel states.
    """

    name: str
    label: str = ""
    unit: str = ""
    scale: float = 1.0
    decimals: int = 3
    step: float | None = None           # display units
    kind: str = KIND_FLOAT
    choices: tuple = ()                 # ((label, value), ...) for KIND_CHOICE
    default: Any = None
    readback: Callable | None = None
    minimum: float | None = None        # hard limits (SI): refused everywhere
    maximum: float | None = None
    warn_below: float | None = None     # soft limits (SI): the GUI asks first
    warn_above: float | None = None
    check: Callable | None = None       # (value, ctx) -> Check | [Check] | None
    tooltip: str = ""
    param: str = ""                     # the ExptParams attribute this field mirrors
    #: The value sent instead when the op goes out long after it was prepared
    #: (a watchdog firing): for a field that carries a measurement taken at
    #: preparation time, which would be stale by then.  None = same value.
    replay: Any = None

    # -- display --------------------------------------------------------------

    @property
    def title(self) -> str:
        return self.label or self.name

    def to_display(self, value: float) -> float:
        return value * self.scale

    def from_display(self, value: float) -> float:
        return value / self.scale

    def format(self, value: float | None) -> str:
        if value is None:
            return "--"
        if self.kind == KIND_CHOICE:
            for label, v in self.choices:
                if float(v) == float(value):
                    return label
        if self.kind == KIND_BOOL:
            return "on" if value > 0.5 else "off"
        text = f"{self.to_display(value):.{self.decimals}f}"
        return f"{text} {self.unit}".strip()

    # -- values -------------------------------------------------------------

    def kernel_expr(self, var: str) -> str:
        """How the placeholder reads in kernel code for argument ``var``."""
        if self.kind in (KIND_INT, KIND_CHOICE):
            return f"int({var})"
        if self.kind == KIND_BOOL:
            return f"({var} > 0.5)"
        return var

    def resolve(self, spec, ctx: Context | None):
        """Evaluate a default/readback spec: a constant or a callable(ctx)."""
        if spec is None:
            return None
        try:
            value = spec(ctx) if callable(spec) else spec
        except Exception:
            return None
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def default_value(self, ctx: Context | None) -> float | None:
        return self.resolve(self.default, ctx)

    def readback_value(self, ctx: Context | None) -> float | None:
        return self.resolve(self.readback, ctx)

    def hard_problems(self, value) -> list[Check]:
        """Findings that refuse the value outright -- the monitor re-runs these."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return [Check.error(f"{self.title}: not a number ({value!r})")]
        if not math.isfinite(value):
            return [Check.error(f"{self.title}: not a finite number")]
        if self.kind in (KIND_INT, KIND_CHOICE) and value != int(value):
            return [Check.error(f"{self.title}: must be a whole number")]
        if self.kind == KIND_CHOICE and self.choices and \
                value not in [float(v) for _, v in self.choices]:
            return [Check.error(f"{self.title}: {value:g} is not one of the choices")]
        if self.minimum is not None and value < self.minimum:
            return [Check.error(f"{self.title} {self.format(value)} is below the "
                                f"limit {self.format(self.minimum)}")]
        if self.maximum is not None and value > self.maximum:
            return [Check.error(f"{self.title} {self.format(value)} is above the "
                                f"limit {self.format(self.maximum)}")]
        return []

    def checks(self, value, ctx: Context | None = None) -> list[Check]:
        """Hard limits, then soft limits, then the custom check."""
        found = self.hard_problems(value)
        if found:
            return found
        value = float(value)
        if self.warn_below is not None and value < self.warn_below:
            found.append(Check.warn(f"{self.title} {self.format(value)} is below "
                                    f"{self.format(self.warn_below)}"))
        if self.warn_above is not None and value > self.warn_above:
            found.append(Check.warn(f"{self.title} {self.format(value)} is above "
                                    f"{self.format(self.warn_above)}"))
        if self.check is not None:
            try:
                found.extend(_as_checks(self.check(value, ctx)))
            except Exception as e:
                found.append(Check.warn(f"{self.title}: check failed ({e!r})"))
        return found


@dataclass(frozen=True)
class Table:
    """A list-valued input (rows of ``columns``) sent as op payload, not as
    kernel arguments.  ``check(rows, ctx)`` sees the whole table; ``info(row,
    ctx)`` returns a derived text per row (e.g. a tweezer position)."""

    name: str
    label: str
    columns: tuple
    default: Any = None                 # list of rows, or callable(ctx)
    check: Callable | None = None
    info: Callable | None = None
    info_label: str = ""
    min_rows: int = 0
    max_rows: int = 64
    tooltip: str = ""

    def default_rows(self, ctx: Context | None) -> list[list[float]]:
        spec = self.default
        try:
            rows = spec(ctx) if callable(spec) else spec
        except Exception:
            rows = None
        return [list(map(float, r)) for r in (rows or [])]

    def checks(self, rows, ctx: Context | None = None) -> list[Check]:
        found: list[Check] = []
        if len(rows) < self.min_rows:
            found.append(Check.error(f"{self.label}: at least {self.min_rows} row(s)"))
        if len(rows) > self.max_rows:
            found.append(Check.error(f"{self.label}: at most {self.max_rows} rows"))
        for i, row in enumerate(rows):
            if len(row) != len(self.columns):
                found.append(Check.error(f"{self.label} row {i + 1}: "
                                         f"{len(row)} values, expected {len(self.columns)}"))
                continue
            for col, value in zip(self.columns, row):
                for c in col.checks(value, ctx):
                    found.append(Check(c.level, f"{self.label} row {i + 1}: {c.message}"))
        if self.check is not None and not any(c.is_error for c in found):
            try:
                found.extend(_as_checks(self.check(rows, ctx)))
            except Exception as e:
                found.append(Check.warn(f"{self.label}: check failed ({e!r})"))
        return found

    def hard_problems(self, rows) -> list[Check]:
        """Row count and per-cell hard limits -- what the monitor re-checks."""
        if not isinstance(rows, (list, tuple)):
            return [Check.error(f"{self.label}: not a list")]
        found = []
        if not (self.min_rows <= len(rows) <= self.max_rows):
            found.append(Check.error(f"{self.label}: {len(rows)} rows, allowed "
                                     f"{self.min_rows}..{self.max_rows}"))
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != len(self.columns):
                found.append(Check.error(f"{self.label}: malformed row {row!r}"))
                continue
            for col, value in zip(self.columns, row):
                found.extend(col.hard_problems(value))
        return found


# --- operations -------------------------------------------------------------

@dataclass(frozen=True)
class Op:
    """One action.  ``code`` runs in the monitor kernel; ``host(expt, args,
    payload)`` (optional) runs in the monitor's host process, before the code
    unless ``host_after``.  ``args`` name the device fields sent with it,
    ``payload`` the device tables.  ``check(args, ctx)`` adds cross-field
    findings; ``confirm`` is always asked; ``danger`` styles it red and always
    confirms."""

    key: str
    label: str
    code: str = ""
    args: tuple = ()
    payload: tuple = ()
    host: Callable | None = None
    host_after: bool = False
    check: Callable | None = None
    confirm: str = ""
    danger: bool = False
    tooltip: str = ""


# --- indicators -------------------------------------------------------------

@dataclass(frozen=True)
class Lamp:
    """On/off of one channel: a DDS RF switch (``sw_state``) or a TTL.
    ``on_level`` is how "on" is coloured: ``"ok"``, ``"warn"`` or ``"err"``."""

    label: str
    dtype: str
    name: str
    on_text: str = "on"
    off_text: str = "off"
    on_level: str = "ok"
    tooltip: str = ""


@dataclass(frozen=True)
class Readout:
    """A computed read-only value: ``value(ctx) -> str | None``."""

    label: str
    value: Callable
    tooltip: str = ""


@dataclass(frozen=True)
class Measured:
    """A measured value from a telemetry source, shown as its own chip next to
    the setpoints (never in place of one).  ``source`` is ``"provider/key"``
    as the lab's telemetry providers publish it.  ``expect(ctx)`` gives the
    commanded value to compare against; a difference above ``tolerance``
    colours the chip.  ``text(sample, ctx)`` formats non-numeric values;
    ``judge(sample, ctx) -> "ok" | "warn"`` colours them."""

    label: str
    source: str
    unit: str = ""
    scale: float = 1.0
    decimals: int = 2
    expect: Callable | None = None
    tolerance: float | None = None
    stale_s: float = 5.0
    text: Callable | None = None
    tooltip: str = ""
    judge: Callable | None = None

    def format(self, sample: "Sample | None", ctx: "Context | None" = None) -> str:
        if sample is None:
            return "--"
        if self.text is not None:
            try:
                return str(self.text(sample, ctx))
            except Exception as e:
                return f"error: {e!r}"
        try:
            return f"{float(sample.value) * self.scale:.{self.decimals}f} {self.unit}".strip()
        except (TypeError, ValueError):
            return str(sample.value)


#: Status levels.  ``hazard`` is for states that must stand out from every
#: ordinary "on" -- an energised coil.
LEVELS = ("on", "off", "partial", "warn", "hazard", "unknown")


@dataclass(frozen=True)
class Status:
    """A device's summary state.  ``level``: one of :data:`LEVELS`.  ``text``
    is the short pill text; ``detail`` the longer explanation (tooltip)."""

    level: str
    text: str
    detail: str = ""


# --- layout -----------------------------------------------------------------

@dataclass(frozen=True)
class Buttons:
    """A row of op buttons; ``main`` makes them the card's large buttons."""

    ops: tuple
    main: bool = False


@dataclass(frozen=True)
class FieldRow:
    """Field editors (spinbox / combo by kind) followed by op buttons."""

    fields: tuple
    ops: tuple = ()


@dataclass(frozen=True)
class ChannelToggle:
    """A single TTL, switched through the ordinary channel path (the same as
    the TTL tab) -- for PID hold / enable lines that are one channel each."""

    name: str
    label: str
    on_text: str = "on"
    off_text: str = "off"
    on_level: str = "warn"
    tooltip: str = ""
    dtype: str = "ttl"


@dataclass(frozen=True)
class TableRow:
    table: str
    ops: tuple = ()


@dataclass(frozen=True)
class Info:
    """A muted line of derived text: ``text(ctx) -> str | None``."""

    text: Callable


@dataclass(frozen=True)
class Menu:
    """Less common ops behind a drop-down (danger ops are drawn red)."""

    ops: tuple
    label: str = "More"


# --- devices ----------------------------------------------------------------

@dataclass(frozen=True)
class CompositeDevice:
    """One card.  Besides fields/ops/layout/indicators (see the module
    docstring):

    * ``group`` -- cards of one group are kept together on the tab.
    * ``hazard(ctx) -> str | None`` -- what about this device is dangerous to
      leave on right now ("ON 188.8 A"), or None.  Drives the hazard strip,
      "Make safe" and the pre-run warning.
    * ``safe_op`` / ``safe_args`` -- the op "Make safe" sends (its other
      arguments come from the fields' defaults).
    * ``max_on_s`` -- after this long hazardous, GUIs warn; with the
      watchdog armed, the server then sends ``safe_op`` itself.
    * ``measured`` -- :class:`Measured` chips from telemetry sources.
    """

    key: str
    title: str
    fields: tuple = ()
    tables: tuple = ()
    ops: tuple = ()
    layout: tuple = ()
    lamps: tuple = ()
    readouts: tuple = ()
    state: Callable | None = None       # ctx -> Status
    doc: str = ""
    sync_cache: Callable | None = None  # (expt, config) -> None; monitor start
    host_state: Callable | None = None  # (expt) -> dict; after this device's ops
    group: str = ""
    hazard: Callable | None = None      # ctx -> str | None
    safe_op: str = ""
    safe_args: Any = None               # {arg: SI value}
    max_on_s: float | None = None
    measured: tuple = ()

    def hazard_text(self, ctx: "Context") -> str | None:
        if self.hazard is None:
            return None
        try:
            text = self.hazard(ctx)
        except Exception as e:
            return f"hazard check failed: {e!r}"
        return str(text) if text else None

    def safe_request(self, ctx: "Context | None" = None,
                     deferred: bool = False) -> tuple[str, dict] | None:
        """(op key, args) of the "Make safe" op, arguments resolved: explicit
        ``safe_args`` first, then each field's default.  ``deferred`` (a
        watchdog, which sends it much later) takes each field's ``replay``
        value where it has one."""
        if not self.safe_op:
            return None
        op = self.get_op(self.safe_op)
        explicit = dict(self.safe_args or {})
        args = {}
        for name in op.args:
            spec = self.get_field(name)
            value = explicit.get(name)
            if deferred and spec.replay is not None:
                value = spec.replay
            if value is None:
                value = spec.default_value(ctx)
            if value is None:
                raise DefinitionError(f"{self.key}: safe op {op.key!r} has no value for {name!r}")
            args[name] = float(value)
        return op.key, args

    def measured_values(self, ctx: "Context") -> list[tuple["Measured", "Sample | None", str]]:
        """Each :class:`Measured` with its sample (None when missing or
        older than its ``stale_s``) and a level: ``"ok"`` (within tolerance
        of the expected value, or nothing to compare), ``"warn"`` (outside
        it), ``"stale"`` (no fresh sample)."""
        out = []
        for m in self.measured:
            sample = ctx.measured(m.source, m.stale_s)
            if sample is None:
                out.append((m, None, "stale"))
                continue
            level = "ok"
            if m.judge is not None:
                try:
                    level = "warn" if m.judge(sample, ctx) == "warn" else "ok"
                except Exception:
                    level = "warn"
            elif m.expect is not None and m.tolerance is not None:
                try:
                    expected = m.expect(ctx)
                    if expected is not None and abs(float(sample.value) - float(expected)) \
                            > m.tolerance:
                        level = "warn"
                except (TypeError, ValueError):
                    pass
                except Exception:
                    level = "warn"
            out.append((m, sample, level))
        return out

    def get_field(self, name: str) -> Arg:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"{self.key}: no field {name!r}")

    def get_table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(f"{self.key}: no table {name!r}")

    def get_op(self, key: str) -> Op:
        for op in self.ops:
            if op.key == key:
                return op
        raise KeyError(f"{self.key}: no op {key!r}")

    def status(self, ctx: Context) -> Status:
        if self.state is None:
            return Status("unknown", "")
        try:
            result = self.state(ctx)
        except Exception as e:
            return Status("unknown", f"state error: {e!r}")
        return result if isinstance(result, Status) else Status("unknown", str(result))

    def validate(self) -> None:
        """Raise :class:`DefinitionError` for anything that would misbehave."""
        where = f"composite device {self.key!r}"
        if not _KEY.match(self.key or "") or self.key == "monitor":
            raise DefinitionError(f"{where}: key must be an identifier other than 'monitor'")
        names = [f.name for f in self.fields] + [t.name for t in self.tables]
        if len(names) != len(set(names)):
            raise DefinitionError(f"{where}: duplicate field/table names {names}")
        for f in self.fields:
            if not _KEY.match(f.name):
                raise DefinitionError(f"{where}: field name {f.name!r} is not an identifier")
            if f.kind not in _KINDS:
                raise DefinitionError(f"{where}: field {f.name!r} has unknown kind {f.kind!r}")
            if f.kind == KIND_CHOICE and not f.choices:
                raise DefinitionError(f"{where}: choice field {f.name!r} has no choices")
            if not f.scale:
                raise DefinitionError(f"{where}: field {f.name!r} has scale 0")
        op_keys = [op.key for op in self.ops]
        if len(op_keys) != len(set(op_keys)):
            raise DefinitionError(f"{where}: duplicate op keys {op_keys}")
        field_names = {f.name for f in self.fields}
        table_names = {t.name for t in self.tables}
        for op in self.ops:
            if not _KEY.match(op.key):
                raise DefinitionError(f"{where}: op key {op.key!r} is not an identifier")
            if len(op.args) > N_OP_ARGS:
                raise DefinitionError(f"{where}: op {op.key!r} sends {len(op.args)} "
                                      f"args (max {N_OP_ARGS})")
            missing = [a for a in op.args if a not in field_names]
            if missing:
                raise DefinitionError(f"{where}: op {op.key!r} sends unknown fields {missing}")
            missing = [p for p in op.payload if p not in table_names]
            if missing:
                raise DefinitionError(f"{where}: op {op.key!r} sends unknown tables {missing}")
            used = set(_PLACEHOLDER.findall(op.code))
            undeclared = sorted(used - set(op.args))
            if undeclared:
                raise DefinitionError(f"{where}: op {op.key!r} code uses {undeclared} "
                                      f"but does not send them (args={op.args})")
            if op.host is not None and not callable(op.host):
                raise DefinitionError(f"{where}: op {op.key!r} host step is not callable")
            if op.payload and op.host is None:
                raise DefinitionError(f"{where}: op {op.key!r} sends a payload but has "
                                      f"no host step to use it")
            if not op.code.strip() and op.host is None:
                raise DefinitionError(f"{where}: op {op.key!r} does nothing")
        if self.safe_op:
            if self.safe_op not in op_keys:
                raise DefinitionError(f"{where}: safe_op {self.safe_op!r} is not an op")
            safe = self.get_op(self.safe_op)
            if safe.payload:
                raise DefinitionError(f"{where}: safe_op {self.safe_op!r} needs a payload")
            unknown = set(self.safe_args or {}) - set(safe.args)
            if unknown:
                raise DefinitionError(f"{where}: safe_args {sorted(unknown)} are not args of "
                                      f"{self.safe_op!r}")
            for name in safe.args:
                if name not in (self.safe_args or {}) and self.get_field(name).default is None:
                    raise DefinitionError(f"{where}: safe op {self.safe_op!r} arg {name!r} has "
                                          f"no safe_args value and no field default")
        if self.hazard is not None and not callable(self.hazard):
            raise DefinitionError(f"{where}: hazard must be callable")
        if self.max_on_s and not self.safe_op:
            raise DefinitionError(f"{where}: max_on_s needs a safe_op for the watchdog to send")
        for m in self.measured:
            if not isinstance(m, Measured) or "/" not in m.source:
                raise DefinitionError(f"{where}: measured entries must be Measured with a "
                                      f"'provider/key' source")
        for row in self.layout:
            refs_ops, refs_fields = (), ()
            if isinstance(row, (Buttons, Menu)):
                refs_ops = row.ops
            elif isinstance(row, FieldRow):
                refs_ops, refs_fields = row.ops, row.fields
            elif isinstance(row, TableRow):
                refs_ops = row.ops
                if row.table not in table_names:
                    raise DefinitionError(f"{where}: layout names unknown table {row.table!r}")
            elif isinstance(row, (ChannelToggle, Info)):
                pass
            else:
                raise DefinitionError(f"{where}: unknown layout row {row!r}")
            bad = [k for k in refs_ops if k not in op_keys]
            if bad:
                raise DefinitionError(f"{where}: layout names unknown ops {bad}")
            bad = [n for n in refs_fields if n not in field_names]
            if bad:
                raise DefinitionError(f"{where}: layout names unknown fields {bad}")


# --- the op table: what the monitor compiles and the server registers -------

def _host_fingerprint(fn) -> str:
    if fn is None:
        return ""
    name = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}"
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = ""
    return name + ":" + hashlib.sha1(source.encode()).hexdigest()


@dataclass(frozen=True)
class OpEntry:
    """One compiled op: its index in the monitor's kernel list, the final
    kernel body, and a signature that changes whenever anything that decides
    what the op does changes."""

    index: int
    name: str
    device: CompositeDevice | None
    op: Op | None
    body: str
    signature: str

    @property
    def arg_names(self) -> tuple:
        return tuple(self.op.args) if self.op is not None else ()

    @property
    def payload_names(self) -> tuple:
        return tuple(self.op.payload) if self.op is not None else ()

    @property
    def has_host(self) -> bool:
        return self.op is not None and self.op.host is not None

    @property
    def host_after(self) -> bool:
        return self.has_host and self.op.host_after

    def arg_specs(self) -> list[Arg]:
        return [self.device.get_field(n) for n in self.arg_names]

    def pack(self, args: Mapping[str, Any]) -> list[float]:
        """The 8 floats the kernel receives, in this op's argument order."""
        values = [float(args[name]) for name in self.arg_names]
        return values + [0.0] * (N_OP_ARGS - len(values))

    def hard_problems(self, args: Mapping[str, Any], payload: Mapping | None = None) -> list[str]:
        """Why the monitor must refuse this request (missing args, hard limits)."""
        problems = []
        for spec in self.arg_specs():
            if spec.name not in args:
                problems.append(f"missing argument {spec.name!r}")
                continue
            problems.extend(c.message for c in spec.hard_problems(args[spec.name]))
        payload = payload or {}
        for name in self.payload_names:
            if name not in payload:
                problems.append(f"missing table {name!r}")
                continue
            table = self.device.get_table(name)
            problems.extend(c.message for c in table.hard_problems(payload[name]))
        return problems


def kernel_body(device: CompositeDevice, op: Op) -> str:
    """The op's kernel body: slack prefix + code with placeholders substituted."""
    positions = {name: i for i, name in enumerate(op.args)}

    def substitute(match):
        name = match.group(1)
        return device.get_field(name).kernel_expr(OP_ARG_NAMES[positions[name]])

    code = textwrap.dedent(op.code).strip("\n")
    code = _PLACEHOLDER.sub(substitute, code)
    return SLACK_PREFIX + ("\n" + code if code.strip() else "")


def op_signature(name: str, body: str, device: CompositeDevice | None, op: Op | None) -> str:
    args = []
    if op is not None:
        for a in op.args:
            spec = device.get_field(a)
            args.append([a, spec.kind, spec.minimum, spec.maximum])
    blob = json.dumps({
        "name": name,
        "body": body,
        "args": args,
        "payload": list(op.payload) if op is not None else [],
        "host": _host_fingerprint(op.host) if op is not None else "",
        "host_after": bool(op.host_after) if op is not None else False,
    }, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class OpTable:
    """Every op of a set of devices, in a fixed order.  Index 0 is always
    :data:`PING_OP`.  The GUI and the monitor build this from the same
    definitions; the signatures are how the server tells whether they did."""

    def __init__(self, devices: Sequence[CompositeDevice] = ()):
        self.devices = tuple(devices)
        keys = [d.key for d in self.devices]
        if len(keys) != len(set(keys)):
            raise DefinitionError(f"duplicate composite device keys {keys}")
        for d in self.devices:
            d.validate()
        entries = [OpEntry(0, PING_OP, None, None, SLACK_PREFIX,
                           op_signature(PING_OP, SLACK_PREFIX, None, None))]
        for d in self.devices:
            for op in d.ops:
                name = f"{d.key}.{op.key}"
                body = kernel_body(d, op)
                entries.append(OpEntry(len(entries), name, d, op, body,
                                       op_signature(name, body, d, op)))
        self.entries = tuple(entries)
        self._by_name = {e.name: e for e in self.entries}
        self.hash = hashlib.sha1(
            "".join(e.signature for e in self.entries).encode()).hexdigest()[:16]

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def get(self, name: str) -> OpEntry | None:
        return self._by_name.get(name)

    def by_index(self, index: int) -> OpEntry | None:
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def device(self, key: str) -> CompositeDevice | None:
        for d in self.devices:
            if d.key == key:
                return d
        return None

    def registration(self, session: str = "") -> dict:
        """What the monitor sends the server (``register_ops``)."""
        return {
            "type": "register_ops",
            "session": session,
            "hash": self.hash,
            "ops": {e.name: {"index": e.index, "sig": e.signature,
                             "args": list(e.arg_names),
                             "payload": list(e.payload_names)}
                    for e in self.entries},
        }


def status_text(status: int, message: str = "") -> str:
    base = OP_STATUS_TEXT.get(int(status), f"status {status}")
    return f"{base}: {message}" if message else base


def hazards(devices: Iterable[CompositeDevice], ctx: Context) -> list[tuple[CompositeDevice, str]]:
    """(device, what is hazardous) for every device that says so now.  Each
    device sees ``ctx`` with its own host state."""
    out = []
    for d in devices:
        if d.hazard is None:
            continue
        own = Context(ctx.config, ctx.params, ctx.frames, host_state=ctx.host_state.get(d.key)
                      if isinstance(ctx.host_state.get(d.key), Mapping) else {},
                      telemetry=ctx.telemetry, trust=ctx.trust)
        text = d.hazard_text(own)
        if text:
            out.append((d, text))
    return out


def compact_state(config: Mapping) -> dict:
    """The device-state JSON without its bookkeeping: DDS as [frequency,
    amplitude, v_pd, switch], DAC voltages, TTL levels -- what a run stamp
    records."""
    out = {"dds": {}, "dac": {}, "ttl": {}}
    for name, c in sorted((config.get("dds") or {}).items()):
        if isinstance(c, Mapping):
            out["dds"][name] = [c.get("frequency"), c.get("amplitude"), c.get("v_pd"),
                                c.get("sw_state")]
    for name, c in sorted((config.get("dac") or {}).items()):
        if isinstance(c, Mapping):
            out["dac"][name] = c.get("voltage")
    for name, c in sorted((config.get("ttl") or {}).items()):
        if isinstance(c, Mapping):
            out["ttl"][name] = c.get("ttl_state")
    return out


def literal_code(device: CompositeDevice, op: Op, args: Mapping) -> str:
    """The op's code with its arguments written in as literals, as it would
    read in an experiment's kernel (``self.`` for ``expt.``).  For "copy as
    code"; not what the monitor compiles (that one takes arguments)."""

    def substitute(match):
        spec = device.get_field(match.group(1))
        value = float(args[match.group(1)])
        if spec.kind in (KIND_INT, KIND_CHOICE):
            return repr(int(value))
        if spec.kind == KIND_BOOL:
            return "True" if value > 0.5 else "False"
        return repr(value)

    code = textwrap.dedent(op.code).strip("\n")
    code = _PLACEHOLDER.sub(substitute, code)
    return re.sub(r"\bexpt\.", "self.", code)


# --- scenes: several ops in order, with holds and a cleanup -----------------

@dataclass(frozen=True)
class Step:
    """One op of a scene, ``"device.op"``.  ``args`` gives values (SI) or
    ``"{field}"`` references to the scene's own fields; arguments not given
    take the device field's default."""

    op: str
    args: Any = None
    label: str = ""


@dataclass(frozen=True)
class Hold:
    """Wait ``seconds`` (a number or ``"{field}"``) before the next step.
    Runs on the monitor server, so closing the GUI does not end it early."""

    seconds: Any
    label: str = ""


@dataclass(frozen=True)
class Scene:
    """Ops run in order by the monitor server, one after the other finished.

    ``finally_`` always runs after the steps -- when they finish, when one
    fails, and when the scene is cancelled -- so a timed scene ("field on
    for two minutes") cannot leave the field on because a GUI closed.  A
    scene must either have a ``finally_`` or say in ``leaves_on`` what it
    deliberately leaves on (shown when it is started).
    """

    key: str
    title: str
    steps: tuple
    finally_: tuple = ()
    fields: tuple = ()
    leaves_on: str = ""
    confirm: str = ""
    doc: str = ""

    def get_field(self, name: str) -> Arg:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"scene {self.key}: no field {name!r}")

    def validate(self, table: OpTable) -> None:
        where = f"scene {self.key!r}"
        if not _KEY.match(self.key or ""):
            raise DefinitionError(f"{where}: key must be an identifier")
        if not self.steps:
            raise DefinitionError(f"{where}: no steps")
        if not self.finally_ and not self.leaves_on.strip():
            raise DefinitionError(f"{where}: needs a finally_ cleanup, or leaves_on saying "
                                  f"what it deliberately leaves on")
        field_names = {f.name for f in self.fields}

        def check_ref(value, what):
            if isinstance(value, str):
                m = _PLACEHOLDER.fullmatch(value.strip())
                if m is None or m.group(1) not in field_names:
                    raise DefinitionError(f"{where}: {what} {value!r} is not a '{{field}}' "
                                          f"reference to one of {sorted(field_names)}")

        for i, step in enumerate(tuple(self.steps) + tuple(self.finally_)):
            in_finally = i >= len(self.steps)
            if isinstance(step, Hold):
                if in_finally:
                    raise DefinitionError(f"{where}: finally_ cannot hold")
                check_ref(step.seconds, "hold")
                continue
            if not isinstance(step, Step):
                raise DefinitionError(f"{where}: unknown step {step!r}")
            entry = table.get(step.op)
            if entry is None or entry.op is None:
                raise DefinitionError(f"{where}: no op {step.op!r}")
            if entry.payload_names:
                raise DefinitionError(f"{where}: {step.op} needs a table payload")
            given = dict(step.args or {})
            unknown = set(given) - set(entry.arg_names)
            if unknown:
                raise DefinitionError(f"{where}: {step.op} has no args {sorted(unknown)}")
            for name in entry.arg_names:
                if name in given:
                    check_ref(given[name], f"{step.op} arg")
                elif entry.device.get_field(name).default is None:
                    raise DefinitionError(f"{where}: {step.op} arg {name!r} has no value "
                                          f"and no field default")

    def resolve(self, table: OpTable, values: Mapping, ctx: Context | None = None) -> dict:
        """The concrete request the server runs: op steps with signatures and
        SI arguments, holds in seconds."""

        def value_of(spec):
            if isinstance(spec, str):
                name = _PLACEHOLDER.fullmatch(spec.strip()).group(1)
                if values.get(name) is None:
                    raise DefinitionError(f"scene {self.key}: field {name!r} has no value")
                return float(values[name])
            return float(spec)

        def one(step):
            if isinstance(step, Hold):
                return {"hold": value_of(step.seconds), "label": step.label or "hold"}
            entry = table.get(step.op)
            given = dict(step.args or {})
            args = {}
            for name in entry.arg_names:
                if name in given:
                    args[name] = value_of(given[name])
                else:
                    args[name] = float(entry.device.get_field(name).default_value(ctx))
            return {"op": entry.name, "sig": entry.signature, "args": args, "payload": {},
                    "label": step.label or f"{entry.device.title}: {entry.op.label}"}

        return {"scene": self.key, "title": self.title,
                "steps": [one(s) for s in self.steps],
                "finally": [one(s) for s in self.finally_]}


def validate_scenes(scenes: Sequence[Scene], table: OpTable) -> None:
    keys = [s.key for s in scenes]
    if len(keys) != len(set(keys)):
        raise DefinitionError(f"duplicate scene keys {keys}")
    for s in scenes:
        s.validate(table)
