"""CalibrationConfig: what a machine tells the calibration framework.

waxx assumes nothing about where things are. The machine (kexp:
``kexp.config.calibration.CALIBRATION_CONFIG``) fills one of these and hands it
to its experiments (``Expt.calibration_config``); the ``kcal`` CLI finds it from
``--config module:attr`` or the ``WAXX_CALIBRATION_CONFIG`` environment variable
(same ``module:attr`` form).
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

ENV_VAR = "WAXX_CALIBRATION_CONFIG"
DEFAULT_BUDGET_S = 30.0


def import_object(spec: str):
    """``pkg.mod:attr.sub`` or ``pkg.mod.attr`` -> the object."""
    if ":" in spec:
        mod_name, attr = spec.split(":", 1)
    else:
        mod_name, _, attr = spec.rpartition(".")
    if not mod_name or not attr:
        raise ValueError(f"{spec!r} is not 'module:attr'")
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


@dataclass
class CalibrationConfig:
    # where the ledger lives: a path, or a zero-argument callable returning one
    # (so a machine can resolve its data root lazily, at first use)
    ledger_dir: Union[None, str, Path, Callable[[], Any]] = None
    # the policy table: a mapping, a module path ('pkg.mod' -> its POLICY), or None
    policy: Union[None, Mapping, str] = None
    # modules searched (in order) for analyses by name
    registry_modules: Sequence[str] = field(default_factory=tuple)
    # the params class write-backs go to when a record names none: a class or
    # 'module:Class'
    params_class: Union[None, type, str] = None
    # run_id -> atomdata-like object; None = waxa.atomdata(run_id, roi_id='auto', lite=False)
    loader: Optional[Callable[[int], Any]] = None
    # seconds an analysis may take inside a run's end() before it is deferred
    budget_s: float = DEFAULT_BUDGET_S

    def get_ledger_dir(self) -> Path:
        d = self.ledger_dir() if callable(self.ledger_dir) else self.ledger_dir
        if not d:
            raise RuntimeError("this CalibrationConfig names no ledger directory")
        return Path(d)

    def get_ledger(self):
        from waxx.calibration.ledger import Ledger
        return Ledger(self.get_ledger_dir())

    def get_policy(self) -> dict:
        from waxx.calibration.policy import load_policy
        return load_policy(self.policy)

    def get_params_class(self, override: Union[None, type, str] = None) -> type:
        spec = override if override is not None else self.params_class
        if spec is None:
            raise RuntimeError("no params class: the CalibrationConfig names none")
        return import_object(spec) if isinstance(spec, str) else spec

    def load_run(self, run_id: int):
        if self.loader is not None:
            return self.loader(int(run_id))
        from waxa import atomdata
        return atomdata(int(run_id), roi_id="auto", lite=False)


def load_config(spec: Optional[str] = None) -> CalibrationConfig:
    """The config named by ``spec`` ('module:attr'), else by $WAXX_CALIBRATION_CONFIG."""
    spec = spec or os.environ.get(ENV_VAR, "").strip()
    if not spec:
        raise RuntimeError(f"no calibration config: pass --config module:attr or set {ENV_VAR} "
                           f"(the K machine: kexp.config.calibration:CALIBRATION_CONFIG)")
    cfg = import_object(spec)
    if not isinstance(cfg, CalibrationConfig):
        raise TypeError(f"{spec} is a {type(cfg).__name__}, not a CalibrationConfig")
    return cfg
