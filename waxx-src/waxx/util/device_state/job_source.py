"""What the run queue knows about a job's experiment file, read from its
source on the PC that holds it -- never imported, never run.

Since 2026-11-15 the PC that submits a job runs it (``kq launch``,
:mod:`~waxx.util.device_state.launcher`), and the monitor server never opens a
job's file: the submitting PC computes these at submit and the job's launcher
again when it attaches, and sends them.  Import-light (standard library only),
so ``kq`` can use it without the server's modules.

* :func:`file_sha256` -- the file's SHA-256 (the launch-time drift check).
* :func:`describe_source` -- the experiment class and the keys of its
  ``self.calibrates('<key>', ...)`` calls.
* :func:`cold_reason` -- why the job must never warm-launch.
* :func:`source_info` -- all three as the submit / attach fields.

Machine-agnostic: nothing here knows the K machine.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

#: Names that make a job cold when its prepare / build / __init__ calls them
#: (hardware or shared lab state touched before GO).
WARM_DENY_NAMES = ("_announce_to_monitor", "announce_run", "withdraw_run", "picomotor",
                   "stage_dev", "objective_stage")

#: The methods whose calls decide a job's warm eligibility (they run before GO).
_PRE_GO_METHODS = ("prepare", "build", "__init__", "finish_prepare")


def file_sha256(path) -> str:
    """The file's SHA-256 (hex).  OSError when it cannot be read."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def sha256_or_none(path) -> str | None:
    """:func:`file_sha256`, or None when the file cannot be read."""
    try:
        return file_sha256(path)
    except OSError:
        return None


def describe_source(path) -> tuple[str, list]:
    """``(expt_class, calibrates_declared)`` from an experiment file's source,
    read with :mod:`ast` (never imported): the single public class that
    subclasses ``EnvExperiment`` or defines ``scan_kernel`` ("" when there is
    none or more than one), and the first argument of every
    ``self.calibrates('<key>', ...)`` call with a string key, in order, once
    each.  ("", []) when the file cannot be read or parsed."""
    try:
        return _read_source(path)
    except Exception as exc:                          # noqa: BLE001
        # a pathological file (RecursionError / MemoryError from ast, ...):
        # the listing goes without; the job is submitted all the same
        log.warning("Run queue: could not read %s's class and calibrations (%r); listed "
                    "without them.", path, exc)
        return "", []


def _read_source(path) -> tuple[str, list]:
    import ast  # noqa: PLC0415
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return "", []
    classes = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        bases = [ast.unparse(b).split(".")[-1] for b in node.bases]
        scan = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == "scan_kernel" for n in node.body)
        if "EnvExperiment" in bases or scan:
            classes.append(node.name)
    keys: list = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "calibrates" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str) and node.args[0].value not in keys):
            keys.append(node.args[0].value)
    return (classes[0] if len(classes) == 1 else ""), keys


def cold_reason(path, deny_names=WARM_DENY_NAMES) -> str:
    """Why a job of this experiment file must never warm-launch ("" when it
    may), read with :mod:`ast` (never imported): a class body that sets
    ``warm_launch = False``, or a call in ``prepare`` / ``build`` /
    ``__init__`` / ``finish_prepare`` whose dotted name contains one of
    ``deny_names`` (hardware or shared lab state touched before GO).  A file
    that cannot be read or parsed is cold (nothing shows it is safe)."""
    import ast  # noqa: PLC0415
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:                          # noqa: BLE001
        return f"its source could not be read for the warm check ({type(exc).__name__})"
    deny = [d.lower() for d in deny_names if d]
    try:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if (isinstance(item, (ast.Assign, ast.AnnAssign))
                        and isinstance(getattr(item, "value", None), ast.Constant)
                        and item.value.value is False
                        and any(isinstance(t, ast.Name) and t.id == "warm_launch"
                                for t in (item.targets if isinstance(item, ast.Assign)
                                          else [item.target]))):
                    return f"class {node.name} sets warm_launch = False"
                if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name in _PRE_GO_METHODS):
                    for sub in ast.walk(item):
                        if not isinstance(sub, ast.Call):
                            continue
                        name = ast.unparse(sub.func).lower()
                        hit = next((d for d in deny if d in name), None)
                        if hit:
                            return (f"{node.name}.{item.name} calls {ast.unparse(sub.func)} "
                                    f"(touches the lab before GO)")
    except Exception as exc:                          # noqa: BLE001
        return f"its source could not be checked for warm start ({type(exc).__name__})"
    return ""


def source_info(path, deny_names=WARM_DENY_NAMES) -> dict:
    """``{"sha256", "expt_class", "calibrates_declared", "cold_why"}`` for the
    file at ``path`` on this PC (``sha256`` None when it cannot be read): the
    fields a submit and a launcher's attach carry."""
    expt_class, calibrates = describe_source(path)
    return {"sha256": sha256_or_none(path), "expt_class": expt_class,
            "calibrates_declared": list(calibrates), "cold_why": cold_reason(path, deny_names)}
