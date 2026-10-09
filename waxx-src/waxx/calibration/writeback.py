"""Write a calibrated value into the params source file, or refuse.

``find_assignment(key, params_cls)`` walks ``params_cls.__mro__`` most-derived
first. In each class's body (its source file, read with ``inspect``) it looks
for every statement that assigns ``self.<key>``. EVERY class is read: an
assignment of the key outside an ``__init__`` in any of them (a compute_*
method that compute_derived re-runs every shot, the class body) refuses the
key as derived. Otherwise the most-derived class that assigns it decides:

- exactly one plain single-line ``self.<key> = <numeric literal>`` inside
  ``__init__`` -> that line is the target;
- more than one -> refused, "ambiguous";
- an assignment that is not plain (augmented, tuple, chained, multi-line, an
  expression, or inside a method other than ``__init__``) -> refused, "not a
  plain assignment (derived or computed)";
- no class assigns it -> refused the same way.

A class in the MRO whose source cannot be read refuses the key (it may assign
it where this cannot see).

``apply`` takes ``<file>.lock`` first, then reads the file once and both parses
and edits from those bytes: it comments the old line in place (``# `` + the
line, same indentation) and inserts the new one directly below, tagged
``#<run_id>, <YYYY-MM-DD>`` and the note. Encoding and newline style are kept.
The original is copied to ``<file>.kcal-backup``; the new bytes go in by temp +
os.replace, only if the file still holds the bytes that were parsed. Then every
module from the target class's down to the run's params class is re-executed
(a reload), and both classes are instantiated and ``compute_derived()`` run:
the attribute must equal the written literal exactly. On any failure
(KeyboardInterrupt included) the original bytes go back; if that restore fails
as well, ``FILE LEFT MODIFIED, original at <backup>`` is printed and reported,
and the backup is kept. The backup is removed only once the file holds
verified new bytes or the original bytes again.

``revert`` re-activates the nearest commented assignment above the active line
the same way, tagged ``#reverted <date>``.

Every call returns a ``WritebackReport`` with the exact old and new lines,
whether it wrote or refused. Write-backs are left uncommitted in git, like hand
edits.
"""

from __future__ import annotations

import ast
import contextlib
import datetime
import inspect
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from waxx.calibration._lock import file_lock, replace_bytes
from waxx.calibration.precision import PrecisionError, format_value, parse_literal


LOCK_TIMEOUT_S = 10.0       # wait this long for <file>.lock, then refuse


class WritebackRefused(Exception):
    pass


@dataclass
class Target:
    key: str
    file: str
    cls_name: str           # the class's __qualname__
    module: str
    line_no: int            # 1-based
    line: str               # the line as in the file, without its newline
    indent: str
    literal: str
    comment: str
    method: str


@dataclass
class WritebackReport:
    ok: bool
    action: str                         # 'apply' | 'revert'
    key: str
    reason: str = ""
    file: Optional[str] = None
    line_no: Optional[int] = None       # 1-based line of the new active line
    old_line: Optional[str] = None      # the active line before
    commented_line: Optional[str] = None
    new_line: Optional[str] = None
    old_value: object = None
    new_value: object = None
    dry_run: bool = False
    written: bool = False
    left_modified: bool = False         # a failed write whose restore also failed
    backup: Optional[str] = None        # the original's copy, kept when left_modified

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        if not self.ok:
            return f"{self.action} {self.key}: REFUSED -- {self.reason}"
        head = (f"{self.action} {self.key}: {'would write' if self.dry_run else 'wrote'} "
                f"{self.file}:{self.line_no}")
        return "\n".join([head, f"  - {self.old_line.strip()}",
                          f"  + {self.commented_line.strip()}", f"  + {self.new_line.strip()}"])


def _active_re(key):
    return re.compile(rf"^(?P<indent>\s*)self\.{re.escape(key)}\s*=\s*(?P<literal>[^#]+?)\s*"
                      rf"(?P<comment>#.*)?$")


def _commented_re(key):
    return re.compile(rf"^(?P<indent>\s*)#\s*self\.{re.escape(key)}\s*=\s*(?P<literal>[^#]+?)\s*"
                      rf"(?P<comment>#.*)?$")


# ---- reading a source file -----------------------------------------------------------

@dataclass
class _Source:
    path: Path
    raw: bytes
    encoding: str
    lines: list = field(default_factory=list)     # with their line endings

    @classmethod
    def read(cls, path):
        path = Path(path)
        raw = path.read_bytes()
        enc = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError as e:
            raise WritebackRefused(f"{path} is not UTF-8 ({e}); not editing it")
        # keepends with only \r\n / \n / \r as separators (str.splitlines also
        # splits on \x0c, \x1c... which would change the file)
        lines = re.findall(r"[^\r\n]*(?:\r\n|\n|\r)|[^\r\n]+$", text)
        return cls(path, raw, enc, lines)

    def text(self):
        return "".join(self.lines)

    def encode(self, lines):
        return "".join(lines).encode(self.encoding)


def _strip_eol(s):
    return s.rstrip("\r\n")


def _eol(s, default="\n"):
    if s.endswith("\r\n"):
        return "\r\n"
    if s.endswith("\n"):
        return "\n"
    if s.endswith("\r"):
        return "\r"
    return default


def _class_node(tree, cls):
    """The ClassDef of ``cls`` (by qualname; by first line when names repeat)."""
    parts = cls.__qualname__.split(".")
    candidates = []

    def walk(body, depth):
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name == parts[depth]:
                if depth == len(parts) - 1:
                    candidates.append(node)
                else:
                    walk(node.body, depth + 1)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "<locals>" in parts:
                pass
    walk(tree.body, 0)
    if not candidates:
        return None
    first = getattr(cls, "__firstlineno__", None)
    for node in candidates:
        starts = [node.lineno] + [d.lineno for d in node.decorator_list]
        if first in starts:
            return node
    return candidates[-1]          # the definition Python kept


def _assigns_key(target, key):
    if isinstance(target, ast.Attribute):
        return (target.attr == key and isinstance(target.value, ast.Name)
                and target.value.id == "self")
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_assigns_key(t, key) for t in target.elts)
    if isinstance(target, ast.Starred):
        return _assigns_key(target.value, key)
    return False


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _scan_class(cls, key, sources=None):
    """(source, class node, [(node, enclosing function name)]) of every
    statement in ``cls``'s body that assigns self.<key>. ``sources`` maps a
    normalised path to an already-read _Source (the bytes held under the lock)."""
    path = inspect.getsourcefile(cls)
    if not path:
        raise WritebackRefused(f"cannot find the source file of {cls.__qualname__}")
    src = (sources or {}).get(_norm(path)) or _Source.read(path)
    tree = ast.parse(src.text(), filename=str(path))
    node = _class_node(tree, cls)
    if node is None:
        raise WritebackRefused(f"cannot find class {cls.__qualname__} in {path}")
    hits = []

    def visit(n, func):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)          # the innermost function owns it
                continue
            if isinstance(child, ast.ClassDef):
                continue                                  # a nested class's self is not ours
            if isinstance(child, ast.Assign) and any(_assigns_key(t, key) for t in child.targets):
                hits.append((child, func))
            elif isinstance(child, (ast.AugAssign, ast.AnnAssign)) and _assigns_key(child.target, key):
                hits.append((child, func))
            visit(child, func)
    visit(node, None)
    return src, node, hits


def find_assignment(key: str, params_cls, _sources=None) -> Target:
    """The one line a write-back of ``key`` may change. Raises WritebackRefused.

    Every class in the MRO is read: an assignment of the key anywhere but an
    ``__init__`` (a compute_* method that compute_derived re-runs, the class
    body) makes the key derived, whichever class it is in. The target is the
    one plain line in the ``__init__`` of the most-derived class that assigns it."""
    rx = _active_re(key)
    target = None
    for cls in params_cls.__mro__:
        if cls is object:
            continue
        try:
            src, node, hits = _scan_class(cls, key, _sources)
        except (TypeError, OSError) as e:
            raise WritebackRefused(f"cannot read the source of {cls.__qualname__} ({e}), in the "
                                   f"MRO of {params_cls.__qualname__}: cannot tell whether it "
                                   f"assigns {key}")
        for n, func in hits:
            if func != "__init__":
                raise WritebackRefused(
                    f"self.{key} at {src.path}:{n.lineno} is assigned in "
                    f"{cls.__qualname__}.{func or '<class body>'}, not __init__: a derived "
                    f"quantity (compute_derived would overwrite a written value)")
        if target is not None or not hits:
            continue
        where = f"{cls.__qualname__} ({src.path})"
        plain = []
        for n, func in hits:
            simple = (isinstance(n, ast.Assign) and len(n.targets) == 1
                      and isinstance(n.targets[0], ast.Attribute) and n.lineno == n.end_lineno)
            m = rx.match(_strip_eol(src.lines[n.lineno - 1])) if simple else None
            if m is None:
                raise WritebackRefused(f"self.{key} at {src.path}:{n.lineno} is not a plain "
                                       f"assignment (derived or computed)")
            plain.append((n, func, m))
        if len(plain) > 1:
            lines = ", ".join(str(n.lineno) for n, _, _ in plain)
            raise WritebackRefused(f"ambiguous: self.{key} is assigned on {len(plain)} active lines "
                                   f"in {where}.__init__ (lines {lines})")
        n, func, m = plain[0]
        literal = m.group("literal")
        try:
            parse_literal(literal)
        except PrecisionError:
            raise WritebackRefused(f"self.{key} = {literal} at {src.path}:{n.lineno} is not a "
                                   f"plain numeric literal (derived or computed)")
        target = Target(key, str(src.path), cls.__qualname__, cls.__module__, n.lineno,
                        _strip_eol(src.lines[n.lineno - 1]), m.group("indent"), literal,
                        m.group("comment") or "", func)
    if target is None:
        raise WritebackRefused(f"self.{key} is not a plain assignment (derived or computed): no "
                               f"class in the MRO of {params_cls.__qualname__} assigns it")
    return target


def history_lines(key: str, params_cls):
    """[(line_no, 'active' | 'commented', line)] for ``key`` in the target
    class's body, in file order (for ``kcal show``)."""
    t = find_assignment(key, params_cls)
    src = _Source.read(t.file)
    tree = ast.parse(src.text())
    cls = _class_by_qualname(params_cls, t.cls_name)
    node = _class_node(tree, cls)
    act, com = _active_re(key), _commented_re(key)
    out = []
    for i in range(node.lineno, node.end_lineno + 1):
        line = _strip_eol(src.lines[i - 1])
        if i == t.line_no:
            out.append((i, "active", line))
        elif com.match(line):
            out.append((i, "commented", line))
        elif act.match(line):
            out.append((i, "other", line))
    return t, out


def _class_by_qualname(params_cls, qualname):
    for c in params_cls.__mro__:
        if c.__qualname__ == qualname:
            return c
    raise WritebackRefused(f"{qualname} is not in the MRO of {params_cls.__qualname__}")


# ---- verify -------------------------------------------------------------------------------

def _chain_modules(target: Target, params_cls):
    """The modules from the target class's down to ``params_cls``'s, base
    first: re-executing them in this order rebuilds every class between them
    on the new source."""
    mro = list(params_cls.__mro__)
    idx = next((i for i, c in enumerate(mro)
                if c.__qualname__ == target.cls_name and c.__module__ == target.module), None)
    if idx is None:
        raise WritebackRefused(f"cannot verify: {target.cls_name} is not in the MRO of "
                               f"{params_cls.__qualname__}")
    mods = []
    for c in reversed(mro[:idx + 1]):
        if c.__module__ in mods:
            continue
        mod = sys.modules.get(c.__module__)
        try:
            src_file = inspect.getsourcefile(c)
        except TypeError:
            src_file = None
        if mod is None or not getattr(mod, "__file__", None) or not src_file or \
                _norm(mod.__file__) != _norm(src_file):
            raise WritebackRefused(f"cannot verify: module {c.__module__} (of "
                                   f"{c.__qualname__}) is not imported from its source file")
        mods.append(c.__module__)
    return mods


def _reexec(module_names):
    """Re-execute each module from its source file into its own module object
    (a reload: fresh code compiled from the file, so no stale bytecode cache
    can stand in for it)."""
    for name in module_names:
        mod = sys.modules[name]
        text = Path(mod.__file__).read_bytes().decode("utf-8-sig")
        exec(compile(text, mod.__file__, "exec"), mod.__dict__)


def _get(module_name, qualname):
    obj = sys.modules[module_name]
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _value_after(cls, key):
    """getattr(cls(), key) after compute_derived(), as a run would see it."""
    inst = cls()
    cd = getattr(inst, "compute_derived", None)
    if callable(cd):
        cd()
    return getattr(inst, key)


def _verify(target: Target, key, expected, params_cls):
    """None when the target class and the run's class both give ``expected``
    after a re-execution of their modules and compute_derived(), else why not."""
    try:
        mods = _chain_modules(target, params_cls)
        _reexec(mods)
        checks = [(target.module, target.cls_name)]
        if (params_cls.__module__, params_cls.__qualname__) not in checks:
            checks.append((params_cls.__module__, params_cls.__qualname__))
        for mod_name, qual in checks:
            got = _value_after(_get(mod_name, qual), key)
            if not _equal(got, expected):
                return (f"verification failed: after the write, {qual}().{key} is {got!r} "
                        f"(after compute_derived), not the written {expected!r}")
        return None
    except WritebackRefused as e:
        return str(e)
    except Exception as e:
        return f"verification failed: re-executing / instantiating raised {e!r}"


def _equal(got, expected) -> bool:
    try:
        if isinstance(expected, int):
            return bool(got == expected) and float(got).is_integer()
        return not isinstance(got, bool) and float(got) == expected
    except (TypeError, ValueError):
        return False


def _restore(path: Path, src: _Source, target, params_cls, backup: Path) -> bool:
    """Put the original bytes back. True once the file holds them again."""
    try:
        replace_bytes(path, src.raw)
        if path.read_bytes() != src.raw:
            raise OSError("the file does not hold the original bytes after the restore")
    except BaseException as e:
        print(f"!! [cal] FILE LEFT MODIFIED: {path} holds an UNVERIFIED write-back and could "
              f"not be restored ({e!r}); original at {backup}", flush=True)
        return False
    try:
        _reexec(_chain_modules(target, params_cls))       # module state back to the original
    except BaseException:
        pass
    return True


def _backup_path(path) -> Path:
    return Path(str(path) + ".kcal-backup")


def _backup_refusal(backup) -> str:
    return (f"a previous write-back was left unverified; restore or delete {backup} first "
            f"(it holds the file as it was before that write-back); nothing written")


def _write_verified(target: Target, src: _Source, new_lines, expected, key, params_cls):
    """Under the caller's lock: write ``new_lines`` if the file still holds
    ``src``'s bytes, verify, restore on any failure (also on KeyboardInterrupt).
    The original goes to ``<file>.kcal-backup`` first; that copy is removed only
    once the file holds verified new bytes or the original bytes again.
    Returns (refusal text or None, left_modified, backup path)."""
    path = Path(target.file)
    backup = _backup_path(path)
    if path.read_bytes() != src.raw:
        return (f"{path} changed since it was read; nothing written (try again)", False, None)
    try:
        with open(backup, "xb") as f:                 # never over an existing backup
            f.write(src.raw)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        return (_backup_refusal(backup), False, None)
    why, ok, touched, left_modified = None, False, False, False
    try:
        replace_bytes(path, src.encode(new_lines))
        why = _verify(target, key, expected, params_cls)
        ok = why is None
    except Exception as e:
        why = f"{type(e).__name__} while writing: {e}"
    finally:
        if not ok:
            # what the file holds decides, not how far the code got: a replace
            # that raised after it landed still has to be undone
            try:
                touched = path.read_bytes() != src.raw
            except OSError:
                touched = True
            if touched:
                left_modified = not _restore(path, src, target, params_cls, backup)
        if not left_modified:
            try:
                backup.unlink()
            except OSError:
                pass
    if why is not None:
        why += (f"; FILE LEFT MODIFIED, original at {backup}" if left_modified
                else "; the original file is back" if touched else "; the file was not changed")
    return why, left_modified, (str(backup) if left_modified else None)


# ---- apply / revert ---------------------------------------------------------------------

def _today():
    return datetime.date.today().isoformat()


def _one_line(note):
    return " ".join(str(note).split())


def _locked_edit(rep, key, params_cls, build, dry_run):
    """Find, edit and write ``key``'s line from one read of the file, all under
    its lock. ``build(t, src)`` returns (new line text without indent, expected
    value). Fills ``rep``; raises WritebackRefused."""
    t0 = find_assignment(key, params_cls)                 # which file to lock
    lock = contextlib.nullcontext() if dry_run else file_lock(t0.file, timeout=LOCK_TIMEOUT_S)
    with lock:
        if _backup_path(t0.file).exists():           # a FILE LEFT MODIFIED is unresolved
            raise WritebackRefused(_backup_refusal(_backup_path(t0.file)))
        src = _Source.read(t0.file)
        t = find_assignment(key, params_cls, _sources={_norm(t0.file): src})
        if _norm(t.file) != _norm(t0.file):
            raise WritebackRefused(f"the assignment of {key} moved to {t.file} while this ran; "
                                   f"nothing written (try again)")
        orig = src.lines[t.line_no - 1]
        if _strip_eol(orig) != t.line:
            raise WritebackRefused(f"{t.file}:{t.line_no} is not the line that was parsed; "
                                   f"nothing written")
        rep.file, rep.old_line = t.file, t.line
        rep.old_value = parse_literal(t.literal).value()
        body, expected = build(t, src)
        eol = _eol(orig, default=_eol(src.lines[0]) if src.lines else "\n")
        commented = t.indent + "# " + _strip_eol(orig).lstrip(" \t")
        new_line = t.indent + body
        rep.commented_line, rep.new_line = commented, new_line
        rep.new_value = expected
        rep.line_no = t.line_no + 1
        if dry_run:
            rep.ok = True
            return rep
        new_lines = list(src.lines)
        new_lines[t.line_no - 1: t.line_no] = [commented + eol, new_line + _eol(orig, "")]
        why, rep.left_modified, rep.backup = _write_verified(t, src, new_lines, expected, key,
                                                             params_cls)
        if why:
            raise WritebackRefused(why)
        rep.ok = rep.written = True
        return rep


def _check_result(key, result):
    if getattr(result, "key", key) != key:
        raise WritebackRefused(f"the result is for {result.key!r}, not {key!r}")
    if getattr(result, "deferred", False):
        raise WritebackRefused("the analysis was deferred; there is no value")
    if result.fit.get("ok", True) is False:
        raise WritebackRefused(f"the analysis failed ({result.fit.get('reason', '')})")
    if result.flags:
        raise WritebackRefused("the result is flagged: " +
                               "; ".join(f["text"] for f in result.flags))
    try:
        if not math.isfinite(float(result.value)):
            raise ValueError
    except (TypeError, ValueError):
        raise WritebackRefused(f"value {result.value!r} is not finite")
    if result.unc is not None:
        try:
            if not math.isfinite(float(result.unc)):
                raise ValueError
        except (TypeError, ValueError):
            raise WritebackRefused(f"uncertainty {result.unc!r} is not finite")


def apply(key: str, result, params_cls, *, dry_run: bool = False, note: str = "",
          allow_no_unc: bool = False, date: Optional[str] = None) -> WritebackReport:
    """Write ``result`` (a CalResult) for ``key``, or refuse. See the module doc."""
    rep = WritebackReport(False, "apply", key, dry_run=dry_run)

    def build(t, src):
        try:
            new_literal = format_value(result.value, result.unc, parse_literal(t.literal),
                                       allow_no_unc=allow_no_unc)
        except PrecisionError as e:
            raise WritebackRefused(str(e))
        tag = f"#{result.run_id}, {date or _today()}" + (f" {_one_line(note)}" if note else "")
        return f"self.{key} = {new_literal} {tag}", parse_literal(new_literal).value()

    try:
        _check_result(key, result)
        return _locked_edit(rep, key, params_cls, build, dry_run)
    except WritebackRefused as e:
        rep.reason = str(e)
        return rep
    except Exception as e:              # a bug or an I/O error: refuse, never half-write
        rep.reason = f"{type(e).__name__}: {e}"
        return rep


def revert(key: str, params_cls, *, dry_run: bool = False,
           date: Optional[str] = None) -> WritebackReport:
    """Re-activate the nearest commented assignment above the active line."""
    rep = WritebackReport(False, "revert", key, dry_run=dry_run)

    def build(t, src):
        node = _class_node(ast.parse(src.text()), _class_by_qualname(params_cls, t.cls_name))
        com = _commented_re(key)
        for i in range(t.line_no - 1, node.lineno, -1):
            m = com.match(_strip_eol(src.lines[i - 1]))
            if m:
                break
        else:
            raise WritebackRefused(f"no commented assignment of self.{key} above line "
                                   f"{t.line_no} in {t.file}: nothing to revert to")
        literal = m.group("literal")
        try:
            lit = parse_literal(literal)
        except PrecisionError:
            raise WritebackRefused(f"the line to revert to ({src.path}:{i}) is not a plain "
                                   f"numeric literal: {literal}")
        was = (m.group("comment") or "").lstrip("#").strip()
        tag = f"#reverted {date or _today()}" + (f"; was #{was}" if was else "")
        return f"self.{key} = {literal} {tag}", lit.value()

    try:
        return _locked_edit(rep, key, params_cls, build, dry_run)
    except WritebackRefused as e:
        rep.reason = str(e)
        return rep
    except Exception as e:
        rep.reason = f"{type(e).__name__}: {e}"
        return rep


def current_value(key: str, params_cls):
    """The value of the target line's literal (None when there is no target)."""
    try:
        return parse_literal(find_assignment(key, params_cls).literal).value()
    except (WritebackRefused, PrecisionError):
        return None
