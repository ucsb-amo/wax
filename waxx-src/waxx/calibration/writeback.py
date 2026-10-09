"""Write a calibrated value into the params source file, or refuse.

``find_assignment(key, params_cls)`` walks ``params_cls.__mro__`` most-derived
first. In each class's body (its source file, read with ``inspect``) it looks
for every statement that assigns ``self.<key>``. The first class that has any
decides:

- exactly one plain single-line ``self.<key> = <numeric literal>`` inside
  ``__init__`` -> that line is the target;
- more than one -> refused, "ambiguous";
- an assignment that is not plain (augmented, tuple, chained, multi-line, an
  expression, or inside a method other than ``__init__``) -> refused, "not a
  plain assignment (derived or computed)";
- no class assigns it -> refused the same way.

A class earlier in the MRO whose source cannot be read stops the walk (it may
assign the key where this cannot see).

``apply`` comments the old line in place (``# `` + the line, same indentation)
and inserts the new one directly below, tagged ``#<run_id>, <YYYY-MM-DD>`` and
the note. The file keeps its encoding and newline style; it is written to a
temp file and os.replace'd under ``<file>.lock``, and only if it has not changed
since it was read. Then the module is re-executed (a reload) and the class
instantiated: the attribute must equal the written literal exactly, else the
original bytes go back and the write-back is refused.

``revert`` re-activates the nearest commented assignment above the active line
the same way, tagged ``#reverted <date>``.

Every call returns a ``WritebackReport`` with the exact old and new lines,
whether it wrote or refused. Write-backs are left uncommitted in git, like hand
edits.
"""

from __future__ import annotations

import ast
import datetime
import inspect
import math
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


def _scan_class(cls, key):
    """(source, class node, [(node, enclosing function name)]) of every
    statement in ``cls``'s body that assigns self.<key>."""
    path = inspect.getsourcefile(cls)
    if not path:
        raise WritebackRefused(f"cannot find the source file of {cls.__qualname__}")
    src = _Source.read(path)
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


def find_assignment(key: str, params_cls) -> Target:
    """The one line a write-back of ``key`` may change. Raises WritebackRefused."""
    rx = _active_re(key)
    for cls in params_cls.__mro__:
        if cls is object:
            continue
        try:
            src, node, hits = _scan_class(cls, key)
        except (TypeError, OSError) as e:
            raise WritebackRefused(f"cannot read the source of {cls.__qualname__} ({e}), which "
                                   f"comes before any assignment of {key} in the MRO of "
                                   f"{params_cls.__qualname__}")
        if not hits:
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
            lines = ", ".join(f"{n.lineno} (in {f or 'the class body'})" for n, f, _ in plain)
            raise WritebackRefused(f"ambiguous: self.{key} is assigned on {len(plain)} active lines "
                                   f"in {where} (lines {lines})")
        n, func, m = plain[0]
        if func != "__init__":
            raise WritebackRefused(f"self.{key} at {src.path}:{n.lineno} is assigned in "
                                   f"{func or 'the class body'}, not __init__: a derived quantity")
        literal = m.group("literal")
        try:
            parse_literal(literal)
        except PrecisionError:
            raise WritebackRefused(f"self.{key} = {literal} at {src.path}:{n.lineno} is not a "
                                   f"plain numeric literal (derived or computed)")
        return Target(key, str(src.path), cls.__qualname__, cls.__module__, n.lineno,
                      _strip_eol(src.lines[n.lineno - 1]), m.group("indent"), literal,
                      m.group("comment") or "", func)
    raise WritebackRefused(f"self.{key} is not a plain assignment (derived or computed): no class "
                           f"in the MRO of {params_cls.__qualname__} assigns it")


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

def _reexec_and_get(target: Target, key: str):
    """Re-execute the target's module from its source (a reload: same module
    object, fresh code -- compiled from the file, so no stale bytecode cache
    can stand in for it), instantiate the class, return the attribute."""
    mod = sys.modules.get(target.module)
    if mod is None or not getattr(mod, "__file__", None) or \
            Path(mod.__file__).resolve() != Path(target.file).resolve():
        raise WritebackRefused(f"cannot verify: module {target.module} is not imported from "
                               f"{target.file}")
    text = Path(target.file).read_bytes().decode("utf-8-sig")
    code = compile(text, target.file, "exec")
    exec(code, mod.__dict__)
    obj = mod
    for part in target.cls_name.split("."):
        obj = getattr(obj, part)
    inst = obj()
    return getattr(inst, key)


def _equal(got, expected) -> bool:
    try:
        if isinstance(expected, int):
            return bool(got == expected) and float(got).is_integer()
        return not isinstance(got, bool) and float(got) == expected
    except (TypeError, ValueError):
        return False


def _write_verified(target: Target, src: _Source, new_lines, expected, key):
    """Write ``new_lines`` if the file is unchanged since ``src`` was read,
    verify, restore on failure. Returns None or the refusal text."""
    with file_lock(target.file, timeout=LOCK_TIMEOUT_S):
        now = Path(target.file).read_bytes()
        if now != src.raw:
            return f"{target.file} changed since it was read; nothing written (try again)"
        replace_bytes(target.file, src.encode(new_lines))
        why = None
        try:
            got = _reexec_and_get(target, key)
            if not _equal(got, expected):
                why = (f"verification failed: after the write, {target.cls_name}().{key} is "
                       f"{got!r}, not the written {expected!r}")
        except Exception as e:
            why = f"verification failed: re-executing {target.module} raised {e!r}"
        if why is None:
            return None
        replace_bytes(target.file, src.raw)
        try:
            _reexec_and_get(target, key)
        except Exception as e:
            why += f"; the original file is back, but re-executing it raised {e!r}"
        else:
            why += "; the original file is back"
        return why


# ---- apply / revert ---------------------------------------------------------------------

def _today():
    return datetime.date.today().isoformat()


def _one_line(note):
    return " ".join(str(note).split())


def apply(key: str, result, params_cls, *, dry_run: bool = False, note: str = "",
          allow_no_unc: bool = False, date: Optional[str] = None) -> WritebackReport:
    """Write ``result`` (a CalResult) for ``key``, or refuse. See the module doc."""
    rep = WritebackReport(False, "apply", key, dry_run=dry_run)
    try:
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
        t = find_assignment(key, params_cls)
        rep.file, rep.old_line = t.file, t.line
        old = parse_literal(t.literal)
        rep.old_value = old.value()
        try:
            new_literal = format_value(result.value, result.unc, old, allow_no_unc=allow_no_unc)
        except PrecisionError as e:
            raise WritebackRefused(str(e))
        tag = f"#{result.run_id}, {date or _today()}"
        if note:
            tag += f" {_one_line(note)}"
        src = _Source.read(t.file)
        orig = src.lines[t.line_no - 1]
        eol = _eol(orig, default=_eol(src.lines[0]) if src.lines else "\n")
        commented = t.indent + "# " + _strip_eol(orig).lstrip(" \t")
        new_line = f"{t.indent}self.{key} = {new_literal} {tag}"
        rep.commented_line, rep.new_line = commented, new_line
        rep.new_value = parse_literal(new_literal).value()
        rep.line_no = t.line_no + 1
        if dry_run:
            rep.ok = True
            return rep
        new_lines = list(src.lines)
        new_lines[t.line_no - 1: t.line_no] = [commented + eol, new_line + _eol(orig, "")]
        why = _write_verified(t, src, new_lines, rep.new_value, key)
        if why:
            raise WritebackRefused(why)
        rep.ok = rep.written = True
        return rep
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
    try:
        t = find_assignment(key, params_cls)
        rep.file, rep.old_line = t.file, t.line
        rep.old_value = parse_literal(t.literal).value()
        src = _Source.read(t.file)
        tree = ast.parse(src.text())
        node = _class_node(tree, _class_by_qualname(params_cls, t.cls_name))
        com = _commented_re(key)
        prev = None
        for i in range(t.line_no - 1, node.lineno, -1):
            m = com.match(_strip_eol(src.lines[i - 1]))
            if m:
                prev = (i, m)
                break
        if prev is None:
            raise WritebackRefused(f"no commented assignment of self.{key} above line "
                                   f"{t.line_no} in {t.file}: nothing to revert to")
        i, m = prev
        literal = m.group("literal")
        try:
            lit = parse_literal(literal)
        except PrecisionError:
            raise WritebackRefused(f"the line to revert to ({src.path}:{i}) is not a plain "
                                   f"numeric literal: {literal}")
        was = (m.group("comment") or "").lstrip("#").strip()
        tag = f"#reverted {date or _today()}" + (f"; was #{was}" if was else "")
        orig = src.lines[t.line_no - 1]
        eol = _eol(orig, default="\n")
        commented = t.indent + "# " + _strip_eol(orig).lstrip(" \t")
        new_line = f"{t.indent}self.{key} = {literal} {tag}"
        rep.commented_line, rep.new_line = commented, new_line
        rep.new_value = lit.value()
        rep.line_no = t.line_no + 1
        if dry_run:
            rep.ok = True
            return rep
        new_lines = list(src.lines)
        new_lines[t.line_no - 1: t.line_no] = [commented + eol, new_line + _eol(orig, "")]
        why = _write_verified(t, src, new_lines, rep.new_value, key)
        if why:
            raise WritebackRefused(why)
        rep.ok = rep.written = True
        return rep
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
