"""How many digits a written-back value gets, and in which format.

The rule (user decision 2026-10-09):

1. Round the value to the second significant digit of its uncertainty.
2. The existing line's LITERAL significant figures are a floor: the rule may only
   add digits, never remove them. Trailing zeros count (``150.0`` = 4,
   ``3.7785e-06`` = 5, ``250.e-6`` = 3); an all-zero literal counts as 1.
3. Never more digits than Python's shortest round-trip repr of the value (those
   are all the digits the float has).
4. The format follows the existing literal: scientific stays scientific (a
   normalised mantissa ``d.ddd`` stays normalised; any other mantissa, e.g.
   ``250.e-6``, keeps its exponent; the exponent keeps its sign style, zero
   padding and ``e``/``E``), plain decimal stays plain, and an int stays an int.
5. An int literal takes only an integral value, written exactly (a count has
   nothing to round); a non-integral value for it is refused.
6. No uncertainty: the full repr when ``allow_no_unc``, else refused.

Rounding is half-to-even on the value's repr. Nothing here nudges a value:
the result is the value rounded at one decimal position, and the position is
chosen by the rule above alone.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Optional

NUM_RE = (r"(?P<sign>[-+]?)(?P<int>\d*)(?:(?P<dot>\.)(?P<frac>\d*))?"
          r"(?:(?P<echar>[eE])(?P<esign>[-+]?)(?P<edigits>\d+))?")
_NUM = re.compile(rf"^{NUM_RE}$")
_WRAPPED = re.compile(r"^(?P<pre>(?P<name>(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)\(\s*)"
                      r"(?P<num>[^()]*?)(?P<post>\s*\))$")
# wrappers a write-back may keep: the last dotted name of the callable
INT_WRAPPERS = ("int", "int32", "int64")
FLOAT_WRAPPERS = ("float", "float64")
KINDS = ("int", "decimal", "sci")


class PrecisionError(ValueError):
    """The value cannot be written into this literal under the rule."""


@dataclass(frozen=True)
class Literal:
    """A numeric literal as written on a params line."""
    text: str           # the whole literal, e.g. 'np.int32(50)' or '3.7785e-06'
    prefix: str         # 'np.int32(' or ''
    number: str         # '50'
    suffix: str         # ')' or ''
    kind: str           # 'int' | 'decimal' | 'sci'
    sign: str
    int_part: str
    dot: str
    frac: str
    echar: str
    esign: str
    edigits: str

    @property
    def sig_figs(self) -> int:
        digits = (self.int_part + self.frac).lstrip("0")
        return max(1, len(digits))

    @property
    def exponent(self) -> int:
        if not self.edigits:
            return 0
        return -int(self.edigits) if self.esign == "-" else int(self.edigits)

    @property
    def normalized_mantissa(self) -> bool:
        """``d.ddd`` (one nonzero digit before the point)."""
        return len(self.int_part) == 1 and self.int_part != "0"

    def value(self):
        """The Python value the literal evaluates to (int or float)."""
        if self.kind == "int":
            return int(self.number)
        return float(self.number)

    def rebuild(self, number: str) -> str:
        return f"{self.prefix}{number}{self.suffix}"


def parse_literal(text: str) -> Literal:
    """Parse a plain numeric literal, optionally inside one of ``INT_WRAPPERS`` /
    ``FLOAT_WRAPPERS`` (``np.int32(50)``). Anything else -- an expression, a
    name, a string -- raises PrecisionError."""
    t = text.strip()
    prefix = suffix = ""
    wrapper = None
    w = _WRAPPED.match(t)
    if w:
        wrapper = w.group("name").split(".")[-1]
        if wrapper not in INT_WRAPPERS + FLOAT_WRAPPERS:
            raise PrecisionError(f"{t!r} is not a plain numeric literal "
                                 f"(wrapper {w.group('name')!r} is not one this helper keeps)")
        prefix, t, suffix = w.group("pre"), w.group("num").strip(), w.group("post")
    m = _NUM.match(t)
    if not m or not (m.group("int") or m.group("frac")):
        raise PrecisionError(f"{text.strip()!r} is not a plain numeric literal")
    if m.group("echar"):
        kind = "sci"
    elif m.group("dot"):
        kind = "decimal"
    else:
        kind = "int"
    if wrapper in INT_WRAPPERS:
        if kind != "int":
            raise PrecisionError(f"{text.strip()!r}: an int wrapper around a float literal")
    elif wrapper in FLOAT_WRAPPERS and kind == "int":
        kind = "decimal"     # float(5): written back as a decimal inside the wrapper
    return Literal(text.strip(), prefix, t, suffix, kind, m.group("sign"), m.group("int") or "",
                   m.group("dot") or "", m.group("frac") or "", m.group("echar") or "",
                   m.group("esign") or "", m.group("edigits") or "")


def _check_number(x, what) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        raise PrecisionError(f"{what} {x!r} is not a number")
    if not math.isfinite(f):
        raise PrecisionError(f"{what} is not finite ({f!r})")
    return f


def format_value(value, unc, old_line_literal, kind: Optional[str] = None, *,
                 allow_no_unc: bool = False) -> str:
    """The new literal for ``value`` (± ``unc``) on a line whose literal is now
    ``old_line_literal`` (a string, wrapper included). ``kind`` overrides the
    literal's own kind ('int', 'decimal' or 'sci'). Raises PrecisionError when the
    rule cannot be met."""
    lit = parse_literal(old_line_literal) if isinstance(old_line_literal, str) else old_line_literal
    kind = kind or lit.kind
    if kind not in KINDS:
        raise PrecisionError(f"kind must be one of {KINDS}, got {kind!r}")
    v = _check_number(value, "value")
    if unc is None:
        if not allow_no_unc:
            raise PrecisionError("no uncertainty: refused (allow_no_unc=True writes the full repr)")
        u = None
    else:
        u = _check_number(unc, "uncertainty")
        if u < 0:
            raise PrecisionError(f"uncertainty is negative ({u!r})")

    if kind == "int":
        if not float(v).is_integer():
            raise PrecisionError(f"value {v!r} is not an integer, and the line holds an int "
                                 f"literal ({lit.text}); refused")
        return lit.rebuild(str(int(v)))

    d = Decimal(repr(float(v)))
    if d == 0:
        p = _position_for_zero(u, lit, kind)
    else:
        e_val = d.adjusted()                                  # position of the first digit
        p_repr = d.normalize().as_tuple().exponent            # position of the last repr digit
        p_floor = e_val - (lit.sig_figs - 1)
        if u is None or u == 0:
            p_rule = p_repr                                   # all the digits there are
        else:
            p_rule = Decimal(repr(u)).adjusted() - 1          # 2nd significant digit of unc
        p = max(min(p_rule, p_floor), p_repr)
    r = d.quantize(Decimal(1).scaleb(p), rounding=ROUND_HALF_EVEN)
    if kind == "decimal":
        return lit.rebuild(_format_decimal(r, p))
    return lit.rebuild(_format_sci(r, p, lit))


def _position_for_zero(u, lit, kind):
    """A value of exactly zero: the uncertainty's resolution, else one decimal."""
    if u is not None and u > 0:
        return Decimal(repr(u)).adjusted() - 1
    if kind == "sci":
        return lit.exponent - 1
    return -1


def _format_decimal(r: Decimal, p: int) -> str:
    decimals = max(0, -p)
    s = format(r, f".{decimals}f")
    if decimals == 0:
        s += "."                       # stays a float literal
    return s


def _format_sci(r: Decimal, p: int, lit: Literal) -> str:
    if lit.kind == "sci" and not lit.normalized_mantissa:
        E = lit.exponent               # e.g. 250.e-6: keep the exponent (it names the unit)
    elif r == 0:
        E = lit.exponent if lit.kind == "sci" else 0
    else:
        E = r.adjusted()               # normalised mantissa d.ddd
    m = r.scaleb(-E)
    decimals = max(0, E - p)
    ms = format(m, f".{decimals}f")
    if decimals == 0 and (lit.dot or lit.kind != "sci"):
        ms += "."
    echar = lit.echar or "e"
    width = max(1, len(lit.edigits))   # 'e-06' / 'e-10' -> two digits, 'e-6' -> as needed
    esign = "-" if E < 0 else ("+" if lit.esign == "+" else "")
    return f"{ms}{echar}{esign}{str(abs(E)).zfill(width)}"
