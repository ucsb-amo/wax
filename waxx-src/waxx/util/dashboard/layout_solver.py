"""Auto-arrange: choose a dock layout for the dashboard that fits a display.

Pure Python (no Qt), so it runs in tests and from the command line::

    python -m waxx.util.dashboard.layout_solver --layout-module kexp.util.dashboard.dashboard_layout \\
        --kind server --display 1920x1080

Model
-----
The dashboard's central widget is empty, so the four Qt dock areas share the
window: top and bottom are full-width bands (Qt's default corners), left and
right sit side by side between them.  Docks in one area stack along it (top /
bottom: left to right; left / right: top to bottom).  A *slot* is one dock or
one tab stack; only its raised tab is on screen.

Every panel has a minimum and a preferred size (the whole dock, header
included) and a tier:

* ``primary``  -- on screen whenever at all possible; spare space goes here first
* ``frequent`` -- on screen when it fits
* ``status``   -- glanced at (status readouts): may sit behind another tab
* ``rare``     -- the first to be stacked

The cost of a layout is a sum over panels: the tier's hide cost
(:data:`TIERS`) for a panel behind another tab, and the tier's space weight
(:data:`SPACE_WEIGHTS`) x ``1 - s`` for a panel on screen, where ``s`` is the
fraction of its preferred size it gets (width and height fractions
multiplied).  Spare space beyond the preferred sizes goes by space weight, so
primary panels grow first.  A tab stack mixing panel families, or holding
more than :data:`MAX_EASY_TABS` tabs, costs a little more, so status panels of
one kind share a stack.  A layout whose minimum sizes do not fit the display
costs more than any layout that fits; among those that do not fit, less
overflow is cheaper.

The solver is an iterated local search from several starting layouts: hill
climbing over single moves (one panel to another slot or to a new slot in any
area, a whole slot to another area, two slots merged), then a few random
moves away from the best layout and a new climb, until an evaluation budget
is spent.  It returns the cheapest layout found, with the dock sizes to apply
and a breakdown of the cost.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

AREAS: tuple[str, ...] = ("top", "left", "right", "bottom")
#: Cost of a panel behind another tab, by tier.
TIERS: dict[str, float] = {"primary": 100.0, "frequent": 12.0, "status": 3.0, "rare": 1.0}
#: Cost of a panel on screen with none of its preferred size, by tier; also
#: how spare space is shared out.  Much smaller than TIERS: showing a panel at
#: its minimum beats hiding it, and a primary panel gives up some of its
#: preferred size before a status panel goes behind a tab.
SPACE_WEIGHTS: dict[str, float] = {"primary": 8.0, "frequent": 3.0, "status": 1.5, "rare": 1.0}

#: Height of the tab bar under a tab stack (px).
TAB_BAR_H = 28
#: Splitter handle between two docks (px).
SEP = 6
#: Cost of each family in a tab stack beyond the first.
FAMILY_MIX = 1.0
#: Cost of each tab beyond MAX_EASY_TABS in one stack.
CROWDED_TAB = 0.5
MAX_EASY_TABS = 4
#: Tie-breaks for displays where everything fits: space weight x this x the
#: fraction of 1.5x its preferred size a panel lacks (so spare room goes to
#: primary panels), and this per extra area a family's on-screen panels span.
COMFORT = 0.1
COHESION = 0.2
#: Cost of a layout that does not fit, plus 1 per pixel of overflow.
OVERFLOW_BASE = 1000.0
#: Window chrome subtracted from a display size to get the dock region:
#: taskbar, title bar, menu bar, toolbar, status bar (Windows 10, 100 % scale).
DEFAULT_CHROME = (16, 40 + 31 + 22 + 32 + 24)
#: Kicks in a row that find nothing better before the search stops.
ILS_PATIENCE = 150


@dataclass(frozen=True)
class PanelInfo:
    """What the solver knows about one panel."""

    id: str
    label: str = ""
    min_size: tuple[int, int] = (320, 200)
    pref_size: tuple[int, int] = (0, 0)      # 0 -> the minimum
    tier: str = "status"
    family: str = ""
    areas: tuple[str, ...] = AREAS           # where it may go

    @property
    def weight(self) -> float:
        return TIERS.get(self.tier, TIERS["status"])

    @property
    def space_weight(self) -> float:
        return SPACE_WEIGHTS.get(self.tier, SPACE_WEIGHTS["status"])

    @property
    def pref(self) -> tuple[int, int]:
        return (max(self.pref_size[0], self.min_size[0]), max(self.pref_size[1], self.min_size[1]))


@dataclass
class SolvedSlot:
    """One dock (or tab stack) of the solution, with its rectangle in the dock region."""

    area: str
    members: list[str]                        # raised panel first
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    @property
    def raised(self) -> str:
        return self.members[0]

    @property
    def stacked(self) -> bool:
        return len(self.members) > 1

    @property
    def content_h(self) -> int:
        return self.h - (TAB_BAR_H if self.stacked else 0)


@dataclass
class LayoutSolution:
    width: int
    height: int
    slots: list[SolvedSlot]
    cost: float
    terms: dict[str, float]
    overflow: tuple[int, int]                 # (px too wide, px too tall); (0, 0) = fits
    satisfaction: dict[str, float]            # panel id -> fraction of preferred size (0 = hidden)
    labels: dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0
    evaluations: int = 0

    @property
    def fits(self) -> bool:
        return self.overflow == (0, 0)

    def placements(self) -> dict[str, dict]:
        """``{panel_id: {placement, dock_area, tab_group}}`` in dock order (raised member first).

        The same shape as the layout module's ``*_PLACEMENT`` tables.
        """
        out: dict[str, dict] = {}
        for s in self.slots:
            for pid in s.members:
                if s.stacked:
                    out[pid] = {"placement": "tab", "dock_area": s.area, "tab_group": f"auto:{s.raised}"}
                else:
                    out[pid] = {"placement": "dock", "dock_area": s.area, "tab_group": f"auto:{pid}"}
        return out

    def band_sizes(self) -> dict[str, int]:
        """Extent of each non-empty area across its stacking direction (height of
        top/bottom, width of left/right)."""
        out: dict[str, int] = {}
        for s in self.slots:
            if s.area in ("top", "bottom"):
                out[s.area] = s.h
            else:
                out[s.area] = s.w
        return out

    def hidden(self) -> list[tuple[str, str]]:
        """``(panel, raised panel it sits behind)`` for every panel not on screen."""
        return [(pid, s.raised) for s in self.slots for pid in s.members[1:]]

    def _label(self, pid: str) -> str:
        return self.labels.get(pid) or pid

    def summary(self) -> str:
        n = sum(len(s.members) for s in self.slots)
        hid = self.hidden()
        parts = [f"{self.width}x{self.height}: {n - len(hid)}/{n} panels on screen in {len(self.slots)} docks"]
        if hid:
            parts.append("behind tabs: " + ", ".join(f"{self._label(p)} (under {self._label(r)})" for p, r in hid))
        if not self.fits:
            parts.append(f"DOES NOT FIT: {self.overflow[0]} px too wide, {self.overflow[1]} px too tall")
        return "; ".join(parts)

    def report(self) -> str:
        lines = [self.summary(), ""]
        for s in self.slots:
            tabs = " | ".join(self._label(p) for p in s.members)
            lines.append(f"  {s.area:<6} {s.w:>5} x {s.h:<5} {tabs}")
        lines.append("")
        lines.append("  preferred-size fraction on screen (0 = behind a tab):")
        for pid, sat in sorted(self.satisfaction.items(), key=lambda kv: kv[1]):
            lines.append(f"    {sat:5.2f}  {self._label(pid)}")
        lines.append("")
        lines.append("  cost " + f"{self.cost:.2f} = " + " + ".join(f"{k} {v:.2f}" for k, v in self.terms.items() if v))
        lines.append(f"  ({self.evaluations} layouts tried in {self.elapsed_s:.2f} s)")
        return "\n".join(lines)

    def render_ascii(self, cols: int = 100) -> str:
        """A box drawing of the layout, ``cols`` characters wide."""
        sx = (cols - 1) / max(1, self.width)
        rows = max(12, int(round(cols * self.height / max(1, self.width) * 0.45)))
        sy = (rows - 1) / max(1, self.height)
        grid = [[" "] * cols for _ in range(rows)]

        def put(r, c, ch):
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = ch

        for s in self.slots:
            c0, c1 = int(round(s.x * sx)), int(round((s.x + s.w) * sx))
            r0, r1 = int(round(s.y * sy)), int(round((s.y + s.h) * sy))
            c1, r1 = max(c1, c0 + 2), max(r1, r0 + 2)
            for c in range(c0, c1 + 1):
                for r in (r0, r1):
                    put(r, c, "+" if grid[r][c] in "+|" or c in (c0, c1) else "-")
            for r in range(r0 + 1, r1):
                for c in (c0, c1):
                    put(r, c, "+" if grid[r][c] in "+-" else "|")
            inner = max(0, c1 - c0 - 2)
            text = [self._label(s.raised)]
            if s.stacked:
                text.append("tabs: " + ", ".join(self._label(p) for p in s.members[1:]))
            text.append(f"{s.w}x{s.h}")
            wrapped: list[str] = []
            for t in text:
                while t and len(wrapped) < max(0, r1 - r0 - 1):
                    wrapped.append(t[:inner])
                    t = t[inner:]
            for i, t in enumerate(wrapped):
                for j, ch in enumerate(t):
                    put(r0 + 1 + i, c0 + 1 + j, ch)
        return "\n".join("".join(row).rstrip() for row in grid)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _distribute(total: float, mins: Sequence[float], wants: Sequence[float], weights: Sequence[float]):
    """Split *total* among items: each gets its min, then the rest goes toward
    each item's want in proportion to weight, then any rest by weight.

    Returns ``(sizes, overflow)``; overflow > 0 when the mins do not fit (the
    sizes are then the mins).
    """
    sizes = [float(m) for m in mins]
    extra = total - sum(sizes)
    if extra < 0:
        return sizes, -extra
    need = [max(0.0, w - m) for w, m in zip(wants, mins)]
    active = [i for i in range(len(sizes)) if need[i] > 0]
    while extra > 1e-6 and active:
        wsum = sum(weights[i] for i in active)
        capped = [i for i in active if extra * weights[i] / wsum >= need[i]]
        if not capped:
            for i in active:
                sizes[i] += extra * weights[i] / wsum
            extra = 0.0
            break
        for i in capped:
            sizes[i] += need[i]
            extra -= need[i]
            need[i] = 0.0
        active = [i for i in active if need[i] > 0]
    if extra > 1e-6 and sizes:
        wsum = sum(weights) or 1.0
        for i in range(len(sizes)):
            sizes[i] += extra * weights[i] / wsum
    return sizes, 0.0


class _Problem:
    def __init__(self, panels: Sequence[PanelInfo], width: int, height: int):
        self.panels = list(panels)
        self.W = int(width)
        self.H = int(height)
        self.n = len(self.panels)
        self.minw = [p.min_size[0] for p in self.panels]
        self.minh = [p.min_size[1] for p in self.panels]
        self.prefw = [p.pref[0] for p in self.panels]
        self.prefh = [p.pref[1] for p in self.panels]
        self.wt = [p.weight for p in self.panels]
        self.sw = [p.space_weight for p in self.panels]
        self.fam = [p.family for p in self.panels]
        self.allowed = [set(a for a in p.areas if a in AREAS) or set(AREAS) for p in self.panels]
        self.evals = 0
        self.max_evals = 30000
        self.deadline = float("inf")
        self._cache: dict = {}

    # A state is a tuple of slots; a slot is (area, members), members sorted
    # raised first (heaviest, then declaration order).
    def norm_slot(self, area: str, members: Iterable[int]) -> tuple:
        return (area, tuple(sorted(members, key=lambda i: (-self.wt[i], i))))

    def norm(self, slots: Iterable[tuple]) -> tuple:
        return tuple(sorted((s for s in slots if s[1]), key=lambda s: (AREAS.index(s[0]), min(s[1]))))

    def legal(self, state: tuple) -> bool:
        return all(area in self.allowed[i] for area, members in state for i in members)

    def geometry(self, state: tuple):
        """Rectangles, overflow and satisfaction of a state."""
        W, H = self.W, self.H
        by_area: dict[str, list[tuple]] = {a: [] for a in AREAS}
        for slot in state:
            by_area[slot[0]].append(slot)

        def dims(members):
            r = members[0]
            mw = max(self.minw[i] for i in members)
            mh = max(self.minh[i] for i in members)
            pw, ph = max(self.prefw[r], mw), max(self.prefh[r], mh)
            if len(members) > 1:
                mh += TAB_BAR_H
                ph += TAB_BAR_H
            return mw, mh, pw, ph, self.sw[r]

        d = {a: [dims(s[1]) for s in by_area[a]] for a in AREAS}
        overflow_w = overflow_h = 0.0

        def seps(k):
            return SEP * max(0, k - 1)

        # Along each area: top/bottom need width, left/right need height.
        for a in ("top", "bottom"):
            if d[a]:
                overflow_w = max(overflow_w, sum(x[0] for x in d[a]) + seps(len(d[a])) - W)
        # Vertical bands: top, middle (left + right), bottom.
        bands = []
        for a in ("top", "mid", "bottom"):
            if a == "mid":
                cols = [c for c in ("left", "right") if d[c]]
                if not cols:
                    continue
                mn = max(sum(x[1] for x in d[c]) + seps(len(d[c])) for c in cols)
                want = max(sum(x[3] for x in d[c]) + seps(len(d[c])) for c in cols)
                wgt = max(x[4] for c in cols for x in d[c])
            else:
                if not d[a]:
                    continue
                mn = max(x[1] for x in d[a])
                want = max(x[3] for x in d[a])
                wgt = max(x[4] for x in d[a])
            bands.append((a, mn, want, wgt))
        bh, ov = _distribute(H - seps(len(bands)), [b[1] for b in bands], [b[2] for b in bands], [b[3] for b in bands])
        overflow_h = max(overflow_h, ov)
        band_h = {b[0]: h for b, h in zip(bands, bh)}
        y = 0.0
        band_y = {}
        for b, h in zip(bands, bh):
            band_y[b[0]] = y
            y += h + SEP
        # Middle band: left and right columns share the width.
        cols = [c for c in ("left", "right") if d[c]]
        col_w: dict[str, float] = {}
        if cols:
            cw, ov = _distribute(W - seps(len(cols)), [max(x[0] for x in d[c]) for c in cols],
                                 [max(x[2] for x in d[c]) for c in cols], [max(x[4] for x in d[c]) for c in cols])
            overflow_w = max(overflow_w, ov)
            col_w = dict(zip(cols, cw))
        rects: list[tuple[tuple, float, float, float, float]] = []
        for a in AREAS:
            if not d[a]:
                continue
            if a in ("top", "bottom"):
                ws, _ = _distribute(W - seps(len(d[a])), [x[0] for x in d[a]], [x[2] for x in d[a]], [x[4] for x in d[a]])
                x = 0.0
                for slot, w in zip(by_area[a], ws):
                    rects.append((slot, x, band_y[a], w, band_h[a]))
                    x += w + SEP
            else:
                hs, ov = _distribute(band_h["mid"] - seps(len(d[a])), [x[1] for x in d[a]], [x[3] for x in d[a]], [x[4] for x in d[a]])
                overflow_h = max(overflow_h, ov)
                x0 = 0.0 if a == "left" else W - col_w[a]
                y = band_y["mid"]
                for slot, h in zip(by_area[a], hs):
                    rects.append((slot, x0, y, col_w[a], h))
                    y += h + SEP
        sat = [0.0] * self.n
        roomy = [0.0] * self.n
        for slot, _x, _y, w, h in rects:
            r = slot[1][0]
            ch = h - (TAB_BAR_H if len(slot[1]) > 1 else 0)
            sat[r] = min(1.0, w / self.prefw[r]) * min(1.0, ch / self.prefh[r])
            roomy[r] = min(1.0, w / (1.5 * self.prefw[r])) * min(1.0, ch / (1.5 * self.prefh[r]))
        self._roomy = roomy
        return rects, (int(round(max(0.0, overflow_w))), int(round(max(0.0, overflow_h)))), sat

    def cost(self, state: tuple) -> float:
        c = self._cache.get(state)
        if c is not None:
            return c
        self.evals += 1
        c = sum(self.terms(state).values())
        self._cache[state] = c
        return c

    def terms(self, state: tuple) -> dict[str, float]:
        _rects, (ow, oh), sat = self.geometry(state)
        roomy = self._roomy
        hidden = shortfall = stacks = tidy = 0.0
        fam_areas: dict[str, set] = {}
        for area, members in state:
            r = members[0]
            shortfall += self.sw[r] * (1.0 - sat[r])
            tidy += COMFORT * self.sw[r] * (1.0 - roomy[r])
            if self.fam[r]:
                fam_areas.setdefault(self.fam[r], set()).add(area)
            for i in members[1:]:
                hidden += self.wt[i]
            if len(members) > 1:
                # the raised panel is on screen anyway; the tabs behind it should be of one kind
                fams = {self.fam[i] or f"_{i}" for i in members[1:]}
                stacks += FAMILY_MIX * (len(fams) - 1)
                stacks += CROWDED_TAB * max(0, len(members) - MAX_EASY_TABS)
        tidy += COHESION * sum(len(a) - 1 for a in fam_areas.values())
        overflow = (OVERFLOW_BASE + ow + oh) if (ow or oh) else 0.0
        return {"hidden": hidden, "shortfall": shortfall, "stacks": stacks, "tidy": tidy, "overflow": overflow}

    # --- moves ------------------------------------------------------------

    def neighbours(self, state: tuple):
        slots = list(state)
        k = len(slots)
        where = {}
        for si, (_a, members) in enumerate(slots):
            for i in members:
                where[i] = si
        # one panel: to another slot, or to a new slot in any area
        for i in range(self.n):
            si = where[i]
            area, members = slots[si]
            rest = [m for m in members if m != i]
            for sj in range(k):
                if sj == si:
                    continue
                a2, m2 = slots[sj]
                new = list(slots)
                new[si] = self.norm_slot(area, rest)
                new[sj] = self.norm_slot(a2, m2 + (i,))
                yield self.norm(new)
            for a in AREAS:
                if not rest and a == area:
                    continue
                new = list(slots)
                new[si] = self.norm_slot(area, rest)
                new.append(self.norm_slot(a, (i,)))
                yield self.norm(new)
        # a whole slot: to another area, or merged into another slot
        for si in range(k):
            area, members = slots[si]
            for a in AREAS:
                if a != area:
                    new = list(slots)
                    new[si] = (a, members)
                    yield self.norm(new)
            for sj in range(k):
                if sj != si:
                    new = list(slots)
                    a2, m2 = slots[sj]
                    new[sj] = self.norm_slot(a2, m2 + members)
                    new[si] = (area, ())
                    yield self.norm(new)

    def out_of_budget(self) -> bool:
        return self.evals >= self.max_evals or time.monotonic() >= self.deadline

    def climb(self, state: tuple) -> tuple[tuple, float]:
        """Steepest descent over single moves until no move improves."""
        best = state
        best_c = self.cost(state)
        while not self.out_of_budget():
            step, step_c = None, best_c
            for cand in self.neighbours(best):
                if not self.legal(cand):
                    continue
                c = self.cost(cand)
                if c < step_c - 1e-9:
                    step, step_c = cand, c
            if step is None:
                break
            best, best_c = step, step_c
        return best, best_c

    def kick(self, state: tuple, rng: random.Random, moves: int) -> tuple:
        """A few random legal moves away from *state*."""
        for _ in range(moves):
            cands = [c for c in self.neighbours(state) if self.legal(c)]
            if not cands:
                break
            state = rng.choice(cands)
        return state

    # --- starting points -----------------------------------------------------

    def _pick_area(self, i: int, prefer: Sequence[str]) -> str:
        for a in prefer:
            if a in self.allowed[i]:
                return a
        return next(iter(self.allowed[i]))

    def starts(self, initial: Optional[Sequence[tuple[str, Sequence[str]]]], rng: random.Random) -> list[tuple]:
        ids = {p.id: i for i, p in enumerate(self.panels)}
        out: list[tuple] = []
        if initial:
            slots, seen = [], set()
            for area, members in initial:
                m = [ids[pid] for pid in members if pid in ids and ids[pid] not in seen]
                seen.update(m)
                if m:
                    slots.append(self.norm_slot(area if area in AREAS else "right", m))
            slots += [self.norm_slot(self._pick_area(i, ("right",)), (i,)) for i in range(self.n) if i not in seen]
            out.append(self.norm(slots))

        def shape_area(i):
            wide = self.prefw[i] >= 1.6 * self.prefh[i]
            if wide:
                return self._pick_area(i, ("bottom", "top") if self.wt[i] >= TIERS["primary"] else ("top", "bottom"))
            return self._pick_area(i, ("right", "left") if self.wt[i] >= TIERS["primary"] else ("left", "right"))

        # every panel its own dock, area by shape
        out.append(self.norm([self.norm_slot(shape_area(i), (i,)) for i in range(self.n)]))
        # status / rare panels stacked by family, the rest on their own
        fam_slots: dict[str, list[int]] = {}
        slots = []
        for i in range(self.n):
            if self.wt[i] <= TIERS["status"]:
                fam_slots.setdefault(self.fam[i] or "_", []).append(i)
            else:
                slots.append(self.norm_slot(shape_area(i), (i,)))
        for members in fam_slots.values():
            slots.append(self.norm_slot(self._pick_area(members[0], ("right", "left")), members))
        out.append(self.norm(slots))
        # random
        for _ in range(4):
            slots = []
            for i in range(self.n):
                if slots and rng.random() < 0.3:
                    j = rng.randrange(len(slots))
                    slots[j] = self.norm_slot(slots[j][0], slots[j][1] + (i,))
                else:
                    slots.append(self.norm_slot(rng.choice(sorted(self.allowed[i])), (i,)))
            out.append(self.norm(slots))
        return [s for s in out if self.legal(s)]


def solve(
    panels: Sequence[PanelInfo],
    width: int,
    height: int,
    *,
    initial: Optional[Sequence[tuple[str, Sequence[str]]]] = None,
    max_evals: int = 30000,
    time_budget_s: float = 10.0,
    seed: int = 0,
) -> LayoutSolution:
    """Arrange *panels* in a dock region *width* x *height* px.

    *initial*: optional starting layout, ``[(area, [panel ids, ...]), ...]``
    (e.g. the dashboard's default placement); the solver also starts from its
    own guesses.  The search stops after *max_evals* distinct layouts (the
    result is then deterministic for a given *seed*) or *time_budget_s*,
    whichever comes first.
    """
    if not panels:
        return LayoutSolution(width, height, [], 0.0, {}, (0, 0), {})
    t0 = time.monotonic()
    prob = _Problem(panels, width, height)
    prob.max_evals = max(1, int(max_evals))
    prob.deadline = t0 + max(0.05, time_budget_s)
    rng = random.Random(seed)
    best, best_c = None, float("inf")
    for start in prob.starts(initial, rng):
        state, c = prob.climb(start)
        if c < best_c:
            best, best_c = state, c
        if prob.out_of_budget():
            break
    # Iterated local search: kick the best layout a little, climb again; stop
    # early once kicks keep finding nothing better (a small problem's whole
    # neighbourhood is soon cached, so the evaluation count stops rising).
    stale = 0
    while not prob.out_of_budget() and stale < ILS_PATIENCE:
        state, c = prob.climb(prob.kick(best, rng, rng.randint(2, 4)))
        if c < best_c - 1e-9:
            best, best_c, stale = state, c, 0
        else:
            stale += 1
    rects, overflow, sat = prob.geometry(best)
    slots = [
        SolvedSlot(area=slot[0], members=[prob.panels[i].id for i in slot[1]],
                   x=int(round(x)), y=int(round(y)), w=int(round(w)), h=int(round(h)))
        for slot, x, y, w, h in rects
    ]
    return LayoutSolution(
        width=int(width), height=int(height), slots=slots, cost=best_c, terms=prob.terms(best),
        overflow=overflow, satisfaction={p.id: sat[i] for i, p in enumerate(prob.panels)},
        labels={p.id: p.label for p in prob.panels if p.label},
        elapsed_s=time.monotonic() - t0, evaluations=prob.evals,
    )


def region_for_display(width: int, height: int, chrome: tuple[int, int] = DEFAULT_CHROME) -> tuple[int, int]:
    """Dock region of a maximized dashboard on a *width* x *height* display."""
    return max(1, width - chrome[0]), max(1, height - chrome[1])


def panels_from_hints(ids: Iterable[str], hints: dict[str, dict], labels: Optional[dict[str, str]] = None) -> list[PanelInfo]:
    """Build :class:`PanelInfo` from a hints table ``{id: {tier, family, min, pref, areas}}``."""
    out = []
    for pid in ids:
        h = hints.get(pid, {})
        mn = tuple(h.get("min", (320, 200)))
        out.append(PanelInfo(
            id=pid,
            label=(labels or {}).get(pid) or h.get("label", ""),
            min_size=(int(mn[0]), int(mn[1])),
            pref_size=tuple(int(v) for v in h.get("pref", (0, 0))),
            tier=h.get("tier", "status"),
            family=h.get("family", ""),
            areas=tuple(h.get("areas", AREAS)),
        ))
    return out


def _parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().replace(" ", "").split("x")
    return int(w), int(h)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Preview the dashboard auto-arrange for a display size.")
    ap.add_argument("--layout-module", required=True, help="module with <KIND>_PLACEMENT and AUTO_LAYOUT_HINTS")
    ap.add_argument("--kind", default="server", choices=("server", "client"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--display", help="display resolution WxH (window chrome is subtracted)")
    g.add_argument("--region", help="dock region WxH (no chrome subtracted)")
    ap.add_argument("--cols", type=int, default=110)
    ap.add_argument("--evals", type=int, default=30000, help="layouts to try")
    args = ap.parse_args(argv)
    mod = importlib.import_module(args.layout_module)
    ids = list(getattr(mod, f"{args.kind.upper()}_PLACEMENT", {}))
    hints = getattr(mod, "AUTO_LAYOUT_HINTS", {})
    w, h = region_for_display(*_parse_size(args.display)) if args.display else _parse_size(args.region)
    sol = solve(panels_from_hints(ids, hints), w, h, max_evals=args.evals)
    print(sol.render_ascii(args.cols))
    print()
    print(sol.report())
    return 0 if sol.fits else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "AREAS", "TIERS", "PanelInfo", "SolvedSlot", "LayoutSolution",
    "solve", "region_for_display", "panels_from_hints", "DEFAULT_CHROME",
]
