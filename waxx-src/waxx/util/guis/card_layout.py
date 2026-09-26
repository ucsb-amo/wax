"""Layouts for card panels: balanced masonry columns, and wrapping rows.

:class:`MasonryLayout` puts cards -- each keeping its own height -- into
columns of equal width.  The width decides the number of columns (as many as
fit at the widest card's minimum width); columns grow with the window up to
``max_column``, beyond which the block is centred.  Which card goes in which
column is *solved* rather than dealt out greedily (:func:`solve_columns`):

* the tallest column is minimised (the panel stays short and even),
* ragged bottoms cost a little (columns end at similar heights),
* reading-order swaps cost more (cards keep roughly the order they are
  defined in, so "the coils are bottom-left" stays true at most widths),
* and while resizing, the previous arrangement is kept unless the new one is
  clearly better, so cards do not jump back and forth under a dragged edge.

Once solved, an arrangement is *pinned*: items changing height (a card
collapsed or expanded, a chip row wrapping) only restack their own column;
nothing moves to another column.  It is solved again only when the width
changes by more than :data:`PIN_TOLERANCE_PX` (a scrollbar appearing is
~15 px) or the set of visible items changes; each visible set keeps its own
pin, so clearing a search puts everything back where it was.  Hidden items
take no space.

:class:`FlowLayout` wraps small widgets (status chips, toggle pills) onto as
many lines as the width needs -- Qt's classic flow layout.

:func:`solve_columns` is pure Python, testable without Qt.
"""

from __future__ import annotations

from typing import Sequence

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QLayoutItem

#: px of tallest-column height one reading-order swap is worth giving up.
ORDER_WEIGHT = 36.
#: fraction of (tallest - shortest column) added to the score.
RAGGED_WEIGHT = 0.15
#: keep the previous arrangement unless the new one scores this much better.
HYSTERESIS_PX = 24.
#: search nodes before settling for the best arrangement found so far.
NODE_BUDGET = 40000

# Choosing the number of columns (MasonryLayout): every count that fits at the
# minimum width is solved and scored on
#   tallest column + NARROW_WEIGHT * (px each card is below the preferred width)
#                  + WASTE_WEIGHT * (px of width left unused)
# and the count in use keeps a COLUMNS_HYSTERESIS_PX advantage, so dragging a
# splitter across a threshold does not flip the layout back and forth.
NARROW_WEIGHT = 1.0
WASTE_WEIGHT = 0.1
# How much width that is depends on how card heights change near the switch
# (chip rows re-wrapping move them in steps): ~10 px of window width at the
# K-machine Composite tab's 3 <-> 4 switch, measured 2026-09-25 -- enough that
# a drag never flips back and forth.
COLUMNS_HYSTERESIS_PX = 60.
#: Width changes up to this keep a pinned arrangement (column count and which
#: item is in which column); only the column width follows.
PIN_TOLERANCE_PX = 40


# --- the solver ------------------------------------------------------------------

def column_heights(columns: Sequence[Sequence[int]], heights: Sequence[float],
                   gap: float) -> list[float]:
    return [sum(heights[i] for i in col) + gap * max(len(col) - 1, 0) for col in columns]


def reading_inversions(columns: Sequence[Sequence[int]], heights: Sequence[float],
                       gap: float) -> int:
    """Pairs of cards that read in the wrong order: row-major by top edge,
    left to right on ties -- how a person scans the panel."""
    tops = {}
    for c, col in enumerate(columns):
        y = 0.
        for i in col:
            tops[i] = (y, c)
            y += heights[i] + gap
    order = sorted(tops, key=lambda i: tops[i])
    inversions = 0
    for a, i in enumerate(order):
        for j in order[a + 1:]:
            if j < i:
                inversions += 1
    return inversions


def score_columns(columns, heights, gap) -> float:
    hs = column_heights(columns, heights, gap)
    tallest = max(hs) if hs else 0.
    return (tallest + RAGGED_WEIGHT * (tallest - min(hs, default=0.))
            + ORDER_WEIGHT * reading_inversions(columns, heights, gap))


def greedy_columns(heights: Sequence[float], k: int, gap: float) -> list[list[int]]:
    """Classic masonry: each card, in order, into the currently shortest column."""
    columns = [[] for _ in range(k)]
    col_h = [0.] * k
    for i, h in enumerate(heights):
        c = min(range(k), key=lambda c: (col_h[c], c))
        col_h[c] += h + (gap if columns[c] else 0.)
        columns[c].append(i)
    return columns


def _canonical(columns) -> list[list[int]]:
    """Stacks ordered by their first card; empty columns last."""
    filled = sorted((list(c) for c in columns if c), key=lambda c: c[0])
    return filled + [[] for _ in range(len(columns) - len(filled))]


def solve_columns(heights: Sequence[float], k: int, gap: float = 0.,
                  previous: Sequence[Sequence[int]] | None = None) -> list[list[int]]:
    """Assign cards (by index, heights given) to ``k`` columns.

    Branch and bound over assignments in which each column keeps its cards
    in definition order and columns are ordered by their first card (every
    other labelling is the same picture), seeded with the greedy masonry
    answer and bounded by the tallest column so far and the average height.
    Returns ``k`` lists of indices (some may be empty when there are fewer
    cards than columns).
    """
    n = len(heights)
    k = max(1, int(k))
    if n == 0:
        return [[] for _ in range(k)]
    if k == 1:
        return [list(range(n))]
    if n <= k:
        return [[i] for i in range(n)] + [[] for _ in range(k - n)]

    best = _canonical(greedy_columns(heights, k, gap))
    best_score = score_columns(best, heights, gap)
    floor = sum(heights) / k            # no arrangement's tallest column is lower
    cols: list[list[int]] = [[] for _ in range(k)]
    col_h = [0.] * k
    nodes = 0

    def search(i: int, used: int) -> None:
        nonlocal best, best_score, nodes
        nodes += 1
        if nodes > NODE_BUDGET:
            return
        if i == n:
            s = score_columns(cols, heights, gap)
            if s < best_score - 1e-9:
                best, best_score = [list(c) for c in cols], s
            return
        for c in range(min(used + 1, k)):
            added = heights[i] + (gap if cols[c] else 0.)
            if max(col_h[c] + added, floor) >= best_score:
                continue
            cols[c].append(i)
            col_h[c] += added
            search(i + 1, max(used, c + 1))
            col_h[c] -= added
            cols[c].pop()

    search(0, 0)

    if previous is not None:
        prev = [list(c) for c in previous]
        flat = sorted(i for c in prev for i in c)
        if len(prev) == k and flat == list(range(n)) and \
                all(c == sorted(c) for c in prev):
            if score_columns(prev, heights, gap) <= best_score + HYSTERESIS_PX:
                return prev
    return best


# --- masonry ------------------------------------------------------------------------

def _item_height(item: QLayoutItem, width: int) -> int:
    h = item.heightForWidth(width) if item.hasHeightForWidth() else -1
    if h < 0:
        h = item.sizeHint().height()
    return max(h, item.minimumSize().height())


class MasonryLayout(QLayout):
    """Cards in balanced columns; see the module docstring.

    ``min_column``: never narrower (nor narrower than the widest card's own
    minimum).  ``preferred_column``: narrower than this costs score, so extra
    columns are only added when they are worth it.  ``max_column``: never
    wider; the block is centred instead.
    """

    def __init__(self, parent=None, *, min_column: int = 360, preferred_column: int = 430,
                 max_column: int = 540, spacing: int = 12):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._min_column = int(min_column)
        self._preferred_column = int(preferred_column)
        self._max_column = int(max_column)
        self._gap = int(spacing)
        self._cache: dict[int, tuple] = {}
        self._previous: dict[tuple, list[list[int]]] = {}
        self._last_k: int | None = None
        # visible item indices -> (width solved at, k, columns in visible indices)
        self._pins: dict[tuple, tuple] = {}
        self.setContentsMargins(0, 0, 0, 0)

    # -- QLayout plumbing -------------------------------------------------------

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)
        self.invalidate()

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        if 0 <= index < len(self._items):
            item = self._items.pop(index)
            self.invalidate()
            return item
        return None

    def invalidate(self) -> None:
        self._cache.clear()
        super().invalidate()

    def expandingDirections(self):
        return Qt.Orientation.Horizontal

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return int(self._plan(width)[3])

    def _visible(self) -> tuple:
        return tuple(i for i, it in enumerate(self._items) if not it.isEmpty())

    def column_minimum(self) -> int:
        widest = max((self._items[i].minimumSize().width() for i in self._visible()),
                     default=0)
        return max(self._min_column, widest)

    def reflow(self) -> None:
        """Forget the pinned arrangements: the next layout is solved afresh."""
        self._pins.clear()
        self._previous.clear()
        self.invalidate()

    def minimumSize(self) -> QSize:
        m = self.contentsMargins()
        return QSize(self.column_minimum() + m.left() + m.right(), 0)

    def sizeHint(self) -> QSize:
        m = self.contentsMargins()
        width = 2 * self.column_minimum() + self._gap + m.left() + m.right()
        return QSize(width, self.heightForWidth(width))

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        rects, columns, k, _, _, _ = self._plan(rect.width())
        visible = self._visible()
        self._previous[(k, visible)] = columns
        self._last_k = k
        pin = self._pins.get(visible)
        if pin is None or pin[1] != k or pin[2] != columns \
                or abs(rect.width() - pin[0]) > PIN_TOLERANCE_PX:
            # a fresh solve: pin it (plans cached under the old pin are stale)
            self._pins[visible] = (rect.width(), k, [list(c) for c in columns])
            self._cache.clear()
        for item, r in zip(self._items, rects):
            if not item.isEmpty():
                item.setGeometry(r.translated(rect.topLeft()))

    # -- planning ---------------------------------------------------------------

    def columns_for(self, width: int) -> tuple[int, int, int]:
        """(columns, column width, left offset) chosen for a total width."""
        _, _, k, _, col_w, left = self._plan(width)
        return k, col_w, left

    def _geometry(self, avail: int, k: int) -> tuple[int, int]:
        """(column width, unused width) for k columns in ``avail`` px."""
        col_w = (avail - (k - 1) * self._gap) // k
        col_w = max(min(col_w, self._max_column), min(self.column_minimum(), avail))
        used = k * col_w + (k - 1) * self._gap
        return col_w, max(avail - used, 0)

    def _plan(self, width: int):
        cached = self._cache.get(width)
        if cached is not None:
            return cached
        m = self.contentsMargins()
        avail = max(width - m.left() - m.right(), 1)
        visible = self._visible()
        items = [self._items[i] for i in visible]
        n = len(items)
        col_min = self.column_minimum()
        k_max = max(1, min(max(n, 1), (avail + self._gap) // (col_min + self._gap)))
        best = None
        pin = self._pins.get(visible)
        if pin is not None and abs(width - pin[0]) <= PIN_TOLERANCE_PX and pin[1] <= k_max:
            # pinned: same columns, only heights (and the column width) follow
            k, columns = pin[1], pin[2]
            col_w, waste = self._geometry(avail, k)
            heights = [_item_height(it, col_w) for it in items]
            best = (0., k, col_w, waste, heights, columns)
        else:
            for k in range(1, k_max + 1):
                col_w, waste = self._geometry(avail, k)
                heights = [_item_height(it, col_w) for it in items]
                columns = solve_columns(heights, k, self._gap,
                                        previous=self._previous.get((k, visible)))
                tallest = max(column_heights(columns, heights, self._gap), default=0.)
                score = (tallest
                         + NARROW_WEIGHT * n * max(self._preferred_column - col_w, 0)
                         + WASTE_WEIGHT * waste)
                if k == self._last_k:
                    score -= COLUMNS_HYSTERESIS_PX
                if best is None or score < best[0] - 1e-9:
                    best = (score, k, col_w, waste, heights, columns)
        _, k, col_w, waste, heights, columns = best
        left = m.left() + waste // 2
        rects = [QRect()] * len(self._items)
        bottom = 0
        for c, col in enumerate(columns):
            x = left + c * (col_w + self._gap)
            y = m.top()
            for j in col:
                rects[visible[j]] = QRect(x, y, col_w, heights[j])
                y += heights[j] + self._gap
            bottom = max(bottom, y - self._gap if col else y)
        plan = (rects, columns, k, bottom + m.bottom(), col_w, left)
        self._cache[width] = plan
        return plan


# --- flow ------------------------------------------------------------------------

class FlowLayout(QLayout):
    """Left-aligned items that wrap onto new lines as the width shrinks."""

    def __init__(self, parent=None, hspacing: int = 6, vspacing: int = 4):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h = hspacing
        self._v = vspacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation.Horizontal

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _arrange(self, rect: QRect, apply: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = area.x(), area.y(), 0
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            if x > area.x() and x + hint.width() > area.right() + 1:
                x = area.x()
                y += line_h + self._v
                line_h = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._h
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()
