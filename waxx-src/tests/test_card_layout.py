"""Card layouts (waxx.util.guis.card_layout): the column solver on its own,
and MasonryLayout / FlowLayout on offscreen Qt widgets."""
import itertools
import os

import pytest
from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from waxx.util.guis import card_layout as cl
from waxx.util.guis.card_layout import (
    FlowLayout, MasonryLayout, column_heights, greedy_columns, reading_inversions,
    score_columns, solve_columns,
)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


# --- solver ----------------------------------------------------------------------

def _brute_force(heights, k, gap):
    best = None
    for assign in itertools.product(range(k), repeat=len(heights)):
        cols = [[i for i, c in enumerate(assign) if c == col] for col in range(k)]
        cols = cl._canonical(cols)
        s = score_columns(cols, heights, gap)
        if best is None or s < best[0]:
            best = (s, cols)
    return best


@pytest.mark.parametrize("heights,k", [
    ([330, 230, 280, 480, 230, 200], 3),
    ([330, 230, 280, 480, 230, 200], 4),
    ([100, 400, 120, 90, 300, 310, 50], 3),
    ([200, 200, 200, 200, 200], 2),
])
def test_solver_matches_brute_force(heights, k):
    cols = solve_columns(heights, k, gap=12)
    assert sorted(i for c in cols for i in c) == list(range(len(heights)))
    assert all(c == sorted(c) for c in cols)                 # definition order kept
    assert score_columns(cols, heights, 12) == pytest.approx(_brute_force(heights, k, 12)[0])


def test_solver_is_never_worse_than_greedy():
    heights = [330, 230, 280, 480, 230, 200, 150, 390]
    for k in (2, 3, 4):
        greedy = cl._canonical(greedy_columns(heights, k, 12))
        assert score_columns(solve_columns(heights, k, 12), heights, 12) <= \
            score_columns(greedy, heights, 12) + 1e-9


def test_degenerate_cases():
    assert solve_columns([], 3) == [[], [], []]
    assert solve_columns([10, 20], 1) == [[0, 1]]
    assert solve_columns([10, 20], 4) == [[0], [1], [], []]


def test_reading_inversions():
    # column 0: card 0 then card 3 below; column 1: card 1, card 2 -- card 2
    # starts above card 3 but after card 1: no inversion.
    assert reading_inversions([[0, 3], [1, 2]], [100, 50, 50, 50], 0) == 0
    # card 2 on top of column 0, card 0 below it, card 1 top of column 1:
    # reads 2, 1, 0 -- three pairs out of order.
    assert reading_inversions([[2, 0], [1]], [50, 50, 50], 0) == 3


def test_hysteresis_keeps_a_near_optimal_previous_arrangement():
    heights = [300, 300, 300, 310]
    best = solve_columns(heights, 2, 0)
    other = [[0, 3], [1, 2]] if best != [[0, 3], [1, 2]] else [[0, 1], [2, 3]]
    gap_px = score_columns(other, heights, 0) - score_columns(best, heights, 0)
    kept = solve_columns(heights, 2, 0, previous=other)
    assert (kept == other) == (gap_px <= cl.HYSTERESIS_PX)
    # a clearly worse previous arrangement is dropped
    assert solve_columns([500, 10, 10, 10], 2, 0, previous=[[0, 1, 2], [3]]) != [[0, 1, 2], [3]]
    # an invalid previous one is ignored
    assert solve_columns(heights, 2, 0, previous=[[0], [1]]) == best


def test_column_heights_include_gaps():
    assert column_heights([[0, 1], [2], []], [10, 20, 30], 5) == [35, 30, 0]


# --- masonry ------------------------------------------------------------------------

class Card(QLabel):
    """A card of fixed height and minimum width."""

    def __init__(self, h, min_w=300):
        super().__init__("card")
        self.setMinimumWidth(min_w)
        self.setFixedHeight(h)


def _host(qapp, heights, **kw):
    host = QWidget()
    layout = MasonryLayout(host, **kw)
    cards = [Card(h) for h in heights]
    for c in cards:
        layout.addWidget(c)
    return host, layout, cards


HEIGHTS = [330, 230, 280, 480, 230, 200]


@pytest.mark.parametrize("width", [320, 700, 900, 1150, 1400, 1900, 2600, 4000])
def test_cards_never_overlap_and_stay_inside(qapp, width):
    host, layout, cards = _host(qapp, HEIGHTS, min_column=360, preferred_column=430,
                                max_column=540, spacing=12)
    height = layout.heightForWidth(width)
    layout.setGeometry(QRect(0, 0, width, height))
    rects = [c.geometry() for c in cards]
    k, col_w, left = layout.columns_for(width)
    for r in rects:
        assert r.width() == col_w
        assert r.left() >= 0 and r.bottom() < height
        if width >= 360:
            assert r.right() < width
    for a, b in itertools.combinations(rects, 2):
        assert not a.intersects(b)
    assert max(r.bottom() for r in rects) + 1 == height
    assert 360 <= col_w <= 540 or width < 360


def test_column_count_follows_width(qapp):
    host, layout, cards = _host(qapp, HEIGHTS, min_column=360, preferred_column=430,
                                max_column=540, spacing=12)
    counts = [layout.columns_for(w)[0] for w in (500, 1150, 1400, 1900, 2600)]
    assert counts == sorted(counts)
    assert counts[0] == 1 and counts[-1] >= 4
    # never more columns than cards; wide windows centre a capped block
    k, col_w, left = layout.columns_for(6000)
    assert k == len(HEIGHTS) and col_w == 540 and left > 0


def test_extra_column_needs_to_earn_its_narrowness(qapp):
    """At 1900 px five 361 px columns fit, but four comfortable ones win."""
    host, layout, cards = _host(qapp, HEIGHTS, min_column=360, preferred_column=430,
                                max_column=540, spacing=12)
    k, col_w, _ = layout.columns_for(1900)
    assert k == 4 and col_w >= 430


def test_column_count_has_hysteresis(qapp):
    """Where two counts score close (the narrowness penalty shrinking as the
    width grows), the count in use holds for a band of widths."""
    host, layout, cards = _host(qapp, HEIGHTS, min_column=360, preferred_column=430,
                                max_column=540, spacing=12)
    # first width at which 4 columns win, with no count in use yet
    threshold = next(w for w in range(1480, 2400, 2) if layout.columns_for(w)[0] >= 4)
    layout.invalidate()
    layout.setGeometry(QRect(0, 0, threshold - 2, 2000))
    assert layout._last_k == 3
    layout.invalidate()
    assert layout.columns_for(threshold)[0] == 3        # held by hysteresis
    layout.invalidate()
    assert layout.columns_for(threshold + 200)[0] == 4  # well past it: switches


def _placement(layout, cards, width):
    layout.setGeometry(QRect(0, 0, width, layout.heightForWidth(width)))
    return [c.geometry().x() for c in cards]


def test_height_changes_never_move_a_card_to_another_column(qapp):
    """Collapsing or opening a card changes its height; the arrangement is
    pinned, so it only restacks its own column."""
    host, layout, cards = _host(qapp, HEIGHTS, min_column=360, preferred_column=430,
                                max_column=540, spacing=12)
    before = _placement(layout, cards, 1400)
    cards[0].setFixedHeight(60)                      # collapsed ...
    cards[3].setFixedHeight(1500)                    # ... and one opened huge
    layout.invalidate()
    assert _placement(layout, cards, 1400) == before
    # the test means something: solved afresh, this would move cards
    layout.reflow()
    assert _placement(layout, cards, 1400) != before


def test_a_scrollbar_appearing_does_not_reflow(qapp):
    host, layout, cards = _host(qapp, HEIGHTS, min_column=360, preferred_column=430,
                                max_column=540, spacing=12)
    before = _placement(layout, cards, 1400)
    cards[3].setFixedHeight(1500)
    layout.invalidate()
    after = _placement(layout, cards, 1400 - 17)     # vertical scrollbar
    k_before = layout.columns_for(1400)[0]
    assert layout.columns_for(1400 - 17)[0] == k_before
    columns = lambda xs: sorted({x for x in xs})
    assert [columns(before).index(x) for x in before] == [columns(after).index(x) for x in after]
    # a real resize is solved again
    layout.invalidate()
    assert layout.columns_for(700)[0] < k_before


def test_hidden_items_take_no_space_and_their_pin_comes_back(qapp):
    host, layout, cards = _host(qapp, HEIGHTS, min_column=360, preferred_column=430,
                                max_column=540, spacing=12)
    host.show()
    before = _placement(layout, cards, 1400)
    cards[1].hide()                                  # a search hides a group
    layout.invalidate()
    _placement(layout, cards, 1400)
    visible = [c for c in cards if not c.isHidden()]
    for a, b in itertools.combinations([c.geometry() for c in visible], 2):
        assert not a.intersects(b)
    cards[1].show()                                  # search cleared
    layout.invalidate()
    assert _placement(layout, cards, 1400) == before
    host.hide()


def test_widest_card_sets_the_column_minimum(qapp):
    host = QWidget()
    layout = MasonryLayout(host, min_column=200, preferred_column=200, max_column=900)
    layout.addWidget(Card(100, min_w=450))
    layout.addWidget(Card(100, min_w=200))
    assert layout.column_minimum() == 450
    assert layout.columns_for(1000)[0] == 2
    assert layout.columns_for(800)[0] == 1


# --- flow ---------------------------------------------------------------------------

def test_flow_wraps_and_reports_its_height(qapp):
    host = QWidget()
    flow = FlowLayout(host, hspacing=5, vspacing=4)
    chips = []
    for _ in range(6):
        chip = QLabel("chip")
        chip.setFixedSize(100, 20)
        flow.addWidget(chip)
        chips.append(chip)
    assert flow.heightForWidth(700) == 20                  # one line
    assert flow.heightForWidth(320) == 20 * 2 + 4          # three per line
    flow.setGeometry(QRect(0, 0, 320, 44))
    assert [c.geometry().top() for c in chips] == [0, 0, 0, 24, 24, 24]
    assert chips[1].geometry().left() == 105
