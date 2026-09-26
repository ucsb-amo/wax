"""Graphics items for the lane stack.

Every item here paints only what is on screen, in device pixels, from
numpy arrays it slices with searchsorted -- never one scene item per pulse.
That is what keeps pan and zoom at frame rate with thousands of pulses and
millions of samples:

* PulseBarItem      translucent bars with labels drawn only when wide enough
* DigitalTraceItem  a level trace from an edge list (run-length encoded)
* AnalogTraceItem   a min/max pyramid of the samples, one resolution per
                    zoom level, swapped on range change
* EventMarkerItem   point markers (phase resets, frequency changes)
* BandItem          background bands (shots, handshake framing, not simulated)
* GapDimensionItem  automatic <-- 2.000 µs --> dimensions between bars
* CursorLine        a measurement cursor that snaps to edges
"""

import bisect

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, QPointF, Qt
from PyQt6.QtGui import (QBrush, QColor, QPen, QFont, QFontMetrics,
                         QPainterPath, QPolygonF)
from PyQt6.QtWidgets import QApplication

from waxx.util.seqview.timefmt import fmt_duration

LABEL_MIN_PX = 40        # bar width below which no label is drawn
GAP_MIN_PX = 44          # gap width below which no dimension is drawn
CLUSTER_PX = 2           # bars narrower than this are drawn as 1 px ticks
SELECT_CARET_PX = 8      # selected bars narrower than this get a caret on top
MAX_LABELS = 400


def qcolor(hexstr, alpha=255):
    c = QColor(hexstr)
    c.setAlpha(int(alpha))
    return c


class ViewItem(pg.GraphicsObject):
    """Base: bounding rect = the visible view, repaint on every range
    change, painting in device pixels."""

    def __init__(self, tmin=0., tmax=1.):
        super().__init__()
        self.tmin, self.tmax = float(tmin), float(max(tmax, tmin + 1.))
        self.setFlag(self.GraphicsItemFlag.ItemUsesExtendedStyleOption, False)

    def boundingRect(self):
        vb = self.getViewBox()
        if vb is None:
            return QRectF(self.tmin, 0., self.tmax - self.tmin, 1.)
        r = vb.viewRect()
        return QRectF(r)

    def viewRangeChanged(self):
        self.prepareGeometryChange()
        self.update()

    # --- device mapping helpers ---
    @staticmethod
    def _affine(p):
        tr = p.transform()
        return tr, tr.m11(), tr.dx(), tr.m22(), tr.dy()

    def _view(self):
        vb = self.getViewBox()
        if vb is None:
            return self.tmin, self.tmax, 0., 1., 1., 1.
        r = vb.viewRect()
        w = max(vb.width(), 1.)
        return r.left(), r.right(), r.top(), r.bottom(), w, max(vb.height(), 1.)

    def _dev(self, p):
        """Device-space mapping for this paint call: (l, r, tr, m11, dx,
        m22, dy, xl, xr) with xl/xr the device x of the view's left and
        right edges. The painter's device origin is the whole widget, not
        the view box, so every clip and label offset uses xl/xr."""
        l, r, _top, _bot, _w, _h = self._view()
        tr = p.transform()
        m11, dx, m22, dy = tr.m11(), tr.dx(), tr.m22(), tr.dy()
        xl, xr = l * m11 + dx, r * m11 + dx
        if xl > xr:
            xl, xr = xr, xl
        return l, r, tr, m11, dx, m22, dy, xl, xr


# ---------------------------------------------------------------------------
# pulses
# ---------------------------------------------------------------------------

class PulseBarItem(ViewItem):
    """Bars for one lane. Arrays sorted by t0."""

    def __init__(self, pulses, tmax, style='bar', y0=0.08, y1=0.92,
                 rows=True, labels=True):
        super().__init__(0., tmax)
        self.style = style            # 'bar' | 'marker' (thin strip on top)
        self.labels_enabled = labels
        self.y0, self.y1 = y0, y1
        self.set_pulses(pulses, rows=rows)
        self.selected = set()         # pulse ids drawn with a white outline
        self.related = set()          # same source line: lighter outline
        self.hovered = None
        self.dim_others = False
        self.font = QFont('Segoe UI', 8)
        self.fm = QFontMetrics(self.font)
        self.setZValue(5)

    def set_pulses(self, pulses, rows=True):
        pulses = sorted(pulses, key=lambda p: (p['t0'], p['t1']))
        self.pulses = pulses
        n = len(pulses)
        self.ids = np.array([p['id'] for p in pulses], dtype=np.int64)
        self.t0 = np.array([p['t0'] for p in pulses], dtype=np.float64)
        self.t1 = np.array([p['t1'] for p in pulses], dtype=np.float64)
        self.maxlen = float(np.max(self.t1 - self.t0)) if n else 0.
        self.texts = [p.get('text', '') for p in pulses]
        self.colors = [p.get('color', '#cccccc') for p in pulses]
        self.brushes = [QBrush(qcolor(c, 115)) for c in self.colors]
        self.pens = [QPen(qcolor(c, 230), 1) for c in self.colors]
        for pen in self.pens:
            pen.setCosmetic(True)
        self.trunc = np.array([bool(p.get('truncated')) for p in pulses])
        # sub-rows for overlapping bars
        self.row = np.zeros(n, dtype=np.int64)
        self.nrows = 1
        if rows and n:
            ends = []
            for i in range(n):
                placed = False
                for r, e in enumerate(ends):
                    if self.t0[i] >= e - 1e-9:
                        ends[r] = self.t1[i]
                        self.row[i] = r
                        placed = True
                        break
                if not placed:
                    ends.append(self.t1[i])
                    self.row[i] = len(ends) - 1
            self.nrows = max(1, len(ends))
        self.prepareGeometryChange()
        self.update()

    # --- hit testing (t in view units, tol in view units) ---
    def hit(self, t, tol=0.):
        if not len(self.t0):
            return None
        lo = bisect.bisect_left(self.t0.tolist(), t - self.maxlen - tol)
        hi = bisect.bisect_right(self.t0.tolist(), t + tol)
        best, bestw = None, None
        for i in range(lo, hi):
            if self.t0[i] - tol <= t <= self.t1[i] + tol:
                w = self.t1[i] - self.t0[i]
                if best is None or w < bestw:     # prefer the narrowest
                    best, bestw = i, w
        return int(self.ids[best]) if best is not None else None

    def index_of(self, pid):
        k = np.flatnonzero(self.ids == pid)
        return int(k[0]) if k.size else None

    def visible_range(self, l, r):
        i0 = int(np.searchsorted(self.t0, l - self.maxlen))
        i1 = int(np.searchsorted(self.t0, r, side='right'))
        return i0, i1

    def paint(self, p, *args):
        n = len(self.t0)
        if not n:
            return
        l, r, tr, m11, dx, m22, dy, xl, xr = self._dev(p)
        i0, i1 = self.visible_range(l, r)
        if i1 <= i0:
            return
        p.resetTransform()
        # y band for the bars, in device px
        ya, yb = m22 * self.y0 + dy, m22 * self.y1 + dy
        if ya > yb:
            ya, yb = yb, ya
        if self.style == 'marker':
            ya, yb = ya, ya + max(6., (yb - ya) * 0.22)
        rowh = (yb - ya) / self.nrows
        x0 = self.t0[i0:i1] * m11 + dx
        x1 = self.t1[i0:i1] * m11 + dx
        x0c = np.clip(x0, xl - 2., xr + 2.)
        x1c = np.clip(x1, xl - 2., xr + 2.)
        rows = self.row[i0:i1]
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        dim = self.dim_others and (self.selected or self.related)
        for k in range(i1 - i0):
            i = i0 + k
            pid = int(self.ids[i])
            w = max(x1c[k] - x0c[k], 1.)
            y = ya + rows[k] * rowh
            rect = QRectF(x0c[k], y + 1., w, max(rowh - 2., 2.))
            brush, pen = self.brushes[i], self.pens[i]
            if pid in self.selected:
                c = QColor(brush.color())
                c.setAlpha(210)
                brush = QBrush(c)
            elif dim and pid not in self.related:
                c = QColor(brush.color())
                c.setAlpha(45)
                brush = QBrush(c)
                pc = QColor(pen.color())
                pc.setAlpha(90)
                pen = QPen(pc, 1)
                pen.setCosmetic(True)
            p.setBrush(brush)
            p.setPen(pen)
            if w < CLUSTER_PX:
                p.drawLine(QPointF(x0c[k], y + 1.), QPointF(x0c[k], y + rowh - 1.))
            elif w >= 6. and self.style == 'bar':
                p.drawRoundedRect(rect, 2.5, 2.5)
            else:
                p.drawRect(rect)
            if self.trunc[i]:
                hp = QPen(QColor('#ff5555'), 1, Qt.PenStyle.DashLine)
                hp.setCosmetic(True)
                p.setPen(hp)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(rect)
        # outlines for selection / relation / hover
        for i in range(i0, i1):
            pid = int(self.ids[i])
            if pid in self.selected or pid in self.related or pid == self.hovered:
                k = i - i0
                w = max(x1c[k] - x0c[k], 2.)
                y = ya + rows[k] * rowh
                rect = QRectF(x0c[k], y + 1., w, max(rowh - 2., 2.))
                if pid in self.selected:
                    pen = QPen(QColor('#ffffff'), 2)
                elif pid == self.hovered:
                    pen = QPen(QColor('#ffffff'), 1)
                else:
                    pen = QPen(QColor(255, 255, 255, 150), 1,
                               Qt.PenStyle.DotLine)
                pen.setCosmetic(True)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(rect)
                if pid in self.selected and w < SELECT_CARET_PX:
                    # too thin to see the outline: a caret over it
                    xc = x0c[k] + 0.5 * w
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QBrush(QColor('#ffffff')))
                    p.drawPolygon(QPolygonF([QPointF(xc - 4., y), QPointF(xc + 4., y),
                                             QPointF(xc, y + 6.)]))
        # labels
        if self.labels_enabled and self.style == 'bar':
            p.setFont(self.font)
            p.setPen(QPen(QColor('#f2f2f2')))
            drawn = 0
            for k in range(i1 - i0):
                w = x1c[k] - x0c[k]
                if w < LABEL_MIN_PX:
                    continue
                i = i0 + k
                txt = self.texts[i]
                if not txt:
                    continue
                y = ya + rows[k] * rowh
                rect = QRectF(x0c[k] + 3., y, w - 6., rowh)
                el = self.fm.elidedText(txt, Qt.TextElideMode.ElideRight,
                                        int(w - 6.))
                p.drawText(rect, int(Qt.AlignmentFlag.AlignVCenter
                                     | Qt.AlignmentFlag.AlignLeft), el)
                drawn += 1
                if drawn >= MAX_LABELS:
                    break
        p.setTransform(tr)


# ---------------------------------------------------------------------------
# digital trace
# ---------------------------------------------------------------------------

class DigitalTraceItem(ViewItem):
    """Level trace from sorted edge times (each edge toggles)."""

    def __init__(self, edges, level0, tmax, color='#9a9a9a', high_label='',
                 low_label='', y_low=0.12, y_high=0.62, fill=True):
        super().__init__(0., tmax)
        self.edges = np.asarray(edges, dtype=np.float64)
        self.level0 = int(level0)
        self.color = color
        self.high_label, self.low_label = high_label, low_label
        self.y_low, self.y_high = y_low, y_high
        self.fill = fill
        self.pen = QPen(qcolor(color, 240), 1)
        self.pen.setCosmetic(True)
        self.brush = QBrush(qcolor(color, 60))
        self.font = QFont('Segoe UI', 7)
        self.setZValue(3)

    def level_at(self, t):
        n = int(np.searchsorted(self.edges, t, side='right'))
        return self.level0 ^ (n & 1)

    def nearest_edge(self, t):
        if not self.edges.size:
            return None
        k = int(np.searchsorted(self.edges, t))
        cands = self.edges[max(k - 1, 0):k + 1]
        return float(cands[np.argmin(np.abs(cands - t))])

    def paint(self, p, *args):
        l, r, tr, m11, dx, m22, dy, xl, xr = self._dev(p)
        wpx = xr - xl
        p.resetTransform()
        yl = m22 * self.y_low + dy
        yh = m22 * self.y_high + dy
        e = self.edges
        k0 = int(np.searchsorted(e, l))
        k1 = int(np.searchsorted(e, r, side='right'))
        lev0 = self.level0 ^ (k0 & 1)
        seg = e[k0:k1]
        if seg.size > 4 * wpx:
            # denser than pixels: draw a solid band (toggling faster than
            # the screen resolves), honest about the coverage
            p.setPen(self.pen)
            p.setBrush(QBrush(qcolor(self.color, 90)))
            xa = max(seg[0] * m11 + dx, xl - 1.)
            xb = min(seg[-1] * m11 + dx, xr + 1.)
            p.drawRect(QRectF(xa, min(yl, yh), xb - xa, abs(yh - yl)))
            p.setTransform(tr)
            return
        xs = np.concatenate(([l], seg, [r])) * m11 + dx
        xs = np.clip(xs, xl - 2., xr + 2.)
        levels = (lev0 ^ (np.arange(seg.size + 1) & 1))
        ys = np.where(levels, yh, yl)
        pts = []
        for i in range(xs.size - 1):
            pts.append(QPointF(xs[i], ys[i]))
            pts.append(QPointF(xs[i + 1], ys[i]))
        poly = QPolygonF(pts)
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        if self.fill:
            path = QPainterPath()
            path.moveTo(xs[0], yl)
            for pt in pts:
                path.lineTo(pt)
            path.lineTo(xs[-1], yl)
            path.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self.brush)
            p.drawPath(path)
        p.setPen(self.pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolyline(poly)
        # level labels at the left edge
        p.setFont(self.font)
        p.setPen(QPen(QColor(220, 220, 220, 170)))
        if self.high_label:
            p.drawText(QPointF(xl + 4., yh - 3.), self.high_label)
        if self.low_label:
            p.drawText(QPointF(xl + 4., yl - 3.), self.low_label)
        p.setTransform(tr)


# ---------------------------------------------------------------------------
# analog trace (min/max pyramid)
# ---------------------------------------------------------------------------

class MinMaxPyramid:
    def __init__(self, y, dt=1.0, base=4, min_len=2048):
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
        self.dt = float(dt)
        self.n = y.size
        self.levels = [(1, y, None)]
        lo = hi = y
        f = 1
        while lo.size > min_len:
            n = lo.size // base * base
            if n < base:
                break
            lo = lo[:n].reshape(-1, base).min(1)
            hi = hi[:n].reshape(-1, base).max(1)
            f *= base
            self.levels.append((f, lo, hi))
        self.amax = float(np.max(np.abs(y))) if y.size else 1.

    def view(self, t0, t1, px):
        spp = (t1 - t0) / self.dt / max(px, 1.)
        f, lo, hi = self.levels[0]
        for lvl in reversed(self.levels):
            if lvl[0] <= max(spp / 2., 1.):
                f, lo, hi = lvl
                break
        i0 = max(int(t0 / self.dt / f) - 1, 0)
        i1 = min(int(t1 / self.dt / f) + 2, lo.size)
        if i1 <= i0:
            return np.zeros(0), np.zeros(0)
        if hi is None:
            return np.arange(i0, i1) * self.dt, lo[i0:i1]
        x = np.repeat(np.arange(i0, i1) * f * self.dt, 2)
        y = np.empty(2 * (i1 - i0), np.float32)
        y[0::2] = lo[i0:i1]
        y[1::2] = hi[i0:i1]
        return x, y

    def value_at(self, t):
        i = int(round(t / self.dt))
        if 0 <= i < self.n:
            return float(self.levels[0][1][i])
        return None


class AnalogTraceItem(pg.PlotCurveItem):
    def __init__(self, samples, dt=1.0, color='#f0e442'):
        pen = QPen(qcolor(color, 120), 1)
        pen.setCosmetic(True)
        super().__init__(pen=pen, connect='all', skipFiniteCheck=True)
        self.pyr = MinMaxPyramid(samples, dt)
        self.setZValue(3)

    def refresh(self, t0, t1, px):
        x, y = self.pyr.view(t0, t1, px)
        self.setData(x, y)


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------

class EventMarkerItem(ViewItem):
    def __init__(self, events, tmax):
        super().__init__(0., tmax)
        self.events = sorted(events, key=lambda e: e['t'])
        self.t = np.array([e['t'] for e in self.events], dtype=np.float64)
        self.ids = np.array([e['id'] for e in self.events], dtype=np.int64)
        self.font = QFont('Segoe UI', 7)
        self.fm = QFontMetrics(self.font)
        self.selected = set()
        self.hovered = None
        self.setZValue(7)

    def hit(self, t, tol):
        if not self.t.size:
            return None
        k = int(np.searchsorted(self.t, t))
        best = None
        for i in (k - 1, k):
            if 0 <= i < self.t.size and abs(self.t[i] - t) <= tol:
                if best is None or abs(self.t[i] - t) < abs(self.t[best] - t):
                    best = i
        return int(self.ids[best]) if best is not None else None

    def paint(self, p, *args):
        if not self.t.size:
            return
        l, r, tr, m11, dx, m22, dy, xl, xr = self._dev(p)
        k0 = int(np.searchsorted(self.t, l))
        k1 = int(np.searchsorted(self.t, r, side='right'))
        if k1 <= k0:
            return
        p.resetTransform()
        ytop = min(m22 * 0. + dy, m22 * 1. + dy)
        ybot = max(m22 * 0. + dy, m22 * 1. + dy)
        p.setFont(self.font)
        last_label_x = -1e9
        for i in range(k0, k1):
            ev = self.events[i]
            x = self.t[i] * m11 + dx
            col = QColor('#ff7f7f') if ev.get('approx') else QColor('#ffd1dc')
            sel = int(self.ids[i]) in self.selected or int(self.ids[i]) == self.hovered
            pen = QPen(col, 2 if sel else 1, Qt.PenStyle.DashLine
                       if ev.get('approx') else Qt.PenStyle.SolidLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.drawLine(QPointF(x, ytop + 2.), QPointF(x, ybot - 2.))
            tri = QPolygonF([QPointF(x - 5., ytop + 1.), QPointF(x + 5., ytop + 1.),
                             QPointF(x, ytop + 8.)])
            p.setBrush(QBrush(col))
            p.drawPolygon(tri)
            label = ev.get('label', '')
            if label and x - last_label_x > 16:
                w = self.fm.horizontalAdvance(label) + 4
                p.setPen(QPen(col))
                p.drawText(QPointF(x + 6., ytop + 12.), label)
                last_label_x = x + w
        p.setTransform(tr)


# ---------------------------------------------------------------------------
# background bands
# ---------------------------------------------------------------------------

BAND_STYLES = {
    'shot_even': (QColor(255, 255, 255, 0), None),
    'shot_odd': (QColor(255, 255, 255, 10), None),
    'handoff': (QColor(140, 140, 140, 38), Qt.BrushStyle.BDiagPattern),
    'handback': (QColor(140, 140, 140, 38), Qt.BrushStyle.FDiagPattern),
    'not_simulated': (QColor(255, 60, 60, 60), Qt.BrushStyle.DiagCrossPattern),
}


class BandItem(ViewItem):
    """Shot shading + framing hatches, with optional labels (header lane)."""

    def __init__(self, shots, framing, tmax, labels=False):
        super().__init__(0., tmax)
        self.shots = shots
        self.framing = framing
        self.labels = labels
        self.font = QFont('Segoe UI', 8)
        self.fontb = QFont('Segoe UI', 8, QFont.Weight.Bold)
        self.fm = QFontMetrics(self.font)
        self.setZValue(-5)

    def paint(self, p, *args):
        l, r, tr, m11, dx, m22, dy, xl, xr = self._dev(p)
        p.resetTransform()
        ytop = min(dy, m22 + dy)
        ybot = max(dy, m22 + dy)
        h = ybot - ytop
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        for s in self.shots:
            if s['t1'] < l or s['t0'] > r:
                continue
            xa = max(s['t0'] * m11 + dx, xl - 1.)
            xb = min(s['t1'] * m11 + dx, xr + 1.)
            col, _ = BAND_STYLES['shot_odd' if s['index'] % 2 else 'shot_even']
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            p.drawRect(QRectF(xa, ytop, xb - xa, h))
            pen = QPen(QColor(255, 255, 255, 70), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.drawLine(QPointF(xa, ytop), QPointF(xa, ybot))
            if self.labels:
                p.setFont(self.fontb)
                p.setPen(QPen(QColor('#e8e8e8')))
                txt = s.get('label', f"shot {s['index']}")
                w = xb - xa - 8.
                if w > 30:
                    el = self.fm.elidedText(txt, Qt.TextElideMode.ElideRight, int(w))
                    p.drawText(QRectF(max(xa, xl) + 4., ytop, w, h),
                               int(Qt.AlignmentFlag.AlignVCenter
                                   | Qt.AlignmentFlag.AlignLeft), el)
        for b in self.framing:
            if b['t1'] < l or b['t0'] > r:
                continue
            xa = max(b['t0'] * m11 + dx, xl - 1.)
            xb = min(b['t1'] * m11 + dx, xr + 1.)
            col, pattern = BAND_STYLES.get(b['kind'], BAND_STYLES['handoff'])
            brush = QBrush(col)
            if pattern is not None:
                brush = QBrush(col, pattern)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(brush)
            p.drawRect(QRectF(xa, ytop, max(xb - xa, 1.), h))
            if self.labels and b['kind'] == 'not_simulated':
                p.setFont(self.font)
                p.setPen(QPen(QColor('#ff8080')))
                p.drawText(QRectF(xa + 4., ytop, max(xb - xa - 8., 0.), h),
                           int(Qt.AlignmentFlag.AlignVCenter
                               | Qt.AlignmentFlag.AlignLeft),
                           self.fm.elidedText(b.get('label', 'not simulated'),
                                              Qt.TextElideMode.ElideRight,
                                              int(max(xb - xa - 8., 0.))))
        p.setTransform(tr)


# ---------------------------------------------------------------------------
# row guides + time grid
# ---------------------------------------------------------------------------

class GridItem(ViewItem):
    """Per-lane guides: a faint row background (alternating), a separator
    along the lane's bottom edge, and vertical time-grid lines at the
    positions the time axis ticks (major brighter than minor)."""

    def __init__(self, tick_fn, tmax, odd=False, header=False):
        super().__init__(0., tmax)
        self.tick_fn = tick_fn        # (l, r, px) -> [(spacing, [values])]
        self.odd = odd
        self.header = header
        self.setZValue(-10)

    def paint(self, p, *args):
        l, r, tr, m11, dx, m22, dy, xl, xr = self._dev(p)
        p.resetTransform()
        ytop = min(dy, m22 + dy)
        ybot = max(dy, m22 + dy)
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        p.setPen(Qt.PenStyle.NoPen)
        if self.header:
            p.setBrush(QBrush(QColor(255, 255, 255, 14)))
        elif self.odd:
            p.setBrush(QBrush(QColor(255, 255, 255, 6)))
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(xl, ytop, xr - xl, ybot - ytop))
        # time grid
        try:
            levels = self.tick_fn(l, r, xr - xl)
        except Exception:
            levels = []
        for k, (spacing, values) in enumerate(levels[:2]):
            alpha = 34 if k == 0 else 16
            pen = QPen(QColor(255, 255, 255, alpha), 1)
            pen.setCosmetic(True)
            p.setPen(pen)
            for v in values:
                x = v * m11 + dx
                if xl <= x <= xr:
                    p.drawLine(QPointF(x, ytop), QPointF(x, ybot))
        # separator
        pen = QPen(QColor(255, 255, 255, 40), 1)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.drawLine(QPointF(xl, ybot - 1.), QPointF(xr, ybot - 1.))
        p.setTransform(tr)


# ---------------------------------------------------------------------------
# gap dimensions
# ---------------------------------------------------------------------------

class GapDimensionItem(ViewItem):
    """Between consecutive bars of a lane: <-- 2.000 µs -->."""

    def __init__(self, bars: PulseBarItem, tmax, y=0.5):
        super().__init__(0., tmax)
        self.bars = bars
        self.y = y
        self.font = QFont('Segoe UI', 7)
        self.fm = QFontMetrics(self.font)
        self.enabled = True
        self.setZValue(6)

    def paint(self, p, *args):
        if not self.enabled or len(self.bars.t0) < 2:
            return
        l, r, tr, m11, dx, m22, dy, xl, xr = self._dev(p)
        p.resetTransform()
        y = m22 * self.y + dy
        t0, t1 = self.bars.t0, self.bars.t1
        # consecutive in time order: gap from the end of i to the start of i+1
        ends = np.maximum.accumulate(t1)
        i0 = max(int(np.searchsorted(t0, l)) - 1, 0)
        i1 = min(int(np.searchsorted(t0, r, side='right')) + 1, t0.size)
        p.setFont(self.font)
        pen = QPen(QColor(200, 200, 200, 160), 1)
        pen.setCosmetic(True)
        for i in range(i0, i1 - 1):
            a, b = ends[i], t0[i + 1]
            if b <= a:
                continue
            if b < l or a > r:
                continue
            xa, xb = a * m11 + dx, b * m11 + dx
            if xb - xa < GAP_MIN_PX or xa < xl - 1 or xb > xr + 1:
                continue
            txt = fmt_duration(b - a)
            w = self.fm.horizontalAdvance(txt)
            mid = (xa + xb) / 2.
            p.setPen(pen)
            p.drawLine(QPointF(xa + 1., y), QPointF(mid - w / 2. - 3., y))
            p.drawLine(QPointF(mid + w / 2. + 3., y), QPointF(xb - 1., y))
            # arrow heads
            p.drawLine(QPointF(xa + 1., y), QPointF(xa + 5., y - 3.))
            p.drawLine(QPointF(xa + 1., y), QPointF(xa + 5., y + 3.))
            p.drawLine(QPointF(xb - 1., y), QPointF(xb - 5., y - 3.))
            p.drawLine(QPointF(xb - 1., y), QPointF(xb - 5., y + 3.))
            p.setPen(QPen(QColor('#dddddd')))
            p.drawText(QPointF(mid - w / 2., y + 4.), txt)
        p.setTransform(tr)


# ---------------------------------------------------------------------------
# cursors
# ---------------------------------------------------------------------------

class CursorLine(pg.InfiniteLine):
    """A draggable vertical cursor that snaps to the nearest edge (within
    a few pixels) unless Alt is held. `edges` is a sorted array shared by
    all lanes; `on_moved(x)` is called with the (snapped) position."""

    def __init__(self, name, color, edges, on_moved):
        pen = QPen(QColor(color), 1)
        pen.setCosmetic(True)
        hpen = QPen(QColor(color), 2)
        hpen.setCosmetic(True)
        super().__init__(angle=90, movable=True, pen=pen, hoverPen=hpen,
                         label=name, labelOpts={'position': 0.92,
                                                'color': QColor(color),
                                                'movable': False})
        self.name = name
        self.edges = np.asarray(edges, dtype=np.float64)
        self.on_moved = on_moved
        self._sync = False
        self.setZValue(20)
        self.sigDragged.connect(self._dragged)
        self.sigPositionChangeFinished.connect(self._dragged)

    def snap(self, x):
        mods = QApplication.keyboardModifiers()
        if mods & Qt.KeyboardModifier.AltModifier or not self.edges.size:
            return x
        vb = self.getViewBox()
        tol = 6. * (vb.viewPixelSize()[0] if vb is not None else 0.)
        k = int(np.searchsorted(self.edges, x))
        cands = [e for e in self.edges[max(k - 1, 0):k + 1] if abs(e - x) <= tol]
        return min(cands, key=lambda e: abs(e - x)) if cands else x

    def _dragged(self, *args):
        if self._sync:
            return
        x = self.snap(self.value())
        self._sync = True
        try:
            self.setValue(x)
        finally:
            self._sync = False
        self.on_moved(x)
