"""Time formatting for the viewer: everything is ns internally."""

import numpy as np
import pyqtgraph as pg


def unit_for(span_ns):
    span = abs(float(span_ns))
    if span >= 2e6:
        return 'ms', 1e-6
    if span >= 2e3:
        return 'µs', 1e-3
    return 'ns', 1.0


def fmt_duration(ns, digits=4):
    ns = float(ns)
    unit, mult = unit_for(ns)
    v = ns * mult
    s = f"{v:.{digits}f}".rstrip('0').rstrip('.')
    if unit == 'ns':
        s = f"{v:.1f}".rstrip('0').rstrip('.')
    return f"{s} {unit}"


def fmt_time(ns, unit=None, digits=3):
    """An absolute time with a fixed unit (for readouts)."""
    ns = float(ns)
    if unit is None:
        unit, mult = unit_for(ns)
    else:
        mult = {'ns': 1.0, 'µs': 1e-3, 'ms': 1e-6}[unit]
    v = ns * mult
    if unit == 'ns':
        return f"{v:.1f} ns".replace('.0 ns', ' ns')
    return f"{v:.{digits}f} {unit}"


class TimeAxis(pg.AxisItem):
    """Bottom axis in ns with automatic ns/µs/ms tick labels and a
    movable origin."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin = 0.0
        self.origin_name = ''
        self.setStyle(tickTextOffset=4)

    def set_origin(self, t, name=''):
        self.origin = float(t)
        self.origin_name = name
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        # one unit for every tick level, from the visible span (the rule the
        # readouts use): picking it per level from the spacing gave the minor
        # ticks ns next to µs major ones, and the label took the last level's
        unit, mult = unit_for(self.range[1] - self.range[0])
        out = []
        for v in values:
            x = (v - self.origin) * mult
            if unit == 'ns':
                s = f"{x:.0f}"
            else:
                d = max(0, int(np.ceil(-np.log10(max(spacing * mult, 1e-12)))))
                s = f"{x:.{d}f}"
            out.append(s)
        lab = f"t ({unit})" + (f" from {self.origin_name}" if self.origin_name else '')
        if self.labelText != lab:
            self.setLabel(lab)
        return out
