"""The exposure delay a triggered camera stream's kernel fires its TTL early
by, decided at the run's claim and handed to the kernel when it starts.

A triggered stream built without a connection (``connect=False``: every
stream of a kexp run since queue phase 2) is connected by the run's claim,
which runs after the kernel was compiled for a warm-launched run and after
the diagnostic frame's ``bind`` for every run. The camera's own exposure
start delay -- what it reports once our trigger settings are in -- is known
only then. ARTIQ bakes the host value of every attribute a kernel reads into
the compiled kernel, and the claim must not change such an attribute (it is
written back at the kernel's end anyway). So the kernel does not read the
delay from the stream: it reads ``ExposureDelayCell.value``, an attribute the
kernel itself assigns at its start:

* each such stream owns one cell, registered on the experiment
  (``register``) in a fixed order before the compile; the experiment's
  cell list always starts with one spare cell (``dummy_cell``), which
  images that take no frames point at, so the list is never empty;
* the cell holds a provisional value at compile time (the camera server's
  last report, else the configured value);
* the claim decides the value (the stream's ``exposure_delay_at_claim``)
  and touches no cell;
* at kernel start, ``Scanner.load_exposure_delays`` makes ONE RPC
  (``exposure_delays_for_kernel``) and writes every cell.

Every cell a kernel reads is a ``ExposureDelayCell`` (one type: a float
``value``), so every image / stream is the same kind of object to the
compiler whatever its camera.
"""

import numpy as np

# experiment attributes (the cell list is read by the kernel; the owner list
# is host-only and parallel to it)
CELLS_ATTR = "_exposure_delay_cells"
OWNERS_ATTR = "_exposure_delay_owners"


class ExposureDelayCell:
    """A kernel-mutable exposure delay (s): ``value`` is always a float."""

    def __init__(self, value=0.):
        self.value = float(value)

    def __repr__(self):
        return f"ExposureDelayCell({self.value!r})"


def _lists(expt):
    cells = getattr(expt, CELLS_ATTR, None)
    owners = getattr(expt, OWNERS_ATTR, None)
    if cells is None or owners is None or len(cells) != len(owners) or not cells:
        cells, owners = [ExposureDelayCell(0.)], [None]
        setattr(expt, CELLS_ATTR, cells)
        setattr(expt, OWNERS_ATTR, owners)
    return cells, owners


def init_cells(expt):
    """Give ``expt`` its cell list (the spare cell only) unless it has one."""
    _lists(expt)


def cells(expt):
    """The experiment's cells, in the order the kernel writes them."""
    return _lists(expt)[0]


def dummy_cell(expt):
    """The spare cell (index 0): what an image that takes no frames reads."""
    return _lists(expt)[0][0]


def register(expt, owner, cell):
    """Add ``owner``'s ``cell`` to the experiment's list (before the compile;
    once). ``owner`` answers ``_exposure_delay_for_kernel() -> (float, bool)``."""
    c, o = _lists(expt)
    if any(x is cell for x in c):
        return
    c.append(cell)
    o.append(owner)


def exposure_delays_for_kernel(expt):
    """The RPC's body: one float per cell (the value the kernel writes into
    it), then 1.0 when every owner's value is final (its claim has decided
    it) else 0.0. Never raises: an owner that fails keeps its cell's current
    value, and the result always has ``len(cells) + 1`` entries."""
    c, o = _lists(expt)
    out = np.empty(len(c) + 1, dtype=np.float64)
    final = True
    for i, (cell, owner) in enumerate(zip(c, o)):
        value = cell.value
        if owner is not None:
            try:
                v, decided = owner._exposure_delay_for_kernel()
                value = float(v)
                final = final and bool(decided)
            except Exception as e:
                final = False
                print(f"[exposure delay] !! {getattr(owner, 'label', owner)}: no value for "
                      f"the kernel ({e!r}); it keeps {value * 1e6:.2f} us")
        out[i] = float(value)
    out[-1] = 1.0 if final else 0.0
    return out
