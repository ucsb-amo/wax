"""Kernel-raisable exceptions shared across waxx device wrappers and the
scan loop. Plain Exception subclasses only: the ARTIQ compiler embeds them
with a constant message plus up to three int64 parameters ({0}..{2}).
"""


class TriggerTimeout(Exception):
    """A gated input wait closed its window without seeing an edge.

    Raised from the kernel by TTL_IN.wait_for_line_trigger
    (waxx.control.artiq.TTL) when its window closes with no edge, and by
    callers of TTL_IN.wait_for_edge whose shot cannot go on without the
    edge (the OPX hand-back wait in kexp). The scan loop
    (waxx.base.scanner.Scanner.scan) treats it like an RTIOUnderflow: the
    shot is abandoned, cleanup_scan_kernel runs and the scan aborts -- or it
    is re-raised under scan(raise_underflow=True).
    """
    pass
