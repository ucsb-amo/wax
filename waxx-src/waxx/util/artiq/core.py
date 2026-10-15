"""The device_db's ``core`` device: ARTIQ's Core, with the run queue's warm-start gate.

A warm-launched job (queue phase 2) starts while the previous run still holds the
machine: it imports, prepares and compiles its kernel, then waits for the queue's GO
before it touches anything shared. This class is where the wait sits -- between the
compile and the first byte sent to the core device:

    upstream Core.run:   compile -> compile_and_upload_subkernels -> _run_compiled
    armed, first kernel: compile -> _wait_for_go -> compile_and_upload_subkernels -> _run_compiled
                                    (queue GO, then the experiment's claim)

* **Armed** iff the process was warm-launched (``queue_gate.is_warm()``, which reads and
  pops ``WAXX_QUEUE_WARM``) when the Core was constructed. Unarmed, every method is
  upstream's: ``run`` hands straight to ``super().run``.
* **One-shot.** Only the first kernel waits. GO and the claim are process-wide
  (``queue_gate``), so a second Core in the same process (``TTLPulser`` builds its own)
  never waits twice and never claims twice; after GO every kernel runs as upstream.
* **Fail-closed connection.** Armed, the comm object is a :class:`GatedCommKernel`
  whose ``open()`` raises :class:`GateNotPassed` before GO -- it never waits and never
  connects. On this lab's core device (artiq-zynq runtime) the TCP accept alone ends the
  running experiment's session, so a stray host kernel, ``precompile`` call or direct
  ``comm.open()`` in a waiting process must fail, not connect.
* **Writeback stays on.** The kernel is compiled with the normal ``compile`` (attribute
  writeback ON), not ``precompile`` (OFF): ``Expt.end`` builds the monitor's end state from
  the written-back host attributes. Consequence: the claim must not set any attribute a
  kernel reads -- the value is already baked into the ELF, and writeback would overwrite
  the host copy at kernel end.

What this depends on in upstream ARTIQ (``artiq/coredevice/core.py`` and
``comm_kernel.py``; pinned by ``tests/test_warm_core_sentinel.py``, re-audit this file
when that test fails):

* ``Core.run`` is exactly compile (with a ``set_result`` async RPC) ->
  ``compile_and_upload_subkernels`` -> ``_run_compiled``; :meth:`Core.run` below copies it
  with one line added.
* ``Core.__init__`` builds ``self.comm = CommKernel(host)`` (or ``CommKernelDummy`` for
  ``host=None``) and sets ``self.comm.core``; the constructor does not connect.
* ``CommKernel`` connects lazily: ``open()`` is the only place ``create_connection`` is
  called, and ``_read_header`` / ``_write_header`` -- the start of every exchange -- call it.
  ``CommKernel.__init__(host, port=1381)`` keeps ``host`` and ``port``.
* ``compile`` is not overridden here, so the startup profiler's wrapper on the base
  ``Core.compile`` (``waxx.util.profiling.startup_timer``) still times it.
"""
from artiq.coredevice.comm_kernel import CommKernel
from artiq.coredevice.core import Core as _ArtiqCore
from artiq.language.core import rpc

from waxx.util.device_state import queue_gate


class GateNotPassed(RuntimeError):
    """A warm-launched process tried to connect to the core device before the queue's GO."""


class GatedCommKernel(CommKernel):
    """``CommKernel`` that refuses to connect until the queue's GO (never waits).

    Installed only on an armed Core. ``open()`` is upstream's lazy connect, called at the
    start of every exchange, so this one check covers check_system_info, load, run,
    subkernel upload and any direct ``comm.open()``."""

    def open(self):
        if not queue_gate.go_passed():
            raise GateNotPassed(
                f"warm-launched process: no connection to the core device ({self.host}) "
                f"before the queue's GO -- opening one would end the running experiment")
        super().open()


class Core(_ArtiqCore):
    """``artiq.coredevice.core.Core`` with the warm-start gate (see the module docstring).

    Same constructor arguments as upstream; the device_db's ``core`` entry names this class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_gate_armed = queue_gate.is_warm()
        if self.queue_gate_armed and type(self.comm) is CommKernel:
            # replaces the (not yet connected) upstream comm; host=None keeps its dummy
            self.comm = GatedCommKernel(self.comm.host, self.comm.port)
            self.comm.core = self

    def run(self, function, args, kwargs):
        if not self.queue_gate_armed:
            return super().run(function, args, kwargs)
        if queue_gate.go_passed():
            # Safety net: a claim registered after GO (the first kernel ran before
            # finish_prepare) still runs before the next kernel. No-op once claimed.
            queue_gate.run_claim()
            return super().run(function, args, kwargs)

        # Upstream Core.run, plus the wait for GO between the compile and the upload.
        result = None
        @rpc(flags={"async"})
        def set_result(new_result):
            nonlocal result
            result = new_result
        embedding_map, kernel_library, symbolizer, demangler, subkernel_arg_types = \
            self.compile(function, args, kwargs, set_result)
        self._wait_for_go()
        self.compile_and_upload_subkernels(embedding_map, args, subkernel_arg_types)
        self._run_compiled(kernel_library, embedding_map, symbolizer, demangler)
        return result

    def _wait_for_go(self):
        """Block until the queue's GO, then run the experiment's claim (liveOD start,
        announce, hardware). May end the process: an abort before GO raises
        ``SystemExit(queue_gate.EXIT_WARM_ABORTED)`` with nothing claimed."""
        queue_gate.ensure_go()
        queue_gate.run_claim()
