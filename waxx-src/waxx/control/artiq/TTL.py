from artiq.language.core import at_mu, delay_mu, delay, kernel, now_mu
from artiq.language.types import TInt64
from artiq.coredevice.ttl import TTLOut, TTLInOut
import artiq.experiment
import numpy as np

from waxx.control.exceptions import TriggerTimeout

T_LINE_TRIGGER_SAMPLE_INTERVAL = 1/60 * 1.05
T_LINE_TRIGGER_RTIO_DELAY = 100.e-6
dv = np.int64(-1)

class TTL():
    def __init__(self,ch):
        self.ch = ch
        self.name = f'ttl{self.ch}'
        self.key = ""

    def get_device(self,expt:artiq.experiment.EnvExperiment):
        self.ttl_device = expt.get_device(self.name)

class TTL_OUT(TTL):
    def __init__(self,ch):
        super().__init__(ch)
        self.ttl_device = TTLOut
        self.state = 0

    @kernel
    def on(self):
        self.ttl_device.on()
        self.state = 1

    @kernel
    def off(self):
        self.ttl_device.off()
        self.state = 0

    @kernel
    def pulse(self,t):
        self.ttl_device.on()
        delay(t)
        self.ttl_device.off()

    @kernel
    def pulse_mu(self,t_mu,compensate_timeline=True):
        t = np.int64(t_mu)
        self.ttl_device.on()
        delay_mu(t)
        self.ttl_device.off()
        if compensate_timeline:
            delay_mu(-t)

    @kernel
    def set_state(self,state=-1):
        self.state = state if state != -1 else self.state
        if self.state == 1:
            self.on()
        else:
            self.off()

class TTL_IN(TTL):
    """A TTLInOut used as an input.

    The gate is opened and closed explicitly (arm / wait_for_edge) instead
    of being scheduled as a fixed window: a caller arms, schedules whatever
    provokes the edge (a trigger pulse to another device), then waits. The
    input is sensitive without a gap from arm() until the edge has come, and
    no assumption about how soon (or late) the edge comes is baked into the
    timeline -- only a sanity deadline for the wait.
    """
    def __init__(self,ch):
        super().__init__(ch)
        self.ttl_device = TTLInOut

        # timeline position where the last gate was closed (clear_input_events)
        self.t_input_gate_end = np.int64(0)

    @kernel
    def arm(self):
        """Opens the rising-edge gate at the cursor; the cursor does not move
        and no close is scheduled. wait_for_edge() closes the gate once the
        edge (or its deadline) has come, so an edge arriving any time after
        this call is caught, however long it takes. The cursor must have the
        usual RTIO slack, as for any output event.

        Any event still queued from an earlier gate is discarded first
        (without blocking), so the edge wait_for_edge() returns is always
        one that came after this call."""
        self.clear_input_events(np.int64(0))
        self.ttl_device._set_sensitivity(1)

    @kernel
    def wait_for_edge(self, t_delay, t_timeout) -> TInt64:
        """Blocks until the first rising edge since arm(), or until t_timeout
        after the current cursor. Parks the cursor t_delay after the edge
        (after the deadline if none came), closes the gate there and returns
        the edge timestamp, or -1 on timeout.

        t_delay is the slack the caller's next events get: it must cover the
        kernel CPU's reaction time (an edge is known ~3 us after it happens
        and each event submitted then costs ~1 us), so keep it >= 10 us.

        The cursor is placed with at_mu rather than delay() on purpose: this
        ARTIQ's I/O-delay estimator crashes on a kernel whose delay depends
        on an argument that some caller leaves at its default.
        """
        t_deadline = now_mu() + self.ttl_device.core.seconds_to_mu(t_timeout)
        t_edge = self.ttl_device.timestamp_mu(t_deadline)
        t_resume = t_deadline
        if t_edge >= 0:
            t_resume = t_edge
        at_mu(t_resume + self.ttl_device.core.seconds_to_mu(t_delay))
        self.ttl_device._set_sensitivity(0)
        self.t_input_gate_end = now_mu()
        return t_edge

    @kernel
    def wait_for_line_trigger(self,
                              t_window=T_LINE_TRIGGER_SAMPLE_INTERVAL,
                              t_delay=T_LINE_TRIGGER_RTIO_DELAY):
        """Parks the cursor t_delay after the next rising edge on this input.

        The gate stays open for at most t_window (default just over one
        60 Hz period, so a line trigger always lands in it). No edge by then
        raises TriggerTimeout: a missing trigger reads as a missing trigger,
        not as the underflow the old re-gating loop produced.
        """
        self.arm()
        if self.wait_for_edge(t_delay, t_window) < 0:
            raise TriggerTimeout("no rising edge on ttl{0} within the gate window",
                                 np.int64(self.ch))

    @kernel
    def clear_input_events(self,t_end=dv):
        if t_end == dv:
            t_end = self.t_input_gate_end
        while True:
            t_other_edge = self.ttl_device.timestamp_mu(t_end)
            if t_other_edge == -1:
                break

class DummyTTL(TTL):
    def __init__(self):
        super().__init__(ch=0)

    @kernel
    def get_device(self,expt:artiq.experiment.EnvExperiment):
        return TTLOut

    @kernel
    def on(self):
        pass

    @kernel
    def off(self):
        pass

    @kernel
    def pulse(self,t):
        pass
