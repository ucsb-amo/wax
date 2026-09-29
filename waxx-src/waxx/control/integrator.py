import numpy as np

from artiq.experiment import kernel, portable, TArray, TFloat, parallel
from artiq.language.core import now_mu, at_mu, delay, delay_mu

from waxx.control.artiq.TTL import TTL_OUT
from waxx.control.artiq.Sampler_CH import Sampler_CH, Sampler_Last_CH
from waxx.util.artiq.async_print import aprint
from waxx.config.expt_params import ExptParams

dv = -0.1
di = 0

T_RESET_RESPONSE_MU = 300
T_RESET_MU = 1500
T_INTEGRATOR_BEGIN_MU = 300
T_SETTLE_MU = 1000
T_ADC_CNVH_PULSE_MU = 30
T_ADC_CONV_MU = 450

class Integrator():
    """Gated integrator read by a Sampler channel.

    Readout window (all ExptParams, so every run file records them):
      p.t_integrator_gate_delay  gate opens this long after begin_integrate's
                                 cursor (the light command); the cursor stays there
      p.t_integrator_gate_extra  stop_and_settle closes the gate at
                                 cursor + gate_delay + gate_extra (cursor = pulse end)
      p.t_integrator_settle      gate close -> end of stop_and_settle (the sample)
    The waxx defaults are 0.7, 1.0, 4.0 us (the original fixed timing was 0, 0,
    1 us). 2026-09-29 tests (runs 83665-83675): the light reaches the integrator ~1 us after its
    command, and the output rings for ~3-4 us after the gate closes.
    """
    def __init__(self,
                 ttl_integrate=TTL_OUT,
                 ttl_reset=TTL_OUT,
                 sampler_ch=Sampler_Last_CH,
                 expt_params=ExptParams()):
        self.ttl_integrate = ttl_integrate # logic inverted -- on=not integrating, off=integrating
        # off = clearing (clear(), init(), and "held in clear" after a read);
        # on = released (begin_integrate)
        self.ttl_reset = ttl_reset
        if not isinstance(sampler_ch, Sampler_Last_CH):
            raise ValueError('For fast readout, use channel 6 or 7 of the sampler and assign as Sampler_Last_CH in sampler_id.py')
        self.sampler_ch = sampler_ch
        self.params = expt_params
        self.p = self.params

    @kernel
    def init(self):
        # I promise this makes sense
        self.ttl_integrate.on()
        self.ttl_reset.off()

    @kernel
    def begin_integrate(self, reset=True):
        """Sample aperture opens p.t_integrator_gate_delay after the current
        cursor (the light command); the cursor is left where it was.
        Pretriggers reset and gate open delay times.
        """
        t_light = now_mu()
        t_gate_open = t_light + np.int64(self.p.t_integrator_gate_delay * 1.e9)
        if reset:
            at_mu(t_gate_open - T_INTEGRATOR_BEGIN_MU - T_RESET_RESPONSE_MU - T_RESET_MU)
            self.ttl_reset.off()

        at_mu(t_gate_open - T_INTEGRATOR_BEGIN_MU - T_RESET_RESPONSE_MU)
        self.ttl_reset.on()

        at_mu(t_gate_open - T_INTEGRATOR_BEGIN_MU)
        self.ttl_integrate.off()

        at_mu(t_light)

    @kernel
    def stop_and_settle(self):
        """Call at the pulse end. Closes the gate p.t_integrator_gate_delay +
        p.t_integrator_gate_extra later and leaves the cursor
        p.t_integrator_settle after the close (where the sample goes)."""
        at_mu(now_mu() + np.int64((self.p.t_integrator_gate_delay
                                   + self.p.t_integrator_gate_extra) * 1.e9))
        self.ttl_integrate.on()
        delay_mu(np.int64(self.p.t_integrator_settle * 1.e9))

    @kernel
    def stop_and_sample(self) -> TFloat:
        """Advances the timeline cursor by gate_delay + gate_extra + settle,
        plus the Sampler conversion and readout.

        Returns:
            TFloat: The sampled value.
        """        
        self.stop_and_settle()
        v = self.sampler_ch.sample_single()
        return v
    
    @kernel
    def sample(self) -> TFloat:
        """
        Samples the current value of the integrator without any timing.
        """
        v = self.sampler_ch.sample_single()
        return v

    @kernel
    def clear(self, t=2*T_RESET_MU):
        self.ttl_reset.off()
        delay_mu(t)

    @kernel
    def reset(self, t=T_RESET_MU):
        self.ttl_reset.on()
        delay_mu(t)

