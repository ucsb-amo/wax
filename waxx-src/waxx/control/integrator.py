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
    def __init__(self,
                 ttl_integrate=TTL_OUT,
                 ttl_reset=TTL_OUT,
                 sampler_ch=Sampler_Last_CH):
        self.ttl_integrate = ttl_integrate # logic inverted -- on=not integrating, off=integrating
        self.ttl_reset = ttl_reset # on=clearing integrator, off=not clearing integrator
        if not isinstance(sampler_ch, Sampler_Last_CH):
            raise ValueError('For fast readout, use channel 6 or 7 of the sampler and assign as Sampler_Last_CH in sampler_id.py')
        self.sampler_ch = sampler_ch

    @kernel
    def init(self):
        # I promise this makes sense
        self.ttl_integrate.on()
        self.ttl_reset.off()

    @kernel
    def begin_integrate(self, reset=True):
        """Sample aperture opens at current position of timeline cursor.
        Pretriggers reset and gate open delay times.
        """        
        t_gate_open = now_mu()
        if reset:
            at_mu(t_gate_open - T_INTEGRATOR_BEGIN_MU - T_RESET_RESPONSE_MU - T_RESET_MU)
            self.ttl_reset.off()

        at_mu(t_gate_open - T_INTEGRATOR_BEGIN_MU - T_RESET_RESPONSE_MU)
        self.ttl_reset.on()

        at_mu(t_gate_open - T_INTEGRATOR_BEGIN_MU)
        self.ttl_integrate.off()

        at_mu(t_gate_open)

    @kernel
    def stop_and_settle(self):
        self.ttl_integrate.on()
        delay_mu(T_SETTLE_MU)

    @kernel
    def stop_and_sample(self) -> TFloat:
        """Advances timeline cursor by 2200 ns.

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

