from artiq.experiment import kernel, rpc
from artiq.language.core import now_mu, at_mu
from artiq.coredevice.zotino import Zotino

from waxx.control.artiq.ramp_math import (linear_step, cubic_coeffs,
                                          exponential_coeffs, adiabatic_coeffs)
from waxx.util.artiq.async_print import aprint

dv = -10432.

class DAC_CH():
    def __init__(self,ch,dac_device=Zotino,max_v=dv):
        self.ch = ch
        self.dac_device = dac_device
        self.v = 0.
        if max_v == dv:
            self.max_v = 9.99
        else:
            self.max_v = max_v
        self.key = ""

    def set_errmessage(self):
        self.errmessage = f"Attempted to set dac ch {self.key} to a voltage > specified maximum voltage ({self.max_v:1.3f}) for that channel. DAC voltage was replaced by zero for these instances."

    @kernel
    def set(self,v=dv,load_dac=True):
        if self.ch < 0:
            return
        if v != dv:
            if v > self.max_v:
                self.v = 0.
                self.max_voltage_error()
            else:
                self.v = v
                
        self.dac_device.write_dac(self.ch,self.v)
        if load_dac:
            self.dac_device.load()

    @rpc(flags={'async'})
    def max_voltage_error(self):
        print(self.errmessage)

    @rpc(flags={'async'})
    def handle_dac_error(self,v):
        if ( v <= -10.) | (v >= 10.):
            print("DAC voltage must be between -10 and 10 V (noninclusive).")
        
    @kernel
    def load(self):
        if self.ch < 0:
            return
        self.dac_device.load()

    @kernel(flags={"fast-math"})
    def linear_ramp(self,t,v_start,v_end,n):
        """Ramp v_start -> v_end linearly.

        Uses absolute timestamps, so the ramp lasts exactly t.

        Args:
            t (float): ramp duration (s).
            v_start (float): starting voltage (V).
            v_end (float): final voltage (V).
            n (int): number of steps.
        """
        if self.ch < 0:
            return
        if (v_start > self.max_v) or (v_end > self.max_v):
            self.max_voltage_error()
            return

        dv_step = linear_step(v_start,v_end,n)
        dt_mu = self.dac_device.core.seconds_to_mu(t / n)

        t_mu = now_mu()
        for i in range(n):
            at_mu(t_mu)
            self.dac_device.write_dac(self.ch, v_start + i*dv_step)
            self.dac_device.load()
            t_mu += dt_mu
        at_mu(t_mu)
        self.v = v_end

    @kernel(flags={"fast-math"})
    def cubic_ramp(self,t,v_start,v_end,n):
        """Ramp v_start -> v_end on a smoothstep (zero slope at both ends).

        Uses absolute timestamps, so the ramp lasts exactly t.

        Args:
            t (float): ramp duration (s).
            v_start (float): starting voltage (V).
            v_end (float): final voltage (V).
            n (int): number of steps.
        """
        if self.ch < 0:
            return
        if (v_start > self.max_v) or (v_end > self.max_v):
            self.max_voltage_error()
            return

        Adt3, Bdt2 = cubic_coeffs(t,v_start,v_end,n)
        dt_mu = self.dac_device.core.seconds_to_mu(t / n)

        t_mu = now_mu()
        for i in range(n):
            at_mu(t_mu)
            self.dac_device.write_dac(self.ch, Adt3 * i**3 + Bdt2 * i**2 + v_start)
            self.dac_device.load()
            t_mu += dt_mu
        at_mu(t_mu)
        self.v = v_end

    @kernel(flags={"fast-math"})
    def exponential_ramp(self,t,v_start,v_end,n,tau=dv):
        """Ramp v_start -> v_end on an exponential of time constant tau.

            v(s) = v_end + (v_start - v_end) * (exp(s/tau) - E) / (1 - E)
            E    = exp(-t/tau)

        i.e. a plain exponential, rescaled so it lands exactly on v_end at s = t
        rather than only asymptoting to it.

        Computed iteratively: exp() is evaluated twice up front and the loop
        carries a running factor e *= k, so each step costs one multiply and one
        multiply-add. No exp/pow inside the loop. See ramp_math for the
        coefficient setup.

        Args:
            t (float): ramp duration (s).
            v_start (float): starting voltage (V).
            v_end (float): final voltage (V).
            n (int): number of steps.
            tau (float): time constant (s), default t/3. Negative tau moves fast
                at the start and slows as it approaches v_end; Positive tau
                flips the curvature (slow start, fast finish). Small |tau| is a
                sharper corner; |tau| >> t is just a linear ramp, so use
                linear_ramp there instead.
        """
        if self.ch < 0:
            return
        if tau == dv:
            tau = - t / 3.
        if (v_start > self.max_v) or (v_end > self.max_v):
            self.max_voltage_error()
            return

        a, k, e_end = exponential_coeffs(t,v_start,v_end,n,tau)

        e = 1.
        dt_mu = self.dac_device.core.seconds_to_mu(t / n)
        t_mu = now_mu()
        for i in range(n):
            at_mu(t_mu)
            self.dac_device.write_dac(self.ch, v_end + a * (e - e_end))
            self.dac_device.load()
            e *= k
            t_mu += dt_mu
        at_mu(t_mu)
        self.v = v_end

    # -------------------------------------------------------------------------
    # constant-adiabaticity ramps (ODT power ramp-up)
    # -------------------------------------------------------------------------
    # See ramp_math.adiabatic_coeffs for the derivation.

    @kernel(flags={"fast-math"})
    def adiabatic_ramp(self,t,v_start,v_end,n,v_offset=0.05):
        """Ramp v_start -> v_end at constant adiabaticity parameter, computing
        the trajectory on the core device.

        Drop-in sibling of linear_ramp / cubic_ramp. Like them, it uses
        absolute timestamps, so the ramp lasts exactly t.

        The Kasli CPU has no hardware FPU, so the per-step divide is soft-float.
        Below ~10 us/step use plan_adiabatic_ramp() + play() instead, which does
        no float math in the loop.

        Args:
            t (float): ramp duration (s).
            v_start (float): starting setpoint (V). Must be above v_offset.
            v_end (float): final setpoint (V). Must be above v_offset.
            n (int): number of steps.
            v_offset (float): setpoint at zero optical power (V). Optical power
                is taken to be proportional to (v - v_offset).
        """
        if self.ch < 0:
            return
        if (v_start > self.max_v) or (v_end > self.max_v):
            self.max_voltage_error()
            return

        u, du = adiabatic_coeffs(v_start,v_end,n,v_offset)
        dt_mu = self.dac_device.core.seconds_to_mu(t / n)

        t_mu = now_mu()
        for i in range(n):
            at_mu(t_mu)
            self.dac_device.write_dac(self.ch, 1. / (u * u) + v_offset)
            self.dac_device.load()
            u += du
            t_mu += dt_mu
        at_mu(t_mu)
        self.v = v_end