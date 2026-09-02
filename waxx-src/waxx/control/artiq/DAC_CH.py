from numpy import sqrt, exp

from artiq.experiment import kernel, rpc, delay
from artiq.language.core import now_mu, at_mu
from artiq.coredevice.zotino import Zotino

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
        if self.ch < 0:
            return
        v0 = v_start
        vf = v_end
        delta_v = (vf-v0)/(n-1)
        dt = t/n
        for i in range(n):
            self.set(v=v0+i*delta_v)
            delay(dt)

    @kernel(flags={"fast-math"})
    def cubic_ramp(self,t,v_start,v_end,n):
        if self.ch < 0:
            return
        v0 = v_start
        vf = v_end
        dt = t/n
        A = -2*(vf-v0)/t**3
        B =  3*(vf-v0)/t**2
        Adt3 = A * dt**3
        Bdt2 = B * dt**2
        for i in range(n):
            self.set(v = Adt3 * i**3 + Bdt2 * i**2 + v0)
            delay(dt)

    @kernel(flags={"fast-math"})
    def exponential_ramp(self,t,v_start,v_end,n,tau=dv):
        """Ramp v_start -> v_end on an exponential of time constant tau.

            v(s) = v_end + (v_start - v_end) * (exp(s/tau) - E) / (1 - E)
            E    = exp(-t/tau)

        i.e. a plain exponential, rescaled so it lands exactly on v_end at s = t
        rather than only asymptoting to it.

        Computed iteratively: exp() is evaluated twice up front and the loop
        carries a running factor e *= k, so each step costs one multiply and one
        multiply-add. No exp/pow inside the loop.

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
        # tau = 0 divides by zero; |tau| >> t sends the 1 - E normalisation to
        # 0/0, and is a linear ramp to well within a DAC LSB anyway
        if (tau == 0.) or (tau > 1000.*t) or (tau < -1000.*t):
            raise ValueError('exponential_ramp needs 0 < |tau| < 1000*t')

        e_end = exp(-t / tau)
        k = exp(-(t / (n - 1)) / tau)   # per-step factor, k**(n-1) == e_end
        a = (v_start - v_end) / (1. - e_end)

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
    #
    # For a dipole trap U ~ P and omega ~ sqrt(U) ~ sqrt(P), so the adiabaticity
    # parameter is
    #
    #     eps = |d(omega)/dt| / omega**2 = |d(1/omega)/dt|
    #
    # Holding eps constant therefore means 1/omega -- and hence 1/sqrt(P) -- is
    # LINEAR in time. With the PID setpoint linear in power, P ~ (v - v_offset),
    # the exact constant-eps trajectory over a ramp of duration t is
    #
    #     u(s) = u0 + (uf - u0)*s ,   u = 1/sqrt(v - v_offset) ,   s = t'/t
    #     v(s) = 1/u(s)**2 + v_offset
    #
    # equivalently, for v_offset = 0,
    #
    #     v(s) = v_start / (1 - s*(1 - sqrt(v_start/v_end)))**2
    #
    # The linear-in-u form is the fast one: one add and one divide per step, no
    # sqrt or pow inside the loop.
    #
    # Note that v_start -> v_offset (zero power) is unreachable: at constant eps
    # it takes infinite time to leave omega = 0. Start from a small but finite
    # setpoint -- the first bit of the ramp is non-adiabatic no matter what.

    @kernel(flags={"fast-math"})
    def adiabatic_ramp(self,t,v_start,v_end,n,v_offset=0.05):
        """Ramp v_start -> v_end at constant adiabaticity parameter, computing
        the trajectory on the core device.

        Drop-in sibling of linear_ramp / cubic_ramp. Uses absolute timestamps,
        so the ramp lasts exactly t (linear_ramp overruns by n*(t_spi+t_ldac)).

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
        w0 = v_start - v_offset
        wf = v_end - v_offset
        if (w0 <= 0.) or (wf <= 0.):
            raise ValueError('ramp cannot go to zero')
            return
        if (v_start > self.max_v) or (v_end > self.max_v):
            self.max_voltage_error()
            return

        u = 1. / sqrt(w0)
        du = (1. / sqrt(wf) - u) / (n - 1)
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