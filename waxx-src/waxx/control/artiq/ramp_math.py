"""Coefficient setup for the ramp trajectories used around the lab.

Every ramp shape here is a cheap recurrence: a handful of coefficients are
computed once up front, and each step of the loop costs an add and a multiply
with no exp/sqrt/pow inside the loop. That matters because the Kasli CPU has no
hardware FPU -- soft-float transcendentals in a ramp loop eat the per-step time
budget.

The coefficient setup is the part that is easy to get subtly wrong (the
off-by-one in n, the sign of tau, the 1 - E normalisation), so it lives here
once and is shared by every implementation: DAC_CH's single-channel ramps and
PaintedBeam's power + painting co-ramps.

Endpoint convention
-------------------
n is the number of steps, spaced dt = t/n apart, so the loop writes at times
i*dt for i = 0 .. n-1 and the caller lands the timeline on t with a final
at_mu(). linear/exponential/adiabatic reach v_end at i = n-1 (one step before
the end of the ramp, where the value then holds for dt). cubic is parameterised
to reach v_end at i = n instead, but its slope is zero there, so v(n-1) falls
short of v_end only by O(1/n**2) -- below a DAC LSB for any sane n.
"""

from numpy import exp, sqrt

from artiq.experiment import portable
from artiq.language.types import TFloat, TTuple

__all__ = ["linear_step", "cubic_coeffs", "exponential_coeffs",
           "adiabatic_coeffs"]


@portable
def linear_step(v_start, v_end, n) -> TFloat:
    """Per-step increment of a linear ramp.

    Step i is  v = v_start + i*dv,  reaching v_end at i = n-1.

    Returns:
        TFloat: dv, the per-step voltage increment.
    """
    return (v_end - v_start) / (n - 1)


@portable
def cubic_coeffs(t, v_start, v_end, n) -> TTuple([TFloat, TFloat]):
    """Coefficients of a smoothstep (zero slope at both ends) ramp.

    Step i is  v = Adt3*i**3 + Bdt2*i**2 + v_start.

    Args:
        t (float): ramp duration (s).
        v_start (float): starting voltage (V).
        v_end (float): final voltage (V).
        n (int): number of steps.

    Returns:
        TTuple([TFloat, TFloat]): (Adt3, Bdt2), the cubic and quadratic
        coefficients with the step spacing already folded in.
    """
    dt = t / n
    Adt3 = -2. * (v_end - v_start) / t**3 * dt**3
    Bdt2 = 3. * (v_end - v_start) / t**2 * dt**2
    return Adt3, Bdt2


@portable
def exponential_coeffs(t, v_start, v_end, n, tau) -> TTuple([TFloat, TFloat, TFloat]):
    """Coefficients of an exponential ramp of time constant tau.

        v(s) = v_end + (v_start - v_end) * (exp(-s/tau) - E) / (1 - E)
        E    = exp(-t/tau)

    i.e. a plain exponential, rescaled so it lands exactly on v_end at s = t
    rather than only asymptoting to it.

    Step i is  v = v_end + a*(e - e_end),  with e starting at 1. and carried
    forward as  e *= k.  exp() is evaluated twice here and never in the loop.

    Args:
        t (float): ramp duration (s).
        v_start (float): starting voltage (V).
        v_end (float): final voltage (V).
        n (int): number of steps.
        tau (float): time constant (s). Negative tau moves fast at the start
            and slows as it approaches v_end; positive tau flips the curvature
            (slow start, fast finish). Small |tau| is a sharper corner;
            |tau| >> t is a linear ramp, so use linear_step there instead.

    Returns:
        TTuple([TFloat, TFloat, TFloat]): (a, k, e_end).
    """
    # tau = 0 divides by zero; |tau| >> t sends the 1 - E normalisation to 0/0,
    # and is a linear ramp to well within a DAC LSB anyway
    if (tau == 0.) or (tau > 1000. * t) or (tau < -1000. * t):
        raise ValueError('exponential ramp needs 0 < |tau| < 1000*t')

    e_end = exp(-t / tau)
    k = exp(-(t / (n - 1)) / tau)   # per-step factor, k**(n-1) == e_end
    a = (v_start - v_end) / (1. - e_end)
    return a, k, e_end


@portable
def adiabatic_coeffs(v_start, v_end, n, v_offset) -> TTuple([TFloat, TFloat]):
    """Coefficients of a constant-adiabaticity ramp.

    For a dipole trap U ~ P and omega ~ sqrt(U) ~ sqrt(P), so the adiabaticity
    parameter is

        eps = |d(omega)/dt| / omega**2 = |d(1/omega)/dt|

    Holding eps constant therefore means 1/omega -- and hence 1/sqrt(P) -- is
    LINEAR in time. With the PID setpoint linear in power, P ~ (v - v_offset),
    the exact constant-eps trajectory is

        u(s) = u0 + (uf - u0)*s ,   u = 1/sqrt(v - v_offset) ,   s = t'/t
        v(s) = 1/u(s)**2 + v_offset

    The linear-in-u form is the fast one: step i is  v = 1/(u*u) + v_offset
    with u carried forward as  u += du.  One add and one divide per step, no
    sqrt inside the loop.

    Note that v_start -> v_offset (zero power) is unreachable: at constant eps
    it takes infinite time to leave omega = 0. Start from a small but finite
    setpoint -- the first bit of the ramp is non-adiabatic no matter what.

    Args:
        v_start (float): starting setpoint (V). Must be above v_offset.
        v_end (float): final setpoint (V). Must be above v_offset.
        n (int): number of steps.
        v_offset (float): setpoint at zero optical power (V). Optical power is
            taken to be proportional to (v - v_offset).

    Returns:
        TTuple([TFloat, TFloat]): (u, du), the initial value and per-step
        increment of u = 1/sqrt(v - v_offset).
    """
    w0 = v_start - v_offset
    wf = v_end - v_offset
    if (w0 <= 0.) or (wf <= 0.):
        raise ValueError('adiabatic ramp cannot go to zero power')

    u = 1. / sqrt(w0)
    du = (1. / sqrt(wf) - u) / (n - 1)
    return u, du
