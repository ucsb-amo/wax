"""Shared ramping + painting machinery for painted optical traps.

A "painted" trap is one whose beam is swept across the atoms fast compared to
the trap period, so the atoms see a time-averaged potential whose width is set
by the sweep amplitude. Two knobs control the trap: the optical power (a PID
setpoint on a DAC) and the painting amplitude (another DAC voltage driving the
FM modulation depth of the AO/AOD source). Ramping the power alone changes both
the depth and the trap frequency; co-ramping the painting amplitude lets you
change the depth at fixed trap frequency.

That co-ramp is the same loop no matter which beam it drives, so it lives here.
A machine-specific subclass supplies the hardware -- which DAC carries the PID
setpoint, what has to happen to turn the beam on, whether there is a second
low-power servo to hand over to -- through four hooks, and supplies its own
ExptParams defaults through thin public wrappers.

Subclass contract
-----------------
Call _init_painting() from __init__, then implement:

    _ramp_begin(v_start, paint, dac_select, dt_mu)
    _set_pd(v, dac_select)      # must NOT load; the co-ramp loads once
    _load_pd(dac_select)
    _ramp_end(v_end, dac_select)

and wrap each _ramp_* core in a public method that resolves ExptParams
defaults. The defaults cannot be resolved here: they are per-beam parameter
names, they are scanned per shot by xvar(), and the ARTIQ compiler has no
dynamic getattr -- so they must be read by name, in-kernel, at call time.

Simultaneous loads
------------------
Every step sets the power DAC and the painting DAC with load_dac=False and then
issues a single load(). On a Zotino, load() is a device-wide LDAC strobe, so
both channels move on the same edge. Setting each with load_dac=True instead
would make the painting amplitude lead or lag the power by one SPI transaction,
which at short step times is a real trap-frequency glitch.
"""

from artiq.coredevice.core import Core
from artiq.experiment import kernel, portable, TFloat
from artiq.language.core import now_mu, at_mu

from waxx.control.artiq.DAC_CH import DAC_CH
from waxx.control.artiq.ramp_math import (linear_step, cubic_coeffs,
                                          exponential_coeffs, adiabatic_coeffs)

# which power servo a ramp drives, for beams that have more than one
DAC_PRIMARY = 0
DAC_SECONDARY = 1


class PaintedBeam():
    """Power + painting co-ramps for a painted optical trap. Mix in alongside
    whatever else the beam's controller inherits; see the module docstring for
    the subclass contract."""

    def _init_painting(self, paint_amp_dac=DAC_CH, v_paint_min=-5., core=Core):
        """Wire up the painting hardware. Call from the subclass __init__.

        Args:
            paint_amp_dac (DAC_CH): DAC channel whose voltage sets the painting
                amplitude, by setting the modulation depth of the FM source
                driving the AO/AOD.
            v_paint_min (float): the voltage on that channel corresponding to
                zero painting. This is both the "painting off" level and the
                bottom of the painting-amplitude rescale, so it is one number
                per beam, fixed by that beam's RF chain (input attenuators and
                dividers differ between beams). Full scale is +6 V.
            core (Core): the core device, for seconds_to_mu.
        """
        self.paint_amp_dac = paint_amp_dac
        self.v_paint_min = v_paint_min
        self.core = core

    # -------------------------------------------------------------------------
    # painting amplitude
    # -------------------------------------------------------------------------

    @portable
    def _paint_amp_v(self, v_pd, v_pd_max, v_awg_am_max) -> TFloat:
        """The painting-amplitude voltage that holds the trap frequency at its
        value for (v_pd_max, v_awg_am_max) while the power sits at v_pd.

        The trap frequency goes as sqrt(P / h**3), where P is the power and h
        the painting amplitude, so holding it constant means h scales as the
        cube root of the fraction by which P changed. That fraction is then
        rescaled onto the DAC range, which runs from self.v_paint_min (no
        painting) to v_awg_am_max (full painting).

        All three arguments are required -- the public wrapper on the subclass
        is where ExptParams defaults get filled in.
        """
        p_frac = v_pd / v_pd_max
        paint_amp_frac = p_frac**0.3333
        return (paint_amp_frac - 0.5)*(v_awg_am_max - self.v_paint_min) \
            + (v_awg_am_max + self.v_paint_min)/2

    @kernel
    def painting_off(self):
        """Sets the painting amplitude to zero."""
        self.paint_amp_dac.set(v=self.v_paint_min)

    # -------------------------------------------------------------------------
    # hooks -- implemented by the machine-specific subclass
    # -------------------------------------------------------------------------
    #
    # Left undecorated and raising here: the base class is never instantiated,
    # so the ARTIQ compiler only ever sees the subclass overrides.

    def _ramp_begin(self, v_start, paint, dac_select, dt_mu):
        """Put the beam in the state the ramp starts from: set the power DAC to
        v_start, engage whichever servo dac_select names, and turn the beam on
        if this beam's ramps are responsible for that. Runs before the ramp
        clock starts, so any delay taken here is on top of the ramp duration t.
        """
        raise NotImplementedError

    def _set_pd(self, v, dac_select):
        """Write v to the power-setpoint DAC selected by dac_select, WITHOUT
        loading -- _load_pd latches it together with the painting DAC."""
        raise NotImplementedError

    def _load_pd(self, dac_select):
        """Latch the writes queued by _set_pd and the painting DAC."""
        raise NotImplementedError

    def _ramp_end(self, v_end, dac_select):
        """Bookkeeping after the last step: record v_end as the DAC's tracked
        voltage so the next relative ramp starts from the right place."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # co-ramps
    # -------------------------------------------------------------------------

    @kernel
    def _write_step(self, v, paint, v_pd_max, v_awg_am_max,
                    keep_trap_frequency_constant, dac_select):
        """One step of a co-ramp: queue the power and (if painting) the painting
        amplitude, then latch both on the same LDAC edge."""
        self._set_pd(v, dac_select)
        if paint:
            if keep_trap_frequency_constant:
                v_awg_amp_mod = self._paint_amp_v(v, v_pd_max, v_awg_am_max)
            else:
                v_awg_amp_mod = v_awg_am_max
            self.paint_amp_dac.set(v_awg_amp_mod, load_dac=False)
        self._load_pd(dac_select)

    @kernel(flags={"fast-math"})
    def _ramp_linear(self, t, v_start, v_end, n_steps, paint,
                     v_awg_am_max, v_pd_max, keep_trap_frequency_constant,
                     dac_select=DAC_PRIMARY):
        """Linear power ramp, co-ramping the painting amplitude.

        All arguments must already be resolved -- no sentinel defaults. Uses
        absolute timestamps, so the ramp lasts exactly t (plus whatever
        _ramp_begin spends turning the beam on).
        """
        if not paint:
            self.painting_off()

        dv_step = linear_step(v_start, v_end, n_steps)
        dt_mu = self.core.seconds_to_mu(t / n_steps)

        self._ramp_begin(v_start, paint, dac_select, dt_mu)

        t_mu = now_mu()
        for i in range(n_steps):
            at_mu(t_mu)
            self._write_step(v_start + i*dv_step, paint, v_pd_max,
                             v_awg_am_max, keep_trap_frequency_constant,
                             dac_select)
            t_mu += dt_mu
        at_mu(t_mu)
        self._ramp_end(v_end, dac_select)

    @kernel(flags={"fast-math"})
    def _ramp_cubic(self, t, v_start, v_end, n_steps, paint,
                    v_awg_am_max, v_pd_max, keep_trap_frequency_constant,
                    dac_select=DAC_PRIMARY):
        """Smoothstep (zero slope at both ends) power ramp, co-ramping the
        painting amplitude. See _ramp_linear for the argument convention."""
        if not paint:
            self.painting_off()

        Adt3, Bdt2 = cubic_coeffs(t, v_start, v_end, n_steps)
        dt_mu = self.core.seconds_to_mu(t / n_steps)

        self._ramp_begin(v_start, paint, dac_select, dt_mu)

        t_mu = now_mu()
        for i in range(n_steps):
            at_mu(t_mu)
            self._write_step(Adt3 * i**3 + Bdt2 * i**2 + v_start, paint,
                             v_pd_max, v_awg_am_max,
                             keep_trap_frequency_constant, dac_select)
            t_mu += dt_mu
        at_mu(t_mu)
        self._ramp_end(v_end, dac_select)

    @kernel(flags={"fast-math"})
    def _ramp_exponential(self, t, v_start, v_end, n_steps, tau, paint,
                          v_awg_am_max, v_pd_max, keep_trap_frequency_constant,
                          dac_select=DAC_PRIMARY):
        """Exponential power ramp of time constant tau, co-ramping the painting
        amplitude. See ramp_math.exponential_coeffs for the sign convention on
        tau, and _ramp_linear for the argument convention."""
        if not paint:
            self.painting_off()

        a, k, e_end = exponential_coeffs(t, v_start, v_end, n_steps, tau)
        e = 1.
        dt_mu = self.core.seconds_to_mu(t / n_steps)

        self._ramp_begin(v_start, paint, dac_select, dt_mu)

        t_mu = now_mu()
        for i in range(n_steps):
            at_mu(t_mu)
            self._write_step(v_end + a * (e - e_end), paint, v_pd_max,
                             v_awg_am_max, keep_trap_frequency_constant,
                             dac_select)
            e *= k
            t_mu += dt_mu
        at_mu(t_mu)
        self._ramp_end(v_end, dac_select)

    @kernel(flags={"fast-math"})
    def _ramp_adiabatic(self, t, v_start, v_end, n_steps, v_offset, paint,
                        v_awg_am_max, v_pd_max, keep_trap_frequency_constant,
                        dac_select=DAC_PRIMARY):
        """Constant-adiabaticity power ramp, co-ramping the painting amplitude.

        Note that the constant-eps trajectory is derived for a trap whose
        frequency follows the power alone; with keep_trap_frequency_constant
        the painting co-ramp is holding the frequency fixed, so the two are
        working against each other. Pair this shape with
        keep_trap_frequency_constant=False unless you know what you want.

        See ramp_math.adiabatic_coeffs for the derivation and _ramp_linear for
        the argument convention.
        """
        if not paint:
            self.painting_off()

        u, du = adiabatic_coeffs(v_start, v_end, n_steps, v_offset)
        dt_mu = self.core.seconds_to_mu(t / n_steps)

        self._ramp_begin(v_start, paint, dac_select, dt_mu)

        t_mu = now_mu()
        for i in range(n_steps):
            at_mu(t_mu)
            self._write_step(1. / (u * u) + v_offset, paint, v_pd_max,
                             v_awg_am_max, keep_trap_frequency_constant,
                             dac_select)
            u += du
            t_mu += dt_mu
        at_mu(t_mu)
        self._ramp_end(v_end, dac_select)
