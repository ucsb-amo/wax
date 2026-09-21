import numpy as np
from artiq.experiment import TFloat

class TweezerMovesLib():
    def cubic_move(self,t,t_move,x_move) -> TFloat:
        """A cubic profile that moves a distance x_move, with zero initial and
        final velocity.

        Args:
            t (float): The time (in s) through the move (from zero).
            t_move (float): The total time (in s) that the move will take.
            x_move (float): The displacement (in m) at time t_move.

        Returns:
            float or np.array: displacement vs time.
        """        
        A = -2*x_move/t_move**3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        B = 3*x_move/t_move**2
        return A*t**3 + B*t**2
    
    def sinusoidal_modulation(self,t,
                              modulation_amplitude,
                              modulation_frequency,
                              t_mod_amplitude_ramp=0.) -> TFloat:
        """A sinusoidal modulation of position vs. time.

        Args:
            t (float): The time (in s) through the move (from zero).
            modulation_amplitude (float): The position-space amplitude (in m)
            that the tweezer should be shaken over.
            modulation_frequency (float): The frequency (in Hz) at which the
            modulation will take place.
            t_mod_amplitude_ramp (float): If nonzero, specifies the time (in s)
            over which the modulation amplitude should be linearly ramped from
            zero to the specified modulation_amplitude.

        Returns:
            float or np.array: displacement vs time.
        """        
        A = modulation_amplitude
        fm = modulation_frequency
        
        A = np.ones_like(t) * modulation_amplitude
        mask = t<t_mod_amplitude_ramp
        A[mask] = A[mask] * (t[mask]/t_mod_amplitude_ramp)

        return A*np.sin(2*np.pi*fm*t + np.pi/2)
        
    def sinusoidal_amplitude_modulation(self,t,
                                        amplitude,
                                        modulation_depth,
                                        modulation_frequency,
                                        t_mod_depth_ramp=0.) -> TFloat:
        """A sinusoidal modulation of amplitude vs. time about a fixed
        amplitude, starting at that amplitude.

        Args:
            t (float): The time (in s) through the modulation (from zero).
            amplitude (float): The DDS amplitude (0 to 1) modulated about.
            modulation_depth (float): The fractional modulation depth (0 to 1):
            the amplitude swings between amplitude*(1 -/+ modulation_depth).
            modulation_frequency (float): The frequency (in Hz) of the
            modulation.
            t_mod_depth_ramp (float): If nonzero, specifies the time (in s)
            over which the modulation depth should be linearly ramped from
            zero to the specified modulation_depth.

        Returns:
            float or np.array: amplitude vs time.
        """
        d = np.ones_like(t) * modulation_depth
        mask = t<t_mod_depth_ramp
        d[mask] = d[mask] * (t[mask]/t_mod_depth_ramp)

        return amplitude * (1. + d*np.sin(2*np.pi*modulation_frequency*t))

    def linear(self,t,t_move,y0,yf) -> TFloat:
        slope = (yf - y0) / t_move
        return slope * t + y0
