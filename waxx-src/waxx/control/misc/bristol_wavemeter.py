from waxx.control.misc.bristol.pyBristolSCPI import pyBristolSCPI

_C_LIGHT = 299792458.0  # m/s

# Default comparison frequency: K-39 D1 line
_F0_DEFAULT_HZ = 389.28617e12  # Hz

class BristolWavemeter:

    def __init__(self, host='192.168.1.105'):
        self._dev = pyBristolSCPI(host)

    def get_wavelength(self) -> float:
        """Return the peak wavelength in meters."""
        wl_nm = self._dev.readWL()
        return wl_nm * 1e-9

    def get_frequency(self) -> float:
        """Return the peak frequency in Hz (c / lambda)."""
        return _C_LIGHT / self.get_wavelength()

    def get_detuning(self, f0: float = _F0_DEFAULT_HZ) -> float:
        """Return detuning from f0 in Hz.

        Parameters
        ----------
        f0 : float
            Reference frequency in Hz. Defaults to 389.28617 THz
            (K-39 D1 line).
        """
        return self.get_frequency() - f0

    def close(self):
        self._dev.close()
