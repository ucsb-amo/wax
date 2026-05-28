import numpy as np


def dBm_to_mW(p: float) -> float:
    """Convert dBm to mW."""
    return 10 ** (p / 10)


def mW_to_dBm(p: float) -> float:
    """Convert mW to dBm."""
    return 10 * np.log10(p)
