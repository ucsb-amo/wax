"""RamanBeamPair AO modes, host side: every AO order and either AO the larger.

RAMAN_AO_BOTH must be the split the pair has always used. RAMAN_AO_SINGLE must give the same
two-photon frequency and the same phase-tracker frequency, sit on the BOTH pair at its reference,
and move only the AO with the larger centre frequency. No hardware: stub DDS objects.
"""
import itertools

import numpy as np
import pytest

from waxx.control.raman_beams import DDS0_IDX, DDS1_IDX, RAMAN_AO_BOTH, RAMAN_AO_SINGLE, RamanBeamPair


class FakeDevice:
    sysclk_per_mu = 1000

    def frequency_to_ftw(self, f):
        return int(f * 2 ** 32 / 1e9)

    def turns_to_pow(self, t):
        return int(t * 65536)

    def amplitude_to_asf(self, a):
        return int(a * 16383)


class FakeDDS:
    def __init__(self, frequency, aom_order):
        self.frequency = frequency
        self.amplitude = .5
        self.aom_order = aom_order
        self.dds_device = FakeDevice()


ORDERS = list(itertools.product((1, -1), (1, -1)))
CENTRES = [(150.e6, 80.e6), (80.e6, 150.e6)]
REF = 119463700.


def pair(a, fc):
    return RamanBeamPair(FakeDDS(fc[0], a[0]), FakeDDS(fc[1], a[1]), FakeDDS(200.e6, 1))


def old_split(a, fc, delta):
    """The split as it was before the AO modes, written out."""
    f = fc[1] / fc[0]
    sgn = float(np.sign(fc[0] - fc[1]))
    if a[0] * a[1] > 0:
        df0, c0 = (delta / 2 - sgn * (fc[0] - fc[1])) / (1 + f), sgn
    else:
        df0, c0 = (delta / 2 - (fc[0] + fc[1])) / (1 + f), 1.
    return fc[0] + c0 * df0, fc[1] - c0 * a[0] * a[1] * df0 * f


def invariant(a, fc, f0, f1):
    """The transition frequency the split solves for."""
    if a[0] * a[1] > 0:
        return 2 * float(np.sign(fc[0] - fc[1])) * (f0 - f1)
    return 2 * (f0 + f1)


def tracker(a, f0, f1):
    """Frequency of phase_tracker: 2 * (a0 * phi0 - a1 * phi1)."""
    return 2 * (a[0] * f0 - a[1] * f1)


@pytest.mark.parametrize("a,fc", list(itertools.product(ORDERS, CENTRES)))
def test_both_is_the_old_split(a, fc):
    p = pair(a, fc)
    for delta in (REF - 3.e6, REF, REF + 1.234e6, 2 * REF):
        assert p.state_splitting_to_ao_frequency(delta) == pytest.approx(old_split(a, fc, delta), abs=0., rel=0.)


@pytest.mark.parametrize("a,fc", list(itertools.product(ORDERS, CENTRES)))
def test_single_sits_on_both_at_the_reference(a, fc):
    p = pair(a, fc)
    want = p.state_splitting_to_ao_frequency(REF)
    p.set_ao_mode(RAMAN_AO_SINGLE, REF)
    assert p.state_splitting_to_ao_frequency(REF) == want


@pytest.mark.parametrize("a,fc", list(itertools.product(ORDERS, CENTRES)))
def test_single_keeps_transition_and_tracker(a, fc):
    both, single = pair(a, fc), pair(a, fc)
    single.set_ao_mode(RAMAN_AO_SINGLE, REF)
    for d in (-30.e3, -1.e3, -.1, 0., .1, 1.e3, 30.e3):
        f0b, f1b = both.state_splitting_to_ao_frequency(REF + d)
        f0s, f1s = single.state_splitting_to_ao_frequency(REF + d)
        assert invariant(a, fc, f0s, f1s) == pytest.approx(invariant(a, fc, f0b, f1b), abs=1.e-6, rel=0.)
        assert tracker(a, f0s, f1s) == pytest.approx(tracker(a, f0b, f1b), abs=1.e-6, rel=0.)


@pytest.mark.parametrize("a,fc", list(itertools.product(ORDERS, CENTRES)))
def test_single_moves_only_the_larger_ao(a, fc):
    p = pair(a, fc)
    want = DDS0_IDX if fc[0] > fc[1] else DDS1_IDX
    assert p._ao_moving == want
    p.set_ao_mode(RAMAN_AO_SINGLE, REF)
    assert (int(p._ao_uses0), int(p._ao_uses1)) == ((1, 0) if want == DDS0_IDX else (0, 1))
    a0, a1 = p.state_splitting_to_ao_frequency(REF - 30.e3)
    b0, b1 = p.state_splitting_to_ao_frequency(REF + 30.e3)
    moved, static = (abs(b0 - a0), abs(b1 - a1)) if want == DDS0_IDX else (abs(b1 - a1), abs(b0 - a0))
    assert static == 0.
    assert moved == pytest.approx(30.e3, abs=1.e-6, rel=0.)


def test_equal_centres_pick_dds0():
    assert pair((1, 1), (100.e6, 100.e6))._ao_moving == DDS0_IDX


def test_back_to_both_restores_the_split():
    p = pair((1, 1), (150.e6, 80.e6))
    want = p.state_splitting_to_ao_frequency(REF + 1.e3)
    p.set_ao_mode(RAMAN_AO_SINGLE, REF)
    p.set_ao_mode(RAMAN_AO_BOTH)
    assert p.state_splitting_to_ao_frequency(REF + 1.e3) == want
    assert (int(p._ao_uses0), int(p._ao_uses1)) == (1, 1)


def test_ao_frequencies_follows_the_mode():
    p = pair((1, 1), (150.e6, 80.e6))
    p.set_ao_mode(RAMAN_AO_SINGLE, REF)
    f0, f1 = p.ao_frequencies(REF + 1.e3)
    assert (float(f0), float(f1)) == p.state_splitting_to_ao_frequency(REF + 1.e3)


@pytest.mark.parametrize("mode,ref", [(RAMAN_AO_SINGLE, None), (RAMAN_AO_SINGLE, 0.), (7, REF)])
def test_refused(mode, ref):
    with pytest.raises(ValueError):
        pair((1, 1), (150.e6, 80.e6)).set_ao_mode(mode, ref)
