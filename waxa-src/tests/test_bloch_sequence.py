"""BlochVector sequences: rotation, attenuation, and the multi-stage plot.

Pure numpy + matplotlib (Agg); nothing here touches hardware or data on disk.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from waxa.plotting.bloch import BlochVector, plot_bloch_sequence


def _raman(b, delta, Omega, t):
    axis = BlochVector.rotation_axis_rotating_frame(0.0, delta, t=0.0, Omega=Omega)
    return b.rotate(axis, -np.hypot(Omega, delta) * t)


@pytest.mark.parametrize("delta_over_Omega", [0.0, 0.3, -0.7, 2.0])
def test_single_pulse_matches_rabi_formula(delta_over_Omega):
    Omega = 2 * np.pi * 135e3
    delta = delta_over_Omega * Omega
    t = 1.7e-6
    b = _raman(BlochVector([0, 0, 1]), delta, Omega, t)
    W = np.hypot(Omega, delta)
    p_flip = (Omega / W) ** 2 * np.sin(W * t / 2) ** 2
    assert b.vector[2] == pytest.approx(1 - 2 * p_flip, abs=1e-12)


def test_rotation_keeps_attenuated_length():
    b = BlochVector([1, 0, 0]).attenuate(0.8, 1.0)
    assert b.length == pytest.approx(0.8)
    b2 = b.rotate_y(0.9).rotate([1, 1, 1], 2.1)
    assert b2.length == pytest.approx(0.8, abs=1e-12)


def test_unit_rotation_unchanged():
    b = BlochVector([0.3, -0.2, 0.9])
    assert b.rotate_x(1.3).length == pytest.approx(1.0, abs=1e-12)


def test_attenuate_scales_components_and_records():
    b = BlochVector([0.6, 0.0, 0.8]).attenuate(0.5, 0.9)
    np.testing.assert_allclose(b.vector, [0.3, 0.0, 0.72])
    assert b.last_rotation is None
    assert b.last_attenuation.transverse == 0.5
    with pytest.raises(ValueError):
        b.attenuate(-0.1)


def test_read_step_matches_kernel_algebra():
    """kexp feedback kernel: n = Rz(-(alpha_z)) h with alpha_z = -alpha_read, then the twist
    rotates by -twist*h_z and x, y are multiplied by C. Our read: rotate_z(alpha - twist*s_z),
    attenuate(C)."""
    alpha, twist, C = 0.948, 0.976, 0.8445
    h = np.array([0.5, -0.3, 0.6])
    h = h / np.linalg.norm(h)
    d = twist * h[2]
    ang = alpha - d  # exact trig version of the kernel's second-order twist
    nx = np.cos(ang) * h[0] - np.sin(ang) * h[1]
    ny = np.sin(ang) * h[0] + np.cos(ang) * h[1]
    b = BlochVector(h).rotate_z(alpha - twist * h[2]).attenuate(C, 1.0)
    np.testing.assert_allclose(b.vector, [C * nx, C * ny, h[2]], atol=1e-12)


def test_history_and_step_path():
    b0 = BlochVector([0, 0, 1])
    b2 = b0.rotate_x(np.pi / 2).attenuate(0.5, 1.0)
    chain = b2.history()
    assert len(chain) == 3 and chain[0] is b0
    arc = chain[1].step_path(50)
    np.testing.assert_allclose(np.linalg.norm(arc, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(arc[-1], chain[1].vector, atol=1e-12)
    chord = b2.step_path(11)
    np.testing.assert_allclose(chord[0], chain[1].vector)
    np.testing.assert_allclose(chord[-1], b2.vector)


def test_plot_bloch_sequence_panels_and_checks():
    Omega = 2 * np.pi * 135e3
    traces = []
    for f in (-40e3, 0.0, 40e3):
        d = 2 * np.pi * f
        b = _raman(BlochVector([0, 0, 1]), d, Omega, 1.7e-6)
        b = b.rotate_z(0.9 - 1.0 * b.vector[2]).attenuate(0.85)
        b = b.rotate_z(-d * 10e-6)
        traces.append(_raman(b, d, Omega, 1.7e-6))
    stages = [("pulse 1", 1), ("read", 2), ("free", 1), ("pulse 2", 1)]
    fig, axes = plot_bloch_sequence(traces, stages=stages, labels=["-40", "0", "+40"], show=False)
    assert len(axes) == len(stages) + 1
    plt.close(fig)
    with pytest.raises(ValueError):
        plot_bloch_sequence(traces, stages=[("all", 4)], show=False)
    with pytest.raises(ValueError):
        plot_bloch_sequence([traces[0], BlochVector([0, 0, 1])], show=False)
