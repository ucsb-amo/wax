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


def test_vivid_color():
    import colorsys
    from matplotlib.colors import to_rgb
    from waxa.plotting.bloch import vivid_color

    np.testing.assert_allclose(vivid_color('#8fb4fe', 1.0)[:3], to_rgb('#8fb4fe'), atol=1e-12)
    g = vivid_color('0.75', 2.0)
    assert g[0] == pytest.approx(g[1]) == pytest.approx(g[2])   # grey stays grey
    assert g[0] < 0.75                                          # pulled toward mid lightness
    h0, l0, s0 = colorsys.rgb_to_hls(*to_rgb('#f4a582'))
    h1, l1, s1 = colorsys.rgb_to_hls(*vivid_color('#f4a582', 1.8)[:3])
    assert h1 == pytest.approx(h0) and l1 < l0 and s1 >= s0
    with pytest.raises(ValueError):
        vivid_color('r', 0.0)


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

    # S_z labels on the read stage: header + one value per trace, from the stage's final vector
    # by default ('end'), the incoming one with 'start'; A_z = 0.9 makes the two differ
    zt = [_raman(BlochVector([0, 0, 1]), 2 * np.pi * f, Omega, 1.7e-6) for f in (-40e3, 40e3)]
    zt = [b.rotate_z(0.9).attenuate(0.85, 0.9) for b in zt]
    for at, idx in ((None, 3), ("end", 3), ("start", 1)):
        kw = {} if at is None else {"sz_label_at": at}
        fig, axes = plot_bloch_sequence(zt, stages=[("pulse", 1), ("read", 2)], overview=False,
                                        sz_label_stages=[1], sz_label_fmt="{:+.4f}", show=False, **kw)
        texts = [t.get_text() for t in axes[1].texts if t.get_position()[0] == 0.99]
        assert texts == ["$S_z$"] + [f"{tr.history()[idx].vector[2]:+.4f}" for tr in zt]
        assert not [t for t in axes[0].texts if t.get_position()[0] == 0.99]
        plt.close(fig)
    assert zt[0].history()[3].vector[2] == pytest.approx(0.9 * zt[0].history()[1].vector[2])
    with pytest.raises(ValueError):
        plot_bloch_sequence(traces, stages=stages, sz_label_stages=[1], sz_label_at="middle", show=False)

    # opacity ramp: each pulse arc is a segment collection whose alpha rises from start to end
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    fig, axes = plot_bloch_sequence(traces[:1], stages=stages, overview=False,
                                    path_alpha=(0.1, 1.0), show=False)
    fig.canvas.draw()  # 3D collections hold their 2D segments only after a projection
    # quiver arrows are Line3DCollections too (3 segments each); the ramp has one per path step
    ramps = [c for c in axes[0].collections
             if isinstance(c, Line3DCollection) and len(c.get_segments()) > 10]
    assert len(ramps) == 1
    a = ramps[0].get_edgecolor()[:, 3]
    assert a[0] == pytest.approx(0.1, abs=0.02) and a[-1] == pytest.approx(1.0, abs=0.02)
    assert np.all(np.diff(a) > 0)
    plt.close(fig)


def test_plot_bloch_sequence_into_subfigures():
    Omega = 2 * np.pi * 135e3
    traces = [_raman(BlochVector([0, 0, 1]), 2 * np.pi * f, Omega, 1.7e-6).rotate_z(0.9).attenuate(0.85)
              for f in (-30e3, 30e3)]
    fig = plt.figure(figsize=(6, 4), layout="constrained")
    rows = fig.subfigures(2, 1)
    for row, tr, lab in zip(rows, traces, ("-30", "+30")):
        f2, axes = plot_bloch_sequence([tr], stages=[("pulse", 1), ("read", 2)], labels=[lab],
                                       overview=False, ncols=2, title=f"row {lab}", fig=row, show=False)
        assert f2 is row and len(axes) == 2
        assert all(ax.get_figure(root=False) is row for ax in axes)
        assert len(row.legends) == 1
    assert len(fig.axes) == 4
    fig.canvas.draw()
    plt.close(fig)


def test_sz_ring_and_projection():
    Omega = 2 * np.pi * 135e3
    tr = _raman(BlochVector([0, 0, 1]), 2 * np.pi * 30e3, Omega, 1.7e-6).rotate_z(0.9).attenuate(0.85)
    stages = [("pulse", 1), ("read", 2)]
    n_lines = {}
    for ring in (False, True):
        fig, axes = plot_bloch_sequence([tr], stages=stages, overview=False, sz_label_stages=[1],
                                        sz_label_ring=ring, show=False)
        assert all(ax._focal_length == np.inf for ax in axes)   # orthographic by default
        n_lines[ring] = len(axes[1].lines)
        if ring:
            ring_line = [l for l in axes[1].lines if len(l.get_data_3d()[0]) == 121][0]
            x, y, z = ring_line.get_data_3d()
            v = tr.vector
            np.testing.assert_allclose(np.hypot(x, y), np.hypot(v[0], v[1]), atol=1e-12)
            np.testing.assert_allclose(z, v[2])
        plt.close(fig)
    assert n_lines[True] == n_lines[False] + 1
    fig, axes = plot_bloch_sequence([tr], stages=stages, overview=False, proj_type="persp", show=False)
    assert axes[0]._focal_length == 1
    plt.close(fig)
