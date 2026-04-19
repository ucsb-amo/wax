from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RotationOperation:
    axis: np.ndarray
    angle_rad: float


class BlochVector:
    """Bloch-sphere vector helper with simple rotation and plotting utilities."""

    STYLE_DEFAULTS = {
        "sphere_color": "steelblue",
        "sphere_alpha": 0.10,
        "sphere_lw": 0.0,
        "sphere_grid_color": "0.35",
        "sphere_grid_alpha": 0.30,
        "sphere_grid_lw": 1.1,
        "equator_front_color": "0.45",
        "equator_back_color": "0.45",
        "equator_front_alpha": 0.82,
        "equator_back_alpha": 0.72,
        "equator_lw": 1.05,
        "equator_front_ls": "-",
        "equator_back_ls": (0, (1.2, 2.2)),
        "state_vector_color": "C0",
        "state_vector_alpha": 1.0,
        "state_vector_lw": 2.3,
        "rotation_color": "tab:red",
        "rotation_alpha": 0.38,
        "rotation_lw": 2.0,
        "rotation_arc_alpha": 0.38,
        "rotation_arc_lw": 1.3,
        "rotation_arc_ls": "-",
        "projection_color": "tab:red",
        "projection_alpha": 0.35,
        "projection_axis_lw": 2.0,
        "projection_connector_lw": 1.8,
        "projection_connector_ls": "--",
        "projection_marker_alpha": 0.95,
        "projection_marker_size": 40,
    }
    STYLE = dict(STYLE_DEFAULTS)

    @classmethod
    def set_style(cls, **kwargs):
        unknown = [k for k in kwargs if k not in cls.STYLE_DEFAULTS]
        if unknown:
            raise ValueError(f"Unknown style keys: {unknown}")
        cls.STYLE.update(kwargs)

    @classmethod
    def reset_style(cls):
        cls.STYLE = dict(cls.STYLE_DEFAULTS)

    @classmethod
    def get_style(cls):
        return dict(cls.STYLE)

    def _resolve_style(self, style: Optional[dict] = None):
        merged = dict(self.STYLE)
        if style:
            merged.update(style)
        return merged

    def __init__(
        self,
        vector: Sequence[float],
        previous: Optional["BlochVector"] = None,
        last_rotation: Optional[RotationOperation] = None,
        normalize: bool = True,
    ):
        v = np.asarray(vector, dtype=float).reshape(3)
        if not np.all(np.isfinite(v)):
            raise ValueError("Bloch vector components must be finite")

        if normalize:
            n = np.linalg.norm(v)
            if n == 0:
                raise ValueError("Bloch vector must be non-zero")
            v = v / n

        self.vector = v
        self.previous = previous
        self.last_rotation = last_rotation

    @staticmethod
    def _unit_axis(axis: Sequence[float]) -> np.ndarray:
        a = np.asarray(axis, dtype=float).reshape(3)
        if not np.all(np.isfinite(a)):
            raise ValueError("Rotation axis components must be finite")
        n = np.linalg.norm(a)
        if n == 0:
            raise ValueError("Rotation axis must be non-zero")
        return a / n

    @staticmethod
    def _rodrigues_rotate(v: np.ndarray, k: np.ndarray, angle_rad: float) -> np.ndarray:
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)

    @staticmethod
    def rotation_axis_rotating_frame(
        omega_0: float,
        omega: float,
        t: float,
        Omega: float,
        phase_zero_time: Optional[float] = None,
    ) -> np.ndarray:
        """Return unit rotation axis in the frame rotating at omega_0."""
        if phase_zero_time is None:
            phase_zero_time = t

        delta_omega = float(omega) - float(omega_0)
        phase = delta_omega * (float(t) - float(phase_zero_time))

        Omega = float(Omega)
        norm_H = np.sqrt(Omega * Omega + delta_omega * delta_omega)
        if norm_H <= 0.0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

        inv_norm_H = 1.0 / norm_H
        ux = (Omega * inv_norm_H) * np.cos(phase)
        uy = (Omega * inv_norm_H) * np.sin(phase)
        uz = delta_omega * inv_norm_H
        return np.array([ux, uy, uz], dtype=np.float64)

    def rotate(self, axis: Sequence[float], angle: float, degrees: bool = False) -> "BlochVector":
        k = self._unit_axis(axis)
        angle_rad = np.deg2rad(angle) if degrees else float(angle)
        v_new = self._rodrigues_rotate(self.vector, k, angle_rad)
        return BlochVector(
            v_new,
            previous=self,
            last_rotation=RotationOperation(axis=k, angle_rad=angle_rad),
            normalize=True,
        )

    def rotate_x(self, angle: float, degrees: bool = False) -> "BlochVector":
        return self.rotate([1.0, 0.0, 0.0], angle, degrees=degrees)

    def rotate_y(self, angle: float, degrees: bool = False) -> "BlochVector":
        return self.rotate([0.0, 1.0, 0.0], angle, degrees=degrees)

    def rotate_z(self, angle: float, degrees: bool = False) -> "BlochVector":
        return self.rotate([0.0, 0.0, 1.0], angle, degrees=degrees)

    @staticmethod
    def _make_axis(ax: Optional[plt.Axes] = None, figsize=(6, 6)):
        if ax is not None:
            return ax, ax.figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        return ax, fig

    @staticmethod
    def _draw_vector(
        ax: plt.Axes,
        v: np.ndarray,
        color: str = "C0",
        alpha: float = 1.0,
        lw: float = 2.0,
    ):
        ax.quiver(
            0,
            0,
            0,
            v[0],
            v[1],
            v[2],
            color=color,
            alpha=alpha,
            linewidth=lw,
            arrow_length_ratio=0.08,
        )

    def _draw_projection_elements(
        self,
        ax: plt.Axes,
        axis: Sequence[float],
        projection_color: str = "tab:red",
        line_alpha: float = 0.35,
        projection_axis_lw: float = 2.0,
        projection_connector_lw: float = 1.8,
        projection_connector_ls: Any = "--",
        projection_marker_alpha: float = 0.95,
        projection_marker_size: float = 40,
    ):
        k = self._unit_axis(axis)
        proj_mag = float(np.dot(self.vector, k))
        proj_vec = proj_mag * k

        ax.plot(
            [0.0, proj_vec[0]],
            [0.0, proj_vec[1]],
            [0.0, proj_vec[2]],
            color=projection_color,
            alpha=line_alpha,
            lw=projection_axis_lw,
        )

        ax.plot(
            [self.vector[0], proj_vec[0]],
            [self.vector[1], proj_vec[1]],
            [self.vector[2], proj_vec[2]],
            color=projection_color,
            alpha=line_alpha,
            lw=projection_connector_lw,
            linestyle=projection_connector_ls,
        )

        ax.scatter(
            [proj_vec[0]],
            [proj_vec[1]],
            [proj_vec[2]],
            color=projection_color,
            s=projection_marker_size,
            alpha=projection_marker_alpha,
            depthshade=False,
        )

    def plot_current(
        self,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        color: str = "C0",
        plot_projection=None,
        style: Optional[dict] = None,
    ):
        s = self._resolve_style(style)
        ax, fig = self._make_axis(ax=ax)
        draw_bloch_sphere(
            ax,
            sphere_alpha=s["sphere_alpha"],
            sphere_color=s["sphere_color"],
            sphere_lw=s["sphere_lw"],
            sphere_grid_color=s["sphere_grid_color"],
            sphere_grid_alpha=s["sphere_grid_alpha"],
            sphere_grid_lw=s["sphere_grid_lw"],
            equator_front_color=s["equator_front_color"],
            equator_back_color=s["equator_back_color"],
            equator_front_alpha=s["equator_front_alpha"],
            equator_back_alpha=s["equator_back_alpha"],
            equator_lw=s["equator_lw"],
            equator_front_ls=s["equator_front_ls"],
            equator_back_ls=s["equator_back_ls"],
        )
        self._draw_vector(
            ax,
            self.vector,
            color=s["state_vector_color"] if color == "C0" else color,
            alpha=s["state_vector_alpha"],
            lw=s["state_vector_lw"],
        )

        if plot_projection is not None:
            self._draw_projection_elements(
                ax,
                axis=plot_projection,
                projection_color=s["projection_color"],
                line_alpha=s["projection_alpha"],
                projection_axis_lw=s["projection_axis_lw"],
                projection_connector_lw=s["projection_connector_lw"],
                projection_connector_ls=s["projection_connector_ls"],
                projection_marker_alpha=s["projection_marker_alpha"],
                projection_marker_size=s["projection_marker_size"],
            )

        if show:
            plt.show()
        return fig, ax

    def plot_last_rotation(
        self,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        vector_color: str = "C0",
        axis_color: str = "tab:red",
        old_alpha: float = 0.25,
        new_alpha: float = 1.0,
        draw_steps: int = 0,
        plot_projection=None,
        style: Optional[dict] = None,
    ):
        if self.previous is None or self.last_rotation is None:
            raise ValueError("No previous state/rotation available on this object")

        s = self._resolve_style(style)
        ax, fig = self._make_axis(ax=ax)
        draw_bloch_sphere(
            ax,
            sphere_alpha=s["sphere_alpha"],
            sphere_color=s["sphere_color"],
            sphere_lw=s["sphere_lw"],
            sphere_grid_color=s["sphere_grid_color"],
            sphere_grid_alpha=s["sphere_grid_alpha"],
            sphere_grid_lw=s["sphere_grid_lw"],
            equator_front_color=s["equator_front_color"],
            equator_back_color=s["equator_back_color"],
            equator_front_alpha=s["equator_front_alpha"],
            equator_back_alpha=s["equator_back_alpha"],
            equator_lw=s["equator_lw"],
            equator_front_ls=s["equator_front_ls"],
            equator_back_ls=s["equator_back_ls"],
        )

        v_old = self.previous.vector
        v_new = self.vector
        axis = self.last_rotation.axis
        angle = self.last_rotation.angle_rad

        ax.quiver(
            0,
            0,
            0,
            axis[0],
            axis[1],
            axis[2],
            color=s["rotation_color"] if axis_color == "tab:red" else axis_color,
            alpha=s["rotation_alpha"],
            linewidth=s["rotation_lw"],
            arrow_length_ratio=0.12,
        )

        self._draw_vector(
            ax,
            v_old,
            color=s["state_vector_color"] if vector_color == "C0" else vector_color,
            alpha=old_alpha,
            lw=max(1.0, s["state_vector_lw"] - 0.4),
        )

        n_steps = max(0, int(draw_steps))
        if n_steps > 0:
            t_steps = np.linspace(0.0, 1.0, n_steps + 2)[1:-1]
            for i, t in enumerate(t_steps, start=1):
                v_step = self._rodrigues_rotate(v_old, axis, t * angle)
                a_step = old_alpha + (new_alpha - old_alpha) * (i / (n_steps + 1))
                self._draw_vector(
                    ax,
                    v_step,
                    color=s["state_vector_color"] if vector_color == "C0" else vector_color,
                    alpha=a_step,
                    lw=max(0.8, s["state_vector_lw"] - 0.6),
                )

        self._draw_vector(
            ax,
            v_new,
            color=s["state_vector_color"] if vector_color == "C0" else vector_color,
            alpha=new_alpha,
            lw=s["state_vector_lw"],
        )

        n_arc = 120
        ts = np.linspace(0.0, 1.0, n_arc)
        arc = np.array([self._rodrigues_rotate(v_old, axis, t * angle) for t in ts])
        arc_half = 0.25 * arc
        ax.plot(
            arc_half[:, 0],
            arc_half[:, 1],
            arc_half[:, 2],
            color=s["rotation_color"] if axis_color == "tab:red" else axis_color,
            lw=s["rotation_arc_lw"],
            alpha=s["rotation_arc_alpha"],
            linestyle=s["rotation_arc_ls"],
        )

        i0 = int(0.68 * (n_arc - 1))
        i1 = int(0.86 * (n_arc - 1))
        p0 = arc_half[i0]
        d = arc_half[i1] - arc_half[i0]
        ax.quiver(
            p0[0],
            p0[1],
            p0[2],
            d[0],
            d[1],
            d[2],
            color=s["rotation_color"] if axis_color == "tab:red" else axis_color,
            alpha=s["rotation_arc_alpha"],
            linewidth=s["rotation_arc_lw"],
            arrow_length_ratio=0.55,
        )

        if plot_projection is not None:
            self._draw_projection_elements(
                ax,
                axis=plot_projection,
                projection_color=s["projection_color"],
                line_alpha=s["projection_alpha"],
                projection_axis_lw=s["projection_axis_lw"],
                projection_connector_lw=s["projection_connector_lw"],
                projection_connector_ls=s["projection_connector_ls"],
                projection_marker_alpha=s["projection_marker_alpha"],
                projection_marker_size=s["projection_marker_size"],
            )

        if show:
            plt.show()
        return fig, ax

    def plot_projection(
        self,
        axis: Sequence[float] = (0.0, 0.0, 1.0),
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        vector_color: str = "C0",
        projection_color: str = "tab:red",
        line_alpha: float = 0.35,
        style: Optional[dict] = None,
    ):
        s = self._resolve_style(style)
        ax, fig = self._make_axis(ax=ax)
        draw_bloch_sphere(
            ax,
            sphere_alpha=s["sphere_alpha"],
            sphere_color=s["sphere_color"],
            sphere_lw=s["sphere_lw"],
            sphere_grid_color=s["sphere_grid_color"],
            sphere_grid_alpha=s["sphere_grid_alpha"],
            sphere_grid_lw=s["sphere_grid_lw"],
            equator_front_color=s["equator_front_color"],
            equator_back_color=s["equator_back_color"],
            equator_front_alpha=s["equator_front_alpha"],
            equator_back_alpha=s["equator_back_alpha"],
            equator_lw=s["equator_lw"],
            equator_front_ls=s["equator_front_ls"],
            equator_back_ls=s["equator_back_ls"],
        )
        self._draw_vector(
            ax,
            self.vector,
            color=s["state_vector_color"] if vector_color == "C0" else vector_color,
            alpha=s["state_vector_alpha"],
            lw=s["state_vector_lw"],
        )
        self._draw_projection_elements(
            ax,
            axis=axis,
            projection_color=s["projection_color"] if projection_color == "tab:red" else projection_color,
            line_alpha=s["projection_alpha"] if line_alpha == 0.35 else line_alpha,
            projection_axis_lw=s["projection_axis_lw"],
            projection_connector_lw=s["projection_connector_lw"],
            projection_connector_ls=s["projection_connector_ls"],
            projection_marker_alpha=s["projection_marker_alpha"],
            projection_marker_size=s["projection_marker_size"],
        )

        if show:
            plt.show()
        return fig, ax


def _viewer_direction(ax: plt.Axes) -> np.ndarray:
    elev = np.deg2rad(ax.elev)
    azim = np.deg2rad(ax.azim)
    v = np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ],
        dtype=float,
    )
    return v / np.linalg.norm(v)


def _plot_masked_curve(ax: plt.Axes, points: np.ndarray, mask: np.ndarray, **kwargs):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    for seg in np.split(idx, cuts):
        ax.plot(points[seg, 0], points[seg, 1], points[seg, 2], **kwargs)


def _plot_circle_with_hidden_style(
    ax: plt.Axes,
    points: np.ndarray,
    front_color: str = "0.45",
    back_color: str = "0.45",
    lw: float = 1.05,
    front_alpha: float = 0.82,
    back_alpha: float = 0.72,
    front_ls: Any = "-",
    back_ls: Any = (0, (1.2, 2.2)),
):
    points_closed = np.vstack([points, points[0]])
    view_dir = _viewer_direction(ax)
    front_mask = points_closed @ view_dir >= 0.0
    back_mask = ~front_mask

    _plot_masked_curve(
        ax,
        points_closed,
        front_mask,
        color=front_color,
        lw=lw,
        alpha=front_alpha,
        linestyle=front_ls,
    )
    _plot_masked_curve(
        ax,
        points_closed,
        back_mask,
        color=back_color,
        lw=lw,
        alpha=back_alpha,
        linestyle=back_ls,
    )


def draw_bloch_sphere(
    ax: plt.Axes,
    sphere_alpha: float = 0.10,
    sphere_color: str = "steelblue",
    sphere_lw: float = 0.0,
    sphere_grid_color: str = "0.35",
    sphere_grid_alpha: float = 0.30,
    sphere_grid_lw: float = 1.1,
    equator_front_color: str = "0.45",
    equator_back_color: str = "0.45",
    equator_front_alpha: float = 0.82,
    equator_back_alpha: float = 0.72,
    equator_lw: float = 1.05,
    equator_front_ls: Any = "-",
    equator_back_ls: Any = (0, (1.2, 2.2)),
):
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 60)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=sphere_color, alpha=sphere_alpha, linewidth=sphere_lw)

    t = np.linspace(-1, 1, 200)
    ax.plot(t, 0 * t, 0 * t, color=sphere_grid_color, lw=sphere_grid_lw, alpha=sphere_grid_alpha)
    ax.plot(0 * t, t, 0 * t, color=sphere_grid_color, lw=sphere_grid_lw, alpha=sphere_grid_alpha)
    ax.plot(0 * t, 0 * t, t, color=sphere_grid_color, lw=sphere_grid_lw, alpha=sphere_grid_alpha)
    ax.text(1.07, 0.0, 0.0, "x", color="0.25", fontsize=9, ha="center", va="center")
    ax.text(0.0, 1.07, 0.0, "y", color="0.25", fontsize=9, ha="center", va="center")
    ax.text(0.0, 0.0, 1.07, "z", color="0.25", fontsize=9, ha="center", va="center")

    th = np.linspace(0, 2 * np.pi, 500)
    circle_xy = np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])
    circle_yz = np.column_stack([np.zeros_like(th), np.cos(th), np.sin(th)])
    circle_xz = np.column_stack([np.cos(th), np.zeros_like(th), np.sin(th)])
    _plot_circle_with_hidden_style(
        ax,
        circle_xy,
        front_color=equator_front_color,
        back_color=equator_back_color,
        lw=equator_lw,
        front_alpha=equator_front_alpha,
        back_alpha=equator_back_alpha,
        front_ls=equator_front_ls,
        back_ls=equator_back_ls,
    )
    _plot_circle_with_hidden_style(
        ax,
        circle_yz,
        front_color=equator_front_color,
        back_color=equator_back_color,
        lw=equator_lw,
        front_alpha=equator_front_alpha,
        back_alpha=equator_back_alpha,
        front_ls=equator_front_ls,
        back_ls=equator_back_ls,
    )
    _plot_circle_with_hidden_style(
        ax,
        circle_xz,
        front_color=equator_front_color,
        back_color=equator_back_color,
        lw=equator_lw,
        front_alpha=equator_front_alpha,
        back_alpha=equator_back_alpha,
        front_ls=equator_front_ls,
        back_ls=equator_back_ls,
    )

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()


def _extract_replay_state_series(replay_result: Any, repeat_index: int = 0):
    """Return (t, state[N,3]) from replay result object or dict."""
    if hasattr(replay_result, "t_rr") and hasattr(replay_result, "state_rr"):
        t_rr = np.asarray(replay_result.t_rr)
        s_rr = np.asarray(replay_result.state_rr)
        if repeat_index < 0 or repeat_index >= t_rr.shape[0]:
            raise ValueError(f"repeat_index={repeat_index} out of range for {t_rr.shape[0]} repeats.")
        t = np.asarray(t_rr[repeat_index], dtype=float)
        state = np.asarray(s_rr[repeat_index], dtype=float)
        if state.ndim != 2 or state.shape[1] != 3:
            raise ValueError(f"replay_result.state_rr must be (N_repeat, N_step, 3), got {s_rr.shape}.")
        return t, state

    if isinstance(replay_result, dict):
        rr = replay_result.get("repeat_results", None)
        if rr is None:
            raise ValueError("Replay dict missing 'repeat_results'.")
        if repeat_index < 0 or repeat_index >= len(rr):
            raise ValueError(f"repeat_index={repeat_index} out of range for {len(rr)} repeats.")
        entry = rr[repeat_index]
        t = np.asarray(entry.get("time_s"), dtype=float)

        if "state" in entry:
            state = np.asarray(entry["state"], dtype=float)
            if state.ndim != 2 or state.shape[1] != 3:
                raise ValueError(f"repeat_results[{repeat_index}]['state'] must have shape (N,3), got {state.shape}.")
            return t, state

        # Backward-compatible reconstruction from z-only trajectory.
        if "state_z_center" in entry:
            z = np.asarray(entry["state_z_center"], dtype=float)
            x = np.zeros_like(z)
            y = np.zeros_like(z)
            state = np.column_stack([x, y, z])
            return t, state

        raise ValueError("Replay dict entry needs either 'state' or 'state_z_center'.")

    raise ValueError("Unsupported replay result type. Pass FeedbackReplayResult or compatible dict.")


def plot_replay_bloch_vs_time(
    replay_result: Any,
    repeat_index: int = 0,
    figsize: tuple[float, float] = (11.0, 4.8),
    title: Optional[str] = None,
    sphere_alpha: float = 0.10,
    show: bool = True,
):
    """
    Composite replay visualization: Bloch trajectory + state-vs-time panel.

    Parameters
    ----------
    replay_result
        FeedbackReplayResult object or compatible replay dict.
    repeat_index
        Which repeat trace to visualize.
    """
    t, state = _extract_replay_state_series(replay_result, repeat_index=repeat_index)
    sx = state[:, 0]
    sy = state[:, 1]
    sz = state[:, 2]

    fig = plt.figure(figsize=figsize)
    ax_bloch = fig.add_subplot(1, 2, 1, projection="3d")
    ax_time = fig.add_subplot(1, 2, 2)

    draw_bloch_sphere(ax_bloch, sphere_alpha=sphere_alpha)

    ax_bloch.plot(sx, sy, sz, color="tab:blue", lw=1.8, alpha=0.9)
    ax_bloch.scatter([sx[0]], [sy[0]], [sz[0]], color="tab:green", s=34, depthshade=False, label="start")
    ax_bloch.scatter([sx[-1]], [sy[-1]], [sz[-1]], color="tab:red", s=36, depthshade=False, label="end")
    ax_bloch.legend(loc="upper left", fontsize="small")

    ax_time.plot(t, sx, label="s_x", lw=1.4)
    ax_time.plot(t, sy, label="s_y", lw=1.4)
    ax_time.plot(t, sz, label="s_z", lw=1.8)
    ax_time.set_xlabel("time (s)")
    ax_time.set_ylabel("Bloch component")
    ax_time.set_ylim(-1.05, 1.05)
    ax_time.grid(alpha=0.25)
    ax_time.legend(loc="best", fontsize="small")

    if title is None:
        run_id = None
        if hasattr(replay_result, "metadata"):
            run_id = replay_result.metadata.get("run_id")
        elif isinstance(replay_result, dict):
            run_id = replay_result.get("metadata", {}).get("run_id")
        if run_id is None:
            title = "Replay Bloch trajectory vs time"
        else:
            title = f"run {run_id} | replay Bloch trajectory vs time"

    fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, (ax_bloch, ax_time)
