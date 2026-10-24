"""Figures and animations for the two-body orbit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from simulations.orbit.core import (
    Trajectory,
    analytic_barycentric_ellipse,
    angular_momentum,
    diagnostics,
    total_energy,
)

_PLANET = "#0072B2"
_STAR = "#E69F00"


def save_orbit_figure(
    trajectory: Trajectory,
    path: Path,
    g: float,
    star_mass: float,
    planet_mass: float,
    semi_major: float,
    eccentricity: float,
) -> None:
    """Ellipse of each body, plus the drift of energy and angular momentum."""
    import matplotlib.pyplot as plt

    report = diagnostics(trajectory, g, star_mass, planet_mass, semi_major)
    energy = total_energy(
        trajectory.r_star,
        trajectory.v_star,
        trajectory.r_planet,
        trajectory.v_planet,
        star_mass,
        planet_mass,
        g,
    )
    angular = angular_momentum(
        trajectory.r_star,
        trajectory.v_star,
        trajectory.r_planet,
        trajectory.v_planet,
        star_mass,
        planet_mass,
    )
    energy_change = (energy - energy[0]) / abs(float(energy[0]))
    angular_change = (angular - angular[0]) / abs(float(angular[0]))
    star_exact, planet_exact = analytic_barycentric_ellipse(semi_major, eccentricity, star_mass, planet_mass)

    fig = plt.figure(figsize=(9.6, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0])
    ax_orbit = fig.add_subplot(grid[:, 0])
    ax_energy = fig.add_subplot(grid[0, 1])
    ax_angular = fig.add_subplot(grid[1, 1])

    ax_orbit.plot(trajectory.r_star[:, 0], trajectory.r_star[:, 1], color=_STAR, linewidth=2.2, label="star", zorder=2)
    ax_orbit.plot(trajectory.r_planet[:, 0], trajectory.r_planet[:, 1], color=_PLANET, linewidth=2.2, label="planet", zorder=2)
    ax_orbit.plot(
        star_exact[:, 0],
        star_exact[:, 1],
        color="black",
        linestyle=(0, (2, 2)),
        linewidth=1.0,
        label="analytic ellipse",
        zorder=3,
    )
    ax_orbit.plot(planet_exact[:, 0], planet_exact[:, 1], color="black", linestyle=(0, (2, 2)), linewidth=1.0, zorder=3)
    ax_orbit.plot(0.0, 0.0, marker="+", color="black", markersize=9, markeredgewidth=1.2, label="barycenter", linestyle="none")
    ax_orbit.plot(trajectory.r_star[0, 0], trajectory.r_star[0, 1], marker="o", color=_STAR, markersize=11, markeredgecolor="black", linestyle="none")
    ax_orbit.plot(trajectory.r_planet[0, 0], trajectory.r_planet[0, 1], marker="o", color=_PLANET, markersize=7, markeredgecolor="black", linestyle="none")
    ax_orbit.set_aspect("equal", adjustable="datalim")
    ax_orbit.set_xlabel("x")
    ax_orbit.set_ylabel("y")
    ax_orbit.set_title(
        f"Two-body ellipses, a = {semi_major:g}, e = {eccentricity:g}\n"
        f"M = {star_mass:g}, m = {planet_mass:g}, G = {g:g}"
    )
    ax_orbit.grid(True, alpha=0.3)
    ax_orbit.legend(loc="best", fontsize=8)

    ax_energy.plot(trajectory.t, energy_change, color=_PLANET, linewidth=1.2)
    ax_energy.set_ylabel("(E - E0) / |E0|")
    ax_energy.set_title(f"energy drift {report['energy_drift']:.2e}")
    ax_energy.grid(True, alpha=0.3)

    ax_angular.plot(trajectory.t, angular_change, color=_PLANET, linewidth=1.2)
    ax_angular.set_xlabel("time t")
    ax_angular.set_ylabel("(L - L0) / |L0|")
    ax_angular.set_title(f"angular momentum drift {report['angular_momentum_drift']:.2e}")
    ax_angular.grid(True, alpha=0.3)

    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_animation(
    trajectory: Trajectory,
    path: Path,
    semi_major: float,
    eccentricity: float,
    star_mass: float,
    planet_mass: float,
    frames: int,
    fps: int,
    animation_format: str,
) -> None:
    """Draw both bodies as they trace their ellipses about the barycenter."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    star_exact, planet_exact = analytic_barycentric_ellipse(semi_major, eccentricity, star_mass, planet_mass)
    frame_index = _frame_indices(len(trajectory.t), frames)
    points = np.vstack((trajectory.r_star, trajectory.r_planet))
    margin = 0.12 * semi_major
    x_min, y_min = points.min(axis=0) - margin
    x_max, y_max = points.max(axis=0) + margin

    fig, ax = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Two-body orbit, a = {semi_major:g}, e = {eccentricity:g}")
    ax.grid(True, alpha=0.3)
    ax.plot(star_exact[:, 0], star_exact[:, 1], color="0.75", linewidth=1.0, zorder=0)
    ax.plot(planet_exact[:, 0], planet_exact[:, 1], color="0.75", linewidth=1.0, zorder=0)
    ax.plot(0.0, 0.0, marker="+", color="black", markersize=8, linestyle="none", label="barycenter")
    star_trail, = ax.plot([], [], color=_STAR, linewidth=1.4, label="star")
    planet_trail, = ax.plot([], [], color=_PLANET, linewidth=1.5, label="planet")
    star_point, = ax.plot([], [], marker="o", color=_STAR, markersize=12, markeredgecolor="black", linestyle="none")
    planet_point, = ax.plot([], [], marker="o", color=_PLANET, markersize=7, markeredgecolor="black", linestyle="none")
    clock = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top")
    ax.legend(loc="upper right", fontsize=8)

    def draw(frame: int):
        i = int(frame_index[frame])
        star_trail.set_data(trajectory.r_star[: i + 1, 0], trajectory.r_star[: i + 1, 1])
        planet_trail.set_data(trajectory.r_planet[: i + 1, 0], trajectory.r_planet[: i + 1, 1])
        star_point.set_data([trajectory.r_star[i, 0]], [trajectory.r_star[i, 1]])
        planet_point.set_data([trajectory.r_planet[i, 0]], [trajectory.r_planet[i, 1]])
        clock.set_text(f"t = {trajectory.t[i]:.2f}")
        return star_trail, planet_trail, star_point, planet_point, clock

    anim = FuncAnimation(fig, draw, frames=len(frame_index), blit=False)
    anim.save(path, writer=_writer(animation_format, fps), dpi=100)
    plt.close(fig)


def _frame_indices(n_samples: int, frames: int) -> np.ndarray:
    count = max(2, min(frames, n_samples))
    return np.linspace(0, n_samples - 1, count).astype(int)


def _writer(animation_format: str, fps: int):
    if animation_format == "gif":
        from matplotlib.animation import PillowWriter

        return PillowWriter(fps=fps)
    if animation_format == "mp4":
        from matplotlib.animation import FFMpegWriter

        return FFMpegWriter(fps=fps)
    raise ValueError(f"unsupported animation format {animation_format!r}")
