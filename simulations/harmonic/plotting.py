"""Figures and animations for the harmonic oscillator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from simulations.harmonic.core import (
    OscillatorParams,
    analytic_position,
    integrate,
    mechanical_energy,
    natural_period,
)
from simulations.harmonic.regimes import REGIME_CYCLES, REGIME_ZETA, params_for_regime

_NUMERIC = "#0072B2"
_NUMERIC_ALT = "#D55E00"


def save_displacement_figure(
    params: OscillatorParams,
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    path: Path,
    regime: str,
) -> None:
    """Displacement against the analytic solution, plus the mechanical energy."""
    import matplotlib.pyplot as plt

    expected = analytic_position(t, params)
    energy = mechanical_energy(x, v, params.mass, params.spring)
    reference = float(energy[0])
    if abs(reference) > 1e-15:
        energy_change = (energy - reference) / abs(reference)
        energy_label = "relative energy change (E - E0) / |E0|"
    else:
        energy_change = energy - reference
        energy_label = "mechanical energy change"

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True, constrained_layout=True)
    ax_x, ax_e = axes
    ax_x.plot(t, expected, color="black", linestyle="--", linewidth=1.4, label="analytic")
    stride = max(1, len(t) // 40)
    ax_x.plot(
        t[::stride],
        x[::stride],
        linestyle="none",
        marker="o",
        markersize=3.5,
        color=_NUMERIC,
        label="velocity Verlet",
    )
    ax_x.set_ylabel("displacement x")
    ax_x.set_title(_title(regime, params))
    low = float(min(np.min(x), np.min(expected)))
    high = float(max(np.max(x), np.max(expected)))
    span = max(high - low, 1e-6)
    ax_x.set_ylim(low - 0.12 * span, high + 0.38 * span)
    ax_x.legend(loc="upper right", frameon=True)
    ax_x.grid(True, alpha=0.3)

    ax_e.plot(t, energy_change, color=_NUMERIC, linewidth=1.2)
    ax_e.set_xlabel("time t")
    ax_e.set_ylabel(energy_label)
    ax_e.grid(True, alpha=0.3)
    peak = float(np.max(np.abs(energy_change)))
    ax_e.set_title(f"mechanical energy, max relative change {peak:.2e}")

    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_regime_comparison(path: Path, mass: float = 1.0, spring: float = 1.0, steps_per_period: int = 500) -> None:
    """Four panels: undamped, underdamped, critical against overdamped, and resonance."""
    import matplotlib.pyplot as plt

    period = natural_period(mass, spring)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)
    _panel_single(axes[0, 0], "undamped", mass, spring, period, steps_per_period)
    _panel_single(axes[0, 1], "underdamped", mass, spring, period, steps_per_period)
    _panel_critical_and_over(axes[1, 0], mass, spring, period, steps_per_period)
    _panel_single(axes[1, 1], "resonance", mass, spring, period, steps_per_period)
    fig.suptitle("Mass on a spring: numerical markers on the analytic solution", fontsize=13)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_animation(
    params: OscillatorParams,
    t: np.ndarray,
    x: np.ndarray,
    path: Path,
    frames: int,
    fps: int,
    animation_format: str,
    regime: str,
) -> None:
    """Animate the mass and a coil spring. The wall is fixed at the left."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle, Rectangle

    frame_index = _frame_indices(len(t), frames)
    amplitude = float(np.max(np.abs(x))) if len(x) else 1.0
    amplitude = max(amplitude, 1e-6)
    equilibrium = 1.7
    scale = min(0.65, 1.0 / amplitude)
    if equilibrium - scale * amplitude < 0.55:
        scale = (equilibrium - 0.55) / amplitude
    visual = equilibrium + scale * x
    right = float(np.max(visual) + 0.55)

    fig, ax = plt.subplots(figsize=(7.4, 3.3), constrained_layout=True)
    ax.set_xlim(-0.25, right)
    ax.set_ylim(-0.9, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_yticks([])
    ax.set_xlabel("position")
    ax.set_title(_title(regime, params))
    ax.add_patch(Rectangle((-0.25, -0.75), 0.18, 1.5, facecolor="#333333", edgecolor="none", zorder=2))
    ax.axvline(equilibrium, color="0.65", linestyle=":", linewidth=1.0, label="equilibrium")
    ax.plot([0.0, right], [0.0, 0.0], color="0.85", linewidth=1.0, zorder=0)
    spring, = ax.plot([], [], color="#444444", linewidth=1.3, zorder=1)
    mass = Circle((visual[0], 0.0), 0.15, facecolor=_NUMERIC, edgecolor="black", linewidth=0.8, zorder=3)
    ax.add_patch(mass)
    clock = ax.text(0.98, 0.92, "", transform=ax.transAxes, ha="right", va="top")
    ax.legend(loc="upper left", frameon=False)

    def draw(frame: int):
        i = frame_index[frame]
        coil_x, coil_y = _spring_polyline(0.0, float(visual[i]))
        spring.set_data(coil_x, coil_y)
        mass.center = (float(visual[i]), 0.0)
        clock.set_text(f"t = {t[i]:.2f}")
        return spring, clock

    anim = FuncAnimation(fig, draw, frames=len(frame_index), blit=False)
    anim.save(path, writer=_writer(animation_format, fps), dpi=100)
    plt.close(fig)


def _panel_single(ax, regime: str, mass: float, spring: float, period: float, steps_per_period: int) -> None:
    params = params_for_regime(regime, mass=mass, spring=spring)
    cycles = REGIME_CYCLES[regime]
    t, x, _v = integrate(params, t_end=cycles * period, dt=period / steps_per_period)
    expected = analytic_position(t, params)
    ax.plot(t, expected, color="black", linestyle="--", linewidth=1.3, zorder=1)
    stride = max(1, len(t) // 35)
    ax.plot(t[::stride], x[::stride], linestyle="none", marker="o", markersize=3.2, color=_NUMERIC, zorder=2)
    zeta = REGIME_ZETA[regime]
    ax.set_title(f"{regime}, ζ = {zeta:g}")
    ax.set_xlabel("time t")
    ax.set_ylabel("displacement x")
    ax.grid(True, alpha=0.3)


def _panel_critical_and_over(ax, mass: float, spring: float, period: float, steps_per_period: int) -> None:
    cycles = REGIME_CYCLES["critical"]
    for regime, color in (("critical", _NUMERIC), ("overdamped", _NUMERIC_ALT)):
        params = params_for_regime(regime, mass=mass, spring=spring)
        t, x, _v = integrate(params, t_end=cycles * period, dt=period / steps_per_period)
        expected = analytic_position(t, params)
        zeta = REGIME_ZETA[regime]
        ax.plot(t, expected, color=color, linestyle="--", linewidth=1.3, zorder=1)
        stride = max(1, len(t) // 30)
        ax.plot(
            t[::stride],
            x[::stride],
            linestyle="none",
            marker="o",
            markersize=3.2,
            color=color,
            label=f"{regime}, ζ = {zeta:g}",
            zorder=2,
        )
    ax.set_title("critical and overdamped")
    ax.set_xlabel("time t")
    ax.set_ylabel("displacement x")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=True)


def _title(regime: str, params: OscillatorParams) -> str:
    return (
        f"Mass on a spring ({regime}): m = {params.mass:g}, k = {params.spring:g}, "
        f"c = {params.damping:g}"
    )


def _spring_polyline(x_wall: float, x_mass: float, coils: int = 12, amplitude: float = 0.22) -> tuple[np.ndarray, np.ndarray]:
    span = x_mass - x_wall
    if abs(span) < 1e-4:
        span = 1e-4 if span >= 0.0 else -1e-4
    lead = 0.08 * span
    xs = [x_wall, x_wall + lead]
    ys = [0.0, 0.0]
    for i in range(coils):
        xx = x_wall + lead + (span - 2.0 * lead) * (i + 0.5) / coils
        yy = amplitude if i % 2 == 0 else -amplitude
        xs.append(xx)
        ys.append(yy)
    xs.extend((x_mass - lead, x_mass))
    ys.extend((0.0, 0.0))
    return np.asarray(xs), np.asarray(ys)


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
