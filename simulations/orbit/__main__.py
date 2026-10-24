"""Command-line entry for the two-body orbit.

Run from the repository root::

    python -m simulations.orbit
    python -m simulations.orbit --eccentricity 0.7 --orbits 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from simulations.orbit.core import (
    diagnostics,
    ellipse_radius_error,
    gravitational_parameter,
    initial_state,
    integrate_two_body,
    kepler_period,
    periapsis_passages,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a planet and a star on elliptical orbits.")
    parser.add_argument("--g", type=float, default=1.0, help="gravitational constant (default: 1)")
    parser.add_argument("--star-mass", type=float, default=1.0, help="star mass M (default: 1)")
    parser.add_argument("--planet-mass", type=float, default=0.2, help="planet mass m (default: 0.2)")
    parser.add_argument("--semi-major", type=float, default=1.0, help="semi-major axis of the relative orbit (default: 1)")
    parser.add_argument("--eccentricity", type=float, default=0.55, help="eccentricity of the relative orbit (default: 0.55)")
    parser.add_argument("--orbits", type=float, default=4.0, help="number of orbital periods to integrate (default: 4)")
    parser.add_argument("--steps-per-orbit", type=int, default=600, help="Yoshida steps per orbit (default: 600)")
    parser.add_argument("--output-dir", type=Path, default=Path("output/orbit"), help="directory for the plot and animation")
    parser.add_argument("--frames", type=int, default=160, help="frames in the animation (default: 160)")
    parser.add_argument("--fps", type=int, default=24, help="animation frame rate (default: 24)")
    parser.add_argument("--animation-format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--show", action="store_true", help="open the figure on screen when a display is available")
    parser.add_argument("--no-animation", action="store_true", help="skip writing the animation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    interactive = _prepare_matplotlib(args.show)
    from simulations.orbit.plotting import save_animation, save_orbit_figure

    if args.orbits <= 0.0 or args.steps_per_orbit < 8:
        print("error: orbits and steps-per-orbit must be positive", file=sys.stderr)
        return 2

    try:
        state = initial_state(args.semi_major, args.eccentricity, args.star_mass, args.planet_mass, args.g)
        mu = gravitational_parameter(args.g, args.star_mass, args.planet_mass)
        period = kepler_period(args.semi_major, mu)
        n_steps = int(round(args.orbits * args.steps_per_orbit))
        trajectory = integrate_two_body(
            *state,
            args.star_mass,
            args.planet_mass,
            period / args.steps_per_orbit,
            n_steps,
            args.g,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = diagnostics(trajectory, args.g, args.star_mass, args.planet_mass, args.semi_major)
    closure, measured_period = _orbit_checks(trajectory, args.semi_major, args.eccentricity)
    shape = ellipse_radius_error(trajectory.relative_position, args.semi_major, args.eccentricity)
    print(_format_report(args, report, closure, measured_period, period, shape))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "orbit.png"
    save_orbit_figure(
        trajectory,
        figure_path,
        args.g,
        args.star_mass,
        args.planet_mass,
        args.semi_major,
        args.eccentricity,
    )
    print(f"Wrote {figure_path}")

    if not args.no_animation:
        animation_path = args.output_dir / f"animation.{args.animation_format}"
        save_animation(
            trajectory,
            animation_path,
            args.semi_major,
            args.eccentricity,
            args.star_mass,
            args.planet_mass,
            args.frames,
            args.fps,
            args.animation_format,
        )
        print(f"Wrote {animation_path}")

    if interactive:
        import matplotlib.pyplot as plt

        plt.show()
    elif args.show:
        print("No display available; figures were saved instead.")
    return 0


def _orbit_checks(trajectory, semi_major: float, eccentricity: float) -> tuple[float, float | None]:
    times, positions = periapsis_passages(trajectory.t, trajectory.relative_position)
    distances = np.linalg.norm(positions - trajectory.relative_position[0], axis=1)
    closure = float(np.max(distances) / semi_major)
    if times.size < 2:
        return closure, None
    return closure, float(np.mean(np.diff(times)))


def _format_report(args, report, closure, measured_period, analytic_period, shape) -> str:
    lines = [
        "Two-body orbit",
        (
            f"  G = {args.g:g}, M = {args.star_mass:g}, m = {args.planet_mass:g}, "
            f"a = {args.semi_major:g}, e = {args.eccentricity:g}"
        ),
        f"  max relative energy drift = {report['energy_drift']:.6e}",
        f"  max relative angular-momentum drift = {report['angular_momentum_drift']:.6e}",
        f"  max |center of mass| / a = {report['com_offset']:.6e}",
        f"  max periapsis displacement / a = {closure:.6e}",
        f"  max ellipse radius error / a = {shape / args.semi_major:.6e}",
    ]
    if measured_period is not None:
        relative = abs(measured_period - analytic_period) / analytic_period
        lines.append(
            "  period = "
            f"{measured_period:.10f} "
            f"(2*pi*sqrt(a^3/mu) = {analytic_period:.10f}, relative error {relative:.6e})"
        )
    return "\n".join(lines)


def _prepare_matplotlib(show: bool) -> bool:
    import matplotlib

    interactive = show and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not interactive:
        matplotlib.use("Agg", force=True)
    return interactive


if __name__ == "__main__":
    raise SystemExit(main())
