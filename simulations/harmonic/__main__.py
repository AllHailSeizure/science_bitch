"""Command-line entry for the harmonic oscillator.

Run from the repository root::

    python -m simulations.harmonic
    python -m simulations.harmonic --compare
    python -m simulations.harmonic --regime resonance
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from simulations.harmonic.core import diagnostics, integrate, natural_period
from simulations.harmonic.regimes import REGIME_CYCLES, REGIMES, params_for_regime


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a mass on a spring and compare it with the analytic solution.")
    parser.add_argument("--regime", choices=REGIMES, default="undamped", help="preset damping and drive (default: undamped)")
    parser.add_argument("--mass", type=float, default=1.0, help="mass m (default: 1)")
    parser.add_argument("--spring", type=float, default=1.0, help="spring constant k (default: 1)")
    parser.add_argument("--damping", type=float, default=None, help="damping coefficient c; overrides the regime preset")
    parser.add_argument("--drive-amplitude", type=float, default=None, help="driving force amplitude F0")
    parser.add_argument("--drive-frequency", type=float, default=None, help="driving frequency omega")
    parser.add_argument("--x0", type=float, default=None, help="initial displacement")
    parser.add_argument("--v0", type=float, default=None, help="initial velocity")
    parser.add_argument("--cycles", type=float, default=None, help="duration in units of the undamped period")
    parser.add_argument("--steps-per-period", type=int, default=800, help="Verlet steps per undamped period (default: 800)")
    parser.add_argument("--output-dir", type=Path, default=Path("output/harmonic"), help="directory for the plot and animation")
    parser.add_argument("--compare", action="store_true", help="also write regimes.png with the four textbook cases")
    parser.add_argument("--frames", type=int, default=160, help="frames in the animation (default: 160)")
    parser.add_argument("--fps", type=int, default=24, help="animation frame rate (default: 24)")
    parser.add_argument("--animation-format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--show", action="store_true", help="open the figures on screen when a display is available")
    parser.add_argument("--no-animation", action="store_true", help="skip writing the animation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    interactive = _prepare_matplotlib(args.show)
    from simulations.harmonic.plotting import save_animation, save_displacement_figure, save_regime_comparison

    try:
        params = params_for_regime(
            args.regime,
            mass=args.mass,
            spring=args.spring,
            damping=args.damping,
            drive_amplitude=args.drive_amplitude,
            drive_frequency=args.drive_frequency,
            x0=args.x0,
            v0=args.v0,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cycles = REGIME_CYCLES[args.regime] if args.cycles is None else args.cycles
    if cycles <= 0.0 or args.steps_per_period < 2:
        print("error: cycles and steps-per-period must be positive", file=sys.stderr)
        return 2

    period = natural_period(params.mass, params.spring)
    t, x, v = integrate(params, t_end=cycles * period, dt=period / args.steps_per_period)
    report = diagnostics(params, t, x, v)
    print(_format_report(args.regime, params, report))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "displacement.png"
    save_displacement_figure(params, t, x, v, figure_path, args.regime)
    print(f"Wrote {figure_path}")

    if args.compare:
        comparison_path = args.output_dir / "regimes.png"
        save_regime_comparison(comparison_path, mass=args.mass, spring=args.spring)
        print(f"Wrote {comparison_path}")

    if not args.no_animation:
        animation_path = args.output_dir / f"animation.{args.animation_format}"
        save_animation(
            params,
            t,
            x,
            animation_path,
            args.frames,
            args.fps,
            args.animation_format,
            args.regime,
        )
        print(f"Wrote {animation_path}")

    if interactive:
        import matplotlib.pyplot as plt

        plt.show()
    elif args.show:
        print("No display available; figures were saved instead.")
    return 0


def _format_report(regime: str, params, report: dict) -> str:
    lines = [
        f"Harmonic oscillator ({regime})",
        (
            f"  m = {params.mass:g}, k = {params.spring:g}, c = {params.damping:g}, "
            f"F0 = {params.drive_amplitude:g}, omega = {params.drive_frequency:g}"
        ),
        f"  max |x - analytic| = {report['position_error']:.6e}",
        f"  max relative energy change = {report['energy_drift']:.6e}",
    ]
    if report["period"] is not None:
        lines.append(
            "  period = "
            f"{report['period']:.10f} "
            f"(2*pi*sqrt(m/k) = {report['analytic_period']:.10f}, "
            f"relative error {report['period_error']:.6e})"
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
