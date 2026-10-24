"""The simulations run headless and write a plot plus an animation."""

from pathlib import Path

from simulations.harmonic.__main__ import main as harmonic_main
from simulations.orbit.__main__ import main as orbit_main


def test_harmonic_cli_writes_plot_animation_and_comparison(tmp_path: Path):
    output = tmp_path / "harmonic"
    code = harmonic_main(
        [
            "--cycles",
            "1",
            "--steps-per-period",
            "40",
            "--frames",
            "8",
            "--compare",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    assert (output / "displacement.png").stat().st_size > 1000
    assert (output / "regimes.png").stat().st_size > 1000
    assert (output / "animation.gif").stat().st_size > 1000


def test_orbit_cli_writes_plot_and_animation(tmp_path: Path):
    output = tmp_path / "orbit"
    code = orbit_main(
        [
            "--orbits",
            "1",
            "--steps-per-orbit",
            "40",
            "--frames",
            "8",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    assert (output / "orbit.png").stat().st_size > 1000
    assert (output / "animation.gif").stat().st_size > 1000


def test_orbit_cli_rejects_unbound_eccentricity(tmp_path: Path):
    code = orbit_main(["--eccentricity", "1", "--no-animation", "--output-dir", str(tmp_path)])
    assert code == 2
