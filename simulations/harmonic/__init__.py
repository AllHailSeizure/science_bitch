"""Mass on a spring: a damped, driven harmonic oscillator."""

from simulations.harmonic.core import (
    OscillatorParams,
    analytic_position,
    diagnostics,
    estimate_period,
    integrate,
    mechanical_energy,
    natural_period,
)

__all__ = [
    "OscillatorParams",
    "analytic_position",
    "diagnostics",
    "estimate_period",
    "integrate",
    "mechanical_energy",
    "natural_period",
]
