"""Named oscillator cases used by the command line and the comparison figure."""

from __future__ import annotations

from simulations.harmonic.core import OscillatorParams, natural_frequency

REGIMES = ("undamped", "underdamped", "critical", "overdamped", "resonance")

# Damping ratio zeta = c / (2 sqrt(k m)) for each preset.
REGIME_ZETA = {
    "undamped": 0.0,
    "underdamped": 0.1,
    "critical": 1.0,
    "overdamped": 2.0,
    "resonance": 0.05,
}

# How many undamped periods to show so each regime is recognizable.
REGIME_CYCLES = {
    "undamped": 6.0,
    "underdamped": 8.0,
    "critical": 5.0,
    "overdamped": 5.0,
    "resonance": 30.0,
}


def params_for_regime(
    regime: str,
    mass: float = 1.0,
    spring: float = 1.0,
    damping: float | None = None,
    drive_amplitude: float | None = None,
    drive_frequency: float | None = None,
    x0: float | None = None,
    v0: float | None = None,
) -> OscillatorParams:
    """Build parameters for a named regime.

    An explicit coefficient overrides the preset. The resonance preset is a
    drive at the natural frequency, released from rest at equilibrium, so the
    transient and the steady oscillation are both visible.
    """
    if regime not in REGIME_ZETA:
        known = ", ".join(REGIMES)
        raise ValueError(f"unknown regime {regime!r}; expected one of {known}")

    zeta = REGIME_ZETA[regime]
    omega0 = natural_frequency(mass, spring)
    preset_damping = 2.0 * zeta * (spring * mass) ** 0.5
    if regime == "resonance":
        preset_force = 0.4
        preset_omega = omega0
        preset_x0 = 0.0
        preset_v0 = 0.0
    else:
        preset_force = 0.0
        preset_omega = 0.0
        preset_x0 = 1.0
        preset_v0 = 0.0

    return OscillatorParams(
        mass=mass,
        spring=spring,
        damping=preset_damping if damping is None else damping,
        drive_amplitude=preset_force if drive_amplitude is None else drive_amplitude,
        drive_frequency=preset_omega if drive_frequency is None else drive_frequency,
        x0=preset_x0 if x0 is None else x0,
        v0=preset_v0 if v0 is None else v0,
    )
