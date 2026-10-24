"""Checks for the harmonic oscillator against the closed-form solution.

Integrator: velocity Verlet with Strang splitting of linear friction.
Position comparisons use 2000 steps per undamped period for 20 periods.
The undamped period check uses 1000 steps per period for 30 periods.
"""

import numpy as np
import pytest

from simulations.harmonic.core import (
    OscillatorParams,
    analytic_position,
    diagnostics,
    estimate_period,
    integrate,
    natural_period,
)

# Measured max |x - analytic| on these trajectories is a few times 10^-5.
POSITION_ABSOLUTE_TOLERANCE = 2e-4
# Measured relative energy excursion of the undamped Verlet step is a few times 10^-6.
ENERGY_RELATIVE_TOLERANCE = 1e-5
# Measured relative period error is about 2e-6.
PERIOD_RELATIVE_TOLERANCE = 1e-5
# Finite-difference residual of the closed form. The difference scheme itself
# is O(dt^2); a wrong formula would leave an O(1) residual.
ODE_RESIDUAL_TOLERANCE = 1e-4

STEPS_PER_PERIOD = 2000
PERIODS = 20


def _damping(mass: float, spring: float, zeta: float) -> float:
    return 2.0 * zeta * np.sqrt(spring * mass)


def test_undamped_solution_matches_analytic():
    params = OscillatorParams(mass=1.3, spring=2.5, damping=0.0, x0=0.8, v0=-0.4)
    error = _position_error(params)
    print(f"undamped max |x - analytic| = {error:.6e}")
    assert error < POSITION_ABSOLUTE_TOLERANCE


def test_underdamped_solution_matches_analytic():
    mass, spring, zeta = 0.8, 1.7, 0.2
    params = OscillatorParams(mass=mass, spring=spring, damping=_damping(mass, spring, zeta), x0=1.1, v0=0.3)
    error = _position_error(params)
    print(f"underdamped max |x - analytic| = {error:.6e}")
    assert error < POSITION_ABSOLUTE_TOLERANCE


def test_critically_damped_solution_matches_analytic():
    mass, spring = 1.5, 2.0
    params = OscillatorParams(mass=mass, spring=spring, damping=_damping(mass, spring, 1.0), x0=0.7, v0=-0.5)
    error = _position_error(params)
    print(f"critical max |x - analytic| = {error:.6e}")
    assert error < POSITION_ABSOLUTE_TOLERANCE


def test_overdamped_solution_matches_analytic():
    mass, spring = 0.7, 3.0
    params = OscillatorParams(mass=mass, spring=spring, damping=_damping(mass, spring, 2.0), x0=1.2, v0=-0.2)
    error = _position_error(params)
    print(f"overdamped max |x - analytic| = {error:.6e}")
    assert error < POSITION_ABSOLUTE_TOLERANCE


def test_driven_off_resonance_matches_analytic():
    mass, spring = 1.1, 4.0
    params = OscillatorParams(
        mass=mass,
        spring=spring,
        damping=_damping(mass, spring, 0.15),
        drive_amplitude=1.2,
        drive_frequency=0.7 * np.sqrt(spring / mass),
        x0=0.2,
        v0=-0.3,
    )
    error = _position_error(params)
    print(f"driven max |x - analytic| = {error:.6e}")
    assert error < POSITION_ABSOLUTE_TOLERANCE


def test_driven_resonant_solution_matches_analytic():
    mass, spring = 0.9, 1.6
    omega0 = np.sqrt(spring / mass)
    params = OscillatorParams(
        mass=mass,
        spring=spring,
        damping=_damping(mass, spring, 0.08),
        drive_amplitude=0.5,
        drive_frequency=omega0,
        x0=0.0,
        v0=0.0,
    )
    error = _position_error(params)
    print(f"resonance max |x - analytic| = {error:.6e}")
    assert error < POSITION_ABSOLUTE_TOLERANCE


def test_period_is_two_pi_sqrt_m_over_k():
    mass, spring = 2.0, 8.0
    params = OscillatorParams(mass=mass, spring=spring, x0=1.0, v0=0.5)
    expected = 2.0 * np.pi * np.sqrt(mass / spring)
    period = natural_period(mass, spring)
    t, x, _v = integrate(params, t_end=30.0 * period, dt=period / 1000.0)
    measured = estimate_period(t, x)
    relative = abs(measured - expected) / expected
    print(f"period measured = {measured:.12f}, analytic = {expected:.12f}, relative error = {relative:.6e}")
    assert relative < PERIOD_RELATIVE_TOLERANCE


def test_undamped_energy_is_conserved():
    params = OscillatorParams(mass=1.0, spring=1.0, damping=0.0, drive_amplitude=0.0, x0=1.0, v0=0.25)
    period = natural_period(params.mass, params.spring)
    t, x, v = integrate(params, t_end=PERIODS * period, dt=period / STEPS_PER_PERIOD)
    report = diagnostics(params, t, x, v)
    print(f"undamped max relative energy drift = {report['energy_drift']:.6e}")
    assert report["energy_drift"] < ENERGY_RELATIVE_TOLERANCE


def test_period_estimator_on_a_known_sine():
    period = 1.7
    t = np.linspace(0.0, 12.0 * period, 8000)
    x = 0.4 * np.cos(2.0 * np.pi * t / period) + 0.2 * np.sin(2.0 * np.pi * t / period)
    relative = abs(estimate_period(t, x) - period) / period
    assert relative < 1e-6


@pytest.mark.parametrize(
    "params",
    [
        OscillatorParams(mass=1.3, spring=2.5, x0=0.8, v0=-0.4),
        OscillatorParams(mass=0.8, spring=1.7, damping=_damping(0.8, 1.7, 0.2), x0=1.1, v0=0.3),
        OscillatorParams(mass=1.5, spring=2.0, damping=_damping(1.5, 2.0, 1.0), x0=0.7, v0=-0.5),
        OscillatorParams(mass=0.7, spring=3.0, damping=_damping(0.7, 3.0, 2.0), x0=1.2, v0=-0.2),
        OscillatorParams(
            mass=1.1,
            spring=4.0,
            damping=_damping(1.1, 4.0, 0.15),
            drive_amplitude=1.2,
            drive_frequency=0.7 * np.sqrt(4.0 / 1.1),
            x0=0.2,
            v0=-0.3,
        ),
        OscillatorParams(
            mass=0.9,
            spring=1.6,
            damping=_damping(0.9, 1.6, 0.08),
            drive_amplitude=0.5,
            drive_frequency=np.sqrt(1.6 / 0.9),
            x0=0.0,
            v0=0.0,
        ),
        OscillatorParams(
            mass=1.0,
            spring=1.0,
            drive_amplitude=0.2,
            drive_frequency=1.0,
            x0=0.0,
            v0=0.0,
        ),
    ],
)
def test_analytic_solution_satisfies_the_ode(params: OscillatorParams):
    period = natural_period(params.mass, params.spring)
    t = np.linspace(0.0, 12.0 * period, 12 * 8000)
    x = analytic_position(t, params)
    residual = _ode_residual(t, x, params)
    print(f"analytic ODE residual = {residual:.6e}")
    assert residual < ODE_RESIDUAL_TOLERANCE


def _position_error(params: OscillatorParams) -> float:
    period = natural_period(params.mass, params.spring)
    t, x, v = integrate(params, t_end=PERIODS * period, dt=period / STEPS_PER_PERIOD)
    return float(diagnostics(params, t, x, v)["position_error"])


def _ode_residual(t: np.ndarray, x: np.ndarray, params: OscillatorParams) -> float:
    dt = float(t[1] - t[0])
    acceleration = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / dt**2
    velocity = (x[2:] - x[:-2]) / (2.0 * dt)
    midpoint = t[1:-1]
    displacement = x[1:-1]
    residual = (
        params.mass * acceleration
        + params.damping * velocity
        + params.spring * displacement
        - params.drive_amplitude * np.cos(params.drive_frequency * midpoint)
    )
    scale = max(params.spring * float(np.max(np.abs(displacement))), abs(params.drive_amplitude), 1e-12)
    return float(np.max(np.abs(residual)) / scale)
