"""Checks for the two-body elliptical orbit.

Integrator: Yoshida's fourth-order symplectic composition of velocity Verlet.
Conservation and closure use 1000 steps per orbit for 20 orbits.
Kepler's third law uses 600 steps per orbit for 8 orbits at several semi-major axes.
"""

import numpy as np

from simulations.orbit.core import (
    angular_momentum,
    center_of_mass,
    diagnostics,
    ellipse_radius_error,
    expected_angular_momentum,
    expected_two_body_energy,
    gravitational_parameter,
    initial_state,
    integrate_two_body,
    kepler_period,
    linear_momentum,
    periapsis_passages,
    total_energy,
)

# About ten times the drift measured with this step size, so a switch back to
# plain second-order Verlet would fail while roundoff does not.
ENERGY_RELATIVE_TOLERANCE = 1e-6
ANGULAR_MOMENTUM_RELATIVE_TOLERANCE = 1e-9
# Periapsis stays within this fraction of the semi-major axis of its start.
CLOSURE_TOLERANCE = 1e-4
# Radial departure from r = a(1-e^2)/(1+e cos theta), divided by a.
ELLIPSE_TOLERANCE = 2e-4
# |T^2 / a^3 - 4 pi^2 / mu| / (4 pi^2 / mu).
KEPLER_RELATIVE_TOLERANCE = 1e-6
CENTER_OF_MASS_TOLERANCE = 1e-8
MOMENTUM_TOLERANCE = 1e-8

STAR_MASS = 1.0
PLANET_MASS = 0.3
G = 1.7


def test_initial_energy_is_minus_g_m1_m2_over_two_a():
    semi_major, eccentricity = 1.4, 0.45
    state = initial_state(semi_major, eccentricity, STAR_MASS, PLANET_MASS, G)
    energy = float(total_energy(*state, STAR_MASS, PLANET_MASS, G))
    expected = -G * STAR_MASS * PLANET_MASS / (2.0 * semi_major)
    relative = abs(energy - expected) / abs(expected)
    print(f"initial energy = {energy:.12e}, vis-viva = {expected:.12e}, relative error = {relative:.6e}")
    assert relative < 1e-12
    helper = expected_two_body_energy(G, STAR_MASS, PLANET_MASS, semi_major)
    assert abs(helper - expected) / abs(expected) < 1e-15


def test_initial_angular_momentum_matches_the_ellipse():
    semi_major, eccentricity = 1.4, 0.45
    state = initial_state(semi_major, eccentricity, STAR_MASS, PLANET_MASS, G)
    angular = float(angular_momentum(*state[:4], STAR_MASS, PLANET_MASS))
    reduced = STAR_MASS * PLANET_MASS / (STAR_MASS + PLANET_MASS)
    mu = G * (STAR_MASS + PLANET_MASS)
    expected = reduced * np.sqrt(mu * semi_major * (1.0 - eccentricity**2))
    relative = abs(angular - expected) / abs(expected)
    print(f"initial L = {angular:.12e}, ellipse = {expected:.12e}, relative error = {relative:.6e}")
    assert relative < 1e-12
    helper = expected_angular_momentum(G, STAR_MASS, PLANET_MASS, semi_major, eccentricity)
    assert abs(helper - expected) / abs(expected) < 1e-12


def test_energy_conserved_over_many_orbits():
    trajectory = _integrate(semi_major=1.4, eccentricity=0.55, steps_per_orbit=1000, n_orbits=20)
    report = diagnostics(trajectory, G, STAR_MASS, PLANET_MASS, 1.4)
    print(f"max relative energy drift = {report['energy_drift']:.6e}")
    assert report["energy_drift"] < ENERGY_RELATIVE_TOLERANCE


def test_angular_momentum_conserved_over_many_orbits():
    trajectory = _integrate(semi_major=1.4, eccentricity=0.55, steps_per_orbit=1000, n_orbits=20)
    report = diagnostics(trajectory, G, STAR_MASS, PLANET_MASS, 1.4)
    print(f"max relative angular-momentum drift = {report['angular_momentum_drift']:.6e}")
    assert report["angular_momentum_drift"] < ANGULAR_MOMENTUM_RELATIVE_TOLERANCE


def test_center_of_mass_does_not_drift():
    semi_major = 1.4
    trajectory = _integrate(semi_major=semi_major, eccentricity=0.55, steps_per_orbit=1000, n_orbits=20)
    com = center_of_mass(trajectory.r_star, trajectory.r_planet, STAR_MASS, PLANET_MASS)
    momentum = linear_momentum(trajectory.v_star, trajectory.v_planet, STAR_MASS, PLANET_MASS)
    offset = float(np.max(np.linalg.norm(com, axis=1)) / semi_major)
    momentum_max = float(np.max(np.linalg.norm(momentum, axis=1)))
    print(f"max |COM| / a = {offset:.6e}, max |P| = {momentum_max:.6e}")
    assert offset < CENTER_OF_MASS_TOLERANCE
    assert momentum_max < MOMENTUM_TOLERANCE


def test_orbit_closes_on_itself():
    semi_major, eccentricity = 1.2, 0.6
    trajectory = _integrate(semi_major=semi_major, eccentricity=eccentricity, steps_per_orbit=1000, n_orbits=20)
    _times, positions = periapsis_passages(trajectory.t, trajectory.relative_position)
    distances = np.linalg.norm(positions - trajectory.relative_position[0], axis=1)
    closure = float(np.max(distances) / semi_major)
    shape = ellipse_radius_error(trajectory.relative_position, semi_major, eccentricity) / semi_major
    print(f"periapsis passages = {len(positions)}, max displacement / a = {closure:.6e}, ellipse error / a = {shape:.6e}")
    assert len(positions) >= 15
    assert closure < CLOSURE_TOLERANCE
    assert shape < ELLIPSE_TOLERANCE


def test_keplers_third_law():
    """T^2 / a^3 equals 4 pi^2 / (G (M + m)) for several ellipses."""
    mu = gravitational_parameter(G, STAR_MASS, PLANET_MASS)
    constant = 4.0 * np.pi**2 / mu
    semi_majors = (0.5, 1.0, 1.6, 2.4)
    ratios = []
    for semi_major in semi_majors:
        measured = _measured_period(semi_major, eccentricity=0.35, steps_per_orbit=600, n_orbits=8)
        ratio = measured**2 / semi_major**3
        ratios.append(ratio)
        relative = abs(ratio - constant) / constant
        print(f"a = {semi_major:.2f}, T^2/a^3 = {ratio:.10f}, target = {constant:.10f}, relative error = {relative:.6e}")
        assert relative < KEPLER_RELATIVE_TOLERANCE
    spread = (max(ratios) - min(ratios)) / np.mean(ratios)
    print(f"spread of T^2/a^3 across semi-major axes = {spread:.6e}")
    assert spread < KEPLER_RELATIVE_TOLERANCE


def test_kepler_period_helper_matches_the_formula():
    semi_major, mu = 1.3, 2.2
    expected = 2.0 * np.pi * np.sqrt(semi_major**3 / mu)
    assert abs(kepler_period(semi_major, mu) - expected) / expected < 1e-15


def _integrate(semi_major: float, eccentricity: float, steps_per_orbit: int, n_orbits: int):
    mu = G * (STAR_MASS + PLANET_MASS)
    period = 2.0 * np.pi * np.sqrt(semi_major**3 / mu)
    state = initial_state(semi_major, eccentricity, STAR_MASS, PLANET_MASS, G)
    return integrate_two_body(
        *state,
        STAR_MASS,
        PLANET_MASS,
        period / steps_per_orbit,
        n_orbits * steps_per_orbit,
        G,
    )


def _measured_period(semi_major: float, eccentricity: float, steps_per_orbit: int, n_orbits: int) -> float:
    trajectory = _integrate(semi_major, eccentricity, steps_per_orbit, n_orbits)
    times, _positions = periapsis_passages(trajectory.t, trajectory.relative_position)
    return float(np.mean(np.diff(times)))
