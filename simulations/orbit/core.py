"""Two-body gravitational orbit.

A star of mass M and a planet of mass m attract each other with Newton's
inverse-square force. Both bodies are integrated. In the barycentric frame
each moves on an ellipse with the barycenter at one focus. The relative
vector r = r_planet - r_star obeys

    r'' = -G (M + m) r / |r|^3

and traces an ellipse of semi-major axis a and eccentricity e. Kepler's
third law for that ellipse is

    T^2 = 4 pi^2 a^3 / (G (M + m)).

The integrator is Yoshida's fourth-order symplectic composition of velocity
Verlet steps (Yoshida, Physics Letters A 150, 262, 1990). Velocity Verlet
alone conserves energy and angular momentum, but its phase error makes a
Keplerian ellipse precess. The fourth-order composition keeps the numerical
orbit closed over many revolutions. One of the three substeps runs backward;
that negative coefficient is part of the method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Yoshida (1990) weights. w1 + w0 + w1 = 1, and w0 is negative.
_YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_W0 = -(2.0 ** (1.0 / 3.0)) * _YOSHIDA_W1


@dataclass
class Trajectory:
    """Barycentric states of both bodies. Arrays have shape (n, 2)."""

    t: np.ndarray
    r_star: np.ndarray
    v_star: np.ndarray
    r_planet: np.ndarray
    v_planet: np.ndarray

    @property
    def relative_position(self) -> np.ndarray:
        return self.r_planet - self.r_star

    @property
    def relative_velocity(self) -> np.ndarray:
        return self.v_planet - self.v_star


def kepler_period(semi_major: float, mu: float) -> float:
    """T = 2 pi sqrt(a^3 / mu), with mu = G (M + m)."""
    return float(2.0 * np.pi * np.sqrt(semi_major**3 / mu))


def gravitational_parameter(g: float, star_mass: float, planet_mass: float) -> float:
    return float(g * (star_mass + planet_mass))


def initial_state(
    semi_major: float,
    eccentricity: float,
    star_mass: float,
    planet_mass: float,
    g: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Barycentric state at periapsis, velocity along +y.

    The relative orbit has semi-major axis ``semi_major`` and the requested
    eccentricity, so the two-body energy is -G M m / (2 a).
    """
    _validate_masses(star_mass, planet_mass, g)
    if semi_major <= 0.0:
        raise ValueError("semi-major axis must be positive")
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must satisfy 0 <= e < 1")

    mu = gravitational_parameter(g, star_mass, planet_mass)
    periapsis = semi_major * (1.0 - eccentricity)
    relative_position = np.array([periapsis, 0.0], dtype=np.float64)
    speed = np.sqrt(mu * (1.0 + eccentricity) / (semi_major * (1.0 - eccentricity)))
    relative_velocity = np.array([0.0, speed], dtype=np.float64)
    total = star_mass + planet_mass
    r_star = -(planet_mass / total) * relative_position
    r_planet = (star_mass / total) * relative_position
    v_star = -(planet_mass / total) * relative_velocity
    v_planet = (star_mass / total) * relative_velocity
    return r_star, v_star, r_planet, v_planet


def integrate_two_body(
    r_star: np.ndarray,
    v_star: np.ndarray,
    r_planet: np.ndarray,
    v_planet: np.ndarray,
    star_mass: float,
    planet_mass: float,
    dt: float,
    n_steps: int,
    g: float = 1.0,
) -> Trajectory:
    """Integrate both bodies for ``n_steps`` of size ``dt``."""
    _validate_masses(star_mass, planet_mass, g)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")

    r_s = np.array(r_star, dtype=np.float64).copy()
    v_s = np.array(v_star, dtype=np.float64).copy()
    r_p = np.array(r_planet, dtype=np.float64).copy()
    v_p = np.array(v_planet, dtype=np.float64).copy()

    out_r_star = np.empty((n_steps + 1, 2), dtype=np.float64)
    out_v_star = np.empty((n_steps + 1, 2), dtype=np.float64)
    out_r_planet = np.empty((n_steps + 1, 2), dtype=np.float64)
    out_v_planet = np.empty((n_steps + 1, 2), dtype=np.float64)
    out_r_star[0] = r_s
    out_v_star[0] = v_s
    out_r_planet[0] = r_p
    out_v_planet[0] = v_p

    for i in range(n_steps):
        for weight in (_YOSHIDA_W1, _YOSHIDA_W0, _YOSHIDA_W1):
            r_s, v_s, r_p, v_p = _velocity_verlet(
                r_s, v_s, r_p, v_p, star_mass, planet_mass, g, weight * dt
            )
        out_r_star[i + 1] = r_s
        out_v_star[i + 1] = v_s
        out_r_planet[i + 1] = r_p
        out_v_planet[i + 1] = v_p

    return Trajectory(
        t=np.arange(n_steps + 1, dtype=np.float64) * dt,
        r_star=out_r_star,
        v_star=out_v_star,
        r_planet=out_r_planet,
        v_planet=out_v_planet,
    )


def total_energy(
    r_star: np.ndarray,
    v_star: np.ndarray,
    r_planet: np.ndarray,
    v_planet: np.ndarray,
    star_mass: float,
    planet_mass: float,
    g: float = 1.0,
) -> np.ndarray:
    """Two-body energy 1/2 M v_star^2 + 1/2 m v_planet^2 - G M m / |r|."""
    kinetic = 0.5 * star_mass * np.sum(v_star * v_star, axis=-1)
    kinetic = kinetic + 0.5 * planet_mass * np.sum(v_planet * v_planet, axis=-1)
    separation = np.linalg.norm(r_planet - r_star, axis=-1)
    return kinetic - g * star_mass * planet_mass / separation


def angular_momentum(
    r_star: np.ndarray,
    v_star: np.ndarray,
    r_planet: np.ndarray,
    v_planet: np.ndarray,
    star_mass: float,
    planet_mass: float,
) -> np.ndarray:
    """Out-of-plane angular momentum of both bodies about the origin."""
    star = r_star[..., 0] * v_star[..., 1] - r_star[..., 1] * v_star[..., 0]
    planet = r_planet[..., 0] * v_planet[..., 1] - r_planet[..., 1] * v_planet[..., 0]
    return star_mass * star + planet_mass * planet


def linear_momentum(
    v_star: np.ndarray,
    v_planet: np.ndarray,
    star_mass: float,
    planet_mass: float,
) -> np.ndarray:
    return star_mass * v_star + planet_mass * v_planet


def center_of_mass(
    r_star: np.ndarray,
    r_planet: np.ndarray,
    star_mass: float,
    planet_mass: float,
) -> np.ndarray:
    total = star_mass + planet_mass
    return (star_mass * r_star + planet_mass * r_planet) / total


def expected_two_body_energy(g: float, star_mass: float, planet_mass: float, semi_major: float) -> float:
    """Vis-viva result E = -G M m / (2 a) for a bound ellipse."""
    return float(-g * star_mass * planet_mass / (2.0 * semi_major))


def expected_angular_momentum(
    g: float,
    star_mass: float,
    planet_mass: float,
    semi_major: float,
    eccentricity: float,
) -> float:
    """L = mu_reduced * sqrt(G (M + m) a (1 - e^2))."""
    reduced = star_mass * planet_mass / (star_mass + planet_mass)
    mu = g * (star_mass + planet_mass)
    return float(reduced * np.sqrt(mu * semi_major * (1.0 - eccentricity**2)))


def periapsis_passages(t: np.ndarray, r_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Times and positions of periapsis, including the initial sample.

    Interior radial minima are refined by quadratic interpolation. Minima far
    above the initial periapsis radius are rejected so a flat numerical wiggle
    cannot register as another passage.
    """
    times = np.asarray(t, dtype=np.float64)
    relative = np.asarray(r_rel, dtype=np.float64)
    radius = np.linalg.norm(relative, axis=1)
    interior = np.where((radius[1:-1] < radius[:-2]) & (radius[1:-1] <= radius[2:]))[0] + 1
    periapsis_radius = float(radius[0])
    gate = periapsis_radius * 1.02 + 1e-10

    passage_times = [float(times[0])]
    passage_positions = [relative[0].copy()]
    for index in interior:
        if radius[index] > gate:
            continue
        y0, y1, y2 = radius[index - 1], radius[index], radius[index + 1]
        curvature = y0 - 2.0 * y1 + y2
        fraction = 0.5 * (y0 - y2) / curvature if curvature != 0.0 else 0.0
        fraction = float(np.clip(fraction, -0.5, 0.5))
        passage_times.append(_quadratic(times[index - 1], times[index], times[index + 1], fraction))
        passage_positions.append(
            np.array(
                [
                    _quadratic(relative[index - 1, 0], relative[index, 0], relative[index + 1, 0], fraction),
                    _quadratic(relative[index - 1, 1], relative[index, 1], relative[index + 1, 1], fraction),
                ]
            )
        )
    return np.asarray(passage_times), np.asarray(passage_positions)


def ellipse_radius_error(r_rel: np.ndarray, semi_major: float, eccentricity: float) -> float:
    """Max |r - a(1-e^2)/(1+e cos theta)|, with theta measured from +x.

    The initial condition puts periapsis on the positive x-axis. A closed
    ellipse with no precession stays on this polar curve.
    """
    relative = np.asarray(r_rel, dtype=np.float64)
    theta = np.arctan2(relative[:, 1], relative[:, 0])
    expected = semi_major * (1.0 - eccentricity**2) / (1.0 + eccentricity * np.cos(theta))
    actual = np.linalg.norm(relative, axis=1)
    return float(np.max(np.abs(actual - expected)))


def diagnostics(
    trajectory: Trajectory,
    g: float,
    star_mass: float,
    planet_mass: float,
    semi_major: float,
) -> dict[str, float]:
    """Relative drifts of energy, angular momentum, and the center of mass."""
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
    com = center_of_mass(trajectory.r_star, trajectory.r_planet, star_mass, planet_mass)
    momentum = linear_momentum(trajectory.v_star, trajectory.v_planet, star_mass, planet_mass)
    energy_scale = abs(float(energy[0]))
    angular_scale = abs(float(angular[0]))
    return {
        "energy_drift": float(np.max(np.abs(energy - energy[0])) / energy_scale),
        "angular_momentum_drift": float(np.max(np.abs(angular - angular[0])) / angular_scale),
        "com_offset": float(np.max(np.linalg.norm(com, axis=1)) / semi_major),
        "momentum": float(np.max(np.linalg.norm(momentum, axis=1))),
    }


def analytic_barycentric_ellipse(
    semi_major: float,
    eccentricity: float,
    star_mass: float,
    planet_mass: float,
    samples: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact elliptical paths of the star and the planet about the barycenter."""
    theta = np.linspace(0.0, 2.0 * np.pi, samples)
    radius = semi_major * (1.0 - eccentricity**2) / (1.0 + eccentricity * np.cos(theta))
    relative = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    total = star_mass + planet_mass
    r_star = -(planet_mass / total) * relative
    r_planet = (star_mass / total) * relative
    return r_star, r_planet


def _velocity_verlet(
    r_star: np.ndarray,
    v_star: np.ndarray,
    r_planet: np.ndarray,
    v_planet: np.ndarray,
    star_mass: float,
    planet_mass: float,
    g: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_star, a_planet = _accelerations(r_star, r_planet, star_mass, planet_mass, g)
    v_star = v_star + 0.5 * step * a_star
    v_planet = v_planet + 0.5 * step * a_planet
    r_star = r_star + step * v_star
    r_planet = r_planet + step * v_planet
    a_star, a_planet = _accelerations(r_star, r_planet, star_mass, planet_mass, g)
    v_star = v_star + 0.5 * step * a_star
    v_planet = v_planet + 0.5 * step * a_planet
    return r_star, v_star, r_planet, v_planet


def _accelerations(
    r_star: np.ndarray,
    r_planet: np.ndarray,
    star_mass: float,
    planet_mass: float,
    g: float,
) -> tuple[np.ndarray, np.ndarray]:
    separation = r_planet - r_star
    distance = float(np.linalg.norm(separation))
    if distance == 0.0:
        raise ValueError("the star and the planet occupy the same position")
    factor = g / distance**3
    a_star = (factor * planet_mass) * separation
    a_planet = -(factor * star_mass) * separation
    return a_star, a_planet


def _quadratic(y0: float, y1: float, y2: float, fraction: float) -> float:
    """Value of the parabola through (-1, y0), (0, y1), (1, y2) at ``fraction``."""
    return float(y1 + 0.5 * fraction * (y2 - y0) + 0.5 * fraction * fraction * (y0 - 2.0 * y1 + y2))


def _validate_masses(star_mass: float, planet_mass: float, g: float) -> None:
    if star_mass <= 0.0 or planet_mass <= 0.0:
        raise ValueError("both masses must be positive")
    if g <= 0.0:
        raise ValueError("G must be positive")
