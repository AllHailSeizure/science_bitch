"""Opt-in post-Newtonian corrections on top of Newtonian gravity.

The module is off unless a :class:`~simulations.star_system.model.RelativisticPolicy`
enables it. Newtonian inverse-square gravity is unchanged when it is off.

Two pairwise terms are added, in harmonic coordinates, for every pair with a
positive total mass. They are the two-body reduction of the Einstein–Infeld–Hoffmann
equations plus the leading quadrupole radiation reaction. Three-body 1PN cross
terms, spin, and 2PN conservative corrections are omitted.

The 1PN acceleration and the 2.5PN radiation reaction are Blanchet, Living
Reviews in Relativity 17, 2 (2014), equation (203), keeping those two orders.
The 1PN piece is the two-body form of Einstein, Infeld, and Hoffmann, Annals
of Mathematics 39, 65 (1938). For a test particle it is the harmonic-coordinate
Schwarzschild correction. Its secular perihelion advance is

    Δω = 6 π G M / (c² a (1 − e²))

radians per radial period (Einstein, Sitzungsberichte der Königlich Preußischen
Akademie der Wissenschaften, 1915, 831). Mercury's orbit gives about 43 arcsec
per Julian century.

The orbit average of the 2.5PN term is Peters, Physical Review 136, B1224
(1964), equation (5.6). The circular coalescence time is equation (5.10):

    T = (5/256) c⁵ a⁴ / (G³ m1 m2 (m1 + m2)).

The fixed-step Yoshida integrator assumes a separable force. At a few percent
of the speed of light the velocity-dependent 1PN term's step error can exceed
the inspiral, so a Peters comparison integrates the 2.5PN term on its own.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math

import numpy as np

from .core import G

# c in AU per Julian year. The metre and the Julian year match core.G.
_METRE_PER_AU = 149597870700.0
_SECONDS_PER_YEAR = 365.25 * 86400.0
SPEED_OF_LIGHT = 299792458.0 * _SECONDS_PER_YEAR / _METRE_PER_AU

_POLICY = ContextVar('sidereel_relativistic_policy', default=None)


def schwarzschild_perihelion_advance(semi_major, eccentricity, central_mass):
    """Radians of perihelion advance per radial period, 6 π G M / (c² a (1−e²))."""
    if semi_major <= 0 or not 0 <= eccentricity < 1 or central_mass <= 0:
        raise ValueError('Perihelion advance needs a > 0, eccentricity in [0, 1), and a positive mass.')
    return 6.0 * math.pi * G * central_mass / (
        SPEED_OF_LIGHT ** 2 * semi_major * (1.0 - eccentricity ** 2))


def peters_semi_major_decay(semi_major, mass_1, mass_2, eccentricity=0.0):
    """Orbit-averaged da/dt in AU per year. Peters 1964, equation (5.6)."""
    if semi_major <= 0 or mass_1 <= 0 or mass_2 <= 0 or not 0 <= eccentricity < 1:
        raise ValueError('Peters decay needs positive masses, a > 0, and eccentricity in [0, 1).')
    e2 = eccentricity * eccentricity
    enhancement = (1.0 + (73.0 / 24.0) * e2 + (37.0 / 96.0) * e2 * e2) / (1.0 - e2) ** 3.5
    return -(64.0 / 5.0) * G ** 3 * mass_1 * mass_2 * (mass_1 + mass_2) / (
        SPEED_OF_LIGHT ** 5 * semi_major ** 3) * enhancement


def peters_coalescence_time(semi_major, mass_1, mass_2):
    """Circular coalescence time in years. Peters 1964, equation (5.10)."""
    if semi_major <= 0 or mass_1 <= 0 or mass_2 <= 0:
        raise ValueError('Peters time needs positive masses and a > 0.')
    return (5.0 / 256.0) * SPEED_OF_LIGHT ** 5 * semi_major ** 4 / (
        G ** 3 * mass_1 * mass_2 * (mass_1 + mass_2))


def post_newtonian_correction(system, *, conservative=True, radiation=True):
    """Pairwise 1PN and 2.5PN acceleration added to the Newtonian field.

    ``n`` in the Blanchet relative acceleration points from body ``j`` to body
    ``i``. The relative correction is split by the mass ratio so a test
    particle feels the whole term and a zero-mass body does not source one.
    """
    if not conservative and not radiation:
        return None
    masses = system.masses
    positions = system.r
    velocities = system.v
    count = len(masses)
    correction = np.zeros_like(positions)
    c2 = SPEED_OF_LIGHT ** 2
    c5 = SPEED_OF_LIGHT ** 5
    any_term = False
    for i in range(count):
        mass_i = float(masses[i])
        for j in range(i + 1, count):
            mass_j = float(masses[j])
            total = mass_i + mass_j
            if total <= 0.0:
                continue
            delta = positions[i] - positions[j]
            distance = float(np.linalg.norm(delta))
            if distance == 0.0 or not math.isfinite(distance):
                continue
            direction = delta / distance
            relative_velocity = velocities[i] - velocities[j]
            speed2 = float(np.dot(relative_velocity, relative_velocity))
            radial_speed = float(np.dot(direction, relative_velocity))
            eta = (mass_i * mass_j) / (total * total) if mass_i > 0.0 and mass_j > 0.0 else 0.0
            gm = G * total
            relative = np.zeros(positions.shape[1], dtype=float)
            if conservative:
                relative = relative + (gm / (distance * distance * c2)) * (
                    direction * ((4.0 + 2.0 * eta) * gm / distance - (1.0 + 3.0 * eta) * speed2
                                 + 1.5 * eta * radial_speed * radial_speed)
                    + (4.0 - 2.0 * eta) * radial_speed * relative_velocity)
            if radiation and eta > 0.0:
                # Blanchet 2014 eq. (203), 2.5PN. The leading piece opposes
                # velocity on a circular orbit, so the pair loses energy.
                relative = relative + (8.0 / 5.0) * (G * G * total * total * eta) / (c5 * distance ** 3) * (
                    radial_speed * direction * (3.0 * speed2 + (17.0 / 3.0) * gm / distance)
                    - relative_velocity * (speed2 + 3.0 * gm / distance))
            if not np.any(relative):
                continue
            any_term = True
            correction[i] += (mass_j / total) * relative
            correction[j] -= (mass_i / total) * relative
    if not any_term:
        return None
    return correction


def correction_for(system):
    """Acceleration hook. ``None`` keeps the Newtonian return value bitwise."""
    policy = _POLICY.get()
    if policy is None or not policy.enabled:
        return None
    return post_newtonian_correction(
        system, conservative=policy.conservative, radiation=policy.radiation)


def _install():
    from . import core
    core._extra_acceleration = correction_for


_install()


@contextmanager
def relativistic_context(setup):
    """Apply ``setup.relativistic`` inside one integration call."""
    policy = None if setup is None else setup.relativistic
    if policy is None or not policy.enabled:
        yield
        return
    token = _POLICY.set(policy)
    try:
        yield
    finally:
        _POLICY.reset(token)
