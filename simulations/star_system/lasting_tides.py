"""Opt-in secular tides: a constant time lag on the spin tracked by body detail.

The module is off unless an experiment sets ``physics.lasting_tides`` true.
That choice requires ``physics.body_detail`` ``spin``. Point mass and fixed
oblateness have no spin rate for a lag to act on, and ``--build`` refuses
them by name.

The force is the quadrupole constant-time-lag tide of Mignard (1979) and
Hut (1981), kept to first order in the lag. For the tide raised on body i
by body j, with ``r`` and ``v`` the position and velocity of j relative to i
and ``Ω`` the spin of i,

    F = -(3 k2 Δt G m_j^2 R_i^5 / r^10) (3 (r · v) r + r^2 (v - Ω × r))

``F`` is the force on j. Body i receives ``-F``. The torque on i's spin is
``-r × F``, so the orbital angular momentum this force removes is stored as
spin. Both directions are evaluated for every pair of positive-mass bodies
that have a radius, a Love number, and a positive lag. Massless bodies are
skipped. There is no constant-Q cutoff and no expansion in eccentricity: a
circular aligned orbit is the special case, not a separate formula.

On that special case the semi-major axis and the spin move together,

    da/dt = 6 k2 Δt m R^5 n (Ω - n) / (M a^4),
    torque = -3 k2 Δt G m^2 R^5 (Ω - n) / a^6,

with ``n`` the mean motion. ``Ω > n`` expands the orbit and slows the spin.
``Ω < n`` does the opposite. ``Ω = n`` is the tidal-lock fixed point: the
torque and ``da/dt`` vanish. A circular orbit with an oblique spin is not
this fixed point.

Earth's default lag is the effective oceanic value that makes this circular,
spin-aligned, Earth-tide-only rate 3.82 cm/yr for the catalog Earth and Moon
at 384400 km. It is not a seismic quality factor. Neron de Surgy & Laskar
(1997) quote ``k2 Δt = 213 s``; with these rounded masses that product
recedes a little faster than 3.82 cm/yr, so the stored lag is the one that
hits 3.82 cm/yr at ``k2 = 0.299``. Other catalog bodies have a zero default
lag and raise no lasting tide. The Moon's own default lag is zero, so the
recession above is the tide raised on Earth.

Close-pass tides (``physics.tides``, ``close-pass-energy-dissipation-v1``)
stay a separate module. Both may be on in one run. The close-pass kick is
still the impulsive spin-free sink at a resolved pericenter inside its
reach. Lasting tides are the continuous force and torque at every
separation, including inside that reach. The losses add. The sum is not one
calibrated tide. Turning one on does not turn the other on. A run that wants
only the secular migration leaves the close-pass kick off.
"""
import math

import numpy as np

from .core import G
from .data import AU_KM, CATALOG

LUNAR_DISTANCE_KM = 384400.0
LUNAR_RECESSION_CM_PER_YEAR = 3.82
_M_PER_AU = AU_KM * 1000.0

TOGETHER = (
    'Close-pass tides and lasting tides may both be on in one run. '
    'Close-pass tides stay an impulsive, spin-free energy sink at a resolved pericenter inside their reach. '
    'Lasting tides are the continuous constant-time-lag force and torque at every separation. '
    'The two models are independent, so their energy losses add, including inside the close-pass reach. '
    'That sum is not one calibrated tide. '
    'Turning one on does not turn the other on. '
    'A run that wants only secular migration leaves close-pass tides off.'
)


def earth_lag_seconds(k2=None, cm_per_year=LUNAR_RECESSION_CM_PER_YEAR):
    """Time lag that makes the aligned circular Earth tide recess at ``cm_per_year``."""
    from .body_detail import EARTH_K2, EARTH_SIDEREAL_SECONDS, YEAR_SECONDS
    if k2 is None:
        k2 = EARTH_K2
    earth, moon, semi_major, mean_motion = lunar_pair()
    spin = 2.0 * math.pi * YEAR_SECONDS / EARTH_SIDEREAL_SECONDS
    per_lag_year = circular_recession_rate(
        earth.mass, moon.mass, earth.radius, semi_major, spin, k2, 1.0)
    target = (cm_per_year / 100.0) / _M_PER_AU
    if per_lag_year == 0.0:
        raise ValueError('The lunar recession formula has a zero coefficient.')
    return (target / per_lag_year) * YEAR_SECONDS


def lunar_pair():
    """Catalog Earth and Moon and a circular 384400 km orbit. Mean motion is radians per year."""
    earth = CATALOG['earth']
    moon = CATALOG['moon']
    semi_major = LUNAR_DISTANCE_KM / AU_KM
    mean_motion = math.sqrt(G * (earth.mass + moon.mass) / semi_major ** 3)
    return earth, moon, semi_major, mean_motion


def circular_recession_rate(primary_mass, companion_mass, radius, semi_major, spin, k2, lag):
    """``da/dt`` in AU per year for a circular spin-aligned tide raised on the primary.

    ``spin`` and the returned rate use radians and years. ``lag`` is in years.
    """
    mean_motion = math.sqrt(G * (primary_mass + companion_mass) / semi_major ** 3)
    return (6.0 * k2 * lag * companion_mass * radius ** 5 * mean_motion * (spin - mean_motion)
            / (primary_mass * semi_major ** 4))


def ctl_acceleration(system, motion):
    """Acceleration from every lasting tide, with the mutual reaction included."""
    acceleration, _torque = _pairs(system, motion)
    return acceleration


def ctl_torque(system, motion):
    """Spin torque from every lasting tide. The sign stores the orbital angular momentum that the force exchanges."""
    _acceleration, torque = _pairs(system, motion)
    return torque


def _pairs(system, motion):
    count = len(system.masses)
    acceleration = np.zeros_like(system.r)
    torque = np.zeros((count, 3))
    if system.r.shape[1] != 3:
        return acceleration, torque
    for index in range(count):
        if motion.k2[index] <= 0.0 or motion.lag[index] <= 0.0 or motion.radius[index] <= 0.0 or system.masses[index] <= 0.0:
            continue
        spin = motion.omega(index)
        for other in range(count):
            if other == index or system.masses[other] <= 0.0:
                continue
            rel = system.r[other] - system.r[index]
            vel = system.v[other] - system.v[index]
            force = _force(rel, vel, spin, motion.k2[index], motion.lag[index], motion.radius[index], system.masses[other])
            if force is None:
                continue
            acceleration[other] += force / system.masses[other]
            acceleration[index] -= force / system.masses[index]
            torque[index] += -np.cross(rel, force)
    return acceleration, torque


def _force(rel, vel, spin, k2, lag, radius, companion_mass):
    r2 = float(np.dot(rel, rel))
    if r2 <= 0.0 or not math.isfinite(r2):
        return None
    factor = 3.0 * k2 * lag * G * companion_mass ** 2 * radius ** 5 / r2 ** 5
    radial = 3.0 * float(np.dot(rel, vel)) * rel
    transverse = r2 * (vel - np.cross(spin, rel))
    force = -factor * (radial + transverse)
    if not np.isfinite(force).all():
        return None
    return force


def measure_lunar_recession(*, orbits=6):
    """Integrate the catalog Earth–Moon pair and report the recession in cm per year.

    Earth's spin is the sidereal day, aligned with the orbit. The Moon is given
    the mean motion and a zero lag, so the tide is the one raised on Earth.
    The state is three-dimensional. The satellite uses the Moon's mass and
    radius with kind ``planet`` because the planar moon-step aid rejects a 3D moon.
    """
    from dataclasses import replace

    from .body_detail import EARTH_K2, EARTH_SIDEREAL_SECONDS, YEAR_SECONDS
    from .initialization import build_system
    from .model import BodyDetailPolicy, InitialBody, InitialSetup, LastingTidePolicy

    earth, moon, semi_major, mean_motion = lunar_pair()
    moon_spec = replace(moon, kind='planet')
    spin = 2.0 * math.pi * YEAR_SECONDS / EARTH_SIDEREAL_SECONDS
    lag = earth_lag_seconds() / YEAR_SECONDS
    analytic = circular_recession_rate(earth.mass, moon.mass, earth.radius, semi_major, spin, EARTH_K2, lag)
    fraction = moon.mass / (earth.mass + moon.mass)
    rel_r = np.array([semi_major, 0.0, 0.0])
    rel_v = np.array([0.0, mean_motion * semi_major, 0.0])
    setup = InitialSetup('Earth-Moon constant time lag', (
        InitialBody('earth', earth, tuple(-fraction * rel_r), tuple(-fraction * rel_v)),
        InitialBody('moon', moon_spec, tuple((1.0 - fraction) * rel_r), tuple((1.0 - fraction) * rel_v)),
    ), (2.0 * math.pi / mean_motion) / 800.0,
        body_detail=BodyDetailPolicy('spin'), lasting_tides=LastingTidePolicy(enabled=True))
    prepared = build_system(setup)
    prepared.system.detail.align_pole(0)
    prepared.system.detail.align_pole(1)
    # The Moon stays at the mean motion. align_pole kept its catalog rate; replace it.
    moment = prepared.system.detail.inertia_c[1]
    prepared.system.detail.angular_momentum[1] = np.array([0.0, 0.0, moment * mean_motion])
    prepared.system.detail.lag[1] = 0.0
    # J2 stays out of this comparison. A bulge changes the shape of the orbit; the
    # 3.82 cm/yr figure is the constant-time-lag tide raised on Earth.
    prepared.system.detail.j2[:] = 0.0
    measured = _recession(prepared, orbits)
    return RecessionComparison(
        analytic * _CM_PER_AU, measured * _CM_PER_AU, earth_lag_seconds(), EARTH_K2, orbits)


def _recession(prepared, orbits):
    system = prepared.system
    mu = G * (system.masses[0] + system.masses[1])
    steps = int(round(orbits * 800))
    times = np.empty(steps + 1)
    axes = np.empty(steps + 1)
    times[0] = system.time
    axes[0] = _semi_major(system, mu)
    for index in range(steps):
        system.step()
        times[index + 1] = system.time
        axes[index + 1] = _semi_major(system, mu)
    return float(np.polyfit(times, axes, 1)[0])


def _semi_major(system, mu):
    rel = system.r[1] - system.r[0]
    vel = system.v[1] - system.v[0]
    distance = float(np.linalg.norm(rel))
    energy = 0.5 * float(np.dot(vel, vel)) - mu / distance
    return -mu / (2.0 * energy)


_CM_PER_AU = _M_PER_AU * 100.0


class RecessionComparison:
    def __init__(self, analytic_cm_per_year, measured_cm_per_year, lag_seconds, k2, orbits):
        self.analytic_cm_per_year = float(analytic_cm_per_year)
        self.measured_cm_per_year = float(measured_cm_per_year)
        self.lag_seconds = float(lag_seconds)
        self.k2 = float(k2)
        self.orbits = orbits
