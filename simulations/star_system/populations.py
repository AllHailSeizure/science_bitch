"""Seeded planar, circular test-particle populations in AU and Julian years."""
from dataclasses import dataclass

import numpy as np

from .core import G, System, kepler_period

MAX_PARTICLES = 2000


@dataclass(frozen=True)
class Population:
    """Ring: one radius; disk: uniform area; belt: uniform radius.

    Radii are initial distances from a massive reference body's center.
    Angles may be random or evenly spaced; all velocities are prograde circular
    two-body guesses relative to that body's current position and velocity.
    """

    kind: str = 'belt'
    count: int = 200
    inner: float = 2.
    outer: float = 3.5
    seed: int = 0
    angles: str = 'random'
    reference: int = 0

    def __post_init__(self):
        if self.kind not in ('ring', 'disk', 'belt'):
            raise ValueError('Population kind must be ring, disk, or belt.')
        if type(self.count) is not int or not 1 <= self.count <= MAX_PARTICLES:
            raise ValueError(f'Population count must be an integer from 1 to {MAX_PARTICLES}.')
        if not np.isfinite([self.inner, self.outer]).all() or not 0 < self.inner <= self.outer:
            raise ValueError('Population radii must be finite, positive, and inner <= outer (AU).')
        if self.kind == 'ring' and self.inner != self.outer:
            raise ValueError('A ring requires equal inner and outer radii.')
        if self.kind != 'ring' and self.inner == self.outer:
            raise ValueError('A disk or belt requires inner < outer.')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('Population seed must be a nonnegative integer.')
        if self.angles not in ('random', 'even'):
            raise ValueError('Population angles must be random or even.')
        if type(self.reference) is not int or self.reference < 0:
            raise ValueError('Population reference must be a nonnegative body index.')


def with_population(system, settings, reference_radius=0.):
    """Return an independent system, preserving all existing states and time.

    The fixed timestep is reduced at construction if the population needs it.
    This is initialization, not an insertion API for a running trajectory.
    The caller supplies the reference's physical radius in AU if known.
    """
    reference = settings.reference
    if reference >= len(system.masses) or system.masses[reference] <= 0:
        raise ValueError('Population reference must identify a massive body.')
    if not np.isfinite(reference_radius) or reference_radius < 0 or settings.inner <= reference_radius:
        raise ValueError('Population must start outside the reference surface (AU).')
    if np.count_nonzero(system.masses == 0) + settings.count > MAX_PARTICLES:
        raise ValueError(f'At most {MAX_PARTICLES} population particles are supported.')
    rng = np.random.Generator(np.random.PCG64(settings.seed))
    if settings.kind == 'disk':
        radii = np.sqrt(rng.uniform(settings.inner**2, settings.outer**2, settings.count))
    else:
        radii = rng.uniform(settings.inner, settings.outer, settings.count)
    if settings.angles == 'even':
        angles = rng.uniform(0, 2*np.pi) + np.arange(settings.count)*2*np.pi/settings.count
    else:
        angles = rng.uniform(0, 2*np.pi, settings.count)
    radial = np.column_stack((np.cos(angles), np.sin(angles)))
    mu = G*system.masses[reference]
    positions = system.r[reference] + radii[:, None]*radial
    velocities = system.v[reference] + np.sqrt(mu/radii)[:, None]*radial[:, ::-1]*[-1, 1]
    dt = min(system.dt, kepler_period(settings.inner, mu)/600)
    return System(np.concatenate((system.masses, np.zeros(settings.count))),
                  np.vstack((system.r, positions)), np.vstack((system.v, velocities)),
                  dt, system.time)
