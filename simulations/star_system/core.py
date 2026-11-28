"""Unsoftened Newtonian gravity in AU, solar masses, Julian years.

Positive masses mutually interact; zero-mass test particles only receive gravity.
Positions are planar (two components) or three-dimensional (three components).
Regular presets store three components with a disclosed inclination.
``make_system`` stays the planar circular helper. A planar recording is not padded.

Fixed-step Yoshida fourth-order composition of velocity Verlet.
Steps stay fixed within each uninterrupted segment; insertion can begin a new
segment with a disclosed smaller step. Close encounters halt instead of silently losing accuracy.
An adaptive run uses the same macro schedule as formation: the authored dt is the maximum.
"""
from dataclasses import dataclass
import math
import numpy as np
from .data import STARS, PLANETS

# Below this floor a close pass is not stepped. The positive Verlet solver
# stays in place; REBOUND is the documented candidate if a run reaches it.
ADAPTIVE_DT_MIN = 1e-10
ADAPTIVE_GROWTH = 2.0

# Yoshida fourth-order composition weights for velocity Verlet.
_YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_W0 = -(2.0 ** (1.0 / 3.0)) * _YOSHIDA_W1


def kepler_period(semi_major: float, mu: float) -> float:
    """T = 2 pi sqrt(a^3 / mu), with mu = G (M + m)."""
    return float(2.0 * np.pi * np.sqrt(semi_major**3 / mu))

G = 6.67430e-11 * 1.98847e30 * (365.25*86400)**2 / (149597870700.)**3
SOLAR_RADIUS_AU = 695700 / 149597870.7


def period(star, planet, distance):
    return kepler_period(distance, G*(STARS[star].mass + PLANETS[planet].mass))


def adaptive_macro_step(current, maximum, floor, limit):
    """Next macro step, or None when ``limit`` is below ``floor``.

    The authored step is the maximum. A closer pair uses the resolution limit.
    After a pass the step grows by at most ``ADAPTIVE_GROWTH`` until it is back
    at the maximum. Callers leave the state unmoved when this returns None.
    """
    if not math.isfinite(limit) or limit >= maximum:
        target = maximum
    elif limit < floor:
        return None
    else:
        target = limit
    if target > current*ADAPTIVE_GROWTH:
        target = current*ADAPTIVE_GROWTH
    if target > maximum:
        target = maximum
    return float(target)


def adaptive_floor_reason(limit, floor):
    """Halt text when a close pass is below the shared adaptive floor."""
    return (
        'Adaptive timestep halt: the pair resolution limit '
        f'{float(limit):.6g} yr is below the floor of {float(floor):.6g} yr. '
        'The state was not advanced. A close-encounter integrator such as REBOUND '
        'remains the candidate if a physical-radius run reaches this floor.')


@dataclass
class System:
    masses: np.ndarray
    r: np.ndarray
    v: np.ndarray
    dt: float
    time: float = 0.

    def accelerations(self, separation_floor=None):
        sources = np.flatnonzero(self.masses > 0)
        delta = self.r[None, sources, :] - self.r[:, None, :]
        d2 = np.sum(delta**2, axis=-1)
        d2[sources, np.arange(len(sources))] = np.inf
        if separation_floor is None:
            return G*np.sum(delta * (self.masses[None, sources]/(d2*np.sqrt(d2)))[:, :, None], axis=1)
        # An unmodeled stellar overlap evaluates gravity at the contact separation
        # so a fixed step can cross it. The direction stays the true separation.
        dist = np.maximum(np.sqrt(d2), np.asarray(separation_floor, dtype=float)[:, sources])
        return G*np.sum(delta * (self.masses[None, sources]/(dist*dist*dist))[:, :, None], axis=1)

    def resolution_limit(self, separation_floor=None):
        """Largest supported step from pair gravity and relative crossing time."""
        if not np.isfinite(self.masses).all() or np.any(self.masses < 0) or not np.isfinite(self.r).all() or not np.isfinite(self.v).all():
            raise ValueError('Nonfinite or negative numeric state; correct the starting vectors.')
        if self.r.ndim != 2 or self.r.shape[0] != len(self.masses) or self.r.shape[1] not in (2, 3) or self.v.shape != self.r.shape:
            raise ValueError('Expected one 2D or 3D position and velocity per mass.')
        sources = np.flatnonzero(self.masses > 0)
        if not len(sources):
            return np.inf
        delta = self.r[None, sources, :] - self.r[:, None, :]
        dist = np.linalg.norm(delta, axis=-1)
        if separation_floor is not None:
            dist = np.maximum(dist, np.asarray(separation_floor, dtype=float)[:, sources])
        dist[sources, np.arange(len(sources))] = np.inf
        speed = np.linalg.norm(self.v[None, sources, :] - self.v[:, None, :], axis=-1)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            gravity = np.sqrt((dist*dist*dist)/(G*(self.masses[:, None]+self.masses[None, sources])))
            crossing = np.divide(dist, speed, out=np.full_like(dist, np.inf), where=speed > 0)
        return float(min(np.min(gravity), np.min(crossing))/10)

    def validate_step(self):
        if not np.isfinite(self.dt) or self.dt <= 0 or not np.isfinite(self.time):
            raise ValueError('Time and positive fixed dt must be finite.')
        if self.dt > self.resolution_limit():
            raise ValueError('Close encounter or fast crossing unresolved: use a smaller fixed dt or wider spacing; state preserved.')

    def step(self, count=1):
        for _ in range(count):
            self.validate_step()
            # Commit only a whole validated Yoshida step. An intermediate failure
            # must not leave a partially kicked/drifted live or recorded state.
            trial = System(self.masses, self.r.copy(), self.v.copy(), self.dt, self.time)
            for w in (_YOSHIDA_W1, _YOSHIDA_W0, _YOSHIDA_W1):
                h = w*self.dt
                trial.v += .5*h*trial.accelerations()
                trial.r += h*trial.v
                trial.validate_step()
                trial.v += .5*h*trial.accelerations()
                trial.validate_step()
            candidate = self.time + self.dt
            if not np.isfinite(candidate) or candidate <= self.time:
                raise ValueError('Yoshida numerical halt: timestep cannot advance clock; state preserved.')
            self.r[:] = trial.r
            self.v[:] = trial.v
            self.time += self.dt

    def invariants(self):
        energy = .5*np.sum(self.masses[:,None]*self.v**2)
        for i in range(len(self.masses)):
            for j in range(i):
                if self.masses[i] > 0 and self.masses[j] > 0:
                    energy -= G*self.masses[i]*self.masses[j]/np.linalg.norm(self.r[i]-self.r[j])
        momentum = np.sum(self.masses[:,None]*self.v, axis=0)
        # Signed z component. It is the whole angular momentum for planar motion.
        angular = np.sum(self.masses*(self.r[:,0]*self.v[:,1]-self.r[:,1]*self.v[:,0]))
        return energy, momentum, angular


def make_system(star='sun', planets=('earth', 'jupiter'), distances=None):
    if not planets or len(set(planets)) != len(planets):
        raise ValueError('Choose at least one planet, without duplicates.')
    distances = np.array(distances if distances is not None else [PLANETS[p].distance for p in planets], dtype=float)
    if len(distances) != len(planets) or not np.all(np.isfinite(distances)) or np.any(distances <= STARS[star].radius*SOLAR_RADIUS_AU):
        raise ValueError('Orbital distances must be finite and outside the stellar surface.')
    masses = np.array([STARS[star].mass] + [PLANETS[p].mass for p in planets])
    angles = np.arange(len(planets))*2.3999632297
    radial = np.column_stack((np.cos(angles), np.sin(angles)))
    r = np.vstack(([0.,0.], distances[:,None]*radial))
    speeds = np.sqrt(G*(masses[0]+masses[1:])/distances)
    v = np.vstack(([0.,0.], speeds[:,None]*radial[:, ::-1]*[-1,1]))
    r -= np.average(r, axis=0, weights=masses)
    v -= np.average(v, axis=0, weights=masses)
    dt = min(period(star,p,a) for p,a in zip(planets,distances))/600
    return System(masses,r,v,dt)
