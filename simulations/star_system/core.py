"""Full unsoftened Newtonian N-body gravity in AU, solar masses, Julian years.

Fixed-step Yoshida composition of Verlet, extending simulations.orbit.core.
Steps are selected at initialization, never varied along a trajectory, preserving
symplecticity. Close encounters halt instead of silently losing accuracy.
"""
from dataclasses import dataclass
import numpy as np
from simulations.orbit.core import _YOSHIDA_W0, _YOSHIDA_W1, kepler_period
from .data import STARS, PLANETS

G = 6.67430e-11 * 1.98847e30 * (365.25*86400)**2 / (149597870700.)**3
SOLAR_RADIUS_AU = 695700 / 149597870.7


def period(star, planet, distance):
    return kepler_period(distance, G*(STARS[star].mass + PLANETS[planet].mass))


@dataclass
class System:
    masses: np.ndarray
    r: np.ndarray
    v: np.ndarray
    dt: float
    time: float = 0.

    def accelerations(self):
        delta = self.r[None, :, :] - self.r[:, None, :]
        d2 = np.sum(delta**2, axis=-1)
        np.fill_diagonal(d2, np.inf)
        return G*np.sum(delta * (self.masses[None, :]/d2**1.5)[:, :, None], axis=1)

    def step(self, count=1):
        for _ in range(count):
            # Reject unresolved encounters before mutating this step. A variable
            # step would compromise the fixed-step symplectic guarantee.
            delta = self.r[None,:,:]-self.r[:,None,:]
            dist = np.linalg.norm(delta, axis=-1)
            np.fill_diagonal(dist, np.inf)
            dynamical = np.sqrt(dist**3/(G*(self.masses[:,None]+self.masses[None,:])))
            if np.min(dynamical) < 10*self.dt:
                raise ValueError('Close encounter unresolved: reset with wider orbital spacing.')
            for w in (_YOSHIDA_W1, _YOSHIDA_W0, _YOSHIDA_W1):
                h = w*self.dt
                self.v += .5*h*self.accelerations()
                self.r += h*self.v
                self.v += .5*h*self.accelerations()
            self.time += self.dt

    def invariants(self):
        energy = .5*np.sum(self.masses[:,None]*self.v**2)
        for i in range(len(self.masses)):
            for j in range(i):
                energy -= G*self.masses[i]*self.masses[j]/np.linalg.norm(self.r[i]-self.r[j])
        momentum = np.sum(self.masses[:,None]*self.v, axis=0)
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
