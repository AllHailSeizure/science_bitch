"""Generic, selection-scoped two-body geometry; never writes physical state."""
from dataclasses import dataclass

import numpy as np

from .core import G


@dataclass(frozen=True)
class SelectionContext:
    pair: tuple | None
    guides: np.ndarray
    orbits: dict


def orbit_curve(frame, index, parent, count=240):
    """Instantaneous osculating bound two-body ellipse, in world AU.

    This freezes the pair's current state and omits other gravitational forces.
    It is a visual estimate, not a predicted N-body trajectory or binding claim.
    """
    r = frame.r[index]-frame.r[parent]
    v = frame.v[index]-frame.v[parent]
    distance = np.linalg.norm(r)
    mu = G*(frame.masses[index]+frame.masses[parent])
    if not distance or mu <= 0:
        return None
    angular = r[0]*v[1]-r[1]*v[0]
    eccentricity_vector = ((np.dot(v, v)-mu/distance)*r-np.dot(r, v)*v)/mu
    e = np.linalg.norm(eccentricity_vector)
    if not np.isfinite(e) or e >= 1-1e-8 or abs(angular) < 1e-12:
        return None
    axis = eccentricity_vector/e if e > 1e-8 else r/distance
    if e <= 1e-8:
        e = 0.
    perpendicular = np.array([-axis[1], axis[0]])
    theta = np.linspace(0, 2*np.pi, count)
    radius = angular**2/mu/(1+e*np.cos(theta))
    return frame.r[parent]+radius[:, None]*(np.cos(theta)[:, None]*axis+np.sin(theta)[:, None]*perpendicular)


def selection_context(frame, selected_id):
    if selected_id not in frame.body_ids:
        return SelectionContext(None, np.empty((0, 2)), {})
    i = frame.body_ids.index(selected_id)
    references = frame.reference_ids or (None,)*len(frame.body_ids)
    orbits = {}
    for j, reference in enumerate(references):
        if reference == selected_id:
            curve = orbit_curve(frame, j, i)
            if curve is not None:
                orbits[frame.body_ids[j]] = curve
    parent = references[i]
    pair, points = None, np.empty((0, 2))
    if parent in frame.body_ids and frame.masses[i] > 0:
        j = frame.body_ids.index(parent)
        delta, velocity = frame.r[i]-frame.r[j], frame.v[i]-frame.v[j]
        if frame.masses[j] > 0 and np.linalg.norm(delta) > 0:
            direction = 1 if delta[0]*velocity[1]-delta[1]*velocity[0] >= 0 else -1
            perpendicular = direction*np.array([-delta[1], delta[0]])
            points = frame.r[j] + .5*delta + np.array([1, -1])[:, None]*np.sqrt(3)/2*perpendicular
            pair = (parent, selected_id)
    return SelectionContext(pair, points, orbits)
