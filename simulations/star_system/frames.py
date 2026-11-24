"""Pure spatial display transforms of world snapshots, before radial compression.

Vectors here are physical inertial vectors expressed along displayed axes, not
derivatives of rotating coordinates. This module has no dynamics dependency.
"""
from dataclasses import dataclass

import numpy as np

MODES = ('sun-relative', 'world', 'rotating')
LABELS = {'sun-relative': 'Ordinary / origin-relative inertial axes',
          'world': 'World / inertial axes', 'rotating': 'Sun–Jupiter rotating axes'}


def reference_id(frame, key):
    if key in frame.body_ids:
        return key
    candidates = [body_id for body_id, spec in zip(frame.body_ids, frame.body_specs) if spec.catalog_key == key]
    if len(candidates) != 1:
        raise ValueError(f'{key.title()} reference is absent or ambiguous; rotating view unavailable.')
    return candidates[0]


@dataclass(frozen=True)
class DisplayTransform:
    origin: np.ndarray
    rotation: np.ndarray

    def positions(self, world):
        return (np.asarray(world)-self.origin) @ self.rotation.T

    def inverse_positions(self, relative):
        return np.asarray(relative) @ self.rotation + self.origin

    def vectors(self, physical):
        return np.asarray(physical) @ self.rotation.T

    def inverse_vectors(self, displayed):
        return np.asarray(displayed) @ self.rotation


def frame_transform(frame, mode='sun-relative', origin_id=None, sun_id=None, jupiter_id=None):
    if mode not in MODES:
        raise ValueError(f'Unknown display frame: {mode}.')
    if mode == 'world':
        return DisplayTransform(np.zeros(2), np.eye(2))
    if mode == 'rotating':
        sun_id = sun_id or reference_id(frame, 'sun')
        jupiter_id = jupiter_id or reference_id(frame, 'jupiter')
        if sun_id not in frame.body_ids or jupiter_id not in frame.body_ids:
            raise ValueError('Sun or Jupiter reference is absent; rotating view unavailable.')
        origin = frame.r[frame.body_ids.index(sun_id)]
        delta = frame.r[frame.body_ids.index(jupiter_id)]-origin
        if np.linalg.norm(delta) == 0:
            raise ValueError('Sun and Jupiter are coincident; rotating view unavailable.')
        theta = np.arctan2(delta[1], delta[0])
        c, s = np.cos(theta), np.sin(theta)
        return DisplayTransform(origin.copy(), np.array([[c, s], [-s, c]]))
    if origin_id is None:
        origin_id = next((key for key, spec in zip(frame.body_ids, frame.body_specs) if spec.kind == 'star'), frame.body_ids[0])
    if origin_id not in frame.body_ids:
        raise ValueError('View origin is missing in this sample.')
    return DisplayTransform(frame.r[frame.body_ids.index(origin_id)].copy(), np.eye(2))


def rotate_about_sun(frame, sun_id, jupiter_id):
    return frame_transform(frame, 'rotating', sun_id=sun_id, jupiter_id=jupiter_id).positions(frame.r)


def order_lagrange_points(origin, delta, velocity):
    """Equilateral vertices with L4 leading the direction of revolution.

    ``delta`` and ``velocity`` are the secondary relative to the primary.
    Signed z angular momentum is ``dx*vy - dy*vx``. Positive keeps +60° then
    -60°; negative swaps them. Zero returns that geometric order and labels
    neither vertex L4 or L5.
    """
    origin = np.asarray(origin, dtype=float)
    delta = np.asarray(delta, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    cosine, sine = .5, np.sqrt(3)/2
    plus = origin+np.array([cosine*delta[0]-sine*delta[1], sine*delta[0]+cosine*delta[1]])
    minus = origin+np.array([cosine*delta[0]+sine*delta[1], -sine*delta[0]+cosine*delta[1]])
    angular = delta[0]*velocity[1]-delta[1]*velocity[0]
    if angular > 0:
        return np.vstack((plus, minus)), ('L4', 'L5')
    if angular < 0:
        return np.vstack((minus, plus)), ('L4', 'L5')
    return np.vstack((plus, minus)), ('Guide', 'Guide')


def labeled_lagrange_guides(frame, sun_id, jupiter_id):
    """Ordered world guides and roles for one Sun/Jupiter snapshot.

    Callers pass a ``StateFrame`` (``.r``, ``.v``, ``.body_ids``). The rotating
    transform validates the pair before velocity is read.
    """
    frame_transform(frame, 'rotating', sun_id=sun_id, jupiter_id=jupiter_id)
    sun = frame.body_ids.index(sun_id)
    jupiter = frame.body_ids.index(jupiter_id)
    delta = frame.r[jupiter]-frame.r[sun]
    return order_lagrange_points(frame.r[sun], delta, frame.v[jupiter]-frame.v[sun])


def lagrange_guides(frame, sun_id, jupiter_id):
    """World positions of ideal circular restricted-three-body equilateral guides.

    These are geometric guides, not detected occupancy or equilibrium claims.
    L4 leads the direction of revolution and is returned first when signed
    angular momentum is nonzero. With no direction of revolution the geometric
    +60° then -60° order is returned and neither vertex is L4 or L5.
    """
    return labeled_lagrange_guides(frame, sun_id, jupiter_id)[0]
