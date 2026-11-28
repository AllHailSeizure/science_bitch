"""Pure spatial display transforms of world snapshots, before radial compression.

Vectors here are physical inertial vectors expressed along displayed axes, not
derivatives of rotating coordinates. This module has no dynamics dependency.
A three-component sample is drawn by dropping the third component. A
two-component sample is left unchanged and is not padded.
"""
from dataclasses import dataclass, replace

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


def _drop_height(value):
    """Overhead axes. A trailing third component is dropped and the input is not written."""
    array = np.asarray(value, dtype=float)
    if array.shape[-1] == 3:
        return array[..., :2]
    return array


@dataclass(frozen=True)
class DisplayTransform:
    origin: np.ndarray
    rotation: np.ndarray

    def positions(self, world):
        return (_drop_height(world)-self.origin) @ self.rotation.T

    def inverse_positions(self, relative):
        return np.asarray(relative) @ self.rotation + self.origin

    def vectors(self, physical):
        return _drop_height(physical) @ self.rotation.T

    def inverse_vectors(self, displayed):
        return np.asarray(displayed) @ self.rotation


def _projected_frame(frame):
    """Read-only xy copy used to build the 2D transform. The caller's sample stays intact."""
    if frame.r.ndim != 2 or frame.r.shape[1] not in (2, 3) or frame.v.shape != frame.r.shape:
        raise ValueError('The overhead view needs one position and velocity per body, with two or three components.')
    if frame.r.shape[1] == 2:
        return frame
    return replace(frame, r=np.array(frame.r[:, :2], dtype=float, copy=True),
                   v=np.array(frame.v[:, :2], dtype=float, copy=True))


def frame_transform(frame, mode='sun-relative', origin_id=None, sun_id=None, jupiter_id=None):
    if mode not in MODES:
        raise ValueError(f'Unknown display frame: {mode}.')
    view = _projected_frame(frame)
    if mode == 'world':
        return DisplayTransform(np.zeros(2), np.eye(2))
    if mode == 'rotating':
        sun_id = sun_id or reference_id(view, 'sun')
        jupiter_id = jupiter_id or reference_id(view, 'jupiter')
        if sun_id not in view.body_ids or jupiter_id not in view.body_ids:
            raise ValueError('Sun or Jupiter reference is absent; rotating view unavailable.')
        origin = view.r[view.body_ids.index(sun_id)]
        delta = view.r[view.body_ids.index(jupiter_id)]-origin
        if np.linalg.norm(delta) == 0:
            raise ValueError('Sun and Jupiter are coincident; rotating view unavailable.')
        theta = np.arctan2(delta[1], delta[0])
        c, s = np.cos(theta), np.sin(theta)
        return DisplayTransform(origin.copy(), np.array([[c, s], [-s, c]]))
    if origin_id is None:
        origin_id = next((key for key, spec in zip(view.body_ids, view.body_specs) if spec.kind == 'star'), view.body_ids[0])
    if origin_id not in view.body_ids:
        raise ValueError('View origin is missing in this sample.')
    return DisplayTransform(view.r[view.body_ids.index(origin_id)].copy(), np.eye(2))


def rotate_about_sun(frame, sun_id, jupiter_id):
    return frame_transform(frame, 'rotating', sun_id=sun_id, jupiter_id=jupiter_id).positions(frame.r)


def order_lagrange_points(origin, delta, velocity):
    """Equilateral vertices with L4 leading the direction of revolution.

    ``delta`` and ``velocity`` are the secondary relative to the primary.
    Signed z angular momentum is ``dx*vy - dy*vx``. Positive keeps +60° then
    -60°; negative swaps them. Zero returns that geometric order and labels
    neither vertex L4 or L5. A three-component pair is rotated in its own
    plane. A vertical angular momentum of zero, or a pair parallel to z,
    stays unlabeled geometry.
    """
    origin = np.asarray(origin, dtype=float)
    delta = np.asarray(delta, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if origin.shape == (2,) and delta.shape == (2,) and velocity.shape == (2,):
        cosine, sine = .5, np.sqrt(3)/2
        plus = origin+np.array([cosine*delta[0]-sine*delta[1], sine*delta[0]+cosine*delta[1]])
        minus = origin+np.array([cosine*delta[0]+sine*delta[1], -sine*delta[0]+cosine*delta[1]])
        angular = delta[0]*velocity[1]-delta[1]*velocity[0]
        if angular > 0:
            return np.vstack((plus, minus)), ('L4', 'L5')
        if angular < 0:
            return np.vstack((minus, plus)), ('L4', 'L5')
        return np.vstack((plus, minus)), ('Guide', 'Guide')
    if origin.shape != (3,) or delta.shape != (3,) or velocity.shape != (3,):
        raise ValueError('Lagrange guides need matching two- or three-component vectors.')
    separation = float(np.linalg.norm(delta))
    if separation == 0:
        raise ValueError('Lagrange guides need a nonzero separation.')
    angular_vector = np.cross(delta, velocity)
    axis = delta/separation
    vertical = float(angular_vector[2])
    scale = max(1.0, float(np.linalg.norm(angular_vector)))
    if abs(vertical) <= 1e-15*scale:
        helper = np.array([1., 0., 0.]) if abs(float(delta[2])) > 1e-12*separation else np.array([0., 0., 1.])
        perpendicular = np.cross(helper, axis)
        perpendicular = perpendicular/np.linalg.norm(perpendicular)
        labels = ('Guide', 'Guide')
        leading = 1
    else:
        perpendicular = np.cross(angular_vector/np.linalg.norm(angular_vector), axis)
        perpendicular = perpendicular/np.linalg.norm(perpendicular)
        labels = ('L4', 'L5')
        leading = 1 if vertical > 0 else -1
    cosine, sine = .5, np.sqrt(3)/2
    plus = origin+cosine*delta+sine*separation*perpendicular
    minus = origin+cosine*delta-sine*separation*perpendicular
    if leading < 0:
        return np.vstack((minus, plus)), labels
    return np.vstack((plus, minus)), labels


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
