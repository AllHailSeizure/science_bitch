"""Opt-in body detail: point mass, a fixed J2 bulge, or spin and wobble.

Point mass is the default. Nothing here runs unless an experiment selects
``fixed_oblateness`` or ``spin``, so a point-mass step is the Newtonian sum
it was before.

Fixed oblateness adds the J2 zonal acceleration of each catalog body that has
a J2. The bulge is constant. Its figure axis is fixed in inertial space: the
catalog obliquity, laid in the x–z plane, or +z for a body with no obliquity.
A planar state cannot carry a tilted pole, so a 2D run pins every axis to +z,
perpendicular to the plane. There is no J3, J4, or tesseral field. Bodies
without a listed J2 stay spherical. The monopole is unchanged. The companion
feels the bulge acceleration and the oblate body receives the opposite force
scaled by the mass ratio, so linear momentum is kept. Massless bodies feel
the acceleration and push back with nothing. Spin is not a state at this
level, so the torque that moves orbital angular momentum is not stored.

Full spin and wobble keeps, for each body, an angular-momentum vector and a
unit figure axis. The polar moment is ``C = k M R^2`` and the equatorial
moment is ``A = C - J2 M R^2``. The same J2 acceleration is evaluated on the
current figure axis, so the bulge can wobble and precess. The companion's
gravity applies the quadrupole torque
``N = -3 G M_other (C - A) (p · r) (p × r) / r^5``. That is minus the orbital
torque of the J2 force, so the change in orbital angular momentum is stored
as spin. It is not left as an unmodeled residual. The figure axis advances by rotating about the angular momentum at
``|L| / A``. Bodies start in principal-axis rotation: the catalog sidereal
period along the obliquity pole. A body with no listed spin starts at rest
and, if it also has no J2, has no figure to wobble. This level needs a 3D
state. A collision remnant does not yet receive a spin or a J2, and neither
does an inserted body or a particle population.

Close-pass tides stay a separate, spin-free kick. They do not write into this
spin state. Lasting tides read the spin from here and add their torque beside
the quadrupole torque.
"""
import math
from dataclasses import dataclass

import numpy as np

from .core import G
from .data import AU_KM, CATALOG
from .model import BODY_DETAIL_LEVELS, BodyDetailPolicy, LastingTidePolicy

YEAR_SECONDS = 365.25 * 86400.0
EARTH_J2 = 1.08263e-3
EARTH_INERTIA = 0.3307
EARTH_SIDEREAL_SECONDS = 86164.0905
EARTH_OBLIQUITY_DEG = 23.439281
EARTH_K2 = 0.299
LEO_ALTITUDE_KM = 400.0
LEO_INCLINATION_DEG = 51.6

LASTING_NEEDS_SPIN = (
    'Lasting tides require physics.body_detail "spin", the full spin-and-wobble level. '
    'Point mass and fixed oblateness do not track a spin rate.'
)
SPIN_NEEDS_3D = (
    'Full spin and wobble needs a 3D state. Choose dimensions 3. '
    'A planar state cannot carry a wobbling figure axis.'
)
NO_REMNANT = (
    'Body detail does not yet assign a spin or a J2 to a collision remnant. '
    'Choose stop_on_contact, or leave the collision module unset.'
)
NO_POPULATION = 'Body detail does not yet include a particle population.'
INSERT_UNSUPPORTED = (
    'Inserting a body is not available while body detail is on. '
    'Spin and J2 are stored one row per body and are not extended by insertion.'
)
_REMNANT_MODES = ('perfect_merge', 'speed_angle', 'binary_star_merger')

# Rounded defaults used only when that body-detail level or lasting tides are on.
# They are not new catalog fields. ``lag`` None marks the Earth value that the
# lasting-tide module calibrates to the lunar recession. A lag of 0 raises no tide.
_DEFAULTS = {
    'sun': {'j2': 2.2e-7, 'inertia': 0.070, 'period': 25.38 * 86400.0, 'obliquity': 7.25, 'k2': 0.0, 'lag': 0.0},
    'mercury': {'j2': 5.03e-5, 'inertia': 0.346, 'period': 58.646 * 86400.0, 'obliquity': 0.034, 'k2': 0.0, 'lag': 0.0},
    'venus': {'j2': 4.458e-6, 'inertia': 0.337, 'period': 243.025 * 86400.0, 'obliquity': 177.36, 'k2': 0.0, 'lag': 0.0},
    'earth': {
        'j2': EARTH_J2, 'inertia': EARTH_INERTIA, 'period': EARTH_SIDEREAL_SECONDS,
        'obliquity': EARTH_OBLIQUITY_DEG, 'k2': EARTH_K2, 'lag': None,
    },
    'moon': {'j2': 2.033e-4, 'inertia': 0.3929, 'period': 27.321661 * 86400.0, 'obliquity': 6.68, 'k2': 0.024, 'lag': 0.0},
    'mars': {'j2': 1.96045e-3, 'inertia': 0.366, 'period': 88642.66, 'obliquity': 25.19, 'k2': 0.0, 'lag': 0.0},
    'jupiter': {'j2': 1.4736e-2, 'inertia': 0.254, 'period': 35730.0, 'obliquity': 3.13, 'k2': 0.0, 'lag': 0.0},
    'saturn': {'j2': 1.6298e-2, 'inertia': 0.210, 'period': 38052.0, 'obliquity': 26.73, 'k2': 0.0, 'lag': 0.0},
    'uranus': {'j2': 3.34343e-3, 'inertia': 0.225, 'period': 62064.0, 'obliquity': 97.77, 'k2': 0.0, 'lag': 0.0},
    'neptune': {'j2': 3.411e-3, 'inertia': 0.23, 'period': 58000.0, 'obliquity': 28.32, 'k2': 0.0, 'lag': 0.0},
}

ASSUMPTIONS = (
    'Point mass is the default and adds nothing to the Newtonian acceleration. '
    'Fixed oblateness adds a constant J2 zonal term on the catalog figure axis. '
    'The axis is fixed: it does not precess, and there is no J3, J4, or tesseral field. '
    'A planar run pins that axis to +z. '
    'Full spin and wobble tracks angular momentum and a figure axis, evaluates J2 on the current axis, '
    'and stores the quadrupole torque as spin instead of an unmodeled residual. '
    'Moments are C = k M R^2 and A = C - J2 M R^2. '
    'The figure axis rotates about the angular momentum at |L| / A. '
    'Catalog sidereal periods and obliquities are the initial principal-axis spins. '
    'Bodies with no listed J2 stay spherical. Bodies with no listed spin start at rest. '
    'A collision remnant, an inserted body, and a particle population are not given a spin or a J2. '
    'Close-pass tides remain spin-free and do not write into this state.'
)


def resolved_level(value):
    """Experiment-file null and the explicit point-mass name are the same level."""
    if value in (None, 'point_mass'):
        return 'point_mass'
    if value not in BODY_DETAIL_LEVELS:
        raise ValueError("physics.body_detail must be null, 'point_mass', 'fixed_oblateness', or 'spin'.")
    return value


def require_experiment_selection(body_detail, lasting, dimensions, collision):
    """Refuse a physics-section combination this module cannot run."""
    level = resolved_level(body_detail)
    if lasting and level != 'spin':
        raise ValueError(LASTING_NEEDS_SPIN)
    if level == 'spin' and dimensions != 3:
        raise ValueError(SPIN_NEEDS_3D)
    if level != 'point_mass' and collision in _REMNANT_MODES:
        raise ValueError(NO_REMNANT)


def require_runnable(setup):
    """The same limits, read off a built setup."""
    require_experiment_selection(
        setup.body_detail.level, setup.lasting_tides.enabled, len(setup.bodies[0].r), setup.collision_mode)
    if setup.body_detail.level != 'point_mass' and setup.population is not None:
        raise ValueError(NO_POPULATION)


def attach(prepared):
    """Hang the body-detail state on the system when the setup asks for it.

    Point mass leaves ``system.detail`` unset, so the acceleration hook is skipped.
    """
    setup = prepared.setup
    if setup is None or (setup.body_detail.level == 'point_mass' and not setup.lasting_tides.enabled):
        return prepared
    dimension = int(prepared.system.r.shape[1])
    prepared.system.detail = BodyMotion.from_bodies(
        prepared.body_ids, prepared.body_specs, setup.body_detail.level, setup.lasting_tides.enabled, dimension)
    return prepared


def pole_from_obliquity(degrees):
    """Unit pole tipped from +z toward +x by ``degrees``."""
    tilt = math.radians(float(degrees))
    return np.array([math.sin(tilt), 0.0, math.cos(tilt)])


def _lookup(spec, body_id):
    key = spec.catalog_key or body_id
    return _DEFAULTS.get(key)


def _lag_years(row):
    if row is None or not row['k2'] or row['lag'] == 0.0:
        return 0.0
    if row['lag'] is None:
        from .lasting_tides import earth_lag_seconds
        seconds = earth_lag_seconds()
    else:
        seconds = float(row['lag'])
    return seconds / YEAR_SECONDS


class BodyMotion:
    """Per-body J2, figure axis, and, at the spin level, angular momentum.

    Arrays follow the system row order. ``lag`` is in years. Spin vectors are
    radians per year.
    """

    def __init__(self, level, lasting, j2, radius, pole, angular_momentum, inertia_a, inertia_c, k2, lag):
        self.level = level
        self.lasting = bool(lasting)
        self.j2 = np.asarray(j2, dtype=float)
        self.radius = np.asarray(radius, dtype=float)
        self.pole = np.asarray(pole, dtype=float)
        self.angular_momentum = None if angular_momentum is None else np.asarray(angular_momentum, dtype=float)
        self.inertia_a = np.asarray(inertia_a, dtype=float)
        self.inertia_c = np.asarray(inertia_c, dtype=float)
        self.k2 = np.asarray(k2, dtype=float)
        self.lag = np.asarray(lag, dtype=float)

    @classmethod
    def from_bodies(cls, body_ids, specs, level, lasting, dimension):
        count = len(body_ids)
        j2 = np.zeros(count)
        radius = np.zeros(count)
        pole = np.zeros((count, 3))
        inertia_a = np.zeros(count)
        inertia_c = np.zeros(count)
        k2 = np.zeros(count)
        lag = np.zeros(count)
        angular = np.zeros((count, 3)) if level == 'spin' else None
        for index, (body_id, spec) in enumerate(zip(body_ids, specs)):
            radius[index] = 0.0 if spec.radius is None else float(spec.radius)
            row = _lookup(spec, body_id)
            mass = 0.0 if spec.mass is None else float(spec.mass)
            if dimension == 2 or row is None:
                pole[index] = (0.0, 0.0, 1.0)
            else:
                pole[index] = pole_from_obliquity(row['obliquity'])
            if row is None or mass <= 0.0 or radius[index] <= 0.0:
                continue
            j2[index] = float(row['j2'])
            inertia_c[index] = float(row['inertia']) * mass * radius[index] ** 2
            inertia_a[index] = inertia_c[index] - j2[index] * mass * radius[index] ** 2
            if inertia_a[index] <= 0.0:
                raise ValueError(f'{body_id} has a J2 larger than its inertia factor. The equatorial moment would not be positive.')
            k2[index] = float(row['k2'])
            lag[index] = _lag_years(row) if lasting else 0.0
            if angular is not None and row['period']:
                omega = 2.0 * math.pi * YEAR_SECONDS / float(row['period'])
                angular[index] = inertia_c[index] * omega * pole[index]
        return cls(level, lasting, j2, radius, pole, angular, inertia_a, inertia_c, k2, lag)

    def copy(self):
        return BodyMotion(
            self.level, self.lasting, self.j2.copy(), self.radius.copy(), self.pole.copy(),
            None if self.angular_momentum is None else self.angular_momentum.copy(),
            self.inertia_a.copy(), self.inertia_c.copy(), self.k2.copy(), self.lag.copy())

    def sliced(self, indices):
        index = np.asarray(list(indices), dtype=int)
        angular = None if self.angular_momentum is None else self.angular_momentum[index].copy()
        return BodyMotion(
            self.level, self.lasting, self.j2[index].copy(), self.radius[index].copy(), self.pole[index].copy(),
            angular, self.inertia_a[index].copy(), self.inertia_c[index].copy(), self.k2[index].copy(),
            self.lag[index].copy())

    def align_pole(self, index):
        """Put body ``index`` on the +z axis and keep its spin rate."""
        rate = 0.0 if self.angular_momentum is None else float(np.linalg.norm(self.omega(index)))
        self.pole[index] = (0.0, 0.0, 1.0)
        if self.angular_momentum is not None:
            self.angular_momentum[index] = np.array([0.0, 0.0, self.inertia_c[index] * rate])

    def omega(self, index):
        """Inertial spin in radians per year. Zero when this level does not track spin."""
        if self.angular_momentum is None:
            return np.zeros(3)
        moment_c = self.inertia_c[index]
        if moment_c <= 0.0:
            return np.zeros(3)
        angular = self.angular_momentum[index]
        moment_a = self.inertia_a[index]
        axis = self.pole[index]
        if moment_a <= 0.0 or abs(moment_c - moment_a) <= 1e-14 * moment_c:
            return angular / moment_c
        parallel = float(np.dot(angular, axis))
        return (angular - parallel * axis) / moment_a + (parallel / moment_c) * axis

    def spin_rates(self):
        """One inertial spin vector per body, or None when spin is not tracked."""
        if self.level != 'spin':
            return None
        return np.vstack([self.omega(index) for index in range(len(self.pole))])

    def acceleration(self, system):
        extra = j2_acceleration(system.r, system.masses, self.j2, self.radius, self.pole)
        if self.lasting:
            from .lasting_tides import ctl_acceleration
            extra = extra + ctl_acceleration(system, self)
        return extra

    def advance(self, system, dt):
        """Store the quadrupole torque, and the lasting-tide torque when that module is on."""
        if self.level != 'spin' or not dt:
            return
        torque = quadrupole_torque(system, self)
        if self.lasting:
            from .lasting_tides import ctl_torque
            torque = torque + ctl_torque(system, self)
        self.angular_momentum += torque * dt
        self._advance_figures(dt)

    def _advance_figures(self, dt):
        for index in range(len(self.pole)):
            moment_a = self.inertia_a[index]
            moment_c = self.inertia_c[index]
            if moment_a <= 0.0 or moment_c <= 0.0 or abs(moment_c - moment_a) <= 1e-14 * moment_c:
                continue
            angular = self.angular_momentum[index]
            magnitude = float(np.linalg.norm(angular))
            if magnitude == 0.0:
                continue
            self.pole[index] = _rodrigues(self.pole[index], angular / magnitude, magnitude / moment_a * dt)
            norm = float(np.linalg.norm(self.pole[index]))
            if norm > 0.0:
                self.pole[index] /= norm

    def document(self):
        angular = None if self.angular_momentum is None else self.angular_momentum.tolist()
        return {
            'level': self.level,
            'lasting': self.lasting,
            'j2': self.j2.tolist(),
            'radius': self.radius.tolist(),
            'pole': self.pole.tolist(),
            'angular_momentum': angular,
            'inertia_a': self.inertia_a.tolist(),
            'inertia_c': self.inertia_c.tolist(),
            'k2': self.k2.tolist(),
            'lag_years': self.lag.tolist(),
        }

    @classmethod
    def from_document(cls, value, count):
        level = value['level']
        if level not in ('fixed_oblateness', 'spin'):
            raise ValueError('Body detail checkpoint level must be fixed_oblateness or spin.')
        j2 = _vector(value['j2'], count, 'j2')
        radius = _vector(value['radius'], count, 'radius')
        pole = _rows(value['pole'], count, 3, 'pole')
        inertia_a = _vector(value['inertia_a'], count, 'inertia_a')
        inertia_c = _vector(value['inertia_c'], count, 'inertia_c')
        k2 = _vector(value['k2'], count, 'k2')
        lag = _vector(value['lag_years'], count, 'lag_years')
        raw_angular = value['angular_momentum']
        if level == 'spin':
            if raw_angular is None:
                raise ValueError('Spin body detail checkpoint is missing angular momentum.')
            angular = _rows(raw_angular, count, 3, 'angular_momentum')
        elif raw_angular is not None:
            raise ValueError('Fixed oblateness does not store angular momentum.')
        else:
            angular = None
        return cls(level, bool(value['lasting']), j2, radius, pole, angular, inertia_a, inertia_c, k2, lag)

    @property
    def disclosure(self):
        tides = ' Lasting tides are on (constant-time-lag-v1).' if self.lasting else ''
        return f'Body detail is {self.level}.{tides} {ASSUMPTIONS}'


def checkpoint_block(prepared):
    motion = prepared.system.detail
    if motion is None:
        return None
    return motion.document()


def policies_and_motion(value, count):
    """Policies and spin state restored from a formation checkpoint. Absent means point mass."""
    if value is None:
        return BodyDetailPolicy(), LastingTidePolicy(), None
    if type(value) is not dict:
        raise ValueError('Body detail checkpoint must be an object.')
    for key in ('level', 'lasting', 'j2', 'radius', 'pole', 'angular_momentum', 'inertia_a', 'inertia_c', 'k2', 'lag_years'):
        if key not in value:
            raise ValueError(f'Body detail checkpoint is missing {key}.')
    level = value['level']
    lasting = value['lasting']
    if level not in ('fixed_oblateness', 'spin'):
        raise ValueError('Body detail checkpoint level must be fixed_oblateness or spin.')
    if type(lasting) is not bool:
        raise ValueError('Body detail checkpoint field lasting must be true or false.')
    if lasting and level != 'spin':
        raise ValueError(LASTING_NEEDS_SPIN)
    motion = BodyMotion.from_document(value, count)
    tides = LastingTidePolicy(enabled=True) if lasting else LastingTidePolicy()
    return BodyDetailPolicy(level), tides, motion


def j2_acceleration(positions, masses, j2, radii, poles):
    """Mutual J2 acceleration. The oblate body receives the reaction."""
    acc = np.zeros_like(positions)
    dimension = positions.shape[1]
    for index in range(len(masses)):
        if j2[index] == 0.0 or radii[index] <= 0.0 or masses[index] <= 0.0:
            continue
        for other in range(len(masses)):
            if other == index:
                continue
            rel = positions[other] - positions[index]
            r2 = float(np.dot(rel, rel))
            if r2 <= 0.0:
                continue
            distance = math.sqrt(r2)
            factor = 1.5 * j2[index] * G * masses[index] * radii[index] ** 2 / (r2 * r2 * distance)
            if dimension == 2:
                delta = -factor * rel
            else:
                projection = float(np.dot(poles[index], rel))
                delta = factor * ((5.0 * projection * projection / r2 - 1.0) * rel - 2.0 * projection * poles[index])
            acc[other] += delta
            if masses[other] > 0.0:
                acc[index] -= delta * (masses[other] / masses[index])
    return acc


def quadrupole_torque(system, motion):
    """Gravitational torque on each spin from the other bodies' monopoles."""
    torque = np.zeros((len(system.masses), 3))
    if system.r.shape[1] != 3:
        return torque
    for index in range(len(system.masses)):
        difference = motion.inertia_c[index] - motion.inertia_a[index]
        if difference == 0.0 or system.masses[index] <= 0.0:
            continue
        axis = motion.pole[index]
        for other in range(len(system.masses)):
            if other == index or system.masses[other] <= 0.0:
                continue
            rel = system.r[other] - system.r[index]
            r2 = float(np.dot(rel, rel))
            if r2 <= 0.0:
                continue
            distance = math.sqrt(r2)
            projection = float(np.dot(axis, rel))
            # Minus the orbital torque r × F of the J2 force, so the pair's angular momentum is kept.
            torque[index] -= (3.0 * G * system.masses[other] * difference * projection / (r2 * r2 * distance)) * np.cross(axis, rel)
    return torque


def nodal_precession_rate(semi_major, inclination, mass, radius, j2, companion_mass=0.0):
    """Secular dΩ/dt for a circular orbit about a fixed J2 bulge whose pole is +z.

    ``-3/2 n J2 (R/a)^2 cos i``, with ``n = sqrt(G(M+m)/a^3)``. Radians per year.
    """
    mean_motion = math.sqrt(G * (mass + companion_mass) / semi_major ** 3)
    return -1.5 * mean_motion * j2 * (radius / semi_major) ** 2 * math.cos(inclination)


def leo_initial_setup(*, level='fixed_oblateness', primary_kind='planet', collision_mode='none',
                      altitude_km=LEO_ALTITUDE_KM, inclination_deg=LEO_INCLINATION_DEG, steps_per_orbit=800):
    """Earth and a tiny inclined satellite. The pole is aligned to +z after ``build_system``."""
    from .model import BodySpec, InitialBody, InitialSetup
    earth = CATALOG['earth']
    if primary_kind != earth.kind:
        earth = _replace_kind(earth, primary_kind)
    # Center-to-center distance is the catalog radius plus the altitude.
    semi_major = float(earth.radius) + altitude_km / AU_KM
    inclination = math.radians(inclination_deg)
    mass = 1e-18
    mean_motion = math.sqrt(G * (earth.mass + mass) / semi_major ** 3)
    speed = mean_motion * semi_major
    satellite = BodySpec('LEO satellite', 'planet', mass, 1e-12, catalog_key=None)
    bodies = (
        InitialBody('earth', earth, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        InitialBody('satellite', satellite, (semi_major, 0.0, 0.0), (0.0, speed * math.cos(inclination), speed * math.sin(inclination))),
    )
    period = 2.0 * math.pi / mean_motion
    policy = BodyDetailPolicy(level)
    return InitialSetup(
        'Low Earth orbit', bodies, period / steps_per_orbit, collision_mode=collision_mode, body_detail=policy), semi_major, inclination


def measure_leo_node(*, orbits=12, level='fixed_oblateness', steps_per_orbit=800, primary_kind='planet',
                     collision_mode='none'):
    """Integrate an inclined 400 km orbit and compare its node to the J2 formula.

    The figure axis is +z, so the inclination is measured from Earth's equator.
    The catalog obliquity is not used for this comparison.
    """
    from .initialization import build_system
    setup, semi_major, inclination = leo_initial_setup(
        level=level, steps_per_orbit=steps_per_orbit, primary_kind=primary_kind, collision_mode=collision_mode)
    prepared = build_system(setup)
    earth = CATALOG['earth']
    if prepared.system.detail is not None:
        prepared.system.detail.align_pole(0)
    analytic = nodal_precession_rate(semi_major, inclination, earth.mass, earth.radius, EARTH_J2 if level != 'point_mass' else 0.0, 1e-18)
    measured = _measure_node(prepared, int(round(orbits * steps_per_orbit)))
    return NodeComparison(analytic * _DEG_PER_DAY, measured * _DEG_PER_DAY, orbits, level, 'plus_z')


@dataclass(frozen=True)
class NodeComparison:
    analytic_deg_per_day: float
    measured_deg_per_day: float
    orbits: float
    level: str
    pole: str


_DEG_PER_DAY = (180.0 / math.pi) / 365.25


def _measure_node(prepared, steps):
    contact = prepared.setup is not None and prepared.setup.collision_mode != 'none'
    times = np.empty(steps + 1)
    nodes = np.empty(steps + 1)
    times[0] = prepared.system.time
    nodes[0] = _node(prepared.system.r[1] - prepared.system.r[0], prepared.system.v[1] - prepared.system.v[0])
    for index in range(steps):
        if contact:
            from .collisions import advance_prepared
            prepared = advance_prepared(prepared).prepared
        else:
            prepared.system.step()
        times[index + 1] = prepared.system.time
        nodes[index + 1] = _node(prepared.system.r[1] - prepared.system.r[0], prepared.system.v[1] - prepared.system.v[0])
    return float(np.polyfit(times, np.unwrap(nodes), 1)[0])


def _node(rel, vel):
    angular = np.cross(rel, vel)
    return math.atan2(float(angular[0]), -float(angular[1]))


def _replace_kind(spec, kind):
    from dataclasses import replace
    return replace(spec, kind=kind)


def _rodrigues(vector, axis, angle):
    cos = math.cos(angle)
    sin = math.sin(angle)
    return vector * cos + np.cross(axis, vector) * sin + axis * float(np.dot(axis, vector)) * (1.0 - cos)


def _vector(value, count, label):
    if type(value) is not list or len(value) != count:
        raise ValueError(f'Body detail checkpoint {label} must have one entry per active body.')
    result = np.empty(count)
    for index, item in enumerate(value):
        if isinstance(item, bool) or type(item) not in (int, float) or not math.isfinite(item):
            raise ValueError(f'Body detail checkpoint {label} must be finite.')
        result[index] = float(item)
    return result


def _rows(value, count, width, label):
    if type(value) is not list or len(value) != count:
        raise ValueError(f'Body detail checkpoint {label} must have one row per active body.')
    result = np.empty((count, width))
    for index, row in enumerate(value):
        if type(row) is not list or len(row) != width:
            raise ValueError(f'Body detail checkpoint {label} rows must have {width} components.')
        for component, item in enumerate(row):
            if isinstance(item, bool) or type(item) not in (int, float) or not math.isfinite(item):
                raise ValueError(f'Body detail checkpoint {label} must be finite.')
            result[index, component] = float(item)
    return result
