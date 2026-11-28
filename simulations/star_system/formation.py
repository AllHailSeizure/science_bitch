"""Late-stage assembly with parcel ancestry, checkpoints, and branched runs.

Gravity is the base. Collision handling, gas drag, and close-pass tides are
separate switches on the experiment setup. The default collision rule is
perfect merge. Drag and tides are off unless the setup stores them. Dimensions
and the adaptive step are separate from those switches.

Planar two-component state and the authored fixed step remain the default.
A run may instead store three components, use a constant-density rocky
radius, and shrink the macro step during a close pass. A collision does not
end the run unless the setup chooses stop on contact. The overhead
observatory still draws the plane. Tree gravity is not used.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
from bisect import bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
import tracemalloc

import numpy as np

from .collisions import CollisionEvent, advance_prepared
from .core import G, SOLAR_RADIUS_AU, System, kepler_period
from .data import AU_KM, PLANETS
from .history import StateFrame
from .initialization import PreparedSystem, build_system
from .model import (
    FORMATION_COLLISION_MODES, SPEED_ANGLE_VERSION, BodySpec, DragPolicy, InitialBody, InitialSetup, TidalPolicy,
)
from .persistence import SOLVER_BUILD, setup_digest
from .playback import Playback
from .recording import RecordedRun

FORMATION_MODEL = 'late-stage-perfect-merge-v1'
REPRESENTATION_ID = 'one-embryo-one-parcel-v1'
COLLISION_MODEL_ID = 'nonstellar-perfect-merge-v1'
IGNORE_CONTACT_ID = 'ignore-contact-v1'
STOP_ON_CONTACT_ID = 'stop-on-contact-v1'
CHECKPOINT_SCHEMA = 'formation-checkpoint-v1'
STAR_RADIUS = float(SOLAR_RADIUS_AU)

DEMONSTRATION_DT = 0.02
DEMONSTRATION_EJECTION_RADIUS = 6.0
BENCHMARK_EMBRYO_COUNT = 80
BENCHMARK_SEED = 48
BENCHMARK_DURATION_YEARS = 1.0
BENCHMARK_DT = 0.025
BENCHMARK_EJECTION_RADIUS = 8.0
BENCHMARK_SAMPLE_STRIDE = 5
BENCHMARK_INNER = 1.5
BENCHMARK_OUTER = 3.2
BENCHMARK_EMBRYO_MASS = 2e-6
BENCHMARK_EMBRYO_RADIUS = 0.04
BENCHMARK_ECCENTRICITY = 0.12
# Below this floor a close pass is not stepped. The positive Verlet solver
# stays in place; REBOUND is the documented candidate if a run reaches it.
ADAPTIVE_DT_MIN = 1e-10
ADAPTIVE_GROWTH = 2.
PHYSICAL_PROBE_COUNT = 8
PHYSICAL_PROBE_SEED = 70
PHYSICAL_PROBE_DURATION_YEARS = 0.25
PHYSICAL_PROBE_INCLINATION = 0.1
ROCKY_YEAR_COUNT = 80
ROCKY_YEAR_SEED = 48
ROCKY_YEAR_DURATION_YEARS = 1.0
ROCKY_YEAR_INCLINATION = 0.1
FLYBY_DURATION_YEARS = 2.0
FLYBY_DT = 0.01

_ASSUMPTIONS = (
    'Late-stage assembly: one star already exists, and every other body is a massive rocky embryo.',
    'Each embryo starts as one material parcel. Parcel ids survive every outcome. Under the speed-angle rule a parcel\'s mass can be shared by fragment bodies and by recorded lost mass.',
    'The collision rule is stored on the experiment setup. Perfect merge keeps one remnant and is unrealistic as a general law: it is not what a real Venus-Earth collision would do.',
    'Perfect merge, when selected, does not fragment. One remnant keeps the mass and linear momentum, and the relative kinetic energy is recorded as lost.',
    'The speed-angle rule merges at or below mutual escape, grazes a faster wider impact, and fragments a faster closer impact. Fragments are bodies. Lost mass is recorded and is not a body.',
    'A nonstellar body is ejected when its two-body specific energy about the star is positive and its star-centric distance has reached the ejection radius. The check runs at macro steps. The body is removed from the active set and its last state is kept.',
    'Ejection is a bookkeeping cut. After it, the frozen body no longer exchanges gravity with the active set.',
    'Stellar contact is logged as unmodeled and the run continues, including when gas drag is on. This prototype does not merge stars. Stop on contact is the only collision choice that ends the run.',
    'Gravity is unsoftened Newtonian gravity. A run is planar unless its positions have a third component. The authored dt is the fixed step, or the maximum step when adaptive control is on. During an unmodeled stellar overlap that pair is evaluated at the contact separation so the step can cross it.',
    'The overhead observatory draws planar motion. A third component is integrated and recorded, and it is not projected into that view.',
)
_QUESTIONS = (
    'Which initial parcels make up a surviving body, and how much mass came from each?',
    'When did mergers, grazes, fragmentations, close encounters, orbital changes, and ejections happen, including for a body that has left an empty region?',
    'If two runs share a checkpoint and one embryo velocity changes, how do their histories differ at the same simulated time?',
    'Do mass, parcel totals, and linear momentum follow the perfect-merge and ejection rules, and does the kinetic-energy drop at a merge match the recorded loss?',
    'How much does the first merger time move when the fixed step changes, while the remnant mass stays the combined parcel mass?',
    'For a disclosed three-dimensional initial state, does vertical motion survive part of an orbit, and does the macro step shrink during a close miss and grow again afterward?',
)
_NOT_MODELED = (
    'A three-dimensional gas disk. Drag uses the existing annular isothermal zones, evaluated at cylindrical radius, with the isothermal scale height as a Gaussian weight in height.',
    'Dust swarms, grain growth, and the swarm-to-body rule. Those stay in the gas-and-dust prototype. Bodies on this path use that prototype\'s quadratic drag when drag is enabled.',
    'Tree-code grouping of distant mass. Direct summation stays in use, and a grouped force would still have to keep parcel identity.',
    'Partial accretion and erosion beyond the disclosed speed-angle lost-mass fraction.',
    'An overhead drawing of the third dimension. The observatory remains a planar view.',
    'Relativity and stellar internal structure. A dedicated stellar merger stays in its own scenario.',
)
_SCOPE = (
    'Keep late-stage embryos, one parcel per body, a stored collision rule, ejection records, ancestry amounts, and checkpoint branching. Perfect merge remains available.',
    'Calculate a run to completion, then scrub and compare it. The same setup can be rerun under each stored rule. Inflated encounter radii make contacts inspectable; they are not rocky surfaces, and the fixed step is not an interactive solver for realistic radii. Rocky radii and an adaptive macro step are a separate schedule and do not replace the stored collision rule.',
    'Log stellar contact and continue. A dedicated stellar-merger model stays in its own scenario.',
)
_GAS = (
    'Gas drag is an opt-in switch. When it is on, this integrator applies the existing quadratic body law and the annular isothermal disk. The standalone gas-and-dust prototype still owns swarms, grain growth, and its own fixed step.',
    'Partial accretion, erosion, hit-and-run, and fragmentation in the literature follow an outcome map such as Stewart and Leinhardt. The stored speed-angle rule is a disclosed speed-and-angle map with tracked fragments and recorded losses; it is not that full model. Perfect merge still cannot send mass to more than one descendant.',
    'Tree gravity is a later optimization. At a few hundred bodies the measured cost is direct summation, and grouping would have to preserve parcel identity to be usable here.',
)
_PROVENANCE_KEYS = ('run_id', 'model', 'representation', 'collision_model', 'seed', 'setup_digest',
                    'parent_checkpoint_id', 'change', 'execution_environment', 'solver_build')
_BODY_KEYS = ('id', 'name', 'kind', 'mass', 'radius', 'status', 'r', 'v', 'contributions', 'reference_id')
_EVENT_KEYS = ('time', 'kind', 'parent_ids', 'remnant_id', 'descendant_ids', 'contributions',
               'lost_relative_kinetic', 'unmodeled_spin', 'location', 'message', 'lost_contributions')
_SAMPLE_KEYS = ('time', 'phase', 'bodies')
_CHANGE_KEYS = ('kind', 'body_id', 'delta_v', 'note')
_DOCUMENT_KEYS = ('schema', 'id', 'model', 'name', 'time', 'dt', 'ejection_radius', 'sample_stride',
                  'status', 'reason', 'completed_steps', 'assumptions', 'provenance', 'active',
                  'removed', 'events', 'samples')
_OPTIONAL_DOCUMENT_KEYS = ('adaptive', 'dt_max', 'dt_min', 'dimensions', 'tides', 'drag')
_COLLISION_MODEL_IDS = (COLLISION_MODEL_ID, SPEED_ANGLE_VERSION, IGNORE_CONTACT_ID, STOP_ON_CONTACT_ID)


@dataclass(frozen=True)
class FormationSpecification:
    """The candidate model this prototype actually runs."""
    starting_stage: str
    representation: str
    collision_model: str
    questions: tuple[str, ...]
    assumptions: tuple[str, ...]
    not_modeled: tuple[str, ...]
    recommended_scope: tuple[str, ...]
    gas_disk_requirements: tuple[str, ...]


SPECIFICATION = FormationSpecification(
    'Late-stage assembly around an existing star, beginning with massive rocky embryos rather than a gas-and-dust disk.',
    'Each simulated object is one body. Every embryo begins as a single material parcel with a stable id and a mass. The prototype does not group parcels into a tree. Gas drag, when enabled, is the annular disk acting on those bodies; dust swarms stay in the gas-and-dust prototype.',
    'The experiment stores one collision rule. Perfect merge (nonstellar-perfect-merge-v1) keeps one remnant and is unrealistic as a general law. The speed-angle rule (speed-angle-merge-graze-fragment-v1) merges, grazes, or fragments from impact speed and impact parameter. Ignore lets an overlap continue with no remnant. Stop on contact ends the run at the first contact and is the only collision choice that does. Fragments stay as bodies and lost mass is recorded. Gas drag and close-pass tides are separate opt-in switches and stay off unless the setup stores them. An unbound body outside the ejection radius leaves the active set and stays in the history.',
    _QUESTIONS, _ASSUMPTIONS, _NOT_MODELED, _SCOPE, _GAS)


def specification():
    """Return the stage, representation, collision model, and questions this prototype can address."""
    return SPECIFICATION


@dataclass(frozen=True)
class Contribution:
    parcel_id: str
    mass: float


@dataclass(frozen=True)
class SampleBody:
    id: str
    name: str
    kind: str
    mass: float
    radius: float
    r: tuple[float, ...]
    v: tuple[float, ...]
    contributions: tuple[Contribution, ...]
    status: str = 'active'
    reference_id: str | None = None


@dataclass(frozen=True)
class Sample:
    time: float
    phase: str | None
    bodies: tuple[SampleBody, ...]


@dataclass(frozen=True)
class FormationEvent:
    time: float
    kind: str
    parent_ids: tuple[str, ...]
    remnant_id: str | None
    contributions: tuple[Contribution, ...]
    descendant_ids: tuple[str, ...]
    lost_relative_kinetic: float
    unmodeled_spin: float
    location: tuple[float, ...]
    message: str
    lost_contributions: tuple[Contribution, ...] = ()


@dataclass(frozen=True)
class VelocityChange:
    """One disclosed change applied when a run branches from a checkpoint."""
    body_id: str
    delta_v: tuple[float, ...]
    note: str

    def __post_init__(self):
        if not isinstance(self.body_id, str) or not self.body_id.strip():
            raise ValueError('Name the embryo whose velocity changes.')
        try:
            components = tuple(self.delta_v)
        except TypeError as exc:
            raise ValueError('delta_v must be two or three finite AU/yr components.') from exc
        if len(components) not in (2, 3):
            raise ValueError('delta_v must be two or three finite AU/yr components.')
        vector = tuple(float(component) for component in components)
        if not all(math.isfinite(component) for component in vector):
            raise ValueError('delta_v must be two or three finite AU/yr components.')
        if all(component == 0.0 for component in vector):
            raise ValueError('A branch needs a nonzero velocity change.')
        if not isinstance(self.note, str) or not self.note.strip():
            raise ValueError('Describe the disclosed change in note.')
        object.__setattr__(self, 'body_id', self.body_id.strip())
        object.__setattr__(self, 'delta_v', vector)
        object.__setattr__(self, 'note', self.note.strip())


@dataclass(frozen=True)
class Ancestry:
    """Parcel amounts, events, and samples for one body, including a removed one."""
    body_id: str
    status: str
    contributions: tuple[Contribution, ...]
    events: tuple[FormationEvent, ...]
    descendant_ids: tuple[str, ...]
    samples: tuple[Sample, ...]

    def highlighted(self, sample):
        """Active bodies in this sample that still carry the selected parcels."""
        parcels = {item.parcel_id for item in self.contributions}
        return tuple(body.id for body in sample.bodies
                     if parcels.intersection(item.parcel_id for item in body.contributions))


@dataclass(frozen=True)
class Accounting:
    initial_mass: float
    closed_mass: float
    parcel_error: float
    merge_momentum_error: float
    merge_kinetic_error: float
    ejection_momentum_error: float
    merge_count: int
    ejection_count: int


@dataclass(frozen=True)
class ResolutionPoint:
    dt: float
    status: str
    first_merge_time: float | None
    remnant_mass: float | None
    parent_ids: tuple[str, ...] | None
    survivor_count: int


@dataclass(frozen=True)
class BenchmarkReport:
    hardware: str
    model: str
    seed: int
    body_count: int
    embryo_count: int
    requested_years: float
    simulated_years: float
    calculation_seconds: float
    peak_python_bytes: int
    recording_bytes: int
    sample_count: int
    merge_count: int
    ejection_count: int
    merge_times: tuple[float, ...]
    ejection_times: tuple[float, ...]
    status: str
    reason: str
    survivor_ids: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalProbeReport:
    """One short rocky-radius run. It does not replace the inflated benchmark."""
    hardware: str
    body_count: int
    embryo_count: int
    requested_years: float
    simulated_years: float
    calculation_seconds: float
    status: str
    reason: str
    merge_count: int
    minimum_step: float
    maximum_step: float
    rocky_radius: float
    encounter_radius: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RockyYearReport:
    """One 80-embryo year at the rocky radius on the combined adaptive path."""
    hardware: str
    body_count: int
    embryo_count: int
    requested_years: float
    simulated_years: float
    calculation_seconds: float
    status: str
    reason: str
    minimum_step: float
    maximum_step: float
    collision_mode: str
    drag: str
    tides: str
    merge_count: int
    tide_count: int
    ejection_count: int
    unmodeled_count: int
    dissipated_kinetic: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class FlybySummary:
    """One Sun–Jupiter flyby under one combination of switches."""
    label: str
    collision_mode: str
    drag: str
    tides: str
    status: str
    simulated_years: float
    initial_sun_energy: float
    sun_energy: float
    jupiter_energy: float
    minimum_jupiter_distance: float
    binding: str
    merged: bool
    tide_count: int
    dissipated_kinetic: float


@dataclass(frozen=True)
class FlybyReport:
    """The same slow flyby with gravity alone and with drag and tides."""
    hardware: str
    free: FlybySummary
    drag: FlybySummary
    tides: FlybySummary
    combined: FlybySummary
    conclusion: str


@dataclass(frozen=True)
class SyncedFrame:
    time: float
    left: StateFrame
    right: StateFrame
    common_start: float
    common_end: float


def rocky_radius(mass):
    """Volume-equivalent radius at the fact-sheet Earth density.

    Earth mass and the 6371 km radius are the rounded values in ``PLANETS['earth']``.
    This is a constant-density comparison, not an interior model. At the benchmark
    embryo mass it is about a thousand times smaller than the 0.04 AU encounter radius.
    """
    mass = _positive('Mass', mass)
    earth = PLANETS['earth']
    return float(earth.radius/AU_KM*(mass/earth.mass)**(1./3.))


def _components(vector):
    return tuple(float(component) for component in np.asarray(vector, dtype=float).reshape(-1))


def _star_body(spatial=False):
    zero = (0., 0., 0.) if spatial else (0., 0.)
    return InitialBody('star', BodySpec('Star', 'star', 1., STAR_RADIUS, temperature=5772., color='#ffe2b0',
                                         provenance='formation prototype'), zero, zero)


def _embryo(body_id, name, mass, radius, position, velocity):
    return InitialBody(body_id, BodySpec(name, 'planet', mass, radius, color='#c9b28d', provenance='formation embryo'),
                       position, velocity)


def make_demonstration_setup(dt=DEMONSTRATION_DT):
    """Star, a head-on embryo pair, and one unbound embryo. Radii are inflated."""
    _positive('Fixed dt', dt)
    bodies = (
        _star_body(),
        _embryo('alpha', 'Alpha', 1e-5, .08, (5., 0.), (2.5, 0.)),
        _embryo('beta', 'Beta', 2e-5, .08, (5.5, 0.), (-2.5, 0.)),
        _embryo('escaper', 'Escaper', 1e-5, .05, (-2., 0.), (-12., .3)),
    )
    return InitialSetup('Formation demonstration', bodies, dt, collision_mode='perfect_merge',
                        provenance={'formation_model': FORMATION_MODEL, 'scenario': 'demonstration'})


def make_merger_probe(dt):
    """The demonstration pair without the escaper, for the resolution check."""
    _positive('Fixed dt', dt)
    bodies = (
        _star_body(),
        _embryo('alpha', 'Alpha', 1e-5, .08, (5., 0.), (2.5, 0.)),
        _embryo('beta', 'Beta', 2e-5, .08, (5.5, 0.), (-2.5, 0.)),
    )
    return InitialSetup('Formation merger probe', bodies, dt, collision_mode='perfect_merge',
                        provenance={'formation_model': FORMATION_MODEL, 'scenario': 'merger-probe'})


def _rotate_orbit(position, velocity, inclination, node):
    """Tilt a planar orbit about x, then rotate the line of nodes about z."""
    cosine_i, sine_i = math.cos(inclination), math.sin(inclination)
    cosine_n, sine_n = math.cos(node), math.sin(node)

    def turn(vector):
        x, y = float(vector[0]), float(vector[1])
        y_tilted = y*cosine_i
        z_tilted = y*sine_i
        return (x*cosine_n-y_tilted*sine_n, x*sine_n+y_tilted*cosine_n, z_tilted)

    return turn(position), turn(velocity)


def _ellipse_state(semi_major, eccentricity, periapsis, anomaly, star_mass):
    mu = G*star_mass
    h = math.sqrt(mu*semi_major*(1-eccentricity*eccentricity))
    radius = semi_major*(1-eccentricity*eccentricity)/(1+eccentricity*math.cos(anomaly))
    radial_speed = mu/h*eccentricity*math.sin(anomaly)
    tangential_speed = mu/h*(1+eccentricity*math.cos(anomaly))
    angle = periapsis+anomaly
    radial = np.array([math.cos(angle), math.sin(angle)])
    tangent = np.array([-radial[1], radial[0]])
    return radius*radial, radial*radial_speed+tangent*tangential_speed


def _clear(position, radius, placed):
    """Keep embryos well outside contact so the run does not start mid-merger."""
    for other, other_radius in placed:
        if float(np.linalg.norm(np.asarray(position)-other)) <= 1.5*(radius+other_radius):
            return False
    return True


def make_late_stage_setup(*, count=BENCHMARK_EMBRYO_COUNT, seed=BENCHMARK_SEED, inner=BENCHMARK_INNER,
                          outer=BENCHMARK_OUTER, embryo_mass=BENCHMARK_EMBRYO_MASS,
                          embryo_radius=BENCHMARK_EMBRYO_RADIUS, eccentricity=BENCHMARK_ECCENTRICITY,
                          dt=BENCHMARK_DT, include_unbound=True, radii='encounter', inclination=0.):
    """Seeded annulus of massive embryos, plus one disclosed unbound body when requested.

    Angles and true anomalies come from PCG64. The initial velocities are
    star-centric ellipses. ``radii='encounter'`` keeps the supplied encounter
    radius and planar vectors. ``radii='rocky'`` uses :func:`rocky_radius`.
    A positive ``inclination`` is the maximum tilt in radians; zero leaves
    every vector two-dimensional so the planar seed is unchanged.
    """
    if type(count) is not int or isinstance(count, bool) or not 1 <= count <= 200:
        raise ValueError('Embryo count must be an integer from 1 to 200.')
    if type(seed) is not int or isinstance(seed, bool) or seed < 0:
        raise ValueError('Formation seed must be a nonnegative integer.')
    if radii not in ('encounter', 'rocky'):
        raise ValueError("radii must be 'encounter' or 'rocky'.")
    _positive('Inner radius', inner)
    _positive('Outer radius', outer)
    if inner >= outer:
        raise ValueError('The annulus needs inner < outer, in AU.')
    _positive('Embryo mass', embryo_mass)
    _positive('Embryo radius', embryo_radius)
    _positive('Fixed dt', dt)
    if not math.isfinite(eccentricity) or not 0 <= eccentricity < 1:
        raise ValueError('Eccentricity must be finite and in [0, 1).')
    if not math.isfinite(inclination) or inclination < 0 or inclination > math.pi/2:
        raise ValueError('Inclination must be finite and in [0, pi/2] radians.')
    if radii == 'rocky':
        embryo_radius = rocky_radius(embryo_mass)
    if inner*(1-eccentricity) <= STAR_RADIUS+embryo_radius:
        raise ValueError('Annulus periapses intersect the star or an embryo surface.')
    spatial = inclination > 0
    rng = np.random.Generator(np.random.PCG64(seed))
    star = _star_body(spatial)
    placed = [(np.zeros(3 if spatial else 2), star.spec.radius)]
    bodies = [star]
    for index in range(count):
        semi_major = inner+(outer-inner)*index/max(count-1, 1)
        ecc = eccentricity*(.75+.25*float(rng.random()))
        tilt = inclination*float(rng.random()) if spatial else 0.
        node = float(rng.random())*2*math.pi if spatial else 0.
        body = None
        for _attempt in range(80):
            position, velocity = _ellipse_state(semi_major, ecc, float(rng.random())*2*math.pi,
                                                float(rng.random())*2*math.pi, 1.)
            if spatial:
                position, velocity = _rotate_orbit(position, velocity, tilt, node)
            if _clear(position, embryo_radius, placed):
                body = _embryo(f'embryo-{index:03d}', f'Embryo {index}', embryo_mass, embryo_radius,
                               _components(position), _components(velocity))
                placed.append((np.array(position, dtype=float), embryo_radius))
                break
        if body is None:
            raise ValueError('Embryos overlap. Reduce count or radius, or widen the annulus.')
        bodies.append(body)
    if include_unbound:
        position = np.array([outer+1.5, 0., 0.] if spatial else [outer+1.5, 0.])
        step = np.array([.5, 0., 0.] if spatial else [.5, 0.])
        while not _clear(position, embryo_radius, placed):
            position = position+step
        velocity = (12., .4, 0.) if spatial else (12., .4)
        bodies.append(_embryo('unbound', 'Unbound', embryo_mass, embryo_radius, _components(position), velocity))
    return InitialSetup('Late-stage assembly', tuple(bodies), dt, collision_mode='perfect_merge',
                        provenance={'formation_model': FORMATION_MODEL, 'scenario': 'late-stage', 'seed': seed,
                                    'generator': 'PCG64', 'embryo_count': count, 'include_unbound': bool(include_unbound),
                                    'radius_model': radii, 'inclination': float(inclination)})


def make_inclined_circular_setup(*, semi_major=1., inclination=math.pi/6, mass=1e-6, dt=None):
    """One embryo on a circular orbit tilted out of the plane."""
    _positive('Semi-major axis', semi_major)
    _positive('Embryo mass', mass)
    if not math.isfinite(inclination) or not 0 < inclination <= math.pi/2:
        raise ValueError('Inclination must be finite and in (0, pi/2] radians.')
    if dt is None:
        dt = kepler_period(semi_major, G*(1.+mass))/600.
    _positive('Fixed dt', dt)
    position, velocity = _rotate_orbit(*_ellipse_state(semi_major, 0., 0., 0., 1.), inclination, 0.)
    bodies = (
        _star_body(spatial=True),
        _embryo('embryo', 'Embryo', mass, rocky_radius(mass), position, velocity),
    )
    return InitialSetup('Inclined circular embryo', bodies, dt, collision_mode='perfect_merge',
                        provenance={'formation_model': FORMATION_MODEL, 'scenario': 'inclined-circular',
                                    'inclination': float(inclination), 'radius_model': 'rocky'})


def make_close_pass_setup(*, dt=0.02, mass=BENCHMARK_EMBRYO_MASS):
    """Two rocky embryos on a fast miss, far enough from the star that the pair sets the step.

    The impact parameter is 0.01 AU and the relative speed is 6 AU/yr. The sum of
    the rocky radii is far below that miss, so perfect merge should not fire.
    """
    _positive('Fixed dt', dt)
    _positive('Embryo mass', mass)
    radius = rocky_radius(mass)
    bodies = (
        _star_body(spatial=True),
        _embryo('alpha', 'Alpha', mass, radius, (40., 0., 0.), (0., 0., 0.)),
        _embryo('beta', 'Beta', mass, radius, (40.01, -0.5, 0.), (0., 6., 0.)),
    )
    return InitialSetup('Physical-radius close miss', bodies, dt, collision_mode='perfect_merge',
                        provenance={'formation_model': FORMATION_MODEL, 'scenario': 'close-pass',
                                    'radius_model': 'rocky'})


def _positive(label, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be finite and positive.')
    return float(value)


def _validated_composition(composition, prepared):
    """Optional starting ledger. The default is one parcel named for each body."""
    if not isinstance(composition, Mapping):
        raise ValueError('Composition must map body ids to parcel masses.')
    if set(composition) != set(prepared.body_ids):
        raise ValueError('Composition must name every active body and no other id.')
    result = {}
    for body_id in prepared.body_ids:
        parcels = composition[body_id]
        if not isinstance(parcels, Mapping) or not parcels:
            raise ValueError(f'{body_id} needs at least one parcel.')
        cleaned = {}
        for parcel_id, mass in parcels.items():
            if not isinstance(parcel_id, str) or not parcel_id.strip():
                raise ValueError('Parcel ids must be nonempty strings.')
            if parcel_id in cleaned:
                raise ValueError(f'{body_id} repeats parcel {parcel_id}.')
            cleaned[parcel_id.strip()] = _positive('Parcel mass', mass)
        total = sum(cleaned.values())
        index = prepared.body_ids.index(body_id)
        body_mass = float(prepared.system.masses[index])
        if abs(total-body_mass) > 1e-9*max(1., abs(body_mass)):
            raise ValueError(f'Parcels for {body_id} must add up to its mass.')
        result[body_id] = cleaned
    return result


def _contributions(mapping):
    return tuple(Contribution(key, float(mapping[key])) for key in sorted(mapping) if mapping[key] > 0)


def _weighted_location(bodies, parent_ids):
    chosen = [next(body for body in bodies if body.id == parent) for parent in parent_ids]
    total = sum(body.mass for body in chosen)
    width = len(chosen[0].r)
    return tuple(sum(body.mass*body.r[axis] for body in chosen)/total for axis in range(width))


def _split_composition(composition, shares):
    """Split parcel masses across descendants and a recorded loss. Amounts sum to the parents."""
    total = sum(mass for _key, mass in shares)
    if total <= 0:
        raise ValueError('A fragment split needs positive mass.')
    largest = max(shares, key=lambda item: item[1])[0]
    assigned = {key: {} for key, _mass in shares}
    for parcel, mass in composition.items():
        remaining = float(mass)
        for key, share in shares:
            if key == largest:
                continue
            portion = float(mass)*float(share)/total
            if portion > 0:
                assigned[key][parcel] = portion
                remaining -= portion
        if remaining > 0:
            assigned[largest][parcel] = remaining
    return assigned


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError('Formation state must be finite.')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Formation state must be finite.')
    return result


def _momentum(bodies):
    width = len(bodies[0].v) if bodies else 2
    total = np.zeros(width)
    for body in bodies:
        total += body.mass*np.asarray(body.v, dtype=float)
    return total


def _kinetic(bodies):
    return sum(.5*body.mass*float(np.dot(body.v, body.v)) for body in bodies)


def _max_abs(vector):
    return float(np.max(np.abs(vector))) if len(np.atleast_1d(vector)) else 0.


def _run_id(setup_digest_value, seed, parent_id, change):
    payload = {'setup_digest': setup_digest_value, 'seed': seed, 'parent_checkpoint_id': parent_id,
               'change': None if change is None else _change_document(change)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:16]


def _change_document(change):
    return {'kind': 'velocity', 'body_id': change['body_id'],
            'delta_v': [_num(component) for component in change['delta_v']], 'note': change['note']}


def _combination_text(collision_mode, drag, tides, dimensions, adaptive):
    drag_text = 'off' if not drag.enabled else f'on ({drag.version}, {drag.disk} disk)'
    tide_text = 'off' if not tides.enabled else f'on ({tides.version})'
    step = 'adaptive' if adaptive else 'fixed'
    width = f'{dimensions}D' if dimensions else 'unspecified dimension'
    return (f'Physics switches: collision {collision_mode}, gas drag {drag_text}, tides {tide_text}, '
            f'{width}, {step} step. A contact does not end the run unless collision is stop_on_contact.')


def with_physics(setup, *, collision_mode=None, tides=None, drag=None):
    """Copy a setup with any combination of the collision, tide, and drag switches."""
    from .model import with_collision_rule
    if collision_mode is not None:
        setup = with_collision_rule(setup, collision_mode)
    updates = {}
    if tides is not None:
        updates['tides'] = tides if isinstance(tides, TidalPolicy) else TidalPolicy(enabled=bool(tides))
    if drag is not None:
        updates['drag'] = drag if isinstance(drag, DragPolicy) else DragPolicy(enabled=True if drag is True else False)
    if not updates:
        return setup
    return replace(setup, **updates)


def collision_model_id(setup):
    """Provenance id for the collision switch actually stored on the setup."""
    mode = setup.collision_mode
    if mode == 'speed_angle':
        return SPEED_ANGLE_VERSION
    if mode == 'ignore':
        return IGNORE_CONTACT_ID
    if mode == 'stop_on_contact':
        return STOP_ON_CONTACT_ID
    return COLLISION_MODEL_ID


def _mode_from_collision_model(collision_model):
    return {SPEED_ANGLE_VERSION: 'speed_angle', IGNORE_CONTACT_ID: 'ignore',
            STOP_ON_CONTACT_ID: 'stop_on_contact'}.get(collision_model, 'perfect_merge')


def _new_provenance(setup, seed):
    provenance = {'run_id': '', 'model': FORMATION_MODEL, 'representation': REPRESENTATION_ID,
                  'collision_model': collision_model_id(setup), 'seed': seed, 'setup_digest': setup_digest(setup),
                  'parent_checkpoint_id': None, 'change': None,
                  'execution_environment': {'numpy': np.__version__}, 'solver_build': SOLVER_BUILD}
    provenance['run_id'] = _run_id(provenance['setup_digest'], seed, None, None)
    return provenance


class FormationExperiment:
    """One formation integration with a parcel ledger and recorded samples."""

    def __init__(self, setup, *, ejection_radius, sample_stride=1, seed=None, adaptive=False, dt_min=None, composition=None):
        if not isinstance(setup, InitialSetup):
            raise TypeError('Formation starts from an InitialSetup.')
        if setup.collision_mode not in FORMATION_COLLISION_MODES:
            raise ValueError('Formation collision modes: ignore, stop_on_contact, perfect_merge, speed_angle.')
        if not isinstance(setup.drag, DragPolicy) or not isinstance(setup.tides, TidalPolicy):
            raise ValueError('Formation stores a drag policy and a tide policy on the setup.')
        if seed is not None and (type(seed) is not int or isinstance(seed, bool) or seed < 0):
            raise ValueError('Formation seed must be a nonnegative integer or omitted.')
        if type(sample_stride) is not int or isinstance(sample_stride, bool) or sample_stride < 1:
            raise ValueError('sample_stride must be a positive integer.')
        if not isinstance(adaptive, bool):
            raise ValueError('adaptive must be a boolean.')
        self.ejection_radius = _positive('Ejection radius', ejection_radius)
        self.sample_stride = sample_stride
        self.adaptive = adaptive
        stars = [body for body in setup.bodies if body.spec.kind == 'star']
        if len(stars) != 1:
            raise ValueError('The formation prototype follows embryos around exactly one star.')
        for body in setup.bodies:
            if body.spec.kind == 'particle' or body.spec.mass is None or body.spec.mass <= 0:
                raise ValueError('Formation bodies must be massive. Test particles are outside this prototype.')
            if body.spec.radius is None:
                raise ValueError(f'{body.spec.name} needs a radius in AU.')
        self.name = setup.name
        self.prepared = build_system(setup)
        if composition is None:
            self.composition = {body_id: {body_id: float(mass)}
                                for body_id, mass in zip(self.prepared.body_ids, self.prepared.system.masses)}
        else:
            self.composition = _validated_composition(composition, self.prepared)
        self.removed = {}
        self.events = []
        self.samples = []
        self.status = 'running'
        self.reason = ''
        self.completed_steps = 0
        self._applied_outcomes = 0
        self.accepted_steps = []
        self.dimensions = int(self.prepared.system.r.shape[1])
        self.dt_max = float(self.prepared.system.dt)
        if self.adaptive:
            self.dt_min = ADAPTIVE_DT_MIN if dt_min is None else _positive('Adaptive floor', dt_min)
            if self.dt_min > self.dt_max:
                raise ValueError('The adaptive floor must be at most the authored maximum step.')
        elif dt_min is not None:
            raise ValueError('dt_min applies only when adaptive control is on.')
        else:
            self.dt_min = self.dt_max
        self.provenance = _new_provenance(setup, seed)
        self.drag_policy = setup.drag
        self.zones = []
        self.drag_parameters = None
        self.drag_star_mass = None
        self.dissipated_kinetic = 0.
        self.drag_momentum_error = 0.
        self.fluid_held = False
        if setup.drag.enabled:
            from .gas_dust import disk_for_policy
            zones, parameters, star_mass = disk_for_policy(setup.drag)
            self.zones = zones
            self.drag_parameters = parameters
            self.drag_star_mass = star_mass
        extras = [f'Ejection radius is {self.ejection_radius:.6g} AU and is evaluated at macro steps.']
        combo = _combination_text(setup.collision_mode, setup.drag, setup.tides, self.dimensions if hasattr(self, 'dimensions') else None, self.adaptive)
        extras.append(combo)
        if self.dimensions == 2:
            extras.append('This run is planar: every position and velocity has two components.')
        else:
            extras.append('This run is three-dimensional. The overhead observatory does not draw the third component.')
        if self.adaptive:
            extras.append(
                f'Adaptive macro step: at most {self.dt_max:.6g} yr, shrinking to the pair resolution limit, '
                f'and never below {self.dt_min:.6g} yr. It grows by at most a factor of {ADAPTIVE_GROWTH:.0f} after the pass.')
        else:
            extras.append(f'The fixed step is {self.dt_max:.6g} yr.')
        radii = [body.spec.radius for body in setup.bodies if body.spec.kind != 'star']
        if radii and max(radii) > .001:
            extras.append('At least one embryo radius exceeds 0.001 AU. Treat a radius this large as an inflated encounter radius, not a rocky surface.')
        self.assumptions = _ASSUMPTIONS+tuple(extras)
        self._record_current(None, force=True)

    def combination(self):
        """The collision, drag, and tide switches this run is actually using."""
        tides = self.prepared.setup.tides if self.prepared.setup is not None else TidalPolicy()
        drag = self.drag_policy
        return {
            'dimensions': self.dimensions,
            'adaptive': self.adaptive,
            'collision_mode': self.prepared.setup.collision_mode if self.prepared.setup is not None else 'perfect_merge',
            'collision_model': self.provenance['collision_model'],
            'drag': 'off' if not drag.enabled else drag.disk,
            'drag_model': 'off' if not drag.enabled else drag.version,
            'tides': 'off' if not tides.enabled else tides.version,
        }

    @property
    def time(self):
        return float(self.prepared.system.time)

    def describe(self):
        """Simulated time, status, and the model assumptions for this run."""
        status = f'Simulated time {self.time:.6g} yr. Status: {self.status}.'
        detail = f'Model {self.provenance["model"]}.'
        if self.reason:
            detail = f'{detail} {self.reason}'
        switches = _combination_text(
            self.combination()['collision_mode'], self.drag_policy,
            self.prepared.setup.tides if self.prepared.setup is not None else TidalPolicy(),
            self.dimensions, self.adaptive)
        return '\n'.join((status, detail, switches, *self.assumptions))

    def integrate(self, years, *, finish=False):
        """Advance by whole macro steps until simulated time has grown by `years`."""
        if self.status != 'running':
            raise ValueError(f'Cannot integrate a {self.status} run. Branch from a checkpoint to calculate a new future.')
        years = _positive('Duration', years)
        target = self.time+years
        guard = 0
        nominal = self.dt_max if self.adaptive else self.prepared.system.dt
        floor = self.dt_min if self.adaptive else nominal
        limit = min(math.ceil(years/floor)+2, math.ceil(years/nominal)*50000+2)
        while self.status == 'running' and self.time < target-1e-12*max(1., abs(target)):
            self.advance(1)
            guard += 1
            if guard > limit:
                raise ValueError('Formation numerical halt: the clock did not reach the requested time.')
        if finish and self.status == 'running':
            self.status = 'complete'
            self.reason = ''
        return self

    def advance(self, max_steps):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError('max_steps must be a nonnegative integer.')
        if self.status != 'running':
            raise ValueError(f'Cannot integrate a {self.status} run. Branch from a checkpoint to calculate a new future.')
        done = 0
        while done < max_steps and self.status == 'running':
            scheduled = self._scheduled_dt()
            if scheduled is None:
                self.status = 'halted'
                limit = self.prepared.system.resolution_limit()
                self.reason = (
                    'Adaptive timestep halt: the pair resolution limit '
                    f'{limit:.6g} yr is below the floor of {self.dt_min:.6g} yr. '
                    'The state was not advanced. A close-encounter integrator such as REBOUND '
                    'remains the candidate if a physical-radius run reaches this floor.')
                break
            previous = float(self.prepared.system.dt)
            self.prepared.system.dt = scheduled
            time_before = float(self.prepared.system.time)
            event_count = len(self.prepared.events)
            try:
                outcome = advance_prepared(self.prepared)
            except ValueError as exc:
                self.prepared.system.dt = previous
                self.status = 'halted'
                self.reason = str(exc)
                break
            previous = event_count
            for frame in outcome.event_frames:
                if frame.event_phase == 'pre':
                    self.samples.append(self._sample_from_frame(frame, 'pre'))
                elif frame.event_phase == 'post':
                    self._consume_outcomes(frame)
                    self.samples.append(self._sample_from_frame(frame, 'post'))
                else:
                    raise ValueError('Unexpected collision frame.')
            for event in outcome.prepared.events[previous:]:
                if isinstance(event, CollisionEvent) and event.kind in ('close_encounter', 'orbital_change'):
                    self._record_passive(event)
            self.prepared = outcome.prepared
            self._record_tides(event_count)
            actual = float(self.prepared.system.time)-time_before
            if outcome.stopped:
                self.status = 'stopped'
                stop = next((event for event in reversed(self.events) if event.kind == 'stop'), None)
                self.reason = stop.message if stop is not None else 'Stop on contact.'
            elif self.drag_policy.enabled and actual > 0 and self.status == 'running':
                self._apply_drag(actual)
            self._eject()
            self.accepted_steps.append(float(self.prepared.system.dt))
            self.completed_steps += 1
            done += 1
            if self.completed_steps % self.sample_stride == 0:
                self._record_current(None)
        return done

    def replay_ancestry(self, body_id):
        """Replay one body. A removed body keeps the samples from when it existed."""
        if not isinstance(body_id, str):
            raise ValueError('Body id must be a string.')
        if body_id in self.composition:
            status = 'active'
            contributions = _contributions(self.composition[body_id])
            parcels = {item.parcel_id for item in contributions}
            samples = tuple(sample for sample in self.samples if _carriers(sample, parcels))
        elif body_id in self.removed:
            record = self.removed[body_id]
            status = record.status
            contributions = record.contributions
            indices = [index for index, sample in enumerate(self.samples)
                       if any(body.id == body_id for body in sample.bodies)]
            if not indices:
                samples = ()
            else:
                chosen = set(indices)
                if indices[-1]+1 < len(self.samples):
                    chosen.add(indices[-1]+1)
                samples = tuple(self.samples[index] for index in sorted(chosen))
        else:
            raise ValueError(f'Unknown body {body_id}.')
        return Ancestry(body_id, status, contributions, self._related_events(body_id, contributions),
                        self.descendant_ids(body_id), samples)

    def descendant_ids(self, body_id):
        children = {}
        for event in self.events:
            if event.kind in ('merge', 'fragment'):
                for parent in event.parent_ids:
                    children[parent] = event.descendant_ids
        if body_id not in self.composition and body_id not in self.removed and body_id not in children:
            raise ValueError(f'Unknown body {body_id}.')
        leaves, stack, seen = [], [body_id], set()
        while stack:
            key = stack.pop()
            if key in seen:
                continue
            seen.add(key)
            if key in children:
                stack.extend(children[key])
            else:
                leaves.append(key)
        return tuple(sorted(leaves))

    def accounting(self):
        """Mass, parcel, momentum, and merge-energy discrepancies for this history."""
        initial = self.samples[0]
        initial_mass = sum(body.mass for body in initial.bodies)
        leaves = self._leaf_records()
        closed_mass = sum(body.mass for body in leaves)+sum(item.mass for event in self.events for item in event.lost_contributions)
        parcel_initial, parcel_now = {}, {}
        for body in initial.bodies:
            for item in body.contributions:
                parcel_initial[item.parcel_id] = parcel_initial.get(item.parcel_id, 0.)+item.mass
        for body in leaves:
            for item in body.contributions:
                parcel_now[item.parcel_id] = parcel_now.get(item.parcel_id, 0.)+item.mass
        for event in self.events:
            for item in event.lost_contributions:
                parcel_now[item.parcel_id] = parcel_now.get(item.parcel_id, 0.)+item.mass
        keys = set(parcel_initial) | set(parcel_now)
        parcel_error = max((abs(parcel_now.get(key, 0.)-parcel_initial.get(key, 0.)) for key in keys), default=0.)
        merge_momentum = merge_kinetic = ejection_momentum = 0.
        merge_count = ejection_count = index = 0
        dynamic = ('merge', 'fragment', 'graze', 'unmodeled')
        for sample_index, sample in enumerate(self.samples[:-1]):
            nxt = self.samples[sample_index+1]
            if sample.phase != 'pre' or nxt.phase != 'post':
                continue
            while index < len(self.events) and self.events[index].kind not in dynamic:
                index += 1
            batch = []
            tolerance = 1e-12*max(1., abs(sample.time))
            while (index < len(self.events) and self.events[index].kind in dynamic
                   and abs(self.events[index].time-sample.time) <= tolerance):
                batch.append(self.events[index])
                index += 1
            merge_count += sum(event.kind == 'merge' for event in batch)
            lost_momentum = np.zeros(len(sample.bodies[0].v))
            carried_kinetic = 0.
            for event in batch:
                lost = sum(item.mass for item in event.lost_contributions)
                if lost <= 0:
                    continue
                parents = [body for body in sample.bodies if body.id in event.parent_ids]
                total = sum(body.mass for body in parents)
                velocity = sum(body.mass*np.array(body.v) for body in parents)/total
                lost_momentum = lost_momentum+lost*velocity
                carried_kinetic += .5*lost*float(np.dot(velocity, velocity))
            merge_momentum = max(merge_momentum, _max_abs(_momentum(nxt.bodies)+lost_momentum-_momentum(sample.bodies)))
            measured = _kinetic(nxt.bodies)-_kinetic(sample.bodies)
            expected = -sum(event.lost_relative_kinetic for event in batch)-carried_kinetic
            merge_kinetic = max(merge_kinetic, abs(measured-expected))
        for sample_index, sample in enumerate(self.samples[:-1]):
            if sample.phase != 'ejection':
                continue
            nxt = self.samples[sample_index+1]
            gone = [body for body in sample.bodies if all(other.id != body.id for other in nxt.bodies)]
            ejection_count += len(gone)
            residual = _momentum(sample.bodies)-_momentum(nxt.bodies)-_momentum(gone)
            ejection_momentum = max(ejection_momentum, _max_abs(residual))
        return Accounting(initial_mass, closed_mass, parcel_error, merge_momentum, merge_kinetic,
                          ejection_momentum, merge_count, ejection_count)

    def branch(self, change):
        """Copy this state and apply one velocity change. The receiver is left as it is."""
        if not isinstance(change, VelocityChange):
            raise TypeError('A branch takes one VelocityChange.')
        parent_id = self.checkpoint_id
        document = self._payload()
        document['id'] = parent_id
        child = _experiment_from_document(document)
        child._diverge(change, parent_id)
        return child

    @property
    def checkpoint_id(self):
        canonical = json.dumps(self._payload(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def _diverge(self, change, parent_id):
        if change.body_id not in self.prepared.body_ids:
            raise ValueError(f'{change.body_id} is not an active body in this checkpoint.')
        index = self.prepared.body_ids.index(change.body_id)
        if self.prepared.body_specs[index].kind == 'star':
            raise ValueError('The disclosed change must name an embryo, not the star.')
        system = self.prepared.system
        if len(change.delta_v) != system.v.shape[1]:
            raise ValueError(
                f'delta_v has {len(change.delta_v)} components; this run stores {system.v.shape[1]}.')
        velocity = system.v.copy()
        velocity[index] = velocity[index]+np.asarray(change.delta_v, dtype=float)
        updated = system.with_state(system.r.copy(), velocity, system.time, masses=system.masses.copy())
        self.prepared = replace(self.prepared, system=updated)
        stored = {'kind': 'velocity', 'body_id': change.body_id, 'delta_v': change.delta_v, 'note': change.note}
        self.provenance = {**self.provenance, 'parent_checkpoint_id': parent_id, 'change': stored,
                           'run_id': _run_id(self.provenance['setup_digest'], self.provenance['seed'], parent_id, stored)}
        self.status = 'running'
        self.reason = ''
        self._record_current('branch', force=True)

    def _consume_outcomes(self, frame):
        dynamic = [event for event in frame.events if isinstance(event, CollisionEvent)
                   and event.kind in ('merge', 'fragment', 'graze', 'unmodeled', 'ignored', 'stop')]
        for event in dynamic[self._applied_outcomes:]:
            if event.kind == 'merge':
                self._apply_merge(event)
            elif event.kind == 'fragment':
                self._apply_fragment(event, frame)
            else:
                self._apply_continuation(event)
        self._applied_outcomes = len(dynamic)

    def _apply_merge(self, event):
        pre = self.samples[-1]
        if pre.phase != 'pre':
            raise ValueError('A merge needs the pre-contact sample before the remnant is recorded.')
        composition = {}
        for parent in event.parent_ids:
            for parcel, mass in self.composition[parent].items():
                composition[parcel] = composition.get(parcel, 0.)+mass
            source = next(body for body in pre.bodies if body.id == parent)
            self.removed[parent] = replace(source, status='merged')
            del self.composition[parent]
        if event.remnant_id in self.composition or event.remnant_id in self.removed:
            raise ValueError('Remnant id collides with an existing formation identity.')
        self.composition[event.remnant_id] = composition
        parents = [next(body for body in pre.bodies if body.id == parent) for parent in event.parent_ids]
        total = sum(body.mass for body in parents)
        width = len(parents[0].r)
        location = tuple(sum(body.mass*body.r[axis] for body in parents)/total for axis in range(width))
        self.events.append(FormationEvent(float(event.time), 'merge', tuple(event.parent_ids), event.remnant_id,
                                          _contributions(composition), (event.remnant_id,), float(event.lost_relative_kinetic),
                                          float(event.unmodeled_spin), location, event.message))

    def _apply_fragment(self, event, frame):
        pre = self.samples[-1]
        if pre.phase != 'pre':
            raise ValueError('A fragment needs the pre-contact sample before the descendants are recorded.')
        composition = {}
        for parent in event.parent_ids:
            for parcel, mass in self.composition[parent].items():
                composition[parcel] = composition.get(parcel, 0.)+mass
            source = next(body for body in pre.bodies if body.id == parent)
            self.removed[parent] = replace(source, status='fragmented')
            del self.composition[parent]
        shares = [(descendant, float(frame.masses[frame.body_ids.index(descendant)])) for descendant in event.descendant_ids]
        if event.lost_mass > 0:
            shares.append(('__lost__', float(event.lost_mass)))
        assigned = _split_composition(composition, shares)
        lost = assigned.pop('__lost__', {})
        for descendant in event.descendant_ids:
            if descendant in self.composition or descendant in self.removed:
                raise ValueError('Fragment id collides with an existing formation identity.')
            self.composition[descendant] = assigned[descendant]
        location = event.location or _weighted_location(pre.bodies, event.parent_ids)
        self.events.append(FormationEvent(float(event.time), 'fragment', tuple(event.parent_ids), event.remnant_id,
                                          _contributions(composition), tuple(event.descendant_ids),
                                          float(event.lost_relative_kinetic), float(event.unmodeled_spin), location,
                                          event.message, _contributions(lost)))

    def _apply_continuation(self, event):
        pre = self.samples[-1]
        if pre.phase != 'pre':
            raise ValueError('A continuation needs the pre-contact sample.')
        contributions = []
        for parent in event.parent_ids:
            contributions.extend(_contributions(self.composition[parent]))
        location = event.location if event.location is not None else _weighted_location(pre.bodies, event.parent_ids)
        descendants = tuple(event.descendant_ids) or tuple(event.parent_ids)
        self.events.append(FormationEvent(float(event.time), event.kind, tuple(event.parent_ids), None,
                                          tuple(contributions), descendants, float(event.lost_relative_kinetic),
                                          float(event.unmodeled_spin), location, event.message))

    def _record_tides(self, previous_count):
        from .tides import TideEvent
        for event in self.prepared.events[previous_count:]:
            if not isinstance(event, TideEvent):
                continue
            location = self._tide_location(event.body_ids)
            self.events.append(FormationEvent(
                float(event.time), 'tide', tuple(event.body_ids), None, (), tuple(event.body_ids),
                float(event.energy_lost), float(event.unmodeled_angular), location, event.message))

    def _tide_location(self, body_ids):
        system = self.prepared.system
        found = [system.r[self.prepared.body_ids.index(body_id)]
                 for body_id in body_ids if body_id in self.prepared.body_ids]
        if not found:
            return tuple(0. for _ in range(self.dimensions))
        midpoint = sum(np.asarray(item, dtype=float) for item in found)/len(found)
        return tuple(float(component) for component in midpoint)

    def _apply_drag(self, dt):
        """One annular fluid subcycle, then quadratic drag, for this macro step."""
        from .gas_dust import advance_annular_fluid, apply_body_drag
        if not advance_annular_fluid(self.zones, dt, self.drag_star_mass, self.drag_parameters.sound_speed):
            self.fluid_held = True
        star = self._star_index()
        system = self.prepared.system
        star_r = system.r[star]
        star_v = system.v[star]
        velocities = np.array(system.v, copy=True)
        for index, body_id in enumerate(self.prepared.body_ids):
            if index == star:
                continue
            spec = self.prepared.body_specs[index]
            updated, dissipated, residual = apply_body_drag(
                self.zones, self.drag_parameters, self.drag_star_mass, star_r, star_v,
                system.r[index], velocities[index], spec.radius or 0., float(system.masses[index]), dt)
            velocities[index] = updated
            self.dissipated_kinetic += dissipated
            self.drag_momentum_error = max(self.drag_momentum_error, residual)
        system.v[:] = velocities

    def _record_passive(self, event):
        contributions = []
        for parent in event.parent_ids:
            if parent in self.composition:
                contributions.extend(_contributions(self.composition[parent]))
            elif parent in self.removed:
                contributions.extend(self.removed[parent].contributions)
        location = event.location if event.location is not None else tuple(0. for _ in range(self.dimensions))
        self.events.append(FormationEvent(float(event.time), event.kind, tuple(event.parent_ids), None,
                                          tuple(contributions), tuple(event.descendant_ids) or tuple(event.parent_ids),
                                          0., 0., location, event.message))

    def _eject(self):
        system = self.prepared.system
        star = self._star_index()
        leaving = []
        star_r, star_v = system.r[star], system.v[star]
        star_mass = float(system.masses[star])
        for index, body_id in enumerate(self.prepared.body_ids):
            if index == star:
                continue
            relative_r = system.r[index]-star_r
            relative_v = system.v[index]-star_v
            distance = float(np.linalg.norm(relative_r))
            if distance == 0:
                continue
            energy = .5*float(np.dot(relative_v, relative_v))-G*star_mass/distance
            if energy > 0 and distance >= self.ejection_radius:
                leaving.append(body_id)
        if not leaving:
            return
        self._record_current('ejection', force=True)
        ejection_sample = self.samples[-1]
        for body_id in sorted(leaving):
            source = next(body for body in ejection_sample.bodies if body.id == body_id)
            self.removed[body_id] = replace(source, status='ejected')
            del self.composition[body_id]
            self.events.append(FormationEvent(
                float(system.time), 'ejection', (body_id,), None, source.contributions, (body_id,), 0., 0., source.r,
                f'{body_id} crossed the ejection radius with positive star-centric energy and left the active set.'))
        self._drop(set(leaving))
        self._record_current(None, force=True)

    def _drop(self, ids):
        keep = [index for index, body_id in enumerate(self.prepared.body_ids) if body_id not in ids]
        system = self.prepared.system
        updated = system.with_state(np.array(system.r[keep], copy=True), np.array(system.v[keep], copy=True),
                                    system.time, masses=np.array(system.masses[keep], copy=True))
        reserves = tuple(CollisionEvent(system.time, 'reserve', (body_id,), None, (body_id,), 0., 0.,
                                        f'Reserved ejected id {body_id}.') for body_id in sorted(ids))
        self.prepared = PreparedSystem(
            updated, tuple(self.prepared.body_ids[index] for index in keep),
            tuple(self.prepared.body_specs[index] for index in keep), self.prepared.setup,
            self.prepared.events+reserves, self.prepared.segment,
            tuple(self.prepared.reference_ids[index] for index in keep))

    def _star_index(self):
        matches = [index for index, spec in enumerate(self.prepared.body_specs) if spec.kind == 'star']
        if len(matches) != 1:
            raise ValueError('Formation ejection bookkeeping needs exactly one active star.')
        return matches[0]

    def _record_current(self, phase, force=False):
        sample = self._sample_from_frame(StateFrame.capture(self.prepared), phase)
        if not force and self.samples and _same_state(self.samples[-1], sample):
            return
        self.samples.append(sample)

    def _sample_from_frame(self, frame, phase):
        bodies = []
        for index, body_id in enumerate(frame.body_ids):
            spec = frame.body_specs[index]
            bodies.append(SampleBody(
                body_id, spec.name, spec.kind, _num(frame.masses[index]), _num(spec.radius),
                tuple(_num(component) for component in frame.r[index]),
                tuple(_num(component) for component in frame.v[index]),
                _contributions(self.composition[body_id]), 'active', frame.reference_ids[index]))
        return Sample(_num(frame.time), phase, tuple(bodies))

    def _active_records(self):
        return self._sample_from_frame(StateFrame.capture(self.prepared), None).bodies

    def _leaf_records(self):
        return list(self._active_records())+[body for body in self.removed.values() if body.status == 'ejected']

    def _related_events(self, body_id, contributions):
        parcels = {item.parcel_id for item in contributions} | {body_id}
        related = []
        for event in self.events:
            named = set(event.parent_ids) | set(event.descendant_ids)
            if event.remnant_id is not None:
                named.add(event.remnant_id)
            moved = {item.parcel_id for item in event.contributions}
            if named & parcels or moved & parcels:
                related.append(event)
        return tuple(related)

    def _removed_records(self):
        return tuple(self.removed[key] for key in self.removed)

    def _payload(self):
        payload = {
            'schema': CHECKPOINT_SCHEMA, 'model': FORMATION_MODEL, 'name': self.name, 'time': _num(self.time),
            'dt': _num(self.prepared.system.dt), 'ejection_radius': _num(self.ejection_radius),
            'sample_stride': self.sample_stride, 'status': self.status, 'reason': self.reason,
            'completed_steps': self.completed_steps, 'assumptions': list(self.assumptions),
            'provenance': _provenance_document(self.provenance),
            'active': [_body_document(body) for body in self._active_records()],
            'removed': [_body_document(body) for body in self._removed_records()],
            'events': [_event_document(event) for event in self.events],
            'samples': [_sample_document(sample) for sample in self.samples],
        }
        payload.update(self._numerics_fields())
        payload.update(self._physics_fields())
        return payload

    def _numerics_fields(self):
        if not self.adaptive and self.dimensions == 2:
            return {}
        fields = {'adaptive': bool(self.adaptive), 'dimensions': int(self.dimensions), 'dt_max': _num(self.dt_max)}
        if self.adaptive:
            fields['dt_min'] = _num(self.dt_min)
        return fields

    def _physics_fields(self):
        """Tide and drag state. Absent when that switch is off, so older checkpoints stay valid."""
        fields = {}
        tides = self.prepared.setup.tides if self.prepared.setup is not None else None
        if tides is not None and tides.enabled:
            fields['tides'] = {'enabled': True, 'version': tides.version,
                               'strength': _num(tides.strength), 'reach': _num(tides.reach)}
        if self.drag_policy.enabled:
            fields['drag'] = {
                'enabled': True, 'version': self.drag_policy.version, 'disk': self.drag_policy.disk,
                'star_mass': _num(self.drag_star_mass), 'sound_speed': _num(self.drag_parameters.sound_speed),
                'drag_coefficient': _num(self.drag_parameters.drag_coefficient),
                'dissipated_kinetic': _num(self.dissipated_kinetic),
                'drag_momentum_error': _num(self.drag_momentum_error), 'fluid_held': bool(self.fluid_held),
                'zones': [{'id': zone.id, 'r_inner': _num(zone.r_inner), 'r_outer': _num(zone.r_outer),
                           'mass': _num(zone.mass), 'radial_momentum': _num(zone.radial_momentum),
                           'angular_momentum': _num(zone.angular_momentum)} for zone in self.zones],
            }
        return fields

    def _scheduled_dt(self):
        """Authored step, or a smaller one while a pair is unresolved."""
        if not self.adaptive:
            return float(self.prepared.system.dt)
        limit = float(self.prepared.system.resolution_limit())
        if not math.isfinite(limit) or limit >= self.dt_max:
            target = self.dt_max
        elif limit < self.dt_min:
            return None
        else:
            target = limit
        previous = float(self.prepared.system.dt)
        if target > previous*ADAPTIVE_GROWTH:
            target = previous*ADAPTIVE_GROWTH
        if target > self.dt_max:
            target = self.dt_max
        return float(target)


def _carriers(sample, parcels):
    return any(parcels.intersection(item.parcel_id for item in body.contributions) for body in sample.bodies)


def _same_state(left, right):
    return (left.time == right.time and tuple((body.id, body.r, body.v) for body in left.bodies)
            == tuple((body.id, body.r, body.v) for body in right.bodies))


def _body_document(body):
    return {'id': body.id, 'name': body.name, 'kind': body.kind, 'mass': _num(body.mass), 'radius': _num(body.radius),
            'status': body.status, 'r': [_num(component) for component in body.r],
            'v': [_num(component) for component in body.v],
            'contributions': [{'parcel_id': item.parcel_id, 'mass': _num(item.mass)}
                              for item in sorted(body.contributions, key=lambda item: item.parcel_id)],
            'reference_id': body.reference_id}


def _event_document(event):
    return {'time': _num(event.time), 'kind': event.kind, 'parent_ids': list(event.parent_ids),
            'remnant_id': event.remnant_id, 'descendant_ids': list(event.descendant_ids),
            'contributions': [{'parcel_id': item.parcel_id, 'mass': _num(item.mass)}
                              for item in sorted(event.contributions, key=lambda item: item.parcel_id)],
            'lost_relative_kinetic': _num(event.lost_relative_kinetic), 'unmodeled_spin': _num(event.unmodeled_spin),
            'location': [_num(component) for component in event.location], 'message': event.message,
            'lost_contributions': [{'parcel_id': item.parcel_id, 'mass': _num(item.mass)}
                                   for item in sorted(event.lost_contributions, key=lambda item: item.parcel_id)]}


def _sample_document(sample):
    return {'time': _num(sample.time), 'phase': sample.phase, 'bodies': [_body_document(body) for body in sample.bodies]}


def _provenance_document(provenance):
    change = provenance['change']
    return {'run_id': provenance['run_id'], 'model': provenance['model'], 'representation': provenance['representation'],
            'collision_model': provenance['collision_model'], 'seed': provenance['seed'],
            'setup_digest': provenance['setup_digest'], 'parent_checkpoint_id': provenance['parent_checkpoint_id'],
            'change': None if change is None else _change_document(change),
            'execution_environment': {'numpy': provenance['execution_environment']['numpy']},
            'solver_build': provenance['solver_build']}


def save_checkpoint(experiment, path):
    """Validate the checkpoint, then atomically replace `path`."""
    if not isinstance(experiment, FormationExperiment):
        raise TypeError('Save a formation experiment.')
    payload = experiment._payload()
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    document = dict(payload)
    document['id'] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    encoded = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)
    _experiment_from_document(json.loads(encoded))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, prefix='.'+path.name+'.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return document['id']


def load_checkpoint(path):
    """Load a formation checkpoint into an experiment that can integrate forward."""
    try:
        with Path(path).open(encoding='utf-8') as handle:
            document = json.load(handle, object_pairs_hook=_pairs,
                                 parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON number: '+value)))
    except (json.JSONDecodeError, TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError(f'Invalid formation checkpoint: {exc}') from exc
    return _experiment_from_document(document)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON field: {key}.')
        result[key] = value
    return result


def _object(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f'{label} must contain exactly {", ".join(keys)}.')
    return value


def _string(value, label, *, nullable=False, hex_length=None):
    if value is None and nullable:
        return None
    if type(value) is not str or not value.strip():
        raise ValueError(f'{label} must be a nonempty string.')
    if hex_length is not None and (len(value) != hex_length or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError(f'{label} must be {hex_length} lowercase hex characters.')
    return value


def _tides_from_document(value):
    if value is None:
        return TidalPolicy()
    if type(value) is not dict or set(value) != {'enabled', 'version', 'strength', 'reach'}:
        raise ValueError('Checkpoint tides must contain enabled, version, strength, and reach.')
    if value['enabled'] is not True:
        raise ValueError('A stored tide block is the enabled model.')
    return TidalPolicy(enabled=True, version=value['version'], strength=value['strength'], reach=value['reach'])


def _drag_from_document(value):
    if value is None:
        return DragPolicy()
    if type(value) is not dict or value.get('enabled') is not True:
        raise ValueError('A stored drag block is the enabled model.')
    return DragPolicy(enabled=True, version=value['version'], disk=value['disk'])


def _restore_drag(experiment, value):
    from .gas_dust import DiskParameters, Zone, default_parameters
    zones = []
    for zone in value['zones']:
        zones.append(Zone(zone['id'], float(zone['r_inner']), float(zone['r_outer']), float(zone['mass']),
                          float(zone['radial_momentum']), float(zone['angular_momentum'])))
    parameters = default_parameters(sound_speed=float(value['sound_speed']), drag_enabled=True)
    if parameters.drag_coefficient != float(value['drag_coefficient']):
        parameters = DiskParameters(
            parameters.sound_speed, parameters.grain_density, parameters.bulk_density, parameters.stokes_min,
            parameters.grain_radius_min, parameters.mass_min, float(value['drag_coefficient']), True)
    experiment.zones = zones
    experiment.drag_parameters = parameters
    experiment.drag_star_mass = float(value['star_mass'])
    experiment.dissipated_kinetic = float(value['dissipated_kinetic'])
    experiment.drag_momentum_error = float(value['drag_momentum_error'])
    experiment.fluid_held = bool(value['fluid_held'])


def _experiment_from_document(document):
    if type(document) is not dict:
        raise ValueError('Formation checkpoint must be a JSON object.')
    if document.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError(f'Unsupported formation checkpoint schema; expected {CHECKPOINT_SCHEMA}.')
    if document.get('model') != FORMATION_MODEL:
        raise ValueError(f'Unsupported formation model; expected {FORMATION_MODEL}.')
    allowed = set(_DOCUMENT_KEYS) | set(_OPTIONAL_DOCUMENT_KEYS)
    if not set(_DOCUMENT_KEYS) <= set(document) <= allowed:
        raise ValueError('Formation checkpoint must contain exactly '+', '.join(_DOCUMENT_KEYS)+'.')
    payload = {key: document[key] for key in document if key != 'id'}
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    identity = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    if document['id'] != identity:
        raise ValueError('Checkpoint id does not match its contents.')
    name = _string(document['name'], 'Experiment name')
    time_value = _finite_number(document['time'], 'Time')
    dt = _positive('Fixed dt', document['dt'])
    ejection_radius = _positive('Ejection radius', document['ejection_radius'])
    if type(document['sample_stride']) is not int or isinstance(document['sample_stride'], bool) or document['sample_stride'] < 1:
        raise ValueError('sample_stride must be a positive integer.')
    if document['status'] not in ('running', 'complete', 'halted', 'stopped'):
        raise ValueError('Unsupported formation status.')
    reason = document['reason']
    if type(reason) is not str:
        raise ValueError('Formation reason must be a string.')
    if type(document['completed_steps']) is not int or isinstance(document['completed_steps'], bool) or document['completed_steps'] < 0:
        raise ValueError('completed_steps must be a nonnegative integer.')
    assumptions = _string_list(document['assumptions'], 'Assumptions')
    provenance = _provenance_from_document(document['provenance'])
    active = tuple(_body_from_document(body, 'Active body') for body in _list(document['active'], 'Active bodies'))
    removed_list = tuple(_body_from_document(body, 'Removed body') for body in _list(document['removed'], 'Removed bodies'))
    if not active or any(body.status != 'active' for body in active):
        raise ValueError('Active checkpoint bodies must be present and marked active.')
    if any(body.status not in ('merged', 'ejected', 'fragmented') for body in removed_list):
        raise ValueError('Removed bodies must be merged, fragmented, or ejected.')
    active_ids = [body.id for body in active]
    removed_ids = [body.id for body in removed_list]
    if len(set(active_ids)) != len(active_ids) or len(set(removed_ids)) != len(removed_ids) or set(active_ids) & set(removed_ids):
        raise ValueError('Formation identities must be unique across the active and removed sets.')
    events = tuple(_event_from_document(event) for event in _list(document['events'], 'Events'))
    samples = tuple(_sample_from_document(sample) for sample in _list(document['samples'], 'Samples'))
    if not samples or any(b.time < a.time for a, b in zip(samples, samples[1:])):
        raise ValueError('Checkpoint samples must be nonempty and nondecreasing in time.')
    if samples[-1].time > time_value+1e-9*max(1., abs(time_value)):
        raise ValueError('A sample is later than the checkpoint time.')
    shell_bodies = tuple(InitialBody(body.id, BodySpec(body.name, body.kind, body.mass, body.radius), body.r, body.v, body.reference_id)
                         for body in active)
    mode = _mode_from_collision_model(provenance['collision_model'])
    tides = _tides_from_document(document.get('tides'))
    drag = _drag_from_document(document.get('drag'))
    shell = InitialSetup(name, shell_bodies, dt, collision_mode=mode, tides=tides, drag=drag)
    collision_events = []
    for event in events:
        if event.kind in ('merge', 'fragment', 'graze', 'unmodeled', 'ignored', 'stop'):
            ancestry = tuple(sorted({item.parcel_id for item in event.contributions}))
            descendants = event.descendant_ids
            collision_events.append(CollisionEvent(
                event.time, event.kind, event.parent_ids, event.remnant_id, ancestry,
                event.lost_relative_kinetic, event.unmodeled_spin, event.message,
                rule=mode, outcome=event.kind, descendant_ids=descendants,
                lost_mass=sum(item.mass for item in event.lost_contributions)))
        elif event.kind == 'ejection':
            collision_events.extend(CollisionEvent(event.time, 'reserve', (parent,), None, (parent,), 0., 0.,
                                                   f'Reserved ejected id {parent}.') for parent in event.parent_ids)
    system = System(np.array([body.mass for body in active], dtype=float),
                    np.array([body.r for body in active], dtype=float),
                    np.array([body.v for body in active], dtype=float), dt, time_value)
    prepared = PreparedSystem(system, tuple(active_ids), tuple(body_spec(body) for body in active), shell,
                              tuple(collision_events), 0, tuple(body.reference_id for body in active))
    experiment = FormationExperiment.__new__(FormationExperiment)
    experiment.name = name
    experiment.ejection_radius = ejection_radius
    experiment.sample_stride = document['sample_stride']
    experiment.assumptions = tuple(assumptions)
    experiment.provenance = provenance
    experiment.prepared = prepared
    experiment.composition = {body.id: {item.parcel_id: item.mass for item in body.contributions} for body in active}
    experiment.removed = {body.id: body for body in removed_list}
    experiment.events = list(events)
    experiment.samples = list(samples)
    experiment.status = document['status']
    experiment.reason = reason
    experiment.completed_steps = document['completed_steps']
    experiment._applied_outcomes = sum(
        event.kind in ('merge', 'fragment', 'graze', 'unmodeled', 'ignored', 'stop') for event in collision_events)
    experiment.accepted_steps = []
    widths = {len(body.r) for body in (*active, *removed_list)}
    widths.update(len(body.v) for body in (*active, *removed_list))
    for sample in samples:
        widths.update(len(body.r) for body in sample.bodies)
        widths.update(len(body.v) for body in sample.bodies)
    widths.update(len(event.location) for event in events)
    if len(widths) != 1:
        raise ValueError('Checkpoint vectors must share one dimension.')
    dimension = widths.pop()
    if 'dimensions' in document:
        stated = document['dimensions']
        if type(stated) is not int or isinstance(stated, bool) or stated != dimension:
            raise ValueError('dimensions must match the stored vectors.')
    adaptive = document.get('adaptive', False)
    if type(adaptive) is not bool:
        raise ValueError('adaptive must be a boolean.')
    dt_max = _positive('Maximum dt', document['dt_max']) if 'dt_max' in document else dt
    if dt_max+1e-12*max(1., dt_max) < dt:
        raise ValueError('dt_max must be at least the stored step.')
    if 'dt_min' in document:
        dt_min = _positive('Adaptive floor', document['dt_min'])
    else:
        dt_min = ADAPTIVE_DT_MIN if adaptive else dt_max
    if adaptive and dt_min > dt_max:
        raise ValueError('The adaptive floor must be at most the maximum step.')
    if not adaptive and 'dt_min' in document:
        raise ValueError('dt_min applies only when adaptive control is on.')
    if provenance['change'] is not None and len(provenance['change']['delta_v']) != dimension:
        raise ValueError('delta_v must match the checkpoint dimension.')
    experiment.adaptive = adaptive
    experiment.dimensions = dimension
    experiment.dt_max = dt_max
    experiment.dt_min = dt_min
    experiment.drag_policy = drag
    experiment.zones = []
    experiment.drag_parameters = None
    experiment.drag_star_mass = None
    experiment.dissipated_kinetic = 0.
    experiment.drag_momentum_error = 0.
    experiment.fluid_held = False
    if drag.enabled:
        _restore_drag(experiment, document['drag'])
    if experiment.checkpoint_id != identity:
        raise ValueError('Checkpoint did not round-trip.')
    return experiment


def body_spec(body):
    return BodySpec(body.name, body.kind, body.mass, body.radius)


def _provenance_from_document(value):
    _object(value, _PROVENANCE_KEYS, 'Provenance')
    if (value['model'] != FORMATION_MODEL or value['representation'] != REPRESENTATION_ID
            or value['collision_model'] not in _COLLISION_MODEL_IDS):
        raise ValueError('Unsupported formation provenance model.')
    seed = value['seed']
    if seed is not None and (type(seed) is not int or isinstance(seed, bool) or seed < 0):
        raise ValueError('Formation seed must be a nonnegative integer or null.')
    environment = value['execution_environment']
    _object(environment, ('numpy',), 'Execution environment')
    change = value['change']
    if change is not None:
        _object(change, _CHANGE_KEYS, 'Velocity change')
        if change['kind'] != 'velocity':
            raise ValueError('The only disclosed branch change is velocity.')
        delta = change['delta_v']
        if type(delta) is not list or len(delta) not in (2, 3):
            raise ValueError('delta_v must contain two or three components.')
        change = {'kind': 'velocity', 'body_id': _string(change['body_id'], 'Changed body'),
                  'delta_v': tuple(_finite_number(item, 'delta_v') for item in delta),
                  'note': _string(change['note'], 'Change note')}
    return {'run_id': _string(value['run_id'], 'Run id', hex_length=16), 'model': value['model'],
            'representation': value['representation'], 'collision_model': value['collision_model'], 'seed': seed,
            'setup_digest': _string(value['setup_digest'], 'Setup digest', hex_length=64),
            'parent_checkpoint_id': _string(value['parent_checkpoint_id'], 'Parent checkpoint', nullable=True, hex_length=64),
            'change': change, 'execution_environment': {'numpy': _string(environment['numpy'], 'NumPy version')},
            'solver_build': _string(value['solver_build'], 'Solver build')}


def _body_from_document(value, label):
    _object(value, _BODY_KEYS, label)
    contributions = tuple(_contribution_from_document(item) for item in _list(value['contributions'], 'Contributions'))
    if not contributions or len({item.parcel_id for item in contributions}) != len(contributions):
        raise ValueError('Contributions need unique parcel ids.')
    body = SampleBody(_string(value['id'], 'Body id'), _string(value['name'], 'Body name'),
                      _string(value['kind'], 'Body kind'), _positive('Mass', value['mass']), _positive('Radius', value['radius']),
                      _vector(value['r'], 'Position'), _vector(value['v'], 'Velocity'), contributions, value['status'],
                      _string(value['reference_id'], 'Reference', nullable=True))
    if body.kind not in ('star', 'planet', 'moon', 'asteroid'):
        raise ValueError('Unsupported formation body kind.')
    if body.status not in ('active', 'merged', 'ejected', 'fragmented'):
        raise ValueError('Unsupported formation body status.')
    if abs(sum(item.mass for item in body.contributions)-body.mass) > 1e-9*max(1., abs(body.mass)):
        raise ValueError(f'Contributions for {body.id} do not add up to its mass.')
    return body


def _contribution_from_document(value):
    _object(value, ('parcel_id', 'mass'), 'Contribution')
    return Contribution(_string(value['parcel_id'], 'Parcel id'), _positive('Contribution mass', value['mass']))


def _event_from_document(value):
    _object(value, _EVENT_KEYS, 'Formation event')
    kind = value['kind']
    if kind not in ('merge', 'fragment', 'graze', 'unmodeled', 'close_encounter', 'orbital_change', 'ejection',
                    'ignored', 'stop', 'tide'):
        raise ValueError('Unsupported formation event.')
    parents = tuple(_string(item, 'Parent id') for item in _list(value['parent_ids'], 'Parent ids'))
    if not parents:
        raise ValueError('A formation event needs a parent id.')
    remnant = _string(value['remnant_id'], 'Remnant id', nullable=True)
    descendants = tuple(_string(item, 'Descendant id') for item in _list(value['descendant_ids'], 'Descendant ids'))
    if kind == 'merge' and (remnant is None or descendants != (remnant,)):
        raise ValueError('A merge event needs its remnant as the only descendant.')
    if kind == 'fragment' and (remnant is None or remnant not in descendants or len(descendants) < 2):
        raise ValueError('A fragment event needs its largest remnant and at least one other descendant.')
    if kind == 'ejection' and (remnant is not None or descendants != parents):
        raise ValueError('An ejection keeps the same body as its descendant.')
    if kind in ('graze', 'unmodeled', 'close_encounter', 'orbital_change', 'ignored', 'stop', 'tide') and remnant is not None:
        raise ValueError('This event keeps the participants and does not name a remnant.')
    contributions = tuple(_contribution_from_document(item) for item in _list(value['contributions'], 'Contributions'))
    lost = tuple(_contribution_from_document(item) for item in _list(value['lost_contributions'], 'Lost contributions'))
    if kind != 'fragment' and lost:
        raise ValueError('Only a fragment event records lost mass.')
    return FormationEvent(_finite_number(value['time'], 'Event time'), kind, parents, remnant, contributions, descendants,
                          _finite_number(value['lost_relative_kinetic'], 'Lost kinetic energy'),
                          _finite_number(value['unmodeled_spin'], 'Unmodeled spin'),
                          _vector(value['location'], 'Event location'), _string(value['message'], 'Event message'), lost)


def _sample_from_document(value):
    _object(value, _SAMPLE_KEYS, 'Sample')
    if value['phase'] not in (None, 'pre', 'post', 'ejection', 'branch'):
        raise ValueError('Unsupported sample phase.')
    bodies = tuple(_body_from_document(body, 'Sample body') for body in _list(value['bodies'], 'Sample bodies'))
    if any(body.status != 'active' for body in bodies):
        raise ValueError('Recorded samples contain active bodies.')
    return Sample(_finite_number(value['time'], 'Sample time'), value['phase'], bodies)


def _list(value, label):
    if type(value) is not list:
        raise ValueError(f'{label} must be a list.')
    return value


def _string_list(value, label):
    items = _list(value, label)
    return [_string(item, label) for item in items]


def _vector(value, label):
    if type(value) is not list or len(value) not in (2, 3):
        raise ValueError(f'{label} must contain two or three components.')
    return tuple(_finite_number(item, label) for item in value)


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be finite.')
    return float(value)


def playback(experiment):
    """Scrub one finished or partial run. Replay does not call the integrator."""
    if not isinstance(experiment, FormationExperiment):
        raise TypeError('Playback needs a formation experiment.')
    return Playback(as_recorded_run(experiment))


def as_recorded_run(experiment):
    frames = tuple(_state_frame(sample, experiment.prepared.system.dt) for sample in experiment.samples)
    if experiment.status == 'complete':
        status, reason = 'complete', experiment.reason
    elif experiment.status in ('halted', 'stopped'):
        status, reason = 'halted', experiment.reason or 'Formation run halted.'
    else:
        status, reason = 'cancelled', 'Partial formation recording; the run is not finished.'
    end = frames[-1].time
    return RecordedRun(experiment.provenance['setup_digest'], experiment.name, end, end, frames, status, reason,
                       experiment.completed_steps, experiment.completed_steps, experiment.sample_stride, None,
                       experiment.provenance['execution_environment'], experiment.provenance['solver_build'])


def _state_frame(sample, dt):
    phase = sample.phase if sample.phase in ('pre', 'post') else None
    return StateFrame(sample.time, dt, tuple(body.id for body in sample.bodies),
                      tuple(body_spec(body) for body in sample.bodies),
                      np.array([body.mass for body in sample.bodies], dtype=float),
                      np.array([body.r for body in sample.bodies], dtype=float),
                      np.array([body.v for body in sample.bodies], dtype=float), (),
                      tuple(body.reference_id for body in sample.bodies), phase)


class SynchronizedPlayback:
    """Side-by-side samples over the simulated times both runs have recorded."""

    def __init__(self, first, second):
        self.left_playback = playback(first)
        self.right_playback = playback(second)
        self.left_times = tuple(frame.time for frame in self.left_playback.recording.frames)
        self.right_times = tuple(frame.time for frame in self.right_playback.recording.frames)
        self.common_start = max(self.left_times[0], self.right_times[0])
        self.common_end = min(self.left_times[-1], self.right_times[-1])
        if self.common_end < self.common_start:
            raise ValueError('The runs have no common simulated-time interval.')
        self.time = self.common_start
        self.playing = False
        self._place()

    def play(self):
        self.playing = self.time < self.common_end

    def pause(self):
        self.playing = False

    def seek(self, time_value):
        if isinstance(time_value, bool) or not isinstance(time_value, (int, float)) or not math.isfinite(time_value):
            raise ValueError('Playback time must be finite.')
        tolerance = 1e-12*max(1., abs(self.common_end))
        if time_value < self.common_start-tolerance or time_value > self.common_end+tolerance:
            raise ValueError('Time is outside the runs\' common simulated interval '
                             f'[{self.common_start:.6g}, {self.common_end:.6g}] yr.')
        self.time = min(max(float(time_value), self.common_start), self.common_end)
        self._place()
        return self.view()

    def advance(self, wall_seconds, years_per_second):
        if (isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float)) or not math.isfinite(wall_seconds)
                or wall_seconds < 0 or isinstance(years_per_second, bool) or not isinstance(years_per_second, (int, float))
                or not math.isfinite(years_per_second) or years_per_second <= 0):
            raise ValueError('Elapsed wall seconds must be nonnegative; playback yr/sec must be positive and finite.')
        if self.playing:
            self.time = min(self.common_end, self.time+wall_seconds*years_per_second)
            self._place()
            if self.time >= self.common_end:
                self.pause()
        return self.view()

    def view(self):
        return SyncedFrame(self.time, self.left_playback.current_frame, self.right_playback.current_frame,
                           self.common_start, self.common_end)

    def _place(self):
        self.left_playback.index = _index_at(self.left_times, self.time)
        self.right_playback.index = _index_at(self.right_times, self.time)


def _index_at(times, time_value):
    index = bisect_right(times, time_value)-1
    if index < 0:
        raise ValueError('No recorded sample at or before this time.')
    return index


def compare_resolution(dts=(.02, .005)):
    """Integrate the same head-on merger at each fixed step and report the first contact."""
    steps = tuple(dts)
    if len(steps) < 2:
        raise ValueError('Compare at least two timesteps.')
    points = []
    for dt in steps:
        experiment = FormationExperiment(make_merger_probe(float(dt)), ejection_radius=100., sample_stride=1)
        experiment.integrate(.25, finish=True)
        merge = next((event for event in experiment.events if event.kind == 'merge'), None)
        points.append(ResolutionPoint(
            float(dt), experiment.status, None if merge is None else merge.time,
            None if merge is None else sum(item.mass for item in merge.contributions),
            None if merge is None else merge.parent_ids,
            sum(body.kind != 'star' for body in experiment._active_records())))
    return tuple(points)


def hardware_description():
    """One line identifying the machine that produced a benchmark."""
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding='utf-8', errors='replace').splitlines():
            if line.lower().startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    cpus = os.cpu_count()
    cpu_count = 'unknown' if cpus is None else str(cpus)
    return (f'{platform.platform()}; {cpu}; {cpu_count} logical CPUs; '
            f'Python {platform.python_version()}; NumPy {np.__version__}')


def run_benchmark():
    """Time the seeded late-stage setup and measure its recording size.

    Peak memory is Python's tracemalloc peak for the setup and integration,
    not the operating-system resident set. Recording size is the checkpoint
    document in bytes.
    """
    tracing = not tracemalloc.is_tracing()
    if tracing:
        tracemalloc.start()
    clock = time.perf_counter()
    try:
        setup = make_late_stage_setup()
        experiment = FormationExperiment(setup, ejection_radius=BENCHMARK_EJECTION_RADIUS,
                                         sample_stride=BENCHMARK_SAMPLE_STRIDE, seed=BENCHMARK_SEED)
        experiment.integrate(BENCHMARK_DURATION_YEARS, finish=True)
        seconds = time.perf_counter()-clock
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if tracing:
            tracemalloc.stop()
    payload = experiment._payload()
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    document = dict(payload)
    document['id'] = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    recording = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n'
    merges = tuple(event.time for event in experiment.events if event.kind == 'merge')
    ejections = tuple(event.time for event in experiment.events if event.kind == 'ejection')
    notes = [
        'Embryo radii are inflated encounter radii so a short window can contain contacts.',
        'One disclosed unbound embryo starts outside the annulus so ejection bookkeeping has a known case.',
        'Annulus mergers are produced by the seeded ellipses. None are inserted by hand.',
    ]
    if experiment.status != 'complete':
        notes.append('The run stopped before the requested time: '+experiment.reason)
    elif len(merges) == 1:
        notes.append('1 perfect merge was recorded.')
    else:
        notes.append(f'{len(merges)} perfect merges were recorded.')
    return BenchmarkReport(
        hardware_description(), FORMATION_MODEL, BENCHMARK_SEED, len(experiment.samples[0].bodies), BENCHMARK_EMBRYO_COUNT,
        BENCHMARK_DURATION_YEARS, experiment.time, seconds, peak, len(recording.encode('utf-8')), len(experiment.samples),
        len(merges), len(ejections), merges, ejections, experiment.status, experiment.reason,
        tuple(body.id for body in experiment._active_records() if body.kind != 'star'), tuple(notes))


def run_physical_probe():
    """Time a short rocky, mildly inclined annulus with the adaptive macro step.

    This is not the inflated 80-embryo benchmark. It reports whether a physical
    radius and a shrinking step stay practical on a small seeded run. Tree
    gravity is not used. Perfect merge is unchanged, so a contact still has
    one descendant.
    """
    radius = rocky_radius(BENCHMARK_EMBRYO_MASS)
    clock = time.perf_counter()
    setup = make_late_stage_setup(
        count=PHYSICAL_PROBE_COUNT, seed=PHYSICAL_PROBE_SEED, include_unbound=False,
        radii='rocky', inclination=PHYSICAL_PROBE_INCLINATION, dt=BENCHMARK_DT)
    experiment = FormationExperiment(
        setup, ejection_radius=BENCHMARK_EJECTION_RADIUS, sample_stride=BENCHMARK_SAMPLE_STRIDE,
        seed=PHYSICAL_PROBE_SEED, adaptive=True)
    experiment.integrate(PHYSICAL_PROBE_DURATION_YEARS, finish=True)
    seconds = time.perf_counter()-clock
    steps = tuple(experiment.accepted_steps) or (experiment.dt_max,)
    ratio = BENCHMARK_EMBRYO_RADIUS/radius
    notes = [
        f'Rocky radius {radius:.6g} AU is {ratio:.0f} times smaller than the {BENCHMARK_EMBRYO_RADIUS} AU encounter radius at the benchmark mass.',
        'Geometric cross section is smaller by about the square of that factor, so contacts are much rarer than in the inflated prototype.',
        'The macro step is the pair resolution limit during a close pass and returns toward the authored 0.025 yr step afterward.',
        'Direct summation is used. Tree gravity remains deferred and would still have to keep parcel identity.',
        'This probe leaves gas drag and tides off. Those switches can be enabled on the same adaptive path. A longer-lived capture is claimed only for a run that turns the relevant switch on.',
    ]
    if experiment.status != 'complete':
        notes.append('The run stopped before the requested time: '+experiment.reason)
    else:
        notes.append(f'{sum(event.kind == "merge" for event in experiment.events)} perfect merges were recorded.')
    return PhysicalProbeReport(
        hardware_description(), len(experiment.samples[0].bodies), PHYSICAL_PROBE_COUNT,
        PHYSICAL_PROBE_DURATION_YEARS, experiment.time, seconds, experiment.status, experiment.reason,
        sum(event.kind == 'merge' for event in experiment.events), min(steps), max(steps), radius,
        BENCHMARK_EMBRYO_RADIUS, tuple(notes))


def run_rocky_year():
    """80 embryos, rocky radii, 3D state, adaptive step, one year.

    The measured combination is perfect merge, the annular disk, and tides.
    Each switch can be turned off by constructing the setup without it.
    """
    setup = with_physics(
        make_late_stage_setup(
            count=ROCKY_YEAR_COUNT, seed=ROCKY_YEAR_SEED, include_unbound=False,
            radii='rocky', inclination=ROCKY_YEAR_INCLINATION, dt=BENCHMARK_DT),
        drag=DragPolicy(enabled=True, disk='annular'), tides=TidalPolicy(enabled=True))
    clock = time.perf_counter()
    experiment = FormationExperiment(
        setup, ejection_radius=BENCHMARK_EJECTION_RADIUS, sample_stride=BENCHMARK_SAMPLE_STRIDE,
        seed=ROCKY_YEAR_SEED, adaptive=True)
    experiment.integrate(ROCKY_YEAR_DURATION_YEARS, finish=True)
    seconds = time.perf_counter()-clock
    steps = tuple(experiment.accepted_steps) or (experiment.dt_max,)
    combo = experiment.combination()
    merges = sum(event.kind == 'merge' for event in experiment.events)
    tides = sum(event.kind == 'tide' for event in experiment.events)
    ejections = sum(event.kind == 'ejection' for event in experiment.events)
    unmodeled = sum(event.kind == 'unmodeled' for event in experiment.events)
    notes = [
        'Collision perfect_merge, gas drag on (annular disk, mid-plane Gaussian), tides on (close-pass-energy-dissipation-v1).',
        'Drag evaluates the existing annular isothermal zones at cylindrical radius. Height only weights the midplane density by the isothermal scale height. There is no vertical gas velocity and no new disk method.',
        'Tides use the existing pericenter formula and apply only to a pair that does not become a contact in that step.',
        'Stellar contact is logged and does not end the run. Stop on contact is not selected.',
        'Direct summation is used. Tree gravity remains deferred.',
        f'Dissipated kinetic energy from drag is {experiment.dissipated_kinetic:.6g}. Tide events: {tides}. Mergers: {merges}. Ejections: {ejections}. Unmodeled stellar contacts: {unmodeled}.',
    ]
    if experiment.fluid_held:
        notes.append('At least one fluid subcycle was held because the Courant limit or a zone mass would not allow it. Drag continued from the zones that remained.')
    if experiment.status != 'complete':
        notes.append('The run stopped before the requested time: '+experiment.reason)
    else:
        notes.append('The requested year completed.')
    return RockyYearReport(
        hardware_description(), len(experiment.samples[0].bodies), ROCKY_YEAR_COUNT,
        ROCKY_YEAR_DURATION_YEARS, experiment.time, seconds, experiment.status, experiment.reason,
        min(steps), max(steps), combo['collision_mode'], combo['drag'], combo['tides'],
        merges, tides, ejections, unmodeled, experiment.dissipated_kinetic, tuple(notes))


def make_sun_giant_flyby(*, collision_mode='perfect_merge', drag=False, tides=False):
    """Sun, catalog Jupiter, and one slow visitor on a three-dimensional approach.

    The visitor radius is a disclosed drag cross-section, not a rocky surface.
    The disk, when enabled, is the annular isothermal disk placed on Jupiter's orbit.
    """
    from .data import CATALOG
    if collision_mode not in FORMATION_COLLISION_MODES:
        raise ValueError('Formation collision modes: ignore, stop_on_contact, perfect_merge, speed_angle.')
    if type(drag) is not bool or type(tides) is not bool:
        raise ValueError('Drag and tides must each be true or false.')
    sun = CATALOG['sun']
    jupiter = CATALOG['jupiter']
    distance = float(PLANETS['jupiter'].distance)
    speed = math.sqrt(G*(sun.mass+jupiter.mass)/distance)
    # Start unbound to the Sun, outside the giant-orbit disk, on a pass that
    # can reach Jupiter. Drag and tides are then able to change the outcome.
    offset = np.array([2.2, 0.5, 0.1])
    relative = np.array([-2.8, -0.6, -0.02])
    position = np.array([distance, 0., 0.])+offset
    velocity = np.array([0., speed, 0.])+relative
    visitor = BodySpec('Visitor', 'planet', 1e-6, 0.05, color='#c9b28d',
                       provenance='Disclosed drag cross-section of 0.05 AU, not a rocky surface.')
    bodies = (
        InitialBody('sun', sun, (0., 0., 0.), (0., 0., 0.)),
        InitialBody('jupiter', jupiter, (distance, 0., 0.), (0., speed, 0.)),
        InitialBody('visitor', visitor, tuple(float(component) for component in position),
                    tuple(float(component) for component in velocity)),
    )
    role = 'drag-on' if drag else 'drag-off'
    tide_role = 'tides-on' if tides else 'tides-off'
    return with_physics(InitialSetup(
        'Sun–Jupiter slow flyby', bodies, FLYBY_DT, collision_mode=collision_mode,
        provenance={'formation_model': FORMATION_MODEL, 'scenario': 'sun-jupiter-flyby',
                    'comparison_role': f'{role}/{tide_role}',
                    'visitor_radius_au': 0.05,
                    'note': 'Slow flyby near Jupiter with the Sun included. The visitor radius is a drag cross-section.'}),
        drag=DragPolicy(enabled=True, disk='giant') if drag else DragPolicy(),
        tides=TidalPolicy(enabled=True) if tides else TidalPolicy())


def _binding_word(energies):
    if not energies:
        return 'unbound'
    bound = [energy < 0. for energy in energies]
    if not any(bound):
        return 'unbound'
    if all(bound):
        return 'bound-throughout'
    if bound[-1]:
        return 'bound-at-end'
    return 'temporary'


def _flyby_summary(label, drag, tides):
    setup = make_sun_giant_flyby(drag=drag, tides=tides)
    experiment = FormationExperiment(setup, ejection_radius=80., sample_stride=1, seed=72, adaptive=True)
    experiment.integrate(FLYBY_DURATION_YEARS, finish=True)
    sun_energies = []
    minimum = math.inf
    last_sun = math.nan
    last_jupiter = math.nan
    initial_sun = math.nan
    for sample in experiment.samples:
        sun_energy, _distance = _sample_energy(sample, 'sun', 'visitor')
        jupiter_energy, distance = _sample_energy(sample, 'jupiter', 'visitor')
        if sun_energy is None:
            continue
        if not sun_energies:
            initial_sun = sun_energy
        sun_energies.append(sun_energy)
        last_sun = sun_energy
        last_jupiter = jupiter_energy
        minimum = min(minimum, distance)
    merged = any(event.kind == 'merge' and 'visitor' in event.parent_ids for event in experiment.events)
    if 'visitor' in experiment.prepared.body_ids and 'sun' in experiment.prepared.body_ids:
        last_sun, _distance = _relative_energy(experiment, 'sun', 'visitor')
        if 'jupiter' in experiment.prepared.body_ids:
            last_jupiter, distance = _relative_energy(experiment, 'jupiter', 'visitor')
            minimum = min(minimum, distance)
        sun_energies.append(last_sun)
    combo = experiment.combination()
    return FlybySummary(
        label, combo['collision_mode'], combo['drag'], combo['tides'], experiment.status, experiment.time,
        initial_sun, last_sun, last_jupiter, minimum, _binding_word(sun_energies), merged,
        sum(event.kind == 'tide' for event in experiment.events), experiment.dissipated_kinetic)


def _sample_energy(sample, primary, secondary):
    bodies = {body.id: body for body in sample.bodies}
    if primary not in bodies or secondary not in bodies:
        return None, None
    separation = np.asarray(bodies[secondary].r, dtype=float)-np.asarray(bodies[primary].r, dtype=float)
    velocity = np.asarray(bodies[secondary].v, dtype=float)-np.asarray(bodies[primary].v, dtype=float)
    distance = float(np.linalg.norm(separation))
    mu = G*(float(bodies[primary].mass)+float(bodies[secondary].mass))
    return .5*float(np.dot(velocity, velocity))-mu/distance, distance


def _relative_energy(experiment, primary, secondary):
    system = experiment.prepared.system
    i = experiment.prepared.body_ids.index(primary)
    j = experiment.prepared.body_ids.index(secondary)
    separation = system.r[j]-system.r[i]
    velocity = system.v[j]-system.v[i]
    distance = float(np.linalg.norm(separation))
    mu = G*(float(system.masses[i])+float(system.masses[j]))
    energy = .5*float(np.dot(velocity, velocity))-mu/distance
    return energy, distance


def _flyby_conclusion(free, drag, tides, combined):
    def phrase(summary):
        if summary.merged:
            binding = f'merged after {summary.binding}'
        else:
            binding = summary.binding
        return (f'{summary.label}: collision {summary.collision_mode}, drag {summary.drag}, tides {summary.tides}, '
                f'status {summary.status}, Sun-specific binding {binding}, '
                f'initial Sun-specific energy {summary.initial_sun_energy:.6g}, '
                f'last Sun-specific energy while the visitor exists {summary.sun_energy:.6g}, '
                f'last Jupiter-specific energy while the visitor exists {summary.jupiter_energy:.6g}, '
                f'minimum Jupiter distance {summary.minimum_jupiter_distance:.6g} AU, '
                f'tide events {summary.tide_count}')
    runs = (free, drag, tides, combined)
    longer = [item.label for item in runs if not item.merged and item.binding in ('bound-at-end', 'bound-throughout')]
    temporary = [item.label for item in runs if not item.merged and item.binding == 'temporary']
    merged = [item.label for item in runs if item.merged]
    claim = ('A longer-lived capture, for this window, is claimed only for the runs that end bound and unmerged: '
             + (', '.join(longer) if longer else 'none') + '. ')
    if temporary:
        claim += 'Temporary binding (bound and then unbound again inside the window) occurred for: '+', '.join(temporary)+'. '
    else:
        claim += 'No unmerged run was bound and then unbound again inside this window. '
    if merged:
        claim += ('A perfect merge removed the visitor as a separate body for: '+', '.join(merged)
                  + '. That merger is not a capture. ')
    if not longer and not temporary:
        claim += 'No selected option leaves a captured survivor. '
    if (combined.tide_count == 0 and not combined.merged and combined.binding == drag.binding
            and combined.minimum_jupiter_distance > 0.3):
        claim += ('On the drag-and-tides run the closest approach stays outside the tidal reach, '
                  'so no tide event is recorded and the end state matches drag alone. ')
    claim += ('The visitor starts unbound to the Sun. This is not a permanent-capture claim. '
              'A later scattering can still unbind a body, and the visitor radius of 0.05 AU is a drag cross-section, '
              'not a rocky surface.')
    return '\n'.join((phrase(free), phrase(drag), phrase(tides), phrase(combined), claim))


def run_sun_giant_flyby():
    """The same slow flyby with each energy sink off, and with drag and tides on."""
    free = _flyby_summary('gravity only', False, False)
    drag = _flyby_summary('drag on', True, False)
    tides = _flyby_summary('tides on', False, True)
    combined = _flyby_summary('drag and tides', True, True)
    return FlybyReport(hardware_description(), free, drag, tides, combined,
                       _flyby_conclusion(free, drag, tides, combined))
