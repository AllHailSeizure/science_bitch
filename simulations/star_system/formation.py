"""Late-stage assembly with parcel ancestry, checkpoints, and branched runs.

The physical stage is rocky embryos around one existing star. Each embryo
starts as one material parcel. Contacts use the disclosed perfect-merge
solver already in this package. Gas, dust, fragmentation, and stellar
mergers are outside this prototype. Ejection is checked at macro steps.
"""
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
from .core import G, SOLAR_RADIUS_AU, System
from .history import StateFrame
from .initialization import PreparedSystem, build_system
from .model import BodySpec, InitialBody, InitialSetup
from .persistence import SOLVER_BUILD, setup_digest
from .playback import Playback
from .recording import RecordedRun

FORMATION_MODEL = 'late-stage-perfect-merge-v1'
REPRESENTATION_ID = 'one-embryo-one-parcel-v1'
COLLISION_MODEL_ID = 'nonstellar-perfect-merge-v1'
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

_ASSUMPTIONS = (
    'Late-stage assembly: one star already exists, and every other body is a massive rocky embryo.',
    'Each embryo starts as one material parcel. Parcels keep their ids and do not split.',
    'Collisions use disclosed perfect merge: one remnant keeps the mass and linear momentum, and the relative kinetic energy is recorded as lost.',
    'Perfect merge does not fragment, partially accrete, or lose mass, so a collision has one descendant.',
    'A nonstellar body is ejected when its two-body specific energy about the star is positive and its star-centric distance has reached the ejection radius. The check runs at macro steps. The body is removed from the active set and its last state is kept.',
    'Ejection is a bookkeeping cut. After it, the frozen body no longer exchanges gravity with the active set.',
    'Stellar contact halts the run. This prototype does not merge stars.',
    'Gravity is planar and unsoftened. The fixed step is the setup dt.',
)
_QUESTIONS = (
    'Which initial parcels make up a surviving body, and how much mass came from each?',
    'When did mergers and ejections happen, including for a body that has left an empty region?',
    'If two runs share a checkpoint and one embryo velocity changes, how do their histories differ at the same simulated time?',
    'Do mass, parcel totals, and linear momentum follow the perfect-merge and ejection rules, and does the kinetic-energy drop at a merge match the recorded loss?',
    'How much does the first merger time move when the fixed step changes, while the remnant mass stays the combined parcel mass?',
)
_NOT_MODELED = (
    'A gas disk, gas drag, and dust as a population.',
    'Grain growth and any rule that turns a swarm into one body.',
    'Tree-code grouping of distant mass.',
    'Fragmentation, hit-and-run, erosion, and partial accretion.',
    'Three-dimensional orbits and relativity. Optional close-pass tidal dissipation is a separate experiment choice and stays off in this prototype.',
    'Stellar mergers and stellar internal structure.',
)
_SCOPE = (
    'Keep the first implementation on late-stage embryos, one parcel per body, perfect merge, ejection records, ancestry amounts, and checkpoint branching.',
    'Calculate a run to completion, then scrub and compare it. Inflated encounter radii make contacts inspectable; they are not rocky surfaces, and this fixed step is not an interactive solver for realistic radii.',
    'Leave stellar contact as a halt.',
)
_GAS = (
    'An earlier gas-and-dust start needs a fluid disk, drag between gas and dust, and an explicit rule for when a swarm becomes one body without dropping mass, momentum, origin, or the internal state the collision model needs.',
    'Partial accretion, erosion, hit-and-run, and fragmentation need an outcome map such as Stewart and Leinhardt, because perfect merge cannot send mass to more than one descendant.',
    'Tree gravity is a later optimization. At a few hundred bodies the measured cost is direct summation, and grouping would have to preserve parcel identity to be usable here.',
)
_PROVENANCE_KEYS = ('run_id', 'model', 'representation', 'collision_model', 'seed', 'setup_digest',
                    'parent_checkpoint_id', 'change', 'execution_environment', 'solver_build')
_BODY_KEYS = ('id', 'name', 'kind', 'mass', 'radius', 'status', 'r', 'v', 'contributions', 'reference_id')
_EVENT_KEYS = ('time', 'kind', 'parent_ids', 'remnant_id', 'descendant_ids', 'contributions',
               'lost_relative_kinetic', 'unmodeled_spin', 'location', 'message')
_SAMPLE_KEYS = ('time', 'phase', 'bodies')
_CHANGE_KEYS = ('kind', 'body_id', 'delta_v', 'note')
_DOCUMENT_KEYS = ('schema', 'id', 'model', 'name', 'time', 'dt', 'ejection_radius', 'sample_stride',
                  'status', 'reason', 'completed_steps', 'assumptions', 'provenance', 'active',
                  'removed', 'events', 'samples')


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
    'Each simulated object is one body. Every embryo begins as a single material parcel with a stable id and a mass. The prototype does not group parcels into a tree and does not represent gas or dust.',
    'Disclosed perfect merge on contact (nonstellar-perfect-merge-v1), using the existing positive-time Verlet contact solver. One remnant keeps the mass and linear momentum. Relative kinetic energy and the orbital spin residual are recorded as lost. An unbound body outside the ejection radius leaves the active set and stays in the history.',
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
    r: tuple[float, float]
    v: tuple[float, float]
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
    location: tuple[float, float]
    message: str


@dataclass(frozen=True)
class VelocityChange:
    """One disclosed change applied when a run branches from a checkpoint."""
    body_id: str
    delta_v: tuple[float, float]
    note: str

    def __post_init__(self):
        if not isinstance(self.body_id, str) or not self.body_id.strip():
            raise ValueError('Name the embryo whose velocity changes.')
        try:
            components = tuple(self.delta_v)
        except TypeError as exc:
            raise ValueError('delta_v must be two finite AU/yr components.') from exc
        if len(components) != 2:
            raise ValueError('delta_v must be two finite AU/yr components.')
        vector = tuple(float(component) for component in components)
        if not all(math.isfinite(component) for component in vector):
            raise ValueError('delta_v must be two finite AU/yr components.')
        if vector == (0.0, 0.0):
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
class SyncedFrame:
    time: float
    left: StateFrame
    right: StateFrame
    common_start: float
    common_end: float


def _star_body():
    return InitialBody('star', BodySpec('Star', 'star', 1., STAR_RADIUS, temperature=5772., color='#ffe2b0',
                                         provenance='formation prototype'), (0., 0.), (0., 0.))


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
                          dt=BENCHMARK_DT, include_unbound=True):
    """Seeded annulus of massive embryos, plus one disclosed unbound body when requested.

    Angles and true anomalies come from PCG64. The initial velocities are
    star-centric ellipses. Embryo radii are encounter radii supplied by the caller.
    """
    if type(count) is not int or isinstance(count, bool) or not 1 <= count <= 200:
        raise ValueError('Embryo count must be an integer from 1 to 200.')
    if type(seed) is not int or isinstance(seed, bool) or seed < 0:
        raise ValueError('Formation seed must be a nonnegative integer.')
    _positive('Inner radius', inner)
    _positive('Outer radius', outer)
    if inner >= outer:
        raise ValueError('The annulus needs inner < outer, in AU.')
    _positive('Embryo mass', embryo_mass)
    _positive('Embryo radius', embryo_radius)
    _positive('Fixed dt', dt)
    if not math.isfinite(eccentricity) or not 0 <= eccentricity < 1:
        raise ValueError('Eccentricity must be finite and in [0, 1).')
    if inner*(1-eccentricity) <= STAR_RADIUS+embryo_radius:
        raise ValueError('Annulus periapses intersect the star or an embryo surface.')
    rng = np.random.Generator(np.random.PCG64(seed))
    star = _star_body()
    placed = [(np.zeros(2), star.spec.radius)]
    bodies = [star]
    for index in range(count):
        semi_major = inner+(outer-inner)*index/max(count-1, 1)
        ecc = eccentricity*(.75+.25*float(rng.random()))
        body = None
        for _attempt in range(80):
            position, velocity = _ellipse_state(semi_major, ecc, float(rng.random())*2*math.pi,
                                                float(rng.random())*2*math.pi, 1.)
            if _clear(position, embryo_radius, placed):
                body = _embryo(f'embryo-{index:03d}', f'Embryo {index}', embryo_mass, embryo_radius,
                               (float(position[0]), float(position[1])), (float(velocity[0]), float(velocity[1])))
                placed.append((np.array(position, dtype=float), embryo_radius))
                break
        if body is None:
            raise ValueError('Embryos overlap. Reduce count or radius, or widen the annulus.')
        bodies.append(body)
    if include_unbound:
        position = np.array([outer+1.5, 0.])
        while not _clear(position, embryo_radius, placed):
            position = position+np.array([.5, 0.])
        bodies.append(_embryo('unbound', 'Unbound', embryo_mass, embryo_radius,
                              (float(position[0]), float(position[1])), (12., .4)))
    return InitialSetup('Late-stage assembly', tuple(bodies), dt, collision_mode='perfect_merge',
                        provenance={'formation_model': FORMATION_MODEL, 'scenario': 'late-stage', 'seed': seed,
                                    'generator': 'PCG64', 'embryo_count': count, 'include_unbound': bool(include_unbound)})


def _positive(label, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be finite and positive.')
    return float(value)


def _contributions(mapping):
    return tuple(Contribution(key, float(mapping[key])) for key in sorted(mapping))


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError('Formation state must be finite.')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Formation state must be finite.')
    return result


def _momentum(bodies):
    total = np.zeros(2)
    for body in bodies:
        total += body.mass*np.array(body.v)
    return total


def _kinetic(bodies):
    return sum(.5*body.mass*(body.v[0]*body.v[0]+body.v[1]*body.v[1]) for body in bodies)


def _max_abs(vector):
    return float(np.max(np.abs(vector))) if len(np.atleast_1d(vector)) else 0.


def _run_id(setup_digest_value, seed, parent_id, change):
    payload = {'setup_digest': setup_digest_value, 'seed': seed, 'parent_checkpoint_id': parent_id,
               'change': None if change is None else _change_document(change)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:16]


def _change_document(change):
    return {'kind': 'velocity', 'body_id': change['body_id'],
            'delta_v': [_num(component) for component in change['delta_v']], 'note': change['note']}


def _new_provenance(setup, seed):
    provenance = {'run_id': '', 'model': FORMATION_MODEL, 'representation': REPRESENTATION_ID,
                  'collision_model': COLLISION_MODEL_ID, 'seed': seed, 'setup_digest': setup_digest(setup),
                  'parent_checkpoint_id': None, 'change': None,
                  'execution_environment': {'numpy': np.__version__}, 'solver_build': SOLVER_BUILD}
    provenance['run_id'] = _run_id(provenance['setup_digest'], seed, None, None)
    return provenance


class FormationExperiment:
    """One formation integration with a parcel ledger and recorded samples."""

    def __init__(self, setup, *, ejection_radius, sample_stride=1, seed=None):
        if not isinstance(setup, InitialSetup):
            raise TypeError('Formation starts from an InitialSetup.')
        if setup.collision_mode != 'perfect_merge':
            raise ValueError('Formation uses the disclosed perfect_merge collision mode.')
        if seed is not None and (type(seed) is not int or isinstance(seed, bool) or seed < 0):
            raise ValueError('Formation seed must be a nonnegative integer or omitted.')
        if type(sample_stride) is not int or isinstance(sample_stride, bool) or sample_stride < 1:
            raise ValueError('sample_stride must be a positive integer.')
        self.ejection_radius = _positive('Ejection radius', ejection_radius)
        self.sample_stride = sample_stride
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
        self.composition = {body_id: {body_id: float(mass)}
                            for body_id, mass in zip(self.prepared.body_ids, self.prepared.system.masses)}
        self.removed = {}
        self.events = []
        self.samples = []
        self.status = 'running'
        self.reason = ''
        self.completed_steps = 0
        self._applied_merges = 0
        self.provenance = _new_provenance(setup, seed)
        extras = [f'Ejection radius is {self.ejection_radius:.6g} AU and is evaluated at macro steps.']
        radii = [body.spec.radius for body in setup.bodies if body.spec.kind != 'star']
        if radii and max(radii) > .001:
            extras.append('At least one embryo radius exceeds 0.001 AU. Treat a radius this large as an inflated encounter radius, not a rocky surface.')
        self.assumptions = _ASSUMPTIONS+tuple(extras)
        self._record_current(None, force=True)

    @property
    def time(self):
        return float(self.prepared.system.time)

    def describe(self):
        """Simulated time, status, and the model assumptions for this run."""
        status = f'Simulated time {self.time:.6g} yr. Status: {self.status}.'
        detail = f'Model {self.provenance["model"]}.'
        if self.reason:
            detail = f'{detail} {self.reason}'
        return '\n'.join((status, detail, *self.assumptions))

    def integrate(self, years, *, finish=False):
        """Advance by whole macro steps until simulated time has grown by `years`."""
        if self.status != 'running':
            raise ValueError(f'Cannot integrate a {self.status} run. Branch from a checkpoint to calculate a new future.')
        years = _positive('Duration', years)
        target = self.time+years
        guard = 0
        limit = math.ceil(years/self.prepared.system.dt)+2
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
            try:
                outcome = advance_prepared(self.prepared)
            except ValueError as exc:
                self.status = 'halted'
                self.reason = str(exc)
                break
            for frame in outcome.event_frames:
                if frame.event_phase == 'pre':
                    self.samples.append(self._sample_from_frame(frame, 'pre'))
                elif frame.event_phase == 'post':
                    self._consume_merges(frame)
                    self.samples.append(self._sample_from_frame(frame, 'post'))
                else:
                    raise ValueError('Unexpected collision frame.')
            self.prepared = outcome.prepared
            self._eject()
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
            if event.kind == 'merge':
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
        closed_mass = sum(body.mass for body in leaves)
        parcel_initial, parcel_now = {}, {}
        for body in initial.bodies:
            for item in body.contributions:
                parcel_initial[item.parcel_id] = parcel_initial.get(item.parcel_id, 0.)+item.mass
        for body in leaves:
            for item in body.contributions:
                parcel_now[item.parcel_id] = parcel_now.get(item.parcel_id, 0.)+item.mass
        keys = set(parcel_initial) | set(parcel_now)
        parcel_error = max((abs(parcel_now.get(key, 0.)-parcel_initial.get(key, 0.)) for key in keys), default=0.)
        merge_momentum = merge_kinetic = ejection_momentum = 0.
        merge_count = ejection_count = index = 0
        for sample_index, sample in enumerate(self.samples[:-1]):
            nxt = self.samples[sample_index+1]
            if sample.phase != 'pre' or nxt.phase != 'post':
                continue
            while index < len(self.events) and self.events[index].kind != 'merge':
                index += 1
            batch = []
            tolerance = 1e-12*max(1., abs(sample.time))
            while (index < len(self.events) and self.events[index].kind == 'merge'
                   and abs(self.events[index].time-sample.time) <= tolerance):
                batch.append(self.events[index])
                index += 1
            merge_count += len(batch)
            merge_momentum = max(merge_momentum, _max_abs(_momentum(sample.bodies)-_momentum(nxt.bodies)))
            measured = _kinetic(nxt.bodies)-_kinetic(sample.bodies)
            expected = -sum(event.lost_relative_kinetic for event in batch)
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
        velocity = system.v.copy()
        velocity[index, 0] += change.delta_v[0]
        velocity[index, 1] += change.delta_v[1]
        updated = System(system.masses.copy(), system.r.copy(), velocity, system.dt, system.time)
        self.prepared = replace(self.prepared, system=updated)
        stored = {'kind': 'velocity', 'body_id': change.body_id, 'delta_v': change.delta_v, 'note': change.note}
        self.provenance = {**self.provenance, 'parent_checkpoint_id': parent_id, 'change': stored,
                           'run_id': _run_id(self.provenance['setup_digest'], self.provenance['seed'], parent_id, stored)}
        self.status = 'running'
        self.reason = ''
        self._record_current('branch', force=True)

    def _consume_merges(self, frame):
        merges = [event for event in frame.events if isinstance(event, CollisionEvent) and event.kind == 'merge']
        for event in merges[self._applied_merges:]:
            self._apply_merge(event)
        self._applied_merges = len(merges)

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
        total = sum(next(body.mass for body in pre.bodies if body.id == parent) for parent in event.parent_ids)
        location = (
            sum(next(body.mass*body.r[0] for body in pre.bodies if body.id == parent) for parent in event.parent_ids)/total,
            sum(next(body.mass*body.r[1] for body in pre.bodies if body.id == parent) for parent in event.parent_ids)/total,
        )
        self.events.append(FormationEvent(float(event.time), 'merge', tuple(event.parent_ids), event.remnant_id,
                                          _contributions(composition), (event.remnant_id,), float(event.lost_relative_kinetic),
                                          float(event.unmodeled_spin), location, event.message))

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
        updated = System(np.array(system.masses[keep], copy=True), np.array(system.r[keep], copy=True),
                         np.array(system.v[keep], copy=True), system.dt, system.time)
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
                (_num(frame.r[index, 0]), _num(frame.r[index, 1])),
                (_num(frame.v[index, 0]), _num(frame.v[index, 1])),
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
        return {
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


def _carriers(sample, parcels):
    return any(parcels.intersection(item.parcel_id for item in body.contributions) for body in sample.bodies)


def _same_state(left, right):
    return (left.time == right.time and tuple((body.id, body.r, body.v) for body in left.bodies)
            == tuple((body.id, body.r, body.v) for body in right.bodies))


def _body_document(body):
    return {'id': body.id, 'name': body.name, 'kind': body.kind, 'mass': _num(body.mass), 'radius': _num(body.radius),
            'status': body.status, 'r': [_num(body.r[0]), _num(body.r[1])], 'v': [_num(body.v[0]), _num(body.v[1])],
            'contributions': [{'parcel_id': item.parcel_id, 'mass': _num(item.mass)}
                              for item in sorted(body.contributions, key=lambda item: item.parcel_id)],
            'reference_id': body.reference_id}


def _event_document(event):
    return {'time': _num(event.time), 'kind': event.kind, 'parent_ids': list(event.parent_ids),
            'remnant_id': event.remnant_id, 'descendant_ids': list(event.descendant_ids),
            'contributions': [{'parcel_id': item.parcel_id, 'mass': _num(item.mass)}
                              for item in sorted(event.contributions, key=lambda item: item.parcel_id)],
            'lost_relative_kinetic': _num(event.lost_relative_kinetic), 'unmodeled_spin': _num(event.unmodeled_spin),
            'location': [_num(event.location[0]), _num(event.location[1])], 'message': event.message}


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


def _experiment_from_document(document):
    if type(document) is not dict:
        raise ValueError('Formation checkpoint must be a JSON object.')
    if document.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError(f'Unsupported formation checkpoint schema; expected {CHECKPOINT_SCHEMA}.')
    if document.get('model') != FORMATION_MODEL:
        raise ValueError(f'Unsupported formation model; expected {FORMATION_MODEL}.')
    _object(document, _DOCUMENT_KEYS, 'Formation checkpoint')
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
    if document['status'] not in ('running', 'complete', 'halted'):
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
    if any(body.status not in ('merged', 'ejected') for body in removed_list):
        raise ValueError('Removed bodies must be merged or ejected.')
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
    shell = InitialSetup(name, shell_bodies, dt, collision_mode='perfect_merge')
    collision_events = []
    for event in events:
        if event.kind == 'merge':
            ancestry = tuple(sorted({item.parcel_id for item in event.contributions}))
            collision_events.append(CollisionEvent(event.time, 'merge', event.parent_ids, event.remnant_id, ancestry,
                                                   event.lost_relative_kinetic, event.unmodeled_spin, event.message))
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
    experiment._applied_merges = sum(event.kind == 'merge' for event in events)
    if experiment.checkpoint_id != identity:
        raise ValueError('Checkpoint did not round-trip.')
    return experiment


def body_spec(body):
    return BodySpec(body.name, body.kind, body.mass, body.radius)


def _provenance_from_document(value):
    _object(value, _PROVENANCE_KEYS, 'Provenance')
    if value['model'] != FORMATION_MODEL or value['representation'] != REPRESENTATION_ID or value['collision_model'] != COLLISION_MODEL_ID:
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
        if type(delta) is not list or len(delta) != 2:
            raise ValueError('delta_v must contain two components.')
        change = {'kind': 'velocity', 'body_id': _string(change['body_id'], 'Changed body'),
                  'delta_v': (_finite_number(delta[0], 'delta_v'), _finite_number(delta[1], 'delta_v')),
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
    if body.status not in ('active', 'merged', 'ejected'):
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
    if kind not in ('merge', 'ejection'):
        raise ValueError('Unsupported formation event.')
    parents = tuple(_string(item, 'Parent id') for item in _list(value['parent_ids'], 'Parent ids'))
    if not parents:
        raise ValueError('A formation event needs a parent id.')
    remnant = _string(value['remnant_id'], 'Remnant id', nullable=True)
    descendants = tuple(_string(item, 'Descendant id') for item in _list(value['descendant_ids'], 'Descendant ids'))
    if kind == 'merge' and (remnant is None or descendants != (remnant,)):
        raise ValueError('A merge event needs its remnant as the only descendant.')
    if kind == 'ejection' and (remnant is not None or descendants != parents):
        raise ValueError('An ejection keeps the same body as its descendant.')
    contributions = tuple(_contribution_from_document(item) for item in _list(value['contributions'], 'Contributions'))
    return FormationEvent(_finite_number(value['time'], 'Event time'), kind, parents, remnant, contributions, descendants,
                          _finite_number(value['lost_relative_kinetic'], 'Lost kinetic energy'),
                          _finite_number(value['unmodeled_spin'], 'Unmodeled spin'),
                          _vector(value['location'], 'Event location'), _string(value['message'], 'Event message'))


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
    if type(value) is not list or len(value) != 2:
        raise ValueError(f'{label} must contain two components.')
    return (_finite_number(value[0], label), _finite_number(value[1], label))


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
    elif experiment.status == 'halted':
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
