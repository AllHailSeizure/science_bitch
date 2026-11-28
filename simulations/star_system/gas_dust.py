"""Gas-and-dust disk, swarm-to-body transition, and aerodynamic capture.

The fluid is a planar annular isothermal disk. Dust is a set of Lagrangian
swarms, each with a material-parcel ledger. Bodies that pass the swarm rule,
and any authored intruder, feel quadratic drag from the same fluid. Contacts
between those bodies reuse the existing perfect-merge primitive. The step
stays fixed. Three-dimensional orbits, adaptive steps, fragmentation, and
tides are outside this prototype.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import tracemalloc

import numpy as np

from .collisions import merge_pair
from .core import G, SOLAR_RADIUS_AU, System
from .formation import Contribution, FormationExperiment, VelocityChange, hardware_description
from .initialization import PreparedSystem
from .model import DRAG_MODEL_ID, BodySpec, InitialBody, InitialSetup
from .persistence import SOLVER_BUILD

DISK_MODEL = 'gas-dust-annular-isothermal-v1'
REPRESENTATION_ID = 'annular-fluid-and-dust-swarms-v1'
GROWTH_MODEL_ID = 'representative-sweep-up-v1'
SWARM_RULE_ID = 'hill-bound-stokes-v1'
COLLISION_MODEL_ID = 'nonstellar-perfect-merge-v1'
CHECKPOINT_SCHEMA = 'gas-dust-checkpoint-v1'

_AU_M = 149597870700.0
_MSUN_KG = 1.98847e30
COURANT = 0.4
STOKES_MIN = 1.0
MASS_MIN = 1e-9
DRAG_COEFFICIENT = 0.44
ROCK_DENSITY_KG_M3 = 3000.0
DISK_SEED = 71
DISK_ZONE_COUNT = 16
DISK_SWARM_COUNT = 6
DISK_INNER = 0.8
DISK_OUTER = 2.4
DISK_SIGMA_REF = 2e-4
DISK_REFERENCE_RADIUS = 1.5
DISK_PRESSURE_INDEX = 1.0
DISK_SCALE_HEIGHT_RATIO = 0.05
DISK_DT = 0.02
DISK_DURATION_YEARS = 0.5
READY_SWARM_ID = 'swarm-ready'
READY_MASS = 1e-8
READY_SPATIAL_RADIUS = 1e-4
READY_INTERNAL_SPEED = 1e-3
ORDINARY_MASS = 1e-10
ORDINARY_SPATIAL_RADIUS = 1e-3

CAPTURE_SEED = 71
CAPTURE_INNER = 0.2
CAPTURE_OUTER = 1.6
CAPTURE_ZONES = 24
CAPTURE_SIGMA0 = 0.05
CAPTURE_SCALE = 0.45
CAPTURE_SOUND_SPEED = 0.35
CAPTURE_DT = 0.002
CAPTURE_YEARS = 0.6
CAPTURE_POSITION = (-1.8, 0.6)
CAPTURE_VELOCITY = (7.0, 0.0)
CAPTURE_MASS = 1e-6
CAPTURE_RADIUS = 0.04
CAPTURE_BODY_ID = 'intruder'

_ASSUMPTIONS = (
    'The star is a pinned point mass. Annular zones store axisymmetric radial and angular momentum, so the closed momentum exchange is that pair, not a free cartesian vector.',
    'Gas is a piecewise-constant annular isothermal fluid. The radial step is a donor-cell flux with no-flux ends, after a source step that balances centrifugal support, stellar gravity, and the pressure gradient. Gas is not self-gravitating.',
    'A dust swarm is one population of equal grains plus a parcel ledger. It feels stellar gravity, gravity from formed bodies, and Epstein drag. Swarms do not source gravity.',
    'Epstein drag and quadratic body drag deposit the opposite momentum on the local annular zone. The isothermal model records the kinetic energy that exchange removes and does not return it as heat.',
    'Grain growth is representative sweep-up from the solid velocity at the start of the step: ds/dt = rho_dust * relative_speed / (4 * grain_density). The fluid update does not change that velocity. Swarm mass and parcel amounts stay put.',
    'Two swarms whose clouds overlap coagulate when the combined internal speed, including their relative motion, stays below the cloud escape speed. Mass, momentum, and parcels add. Relative kinetic energy becomes internal kinetic energy.',
    'A swarm becomes one body only when the Stokes number, grain radius, mass, Hill compactness, and internal binding all pass. The body keeps the swarm mass, velocity, parcels, grain radius, and internal speed. Its radius is the compact-sphere radius at the disclosed bulk density, which is the radius perfect merge uses.',
    'Body-body contact calls the existing perfect-merge primitive. That remnant keeps the collision model volume radius and the parcel ledger. Stellar contact halts. This step is not adapted when the Courant limit is crossed.',
)
_QUESTIONS = (
    'Which solid parcels sit in a swarm or a body, and how much mass came from each?',
    'Does gas drag move angular momentum from a Keplerian dust swarm onto sub-Keplerian gas?',
    'Does representative grain growth change the grain radius while the swarm mass and parcel amounts stay the same?',
    'When the swarm rule passes, does the new body keep the swarm mass, momentum, parcels, grain radius, and internal speed, and a positive radius the perfect-merge model can use?',
    'Does a checkpoint resume the same gas, swarms, and parcels, and does one velocity branch leave the source run unchanged?',
    'Does quadratic drag from a gaseous envelope bind a hyperbolic flyby that stays unbound when the same drag is turned off?',
)
_NOT_MODELED = (
    'Three-dimensional motion, vertical settling, and a turbulence spectrum.',
    'SPH or GADGET hydrodynamics, gas self-gravity, and the streaming instability.',
    'A multi-bin Smoluchowski grain distribution.',
    'Fragmentation, hit-and-run, erosion, and partial accretion.',
    'An adaptive timestep, a close-encounter integrator, and non-planar orbits.',
    'Tides, relativity, and stellar mergers.',
    'Tree-code grouping of distant mass.',
    'A permanent-capture claim. Aerodynamic binding in this planar fixed-step model is not that result.',
)
_CAPTURE = (
    'Gas drag pulls a body toward the local gas velocity of the disk or envelope and removes hyperbolic excess energy. The disclosed envelope flyby is integrated twice. With quadratic drag the intruder can become bound. With drag disabled the same initial state stays unbound.',
    'That bound orbit is not a permanent capture. Fragmenting collisions remain #69, three-dimensional orbits and an adaptive or close-encounter integrator remain #70, and tides remain #79.',
    'A sweep of temporary three-body captures can still use the existing flyby behavior. This drag result does not replace that sweep.',
)

_PROVENANCE_KEYS = ('run_id', 'model', 'representation', 'drag_model', 'growth_model', 'swarm_rule',
                    'collision_model', 'seed', 'setup_digest', 'parent_checkpoint_id', 'change',
                    'execution_environment', 'solver_build')
_PARAMETER_KEYS = ('sound_speed', 'grain_density', 'bulk_density', 'stokes_min', 'grain_radius_min',
                   'mass_min', 'drag_coefficient', 'drag_enabled')
_STAR_KEYS = ('mass', 'radius')
_ZONE_KEYS = ('id', 'r_inner', 'r_outer', 'mass', 'radial_momentum', 'angular_momentum')
_SWARM_KEYS = ('id', 'name', 'mass', 'grain_radius', 'internal_speed', 'spatial_radius', 'r', 'v',
               'contributions', 'status')
_BODY_KEYS = ('id', 'name', 'mass', 'radius', 'grain_radius', 'internal_speed', 'bulk_density', 'r', 'v',
              'contributions', 'status')
_EVENT_KEYS = ('time', 'kind', 'parent_ids', 'remnant_id', 'descendant_ids', 'contributions', 'mass',
               'momentum', 'grain_radius', 'internal_speed', 'internal_kinetic', 'radius',
               'lost_relative_kinetic', 'unmodeled_spin', 'message')
_SAMPLE_KEYS = ('time', 'phase', 'zones', 'swarms', 'bodies')
_CHANGE_KEYS = ('kind', 'body_id', 'delta_v', 'note')
_DOCUMENT_KEYS = ('schema', 'id', 'model', 'name', 'time', 'dt', 'sample_stride', 'status', 'reason',
                  'completed_steps', 'assumptions', 'provenance', 'parameters', 'star', 'zones', 'swarms',
                  'bodies', 'removed_swarms', 'removed_bodies', 'events', 'samples', 'dissipated_kinetic',
                  'drag_momentum_error', 'transition_momentum_error')


def meters_to_au(meters):
    """Convert a length in meters to AU using the same meter as this package's G."""
    value = _positive('Length', meters)
    return value/_AU_M


def solar_density(kg_per_m3):
    """Convert kg/m^3 to solar masses per AU^3 using the constants inside G."""
    value = _positive('Density', kg_per_m3)
    return value*(_AU_M**3)/_MSUN_KG


@dataclass(frozen=True)
class GasDustSpecification:
    """The gas-and-dust model this module actually runs."""
    starting_distribution: str
    representation: str
    fluid: str
    drag: str
    growth: str
    swarm_rule: str
    collision_model: str
    questions: tuple[str, ...]
    assumptions: tuple[str, ...]
    not_modeled: tuple[str, ...]
    capture: tuple[str, ...]


SPECIFICATION = GasDustSpecification(
    'A pinned star, a piecewise-constant annular gas disk, and Lagrangian dust swarms on Keplerian orbits. One disclosed swarm starts already past the body rule. The capture setup replaces the power-law disk with a Gaussian gaseous envelope and one authored intruder.',
    'A gas zone is a vertically integrated annular fluid element, not a molecule and not a 3D particle. A dust swarm is a single-size grain population with a parcel ledger, not yet a gravitating body. A body is one planetesimal or intruder and one perfect-merge participant. Solid parcels stay Lagrangian. Gas identity is the zone id; gas mass is conserved in total because fluxes move it between zones.',
    'Planar isothermal annular hydrodynamics: centrifugal support, stellar gravity, a finite-difference pressure force, and donor-cell advection. No gas self-gravity and no vertical velocity.',
    'Swarms use Epstein drag. Bodies use quadratic drag with the disclosed sphere coefficient. Both return the opposite radial and angular momentum to the zone that contains them.',
    'Representative sweep-up. The grain radius grows. The swarm mass and the parcel amounts do not.',
    'hill-bound-stokes-v1. Stokes number, grain radius, mass, Hill compactness, and internal binding must all pass. The transition copies mass, velocity, parcels, grain radius, and internal speed, and sets the compact-sphere radius.',
    'Existing nonstellar-perfect-merge-v1 for body-body contact only. The collision rule is unchanged. Stellar contact halts.',
    _QUESTIONS, _ASSUMPTIONS, _NOT_MODELED, _CAPTURE)


def specification():
    """Return the distribution, rules, questions, and capture limits this module runs."""
    return SPECIFICATION


@dataclass(frozen=True)
class DiskParameters:
    sound_speed: float
    grain_density: float
    bulk_density: float
    stokes_min: float
    grain_radius_min: float
    mass_min: float
    drag_coefficient: float
    drag_enabled: bool

    def __post_init__(self):
        _positive('Sound speed', self.sound_speed)
        _positive('Grain density', self.grain_density)
        _positive('Bulk density', self.bulk_density)
        _positive('Stokes threshold', self.stokes_min)
        _positive('Grain-radius threshold', self.grain_radius_min)
        _positive('Mass threshold', self.mass_min)
        _positive('Drag coefficient', self.drag_coefficient)
        if type(self.drag_enabled) is not bool:
            raise ValueError('drag_enabled must be true or false.')


@dataclass
class Zone:
    id: str
    r_inner: float
    r_outer: float
    mass: float
    radial_momentum: float
    angular_momentum: float

    @property
    def area(self):
        return math.pi*(self.r_outer**2-self.r_inner**2)

    @property
    def radius(self):
        return math.sqrt(.5*(self.r_inner**2+self.r_outer**2))

    @property
    def sigma(self):
        return self.mass/self.area

    @property
    def v_r(self):
        return self.radial_momentum/self.mass

    @property
    def specific_angular(self):
        return self.angular_momentum/self.mass

    @property
    def v_phi(self):
        return self.specific_angular/self.radius


@dataclass
class Swarm:
    id: str
    name: str
    mass: float
    grain_radius: float
    internal_speed: float
    spatial_radius: float
    r: np.ndarray
    v: np.ndarray
    contributions: dict

    def copy(self):
        return Swarm(self.id, self.name, float(self.mass), float(self.grain_radius), float(self.internal_speed),
                     float(self.spatial_radius), np.array(self.r, dtype=float, copy=True),
                     np.array(self.v, dtype=float, copy=True), dict(self.contributions))


@dataclass
class Body:
    id: str
    name: str
    mass: float
    radius: float
    grain_radius: float
    internal_speed: float
    bulk_density: float
    r: np.ndarray
    v: np.ndarray
    contributions: dict

    def copy(self):
        return Body(self.id, self.name, float(self.mass), float(self.radius), float(self.grain_radius),
                    float(self.internal_speed), float(self.bulk_density), np.array(self.r, dtype=float, copy=True),
                    np.array(self.v, dtype=float, copy=True), dict(self.contributions))


@dataclass(frozen=True)
class DiskSetup:
    name: str
    seed: int
    dt: float
    star_mass: float
    star_radius: float
    parameters: DiskParameters
    zones: tuple[Zone, ...]
    swarms: tuple[Swarm, ...]
    bodies: tuple[Body, ...]


@dataclass(frozen=True)
class RuleVerdict:
    accepted: bool
    failed: tuple[str, ...]
    stokes: float
    hill_radius: float
    escape_speed: float


@dataclass(frozen=True)
class SolidView:
    id: str
    name: str
    kind: str
    mass: float
    radius: float
    grain_radius: float
    internal_speed: float
    spatial_radius: float
    bulk_density: float
    r: tuple[float, float]
    v: tuple[float, float]
    contributions: tuple[Contribution, ...]
    status: str


@dataclass(frozen=True)
class ZoneView:
    id: str
    r_inner: float
    r_outer: float
    mass: float
    radial_momentum: float
    angular_momentum: float


@dataclass(frozen=True)
class DiskSample:
    time: float
    phase: str | None
    zones: tuple[ZoneView, ...]
    swarms: tuple[SolidView, ...]
    bodies: tuple[SolidView, ...]


@dataclass(frozen=True)
class DiskEvent:
    time: float
    kind: str
    parent_ids: tuple[str, ...]
    remnant_id: str | None
    descendant_ids: tuple[str, ...]
    contributions: tuple[Contribution, ...]
    mass: float
    momentum: tuple[float, float]
    grain_radius: float
    internal_speed: float
    internal_kinetic: float
    radius: float
    lost_relative_kinetic: float
    unmodeled_spin: float
    message: str


@dataclass(frozen=True)
class ParcelStory:
    parcel_id: str
    mass: float
    holder_ids: tuple[str, ...]
    events: tuple[DiskEvent, ...]
    samples: tuple[DiskSample, ...]


@dataclass(frozen=True)
class DiskAccounting:
    initial_gas_mass: float
    gas_mass: float
    initial_solid_mass: float
    solid_mass: float
    parcel_error: float
    angular_momentum_error: float
    drag_momentum_error: float
    transition_momentum_error: float
    dissipated_kinetic: float
    promotion_count: int
    coagulation_count: int
    merge_count: int


@dataclass(frozen=True)
class CaptureReport:
    model: str
    drag_model: str
    years: float
    dt: float
    intruder_mass: float
    intruder_radius: float
    drag_specific_energy: float
    free_specific_energy: float
    drag_initial_energy: float
    free_initial_energy: float
    drag_bound: bool
    free_bound: bool
    drag_dissipated_kinetic: float
    free_dissipated_kinetic: float
    drag_minimum_radius: float
    free_minimum_radius: float
    drag_status: str
    free_status: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class DiskReport:
    hardware: str
    model: str
    representation: str
    swarm_rule: str
    seed: int
    zone_count: int
    swarm_count: int
    parcel_count: int
    body_count: int
    requested_years: float
    simulated_years: float
    calculation_seconds: float
    peak_python_bytes: int
    recording_bytes: int
    sample_count: int
    promotion_count: int
    coagulation_count: int
    status: str
    questions: tuple[str, ...]
    notes: tuple[str, ...]
    capture: CaptureReport


def default_parameters(*, sound_speed, drag_enabled=True):
    """Thresholds and densities shared by the disk and envelope setups."""
    return DiskParameters(sound_speed, GRAIN_DENSITY, GRAIN_DENSITY, STOKES_MIN, GRAIN_RADIUS_MIN, MASS_MIN,
                          DRAG_COEFFICIENT, bool(drag_enabled))


def make_disk_setup(*, seed=DISK_SEED, zone_count=DISK_ZONE_COUNT, swarm_count=DISK_SWARM_COUNT,
                    include_ready=True, dt=DISK_DT):
    """Power-law gas annulus and seeded Keplerian swarms.

    Angles come from PCG64. One optional swarm starts past the body rule so a
    short run has a known promotion. Ordinary swarms start at millimeter grains.
    """
    if type(seed) is not int or isinstance(seed, bool) or seed < 0:
        raise ValueError('Disk seed must be a nonnegative integer.')
    if type(zone_count) is not int or isinstance(zone_count, bool) or zone_count < 2:
        raise ValueError('Zone count must be an integer of at least 2.')
    if type(swarm_count) is not int or isinstance(swarm_count, bool) or swarm_count < 0:
        raise ValueError('Swarm count must be a nonnegative integer.')
    star_mass = 1.
    reference = DISK_REFERENCE_RADIUS
    omega = math.sqrt(G*star_mass/reference**3)
    sound_speed = DISK_SCALE_HEIGHT_RATIO*reference*omega
    parameters = default_parameters(sound_speed=sound_speed)
    zones = _power_law_zones(zone_count, DISK_INNER, DISK_OUTER, DISK_SIGMA_REF, reference, DISK_PRESSURE_INDEX)
    assign_equilibrium(zones, star_mass, sound_speed)
    rng = np.random.Generator(np.random.PCG64(seed))
    swarms = []
    for index in range(swarm_count):
        radius = DISK_INNER+(DISK_OUTER-DISK_INNER)*(index+.5)/max(swarm_count, 1)
        angle = float(rng.random())*2*math.pi
        position = np.array([radius*math.cos(angle), radius*math.sin(angle)])
        swarms.append(_swarm(f'swarm-{index:03d}', f'Swarm {index}', ORDINARY_MASS, MILLIMETER, 0.,
                             ORDINARY_SPATIAL_RADIUS, position, _kepler_velocity(position, star_mass),
                             {f'swarm-{index:03d}': ORDINARY_MASS}))
    if include_ready:
        position = np.array([reference, 0.])
        swarms.append(_swarm(READY_SWARM_ID, 'Ready swarm', READY_MASS, GRAIN_RADIUS_MIN, READY_INTERNAL_SPEED,
                             READY_SPATIAL_RADIUS, position, _kepler_velocity(position, star_mass),
                             {READY_SWARM_ID: READY_MASS}))
    return DiskSetup('Gas-and-dust disk', seed, dt, star_mass, float(SOLAR_RADIUS_AU), parameters,
                     tuple(zones), tuple(swarms), ())


def make_uniform_disk(*, inner=1., outer=2., zones=8, sigma=1e-4, sound_speed=.2, dt=.01, star_mass=1.,
                      drag_enabled=False):
    """Flat surface density in equilibrium, with no solids. Useful for fluid checks."""
    built = _uniform_zones(zones, inner, outer, sigma)
    assign_equilibrium(built, star_mass, sound_speed)
    parameters = default_parameters(sound_speed=sound_speed, drag_enabled=drag_enabled)
    return DiskSetup('Uniform annular disk', 0, dt, star_mass, float(SOLAR_RADIUS_AU), parameters, tuple(built), (), ())


def make_capture_setup(*, drag=True):
    """Gaussian envelope and one hyperbolic intruder. `drag` is the only switch."""
    if type(drag) is not bool:
        raise ValueError('drag must be true or false.')
    star_mass = 1.
    zones = _gaussian_zones(CAPTURE_ZONES, CAPTURE_INNER, CAPTURE_OUTER, CAPTURE_SIGMA0, CAPTURE_SCALE)
    assign_equilibrium(zones, star_mass, CAPTURE_SOUND_SPEED)
    position = np.array(CAPTURE_POSITION, dtype=float)
    velocity = np.array(CAPTURE_VELOCITY, dtype=float)
    body = _body(CAPTURE_BODY_ID, 'Intruder', CAPTURE_MASS, CAPTURE_RADIUS, CAPTURE_RADIUS, 0., GRAIN_DENSITY,
                 position, velocity, {CAPTURE_BODY_ID: CAPTURE_MASS})
    parameters = default_parameters(sound_speed=CAPTURE_SOUND_SPEED, drag_enabled=drag)
    name = 'Envelope capture' if drag else 'Envelope capture, drag off'
    return DiskSetup(name, CAPTURE_SEED, CAPTURE_DT, star_mass, float(SOLAR_RADIUS_AU), parameters, tuple(zones),
                     (), (body,))


def setup_digest(setup):
    """Stable hash of a disk setup. The evolved state is not part of it."""
    if not isinstance(setup, DiskSetup):
        raise TypeError('Digest a disk setup.')
    payload = json.dumps(_setup_document(setup), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def specific_energy(star_mass, position, velocity):
    """Two-body specific energy about a star at the origin."""
    radius = float(np.linalg.norm(position))
    if radius <= 0:
        raise ValueError('Specific energy needs a positive star-centric distance.')
    speed2 = float(np.dot(velocity, velocity))
    return .5*speed2-G*float(star_mass)/radius


def pressure_accel(zones, index, sound_speed):
    """Radial pressure acceleration (1/Sigma) dP/dr for an isothermal zone."""
    pressures = [sound_speed**2*zone.sigma for zone in zones]
    radii = [zone.radius for zone in zones]
    if index == 0:
        slope = (pressures[1]-pressures[0])/(radii[1]-radii[0])
    elif index == len(zones)-1:
        slope = (pressures[-1]-pressures[-2])/(radii[-1]-radii[-2])
    else:
        slope = (pressures[index+1]-pressures[index-1])/(radii[index+1]-radii[index-1])
    return slope/zones[index].sigma


def equilibrium_v_phi(zones, index, star_mass, sound_speed):
    """Azimuthal speed that cancels gravity, centrifugal support, and pressure."""
    zone = zones[index]
    supported = G*star_mass/zone.radius+pressure_accel(zones, index, sound_speed)*zone.radius
    if not math.isfinite(supported) or supported <= 0:
        raise ValueError('This surface-density profile is not rotationally supported.')
    return math.sqrt(supported)


def assign_equilibrium(zones, star_mass, sound_speed):
    """Set each zone's rotation to the discrete radial balance and clear radial flow."""
    for index, zone in enumerate(zones):
        speed = equilibrium_v_phi(zones, index, star_mass, sound_speed)
        zone.angular_momentum = zone.mass*zone.radius*speed
        zone.radial_momentum = 0.


def sweep_up_increment(dust_density, relative_speed, grain_density, dt):
    """ds = rho_dust * relative_speed / (4 * grain_density) * dt."""
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in (dust_density, relative_speed, dt)):
        raise ValueError('Sweep-up inputs must be finite and nonnegative.')
    density = _positive('Grain density', grain_density)
    return float(dust_density)*float(relative_speed)/(4*density)*float(dt)


def evaluate_swarm_rule(swarm, zone, star_mass, parameters):
    """Return every failed gate. A missing zone means the swarm is decoupled, so Stokes passes."""
    radius = float(np.linalg.norm(swarm.r))
    if radius <= 0:
        raise ValueError('The swarm rule needs a positive star-centric distance.')
    omega = math.sqrt(G*star_mass/radius**3)
    if zone is None:
        stokes = math.inf
    else:
        stopping = _stopping_time(swarm.grain_radius, zone.sigma, zone.radius, star_mass, parameters)
        stokes = stopping*omega
    hill = radius*(swarm.mass/(3*star_mass))**(1/3)
    escape = math.sqrt(2*G*swarm.mass/swarm.spatial_radius)
    failed = []
    if not stokes >= parameters.stokes_min:
        failed.append('stokes')
    if not swarm.grain_radius >= parameters.grain_radius_min:
        failed.append('grain_radius')
    if not swarm.mass >= parameters.mass_min:
        failed.append('mass')
    if not swarm.spatial_radius <= hill:
        failed.append('hill')
    if not swarm.internal_speed <= escape:
        failed.append('internal')
    return RuleVerdict(not failed, tuple(failed), float(stokes), float(hill), float(escape))


class GasDustExperiment:
    """One gas-and-dust integration with a parcel ledger and recorded samples."""

    def __init__(self, setup, *, sample_stride=1):
        if not isinstance(setup, DiskSetup):
            raise TypeError('Gas-and-dust formation starts from a DiskSetup.')
        if type(sample_stride) is not int or isinstance(sample_stride, bool) or sample_stride < 1:
            raise ValueError('sample_stride must be a positive integer.')
        self.name = setup.name
        self.dt = _positive('Fixed dt', setup.dt)
        self.star_mass = _positive('Star mass', setup.star_mass)
        self.star_radius = _positive('Star radius', setup.star_radius)
        self.parameters = setup.parameters
        self.sample_stride = sample_stride
        self.seed = setup.seed
        self.zones = [replace(zone) for zone in setup.zones]
        self.swarms = [swarm.copy() for swarm in setup.swarms]
        self.bodies = [body.copy() for body in setup.bodies]
        self.removed_swarms = {}
        self.removed_bodies = {}
        self.events = []
        self.samples = []
        self.time = 0.
        self.status = 'running'
        self.reason = ''
        self.completed_steps = 0
        self.dissipated_kinetic = 0.
        self.drag_momentum_error = 0.
        self.transition_momentum_error = 0.
        self.minimum_radius = math.inf
        _validate_population(self)
        self.assumptions = _run_assumptions(self)
        self.provenance = _new_provenance(setup)
        self._record(None, force=True)
        self._note_radius()

    def describe(self):
        """Simulated time, status, and the model assumptions for this run."""
        status = f'Simulated time {self.time:.6g} yr. Status: {self.status}.'
        detail = f'Model {self.provenance["model"]}.'
        if self.reason:
            detail = f'{detail} {self.reason}'
        return '\n'.join((status, detail, *self.assumptions))

    def integrate(self, years, *, finish=False):
        """Advance by whole fixed steps until simulated time has grown by `years`."""
        if self.status != 'running':
            raise ValueError(f'Cannot integrate a {self.status} run. Branch from a checkpoint to calculate a new future.')
        years = _positive('Duration', years)
        target = self.time+years
        guard = 0
        limit = math.ceil(years/self.dt)+2
        while self.status == 'running' and self.time < target-1e-12*max(1., abs(target)):
            self.advance(1)
            guard += 1
            if guard > limit:
                raise ValueError('Disk numerical halt: the clock did not reach the requested time.')
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
            if self._stellar_contact():
                self._halt('Stellar contact halts the run. This model does not merge stars.')
                break
            limit = _cfl_dt(self.zones, self.parameters.sound_speed)
            if self.dt > COURANT*limit:
                self._halt(f'Fixed step {self.dt:.6g} yr exceeds the Courant limit {COURANT*limit:.6g} yr. '
                           'The step is not reduced.')
                break
            if not self._fluid(self.dt):
                self._halt('Fluid transport would empty a zone. The fixed step is too long for this radial flow, and the step is not reduced.')
                break
            self._grow(self.dt)
            self._coagulate()
            self._promote()
            if self.status != 'running':
                break
            self._kick(.5)
            self._drift()
            self._drag(self.dt)
            self._kick(.5)
            self.time += self.dt
            self.completed_steps += 1
            done += 1
            self._merge_contacts()
            self._note_radius()
            if self.status != 'running':
                break
            if self.completed_steps % self.sample_stride == 0:
                self._record(None)
        return done

    def zone_at(self, radius):
        """Return the annular zone containing `radius`, or None outside the fluid."""
        return find_zone(self.zones, _finite('Radius', radius))

    def replay_parcel(self, parcel_id):
        """Parcels, events, and samples that still carry this solid material."""
        if not isinstance(parcel_id, str) or not parcel_id.strip():
            raise ValueError('Parcel id must be a nonempty string.')
        parcel_id = parcel_id.strip()
        if not self._known_parcel(parcel_id):
            raise ValueError(f'Unknown parcel {parcel_id}.')
        holders = [swarm.id for swarm in self.swarms if parcel_id in swarm.contributions]
        holders.extend(body.id for body in self.bodies if parcel_id in body.contributions)
        events = tuple(event for event in self.events if parcel_id in _event_names(event))
        samples = tuple(sample for sample in self.samples if _sample_carries(sample, parcel_id))
        return ParcelStory(parcel_id, self._initial_parcel_mass(parcel_id), tuple(holders), events, samples)

    def accounting(self):
        """Mass, parcel, angular-momentum, drag, and transition discrepancies."""
        initial = self.samples[0]
        initial_gas = sum(zone.mass for zone in initial.zones)
        gas = sum(zone.mass for zone in self.zones)
        initial_solids = list(initial.swarms)+list(initial.bodies)
        leaves = [*self.swarms, *self.bodies]
        initial_solid = sum(item.mass for item in initial_solids)
        solid = sum(item.mass for item in leaves)
        parcel_error = _parcel_error(initial_solids, leaves)
        angular_error = abs(self._angular_momentum()-_sample_angular(initial))
        return DiskAccounting(initial_gas, gas, initial_solid, solid, parcel_error, angular_error,
                              self.drag_momentum_error, self.transition_momentum_error, self.dissipated_kinetic,
                              sum(event.kind == 'swarm-to-body' for event in self.events),
                              sum(event.kind == 'coagulation' for event in self.events),
                              sum(event.kind == 'merge' for event in self.events))

    def specific_energy(self, body_id):
        body = self._require_body(body_id)
        return specific_energy(self.star_mass, body.r, body.v)

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
        target = self._find_solid(change.body_id)
        if target is None:
            raise ValueError(f'{change.body_id} is not an active swarm or body in this checkpoint.')
        target.v = np.array(target.v, dtype=float)+np.array(change.delta_v, dtype=float)
        stored = {'kind': 'velocity', 'body_id': change.body_id, 'delta_v': change.delta_v, 'note': change.note}
        self.provenance = {**self.provenance, 'parent_checkpoint_id': parent_id, 'change': stored,
                           'run_id': _run_id(self.provenance['setup_digest'], self.seed, parent_id, stored)}
        self.status = 'running'
        self.reason = ''
        self._record('branch', force=True)

    def _grow(self, dt):
        grouped = {zone.id: [] for zone in self.zones}
        for swarm in self.swarms:
            zone = self.zone_at(float(np.linalg.norm(swarm.r)))
            if zone is not None:
                grouped[zone.id].append(swarm)
        for zone in self.zones:
            members = grouped[zone.id]
            if not members:
                continue
            dust = _dust_density(sum(swarm.mass for swarm in members), zone, self.star_mass, self.parameters)
            for swarm in members:
                gas = _gas_cartesian(zone, swarm.r)
                relative = max(swarm.internal_speed, float(np.linalg.norm(swarm.v-gas)))
                swarm.grain_radius += sweep_up_increment(dust, relative, self.parameters.grain_density, dt)

    def _coagulate(self):
        pair = _coagulation_pair(self.swarms)
        if pair is None:
            return
        left, right = pair
        mass = left.mass+right.mass
        velocity = (left.mass*np.asarray(left.v)+right.mass*np.asarray(right.v))/mass
        position = (left.mass*np.asarray(left.r)+right.mass*np.asarray(right.r))/mass
        kinetic = _internal_kinetic(left)+_internal_kinetic(right)+_relative_kinetic((left, right), velocity)
        internal = math.sqrt(2*kinetic/mass)
        spatial = (left.spatial_radius**3+right.spatial_radius**3)**(1/3)
        grain = (left.mass*left.grain_radius+right.mass*right.grain_radius)/mass
        contributions = _sum_maps(left.contributions, right.contributions)
        child_id = _derived_id('coag', (left.id, right.id), self._used_ids())
        child = Swarm(child_id, left.name+' + '+right.name, mass, grain, internal, spatial,
                      np.array(position, dtype=float), np.array(velocity, dtype=float), contributions)
        self._retire_swarm(left, 'coagulated')
        self._retire_swarm(right, 'coagulated')
        self.swarms.append(child)
        self._remember_transition('coagulation', (left, right), child, internal, spatial, 0., 0.,
                                  'Overlapping dust swarms coagulated. Relative kinetic energy became internal kinetic energy.')

    def _promote(self):
        ready = []
        for swarm in self.swarms:
            zone = self.zone_at(float(np.linalg.norm(swarm.r)))
            verdict = evaluate_swarm_rule(swarm, zone, self.star_mass, self.parameters)
            if verdict.accepted:
                ready.append((swarm, verdict))
        for swarm, verdict in ready:
            radius = _compact_radius(swarm.mass, self.parameters.bulk_density)
            body_id = _derived_id('body', (swarm.id,), self._used_ids())
            body = Body(body_id, swarm.name, swarm.mass, radius, swarm.grain_radius, swarm.internal_speed,
                        self.parameters.bulk_density, np.array(swarm.r, dtype=float, copy=True),
                        np.array(swarm.v, dtype=float, copy=True), dict(swarm.contributions))
            self._retire_swarm(swarm, 'became-body')
            self.bodies.append(body)
            gates = (f'Stokes {verdict.stokes:.6g} >= {self.parameters.stokes_min:.6g}',
                     f'grain radius {swarm.grain_radius:.6g} AU', f'mass {swarm.mass:.6g}',
                     f'spatial radius <= Hill radius {verdict.hill_radius:.6g} AU',
                     f'internal speed <= escape speed {verdict.escape_speed:.6g} AU/yr')
            self._remember_transition('swarm-to-body', (swarm,), body, swarm.internal_speed, radius, 0., 0.,
                                      'Swarm became one body ('+SWARM_RULE_ID+'): '+'; '.join(gates)+'.')

    def _fluid(self, dt):
        trial = [replace(zone) for zone in self.zones]
        _source(trial, dt, self.star_mass, self.parameters.sound_speed)
        if not _transport(trial, dt):
            return False
        self.zones = trial
        return True

    def _kick(self, fraction):
        accelerations = self._accelerations()
        for obj in (*self.swarms, *self.bodies):
            obj.v = np.array(obj.v, dtype=float)+fraction*self.dt*accelerations[obj.id]

    def _drift(self):
        for obj in (*self.swarms, *self.bodies):
            obj.r = np.array(obj.r, dtype=float)+self.dt*np.array(obj.v, dtype=float)

    def _drag(self, dt):
        if not self.parameters.drag_enabled:
            return
        for obj in (*self.swarms, *self.bodies):
            zone = self.zone_at(float(np.linalg.norm(obj.r)))
            if zone is None:
                continue
            gas = _gas_cartesian(zone, obj.r)
            before = np.array(obj.v, dtype=float)
            if isinstance(obj, Swarm):
                stopping = _stopping_time(obj.grain_radius, zone.sigma, zone.radius, self.star_mass, self.parameters)
                if not math.isfinite(stopping):
                    continue
                after = gas+(before-gas)*math.exp(-dt/stopping)
            else:
                after = _quadratic_drag(before, gas, zone, obj, self.star_mass, self.parameters, dt)
            delta_p = obj.mass*(after-before)
            r_hat, phi_hat, radius = _basis(obj.r)
            gas_radial = -float(np.dot(delta_p, r_hat))
            gas_angular = -radius*float(np.dot(delta_p, phi_hat))
            residual = abs(float(np.dot(delta_p, r_hat))+gas_radial)+abs(radius*float(np.dot(delta_p, phi_hat))+gas_angular)
            self.drag_momentum_error = max(self.drag_momentum_error, residual)
            kinetic_before = .5*obj.mass*float(np.dot(before, before))+_zone_kinetic(zone)
            zone.radial_momentum += gas_radial
            zone.angular_momentum += gas_angular
            obj.v = np.array(after, dtype=float)
            kinetic_after = .5*obj.mass*float(np.dot(after, after))+_zone_kinetic(zone)
            self.dissipated_kinetic += kinetic_before-kinetic_after

    def _merge_contacts(self):
        if self._stellar_contact():
            self._halt('Stellar contact halts the run. This model does not merge stars.')
            self._record(None, force=True)
            return
        pair = _overlapping_bodies(self.bodies)
        if pair is None:
            return
        self._record('pre', force=True)
        self._merge_bodies(*pair)
        self._record('post', force=True)

    def _merge_bodies(self, left, right):
        prepared = PreparedSystem(
            System(np.array([left.mass, right.mass], dtype=float), np.vstack((left.r, right.r)),
                   np.vstack((left.v, right.v)), self.dt, self.time),
            (left.id, right.id), (_body_spec(left), _body_spec(right)), None, (), 0, (None, None))
        merged, event = merge_pair(prepared, left.id, right.id)
        remnant_mass = float(merged.system.masses[-1])
        remnant_radius = float(merged.body_specs[-1].radius)
        position = np.array(merged.system.r[-1], dtype=float)
        velocity = np.array(merged.system.v[-1], dtype=float)
        internal_kinetic = _internal_kinetic(left)+_internal_kinetic(right)
        internal = math.sqrt(2*internal_kinetic/remnant_mass)
        grain = (left.mass*left.grain_radius+right.mass*right.grain_radius)/remnant_mass
        volume = 4/3*math.pi*remnant_radius**3
        body = Body(event.remnant_id, merged.body_specs[-1].name, remnant_mass, remnant_radius, grain, internal,
                    remnant_mass/volume, position, velocity, _sum_maps(left.contributions, right.contributions))
        self._retire_body(left, 'merged')
        self._retire_body(right, 'merged')
        self.bodies.append(body)
        self._remember_transition('merge', (left, right), body, internal, remnant_radius,
                                  float(event.lost_relative_kinetic), float(event.unmodeled_spin), event.message)

    def _accelerations(self):
        objects = [*self.swarms, *self.bodies]
        if not objects:
            return {}
        positions = np.stack([np.asarray(obj.r, dtype=float) for obj in objects])
        radii = np.linalg.norm(positions, axis=1)
        if np.any(radii == 0):
            self._halt('A solid reached the pinned star. The run stops before a singular acceleration.')
            return {obj.id: np.zeros(2) for obj in objects}
        acc = -G*self.star_mass*positions/radii[:, None]**3
        if self.bodies:
            body_positions = np.stack([np.asarray(body.r, dtype=float) for body in self.bodies])
            body_masses = np.array([body.mass for body in self.bodies])
            delta = body_positions[None, :, :]-positions[:, None, :]
            dist2 = np.sum(delta**2, axis=-1)
            for index, obj in enumerate(objects):
                for body_index, body in enumerate(self.bodies):
                    if obj is body:
                        dist2[index, body_index] = np.inf
            if np.any(dist2 == 0):
                self._halt('A dust swarm sits on a body. This model has no swarm-body collision.')
                return {obj.id: np.zeros(2) for obj in objects}
            acc = acc+G*np.sum(delta*(body_masses[None, :]/(dist2*np.sqrt(dist2)))[:, :, None], axis=1)
        if not np.isfinite(acc).all():
            self._halt('A non-finite acceleration stopped the run.')
            return {obj.id: np.zeros(2) for obj in objects}
        return {obj.id: acc[index] for index, obj in enumerate(objects)}

    def _stellar_contact(self):
        return any(float(np.linalg.norm(body.r)) <= self.star_radius+body.radius for body in self.bodies)

    def _halt(self, reason):
        self.status = 'halted'
        self.reason = reason

    def _note_radius(self):
        for obj in (*self.swarms, *self.bodies):
            self.minimum_radius = min(self.minimum_radius, float(np.linalg.norm(obj.r)))

    def _remember_transition(self, kind, parents, child, internal_speed, radius, lost, spin, message):
        parent_momentum = np.zeros(2)
        for parent in parents:
            parent_momentum += parent.mass*np.asarray(parent.v, dtype=float)
        child_momentum = child.mass*np.asarray(child.v, dtype=float)
        self.transition_momentum_error = max(self.transition_momentum_error,
                                             float(np.max(np.abs(child_momentum-parent_momentum))))
        contributions = _contribution_tuple(child.contributions)
        self.events.append(DiskEvent(
            self.time, kind, tuple(parent.id for parent in parents), child.id, (child.id,), contributions,
            float(child.mass), (float(child_momentum[0]), float(child_momentum[1])), float(child.grain_radius),
            float(internal_speed), .5*child.mass*internal_speed**2, float(radius), float(lost), float(spin), message))

    def _retire_swarm(self, swarm, status):
        self.removed_swarms[swarm.id] = _solid_view(swarm, 'swarm', status)
        self.swarms = [item for item in self.swarms if item is not swarm]

    def _retire_body(self, body, status):
        self.removed_bodies[body.id] = _solid_view(body, 'body', status)
        self.bodies = [item for item in self.bodies if item is not body]

    def _used_ids(self):
        used = {zone.id for zone in self.zones}
        used.update(swarm.id for swarm in self.swarms)
        used.update(body.id for body in self.bodies)
        used.update(self.removed_swarms)
        used.update(self.removed_bodies)
        return used

    def _find_solid(self, solid_id):
        for obj in (*self.swarms, *self.bodies):
            if obj.id == solid_id:
                return obj
        return None

    def _require_body(self, body_id):
        for body in self.bodies:
            if body.id == body_id:
                return body
        raise ValueError(f'{body_id} is not an active body.')

    def _angular_momentum(self):
        total = sum(zone.angular_momentum for zone in self.zones)
        for obj in (*self.swarms, *self.bodies):
            total += obj.mass*(obj.r[0]*obj.v[1]-obj.r[1]*obj.v[0])
        return float(total)

    def _known_parcel(self, parcel_id):
        return any(_carries(item, parcel_id) for item in (*self.swarms, *self.bodies, *self.removed_swarms.values(),
                                                          *self.removed_bodies.values()))

    def _initial_parcel_mass(self, parcel_id):
        total = 0.
        for item in (*self.samples[0].swarms, *self.samples[0].bodies):
            for contribution in item.contributions:
                if contribution.parcel_id == parcel_id:
                    total += contribution.mass
        return total

    def _record(self, phase, force=False):
        sample = DiskSample(self.time, phase, tuple(_zone_view(zone) for zone in self.zones),
                            tuple(_solid_view(swarm, 'swarm', 'active') for swarm in self.swarms),
                            tuple(_solid_view(body, 'body', 'active') for body in self.bodies))
        if not force and self.samples and _same_sample(self.samples[-1], sample):
            return
        self.samples.append(sample)

    def _payload(self):
        return {
            'schema': CHECKPOINT_SCHEMA, 'model': DISK_MODEL, 'name': self.name, 'time': _num(self.time),
            'dt': _num(self.dt), 'sample_stride': self.sample_stride, 'status': self.status, 'reason': self.reason,
            'completed_steps': self.completed_steps, 'assumptions': list(self.assumptions),
            'provenance': _provenance_document(self.provenance), 'parameters': _parameter_document(self.parameters),
            'star': {'mass': _num(self.star_mass), 'radius': _num(self.star_radius)},
            'zones': [_zone_document(zone) for zone in self.zones],
            'swarms': [_swarm_document(swarm, 'active') for swarm in self.swarms],
            'bodies': [_body_document(body, 'active') for body in self.bodies],
            'removed_swarms': [_swarm_document(view, view.status) for view in self.removed_swarms.values()],
            'removed_bodies': [_body_document(view, view.status) for view in self.removed_bodies.values()],
            'events': [_event_document(event) for event in self.events],
            'samples': [_sample_document(sample) for sample in self.samples],
            'dissipated_kinetic': _num(self.dissipated_kinetic),
            'drag_momentum_error': _num(self.drag_momentum_error),
            'transition_momentum_error': _num(self.transition_momentum_error),
        }


def continue_as_late_stage(experiment, *, ejection_radius):
    """Hand the star and current bodies to the late-stage prototype with their parcels."""
    if not isinstance(experiment, GasDustExperiment):
        raise TypeError('Continue a gas-and-dust experiment.')
    if experiment.status == 'halted':
        raise ValueError('A halted disk run has no late-stage continuation.')
    if not experiment.bodies:
        raise ValueError('Late-stage continuation needs a formed body.')
    star = InitialBody('star', BodySpec('Star', 'star', experiment.star_mass, experiment.star_radius,
                                        provenance='gas-dust pinned star'), (0., 0.), (0., 0.))
    bodies = [star]
    composition = {'star': {'star': experiment.star_mass}}
    for body in experiment.bodies:
        bodies.append(InitialBody(body.id, _body_spec(body), (float(body.r[0]), float(body.r[1])),
                                  (float(body.v[0]), float(body.v[1]))))
        composition[body.id] = dict(body.contributions)
    setup = InitialSetup(experiment.name+' late stage', tuple(bodies), experiment.dt, collision_mode='perfect_merge',
                        provenance={'formation_model': 'late-stage-perfect-merge-v1', 'scenario': 'gas-dust-handoff',
                                    'parent_run_id': experiment.provenance['run_id']})
    return FormationExperiment(setup, ejection_radius=ejection_radius, seed=experiment.seed, composition=composition)


def save_checkpoint(experiment, path):
    """Validate the checkpoint, then atomically replace `path`."""
    if not isinstance(experiment, GasDustExperiment):
        raise TypeError('Save a gas-and-dust experiment.')
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
    """Load a gas-and-dust checkpoint into an experiment that can integrate forward."""
    try:
        with Path(path).open(encoding='utf-8') as handle:
            document = json.load(handle, object_pairs_hook=_pairs,
                                 parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON number: '+value)))
    except (json.JSONDecodeError, TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError(f'Invalid gas-and-dust checkpoint: {exc}') from exc
    return _experiment_from_document(document)


def run_capture_comparison():
    """Integrate the envelope flyby with drag and with drag disabled."""
    drag = GasDustExperiment(make_capture_setup(drag=True))
    free = GasDustExperiment(make_capture_setup(drag=False))
    drag_initial = drag.specific_energy(CAPTURE_BODY_ID)
    free_initial = free.specific_energy(CAPTURE_BODY_ID)
    drag.integrate(CAPTURE_YEARS, finish=True)
    free.integrate(CAPTURE_YEARS, finish=True)
    drag_energy = drag.specific_energy(CAPTURE_BODY_ID) if any(body.id == CAPTURE_BODY_ID for body in drag.bodies) else math.nan
    free_energy = free.specific_energy(CAPTURE_BODY_ID) if any(body.id == CAPTURE_BODY_ID for body in free.bodies) else math.nan
    notes = [
        'The envelope is a Gaussian surface-density fluid. Its gravity is not applied, so an energy change comes from drag and from the pinned star.',
        f'The intruder drag radius is {CAPTURE_RADIUS:.3g} AU. A radius this large is a demonstration cross-section, not a rocky surface.',
        'The drag-free run is the same initial state with drag_enabled false. Binding in the drag run is aerodynamic and planar. It is not a permanent capture.',
        'Fragmenting collisions remain #69, three-dimensional orbits and an adaptive or close-encounter integrator remain #70, and tides remain #79.',
    ]
    return CaptureReport(DISK_MODEL, DRAG_MODEL_ID, CAPTURE_YEARS, CAPTURE_DT, CAPTURE_MASS, CAPTURE_RADIUS,
                         drag_energy, free_energy, drag_initial, free_initial, bool(drag_energy < 0),
                         bool(free_energy < 0), drag.dissipated_kinetic, free.dissipated_kinetic,
                         drag.minimum_radius, free.minimum_radius, drag.status, free.status, tuple(notes))


def run_disk_experiment():
    """Time the seeded disk and the envelope comparison.

    Peak memory is Python's tracemalloc peak for both runs, not the operating-system
    resident set. Recording size is the disk checkpoint document in bytes.
    """
    tracing = not tracemalloc.is_tracing()
    if tracing:
        tracemalloc.start()
    clock = time.perf_counter()
    try:
        experiment = GasDustExperiment(make_disk_setup(), sample_stride=1)
        experiment.integrate(DISK_DURATION_YEARS, finish=True)
        capture = run_capture_comparison()
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
    books = experiment.accounting()
    notes = [
        'Gas zones are piecewise-constant annuli. Dust swarms are single-size populations with one parcel each, except where coagulation has combined them.',
        'One seeded swarm starts past the Hill, Stokes, mass, grain-size, and binding gates so the run contains a known swarm-to-body transition.',
        'Ordinary grains start at 1 mm. This run does not carry them across the 1 km gate.',
        'The fixed step is not adapted. A Courant violation halts.',
        'Wall time includes the seeded disk and both envelope flybys. It does not include writing the checkpoint.',
        f'Gas mass closed to {abs(books.gas_mass-books.initial_gas_mass):.3g}. Parcel error {books.parcel_error:.3g}. '
        f'Angular-momentum error {books.angular_momentum_error:.3g}.',
    ]
    if experiment.status != 'complete':
        notes.append('The disk run stopped before the requested time: '+experiment.reason)
    if capture.drag_bound and not capture.free_bound:
        notes.append('The envelope flyby became bound with drag and stayed unbound without it.')
    else:
        notes.append('The envelope comparison did not produce the disclosed bound-versus-unbound split.')
    parcel_count = sum(len(swarm.contributions) for swarm in make_disk_setup().swarms)
    return DiskReport(hardware_description(), DISK_MODEL, REPRESENTATION_ID, SWARM_RULE_ID, DISK_SEED,
                      DISK_ZONE_COUNT, DISK_SWARM_COUNT+1, parcel_count, len(experiment.bodies),
                      DISK_DURATION_YEARS, experiment.time, seconds, peak, len(recording.encode('utf-8')),
                      len(experiment.samples), books.promotion_count, books.coagulation_count, experiment.status,
                      _QUESTIONS, tuple(notes), capture)


def _power_law_zones(count, inner, outer, sigma_ref, reference, index):
    edges = np.linspace(inner, outer, count+1)
    zones = []
    for number, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        radius = math.sqrt(.5*(float(left)**2+float(right)**2))
        sigma = sigma_ref*(radius/reference)**(-index)
        area = math.pi*(float(right)**2-float(left)**2)
        zones.append(Zone(f'gas-{number:03d}', float(left), float(right), sigma*area, 0., 0.))
    return zones


def _uniform_zones(count, inner, outer, sigma):
    edges = np.linspace(inner, outer, count+1)
    zones = []
    for number, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        area = math.pi*(float(right)**2-float(left)**2)
        zones.append(Zone(f'gas-{number:03d}', float(left), float(right), sigma*area, 0., 0.))
    return zones


def _gaussian_zones(count, inner, outer, sigma0, scale):
    edges = np.linspace(inner, outer, count+1)
    zones = []
    for number, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        radius = math.sqrt(.5*(float(left)**2+float(right)**2))
        sigma = sigma0*math.exp(-(radius/scale)**2)
        area = math.pi*(float(right)**2-float(left)**2)
        zones.append(Zone(f'gas-{number:03d}', float(left), float(right), sigma*area, 0., 0.))
    return zones


def _swarm(swarm_id, name, mass, grain_radius, internal_speed, spatial_radius, position, velocity, contributions):
    return Swarm(swarm_id, name, mass, grain_radius, internal_speed, spatial_radius,
                 np.array(position, dtype=float), np.array(velocity, dtype=float), dict(contributions))


def _body(body_id, name, mass, radius, grain_radius, internal_speed, bulk_density, position, velocity, contributions):
    return Body(body_id, name, mass, radius, grain_radius, internal_speed, bulk_density,
                np.array(position, dtype=float), np.array(velocity, dtype=float), dict(contributions))


def _kepler_velocity(position, star_mass):
    radius = float(np.linalg.norm(position))
    speed = math.sqrt(G*star_mass/radius)
    tangent = np.array([-position[1], position[0]], dtype=float)/radius
    return tangent*speed


def _source(zones, dt, star_mass, sound_speed):
    accels = [pressure_accel(zones, index, sound_speed) for index in range(len(zones))]
    for zone, accel in zip(zones, accels):
        radial = zone.v_phi**2/zone.radius-G*star_mass/zone.radius**2-accel
        zone.radial_momentum = zone.mass*(zone.v_r+dt*radial)


def _transport(zones, dt):
    fluxes = []
    for index in range(len(zones)-1):
        left, right = zones[index], zones[index+1]
        face_speed = .5*(left.v_r+right.v_r)
        if face_speed == 0:
            fluxes.append((0., 0., 0.))
            continue
        donor = left if face_speed > 0 else right
        mass_flux = 2*math.pi*left.r_outer*donor.sigma*face_speed
        fluxes.append((mass_flux, mass_flux*donor.v_r, mass_flux*donor.specific_angular))
    changes = [[0., 0., 0.] for _ in zones]
    for index, (mass_flux, radial_flux, angular_flux) in enumerate(fluxes):
        for slot, flux in enumerate((mass_flux, radial_flux, angular_flux)):
            changes[index][slot] -= flux*dt
            changes[index+1][slot] += flux*dt
    if any(zone.mass+change[0] <= 0 for zone, change in zip(zones, changes)):
        return False
    for zone, change in zip(zones, changes):
        zone.mass += change[0]
        zone.radial_momentum += change[1]
        zone.angular_momentum += change[2]
    return True


def _cfl_dt(zones, sound_speed):
    limit = math.inf
    for zone in zones:
        width = zone.r_outer-zone.r_inner
        limit = min(limit, width/(abs(zone.v_r)+sound_speed))
    return limit


def _stopping_time(grain_radius, sigma, radius, star_mass, parameters):
    density = _midplane_density(sigma, radius, star_mass, parameters.sound_speed)
    if density <= 0:
        return math.inf
    return parameters.grain_density*grain_radius/(density*parameters.sound_speed)


def _midplane_density(sigma, radius, star_mass, sound_speed):
    omega = math.sqrt(G*star_mass/radius**3)
    scale_height = sound_speed/omega
    return sigma/(math.sqrt(2*math.pi)*scale_height)


def _dust_density(mass, zone, star_mass, parameters):
    omega = math.sqrt(G*star_mass/zone.radius**3)
    scale_height = parameters.sound_speed/omega
    return mass/(zone.area*math.sqrt(2*math.pi)*scale_height)


def quadratic_velocity_update(velocity, gas, density, radius, mass, coefficient, dt):
    """Implicit quadratic drag. The speed law is the one ``_quadratic_drag`` uses."""
    if density <= 0 or radius <= 0 or mass <= 0 or coefficient <= 0:
        return np.array(velocity, dtype=float)
    area = math.pi*float(radius)**2
    drag_coefficient = float(coefficient)*float(density)*area/(2*float(mass))
    relative = np.array(velocity, dtype=float)-np.asarray(gas, dtype=float)
    speed = float(np.linalg.norm(relative))
    if speed == 0 or drag_coefficient == 0:
        return np.array(velocity, dtype=float)
    slowed = speed/(1+drag_coefficient*speed*dt)
    return np.asarray(gas, dtype=float)+relative*(slowed/speed)


def _quadratic_drag(velocity, gas, zone, body, star_mass, parameters, dt):
    density = _midplane_density(zone.sigma, zone.radius, star_mass, parameters.sound_speed)
    return quadratic_velocity_update(velocity, gas, density, body.radius, body.mass, parameters.drag_coefficient, dt)


def find_zone(zones, radius):
    """Annular zone containing ``radius``, or None outside the fluid."""
    value = float(radius)
    if not zones or not math.isfinite(value):
        return None
    for zone in zones:
        if zone.r_inner <= value < zone.r_outer or (zone is zones[-1] and value == zone.r_outer):
            return zone
    return None


def vertical_density(zone, height, star_mass, sound_speed):
    """Midplane density times a Gaussian in height. Zero height is the midplane value."""
    midplane = _midplane_density(zone.sigma, zone.radius, star_mass, sound_speed)
    if height == 0 or midplane <= 0:
        return midplane
    omega = math.sqrt(G*star_mass/zone.radius**3)
    scale = sound_speed/omega
    if scale <= 0 or not math.isfinite(scale):
        return midplane
    return midplane*math.exp(-.5*(float(height)/scale)**2)


def cylindrical_geometry(relative):
    """Cylindrical radius, height, and in-plane basis of a star-centric offset.

    A two-component offset has zero height. The basis matches ``_basis`` in the plane.
    """
    vector = np.asarray(relative, dtype=float)
    if vector.shape == (2,):
        radius = float(np.linalg.norm(vector))
        height = 0.
        if radius <= 0:
            return radius, height, None, None
        r_hat = vector/radius
        phi_hat = np.array([-r_hat[1], r_hat[0]])
        return radius, height, r_hat, phi_hat
    radius = float(math.hypot(float(vector[0]), float(vector[1])))
    height = float(vector[2])
    if radius <= 0:
        return radius, height, None, None
    r_hat = np.array([vector[0]/radius, vector[1]/radius, 0.])
    phi_hat = np.array([-vector[1]/radius, vector[0]/radius, 0.])
    return radius, height, r_hat, phi_hat


def advance_annular_fluid(zones, dt, star_mass, sound_speed):
    """Subcycle the existing source and transport steps inside ``dt``.

    Returns False when a piece would empty a zone or the Courant limit is not
    usable. Zones already updated stay updated. The caller does not halt.
    """
    remaining = float(dt)
    guard = 0
    while remaining > 1e-15*max(1., abs(float(dt))):
        guard += 1
        if guard > 100000:
            return False
        limit = _cfl_dt(zones, sound_speed)
        step = min(remaining, COURANT*limit)
        if not math.isfinite(step) or step <= 0 or not math.isfinite(limit):
            return False
        trial = [replace(zone) for zone in zones]
        _source(trial, step, star_mass, sound_speed)
        if not _transport(trial, step):
            return False
        for zone, updated in zip(zones, trial):
            zone.mass = updated.mass
            zone.radial_momentum = updated.radial_momentum
            zone.angular_momentum = updated.angular_momentum
        remaining -= step
    return True


def apply_body_drag(zones, parameters, star_mass, star_position, star_velocity, position, velocity, radius, mass, dt):
    """Quadratic drag on one body about the current star, using the mid-plane disk.

    Height enters only as the Gaussian weight on the existing midplane density.
    Gas has no vertical velocity. Vertical momentum taken from the body is not
    stored on a zone. Returns ``(velocity, dissipated, momentum_error)``.
    """
    relative_r = np.asarray(position, dtype=float)-np.asarray(star_position, dtype=float)
    relative_v = np.asarray(velocity, dtype=float)-np.asarray(star_velocity, dtype=float)
    cyl, height, r_hat, phi_hat = cylindrical_geometry(relative_r)
    if r_hat is None:
        return np.array(velocity, dtype=float), 0., 0.
    zone = find_zone(zones, cyl)
    if zone is None or not parameters.drag_enabled:
        return np.array(velocity, dtype=float), 0., 0.
    density = vertical_density(zone, height, star_mass, parameters.sound_speed)
    gas = zone.v_r*r_hat+zone.v_phi*phi_hat
    before = relative_v
    after = quadratic_velocity_update(before, gas, density, radius, mass, parameters.drag_coefficient, dt)
    delta_p = float(mass)*(after-before)
    gas_radial = -float(np.dot(delta_p, r_hat))
    gas_angular = -cyl*float(np.dot(delta_p, phi_hat))
    residual = abs(float(np.dot(delta_p, r_hat))+gas_radial)+abs(cyl*float(np.dot(delta_p, phi_hat))+gas_angular)
    kinetic_before = .5*float(mass)*float(np.dot(before, before))+_zone_kinetic(zone)
    zone.radial_momentum += gas_radial
    zone.angular_momentum += gas_angular
    kinetic_after = .5*float(mass)*float(np.dot(after, after))+_zone_kinetic(zone)
    inertial = np.asarray(star_velocity, dtype=float)+after
    return inertial, kinetic_before-kinetic_after, residual


GIANT_DISK_INNER = 4.2
GIANT_DISK_OUTER = 6.4
GIANT_DISK_SIGMA = 0.02
GIANT_DISK_SOUND = 0.35
GIANT_DISK_ZONES = 16


def make_giant_orbit_disk():
    """Uniform annular disk on a giant planet's orbit. Same isothermal method."""
    zones = _uniform_zones(GIANT_DISK_ZONES, GIANT_DISK_INNER, GIANT_DISK_OUTER, GIANT_DISK_SIGMA)
    assign_equilibrium(zones, 1., GIANT_DISK_SOUND)
    parameters = default_parameters(sound_speed=GIANT_DISK_SOUND, drag_enabled=True)
    return zones, parameters, 1.


def disk_for_policy(policy):
    """Zones, parameters, and star mass for one stored drag disk. Zones are copies."""
    if policy.disk == 'annular':
        setup = make_disk_setup(swarm_count=0, include_ready=False)
    elif policy.disk == 'envelope':
        setup = make_capture_setup(drag=True)
    elif policy.disk == 'giant':
        zones, parameters, star_mass = make_giant_orbit_disk()
        return [replace(zone) for zone in zones], parameters, star_mass
    else:
        raise ValueError("Drag disk must be 'annular', 'envelope', or 'giant'.")
    return [replace(zone) for zone in setup.zones], setup.parameters, setup.star_mass


def _gas_cartesian(zone, position):
    r_hat, phi_hat, _radius = _basis(position)
    return zone.v_r*r_hat+zone.v_phi*phi_hat


def _basis(position):
    vector = np.asarray(position, dtype=float)
    radius = float(np.linalg.norm(vector))
    if radius <= 0:
        raise ValueError('Drag needs a positive star-centric distance.')
    r_hat = vector/radius
    phi_hat = np.array([-r_hat[1], r_hat[0]])
    return r_hat, phi_hat, radius


def _zone_kinetic(zone):
    return .5*zone.mass*(zone.v_r**2+zone.v_phi**2)


def _compact_radius(mass, bulk_density):
    volume = mass/bulk_density
    return (3*volume/(4*math.pi))**(1/3)


def _coagulation_pair(swarms):
    ordered = sorted(swarms, key=lambda swarm: swarm.id)
    for index, left in enumerate(ordered):
        for right in ordered[index+1:]:
            if _can_coagulate(left, right):
                return left, right
    return None


def _can_coagulate(left, right):
    separation = float(np.linalg.norm(np.asarray(left.r)-np.asarray(right.r)))
    reach = left.spatial_radius+right.spatial_radius
    if separation > reach:
        return False
    distance = max(separation, reach)
    escape = math.sqrt(2*G*(left.mass+right.mass)/distance)
    relative = float(np.linalg.norm(np.asarray(left.v)-np.asarray(right.v)))
    if relative >= escape:
        return False
    velocity = (left.mass*np.asarray(left.v)+right.mass*np.asarray(right.v))/(left.mass+right.mass)
    kinetic = _internal_kinetic(left)+_internal_kinetic(right)+_relative_kinetic((left, right), velocity)
    mass = left.mass+right.mass
    internal = math.sqrt(2*kinetic/mass)
    spatial = (left.spatial_radius**3+right.spatial_radius**3)**(1/3)
    return internal <= math.sqrt(2*G*mass/spatial)


def _overlapping_bodies(bodies):
    ordered = sorted(bodies, key=lambda body: body.id)
    for index, left in enumerate(ordered):
        for right in ordered[index+1:]:
            separation = float(np.linalg.norm(np.asarray(left.r)-np.asarray(right.r)))
            if separation <= left.radius+right.radius:
                return left, right
    return None


def _internal_kinetic(obj):
    return .5*obj.mass*obj.internal_speed**2


def _relative_kinetic(objects, velocity):
    total = 0.
    for obj in objects:
        delta = np.asarray(obj.v, dtype=float)-np.asarray(velocity, dtype=float)
        total += .5*obj.mass*float(np.dot(delta, delta))
    return total


def _sum_maps(*maps):
    total = {}
    for mapping in maps:
        for key, mass in mapping.items():
            total[key] = total.get(key, 0.)+float(mass)
    return total


def _derived_id(prefix, parents, used):
    digest = hashlib.sha256(('\0'.join(sorted(parents))).encode()).hexdigest()[:16]
    key = f'{prefix}-{digest}'
    ordinal = 0
    while key in used:
        ordinal += 1
        key = f'{prefix}-{digest}-{ordinal}'
    return key


def _body_spec(body):
    internal_kinetic = .5*body.mass*body.internal_speed**2
    provenance = (f'gas-dust {SWARM_RULE_ID}\n'
                  f'grain_radius_au={body.grain_radius:.16e}\n'
                  f'internal_speed_au_per_yr={body.internal_speed:.16e}\n'
                  f'internal_kinetic={internal_kinetic:.16e}\n'
                  f'bulk_density_solar_per_au3={body.bulk_density:.16e}')
    return BodySpec(body.name, 'planet', body.mass, body.radius, color='#c9b28d', provenance=provenance)


def _validate_population(experiment):
    if len(experiment.zones) < 2:
        raise ValueError('The annular fluid needs at least two zones.')
    previous = None
    seen = set()
    for zone in experiment.zones:
        if not isinstance(zone.id, str) or not zone.id.strip() or zone.id in seen:
            raise ValueError('Zone ids must be unique nonempty strings.')
        if not math.isfinite(zone.r_inner) or not math.isfinite(zone.r_outer) or zone.r_inner >= zone.r_outer:
            raise ValueError('Each zone needs r_inner < r_outer.')
        if zone.r_inner <= experiment.star_radius:
            raise ValueError('The fluid starts outside the star.')
        if previous is not None and abs(zone.r_inner-previous) > 1e-9*max(1., abs(previous)):
            raise ValueError('Zones must meet without gaps or overlaps.')
        if not math.isfinite(zone.mass) or zone.mass <= 0:
            raise ValueError('Each zone needs positive mass.')
        if not math.isfinite(zone.radial_momentum) or not math.isfinite(zone.angular_momentum):
            raise ValueError('Zone momenta must be finite.')
        previous = zone.r_outer
        seen.add(zone.id)
    for swarm in experiment.swarms:
        _validate_solid_identity(swarm.id, seen, 'Swarm')
        _validate_vectors(swarm)
        if swarm.mass <= 0 or swarm.grain_radius <= 0 or swarm.spatial_radius <= 0 or swarm.internal_speed < 0:
            raise ValueError(f'{swarm.id} needs positive mass, grain radius, and spatial radius, and a nonnegative internal speed.')
        _validate_contributions(swarm.id, swarm.mass, swarm.contributions)
        seen.add(swarm.id)
    for body in experiment.bodies:
        _validate_solid_identity(body.id, seen, 'Body')
        _validate_vectors(body)
        if body.mass <= 0 or body.radius <= 0 or body.grain_radius <= 0 or body.bulk_density <= 0 or body.internal_speed < 0:
            raise ValueError(f'{body.id} needs positive mass, radius, grain radius, and bulk density.')
        _validate_contributions(body.id, body.mass, body.contributions)
        seen.add(body.id)


def _validate_solid_identity(identity, seen, label):
    if not isinstance(identity, str) or not identity.strip() or identity in seen:
        raise ValueError(f'{label} ids must be unique nonempty strings.')


def _validate_vectors(obj):
    for vector in (obj.r, obj.v):
        array = np.asarray(vector, dtype=float)
        if array.shape != (2,) or not np.isfinite(array).all():
            raise ValueError(f'{obj.id} needs finite planar position and velocity.')


def _validate_contributions(identity, mass, contributions):
    if not isinstance(contributions, Mapping) or not contributions:
        raise ValueError(f'{identity} needs at least one parcel.')
    total = 0.
    for parcel_id, parcel_mass in contributions.items():
        if not isinstance(parcel_id, str) or not parcel_id.strip():
            raise ValueError('Parcel ids must be nonempty strings.')
        if not isinstance(parcel_mass, (int, float)) or isinstance(parcel_mass, bool) or not math.isfinite(parcel_mass) or parcel_mass <= 0:
            raise ValueError('Parcel masses must be finite and positive.')
        total += float(parcel_mass)
    if abs(total-mass) > 1e-9*max(1., abs(mass)):
        raise ValueError(f'Parcels for {identity} must add up to its mass.')


def _run_assumptions(experiment):
    lines = list(_ASSUMPTIONS)
    lines.append(f'Sound speed is {experiment.parameters.sound_speed:.6g} AU/yr. Drag is '
                 f'{"on" if experiment.parameters.drag_enabled else "off"}. '
                 f'The Courant factor is {COURANT:.2g}, and a crossing halts instead of shrinking the step.')
    if any(body.radius > .001 for body in experiment.bodies):
        lines.append('At least one body radius exceeds 0.001 AU. Treat that radius as an inflated drag cross-section, not a rocky surface.')
    return tuple(lines)


def _new_provenance(setup):
    digest = setup_digest(setup)
    provenance = {'run_id': '', 'model': DISK_MODEL, 'representation': REPRESENTATION_ID, 'drag_model': DRAG_MODEL_ID,
                  'growth_model': GROWTH_MODEL_ID, 'swarm_rule': SWARM_RULE_ID, 'collision_model': COLLISION_MODEL_ID,
                  'seed': setup.seed, 'setup_digest': digest, 'parent_checkpoint_id': None, 'change': None,
                  'execution_environment': {'numpy': np.__version__}, 'solver_build': SOLVER_BUILD}
    provenance['run_id'] = _run_id(digest, setup.seed, None, None)
    return provenance


def _run_id(digest, seed, parent_id, change):
    payload = {'setup_digest': digest, 'seed': seed, 'parent_checkpoint_id': parent_id,
               'change': None if change is None else _change_document(change)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:16]


def _setup_document(setup):
    return {'name': setup.name, 'seed': setup.seed, 'dt': _num(setup.dt), 'star_mass': _num(setup.star_mass),
            'star_radius': _num(setup.star_radius), 'parameters': _parameter_document(setup.parameters),
            'zones': [_zone_document(zone) for zone in setup.zones],
            'swarms': [_swarm_document(swarm, 'active') for swarm in setup.swarms],
            'bodies': [_body_document(body, 'active') for body in setup.bodies]}


def _parameter_document(parameters):
    return {'sound_speed': _num(parameters.sound_speed), 'grain_density': _num(parameters.grain_density),
            'bulk_density': _num(parameters.bulk_density), 'stokes_min': _num(parameters.stokes_min),
            'grain_radius_min': _num(parameters.grain_radius_min), 'mass_min': _num(parameters.mass_min),
            'drag_coefficient': _num(parameters.drag_coefficient), 'drag_enabled': parameters.drag_enabled}


def _zone_document(zone):
    return {'id': zone.id, 'r_inner': _num(zone.r_inner), 'r_outer': _num(zone.r_outer), 'mass': _num(zone.mass),
            'radial_momentum': _num(zone.radial_momentum), 'angular_momentum': _num(zone.angular_momentum)}


def _contribution_document(contributions):
    return [{'parcel_id': key, 'mass': _num(contributions[key])} for key in sorted(contributions)]


def _swarm_document(swarm, status):
    contributions = swarm.contributions if isinstance(swarm.contributions, Mapping) else {
        item.parcel_id: item.mass for item in swarm.contributions}
    return {'id': swarm.id, 'name': swarm.name, 'mass': _num(swarm.mass), 'grain_radius': _num(swarm.grain_radius),
            'internal_speed': _num(swarm.internal_speed), 'spatial_radius': _num(swarm.spatial_radius),
            'r': _vector_list(swarm.r), 'v': _vector_list(swarm.v), 'contributions': _contribution_document(contributions),
            'status': status}


def _body_document(body, status):
    contributions = body.contributions if isinstance(body.contributions, Mapping) else {
        item.parcel_id: item.mass for item in body.contributions}
    return {'id': body.id, 'name': body.name, 'mass': _num(body.mass), 'radius': _num(body.radius),
            'grain_radius': _num(body.grain_radius), 'internal_speed': _num(body.internal_speed),
            'bulk_density': _num(body.bulk_density), 'r': _vector_list(body.r), 'v': _vector_list(body.v),
            'contributions': _contribution_document(contributions), 'status': status}


def _event_document(event):
    return {'time': _num(event.time), 'kind': event.kind, 'parent_ids': list(event.parent_ids),
            'remnant_id': event.remnant_id, 'descendant_ids': list(event.descendant_ids),
            'contributions': [{'parcel_id': item.parcel_id, 'mass': _num(item.mass)} for item in event.contributions],
            'mass': _num(event.mass), 'momentum': [_num(event.momentum[0]), _num(event.momentum[1])],
            'grain_radius': _num(event.grain_radius), 'internal_speed': _num(event.internal_speed),
            'internal_kinetic': _num(event.internal_kinetic), 'radius': _num(event.radius),
            'lost_relative_kinetic': _num(event.lost_relative_kinetic), 'unmodeled_spin': _num(event.unmodeled_spin),
            'message': event.message}


def _sample_document(sample):
    return {'time': _num(sample.time), 'phase': sample.phase,
            'zones': [_zone_document(zone) for zone in sample.zones],
            'swarms': [_swarm_document(swarm, swarm.status) for swarm in sample.swarms],
            'bodies': [_body_document(body, body.status) for body in sample.bodies]}


def _provenance_document(provenance):
    change = provenance['change']
    return {'run_id': provenance['run_id'], 'model': provenance['model'], 'representation': provenance['representation'],
            'drag_model': provenance['drag_model'], 'growth_model': provenance['growth_model'],
            'swarm_rule': provenance['swarm_rule'], 'collision_model': provenance['collision_model'],
            'seed': provenance['seed'], 'setup_digest': provenance['setup_digest'],
            'parent_checkpoint_id': provenance['parent_checkpoint_id'],
            'change': None if change is None else _change_document(change),
            'execution_environment': {'numpy': provenance['execution_environment']['numpy']},
            'solver_build': provenance['solver_build']}


def _change_document(change):
    return {'kind': 'velocity', 'body_id': change['body_id'],
            'delta_v': [_num(component) for component in change['delta_v']], 'note': change['note']}


def _experiment_from_document(document):
    if type(document) is not dict:
        raise ValueError('Gas-and-dust checkpoint must be a JSON object.')
    if document.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError(f'Unsupported gas-and-dust checkpoint schema; expected {CHECKPOINT_SCHEMA}.')
    if document.get('model') != DISK_MODEL:
        raise ValueError(f'Unsupported gas-and-dust model; expected {DISK_MODEL}.')
    _object(document, _DOCUMENT_KEYS, 'Gas-and-dust checkpoint')
    payload = {key: document[key] for key in document if key != 'id'}
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    identity = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    if document['id'] != identity:
        raise ValueError('Checkpoint id does not match its contents.')
    name = _string(document['name'], 'Experiment name')
    time_value = _finite('Time', document['time'])
    dt = _positive('Fixed dt', document['dt'])
    if type(document['sample_stride']) is not int or isinstance(document['sample_stride'], bool) or document['sample_stride'] < 1:
        raise ValueError('sample_stride must be a positive integer.')
    if document['status'] not in ('running', 'complete', 'halted'):
        raise ValueError('Unsupported gas-and-dust status.')
    if type(document['reason']) is not str:
        raise ValueError('Gas-and-dust reason must be a string.')
    if type(document['completed_steps']) is not int or isinstance(document['completed_steps'], bool) or document['completed_steps'] < 0:
        raise ValueError('completed_steps must be a nonnegative integer.')
    assumptions = tuple(_string(item, 'Assumption') for item in _list(document['assumptions'], 'Assumptions'))
    provenance = _provenance_from_document(document['provenance'])
    parameters = _parameters_from_document(document['parameters'])
    star = _object(document['star'], _STAR_KEYS, 'Star')
    star_mass = _positive('Star mass', star['mass'])
    star_radius = _positive('Star radius', star['radius'])
    zones = [_zone_from_document(item) for item in _list(document['zones'], 'Zones')]
    swarms = [_swarm_from_document(item, active=True) for item in _list(document['swarms'], 'Swarms')]
    bodies = [_body_from_document(item, active=True) for item in _list(document['bodies'], 'Bodies')]
    removed_swarms = [_swarm_from_document(item, active=False) for item in _list(document['removed_swarms'], 'Removed swarms')]
    removed_bodies = [_body_from_document(item, active=False) for item in _list(document['removed_bodies'], 'Removed bodies')]
    events = tuple(_event_from_document(item) for item in _list(document['events'], 'Events'))
    samples = tuple(_sample_from_document(item) for item in _list(document['samples'], 'Samples'))
    if not samples or any(b.time < a.time-1e-12 for a, b in zip(samples, samples[1:])):
        raise ValueError('Checkpoint samples must be nonempty and nondecreasing in time.')
    if samples[-1].time > time_value+1e-9*max(1., abs(time_value)):
        raise ValueError('A sample is later than the checkpoint time.')
    experiment = GasDustExperiment.__new__(GasDustExperiment)
    experiment.name = name
    experiment.dt = dt
    experiment.star_mass = star_mass
    experiment.star_radius = star_radius
    experiment.parameters = parameters
    experiment.sample_stride = document['sample_stride']
    experiment.seed = provenance['seed']
    experiment.zones = zones
    experiment.swarms = swarms
    experiment.bodies = bodies
    experiment.removed_swarms = {item.id: item for item in removed_swarms}
    experiment.removed_bodies = {item.id: item for item in removed_bodies}
    experiment.events = list(events)
    experiment.samples = list(samples)
    experiment.time = time_value
    experiment.status = document['status']
    experiment.reason = document['reason']
    experiment.completed_steps = document['completed_steps']
    experiment.dissipated_kinetic = _finite('Dissipated kinetic energy', document['dissipated_kinetic'])
    experiment.drag_momentum_error = _finite('Drag momentum error', document['drag_momentum_error'])
    experiment.transition_momentum_error = _finite('Transition momentum error', document['transition_momentum_error'])
    experiment.minimum_radius = min((float(np.linalg.norm(item.r)) for item in (*swarms, *bodies)), default=math.inf)
    experiment.assumptions = assumptions
    experiment.provenance = provenance
    _validate_population(experiment)
    if experiment.checkpoint_id != identity:
        raise ValueError('Checkpoint did not round-trip.')
    return experiment


def _provenance_from_document(value):
    _object(value, _PROVENANCE_KEYS, 'Provenance')
    if (value['model'] != DISK_MODEL or value['representation'] != REPRESENTATION_ID or value['drag_model'] != DRAG_MODEL_ID
            or value['growth_model'] != GROWTH_MODEL_ID or value['swarm_rule'] != SWARM_RULE_ID
            or value['collision_model'] != COLLISION_MODEL_ID):
        raise ValueError('Unsupported gas-and-dust provenance model.')
    seed = value['seed']
    if type(seed) is not int or isinstance(seed, bool) or seed < 0:
        raise ValueError('Disk seed must be a nonnegative integer.')
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
                  'delta_v': (_finite('delta_v', delta[0]), _finite('delta_v', delta[1])),
                  'note': _string(change['note'], 'Change note')}
    return {'run_id': _string(value['run_id'], 'Run id', hex_length=16), 'model': value['model'],
            'representation': value['representation'], 'drag_model': value['drag_model'],
            'growth_model': value['growth_model'], 'swarm_rule': value['swarm_rule'],
            'collision_model': value['collision_model'], 'seed': seed,
            'setup_digest': _string(value['setup_digest'], 'Setup digest', hex_length=64),
            'parent_checkpoint_id': _string(value['parent_checkpoint_id'], 'Parent checkpoint', nullable=True, hex_length=64),
            'change': change, 'execution_environment': {'numpy': _string(environment['numpy'], 'NumPy version')},
            'solver_build': _string(value['solver_build'], 'Solver build')}


def _parameters_from_document(value):
    _object(value, _PARAMETER_KEYS, 'Parameters')
    if type(value['drag_enabled']) is not bool:
        raise ValueError('drag_enabled must be true or false.')
    return DiskParameters(_positive('Sound speed', value['sound_speed']), _positive('Grain density', value['grain_density']),
                          _positive('Bulk density', value['bulk_density']), _positive('Stokes threshold', value['stokes_min']),
                          _positive('Grain-radius threshold', value['grain_radius_min']),
                          _positive('Mass threshold', value['mass_min']),
                          _positive('Drag coefficient', value['drag_coefficient']), value['drag_enabled'])


def _zone_from_document(value):
    _object(value, _ZONE_KEYS, 'Zone')
    return Zone(_string(value['id'], 'Zone id'), _finite('Inner radius', value['r_inner']),
                _finite('Outer radius', value['r_outer']), _positive('Zone mass', value['mass']),
                _finite('Radial momentum', value['radial_momentum']),
                _finite('Angular momentum', value['angular_momentum']))


def _swarm_from_document(value, *, active):
    _object(value, _SWARM_KEYS, 'Swarm')
    status = value['status']
    if active and status != 'active':
        raise ValueError('Active swarms must be marked active.')
    if not active and status not in ('coagulated', 'became-body'):
        raise ValueError('Removed swarms must be coagulated or became-body.')
    contributions = _contributions_from_document(value['contributions'])
    swarm = Swarm(_string(value['id'], 'Swarm id'), _string(value['name'], 'Swarm name'),
                  _positive('Swarm mass', value['mass']), _positive('Grain radius', value['grain_radius']),
                  _nonnegative('Internal speed', value['internal_speed']),
                  _positive('Spatial radius', value['spatial_radius']), np.array(_vector(value['r'], 'Position')),
                  np.array(_vector(value['v'], 'Velocity')), contributions)
    if not active:
        return _solid_view(swarm, 'swarm', status)
    return swarm


def _body_from_document(value, *, active):
    _object(value, _BODY_KEYS, 'Body')
    status = value['status']
    if active and status != 'active':
        raise ValueError('Active bodies must be marked active.')
    if not active and status != 'merged':
        raise ValueError('Removed bodies must be merged.')
    contributions = _contributions_from_document(value['contributions'])
    body = Body(_string(value['id'], 'Body id'), _string(value['name'], 'Body name'),
                _positive('Body mass', value['mass']), _positive('Body radius', value['radius']),
                _positive('Grain radius', value['grain_radius']), _nonnegative('Internal speed', value['internal_speed']),
                _positive('Bulk density', value['bulk_density']), np.array(_vector(value['r'], 'Position')),
                np.array(_vector(value['v'], 'Velocity')), contributions)
    if not active:
        return _solid_view(body, 'body', status)
    return body


def _event_from_document(value):
    _object(value, _EVENT_KEYS, 'Disk event')
    kind = value['kind']
    if kind not in ('coagulation', 'swarm-to-body', 'merge'):
        raise ValueError('Unsupported gas-and-dust event.')
    parents = tuple(_string(item, 'Parent id') for item in _list(value['parent_ids'], 'Parent ids'))
    if not parents:
        raise ValueError('A disk event needs a parent id.')
    remnant = _string(value['remnant_id'], 'Remnant id', nullable=True)
    descendants = tuple(_string(item, 'Descendant id') for item in _list(value['descendant_ids'], 'Descendant ids'))
    if remnant is None or descendants != (remnant,):
        raise ValueError('A disk event needs its product as the only descendant.')
    contributions = tuple(_contribution_from_document(item) for item in _list(value['contributions'], 'Contributions'))
    return DiskEvent(_finite('Event time', value['time']), kind, parents, remnant, descendants, contributions,
                     _positive('Event mass', value['mass']), _vector(value['momentum'], 'Momentum'),
                     _positive('Grain radius', value['grain_radius']), _nonnegative('Internal speed', value['internal_speed']),
                     _nonnegative('Internal kinetic energy', value['internal_kinetic']), _positive('Event radius', value['radius']),
                     _finite('Lost kinetic energy', value['lost_relative_kinetic']),
                     _finite('Unmodeled spin', value['unmodeled_spin']), _string(value['message'], 'Event message'))


def _sample_from_document(value):
    _object(value, _SAMPLE_KEYS, 'Sample')
    if value['phase'] not in (None, 'branch', 'pre', 'post'):
        raise ValueError('Unsupported sample phase.')
    zones = tuple(_zone_view(_zone_from_document(item)) for item in _list(value['zones'], 'Sample zones'))
    swarms = tuple(_swarm_from_document(item, active=True) for item in _list(value['swarms'], 'Sample swarms'))
    bodies = tuple(_body_from_document(item, active=True) for item in _list(value['bodies'], 'Sample bodies'))
    return DiskSample(_finite('Sample time', value['time']), value['phase'], zones,
                      tuple(_solid_view(item, 'swarm', 'active') for item in swarms),
                      tuple(_solid_view(item, 'body', 'active') for item in bodies))


def _contributions_from_document(value):
    items = tuple(_contribution_from_document(item) for item in _list(value, 'Contributions'))
    if not items or len({item.parcel_id for item in items}) != len(items):
        raise ValueError('Contributions need unique parcel ids.')
    return {item.parcel_id: item.mass for item in items}


def _contribution_from_document(value):
    _object(value, ('parcel_id', 'mass'), 'Contribution')
    return Contribution(_string(value['parcel_id'], 'Parcel id'), _positive('Contribution mass', value['mass']))


def _contribution_tuple(contributions):
    return tuple(Contribution(key, float(contributions[key])) for key in sorted(contributions))


def _solid_view(obj, kind, status):
    contributions = obj.contributions
    if isinstance(contributions, tuple):
        stored = contributions
    else:
        stored = _contribution_tuple(contributions)
    spatial = obj.spatial_radius if kind == 'swarm' else obj.radius
    return SolidView(obj.id, obj.name, kind, float(obj.mass), float(obj.radius) if kind == 'body' else float(spatial),
                     float(obj.grain_radius), float(obj.internal_speed), float(obj.spatial_radius) if kind == 'swarm' else float(obj.radius),
                     float(obj.bulk_density) if kind == 'body' else 0., _pair(obj.r), _pair(obj.v), stored, status)


def _zone_view(zone):
    return ZoneView(zone.id, float(zone.r_inner), float(zone.r_outer), float(zone.mass), float(zone.radial_momentum),
                    float(zone.angular_momentum))


def _parcel_error(initial, current):
    before, after = {}, {}
    for item in initial:
        for contribution in _contribution_tuple(item.contributions) if not isinstance(item.contributions, tuple) else item.contributions:
            before[contribution.parcel_id] = before.get(contribution.parcel_id, 0.)+contribution.mass
    for item in current:
        for contribution in _contribution_tuple(item.contributions):
            after[contribution.parcel_id] = after.get(contribution.parcel_id, 0.)+contribution.mass
    keys = set(before) | set(after)
    return max((abs(after.get(key, 0.)-before.get(key, 0.)) for key in keys), default=0.)


def _sample_angular(sample):
    total = sum(zone.angular_momentum for zone in sample.zones)
    for item in (*sample.swarms, *sample.bodies):
        total += item.mass*(item.r[0]*item.v[1]-item.r[1]*item.v[0])
    return total


def _sample_carries(sample, parcel_id):
    for item in (*sample.swarms, *sample.bodies):
        if any(contribution.parcel_id == parcel_id for contribution in item.contributions):
            return True
    return False


def _carries(obj, parcel_id):
    contributions = obj.contributions
    if isinstance(contributions, Mapping):
        return parcel_id in contributions
    return any(item.parcel_id == parcel_id for item in contributions)


def _event_names(event):
    names = set(event.parent_ids) | set(event.descendant_ids)
    names.update(item.parcel_id for item in event.contributions)
    return names


def _same_sample(left, right):
    return (left.time == right.time and left.phase == right.phase
            and tuple((zone.id, zone.mass, zone.radial_momentum, zone.angular_momentum) for zone in left.zones)
            == tuple((zone.id, zone.mass, zone.radial_momentum, zone.angular_momentum) for zone in right.zones)
            and tuple((item.id, item.r, item.v) for item in left.swarms) == tuple((item.id, item.r, item.v) for item in right.swarms)
            and tuple((item.id, item.r, item.v) for item in left.bodies) == tuple((item.id, item.r, item.v) for item in right.bodies))


def _positive(label, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be finite and positive.')
    return float(value)


GRAIN_DENSITY = solar_density(ROCK_DENSITY_KG_M3)
GRAIN_RADIUS_MIN = meters_to_au(1000.0)
MILLIMETER = meters_to_au(1e-3)


def _nonnegative(label, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label} must be finite and nonnegative.')
    return float(value)


def _finite(label, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)) or not math.isfinite(value):
        raise ValueError(f'{label} must be finite.')
    return float(value)


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError('Disk state must be finite.')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Disk state must be finite.')
    return result


def _vector_list(vector):
    array = np.asarray(vector, dtype=float)
    return [_num(array[0]), _num(array[1])]


def _pair(vector):
    array = np.asarray(vector, dtype=float)
    return (_num(array[0]), _num(array[1]))


def _vector(value, label):
    if type(value) is not list or len(value) != 2:
        raise ValueError(f'{label} must contain two components.')
    return (_finite(label, value[0]), _finite(label, value[1]))


def _object(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f'{label} must contain exactly {", ".join(keys)}.')
    return value


def _list(value, label):
    if type(value) is not list:
        raise ValueError(f'{label} must be a list.')
    return value


def _string(value, label, *, nullable=False, hex_length=None):
    if value is None and nullable:
        return None
    if type(value) is not str or not value.strip():
        raise ValueError(f'{label} must be a nonempty string.')
    if hex_length is not None and (len(value) != hex_length or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError(f'{label} must be {hex_length} lowercase hex characters.')
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON field: {key}.')
        result[key] = value
    return result
