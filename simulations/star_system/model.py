"""Immutable authored definitions. Distances AU, masses solar, velocities AU/year."""
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from .populations import Population

def freeze_metadata(value):
    """Defensive immutable copy of JSON provenance (never executable content)."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError('Provenance keys must be strings.')
        return MappingProxyType({key: freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_metadata(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and np.isfinite(value):
        return value
    raise ValueError('Provenance must contain finite JSON values.')

@dataclass(frozen=True)
class PropertySource:
    url: str
    version: str
    retrieved: str
    original_value: float | None
    original_units: str
    conversion: str
    uncertainty: float | str | tuple[float, ...] | None = None
    status: str = 'measurement'
    note: str = ''

    def __post_init__(self):
        if isinstance(self.uncertainty, list):
            object.__setattr__(self, 'uncertainty', tuple(self.uncertainty))

@dataclass(frozen=True)
class BodySpec:
    name: str
    kind: str
    mass: float | None
    radius: float | None
    temperature: float | None = None
    color: str | None = None
    catalog_key: str | None = None
    provenance: str = 'custom'
    sources: Mapping[str, PropertySource] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if not self.name.strip() or self.kind not in ('star', 'planet', 'moon', 'asteroid', 'particle'):
            raise ValueError('A body needs a name and a supported kind.')
        for key in ('mass', 'radius', 'temperature'):
            value = getattr(self, key)
            if value is not None and (not np.isfinite(value) or value <= 0):
                if key == 'mass' and self.kind == 'particle' and value == 0:
                    continue
                raise ValueError(f'{key} must be finite and positive, or unknown.')
        object.__setattr__(self, 'sources', MappingProxyType(dict(self.sources)))

@dataclass(frozen=True)
class InitialBody:
    id: str
    spec: BodySpec
    r: tuple[float, float]
    v: tuple[float, float]
    reference_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError('Body IDs must be nonempty strings.')
        for key in ('r', 'v'):
            vector = np.asarray(getattr(self, key), dtype=float)
            if vector.shape not in ((2,), (3,)) or not np.isfinite(vector).all():
                raise ValueError('Position and velocity must each contain two or three finite numbers.')
            object.__setattr__(self, key, tuple(float(x) for x in vector))

@dataclass(frozen=True)
class PopulationSpec:
    """Authored identity; Population.reference is only a legacy engine adapter."""
    id: str
    reference_id: str
    settings: 'Population'

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip() or not isinstance(self.reference_id, str) or not self.reference_id.strip():
            raise ValueError('Population and reference IDs are required.')
        # A numeric row has no meaning in authored metadata or a reordered setup.
        object.__setattr__(self, 'settings', replace(self.settings, reference=0))

# Optional close-pass energy sink. Off unless an experiment enables it.
# Strength is a dimensionless stand-in, not a measured k2/Q.
TIDAL_MODEL_ID = 'close-pass-energy-dissipation-v1'
TIDAL_STRENGTH = 80.0
TIDAL_REACH = 6.0
# Same id as the gas-and-dust drag laws. Off unless an experiment enables it.
DRAG_MODEL_ID = 'epstein-swarm-quadratic-body-v1'
DRAG_DISKS = ('annular', 'envelope', 'giant')


@dataclass(frozen=True)
class TidalPolicy:
    """Energy removed at resolved close pericenters. Disabled policies are not stored."""
    enabled: bool = False
    version: str = TIDAL_MODEL_ID
    strength: float = TIDAL_STRENGTH
    reach: float = TIDAL_REACH

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('Tidal dissipation must be explicitly on or off.')
        if self.version != TIDAL_MODEL_ID:
            raise ValueError('Unsupported tidal model.')
        if type(self.strength) not in (int, float) or not np.isfinite(self.strength) or not 0 < self.strength <= 1e6:
            raise ValueError('Tidal strength must be a finite number in (0, 1000000].')
        if type(self.reach) not in (int, float) or not np.isfinite(self.reach) or not 1 <= self.reach <= 100:
            raise ValueError('Tidal reach must be a finite number from 1 to 100, in units of the larger radius.')
        if not self.enabled and (self.strength != TIDAL_STRENGTH or self.reach != TIDAL_REACH):
            raise ValueError('Turn tidal dissipation on to store a strength or reach. Off uses the default policy only.')


@dataclass(frozen=True)
class DragPolicy:
    """Quadratic body drag from an annular disk. Disabled policies are not stored.

    ``disk`` selects which existing annular profile supplies the zones:
    ``annular`` is the seeded power-law disk, ``envelope`` is the Gaussian
    capture envelope, and ``giant`` is that same isothermal disk placed on a
    giant planet's orbit. The drag laws themselves do not change.
    """
    enabled: bool = False
    version: str = DRAG_MODEL_ID
    disk: str = 'annular'

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('Gas drag must be explicitly on or off.')
        if self.version != DRAG_MODEL_ID:
            raise ValueError('Unsupported drag model.')
        if self.disk not in DRAG_DISKS:
            raise ValueError("Drag disk must be 'annular', 'envelope', or 'giant'.")
        if not self.enabled and self.disk != 'annular':
            raise ValueError('Turn gas drag on to store a disk. Off uses the default policy only.')


# Body detail is off, at point mass, unless an experiment selects a richer level.
BODY_DETAIL_MODEL_ID = 'body-detail-v1'
BODY_DETAIL_LEVELS = ('point_mass', 'fixed_oblateness', 'spin')
# Secular constant-time-lag tides. Off unless an experiment enables them.
LASTING_TIDE_MODEL_ID = 'constant-time-lag-v1'


@dataclass(frozen=True)
class BodyDetailPolicy:
    """Which body interior an experiment uses. Point mass is the default and is not stored.

    ``fixed_oblateness`` adds a constant J2 bulge. ``spin`` tracks spin and a figure
    axis and applies the torques that go with them. The physical constants live with
    the catalog defaults in ``body_detail``, not in this policy.
    """
    level: str = 'point_mass'
    version: str = BODY_DETAIL_MODEL_ID

    def __post_init__(self):
        if self.level not in BODY_DETAIL_LEVELS:
            raise ValueError("Body detail level must be 'point_mass', 'fixed_oblateness', or 'spin'.")
        if self.version != BODY_DETAIL_MODEL_ID:
            raise ValueError('Unsupported body-detail model.')


@dataclass(frozen=True)
class LastingTidePolicy:
    """Secular constant-time-lag tides. A disabled policy is not stored."""
    enabled: bool = False
    version: str = LASTING_TIDE_MODEL_ID

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('Lasting tides must be explicitly on or off.')
        if self.version != LASTING_TIDE_MODEL_ID:
            raise ValueError('Unsupported lasting-tide model.')

# Pairwise 1PN and 2.5PN terms on top of Newtonian gravity. Off unless enabled.
RELATIVISTIC_MODEL_ID = 'pairwise-1pn-2.5pn-v1'
# Constant fractional isotropic wind. Off unless enabled.
MASS_LOSS_MODEL_ID = 'isotropic-jeans-wind-v1'
# Per year. Same order as a weak stellar wind, not a fitted solar-wind history.
MASS_LOSS_RATE_PER_YEAR = 1e-13


@dataclass(frozen=True)
class RelativisticPolicy:
    """Post-Newtonian corrections added to Newtonian gravity. Disabled unless enabled.

    ``conservative`` is the pairwise 1PN term (perihelion precession). ``radiation``
    is the pairwise 2.5PN radiation reaction (gravitational-wave inspiral). The
    experiment file turns both on together. A disabled policy does not store a
    narrower choice.
    """
    enabled: bool = False
    version: str = RELATIVISTIC_MODEL_ID
    conservative: bool = True
    radiation: bool = True

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.conservative) is not bool or type(self.radiation) is not bool:
            raise ValueError('Relativistic corrections must be explicitly on or off.')
        if self.version != RELATIVISTIC_MODEL_ID:
            raise ValueError('Unsupported relativistic model.')
        if not self.enabled and (self.conservative is not True or self.radiation is not True):
            raise ValueError(
                'Turn relativistic corrections on to store a term choice. Off applies neither term.')
        if self.enabled and not self.conservative and not self.radiation:
            raise ValueError('Relativistic corrections need the 1PN term, the 2.5PN term, or both.')


@dataclass(frozen=True)
class StellarMassLossPolicy:
    """Isotropic stellar wind. Disabled policies are not stored.

    Each star loses a constant fraction of its mass per year. The wind leaves
    the system and does not change the star's velocity. Planets are unchanged.
    """
    enabled: bool = False
    version: str = MASS_LOSS_MODEL_ID
    rate_per_year: float = MASS_LOSS_RATE_PER_YEAR

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('Stellar mass loss must be explicitly on or off.')
        if self.version != MASS_LOSS_MODEL_ID:
            raise ValueError('Unsupported stellar mass-loss model.')
        rate = self.rate_per_year
        if type(rate) not in (int, float) or isinstance(rate, bool) or not np.isfinite(rate) or not 0 < rate <= 1e6:
            raise ValueError('Stellar mass-loss rate must be a finite number in (0, 1000000] per year.')
        if not self.enabled and rate != MASS_LOSS_RATE_PER_YEAR:
            raise ValueError(
                'Turn stellar mass loss on to store a rate. Off uses the default policy only.')


# Perfect merge is one disclosed rule. The legacy pairs id said stellar contact
# halted the run; that behavior is retired, and loaded documents adopt the id below.
PERFECT_MERGE_VERSION = 'nonstellar-perfect-merge-v1'
PERFECT_MERGE_PAIRS = 'nonstellar-massive-merge-point-tracer-absorb-star-unmodeled-v1'
_LEGACY_PERFECT_MERGE_PAIRS = 'nonstellar-massive-merge-point-tracer-absorb-star-halt-v1'
PERFECT_MERGE_REMNANT = 'volume-com-planet-reference-map-v1'
COLLISION_NUMERICAL = 'positive-verlet-swept-parabola-v1'
SPEED_ANGLE_VERSION = 'speed-angle-merge-graze-fragment-v1'
SPEED_ANGLE_PAIRS = 'nonstellar-speed-angle-merge-graze-fragment-star-unmodeled-v1'
SPEED_ANGLE_REMNANT = 'fragments-tracked-losses-recorded-v1'
_PERFECT_MERGE_IDENTITY = (PERFECT_MERGE_VERSION, PERFECT_MERGE_PAIRS, PERFECT_MERGE_REMNANT, COLLISION_NUMERICAL)
_SPEED_ANGLE_IDENTITY = (SPEED_ANGLE_VERSION, SPEED_ANGLE_PAIRS, SPEED_ANGLE_REMNANT, COLLISION_NUMERICAL)
# ``none`` keeps the preset Yoshida step, which can halt on an unresolved encounter.
# ``ignore`` uses the contact stepper and lets bodies overlap without a remnant.
# ``stop_on_contact`` is the only mode that ends a run because two bodies touch.
COLLISION_MODES = ('none', 'ignore', 'stop_on_contact', 'perfect_merge', 'speed_angle', 'binary_star_merger')
SELECTABLE_COLLISION_MODES = ('none', 'ignore', 'stop_on_contact', 'perfect_merge', 'speed_angle')
CONTACT_SOLVER_MODES = ('ignore', 'stop_on_contact', 'perfect_merge', 'speed_angle', 'binary_star_merger')
FORMATION_COLLISION_MODES = ('ignore', 'stop_on_contact', 'perfect_merge', 'speed_angle')
_COLLISION_MODE_LIST = 'none, ignore, stop_on_contact, perfect_merge, speed_angle, binary_star_merger'


def collision_policy_identity(policy):
    """Which stored rule a policy names. ``None`` when the policy is not a known rule."""
    identity = (policy.version, policy.pairs, policy.remnant, policy.numerical)
    if identity == _PERFECT_MERGE_IDENTITY:
        return 'perfect_merge'
    if identity == _SPEED_ANGLE_IDENTITY:
        return 'speed_angle'
    return None


def retarget_collision_policy(policy, mode):
    """Keep the numerical bounds and store the named rule's version identifiers."""
    if not isinstance(policy, CollisionPolicy):
        raise ValueError('A supported CollisionPolicy is required.')
    if mode == 'speed_angle':
        version, pairs, remnant, numerical = _SPEED_ANGLE_IDENTITY
    elif mode in ('none', 'ignore', 'stop_on_contact', 'perfect_merge', 'binary_star_merger'):
        version, pairs, remnant, numerical = _PERFECT_MERGE_IDENTITY
    else:
        raise ValueError(f'Supported collision modes: {_COLLISION_MODE_LIST}.')
    return replace(policy, version=version, pairs=pairs, remnant=remnant, numerical=numerical)


def with_collision_rule(setup, mode):
    """Copy a setup onto one stored collision rule, including ``none``."""
    if mode not in COLLISION_MODES:
        raise ValueError(f'Supported collision modes: {_COLLISION_MODE_LIST}.')
    policy = retarget_collision_policy(setup.collision_policy, mode)
    return replace(setup, collision_mode=mode, collision_policy=policy)


@dataclass(frozen=True)
class CollisionPolicy:
    """Versioned, deliberately bounded contact model stored on the experiment setup."""
    version: str = PERFECT_MERGE_VERSION
    pairs: str = PERFECT_MERGE_PAIRS
    remnant: str = PERFECT_MERGE_REMNANT
    numerical: str = COLLISION_NUMERICAL
    max_substeps: int = 100000
    max_refinements: int = 64
    macro_fraction: float = .125
    guard_fraction: float = .5
    time_tolerance: float = 1e-12
    distance_rtol: float = 1e-12

    def __post_init__(self):
        if (self.version, self.pairs, self.remnant, self.numerical) == (
                PERFECT_MERGE_VERSION, _LEGACY_PERFECT_MERGE_PAIRS, PERFECT_MERGE_REMNANT, COLLISION_NUMERICAL):
            object.__setattr__(self, 'pairs', PERFECT_MERGE_PAIRS)
        if (self.version, self.pairs, self.remnant, self.numerical) not in (_PERFECT_MERGE_IDENTITY, _SPEED_ANGLE_IDENTITY):
            raise ValueError('Unsupported collision policy / model / numerical version.')
        if type(self.max_substeps) is not int or not 1 <= self.max_substeps <= 100000:
            raise ValueError('Collision substep bound must be an integer from 1 to 100000.')
        if type(self.max_refinements) is not int or not 16 <= self.max_refinements <= 64:
            raise ValueError('Collision refinement bound must be an integer from 16 to 64.')
        for name, low, high in (('macro_fraction', 1e-6, .125), ('guard_fraction', 1e-6, .5),
                                ('time_tolerance', 1e-15, 1e-12), ('distance_rtol', 1e-15, 1e-12)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f'Unsupported collision setting {name}.')


@dataclass(frozen=True)
class InitialSetup:
    name: str
    bodies: tuple[InitialBody, ...]
    dt: float
    preset_id: str | None = None
    population: PopulationSpec | None = None
    collision_mode: str = 'none'
    provenance: Mapping[str, object] = field(default_factory=dict, compare=False, hash=False)
    collision_policy: CollisionPolicy = field(default_factory=CollisionPolicy)
    tides: TidalPolicy = field(default_factory=TidalPolicy)
    drag: DragPolicy = field(default_factory=DragPolicy)
    relativistic: RelativisticPolicy = field(default_factory=RelativisticPolicy)
    stellar_mass_loss: StellarMassLossPolicy = field(default_factory=StellarMassLossPolicy)
    # NumPy provenance of the document that produced this setup. Absent on a
    # setup built in memory; load_setup copies it from the file. Not physical
    # state, so equality ignores it.
    source_environment: Mapping[str, str] | None = field(default=None, compare=False, hash=False)
    body_detail: BodyDetailPolicy = field(default_factory=BodyDetailPolicy)
    lasting_tides: LastingTidePolicy = field(default_factory=LastingTidePolicy)

    # Close-pass macro step. Off keeps the authored fixed step and is omitted
    # from the setup document so an existing planar digest stays unchanged.
    adaptive: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'bodies', tuple(self.bodies))
        if not self.name.strip() or not self.bodies:
            raise ValueError('A setup needs a name and at least one body.')
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('Fixed dt must be finite and positive (years).')
        if self.collision_mode not in COLLISION_MODES:
            raise ValueError(f'Supported collision modes: {_COLLISION_MODE_LIST}.')
        if not isinstance(self.collision_policy, CollisionPolicy):
            raise ValueError('A supported CollisionPolicy is required.')
        if not isinstance(self.tides, TidalPolicy):
            raise ValueError('A supported TidalPolicy is required.')
        if not isinstance(self.drag, DragPolicy):
            raise ValueError('A supported DragPolicy is required.')
        if not isinstance(self.body_detail, BodyDetailPolicy):
            raise ValueError('A supported BodyDetailPolicy is required.')
        if not isinstance(self.lasting_tides, LastingTidePolicy):
            raise ValueError('A supported LastingTidePolicy is required.')

        if type(self.adaptive) is not bool:
            raise ValueError('adaptive must be a boolean.')
        if not isinstance(self.relativistic, RelativisticPolicy):
            raise ValueError('A supported RelativisticPolicy is required.')
        if not isinstance(self.stellar_mass_loss, StellarMassLossPolicy):
            raise ValueError('A supported StellarMassLossPolicy is required.')
        family = collision_policy_identity(self.collision_policy)
        # A perfect-merge policy is the default object, so choosing speed_angle
        # without a second argument still stores that rule on the setup.
        if self.collision_mode == 'speed_angle' and family == 'perfect_merge':
            object.__setattr__(self, 'collision_policy', retarget_collision_policy(self.collision_policy, 'speed_angle'))
            family = 'speed_angle'
        if self.collision_mode == 'speed_angle' and family != 'speed_angle':
            raise ValueError('speed_angle stores the speed-angle collision policy on the setup.')
        if self.collision_mode != 'speed_angle' and family == 'speed_angle':
            raise ValueError('The speed-angle policy is stored only with collision mode speed_angle.')
        if not isinstance(self.provenance, Mapping):
            raise ValueError('Setup provenance must be an object.')
        object.__setattr__(self, 'provenance', freeze_metadata(self.provenance))
        if self.source_environment is not None:
            environment = self.source_environment
            if not isinstance(environment, Mapping) or set(environment) != {'numpy'}:
                raise ValueError('Source environment must record exactly a NumPy version.')
            numpy_version = environment['numpy']
            if type(numpy_version) is not str or not numpy_version.strip():
                raise ValueError('Source environment NumPy version must be a nonempty string.')
            object.__setattr__(self, 'source_environment', MappingProxyType({'numpy': numpy_version}))
