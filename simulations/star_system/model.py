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
            if vector.shape != (2,) or not np.isfinite(vector).all():
                raise ValueError('Position and velocity must each contain two finite numbers.')
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

@dataclass(frozen=True)
class CollisionPolicy:
    """Versioned, deliberately bounded nonstellar contact model."""
    version: str = 'nonstellar-perfect-merge-v1'
    pairs: str = 'nonstellar-massive-merge-point-tracer-absorb-star-halt-v1'
    remnant: str = 'volume-com-planet-reference-map-v1'
    numerical: str = 'positive-verlet-swept-parabola-v1'
    max_substeps: int = 100000
    max_refinements: int = 64
    macro_fraction: float = .125
    guard_fraction: float = .5
    time_tolerance: float = 1e-12
    distance_rtol: float = 1e-12

    def __post_init__(self):
        expected = ('nonstellar-perfect-merge-v1', 'nonstellar-massive-merge-point-tracer-absorb-star-halt-v1',
                    'volume-com-planet-reference-map-v1', 'positive-verlet-swept-parabola-v1')
        if (self.version, self.pairs, self.remnant, self.numerical) != expected:
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

    def __post_init__(self):
        object.__setattr__(self, 'bodies', tuple(self.bodies))
        if not self.name.strip() or not self.bodies:
            raise ValueError('A setup needs a name and at least one body.')
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('Fixed dt must be finite and positive (years).')
        if self.collision_mode not in ('none', 'perfect_merge'):
            raise ValueError('Supported collision modes: none, perfect_merge.')
        if not isinstance(self.collision_policy, CollisionPolicy):
            raise ValueError('A supported CollisionPolicy is required.')
        if not isinstance(self.provenance, Mapping):
            raise ValueError('Setup provenance must be an object.')
        object.__setattr__(self, 'provenance', freeze_metadata(self.provenance))
