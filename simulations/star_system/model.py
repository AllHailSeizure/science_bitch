"""Immutable authored definitions. Distances AU, masses solar, velocities AU/year."""
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from .populations import Population

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
class InitialSetup:
    name: str
    bodies: tuple[InitialBody, ...]
    dt: float
    preset_id: str | None = None
    population: PopulationSpec | None = None
    collision_mode: str = 'none'

    def __post_init__(self):
        object.__setattr__(self, 'bodies', tuple(self.bodies))
        if not self.name.strip() or not self.bodies:
            raise ValueError('A setup needs a name and at least one body.')
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('Fixed dt must be finite and positive (years).')
        if self.collision_mode != 'none':
            raise ValueError('Only collision mode none is currently supported.')
