"""Resolve authored reference vectors once; never recenter a live trajectory."""
from dataclasses import dataclass, replace
import numpy as np
from .core import System, G
from .model import BodySpec, InitialBody, InitialSetup

@dataclass(frozen=True)
class IntegrationEvent:
    time: float
    kind: str
    body_id: str
    old_dt: float
    new_dt: float
    message: str

@dataclass(frozen=True)
class PreparedSystem:
    system: System
    body_ids: tuple[str, ...]
    body_specs: tuple[BodySpec, ...]
    setup: InitialSetup | None = None
    events: tuple[IntegrationEvent, ...] = ()
    segment: int = 0

    def __post_init__(self):
        object.__setattr__(self, 'body_ids', tuple(self.body_ids))
        object.__setattr__(self, 'body_specs', tuple(self.body_specs))
        if len(self.body_ids) != len(self.system.masses) or len(self.body_specs) != len(self.body_ids):
            raise ValueError('Body identities and definitions must align with numeric rows.')
        if len(set(self.body_ids)) != len(self.body_ids):
            raise ValueError('Body IDs must be unique.')

def circular_relative_velocity(relative_position, reference_mass, body_mass):
    r = np.asarray(relative_position, dtype=float)
    if r.shape != (2,) or not np.isfinite(r).all() or np.linalg.norm(r) == 0:
        raise ValueError('Circular aid needs a finite nonzero relative position.')
    if reference_mass is None or body_mass is None or not np.isfinite([reference_mass, body_mass]).all() or reference_mass <= 0 or body_mass < 0:
        raise ValueError('Circular aid needs a positive reference mass and known body mass.')
    distance = np.linalg.norm(r)
    return np.array([-r[1], r[0]])/distance*np.sqrt(G*(reference_mass+body_mass)/distance)

def _validate_surfaces(r, specs):
    sources = np.array([i for i,s in enumerate(specs) if s.mass and s.mass > 0], dtype=int)
    if not len(sources):
        return
    radii = np.array([s.radius or 0. for s in specs])
    dist = np.linalg.norm(r[:,None,:]-r[None,sources,:],axis=-1)
    dist[sources,np.arange(len(sources))] = np.inf
    overlaps = np.argwhere(dist <= radii[:,None]+radii[None,sources])
    if len(overlaps):
        i,j = overlaps[0]
        raise ValueError(f'Bodies overlap: {specs[i].name} and {specs[sources[j]].name}. Move them apart.')


def build_system(setup):
    bodies = {body.id: body for body in setup.bodies}
    if len(bodies) != len(setup.bodies):
        raise ValueError('Body IDs must be unique, including repeated catalog choices.')
    resolved, visiting = {}, set()
    def resolve(key):
        if key in resolved:
            return resolved[key]
        if key not in bodies:
            raise ValueError(f'Missing reference body: {key}.')
        if key in visiting:
            raise ValueError('Reference cycle: choose an acyclic initial reference chain.')
        visiting.add(key)
        body = bodies[key]
        r, v = np.array(body.r), np.array(body.v)
        if body.reference_id is not None:
            rr, vv = resolve(body.reference_id)
            r, v = r+rr, v+vv
        visiting.remove(key)
        resolved[key] = r, v
        return r, v
    states = [resolve(body.id) for body in setup.bodies]
    specs = tuple(body.spec for body in setup.bodies)
    if any(spec.mass is None for spec in specs):
        raise ValueError('Gravity requires known mass: supply a custom mass for unknown catalog values.')
    r, v = (np.array([state[i] for state in states]) for i in (0, 1))
    _validate_surfaces(r, specs)
    system = System(np.array([s.mass for s in specs]), r, v, setup.dt)
    ids = tuple(bodies)
    if setup.population is not None:
        from .populations import with_population
        pop = setup.population
        if pop.reference_id not in ids:
            raise ValueError('Population reference body is missing.')
        index = ids.index(pop.reference_id)
        if specs[index].radius is None:
            raise ValueError('Population surface validation requires a known reference radius.')
        system = with_population(system, replace(pop.settings, reference=index), specs[index].radius)
        # Explicit dt is authoritative for general setups; never silently shorten here.
        system.dt = setup.dt
        particle = BodySpec('Test particle', 'particle', 0., None, color='#d8dfb3', provenance='generated population')
        ids += tuple(f'{pop.id}:{i}' for i in range(pop.settings.count))
        specs += (particle,)*pop.settings.count
    _validate_surfaces(system.r, specs)
    system.validate_step()
    return PreparedSystem(system, ids, specs, setup)

def insert_body(prepared, body):
    if body.id in prepared.body_ids:
        raise ValueError('Body ID already exists; choose a unique instance ID.')
    if body.spec.mass is None:
        raise ValueError('Gravity requires a known mass; enter a custom mass.')
    r, v = np.array(body.r), np.array(body.v)
    if body.reference_id is not None:
        if body.reference_id not in prepared.body_ids:
            raise ValueError('Insertion reference body is missing.')
        index = prepared.body_ids.index(body.reference_id)
        r, v = r+prepared.system.r[index], v+prepared.system.v[index]
    old = prepared.system
    system = System(np.append(old.masses, body.spec.mass), np.vstack((old.r, r)), np.vstack((old.v, v)), old.dt, old.time)
    specs = prepared.body_specs + (body.spec,)
    _validate_surfaces(system.r, specs)
    limit = system.resolution_limit()
    if np.isfinite(limit):
        system.dt = min(old.dt, limit*.5)
    system.validate_step()
    message = f'Inserted {body.spec.name}; starting setup unchanged.'
    if system.dt != old.dt:
        message += f' New fixed-step segment: dt {old.dt:.6g} -> {system.dt:.6g} yr.'
    event = IntegrationEvent(old.time, 'insertion', body.id, old.dt, system.dt, message)
    return PreparedSystem(system, prepared.body_ids+(body.id,), specs, prepared.setup, prepared.events+(event,), prepared.segment+1)

def legacy_setup(star='sun', planets=('earth', 'jupiter'), distances=None, population=None):
    """Retain the original circular/barycentric construction and step selection."""
    from .core import make_system, SOLAR_RADIUS_AU
    from .data import CATALOG, STARS
    from .model import PopulationSpec
    from .populations import with_population
    system = make_system(star, planets, distances)
    bodies = tuple(InitialBody(key, CATALOG[key], tuple(r), tuple(v))
                   for key, r, v in zip((star, *planets), system.r, system.v))
    pop = None
    if population is not None:
        system = with_population(system, population, STARS[star].radius*SOLAR_RADIUS_AU)
        pop = PopulationSpec('population', star, population)
    return InitialSetup(' / '.join(b.spec.name for b in bodies), bodies, system.dt, population=pop)
