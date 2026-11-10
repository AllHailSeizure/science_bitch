"""Solar System starting template, prepared variations, and validated atomic edits."""
from dataclasses import dataclass, replace
from uuid import uuid4

import numpy as np

from .data import AU_KM, CATALOG
from .experiments import custom_copy
from .initialization import build_system, circular_relative_velocity, legacy_setup, recommended_moon_step
from .model import InitialBody, PopulationSpec, PropertySource
from .populations import Population

SOLAR_PLANETS = ('mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')


@dataclass(frozen=True)
class ExperimentTemplate:
    key: str
    title: str
    description: str
    observe: str


EXPERIMENTS = (
    ExperimentTemplate('solar-system', 'Solar System',
        'The Sun and all eight planets. The starting template for your experiments.',
        'Run the system, select a body to inspect its surroundings, or edit a copy. '
        'This is an illustrative 2D circular starting model, not current astronomical positions.'),
    ExperimentTemplate('heavier-star', 'A more massive star',
        'The same Solar System positions and velocities, with twice the Sun\'s mass.',
        'Compare with Solar System: changing gravity without changing starting motion changes the orbits.'),
    ExperimentTemplate('earth-moon', 'A planet and its moon',
        'The Solar System with the Moon added around Earth at 384,400 km.',
        'Select Earth and zoom in to see its moon\'s orbit. Replace the planet or edit its mass to investigate the effect.'),
    ExperimentTemplate('asteroid-ring', 'Asteroid perturbations',
        'The Solar System with 32 massless test particles in a ring at 3.2 AU.',
        'Watch the initially circular ring change under planetary gravity. '
        'This demonstrates perturbations; it does not promise asteroid capture.'),
)


def _solar_system():
    setup = legacy_setup(planets=SOLAR_PLANETS)
    origin = setup.bodies[0]
    bodies = (origin,) + tuple(replace(body,
        r=tuple(np.array(body.r)-origin.r), v=tuple(np.array(body.v)-origin.v), reference_id=origin.id)
        for body in setup.bodies[1:])
    return replace(setup, name='Solar System', bodies=bodies, preset_id='solar-system',
                   provenance={'starting_template': 'solar-system', 'origin_preset': 'solar-system'})


def get_experiment(key='solar-system'):
    definition = next((item for item in EXPERIMENTS if item.key == key), None)
    if definition is None:
        raise ValueError(f'Unknown experiment: {key}.')
    setup = _solar_system()
    if key == 'heavier-star':
        setup = update_body(setup, 'sun', spec=override_properties(CATALOG['sun'], mass=2.))
    elif key == 'earth-moon':
        prepared = build_system(setup)
        parent = prepared.system.r[prepared.body_ids.index('earth')]
        setup = place_body(setup, CATALOG['moon'], parent + (384400/AU_KM, 0), 'earth', body_id='moon')
    elif key == 'asteroid-ring':
        setup = replace(setup, population=PopulationSpec('asteroids', 'sun',
            Population(kind='ring', count=32, inner=3.2, outer=3.2, angles='even', seed=7)))
        prepared = build_system(setup)
        parent = prepared.body_ids.index('sun')
        explicit = len(setup.bodies)
        particles = tuple(InitialBody(key, replace(spec, name=f'Test particle {i+1}'),
            tuple(r-prepared.system.r[parent]), tuple(v-prepared.system.v[parent]), 'sun')
            for i, (key, spec, r, v) in enumerate(zip(prepared.body_ids[explicit:], prepared.body_specs[explicit:],
                prepared.system.r[explicit:], prepared.system.v[explicit:])))
        setup = replace(setup, bodies=setup.bodies+particles, population=None)
    result = replace(setup, name=definition.title, preset_id=key,
        provenance={'starting_template': 'solar-system', 'origin_preset': key,
                    'description': definition.description, 'observe': definition.observe})
    build_system(result)
    return result


def override_properties(spec, **values):
    sources = dict(spec.sources)
    for field, value in values.items():
        if field in ('mass', 'radius', 'temperature') and value != getattr(spec, field):
            sources[field] = PropertySource('user://override', 'desktop input', '', value,
                {'mass': 'solar masses', 'radius': 'AU', 'temperature': 'K'}[field],
                'identity', status='user-authored override')
    return replace(spec, **values, sources=sources,
                   provenance='User-edited '+(spec.catalog_key or spec.name))


def _validated_edit(candidate, original):
    # Validate shapes, references, surfaces and model first at a small fixed dt.
    # Then choose a conservative fixed step for this NEW starting setup only.
    checked = build_system(replace(candidate, dt=min(candidate.dt, 1e-12)))
    dt = min(candidate.dt, checked.system.resolution_limit()*.5)
    for i, parent in enumerate(checked.reference_ids):
        if parent in checked.body_ids:
            j = checked.body_ids.index(parent)
            if checked.system.masses[j] > 0:
                dt = min(dt, recommended_moon_step(checked.system.r[i]-checked.system.r[j],
                    checked.system.masses[j], checked.system.masses[i]))
    result = custom_copy(replace(candidate, dt=dt), original)
    build_system(result)
    return result


def update_body(setup, body_id, *, spec=None, position=None, velocity=None):
    """Replace properties or world vectors, preserving all other authored values."""
    prepared = build_system(setup)
    original = next((body for body in setup.bodies if body.id == body_id), None)
    if original is None:
        raise ValueError('This generated particle is not an editable starting body.')
    r, v = original.r, original.v
    if position is not None or velocity is not None:
        parent = prepared.body_ids.index(original.reference_id) if original.reference_id else None
        if position is not None:
            r = tuple(np.asarray(position)-(prepared.system.r[parent] if parent is not None else 0))
        if velocity is not None:
            v = tuple(np.asarray(velocity)-(prepared.system.v[parent] if parent is not None else 0))
    body = replace(original, spec=spec if spec is not None else original.spec, r=r, v=v)
    return _validated_edit(replace(setup, bodies=tuple(body if b.id == body_id else b for b in setup.bodies)), setup)


def place_body(setup, spec, position, reference_id=None, *, circular=True, velocity=(0., 0.), body_id=None):
    """Scene position is world AU; manual velocity is relative AU/year."""
    prepared = build_system(setup)
    parent = prepared.body_ids.index(reference_id) if reference_id in prepared.body_ids else None
    if reference_id is not None and parent is None:
        raise ValueError('Choose an existing body to place this relative to.')
    r = np.asarray(position)-(prepared.system.r[parent] if parent is not None else 0)
    if circular:
        if parent is None:
            raise ValueError('Choose a body for the initial circular orbit.')
        velocity = circular_relative_velocity(r, prepared.system.masses[parent], spec.mass)
    body = InitialBody(body_id or (spec.catalog_key or 'body')+'-'+uuid4().hex[:10], spec,
                       tuple(r), tuple(velocity), reference_id)
    return _validated_edit(replace(setup, bodies=setup.bodies+(body,)), setup)


def remove_body(setup, body_id):
    if any(b.reference_id == body_id for b in setup.bodies) or (setup.population and setup.population.reference_id == body_id):
        raise ValueError('Other bodies reference this body. Remove its orbiting bodies first.')
    return _validated_edit(replace(setup, bodies=tuple(b for b in setup.bodies if b.id != body_id)), setup)
