"""Preset sweep experiments the CLI writes, then ``sidereel --build`` records.

Each current preset is one file: a shared base setup, one swept variable, a
step, a run count, and a duration. Where the preset itself does not imply a
range, the file is one run at the value the preset already uses. Those ranges
are a starting point, not a locked grid. The capture flyby is an example in
the same shape, not a menu preset.

Building a file integrates with the existing solver and writes a
``.sidereel`` playback. Current presets are the three-dimensional ones.
The capture example stays on the planar legacy setup. The player reads
the file, draws the horizontal projection, and does not integrate.
"""
import math
from dataclasses import dataclass, replace

import numpy as np

from .data import CATALOG
from .experiment_file import PLAYBACK_SCHEMA, _atomic_text, _dumps, _with_mass, collect_watch_events
from .initialization import build_system, legacy_setup
from .jupiter_experiment import ring_setup
from .model import BodySpec, InitialBody
from .persistence import setup_digest
from .recording import Calculation, CalculationRequest
from .scenarios import DESCRIPTIONS
from .stellar_merger import binary_star_merger_setup, observation_years
from .templates import EXPERIMENTS, get_experiment

EXPERIMENT_SCHEMA = 'sidereel-experiment-v1'
PRESET_SCHEMA = 'sidereel-preset-sweep-v1'
PROPOSED_YEARS = 1.0
_MAX_RUNS = 100
_DOCUMENT_KEYS = ('schema', 'preset', 'dimensions', 'physics', 'experiment', 'proposal')
_PHYSICS_KEYS = ('collision', 'tides', 'gas_dust_drag')
_EXPERIMENT_KEYS = ('independent_variable', 'fixed_duration', 'open_scan', 'watch_for')
_VARIABLE_KEYS = ('parameter', 'body', 'start', 'stop', 'step')
_SOLAR = ('sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')
_UNITS = {'mass': 'solar_mass', 'distance': 'AU', 'speed': 'catalog_velocity'}


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    summary: str
    menu: bool


def _catalog_presets():
    earth = float(CATALOG['earth'].mass)
    jupiter = float(CATALOG['jupiter'].mass)
    return (
        Preset('solar-system', 'Solar System',
               'Sun mass, one run at the catalog mass (1). This preset implies no end or step, '
               'so the file does not invent a range. Proposed duration 1 year.', True),
        Preset('heavier-star', 'A more massive star',
               'Sun mass from 1 to 2 in steps of 0.1. Catalog positions and velocities stay fixed. '
               'The last run is twice the catalog solar mass. Proposed duration 1 year.', True),
        Preset('earth-moon', 'A planet and its moon',
               f"Earth's mass, one run at the catalog mass ({earth:.6g} solar masses). "
               'This preset implies no range. The Moon stays at 384,400 km. Proposed duration 1 year.', True),
        Preset('asteroid-ring', 'Asteroid perturbations',
               'Ring radius, one run at 3.2 AU. This preset implies no endpoints. '
               'Count 32, even angles, seed 7, and the planets stay fixed. Proposed duration 1 year.', True),
        Preset('sun-earth-jupiter', 'Sun, Earth, and Jupiter',
               'Sun mass, one run at the catalog mass (1). This preset implies no end or step. '
               "Earth and Jupiter keep this preset's starting vectors. Proposed duration 1 year.", True),
        Preset('sun-jupiter-ring', 'Sun–Jupiter ring',
               "Jupiter's mass from 0 (Jupiter removed, the matched control) to the catalog mass "
               f'({jupiter:.6g} solar masses) in five equal steps. The ring stays at 3.2 AU, '
               '32 rocks, seed 7. Proposed duration 1 year.', True),
        Preset('binary-star-merger', 'Binary-star merger',
               'A sweep of one, with no swept variable, over the published head-on window. '
               'The run slider is trivial.', True),
        Preset('capture-flyby', 'Capture flyby example',
               'Not a menu preset. The Sun and Jupiter, plus a massless tracer starting 0.15 AU '
               'farther out than Jupiter. Approach speed runs from 0.5 to 2 AU/yr in steps of 0.5. '
               'Duration 0.5 year. This can show a temporary flyby. It does not claim a permanent capture.', False),
    )


PRESETS = _catalog_presets()
PRESET_KEYS = tuple(item.key for item in PRESETS)


def preset(key):
    found = next((item for item in PRESETS if item.key == key), None)
    if found is None:
        known = ', '.join(PRESET_KEYS)
        raise ValueError(f'Unknown preset {key!r}. Known presets: {known}.')
    return found


def document(key):
    """The experiment file for one preset. Catalog presets use the CLI experiment schema."""
    preset(key)
    if key == 'solar-system':
        return _experiment(_SOLAR, _variable('mass', 'sun', 1.0, 1.0, 0.1), PROPOSED_YEARS)
    if key == 'heavier-star':
        return _experiment(_SOLAR, _variable('mass', 'sun', 1.0, 2.0, 0.1), PROPOSED_YEARS)
    if key == 'sun-earth-jupiter':
        return _experiment(('sun', 'earth', 'jupiter'), _variable('mass', 'sun', 1.0, 1.0, 0.1), PROPOSED_YEARS)
    if key == 'earth-moon':
        mass = float(CATALOG['earth'].mass)
        return _preset_document(
            key, _variable('mass', 'earth', mass, mass, mass), PROPOSED_YEARS, ('collision',),
            "Earth's mass range is not fixed by the preset. This file is one run at the catalog mass. "
            'The Moon stays at 384,400 km.')
    if key == 'asteroid-ring':
        return _preset_document(
            key, _variable('distance', 'ring', 3.2, 3.2, 0.1), PROPOSED_YEARS, ('collision',),
            'The ring radius is 3.2 AU. Endpoints are not fixed by the preset, so this file is one run there. '
            'Particle count, angles, seed, and the planets stay fixed.')
    if key == 'sun-jupiter-ring':
        mass = float(CATALOG['jupiter'].mass)
        return _preset_document(
            key, _variable('mass', 'jupiter', 0.0, mass, mass / 5.0), PROPOSED_YEARS, ('collision',),
            "Jupiter's mass runs from 0, where Jupiter is removed, to the catalog mass in five equal steps. "
            'The ring radius, count, seed, and rock vectors stay fixed.')
    if key == 'binary-star-merger':
        years = float(observation_years(binary_star_merger_setup()))
        return _preset_document(
            key, None, years, ('collision',),
            'A sweep of one. There is no swept variable. The duration is the published head-on window.',
            collision='binary_star_merger')
    years = 0.5
    return _preset_document(
        'capture-flyby', _variable('speed', 'tracer', 0.5, 2.0, 0.5), years,
        ('capture', 'collision', 'escape'),
        'A slow flyby near Jupiter with the Sun included, sweeping approach speed. '
        'Not a menu preset, and not a claim that the tracer is captured for good. '
        'This example stays on the planar legacy setup.',
        dimensions=2,
        )


def write_document(key, path):
    from pathlib import Path
    destination = Path(path)
    _atomic_text(destination, _dumps(document(key)))
    return destination


def write_all(directory):
    from pathlib import Path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for key in PRESET_KEYS:
        written.append(write_document(key, directory / f'{key}.json'))
    return tuple(written)


def build_preset_playback(document_):
    """Integrate a preset-sweep file with the existing solver and return playback JSON."""
    key, variable, years, watch, selected = _checked(document_)
    base = _base_setup(key)
    base_digest = setup_digest(base)
    if variable is None:
        values = (None,)
    else:
        values = _values(variable)
    runs = []
    for index, value in enumerate(values):
        setup = _setup_for_value(key, value)
        recorded = {
            'parameter': None if variable is None else variable['parameter'],
            'body': None if variable is None else variable['body'],
            'value': value,
            'unit': None if variable is None else _unit(key, variable['parameter']),
        }
        runs.append(_record(setup, years, index, None if value is None else recorded, watch, base_digest))
    return {
        'schema': PLAYBACK_SCHEMA,
        'encoding': 'json',
        'source_name': None,
        'units': {'length': 'AU', 'mass': 'solar_mass', 'time': 'Julian_year', 'velocity': 'AU/Julian_year'},
        'dimensions': len(base.bodies[0].r),
        'gravity_note': (
            'Newtonian inverse-square gravity. '
            'A three-dimensional recording is drawn by dropping the third component.'
            if len(base.bodies[0].r) == 3 else
            'Newtonian inverse-square gravity. Dimension 2 stores that motion in a plane.'),
        'physics': {
            'collision': selected,
            'engine_collision_mode': runs[0]['collision_mode'],
            'tides': False,
            'gas_dust_drag': False,
            'path': 'yoshida',
            'note': (
                'Preset sweep recorded by the existing integrator from one shared base setup. '
                'Only the swept value changes from run to run.'),
        },
        'experiment': document_,
        'base_setup': {'preset': key, 'name': base.name, 'digest': base_digest},
        'requested_years': years,
        'scan': {
            'mode': 'fixed' if variable is None else 'sweep',
            'until': None,
            'stopped_because': None,
        },
        'runs': runs,
    }


def _experiment(bodies, variable, years):
    return {
        'schema': EXPERIMENT_SCHEMA,
        'dimensions': 3,
        'bodies': {'catalog': list(bodies), 'new': None},
        'physics': {'collision': None, 'tides': None, 'gas_dust_drag': None},
        'experiment': {
            'independent_variable': variable,
            'fixed_duration': {'years': years},
            'open_scan': None,
            'watch_for': ['collision'],
        },
    }


def _preset_document(key, variable, years, watch, proposal, collision=None, dimensions=3):
    return {
        'schema': PRESET_SCHEMA,
        'preset': key,
        'dimensions': dimensions,
        'physics': {'collision': collision, 'tides': None, 'gas_dust_drag': None},
        'experiment': {
            'independent_variable': variable,
            'fixed_duration': {'years': float(years)},
            'open_scan': None,
            'watch_for': list(watch),
        },
        'proposal': proposal,
    }


def _variable(parameter, body, start, stop, step):
    return {'parameter': parameter, 'body': body, 'start': float(start), 'stop': float(stop), 'step': float(step)}


def _unit(key, parameter):
    if key == 'capture-flyby' and parameter == 'speed':
        return 'AU/Julian_year'
    return _UNITS[parameter]


def _checked(document_):
    if type(document_) is not dict or set(document_) != set(_DOCUMENT_KEYS):
        raise ValueError('Preset sweep: expected schema, preset, dimensions, physics, experiment, and proposal.')
    if document_['schema'] != PRESET_SCHEMA:
        raise ValueError(f'Unsupported preset schema. Expected {PRESET_SCHEMA}.')
    key = document_['preset']
    preset(key)
    expected_dimensions = 2 if key == 'capture-flyby' else 3
    if document_['dimensions'] != expected_dimensions:
        if key == 'capture-flyby':
            raise ValueError('The capture example uses the planar legacy setup. dimensions must be 2.')
        raise ValueError('This preset sweep uses the three-dimensional preset. dimensions must be 3.')
    physics = document_['physics']
    if type(physics) is not dict or set(physics) != set(_PHYSICS_KEYS):
        raise ValueError('Preset physics: expected collision, tides, and gas_dust_drag.')
    if physics['tides'] is not None or physics['gas_dust_drag'] is not None:
        raise ValueError('These preset files leave tides and gas-dust drag off.')
    selected = physics['collision']
    if key == 'binary-star-merger':
        if selected != 'binary_star_merger':
            raise ValueError("The binary-star merger preset records collision mode binary_star_merger.")
    elif selected is not None:
        raise ValueError('This preset keeps the template collision mode. Leave physics.collision null.')
    experiment = document_['experiment']
    if type(experiment) is not dict or set(experiment) != set(_EXPERIMENT_KEYS):
        raise ValueError('Preset experiment: expected independent_variable, fixed_duration, open_scan, and watch_for.')
    if experiment['open_scan'] is not None:
        raise ValueError('A preset sweep file uses a fixed duration.')
    duration = experiment['fixed_duration']
    if type(duration) is not dict or set(duration) != {'years'}:
        raise ValueError('Fixed duration: expected years.')
    years = duration['years']
    if isinstance(years, bool) or type(years) not in (int, float) or not math.isfinite(years) or years <= 0:
        raise ValueError('fixed_duration.years must be a finite number greater than 0.')
    watch = experiment['watch_for']
    if type(watch) is not list or any(type(item) is not str for item in watch):
        raise ValueError('watch_for must be a list of outcome names.')
    variable = experiment['independent_variable']
    expected = {
        'earth-moon': ('mass', 'earth'),
        'asteroid-ring': ('distance', 'ring'),
        'sun-jupiter-ring': ('mass', 'jupiter'),
        'capture-flyby': ('speed', 'tracer'),
        'binary-star-merger': None,
    }[key]
    if expected is None:
        if variable is not None:
            raise ValueError('The binary-star merger is a sweep of one and has no swept variable.')
    else:
        if type(variable) is not dict or set(variable) != set(_VARIABLE_KEYS):
            raise ValueError('Independent variable: expected parameter, body, start, stop, and step.')
        if (variable['parameter'], variable['body']) != expected:
            raise ValueError(f'{key} sweeps {expected[0]} of {expected[1]}.')
        _values(variable)
    if type(document_['proposal']) is not str or not document_['proposal'].strip():
        raise ValueError('A preset file needs a proposal note for ranges the preset itself does not fix.')
    return key, variable, float(years), tuple(watch), selected


def _values(variable):
    start = _finite(variable['start'], 'start')
    stop = _finite(variable['stop'], 'stop')
    step = _finite(variable['step'], 'step')
    if step == 0:
        raise ValueError('step must be a nonzero finite number.')
    if start == stop:
        return (start,)
    span = stop - start
    if (span > 0) != (step > 0):
        raise ValueError('step must move from start toward stop.')
    count = span / step
    nearest = round(count)
    if nearest < 1 or abs(count - nearest) > 1e-6 * max(1.0, abs(count)):
        raise ValueError('step must land on stop.')
    if nearest + 1 > _MAX_RUNS:
        raise ValueError(f'This sweep is more than {_MAX_RUNS} runs.')
    return tuple(start + step * index for index in range(nearest)) + (stop,)


def _finite(value, label):
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number.')
    return float(value)


def _base_setup(key):
    if key == 'earth-moon':
        return get_experiment('earth-moon')
    if key == 'asteroid-ring':
        return _ring_on_solar(3.2)
    if key == 'sun-jupiter-ring':
        return _fit_step(ring_setup())
    if key == 'binary-star-merger':
        return binary_star_merger_setup()
    if key == 'capture-flyby':
        return legacy_setup(planets=('jupiter',))
    raise ValueError(f'Unknown preset {key!r}.')


def _setup_for_value(key, value):
    if key == 'earth-moon':
        setup = get_experiment('earth-moon')
        current = next(body.spec.mass for body in setup.bodies if body.id == 'earth')
        if not math.isclose(current, value, rel_tol=0.0, abs_tol=0.0):
            setup = _set_mass(setup, 'earth', value)
        return _fit_step(setup)
    if key == 'asteroid-ring':
        return _ring_on_solar(value)
    if key == 'sun-jupiter-ring':
        setup = ring_setup()
        if value == 0:
            setup = _set_mass(setup, 'jupiter', 0.0)
        else:
            current = next(body.spec.mass for body in setup.bodies if body.id == 'jupiter')
            if not math.isclose(current, value, rel_tol=0.0, abs_tol=0.0):
                setup = _set_mass(setup, 'jupiter', value)
        return _fit_step(setup)
    if key == 'binary-star-merger':
        return binary_star_merger_setup()
    return _capture_setup(value)


def _set_mass(setup, body_id, mass):
    if mass == 0:
        if body_id != 'jupiter':
            raise ValueError('A mass of 0 removes Jupiter for the matched control. Other masses stay positive.')
        bodies = tuple(body for body in setup.bodies if body.id != body_id)
        if len(bodies) == len(setup.bodies):
            raise ValueError('Jupiter is not in this setup.')
        return replace(setup, name=setup.name + ' / no Jupiter', bodies=bodies)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('mass must be 0, to remove Jupiter, or a finite solar mass greater than 0.')
    found = False
    bodies = []
    for body in setup.bodies:
        if body.id == body_id:
            found = True
            bodies.append(replace(body, spec=_with_mass(body.spec, mass)))
        else:
            bodies.append(body)
    if not found:
        raise ValueError(f'{body_id} is not in this setup.')
    return replace(setup, bodies=tuple(bodies))


def _ring_on_solar(radius):
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Ring radius must be a finite AU value greater than 0.')
    from .model import PopulationSpec
    from .populations import Population
    setup = get_experiment('solar-system')
    staged = replace(
        setup,
        name='Asteroid perturbations',
        population=PopulationSpec('asteroids', 'sun', Population(
            kind='ring', count=32, inner=float(radius), outer=float(radius), angles='even', seed=7)))
    prepared = build_system(staged)
    parent = prepared.body_ids.index('sun')
    explicit = len(staged.bodies)
    particles = []
    for index, (body_id, spec, position, velocity) in enumerate(zip(
            prepared.body_ids[explicit:], prepared.body_specs[explicit:],
            prepared.system.r[explicit:], prepared.system.v[explicit:])):
        particles.append(InitialBody(
            body_id,
            replace(spec, name=f'Test particle {index + 1}'),
            tuple(float(component) for component in (position - prepared.system.r[parent])),
            tuple(float(component) for component in (velocity - prepared.system.v[parent])),
            'sun'))
    return _fit_step(replace(staged, bodies=staged.bodies + tuple(particles), population=None,
                             preset_id='asteroid-ring'))


def _capture_setup(speed):
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('Approach speed must be a finite AU/yr value greater than 0.')
    base = legacy_setup(planets=('jupiter',))
    prepared = build_system(base)
    sun = prepared.body_ids.index('sun')
    jupiter = prepared.body_ids.index('jupiter')
    offset = prepared.system.r[jupiter] - prepared.system.r[sun]
    direction = offset / np.linalg.norm(offset)
    tangent = np.array([-direction[1], direction[0]])
    position = prepared.system.r[jupiter] + direction * 0.15
    velocity = prepared.system.v[jupiter] + tangent * float(speed)
    tracer = InitialBody(
        'tracer',
        BodySpec('Flyby tracer', 'particle', 0.0, None, color='#f2d48a', provenance='capture-flyby example'),
        tuple(float(component) for component in position),
        tuple(float(component) for component in velocity))
    provenance = dict(base.provenance)
    provenance['origin_preset'] = 'capture-flyby'
    provenance['approach_speed_au_per_year'] = float(speed)
    setup = replace(
        base, name=f'Capture flyby / approach {float(speed):.6g} AU/yr', bodies=base.bodies + (tracer,),
        preset_id='capture-flyby', provenance=provenance)
    return _fit_step(setup)


def _fit_step(setup):
    probed = build_system(replace(setup, dt=1e-12))
    limit = float(probed.system.resolution_limit())
    dt = setup.dt
    if math.isfinite(limit) and dt > limit:
        dt = float(limit) * 0.5
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('The varied initial state has no usable fixed step.')
    fitted = replace(setup, dt=dt)
    build_system(fitted)
    return fitted


def _record(setup, years, index, variable, watch, base_digest):
    calculation = Calculation(setup, CalculationRequest(float(years), 1))
    while not calculation.done:
        calculation.advance(250)
    result = calculation.result
    primary = next((body.id for body in setup.bodies if body.spec.kind == 'star'), setup.bodies[0].id)
    user_collision = 'stop_on_contact' if setup.collision_mode == 'none' else setup.collision_mode
    return {
        'index': index,
        'variable': variable,
        'status': result.status,
        'reason': result.reason,
        'requested_years': float(result.requested_end_time),
        'duration_years': float(result.actual_end_time - result.frames[0].time),
        'time_start': float(result.frames[0].time),
        'time_end': float(result.actual_end_time),
        'completed_steps': result.completed_steps,
        'sample_stride': result.sample_stride,
        'setup_digest': result.setup_digest,
        'base_setup_digest': base_digest,
        'collision_mode': setup.collision_mode,
        'solver_build': result.solver_build,
        'dissipated_kinetic': 0.0,
        'bodies': _body_definitions(result.frames),
        'samples': [_sample(frame) for frame in result.frames],
        'events': collect_watch_events(
            result.frames, watch, status=result.status, reason=result.reason,
            user_collision=user_collision, primary_id=primary),
    }


def _body_definitions(frames):
    definitions = []
    seen = set()
    for frame in frames:
        for body_id, spec in zip(frame.body_ids, frame.body_specs):
            if body_id in seen:
                continue
            seen.add(body_id)
            definitions.append({
                'id': body_id,
                'name': spec.name,
                'kind': spec.kind,
                'catalog_key': spec.catalog_key,
                'color': spec.color,
                'radius': None if spec.radius is None else float(spec.radius),
                'mass': None if spec.mass is None else float(spec.mass),
            })
    return definitions


def _sample(frame):
    return {
        'time': float(frame.time),
        'dt': float(frame.dt),
        'phase': frame.event_phase,
        'ids': list(frame.body_ids),
        'mass': [float(mass) for mass in frame.masses],
        'r': [[float(component) for component in row] for row in frame.r],
        'v': [[float(component) for component in row] for row in frame.v],
    }


def menu_preset_keys():
    """Current menu and legacy preset keys. The capture example is not one of them."""
    return {item.key for item in EXPERIMENTS} | set(DESCRIPTIONS)
