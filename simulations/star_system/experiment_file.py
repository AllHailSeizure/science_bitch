"""Blank experiment JSON and the playback file ``sidereel --build`` writes.

``sidereel --new`` writes every section, blank, in the agreed order:
dimensions, bodies, physics, and the run. Newtonian gravity is what
Sidereel uses. A ``gravity_model`` field is refused (see #77).
Relativistic corrections and stellar mass loss are opt-in keys in the
physics section and are off until set true.
``sidereel --build`` checks that file and integrates it with the existing
solver. The ``.sidereel`` result is JSON: one run, or several when the file
sweeps a variable or scans until a watched outcome.

Custom bodies and collision mode ``ignore`` are reserved and refused.
Collision rules, gas-dust drag, and tides, including a 3D state, run
together on the combined formation path. Post-Newtonian terms are
``physics.relativistic_corrections``. The relativistic and mass-loss
modules run on that path and on the Yoshida path.
"""
import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .collisions import CollisionEvent
from .core import ADAPTIVE_DT_MIN, ADAPTIVE_GROWTH, G, SOLAR_RADIUS_AU, make_system
from .data import CATALOG, PLANETS, STARS
from .initialization import PRESET_INCLINATION, build_system, with_preset_inclination
from .model import (
    MASS_LOSS_RATE_PER_YEAR, DragPolicy, InitialBody, InitialSetup, PropertySource,
    RelativisticPolicy, StellarMassLossPolicy, TidalPolicy,
)
from .recording import Calculation, CalculationRequest
from .tides import TideEvent

EXPERIMENT_SCHEMA = 'sidereel-experiment-v1'
PLAYBACK_SCHEMA = 'sidereel-playback-v1'
PLAYBACK_SUFFIX = '.sidereel'
COLLISION_MODES = ('stop_on_contact', 'perfect_merge', 'speed_angle', 'ignore')
_FORMATION_COLLISION = ('stop_on_contact', 'perfect_merge', 'speed_angle')
# Far enough that a short catalog orbit is not dropped as an ejection.
_EJECTION_AU = 10000.0
PARAMETERS = ('distance', 'mass', 'speed')
PARAMETER_UNITS = {'distance': 'AU', 'mass': 'solar_mass', 'speed': 'catalog_velocity'}
WATCH_KINDS = ('capture', 'collision', 'merger', 'escape', 'graze', 'fragment', 'tide', 'close_encounter')
_CONTACT_KINDS = {
    'merge': ('merger', 'collision'),
    'stellar_merger': ('merger', 'collision'),
    'fragment': ('fragment', 'collision'),
    'graze': ('graze', 'collision'),
    'absorption': ('collision',),
    'unmodeled': ('collision',),
    'close_encounter': ('close_encounter',),
}
_MAX_RUNS = 100
_DOCUMENT_KEYS = ('schema', 'dimensions', 'bodies', 'physics', 'experiment')
_BODY_KEYS = ('catalog', 'new')
_PHYSICS_KEYS = ('collision', 'tides', 'gas_dust_drag', 'relativistic_corrections', 'stellar_mass_loss')
_EXPERIMENT_KEYS = ('independent_variable', 'fixed_duration', 'open_scan', 'watch_for')
_SWEEP_KEYS = ('parameter', 'body', 'start', 'stop', 'step')
_SCAN_VARIABLE_KEYS = ('parameter', 'body', 'start')
_DURATION_KEYS = ('years',)
_SCAN_KEYS = ('increment', 'until', 'years', 'max_runs')


def is_experiment_command(arguments):
    """True when the argv list is ``--new`` or ``--build``."""
    return any(argument.split('=', 1)[0] in ('--new', '--build') for argument in arguments)


def empty_experiment():
    """Every section present and blank, in the agreed order."""
    return {
        'schema': EXPERIMENT_SCHEMA,
        'dimensions': None,
        'bodies': {'catalog': [], 'new': None},
        'physics': {
            'collision': None,
            'tides': None,
            'gas_dust_drag': None,
            'relativistic_corrections': None,
            'stellar_mass_loss': None,
        },
        'experiment': {
            'independent_variable': None,
            'fixed_duration': None,
            'open_scan': None,
            'watch_for': [],
        },
    }


@dataclass(frozen=True)
class _Variable:
    parameter: str
    body: str
    start: float
    stop: float | None = None
    step: float | None = None

    def values(self, limit):
        if self.stop is None:
            return tuple(self.start + self.step * index for index in range(limit))
        return _sweep_values(self.start, self.stop, self.step)

    def document(self):
        value = {'parameter': self.parameter, 'body': self.body, 'start': self.start}
        if self.stop is not None:
            value['stop'] = self.stop
            value['step'] = self.step
        return value


@dataclass(frozen=True)
class _Scan:
    increment: float
    until: str
    years: float
    max_runs: int

    def document(self):
        return {'increment': self.increment, 'until': self.until, 'years': self.years, 'max_runs': self.max_runs}


@dataclass(frozen=True)
class Experiment:
    dimensions: int
    catalog: tuple[str, ...]
    collision: str | None
    tides: bool | None
    gas_dust_drag: bool | None
    relativistic_corrections: bool | None
    stellar_mass_loss: bool | None
    variable: _Variable | None
    years: float | None
    scan: _Scan | None
    watch_for: tuple[str, ...]
    adaptive: bool = False

    @property
    def tides_on(self):
        return self.tides is True

    @property
    def drag_on(self):
        return self.gas_dust_drag is True

    @property
    def relativistic_on(self):
        return self.relativistic_corrections is True

    @property
    def mass_loss_on(self):
        return self.stellar_mass_loss is True

    def document(self):
        return {
            'schema': EXPERIMENT_SCHEMA,
            'dimensions': self.dimensions,
            'bodies': {'catalog': list(self.catalog), 'new': None},
            'physics': {
                'collision': self.collision,
                'tides': self.tides,
                'gas_dust_drag': self.gas_dust_drag,
                'relativistic_corrections': self.relativistic_corrections,
                'stellar_mass_loss': self.stellar_mass_loss,
            },
            'experiment': self._experiment_section(),
        }

    def _experiment_section(self):
        section = {
            'independent_variable': None if self.variable is None else self.variable.document(),
            'fixed_duration': None if self.years is None else {'years': self.years},
            'open_scan': None if self.scan is None else self.scan.document(),
            'watch_for': list(self.watch_for),
        }
        if self.adaptive:
            section['adaptive'] = True
        return section


def write_empty(path):
    destination = Path(path)
    if destination.exists() and destination.is_dir():
        destination = destination / 'experiment.json'
    if destination.exists():
        raise ValueError(f'{destination} already exists.')
    _atomic_text(destination, _dumps(empty_experiment()))
    return destination


def load_experiment(path):
    try:
        with Path(path).open(encoding='utf-8') as handle:
            document = json.load(handle, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f'Invalid experiment JSON: {exc}') from exc
    return _decode(document)


def playback_path(source):
    return Path(source).with_suffix(PLAYBACK_SUFFIX)


def build_experiment(experiment):
    """Integrate a validated experiment and return the playback document."""
    base = _catalog_setup(experiment.catalog, experiment.dimensions)
    if experiment.scan is not None:
        runs, stopped = _open_scan(experiment, base)
        scan = {'mode': 'open', 'until': experiment.scan.until, 'stopped_because': stopped}
        years = experiment.scan.years
    elif experiment.variable is not None:
        runs = _sweep(experiment, base)
        scan = {'mode': 'sweep', 'until': None, 'stopped_because': None}
        years = experiment.years
    else:
        runs = [_one_run(experiment, base, None, 0, experiment.years)]
        scan = {'mode': 'fixed', 'until': None, 'stopped_because': None}
        years = experiment.years
    playback = {
        'schema': PLAYBACK_SCHEMA,
        'encoding': 'json',
        'source_name': None,
        'units': {'length': 'AU', 'mass': 'solar_mass', 'time': 'Julian_year', 'velocity': 'AU/Julian_year'},
        'dimensions': experiment.dimensions,
        'gravity_note': _gravity_note(experiment),
        'physics': _physics_report(experiment),
        'experiment': experiment.document(),
        'requested_years': years,
        'scan': scan,
        'runs': runs,
    }
    if experiment.dimensions == 3:
        playback['inclination_rad'] = float(PRESET_INCLINATION)
    return playback


def _gravity_note(experiment):
    correction = ' physics.relativistic_corrections adds pairwise 1PN and 2.5PN terms when it is on.'
    if experiment.dimensions == 3:
        return (
            'Newtonian inverse-square gravity.'
            + correction
            + f' Dimension 3 stores three components, inclined by {PRESET_INCLINATION:g} rad about the x axis. '
            'A two-component recording is not given a zero z component.'
        )
    return (
        'Newtonian inverse-square gravity.'
        + correction
        + ' Dimension 2 stores that motion in a plane.'
    )


def collect_watch_events(frames, watch, *, status, reason, user_collision, primary_id):
    """Watch-for rows from one finished run, in time order.

    Engine collision and tide events keep their own times. Escape is the first
    sample on which a body is unbound from the primary. Capture is a later
    sample on which a body is bound to a mass it was unbound from at first sight.
    """
    wanted = set(watch)
    found = []
    if frames:
        for event in frames[-1].events:
            found.extend(_engine_watch(event, wanted))
        found.extend(_trajectory_watch(frames, wanted, primary_id))
    if user_collision == 'stop_on_contact' and status == 'halted' and 'collision' in wanted and _encounter_halt(reason):
        when = float(frames[-1].time) if frames else 0.0
        found.append({'time': when, 'kind': 'collision', 'bodies': [], 'message': reason})
    found.sort(key=lambda item: (item['time'], item['kind'], tuple(item['bodies']), item['message']))
    unique = []
    seen = set()
    for item in found:
        key = (item['time'], item['kind'], tuple(item['bodies']), item['message'])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog='sidereel', description='Write a blank experiment, or build it into a .sidereel playback file.',
        allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--new', nargs='?', const='experiment.json', default=None,
                       help='Write a blank experiment JSON (default: experiment.json)')
    group.add_argument('--build', type=Path, help='Validate an experiment JSON and write a .sidereel file beside it')
    args = parser.parse_args(arguments)
    try:
        if args.new is not None:
            destination = write_empty(args.new)
            print(destination)
            return 0
        source = args.build
        experiment = load_experiment(source)
        playback = build_experiment(experiment)
        playback['source_name'] = source.name
        destination = playback_path(source)
        _atomic_text(destination, _dumps(playback))
        print(destination)
        return 0
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _decode(document):
    if type(document) is dict and 'gravity_model' in document:
        raise ValueError(
            'gravity_model is not an experiment field (see #77). '
            'Newtonian gravity is what Sidereel uses. Remove gravity_model. '
            'Turn on physics.relativistic_corrections for post-Newtonian precession and gravitational-wave inspiral.')
    _object(document, _DOCUMENT_KEYS, 'Experiment')
    if document['schema'] != EXPERIMENT_SCHEMA:
        raise ValueError(f'Unsupported experiment schema. Expected {EXPERIMENT_SCHEMA}.')
    dimensions = _dimensions(document['dimensions'])
    catalog = _bodies(document['bodies'])
    collision, tides, drag, relativistic, mass_loss = _physics(document['physics'])
    variable, years, scan, watch, adaptive = _details(document['experiment'])
    if variable is not None and variable.body not in catalog:
        raise ValueError(f'independent_variable.body {variable.body!r} is not in bodies.catalog.')
    return Experiment(
        dimensions, catalog, collision, tides, drag, relativistic, mass_loss,
        variable, years, scan, watch, adaptive)


def _dimensions(value):
    if value is None:
        raise ValueError('Choose dimensions 2 or 3.')
    if type(value) is not int or value not in (2, 3):
        raise ValueError('Dimensions must be 2 or 3.')
    return value


def _bodies(value):
    _object(value, _BODY_KEYS, 'Bodies')
    if value['new'] is not None:
        raise ValueError(
            'Custom bodies are reserved and not implemented yet. Leave bodies.new null and list catalog names in bodies.catalog.')
    catalog = value['catalog']
    if type(catalog) is not list:
        raise ValueError('bodies.catalog must be a list of catalog names.')
    names = []
    for item in catalog:
        if type(item) is not str or not item.strip():
            raise ValueError(
                'Each catalog entry must be a catalog name. A custom body object is reserved and not implemented yet.')
        names.append(item)
    if len(names) != len(set(names)):
        duplicate = next(name for name in names if names.count(name) > 1)
        raise ValueError(f'Duplicate catalog body {duplicate}.')
    for name in names:
        if name not in CATALOG:
            raise ValueError(f'Unknown catalog body {name!r}. Use a catalog key such as sun, earth, or jupiter.')
    stars = [name for name in names if CATALOG[name].kind == 'star']
    if len(stars) != 1 or stars[0] not in STARS:
        placed = ', '.join(sorted(STARS))
        if len(stars) > 1:
            raise ValueError(
                f'Choose one star. {stars[1]} would need a custom placement, which is not implemented yet.')
        if len(stars) == 1:
            raise ValueError(
                f'{stars[0]} is a catalog star without a placed circular system. Placed stars: {placed}. '
                'A custom arrangement belongs under bodies.new and is not implemented yet.')
        raise ValueError(f'Choose one star: {placed}.')
    planets = [name for name in names if name != stars[0]]
    if not planets:
        raise ValueError('Choose at least one planet from the catalog.')
    for name in planets:
        if name not in PLANETS:
            raise ValueError(
                f'{name} is in the catalog and has no circular orbit in the existing placement. '
                'Custom placement belongs under bodies.new and is not implemented yet.')
    return tuple(names)


def _physics(value):
    if type(value) is not dict:
        raise ValueError('Physics: expected an object.')
    # Older files omit the later modules. A missing key is off. --new still writes every key.
    filled = dict(value)
    for key in ('relativistic_corrections', 'stellar_mass_loss'):
        filled.setdefault(key, None)
    _object(filled, _PHYSICS_KEYS, 'Physics')
    value = filled
    collision = value['collision']
    if collision is not None:
        if type(collision) is not str or collision not in COLLISION_MODES:
            choices = ', '.join(COLLISION_MODES)
            raise ValueError(f'Collision mode must be one of: {choices}.')
        if collision == 'ignore':
            raise ValueError(
                "Collision mode 'ignore' is not implemented yet. The engine halts on an unresolved close encounter "
                'or applies perfect merge or speed-angle; it does not let bodies pass through.')
    tides = _optional_bool(value['tides'], 'physics.tides')
    drag = _optional_bool(value['gas_dust_drag'], 'physics.gas_dust_drag')
    relativistic = _optional_bool(value['relativistic_corrections'], 'physics.relativistic_corrections')
    mass_loss = _optional_bool(value['stellar_mass_loss'], 'physics.stellar_mass_loss')
    return collision, tides, drag, relativistic, mass_loss


def _details(value):
    if type(value) is not dict:
        raise ValueError(
            'Experiment: expected exactly fields independent_variable, fixed_duration, open_scan, watch_for.')
    adaptive = False
    if 'adaptive' in value:
        if type(value['adaptive']) is not bool:
            raise ValueError('experiment.adaptive must be true or false.')
        adaptive = value['adaptive']
        value = {key: item for key, item in value.items() if key != 'adaptive'}
    _object(value, _EXPERIMENT_KEYS, 'Experiment')
    watch = _watch(value['watch_for'])
    duration = value['fixed_duration']
    scan = value['open_scan']
    if duration is None and scan is None:
        raise ValueError('Choose a fixed duration or an open scan.')
    if duration is not None and scan is not None:
        raise ValueError('Choose a fixed duration or an open scan, not both.')
    variable = value['independent_variable']
    if scan is not None:
        parsed_scan = _open(scan, watch)
        if variable is None:
            raise ValueError('An open scan needs an independent variable.')
        return _variable(variable, _SCAN_VARIABLE_KEYS, parsed_scan.increment), None, parsed_scan, watch, adaptive
    years = _duration(duration)
    if variable is None:
        return None, years, None, watch, adaptive
    return _variable(variable, _SWEEP_KEYS, None), years, None, watch, adaptive


def _watch(value):
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError('watch_for must be a list of outcome names.')
    if len(value) != len(set(value)):
        raise ValueError('watch_for names an outcome more than once.')
    unknown = [item for item in value if item not in WATCH_KINDS]
    if unknown:
        choices = ', '.join(WATCH_KINDS)
        raise ValueError(f'Unknown watch_for outcome {unknown[0]!r}. Choose from: {choices}.')
    return tuple(value)


def _duration(value):
    _object(value, _DURATION_KEYS, 'Fixed duration')
    return _positive(value['years'], 'fixed_duration.years')


def _open(value, watch):
    _object(value, _SCAN_KEYS, 'Open scan')
    increment = _finite(value['increment'], 'open_scan.increment')
    if increment == 0:
        raise ValueError('open_scan.increment must be a nonzero finite number.')
    until = value['until']
    if type(until) is not str or until not in WATCH_KINDS:
        raise ValueError(f'open_scan.until must name a watched outcome. Choose from: {", ".join(WATCH_KINDS)}.')
    if until not in watch:
        raise ValueError('open_scan.until must be listed in watch_for.')
    years = _positive(value['years'], 'open_scan.years')
    limit = value['max_runs']
    if type(limit) is not int or isinstance(limit, bool) or not 1 <= limit <= _MAX_RUNS:
        raise ValueError(f'open_scan.max_runs must be an integer from 1 to {_MAX_RUNS}.')
    return _Scan(increment, until, years, limit)


def _variable(value, keys, increment):
    _object(value, keys, 'Independent variable')
    parameter = value['parameter']
    body = value['body']
    if type(parameter) is not str or parameter not in PARAMETERS:
        raise ValueError(f'Independent variable must be one of: {", ".join(PARAMETERS)}.')
    if type(body) is not str or not body.strip():
        raise ValueError('independent_variable.body must be a catalog name from bodies.catalog.')
    start = _finite(value['start'], 'independent_variable.start')
    stop = step = None
    if 'stop' in keys:
        stop = _finite(value['stop'], 'independent_variable.stop')
        step = _finite(value['step'], 'independent_variable.step')
        _sweep_values(start, stop, step)
    else:
        step = increment
    return _Variable(parameter, body, start, stop, step)


def _uses_combined_path(experiment):
    """Drag, or tides with a contact rule or a 3D state, uses the formation path."""
    if experiment.drag_on:
        return True
    if not experiment.tides_on:
        return False
    return experiment.dimensions == 3 or experiment.collision in ('perfect_merge', 'speed_angle')


def _engine_mode(experiment):
    """Collision mode the integrator actually stores for this experiment."""
    if _uses_combined_path(experiment):
        if experiment.collision in _FORMATION_COLLISION:
            return experiment.collision
        # A blank collision module does not merge and does not let bodies pass through.
        return 'stop_on_contact'
    if experiment.collision in (None, 'stop_on_contact'):
        return 'none'
    return experiment.collision


def _catalog_setup(names, dimensions):
    star = next(name for name in names if name in STARS)
    planets = tuple(name for name in names if name != star)
    system = make_system(star, planets)
    bodies = []
    for key, position, velocity in zip((star, *planets), system.r, system.v):
        r = tuple(float(component) for component in position)
        v = tuple(float(component) for component in velocity)
        bodies.append(InitialBody(key, CATALOG[key], r, v))
    name = ' / '.join(body.spec.name for body in bodies)
    setup = InitialSetup(name, tuple(bodies), float(system.dt), provenance={'origin': EXPERIMENT_SCHEMA})
    if dimensions == 2:
        return setup
    inclined = with_preset_inclination(setup)
    return replace(inclined, provenance={
        **dict(inclined.provenance),
        'dimensions': 3,
        'inclination_rad': float(PRESET_INCLINATION),
        'adaptive': False,
    })


def _setup_for_value(experiment, base, value):
    setup = base
    if experiment.variable is not None:
        setup = _apply_variable(setup, experiment.variable.parameter, experiment.variable.body, value)
    mode = _engine_mode(experiment)
    tides = TidalPolicy(enabled=True) if experiment.tides_on else TidalPolicy()
    drag = DragPolicy(enabled=True) if experiment.drag_on else DragPolicy()
    relativistic = RelativisticPolicy(enabled=True) if experiment.relativistic_on else RelativisticPolicy()
    mass_loss = StellarMassLossPolicy(enabled=True) if experiment.mass_loss_on else StellarMassLossPolicy()
    probed = build_system(replace(setup, dt=1e-12, collision_mode='none', tides=TidalPolicy()))
    limit = probed.system.resolution_limit()
    # Keep the catalog step when it still resolves the initial state. Shrink it
    # only after a variation makes that step too coarse for the Yoshida check.
    if math.isfinite(limit) and setup.dt > limit:
        dt = float(limit) * 0.5
    else:
        dt = setup.dt
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('The varied initial state has no usable fixed step.')
    prepared = replace(
        setup, dt=dt, collision_mode=mode, tides=tides, drag=drag,
        relativistic=relativistic, stellar_mass_loss=mass_loss, adaptive=experiment.adaptive)
    build_system(prepared)
    return prepared


def _apply_variable(setup, parameter, body_id, value):
    if body_id not in {body.id for body in setup.bodies}:
        raise ValueError(f'independent_variable.body {body_id!r} is not in bodies.catalog.')
    star = next(body for body in setup.bodies if body.id in STARS)
    original = next(body for body in setup.bodies if body.id == body_id)
    if parameter == 'distance':
        if body_id == star.id:
            raise ValueError('distance varies a planet. The star stays where the catalog placement put it.')
        if not math.isfinite(value) or value <= 0:
            raise ValueError('distance must be a finite orbital radius in AU, greater than 0.')
        surface = STARS[star.id].radius * SOLAR_RADIUS_AU
        if value <= surface:
            raise ValueError('distance must sit outside the stellar surface.')
        offset = np.asarray(original.r, dtype=float) - np.asarray(star.r, dtype=float)
        norm = float(np.linalg.norm(offset))
        if norm == 0:
            raise ValueError(f'{body_id} has no direction from the star to move along.')
        updated = replace(original, r=tuple(float(component) for component in (np.asarray(star.r) + offset / norm * value)))
    elif parameter == 'mass':
        if not math.isfinite(value) or value <= 0:
            raise ValueError('mass must be a finite solar mass greater than 0.')
        updated = replace(original, spec=_with_mass(original.spec, value))
    else:
        if body_id == star.id:
            raise ValueError('speed varies a planet relative to the star.')
        relative = np.asarray(original.v, dtype=float) - np.asarray(star.v, dtype=float)
        updated = replace(original, v=tuple(float(component) for component in (np.asarray(star.v) + value * relative)))
    bodies = tuple(updated if body.id == body_id else body for body in setup.bodies)
    label = f'{setup.name} / {body_id} {parameter} {value:.8g}'
    return replace(setup, name=label, bodies=bodies)


def _with_mass(spec, mass):
    sources = dict(spec.sources)
    sources['mass'] = PropertySource(
        'user://experiment', 'sidereel --build', '', float(mass), 'solar masses', 'identity',
        status='user-authored override')
    return replace(spec, mass=float(mass), sources=sources, provenance='User-edited ' + (spec.catalog_key or spec.name))


def _integrate(setup, years):
    calculation = Calculation(setup, CalculationRequest(years, 1))
    while not calculation.done:
        calculation.advance(250)
    return calculation.result


def _integrate_formation(setup, years):
    from .formation import FormationExperiment
    experiment = FormationExperiment(
        setup, ejection_radius=_EJECTION_AU, sample_stride=1, adaptive=setup.adaptive)
    experiment.integrate(years, finish=True)
    return experiment


def _one_run(experiment, base, value, index, years):
    try:
        setup = _setup_for_value(experiment, base, value)
        if _uses_combined_path(experiment):
            record = _formation_record(experiment, setup, years)
        else:
            record = _yoshida_record(experiment, setup, years)
    except ValueError as exc:
        if value is None:
            raise
        raise ValueError(f'{experiment.variable.parameter}={value}: {exc}') from exc
    variable = None
    if experiment.variable is not None:
        variable = {
            'parameter': experiment.variable.parameter,
            'body': experiment.variable.body,
            'value': value,
            'unit': PARAMETER_UNITS[experiment.variable.parameter],
        }
    record['index'] = index
    record['variable'] = variable
    return record


def _yoshida_record(experiment, setup, years):
    result = _integrate(setup, years)
    primary = next(body.id for body in setup.bodies if body.id in STARS)
    return {
        'status': result.status,
        'reason': result.reason,
        'requested_years': result.requested_end_time,
        'duration_years': float(result.actual_end_time - result.frames[0].time),
        'time_start': float(result.frames[0].time),
        'time_end': float(result.actual_end_time),
        'completed_steps': result.completed_steps,
        'sample_stride': result.sample_stride,
        'setup_digest': result.setup_digest,
        'solver_build': result.solver_build,
        'dissipated_kinetic': 0.0,
        'bodies': _body_definitions(result.frames),
        'samples': [_sample(frame) for frame in result.frames],
        'events': collect_watch_events(
            result.frames, experiment.watch_for, status=result.status, reason=result.reason,
            user_collision=experiment.collision, primary_id=primary),
    }


def _formation_record(experiment, setup, years):
    result = _integrate_formation(setup, years)
    primary = next(body.id for body in setup.bodies if body.id in STARS)
    views = [_FormationView(sample) for sample in result.samples]
    wanted = set(experiment.watch_for)
    events = _formation_watch(result.events, wanted)
    events.extend(_trajectory_watch(views, wanted, primary))
    events.sort(key=lambda event: (event['time'], event['kind'], event['bodies']))
    return {
        'status': result.status,
        'reason': result.reason,
        'requested_years': float(years),
        'duration_years': float(views[-1].time - views[0].time),
        'time_start': float(views[0].time),
        'time_end': float(views[-1].time),
        'completed_steps': result.completed_steps,
        'sample_stride': result.sample_stride,
        'setup_digest': result.provenance['setup_digest'],
        'solver_build': result.provenance['solver_build'],
        'dissipated_kinetic': float(result.dissipated_kinetic),
        'bodies': _formation_bodies(result.samples),
        'samples': [_formation_sample(sample, result.dt_max) for sample in result.samples],
        'events': events,
    }


def _sweep(experiment, base):
    values = experiment.variable.values(_MAX_RUNS)
    return [_one_run(experiment, base, value, index, experiment.years) for index, value in enumerate(values)]


def _open_scan(experiment, base):
    runs = []
    stopped = 'max_runs'
    for index in range(experiment.scan.max_runs):
        value = experiment.variable.start + experiment.scan.increment * index
        run = _one_run(experiment, base, value, index, experiment.scan.years)
        runs.append(run)
        if any(event['kind'] == experiment.scan.until for event in run['events']):
            stopped = 'condition'
            break
    return runs, stopped


def _step_text(experiment):
    if experiment.adaptive:
        return (
            f'adaptive step (floor {ADAPTIVE_DT_MIN:g} yr, growth at most {ADAPTIVE_GROWTH:g})'
        )
    return 'fixed step'


def _physics_report(experiment):
    mode = experiment.collision
    engine = _engine_mode(experiment)
    combined = _uses_combined_path(experiment)
    step = _step_text(experiment)
    if combined:
        drag = 'annular' if experiment.drag_on else 'off'
        tides = 'close-pass-energy-dissipation-v1' if experiment.tides_on else 'off'
        note = (
            f'Combined formation path: collision {engine}, gas drag {drag}, tides {tides}, '
            f'{experiment.dimensions}D, {step}.')
        if mode is None:
            note += ' No collision module was selected, so contact ends the run and does not merge.'
    elif mode == 'stop_on_contact':
        note = (
            'stop_on_contact uses the existing Yoshida integrator (collision mode none). '
            'An unresolved close encounter halts the run and keeps the last good state.')
    elif mode in ('perfect_merge', 'speed_angle'):
        note = f'{mode} uses the existing contact solver. Contact does not end the run by itself.'
    else:
        note = (
            'No collision module was selected. The run uses the existing Yoshida integrator '
            'and halts if a close encounter is unresolved.')
    if experiment.adaptive and not combined:
        note += f' The close-pass schedule is an {step}.'
    elif not experiment.adaptive and not combined:
        note += ' The run uses a fixed step.'
    if experiment.tides_on and not combined:
        path = 'Yoshida path' if experiment.adaptive else 'fixed-step planar path'
        note += f' Tides use close-pass-energy-dissipation-v1 on the {path}.'
    if experiment.relativistic_on:
        note += (
            ' Relativistic corrections are on: pairwise 1PN and 2.5PN terms are added to Newtonian gravity. '
            'Perihelion precession and gravitational-wave inspiral are claimed only while this module is on.')
    else:
        note += ' Relativistic corrections are off, so this run does not claim precession or inspiral.'
    if experiment.mass_loss_on:
        note += (
            ' Stellar mass loss is on: each star loses mass at a constant fractional rate of '
            f'{MASS_LOSS_RATE_PER_YEAR:.0e} per year. The isotropic wind leaves the system and the star '
            'velocity is unchanged. An orbit widens because of mass loss only while this module is on.')
    else:
        note += ' Stellar mass loss is off, so stellar masses, including a merger remnant, stay constant.'
    return {
        'collision': mode,
        'engine_collision_mode': engine,
        'tides': experiment.tides_on,
        'gas_dust_drag': experiment.drag_on if combined else False,
        'adaptive': experiment.adaptive,
        'relativistic_corrections': experiment.relativistic_on,
        'stellar_mass_loss': experiment.mass_loss_on,
        'path': 'formation' if combined else 'yoshida',
        'note': note,
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


class _FormationView:
    """Sample shaped like the fields the escape and capture scan reads."""

    def __init__(self, sample):
        self.time = float(sample.time)
        self.body_ids = tuple(body.id for body in sample.bodies)
        self.masses = np.array([body.mass for body in sample.bodies], dtype=float)
        self.r = np.array([body.r for body in sample.bodies], dtype=float)
        self.v = np.array([body.v for body in sample.bodies], dtype=float)


def _formation_sample(sample, dt):
    return {
        'time': float(sample.time),
        'dt': float(dt),
        'phase': sample.phase,
        'ids': [body.id for body in sample.bodies],
        'mass': [float(body.mass) for body in sample.bodies],
        'r': [list(body.r) for body in sample.bodies],
        'v': [list(body.v) for body in sample.bodies],
    }


def _formation_bodies(samples):
    definitions = []
    seen = set()
    for sample in samples:
        for body in sample.bodies:
            if body.id in seen:
                continue
            seen.add(body.id)
            catalog = CATALOG.get(body.id)
            definitions.append({
                'id': body.id,
                'name': body.name,
                'kind': body.kind,
                'catalog_key': body.id if catalog is not None else None,
                'color': None if catalog is None else catalog.color,
                'radius': None if body.radius is None else float(body.radius),
                'mass': None if body.mass is None else float(body.mass),
            })
    return definitions


_FORMATION_KINDS = {
    'merge': ('merger', 'collision'),
    'fragment': ('fragment', 'collision'),
    'graze': ('graze', 'collision'),
    'unmodeled': ('collision',),
    'stop': ('collision',),
    'tide': ('tide',),
    'close_encounter': ('close_encounter',),
}


def _formation_watch(events, wanted):
    recorded = []
    for event in events:
        for kind in _FORMATION_KINDS.get(event.kind, ()):
            if kind not in wanted:
                continue
            recorded.append({
                'time': float(event.time),
                'kind': kind,
                'bodies': list(event.parent_ids),
                'message': event.message,
            })
    return recorded


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


def _engine_watch(event, wanted):
    if isinstance(event, CollisionEvent):
        kinds = _CONTACT_KINDS.get(event.kind, ())
        bodies = [body_id for body_id in event.parent_ids if isinstance(body_id, str)]
        message = event.message
        when = float(event.time)
    elif isinstance(event, TideEvent):
        kinds = ('tide',)
        bodies = [body_id for body_id in event.body_ids if isinstance(body_id, str)]
        message = event.message
        when = float(event.time)
    else:
        return []
    return [{'time': when, 'kind': kind, 'bodies': bodies, 'message': message} for kind in kinds if kind in wanted]


def _trajectory_watch(frames, wanted, primary_id):
    if not wanted.intersection(('escape', 'capture')):
        return []
    found = []
    escaped = set()
    baseline = {}
    captured = set()
    for frame in frames:
        if primary_id not in frame.body_ids:
            continue
        primary = frame.body_ids.index(primary_id)
        indexes = {body_id: index for index, body_id in enumerate(frame.body_ids)}
        for body_id, index in indexes.items():
            if body_id == primary_id or frame.masses[index] <= 0:
                continue
            if 'escape' in wanted and body_id not in escaped:
                energy = _specific_energy(frame, primary, index)
                if energy is not None and energy >= 0:
                    escaped.add(body_id)
                    found.append({
                        'time': float(frame.time),
                        'kind': 'escape',
                        'bodies': [body_id],
                        'message': f'{body_id} is unbound from {primary_id}.',
                    })
            if 'capture' not in wanted:
                continue
            for other_id, other in indexes.items():
                if other_id == body_id or frame.masses[other] <= 0:
                    continue
                energy = _specific_energy(frame, other, index)
                if energy is None:
                    continue
                key = (body_id, other_id)
                bound = energy < 0
                if key not in baseline:
                    baseline[key] = bound
                    continue
                if key not in captured and not baseline[key] and bound:
                    captured.add(key)
                    found.append({
                        'time': float(frame.time),
                        'kind': 'capture',
                        'bodies': [body_id, other_id],
                        'message': f'{body_id} is bound to {other_id} after being unbound from it.',
                    })
    return found


def _specific_energy(frame, primary, secondary):
    separation = frame.r[secondary] - frame.r[primary]
    velocity = frame.v[secondary] - frame.v[primary]
    distance = float(np.linalg.norm(separation))
    if distance == 0 or not math.isfinite(distance):
        return None
    mu = G * (float(frame.masses[primary]) + float(frame.masses[secondary]))
    energy = float(0.5 * np.dot(velocity, velocity) - mu / distance)
    if not math.isfinite(energy):
        return None
    return energy


def _encounter_halt(reason):
    text = reason.lower()
    return 'unresolved' in text or 'close encounter' in text


def _sweep_values(start, stop, step):
    if step == 0 or not math.isfinite(step):
        raise ValueError('independent_variable.step must be a nonzero finite number.')
    if start == stop:
        return (start,)
    span = stop - start
    if (span > 0) != (step > 0):
        raise ValueError('independent_variable.step must move from start toward stop.')
    count = span / step
    nearest = round(count)
    if abs(count - nearest) > 1e-8 * max(1.0, abs(count)):
        raise ValueError('independent_variable.step must land on stop.')
    runs = nearest + 1
    if runs > _MAX_RUNS:
        raise ValueError(f'This sweep is more than {_MAX_RUNS} runs. Use a wider step.')
    return tuple([start + step * index for index in range(nearest)] + [stop])


def _optional_bool(value, label):
    if value is None or type(value) is bool:
        return value
    raise ValueError(f'{label} must be true, false, or null.')


def _positive(value, label):
    number = _finite(value, label)
    if number <= 0:
        raise ValueError(f'{label} must be a finite number greater than 0.')
    return number


def _finite(value, label):
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number.')
    return float(value)


def _object(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f'{label}: expected exactly fields {", ".join(keys)}.')
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON field: {key}.')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('Nonfinite JSON number: ' + value)


def _dumps(document):
    return json.dumps(document, indent=2, allow_nan=False, ensure_ascii=True) + '\n'


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
