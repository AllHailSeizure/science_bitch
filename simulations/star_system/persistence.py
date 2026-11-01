"""Strict, versioned initial-epoch JSON. No mutable trajectories or recordings."""
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

import numpy as np

from .model import BodySpec, InitialBody, InitialSetup, PopulationSpec, PropertySource, CollisionPolicy
from .populations import Population
from .initialization import build_system

SCHEMA_VERSION = 1
SOURCE_VERSION = 'star-system-setup-v1'
MODEL_VERSION = 'planar-newtonian-point-mass-v1'
NUMERICAL_VERSION = 'fixed-yoshida4-v1'
GENERATOR_VERSION = 'planar-circular-population-v1'
UNITS = {'length': 'AU', 'mass': 'solar_mass', 'time': 'Julian_year',
         'velocity': 'AU/Julian_year', 'temperature': 'K'}


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _record(value):
    return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}


def _population(pop):
    if pop is None:
        return None
    return {'id': pop.id, 'reference_id': pop.reference_id, 'generator': 'PCG64',
            'version': GENERATOR_VERSION, 'settings': _record(pop.settings)}


def _document(setup):
    bodies = []
    for body in setup.bodies:
        spec = {field.name: getattr(body.spec, field.name) for field in fields(BodySpec) if field.name != 'sources'}
        spec['sources'] = {key: _record(source) for key, source in body.spec.sources.items()}
        bodies.append({'id': body.id, 'spec': spec, 'r': list(body.r), 'v': list(body.v),
                       'reference_id': body.reference_id})
    document = {'schema_version': SCHEMA_VERSION, 'units': UNITS, 'source_version': SOURCE_VERSION,
            'model_version': MODEL_VERSION, 'numerical_version': (setup.collision_policy.numerical if setup.collision_mode == 'perfect_merge' else NUMERICAL_VERSION),
            'environment': {'numpy': np.__version__},
            'setup': {'name': setup.name, 'preset_id': setup.preset_id, 'dt': setup.dt,
                      'collision_mode': setup.collision_mode, 'bodies': bodies,
                      'population': _population(setup.population), 'provenance': _plain(setup.provenance)}}
    # Preserve byte-for-byte canonical legacy mode-none documents/digests.
    if setup.collision_mode != 'none' or setup.collision_policy != CollisionPolicy():
        document['setup']['collision_policy'] = _record(setup.collision_policy)
    return document


def _object(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f'{label}: expected exactly fields {", ".join(sorted(keys))}.')
    return value


def _string(value, label, *, nullable=False, empty=False):
    if value is None and nullable:
        return value
    if type(value) is not str or (not empty and not value.strip()):
        raise ValueError(f'{label} must be a {"nonempty " if not empty else ""}string.')
    return value


def _number(value, label, *, nullable=False, minimum=None, positive=False):
    if value is None and nullable:
        return value
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number.')
    if (minimum is not None and value < minimum) or (positive and value <= 0):
        raise ValueError(f'{label} is outside its supported range.')
    return value


def _vector(value, label):
    if type(value) is not list or len(value) != 2:
        raise ValueError(f'{label} must contain exactly two numbers.')
    return tuple(_number(item, label) for item in value)


def _source(value):
    _object(value, {field.name for field in fields(PropertySource)}, 'Property source')
    for key in ('url', 'version', 'retrieved', 'original_units', 'conversion', 'status', 'note'):
        _string(value[key], 'Source '+key, empty=True)
    _number(value['original_value'], 'Source original_value', nullable=True)
    uncertainty = value['uncertainty']
    if type(uncertainty) is list:
        for item in uncertainty:
            _number(item, 'Source uncertainty', minimum=0)
    elif uncertainty is not None and type(uncertainty) is not str:
        _number(uncertainty, 'Source uncertainty', minimum=0)
    return PropertySource(**value)


def _spec(value):
    _object(value, {field.name for field in fields(BodySpec)}, 'Body definition')
    for key in ('name', 'kind', 'provenance'):
        _string(value[key], 'Body '+key)
    for key in ('color', 'catalog_key'):
        _string(value[key], 'Body '+key, nullable=True)
    if value['color'] is not None:
        from matplotlib.colors import is_color_like
        if not is_color_like(value['color']):
            raise ValueError('Body color must be a supported Matplotlib color.')
    for key in ('mass', 'radius', 'temperature'):
        _number(value[key], 'Body '+key, nullable=True)
    if type(value['sources']) is not dict or any(key not in ('mass', 'radius', 'temperature') for key in value['sources']):
        raise ValueError('Unknown physical source property.')
    if value['kind'] == 'particle' and value['mass'] != 0:
        raise ValueError('Test particles must have mass zero.')
    if value['kind'] != 'particle' and value['mass'] == 0:
        raise ValueError('Only test particles may have mass zero.')
    return BodySpec(**{**value, 'sources': {key: _source(source) for key, source in value['sources'].items()}})


def _decode_population(value):
    if value is None:
        return None
    _object(value, ('id', 'reference_id', 'generator', 'version', 'settings'), 'Population')
    if value['generator'] != 'PCG64' or value['version'] != GENERATOR_VERSION:
        raise ValueError('Unsupported population generator/version.')
    _string(value['id'], 'Population ID'); _string(value['reference_id'], 'Population reference')
    settings = _object(value['settings'], {field.name for field in fields(Population)}, 'Population settings')
    for key in ('count', 'seed', 'reference'):
        if type(settings[key]) is not int:
            raise ValueError(f'Population {key} must be an integer.')
    if settings['reference'] != 0:
        raise ValueError('Authored population numeric reference must be canonical zero; use reference_id.')
    for key in ('inner', 'outer'):
        _number(settings[key], 'Population '+key, positive=True)
    for key in ('kind', 'angles'):
        _string(settings[key], 'Population '+key)
    return PopulationSpec(value['id'], value['reference_id'], Population(**settings))


def _decode(document):
    _object(document, ('schema_version', 'units', 'source_version', 'model_version', 'numerical_version', 'environment', 'setup'), 'Document')
    if type(document['schema_version']) is not int or document['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported setup schema version; expected 1.')
    for key, expected in (('units', UNITS), ('source_version', SOURCE_VERSION),
                          ('model_version', MODEL_VERSION)):
        if document[key] != expected:
            raise ValueError(f'Unsupported {key}.')
    _object(document['environment'], ('numpy',), 'Environment')
    _string(document['environment']['numpy'], 'NumPy version')
    keys = {'name', 'preset_id', 'dt', 'collision_mode', 'bodies', 'population', 'provenance'}
    if type(document['setup']) is dict and 'collision_policy' in document['setup']:
        keys.add('collision_policy')
    value = _object(document['setup'], keys, 'Setup')
    _string(value['name'], 'Setup name'); _string(value['preset_id'], 'Preset ID', nullable=True)
    _string(value['collision_mode'], 'Collision mode')
    if value['collision_mode'] == 'perfect_merge' and 'collision_policy' not in value:
        raise ValueError('Physical collision mode requires an explicit versioned collision policy.')
    policy = CollisionPolicy(**_object(value['collision_policy'], {field.name for field in fields(CollisionPolicy)}, 'Collision policy')) if 'collision_policy' in value else CollisionPolicy()
    if document['numerical_version'] != (policy.numerical if value['collision_mode'] == 'perfect_merge' else NUMERICAL_VERSION):
        raise ValueError('Unsupported numerical_version for selected collision mode.')
    _number(value['dt'], 'Fixed dt', positive=True)
    if type(value['bodies']) is not list or not value['bodies']:
        raise ValueError('Setup bodies must be a nonempty list.')
    if type(value['provenance']) is not dict:
        raise ValueError('Setup provenance must be an object.')
    bodies = []
    for body in value['bodies']:
        _object(body, ('id', 'spec', 'r', 'v', 'reference_id'), 'Initial body')
        bodies.append(InitialBody(_string(body['id'], 'Body ID'), _spec(body['spec']),
                                  _vector(body['r'], 'Position'), _vector(body['v'], 'Velocity'),
                                  _string(body['reference_id'], 'Body reference', nullable=True)))
    setup = InitialSetup(value['name'], tuple(bodies), value['dt'], value['preset_id'],
                         _decode_population(value['population']), value['collision_mode'], value['provenance'], policy)
    # Validate references, overlap, known masses and numerical resolution before install.
    build_system(setup)
    return setup


def _canonical(setup):
    # Apply the identical strict codec to programmatic saves as to external loads.
    document = _document(setup)
    encoded = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=True)
    _decode(json.loads(encoded))
    return encoded


def setup_digest(setup):
    """Stable SHA-256 identity for the full configured epoch and its provenance."""
    return hashlib.sha256(_canonical(setup).encode('utf-8')).hexdigest()


def save_setup(setup, path):
    """Validate first, then atomically replace through one owned sibling tempfile."""
    encoded = _canonical(setup)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.'+path.name+'.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON field: {key}.')
        result[key] = value
    return result


def load_setup(path):
    """Return a fully validated setup, without touching an active experiment."""
    try:
        with Path(path).open(encoding='utf-8') as handle:
            document = json.load(handle, object_pairs_hook=_pairs,
                                 parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON number: '+value)))
        return _decode(document)
    except (TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError(f'Invalid setup JSON: {exc}') from exc
