"""Saved setups are complete initial epochs, independent of catalog/restart state."""
from dataclasses import replace
import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from simulations.star_system.data import CATALOG, AU_KM
from simulations.star_system.initialization import build_system, legacy_setup, insert_body, circular_relative_velocity
from simulations.star_system.model import InitialBody, InitialSetup, PopulationSpec
from simulations.star_system.populations import Population
from simulations.star_system.persistence import save_setup, load_setup, setup_digest


def fixtures():
    ordinary = legacy_setup()
    binary = InitialSetup('Binary', (InitialBody('a', CATALOG['sun'], (-3, 1), (0, -1)),
                                   InitialBody('b', CATALOG['sun'], (3, 1), (0, 1))), .0001)
    population = replace(binary, population=PopulationSpec('belt', 'b', Population(count=12, seed=42)))
    separation = 384400 / AU_KM
    moon = InitialBody('moon', CATALOG['moon'], (separation, 0),
                       tuple(.99*circular_relative_velocity((separation, 0), CATALOG['earth'].mass, CATALOG['moon'].mass)), 'earth')
    moons = InitialSetup('Edited Moon', (InitialBody('earth', CATALOG['earth'], (3, 2), (.2, -.1)), moon), .00001)
    return ordinary, binary, population, moons


@pytest.mark.parametrize('setup', fixtures(), ids=['ordinary', 'binary', 'second-star-population', 'edited-moon'])
def test_roundtrip_initial_arrays_sources_and_fifty_steps(tmp_path, setup):
    path = tmp_path/'setup.json'
    save_setup(setup, path)
    loaded = load_setup(path)
    assert setup_digest(setup) == setup_digest(loaded)
    assert loaded.population == setup.population
    for a, b in zip(setup.bodies, loaded.bodies):
        assert dict(a.spec.sources) == dict(b.spec.sources)
    a, b = build_system(setup), build_system(loaded)
    assert a.body_ids == b.body_ids and a.reference_ids == b.reference_ids
    for key in ('masses', 'r', 'v'):
        np.testing.assert_array_equal(getattr(a.system, key), getattr(b.system, key))
    a.system.step(50); b.system.step(50)
    np.testing.assert_allclose(a.system.r, b.system.r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.system.v, b.system.v, rtol=1e-12, atol=1e-12)


def test_subprocess_load_is_independent(tmp_path):
    setup = replace(fixtures()[3], population=PopulationSpec('ring', 'earth', Population('ring', 8, .01, .01, 9)))
    path, image = tmp_path/'setup.json', tmp_path/'loaded.png'
    save_setup(setup, path)
    result = subprocess.run([sys.executable, '-m', 'simulations.star_system', '--load', str(path),
                             '--years', '.01', '--output', str(image)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with Image.open(image) as png:
        assert png.format == 'PNG' and png.width > 100 and png.height > 100


def test_save_creates_explicit_parent_path_after_validation(tmp_path):
    path = tmp_path/'new-output'/'setup.json'
    save_setup(fixtures()[0], path)
    assert setup_digest(load_setup(path)) == setup_digest(fixtures()[0])


@pytest.mark.parametrize('mutate', [
    lambda d: d.update(schema_version=2),
    lambda d: d.update(schema_version=True),
    lambda d: d.update(model_version='unknown'),
    lambda d: d.update(numerical_version='adaptive'),
    lambda d: d.update(source_version='unknown'),
    lambda d: d.update(extra_setting=True),
    lambda d: d['units'].update(length='km'),
    lambda d: d['setup'].update(extra_setting=True),
    lambda d: d['setup'].update(collision_mode='merge'),
    lambda d: d['setup'].update(dt=True),
    lambda d: d['setup'].update(dt=0),
    lambda d: d['setup'].update(dt=float('inf')),
    lambda d: d['setup']['bodies'][0].update(r=[0, 0, 0]),
    lambda d: d['setup']['bodies'][0].update(r=['1', 0]),
    lambda d: d['setup']['bodies'][0].update(reference_id='missing'),
    lambda d: d['setup']['bodies'][0]['spec'].update(mass=True),
    lambda d: d['setup']['bodies'][0]['spec'].update(mass=None),
    lambda d: d['setup']['bodies'][0]['spec'].update(radius=-1),
    lambda d: d['setup']['bodies'][0]['spec'].update(color='invalid-color'),
    lambda d: d['setup']['bodies'][0]['spec']['sources']['mass'].update(original_value=float('nan')),
    lambda d: d['setup']['bodies'][0]['spec']['sources']['mass'].update(extra_field=True),
    lambda d: d['setup']['bodies'][0]['spec'].update(new_physics=1),
    lambda d: d['setup']['bodies'][1].update(id='sun'),
])
def test_invalid_or_unknown_settings_rejected(tmp_path, mutate):
    path = tmp_path/'setup.json'; save_setup(fixtures()[0], path)
    document = json.loads(path.read_text()); mutate(document); path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_setup(path)


@pytest.mark.parametrize('content', ["__import__('os').system('no')", '{', 'null', '[]',
                                   '{"schema_version":1,"schema_version":1}',
                                   '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'])
def test_non_json_duplicate_and_nonfinite_rejected(tmp_path, content):
    path = tmp_path/'bad.json'; path.write_text(content)
    with pytest.raises(ValueError): load_setup(path)


@pytest.mark.parametrize('key,value', [('generator', 'unknown'), ('version', 'unknown'),
                                     ('settings', {'bad': 'ignored?'})])
def test_unknown_population_configuration_rejected(tmp_path, key, value):
    path = tmp_path/'setup.json'; save_setup(fixtures()[2], path)
    document = json.loads(path.read_text()); document['setup']['population'][key] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError): load_setup(path)


@pytest.mark.parametrize('operation', ['replace', 'fsync'])
def test_failed_atomic_replace_and_invalid_save_preserve_destination(tmp_path, monkeypatch, operation):
    import simulations.star_system.persistence as codec
    path = tmp_path/'setup.json'; original = fixtures()[0]; save_setup(original, path)
    content = path.read_bytes()
    unrelated = tmp_path/'.setup.json.someone-else.tmp'; unrelated.write_text('preserve')
    with pytest.raises(ValueError):
        save_setup(replace(original, bodies=(replace(original.bodies[0], reference_id='missing'),)), path)
    assert path.read_bytes() == content
    def fail(*args): raise OSError('interrupted replace')
    monkeypatch.setattr(codec.os, operation, fail)
    with pytest.raises(OSError, match='interrupted'):
        save_setup(replace(original, name='new'), path)
    assert path.read_bytes() == content and load_setup(path).name == original.name
    assert sorted(file.name for file in tmp_path.iterdir()) == sorted([path.name, unrelated.name])


@pytest.mark.parametrize('flag', ['--star=sun', '--planet=earth', '--population=ring', '--count=200',
                                 '--inner=2', '--outer=3.5', '--seed=0', '--angles=random', '--sta=sun', '--co=200'])
def test_cli_load_rejects_explicit_construction_defaults(flag):
    from simulations.star_system.__main__ import main
    with pytest.raises(SystemExit) as error:
        main(['--load', 'irrelevant.json', flag, '--output', 'irrelevant.png'])
    assert error.value.code == 2
