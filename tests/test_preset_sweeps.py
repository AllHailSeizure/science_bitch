"""Preset sweep files: every current preset, plus the capture example."""
import json
from pathlib import Path

import pytest

from simulations.star_system.data import CATALOG
from simulations.star_system.experiment_file import load_experiment, main
from simulations.star_system.player import Player
from simulations.star_system.preset_sweeps import (
    PRESET_KEYS, build_preset_playback, document, menu_preset_keys, write_all,
)

ROOT = Path(__file__).resolve().parents[1]


def test_every_current_preset_has_a_sweep_file_and_capture_is_only_an_example():
    assert menu_preset_keys() <= set(PRESET_KEYS)
    assert 'capture-flyby' in PRESET_KEYS
    assert 'capture-flyby' not in menu_preset_keys()
    for key in menu_preset_keys():
        assert (ROOT / 'experiments' / f'{key}.json').is_file()


def test_shipped_files_match_the_cli_writer(tmp_path):
    written = {path.name: json.loads(path.read_text(encoding='utf-8')) for path in write_all(tmp_path)}
    for key in PRESET_KEYS:
        assert written[f'{key}.json'] == document(key)
        assert json.loads((ROOT / 'experiments' / f'{key}.json').read_text(encoding='utf-8')) == document(key)


def test_catalog_presets_load_as_experiment_files_without_integrating():
    solar = load_experiment(ROOT / 'experiments' / 'solar-system.json')
    assert solar.variable.values(5) == (1.0,)
    assert solar.catalog == ('sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')
    heavier = load_experiment(ROOT / 'experiments' / 'heavier-star.json')
    assert list(heavier.variable.values(20)) == pytest.approx([1 + 0.1 * index for index in range(11)])
    pair = load_experiment(ROOT / 'experiments' / 'sun-earth-jupiter.json')
    assert pair.catalog == ('sun', 'earth', 'jupiter')
    assert pair.variable.values(5) == (1.0,)


def test_build_plays_a_capture_sweep_from_the_recording(tmp_path):
    source = dict(document('capture-flyby'))
    source['experiment'] = dict(source['experiment'], fixed_duration={'years': 0.02})
    path = tmp_path / 'capture-flyby.json'
    path.write_text(json.dumps(source), encoding='utf-8')
    assert main(['--build', str(path)]) == 0
    playback = json.loads((tmp_path / 'capture-flyby.sidereel').read_text(encoding='utf-8'))
    assert playback['schema'] == 'sidereel-playback-v1'
    assert [run['variable']['value'] for run in playback['runs']] == pytest.approx([0.5, 1.0, 1.5, 2.0])
    assert {run['base_setup_digest'] for run in playback['runs']} == {playback['base_setup']['digest']}
    first, second = playback['runs'][0], playback['runs'][1]
    ids = first['samples'][0]['ids']
    sun, jupiter, tracer = (ids.index(name) for name in ('sun', 'jupiter', 'tracer'))
    assert first['samples'][0]['r'][sun] == pytest.approx(second['samples'][0]['r'][sun])
    assert first['samples'][0]['r'][jupiter] == pytest.approx(second['samples'][0]['r'][jupiter])
    assert first['samples'][0]['v'][tracer] != pytest.approx(second['samples'][0]['v'][tracer])
    player = Player(playback)
    player.advance(0.01)
    assert player.time > 0
    assert playback['runs'][0]['samples'][0]['r'][tracer] == pytest.approx(first['samples'][0]['r'][tracer])


def test_earth_moon_and_binary_merger_keep_their_shapes(tmp_path):
    moon = dict(document('earth-moon'))
    moon['experiment'] = dict(moon['experiment'], fixed_duration={'years': 0.01})
    playback = build_preset_playback(moon)
    assert 'gravity_model' not in playback
    run = playback['runs'][0]
    assert run['variable']['value'] == pytest.approx(CATALOG['earth'].mass)
    assert run['collision_mode'] == 'none'
    assert run['base_setup_digest'] == playback['base_setup']['digest']
    kinds = {body['id']: body['kind'] for body in run['bodies']}
    assert kinds['moon'] == 'moon'
    merger = dict(document('binary-star-merger'))
    merger['experiment'] = dict(merger['experiment'], fixed_duration={'years': 0.001})
    built = build_preset_playback(merger)
    assert len(built['runs']) == 1
    assert built['runs'][0]['variable'] is None
    assert built['runs'][0]['collision_mode'] == 'binary_star_merger'
    assert built['scan']['mode'] == 'fixed'


def test_jupiter_mass_sweep_keeps_the_ring_and_drops_jupiter_at_zero():
    source = dict(document('sun-jupiter-ring'))
    source['experiment'] = dict(source['experiment'], fixed_duration={'years': 0.01})
    playback = build_preset_playback(source)
    masses = [run['variable']['value'] for run in playback['runs']]
    assert masses[0] == 0
    assert masses[-1] == pytest.approx(CATALOG['jupiter'].mass)
    assert len(masses) == 6
    empty = playback['runs'][0]['samples'][0]
    full = playback['runs'][-1]['samples'][0]
    assert 'jupiter' not in empty['ids']
    assert 'jupiter' in full['ids']
    rock = next(body_id for body_id in empty['ids'] if body_id.startswith('population:'))
    assert empty['r'][empty['ids'].index(rock)] == pytest.approx(full['r'][full['ids'].index(rock)])
    assert all(run['collision_mode'] == 'none' for run in playback['runs'])


def test_ring_radius_changes_only_the_ring():
    source = dict(document('asteroid-ring'))
    experiment = dict(source['experiment'])
    experiment['fixed_duration'] = {'years': 0.005}
    experiment['independent_variable'] = dict(experiment['independent_variable'], start=3.0, stop=3.2, step=0.2)
    source['experiment'] = experiment
    playback = build_preset_playback(source)
    first, second = (run['samples'][0] for run in playback['runs'])
    earth = first['ids'].index('earth')
    assert first['r'][earth] == pytest.approx(second['r'][earth])
    rock = next(body_id for body_id in first['ids'] if body_id.startswith('asteroids:'))
    sun = first['ids'].index('sun')
    import numpy as np
    near = np.linalg.norm(np.array(first['r'][first['ids'].index(rock)]) - np.array(first['r'][sun]))
    far = np.linalg.norm(np.array(second['r'][second['ids'].index(rock)]) - np.array(second['r'][sun]))
    assert near == pytest.approx(3.0, abs=1e-6)
    assert far == pytest.approx(3.2, abs=1e-6)


def test_preset_build_refuses_a_different_variable():
    source = document('capture-flyby')
    source['experiment']['independent_variable'] = dict(source['experiment']['independent_variable'], body='jupiter')
    with pytest.raises(ValueError, match='tracer'):
        build_preset_playback(source)
