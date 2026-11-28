"""Regular presets and experiment-file sweeps integrate in three dimensions."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from simulations.star_system.core import ADAPTIVE_DT_MIN, ADAPTIVE_GROWTH, adaptive_macro_step, make_system
from simulations.star_system.data import AU_KM
from simulations.star_system.experiment_file import main
from simulations.star_system.frames import frame_transform
from simulations.star_system.history import StateFrame
from simulations.star_system.initialization import (
    PRESET_INCLINATION, build_system, legacy_setup, recommended_moon_step,
)
from simulations.star_system.model import BodySpec, InitialBody, InitialSetup
from simulations.star_system.persistence import GENERATOR_VERSION, SPATIAL_GENERATOR_VERSION, MODEL_VERSION, save_setup
from simulations.star_system.recording import Calculation, CalculationRequest
from simulations.star_system.rendering import draw_system
from simulations.star_system.scenarios import DESCRIPTIONS, get_scenario
from simulations.star_system.templates import EXPERIMENTS, get_experiment


def _text(setup):
    return ' '.join(str(setup.provenance.get(key, '')) for key in (
        'observe', 'description', 'model_assumption')).lower()


@pytest.mark.parametrize('key', [item.key for item in EXPERIMENTS])
def test_regular_presets_are_three_dimensional_and_say_so(key):
    setup = get_experiment(key)
    assert all(len(body.r) == 3 and len(body.v) == 3 for body in setup.bodies)
    assert setup.provenance['dimensions'] == 3
    assert setup.provenance['inclination_rad'] == pytest.approx(PRESET_INCLINATION)
    assert setup.adaptive is False
    assert setup.provenance['adaptive'] is False
    text = _text(setup)
    assert 'three-dimensional' in text or 'three dimensions' in text
    assert 'adaptive' in text
    prepared = build_system(setup)
    assert prepared.system.r.shape[1] == 3
    assert np.any(np.abs(prepared.system.v[:, 2]) > 0) or np.any(np.abs(prepared.system.r[:, 2]) > 0)


@pytest.mark.parametrize('key', list(DESCRIPTIONS))
def test_named_scenarios_use_the_same_disclosed_inclination(key):
    setup = get_scenario(key)
    assert all(len(body.r) == 3 and len(body.v) == 3 for body in setup.bodies)
    assert setup.provenance['dimensions'] == 3
    assert setup.provenance['inclination_rad'] == pytest.approx(PRESET_INCLINATION)
    assert setup.adaptive is False
    assert 'three-dimensional' in _text(setup)
    assert 'adaptive' in _text(setup)


def test_preset_motion_leaves_the_plane_and_the_moon_keeps_its_period():
    solar = build_system(get_experiment('solar-system'))
    assert np.max(np.abs(solar.system.v[:, 2])) > 0
    before = solar.system.r.copy()
    solar.system.step(2)
    assert solar.system.time > 0
    assert not np.allclose(solar.system.r, before)

    earth_moon = build_system(get_experiment('earth-moon'))
    earth = earth_moon.body_ids.index('earth')
    moon = earth_moon.body_ids.index('moon')
    separation = np.linalg.norm(earth_moon.system.r[moon] - earth_moon.system.r[earth])
    assert separation == pytest.approx(384400 / AU_KM, rel=1e-6)
    relative = earth_moon.system.r[moon] - earth_moon.system.r[earth]
    limit = recommended_moon_step(relative, earth_moon.system.masses[earth], earth_moon.system.masses[moon])
    assert earth_moon.system.dt <= limit * (1 + 1e-12)
    planar = recommended_moon_step((separation, 0.), earth_moon.system.masses[earth], earth_moon.system.masses[moon])
    assert recommended_moon_step(relative, earth_moon.system.masses[earth], earth_moon.system.masses[moon]) == pytest.approx(planar)

    ring = build_system(get_experiment('asteroid-ring'))
    sun = ring.body_ids.index('sun')
    particles = [index for index, spec in enumerate(ring.body_specs) if spec.kind == 'particle']
    relative_ring = ring.system.r[particles] - ring.system.r[sun]
    assert relative_ring.shape[1] == 3
    assert np.allclose(relative_ring[:, 2], 0, atol=1e-12)
    assert np.allclose(np.linalg.norm(relative_ring, axis=1), 3.2, rtol=1e-6)

    jupiter = build_system(get_experiment('sun-jupiter-ring'))
    sun = jupiter.body_ids.index('sun')
    rocks = [index for index, spec in enumerate(jupiter.body_specs) if spec.kind == 'particle']
    relative_rocks = jupiter.system.r[rocks] - jupiter.system.r[sun]
    assert np.allclose(relative_rocks[:, 2], 0, atol=1e-12)
    assert np.allclose(np.linalg.norm(relative_rocks, axis=1), 3.2, rtol=1e-6)

    merger = build_system(get_experiment('binary-star-merger'))
    assert merger.system.r[0, 2] == pytest.approx(-merger.system.r[1, 2])
    assert abs(merger.system.r[0, 2]) > 0
    np.testing.assert_allclose(merger.system.v, 0)
    separation = float(np.linalg.norm(merger.system.r[0] - merger.system.r[1]))
    merger.system.step(2)
    assert merger.system.time > 0
    assert abs(float(merger.system.v[0, 2])) > 0
    assert float(np.linalg.norm(merger.system.r[0] - merger.system.r[1])) < separation


def test_planar_helpers_and_recordings_are_not_padded(tmp_path):
    system = make_system()
    assert system.r.shape[1] == 2 and system.v.shape[1] == 2
    planar = legacy_setup()
    assert all(len(body.r) == 2 and len(body.v) == 2 for body in planar.bodies)
    path = tmp_path / 'planar.json'
    save_setup(planar, path)
    document = json.loads(path.read_text(encoding='utf-8'))
    assert document['model_version'] == MODEL_VERSION
    assert 'adaptive' not in document['setup']
    assert document['setup']['population'] is None
    frame = StateFrame.capture(build_system(planar))
    shown = frame_transform(frame, 'world').positions(frame.r)
    assert shown.shape == (len(frame.r), 2)
    assert frame.r.shape[1] == 2
    spatial = get_experiment('solar-system')
    spatial_path = tmp_path / 'spatial.json'
    save_setup(spatial, spatial_path)
    spatial_document = json.loads(spatial_path.read_text(encoding='utf-8'))
    assert 'adaptive' not in spatial_document['setup']
    preset = StateFrame.capture(build_system(spatial))
    original = preset.r.copy()
    projected = frame_transform(preset, 'world').positions(preset.r)
    assert projected.shape == (len(preset.r), 2)
    np.testing.assert_allclose(projected, preset.r[:, :2])
    np.testing.assert_array_equal(preset.r, original)
    figure, axes = plt.subplots()
    draw_system(axes, build_system(get_experiment('earth-moon')))
    plt.close(figure)


def test_population_generator_id_follows_the_setup_dimension(tmp_path):
    from simulations.star_system.populations import Population
    spatial = get_scenario('sun-jupiter-ring')
    spatial_path = tmp_path / 'ring.json'
    save_setup(spatial, spatial_path)
    spatial_document = json.loads(spatial_path.read_text(encoding='utf-8'))
    assert spatial_document['setup']['population']['version'] == SPATIAL_GENERATOR_VERSION
    planar = legacy_setup(planets=('jupiter',), population=Population(
        kind='ring', inner=3.2, outer=3.2, count=8, angles='even', seed=7))
    planar_path = tmp_path / 'planar-ring.json'
    save_setup(planar, planar_path)
    planar_document = json.loads(planar_path.read_text(encoding='utf-8'))
    assert planar_document['setup']['population']['version'] == GENERATOR_VERSION
    assert all(len(body['r']) == 2 for body in planar_document['setup']['bodies'])


def test_three_dimensional_sweep_keeps_the_shared_base_and_duration(tmp_path):
    source = tmp_path / 'experiment.json'
    source.write_text(json.dumps({
        'schema': 'sidereel-experiment-v1',
        'dimensions': 3,
        'gravity_model': 'newtonian',
        'bodies': {'catalog': ['sun', 'earth', 'jupiter'], 'new': None},
        'physics': {'collision': None, 'tides': False, 'gas_dust_drag': False},
        'experiment': {
            'independent_variable': {
                'parameter': 'mass', 'body': 'sun', 'start': 1, 'stop': 1.2, 'step': 0.2,
            },
            'fixed_duration': {'years': 0.005},
            'open_scan': None,
            'watch_for': [],
        },
    }), encoding='utf-8')
    assert main(['--build', str(source)]) == 0
    playback = json.loads((tmp_path / 'experiment.sidereel').read_text(encoding='utf-8'))
    assert playback['dimensions'] == 3
    assert playback['inclination_rad'] == pytest.approx(PRESET_INCLINATION)
    assert 'not given a zero z' in playback['gravity_note']
    assert playback['requested_years'] == pytest.approx(0.005)
    assert len(playback['runs']) == 2
    first, second = playback['runs']
    assert first['status'] == second['status'] == 'complete'
    assert first['requested_years'] == pytest.approx(0.005)
    ids = first['samples'][0]['ids']
    earth = ids.index('earth')
    jupiter = ids.index('jupiter')
    assert len(first['samples'][0]['r'][earth]) == 3
    np.testing.assert_allclose(first['samples'][0]['r'][earth], second['samples'][0]['r'][earth])
    np.testing.assert_allclose(first['samples'][0]['v'][jupiter], second['samples'][0]['v'][jupiter])
    assert any(abs(row[2]) > 0 for row in first['samples'][0]['v'])
    assert playback['physics']['adaptive'] is False
    assert 'fixed step' in playback['physics']['note']


def test_experiment_file_can_select_the_adaptive_step(tmp_path):
    source = tmp_path / 'experiment.json'
    document = {
        'schema': 'sidereel-experiment-v1',
        'dimensions': 3,
        'gravity_model': 'newtonian',
        'bodies': {'catalog': ['sun', 'earth'], 'new': None},
        'physics': {'collision': None, 'tides': False, 'gas_dust_drag': False},
        'experiment': {
            'independent_variable': None,
            'fixed_duration': {'years': 0.005},
            'open_scan': None,
            'watch_for': [],
            'adaptive': True,
        },
    }
    source.write_text(json.dumps(document), encoding='utf-8')
    assert main(['--build', str(source)]) == 0
    playback = json.loads((tmp_path / 'experiment.sidereel').read_text(encoding='utf-8'))
    note = playback['physics']['note']
    assert playback['physics']['adaptive'] is True
    assert playback['physics']['path'] == 'yoshida'
    assert 'adaptive' in note
    assert f'{ADAPTIVE_DT_MIN:g}' in note
    assert f'{ADAPTIVE_GROWTH:g}' in note
    assert playback['experiment']['experiment']['adaptive'] is True
    steps = [sample['dt'] for sample in playback['runs'][0]['samples']]
    assert all(step == pytest.approx(steps[0]) for step in steps)
    assert playback['runs'][0]['status'] == 'complete'


def test_adaptive_macro_shrinks_on_a_close_pass_and_halts_below_the_floor():
    limit_above = 0.01
    assert adaptive_macro_step(0.002, 0.02, ADAPTIVE_DT_MIN, limit_above) == pytest.approx(0.004)
    assert adaptive_macro_step(0.02, 0.02, ADAPTIVE_DT_MIN, 1.) == pytest.approx(0.02)
    assert adaptive_macro_step(1e-12, 0.02, ADAPTIVE_DT_MIN, 1e-16) is None

    spec = BodySpec('Passer', 'planet', 1e-6, 1e-6)
    setup = InitialSetup('Close pass', (
        InitialBody('a', spec, (0., 0., 0.), (0., 0., 0.)),
        InitialBody('b', spec, (0.2, 0.01, 0.), (-1., 0., 0.)),
    ), dt=0.01, adaptive=True)
    calculation = Calculation(setup, CalculationRequest(0.25, 1))
    authored = calculation.dt_max
    for _ in range(40):
        if calculation.done:
            break
        calculation.advance(10)
    assert calculation.status == 'complete'
    seen = [frame.dt for frame in calculation.result.frames]
    assert min(seen) < authored
    assert min(seen) >= ADAPTIVE_DT_MIN

    wide = InitialSetup('Below the floor', (
        InitialBody('a', BodySpec('A', 'planet', 1e-6, 1e-4), (0., 0., 0.), (0., 0., 0.)),
        InitialBody('b', BodySpec('B', 'planet', 1e-6, 1e-4), (1., 0., 0.), (0., 0., 0.)),
    ), dt=0.01, adaptive=True)
    halted = Calculation(wide, CalculationRequest(0.05, 1))
    halted.prepared.system.r[1] = (1e-8, 0., 0.)
    halted.prepared.system.v[1] = (-100., 0., 0.)
    assert halted.prepared.system.resolution_limit() < ADAPTIVE_DT_MIN
    before = halted.prepared.system.r.copy(), halted.prepared.system.v.copy(), halted.prepared.system.time
    halted.advance(1)
    assert halted.status == 'halted'
    assert 'floor' in halted.reason
    np.testing.assert_array_equal(halted.prepared.system.r, before[0])
    np.testing.assert_array_equal(halted.prepared.system.v, before[1])
    assert halted.prepared.system.time == before[2]
