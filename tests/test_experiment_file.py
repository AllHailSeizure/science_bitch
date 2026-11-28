"""CLI experiment files: a blank JSON template and a .sidereel playback file."""
import json
from dataclasses import replace

import numpy as np
import pytest

from simulations.star_system.experiment_file import main


def _write(path, *, dimensions=2, gravity='newtonian', bodies=('sun', 'earth', 'jupiter'), collision=None,
           tides=False, drag=False, variable=None, years=0.01, scan=None, watch=(), custom=None):
    document = {
        'schema': 'sidereel-experiment-v1',
        'dimensions': dimensions,
        'gravity_model': gravity,
        'bodies': {'catalog': list(bodies), 'new': custom},
        'physics': {'collision': collision, 'tides': tides, 'gas_dust_drag': drag},
        'experiment': {
            'independent_variable': variable,
            'fixed_duration': None if scan is not None else {'years': years},
            'open_scan': scan,
            'watch_for': list(watch),
        },
    }
    path.write_text(json.dumps(document), encoding='utf-8')
    return path


def _build(tmp_path, **kwargs):
    source = _write(tmp_path / 'experiment.json', **kwargs)
    assert main(['--build', str(source)]) == 0
    output = tmp_path / 'experiment.sidereel'
    assert output.is_file()
    return json.loads(output.read_text(encoding='utf-8')), output


def _refuse(tmp_path, capsys, **kwargs):
    source = _write(tmp_path / 'experiment.json', **kwargs)
    assert main(['--build', str(source)]) == 2
    assert not (tmp_path / 'experiment.sidereel').exists()
    return capsys.readouterr().err


def test_new_writes_every_section_blank_and_in_order(tmp_path):
    path = tmp_path / 'blank.json'
    assert main(['--new', str(path)]) == 0
    document = json.loads(path.read_text(encoding='utf-8'))
    assert list(document) == ['schema', 'dimensions', 'gravity_model', 'bodies', 'physics', 'experiment']
    assert document['schema'] == 'sidereel-experiment-v1'
    assert document['dimensions'] is None
    assert document['gravity_model'] is None
    assert list(document['bodies']) == ['catalog', 'new']
    assert document['bodies'] == {'catalog': [], 'new': None}
    assert list(document['physics']) == ['collision', 'tides', 'gas_dust_drag']
    assert document['physics'] == {'collision': None, 'tides': None, 'gas_dust_drag': None}
    assert list(document['experiment']) == ['independent_variable', 'fixed_duration', 'open_scan', 'watch_for']
    assert document['experiment'] == {
        'independent_variable': None, 'fixed_duration': None, 'open_scan': None, 'watch_for': []}


def test_new_refuses_to_overwrite(tmp_path, capsys):
    path = tmp_path / 'experiment.json'
    assert main(['--new', str(path)]) == 0
    original = path.read_text(encoding='utf-8')
    assert main(['--new', str(path)]) == 2
    assert 'already exists' in capsys.readouterr().err
    assert path.read_text(encoding='utf-8') == original


def test_sidereel_new_does_not_open_the_desktop(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError('desktop opened')

    monkeypatch.setattr('simulations.star_system.desktop.QApplication', boom)
    from simulations.star_system.desktop import main as desktop_main
    path = tmp_path / 'experiment.json'
    assert desktop_main(['--new', str(path)]) == 0
    assert json.loads(path.read_text(encoding='utf-8'))['schema'] == 'sidereel-experiment-v1'


def test_sidereel_cli_build_writes_the_playback_file(tmp_path):
    source = _write(tmp_path / 'experiment.json', bodies=('sun', 'earth'), years=0.005)
    from simulations.star_system.__main__ import main as cli_main
    assert cli_main(['--build', str(source)]) == 0
    playback = json.loads((tmp_path / 'experiment.sidereel').read_text(encoding='utf-8'))
    assert playback['schema'] == 'sidereel-playback-v1'
    assert playback['encoding'] == 'json'
    assert playback['source_name'] == 'experiment.json'


def test_build_of_a_blank_file_asks_for_dimensions(tmp_path, capsys):
    path = tmp_path / 'experiment.json'
    assert main(['--new', str(path)]) == 0
    assert main(['--build', str(path)]) == 2
    assert 'dimensions' in capsys.readouterr().err
    assert not (tmp_path / 'experiment.sidereel').exists()


def test_build_rejects_relativistic_gravity(tmp_path, capsys):
    error = _refuse(tmp_path, capsys, gravity='relativistic')
    assert 'not implemented yet' in error
    assert 'newtonian' in error


def test_build_rejects_a_custom_body(tmp_path, capsys):
    error = _refuse(tmp_path, capsys, custom={'name': 'rock'})
    assert 'not implemented yet' in error
    assert 'bodies.new' in error


def test_build_rejects_a_catalog_body_without_a_placed_orbit(tmp_path, capsys):
    error = _refuse(tmp_path, capsys, bodies=('sun', 'moon'))
    assert 'not implemented yet' in error


def test_build_rejects_ignore(tmp_path, capsys):
    error = _refuse(tmp_path, capsys, collision='ignore')
    assert 'not implemented yet' in error
    assert 'ignore' in error


@pytest.mark.parametrize('collision', ['perfect_merge', 'speed_angle'])
def test_build_rejects_tides_with_a_contact_rule_and_names_84(tmp_path, capsys, collision):
    error = _refuse(tmp_path, capsys, collision=collision, tides=True)
    assert '#84' in error
    assert collision in error


def test_build_rejects_tides_in_three_dimensions_and_names_84(tmp_path, capsys):
    error = _refuse(tmp_path, capsys, dimensions=3, bodies=('sun', 'earth'), tides=True, collision='stop_on_contact')
    assert '#84' in error


def test_build_rejects_gas_dust_drag_and_names_84(tmp_path, capsys):
    error = _refuse(tmp_path, capsys, drag=True, collision='perfect_merge')
    assert '#84' in error
    assert 'gas' in error.lower()


def test_build_requires_a_duration_or_an_open_scan(tmp_path, capsys):
    source = _write(tmp_path / 'experiment.json')
    document = json.loads(source.read_text(encoding='utf-8'))
    document['experiment']['fixed_duration'] = None
    source.write_text(json.dumps(document), encoding='utf-8')
    assert main(['--build', str(source)]) == 2
    assert 'fixed duration' in capsys.readouterr().err


def test_build_rejects_a_step_that_skips_the_end(tmp_path, capsys):
    error = _refuse(tmp_path, capsys, variable={
        'parameter': 'speed', 'body': 'earth', 'start': 1, 'stop': 1.2, 'step': 0.15})
    assert 'land on stop' in error


def test_build_fixed_run_matches_the_existing_solver_and_stores_trails(tmp_path):
    from simulations.star_system.initialization import legacy_setup
    from simulations.star_system.recording import Calculation, CalculationRequest
    playback, _output = _build(tmp_path, years=0.01)
    setup = legacy_setup()
    calculation = Calculation(setup, CalculationRequest(0.01, 1))
    while not calculation.done:
        calculation.advance(100)
    result = calculation.result
    run = playback['runs'][0]
    assert playback['scan'] == {'mode': 'fixed', 'until': None, 'stopped_because': None}
    assert run['status'] == 'complete'
    assert run['samples'][0]['time'] == 0
    assert run['duration_years'] == pytest.approx(run['time_end'] - run['time_start'])
    assert run['time_end'] == pytest.approx(result.actual_end_time)
    assert [sample['time'] for sample in run['samples']] == pytest.approx([frame.time for frame in result.frames])
    np.testing.assert_allclose(run['samples'][-1]['r'], result.frames[-1].r)
    np.testing.assert_allclose(run['samples'][-1]['v'], result.frames[-1].v)
    assert run['samples'][0]['ids'] == ['sun', 'earth', 'jupiter']
    assert run['events'] == []
    assert playback['physics']['gas_dust_drag'] is False
    assert playback['units']['length'] == 'AU'


def test_build_sweep_keeps_unvaried_bodies_fixed(tmp_path):
    playback, _output = _build(tmp_path, years=0.005, variable={
        'parameter': 'speed', 'body': 'earth', 'start': 1, 'stop': 1.2, 'step': 0.2})
    assert playback['scan']['mode'] == 'sweep'
    assert len(playback['runs']) == 2
    assert [run['variable']['value'] for run in playback['runs']] == pytest.approx([1, 1.2])
    first, second = playback['runs']
    ids = first['samples'][0]['ids']
    earth, jupiter, sun = (ids.index(name) for name in ('earth', 'jupiter', 'sun'))
    np.testing.assert_allclose(first['samples'][0]['r'][jupiter], second['samples'][0]['r'][jupiter])
    np.testing.assert_allclose(first['samples'][0]['v'][sun], second['samples'][0]['v'][sun])
    relative_first = np.array(first['samples'][0]['v'][earth]) - np.array(first['samples'][0]['v'][sun])
    relative_second = np.array(second['samples'][0]['v'][earth]) - np.array(second['samples'][0]['v'][sun])
    assert np.linalg.norm(relative_second) == pytest.approx(np.linalg.norm(relative_first) * 1.2)


def test_build_open_scan_stops_when_escape_is_recorded(tmp_path):
    playback, _output = _build(
        tmp_path, bodies=('sun', 'earth'), years=None, watch=('escape', 'capture'),
        variable={'parameter': 'speed', 'body': 'earth', 'start': 1},
        scan={'increment': 0.5, 'until': 'escape', 'years': 0.005, 'max_runs': 6})
    assert playback['scan'] == {'mode': 'open', 'until': 'escape', 'stopped_because': 'condition'}
    assert len(playback['runs']) == 2
    assert playback['runs'][0]['events'] == []
    event = playback['runs'][1]['events'][0]
    assert event['kind'] == 'escape'
    assert event['bodies'] == ['earth']
    assert event['time'] == pytest.approx(0)
    assert 'capture' not in {item['kind'] for item in playback['runs'][1]['events']}


def test_build_open_scan_stops_at_max_runs_when_the_condition_is_absent(tmp_path):
    playback, _output = _build(
        tmp_path, bodies=('sun', 'earth'), watch=('escape',),
        variable={'parameter': 'speed', 'body': 'earth', 'start': 1},
        scan={'increment': 0.01, 'until': 'escape', 'years': 0.005, 'max_runs': 2})
    assert len(playback['runs']) == 2
    assert playback['scan']['stopped_because'] == 'max_runs'
    assert all(run['events'] == [] for run in playback['runs'])


def test_build_three_dimensional_speed_angle_stores_planar_samples(tmp_path):
    playback, _output = _build(
        tmp_path, dimensions=3, bodies=('sun', 'earth'), collision='speed_angle', years=0.005)
    run = playback['runs'][0]
    assert run['status'] == 'complete'
    assert len(run['samples'][0]['r'][0]) == 3
    assert run['samples'][0]['r'][0][2] == 0
    assert playback['physics']['engine_collision_mode'] == 'speed_angle'
    assert '#84' not in playback['physics']['note']


def test_build_stop_on_contact_records_the_halt_time(tmp_path):
    playback, _output = _build(
        tmp_path, bodies=('sun', 'earth'), collision='stop_on_contact', years=0.2, watch=('collision',),
        variable={'parameter': 'speed', 'body': 'earth', 'start': 0, 'stop': 0, 'step': 1})
    run = playback['runs'][0]
    assert run['status'] == 'halted'
    assert run['events'][0]['kind'] == 'collision'
    assert run['events'][0]['time'] == pytest.approx(run['time_end'])
    assert run['time_end'] > 0
    assert 'Yoshida' in playback['physics']['note']


def test_build_runs_planar_tides_with_stop_on_contact(tmp_path):
    playback, _output = _build(
        tmp_path, bodies=('sun', 'earth'), collision='stop_on_contact', tides=True, years=0.005)
    assert playback['runs'][0]['status'] == 'complete'
    assert playback['physics']['tides'] is True
    assert playback['physics']['engine_collision_mode'] == 'none'


def test_watch_list_records_a_merger_time():
    from simulations.star_system.collisions import CollisionEvent
    from simulations.star_system.experiment_file import collect_watch_events
    from simulations.star_system.history import StateFrame
    from simulations.star_system.initialization import build_system
    from simulations.star_system.scenarios import get_scenario
    prepared = build_system(get_scenario('sun-earth-jupiter'))
    frame = replace(StateFrame.capture(prepared), time=0.25, events=(
        CollisionEvent(0.25, 'merge', ('earth', 'jupiter'), 'remnant', ('earth', 'jupiter'), 0.0, 0.0, 'merged'),))
    found = collect_watch_events(
        (frame,), ('merger',), status='complete', reason='', user_collision='perfect_merge', primary_id='sun')
    assert found == [{'time': 0.25, 'kind': 'merger', 'bodies': ['earth', 'jupiter'], 'message': 'merged'}]


def test_watch_list_records_a_tide_without_calling_it_a_collision():
    from simulations.star_system.experiment_file import collect_watch_events
    from simulations.star_system.history import StateFrame
    from simulations.star_system.initialization import build_system
    from simulations.star_system.scenarios import get_scenario
    from simulations.star_system.tides import TideEvent
    prepared = build_system(get_scenario('sun-earth-jupiter'))
    frame = replace(StateFrame.capture(prepared), events=(
        TideEvent(0.1, 'close-pass-dissipation', ('earth', 'jupiter'), 0.01, 1.0, 0.0, 'a tide'),))
    found = collect_watch_events(
        (frame,), ('tide', 'collision'), status='complete', reason='', user_collision='stop_on_contact',
        primary_id='sun')
    assert [item['kind'] for item in found] == ['tide']
    assert found[0]['time'] == pytest.approx(0.1)
