"""Galactic tide magnitude and seeded stellar flybys for the outside-pulls module."""
import json
import math

import numpy as np
import pytest

from simulations.star_system.core import G, SOLAR_RADIUS_AU, System
from simulations.star_system.experiment_file import main
from simulations.star_system.formation import FormationExperiment
from simulations.star_system.initialization import build_system
from simulations.star_system.model import BodySpec, InitialBody, InitialSetup
from simulations.star_system.outside_pulls import (
    ENCOUNTERS_PER_MYR_WITHIN_1_PC,
    MEAN_ENCOUNTER_SPEED_KM_S,
    MIDPLANE_DENSITY,
    OORT_A,
    OORT_B,
    OutsidePulls,
    attach,
    draw_encounters,
    encounter_rate_per_year,
    galactic_tide_acceleration,
    summarize_flybys,
    tidal_coefficients,
)


def _write(path, *, dimensions=3, outside=None, years=0.005, bodies=('sun', 'earth'), collision=None,
           tides=False, drag=False, variable=None, scan=None, watch=()):
    document = {
        'schema': 'sidereel-experiment-v1',
        'dimensions': dimensions,
        'bodies': {'catalog': list(bodies), 'new': None},
        'physics': {
            'collision': collision,
            'tides': tides,
            'gas_dust_drag': drag,
            'outside_pulls': outside,
        },
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
    return json.loads((tmp_path / 'experiment.sidereel').read_text(encoding='utf-8'))


def _wide_pair(mode='none'):
    star = InitialBody('star', BodySpec('Star', 'star', 1.0, SOLAR_RADIUS_AU), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    rock = InitialBody('rock', BodySpec('Rock', 'planet', 1.0e-6, 1.0e-5), (0.0, 0.0, 10000.0), (0.0, 0.0, 0.0))
    return InitialSetup('Wide', (star, rock), 1.0, collision_mode=mode)


def test_galactic_tide_magnitude_matches_the_heisler_tremaine_solar_neighborhood():
    # Heisler & Tremaine (1986) with Kerr & Lynden-Bell (1986) Oort constants and Bahcall (1984) density.
    # K_x = (A-B)(3A+B), K_y = -(A-B)^2, K_z = -[4πGρ - 2(B^2-A^2)], evaluated independently of the module.
    assert (OORT_A, OORT_B, MIDPLANE_DENSITY) == (14.4, -12.0, 0.185)
    radial = (OORT_A - OORT_B) * (3.0 * OORT_A + OORT_B)
    azimuthal = -((OORT_A - OORT_B) ** 2)
    au_km = 149597870700.0 / 1000.0
    year_s = 365.25 * 86400.0
    pc_au = 648000.0 / math.pi
    g_pc = G * (au_km ** 3) / (year_s ** 2) / (pc_au * au_km)
    vertical = -(4.0 * math.pi * g_pc * MIDPLANE_DENSITY * 1.0e6 - 2.0 * (OORT_B ** 2 - OORT_A ** 2))
    got = tidal_coefficients()
    assert got[0] == pytest.approx(radial, rel=1e-12)
    assert got[1] == pytest.approx(azimuthal, rel=1e-12)
    assert got[2] == pytest.approx(vertical, rel=1e-12)
    assert got[2] == pytest.approx(-10125.702632462857, rel=1e-9)
    height = 10000.0
    kpc_au = 1000.0 * pc_au
    per_au = vertical / (kpc_au * kpc_au * au_km) * (year_s ** 2) / au_km
    above = galactic_tide_acceleration(np.array([0.0, 0.0, height]))
    outward = galactic_tide_acceleration(np.array([height, 0.0, 0.0]))
    assert above[2] == pytest.approx(per_au * height, rel=1e-12)
    assert above[2] == pytest.approx(-1.0590879158410512e-10, rel=1e-9)
    assert above[2] < 0.0
    assert outward[0] > 0.0
    # The vertical tide is an order of magnitude stronger than the radial tide at the Sun.
    assert abs(above[2]) > 10.0 * outward[0]


def test_three_dimensional_step_adds_the_galactic_tide_on_a_wide_orbit():
    def once(enabled, mode):
        setup = _wide_pair(mode)
        if mode == 'none':
            prepared = build_system(setup)
            if enabled:
                attach(prepared.system, OutsidePulls(seed=None), 0, setup.dt)
            prepared.system.step()
            return float(prepared.system.v[1, 2]), float(prepared.system.dt)
        experiment = FormationExperiment(setup, ejection_radius=1.0e6)
        if enabled:
            attach(experiment.prepared.system, OutsidePulls(seed=None), 0, setup.dt)
        experiment.advance(1)
        rock = experiment.prepared.body_ids.index('rock')
        elapsed = float(experiment.time)
        return float(experiment.prepared.system.v[rock, 2]), elapsed

    expected = float(galactic_tide_acceleration(np.array([0.0, 0.0, 10000.0]))[2])
    for mode in ('none', 'perfect_merge'):
        off, elapsed = once(False, mode)
        on, same = once(True, mode)
        assert same == pytest.approx(elapsed)
        assert on < off
        assert on - off == pytest.approx(expected * elapsed, rel=1e-6)


def test_flyby_force_pulls_the_primary_toward_the_passer():
    from simulations.star_system.outside_pulls import Flyby, make_acceleration
    distance = 50000.0
    flyby = Flyby(0.0, distance, 10.0, 0.5, (distance, 0.0, 0.0), (0.0, 10.0, 0.0))
    system = System(np.array([1.0]), np.zeros((1, 3)), np.zeros((1, 3)), 0.01)
    system.external_acceleration = make_acceleration((flyby,))
    acc = system.accelerations()
    assert acc[0, 0] == pytest.approx(G * 0.5 / distance ** 2, rel=1e-12)
    assert acc[0, 1] == pytest.approx(0.0, abs=1e-18)
    assert acc[0, 2] == pytest.approx(0.0, abs=1e-18)


def test_flybys_replay_for_the_same_seed_and_change_with_the_run_index():
    assert ENCOUNTERS_PER_MYR_WITHIN_1_PC == 11.7
    assert MEAN_ENCOUNTER_SPEED_KM_S == 45.7
    assert encounter_rate_per_year(1.0) == pytest.approx(11.7e-6)
    assert encounter_rate_per_year(2.0) == pytest.approx(11.7e-6 * 4.0)
    first = draw_encounters(7, 0, 2_000_000.0)
    assert first == draw_encounters(7, 0, 2_000_000.0)
    assert first
    assert draw_encounters(7, 1, 2_000_000.0) != first
    assert draw_encounters(8, 0, 2_000_000.0) != first
    flyby = first[0]
    impact = np.array(flyby.impact_au)
    velocity = np.array(flyby.velocity_au_per_year)
    assert np.linalg.norm(impact) == pytest.approx(flyby.closest_approach_au, rel=1e-12)
    assert abs(float(np.dot(impact, velocity))) <= 1e-8 * flyby.closest_approach_au * flyby.speed_au_per_year


def test_flyby_draws_follow_the_published_rate_distance_and_speed():
    pc_au = 648000.0 / math.pi
    km_s = (149597870700.0 / 1000.0) / (365.25 * 86400.0)
    sequences = [draw_encounters(seed, 0, 10_000_000.0) for seed in range(40)]
    flat = [flyby for sequence in sequences for flyby in sequence]
    expected = 40 * 11.7 * 10.0
    assert sum(len(sequence) for sequence in sequences) == pytest.approx(expected, rel=0.08)
    distances = np.array([flyby.closest_approach_au / pc_au for flyby in flat])
    speeds = np.array([flyby.speed_au_per_year * km_s for flyby in flat])
    assert distances.mean() == pytest.approx(2.0 / 3.0, rel=0.05)
    assert speeds.mean() == pytest.approx(45.7, rel=0.05)
    assert {flyby.mass_solar for flyby in flat} == {0.5}


def test_flyby_counts_are_reported_across_runs_and_one_sequence_is_not_the_result():
    summary = summarize_flybys((('a', 'b'), (), ('c',)))
    assert summary['runs'] == 3
    assert summary['flyby_count'] == 3
    assert summary['runs_with_a_flyby'] == 2
    assert 'not a general result' in summary['note']
    assert 'across runs' in summary['note']


def test_new_file_leaves_outside_pulls_off(tmp_path):
    path = tmp_path / 'blank.json'
    assert main(['--new', str(path)]) == 0
    document = json.loads(path.read_text(encoding='utf-8'))
    assert document['physics']['outside_pulls'] is None


def test_build_with_the_module_off_does_not_claim_an_outside_pull(tmp_path):
    playback = _build(tmp_path)
    report = playback['physics']['outside_pulls']
    assert report['enabled'] is False
    assert 'no galactic tide' in report['note']
    assert 'stripped' not in report['note']
    assert 'outside_pulls' not in playback['runs'][0]


def test_build_applies_outside_pulls_on_the_three_dimensional_path(tmp_path):
    playback = _build(tmp_path, outside={'flybys': None})
    run = playback['runs'][0]
    report = playback['physics']['outside_pulls']
    assert run['status'] == 'complete'
    assert len(run['samples'][0]['r'][0]) == 3
    assert any(abs(component[2]) > 0 for component in run['samples'][0]['v'])
    assert 'not given a zero z' in playback['gravity_note']
    assert playback['physics']['path'] == 'yoshida'
    assert '3D path' in playback['physics']['note']
    assert report['enabled'] is True
    assert report['galactic_tide'] == 'heisler-tremaine-1986'
    assert 'very wide orbits' in report['note']
    assert report['flybys'] is False
    assert 'no field star' in report['note']
    assert run['outside_pulls']['flybys'] is None
    assert run['outside_pulls']['seed'] is None
    assert playback['experiment']['physics']['outside_pulls'] == {'flybys': None}


def test_build_refuses_outside_pulls_in_a_plane(tmp_path, capsys):
    source = _write(tmp_path / 'experiment.json', dimensions=2, outside={'flybys': None})
    assert main(['--build', str(source)]) == 2
    error = capsys.readouterr().err
    assert 'dimensions 3' in error
    assert 'very wide orbits' in error
    assert not (tmp_path / 'experiment.sidereel').exists()


def test_build_stores_seeded_flybys_for_every_run(tmp_path):
    playback = _build(
        tmp_path, outside={'flybys': {'seed': 11}}, years=0.005, watch=('flyby',),
        variable={'parameter': 'speed', 'body': 'earth', 'start': 1, 'stop': 1.2, 'step': 0.2})
    report = playback['physics']['outside_pulls']
    assert report['flybys'] is True
    assert report['seed'] == 11
    assert report['runs'] == 2
    assert 'not a general result' in report['note']
    assert 'across runs' in report['note']
    for run in playback['runs']:
        expected = [flyby.document() for flyby in draw_encounters(11, run['index'], 0.005)]
        assert run['outside_pulls']['flybys'] == expected
        assert run['outside_pulls']['run_index'] == run['index']
        assert [event['kind'] for event in run['events'] if event['kind'] == 'flyby'] == []


def test_build_composes_outside_pulls_with_collision_drag_and_tides(tmp_path):
    playback = _build(
        tmp_path, collision='speed_angle', tides=True, drag=True, outside={'flybys': {'seed': 3}}, years=0.02)
    run = playback['runs'][0]
    assert run['status'] == 'complete'
    assert len(run['samples'][0]['r'][0]) == 3
    assert playback['physics']['path'] == 'formation'
    assert playback['physics']['tides'] is True
    assert playback['physics']['gas_dust_drag'] is True
    assert playback['physics']['outside_pulls']['enabled'] is True
    assert playback['physics']['outside_pulls']['flybys'] is True
    assert run['dissipated_kinetic'] > 0
    assert run['outside_pulls']['flybys'] == [flyby.document() for flyby in draw_encounters(3, 0, 0.02)]


@pytest.mark.parametrize('outside', [True, {'flybys': {'seed': -1}}, {'flybys': {'seed': True}}])
def test_build_rejects_an_unusable_outside_pulls_value(tmp_path, capsys, outside):
    source = _write(tmp_path / 'experiment.json', outside=outside)
    assert main(['--build', str(source)]) == 2
    assert capsys.readouterr().err
    assert not (tmp_path / 'experiment.sidereel').exists()
