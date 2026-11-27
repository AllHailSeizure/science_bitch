"""The one supported stellar merger, and the contacts that must not use it."""
import json
import math
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from simulations.star_system.collisions import advance_prepared
from simulations.star_system.core import G
from simulations.star_system.data import CATALOG
from simulations.star_system.experiments import capture_current_setup
from simulations.star_system.initialization import build_system
from simulations.star_system.persistence import load_setup, save_setup, setup_digest
from simulations.star_system.recording import Calculation, CalculationRequest
from simulations.star_system.stellar_merger import (
    MODE, MODEL_ID, OBSERVATION_FACTOR, REMNANT_RADIUS_EXPONENT, SEPARATION_FACTOR,
    analytic_contact_time, binary_star_merger_setup, estimated_remnant_radius,
    head_on_rest_contact_time, observation_years, outcome_text,
)
from simulations.star_system.templates import EXPERIMENTS, get_experiment


def integrate(setup, years=None):
    duration = observation_years(binary_star_merger_setup()) if years is None else years
    calc = Calculation(setup, CalculationRequest(duration, 1))
    while not calc.done:
        calc.advance(1000)
    return calc


def merger_event(prepared):
    events = [event for event in prepared.events if event.kind == 'stellar_merger']
    assert len(events) == 1
    return events[0]


def test_published_scenario_is_two_solar_type_stars_released_from_rest():
    setup = get_experiment('binary-star-merger')
    assert setup is not binary_star_merger_setup()
    assert setup.name == 'Binary-star merger'
    assert setup.preset_id == 'binary-star-merger'
    assert setup.collision_mode == MODE
    assert setup.provenance['merger_model'] == MODEL_ID
    assert setup.provenance['starting_template'] == 'binary-star-merger'
    assert setup.provenance['separation_over_contact'] == SEPARATION_FACTOR
    assert 'head-on' in setup.provenance['mechanism']
    assert any(item.key == 'binary-star-merger' and item.title == setup.name for item in EXPERIMENTS)
    sun = CATALOG['sun']
    assert [body.id for body in setup.bodies] == ['primary', 'secondary']
    prepared = build_system(setup)
    assert prepared.system.time == 0
    separation = float(np.linalg.norm(prepared.system.r[0] - prepared.system.r[1]))
    assert separation == pytest.approx(SEPARATION_FACTOR * 2 * sun.radius)
    np.testing.assert_allclose(prepared.system.v, 0)
    np.testing.assert_allclose(prepared.system.masses, sun.mass)
    assert all(spec.kind == 'star' and spec.radius == sun.radius for spec in prepared.body_specs)
    assert observation_years(setup) == pytest.approx(analytic_contact_time(setup) * OBSERVATION_FACTOR)
    assert observation_years(setup) > analytic_contact_time(setup)


def test_center_fall_limit_matches_the_textbook_radial_time():
    separation = 0.1
    total_mass = 2.
    textbook = math.pi * math.sqrt(separation ** 3 / (8 * G * total_mass))
    close = head_on_rest_contact_time(separation, separation * 1e-6, 1., 1.)
    assert close == pytest.approx(textbook, rel=1e-8)


def test_supported_run_records_parents_time_and_estimated_remnant():
    setup = binary_star_merger_setup()
    analytic = analytic_contact_time(setup)
    calc = integrate(setup)
    result = calc.result
    assert result.status == 'complete', result.reason
    event = merger_event(calc.prepared)
    assert event.time == pytest.approx(analytic, abs=1e-7)
    assert event.parent_ids == ('primary', 'secondary')
    assert event.remnant_id.startswith('stellar-merger:')
    assert 'Modeled stellar merger' in event.message
    assert 'primary' in event.message and 'secondary' in event.message
    assert f'{event.time:.8g}' in event.message
    assert MODEL_ID in event.message
    assert 'temperature' in event.message and 'omitted' in event.message
    pre, post = next(
        (a, b) for a, b in zip(result.frames, result.frames[1:])
        if a.event_phase == 'pre' and b.event_phase == 'post')
    assert pre.time == post.time == event.time
    assert pre.body_ids == ('primary', 'secondary')
    assert post.body_ids == (event.remnant_id,)
    remnant = post.body_specs[0]
    assert remnant.kind == 'star'
    assert remnant.name == 'Solar-type merger remnant'
    assert remnant.temperature is None and not remnant.sources and remnant.catalog_key is None
    assert remnant.mass == pytest.approx(2.)
    assert remnant.radius == pytest.approx(estimated_remnant_radius(2.))
    assert remnant.radius == pytest.approx(CATALOG['sun'].radius * 2 ** REMNANT_RADIUS_EXPONENT)
    np.testing.assert_allclose(post.masses.sum(), pre.masses.sum())
    np.testing.assert_allclose(post.r[0], 0, atol=1e-12)
    np.testing.assert_allclose(post.v[0], 0, atol=1e-12)
    np.testing.assert_allclose((post.masses[:, None]*post.v).sum(0), (pre.masses[:, None]*pre.v).sum(0), atol=1e-12)
    assert event.lost_relative_kinetic > 0
    assert abs(event.unmodeled_spin) < 1e-12
    assert outcome_text(calc.prepared) == event.message


def test_contact_time_moves_toward_the_analytic_fall_as_the_step_shrinks():
    setup = binary_star_merger_setup()
    analytic = analytic_contact_time(setup)
    errors = []
    for factor in (1., .5, .25):
        calc = integrate(replace(setup, dt=setup.dt * factor))
        assert calc.result.status == 'complete', calc.result.reason
        errors.append(abs(merger_event(calc.prepared).time - analytic))
    assert errors[2] < errors[0]
    assert errors[2] < 1e-7


def test_save_reload_and_row_order_reproduce_the_same_merger(tmp_path):
    setup = binary_star_merger_setup()
    path = tmp_path / 'binary-star-merger.json'
    save_setup(setup, path)
    loaded = load_setup(path)
    assert setup_digest(loaded) == setup_digest(setup)
    document = json.loads(path.read_text())
    assert document['numerical_version'] == setup.collision_policy.numerical
    assert document['numerical_version'] == 'positive-verlet-swept-parabola-v1'
    assert document['setup']['collision_mode'] == MODE
    first, second = integrate(setup), integrate(loaded)
    assert first.result.status == second.result.status == 'complete'
    assert merger_event(first.prepared) == merger_event(second.prepared)
    flipped = replace(setup, bodies=(setup.bodies[1], setup.bodies[0]))
    assert merger_event(integrate(flipped).prepared) == merger_event(first.prepared)


def test_other_contacts_halt_without_a_stellar_remnant():
    setup = binary_star_merger_setup()
    duration = observation_years(setup)
    no_model = replace(setup, provenance={'note': 'not the published scenario'})
    heavier = replace(setup.bodies[0], spec=replace(setup.bodies[0].spec, mass=1.2))
    sirius = CATALOG['sirius']
    foreign = replace(setup, bodies=tuple(
        replace(body, spec=replace(sirius, name=body.spec.name)) for body in setup.bodies))
    cases = {
        'perfect_merge': replace(setup, collision_mode='perfect_merge'),
        'missing model': no_model,
        'edited mass': replace(setup, bodies=(heavier, setup.bodies[1])),
        'sirius': foreign,
    }
    for label, candidate in cases.items():
        calc = integrate(candidate, duration)
        assert calc.result.status == 'halted', label
        assert 'Unsupported' in calc.result.reason and '#14' in calc.result.reason, label
        assert 'Modeled stellar merger' not in calc.result.reason, label
        assert 'numerical halt' not in calc.result.reason, label
        assert not any(getattr(event, 'kind', None) == 'stellar_merger' for event in calc.prepared.events), label
        assert len(calc.prepared.body_ids) == 2, label
        assert all(spec.kind == 'star' for spec in calc.prepared.body_specs), label
    limited = replace(setup, collision_policy=replace(setup.collision_policy, max_substeps=1))
    numerical = integrate(limited, duration)
    assert numerical.result.status == 'halted'
    assert 'numerical halt' in numerical.result.reason
    assert 'Unsupported' not in numerical.result.reason
    assert 'Modeled' not in numerical.result.reason
    assert numerical.result.actual_end_time == 0
    assert numerical.prepared.body_ids == ('primary', 'secondary')
    free = integrate(replace(setup, collision_mode='none'), duration)
    assert free.result.status == 'halted'
    assert 'Modeled stellar merger' not in free.result.reason
    assert 'Unsupported contact' not in free.result.reason
    assert free.prepared.body_ids == ('primary', 'secondary')


def test_captured_epoch_keeps_the_model_and_still_merges():
    setup = binary_star_merger_setup()
    prepared = build_system(setup)
    for _ in range(4):
        prepared = advance_prepared(prepared).prepared
    assert prepared.system.time > 0 and not prepared.events
    captured = capture_current_setup(prepared, setup, 'Closer epoch')
    assert captured.collision_mode == MODE
    assert captured.provenance['merger_model'] == MODEL_ID
    calc = integrate(captured)
    assert calc.result.status == 'complete', calc.result.reason
    event = merger_event(calc.prepared)
    assert event.parent_ids == ('primary', 'secondary')
    assert calc.prepared.body_specs[-1].mass == pytest.approx(2.)


def test_saved_merger_cli_completes_and_other_contact_halts(tmp_path):
    setup = binary_star_merger_setup()
    path, image = tmp_path / 'merger.json', tmp_path / 'merger.png'
    save_setup(setup, path)
    env = dict(os.environ, MPLBACKEND='Agg')
    completed = subprocess.run(
        [sys.executable, '-m', 'simulations.star_system', '--load', str(path),
         '--years', str(observation_years(setup)), '--sample-stride', '1', '--output', str(image)],
        capture_output=True, text=True, env=env)
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ''
    assert image.read_bytes().startswith(b'\x89PNG')
    edited = replace(setup, bodies=(
        replace(setup.bodies[0], spec=replace(setup.bodies[0].spec, mass=1.2)), setup.bodies[1]))
    other, other_image = tmp_path / 'other.json', tmp_path / 'other.png'
    save_setup(edited, other)
    halted = subprocess.run(
        [sys.executable, '-m', 'simulations.star_system', '--load', str(other),
         '--years', str(observation_years(setup)), '--sample-stride', '5', '--output', str(other_image)],
        capture_output=True, text=True, env=env)
    assert halted.returncode == 2, halted.stderr
    assert 'Unsupported' in halted.stderr and '#14' in halted.stderr
    assert 'Modeled stellar merger' not in halted.stderr
    assert other_image.read_bytes().startswith(b'\x89PNG')


@pytest.fixture(scope='module')
def qtapp():
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qtapp):
    from simulations.star_system.desktop import Observatory
    instance = Observatory()
    instance.timer.stop()
    instance.show()
    qtapp.processEvents()
    yield instance
    instance.dirty = False
    instance.close()
    qtapp.processEvents()


def test_desktop_experiment_shows_the_merger_outcome(window, qtapp):
    window.choose_button.click()
    titles = [window.experiment_list.item(i).text() for i in range(window.experiment_list.count())]
    row = titles.index('Binary-star merger')
    window.experiment_list.setCurrentRow(row)
    qtapp.processEvents()
    assert 'solar-type' in window.experiment_description.text()
    assert 'remnant' in window.experiment_observe.text()
    window.open_button.click()
    qtapp.processEvents()
    assert window.setup.collision_mode == MODE
    assert 'unsupported' in window.notice.text().lower()
    assert 'numerically' in window.notice.text().lower()
    names = [window.body_list.item(i).text() for i in range(window.body_list.count())]
    assert names == ['Solar-type primary', 'Solar-type secondary']
    window.duration.setValue(observation_years(window.setup))
    window.stride.setValue(1)
    window.start_calculation()
    for _ in range(100):
        if window.calculation.done:
            break
        window.tick()
    assert window.calculation.done
    assert window.playback.recording.status == 'complete'
    window.playback.index = len(window.playback.recording.frames) - 1
    window.display_recorded_frame()
    qtapp.processEvents()
    readout = window.time_readout.text()
    assert 'Modeled stellar merger' in readout
    assert 'primary' in readout and 'secondary' in readout
    assert 'yr' in readout
    remnant = window.prepared.body_ids[0]
    window.select_body(remnant)
    assert window.body_information.name.text() == 'Solar-type merger remnant'
    details = window.body_information.details.text()
    assert details.startswith('Star')
    assert 'Temperature: unknown' in details
    assert 'R_sun' in details
    assert 'mass loss omitted' in details
