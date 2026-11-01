"""Real save/calculate/live/replay boundaries for changing physical membership."""
from dataclasses import replace
import json
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from simulations.star_system.core import System
from simulations.star_system.collisions import advance_prepared
from simulations.star_system.experiments import capture_current_setup
from simulations.star_system.initialization import build_system, insert_body
from simulations.star_system.model import BodySpec, InitialBody, InitialSetup, PopulationSpec
from simulations.star_system.persistence import save_setup, load_setup, setup_digest
from simulations.star_system.plotting import App
from simulations.star_system.populations import Population
from simulations.star_system.recording import Calculation, CalculationRequest, frame_storage_bytes
from simulations.star_system.scenarios import get_scenario


def setup_fixture(kind='planet'):
    return InitialSetup('Saved contact', (
        InitialBody('a', BodySpec('A', kind, 1e-14, .1), (-1., 0.), (10., 0.)),
        InitialBody('b', BodySpec('B', 'planet', 2e-14, .1), (1., 0.), (0., 0.)),
    ), .2, collision_mode='perfect_merge')


def calculate(setup, steps=2, stride=100):
    calc = Calculation(setup, CalculationRequest(setup.dt*steps, stride))
    while not calc.done:
        calc.advance(4)
    return calc


@pytest.fixture
def app():
    value = App(setup=setup_fixture())
    value.timer.stop()
    yield value
    value.close(); plt.close(value.fig)


def test_save_reload_reproduces_contact_times_membership_properties_and_states(tmp_path):
    setup = setup_fixture()
    path = tmp_path/'contact.json'; save_setup(setup, path)
    loaded = load_setup(path)
    assert setup_digest(setup) == setup_digest(loaded)
    assert loaded.collision_policy == setup.collision_policy
    document = json.loads(path.read_text())
    assert document['numerical_version'] == setup.collision_policy.numerical
    assert document['setup']['collision_policy']['pairs'] == setup.collision_policy.pairs
    first, second = calculate(setup), calculate(loaded)
    assert first.result.status == second.result.status == 'complete'
    assert [f.time for f in first.result.frames] == [f.time for f in second.result.frames]
    for a, b in zip(first.result.frames, second.result.frames):
        assert a.body_ids == b.body_ids and a.body_specs == b.body_specs
        assert a.events == b.events and a.reference_ids == b.reference_ids
        np.testing.assert_array_equal(a.r, b.r); np.testing.assert_array_equal(a.v, b.v)
    pre, post = first.result.frames[1:3]
    assert pre.event_phase == 'pre' and post.event_phase == 'post'
    assert pre.time == post.time == first.prepared.events[0].time
    assert pre.time == pytest.approx(.18, abs=1e-10)
    assert pre.events == post.events and len(pre.body_ids) == 2 and len(post.body_ids) == 1
    assert first.result.actual_end_time == .4
    for frame in first.result.frames:
        with pytest.raises(ValueError): frame.r.setflags(write=True)


def test_legacy_mode_none_schema1_keeps_its_numerical_version(tmp_path):
    path = tmp_path/'legacy.json'
    original = get_scenario('sun-earth-jupiter'); save_setup(original, path)
    document = json.loads(path.read_text())
    assert document['schema_version'] == 1 and 'collision_policy' not in document['setup']
    assert document['numerical_version'] == 'fixed-yoshida4-v1'
    loaded = load_setup(path)
    a, b = build_system(original), build_system(loaded)
    a.system.step(3); b.system.step(3)
    np.testing.assert_array_equal(a.system.r, b.system.r)


@pytest.mark.parametrize('field,value', [('version','unknown'), ('pairs','star-merge'), ('remnant','inherit-temperature'),
                                         ('numerical','negative-yoshida-contact'), ('max_substeps',True), ('guard_fraction',1),
                                         ('time_tolerance',1), ('max_refinements',2)])
def test_unsupported_policy_rejected_before_install(app, tmp_path, field, value):
    path = tmp_path/'invalid.json'; save_setup(app.initial_setup, path)
    document = json.loads(path.read_text()); document['setup']['collision_policy'][field] = value
    path.write_text(json.dumps(document))
    old, initial = app.prepared, app.initial_setup
    with pytest.raises(ValueError): app.load_setup(path)
    assert app.prepared is old and app.initial_setup is initial


def test_current_capture_retains_mode_policy_ancestry_and_valid_moon_parent(tmp_path):
    setup = setup_fixture()
    moon = InitialBody('moon', BodySpec('Far moon', 'moon', 1e-17, .001), (10., 0.), (0., .001), 'b')
    setup = replace(setup, bodies=setup.bodies+(moon,))
    calc = calculate(setup, steps=1)
    event = calc.prepared.events[0]
    assert calc.prepared.reference_ids[calc.prepared.body_ids.index('moon')] == event.remnant_id
    captured = capture_current_setup(calc.prepared, setup, 'New contact epoch')
    assert captured.collision_mode == 'perfect_merge' and captured.collision_policy == setup.collision_policy
    assert captured.provenance['collision_events'][0]['ancestry'] == ('a', 'b')
    path = tmp_path/'epoch.json'; save_setup(captured, path)
    rebuilt = build_system(load_setup(path))
    assert rebuilt.body_ids == calc.prepared.body_ids
    np.testing.assert_allclose(rebuilt.system.r, calc.prepared.system.r, atol=1e-14, rtol=0)
    assert rebuilt.reference_ids == calc.prepared.reference_ids
    assert setup.bodies[0].id == 'a'  # original source unchanged


def test_memory_budget_rejects_event_macro_without_losing_policy_or_metadata():
    calc = Calculation(setup_fixture(), CalculationRequest(.4, 100))
    before = calc._checkpoint
    calc.request = replace(calc.request, max_bytes=calc._bytes+frame_storage_bytes(before)+300)
    calc.advance(1)
    assert calc.result.status == 'halted' and 'memory budget' in calc.result.reason
    assert calc.completed_steps == 0 and calc.prepared.system.time == 0
    assert calc.prepared.body_ids == before.body_ids and not calc.prepared.events
    assert calc.prepared.setup.collision_mode == 'perfect_merge'
    np.testing.assert_array_equal(calc.prepared.system.r, before.r)
    assert sum(frame_storage_bytes(frame) for frame in calc.result.frames) <= calc.request.max_bytes


def test_star_and_numerical_halts_remain_partial_results():
    star = calculate(setup_fixture('star'))
    assert star.result.status == 'halted' and '#14' in star.result.reason
    assert star.result.completed_steps == 0 and star.result.actual_end_time == 0
    setup = setup_fixture()
    limited = calculate(replace(setup, collision_policy=replace(setup.collision_policy, max_substeps=1)))
    assert limited.result.status == 'halted' and 'numerical halt' in limited.result.reason
    assert not limited.prepared.events and limited.prepared.body_ids == ('a', 'b')


@pytest.mark.parametrize('blit', [True, False])
def test_live_and_replay_selection_event_frames_trails_and_prior_membership(app, monkeypatch, blit):
    calc = calculate(app.initial_setup, steps=1)
    app.painter.use_blit = blit
    app.set_frame_mode('world')
    app.selected_id = 'a'; app.following = True
    app.step_once()
    remnant = app.prepared.events[0].remnant_id
    assert app.selected_id == remnant and app.following and app.paused
    assert 'explicit merge mapping' in app.status.get_text()
    assert 'temperature and composition unknown' in app.inspector.get_text()
    assert [frame.event_phase for frame in app.history] == [None, 'pre', 'post', None]
    # A new identity starts at its own event frame, never on a constituent path.
    remnant_segments = app.scene.trails_by_id[remnant].get_segments()
    assert len(remnant_segments) == 1
    assert len(app.scene.trails_by_id['a'].get_segments()) == 1
    def unexpected(*args, **kwargs): raise AssertionError('Replay must not integrate either numerical mode')
    monkeypatch.setattr(System, 'step', unexpected)
    monkeypatch.setattr('simulations.star_system.collisions.advance_prepared', unexpected)
    monkeypatch.setattr('simulations.star_system.plotting.advance_prepared', unexpected)
    app.show_recording(calc.result); app.selected_id = 'a'; app.following = True
    app.step_once()  # pre-event frame still has the original ID
    assert app.selected_id == 'a' and app.prepared.body_ids == ('a', 'b')
    app.step_once()  # exact same time, post-event membership
    assert app.selected_id == remnant and app.following
    assert 'explicit merge mapping' in app.status.get_text()
    app.fig.canvas.draw(); app.step_once()
    incremental = np.asarray(app.fig.canvas.buffer_rgba()).copy()
    app.fig.canvas.draw()
    np.testing.assert_array_equal(incremental, np.asarray(app.fig.canvas.buffer_rgba()))
    assert app._advance(1) == 0
    app.playback.index = 0; app._display_recorded_frame()
    assert app.prepared.body_ids == ('a', 'b') and app.prepared.body_specs[0].radius == .1
    np.testing.assert_array_equal(app.system.r, calc.result.frames[0].r)


def test_absorbed_tracer_selection_reports_removal_without_remnant(app):
    setup = InitialSetup('Tracer', (
        InitialBody('host', BodySpec('Host', 'planet', 1e-14, .1), (0,0), (0,0)),
        InitialBody('tracer', BodySpec('Tracer', 'particle', 0., None), (-1,0), (10,0))),
        .2, collision_mode='perfect_merge')
    app.select_setup(setup); app.selected_id = 'tracer'; app.step_once()
    assert app.selected_id == 'tracer' and 'missing / removed' in app.inspector.get_text()
    assert 'absorbed' in app.status.get_text() and 'zero mass/momentum' in app.status.get_text()


def test_policy_dialog_is_explicit_reset_and_previous_result_stays_immutable(app):
    old = calculate(app.initial_setup).result
    app.session.retain_recording(old)
    app.step_once(); assert app.system.time > 0
    app.edit_experiment(); editor = app.experiment_editor
    assert editor.collision_picker.value_selected == 'perfect_merge'
    editor.collision_picker.set_active(0)
    # Collision-free guard rejects this large dt; failed policy application leaves live state.
    editor.collisions()
    assert app.experiment_editor is editor and 'unresolved' in editor.error.get_text()
    assert app.system.time > 0 and app.initial_setup.collision_mode == 'perfect_merge'
    editor.finish()
    app.select_setup(get_scenario('sun-earth-jupiter'))
    app.edit_experiment(); editor = app.experiment_editor
    editor.collision_picker.set_active(1); editor.collisions()
    assert app.experiment_editor is None and app.preparing and app.paused and app.system.time == 0
    assert app.initial_setup.collision_mode == 'perfect_merge'
    assert app.initial_setup.preset_id is None and app.initial_setup.provenance['origin_preset'] == 'sun-earth-jupiter'
    assert get_scenario('sun-earth-jupiter').collision_mode == 'none'
    assert app.session.recordings[0] is old and old.source_setup.collision_mode == 'perfect_merge'


def test_population_moon_insertion_and_display_scale_do_not_change_physics(app, monkeypatch):
    setup = replace(get_scenario('sun-jupiter-ring'), collision_mode='perfect_merge')
    pop = calculate(setup, steps=2)
    assert pop.result.status == 'complete' and 'population:7' in pop.prepared.body_ids
    host = InitialBody('host', BodySpec('Host', 'planet', 1e-14, .1), (0,0), (0,0))
    moon = InitialBody('moon', BodySpec('Moon', 'moon', 1e-16, .01), (-.3,0), (2,0), 'host')
    moons = InitialSetup('Moon contact', (host, moon), .2, collision_mode='perfect_merge')
    assert calculate(moons, steps=1).prepared.events[0].kind == 'merge'
    original = calculate(app.initial_setup, steps=1)
    app.ax.set_xlim(-.01, .01); app.ax.set_ylim(-.02, .02)
    monkeypatch.setattr('simulations.star_system.rendering.body_size', lambda spec: 200.)
    app.step_once()
    assert app.prepared.events == original.prepared.events
    np.testing.assert_array_equal(app.system.r, original.prepared.system.r)
    inserted = insert_body(app.prepared, InitialBody('new', BodySpec('New', 'asteroid', 1e-15, .001), (10,1), (0,0)))
    assert inserted.setup.collision_mode == 'perfect_merge' and inserted.events[:-1] == app.prepared.events
    assert inserted.system.time == app.system.time


def test_saved_collision_cli_and_unsupported_contact_export_status(tmp_path):
    for kind, expected in (('planet', 0), ('star', 2)):
        path, image = tmp_path/(kind+'.json'), tmp_path/(kind+'.png')
        save_setup(setup_fixture(kind), path)
        result = subprocess.run([sys.executable, '-m', 'simulations.star_system', '--load', str(path),
            '--years', '.2', '--sample-stride', '100', '--output', str(image)], capture_output=True, text=True)
        assert result.returncode == expected, result.stderr
        if expected: assert 'halted' in result.stderr and '#14' in result.stderr
        with Image.open(image) as png:
            assert png.format == 'PNG' and png.size == (1350, 1350)


def test_supported_2000_tracer_macro_performance_is_observed():
    setup = get_scenario('sun-jupiter-ring')
    setup = replace(setup, collision_mode='perfect_merge', population=replace(setup.population,
        settings=replace(setup.population.settings, count=2000)))
    prepared = build_system(setup)
    start = time.perf_counter()
    outcome = advance_prepared(prepared)
    elapsed = time.perf_counter()-start
    print(f'2000-tracer perfect_merge macro: {elapsed:.6f}s, dt={setup.dt:.12g}yr, bodies={len(outcome.prepared.body_ids)}')
    assert len(outcome.prepared.body_ids) == 2002 and not outcome.prepared.events
    assert outcome.prepared.system.time == setup.dt
    assert np.isfinite(outcome.prepared.system.r).all()
