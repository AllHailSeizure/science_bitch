"""Preparation ownership and Agg callbacks, not native desktop input evidence."""
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from simulations.star_system.experiments import ExperimentSession, capture_current_setup
from simulations.star_system.initialization import build_system, insert_body, circular_relative_velocity
from simulations.star_system.model import InitialBody, InitialSetup, PopulationSpec
from simulations.star_system.data import CATALOG, AU_KM
from simulations.star_system.persistence import save_setup, load_setup, setup_digest
from simulations.star_system.plotting import App
from simulations.star_system.populations import Population
from simulations.star_system.scenarios import COMPATIBILITY, get_scenario


@pytest.fixture
def app():
    instance = App()
    instance.timer.stop()
    yield instance
    instance.close(); plt.close(instance.fig)


def test_scenario_factories_are_independent_and_session_drafts_cancel():
    first, second = get_scenario(COMPATIBILITY), get_scenario(COMPATIBILITY)
    assert first is not second and first.bodies[0] is not second.bodies[0]
    session = ExperimentSession(first)
    session.update_draft(replace(first, name='My experiment'))
    assert session.initial_setup is first and session.draft.preset_id is None
    assert session.draft.provenance['origin_preset'] == COMPATIBILITY
    session.cancel_draft()
    assert session.draft is first
    session.update_draft(replace(first, name='My experiment'))
    session.apply_draft()
    assert session.initial_setup.name == 'My experiment'
    assert get_scenario(COMPATIBILITY).name == second.name
    with pytest.raises(TypeError):
        first.provenance['description'] = 'changed'


def test_rebuilt_custom_drafts_keep_original_preset_origin():
    session = ExperimentSession(get_scenario(COMPATIBILITY))
    for name in ('First circular edit', 'Second circular edit'):
        # Legacy circular construction creates a fresh setup rather than replace.
        candidate = InitialSetup(name, session.initial_setup.bodies, session.initial_setup.dt)
        session.replace_initial(candidate)
        assert session.initial_setup.preset_id is None
        assert session.initial_setup.provenance['origin_preset'] == COMPATIBILITY


def test_app_begins_paused_and_prepare_never_advances(app):
    assert app.preparing and app.paused and app.system.time == 0
    app.update(None)
    assert app.system.time == 0 and app.pause.label.get_text() == 'Run live'
    app.toggle_pause()
    assert not app.preparing and not app.paused
    app.step_once()
    assert app.system.time > 0
    app.prepare_setup()
    assert app.preparing and app.paused and app.system.time == 0


def test_load_failure_and_experiment_dialog_cancel_preserve_live_state(app, tmp_path):
    app.toggle_pause(); app._advance(3)
    before, initial = app.system, app.initial_setup
    r, v, time = before.r.copy(), before.v.copy(), before.time
    path = tmp_path/'bad.json'; path.write_text('{"not":"a setup"}')
    with pytest.raises(ValueError):
        app.load_setup(path)
    assert app.system is before and app.initial_setup is initial
    app.edit_experiment()
    editor = app.experiment_editor
    assert not app.pause.active
    editor.fields['path'].set_val(str(path))
    editor.load()
    assert app.experiment_editor is editor and editor.error.get_text()
    editor.fields['name'].set_val('Unapplied draft')
    app.step_once(); app.update(None); app.toggle_pause(); app.prepare_setup(); app.restart()
    app.edit_body(); app.edit_population()
    assert app.body_editor is None and app.population_editor is None
    editor.finish()
    assert not app.paused and app.initial_setup is initial and app.system is before
    np.testing.assert_array_equal(app.system.r, r); np.testing.assert_array_equal(app.system.v, v)
    assert app.system.time == time


def test_ui_name_preset_save_load_and_restart_use_selected_initial(app, tmp_path):
    app.rename_setup('My starting setup')
    assert app.initial_setup.preset_id is None
    assert app.sidebar.title.get_text() == 'My starting setup'
    assert app.initial_setup.provenance['origin_preset'] == COMPATIBILITY
    assert app.prepared is app.session.prepared and app.prepared.setup is app.initial_setup
    app.step_once()
    expected = build_system(app.initial_setup)
    app.edit_experiment(); editor = app.experiment_editor
    path = tmp_path/'mine.json'; editor.fields['path'].set_val(str(path)); editor.save_initial()
    assert load_setup(path).name == 'My starting setup'
    editor.load()
    assert app.experiment_editor is None and app.preparing and app.paused
    np.testing.assert_array_equal(app.system.r, expected.system.r)
    app.step_once(); app.restart()
    np.testing.assert_array_equal(app.system.v, expected.system.v)
    app.edit_experiment(); app.experiment_editor.preset()
    assert app.initial_setup.preset_id == COMPATIBILITY and app.preparing and app.paused


def test_capture_preserves_evolved_particles_live_insertion_and_original(tmp_path):
    initial = replace(get_scenario(COMPATIBILITY), population=PopulationSpec('rocks', 'sun', Population(count=8, seed=123)))
    session = ExperimentSession(initial)
    session.prepared.system.step(6)
    session.prepared = insert_body(session.prepared, InitialBody('live', CATALOG['earth'], (8, 3), (.3, .2), 'sun'))
    initial_path, current_path = tmp_path/'initial.json', tmp_path/'current.json'
    session.save_initial(initial_path)
    current = session.save_current(current_path, 'Evolved custom epoch')
    assert session.initial_setup is initial
    assert setup_digest(load_setup(initial_path)) == setup_digest(initial)
    assert current.population is None and len(current.bodies) == len(session.prepared.body_ids)
    particles = [body for body in current.bodies if body.spec.kind == 'particle']
    assert len(particles) == 8 and all(body.spec.mass == 0 for body in particles)
    assert current.provenance['source_population']['settings']['seed'] == 123
    rebuilt = build_system(load_setup(current_path))
    assert rebuilt.system.time == 0
    assert rebuilt.body_ids == session.prepared.body_ids and rebuilt.reference_ids == session.prepared.reference_ids
    np.testing.assert_allclose(rebuilt.system.r, session.prepared.system.r, rtol=0, atol=1e-14)
    np.testing.assert_allclose(rebuilt.system.v, session.prepared.system.v, rtol=0, atol=1e-14)
    assert current.provenance['source_time'] == session.prepared.system.time
    assert 'live' not in build_system(load_setup(initial_path)).body_ids


def test_ui_separate_capture_path_preserves_initial_file(app, tmp_path):
    initial, captured = tmp_path/'initial.json', tmp_path/'captured.json'
    app.step_once(); app.edit_experiment(); editor = app.experiment_editor
    editor.fields['path'].set_val(str(initial))
    editor.save_initial()
    content = initial.read_bytes()
    editor.fields['capture_path'].set_val(str(initial))
    editor.save_current()
    assert 'different' in editor.error.get_text() and initial.read_bytes() == content
    editor.fields['capture_path'].set_val(str(captured))
    editor.save_current()
    assert captured.exists() and initial.read_bytes() == content
    assert load_setup(captured).name != load_setup(initial).name


def test_current_moon_capture_subtracts_parent_once_and_discloses_smaller_dt(tmp_path):
    separation = 384400/AU_KM
    velocity = .7*circular_relative_velocity((separation, 0), CATALOG['earth'].mass, CATALOG['moon'].mass)
    initial = InitialSetup('Earth', (InitialBody('earth', CATALOG['earth'], (3, 2), (.2, -.1)),), .001)
    live = insert_body(build_system(initial), InitialBody('moon', CATALOG['moon'], (separation, 0), tuple(velocity), 'earth'))
    live.system.step(80)
    captured = capture_current_setup(live, initial, 'Captured Moon')
    assert captured.dt < live.system.dt
    assert 'Smaller fixed dt' in captured.provenance['capture_note']
    assert captured.bodies[1].reference_id == 'earth'
    np.testing.assert_array_equal(captured.bodies[1].r, live.system.r[1]-live.system.r[0])
    np.testing.assert_array_equal(captured.bodies[1].v, live.system.v[1]-live.system.v[0])
    path = tmp_path/'moon.json'; save_setup(captured, path)
    rebuilt = build_system(load_setup(path))
    np.testing.assert_allclose(rebuilt.system.r, live.system.r, atol=1e-15)
    np.testing.assert_allclose(rebuilt.system.v, live.system.v, atol=1e-15)
    assert rebuilt.reference_ids == (None, 'earth')
    with pytest.raises(ValueError, match='different'):
        capture_current_setup(live, initial, initial.name)


def finish_calculation(app):
    for _ in range(1000):
        app.update(None)
        if app.mode_name == 'RECORDED':
            return
    raise AssertionError('Calculation did not yield a result')


def test_calculation_lifecycle_retains_source_through_draft_and_live(app):
    from simulations.star_system.recording import CalculationRequest
    original = app.initial_setup
    app.start_calculation(CalculationRequest(original.dt*9, 4))
    assert app.mode_name == 'CALCULATING' and app.sidebar.calculate.active
    assert not app.sidebar.add.active and not app.pause.active
    finish_calculation(app)
    run = app.playback.recording
    assert app.mode_name == 'RECORDED' and run.status == 'complete'
    assert run.setup_digest == setup_digest(original) and run.source_name == original.name
    assert app.session.recordings == (run,)
    app.rename_setup('A different draft')
    assert app.preparing and app.paused and app.playback is None and app.system.time == 0
    assert run.source_name == original.name and run.setup_digest == setup_digest(original)
    app.show_recording(run)
    assert app.sidebar.title.get_text() == original.name
    app.run_live()
    assert app.mode_name == 'LIVE' and not app.paused and app.system.time == 0
    app.step_once()
    assert app.paused and app.system.time == app.initial_setup.dt
    assert app.session.recordings == (run,)


def click(app, axes):
    from matplotlib.backend_bases import MouseEvent
    app.fig.canvas.draw()
    x, y = axes.transAxes.transform((.5,.5))
    for event_name in ('button_press_event', 'button_release_event'):
        app.fig.canvas.callbacks.process(event_name, MouseEvent(event_name, app.fig.canvas, x, y, button=1))


def test_calculate_and_cancel_buttons_use_bounded_lifecycle(app):
    click(app, app.sidebar.calculate.ax)
    editor = app.calculation_editor
    editor.fields['duration'].set_val(str(app.initial_setup.dt*100))
    from matplotlib.backend_bases import MouseEvent
    x, y = editor.start.ax.transAxes.transform((.5,.5))
    for name in ('button_press_event', 'button_release_event'):
        editor.fig.canvas.callbacks.process(name, MouseEvent(name, editor.fig.canvas, x, y, button=1))
    assert app.mode_name == 'CALCULATING'
    app.calculation.advance(3)
    click(app, app.sidebar.calculate.ax)
    assert app.mode_name == 'RECORDED' and app.playback.recording.status == 'cancelled'


def test_calculation_cancel_halt_and_previous_results(app):
    from simulations.star_system.recording import CalculationRequest
    request = CalculationRequest(app.initial_setup.dt*100, 9)
    app.start_calculation(request)
    app.calculation.advance(3)
    app.cancel_calculation()
    cancelled = app.playback.recording
    assert cancelled.status == 'cancelled' and cancelled.completed_steps == 3
    assert cancelled.frames[-1].time > 0
    app.start_calculation(request)
    app.calculation.advance(2)
    app.calculation.prepared.system.v[1] = (1e8,0)
    finish_calculation(app)
    halted = app.playback.recording
    assert halted.status == 'halted' and halted.completed_steps == 2
    assert 'unresolved' in app.status.get_text()
    app.show_recording()
    assert app.playback.recording is cancelled
    app.show_recording()
    assert app.playback.recording is halted


def test_calculate_dialog_preflight_overlay_and_cancel_is_recoverable(app):
    app.edit_calculation()
    editor = app.calculation_editor
    assert editor.start.active and 'whole steps' in editor.summary.get_text()
    app.step_once(); app.toggle_pause(); app.edit_body(); app.edit_population()
    assert app.system.time == 0 and app.body_editor is None and app.population_editor is None
    editor.fields['duration'].set_val('nan')
    assert editor.error.get_text() and not editor.start.active
    editor.fields['duration'].set_val(str(app.initial_setup.dt*4))
    assert editor.start.active
    editor.begin()
    assert app.calculation_editor is None and app.mode_name == 'CALCULATING'
    assert all(widget.ax in app.painter.overlay_axes for widget in (app.sidebar.calculate, app.sidebar.results, app.sidebar.run_live))
    app.cancel_calculation()
    assert app.playback.recording.status == 'cancelled'


def test_recorded_authoring_returns_to_preparation_without_insertion(app):
    from simulations.star_system.recording import CalculationRequest
    app.start_calculation(CalculationRequest(app.initial_setup.dt*3, 2))
    finish_calculation(app)
    result = app.playback.recording
    app.edit_body()
    assert app.preparing and app.playback is None and app.system.time == 0
    assert app.body_editor is not None
    app.body_editor.finish()
    app.show_recording(result)
    app.edit_population()
    assert app.preparing and app.playback is None and app.system.time == 0
    app.population_editor.finish()
    assert app.session.recordings == (result,)


def test_recorded_capture_uses_displayed_sample_and_true_source_after_draft_edit(app, tmp_path):
    from simulations.star_system.recording import CalculationRequest
    original = app.initial_setup
    app.start_calculation(CalculationRequest(original.dt*9, 4))
    finish_calculation(app)
    result = app.playback.recording
    app.rename_setup('Another preparation draft')
    app.show_recording(result); app.step_once(); app.step_once()
    frame = app.playback.current_frame
    assert frame.time > 0
    app.edit_experiment()
    editor = app.experiment_editor
    assert app.playback.current_frame is frame
    path = tmp_path/'recorded-capture.json'
    editor.fields['capture_name'].set_val('Recorded epoch')
    editor.fields['capture_path'].set_val(str(path))
    editor.save_current()
    captured = load_setup(path)
    assert captured.provenance['source_time'] == frame.time
    assert captured.provenance['source_name'] == original.name
    assert captured.provenance['source_setup_digest'] == result.setup_digest
    assert app.initial_setup.name == 'Another preparation draft'
    rebuilt = build_system(captured)
    np.testing.assert_allclose(rebuilt.system.r, frame.r, rtol=0, atol=1e-14)
    np.testing.assert_allclose(rebuilt.system.v, frame.v, rtol=0, atol=1e-14)
    editor.finish()
