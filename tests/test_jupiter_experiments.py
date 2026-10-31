"""Matched initial conditions and reproducible perturbations through shared APIs."""
from dataclasses import replace

import numpy as np
import pytest

from simulations.star_system.initialization import build_system
from simulations.star_system.scenarios import get_scenario, make_jupiter_comparison, compare_recordings
from simulations.star_system.persistence import save_setup, load_setup
from simulations.star_system.recording import Calculation, CalculationRequest
from simulations.star_system.model import InitialBody, BodySpec


def calculate(setup, steps=600, stride=30):
    calc = Calculation(setup, CalculationRequest(setup.dt*steps, stride))
    while not calc.done:
        calc.advance(150)
    return calc.result


def test_scenario_is_independent_reproducible_and_saveable(tmp_path):
    first, second = (get_scenario('sun-jupiter-ring') for _ in range(2))
    assert first is not second and first.population is not second.population
    assert first.population.settings.count == 32 and first.population.settings.seed == 7
    assert first.population.settings.inner == 3.2 and first.population.settings.angles == 'even'
    path = tmp_path/'jupiter.json'; save_setup(first, path)
    built = build_system(first); restored = build_system(load_setup(path))
    np.testing.assert_array_equal(built.system.r, restored.system.r)
    np.testing.assert_array_equal(built.system.v, restored.system.v)
    np.testing.assert_array_equal(built.system.r, build_system(second).system.r)


def test_control_uses_resolved_identical_vectors_without_hidden_jupiter(tmp_path):
    source = get_scenario('sun-jupiter-ring')
    source = replace(source, bodies=source.bodies[::-1])
    with_jupiter, control = make_jupiter_comparison(source)
    first, second = build_system(with_jupiter), build_system(control)
    assert control.population is None and 'jupiter' not in second.body_ids
    assert second.system.dt == first.system.dt
    for key in second.body_ids:
        i, j = first.body_ids.index(key), second.body_ids.index(key)
        np.testing.assert_array_equal(first.system.r[i], second.system.r[j])
        np.testing.assert_array_equal(first.system.v[i], second.system.v[j])
    path = tmp_path/'control.json'; save_setup(control, path)
    again = build_system(load_setup(path))
    assert again.body_ids == second.body_ids
    np.testing.assert_array_equal(again.system.r, second.system.r)


def test_control_preserves_unrelated_moon_parent_and_rejects_removing_required_parent():
    from simulations.star_system.data import CATALOG, AU_KM
    from simulations.star_system.initialization import circular_relative_velocity
    setup = get_scenario('sun-jupiter-ring')
    separation = (384400/AU_KM, 0.)
    moon_v = circular_relative_velocity(separation, CATALOG['earth'].mass, CATALOG['moon'].mass)
    earth = InitialBody('earth', CATALOG['earth'], (1., 0.), (0., 6.28), 'sun')
    moon = InitialBody('moon', CATALOG['moon'], separation, tuple(moon_v), 'earth')
    setup = replace(setup, bodies=setup.bodies+(earth, moon), dt=.0001)
    first, control = make_jupiter_comparison(setup)
    original, second = build_system(first), build_system(control)
    assert second.reference_ids[second.body_ids.index('moon')] == 'earth'
    for key in second.body_ids:
        np.testing.assert_array_equal(original.system.r[original.body_ids.index(key)], second.system.r[second.body_ids.index(key)])
        np.testing.assert_array_equal(original.system.v[original.body_ids.index(key)], second.system.v[second.body_ids.index(key)])
    invalid_control = replace(setup, bodies=setup.bodies[:-1]+(replace(moon, reference_id='jupiter'),), dt=1e-6)
    with pytest.raises(ValueError, match='Jupiter.*moon parent'):
        make_jupiter_comparison(invalid_control)


def test_matched_evolution_exhibits_unscripted_jupiter_perturbation():
    first, control = make_jupiter_comparison(get_scenario('sun-jupiter-ring'))
    runs = calculate(first), calculate(control)
    summary = compare_recordings(*runs)
    assert runs[0].status == runs[1].status == 'complete'
    assert runs[0].completed_steps == runs[1].completed_steps == 600
    assert runs[0].actual_end_time == runs[1].actual_end_time
    assert summary.max_displacement > .01
    assert summary.end_time == runs[0].actual_end_time
    assert len(summary.particle_ids) == 32


def test_partial_comparison_reports_only_common_sampled_interval():
    first, control = make_jupiter_comparison(get_scenario('sun-jupiter-ring'))
    complete = calculate(first, 60, 10)
    calc = Calculation(control, CalculationRequest(control.dt*60, 10))
    calc.advance(23); partial = calc.cancel()
    summary = compare_recordings(complete, partial)
    assert summary.end_time == complete.frames[2].time < partial.actual_end_time
    assert 'incomplete' in summary.description.lower()
    assert 'cancelled' in summary.description.lower()
    assert 'common' in summary.description.lower()
    with pytest.raises(ValueError, match='matched'):
        compare_recordings(complete, calculate(get_scenario('sun-earth-jupiter'), 60, 10))


def test_halted_comparison_preserves_real_common_time():
    setup = get_scenario('sun-jupiter-ring')
    sun = setup.bodies[0]
    rock = InitialBody('falling-rock', BodySpec('Falling rock', 'particle', 0., None),
                       (.1, 0.), (0., 0.), sun.id)
    setup = replace(setup, name='Initially valid radial infall with Jupiter',
                    bodies=setup.bodies+(rock,), population=None, dt=1e-5)
    first, control = make_jupiter_comparison(setup)
    runs = calculate(first, 1000, 20), calculate(control, 1000, 20)
    assert runs[0].status == runs[1].status == 'halted'
    summary = compare_recordings(*runs)
    assert summary.end_time <= min(run.actual_end_time for run in runs)
    assert summary.end_time < runs[0].requested_end_time
    assert 'halted' in summary.description and 'Incomplete' in summary.description
    for run in runs:
        assert np.isfinite(run.frames[-1].r).all()


def test_original_recording_pairs_without_rewriting_source_and_edited_control_is_rejected():
    original = get_scenario('sun-jupiter-ring')
    _, control = make_jupiter_comparison(original)
    first = calculate(original, 5, 1)
    summary = compare_recordings(first, calculate(control, 5, 1))
    assert summary.end_time == first.actual_end_time
    assert first.source_setup is original
    assert 'comparison_role' not in original.provenance
    changed = replace(control, bodies=control.bodies[:-1]+(replace(control.bodies[-1], v=(0., 0.)),))
    with pytest.raises(ValueError, match='initial'):
        compare_recordings(first, calculate(changed, 5, 1))


def test_selected_particle_history_uses_stable_identity():
    import matplotlib.pyplot as plt
    from simulations.star_system.history import StateFrame
    from simulations.star_system.rendering import draw_system, coordinates
    from simulations.star_system.frames import frame_transform
    prepared = build_system(get_scenario('sun-jupiter-ring'))
    first = StateFrame.capture(prepared)
    prepared.system.step(3)
    second = StateFrame.capture(prepared)
    order = np.arange(len(second.body_ids))[::-1]
    reordered = replace(second, body_ids=tuple(second.body_ids[i] for i in order),
        body_specs=tuple(second.body_specs[i] for i in order), masses=second.masses[order],
        r=second.r[order], v=second.v[order], reference_ids=tuple(second.reference_ids[i] for i in order))
    fig, ax = plt.subplots()
    try:
        scene = draw_system(ax, prepared, (first, reordered), frame_mode='rotating')
        scene.select('population:3')
        expected = [coordinates(frame_transform(frame, 'rotating').positions(frame.r[frame.body_ids.index('population:3')]))
                    for frame in (first, reordered)]
        np.testing.assert_allclose(scene.selected_trail.get_segments()[0], expected)
        assert len(scene.trails_by_id) == 1  # Jupiter; no unselected particle trails
        assert scene.selected_trail_times == (first.time, second.time)
        scene.select('population:4')
        assert not np.array_equal(scene.selected_trail.get_segments()[0], expected)
        scene.select('jupiter')
        assert not scene.selected_trail.get_visible()
        history = tuple(replace(first, time=1+i*.02) for i in range(1001))
        scene.update(prepared, history); scene.select('population:3')
        assert len(scene.selected_trail.get_segments()) == 399
        assert scene.selected_trail_vertices == 400
        assert scene.selected_trail_times == tuple(frame.time for frame in history)
        assert 'earlier history truncated' in scene.trail_info.get_text()
    finally:
        plt.close(fig)


def test_jupiter_workflow_preserves_saved_initial_conditions(tmp_path):
    import subprocess
    import sys
    import matplotlib.pyplot as plt
    from simulations.star_system.plotting import App
    setup = get_scenario('sun-jupiter-ring')
    initial = build_system(setup)
    path, image = tmp_path/'preset.json', tmp_path/'preset.png'
    save_setup(setup, path)
    restored = load_setup(path)
    run = calculate(restored, 12, 2)
    app = App(setup=restored); app.timer.stop()
    try:
        app.session.retain_recording(run); app.show_recording(run)
        app.selected_id = 'population:5'; app.toggle_follow()
        app.set_frame_mode('rotating')
        app.step_once()
        assert app.selected_id == 'population:5' and app.following
        assert 'Jupiter separation' in app.inspector.get_text()
        assert 'physical inertial v' in app.inspector.get_text()
        app.toggle_pause(); assert app.playback.playing
        app.toggle_pause(); assert not app.playback.playing
        app.run_live(); app.toggle_pause(); app.step_once()
        assert app.paused and app.system.time == setup.dt
        app.restart()
        np.testing.assert_array_equal(app.system.r, initial.system.r)
        np.testing.assert_array_equal(app.system.v, initial.system.v)
        np.testing.assert_array_equal(build_system(load_setup(path)).system.r, initial.system.r)
    finally:
        app.close(); plt.close(app.fig)
    result = subprocess.run([sys.executable, '-m', 'simulations.star_system', '--load', str(path),
        '--years', str(setup.dt*12), '--sample-stride', '2', '--output', str(image)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert image.stat().st_size > 10000


def test_shared_picker_calculate_control_and_result_switching():
    import matplotlib.pyplot as plt
    from simulations.star_system.plotting import App
    from simulations.star_system.scenarios import JUPITER_RING
    app = App(); app.timer.stop()
    try:
        app.edit_experiment()
        editor = app.experiment_editor
        editor.preset_picker.set_active(1)
        assert '3.2 AU' in editor.error.get_text() and 'capture' in editor.error.get_text()
        editor.preset()
        assert app.initial_setup.preset_id == JUPITER_RING
        rows = dict(app.sidebar.rows)
        assert rows['population:0'] == 'Test particle [population:0]'
        assert rows['population:1'] == 'Test particle [population:1]'
        assert rows['population:0'] != rows['population:1']
        original = app.initial_setup
        request = CalculationRequest(original.dt*31, 4)
        app.start_calculation(request)
        app.calculation.advance(100); app._finish_calculation()
        first = app.playback.recording
        app.edit_experiment(); app.experiment_editor.control()
        assert app.preparing and app.paused
        assert app.initial_setup.population is None and 'jupiter' not in app.prepared.body_ids
        assert not app.sidebar.rotating_view.active
        assert 'Jupiter' in app.scene.rotating_reason
        assert first.source_setup is original
        app.start_calculation(request)
        app.calculation.advance(100); app._finish_calculation()
        assert len(app.session.recordings) == 2
        assert 'Matched common sampled interval' in app.scene.result_info.get_text()
        assert 'compared at' in app.scene.result_info.get_text()
        app.show_recording()
        assert app.playback.recording is first
        assert 'Matched common sampled interval' in app.scene.result_info.get_text()
        app.set_frame_mode('rotating')
        assert app.scene.frame_mode == 'rotating'
        app.show_recording()
        assert app.scene.frame_mode == 'sun-relative'
        assert 'Jupiter' in app.scene.guide_note.get_text()
        assert app.sidebar.rotating_view.ax in app.painter.overlay_axes
        app.toggle_sidebar()
        assert not app.sidebar.rotating_view.ax.get_visible()
        app.toggle_sidebar()
        assert app.sidebar.rotating_view.ax.get_visible() and not app.sidebar.rotating_view.active
    finally:
        app.close(); plt.close(app.fig)
