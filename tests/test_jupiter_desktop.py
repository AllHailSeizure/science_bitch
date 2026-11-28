"""Sun–Jupiter ring behavior through the desktop observatory."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from simulations.star_system.frames import frame_transform, labeled_lagrange_guides
from simulations.star_system.initialization import build_system
from simulations.star_system.persistence import load_setup, save_setup
from simulations.star_system.rendering import coordinates
from simulations.star_system.scenarios import compare_recordings
from simulations.star_system.templates import get_experiment


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


def open_ring(window, qtapp):
    window.choose_button.click()
    titles = [window.experiment_list.item(i).text() for i in range(window.experiment_list.count())]
    window.experiment_list.setCurrentRow(titles.index('Sun–Jupiter ring'))
    qtapp.processEvents()
    window.open_button.click()
    qtapp.processEvents()


def rock_angles(prepared):
    sun = prepared.system.r[prepared.body_ids.index('sun')]
    rocks = np.array([prepared.system.r[i] for i, spec in enumerate(prepared.body_specs) if spec.kind == 'particle'])
    angles = np.sort(np.arctan2(rocks[:, 1]-sun[1], rocks[:, 0]-sun[0]))
    return np.diff(np.r_[angles, angles[0]+2*np.pi])


def test_library_states_ring_settings_and_limits(window, qtapp):
    window.choose_button.click()
    titles = [window.experiment_list.item(i).text() for i in range(window.experiment_list.count())]
    window.experiment_list.setCurrentRow(titles.index('Sun–Jupiter ring'))
    qtapp.processEvents()
    text = window.experiment_description.text()+' '+window.experiment_observe.text()
    for phrase in ('32', '3.2', 'seed 7', 'evenly', 'collision mode is none', 'capture'):
        assert phrase in text.lower() if phrase != '3.2' else phrase in text
    assert 'rotating view' in text.lower()
    assert 'l4' in text.lower() and 'l5' in text.lower()


def test_ring_rocks_are_evenly_spaced_reproducible_and_saveable(tmp_path):
    first, second = get_experiment('sun-jupiter-ring'), get_experiment('sun-jupiter-ring')
    assert first is not second and first.population is not second.population
    settings = first.population.settings
    assert first.population.reference_id == 'sun'
    assert settings.count == 32 and settings.seed == 7 and settings.angles == 'even'
    assert settings.inner == settings.outer == 3.2
    assert first.collision_mode == 'none'
    assert {body.id for body in first.bodies} == {'sun', 'jupiter'}
    prepared = build_system(first)
    np.testing.assert_allclose(rock_angles(prepared), 2*np.pi/32)
    radii = np.linalg.norm(prepared.system.r[2:]-prepared.system.r[0], axis=1)
    np.testing.assert_allclose(radii, 3.2)
    path = tmp_path/'ring.json'
    save_setup(first, path)
    restored = build_system(load_setup(path))
    np.testing.assert_array_equal(restored.system.r, prepared.system.r)
    np.testing.assert_array_equal(restored.system.v, prepared.system.v)
    np.testing.assert_array_equal(prepared.system.r, build_system(second).system.r)


def test_solar_system_does_not_show_ring_controls(window, qtapp):
    window.choose_button.click()
    window.open_button.click()
    qtapp.processEvents()
    assert window.setup.name == 'Solar System'
    assert not window.rotating_button.isVisible()
    assert not window.control_button.isVisible()
    assert not window.scene.reference_guides.get_visible()


def test_live_pause_step_reset_and_recorded_status(window, qtapp):
    open_ring(window, qtapp)
    assert '32' in window.assumptions.text() and '3.2' in window.assumptions.text()
    assert 'seed 7' in window.assumptions.text() and 'none' in window.assumptions.text().lower()
    assert 'capture' in window.assumptions.text().lower()
    start = window.prepared.system.r.copy()
    dt = window.setup.dt
    window.step_button.click()
    qtapp.processEvents()
    assert window.prepared.system.time == pytest.approx(dt)
    assert f'{dt:,.3f}' in window.time_readout.text()
    assert 'elapsed simulated years' in window.time_readout.text()
    window.run_button.click()
    assert window.running and window.run_button.text() == 'Pause'
    window.run_button.click()
    assert not window.running and window.prepared.system.time == pytest.approx(dt)
    window.reset_button.click()
    qtapp.processEvents()
    assert window.prepared.system.time == 0 and window.editing
    np.testing.assert_array_equal(window.prepared.system.r, start)
    assert '0.000' in window.time_readout.text()
    window.duration.setValue(dt*12)
    window.stride.setValue(4)
    window.start_calculation()
    window.calculation.advance(100)
    assert window.calculation.done and window.calculation.status == 'complete'
    window.finish_calculation()
    run = window.session.recordings[-1]
    assert 'complete' in window.result_combo.currentText().lower()
    info = window.scene.result_info.get_text()
    assert 'COMPLETE' in info
    assert f'{run.actual_end_time:.6g}' in info
    assert run.actual_end_time == pytest.approx(run.resolved_end_time)


def test_rotating_view_fixes_jupiter_keeps_state_and_labels_guides(window, qtapp):
    open_ring(window, qtapp)
    before_r, before_v = window.prepared.system.r.copy(), window.prepared.system.v.copy()
    before_t = window.prepared.system.time
    assert window.rotating_button.text() == 'Rotating view'
    assert 'inertial' in window.scene.heading.get_text()
    window.rotating_button.click()
    qtapp.processEvents()
    jupiter = window.prepared.body_ids.index('jupiter')
    displayed = window.scene.positions[jupiter]
    assert displayed[0] > 0
    assert displayed[1] == pytest.approx(0, abs=1e-8)
    np.testing.assert_array_equal(window.prepared.system.r, before_r)
    np.testing.assert_array_equal(window.prepared.system.v, before_v)
    assert window.prepared.system.time == before_t
    assert window.scene.frame_mode == 'rotating'
    assert 'rotating' in window.scene.heading.get_text().lower()
    assert window.rotating_button.text() == 'Ordinary view'
    assert [item.get_text() for item in window.scene.guide_labels] == ['L4 guide', 'L5 guide']
    assert window.scene.reference_guides.get_visible()
    assert len(window.scene.reference_guides.get_facecolor()) == 0
    assert window.scene.reference_guides.get_edgecolor()[0, 3] == 1
    assert len(window.scene.reference_guides.get_paths()[0].vertices) == 5
    assert window.scene.particles.get_facecolor()[0, 3] > 0
    assert len(window.scene.particles.get_paths()[0].vertices) > 8
    guides = window.scene.display_to_world(np.asarray(window.scene.reference_guides.get_offsets(), dtype=float))
    rocks = window.scene.display_to_world(np.asarray(window.scene.particles.get_offsets(), dtype=float))
    world_guides = labeled_lagrange_guides(window.scene.frame, 'sun', 'jupiter')[0]
    np.testing.assert_allclose(guides, world_guides[:, :guides.shape[1]], atol=1e-8)
    distances = np.linalg.norm(guides[:, None, :]-rocks[None, :, :], axis=2)
    assert distances.min() > .5
    note = window.scene.guide_note.get_text()
    assert 'ideal circular' in note.lower() or 'reference geometry' in note.lower()
    assert 'not detected' in note.lower()
    window.rotating_button.click()
    qtapp.processEvents()
    assert window.scene.frame_mode == 'sun-relative'
    np.testing.assert_array_equal(window.prepared.system.r, before_r)
    np.testing.assert_array_equal(window.prepared.system.v, before_v)


def test_selected_rock_trail_uses_each_samples_angle_and_reports_duration(window, qtapp):
    open_ring(window, qtapp)
    window.rotating_button.click()
    window.step_button.click()
    window.step_button.click()
    window.select_body('population:3')
    qtapp.processEvents()
    report = window.context_label.text()
    assert 'Jupiter separation:' in report
    assert 'physical inertial v' in report
    assert 'population:3' in report
    assert 'duration' in report
    assert 'not permanent capture' in report
    times = window.scene.selected_trail_times
    assert times[0] == 0 and times[-1] == pytest.approx(2*window.setup.dt)
    assert f'{times[-1]-times[0]:.6g}' in report
    expected = [coordinates(frame_transform(frame, 'rotating').positions(
        frame.r[frame.body_ids.index('population:3')])) for frame in window.scene.history]
    segments = window.scene.selected_trail.get_segments()
    shown = np.vstack([segments[0][0], *[segment[1] for segment in segments]])
    np.testing.assert_allclose(shown, expected)
    window.select_body('population:4')
    qtapp.processEvents()
    assert 'population:4' in window.context_label.text()
    assert 'population:3' not in window.context_label.text()


def test_rotating_velocity_edit_stores_the_physical_vector(window, qtapp):
    open_ring(window, qtapp)
    window.rotating_button.click()
    qtapp.processEvents()
    window.select_body('jupiter')
    index = window.prepared.body_ids.index('jupiter')
    displayed = np.array([.35, -.2])
    previous = window.prepared.system.v[index].copy()
    anchor = window.ax.transData.transform(window.scene.positions[index])
    tip = window.ax.transData.transform(window.velocity_tips['jupiter'])
    target = anchor+displayed*12
    window.on_press(SimpleNamespace(inaxes=window.ax, button=1, xdata=0., ydata=0., x=tip[0], y=tip[1]))
    window.on_motion(SimpleNamespace(inaxes=window.ax, xdata=0., ydata=0., x=target[0], y=target[1]))
    window.on_release(SimpleNamespace(inaxes=window.ax, xdata=0., ydata=0., x=target[0], y=target[1]))
    physical = np.asarray(window.scene.transform.inverse_vectors(displayed), dtype=float)
    reference = window.prepared.reference_ids[index]
    parent_v = (window.prepared.system.v[window.prepared.body_ids.index(reference)]
                if reference in window.prepared.body_ids else np.zeros(previous.shape[0]))
    current = previous-parent_v
    expected = np.array(parent_v, dtype=float, copy=True)
    expected[:2] = parent_v[:2]+physical
    if expected.shape[0] == 3:
        expected[2] = parent_v[2]+current[2]
    np.testing.assert_allclose(window.prepared.system.v[index], expected, atol=1e-12)
    stored = window.prepared.system.v.copy()
    window.rotating_button.click()
    qtapp.processEvents()
    np.testing.assert_array_equal(window.prepared.system.v, stored)
    assert window.scene.frame_mode == 'sun-relative'


def test_matched_control_compares_equal_time_and_labels_a_shorter_run(window, qtapp):
    open_ring(window, qtapp)
    ring = window.setup
    assert window.tabs.isTabEnabled(2) is False
    window.select_body('jupiter')
    assert window.tabs.isTabEnabled(2)
    dt = window.setup.dt
    window.duration.setValue(dt*12)
    window.stride.setValue(4)
    window.start_calculation()
    window.calculation.advance(100)
    window.finish_calculation()
    window.control_button.click()
    qtapp.processEvents()
    assert window.editing and window.prepared.system.time == 0
    assert window.setup.population is None and 'jupiter' not in window.prepared.body_ids
    assert any(body.id == 'population:0' for body in window.setup.bodies)
    original, control = build_system(ring), build_system(window.setup)
    for key in control.body_ids:
        i, j = original.body_ids.index(key), control.body_ids.index(key)
        np.testing.assert_array_equal(original.system.r[i], control.system.r[j])
        np.testing.assert_array_equal(original.system.v[i], control.system.v[j])
    assert control.system.dt == original.system.dt
    assert not window.rotating_button.isEnabled()
    assert 'Jupiter' in window.scene.rotating_reason
    assert 'Jupiter' in window.scene.guide_note.get_text()
    assert not window.scene.reference_guides.get_visible()
    window.select_body('population:1')
    assert 'Jupiter is absent' in window.context_label.text()
    assert 'duration' in window.context_label.text()
    window.start_calculation()
    window.calculation.advance(100)
    window.finish_calculation()
    summary = compare_recordings(*window.session.recordings)
    assert window.session.recordings[0].status == window.session.recordings[1].status == 'complete'
    assert summary.end_time == window.session.recordings[0].actual_end_time
    assert summary.max_displacement > 0
    assert summary.description == window.comparison_label.text()
    assert 'Incomplete' not in summary.description
    assert summary.description in window.scene.result_info.get_text()
    window.original_button.click()
    qtapp.processEvents()
    assert 'jupiter' in window.prepared.body_ids
    window.control_button.click()
    window.start_calculation()
    window.calculation.advance(5)
    window.calculation.cancel()
    window.finish_calculation()
    text = window.comparison_label.text()
    assert 'incomplete' in text.lower() and 'cancelled' in text.lower() and 'common' in text.lower()
    assert text in window.scene.result_info.get_text()
