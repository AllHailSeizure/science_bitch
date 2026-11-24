"""Exercise native widgets and scene callbacks through the normal desktop workflow."""
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from simulations.star_system.model import BodySpec, InitialBody, InitialSetup
from simulations.star_system.persistence import setup_digest


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


def open_solar(window, qtapp):
    window.choose_button.click()
    window.open_button.click()
    qtapp.processEvents()


def widget_point(window, world):
    """Matplotlib event pixels are physical; Qt mouse positions are logical."""
    from PyQt6.QtCore import QPoint
    display = window.scene.world_to_display(world)
    pixel = window.ax.transData.transform(display)
    ratio = float(getattr(window.canvas, 'device_pixel_ratio', 1) or 1)
    return QPoint(round(pixel[0]/ratio), round((window.figure.bbox.height-pixel[1])/ratio))


def test_launch_menu_then_frozen_solar_system(window, qtapp):
    assert window.session is None
    assert window.stack.currentWidget() is window.menu_page
    window.choose_button.click()
    assert window.stack.currentWidget() is window.library_page
    assert window.session is None
    window.open_button.click()
    qtapp.processEvents()
    assert window.setup.name == 'Solar System'
    assert len(window.prepared.body_ids) == 9
    assert window.editing and not window.running
    assert window.prepared.system.time == 0
    assert len(window.context_points.get_offsets()) == 0
    assert not window.scene.reference_guides.get_visible()


def test_module_default_uses_desktop_entry(monkeypatch):
    from simulations.star_system import desktop
    from simulations.star_system.__main__ import main
    monkeypatch.setattr(desktop, 'main', lambda: 23)
    assert main([]) == 23


def test_dropdown_info_add_then_scene_placement(window, qtapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    open_solar(window, qtapp)
    original = setup_digest(window.setup)
    window.tabs.setCurrentIndex(1)
    window.catalog.type_combo.setCurrentText('Other')
    window.catalog.body_combo.setCurrentIndex(window.catalog.body_combo.findData('vesta'))
    qtapp.processEvents()
    assert window.catalog.information.name.text() == 'Vesta'
    window.catalog.add_button.click()
    assert window.pending_body is not None
    assert setup_digest(window.setup) == original
    world = window.prepared.system.r[0]+(2.1, 1.2)
    QTest.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=widget_point(window, world))
    qtapp.processEvents()
    assert window.pending_body is None
    assert len(window.setup.bodies) == 10
    assert window.setup.bodies[-1].spec.name == 'Vesta'
    assert window.selected_id == window.setup.bodies[-1].id
    np.testing.assert_allclose(window.prepared.system.r[-1], world, atol=.05)
    assert len(window.original_setup.bodies) == 9


def test_escape_cancels_placement_without_editing(window, qtapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    open_solar(window, qtapp)
    before = setup_digest(window.setup)
    window.catalog.add_button.click()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    qtapp.processEvents()
    assert window.pending_body is None
    assert setup_digest(window.setup) == before
    assert not window.dirty


@pytest.mark.parametrize('scale', ['1', '2'])
def test_scene_placement_works_at_device_scale(scale):
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env['QT_SCALE_FACTOR'] = scale
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    completed = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '--tb=line', '-p', 'no:cacheprovider',
         'tests/test_desktop.py::test_dropdown_info_add_then_scene_placement'],
        cwd=root, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout+'\n'+completed.stderr


def test_mass_edit_restore_and_undo_preserve_frozen_original(window, qtapp):
    open_solar(window, qtapp)
    original = setup_digest(window.original_setup)
    velocity = window.prepared.system.v.copy()
    window.select_body('sun')
    window.tabs.setCurrentIndex(2)
    window.properties.mass_factor.setValue(2)
    window.properties.apply_button.click()
    assert window.setup.bodies[0].spec.mass == 2
    np.testing.assert_array_equal(window.prepared.system.v, velocity)
    assert setup_digest(window.original_setup) == original
    window.original_button.click()
    assert setup_digest(window.setup) == original
    window.undo_button.click()
    assert window.setup.bodies[0].spec.mass == 2


def test_replace_body_keeps_starting_motion_and_identity(window, qtapp):
    open_solar(window, qtapp)
    window.select_body('jupiter')
    old_r, old_v = window.prepared.system.r.copy(), window.prepared.system.v.copy()
    widget = window.properties
    widget.replacement.setCurrentIndex(widget.replacement.findData('saturn'))
    widget.apply_button.click()
    index = window.prepared.body_ids.index('jupiter')
    assert window.prepared.body_specs[index].name == 'Saturn'
    np.testing.assert_array_equal(window.prepared.system.r, old_r)
    np.testing.assert_array_equal(window.prepared.system.v, old_v)


def test_run_step_and_reset_use_edited_starting_setup(window, qtapp):
    open_solar(window, qtapp)
    window.run_button.click()
    assert window.running and not window.editing
    assert not window.tabs.isTabEnabled(1)
    window.step_button.click()
    assert not window.running and window.prepared.system.time > 0
    window.reset_button.click()
    assert window.editing and window.prepared.system.time == 0
    assert window.tabs.isTabEnabled(1)


def test_selection_replaces_and_clears_contextual_guides(window, qtapp):
    window.open_experiment('earth-moon')
    qtapp.processEvents()
    window.select_body('earth')
    first = window.context_points.get_offsets().copy()
    assert len(first) == 2
    assert len(window.context_orbits.get_segments()) == 1
    window.select_body('moon')
    assert not np.allclose(window.context_points.get_offsets(), first)
    assert len(window.context_orbits.get_segments()) == 0
    window.select_body(None)
    assert len(window.context_points.get_offsets()) == 0
    assert not any(item.get_visible() for item in window.context_names)


def test_focus_frames_the_selected_planets_moon_without_changing_state(window, qtapp):
    window.open_experiment('earth-moon')
    qtapp.processEvents()
    window.select_body('earth')
    before = window.prepared.system.r.copy()
    old_width = np.ptp(window.ax.get_xlim())
    window.focus_button.click()
    assert np.ptp(window.ax.get_xlim()) < old_width/100
    assert window.following
    np.testing.assert_array_equal(window.prepared.system.r, before)


def test_planetary_view_shows_selected_planet_and_its_moon(window, qtapp):
    from PyQt6.QtCore import Qt
    window.open_experiment('earth-moon')
    before = window.prepared.system.r.copy()
    window.select_body('earth')
    assert window.planetary_button.isEnabled()
    window.planetary_button.click()
    assert window.planetary_id == 'earth'
    assert window.system_view_button.isVisible()
    assert window.scene.origin_id == 'earth'
    assert {window.body_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(window.body_list.count())} == {'earth', 'moon'}
    assert not window.scene.artists_by_id['sun'].get_visible()
    assert window.scene.artists_by_id['moon'].get_visible()
    assert len(window.context_orbits.get_segments()) == 1
    assert 'planetary view' in window.scene.heading.get_text()
    window.select_body('moon')
    assert window.selected_id == 'moon'
    assert window.system_view_button.isVisible()
    window.system_view_button.click()
    assert window.planetary_id is None
    assert not window.system_view_button.isVisible()
    assert window.scene.origin_id == 'sun'
    assert window.scene.artists_by_id['sun'].get_visible()
    np.testing.assert_array_equal(window.prepared.system.r, before)


def test_planetary_view_return_is_visible_without_a_moon(window, qtapp):
    from PyQt6.QtCore import Qt
    open_solar(window, qtapp)
    window.select_body('mercury')
    window.planetary_button.click()
    assert window.planetary_id == 'mercury'
    assert window.system_view_button.isVisible()
    assert {window.body_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(window.body_list.count())} == {'mercury'}
    window.system_view_button.click()
    assert window.planetary_id is None
    assert not window.system_view_button.isVisible()
    assert window.scene.artists_by_id['sun'].get_visible()


def test_planetary_view_membership_is_the_planet_and_its_moons(window, qtapp):
    from PyQt6.QtCore import Qt
    from simulations.star_system.model import BodySpec
    from simulations.star_system.templates import place_body
    window.open_experiment('earth-moon')
    earth = window.prepared.body_ids.index('earth')
    tracer = BodySpec('tracer', 'particle', 0., None)
    setup = place_body(window.setup, tracer, window.prepared.system.r[earth]+(0.002, 0.001),
                       'earth', circular=False, velocity=(0., 0.), body_id='tracer')
    window.install_experiment(setup)
    window.select_body('earth')
    window.planetary_button.click()
    listed = {window.body_list.item(i).data(Qt.ItemDataRole.UserRole)
              for i in range(window.body_list.count())}
    assert listed == window.visible_body_ids() == {'earth', 'moon'}
    assert window.scene.artists_by_id['earth'].get_visible()
    assert window.scene.artists_by_id['moon'].get_visible()
    assert not window.scene.particles.get_visible()
    window.select_body('tracer')
    assert window.selected_id is None


def test_save_reload_and_bad_load_are_transactional(window, qtapp, tmp_path):
    open_solar(window, qtapp)
    window.select_body('sun')
    window.properties.mass_factor.setValue(1.5)
    window.properties.apply_button.click()
    filename = tmp_path/'mine.json'
    window.save_to(filename)
    expected = setup_digest(window.setup)
    window.load_from(filename)
    assert setup_digest(window.setup) == expected
    assert window.editing and not window.dirty
    bad = tmp_path/'invalid.json'
    bad.write_text('{"invalid": true}')
    with pytest.raises(ValueError):
        window.load_from(bad)
    assert setup_digest(window.setup) == expected


def test_recordings_keep_their_source_when_edited_setup_changes(window, qtapp):
    open_solar(window, qtapp)
    original_digest = setup_digest(window.setup)
    window.duration.setValue(.01)
    window.stride.setValue(2)
    window.start_calculation()
    for _ in range(100):
        if window.calculation.done:
            break
        window.tick()
    assert window.calculation.done
    assert window.playback.recording.setup_digest == original_digest
    window.step_once()
    assert window.prepared.system.time > 0
    window.edit_starting_setup()
    window.select_body('sun')
    window.properties.mass_factor.setValue(2)
    window.properties.apply_button.click()
    edited_digest = setup_digest(window.setup)
    assert edited_digest != original_digest
    assert window.playback is None
    assert window.result_combo.currentIndex() == -1
    opened = window.session.recordings[0]
    window.result_combo.setCurrentIndex(0)
    assert window.playback.recording.setup_digest == original_digest
    assert window.prepared.system.time == opened.frames[0].time
    assert window.prepared.system.masses[0] == 1
    assert setup_digest(window.setup) == edited_digest
    window.edit_starting_setup()
    assert window.prepared.system.masses[0] == 2
    assert window.playback is None
    window.result_combo.blockSignals(True)
    window.result_combo.setCurrentIndex(0)
    window.result_combo.blockSignals(False)
    assert window.playback is None
    window.result_combo.activated.emit(0)
    assert window.playback.recording.setup_digest == original_digest
    assert window.prepared.system.time == opened.frames[0].time
    assert window.prepared.system.masses[0] == 1


def test_invalid_property_and_overlapping_placement_preserve_setup(window, qtapp):
    open_solar(window, qtapp)
    before = setup_digest(window.setup)
    window.select_body('sun')
    window.properties.fields['mass'].setText('-5')
    window.properties.apply_button.click()
    assert setup_digest(window.setup) == before
    window.catalog.add_button.click()
    window.place_at(window.prepared.system.r[0])
    assert window.pending_body is not None
    assert setup_digest(window.setup) == before
    window.cancel_interaction()


def test_hidden_velocity_arrow_does_not_change_starting_motion(window, qtapp):
    open_solar(window, qtapp)
    window.select_body('earth')
    tip = window.ax.transData.transform(window.velocity_tips['earth'])
    before = window.prepared.system.v.copy()
    undo = len(window.undo_stack)
    window.arrows_checkbox.setChecked(False)
    qtapp.processEvents()
    assert window.velocity_tips == {}
    window.on_press(SimpleNamespace(inaxes=window.ax, button=1, xdata=1., ydata=0., x=tip[0], y=tip[1]))
    moved = tip+(24, 0)
    window.on_motion(SimpleNamespace(inaxes=window.ax, xdata=1., ydata=0., x=moved[0], y=moved[1]))
    window.on_release(SimpleNamespace(inaxes=window.ax, xdata=1., ydata=0., x=moved[0], y=moved[1]))
    np.testing.assert_array_equal(window.prepared.system.v, before)
    assert len(window.undo_stack) == undo


def test_dragging_velocity_changes_starting_motion_only_on_release(window, qtapp):
    open_solar(window, qtapp)
    window.select_body('earth')
    i = window.prepared.body_ids.index('earth')
    before = window.prepared.system.v.copy()
    tip = window.ax.transData.transform(window.velocity_tips['earth'])
    window.on_press(SimpleNamespace(inaxes=window.ax, button=1, xdata=1., ydata=0., x=tip[0], y=tip[1]))
    moved = tip+(15, 5)
    window.on_motion(SimpleNamespace(inaxes=window.ax, xdata=1., ydata=0., x=moved[0], y=moved[1]))
    np.testing.assert_array_equal(window.prepared.system.v, before)
    window.on_release(SimpleNamespace(inaxes=window.ax))
    assert not np.allclose(window.prepared.system.v[i], before[i])
    assert window.prepared.system.time == 0


def test_asteroid_preset_has_editable_starting_particles(window, qtapp):
    window.open_experiment('asteroid-ring')
    qtapp.processEvents()
    window.select_body('asteroids:0')
    assert window.tabs.isTabEnabled(2)
    assert 'asteroids:0' in window.velocity_tips
    assert window.properties.body_id == 'asteroids:0'
    replacement = window.properties.replacement
    replacement.setCurrentIndex(replacement.findData('vesta'))
    window.properties.apply_button.click()
    index = window.prepared.body_ids.index('asteroids:0')
    assert window.prepared.body_specs[index].name == 'Vesta'


def _context_world(window):
    return window.scene.display_to_world(np.asarray(window.context_points.get_offsets(), dtype=float))


def test_desktop_labels_l4_ahead_of_motion_and_omits_it_without_revolution(window, qtapp):
    plus = np.array((.5, np.sqrt(3)/2))
    minus = np.array((.5, -np.sqrt(3)/2))
    window.install_experiment(InitialSetup('Sun Jupiter pair', (
        InitialBody('sun', BodySpec('Sun', 'star', 1., 1e-4, temperature=5772.), (0., 0.), (0., 0.)),
        InitialBody('jupiter', BodySpec('Jupiter', 'planet', 1e-3, 1e-4, color='#c9b27c', catalog_key='jupiter'),
                    (1., 0.), (0., 1.), 'sun'),
    ), .01))
    qtapp.processEvents()
    window.select_body('jupiter')
    np.testing.assert_allclose(_context_world(window)[0], plus)
    assert [item.get_text() for item in window.context_names] == ['L4', 'L5']

    window.install_experiment(InitialSetup('Sun Jupiter pair', (
        InitialBody('sun', BodySpec('Sun', 'star', 1., 1e-4, temperature=5772.), (0., 0.), (0., 0.)),
        InitialBody('jupiter', BodySpec('Jupiter', 'planet', 1e-3, 1e-4, color='#c9b27c', catalog_key='jupiter'),
                    (1., 0.), (0., -1.), 'sun'),
    ), .01))
    qtapp.processEvents()
    window.select_body('jupiter')
    np.testing.assert_allclose(_context_world(window)[0], minus)
    assert [item.get_text() for item in window.context_names] == ['L4', 'L5']

    window.install_experiment(InitialSetup('Sun Jupiter pair', (
        InitialBody('sun', BodySpec('Sun', 'star', 1., 1e-4, temperature=5772.), (0., 0.), (0., 0.)),
        InitialBody('jupiter', BodySpec('Jupiter', 'planet', 1e-3, 1e-4, color='#c9b27c', catalog_key='jupiter'),
                    (1., 0.), (0., 0.), 'sun'),
    ), .01))
    qtapp.processEvents()
    window.select_body('jupiter')
    np.testing.assert_allclose(_context_world(window), np.vstack((plus, minus)))
    texts = [item.get_text() for item in window.context_names]
    assert all('L4' not in text and 'L5' not in text and 'guide' in text.lower() for text in texts)
    note = window.context_note.get_text()
    assert 'no direction of revolution' in note
    assert 'L4' not in note and 'L5' not in note
