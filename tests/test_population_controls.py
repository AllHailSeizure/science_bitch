"""Draft editing must preserve evolved state until an explicit restart."""
from dataclasses import replace
import matplotlib.pyplot as plt
from matplotlib.backend_bases import CloseEvent, KeyEvent, MouseEvent
import numpy as np
import pytest

from simulations.star_system.plotting import App
from simulations.star_system.populations import Population, with_population
from simulations.star_system.core import G, kepler_period, make_system
from simulations.star_system.model import BodySpec, InitialBody, InitialSetup, PopulationSpec
from simulations.star_system.initialization import build_system, insert_body


@pytest.fixture
def app():
    instance = App(population=Population(count=12, seed=17))
    yield instance
    instance.close()
    plt.close(instance.fig)


def click(button):
    canvas = button.ax.figure.canvas
    canvas.draw()
    x, y = button.ax.transAxes.transform((.5, .5))
    for name in ('button_press_event', 'button_release_event'):
        canvas.callbacks.process(name, MouseEvent(name, canvas, x, y, button=1))


@pytest.fixture
def authored_app():
    setup = InitialSetup('Authored binary', (
        InitialBody('first', BodySpec('First star', 'star', 1., .00465), (-10, 0), (.1, -.2)),
        InitialBody('second', BodySpec('Second star', 'star', .6, .003), (10, 2), (-.3, .4)),
        InitialBody('custom', BodySpec('Custom planet', 'planet', 3e-6, .00004), (1, 0), (0, 6.28), 'first'),
        InitialBody('dust', BodySpec('Explicit particle', 'particle', 0., None), (50, 0), (0, 0)),
        InitialBody('unknown', BodySpec('Unknown radius', 'planet', 1e-6, None), (-50, 0), (0, 0))),
        dt=.0001)
    instance = App(setup=setup)
    instance.timer.stop()
    yield instance
    instance.close()
    plt.close(instance.fig)


def test_population_editor_uses_selected_initial_reference_and_reopens(authored_app):
    app = authored_app
    starting = app.initial_setup
    app.selected_id = 'second'
    live = InitialBody('live-only', BodySpec('Live only', 'planet', 1e-7, .00001), (60, 0), (0, 0))
    app._install_prepared(insert_body(app.prepared, live))
    app._advance(5)
    app.edit_population()
    editor = app.population_editor
    assert editor.fields['reference'].text == 'second'
    assert 'Second star' in editor.reference_label.get_text()
    assert 'second' in editor.reference_label.get_text()
    assert 'initial' in editor.reference_label.get_text().lower()
    assert editor.apply.label.get_text() == 'Apply and restart'
    editor.fields['count'].set_val('8')
    editor.fields['seed'].set_val('72')
    click(editor.apply)
    assert app.population_editor is None
    assert app.initial_setup.bodies == starting.bodies
    assert app.initial_setup.population.reference_id == 'second'
    assert app.system.time == 0
    assert 'live-only' not in app.prepared.body_ids
    expected = build_system(app.initial_setup)
    np.testing.assert_array_equal(app.system.r, expected.system.r)
    np.testing.assert_array_equal(app.system.v, expected.system.v)
    initial = build_system(starting)
    np.testing.assert_array_equal(app.system.r[:5], initial.system.r)
    np.testing.assert_array_equal(app.system.v[:5], initial.system.v)
    app.selected_id = 'first'
    app.edit_population()
    editor = app.population_editor
    assert editor.fields['reference'].text == 'second'
    assert editor.fields['seed'].text == '72'
    editor.fields['seed'].set_val('73')
    click(editor.apply)
    assert app.population.seed == 73
    assert app.initial_setup.population.id == 'population'
    assert not np.array_equal(app.system.r[5:], expected.system.r[5:])


def test_existing_population_identity_and_settings_survive_edit(authored_app):
    app = authored_app
    settings = Population(kind='ring', count=8, inner=1., outer=1., seed=29)
    app.initial_setup = replace(app.initial_setup, population=PopulationSpec('stable-rocks', 'second', settings))
    app.restart()
    assert app.population == settings
    app.selected_id = 'first'
    app.edit_population()
    editor = app.population_editor
    assert editor.fields['reference'].text == 'second'
    assert editor.fields['seed'].text == '29'
    editor.fields['seed'].set_val('30')
    click(editor.apply)
    assert app.initial_setup.population.id == 'stable-rocks'
    assert app.prepared.body_ids[5:] == tuple(f'stable-rocks:{i}' for i in range(8))


def test_population_preparation_discloses_fixed_dt_and_stays_paused(authored_app):
    app = authored_app
    app.prepare_setup()
    app.selected_id = 'second'
    app.edit_population()
    editor = app.population_editor
    assert editor.apply.label.get_text() == 'Apply'
    assert editor.remove.label.get_text() == 'Remove'
    editor.kind.set_active(0)
    editor.fields['inner'].set_val('.01')
    editor.fields['count'].set_val('8')
    assert 'Fixed dt' in editor.timestep_label.get_text()
    click(editor.apply)
    assert app.preparing and app.paused
    assert app.system.time == 0
    assert app.system.dt == pytest.approx(kepler_period(.01, G*.6)/600)
    assert app.initial_setup.dt == app.system.dt
    assert 'dt' in app.status.get_text()
    app.update(None)
    assert app.system.time == 0
    app.edit_population()
    click(app.population_editor.remove)
    assert app.preparing and app.paused
    assert app.initial_setup.population is None
    assert len(app.system.masses) == 5


@pytest.mark.parametrize('reference', ['missing', 'dust', 'unknown'])
def test_invalid_population_reference_preserves_active_setup_and_draft(authored_app, reference):
    app = authored_app
    app.selected_id = 'second'
    app.edit_population()
    editor = app.population_editor
    previous, setup = app.system, app.initial_setup
    editor.fields['seed'].set_val('91')
    editor.fields['reference'].set_val(reference)
    click(editor.apply)
    assert app.population_editor is editor
    assert editor.error.get_text()
    assert app.system is previous and app.initial_setup is setup
    assert editor.fields['seed'].text == '91'
    assert editor.fields['reference'].text == reference
    editor.fields['reference'].set_val('second')
    click(editor.apply)
    assert app.population_editor is None
    assert app.population.seed == 91


def test_particle_selection_requires_explicit_population_reference(authored_app):
    app = authored_app
    app.selected_id = 'dust'
    app.edit_population()
    editor = app.population_editor
    assert editor.fields['reference'].text == ''
    click(editor.apply)
    assert app.population_editor is editor
    assert 'reference' in editor.error.get_text().lower()
    click(editor.reference_next)
    assert editor.fields['reference'].text == 'first'


def test_population_reference_click_cannot_place_or_select_in_scene(authored_app):
    app = authored_app
    app.selected_id = 'first'
    app.edit_population()
    editor = app.population_editor
    before = app.system.r.copy(), app.system.v.copy(), app.prepared.body_ids
    click(editor.reference_next)
    assert editor.fields['reference'].text == 'second'
    assert app.body_editor is None and app._drag is None
    app.fig.canvas.draw()
    x, y = app.ax.transAxes.transform((.8, .7))
    for name in ('button_press_event', 'button_release_event'):
        app.fig.canvas.callbacks.process(name, MouseEvent(name, app.fig.canvas, x, y, button=1))
    assert app._drag is None and app.selected_id == 'first'
    np.testing.assert_array_equal(app.system.r, before[0])
    np.testing.assert_array_equal(app.system.v, before[1])
    assert app.prepared.body_ids == before[2]


def test_population_reference_cannot_be_removed_while_drafting_or_preparing(authored_app):
    app = authored_app
    app.prepare_setup()
    app.selected_id = 'second'
    app.edit_population()
    editor = app.population_editor
    previous = app.initial_setup
    app.remove_body()
    assert app.initial_setup is previous
    editor.fields['count'].set_val('8')
    click(editor.apply)
    app.selected_id = 'second'
    valid, previous = app.system, app.initial_setup
    app.remove_body()
    assert app.initial_setup is previous and app.system is valid
    assert 'reference' in app.status.get_text().lower()


@pytest.mark.parametrize('paused', [False, True])
def test_escape_cancel_retains_authored_evolved_state_exactly(authored_app, paused):
    app = authored_app
    app._advance(5)
    app.paused = paused
    before, setup = app.system, app.initial_setup
    r, v, elapsed = before.r.copy(), before.v.copy(), before.time
    app.edit_population()
    editor = app.population_editor
    editor.fields['seed'].set_val('999')
    app.fig.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', app.fig.canvas, key='escape'))
    assert app.population_editor is None
    assert app.paused == paused and app.system is before and app.initial_setup is setup
    np.testing.assert_array_equal(app.system.r, r)
    np.testing.assert_array_equal(app.system.v, v)
    assert app.system.time == elapsed


def test_open_cancel_and_window_close_preserve_evolved_state(app):
    app.toggle_pause()  # Startup is preparation; explicitly run before cancel testing.
    app._advance(5)
    previous, positions, velocities = app.system, app.system.r.copy(), app.system.v.copy()
    elapsed = app.system.time
    app.following = True
    click(app.population_button)
    editor = app.population_editor
    assert app.paused
    assert not app.pause.active
    editor.fields['seed'].set_val('99')
    app.on_key(type('Event', (), {'key': ' '})())
    app.step_once()
    app.update(None)
    assert app.system.time == elapsed
    click(editor.cancel)
    assert app.population_editor is None
    assert not app.paused
    assert app.following
    assert app.pause.active
    assert app.system is previous
    np.testing.assert_array_equal(app.system.r, positions)
    np.testing.assert_array_equal(app.system.v, velocities)
    assert app.population.seed == 17
    app.edit_population()
    editor = app.population_editor
    editor.fig.canvas.callbacks.process('close_event', CloseEvent('close_event', editor.fig.canvas))
    assert app.population_editor is None
    assert not app.paused
    assert app.system is previous


@pytest.mark.parametrize('kind', ['ring', 'disk', 'belt'])
def test_apply_restarts_paused_with_authored_settings(app, kind):
    app._advance(5)
    app.selected_index = 3
    app.following = True
    app.edit_population()
    editor = app.population_editor
    editor.kind.set_active(['ring', 'disk', 'belt'].index(kind))
    editor.angles.set_active(1)
    for name, value in dict(count='24', inner='2.5', outer='4', seed='42').items():
        editor.fields[name].set_val(value)
    click(editor.apply)
    assert app.population_editor is None
    assert app.population == Population(kind=kind, count=24, inner=2.5,
                                        outer=2.5 if kind == 'ring' else 4., seed=42, angles='even')
    expected = with_population(make_system(), app.population)
    np.testing.assert_array_equal(app.system.r, expected.r)
    np.testing.assert_array_equal(app.system.v, expected.v)
    assert app.system.time == 0
    assert app.paused
    assert not app.following
    assert app.selected_index == 0
    app.restart()
    np.testing.assert_array_equal(app.system.r, expected.r)
    app.edit_population()
    assert app.population_editor.fields['seed'].text == '42'
    app.population_editor.finish()
    assert app.paused


@pytest.mark.parametrize('field,value', [('count', '1.5'), ('count', '2001'),
    ('seed', '-1'), ('inner', 'nan'), ('inner', '1e308'), ('inner', '.001'), ('outer', '1')])
def test_invalid_draft_is_recoverable(app, field, value):
    app._advance(2)
    previous, settings = app.system, app.population
    app.edit_population()
    editor = app.population_editor
    editor.fields[field].set_val(value)
    click(editor.apply)
    assert app.population_editor is editor
    assert editor.error.get_text()
    assert app.system is previous
    assert app.population == settings
    assert app.paused
    editor.fields[field].set_val(str(getattr(settings, field)))
    click(editor.apply)
    assert app.population_editor is None


def test_remove_restarts_without_population(app):
    app.selected_index = 10
    app.following = True
    app.edit_population()
    click(app.population_editor.remove)
    assert app.population is None
    assert app.paused
    assert app.system.time == 0
    assert len(app.system.masses) == 3
    assert len(app.scene.particles.get_offsets()) == 0
    assert app.selected_index == 0
    app.edit_population()
    assert not app.population_editor.remove.active
    click(app.population_editor.apply)
    assert app.population is not None
    assert len(app.system.masses) == 203


def test_editor_single_instance_and_main_window_close(app):
    app.edit_population()
    editor = app.population_editor
    app.edit_population()
    assert app.population_editor is editor
    app.close()
    assert app.population_editor is None
    assert not plt.fignum_exists(editor.fig.number)


def test_ring_has_only_one_editable_radius(app):
    app.edit_population()
    editor = app.population_editor
    editor.kind.set_active(0)
    assert not editor.fields['outer'].active
    editor.kind.set_active(2)
    assert editor.fields['outer'].active


def test_population_editor_preserves_sidebar_visibility_and_disables_catalog(app):
    app.sidebar.prepare_circular()
    app.edit_population()
    canvas = app.fig.canvas
    editor = app.population_editor
    assert not app.add.active
    assert not app.star_picker.active
    assert not any(slider.active for slider in app.sliders)
    for key in ('m', '.', ' ', 'home'):
        canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', canvas, key=key))
    assert app.sidebar.visible
    assert app.system.time == 0
    editor.finish()
    assert app.add.active
    assert all(slider.active for slider in app.sliders)
    app.toggle_sidebar()
    app.edit_population()
    app.population_editor.finish()
    assert not app.sidebar.visible
    assert not app.pause.active
    assert not app.add.active
