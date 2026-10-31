"""Draft editing must preserve evolved state until an explicit restart."""
import matplotlib.pyplot as plt
from matplotlib.backend_bases import CloseEvent, MouseEvent
import numpy as np
import pytest

from simulations.star_system.plotting import App
from simulations.star_system.populations import Population, with_population
from simulations.star_system.core import make_system


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


def test_open_cancel_and_window_close_preserve_evolved_state(app):
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
    ('seed', '-1'), ('inner', 'nan'), ('inner', '.001'), ('outer', '1')])
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
