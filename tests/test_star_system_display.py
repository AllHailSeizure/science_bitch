"""Observable rendering and control checks using the repository's Agg backend."""
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
import numpy as np
import pytest
from PIL import Image

from simulations.star_system.core import make_system, SOLAR_RADIUS_AU
from simulations.star_system.data import STARS, PLANETS
from simulations.star_system.plotting import App
from simulations.star_system.rendering import (
    MAX_TRAIL_VERTICES, coordinates, draw, planet_sprite, star_sprite,
)


@pytest.fixture
def app():
    instance = App()
    instance.paused = True
    instance.fig.canvas.draw()
    instance.timer.stop()
    yield instance
    plt.close(instance.fig)


def click(canvas, ax, point=None):
    pixel = ax.transAxes.transform((.5, .5)) if point is None else ax.transData.transform(point)
    for name in ('button_press_event', 'button_release_event'):
        event = MouseEvent(name, canvas, *pixel, button=1)
        canvas.callbacks.process(name, event)


@pytest.mark.parametrize('direction, bright_half', [((1, 0), 'right'), ((-1, 0), 'left'),
                                                   ((0, 1), 'top'), ((0, -1), 'bottom')])
def test_planets_are_illuminated_toward_the_star(direction, bright_half):
    sprite = planet_sprite('earth', direction)
    halves = {'left': sprite[25:55, 12:32, :3], 'right': sprite[25:55, 48:68, :3],
              'bottom': sprite[12:32, 25:55, :3], 'top': sprite[48:68, 25:55, :3]}
    opposite = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}
    assert halves[bright_half].mean() > 3*halves[opposite[bright_half]].mean()
    assert sprite[0, 0, 3] == 0
    assert np.all(np.isfinite(sprite))


def test_stars_have_a_luminous_disk_and_translucent_halo():
    sprite = star_sprite('sun')
    assert sprite[80, 80, 3] == 1
    assert 0 < sprite[80, 125, 3] < .2
    assert sprite[0, 0, 3] == 0
    assert sprite[80, 80, :3].mean() > .7


def test_scene_keeps_artists_and_camera_and_places_bodies_on_their_orbits():
    system = make_system()
    fig, ax = plt.subplots()
    try:
        scene = draw(ax, system, 'sun', ['earth', 'jupiter'])
        artists = tuple(ax.get_children())
        ax.set_xlim(-5, 7)
        ax.set_ylim(-6, 6)
        history = []
        for _ in range(5):
            system.step()
            history.append((system.r-system.r[0]).copy())
        scene.update(system, history)
        fig.canvas.draw()
        assert tuple(ax.get_children()) == artists
        assert ax.get_xlim() == (-5, 7)
        assert ax.get_ylim() == (-6, 6)
        expected = coordinates(system.r-system.r[0])
        for index, artist in enumerate(scene.bodies, 1):
            bounds = artist.offsetbox.get_window_extent(fig.canvas.get_renderer())
            actual_center = (bounds.x0+bounds.width/2, bounds.y0+bounds.height/2)
            np.testing.assert_allclose(actual_center, ax.transData.transform(expected[index]), atol=.01)
        assert len(scene.trails[0].get_segments()) == 4
        alpha = scene.trails[0].get_colors()[:, 3]
        assert np.all(np.diff(alpha) > 0)
    finally:
        plt.close(fig)


def test_pause_and_single_step_buttons(app):
    initial_time = app.system.time
    app.update(0)
    assert app.system.time == initial_time
    click(app.fig.canvas, app.step.ax)
    assert app.paused
    assert app.system.time == pytest.approx(initial_time+app.system.dt)
    assert len(app.trails) == 1
    click(app.fig.canvas, app.pause.ax)
    assert not app.paused
    assert app.pause.label.get_text() == 'Pause'


def test_keyboard_controls(app):
    canvas = app.fig.canvas
    canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', canvas, key=' '))
    assert not app.paused
    canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', canvas, key='.'))
    assert app.paused
    assert app.system.time == pytest.approx(app.system.dt)


def test_clock_advances_only_when_running(app, monkeypatch):
    app.toggle_pause()
    monkeypatch.setattr('simulations.star_system.plotting.time.monotonic', lambda: app.last+.05)
    app.update(1)
    assert app.system.time > 0
    app.toggle_pause()
    stopped = app.system.time
    app.update(2)
    assert app.system.time == stopped
    assert app.pending == 0


def test_zoom_anchors_at_cursor_and_fit_restores_view(app):
    original = app.ax.get_xlim()
    point = app.scene.positions[1]
    pixel = app.ax.transData.transform(point)
    event = MouseEvent('scroll_event', app.fig.canvas, *pixel, button='up', step=1)
    app.fig.canvas.callbacks.process('scroll_event', event)
    assert np.diff(app.ax.get_xlim())[0] < np.diff(original)[0]
    np.testing.assert_allclose(app.ax.transData.transform(point), pixel, atol=1)
    click(app.fig.canvas, app.home.ax)
    np.testing.assert_allclose(app.ax.get_xlim(), original)


def test_click_inspects_body_and_follow_tracks_its_motion(app):
    click(app.fig.canvas, app.ax, app.scene.positions[1])
    assert app.selected_index == 1
    assert app.inspector.get_text().startswith('Earth\n')
    click(app.fig.canvas, app.follow.ax)
    assert app.following
    app.step_once()
    center = (np.mean(app.ax.get_xlim()), np.mean(app.ax.get_ylim()))
    np.testing.assert_allclose(center, app.scene.positions[1])
    app.fit_view()
    assert not app.following


def test_drag_pans_without_selecting_a_body(app):
    app.following = True
    app.fig.canvas.draw()
    pixel = app.ax.transAxes.transform((.45, .6))
    old_xlim = app.ax.get_xlim()
    press = MouseEvent('button_press_event', app.fig.canvas, *pixel, button=1)
    motion = MouseEvent('motion_notify_event', app.fig.canvas, pixel[0]+30, pixel[1]+20, button=1)
    release = MouseEvent('button_release_event', app.fig.canvas, pixel[0]+30, pixel[1]+20, button=1)
    for event in (press, motion, release):
        app.fig.canvas.callbacks.process(event.name, event)
    assert app.ax.get_xlim()[0] < old_xlim[0]
    assert not app.following
    assert app.selected_index == 0
    assert app._drag is None


def test_existing_add_distance_and_star_controls_still_rebuild(app):
    original_axes = len(app.fig.axes)
    app.planet_picker.set_active(list(PLANETS).index('mercury'))
    click(app.fig.canvas, app.add.ax)
    assert app.planets[-1] == 'mercury'
    click(app.fig.canvas, app.add.ax)
    assert app.planets.count('mercury') == 1
    assert len(app.fig.axes) == original_axes+1
    app.step_once()
    app.sliders[0].set_val(np.log10(.01))
    assert app.system.time == 0
    assert app.trails == []
    app.star_picker.set_active(list(STARS).index('rigel'))
    minimum = STARS['rigel'].radius*SOLAR_RADIUS_AU*1.02
    assert app.distances[0] == pytest.approx(minimum)
    assert app.system.time == 0
    assert app.paused
    assert app.inspector.get_text().startswith('Rigel\n')


def test_all_planets_fit_in_control_panel(app):
    for planet in PLANETS:
        app.planet_picker.set_active(list(PLANETS).index(planet))
        app.add_planet(None)
    app.fig.canvas.draw()
    assert len(app.sliders) == 10
    for slider in app.sliders:
        assert .245 < slider.ax.get_position().y0 < .48
    assert len(app.scene.bodies) == 10


def test_close_encounter_stays_halted_until_rebuilt(app):
    app.system.r[1] = app.system.r[0] + (.00001, 0)
    app.step_once()
    assert app.halted
    assert 'Close encounter' in app.status.get_text()
    assert app.mode.get_text() == 'HALTED'
    app.step_once()
    assert app.system.time == 0
    app.restart()
    assert not app.halted
    assert app.status.get_text() == ''


def test_running_frames_reuse_background_without_full_redraw(app, monkeypatch):
    """Ordinary motion must not rerender every static control and guide."""
    full_draws = []
    original_draw = app.fig.canvas.draw
    def record_draw(*args, **kwargs):
        full_draws.append(True)
        return original_draw(*args, **kwargs)
    monkeypatch.setattr(app.fig.canvas, 'draw', record_draw)
    app.paused = False
    for frame in range(3):
        app.last -= .02
        app.update(frame)
    assert app.system.time > 0
    assert full_draws == []


def test_trail_display_is_bounded_without_losing_physics_history(app):
    app._advance(1500)
    app._refresh()
    assert len(app.trails) == 1500
    trail = app.scene.trails[0].get_segments()
    assert len(trail) == MAX_TRAIL_VERTICES-1
    np.testing.assert_allclose(trail[0][0], coordinates(app.trails[0][1]))
    np.testing.assert_allclose(trail[-1][-1], app.scene.positions[1])


def test_budgeted_frames_account_for_only_completed_physics_steps(app, monkeypatch):
    elapsed = [0.]
    original_step = app.system.step
    def slow_step():
        original_step()
        elapsed[0] += .004
    app.speed.set_val(10.)
    monkeypatch.setattr(app.system, 'step', slow_step)
    monkeypatch.setattr('simulations.star_system.plotting.time.perf_counter', lambda: elapsed[0])
    monkeypatch.setattr('simulations.star_system.plotting.time.monotonic', lambda: 1.1)
    app.paused = False
    app.last = 1.
    app.update(0)
    assert app.system.time == pytest.approx(2*app.system.dt)
    assert len(app.trails) == 2
    assert app.pending == pytest.approx(1.-2*app.system.dt)


def test_partial_redraw_matches_full_redraw_after_motion_and_resize(app):
    app._advance(25)
    app._refresh()
    incremental = np.asarray(app.fig.canvas.buffer_rgba()).copy()
    app.fig.canvas.draw()
    np.testing.assert_array_equal(incremental, np.asarray(app.fig.canvas.buffer_rgba()))
    app.fig.set_size_inches(12, 8)
    app.fig.canvas.draw()
    app._advance(10)
    app._refresh()
    incremental = np.asarray(app.fig.canvas.buffer_rgba()).copy()
    app.fig.canvas.draw()
    np.testing.assert_array_equal(incremental, np.asarray(app.fig.canvas.buffer_rgba()))


def test_export_keeps_inspector_text_and_recovers_live_cache(app):
    app.selected_index = 1
    app.step_once()
    live = np.asarray(app.fig.canvas.buffer_rgba()).copy()
    output = BytesIO()
    app.fig.savefig(output, format='png', dpi=app.fig.dpi)
    output.seek(0)
    saved = np.asarray(Image.open(output).convert('RGBA'))
    height, width = live.shape[:2]
    inspector = np.s_[int(height*.81):int(height*.9), int(width*.03):int(width*.28)]
    np.testing.assert_array_equal(live[inspector], saved[inspector])
    app.step_once()
    assert app.painter.background is not None
    assert app.system.time == pytest.approx(2*app.system.dt)


def test_backends_without_blitting_still_draw(app, monkeypatch):
    app.painter.use_blit = False
    app.painter.set_artists(app.painter.artists)
    full_draws = []
    original_draw = app.fig.canvas.draw
    def record_draw(*args, **kwargs):
        full_draws.append(True)
        return original_draw(*args, **kwargs)
    monkeypatch.setattr(app.fig.canvas, 'draw', record_draw)
    app.step_once()
    assert full_draws
    assert app.system.time == pytest.approx(app.system.dt)
