"""Recorded time, identity and renderer checks under Agg, without reintegration."""
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from simulations.star_system.core import System
from simulations.star_system.history import StateFrame, relative_state
from simulations.star_system.initialization import build_system, circular_relative_velocity
from simulations.star_system.data import CATALOG, AU_KM
from simulations.star_system.model import InitialBody, InitialSetup, PopulationSpec
from simulations.star_system.persistence import save_setup, load_setup
from simulations.star_system.populations import Population
from simulations.star_system.recording import Calculation, CalculationRequest, RecordedRun
from simulations.star_system.playback import Playback
from simulations.star_system.plotting import App
from simulations.star_system.scenarios import get_scenario


def calculate(setup, steps=11, stride=4):
    calc = Calculation(setup, CalculationRequest(setup.dt*steps, stride))
    while not calc.done:
        calc.advance(3)
    return calc.result


def unexpected_step(*args, **kwargs):
    raise AssertionError('Recorded playback must not integrate')


def test_recorded_rotating_history_matches_live_samples(monkeypatch):
    setup = get_scenario('sun-jupiter-ring')
    run = calculate(setup, steps=41, stride=1)
    app = App(setup=setup); app.timer.stop()
    try:
        app.selected_id = 'population:7'
        app._advance(41); app._refresh()
        original_r, original_v, original_time = app.system.r.copy(), app.system.v.copy(), app.system.time
        app.set_frame_mode('rotating')
        live = np.asarray(app.scene.selected_trail.get_segments()).copy()
        np.testing.assert_array_equal(app.system.r, original_r)
        np.testing.assert_array_equal(app.system.v, original_v)
        assert app.system.time == original_time
        monkeypatch.setattr(System, 'step', unexpected_step)
        app.session.retain_recording(run); app.show_recording(run)
        app.selected_id = 'population:7'; app.following = True
        app.playback.play(); app.playback.advance(100, 1); app._display_recorded_frame()
        np.testing.assert_allclose(app.scene.selected_trail.get_segments(), live)
        assert app.selected_id == 'population:7' and app.following
        assert len(app.scene.selected_trail.get_segments()) == 41
        for mode in ('world', 'sun-relative', 'rotating'):
            app.set_frame_mode(mode)
            np.testing.assert_array_equal(app.system.r, run.frames[-1].r)
            np.testing.assert_array_equal(app.system.v, run.frames[-1].v)
            assert app.system.time == run.frames[-1].time
    finally:
        app.close(); plt.close(app.fig)


@pytest.fixture
def app():
    app = App()
    app.timer.stop()
    yield app
    app.close(); plt.close(app.fig)


def test_recorded_time_speed_pause_next_and_end(monkeypatch):
    run = calculate(get_scenario('sun-earth-jupiter'))
    monkeypatch.setattr(System, 'step', unexpected_step)
    playback = Playback(run)
    dt = run.frames[0].dt
    playback.advance(100, 100)
    assert playback.current_frame.time == 0
    playback.play(); playback.advance(dt*3, 1)
    assert playback.index == 0  # no interpolation
    playback.advance(dt, 1)
    assert playback.current_frame.time == run.frames[1].time
    playback.pause(); playback.advance(100, 100)
    assert playback.index == 1
    assert playback.next_sample() is run.frames[2] and not playback.playing
    playback.play(); playback.advance(100, 10)
    assert playback.current_frame is run.frames[-1] and not playback.playing
    playback.next_sample(); playback.play()
    assert not playback.playing


def test_variable_membership_uses_frame_identity_and_explicit_removal(app, monkeypatch):
    first = StateFrame.capture(build_system(app.initial_setup))
    # Remove the view origin and selected planet, reorder rows, change properties.
    j = first.body_ids.index('jupiter')
    second = StateFrame(first.dt*7, first.dt/2, ('jupiter',),
        (replace(first.body_specs[j], name='Remnant', radius=.02),), first.masses[[j]],
        first.r[[j]]+1, first.v[[j]], reference_ids=(None,))
    run = RecordedRun('synthetic', 'Variable membership', second.time, second.time,
                      (first, second), 'halted', 'Synthetic removal event')
    app.session.retain_recording(run)
    monkeypatch.setattr(System, 'step', unexpected_step)
    app.show_recording(run)
    app.selected_id = 'earth'; app.toggle_follow()
    camera = app.ax.get_xlim(), app.ax.get_ylim()
    app.step_once()
    assert app.selected_id == 'earth' and 'missing / removed' in app.inspector.get_text()
    assert app.prepared.body_specs[0].name == 'Remnant'
    assert not app.scene.artists_by_id['earth'].get_visible()
    assert app.scene.origin_id == 'jupiter'
    assert (app.ax.get_xlim(), app.ax.get_ylim()) == camera
    assert app.system.time == second.time and app.mode.get_text() == 'RECORDED'
    assert 'HALTED' in app.status.get_text() and 'Synthetic removal' in app.status.get_text()
    assert app.step.label.get_text() == 'Next sample'
    assert f'{second.time:.5g}' in app.sidebar.readout.get_text()
    app.selected_id = 'jupiter'; app._refresh()
    assert 'Remnant' in app.inspector.get_text()
    assert f'World r = ({second.r[0,0]:.5g}, {second.r[0,1]:.5g}) AU' in app.inspector.get_text()
    assert f'World v = ({second.v[0,0]:.5g}, {second.v[0,1]:.5g}) AU/yr' in app.inspector.get_text()


def test_replay_camera_stable_history_uses_timestamps_and_is_bounded(app, monkeypatch):
    base = StateFrame.capture(app.prepared)
    frames = tuple(replace(base, time=i*i*.0001, dt=.9) for i in range(1602))
    run = RecordedRun('irregular', 'Unequal intervals', frames[-1].time, frames[-1].time, frames, 'complete')
    monkeypatch.setattr(System, 'step', unexpected_step)
    app.show_recording(run)
    app.ax.set_xlim(-4, 8); app.ax.set_ylim(-3, 5)
    app.playback.play(); app.playback.advance(1000, 1); app._display_recorded_frame()
    assert len(app.history) == 1500
    assert app.history[0].time == frames[-1500].time
    assert app.ax.get_xlim() == (-4, 8) and app.ax.get_ylim() == (-3, 5)
    text = app.sidebar.readout.get_text()
    assert f'{frames[-1].time-frames[-1500].time:.5g}' in text
    assert all(len(trail.get_segments()) <= 399 for trail in app.scene.trails)


@pytest.mark.parametrize('kind', ['population', 'moon'])
def test_save_reload_calculate_inspect_population_and_moon(app, tmp_path, monkeypatch, kind):
    if kind == 'population':
        setup = replace(get_scenario('sun-earth-jupiter'), population=PopulationSpec('rocks', 'sun', Population(count=8, seed=87)))
        selected = 'rocks:3'
    else:
        separation = 384400/AU_KM
        velocity = circular_relative_velocity((separation, 0), CATALOG['earth'].mass, CATALOG['moon'].mass)
        setup = InitialSetup('Moon recording', (InitialBody('earth', CATALOG['earth'], (3,2), (.2,-.1)),
            InitialBody('moon', CATALOG['moon'], (separation,0), tuple(velocity), 'earth')), .0001)
        selected = 'moon'
    path = tmp_path/'setup.json'; save_setup(setup, path)
    restored = load_setup(path)
    initial = build_system(setup)
    run = calculate(restored, steps=13, stride=5)
    assert run.status == 'complete'
    np.testing.assert_array_equal(run.frames[0].r, initial.system.r)
    np.testing.assert_array_equal(run.frames[0].v, initial.system.v)
    np.testing.assert_allclose([f.time for f in run.frames], np.array([0,5,10,13])*setup.dt)
    initial.system.step(13)
    np.testing.assert_array_equal(run.frames[-1].r, initial.system.r)
    monkeypatch.setattr(System, 'step', unexpected_step)
    app.show_recording(run); app.selected_id = selected; app.toggle_follow()
    app.step_once()
    assert app.selected_id == selected and app.following
    if kind == 'moon':
        relative = relative_state(app.playback.current_frame, selected)
        assert relative.reference_id == 'earth' and relative.separation > 0
        assert f'{relative.separation*AU_KM:,.7g} km' in app.inspector.get_text()
    else:
        assert selected in app.inspector.get_text()
    assert app.system.time == run.frames[1].time

@pytest.mark.parametrize('use_blit', [True, False])
def test_recorded_render_cache_matches_full_draw_and_preserves_overlay(app, monkeypatch, use_blit):
    run = calculate(app.initial_setup)
    monkeypatch.setattr(System, 'step', unexpected_step)
    app.painter.use_blit = use_blit
    app.show_recording(run)
    app.step_once()
    app.fig.canvas.draw()
    app.step_once()
    incremental = np.asarray(app.fig.canvas.buffer_rgba()).copy()
    app.fig.canvas.draw()
    np.testing.assert_array_equal(incremental, np.asarray(app.fig.canvas.buffer_rgba()))
    assert app.sidebar.calculate.ax in app.painter.overlay_axes
    assert app.sidebar.results.ax in app.painter.overlay_axes


def test_ordinary_replay_frame_reuses_cached_background(app, monkeypatch):
    run = calculate(app.initial_setup, steps=20, stride=2)
    app.show_recording(run)
    app.playback.play(); app._display_recorded_frame()
    app.fig.canvas.draw()
    original = app.fig.canvas.draw
    draws = []
    def draw(*args, **kwargs):
        draws.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(app.fig.canvas, 'draw', draw)
    monkeypatch.setattr(System, 'step', unexpected_step)
    app.playback.advance(run.frames[0].dt*5,1)
    app._display_recorded_frame()
    assert app.playback.index > 0 and not draws


def test_natural_radial_infall_halts_with_partial_inspectable_result(app, monkeypatch):
    setup = InitialSetup('Unresolved radial infall', (
        InitialBody('sun', CATALOG['sun'], (0,0), (0,0)),
        InitialBody('earth', CATALOG['earth'], (.1,0), (0,0))), 1e-5)
    initial = build_system(setup)
    initial.system.validate_step()  # initially valid; no injected state corruption
    calc = Calculation(setup, CalculationRequest(.01, 37))
    while not calc.done:
        calc.advance(50)
    run = calc.result
    assert run.status == 'halted' and 'unresolved' in run.reason
    assert 0 < run.completed_steps < run.requested_steps
    assert 0 < run.actual_end_time < run.requested_end_time
    assert run.frames[-1].time == calc.prepared.system.time
    np.testing.assert_array_equal(run.frames[-1].r, calc.prepared.system.r)
    np.testing.assert_array_equal(run.frames[-1].v, calc.prepared.system.v)
    assert np.isfinite(run.frames[-1].r).all() and np.isfinite(run.frames[-1].v).all()
    monkeypatch.setattr(System, 'step', unexpected_step)
    app.show_recording(run)
    app.playback.play(); app.playback.advance(10,1); app._display_recorded_frame()
    assert 'HALTED' in app.scene.result_info.get_text() and 'unresolved' in app.status.get_text()
    assert app.system.time == run.actual_end_time
