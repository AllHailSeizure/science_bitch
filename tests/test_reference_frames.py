"""Display transforms preserve physical state and use each sample's geometry."""
from dataclasses import replace

import numpy as np
import pytest

from simulations.star_system.history import StateFrame
from simulations.star_system.scenarios import get_scenario
from simulations.star_system.frames import frame_transform, rotate_about_sun, lagrange_guides
from simulations.star_system.initialization import build_system
from simulations.star_system.model import BodySpec, InitialBody, InitialSetup
from simulations.star_system.rendering import coordinates, inverse_coordinates, draw_system


def sample():
    from simulations.star_system.populations import Population
    from simulations.star_system.initialization import legacy_setup
    setup = legacy_setup(planets=('jupiter',), population=Population(
        kind='ring', inner=3.2, outer=3.2, count=32, angles='even', seed=7))
    return StateFrame.capture(build_system(setup))


def test_rotation_preserves_distances_and_places_jupiter_on_positive_axis():
    frame = sample()
    moved = replace(frame, r=frame.r @ np.array([[0., 1.], [-1., 0.]]) + (3, -9))
    before = moved.r.copy()
    rotated = rotate_about_sun(moved, 'sun', 'jupiter')
    assert rotated[1, 0] > 0
    assert rotated[1, 1] == pytest.approx(0., abs=1e-14)
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(moved.r-moved.r[0], axis=1))
    np.testing.assert_array_equal(moved.r, before)


@pytest.mark.parametrize('mode', ['world', 'sun-relative', 'rotating'])
def test_inverse_mapping_undoes_compression_then_frame(mode):
    frame = sample()
    transform = frame_transform(frame, mode, 'sun')
    world = np.array([[2., -4.], [-3., 1.]])
    display = coordinates(transform.positions(world))
    np.testing.assert_allclose(transform.inverse_positions(inverse_coordinates(display)), world, atol=1e-13)
    velocity = np.array([.8, -2.3])
    np.testing.assert_allclose(transform.inverse_vectors(transform.vectors(velocity)), velocity, atol=1e-15)


def test_historical_samples_use_their_own_angle():
    first = sample()
    quarter = np.array([[0., -1.], [1., 0.]])
    second = replace(first, time=2., r=first.r @ quarter.T)
    np.testing.assert_allclose(rotate_about_sun(first, 'sun', 'jupiter'),
                               rotate_about_sun(second, 'sun', 'jupiter'), atol=1e-14)
    wrong = frame_transform(second, 'rotating').positions(first.r)
    assert not np.allclose(wrong, rotate_about_sun(first, 'sun', 'jupiter'))


def test_guides_are_equilateral_geometry_and_missing_references_are_explicit():
    frame = sample()
    guides = lagrange_guides(frame, 'sun', 'jupiter')
    separation = np.linalg.norm(frame.r[1]-frame.r[0])
    np.testing.assert_allclose(np.linalg.norm(guides-frame.r[0], axis=1), separation)
    np.testing.assert_allclose(np.linalg.norm(guides-frame.r[1], axis=1), separation)
    delta, l4 = frame.r[1]-frame.r[0], guides[0]-frame.r[0]
    assert delta[0]*l4[1]-delta[1]*l4[0] > 0
    with pytest.raises(ValueError, match='Jupiter'):
        frame_transform(replace(frame, body_ids=tuple('absent' if key == 'jupiter' else key for key in frame.body_ids),
                                body_specs=tuple(replace(s, catalog_key=None) if i == 1 else s for i, s in enumerate(frame.body_specs))), 'rotating')
    same = frame.r.copy()
    same[1] = same[0]
    with pytest.raises(ValueError, match='coincident'):
        frame_transform(replace(frame, r=same), 'rotating')


def test_rotating_authoring_inverse_and_physical_vector_survive_view_switch():
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    from simulations.star_system.plotting import App
    from simulations.star_system.authoring import VELOCITY_PIXELS_PER_AUYR
    app = App(setup=get_scenario('sun-jupiter-ring'))
    app.timer.stop()
    before_r, before_v = app.system.r.copy(), app.system.v.copy()
    try:
        app.set_frame_mode('rotating')
        app.edit_body()
        editor = app.body_editor
        offset = np.zeros(app.system.r.shape[1])
        offset[:2] = (-1.2, 1.7)
        world = offset + app.system.r[0]
        display = app.scene.world_to_display(world)
        pixel = app.ax.transData.transform(display)
        editor.on_press(SimpleNamespace(button=1, inaxes=app.ax, xdata=display[0], ydata=display[1], x=pixel[0], y=pixel[1]))
        r, _ = editor.vectors()
        np.testing.assert_allclose(r+app.system.r[0], world, atol=1e-12)
        physical_v = np.zeros(app.system.r.shape[1])
        physical_v[:2] = (.25, -.8)
        tip = app.ax.transData.transform(editor.anchor)+app.scene.transform.vectors(physical_v)*VELOCITY_PIXELS_PER_AUYR
        editor.dragging = True
        editor.on_motion(SimpleNamespace(x=tip[0], y=tip[1]))
        _, velocity = editor.vectors()
        np.testing.assert_allclose(velocity, physical_v, atol=1e-14)
        fields = tuple(editor.fields[k].text for k in ('x', 'y', 'vx', 'vy'))
        app.scene.set_frame_mode('world')
        editor.preview()
        assert tuple(editor.fields[k].text for k in ('x', 'y', 'vx', 'vy')) == fields
        app.set_frame_mode('rotating')  # modal callback guard
        assert app.scene.frame_mode == 'world'
        np.testing.assert_array_equal(app.system.r, before_r)
        np.testing.assert_array_equal(app.system.v, before_v)
        assert app.system.time == 0
        editor.finish()
    finally:
        app.close()
        plt.close(app.fig)


PLUS_60 = np.array((.5, np.sqrt(3)/2))
MINUS_60 = np.array((.5, -np.sqrt(3)/2))


def sun_jupiter_pair(velocity):
    """Sun at the origin, Jupiter at (1, 0), with Jupiter's absolute velocity."""
    sun = BodySpec('Sun', 'star', 1., 1e-4, temperature=5772.)
    jupiter = BodySpec('Jupiter', 'planet', 1e-3, 1e-4, color='#c9b27c', catalog_key='jupiter')
    return InitialSetup('Sun Jupiter pair', (
        InitialBody('sun', sun, (0., 0.), (0., 0.)),
        InitialBody('jupiter', jupiter, (1., 0.), (float(velocity[0]), float(velocity[1])), 'sun'),
    ), .01)


def _guide_labels(scene):
    displayed = np.vstack([np.asarray(label.xy, dtype=float) for label in scene.guide_labels])
    return scene.display_to_world(displayed), [label.get_text() for label in scene.guide_labels]


def test_prograde_l4_guide_is_the_leading_equilateral_vertex():
    prepared = build_system(sun_jupiter_pair((0., 1.)))
    frame = StateFrame.capture(prepared)
    np.testing.assert_allclose(lagrange_guides(frame, 'sun', 'jupiter')[0], PLUS_60)
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots()
    try:
        points, texts = _guide_labels(draw_system(ax, prepared))
        np.testing.assert_allclose(points[0], PLUS_60)
        np.testing.assert_allclose(points[1], MINUS_60)
        assert texts == ['L4 guide', 'L5 guide']
    finally:
        plt.close(figure)


def test_retrograde_l4_guide_leads_clockwise_motion():
    prepared = build_system(sun_jupiter_pair((0., -1.)))
    frame = StateFrame.capture(prepared)
    guides = lagrange_guides(frame, 'sun', 'jupiter')
    np.testing.assert_allclose(guides[0], MINUS_60)
    np.testing.assert_allclose(guides[1], PLUS_60)
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots()
    try:
        points, texts = _guide_labels(draw_system(ax, prepared))
        np.testing.assert_allclose(points[0], MINUS_60)
        np.testing.assert_allclose(points[1], PLUS_60)
        assert texts == ['L4 guide', 'L5 guide']
    finally:
        plt.close(figure)


@pytest.mark.parametrize('velocity', [(0., 0.), (1., 0.)])
def test_zero_angular_momentum_guides_are_not_labeled_l4_or_l5(velocity):
    prepared = build_system(sun_jupiter_pair(velocity))
    frame = StateFrame.capture(prepared)
    guides = lagrange_guides(frame, 'sun', 'jupiter')
    np.testing.assert_allclose(guides, np.vstack((PLUS_60, MINUS_60)))
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots()
    try:
        scene = draw_system(ax, prepared)
        points, texts = _guide_labels(scene)
        np.testing.assert_allclose(points, np.vstack((PLUS_60, MINUS_60)))
        assert all('L4' not in text and 'L5' not in text and 'guide' in text.lower() for text in texts)
        note = scene.guide_note.get_text()
        assert 'no direction of revolution' in note
        assert 'L4' not in note and 'L5' not in note
    finally:
        plt.close(figure)
