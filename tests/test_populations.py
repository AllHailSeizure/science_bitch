"""Reproducibility, distribution, restricted gravity, and playback contracts."""
from dataclasses import replace

import numpy as np
import pytest

from simulations.star_system.core import G, System, make_system
from simulations.star_system.populations import MAX_PARTICLES, Population, with_population
from simulations.star_system.model import BodySpec, InitialBody, InitialSetup, PopulationSpec
from simulations.star_system.initialization import build_system


def binary_population_setup():
    first = BodySpec('First star', 'star', 1., .00465)
    second = BodySpec('Second star', 'star', .6, .003)
    settings = Population(kind='ring', inner=1., outer=1., count=12, seed=23, angles='even')
    return InitialSetup('Binary ring', (
        InitialBody('first', first, (-20., 3.), (.2, -.1)),
        InitialBody('second', second, (40., -7.), (-.3, .4))),
        dt=.0001, population=PopulationSpec('rocks', 'second', settings))


def test_population_reference_survives_body_reordering():
    setup = binary_population_setup()
    original = build_system(setup)
    reordered = build_system(replace(setup, bodies=setup.bodies[::-1]))
    for key in original.body_ids:
        i, j = original.body_ids.index(key), reordered.body_ids.index(key)
        np.testing.assert_array_equal(original.system.r[i], reordered.system.r[j])
        np.testing.assert_array_equal(original.system.v[i], reordered.system.v[j])
        assert original.body_specs[i] == reordered.body_specs[j]
    assert original.body_ids[2:] == tuple(f'rocks:{i}' for i in range(12))


def test_population_around_second_star_uses_its_state():
    setup = binary_population_setup()
    prepared = build_system(setup)
    r = prepared.system.r[2:]-setup.bodies[1].r
    v = prepared.system.v[2:]-setup.bodies[1].v
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), 1., atol=1e-14)
    np.testing.assert_allclose(np.sum(v*v, axis=1), G*.6, rtol=1e-14)
    np.testing.assert_allclose(np.sum(r*v, axis=1), 0., atol=5e-14)
    assert np.all(r[:, 0]*v[:, 1]-r[:, 1]*v[:, 0] > 0)
    for i, body in enumerate(setup.bodies):
        np.testing.assert_array_equal(prepared.system.r[i], body.r)
        np.testing.assert_array_equal(prepared.system.v[i], body.v)
    assert prepared.system.dt == setup.dt


@pytest.mark.parametrize('problem', ['missing', 'zero mass', 'unknown radius', 'surface'])
def test_missing_population_reference_is_recoverable(problem):
    setup = binary_population_setup()
    valid = build_system(setup)
    r, v = valid.system.r.copy(), valid.system.v.copy()
    if problem == 'missing':
        invalid = replace(setup, bodies=setup.bodies[:1])
    else:
        spec = replace(setup.bodies[1].spec, **{
            'zero mass': dict(kind='particle', mass=0.),
            'unknown radius': dict(radius=None), 'surface': dict(radius=2.)}[problem])
        invalid = replace(setup, bodies=(setup.bodies[0], replace(setup.bodies[1], spec=spec)))
    with pytest.raises(ValueError, match='Population.*(reference|surface)'):
        build_system(invalid)
    np.testing.assert_array_equal(valid.system.r, r)
    np.testing.assert_array_equal(valid.system.v, v)
    assert valid.setup is setup


def test_authored_population_does_not_persist_numeric_reference():
    setup = binary_population_setup()
    pop = replace(setup.population, settings=replace(setup.population.settings, reference=19))
    assert pop.settings.reference == 0  # compatibility field, never a saved row identity
    built = build_system(replace(setup, population=pop))
    np.testing.assert_array_equal(built.system.r, build_system(setup).system.r)


@pytest.mark.parametrize('kind', ['ring', 'disk', 'belt'])
def test_seeded_initial_state_and_evolution(kind):
    settings = Population(kind=kind, outer=2. if kind == 'ring' else 3.5)
    base = make_system()
    first = with_population(base, settings)
    again = with_population(base, settings)
    np.testing.assert_array_equal(first.r, again.r)
    np.testing.assert_array_equal(first.v, again.v)
    np.testing.assert_array_equal(first.r[:3], base.r)
    np.testing.assert_array_equal(first.v[:3], base.v)
    assert not np.shares_memory(first.r, base.r)
    changed = with_population(base, replace(settings, seed=1))
    assert not np.array_equal(first.r[3:], changed.r[3:])
    first.step(20)
    again.step(20)
    np.testing.assert_array_equal(first.r, again.r)
    assert first.time == again.time
    restarted = with_population(base, settings)
    assert restarted.time == 0
    np.testing.assert_array_equal(restarted.r, with_population(make_system(), settings).r)


@pytest.mark.parametrize('kind', ['ring', 'disk', 'belt'])
def test_distribution_and_circular_velocities(kind):
    settings = Population(kind=kind, count=MAX_PARTICLES, inner=2.,
                          outer=2. if kind == 'ring' else 4., angles='even')
    base = make_system()
    system = with_population(base, settings)
    r = system.r[3:]-system.r[0]
    v = system.v[3:]-system.v[0]
    radius = np.linalg.norm(r, axis=1)
    assert np.all((radius >= settings.inner-1e-14) & (radius <= settings.outer+1e-14))
    np.testing.assert_allclose(np.sum(r*v, axis=1), 0, atol=1e-14)
    np.testing.assert_allclose(np.sum(v*v, axis=1), G/radius, rtol=1e-14)
    theta = np.sort(np.mod(np.arctan2(r[:, 1], r[:, 0]), 2*np.pi))
    np.testing.assert_allclose(np.diff(np.r_[theta, theta[0]+2*np.pi]),
                               2*np.pi/settings.count, atol=2e-14)
    if kind == 'disk':
        assert np.mean(radius**2) == pytest.approx(10., abs=.2)
    elif kind == 'belt':
        assert np.mean(radius) == pytest.approx(3., abs=.04)


@pytest.mark.parametrize('kwargs', [dict(count=0), dict(count=2001), dict(count=True),
    dict(seed=-1), dict(seed=1.5), dict(inner=float('nan')), dict(outer=float('inf')),
    dict(inner=0), dict(inner=4), dict(kind='ring'), dict(kind='unknown'),
    dict(angles='unknown'), dict(reference=-1)])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Population(**kwargs)


def test_reference_frame_resolution_and_validation():
    base = make_system()
    base.time = 3.
    settings = Population(kind='ring', inner=.01, outer=.01, reference=1, count=10)
    result = with_population(base, settings)
    assert result.time == 3.
    assert result.dt < base.dt
    np.testing.assert_allclose(np.linalg.norm(result.r[3:]-base.r[1], axis=1), .01)
    np.testing.assert_allclose(np.sum((result.v[3:]-base.v[1])**2, axis=1),
                               G*base.masses[1]/.01)
    for reference in (3, 50):
        with pytest.raises(ValueError, match='massive body'):
            with_population(base, replace(settings, reference=reference))
    with pytest.raises(ValueError, match='surface'):
        with_population(base, settings, reference_radius=.02)
    with pytest.raises(ValueError, match='At most'):
        with_population(with_population(base, Population(count=2000)), Population(count=1))


def test_particles_do_not_backreact_or_interact_and_can_overlap():
    base = make_system()
    populated = with_population(base, Population(count=20))
    populated.r[-1] = populated.r[-2]
    populated.v[-1] = populated.v[-2]
    base.step(100)
    populated.step(100)
    np.testing.assert_allclose(populated.r[:3], base.r, rtol=0, atol=1e-14)
    np.testing.assert_allclose(populated.v[:3], base.v, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(populated.r[-1], populated.r[-2])
    assert np.isfinite(populated.invariants()[0])


def test_isolated_ring_period_and_close_encounter():
    base = System(np.array([1.]), np.zeros((1, 2)), np.zeros((1, 2)), 1.)
    system = with_population(base, Population(kind='ring', inner=1., outer=1., count=12))
    initial = system.r.copy()
    system.step(600)
    np.testing.assert_allclose(system.r, initial, atol=1e-7)
    system.r[-1] = system.r[0]
    previous = system.r.copy()
    with pytest.raises(ValueError, match='Close encounter'):
        system.step()
    np.testing.assert_array_equal(system.r, previous)


def test_jupiter_perturbs_identical_population():
    # Match the star-relative initial particles and integration times exactly.
    base = make_system(planets=['jupiter'])
    settings = Population(kind='ring', inner=3.2, outer=3.2, count=32, angles='even', seed=7)
    perturbed = with_population(base, settings)
    control = System(perturbed.masses.copy(), perturbed.r.copy(), perturbed.v.copy(), perturbed.dt)
    control.masses[1] = 0.
    np.testing.assert_array_equal(perturbed.r, control.r)
    np.testing.assert_array_equal(perturbed.v, control.v)
    for system in (perturbed, control):
        system.step(600)
    assert perturbed.time == control.time
    displacement = np.linalg.norm((perturbed.r[2:]-perturbed.r[0]) -
                                  (control.r[2:]-control.r[0]), axis=1)
    assert np.max(displacement) > .01  # AU, after one initial ring period


def test_population_playback_restart_and_render(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from simulations.star_system.plotting import App
    app = App(population=Population(count=16))
    try:
        initial = app.system.r.copy()
        app.toggle_pause()
        app.update(None)
        np.testing.assert_array_equal(app.system.r, initial)
        app.step_once()
        assert app.paused
        assert app.system.time == app.system.dt
        app.toggle_pause()
        monkeypatch.setattr('simulations.star_system.plotting.time.monotonic', lambda: app.last+.05)
        app.update(None)
        assert app.system.time > app.system.dt
        assert len(app.scene.particles.get_offsets()) == 16
        app.selected_index = 3
        app.toggle_follow()
        assert 'Test particle' in app.inspector.get_text()
        app.restart()
        np.testing.assert_array_equal(app.system.r, initial)
        assert app.system.time == 0
        app.fig.savefig(tmp_path/'population.png')
        assert (tmp_path/'population.png').stat().st_size > 10000
    finally:
        app.timer.stop()
        plt.close(app.fig)


def test_incompatible_star_leaves_population_recoverable():
    import matplotlib.pyplot as plt
    from simulations.star_system.plotting import App
    app = App(population=Population(kind='ring', inner=.1, outer=.1, count=8))
    try:
        previous = app.system
        app.star_picker.set_active(3)  # Rigel's surface encloses the ring.
        assert app.star == 'sun'
        assert app.star_picker.value_selected == 'sun'
        assert app.system is previous
        assert 'inside' in app.status.get_text()
        app.step_once()
        assert not app.halted
    finally:
        app.timer.stop()
        plt.close(app.fig)


def test_population_cli(tmp_path):
    from simulations.star_system.__main__ import main
    output = tmp_path/'ring.png'
    assert main(['--population', 'ring', '--count', '24', '--inner', '2.5',
                 '--angles', 'even', '--years', '.01', '--output', str(output)]) == 0
    assert output.stat().st_size > 10000
    with pytest.raises(SystemExit) as exc:
        main(['--population', 'belt', '--count', '0', '--output', str(output)])
    assert exc.value.code == 2
