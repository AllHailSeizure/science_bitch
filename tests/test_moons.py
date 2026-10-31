"""Planet-relative initial conditions share the unrestricted Newtonian solver."""
from dataclasses import replace

import numpy as np
import pytest

from simulations.star_system.core import G, System, kepler_period
from simulations.star_system.data import AU_KM, CATALOG
from simulations.star_system.history import StateFrame, relative_state
from simulations.star_system.initialization import (
    build_system, circular_relative_velocity, insert_body, recommended_moon_step,
)
from simulations.star_system.model import InitialBody, InitialSetup

# NASA Moon Fact Sheet mean semimajor axis; an illustrative circular fixture,
# not the eccentric, inclined Moon's ephemeris. Physical values come from CATALOG.
SEPARATION = 384400 / AU_KM
EARTH, MOON = CATALOG['earth'], CATALOG['moon']
PERIOD = kepler_period(SEPARATION, G * (EARTH.mass + MOON.mass))


def moon_body(speed=1.):
    velocity = circular_relative_velocity((SEPARATION, 0), EARTH.mass, MOON.mass)
    return InitialBody('moon', MOON, (SEPARATION, 0), tuple(speed * velocity), 'earth')


def pair(dt=PERIOD / 600):
    return InitialSetup('Illustrative circular Earth-Moon', (
        InitialBody('earth', EARTH, (0, 0), (0, 0)), moon_body()), dt)


def test_moon_initial_state_adds_parent_position_and_velocity():
    setup = replace(pair(), bodies=(
        InitialBody('earth', EARTH, (3, 2), (.2, -.1)), moon_body()))
    prepared = build_system(setup)
    np.testing.assert_array_equal(prepared.system.r[0], (3, 2))
    np.testing.assert_array_equal(prepared.system.r[1], prepared.system.r[0] + moon_body().r)
    np.testing.assert_array_equal(prepared.system.v[1], prepared.system.v[0] + moon_body().v)
    assert prepared.reference_ids == (None, 'earth')
    assert recommended_moon_step(moon_body().r, EARTH.mass, MOON.mass) == pytest.approx(PERIOD / 600)


def test_live_moon_insertion_preserves_existing_state_and_shortens_fixed_segment():
    before = build_system(replace(pair(.001), bodies=pair().bodies[:1]))
    before.system.step(7)
    saved = before.system.r.copy(), before.system.v.copy(), before.system.time
    after = insert_body(before, moon_body())
    np.testing.assert_array_equal(after.system.r[:-1], saved[0])
    np.testing.assert_array_equal(after.system.v[:-1], saved[1])
    np.testing.assert_array_equal(after.system.r[-1], saved[0][0] + moon_body().r)
    assert after.system.time == saved[2] and after.setup is before.setup
    assert after.system.dt == pytest.approx(PERIOD / 600)
    assert after.segment == before.segment + 1
    assert after.events[-1].old_dt == .001
    assert after.events[-1].new_dt == after.system.dt
    frame = StateFrame.capture(after)
    assert frame.reference_ids == (None, 'earth')
    assert relative_state(frame, 'moon').separation == pytest.approx(SEPARATION)


def test_moon_parent_does_not_modify_force_law_or_reattach_fast_moon():
    p = build_system(replace(pair(), bodies=(pair().bodies[0], moon_body(5.))))
    numeric = System(p.system.masses.copy(), p.system.r.copy(), p.system.v.copy(), p.system.dt)
    p.system.step(300); numeric.step(300)
    np.testing.assert_array_equal(p.system.r, numeric.r)
    np.testing.assert_array_equal(p.system.v, numeric.v)
    assert np.linalg.norm(p.system.r[1] - p.system.r[0]) > 5 * SEPARATION
    assert p.reference_ids[1] == 'earth'


def measure_pair(divisor):
    p = build_system(pair(PERIOD / divisor))
    angles, times, energies, angular = [], [], [], []
    for step in range(3 * divisor + 2):
        r = p.system.r[1] - p.system.r[0]
        angles.append(np.arctan2(r[1], r[0])); times.append(p.system.time)
        energy, _, momentum = p.system.invariants()
        energies.append(energy); angular.append(momentum)
        if step == 3 * divisor:
            final = r.copy()
        p.system.step()
    theta = np.unwrap(angles)
    crossings = np.interp(2 * np.pi * np.arange(1, 4), theta, times)
    periods = np.diff(np.r_[0., crossings])
    period_error = np.max(np.abs(periods / PERIOD - 1))
    energy_error = np.max(np.abs(np.asarray(energies) / energies[0] - 1))
    angular_error = np.max(np.abs(np.asarray(angular) / angular[0] - 1))
    position_error = np.linalg.norm(final - (SEPARATION, 0))
    return period_error, energy_error, angular_error, position_error


def test_isolated_earth_moon_orbit_converges():
    coarse, fine = measure_pair(600), measure_pair(1200)
    print(f'Earth-Moon period={PERIOD:.12g} yr / {PERIOD*365.25:.12g} days; dt={PERIOD/600:.12g} yr')
    print(f'Earth-Moon period/600: period={coarse[0]:.12g}, energy={coarse[1]:.12g}, angular={coarse[2]:.12g}, position={coarse[3]:.12g} AU')
    print(f'Earth-Moon period/1200: period={fine[0]:.12g}, energy={fine[1]:.12g}, angular={fine[2]:.12g}, position={fine[3]:.12g} AU')
    assert coarse[0] < 1e-6 and coarse[1] < 1e-7 and coarse[2] < 1e-10
    assert fine[3] < coarse[3] / 5


@pytest.mark.parametrize('second_star', [False, True])
def test_sun_earth_moon_and_second_star_are_finite_mutually_forced(second_star):
    bodies = (InitialBody('sun', CATALOG['sun'], (0, 0), (0, 0)),
              InitialBody('earth', EARTH, (1, 0), (0, np.sqrt(G)), 'sun'), moon_body())
    if second_star:
        bodies += (InitialBody('companion', CATALOG['sun'], (8, 5), (0, -1)),)
    p = build_system(InitialSetup('Forced moons', bodies, PERIOD / 600))
    pair_acceleration = build_system(pair()).system.accelerations()
    assert not np.allclose(p.system.accelerations()[1:3], pair_acceleration)
    if second_star:
        without = System(p.system.masses[:-1], p.system.r[:-1], p.system.v[:-1], p.system.dt)
        assert np.linalg.norm(p.system.accelerations()[0] - without.accelerations()[0]) > 0
    p.system.step(600)
    assert np.isfinite(p.system.r).all() and np.isfinite(p.system.v).all()
    assert p.system.dt <= PERIOD / 600


@pytest.mark.parametrize('body', [
    replace(moon_body(), reference_id=None),
    replace(moon_body(), reference_id='missing'),
    replace(moon_body(), reference_id='sun'),
    replace(moon_body(), r=(EARTH.radius, 0)),
])
def test_invalid_moon_parent_or_overlap_is_recoverable_without_mutation(body):
    p = build_system(replace(pair(.001), bodies=pair().bodies[:1]))
    saved = p.system.r.copy(), p.system.v.copy(), p.system.time
    with pytest.raises(ValueError): insert_body(p, body)
    np.testing.assert_array_equal(p.system.r, saved[0]); np.testing.assert_array_equal(p.system.v, saved[1])
    assert p.system.time == saved[2]


def test_moon_unresolved_step_cycle_nonfinite_and_missing_parent_are_rejected():
    with pytest.raises(ValueError, match='moon|Moon'):
        build_system(pair(.001))
    with pytest.raises(ValueError, match='cycle'):
        build_system(replace(pair(), bodies=(replace(pair().bodies[0], reference_id='moon'), moon_body())))
    with pytest.raises(ValueError, match='finite'):
        replace(moon_body(), v=(float('inf'), 0))
    with pytest.raises(ValueError, match='Missing reference'):
        build_system(replace(pair(), bodies=(replace(moon_body(), reference_id='removed'),)))


def test_separation_inspection_uses_frame_world_state_and_reports_missing_parent():
    p = build_system(pair()); p.system.step(30)
    p.system.r[1] += (SEPARATION, 0)
    frame = StateFrame.capture(p)
    result = relative_state(frame, 'moon')
    assert result.reference_id == 'earth' and result.reference_name == 'Earth'
    assert result.separation == pytest.approx(np.linalg.norm(frame.r[1] - frame.r[0]))
    assert result.separation != pytest.approx(SEPARATION)
    missing = replace(frame, body_ids=('moon',), body_specs=(MOON,), masses=frame.masses[1:], r=frame.r[1:], v=frame.v[1:], reference_ids=('earth',))
    assert relative_state(missing, 'moon').missing
    assert relative_state(missing, 'moon').reference_id == 'earth'


def test_live_moon_uses_planet_identity_appended_after_population_particles():
    from simulations.star_system.initialization import legacy_setup
    from simulations.star_system.populations import Population
    p = build_system(legacy_setup(population=Population(count=4)))
    p = insert_body(p, InitialBody('other-earth', EARTH, (4, 3), (.1, .2)))
    saved_r, saved_v, saved_time = p.system.r.copy(), p.system.v.copy(), p.system.time
    after = insert_body(p, replace(moon_body(), reference_id='other-earth'))
    assert after.body_ids[-2:] == ('other-earth', 'moon')
    np.testing.assert_array_equal(after.system.r[:-1], saved_r)
    np.testing.assert_array_equal(after.system.v[:-1], saved_v)
    np.testing.assert_array_equal(after.system.r[-1], saved_r[-1] + moon_body().r)
    assert after.system.time == saved_time
    frame = StateFrame.capture(after)
    assert relative_state(frame, 'moon').reference_id == 'other-earth'
    assert frame.reference_ids[3:7] == (None,)*4
