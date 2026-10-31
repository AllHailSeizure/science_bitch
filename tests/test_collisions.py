from dataclasses import replace

import numpy as np
import pytest

from simulations.star_system.model import BodySpec, InitialBody, InitialSetup
from simulations.star_system.initialization import build_system
from simulations.star_system.collisions import advance_prepared, merge_pair, CollisionEvent
from simulations.star_system.core import System


def fixture(dt=.2, *, y=0., speed=10., mass=1e-14, kind='planet'):
    return InitialSetup('Contact fixture', (
        InitialBody('a', BodySpec('A', kind, mass, .1), (-1., y), (speed, 0.)),
        InitialBody('b', BodySpec('B', 'planet', mass*2, .1), (1., 0.), (0., 0.)),
    ), dt, collision_mode='perfect_merge')


def test_merge_conserves_mass_momentum_and_records_unmodeled_spin():
    p = build_system(fixture(dt=.001, y=.05, speed=2.))
    old = p.system
    merged, event = merge_pair(p, 'b', 'a')
    assert event.parent_ids == ('a', 'b')
    assert merged.body_ids == (event.remnant_id,)
    np.testing.assert_allclose(merged.system.masses.sum(), old.masses.sum(), rtol=1e-15)
    np.testing.assert_allclose((merged.system.masses[:, None]*merged.system.v).sum(0),
                               (old.masses[:, None]*old.v).sum(0), rtol=1e-15)
    assert merged.body_specs[0].radius == pytest.approx(np.cbrt(.002))
    assert merged.body_specs[0].temperature is None and not merged.body_specs[0].sources
    assert merged.body_specs[0].kind == 'planet'
    assert event.lost_relative_kinetic == pytest.approx(.5*(2e-28/3e-14)*4.)
    assert event.unmodeled_spin == pytest.approx(old.invariants()[2]-merged.system.invariants()[2])
    np.testing.assert_array_equal(p.system.r, old.r)


def test_high_speed_crossing_resolves_contact_and_macro_remainder():
    p = build_system(fixture())
    result = advance_prepared(p)
    assert len(result.prepared.body_ids) == 1
    assert result.prepared.system.time == .2
    event = result.prepared.events[-1]
    assert event.time == pytest.approx(.18, abs=1e-10)
    assert [f.time for f in result.event_frames] == [event.time, event.time]
    assert len(result.event_frames[0].body_ids) == 2
    assert len(result.event_frames[1].body_ids) == 1
    assert p.system.time == 0 and p.body_ids == ('a', 'b')


def test_near_miss_and_grazing():
    miss = advance_prepared(build_system(fixture(y=.20001))).prepared
    assert len(miss.body_ids) == 2 and not miss.events
    graze = advance_prepared(build_system(fixture(y=.2))).prepared
    assert len(graze.body_ids) == 1


def test_star_contact_halts_whole_macro_transactionally():
    p = build_system(fixture(kind='star'))
    old = (p.system.r.copy(), p.system.v.copy())
    with pytest.raises(ValueError, match='star.*#14'):
        advance_prepared(p)
    assert p.system.time == 0 and p.body_ids == ('a', 'b') and not p.events
    np.testing.assert_array_equal(p.system.r, old[0])
    np.testing.assert_array_equal(p.system.v, old[1])


@pytest.mark.parametrize('other_kind', ['star', 'particle'])
def test_star_star_and_star_point_contact_are_both_unsupported(other_kind):
    setup = fixture(kind='star')
    other = replace(setup.bodies[1], spec=BodySpec('Other', other_kind,
        0. if other_kind == 'particle' else 1e-14, None if other_kind == 'particle' else .1))
    p = build_system(replace(setup, bodies=(setup.bodies[0], other)))
    with pytest.raises(ValueError, match='star.*#14'):
        advance_prepared(p)
    assert p.system.time == 0 and len(p.body_ids) == 2 and not p.events


def test_missing_radius_and_initial_overlap_block_recoverably():
    setup = fixture()
    with pytest.raises(ValueError, match='radius'):
        build_system(replace(setup, bodies=(replace(setup.bodies[0], spec=replace(setup.bodies[0].spec, radius=None)), setup.bodies[1])))
    with pytest.raises(ValueError, match='overlap'):
        build_system(replace(setup, bodies=(replace(setup.bodies[0], r=(.9, 0)), setup.bodies[1])))


def test_simultaneous_connected_group_is_row_order_invariant_and_preserves_other_rows():
    spec = BodySpec('Rock', 'asteroid', 1e-14, .1)
    bodies = (InitialBody('c', spec, (1., 0), (-10., 0)),
              InitialBody('b', spec, (0., 0), (0., 0)),
              InitialBody('a', spec, (-1., 0), (10., 0)),
              InitialBody('unaffected', spec, (0., 100.), (0., 0)))
    setup = InitialSetup('Triple', bodies, .1, collision_mode='perfect_merge')
    first = advance_prepared(build_system(setup)).prepared
    second = advance_prepared(build_system(replace(setup, bodies=bodies[::-1]))).prepared
    assert first.body_ids == second.body_ids
    event = first.events[0]
    assert event.parent_ids == event.ancestry == ('a', 'b', 'c')
    assert first.body_ids[0] == 'unaffected'
    assert event.remnant_id == second.events[0].remnant_id
    np.testing.assert_allclose(first.system.r, second.system.r, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(first.system.v, second.system.v, atol=1e-14, rtol=1e-14)
    assert first.system.r[0, 1] == pytest.approx(100., abs=1e-10)


def test_time_tie_tolerance_never_merges_a_still_separated_pair():
    spec = BodySpec('Rock', 'asteroid', 1e-14, .1)
    setup = InitialSetup('Short-time separation', (
        InitialBody('a', spec, (-1,0), (1e13,0)), InitialBody('b', spec, (1,0), (0,0)),
        InitialBody('c', spec, (-1.002,100), (1e13,0)), InitialBody('d', spec, (1,100), (0,0))),
        4e-13, collision_mode='perfect_merge')
    result = advance_prepared(build_system(setup))
    events = result.prepared.events
    assert len(events) == 2 and events[0].time < events[1].time
    assert events[1].time-events[0].time == pytest.approx(2e-16, rel=1e-8)
    for frame in result.event_frames:
        if frame.event_phase == 'pre':
            event = frame.events[-1]
            a, b = [frame.body_ids.index(key) for key in event.parent_ids]
            assert np.linalg.norm(frame.r[a]-frame.r[b]) == pytest.approx(.2, abs=3e-13)


def test_ancestry_and_parent_references_map_to_a_valid_planet_remnant():
    setup = fixture(dt=.001)
    moon = InitialBody('moon', BodySpec('Moon', 'moon', 1e-16, .001), (.5, 0.), (0., 0.), 'b')
    p = build_system(replace(setup, bodies=setup.bodies+(moon,), dt=1e-6))
    p, event = merge_pair(p, 'a', 'b')
    assert p.reference_ids[p.body_ids.index('moon')] == event.remnant_id
    assert p.body_specs[p.body_ids.index(event.remnant_id)].kind == 'planet'
    p, second = merge_pair(p, 'moon', event.remnant_id)
    assert second.ancestry == ('a', 'b', 'moon')


def test_remnant_id_avoids_current_and_historical_identities_deterministically():
    setup = fixture(dt=.001)
    _, ordinary = merge_pair(build_system(setup), 'a', 'b')
    stem = ordinary.remnant_id
    occupied = InitialBody(stem, BodySpec('Existing custom ID', 'asteroid', 1e-15, .001), (100,100), (0,0))
    with_key = replace(setup, bodies=setup.bodies+(occupied,))
    p, event = merge_pair(build_system(with_key), 'a', 'b')
    other, reordered = merge_pair(build_system(replace(with_key, bodies=with_key.bodies[::-1])), 'b', 'a')
    assert event.remnant_id == reordered.remnant_id == stem+':1'
    assert stem in p.body_ids and len(set(p.body_ids)) == 2
    previous = CollisionEvent(-1., 'merge', ('past-a', 'past-b'), stem, ('past-a', 'past-b'), 0., 0., 'Earlier identity')
    _, historical = merge_pair(replace(build_system(setup), events=(previous,)), 'a', 'b')
    assert historical.remnant_id == stem+':1'
    captured = replace(setup, provenance={'collision_events': [{'remnant_id': stem, 'ancestry': ['past-a','past-b']}]})
    _, from_capture = merge_pair(build_system(captured), 'a', 'b')
    assert from_capture.remnant_id == stem+':1'
    inherited = replace(setup, provenance={'collision_events': [{'remnant_id': 'a', 'ancestry': ['past-a','past-b']}]})
    _, ancestry = merge_pair(build_system(inherited), 'a', 'b')
    assert ancestry.ancestry == ('b', 'past-a', 'past-b')


def test_point_tracer_absorption_has_zero_feedback_and_tracer_tracer_is_ignored():
    rock = InitialBody('rock', BodySpec('Rock', 'asteroid', 1e-14, .1), (0., 0.), (0., 0.))
    particle = BodySpec('Tracer', 'particle', 0., 100.)  # marker/declared tracer radius is irrelevant
    setup = InitialSetup('Points', (rock, InitialBody('p', particle, (-1., 0.), (10., 0.)),
                                  InitialBody('q', particle, (1., .3), (-10., 0.))), .2, collision_mode='perfect_merge')
    result = advance_prepared(build_system(setup)).prepared
    assert result.body_ids == ('rock', 'q')
    np.testing.assert_array_equal(result.system.r[0], (0., 0.))
    np.testing.assert_array_equal(result.system.v[0], (0., 0.))
    assert result.events[0].kind == 'absorption' and result.events[0].remnant_id is None
    assert result.events[0].time == pytest.approx(.09, abs=1e-10)
    only = replace(setup, bodies=setup.bodies[1:], dt=.2)
    p = advance_prepared(build_system(only)).prepared
    assert len(p.body_ids) == 2 and not p.events


def test_earliest_contact_then_remainder_produces_ordered_event_ancestry():
    setup = fixture(dt=.4)
    extra = InitialBody('c', BodySpec('C', 'asteroid', 1e-14, .1), (1.6, 0), (0., 0.))
    result = advance_prepared(build_system(replace(setup, bodies=setup.bodies+(extra,)))).prepared
    assert len(result.events) == 2
    assert result.events[0].time < result.events[1].time < .4
    assert result.events[-1].ancestry == ('a', 'b', 'c')
    assert result.system.time == .4


def test_bounded_numerical_halt_and_late_stellar_contact_roll_back_prior_merges():
    setup = fixture(dt=.4)
    limited = replace(setup, collision_policy=replace(setup.collision_policy, max_substeps=1))
    p = build_system(limited)
    with pytest.raises(ValueError, match='numerical halt.*budget'):
        advance_prepared(p)
    assert p.system.time == 0 and not p.events and p.body_ids == ('a', 'b')
    star = InitialBody('star', BodySpec('Star', 'star', 1e-14, .1), (1.6, 0), (0., 0.))
    p = build_system(replace(setup, bodies=setup.bodies+(star,)))
    with pytest.raises(ValueError, match='star.*#14'):
        advance_prepared(p)
    assert p.system.time == 0 and not p.events and len(p.body_ids) == 3


def test_insufficient_contact_refinement_halts_instead_of_claiming_tolerance():
    setup = fixture()
    setup = replace(setup, collision_policy=replace(setup.collision_policy, max_refinements=16))
    p = build_system(setup)
    with pytest.raises(ValueError, match='refinement tolerance'):
        advance_prepared(p)
    assert p.system.time == 0 and p.body_ids == ('a', 'b') and not p.events


def test_natural_close_approach_resolves_substeps_before_physical_contact():
    setup = InitialSetup('Nonstellar radial infall', (
        InitialBody('large', BodySpec('Large', 'planet', 1., .0001), (0,0), (0,0)),
        InitialBody('small', BodySpec('Small', 'planet', 3e-6, .0001), (.1,0), (0,0))),
        1e-5, collision_mode='perfect_merge')
    ordinary = build_system(replace(setup, collision_mode='none'))
    with pytest.raises(ValueError, match='Close encounter'):
        ordinary.system.step(1000)
    p = build_system(setup)
    while not p.events and p.system.time < .01:
        p = advance_prepared(p).prepared
    assert p.events and p.events[0].time > ordinary.system.time
    assert p.events[0].kind == 'merge' and p.system.time > ordinary.system.time


def test_collision_substeps_obey_start_and_end_guard(monkeypatch):
    from simulations.star_system import collisions
    original = System.accelerations
    states = []
    def capture(s):
        states.append((s.time, s.resolution_limit()))
        return original(s)
    monkeypatch.setattr(System, 'accelerations', capture)
    result = collisions.advance_prepared(build_system(fixture()))
    # Accepted accelerations occur in start/end pairs; rejected trials can add
    # extra end calls, whose elapsed interval must still be within the start bound.
    assert len(result.prepared.body_ids) == 1 and len(states) > 10
    for (start, limit), (end, _) in zip(states[::2], states[1::2]):
        if end > start:
            assert end-start <= .5*limit*(1+1e-12)


def contact_time(dt):
    setup = fixture(dt=dt, speed=1., mass=1e-4)
    p = build_system(setup)
    while not p.events and p.system.time < 3.:
        p = advance_prepared(p).prepared
    assert p.events
    return p.events[0].time


def test_contact_time_converges_under_macro_dt_halving():
    times = [contact_time(dt) for dt in (.08, .04, .02, .0025)]
    errors = [abs(value-times[-1]) for value in times[:-1]]
    print('contact convergence (dt=.08,.04,.02,reference=.0025):', times, 'errors:', errors)
    assert errors[2] < errors[1] < errors[0]
    assert errors[0] < 2e-5 and errors[2] < 2e-6
