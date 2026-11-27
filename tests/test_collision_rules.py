"""Selectable collision rules: perfect merge, and speed-and-angle merge/graze/fragment."""
import json
import math

import numpy as np
import pytest

from simulations.star_system.collisions import (
    advance_prepared, classify_impact, fragment_mass_fractions, mutual_escape_speed,
)
from simulations.star_system.core import G, SOLAR_RADIUS_AU
from simulations.star_system.initialization import build_system
from simulations.star_system.model import (
    PERFECT_MERGE_PAIRS, SPEED_ANGLE_VERSION, BodySpec, InitialBody, InitialSetup, with_collision_rule,
)
from simulations.star_system.persistence import load_setup, save_setup


def pair(mode, *, y=0., speed=5., mass=1e-3, radius=.05, dt=.05, gap=.1001):
    return InitialSetup('Impact', (
        InitialBody('a', BodySpec('A', 'planet', mass, radius), (-gap/2, y), (speed, 0.)),
        InitialBody('b', BodySpec('B', 'planet', mass, radius), (gap/2, 0.), (0., 0.)),
    ), dt, collision_mode=mode)


def test_speed_and_angle_map_boundaries_and_fragment_fractions():
    escape = mutual_escape_speed(1e-3, 1e-3, .1)
    assert classify_impact(escape, escape, 0.) == 'merge'
    assert classify_impact(escape*1.01, escape, .5) == 'graze'
    assert classify_impact(escape*1.01, escape, .49) == 'fragment'
    largest, each, lost = fragment_mass_fractions(escape*4, escape, 0.)
    assert largest+2*each+lost == pytest.approx(1.)
    assert largest >= .2 and each > 0 and lost > 0
    with pytest.raises(ValueError, match='fragment'):
        fragment_mass_fractions(escape, escape, 0.)


def test_fast_head_on_fragments_and_accounts_for_lost_mass():
    setup = pair('speed_angle', dt=1e-4)
    prepared = build_system(setup)
    before = prepared.system
    momentum = (before.masses[:, None]*before.v).sum(0)
    outcome = advance_prepared(prepared)
    result = outcome.prepared
    event = next(item for item in result.events if item.kind == 'fragment')
    assert event.rule == 'speed_angle' and event.outcome == 'fragment'
    assert event.parent_ids == ('a', 'b') and event.remnant_id in event.descendant_ids
    assert len(event.descendant_ids) == 3 and set(event.descendant_ids) <= set(result.body_ids)
    assert 'a' not in result.body_ids and 'b' not in result.body_ids
    assert result.system.time == pytest.approx(setup.dt)
    np.testing.assert_allclose(result.system.masses.sum()+event.lost_mass, before.masses.sum(), rtol=1e-12)
    pre = next(frame for frame in outcome.event_frames if frame.event_phase == 'pre')
    indices = [pre.body_ids.index(key) for key in event.parent_ids]
    total = float(pre.masses[indices].sum())
    com_velocity = (pre.masses[indices, None]*pre.v[indices]).sum(0)/total
    after = (result.system.masses[:, None]*result.system.v).sum(0)
    np.testing.assert_allclose(after+event.lost_mass*com_velocity, momentum, rtol=1e-12, atol=1e-15)
    post = next(frame for frame in outcome.event_frames if frame.event_phase == 'post')
    pre_kinetic = float(.5*np.sum(pre.masses*np.sum(pre.v**2, axis=1)))
    after_kinetic = float(.5*np.sum(post.masses*np.sum(post.v**2, axis=1)))
    carried = .5*event.lost_mass*float(np.dot(com_velocity, com_velocity))
    assert after_kinetic-pre_kinetic == pytest.approx(-event.lost_relative_kinetic-carried, rel=1e-9, abs=1e-15)
    assert prepared.system.time == 0 and prepared.body_ids == ('a', 'b')


def test_fast_wide_impact_grazes_and_slow_impact_merges():
    grazed = advance_prepared(build_system(pair('speed_angle', y=.07, speed=5., gap=.3, dt=.08))).prepared
    assert grazed.body_ids == ('a', 'b')
    graze = next(event for event in grazed.events if event.kind == 'graze')
    assert graze.rule == 'speed_angle' and graze.outcome == 'graze' and graze.lost_mass == 0
    assert graze.impact_parameter >= .5 and graze.descendant_ids == ('a', 'b')
    merged = advance_prepared(build_system(pair('speed_angle', speed=.2))).prepared
    assert len(merged.body_ids) == 1
    merge = next(event for event in merged.events if event.kind == 'merge')
    assert merge.rule == 'speed_angle' and merge.outcome == 'merge' and merge.descendant_ids == (merge.remnant_id,)


def test_same_fast_impact_still_merges_under_perfect_merge():
    result = advance_prepared(build_system(pair('perfect_merge'))).prepared
    assert len(result.body_ids) == 1
    event = next(item for item in result.events if item.kind == 'merge')
    assert event.rule == 'perfect_merge' and event.outcome == 'merge'
    assert 'not a general collision law' in event.message.lower() or 'disclosed option' in event.message.lower()


def test_stellar_overlap_is_logged_once_and_later_steps_continue():
    setup = InitialSetup('Through a star', (
        InitialBody('star', BodySpec('Star', 'star', 1., 1.), (0., 0.), (0., 0.)),
        InitialBody('rock', BodySpec('Rock', 'planet', 1e-6, .05), (1.051, 0.), (-.2, 0.)),
    ), .05, collision_mode='speed_angle')
    first = advance_prepared(build_system(setup)).prepared
    assert first.system.time == pytest.approx(.05) and first.body_ids == ('star', 'rock')
    logged = [event for event in first.events if event.kind == 'unmodeled']
    assert len(logged) == 1 and logged[0].rule == 'speed_angle' and logged[0].outcome == 'unmodeled'
    assert logged[0].parent_ids == ('rock', 'star')
    second = advance_prepared(first).prepared
    assert second.system.time == pytest.approx(.1) and second.body_ids == ('star', 'rock')
    assert [event.kind for event in second.events].count('unmodeled') == 1
    assert not any(event.kind == 'merge' for event in second.events)


def test_close_encounter_and_orbital_change_are_logged_without_contact():
    setup = InitialSetup('Flyby', (
        InitialBody('star', BodySpec('Star', 'star', 1., SOLAR_RADIUS_AU), (0., 0.), (0., 0.)),
        InitialBody('planet', BodySpec('Planet', 'planet', 1e-4, .05), (2., 0.), (0., math.sqrt(G*1.0001/2.))),
        InitialBody('intruder', BodySpec('Intruder', 'planet', .08, .05), (2.9, .18), (-10., 0.)),
    ), .12, collision_mode='perfect_merge')
    result = advance_prepared(build_system(setup)).prepared
    assert result.system.time == pytest.approx(.12)
    assert result.body_ids == ('star', 'planet', 'intruder')
    encounter = next(event for event in result.events if event.kind == 'close_encounter')
    assert encounter.rule == 'perfect_merge' and encounter.outcome == 'close_encounter'
    assert set(encounter.parent_ids) == {'planet', 'intruder'}
    assert encounter.time > 0
    change = next(event for event in result.events if event.kind == 'orbital_change')
    assert change.outcome == 'orbital_change' and change.rule == 'perfect_merge'
    assert change.parent_ids[0] in ('planet', 'intruder')


def test_rule_is_stored_on_the_setup_and_a_legacy_halt_id_loads(tmp_path):
    stored = with_collision_rule(pair('perfect_merge'), 'speed_angle')
    assert stored.collision_mode == 'speed_angle'
    assert stored.collision_policy.version == SPEED_ANGLE_VERSION
    path = tmp_path/'rule.json'
    save_setup(stored, path)
    loaded = load_setup(path)
    assert loaded.collision_mode == 'speed_angle'
    assert loaded.collision_policy.version == SPEED_ANGLE_VERSION
    assert loaded.collision_policy.pairs != PERFECT_MERGE_PAIRS
    document = json.loads(path.read_text())
    merged = with_collision_rule(stored, 'perfect_merge')
    save_setup(merged, path)
    document = json.loads(path.read_text())
    document['setup']['collision_policy']['pairs'] = 'nonstellar-massive-merge-point-tracer-absorb-star-halt-v1'
    path.write_text(json.dumps(document))
    legacy = load_setup(path)
    assert legacy.collision_mode == 'perfect_merge'
    assert legacy.collision_policy.pairs == PERFECT_MERGE_PAIRS
    assert legacy.collision_policy.version == merged.collision_policy.version
