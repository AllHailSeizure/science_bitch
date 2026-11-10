"""Prepared experiments share a Solar System origin and preserve frozen originals."""
from dataclasses import replace

import numpy as np
import pytest

from simulations.star_system.data import CATALOG
from simulations.star_system.initialization import build_system
from simulations.star_system.persistence import load_setup, save_setup, setup_digest
from simulations.star_system.templates import (
    EXPERIMENTS, SOLAR_PLANETS, get_experiment, update_body, place_body, remove_body,
)


def test_solar_system_has_eight_planets_with_resolved_sun_references():
    first, second = get_experiment('solar-system'), get_experiment('solar-system')
    assert first is not second
    assert tuple(b.id for b in first.bodies) == ('sun', *SOLAR_PLANETS)
    assert all(b.reference_id == 'sun' for b in first.bodies[1:])
    assert all(e.key != 'sun-earth-jupiter' for e in EXPERIMENTS)
    prepared = build_system(first)
    assert prepared.system.time == 0
    np.testing.assert_allclose(np.sum(prepared.system.masses[:, None]*prepared.system.v, axis=0), 0, atol=1e-17)


@pytest.mark.parametrize('key', ['solar-system', 'heavier-star', 'earth-moon', 'asteroid-ring'])
def test_prepared_experiments_are_valid_solar_system_variations(key):
    setup = get_experiment(key)
    assert set(('sun', *SOLAR_PLANETS)) <= {b.id for b in setup.bodies}
    assert setup.provenance['starting_template'] == 'solar-system'
    prepared = build_system(setup)
    prepared.system.step(4)
    assert prepared.system.time > 0


def test_mass_edit_keeps_starting_vectors_and_original_but_changes_outcome():
    original = get_experiment('solar-system')
    digest = setup_digest(original)
    edited = update_body(original, 'sun', spec=replace(CATALOG['sun'], mass=2.))
    a, b = build_system(original), build_system(edited)
    np.testing.assert_array_equal(a.system.r, b.system.r)
    np.testing.assert_array_equal(a.system.v, b.system.v)
    assert edited.preset_id is None
    assert edited.provenance['origin_preset'] == original.preset_id
    a.system.dt = b.system.dt = min(a.system.dt, b.system.dt)
    a.system.step(20)
    b.system.step(20)
    assert not np.allclose(a.system.r, b.system.r, rtol=1e-10, atol=1e-10)
    assert setup_digest(original) == digest


def test_replace_body_preserves_identity_and_can_save_and_reload(tmp_path):
    original = get_experiment('solar-system')
    edited = update_body(original, 'jupiter', spec=CATALOG['saturn'])
    before, after = build_system(original), build_system(edited)
    assert after.body_specs[after.body_ids.index('jupiter')].name == 'Saturn'
    np.testing.assert_array_equal(after.system.r, before.system.r)
    np.testing.assert_array_equal(after.system.v, before.system.v)
    path = tmp_path/'variation.json'
    save_setup(edited, path)
    assert setup_digest(load_setup(path)) == setup_digest(edited)


def test_scene_placement_is_relative_once_and_moon_step_is_automatic():
    original = get_experiment('solar-system')
    prepared = build_system(original)
    parent = prepared.body_ids.index('earth')
    world = prepared.system.r[parent] + np.array([.00257, 0.])
    edited = place_body(original, CATALOG['moon'], world, 'earth', circular=True)
    after = build_system(edited)
    assert len(edited.bodies) == len(original.bodies)+1
    assert edited.bodies[-1].reference_id == 'earth'
    np.testing.assert_allclose(after.system.r[-1], world, atol=1e-15)
    np.testing.assert_array_equal(after.system.r[:-1], prepared.system.r)
    assert edited.dt < original.dt


def test_invalid_placement_and_removal_leave_original_usable():
    original = get_experiment('solar-system')
    digest = setup_digest(original)
    with pytest.raises(ValueError, match='overlap|nonzero'):
        place_body(original, CATALOG['earth'], build_system(original).system.r[0], 'sun', circular=True)
    with pytest.raises(ValueError, match='reference|referenc'):
        remove_body(original, 'sun')
    assert setup_digest(original) == digest
    build_system(original)
