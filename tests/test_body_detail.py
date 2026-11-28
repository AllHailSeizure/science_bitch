"""Body detail: point mass, a fixed J2 bulge, and spin with torques."""
import math

import numpy as np
import pytest

from simulations.star_system.body_detail import (
    ASSUMPTIONS, INSERT_UNSUPPORTED, NO_POPULATION, NO_REMNANT, SPIN_NEEDS_3D,
    j2_acceleration, measure_leo_node, quadrupole_torque,
)
from simulations.star_system.data import CATALOG
from simulations.star_system.initialization import build_system, insert_body
from simulations.star_system.model import (
    BODY_DETAIL_MODEL_ID, BodyDetailPolicy, BodySpec, InitialBody, InitialSetup, PopulationSpec,
)
from simulations.star_system.persistence import _decode, _document, save_setup, load_setup, setup_digest
from simulations.star_system.populations import Population
from simulations.star_system.scenarios import get_scenario


def _sun_earth(*, level='point_mass', dimensions=3):
    sun, earth = CATALOG['sun'], CATALOG['earth']
    bodies = (
        InitialBody('sun', sun, (0.0, 0.0, 0.0)[:dimensions], (0.0, 0.0, 0.0)[:dimensions]),
        InitialBody('earth', earth, (1.0, 0.0, 0.0)[:dimensions], (0.0, 6.28, 0.0)[:dimensions]),
    )
    return InitialSetup('Sun Earth', bodies, 1e-4, body_detail=BodyDetailPolicy(level))


def test_point_mass_is_the_default_and_leaves_the_force_and_state_alone():
    plain = build_system(_sun_earth())
    named = build_system(_sun_earth())
    assert plain.system.detail is None and named.system.detail is None
    start = plain.system.r.copy()
    for _ in range(30):
        plain.system.step()
        named.system.step()
    assert plain.system.detail is None
    np.testing.assert_array_equal(plain.system.r, named.system.r)
    np.testing.assert_array_equal(plain.system.v, named.system.v)
    assert not np.array_equal(plain.system.r, start)


def test_explicit_point_mass_matches_the_unset_level():
    unset = build_system(get_scenario('sun-earth-jupiter'))
    explicit = build_system(replace_detail(get_scenario('sun-earth-jupiter'), 'point_mass'))
    assert explicit.system.detail is None
    for _ in range(20):
        unset.system.step()
        explicit.system.step()
    np.testing.assert_array_equal(unset.system.r, explicit.system.r)
    np.testing.assert_array_equal(unset.system.v, explicit.system.v)


def test_a_tilted_bulge_accelerates_out_of_the_plane_and_keeps_linear_momentum():
    prepared = build_system(_sun_earth(level='fixed_oblateness'))
    motion = prepared.system.detail
    assert motion.level == 'fixed_oblateness'
    assert motion.angular_momentum is None
    assert motion.pole[1, 0] != 0.0
    acc = prepared.system.accelerations()
    assert acc[1, 2] != 0.0
    masses = prepared.system.masses
    np.testing.assert_allclose(masses[:, None] * acc, masses[:, None] * prepared.system.accelerations(), rtol=0, atol=0)
    np.testing.assert_allclose(np.sum(masses[:, None] * acc, axis=0), 0.0, atol=1e-18)


def test_fixed_oblateness_in_a_plane_pins_the_pole_to_plus_z():
    prepared = build_system(_sun_earth(level='fixed_oblateness', dimensions=2))
    np.testing.assert_allclose(prepared.system.detail.pole[1], (0.0, 0.0, 1.0))
    assert prepared.system.accelerations().shape == (2, 2)


def test_j2_nodal_precession_of_a_low_earth_orbit():
    fixed = measure_leo_node(orbits=8, level='fixed_oblateness')
    spinning = measure_leo_node(orbits=8, level='spin')
    point = measure_leo_node(orbits=4, level='point_mass')
    assert fixed.analytic_deg_per_day == pytest.approx(-5.008676911936782, rel=1e-12)
    assert fixed.measured_deg_per_day == pytest.approx(fixed.analytic_deg_per_day, rel=0.01, abs=0.02)
    assert spinning.measured_deg_per_day == pytest.approx(fixed.analytic_deg_per_day, rel=0.01, abs=0.02)
    assert abs(point.measured_deg_per_day) < 0.02


def test_the_contact_path_precesses_the_same_node():
    contact = measure_leo_node(
        orbits=6, level='fixed_oblateness', steps_per_orbit=200, collision_mode='ignore')
    assert contact.measured_deg_per_day == pytest.approx(contact.analytic_deg_per_day, rel=0.02, abs=0.05)


def test_spin_stores_the_quadrupole_torque_and_wobbles_about_angular_momentum():
    prepared = build_system(_sun_earth(level='spin'))
    system = prepared.system
    motion = system.detail
    assert motion.spin_rates() is not None
    torque = quadrupole_torque(system, motion)
    acc = j2_acceleration(system.r, system.masses, motion.j2, motion.radius, motion.pole)
    orbital = np.zeros(3)
    for index in range(len(system.masses)):
        orbital += system.masses[index] * np.cross(system.r[index], acc[index])
    np.testing.assert_allclose(torque.sum(axis=0) + orbital, 0.0, atol=1e-20)
    assert np.linalg.norm(torque) > 0.0
    before = motion.angular_momentum.copy()
    system.step()
    assert not np.array_equal(motion.angular_momentum, before)

    lone = build_system(InitialSetup(
        'Earth', (InitialBody('earth', CATALOG['earth'], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),),
        1e-5, body_detail=BodyDetailPolicy('spin')))
    wobble = lone.system.detail
    pole = wobble.pole[0].copy()
    moment = wobble.inertia_a[0]
    angular = moment * 2.0 * np.array([0.2, 0.0, 1.0])
    wobble.angular_momentum[0] = angular
    wobble.pole[0] = pole
    angle = math.degrees(math.acos(np.clip(np.dot(angular, pole) / (np.linalg.norm(angular) * np.linalg.norm(pole)), -1, 1)))
    dt = 1e-4
    wobble.advance(lone.system, dt)
    later = wobble.pole[0]
    kept = math.degrees(math.acos(np.clip(
        np.dot(wobble.angular_momentum[0], later) / (np.linalg.norm(wobble.angular_momentum[0]) * np.linalg.norm(later)), -1, 1)))
    assert kept == pytest.approx(angle, abs=1e-8)
    axis = angular / np.linalg.norm(angular)
    perpendicular = pole - np.dot(pole, axis) * axis
    turned = later - np.dot(later, axis) * axis
    swept = math.atan2(np.dot(axis, np.cross(perpendicular, turned)), np.dot(perpendicular, turned))
    assert swept == pytest.approx(np.linalg.norm(angular) / moment * dt, rel=1e-9)


def test_spin_needs_three_dimensions_and_refuses_a_remnant_a_population_and_insertion():
    with pytest.raises(ValueError, match=SPIN_NEEDS_3D):
        build_system(_sun_earth(level='spin', dimensions=2))
    planar = get_scenario('sun-earth-jupiter')
    for mode in ('perfect_merge', 'speed_angle'):
        with pytest.raises(ValueError, match=NO_REMNANT):
            build_system(replace_detail(planar, 'fixed_oblateness', collision_mode=mode))
    population = PopulationSpec('rocks', 'sun', Population(count=4, seed=3))
    with pytest.raises(ValueError, match=NO_POPULATION):
        build_system(replace_detail(planar, 'fixed_oblateness', population=population))
    prepared = build_system(_sun_earth(level='spin'))
    extra = InitialBody('rock', BodySpec('rock', 'planet', 1e-6, 1e-5), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    with pytest.raises(ValueError, match=INSERT_UNSUPPORTED):
        insert_body(prepared, extra)


def test_a_memory_budget_halt_restores_the_spin():
    from dataclasses import replace
    from simulations.star_system.recording import Calculation, CalculationRequest
    setup = _sun_earth(level='spin')
    initial = build_system(setup).system.detail.angular_momentum.copy()
    calculation = Calculation(setup, CalculationRequest(1e-4, 1))
    calculation.request = replace(calculation.request, max_bytes=calculation._bytes + 1)
    calculation.advance(1)
    assert calculation.result.status == 'halted'
    np.testing.assert_array_equal(calculation.prepared.system.detail.angular_momentum, initial)
    np.testing.assert_array_equal(calculation.prepared.system.r, build_system(setup).system.r)


def test_point_mass_documents_omit_the_policy_and_a_selected_level_round_trips(tmp_path):
    baseline = get_scenario('sun-earth-jupiter')
    assert 'body_detail' not in _document(baseline)['setup']
    assert setup_digest(baseline) == setup_digest(load_setup(save(tmp_path, baseline)))
    selected = replace_detail(baseline, 'fixed_oblateness')
    loaded = load_setup(save(tmp_path, selected))
    assert loaded.body_detail.level == 'fixed_oblateness'
    assert loaded.body_detail.version == BODY_DETAIL_MODEL_ID
    assert setup_digest(loaded) == setup_digest(selected)
    stored = _document(baseline)
    stored['setup']['body_detail'] = {'level': 'point_mass', 'version': BODY_DETAIL_MODEL_ID}
    with pytest.raises(ValueError, match='Omit body_detail'):
        _decode(stored)
    prepared = build_system(loaded)
    assert ASSUMPTIONS in prepared.system.detail.disclosure


def replace_detail(setup, level, **updates):
    from dataclasses import replace
    return replace(setup, body_detail=BodyDetailPolicy(level), **updates)


def save(tmp_path, setup):
    path = tmp_path / 'setup.json'
    save_setup(setup, path)
    return path
