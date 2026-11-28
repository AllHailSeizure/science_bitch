"""Lasting tides: constant time lag, separate from the close-pass kick."""
from dataclasses import replace

import numpy as np
import pytest

from simulations.star_system.body_detail import LASTING_NEEDS_SPIN
from simulations.star_system.core import G
from simulations.star_system.initialization import build_system
from simulations.star_system.lasting_tides import (
    TOGETHER, circular_recession_rate, ctl_acceleration, ctl_torque, earth_lag_seconds,
    lunar_pair, measure_lunar_recession,
)
from simulations.star_system.model import BodyDetailPolicy, LastingTidePolicy, TidalPolicy
from simulations.star_system.persistence import _decode, _document, load_setup, save_setup, setup_digest
from simulations.star_system.tides import dissipate_close_passes


def test_the_time_lag_that_hits_the_lunar_recession_is_an_effective_oceanic_lag():
    lag = earth_lag_seconds()
    earth, moon, semi_major, mean_motion = lunar_pair()
    from simulations.star_system.body_detail import EARTH_K2, EARTH_SIDEREAL_SECONDS, YEAR_SECONDS
    spin = 2.0 * np.pi * YEAR_SECONDS / EARTH_SIDEREAL_SECONDS
    rate = circular_recession_rate(earth.mass, moon.mass, earth.radius, semi_major, spin, EARTH_K2, lag / YEAR_SECONDS)
    cm = rate * (149597870.7 * 1e5)
    assert cm == pytest.approx(3.82, rel=1e-9)
    assert 500.0 < lag < 800.0
    quoted = circular_recession_rate(
        earth.mass, moon.mass, earth.radius, semi_major, spin, EARTH_K2, (213.0 / EARTH_K2) / YEAR_SECONDS)
    assert quoted * (149597870.7 * 1e5) == pytest.approx(4.466, rel=0.01)
    assert mean_motion > spin * 0.01


def test_the_integrated_moon_recedes_and_the_spin_slows_until_the_rates_match():
    comparison = measure_lunar_recession(orbits=4)
    assert comparison.analytic_cm_per_year == pytest.approx(3.82, rel=1e-9)
    assert comparison.measured_cm_per_year == pytest.approx(3.82, rel=0.01, abs=0.05)
    assert comparison.k2 == pytest.approx(0.299)

    prepared = _aligned_moon()
    system = prepared.system
    motion = system.detail
    mu = G * (system.masses[0] + system.masses[1])
    rel_v = system.v[1] - system.v[0]
    rel_a = ctl_acceleration(system, motion)
    rel_a = rel_a[1] - rel_a[0]
    energy_rate = float(np.dot(rel_v, rel_a))
    distance = float(np.linalg.norm(system.r[1] - system.r[0]))
    axis = _axis(system, mu)
    da = energy_rate * 2.0 * axis * axis / mu
    assert da * (149597870.7 * 1e5) == pytest.approx(3.82, rel=0.01)
    assert distance > 0.0

    locked = ctl_torque(system, motion)
    mean = np.sqrt(mu / distance ** 3)
    motion.angular_momentum[0] = np.array([0.0, 0.0, motion.inertia_c[0] * mean])
    assert abs(ctl_torque(system, motion)[0, 2]) <= abs(locked[0, 2]) * 1e-6 + 1e-30
    motion.angular_momentum[0] *= 2.0
    assert ctl_torque(system, motion)[0, 2] < 0.0
    motion.angular_momentum[0] *= 0.0
    assert ctl_torque(system, motion)[0, 2] > 0.0
    assert 'both be on' in TOGETHER and 'losses add' in TOGETHER


def test_lasting_tides_require_spin_and_do_not_switch_on_the_close_pass_kick():
    with pytest.raises(ValueError, match=LASTING_NEEDS_SPIN):
        build_system(_policy('point_mass', lasting=True))
    with pytest.raises(ValueError, match=LASTING_NEEDS_SPIN):
        build_system(_policy('fixed_oblateness', lasting=True))
    prepared = build_system(_policy('spin', lasting=True))
    assert prepared.setup.tides.enabled is False
    assert prepared.setup.lasting_tides.enabled is True
    assert prepared.system.detail.lasting is True
    close = build_system(replace(_policy('spin', lasting=False), tides=TidalPolicy(enabled=True)))
    assert close.setup.tides.enabled is True
    assert close.system.detail.lasting is False


def test_close_pass_and_lasting_tides_add_and_the_kick_stays_spin_free():
    lasting = build_system(_policy('spin', lasting=True))
    both = build_system(replace(_policy('spin', lasting=True), tides=TidalPolicy(enabled=True)))
    np.testing.assert_array_equal(lasting.system.accelerations(), both.system.accelerations())
    point = build_system(_policy('point_mass', lasting=False))
    assert not np.allclose(lasting.system.accelerations(), point.system.accelerations())

    from simulations.star_system.data import CATALOG
    from simulations.star_system.model import InitialBody, InitialSetup
    jupiter, earth = CATALOG['jupiter'], CATALOG['earth']
    radius = float(jupiter.radius)
    setup = InitialSetup('Close pass beside spin', (
        InitialBody('jupiter', jupiter, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        InitialBody('earth', earth, (2.0 * radius, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ), 1e-8, body_detail=BodyDetailPolicy('spin'), tides=TidalPolicy(enabled=True))
    prepared = build_system(setup)
    offset = 2.0 * radius
    half = 0.2 * radius
    start_r = np.array([[0.0, 0.0, 0.0], [offset, -half, 0.0]])
    start_v = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    prepared.system.r[:] = np.array([[0.0, 0.0, 0.0], [offset, half, 0.0]])
    prepared.system.v[:] = start_v
    before = prepared.system.detail.angular_momentum.copy()
    velocities = prepared.system.v.copy()
    updated = dissipate_close_passes(prepared, start_r, start_v)
    assert any(getattr(event, 'kind', None) == 'close-pass-dissipation' for event in updated.events)
    assert not np.array_equal(updated.system.v, velocities)
    np.testing.assert_array_equal(updated.system.detail.angular_momentum, before)


def test_a_disabled_lasting_tide_is_omitted_and_an_enabled_one_round_trips(tmp_path):
    baseline = _policy('point_mass', lasting=False)
    assert 'lasting_tides' not in _document(baseline)['setup']
    path = tmp_path / 'setup.json'
    save_setup(baseline, path)
    assert setup_digest(load_setup(path)) == setup_digest(baseline)
    selected = _policy('spin', lasting=True)
    save_setup(selected, path)
    loaded = load_setup(path)
    assert loaded.lasting_tides.enabled is True
    assert loaded.body_detail.level == 'spin'
    assert setup_digest(loaded) == setup_digest(selected)
    stored = _document(baseline)
    stored['setup']['lasting_tides'] = {'enabled': False, 'version': 'constant-time-lag-v1'}
    with pytest.raises(ValueError, match='Omit lasting_tides'):
        _decode(stored)


def test_a_formation_checkpoint_keeps_the_spin(tmp_path):
    from simulations.star_system.body_detail import leo_initial_setup
    from simulations.star_system.formation import FormationExperiment, load_checkpoint, save_checkpoint
    setup, _axis, _inclination = leo_initial_setup(level='spin', primary_kind='star', collision_mode='ignore')
    setup = replace(setup, lasting_tides=LastingTidePolicy(enabled=True))
    experiment = FormationExperiment(setup, ejection_radius=1.0)
    experiment.integrate(float(setup.dt) * 3, finish=True)
    assert experiment.combination()['body_detail'] == 'spin'
    assert experiment.combination()['lasting_tides'] == 'constant-time-lag-v1'
    path = tmp_path / 'checkpoint.json'
    save_checkpoint(experiment, path)
    restored = load_checkpoint(path)
    assert restored.combination() == experiment.combination()
    np.testing.assert_allclose(
        restored.prepared.system.detail.angular_momentum, experiment.prepared.system.detail.angular_momentum)
    np.testing.assert_allclose(restored.prepared.system.detail.pole, experiment.prepared.system.detail.pole)


def _policy(level, *, lasting):
    from simulations.star_system.body_detail import leo_initial_setup
    setup, _semi, _inclination = leo_initial_setup(level='point_mass')
    return replace(setup, body_detail=BodyDetailPolicy(level), lasting_tides=LastingTidePolicy(enabled=lasting))


def _aligned_moon():
    """The same initial Earth–Moon state the recession measurement integrates."""
    from simulations.star_system.body_detail import EARTH_SIDEREAL_SECONDS, YEAR_SECONDS
    from simulations.star_system.model import InitialBody, InitialSetup
    earth, moon, semi_major, mean_motion = lunar_pair()
    moon_spec = replace(moon, kind='planet')
    fraction = moon.mass / (earth.mass + moon.mass)
    rel_r = np.array([semi_major, 0.0, 0.0])
    rel_v = np.array([0.0, mean_motion * semi_major, 0.0])
    setup = InitialSetup('Earth-Moon constant time lag', (
        InitialBody('earth', earth, tuple(-fraction * rel_r), tuple(-fraction * rel_v)),
        InitialBody('moon', moon_spec, tuple((1.0 - fraction) * rel_r), tuple((1.0 - fraction) * rel_v)),
    ), (2.0 * np.pi / mean_motion) / 800.0,
        body_detail=BodyDetailPolicy('spin'), lasting_tides=LastingTidePolicy(enabled=True))
    prepared = build_system(setup)
    prepared.system.detail.align_pole(0)
    prepared.system.detail.align_pole(1)
    moment = prepared.system.detail.inertia_c[1]
    prepared.system.detail.angular_momentum[1] = np.array([0.0, 0.0, moment * mean_motion])
    prepared.system.detail.lag[1] = 0.0
    prepared.system.detail.j2[:] = 0.0
    assert abs(np.linalg.norm(prepared.system.detail.omega(0)) - 2.0 * np.pi * YEAR_SECONDS / EARTH_SIDEREAL_SECONDS) < 1e-6
    return prepared


def _axis(system, mu):
    rel = system.r[1] - system.r[0]
    vel = system.v[1] - system.v[0]
    distance = float(np.linalg.norm(rel))
    energy = 0.5 * float(np.dot(vel, vel)) - mu / distance
    return -mu / (2.0 * energy)
