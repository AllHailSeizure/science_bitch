"""Opt-in 1PN precession and 2.5PN inspiral, off unless the policy is enabled."""
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from simulations.star_system.collisions import advance_prepared
from simulations.star_system.core import G, System
from simulations.star_system.initialization import build_system
from simulations.star_system.model import BodySpec, InitialBody, InitialSetup, RelativisticPolicy
from simulations.star_system.persistence import load_setup, save_setup, setup_digest
from simulations.star_system.scenarios import get_scenario
from simulations.star_system.relativity import (
    peters_coalescence_time, peters_semi_major_decay, relativistic_context, schwarzschild_perihelion_advance,
)

# NASA planetary fact sheet values used for the classic GR perihelion.
MERCURY_A = 0.38709893
MERCURY_E = 0.20563069
_ARCSEC = (180.0 / math.pi) * 3600.0


def _arcsec_per_century(radians_per_year):
    return radians_per_year * 100.0 * _ARCSEC


def _eccentricity_angle(system):
    delta = system.r[1] - system.r[0]
    velocity = system.v[1] - system.v[0]
    distance = np.linalg.norm(delta)
    mu = G * float(system.masses[0] + system.masses[1])
    eccentricity = (np.dot(velocity, velocity) * delta - np.dot(velocity, delta) * velocity) / mu - delta / distance
    return math.atan2(float(eccentricity[1]), float(eccentricity[0]))


def _mercury_system(divisor):
    mu = G * 1.0
    perihelion = MERCURY_A * (1.0 - MERCURY_E)
    speed = math.sqrt(mu * (2.0 / perihelion - 1.0 / MERCURY_A))
    masses = np.array([1.0, 0.0])
    position = np.zeros((2, 3))
    velocity = np.zeros((2, 3))
    position[1, 0] = perihelion
    velocity[1, 1] = speed
    period = 2.0 * math.pi * math.sqrt(MERCURY_A ** 3 / mu)
    return System(masses, position, velocity, period / divisor), period


def _advance_angle(policy, divisor, orbits):
    system, period = _mercury_system(divisor)
    setup = InitialSetup('Mercury', (
        InitialBody('sun', BodySpec('Sun', 'star', 1.0, 0.00465047), (0., 0., 0.), (0., 0., 0.)),
        InitialBody('mercury', BodySpec('Mercury', 'particle', 0.0, None), (1., 0., 0.), (0., 1., 0.)),
    ), system.dt, relativistic=policy)
    start = _eccentricity_angle(system)
    steps = int(round(orbits * period / system.dt))
    with relativistic_context(setup):
        system.step(steps)
    elapsed = steps * system.dt
    return (_eccentricity_angle(system) - start) / elapsed, elapsed


def test_mercury_formula_is_about_43_arcsec_per_century():
    advance = schwarzschild_perihelion_advance(MERCURY_A, MERCURY_E, 1.0)
    period = 2.0 * math.pi * math.sqrt(MERCURY_A ** 3 / G)
    arcsec = _arcsec_per_century(advance / period)
    assert arcsec == pytest.approx(42.98, abs=0.02)
    assert arcsec == pytest.approx(43.0, abs=0.1)


def test_mercury_orbit_precesses_only_when_the_module_is_on():
    divisor, orbits = 600, 12
    off, _elapsed = _advance_angle(RelativisticPolicy(), divisor, orbits)
    on, _elapsed = _advance_angle(RelativisticPolicy(enabled=True), divisor, orbits)
    measured = _arcsec_per_century(on - off)
    analytic = _arcsec_per_century(
        schwarzschild_perihelion_advance(MERCURY_A, MERCURY_E, 1.0)
        / (2.0 * math.pi * math.sqrt(MERCURY_A ** 3 / G)))
    assert measured == pytest.approx(analytic, abs=0.05)
    assert measured == pytest.approx(43.0, abs=0.1)
    assert abs(_arcsec_per_century(off)) > abs(measured) * 0.01


def _circular_binary(separation):
    mu = G * 2.0
    speed = math.sqrt(mu / separation) / 2.0
    masses = np.array([1.0, 1.0])
    position = np.array([[-separation / 2.0, 0.0, 0.0], [separation / 2.0, 0.0, 0.0]])
    velocity = np.array([[0.0, -speed, 0.0], [0.0, speed, 0.0]])
    period = 2.0 * math.pi * math.sqrt(separation ** 3 / mu)
    return System(masses, position, velocity, period / 200.0), period


def _semi_major(system):
    delta = system.r[0] - system.r[1]
    velocity = system.v[0] - system.v[1]
    distance = float(np.linalg.norm(delta))
    mu = G * float(np.sum(system.masses))
    energy = 0.5 * float(np.dot(velocity, velocity)) - mu / distance
    return -mu / (2.0 * energy)


def test_peters_coalescence_time_matches_the_orbit_averaged_decay():
    separation = 1.0e-5
    decay = peters_semi_major_decay(separation, 1.0, 1.0)
    lifetime = peters_coalescence_time(separation, 1.0, 1.0)
    assert decay < 0.0
    assert separation / (4.0 * abs(decay)) == pytest.approx(lifetime)


def test_radiation_reaction_inspiral_matches_the_peters_time():
    separation = 1.0e-5
    system, period = _circular_binary(separation)
    orbits = 20
    setup = InitialSetup('Pair', (
        InitialBody('a', BodySpec('A', 'star', 1.0, 1.0e-8), (-separation / 2.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        InitialBody('b', BodySpec('B', 'star', 1.0, 1.0e-8), (separation / 2.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ), system.dt, relativistic=RelativisticPolicy(enabled=True, conservative=False, radiation=True))
    start = _semi_major(system)
    steps = int(round(orbits * period / system.dt))
    with relativistic_context(setup):
        system.step(steps)
    elapsed = steps * system.dt
    decay = (_semi_major(system) - start) / elapsed
    assert decay == pytest.approx(peters_semi_major_decay(start, 1.0, 1.0), rel=0.01)
    assert start / (4.0 * abs(decay)) == pytest.approx(peters_coalescence_time(start, 1.0, 1.0), rel=0.01)
    assert system.r.shape[1] == 3


def test_advance_prepared_applies_the_module_on_a_three_dimensional_step():
    separation = 0.2
    speed = math.sqrt(G * 1.0 / separation)
    bare = InitialSetup('Bare', (
        InitialBody('sun', BodySpec('Sun', 'star', 1.0, 0.00465047), (0., 0., 0.), (0., 0., 0.)),
        InitialBody('rock', BodySpec('Rock', 'planet', 1.0e-6, 1.0e-5), (separation, 0., 0.05), (0., speed, 0.)),
    ), 1.0e-4)
    enabled = InitialSetup('Enabled', bare.bodies, bare.dt, relativistic=RelativisticPolicy(enabled=True))
    plain = build_system(bare)
    corrected = build_system(enabled)
    for _ in range(30):
        advance_prepared(plain)
        advance_prepared(corrected)
    assert plain.system.r.shape[1] == 3
    assert np.max(np.abs(plain.system.r - corrected.system.r)) > 1.0e-12
    assert corrected.setup.relativistic.enabled is True


def test_disabled_relativistic_policy_is_omitted_and_an_enabled_one_round_trips(tmp_path):
    base = get_scenario('sun-earth-jupiter')
    assert setup_digest(replace(base, relativistic=RelativisticPolicy())) == setup_digest(base)
    enabled = replace(base, relativistic=RelativisticPolicy(enabled=True, conservative=True, radiation=False))
    path = tmp_path / 'on.json'
    save_setup(enabled, path)
    assert load_setup(path).relativistic == enabled.relativistic
    document = json.loads(path.read_text())
    assert document['setup']['relativistic']['enabled'] is True
    assert document['setup']['relativistic']['radiation'] is False
    plain = tmp_path / 'off.json'
    save_setup(base, plain)
    assert 'relativistic' not in json.loads(plain.read_text())['setup']
    document['setup']['relativistic']['enabled'] = False
    rejected = tmp_path / 'rejected.json'
    rejected.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='enabled model'):
        load_setup(rejected)
