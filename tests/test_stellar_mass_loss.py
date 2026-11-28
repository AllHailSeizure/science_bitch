"""Opt-in isotropic stellar wind. Stellar mass stays constant while it is off."""
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from simulations.star_system.collisions import advance_prepared
from simulations.star_system.core import G, SOLAR_RADIUS_AU
from simulations.star_system.data import CATALOG
from simulations.star_system.formation import FormationExperiment, load_checkpoint, save_checkpoint
from simulations.star_system.initialization import build_system
from simulations.star_system.model import (
    MASS_LOSS_MODEL_ID, BodySpec, InitialBody, InitialSetup, RelativisticPolicy, StellarMassLossPolicy,
)
from simulations.star_system.persistence import load_setup, save_setup, setup_digest
from simulations.star_system.scenarios import get_scenario
from simulations.star_system.recording import Calculation, CalculationRequest
from simulations.star_system.stellar_merger import analytic_contact_time, binary_star_merger_setup


def _star_planet(rate=None):
    separation = 1.0
    speed = math.sqrt(G * 1.0 / separation)
    policy = StellarMassLossPolicy() if rate is None else StellarMassLossPolicy(enabled=True, rate_per_year=rate)
    return InitialSetup('Wind', (
        InitialBody('sun', BodySpec('Sun', 'star', 1.0, SOLAR_RADIUS_AU), (0., 0., 0.), (0., 0., 0.)),
        InitialBody('rock', BodySpec('Rock', 'planet', 1.0e-8, 1.0e-5), (separation, 0., 0.), (0., speed, 0.)),
    ), 0.01, stellar_mass_loss=policy)


def _run(setup, steps):
    prepared = build_system(setup)
    for _ in range(steps):
        prepared = advance_prepared(prepared).prepared
    return prepared


def test_mass_and_separation_stay_put_when_the_module_is_off():
    start = build_system(_star_planet())
    finished = _run(_star_planet(), 40)
    assert finished.system.masses[0] == pytest.approx(1.0)
    assert finished.body_specs[0].mass == pytest.approx(1.0)
    assert np.linalg.norm(finished.system.r[1] - finished.system.r[0]) == pytest.approx(
        np.linalg.norm(start.system.r[1] - start.system.r[0]), rel=1.0e-4)
    assert len(finished.body_ids) == 2


def test_isotropic_wind_expands_a_slow_circular_orbit():
    rate = 0.01
    steps = 1000
    years = steps * 0.01
    finished = _run(_star_planet(rate), steps)
    expected_mass = math.exp(-rate * years)
    assert finished.system.masses[0] == pytest.approx(expected_mass, rel=1.0e-12)
    assert finished.body_specs[0].mass == pytest.approx(expected_mass, rel=1.0e-12)
    assert finished.system.masses[1] == pytest.approx(1.0e-8)
    # Adiabatic Jeans result for a light companion: a M stays constant.
    separation = float(np.linalg.norm(finished.system.r[1] - finished.system.r[0]))
    assert separation == pytest.approx(1.0 / expected_mass, rel=0.02)
    assert len(finished.body_ids) == 2


def test_merger_remnant_keeps_the_sum_until_mass_loss_is_on():
    setup = binary_star_merger_setup()
    calculation = Calculation(setup, CalculationRequest(analytic_contact_time(setup) * 1.2, 20))
    while not calculation.done:
        calculation.advance(2000)
    remnant = calculation.prepared
    assert remnant.system.masses[0] == pytest.approx(2.0 * CATALOG['sun'].mass)
    assert 'mass loss' in remnant.events[-1].message
    assert 'omitted' in remnant.events[-1].message
    held = advance_prepared(remnant).prepared
    assert held.system.masses[0] == pytest.approx(2.0 * CATALOG['sun'].mass)
    enabled = replace(remnant.setup, stellar_mass_loss=StellarMassLossPolicy(enabled=True, rate_per_year=0.2))
    losing = replace(remnant, setup=enabled)
    stepped = advance_prepared(losing).prepared
    elapsed = stepped.system.time - remnant.system.time
    assert elapsed > 0.0
    assert stepped.system.masses[0] == pytest.approx(2.0 * math.exp(-0.2 * elapsed), rel=1.0e-12)
    assert stepped.body_specs[0].kind == 'star'


def test_disabled_mass_loss_is_omitted_and_an_enabled_rate_round_trips(tmp_path):
    base = get_scenario('sun-earth-jupiter')
    assert setup_digest(replace(base, stellar_mass_loss=StellarMassLossPolicy())) == setup_digest(base)
    enabled = replace(base, stellar_mass_loss=StellarMassLossPolicy(enabled=True, rate_per_year=1.0e-6))
    path = tmp_path / 'on.json'
    save_setup(enabled, path)
    assert load_setup(path).stellar_mass_loss == enabled.stellar_mass_loss
    document = json.loads(path.read_text())
    assert document['setup']['stellar_mass_loss']['rate_per_year'] == pytest.approx(1.0e-6)
    plain = tmp_path / 'off.json'
    save_setup(base, plain)
    assert 'stellar_mass_loss' not in json.loads(plain.read_text())['setup']
    document['setup']['stellar_mass_loss']['enabled'] = False
    rejected = tmp_path / 'rejected.json'
    rejected.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='enabled model'):
        load_setup(rejected)


def test_three_dimensional_formation_path_records_both_modules(tmp_path):
    separation = 1.0
    speed = math.sqrt(G / separation)
    setup = InitialSetup('Spatial', (
        InitialBody('star', BodySpec('Star', 'star', 1.0, SOLAR_RADIUS_AU), (0., 0., 0.), (0., 0., 0.)),
        InitialBody('planet', BodySpec('Planet', 'planet', 1.0e-6, 1.0e-4), (separation, 0., 0.1), (0., speed, 0.)),
    ), 0.01, collision_mode='perfect_merge',
        relativistic=RelativisticPolicy(enabled=True),
        stellar_mass_loss=StellarMassLossPolicy(enabled=True, rate_per_year=1.0e-3))
    experiment = FormationExperiment(setup, ejection_radius=30.)
    experiment.advance(4)
    assert experiment.prepared.system.r.shape[1] == 3
    assert experiment.combination()['relativistic_corrections'].startswith('pairwise-1pn')
    assert experiment.combination()['stellar_mass_loss'] == MASS_LOSS_MODEL_ID
    assert experiment.prepared.system.masses[0] < 1.0
    assert 'relativistic corrections on' in experiment.describe()
    assert 'stellar mass loss on' in experiment.describe()
    path = tmp_path / 'checkpoint.json'
    save_checkpoint(experiment, path)
    loaded = load_checkpoint(path)
    assert loaded.prepared.setup.relativistic.enabled is True
    assert loaded.prepared.setup.stellar_mass_loss.rate_per_year == pytest.approx(1.0e-3)
    assert loaded.prepared.system.masses[0] == pytest.approx(float(experiment.prepared.system.masses[0]))
