"""Close-pass tidal dissipation stays off unless an experiment selects it."""
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from simulations.star_system.collisions import advance_prepared
from simulations.star_system.core import G
from simulations.star_system.data import CATALOG
from simulations.star_system.experiments import capture_current_setup
from simulations.star_system.formation import specification
from simulations.star_system.initialization import build_system
from simulations.star_system.model import TIDAL_MODEL_ID, BodySpec, InitialBody, InitialSetup, TidalPolicy
from simulations.star_system.persistence import load_setup, save_setup, setup_digest
from simulations.star_system.scenarios import compare_recordings, get_scenario, make_jupiter_comparison
from simulations.star_system.tides import (
    INTERACTION, MODEL_DISCLOSURE, TideEvent, compare_jupiter_earth_passes, contact_separation,
    dissipate_close_passes, jupiter_earth_pass, pair_energy_loss,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def comparison():
    return compare_jupiter_earth_passes()


def _pair(radius=0.05, mass=1e-3, other=2e-4, separation=(0.12, 0.03), speed=(0.0, -0.5), dt=1e-3, tides=None):
    policy = TidalPolicy() if tides is None else tides
    return InitialSetup('Kick', (
        InitialBody('a', BodySpec('A', 'planet', mass, radius), (0., 0.), (0., 0.)),
        InitialBody('b', BodySpec('B', 'planet', other, radius), separation, speed),
    ), dt, tides=policy)


def _crossed(setup):
    prepared = build_system(setup)
    start_r = prepared.system.r.copy()
    start_v = prepared.system.v.copy()
    prepared.system.r[1] = np.array([setup.bodies[1].r[0], -setup.bodies[1].r[1]])
    prepared.system.v[1] = np.array(setup.bodies[1].v)
    return prepared, start_r, start_v


def test_disabled_tides_match_the_yoshida_step():
    setup = get_scenario('sun-earth-jupiter')
    stepped = build_system(setup)
    advanced = build_system(setup)
    for _ in range(8):
        advanced = advance_prepared(advanced).prepared
        stepped.system.step()
    np.testing.assert_array_equal(advanced.system.r, stepped.system.r)
    np.testing.assert_array_equal(advanced.system.v, stepped.system.v)
    assert advanced.system.time == stepped.system.time
    assert not advanced.events


def test_tide_policy_rejects_a_hidden_or_unknown_choice():
    with pytest.raises(ValueError, match='strength or reach'):
        TidalPolicy(enabled=False, strength=2)
    with pytest.raises(ValueError, match='tidal model'):
        TidalPolicy(enabled=True, version='other')
    with pytest.raises(ValueError, match='Tidal strength'):
        TidalPolicy(enabled=True, strength=0)
    with pytest.raises(ValueError, match='TidalPolicy'):
        InitialSetup('Bad', (InitialBody('a', CATALOG['earth'], (0., 0.), (0., 0.)),), 1e-4, tides=None)


def test_pair_energy_loss_follows_the_disclosed_formula_and_floors_overlap():
    expected = 80 * G * ((1e-3**2 + 2e-4**2) * 0.05**5) / 0.12**6
    assert pair_energy_loss(2e-4, 0.05, 1e-3, 0.05, 0.12, 80) == pytest.approx(expected)
    assert pair_energy_loss(1, 0.2, 1, 0.2, 0.01, 1) == pair_energy_loss(1, 0.2, 1, 0.2, 0.2, 1)
    assert pair_energy_loss(0, 0.2, 1, 0.2, 0.1, 1) == 0
    assert pair_energy_loss(1, None, 1, None, 0.1, 1) == 0


def test_close_pass_removes_only_the_available_relative_energy_and_keeps_momentum():
    prepared, start_r, start_v = _crossed(_pair(tides=TidalPolicy(enabled=True, strength=0.01)))
    before = prepared.system.invariants()
    updated = dissipate_close_passes(prepared, start_r, start_v)
    after = updated.system.invariants()
    event = updated.events[-1]
    assert isinstance(event, TideEvent) and event.kind == 'close-pass-dissipation'
    assert event.body_ids == ('a', 'b')
    assert event.energy_lost == pytest.approx(
        pair_energy_loss(1e-3, 0.05, 2e-4, 0.05, event.pericenter, 0.01))
    assert before[0] - after[0] == pytest.approx(event.energy_lost, rel=1e-12)
    np.testing.assert_allclose(after[1], before[1], rtol=1e-12, atol=1e-18)
    np.testing.assert_array_equal(updated.system.r, prepared.system.r)
    assert np.linalg.norm(updated.system.v[0] - updated.system.v[1]) > 0

    clamped, start_r, start_v = _crossed(_pair(tides=TidalPolicy(enabled=True)))
    momentum = clamped.system.invariants()[1].copy()
    kinetic = 0.5 * (1e-3 * 2e-4 / 1.2e-3) * 0.5**2
    updated = dissipate_close_passes(clamped, start_r, start_v)
    assert updated.events[-1].energy_lost == pytest.approx(kinetic, rel=1e-12)
    assert np.linalg.norm(updated.system.v[0] - updated.system.v[1]) == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(updated.system.invariants()[1], momentum, rtol=1e-12, atol=1e-18)


def test_wide_massless_and_radiiless_passes_do_not_dissipate():
    wide, start_r, start_v = _crossed(_pair(separation=(5., 0.2), speed=(0., -0.2), dt=0.01,
                                            tides=TidalPolicy(enabled=True)))
    velocities = wide.system.v.copy()
    updated = dissipate_close_passes(wide, start_r, start_v)
    assert updated.events == ()
    np.testing.assert_array_equal(updated.system.v, velocities)

    barren = InitialSetup('Barren', (
        InitialBody('a', BodySpec('A', 'planet', 1e-3, None), (0., 0.), (0., 0.)),
        InitialBody('b', BodySpec('B', 'planet', 2e-4, None), (0.12, 0.03), (0., -0.5)),
    ), 1e-3, tides=TidalPolicy(enabled=True))
    prepared, start_r, start_v = _crossed(barren)
    assert dissipate_close_passes(prepared, start_r, start_v).events == ()

    particle = InitialSetup('Tracer', (
        InitialBody('a', BodySpec('A', 'planet', 1e-3, 0.05), (0., 0.), (0., 0.)),
        InitialBody('b', BodySpec('B', 'particle', 0., None), (0.12, 0.03), (0., -0.5)),
    ), 1e-3, tides=TidalPolicy(enabled=True))
    prepared, start_r, start_v = _crossed(particle)
    assert dissipate_close_passes(prepared, start_r, start_v).events == ()

    off, start_r, start_v = _crossed(_pair())
    assert dissipate_close_passes(off, start_r, start_v) is off


def test_unresolved_encounter_still_halts_when_tides_are_on():
    with pytest.raises(ValueError, match='Close encounter'):
        build_system(replace(jupiter_earth_pass(True), dt=1.0))


def test_enabled_tides_leave_a_wide_orbit_unchanged():
    base = get_scenario('sun-earth-jupiter')
    on = replace(base, tides=TidalPolicy(enabled=True))
    plain, tidal = build_system(base), build_system(on)
    for _ in range(3):
        plain = advance_prepared(plain).prepared
        tidal = advance_prepared(tidal).prepared
    np.testing.assert_array_equal(tidal.system.r, plain.system.r)
    np.testing.assert_array_equal(tidal.system.v, plain.system.v)
    assert not any(isinstance(event, TideEvent) for event in tidal.events)


def test_same_jupiter_earth_vectors_differ_only_by_the_stored_policy(tmp_path):
    off = jupiter_earth_pass(False)
    on = jupiter_earth_pass(True)
    assert off.tides.enabled is False and on.tides == TidalPolicy(enabled=True)
    assert [(body.r, body.v) for body in off.bodies] == [(body.r, body.v) for body in on.bodies]
    assert off.dt == on.dt
    assert setup_digest(off) != setup_digest(on)
    path = tmp_path / 'pass.json'
    save_setup(on, path)
    loaded = load_setup(path)
    assert loaded.tides == on.tides
    assert setup_digest(loaded) == setup_digest(on)
    document = json.loads(path.read_text())
    assert document['setup']['tides']['version'] == TIDAL_MODEL_ID
    assert document['setup']['tides']['enabled'] is True
    plain = tmp_path / 'plain.json'
    save_setup(get_scenario('sun-earth-jupiter'), plain)
    assert 'tides' not in json.loads(plain.read_text())['setup']
    document['setup']['tides']['enabled'] = False
    rejected = tmp_path / 'rejected.json'
    rejected.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='enabled model'):
        load_setup(rejected)
    captured = capture_current_setup(build_system(on), on, 'Captured close pass')
    assert captured.tides == on.tides


def test_contact_solver_ignores_a_stored_tide_policy():
    def make(enabled):
        return InitialSetup('Contact', (
            InitialBody('a', BodySpec('A', 'planet', 1e-4, .1), (-1., 0.), (10., 0.)),
            InitialBody('b', BodySpec('B', 'planet', 2e-4, .1), (1., 0.), (0., 0.)),
        ), .2, collision_mode='perfect_merge', tides=TidalPolicy(enabled=True) if enabled else TidalPolicy())
    left = advance_prepared(build_system(make(False))).prepared
    right = advance_prepared(build_system(make(True))).prepared
    assert len(left.body_ids) == 1 and left.body_ids == right.body_ids
    np.testing.assert_array_equal(left.system.r, right.system.r)
    np.testing.assert_array_equal(left.system.v, right.system.v)
    assert left.system.time == right.system.time
    assert not any(isinstance(event, TideEvent) for event in right.events)


def test_flyby_without_tides_stays_unbound(comparison):
    assert comparison.flyby_off.specific_energy > 0
    assert comparison.flyby_off.eccentricity > 1
    assert comparison.flyby_off.pericenters == 1
    assert comparison.flyby_off.tide_events == 0
    assert comparison.flyby_off.energy_lost == 0


def test_flyby_with_tides_returns_for_a_second_pericenter(comparison):
    assert comparison.flyby_on.steps == comparison.flyby_off.steps
    assert comparison.flyby_on.time == comparison.flyby_off.time
    assert comparison.flyby_on.specific_energy < 0
    assert comparison.flyby_on.pericenters >= 2
    assert comparison.flyby_on.tide_events >= 1
    assert comparison.flyby_on.energy_lost > 0
    assert comparison.flyby_on.min_separation > contact_separation()
    assert comparison.flyby_off.min_separation > contact_separation()


def test_eccentric_pass_circularizes_only_with_tides(comparison):
    assert comparison.eccentric_on.steps == comparison.eccentric_off.steps
    assert comparison.eccentric_on.pericenters == comparison.eccentric_off.pericenters == 1
    assert comparison.eccentric_off.eccentricity == pytest.approx(0.65, abs=1e-4)
    assert comparison.eccentric_on.eccentricity < comparison.eccentric_off.eccentricity - 0.05
    assert comparison.eccentric_on.specific_energy < comparison.eccentric_off.specific_energy
    assert comparison.eccentric_on.tide_events == 1
    assert comparison.eccentric_off.tide_events == 0
    assert comparison.eccentric_on.min_separation > contact_separation()


def test_disclosure_limits_the_claim_to_the_selected_model(comparison):
    text = comparison.disclosure
    assert MODEL_DISCLOSURE in text
    assert INTERACTION in text
    assert 'unbound' in text and 'bound' in text
    assert 'only when that policy is on' in text
    assert '#69' in text and '#70' in text and '#71' in text
    readme = (ROOT / 'simulations/star_system/README.md').read_text(encoding='utf-8')
    for phrase in (TIDAL_MODEL_ID, 'off unless', '#69', '#70', '#71',
                   'not the dominant capture path', 'not gas drag', 'does not change collision rules'):
        assert phrase in readme
    note = ' '.join(specification().not_modeled)
    assert 'close-pass' in note and 'stays off' in note


def test_matched_jupiter_control_rejects_a_different_tide_policy():
    from simulations.star_system.recording import Calculation, CalculationRequest
    treatment, control = make_jupiter_comparison(get_scenario('sun-jupiter-ring'))
    treatment = replace(treatment, tides=TidalPolicy(enabled=True))

    def once(setup):
        calc = Calculation(setup, CalculationRequest(setup.dt))
        calc.advance(1)
        return calc.result

    with pytest.raises(ValueError, match='model settings'):
        compare_recordings(once(treatment), once(control))


def test_legacy_controller_stores_the_tide_choice():
    import matplotlib.pyplot as plt
    from simulations.star_system.plotting import App
    app = App(setup=jupiter_earth_pass(False))
    try:
        app.set_tides(True)
        assert app.initial_setup.tides.enabled is True
        assert TIDAL_MODEL_ID in app.status.get_text().replace('\n', '')
        app.set_tides(False)
        assert app.initial_setup.tides.enabled is False
        with pytest.raises(ValueError, match='on or off'):
            app.set_tides(1)
    finally:
        plt.close('all')


@pytest.fixture(scope='module')
def qtapp():
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qtapp):
    from simulations.star_system.desktop import Observatory
    instance = Observatory()
    instance.timer.stop()
    instance.show()
    qtapp.processEvents()
    yield instance
    instance.dirty = False
    instance.close()
    qtapp.processEvents()


def test_desktop_checkbox_stores_tides_with_the_experiment(window, tmp_path):
    window.install_experiment(get_scenario('sun-earth-jupiter'))
    assert window.tides_check.isChecked() is False
    assert 'Tidal dissipation is off' in window.collision_note.text()
    window.tides_check.setChecked(True)
    window.collision_button.click()
    assert window.setup.tides.enabled is True
    assert window.setup.collision_mode == 'none'
    assert TIDAL_MODEL_ID in window.collision_note.text()
    assert TIDAL_MODEL_ID in window.scene.heading.get_text()
    path = tmp_path / 'tides.json'
    window.save_to(path)
    assert load_setup(path).tides == window.setup.tides
    window.tides_check.setChecked(False)
    window.collision_button.click()
    assert window.setup.tides.enabled is False
    window.undo_button.click()
    assert window.setup.tides.enabled is True
