"""Collision, drag, and tides as separate switches on the formation path."""
from pathlib import Path

import numpy as np
import pytest

from simulations.star_system.core import SOLAR_RADIUS_AU
from simulations.star_system.formation import (
    ROCKY_YEAR_COUNT, FormationExperiment, VelocityChange, load_checkpoint, make_late_stage_setup,
    make_sun_giant_flyby, run_rocky_year, run_sun_giant_flyby, save_checkpoint, with_physics,
)
from simulations.star_system.model import DragPolicy, TidalPolicy, BodySpec, InitialBody, InitialSetup
from simulations.star_system.persistence import save_setup
from simulations.star_system.scenarios import get_scenario

ROOT = Path(__file__).resolve().parents[1]
NUMERICS = ROOT / 'docs' / 'formation-numerics.md'


def _star(position=(0., 0.)):
    return InitialBody('star', BodySpec('Star', 'star', 1., SOLAR_RADIUS_AU), position, tuple(0. for _ in position))


def test_defaults_leave_drag_and_tides_off_and_omit_them_from_the_document(tmp_path):
    setup = make_late_stage_setup()
    assert setup.collision_mode == 'perfect_merge'
    assert setup.drag.enabled is False and setup.tides.enabled is False
    experiment = FormationExperiment(setup, ejection_radius=8.)
    combo = experiment.combination()
    assert combo['drag'] == 'off' and combo['tides'] == 'off'
    assert combo['dimensions'] == 2 and combo['adaptive'] is False
    assert 'gas drag off' in experiment.describe() and 'tides off' in experiment.describe()
    path = tmp_path / 'plain.json'
    save_setup(get_scenario('sun-earth-jupiter'), path)
    document = path.read_text(encoding='utf-8')
    assert '"drag"' not in document
    spatial = with_physics(make_late_stage_setup(count=4, include_unbound=False, inclination=0.1),
                           collision_mode='speed_angle')
    held = FormationExperiment(spatial, ejection_radius=8., adaptive=True)
    assert held.combination()['collision_mode'] == 'speed_angle'
    assert held.combination()['drag'] == 'off' and held.combination()['tides'] == 'off'
    assert held.combination()['dimensions'] == 3 and held.combination()['adaptive'] is True


def test_each_switch_can_be_left_off():
    base = make_late_stage_setup(count=4, include_unbound=False, radii='rocky', inclination=0.1, dt=0.05)
    drag_only = FormationExperiment(
        with_physics(base, drag=DragPolicy(enabled=True, disk='annular')), ejection_radius=8., adaptive=True)
    tides_only = FormationExperiment(
        with_physics(base, tides=TidalPolicy(enabled=True)), ejection_radius=8., adaptive=True)
    assert drag_only.combination()['drag'] == 'annular' and drag_only.combination()['tides'] == 'off'
    assert tides_only.combination()['tides'].startswith('close-pass') and tides_only.combination()['drag'] == 'off'
    drag_only.integrate(0.05, finish=True)
    tides_only.integrate(0.05, finish=True)
    assert drag_only.status == 'complete' and tides_only.status == 'complete'


def test_three_dimensional_adaptive_run_enables_all_three_switches():
    setup = with_physics(
        make_late_stage_setup(count=4, include_unbound=False, radii='rocky', inclination=0.1, dt=0.05),
        collision_mode='speed_angle', drag=DragPolicy(enabled=True, disk='annular'), tides=TidalPolicy(enabled=True))
    experiment = FormationExperiment(setup, ejection_radius=8., adaptive=True, seed=48)
    experiment.integrate(0.05, finish=True)
    combo = experiment.combination()
    assert experiment.status == 'complete'
    assert combo['dimensions'] == 3 and combo['adaptive'] is True
    assert combo['collision_mode'] == 'speed_angle'
    assert combo['drag'] == 'annular' and combo['drag_model'].startswith('epstein-swarm')
    assert combo['tides'].startswith('close-pass')
    text = experiment.describe()
    assert 'speed_angle' in text and 'gas drag on' in text and 'tides on' in text


def test_drag_changes_a_body_inside_the_annulus():
    def once(enabled):
        rock = InitialBody('rock', BodySpec('Rock', 'planet', 2e-6, .05), (1.5, 0., 0.), (0., 3., 0.))
        setup = InitialSetup('Annulus', (_star((0., 0., 0.)), rock), .025, collision_mode='perfect_merge',
                              drag=DragPolicy(enabled=True) if enabled else DragPolicy())
        experiment = FormationExperiment(setup, ejection_radius=20., adaptive=True)
        experiment.advance(1)
        return experiment
    held = once(False)
    dragged = once(True)
    assert dragged.status == 'running' and dragged.dissipated_kinetic > 0
    assert not np.allclose(dragged.prepared.system.v, held.prepared.system.v)


def test_noncontacting_pass_loses_energy_only_when_tides_are_on():
    from simulations.star_system.tides import jupiter_earth_pass
    passage = jupiter_earth_pass(True)
    bodies = (_star((20., 0.)), *passage.bodies)

    def once(enabled, steps):
        setup = InitialSetup('Pass', bodies, passage.dt, collision_mode='perfect_merge',
                              tides=TidalPolicy(enabled=True) if enabled else TidalPolicy())
        experiment = FormationExperiment(setup, ejection_radius=50.)
        taken = 0
        while taken < steps and experiment.status == 'running':
            if enabled and any(event.kind == 'tide' for event in experiment.events):
                break
            experiment.advance(1)
            taken += 1
        return experiment, taken
    on, steps = once(True, 400)
    off, _taken = once(False, steps)
    assert any(event.kind == 'tide' for event in on.events)
    assert not any(event.kind == 'tide' for event in off.events)
    assert not any(event.kind == 'merge' for event in on.events)
    assert on.status == 'running' and off.status == 'running'
    earth = on.prepared.body_ids.index('earth')
    assert not np.allclose(on.prepared.system.v[earth], off.prepared.system.v[earth])


def test_a_merge_does_not_stop_the_run_and_stop_on_contact_does():
    def pair(mode):
        star = _star((40., 0.))
        left = InitialBody('left', BodySpec('Left', 'planet', 1e-4, .1), (-1., 0.), (8., 0.))
        right = InitialBody('right', BodySpec('Right', 'planet', 2e-4, .1), (1., 0.), (0., 0.))
        setup = InitialSetup('Pair', (star, left, right), .05, collision_mode=mode)
        return FormationExperiment(setup, ejection_radius=80.)
    merged = pair('perfect_merge')
    merged.integrate(.3, finish=True)
    assert merged.status == 'complete'
    assert any(event.kind == 'merge' for event in merged.events)
    assert merged.time >= .3
    stopped = pair('stop_on_contact')
    stopped.integrate(.3, finish=True)
    assert stopped.status == 'stopped'
    assert stopped.time < .3
    assert any(event.kind == 'stop' for event in stopped.events)
    assert not any(event.kind == 'merge' for event in stopped.events)
    assert 'left' in stopped.composition and 'right' in stopped.composition
    ignored = pair('ignore')
    ignored.integrate(.3, finish=True)
    assert ignored.status == 'complete'
    assert ignored.time >= .3
    assert any(event.kind == 'ignored' for event in ignored.events)
    assert 'left' in ignored.composition and 'right' in ignored.composition
    assert not any(event.kind == 'merge' for event in ignored.events)


def test_stellar_contact_on_a_gas_bearing_run_finishes_the_requested_time():
    rock = InitialBody('rock', BodySpec('Rock', 'planet', 1e-6, .02), (.15, 0.), (-6., 0.))
    setup = InitialSetup('Starward gas', (_star(), rock), .002, collision_mode='perfect_merge',
                          drag=DragPolicy(enabled=True, disk='annular'))
    experiment = FormationExperiment(setup, ejection_radius=20.)
    experiment.integrate(.05, finish=True)
    assert experiment.status == 'complete'
    assert experiment.time == pytest.approx(.05)
    assert experiment.combination()['drag'] == 'annular'
    logged = [event for event in experiment.events if event.kind == 'unmodeled']
    assert logged and logged[0].parent_ids == ('rock', 'star')
    assert 'star' in experiment.composition and 'rock' in experiment.composition


def test_checkpoint_and_branch_keep_drag_tides_and_parcels(tmp_path):
    rock = InitialBody('rock', BodySpec('Rock', 'planet', 2e-6, .05), (1.5, 0.1, 0.02), (0.2, 4.5, 0.05))
    setup = InitialSetup(
        'Combined ledger', (_star((0., 0., 0.)), rock), .025, collision_mode='perfect_merge',
        drag=DragPolicy(enabled=True, disk='annular'), tides=TidalPolicy(enabled=True))
    experiment = FormationExperiment(setup, ejection_radius=20., adaptive=True, seed=7)
    experiment.integrate(.05)
    path = tmp_path / 'combined.json'
    saved = save_checkpoint(experiment, path)
    restored = load_checkpoint(path)
    assert restored.checkpoint_id == saved
    assert restored.combination() == experiment.combination()
    assert restored.drag_policy.enabled and restored.prepared.setup.tides.enabled
    assert len(restored.zones) == len(experiment.zones)
    experiment.integrate(.05)
    restored.integrate(.05)
    np.testing.assert_allclose(restored.prepared.system.r, experiment.prepared.system.r, rtol=0, atol=0)
    np.testing.assert_allclose(restored.prepared.system.v, experiment.prepared.system.v, rtol=0, atol=0)
    assert restored.composition['rock'] == experiment.composition['rock']
    source = load_checkpoint(path)
    child = source.branch(VelocityChange('rock', (0., 0.2, 0.), 'A disclosed nudge.'))
    assert child.provenance['parent_checkpoint_id'] == saved
    fresh = load_checkpoint(path)
    np.testing.assert_array_equal(source.prepared.system.v, fresh.prepared.system.v)
    assert child.prepared.system.v[1, 1] == pytest.approx(fresh.prepared.system.v[1, 1] + 0.2)


def test_sun_giant_flyby_discloses_the_capture_under_each_switch():
    report = run_sun_giant_flyby()
    text = NUMERICS.read_text(encoding='utf-8')
    assert report.conclusion in text
    lowered = report.conclusion.lower()
    assert 'permanent' in lowered
    assert any(word in lowered for word in ('unbound', 'temporary', 'longer-lived'))
    for summary in (report.free, report.drag, report.tides, report.combined):
        assert summary.collision_mode == 'perfect_merge'
        assert summary.status == 'complete'
        assert summary.simulated_years >= 2.0
    assert report.free.drag == 'off' and report.free.tides == 'off'
    assert report.drag.drag == 'giant' and report.drag.tides == 'off'
    assert report.tides.drag == 'off' and report.tides.tides.startswith('close-pass')
    assert report.combined.drag == 'giant' and report.combined.tides.startswith('close-pass')
    flyby = make_sun_giant_flyby(drag=False, tides=False)
    assert all(len(body.r) == 3 and len(body.v) == 3 for body in flyby.bodies)


def test_rocky_year_completes_at_physical_radii():
    report = run_rocky_year()
    text = NUMERICS.read_text(encoding='utf-8')
    assert report.status == 'complete'
    assert report.body_count == ROCKY_YEAR_COUNT + 1
    assert report.embryo_count == ROCKY_YEAR_COUNT
    assert report.requested_years == 1.0
    assert report.simulated_years >= 1.0
    assert report.simulated_years < 1.0 + 0.025 + 1e-9
    assert report.minimum_step <= report.maximum_step
    assert report.maximum_step <= 0.025 + 1e-15
    assert report.collision_mode == 'perfect_merge'
    assert report.drag == 'annular'
    assert report.tides.startswith('close-pass')
    assert report.calculation_seconds > 0
    # The last stored digit can move by one unit between hosts. Both measured
    # clocks agree through these digits, and a real trajectory change would
    # move the clock by a whole macro step.
    assert '1.005128483931' in text
    assert abs(report.simulated_years - 1.0051284839311654) < 1e-14
    assert 'perfect_merge' in text and 'annular' in text
