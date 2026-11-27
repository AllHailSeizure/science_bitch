"""Gas-and-dust disk: fluid, drag, growth, swarm-to-body, and envelope capture."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from simulations.star_system.core import G, SOLAR_RADIUS_AU
from simulations.star_system.formation import VelocityChange
from simulations.star_system.gas_dust import (
    CAPTURE_RADIUS, DISK_DURATION_YEARS, DISK_MODEL, DISK_SEED, DISK_SWARM_COUNT, DISK_ZONE_COUNT,
    GRAIN_DENSITY, GRAIN_RADIUS_MIN, MILLIMETER, READY_INTERNAL_SPEED, READY_MASS, READY_SWARM_ID,
    SWARM_RULE_ID, Body, GasDustExperiment, Swarm, continue_as_late_stage, evaluate_swarm_rule,
    load_checkpoint, make_capture_setup, make_disk_setup, make_uniform_disk, run_capture_comparison,
    run_disk_experiment, save_checkpoint, setup_digest, specification, sweep_up_increment,
)


def _momentum(items):
    total = np.zeros(2)
    for item in items:
        total += item.mass*np.asarray(item.v, dtype=float)
    return total


def test_specification_names_the_fluid_drag_growth_rule_and_capture_limits():
    spec = specification()
    assert 'annular' in spec.starting_distribution.lower()
    assert 'swarm' in spec.representation.lower()
    assert 'isothermal' in spec.fluid.lower()
    assert 'epstein' in spec.drag.lower()
    assert 'sweep-up' in spec.growth.lower()
    assert SWARM_RULE_ID in spec.swarm_rule
    assert 'perfect merge' in spec.collision_model.lower() or 'perfect-merge' in spec.collision_model.lower()
    text = ' '.join(spec.capture).lower()
    assert 'envelope' in text and '#69' in text and '#70' in text and '#79' in text
    assert 'permanent' in text
    described = GasDustExperiment(make_disk_setup(swarm_count=0, include_ready=False)).describe()
    assert DISK_MODEL in described
    assert 'donor-cell' in described.lower() or 'annular' in described.lower()


def test_disk_setup_repeats_for_the_same_seed():
    first = make_disk_setup()
    second = make_disk_setup()
    other = make_disk_setup(seed=72)
    assert setup_digest(first) == setup_digest(second)
    assert setup_digest(first) != setup_digest(other)
    assert first.seed == DISK_SEED
    assert len(first.zones) == DISK_ZONE_COUNT
    assert len(first.swarms) == DISK_SWARM_COUNT+1
    with pytest.raises(ValueError, match='gap|meet'):
        broken = replace(first, zones=(first.zones[0], replace(first.zones[1], r_inner=first.zones[1].r_inner+.05)))
        GasDustExperiment(broken)


def test_equilibrium_disk_keeps_mass_and_stays_quiet():
    experiment = GasDustExperiment(make_uniform_disk())
    initial_mass = sum(zone.mass for zone in experiment.zones)
    initial_angular = sum(zone.angular_momentum for zone in experiment.zones)
    experiment.integrate(.2, finish=True)
    assert experiment.status == 'complete'
    assert sum(zone.mass for zone in experiment.zones) == pytest.approx(initial_mass, rel=0, abs=1e-15)
    assert sum(zone.angular_momentum for zone in experiment.zones) == pytest.approx(initial_angular, rel=0, abs=1e-12)
    assert max(abs(zone.v_r) for zone in experiment.zones) < 1e-12


def test_radial_flow_moves_zone_mass_and_keeps_the_total():
    setup = make_uniform_disk()
    setup.zones[0].radial_momentum = setup.zones[0].mass*.25
    before = [zone.mass for zone in setup.zones]
    experiment = GasDustExperiment(setup)
    experiment.advance(1)
    assert experiment.status == 'running'
    assert experiment.zones[0].mass < before[0]
    assert experiment.zones[1].mass > before[1]
    assert sum(zone.mass for zone in experiment.zones) == pytest.approx(sum(before), rel=0, abs=0)


def test_drag_moves_swarm_angular_momentum_onto_the_gas():
    experiment = GasDustExperiment(make_disk_setup(swarm_count=1, include_ready=False))
    swarm = experiment.swarms[0]
    radius = float(np.linalg.norm(swarm.r))
    before = swarm.mass*(swarm.r[0]*swarm.v[1]-swarm.r[1]*swarm.v[0])
    experiment.advance(1)
    swarm = experiment.swarms[0]
    after = swarm.mass*(swarm.r[0]*swarm.v[1]-swarm.r[1]*swarm.v[0])
    assert after < before
    books = experiment.accounting()
    assert books.drag_momentum_error == pytest.approx(0., abs=1e-12)
    assert books.angular_momentum_error == pytest.approx(0., abs=1e-9)
    assert books.dissipated_kinetic > 0
    assert books.promotion_count == 0
    assert books.gas_mass == pytest.approx(books.initial_gas_mass, rel=0, abs=0)
    assert radius > 0


def test_envelope_drag_slows_a_fast_body():
    base = make_capture_setup(drag=True)
    source = base.bodies[0]
    body = Body(source.id, source.name, source.mass, source.radius, source.grain_radius, source.internal_speed,
                source.bulk_density, np.array([.45, 0.]), np.array([6., 0.]), dict(source.contributions))
    experiment = GasDustExperiment(replace(base, dt=1e-4, bodies=(body,)))
    experiment.advance(1)
    assert experiment.bodies[0].v[0] < 6.
    assert experiment.dissipated_kinetic > 0
    assert experiment.drag_momentum_error == pytest.approx(0., abs=1e-12)
    assert experiment.accounting().gas_mass == pytest.approx(experiment.accounting().initial_gas_mass, abs=0)


def test_grain_growth_increases_radius_and_keeps_mass():
    setup = make_uniform_disk(inner=.8, outer=1.6, zones=4, sigma=2e-4, sound_speed=.3, dt=.01, drag_enabled=True)
    position = np.array([1.2, 0.])
    speed = math.sqrt(G*setup.star_mass/1.2)
    swarm = Swarm('pebble', 'Pebble', 1e-10, MILLIMETER, 1., 1e-3, position,
                  np.array([0., speed]), {'pebble': 1e-10})
    experiment = GasDustExperiment(replace(setup, swarms=(swarm,)))
    zone = experiment.zone_at(1.2)
    omega = math.sqrt(G*setup.star_mass/zone.radius**3)
    scale_height = setup.parameters.sound_speed/omega
    dust = swarm.mass/(zone.area*math.sqrt(2*math.pi)*scale_height)
    expected = MILLIMETER+sweep_up_increment(dust, 1., GRAIN_DENSITY, setup.dt)
    experiment.advance(1)
    grown = experiment.swarms[0]
    assert grown.mass == pytest.approx(1e-10, rel=0, abs=0)
    assert grown.grain_radius == pytest.approx(expected, rel=1e-12, abs=0)
    assert grown.contributions == {'pebble': 1e-10}


def test_overlapping_swarms_coagulate_without_dropping_parcels():
    setup = make_uniform_disk(drag_enabled=False, dt=1e-4)
    left = Swarm('left', 'Left', 1e-6, MILLIMETER, 0., .02, np.array([1.2, 0.]), np.array([0., .01]),
                 {'grain-a': 1e-6})
    right = Swarm('right', 'Right', 2e-6, MILLIMETER, 0., .02, np.array([1.21, 0.]), np.array([0., -.01]),
                  {'grain-b': 2e-6})
    opening_momentum = _momentum((left, right))
    experiment = GasDustExperiment(replace(setup, swarms=(left, right)))
    experiment.advance(1)
    assert len(experiment.swarms) == 1
    child = experiment.swarms[0]
    event = experiment.events[0]
    assert event.kind == 'coagulation'
    assert event.parent_ids == ('left', 'right')
    assert {item.parcel_id: item.mass for item in event.contributions} == {'grain-a': 1e-6, 'grain-b': 2e-6}
    assert child.mass == pytest.approx(3e-6, rel=0, abs=0)
    assert event.momentum == pytest.approx(tuple(opening_momentum), rel=0, abs=1e-15)
    assert event.internal_kinetic == pytest.approx(.5*child.mass*child.internal_speed**2, rel=1e-12, abs=0)
    assert experiment.accounting().parcel_error == pytest.approx(0., abs=1e-15)
    assert experiment.removed_swarms['left'].status == 'coagulated'


def test_swarm_rule_requires_every_condition():
    experiment = GasDustExperiment(make_disk_setup(swarm_count=0, include_ready=True))
    swarm = experiment.swarms[0]
    zone = experiment.zone_at(float(np.linalg.norm(swarm.r)))
    verdict = evaluate_swarm_rule(swarm, zone, experiment.star_mass, experiment.parameters)
    assert verdict.accepted and verdict.failed == ()
    assert evaluate_swarm_rule(swarm, zone, experiment.star_mass,
                               replace(experiment.parameters, stokes_min=1e12)).failed == ('stokes',)
    assert evaluate_swarm_rule(swarm, zone, experiment.star_mass,
                               replace(experiment.parameters, grain_radius_min=swarm.grain_radius*2)).failed == ('grain_radius',)
    assert evaluate_swarm_rule(swarm, zone, experiment.star_mass,
                               replace(experiment.parameters, mass_min=swarm.mass*2)).failed == ('mass',)
    wide = swarm.copy()
    wide.spatial_radius = 1.
    wide.internal_speed = 0.
    assert evaluate_swarm_rule(wide, zone, experiment.star_mass, experiment.parameters).failed == ('hill',)
    hot = swarm.copy()
    hot.internal_speed = 10.
    assert evaluate_swarm_rule(hot, zone, experiment.star_mass, experiment.parameters).failed == ('internal',)


def test_swarm_to_body_conserves_mass_momentum_origin_and_internal_state():
    experiment = GasDustExperiment(make_disk_setup(swarm_count=0, include_ready=True))
    swarm = experiment.swarms[0]
    opening_momentum = swarm.mass*np.asarray(swarm.v, dtype=float)
    opening_grain = swarm.grain_radius
    experiment.advance(1)
    assert experiment.swarms == []
    body = experiment.bodies[0]
    event = experiment.events[0]
    assert event.kind == 'swarm-to-body'
    assert event.parent_ids == (READY_SWARM_ID,)
    assert event.mass == pytest.approx(READY_MASS, rel=0, abs=0)
    assert event.momentum == pytest.approx(tuple(opening_momentum), rel=0, abs=1e-15)
    assert {item.parcel_id: item.mass for item in event.contributions} == {READY_SWARM_ID: READY_MASS}
    assert event.grain_radius == pytest.approx(body.grain_radius, rel=0, abs=0)
    assert event.grain_radius >= opening_grain
    assert event.internal_speed == pytest.approx(READY_INTERNAL_SPEED, rel=0, abs=0)
    assert event.internal_kinetic == pytest.approx(.5*READY_MASS*READY_INTERNAL_SPEED**2, rel=1e-12, abs=0)
    assert body.mass == pytest.approx(READY_MASS, rel=0, abs=0)
    assert body.bulk_density*4/3*math.pi*body.radius**3 == pytest.approx(body.mass, rel=1e-12, abs=0)
    assert body.radius > 0
    assert experiment.removed_swarms[READY_SWARM_ID].status == 'became-body'
    assert experiment.accounting().transition_momentum_error == pytest.approx(0., abs=1e-15)
    assert experiment.accounting().parcel_error == pytest.approx(0., abs=1e-15)
    story = experiment.replay_parcel(READY_SWARM_ID)
    assert story.holder_ids == (body.id,)
    assert story.mass == pytest.approx(READY_MASS, rel=0, abs=0)
    assert any(item.kind == 'swarm-to-body' for item in story.events)
    assert GRAIN_RADIUS_MIN <= opening_grain


def test_body_contact_uses_perfect_merge_and_keeps_parcels():
    setup = make_uniform_disk(drag_enabled=False, dt=1e-4)
    left = Body('left', 'Left', 1e-5, .05, MILLIMETER, 0., GRAIN_DENSITY, np.array([1.2, 0.]), np.array([0., .2]),
                {'dust-a': 1e-5})
    right = Body('right', 'Right', 2e-5, .05, MILLIMETER, 0., GRAIN_DENSITY, np.array([1.22, 0.]), np.array([0., -.1]),
                 {'dust-b': 2e-5})
    experiment = GasDustExperiment(replace(setup, bodies=(left, right)))
    experiment.advance(1)
    assert len(experiment.bodies) == 1
    remnant = experiment.bodies[0]
    event = next(item for item in experiment.events if item.kind == 'merge')
    pre = next(sample for sample in experiment.samples if sample.phase == 'pre')
    post = next(sample for sample in experiment.samples if sample.phase == 'post')
    assert {item.parcel_id: item.mass for item in event.contributions} == {'dust-a': 1e-5, 'dust-b': 2e-5}
    assert remnant.mass == pytest.approx(3e-5, rel=1e-12, abs=0)
    assert _momentum(post.bodies) == pytest.approx(_momentum(pre.bodies), rel=0, abs=1e-12)
    assert remnant.radius == pytest.approx(.05*2**(1/3), rel=1e-12, abs=0)
    assert event.lost_relative_kinetic > 0
    assert experiment.accounting().parcel_error == pytest.approx(0., abs=1e-15)
    assert experiment.accounting().transition_momentum_error == pytest.approx(0., abs=1e-12)


def test_checkpoint_roundtrip_resumes_the_disk(tmp_path):
    experiment = GasDustExperiment(make_disk_setup())
    experiment.integrate(.04)
    path = tmp_path/'disk.json'
    saved = save_checkpoint(experiment, path)
    restored = load_checkpoint(path)
    assert restored.checkpoint_id == saved == experiment.checkpoint_id
    experiment.integrate(.1)
    restored.integrate(.1)
    assert restored.time == experiment.time
    np.testing.assert_allclose([zone.mass for zone in restored.zones], [zone.mass for zone in experiment.zones], rtol=0, atol=0)
    np.testing.assert_allclose([zone.angular_momentum for zone in restored.zones],
                               [zone.angular_momentum for zone in experiment.zones], rtol=0, atol=0)
    assert [(swarm.id, tuple(np.asarray(swarm.r, dtype=float))) for swarm in restored.swarms] == [
        (swarm.id, tuple(np.asarray(swarm.r, dtype=float))) for swarm in experiment.swarms]
    assert [(body.id, tuple(np.asarray(body.r, dtype=float)), tuple(body.contributions)) for body in restored.bodies] == [
        (body.id, tuple(np.asarray(body.r, dtype=float)), tuple(body.contributions)) for body in experiment.bodies]
    assert [(event.kind, event.parent_ids, tuple((item.parcel_id, item.mass) for item in event.contributions))
            for event in restored.events] == [
                (event.kind, event.parent_ids, tuple((item.parcel_id, item.mass) for item in event.contributions))
                for event in experiment.events]


def test_branch_changes_one_velocity_and_leaves_the_source(tmp_path):
    experiment = GasDustExperiment(make_disk_setup())
    experiment.integrate(.04)
    saved = save_checkpoint(experiment, tmp_path/'branch.json')
    swarm = next(item for item in experiment.swarms if item.id == 'swarm-000')
    velocity = np.array(swarm.v, copy=True)
    change = VelocityChange('swarm-000', (0., 1.), 'Add a +y velocity to this dust swarm.')
    child = experiment.branch(change)
    np.testing.assert_allclose(swarm.v, velocity, rtol=0, atol=0)
    assert experiment.provenance['change'] is None
    assert child.provenance['parent_checkpoint_id'] == saved
    assert child.provenance['change']['note'] == change.note
    assert child.provenance['setup_digest'] == experiment.provenance['setup_digest']
    assert child.provenance['run_id'] != experiment.provenance['run_id']
    branched = next(item for item in child.swarms if item.id == 'swarm-000')
    assert branched.v[1] == pytest.approx(velocity[1]+1.)
    experiment.integrate(.06)
    child.integrate(.06)
    source = next(item for item in experiment.swarms if item.id == 'swarm-000')
    moved = next(item for item in child.swarms if item.id == 'swarm-000')
    assert not np.allclose(source.r, moved.r)
    with pytest.raises(ValueError, match='not an active'):
        experiment.branch(VelocityChange('missing', (0., 1.), 'Nudge a missing swarm.'))


def test_invalid_checkpoint_is_rejected(tmp_path):
    experiment = GasDustExperiment(make_uniform_disk())
    experiment.advance(1)
    path = tmp_path/'checkpoint.json'
    save_checkpoint(experiment, path)
    document = json.loads(path.read_text())
    document['id'] = '0'*64
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='id'):
        load_checkpoint(path)
    document['schema'] = 'gas-dust-checkpoint-v0'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='schema'):
        load_checkpoint(path)
    path.write_text('{')
    with pytest.raises(ValueError, match='Invalid gas-and-dust checkpoint'):
        load_checkpoint(path)


def test_drag_binds_the_envelope_flyby_and_the_control_stays_unbound():
    report = run_capture_comparison()
    assert report.drag_status == 'complete'
    assert report.free_status == 'complete'
    assert report.free_initial_energy > 0
    assert report.free_specific_energy > 0
    assert report.drag_specific_energy < 0
    assert report.free_specific_energy == pytest.approx(report.free_initial_energy, rel=1e-3, abs=1e-3)
    assert report.drag_dissipated_kinetic > 0
    assert report.free_dissipated_kinetic == 0
    assert report.drag_bound and not report.free_bound
    assert report.drag_minimum_radius > SOLAR_RADIUS_AU+CAPTURE_RADIUS
    assert any('#69' in note and '#70' in note and '#79' in note for note in report.notes)
    drag = make_capture_setup(drag=True)
    free = make_capture_setup(drag=False)
    assert [(zone.mass, zone.angular_momentum) for zone in drag.zones] == [
        (zone.mass, zone.angular_momentum) for zone in free.zones]
    assert drag.bodies[0].r.tolist() == free.bodies[0].r.tolist()
    assert drag.parameters.drag_enabled and not free.parameters.drag_enabled


def test_handoff_keeps_parcels_for_the_late_stage_solver():
    experiment = GasDustExperiment(make_disk_setup(swarm_count=0, include_ready=True))
    experiment.advance(1)
    body = experiment.bodies[0]
    late = continue_as_late_stage(experiment, ejection_radius=20.)
    assert late.composition[body.id] == {READY_SWARM_ID: READY_MASS}
    assert 'grain_radius_au' in late.prepared.body_specs[late.prepared.body_ids.index(body.id)].provenance
    assert 'internal_speed' in late.prepared.body_specs[late.prepared.body_ids.index(body.id)].provenance
    late.integrate(late.prepared.system.dt)
    assert late.prepared.system.masses[late.prepared.body_ids.index(body.id)] == pytest.approx(READY_MASS, rel=0, abs=0)
    assert late.accounting().parcel_error == pytest.approx(0., abs=1e-15)


def test_cfl_violation_halts_without_adapting_the_step():
    experiment = GasDustExperiment(make_uniform_disk(dt=1.))
    masses = [zone.mass for zone in experiment.zones]
    experiment.advance(1)
    assert experiment.status == 'halted'
    assert 'not reduced' in experiment.reason
    assert experiment.time == 0
    assert [zone.mass for zone in experiment.zones] == masses
    assert experiment.dt == 1.


def test_stellar_contact_halts_without_a_stellar_merge():
    setup = make_uniform_disk(dt=1e-3)
    body = Body('rock', 'Rock', 1e-6, .02, MILLIMETER, 0., GRAIN_DENSITY, np.array([.01, 0.]), np.array([0., 0.]),
                {'rock': 1e-6})
    experiment = GasDustExperiment(replace(setup, bodies=(body,)))
    experiment.advance(1)
    assert experiment.status == 'halted'
    assert 'star' in experiment.reason.lower()
    assert experiment.time == 0
    assert [item.id for item in experiment.bodies] == ['rock']
    assert experiment.events == []


def test_disk_experiment_reports_counts_duration_and_cost():
    report = run_disk_experiment()
    assert report.model == DISK_MODEL
    assert report.representation
    assert report.swarm_rule == SWARM_RULE_ID
    assert report.seed == DISK_SEED
    assert report.zone_count == DISK_ZONE_COUNT
    assert report.swarm_count == DISK_SWARM_COUNT+1
    assert report.parcel_count == report.swarm_count
    assert report.body_count == 1
    assert report.requested_years == DISK_DURATION_YEARS
    assert report.simulated_years == pytest.approx(DISK_DURATION_YEARS, rel=0, abs=1e-12)
    assert report.calculation_seconds > 0
    assert report.peak_python_bytes > 0
    assert report.recording_bytes > 0
    assert report.sample_count > 1
    assert report.promotion_count == 1
    assert report.status == 'complete'
    assert report.questions
    assert report.capture.drag_bound and not report.capture.free_bound
    assert 'Python' in report.hardware


def test_gas_dust_note_states_the_model_and_the_open_capture_gaps():
    text = (__import__('pathlib').Path(__file__).resolve().parents[1]/'docs'/'gas-dust-formation.md').read_text(encoding='utf-8')
    lowered = text.lower()
    assert 'annular' in lowered
    assert 'epstein' in lowered
    assert 'swarm' in lowered
    assert 'checkpoint' in lowered
    assert '#69' in text and '#70' in text and '#79' in text
    assert 'permanent' in lowered
