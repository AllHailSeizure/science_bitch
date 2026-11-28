"""Top-level formation behavior: ancestry, checkpoints, branches, and accounting."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from simulations.star_system.formation import (
    BENCHMARK_EMBRYO_COUNT, BENCHMARK_EMBRYO_MASS, BENCHMARK_EMBRYO_RADIUS,
    DEMONSTRATION_EJECTION_RADIUS, FORMATION_MODEL, PHYSICAL_PROBE_COUNT,
    FormationExperiment, SynchronizedPlayback, VelocityChange, compare_resolution,
    load_checkpoint, make_close_pass_setup, make_demonstration_setup,
    make_inclined_circular_setup, make_late_stage_setup, playback, rocky_radius,
    run_benchmark, run_physical_probe, save_checkpoint, specification,
)
from simulations.star_system.frames import frame_transform
from simulations.star_system.history import StateFrame
from simulations.star_system.model import SPEED_ANGLE_VERSION, BodySpec, InitialBody, InitialSetup, with_collision_rule
from simulations.star_system.persistence import (
    MODEL_VERSION, SPATIAL_MODEL_VERSION, load_setup, save_setup, setup_digest,
)
from simulations.star_system.playback import Playback
from simulations.star_system.core import G, SOLAR_RADIUS_AU, kepler_period


def demonstration():
    return FormationExperiment(make_demonstration_setup(), ejection_radius=DEMONSTRATION_EJECTION_RADIUS)


def test_specification_names_the_stage_parcels_merge_model_and_questions():
    spec = specification()
    assert 'late-stage' in spec.starting_stage.lower()
    assert 'parcel' in spec.representation.lower()
    assert 'perfect merge' in spec.collision_model.lower()
    assert any('parcel' in question.lower() for question in spec.questions)
    assert any('checkpoint' in question.lower() for question in spec.questions)
    assert any('gas' in item.lower() for item in spec.gas_disk_requirements)
    assert any('perfect merge' in item.lower() or 'perfect-merge' in item.lower() for item in spec.recommended_scope)
    text = demonstration().describe()
    assert 'Status: running' in text
    assert '0 yr' in text
    assert FORMATION_MODEL in text
    assert 'perfect merge' in text.lower()


def test_late_stage_setup_repeats_for_the_same_seed():
    first = make_late_stage_setup(count=8, include_unbound=False)
    second = make_late_stage_setup(count=8, include_unbound=False)
    other = make_late_stage_setup(count=8, seed=49, include_unbound=False)
    assert setup_digest(first) == setup_digest(second)
    assert setup_digest(first) != setup_digest(other)
    assert first.provenance['seed'] == 48
    assert len(first.bodies) == 9


def test_formation_requires_one_star_and_massive_embryos():
    star = InitialBody('star', BodySpec('Star', 'star', 1., SOLAR_RADIUS_AU), (0., 0.), (0., 0.))
    particle = InitialBody('rock', BodySpec('Rock', 'particle', 0., None), (1., 0.), (0., 1.))
    with pytest.raises(ValueError, match='massive'):
        FormationExperiment(InitialSetup('particles', (star, particle), .01, collision_mode='perfect_merge'),
                            ejection_radius=10.)
    embryo = InitialBody('rock', BodySpec('Rock', 'planet', 1e-6, .01), (1., 0.), (0., 1.))
    with pytest.raises(ValueError, match='one star'):
        FormationExperiment(InitialSetup('no star', (embryo,), .01, collision_mode='perfect_merge'), ejection_radius=10.)


def test_merge_records_parcel_amounts_and_one_descendant():
    experiment = demonstration()
    experiment.integrate(.2)
    merges = [event for event in experiment.events if event.kind == 'merge']
    assert len(merges) == 1
    merge = merges[0]
    assert merge.parent_ids == ('alpha', 'beta')
    assert merge.descendant_ids == (merge.remnant_id,)
    amounts = {item.parcel_id: item.mass for item in merge.contributions}
    assert amounts == pytest.approx({'alpha': 1e-5, 'beta': 2e-5})
    assert merge.remnant_id in experiment.composition
    assert 'alpha' not in experiment.composition and 'beta' not in experiment.composition
    assert experiment.removed['alpha'].status == 'merged'
    story = experiment.replay_ancestry(merge.remnant_id)
    assert story.descendant_ids == (merge.remnant_id,)
    assert {item.parcel_id for item in story.contributions} == {'alpha', 'beta'}


def test_replay_highlights_the_bodies_that_built_a_survivor():
    experiment = demonstration()
    experiment.integrate(.2)
    remnant = next(event.remnant_id for event in experiment.events if event.kind == 'merge')
    story = experiment.replay_ancestry(remnant)
    opening = story.samples[0]
    assert set(story.highlighted(opening)) == {'alpha', 'beta'}
    assert remnant not in story.highlighted(opening)
    assert any(remnant in story.highlighted(sample) for sample in story.samples)


def test_ejected_body_keeps_a_history_after_its_region_is_empty():
    experiment = demonstration()
    experiment.integrate(1., finish=True)
    assert experiment.status == 'complete'
    story = experiment.replay_ancestry('escaper')
    assert story.status == 'ejected'
    assert story.descendant_ids == ('escaper',)
    occupied = [sample for sample in story.samples if any(body.id == 'escaper' for body in sample.bodies)]
    assert occupied
    escaper = next(body for body in occupied[-1].bodies if body.id == 'escaper')
    assert math.hypot(*escaper.r) >= DEMONSTRATION_EJECTION_RADIUS-1e-8
    empty = next(sample for sample in story.samples if all(body.id != 'escaper' for body in sample.bodies))
    assert all(math.hypot(body.r[0]-escaper.r[0], body.r[1]-escaper.r[1]) > .5 for body in empty.bodies)
    assert any(event.kind == 'ejection' and event.parent_ids == ('escaper',) for event in story.events)


def test_checkpoint_roundtrip_resumes_the_same_future(tmp_path):
    experiment = demonstration()
    experiment.integrate(.04)
    path = tmp_path/'formation.json'
    saved = save_checkpoint(experiment, path)
    restored = load_checkpoint(path)
    assert restored.checkpoint_id == saved == experiment.checkpoint_id
    assert not any(event.kind == 'merge' for event in experiment.events)
    assert [(event.kind, event.parent_ids) for event in restored.events] == [
        (event.kind, event.parent_ids) for event in experiment.events]
    experiment.integrate(.3)
    restored.integrate(.3)
    np.testing.assert_allclose(restored.prepared.system.r, experiment.prepared.system.r, rtol=0, atol=0)
    np.testing.assert_allclose(restored.prepared.system.v, experiment.prepared.system.v, rtol=0, atol=0)
    np.testing.assert_allclose(restored.prepared.system.masses, experiment.prepared.system.masses, rtol=0, atol=0)
    assert restored.time == experiment.time
    assert [(event.kind, event.parent_ids, tuple((item.parcel_id, item.mass) for item in event.contributions))
            for event in restored.events] == [
                (event.kind, event.parent_ids, tuple((item.parcel_id, item.mass) for item in event.contributions))
                for event in experiment.events]


def test_branch_records_one_change_and_does_not_edit_the_source(tmp_path):
    experiment = demonstration()
    experiment.integrate(.04)
    saved = save_checkpoint(experiment, tmp_path/'branch.json')
    velocity = experiment.prepared.system.v.copy()
    change = VelocityChange('alpha', (0., 6.), 'Add a +y velocity to alpha so it misses beta.')
    child = experiment.branch(change)
    np.testing.assert_array_equal(experiment.prepared.system.v, velocity)
    assert experiment.provenance['change'] is None
    assert child.provenance['parent_checkpoint_id'] == saved
    assert child.provenance['change']['note'] == change.note
    assert child.provenance['change']['body_id'] == 'alpha'
    assert child.provenance['setup_digest'] == experiment.provenance['setup_digest']
    assert child.provenance['run_id'] != experiment.provenance['run_id']
    experiment.integrate(.2)
    child.integrate(.12)
    assert any(event.kind == 'merge' for event in experiment.events)
    assert not any(event.kind == 'merge' for event in child.events)
    assert 'alpha' in child.composition
    with pytest.raises(ValueError, match='nonzero'):
        VelocityChange('alpha', (0., 0.), 'No change.')
    with pytest.raises(ValueError, match='not an active'):
        experiment.branch(VelocityChange('missing', (0., 1.), 'Nudge a missing embryo.'))


def test_synchronized_playback_stays_inside_the_common_interval():
    experiment = demonstration()
    experiment.integrate(.04)
    child = experiment.branch(VelocityChange('alpha', (0., 6.), 'Add a +y velocity to alpha so it misses beta.'))
    experiment.integrate(.2)
    child.integrate(.12)
    sync = SynchronizedPlayback(experiment, child)
    assert isinstance(sync.left_playback, Playback)
    early = sync.seek(.02)
    assert early.left.body_ids == early.right.body_ids
    np.testing.assert_allclose(early.left.r, early.right.r, rtol=0, atol=0)
    at_change = sync.seek(.04)
    alpha = at_change.left.body_ids.index('alpha')
    assert at_change.right.v[alpha, 1] == pytest.approx(at_change.left.v[alpha, 1]+6.)
    later = sync.seek(.12)
    assert 'alpha' in later.right.body_ids
    assert 'alpha' not in later.left.body_ids
    with pytest.raises(ValueError, match='common'):
        sync.seek(sync.common_end+1.)
    clock = experiment.time
    sync.play()
    sync.advance(1., 1.)
    assert experiment.time == clock
    assert child.time == pytest.approx(experiment.time-.08)


def test_playback_scrubs_recorded_frames_without_integrating():
    experiment = demonstration()
    experiment.integrate(.06)
    clock = experiment.time
    player = playback(experiment)
    assert isinstance(player, Playback)
    frame = player.next_sample()
    assert frame.time > 0
    assert experiment.time == clock
    assert experiment.prepared.system.r is not frame.r


def test_accounting_matches_mass_momentum_and_merge_energy():
    experiment = demonstration()
    experiment.integrate(1., finish=True)
    books = experiment.accounting()
    assert books.merge_count == 1
    assert books.ejection_count == 1
    assert books.closed_mass == pytest.approx(books.initial_mass, rel=1e-12, abs=1e-15)
    assert books.parcel_error == pytest.approx(0., abs=1e-15)
    assert books.merge_momentum_error == pytest.approx(0., abs=1e-12)
    assert books.merge_kinetic_error == pytest.approx(0., abs=1e-12)
    assert books.ejection_momentum_error == pytest.approx(0., abs=1e-12)
    with pytest.raises(ValueError, match='complete'):
        experiment.integrate(.02)


def test_resolution_keeps_remnant_mass_across_two_steps():
    points = compare_resolution()
    assert [point.dt for point in points] == [.02, .005]
    assert all(point.status == 'complete' for point in points)
    assert all(point.parent_ids == ('alpha', 'beta') for point in points)
    assert points[0].remnant_mass == pytest.approx(points[1].remnant_mass, rel=1e-12, abs=0)
    assert points[0].remnant_mass == pytest.approx(3e-5)
    assert points[0].first_merge_time == pytest.approx(points[1].first_merge_time, abs=.02)
    assert all(point.survivor_count == 1 for point in points)


def test_explicit_parcels_survive_a_merge():
    star = InitialBody('star', BodySpec('Star', 'star', 1., SOLAR_RADIUS_AU), (0., 0.), (0., 0.))
    left = InitialBody('left', BodySpec('Left', 'planet', 3e-5, .08), (5., 0.), (2.5, 0.))
    right = InitialBody('right', BodySpec('Right', 'planet', 2e-5, .08), (5.5, 0.), (-2.5, 0.))
    composition = {'star': {'star': 1.}, 'left': {'dust-a': 1e-5, 'dust-b': 2e-5}, 'right': {'dust-c': 2e-5}}
    experiment = FormationExperiment(InitialSetup('Parcel handoff', (star, left, right), .02, collision_mode='perfect_merge'),
                                     ejection_radius=30., composition=composition)
    experiment.integrate(.2)
    merge = next(event for event in experiment.events if event.kind == 'merge')
    amounts = {item.parcel_id: item.mass for item in merge.contributions}
    assert amounts == pytest.approx({'dust-a': 1e-5, 'dust-b': 2e-5, 'dust-c': 2e-5})
    assert experiment.accounting().parcel_error == pytest.approx(0., abs=1e-15)


def test_composition_must_match_body_mass():
    star = InitialBody('star', BodySpec('Star', 'star', 1., SOLAR_RADIUS_AU), (0., 0.), (0., 0.))
    rock = InitialBody('rock', BodySpec('Rock', 'planet', 1e-5, .05), (2., 0.), (0., 3.))
    setup = InitialSetup('Short ledger', (star, rock), .02, collision_mode='perfect_merge')
    with pytest.raises(ValueError, match='add up'):
        FormationExperiment(setup, ejection_radius=20., composition={'star': {'star': 1.}, 'rock': {'dust-a': 1e-6}})


def test_stellar_contact_continues_without_merging():
    star = InitialBody('star', BodySpec('Star', 'star', 1., SOLAR_RADIUS_AU), (0., 0.), (0., 0.))
    rock = InitialBody('rock', BodySpec('Rock', 'planet', 1e-6, .02), (.15, 0.), (-6., 0.))
    experiment = FormationExperiment(InitialSetup('Starward', (star, rock), .002, collision_mode='perfect_merge'),
                                     ejection_radius=20.)
    experiment.integrate(.05, finish=True)
    assert experiment.status == 'complete'
    assert experiment.time == pytest.approx(.05)
    logged = [event for event in experiment.events if event.kind == 'unmodeled']
    assert logged and logged[0].parent_ids == ('rock', 'star')
    assert not any(event.kind == 'merge' for event in experiment.events)
    assert 'star' in experiment.composition and 'rock' in experiment.composition
    assert experiment.replay_ancestry('rock').samples


def test_speed_angle_fragments_keep_parcels_and_survive_a_checkpoint(tmp_path):
    experiment = FormationExperiment(with_collision_rule(make_demonstration_setup(), 'speed_angle'),
                                     ejection_radius=DEMONSTRATION_EJECTION_RADIUS)
    experiment.integrate(.2)
    fragments = [event for event in experiment.events if event.kind == 'fragment']
    assert len(fragments) == 1
    fragment = fragments[0]
    assert fragment.parent_ids == ('alpha', 'beta')
    assert fragment.remnant_id in fragment.descendant_ids and len(fragment.descendant_ids) == 3
    assert fragment.lost_contributions
    tracked = {}
    for body_id in fragment.descendant_ids:
        for parcel, mass in experiment.composition[body_id].items():
            tracked[parcel] = tracked.get(parcel, 0.)+mass
    for item in fragment.lost_contributions:
        tracked[item.parcel_id] = tracked.get(item.parcel_id, 0.)+item.mass
    assert tracked == pytest.approx({'alpha': 1e-5, 'beta': 2e-5})
    assert experiment.removed['alpha'].status == 'fragmented'
    story = experiment.replay_ancestry('alpha')
    assert story.status == 'fragmented'
    assert any(event.kind == 'fragment' and event.descendant_ids == fragment.descendant_ids for event in story.events)
    assert story.descendant_ids
    books = experiment.accounting()
    assert books.merge_count == 0
    assert books.parcel_error == pytest.approx(0., abs=1e-15)
    assert books.merge_momentum_error == pytest.approx(0., abs=1e-12)
    path = tmp_path/'fragments.json'
    save_checkpoint(experiment, path)
    restored = load_checkpoint(path)
    assert restored.provenance['collision_model'] == SPEED_ANGLE_VERSION
    assert restored.prepared.setup.collision_mode == 'speed_angle'
    restored_event = next(event for event in restored.events if event.kind == 'fragment')
    assert restored_event.descendant_ids == fragment.descendant_ids
    assert {(item.parcel_id, item.mass) for item in restored_event.lost_contributions} == {
        (item.parcel_id, item.mass) for item in fragment.lost_contributions}
    child = restored.branch(VelocityChange(fragment.descendant_ids[0], (0., .1), 'Nudge one fragment.'))
    assert child.provenance['parent_checkpoint_id'] == restored.checkpoint_id
    assert fragment.descendant_ids[0] in child.composition


def test_same_experiment_can_be_rerun_under_each_stored_rule():
    perfect = FormationExperiment(make_demonstration_setup(), ejection_radius=100.)
    angled = FormationExperiment(with_collision_rule(make_demonstration_setup(), 'speed_angle'), ejection_radius=100.)
    perfect.integrate(.2, finish=True)
    angled.integrate(.2, finish=True)
    assert perfect.status == angled.status == 'complete'
    assert any(event.kind == 'merge' and len(event.descendant_ids) == 1 for event in perfect.events)
    assert any(event.kind == 'fragment' and len(event.descendant_ids) == 3 for event in angled.events)
    compared = SynchronizedPlayback(perfect, angled)
    view = compared.seek(min(perfect.time, angled.time))
    assert view.common_start <= view.time <= view.common_end


def test_invalid_checkpoint_is_rejected(tmp_path):
    experiment = demonstration()
    experiment.integrate(.02)
    path = tmp_path/'checkpoint.json'
    save_checkpoint(experiment, path)
    document = json.loads(path.read_text())
    document['id'] = '0'*64
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='id'):
        load_checkpoint(path)
    document['schema'] = 'formation-checkpoint-v0'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='schema'):
        load_checkpoint(path)
    path.write_text('{')
    with pytest.raises(ValueError, match='Invalid formation checkpoint'):
        load_checkpoint(path)


def test_benchmark_reports_cost_memory_and_recording_size():
    report = run_benchmark()
    assert report.model == FORMATION_MODEL
    assert report.embryo_count == BENCHMARK_EMBRYO_COUNT
    assert report.body_count == BENCHMARK_EMBRYO_COUNT+2
    assert report.seed == 48
    assert report.requested_years == 1.
    assert report.simulated_years > 0
    assert report.calculation_seconds > 0
    assert report.peak_python_bytes > 0
    assert report.recording_bytes > 0
    assert report.sample_count >= 1
    assert report.hardware
    assert 'Python' in report.hardware
    assert report.status in ('complete', 'halted')
    assert report.notes


def test_feasibility_note_states_the_measured_experiment():
    text = (Path(__file__).resolve().parents[1]/'docs'/'formation-feasibility.md').read_text(encoding='utf-8')
    lowered = text.lower()
    assert str(BENCHMARK_EMBRYO_COUNT) in text
    assert 'perfect merge' in lowered
    assert 'gas' in lowered
    assert 'checkpoint' in lowered
    assert 'stewart' in lowered
    assert 'formation-numerics.md' in text


def test_default_late_stage_setup_stays_planar_and_inflated():
    setup = make_late_stage_setup(count=4, include_unbound=False)
    assert all(len(body.r) == 2 and len(body.v) == 2 for body in setup.bodies)
    embryos = [body for body in setup.bodies if body.spec.kind != 'star']
    assert all(body.spec.radius == BENCHMARK_EMBRYO_RADIUS for body in embryos)
    rocky = make_late_stage_setup(count=4, include_unbound=False, radii='rocky')
    radius = rocky_radius(BENCHMARK_EMBRYO_MASS)
    assert rocky.bodies[1].spec.radius == pytest.approx(radius)
    assert BENCHMARK_EMBRYO_RADIUS/radius > 500


def test_mixed_dimensions_are_rejected():
    star = InitialBody('star', BodySpec('Star', 'star', 1., SOLAR_RADIUS_AU), (0., 0.), (0., 0.))
    embryo = InitialBody('rock', BodySpec('Rock', 'planet', 1e-6, .01), (1., 0., 0.), (0., 1., 0.))
    setup = InitialSetup('Mixed', (star, embryo), .01, collision_mode='perfect_merge')
    with pytest.raises(ValueError, match='three'):
        FormationExperiment(setup, ejection_radius=10.)


def test_inclined_circular_orbit_keeps_a_vertical_component():
    mass = 1e-6
    setup = make_inclined_circular_setup(mass=mass)
    experiment = FormationExperiment(setup, ejection_radius=5.)
    assert experiment.dimensions == 3
    relative_r = experiment.prepared.system.r[1]-experiment.prepared.system.r[0]
    relative_v = experiment.prepared.system.v[1]-experiment.prepared.system.v[0]
    assert relative_r[2] == pytest.approx(0., abs=1e-12)
    assert abs(relative_v[2]) > .1
    period = kepler_period(1., G*(1.+mass))
    experiment.integrate(period/4)
    assert experiment.status == 'running'
    later_r = experiment.prepared.system.r[1]-experiment.prepared.system.r[0]
    later_v = experiment.prepared.system.v[1]-experiment.prepared.system.v[0]
    assert later_r[2] > .3
    initial_h = np.cross(relative_r, relative_v)
    later_h = np.cross(later_r, later_v)
    cosine = float(np.dot(initial_h, later_h)/(np.linalg.norm(initial_h)*np.linalg.norm(later_h)))
    assert cosine > math.cos(math.radians(1.5))


def test_adaptive_step_shrinks_on_a_close_miss_and_recovers():
    setup = make_close_pass_setup()
    adaptive = FormationExperiment(setup, ejection_radius=80., adaptive=True)
    steps = []
    while adaptive.status == 'running' and adaptive.time < .4:
        adaptive.advance(1)
        steps.append(adaptive.prepared.system.dt)
    assert adaptive.status == 'running'
    assert min(steps) < setup.dt/5
    assert steps[-1] == pytest.approx(setup.dt)
    assert not any(event.kind == 'merge' for event in adaptive.events)
    separations = []
    for sample in adaptive.samples:
        bodies = {body.id: body for body in sample.bodies}
        if 'alpha' in bodies and 'beta' in bodies:
            separations.append(math.dist(bodies['alpha'].r, bodies['beta'].r))
    assert min(separations) < .05
    assert min(separations) > 10*(rocky_radius(BENCHMARK_EMBRYO_MASS)*2)
    fixed = FormationExperiment(make_close_pass_setup(), ejection_radius=80., adaptive=False)
    fixed.integrate(.05)
    assert fixed.status == 'running'
    assert fixed.prepared.system.dt == setup.dt


def test_adaptive_floor_halts_before_taking_a_step():
    setup = make_close_pass_setup()
    experiment = FormationExperiment(setup, ejection_radius=80., adaptive=True, dt_min=.01)
    positions = experiment.prepared.system.r.copy()
    experiment.advance(1)
    assert experiment.status == 'halted'
    assert 'floor' in experiment.reason.lower()
    np.testing.assert_array_equal(experiment.prepared.system.r, positions)
    assert experiment.time == 0.


def test_spatial_merge_conserves_vector_momentum():
    star = InitialBody('star', BodySpec('Star', 'star', 1., SOLAR_RADIUS_AU), (0., 0., 0.), (0., 0., 0.))
    alpha = InitialBody('alpha', BodySpec('Alpha', 'planet', 1e-5, .08), (5., 0., 0.), (2.5, 0., .05))
    beta = InitialBody('beta', BodySpec('Beta', 'planet', 2e-5, .08), (5.5, 0., 0.), (-2.5, 0., -.05))
    setup = InitialSetup('Spatial merge', (star, alpha, beta), .02, collision_mode='perfect_merge')
    experiment = FormationExperiment(setup, ejection_radius=30.)
    experiment.integrate(.2, finish=True)
    merges = [event for event in experiment.events if event.kind == 'merge']
    assert len(merges) == 1
    assert len(merges[0].location) == 3
    assert merges[0].unmodeled_spin > 0
    assert experiment.accounting().merge_momentum_error == pytest.approx(0., abs=1e-12)


def test_spatial_checkpoint_and_branch_keep_the_third_component(tmp_path):
    experiment = FormationExperiment(make_inclined_circular_setup(), ejection_radius=5., adaptive=True)
    experiment.integrate(.01)
    path = tmp_path/'spatial.json'
    save_checkpoint(experiment, path)
    restored = load_checkpoint(path)
    assert restored.adaptive and restored.dimensions == 3
    assert restored.dt_max == pytest.approx(experiment.dt_max)
    np.testing.assert_array_equal(restored.prepared.system.r, experiment.prepared.system.r)
    experiment.integrate(.01)
    restored.integrate(.01)
    np.testing.assert_allclose(restored.prepared.system.r, experiment.prepared.system.r, rtol=0, atol=0)
    np.testing.assert_allclose(restored.prepared.system.v, experiment.prepared.system.v, rtol=0, atol=0)
    embryo = experiment.prepared.body_ids.index('embryo')
    change = VelocityChange('embryo', (0., 0., .01), 'Add a small vertical velocity.')
    child = experiment.branch(change)
    assert child.prepared.system.v[embryo, 2] == pytest.approx(experiment.prepared.system.v[embryo, 2]+.01)
    assert experiment.provenance['change'] is None
    with pytest.raises(ValueError, match='components'):
        experiment.branch(VelocityChange('embryo', (0., .01), 'Planar nudge on a spatial run.'))


def test_planar_and_spatial_setup_documents_disclose_dimension(tmp_path):
    planar = make_demonstration_setup()
    planar_path = tmp_path/'planar.json'
    save_setup(planar, planar_path)
    document = json.loads(planar_path.read_text())
    assert document['model_version'] == MODEL_VERSION
    assert all(len(body['r']) == 2 and len(body['v']) == 2 for body in document['setup']['bodies'])
    assert setup_digest(load_setup(planar_path)) == setup_digest(planar)
    spatial = make_inclined_circular_setup()
    spatial_path = tmp_path/'spatial.json'
    save_setup(spatial, spatial_path)
    spatial_document = json.loads(spatial_path.read_text())
    assert spatial_document['model_version'] == SPATIAL_MODEL_VERSION
    assert all(len(body['r']) == 3 and len(body['v']) == 3 for body in spatial_document['setup']['bodies'])
    loaded = load_setup(spatial_path)
    assert loaded.bodies[1].r[2] == pytest.approx(spatial.bodies[1].r[2])
    assert setup_digest(loaded) == setup_digest(spatial)


def test_overhead_view_projects_a_spatial_sample():
    experiment = FormationExperiment(make_inclined_circular_setup(), ejection_radius=5.)
    frame = StateFrame.capture(experiment.prepared)
    assert frame.r.shape[1] == 3
    projected = frame_transform(frame, 'world').positions(frame.r)
    assert projected.shape == (len(frame.r), 2)
    np.testing.assert_allclose(projected, frame.r[:, :2])


def test_physical_probe_reports_a_short_rocky_run():
    report = run_physical_probe()
    assert report.embryo_count == PHYSICAL_PROBE_COUNT
    assert report.body_count == PHYSICAL_PROBE_COUNT+1
    assert report.requested_years == .25
    assert report.calculation_seconds > 0
    assert report.rocky_radius < report.encounter_radius/500
    assert report.maximum_step >= report.minimum_step
    assert report.status in ('complete', 'halted')
    assert any('tree' in note.lower() for note in report.notes)
    assert any('drag' in note.lower() and 'tide' in note.lower() for note in report.notes)


def test_numerics_note_states_the_integrator_decision():
    text = (Path(__file__).resolve().parents[1]/'docs'/'formation-numerics.md').read_text(encoding='utf-8')
    lowered = text.lower()
    assert 'rebound' in lowered
    assert 'adaptive' in lowered
    assert 'rocky' in lowered
    assert 'three-dimensional' in lowered or '3d' in lowered
    assert '#69' in text and '#71' in text and '#79' in text
    assert 'tree' in lowered
    assert 'planar' in lowered
