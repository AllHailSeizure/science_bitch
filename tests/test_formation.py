"""Top-level formation behavior: ancestry, checkpoints, branches, and accounting."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from simulations.star_system.formation import (
    BENCHMARK_EMBRYO_COUNT, DEMONSTRATION_EJECTION_RADIUS, FORMATION_MODEL,
    FormationExperiment, SynchronizedPlayback, VelocityChange, compare_resolution,
    load_checkpoint, make_demonstration_setup, make_late_stage_setup, playback,
    run_benchmark, save_checkpoint, specification,
)
from simulations.star_system.model import BodySpec, InitialBody, InitialSetup
from simulations.star_system.persistence import setup_digest
from simulations.star_system.playback import Playback
from simulations.star_system.core import SOLAR_RADIUS_AU


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
    assert [(event.kind, event.parent_ids) for event in restored.events] == []
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


def test_stellar_contact_halts_without_merging():
    star = InitialBody('star', BodySpec('Star', 'star', 1., SOLAR_RADIUS_AU), (0., 0.), (0., 0.))
    rock = InitialBody('rock', BodySpec('Rock', 'planet', 1e-6, .02), (.15, 0.), (-6., 0.))
    experiment = FormationExperiment(InitialSetup('Starward', (star, rock), .002, collision_mode='perfect_merge'),
                                     ejection_radius=20.)
    experiment.integrate(.05, finish=True)
    assert experiment.status == 'halted'
    assert experiment.reason
    assert not any(event.kind == 'merge' for event in experiment.events)
    assert 'star' in experiment.composition
    assert experiment.replay_ancestry('rock').samples


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
