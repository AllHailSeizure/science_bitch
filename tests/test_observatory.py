"""Formation observatory exercises only stored samples after fixture creation."""
import copy
import json
import os

import pytest
from matplotlib.figure import Figure

from simulations.star_system.observatory import (
    RecordedComparison, carriers, contributions, formation_metadata, history_indices,
)
from simulations.star_system.player import Player, render


def formation_run():
    def sample(time, ids, amounts, phase=None):
        return dict(time=time, ids=ids, mass=[1.] * len(ids),
                    r=[[i, time, 9.] for i in range(len(ids))],
                    v=[[0., 0., 0.] for _ in ids], phase=phase,
                    contributions={key: [dict(parcel_id=p, mass=m) for p, m in values]
                                   for key, values in amounts.items()})
    return dict(status='complete', time_start=0., time_end=2.,
                bodies=[dict(id=key, name=key, kind='star' if key == 'star' else 'planet')
                        for key in ('star', 'a', 'b', 'c')],
                samples=[sample(0., ['star', 'a', 'b'], {'a': [('p', 1.)], 'b': [('q', 2.)]}),
                         sample(1., ['star', 'a', 'b'], {'a': [('p', 1.)], 'b': [('q', 2.)]}, 'pre'),
                         sample(1., ['star', 'c'], {'c': [('p', 1.), ('q', 2.)]}, 'post'),
                         sample(2., ['star', 'c'], {'c': [('p', 1.), ('q', 2.)]})],
                formation_events=[dict(time=1., kind='merge', parent_ids=['a', 'b'],
                                       descendant_ids=['c'], message='a and b merged')],
                removed_bodies=[dict(id='a', status='merged'), dict(id='b', status='merged')],
                assumptions=['inflated radii; perfect merge'],
                provenance=dict(parent_checkpoint_id='checkpoint-1', setup_digest='setup',
                                run_id='left', change=None))


def test_recorded_ancestry_and_exact_pre_post_samples():
    run = formation_run()
    player = Player(dict(schema='sidereel-playback-v1', runs=[run]))
    original = copy.deepcopy(run)
    assert contributions(run, 'c') == [dict(parcel_id='p', mass=1.), dict(parcel_id='q', mass=2.)]
    assert carriers(run, run['samples'][0], 'c') == ['a', 'b']
    assert history_indices(run, 'a') == [0, 1, 2]
    player.selected_body = 'c'
    player.seek_sample(0)
    figure = Figure()
    render(player, figure)
    rings = [artist for artist in figure.axes[0].collections if artist.get_gid() == 'ancestry']
    assert len(rings[0].get_offsets()) == 2
    player.selected_body = 'a'
    player.seek_sample(1)
    render(player, figure)
    assert len(figure.axes[0].collections) == 4  # three bodies plus ancestry ring
    player.seek_sample(2)
    render(player, figure)
    assert len(figure.axes[0].collections) == 2  # removed parents have trails, no live dots
    assert run == original


def test_comparison_uses_shared_interval_and_stops_without_switching_runs():
    left, right = formation_run(), formation_run()
    right['samples'] = right['samples'][2:]
    right['time_start'] = 1.
    right['provenance']['change'] = dict(body_id='c', delta_v=[.1, 0.], note='disclosed kick')
    comparison = RecordedComparison(left, right)
    assert comparison.common_start == 1.
    with pytest.raises(ValueError, match='common simulated interval'):
        comparison.seek(0.)
    player = Player(dict(schema='sidereel-playback-v1', runs=[left, right]))
    player.comparing = True
    player.seek(-1.)
    assert player.time == 1.
    player.play()
    player.advance(10.)
    assert player.time == 2.
    assert player.paused and player.run_index == 0
    assert player.comparison().view().left.time == player.comparison().view().right.time == 2.


def test_offscreen_observatory_scrubs_events_removed_history_and_branches(tmp_path, monkeypatch):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PyQt6.QtWidgets import QApplication
    from simulations.star_system.player_window import PlayerShell
    from simulations.star_system import core
    monkeypatch.setattr(core.System, 'step', lambda *a, **kw: pytest.fail('player called solver'))
    application = QApplication.instance() or QApplication([])
    left, right = formation_run(), formation_run()
    right['provenance']['run_id'] = 'right'
    right['provenance']['change'] = dict(body_id='a', delta_v=[.1, 0.], note='kick')
    path = tmp_path / 'formation.sidereel'
    path.write_text(json.dumps(dict(schema='sidereel-playback-v1', runs=[left, right])))
    window = PlayerShell(path)
    window.timer.stop()
    window.body_combo.setCurrentIndex(window.body_combo.findData('c'))
    assert 'p: 1' in window.inspection.toPlainText()
    assert 'inflated radii' in window.inspection.toPlainText()
    window.state_slider.setValue(0)
    window._scrub_event(1)
    assert window.player.time == 1. and window.player.paused
    window.body_combo.setCurrentIndex(window.body_combo.findData('a'))
    assert 'merged' in window.inspection.toPlainText()
    window._scrub_history(window.history_combo.count() - 1)
    assert window.player.sample_index == 2
    assert 'absent' in window.history_combo.itemText(window.history_combo.count() - 1)
    window.compare_check.setChecked(True)
    application.processEvents()
    assert 'checkpoint-1' in window.inspection.toPlainText()
    assert 'kick' in window.inspection.toPlainText()
    assert len(window.figure.axes) == 2
    window.close()


def test_headless_writer_preserves_prototype_metadata():
    from simulations.star_system.formation import FormationExperiment, make_demonstration_setup
    from simulations.star_system.experiment_file import _formation_sample
    experiment = FormationExperiment(make_demonstration_setup(), ejection_radius=6.)
    metadata = formation_metadata(experiment)
    assert metadata['assumptions']
    assert metadata['provenance']['setup_digest']
    sample = _formation_sample(experiment.samples[0], experiment.dt_max)
    assert sample['contributions']['alpha'][0]['mass'] > 0
    json.dumps(metadata)


@pytest.mark.parametrize('status,kind', [('merged', 'merge'), ('fragmented', 'fragment'), ('ejected', 'ejection')])
def test_removed_body_history_includes_following_empty_region(status, kind):
    run = formation_run()
    run['removed_bodies'][0]['status'] = status
    run['formation_events'][0]['kind'] = kind
    indices = history_indices(run, 'a')
    assert 'a' in run['samples'][indices[-2]]['ids']
    assert 'a' not in run['samples'][indices[-1]]['ids']
    assert carriers(run, run['samples'][indices[-1]], 'a') == []


def test_calculated_branches_export_without_further_integration(monkeypatch, tmp_path):
    from simulations.star_system.formation import FormationExperiment, VelocityChange, make_demonstration_setup
    from simulations.star_system.observatory import formation_playback
    from simulations.star_system.player import load_playback
    parent = FormationExperiment(make_demonstration_setup(), ejection_radius=6.)
    parent.integrate(.02, finish=True)
    left = parent.branch(VelocityChange('alpha', (.01, 0.), 'left kick'))
    right = parent.branch(VelocityChange('beta', (-.01, 0.), 'right kick'))
    left.integrate(.2, finish=True)
    right.integrate(.3, finish=True)
    monkeypatch.setattr(FormationExperiment, 'integrate', lambda *a, **kw: pytest.fail('recalculated'))
    document = formation_playback(left, right)
    path = tmp_path / 'branches.sidereel'
    path.write_text(json.dumps(document))
    player = Player(load_playback(path))
    player.comparing = True
    player.seek(.1)
    figure = Figure()
    render(player, figure)
    assert len(figure.axes) == 2
    assert document['runs'][0]['provenance']['parent_checkpoint_id'] == parent.checkpoint_id
    assert document['runs'][1]['provenance']['change']['note'] == 'right kick'
    assert document['runs'][0]['formation_events']
    point = document['runs'][0]['branch_point']
    assert point['time'] == pytest.approx(.02)
    common = next(body for body in point['common_state']['bodies'] if body['id'] == 'alpha')
    changed = next(body for body in point['changed_state']['bodies'] if body['id'] == 'alpha')
    assert changed['v'][0] - common['v'][0] == pytest.approx(.01)
