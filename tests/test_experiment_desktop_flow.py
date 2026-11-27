"""Top-level desktop flow for preparation, recording, collision, and saves."""
import os
import time

import numpy as np
import pytest

from simulations.star_system.initialization import build_system
from simulations.star_system.model import BodySpec, InitialBody, InitialSetup
from simulations.star_system.persistence import load_setup, setup_digest
from simulations.star_system.scenarios import get_scenario


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


def quiet_pair():
    return InitialSetup('Quiet pair', (
        InitialBody('a', BodySpec('A', 'planet', 1e-6, .01), (0., 0.), (0., 0.)),
        InitialBody('b', BodySpec('B', 'planet', 1e-6, .01), (5., 0.), (0., .1)),
    ), .01)


def contact_pair(mode='none', kind='planet'):
    return InitialSetup('Contact', (
        InitialBody('a', BodySpec('A', kind, 1e-14, .1, temperature=5000. if kind == 'star' else None),
                    (-1., 0.), (10., 0.)),
        InitialBody('b', BodySpec('B', 'planet', 2e-14, .1), (1., 0.), (0., 0.)),
    ), .2, collision_mode=mode)


def bare_pair():
    return InitialSetup('Bare rock', (
        InitialBody('star', BodySpec('Star', 'star', 1., .01, temperature=5000.), (0., 0.), (0., 0.)),
        InitialBody('rock', BodySpec('Rock', 'planet', 1e-6, None), (1., 0.), (0., 1.)),
    ), .01)


def calculate(window, years):
    window.duration.setValue(years)
    window.stride.setValue(1)
    window.start_calculation()
    for _ in range(20):
        if window.calculation.done:
            break
        window.tick()
    assert window.calculation.done


def test_jupiter_scenario_opens_in_the_sandbox_without_changing_the_preset(window, qtapp, tmp_path):
    window.choose_button.click()
    titles = [window.experiment_list.item(i).text() for i in range(window.experiment_list.count())]
    scenario = get_scenario('sun-jupiter-ring')
    frozen = setup_digest(scenario)
    row = titles.index(scenario.name)
    window.experiment_list.setCurrentRow(row)
    qtapp.processEvents()
    assert 'Trojan' in window.experiment_description.text()
    assert 'not steered' in window.experiment_observe.text()
    window.open_button.click()
    qtapp.processEvents()
    assert window.heading_label.text() == scenario.name
    assert window.setup.preset_id == 'sun-jupiter-ring'
    assert window.setup.population.settings.seed == 7
    assert window.setup.population.settings.count == 32
    assert window.editing and window.prepared.system.time == 0
    path = tmp_path/'ring.json'
    window.save_to(path)
    loaded = load_setup(path)
    assert loaded.population.settings.seed == 7
    assert loaded.collision_mode == 'none'
    window.name_field.setText('My Jupiter ring')
    window.rename_experiment()
    assert window.setup.preset_id is None
    assert window.setup.provenance['origin_preset'] == 'sun-jupiter-ring'
    assert window.heading_label.text() == 'My Jupiter ring'
    assert setup_digest(get_scenario('sun-jupiter-ring')) == frozen
    assert setup_digest(load_setup(path)) == frozen


def test_save_initial_and_current_state_are_separate_epochs(window, tmp_path):
    window.install_experiment(quiet_pair())
    initial = tmp_path/'initial.json'
    captured = tmp_path/'captured.json'
    window.save_to(initial)
    window.step_button.click()
    evolved = window.prepared.system.r.copy()
    elapsed = window.prepared.system.time
    result = window.save_current_to(captured, 'After one step')
    assert result.name == 'After one step'
    assert result.population is None
    assert result.provenance['source_name'] == 'Quiet pair'
    assert result.provenance['source_time'] == pytest.approx(elapsed)
    assert 'time zero' in result.provenance['capture_note']
    assert window.setup.name == 'Quiet pair'
    assert window.prepared.system.time == pytest.approx(elapsed)
    np.testing.assert_array_equal(window.prepared.system.r, evolved)
    restored = build_system(load_setup(initial))
    assert restored.system.time == 0
    np.testing.assert_array_equal(restored.system.r, build_system(quiet_pair()).system.r)
    rebuilt = build_system(load_setup(captured))
    np.testing.assert_array_equal(rebuilt.system.r, evolved)
    assert rebuilt.system.time == 0


def test_rejected_capture_leaves_the_open_experiment_and_initial_file(window, tmp_path):
    window.install_experiment(quiet_pair())
    initial = tmp_path/'initial.json'
    window.save_to(initial)
    window.step_button.click()
    digest = setup_digest(window.setup)
    elapsed = window.prepared.system.time
    with pytest.raises(ValueError, match='different, nonempty'):
        window.save_current_to(tmp_path/'other.json', window.setup.name)
    with pytest.raises(ValueError, match='different from the initial setup path'):
        window.save_current_to(initial, 'After one step')
    assert not (tmp_path/'other.json').exists()
    assert setup_digest(load_setup(initial)) == digest
    assert setup_digest(window.setup) == digest
    assert window.prepared.system.time == pytest.approx(elapsed)


def test_reloaded_setup_matches_the_original_evolution(window, tmp_path):
    window.install_experiment(quiet_pair())
    path = tmp_path/'quiet.json'
    window.save_to(path)
    reference = build_system(load_setup(path))
    window.load_from(path)
    assert window.initial_path == path
    assert window.prepared.system.time == 0
    for _ in range(5):
        reference.system.step()
        window.step_button.click()
    np.testing.assert_allclose(window.prepared.system.r, reference.system.r, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(window.prepared.system.v, reference.system.v, rtol=1e-12, atol=1e-12)


def test_jupiter_capture_keeps_evolved_particles_without_regenerating_them(window, tmp_path):
    window.install_experiment(get_scenario('sun-jupiter-ring'))
    initial = tmp_path/'ring.json'
    window.save_to(initial)
    window.step_button.click()
    evolved = window.prepared.system.r.copy()
    count = len(window.prepared.body_ids)
    captured = tmp_path/'ring-now.json'
    window.save_current_to(captured, 'Evolved ring')
    assert load_setup(initial).population.settings.seed == 7
    now = load_setup(captured)
    assert now.population is None
    assert len(now.bodies) == count
    np.testing.assert_array_equal(build_system(now).system.r, evolved)
    assert window.setup.population is not None
    assert window.setup.population.settings.seed == 7


def test_playback_speed_selects_recorded_samples_without_integrating(window, monkeypatch):
    window.install_experiment(quiet_pair())
    window.duration.setValue(.03)
    window.stride.setValue(1)
    window.start_calculation()
    assert window.progress.isVisible()
    assert 'Calculating' in window.notice.text()
    assert window.calculation.total_steps > 1
    while not window.calculation.done:
        window.tick()
    recording = window.playback.recording
    assert recording.status == 'complete'
    assert window.prepared.system.time == 0
    assert f'{0:,.3f} elapsed simulated years' in window.time_readout.text()

    def unexpected(*_args, **_kwargs):
        raise AssertionError('Recorded playback must not integrate')

    monkeypatch.setattr('simulations.star_system.desktop.advance_prepared', unexpected)
    monkeypatch.setattr('simulations.star_system.core.System.step', unexpected)
    window.speed.setValue(1000)
    window.run_button.click()
    assert window.run_button.text() == 'Pause'
    window.last_tick = time.monotonic()-1
    window.tick()
    last = recording.frames[-1]
    assert window.playback.index == len(recording.frames)-1
    assert window.prepared.system.time == pytest.approx(last.time)
    np.testing.assert_array_equal(window.prepared.system.r, last.r)
    assert f'{last.time:,.3f} elapsed simulated years' in window.time_readout.text()
    assert 'COMPLETE' in window.scene.result_info.get_text()
    window.edit_button.click()
    assert window.playback is None
    assert window.prepared.system.time == 0
    assert window.run_button.text() == 'Run'


def test_recorded_sample_capture_uses_that_sample_and_keeps_the_recording(window, tmp_path):
    window.install_experiment(quiet_pair())
    calculate(window, .03)
    window.step_button.click()
    frame = window.playback.current_frame
    recording = window.playback.recording
    path = tmp_path/'sample.json'
    window.save_current_to(path, 'Sample epoch')
    loaded = load_setup(path)
    assert loaded.provenance['source_setup_digest'] == recording.setup_digest
    assert loaded.provenance['source_name'] == recording.source_name
    assert loaded.provenance['source_time'] == pytest.approx(frame.time)
    np.testing.assert_array_equal(build_system(loaded).system.r, frame.r)
    assert window.playback.recording is recording
    assert window.playback.current_frame.time == pytest.approx(frame.time)
    assert window.setup.name == 'Quiet pair'


def test_collision_reset_keeps_the_recording(window):
    window.install_experiment(quiet_pair())
    original = window.setup
    calculate(window, .02)
    recording = window.session.recordings[0]
    assert recording.status == 'complete'
    assert recording.source_setup.collision_mode == 'none'
    window.collision_mode.setCurrentIndex(window.collision_mode.findData('perfect_merge'))
    window.collision_button.click()
    assert original.collision_mode == 'none'
    assert window.setup is not original
    assert window.setup.collision_mode == 'perfect_merge'
    assert window.setup.preset_id is None
    assert recording.source_setup.collision_mode == 'none'
    assert recording.setup_digest == setup_digest(original)
    assert len(window.session.recordings) == 1
    assert window.playback is None
    assert window.editing and window.prepared.system.time == 0
    assert len(window.prepared.body_ids) == 2
    assert 'Active collision mode: perfect_merge' in window.collision_note.text()
    assert 'Model: perfect_merge' in window.scene.heading.get_text()
    window.undo_button.click()
    assert window.setup.collision_mode == 'none'
    assert window.prepared.system.time == 0


def test_saved_perfect_merge_joins_nonstellar_bodies(window, tmp_path):
    window.install_experiment(contact_pair(mode='perfect_merge'))
    path = tmp_path/'merge.json'
    window.save_to(path)
    window.load_from(path)
    assert window.collision_mode.currentData() == 'perfect_merge'
    assert 'Model: perfect_merge' in window.scene.heading.get_text()
    window.step_button.click()
    assert len(window.prepared.body_ids) == 1
    assert window.prepared.system.time == pytest.approx(.2)


def test_missing_radius_does_not_replace_the_experiment(window):
    window.install_experiment(bare_pair())
    digest = setup_digest(window.setup)
    window.collision_mode.setCurrentIndex(window.collision_mode.findData('perfect_merge'))
    window.collision_button.click()
    assert window.setup.collision_mode == 'none'
    assert setup_digest(window.setup) == digest
    assert window.collision_mode.currentData() == 'none'
    assert 'radius' in window.notice.text().lower()
    assert 'unchanged' in window.notice.text().lower()


def test_star_contact_continues_the_live_step_and_the_calculation(window):
    window.install_experiment(contact_pair(mode='perfect_merge', kind='star'))
    window.step_button.click()
    assert window.prepared.system.time == pytest.approx(.2)
    assert window.prepared.body_ids == ('a', 'b')
    assert any(event.kind == 'unmodeled' for event in window.prepared.events)
    assert 'Unmodeled stellar contact' in window.notice.text()
    assert '#14' in window.notice.text()
    window.edit_button.click()
    window.duration.setValue(1)
    window.start_calculation()
    assert window.progress.isVisible()
    for _ in range(30):
        if window.calculation.done:
            break
        window.tick()
    result = window.calculation.result
    assert result.status == 'complete'
    assert result.actual_end_time == pytest.approx(1.)
    assert window.notice.text().startswith('Complete:')
    assert not any(event.kind == 'merge' for frame in result.frames for event in frame.events)
