"""Headless player: recorded samples, views, and the control-board transport."""
import json
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from simulations.star_system.player import (
    GOLD, INNER_LIMIT_AU, OUTER_LIMIT_AU, VIEW_INNER_OUTER, VIEW_LIVE, VIEW_OVERLAY,
    Player, equal_time, export_overlay_csv, export_overlay_png, export_overlay_svg,
    export_split_gif, export_split_png, load_playback, render, run_caption, run_notes,
)


def _body(body_id, kind, color):
    return {'id': body_id, 'name': body_id, 'kind': kind, 'color': color}


def _run(index, *, status='complete', reason='', end=1.0, variable=None, events=(), z=None, earth_x=1.0):
    def point(x, y):
        return [x, y, z] if z is not None else [x, y]

    samples = []
    for step in range(5):
        time = end * step / 4
        samples.append({
            'time': time,
            'ids': ['sun', 'earth'],
            'mass': [1.0, 3e-6],
            'r': [point(0.0, 0.0), point(earth_x + 0.1 * index, time)],
            'v': [[0.0, 0.0], [0.0, 1.0]],
        })
    return {
        'index': index,
        'status': status,
        'reason': reason,
        'requested_years': 1.0,
        'duration_years': end,
        'time_start': 0.0,
        'time_end': end,
        'variable': variable,
        'bodies': [_body('sun', 'star', '#fff0d9'), _body('earth', 'planet', '#51a9ef')],
        'samples': samples,
        'events': list(events),
    }


def _playback(*runs):
    return {'schema': 'sidereel-playback-v1', 'runs': list(runs)}


def _texts(figure):
    return [artist.get_text() for artist in figure.findobj(lambda artist: hasattr(artist, 'get_text')) if artist.get_text()]


def test_playback_rejects_a_file_that_is_not_the_recording_schema():
    with pytest.raises(ValueError, match='sidereel-playback-v1'):
        load_playback({'schema': 'sidereel-experiment-v1', 'runs': []})


def test_transport_selects_samples_and_does_not_change_the_recording():
    sun_mass = {'parameter': 'mass', 'body': 'sun', 'value': 1.3, 'unit': 'solar_mass'}
    playback = _playback(_run(0, variable=sun_mass), _run(1, variable=dict(sun_mass, value=1.4)))
    before = json.dumps(playback)
    player = Player(playback)
    assert run_caption(player.current) == 'Sun mass: 130%'
    player.years_per_second = 1
    player.runs_per_second = 2
    player.advance(0.25)
    assert player.run_index == 0
    assert player.time == pytest.approx(0.25)
    player.advance(1.25)
    assert player.run_index == 1
    assert player.time == pytest.approx(0.0)
    assert json.dumps(playback) == before


def test_pause_holds_the_playhead_and_the_last_run_stops():
    player = Player(_playback(_run(0), _run(1)))
    player.pause()
    player.advance(2)
    assert player.time == 0
    player.play()
    player.years_per_second = 100
    player.runs_per_second = 10
    player.advance(10)
    assert player.paused
    assert player.run_index == 1


def test_reset_checkbox_keeps_the_timestamp_when_it_is_off():
    player = Player(_playback(_run(0), _run(1)))
    player.reset_each_run = False
    player.years_per_second = 1
    player.runs_per_second = 2
    player.advance(1.5)
    assert player.run_index == 1
    assert player.time == pytest.approx(1.0)


def test_run_slider_switches_member_without_recalculating():
    playback = _playback(_run(0), _run(1))
    player = Player(playback)
    player.time = 0.4
    player.set_run(1)
    assert player.run_index == 1
    assert player.left_index == 1
    assert player.time == 0
    assert playback['runs'][1]['samples'][-1]['r'][1][0] == pytest.approx(1.1)


def test_a_finished_collision_is_not_an_early_end_and_a_halt_is():
    collision = {'time': 0.2, 'kind': 'collision', 'bodies': ['venus', 'earth'], 'message': 'Venus and Earth met'}
    finished = _run(0, events=[collision])
    halted = _run(1, status='halted', reason='step below the floor', end=0.3)
    assert equal_time(finished)
    assert not any(note.startswith('Ended early') for note in run_notes(finished))
    assert any(note.startswith('Collision at 0.2 yr') for note in run_notes(finished))
    assert not equal_time(halted)
    assert run_notes(halted)[0].startswith('Ended early (halted)')


def test_live_view_is_one_plot_without_axis_values_and_a_duration():
    player = Player(_playback(_run(0, variable={'parameter': 'mass', 'body': 'sun', 'value': 1, 'unit': 'solar_mass'})))
    player.time = 1
    figure = Figure()
    render(player, figure)
    assert player.view == VIEW_LIVE
    assert len(figure.axes) == 1
    axis = figure.axes[0]
    assert axis.get_xticks().size == 0
    assert axis.get_yticks().size == 0
    assert axis.title.get_color() == GOLD
    assert axis.title.get_text() == 'Sun mass: 100%'
    assert any(text.startswith('Duration') and 'yr' in text for text in _texts(figure))
    assert list(axis.lines[1].get_xdata()) == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])


def test_three_dimensional_playback_draws_the_horizontal_projection():
    player = Player(_playback(_run(0, z=5.0)))
    player.time = 1
    figure = Figure()
    render(player, figure)
    earth = figure.axes[0].lines[1]
    assert list(earth.get_xdata()) == pytest.approx([1.0] * 5)
    assert max(earth.get_ydata()) == pytest.approx(1.0)
    assert all(value != pytest.approx(5.0) for value in list(earth.get_xdata()) + list(earth.get_ydata()))


def test_overlay_omits_a_shorter_run_and_exports_it_as_not_equal_time(tmp_path):
    finished = _run(0, variable={'parameter': 'mass', 'body': 'sun', 'value': 1.0, 'unit': 'solar_mass'})
    other = _run(1, variable={'parameter': 'mass', 'body': 'sun', 'value': 1.1, 'unit': 'solar_mass'})
    halted = _run(2, status='cancelled', reason='cancelled', end=0.2,
                  variable={'parameter': 'mass', 'body': 'sun', 'value': 1.2, 'unit': 'solar_mass'})
    player = Player(_playback(finished, other, halted))
    player.set_view(VIEW_OVERLAY)
    player.time = 1
    figure = Figure()
    render(player, figure)
    labels = figure.axes[0].get_legend().get_texts()
    assert [text.get_text() for text in labels] == ['Sun mass: 100%', 'Sun mass: 110%']
    assert any('Not an equal-time result' in text and '120%' in text for text in _texts(figure))
    png = tmp_path / 'overlay.png'
    svg = tmp_path / 'overlay.svg'
    csv_path = tmp_path / 'overlay.csv'
    export_overlay_png(player, png)
    export_overlay_svg(player, svg)
    export_overlay_csv(player, csv_path)
    assert png.read_bytes().startswith(b'\x89PNG')
    assert b'<svg' in svg.read_bytes()
    rows = csv_path.read_text(encoding='utf-8').splitlines()
    assert rows[0].split(',')[:3] == ['run_index', 'status', 'equal_time']
    assert any(row.split(',')[2] == 'no' and 'Ended early' in row for row in rows[1:])
    assert any(row.split(',')[2] == 'yes' for row in rows[1:])


def test_inner_outer_limits_stay_fixed_and_export(tmp_path):
    player = Player(_playback(_run(0, earth_x=10.0), _run(1, earth_x=12.0)))
    player.set_view(VIEW_INNER_OUTER)
    player.time = 1
    figure = Figure()
    render(player, figure)
    assert [axis.get_xlim() for axis in figure.axes] == [(-INNER_LIMIT_AU, INNER_LIMIT_AU), (-OUTER_LIMIT_AU, OUTER_LIMIT_AU)]
    assert [axis.get_ylim() for axis in figure.axes] == [(-INNER_LIMIT_AU, INNER_LIMIT_AU), (-OUTER_LIMIT_AU, OUTER_LIMIT_AU)]
    player.set_run(1)
    player.time = 1
    render(player, figure)
    assert figure.axes[0].get_xlim() == (-INNER_LIMIT_AU, INNER_LIMIT_AU)
    assert any(text == 'Inner solar system' for text in _texts(figure))
    assert any(text == 'Outer solar system' for text in _texts(figure))
    png = tmp_path / 'split.png'
    gif = tmp_path / 'split.gif'
    export_split_png(player, png)
    export_split_gif(player, gif)
    assert png.read_bytes().startswith(b'\x89PNG')
    assert gif.read_bytes().startswith(b'GIF')
    assert player.run_index == 1


def test_comparison_splits_one_chosen_run_from_another():
    player = Player(_playback(
        _run(0, variable={'parameter': 'mass', 'body': 'sun', 'value': 1.0, 'unit': 'solar_mass'}),
        _run(1, variable={'parameter': 'mass', 'body': 'sun', 'value': 2.0, 'unit': 'solar_mass'})))
    player.comparing = True
    player.set_right(1)
    player.time = 1
    figure = Figure()
    render(player, figure)
    assert len(figure.axes) == 2
    assert figure.axes[0].title.get_text().startswith('Left ·')
    assert figure.axes[1].title.get_text().startswith('Right ·')
    assert '100%' in figure.axes[0].title.get_text()
    assert '200%' in figure.axes[1].title.get_text()
    assert figure.axes[0].get_xticks().size == 0


def test_player_window_opens_a_file_on_live_view_with_the_control_board(tmp_path):
    import os
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PyQt6.QtWidgets import QApplication
    from simulations.star_system.player_window import PlayerShell, launch
    playback = _playback(
        _run(0, variable={'parameter': 'mass', 'body': 'sun', 'value': 1.0, 'unit': 'solar_mass'}),
        _run(1, variable={'parameter': 'mass', 'body': 'sun', 'value': 1.1, 'unit': 'solar_mass'}))
    path = tmp_path / 'sweep.sidereel'
    path.write_text(json.dumps(playback), encoding='utf-8')
    application = QApplication.instance() or QApplication([])
    window = PlayerShell(path)
    window.timer.stop()
    window.player.pause()
    window.show()
    application.processEvents()
    assert window.view_combo.currentText() == 'Live view'
    assert window.view_combo.itemText(2) == 'Inner/outer split view (name TBD)'
    assert window.reset_each.isChecked()
    assert window.pause_button.text() == 'Pause'
    assert window.run_slider.isVisible()
    assert window.run_slider.maximum() == 1
    assert window.compare_check.text() == 'Compare'
    assert window.within_speed.suffix() == ' yr/s'
    assert window.between_speed.suffix() == ' runs/s'
    window.run_slider.setValue(1)
    application.processEvents()
    assert window.player.run_index == 1
    window.compare_check.setChecked(True)
    application.processEvents()
    assert window.player.comparing
    assert window.player.right_index == 1
    window.close()
    home = PlayerShell(None)
    home.show()
    application.processEvents()
    assert home.player is None
    assert home.stack.currentWidget() is home.home
    assert home.preset_list.count() == 8
    labels = home.findChildren(type(home.preset_summary))
    assert any('does not calculate' in label.text() for label in labels)
    home.close()
    assert launch(['--help']) == 0
    assert launch([str(tmp_path / 'notes.json')]) == 2


def test_cli_help_does_not_import_the_qt_desktop():
    import sys
    from simulations.star_system.__main__ import main
    sys.modules.pop('simulations.star_system.desktop', None)
    with pytest.raises(SystemExit) as caught:
        main(['--help'])
    assert caught.value.code == 0
    assert 'simulations.star_system.desktop' not in sys.modules


def test_module_entry_sends_a_playback_path_to_the_player(tmp_path):
    from simulations.star_system.__main__ import main
    notes = tmp_path / 'notes.json'
    notes.write_text('{}', encoding='utf-8')
    assert main([str(notes)]) == 2


def test_player_source_does_not_call_the_integrator():
    source = Path(__import__('simulations.star_system.player', fromlist=['player']).__file__).read_text(encoding='utf-8')
    assert 'Calculation' not in source
    assert 'build_experiment' not in source
    assert 'from .recording' not in source
