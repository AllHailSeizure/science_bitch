"""Play a finished ``.sidereel`` recording.

The player selects recorded samples. It does not call the integrator. Live view
is the default: one top-down plot, no axis values, orbit trails as lines, and
the duration underneath. Overlay and the inner/outer split view are optional.
A third position component is dropped; that is the top-down projection, not a
perspective render.
"""
import csv
from copy import copy
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

PLAYBACK_SCHEMA = 'sidereel-playback-v1'
VIEW_LIVE = 'live'
VIEW_OVERLAY = 'overlay'
VIEW_INNER_OUTER = 'inner-outer'
VIEW_CHOICES = (
    (VIEW_LIVE, 'Live view'),
    (VIEW_OVERLAY, 'Overlay'),
    (VIEW_INNER_OUTER, 'Inner/outer split view (name TBD)'),
)
OBSERVABLES = ('trajectory', 'radius', 'speed')
OBSERVABLE_LABELS = {
    'trajectory': 'Trajectories',
    'radius': 'Heliocentric distance',
    'speed': 'Speed',
}
# Inner panel covers Mercury through Mars. Outer panel covers Jupiter through Neptune.
# The limits stay fixed so one run of a sweep can be compared with another.
INNER_LIMIT_AU = 2.0
OUTER_LIMIT_AU = 35.0
BACKGROUND = '#070d18'
GOLD = '#e6c36a'
DURATION = '#d5dde8'
NOTE = '#dfbd80'
MUTED = '#9eb2c6'
_COLLISION_KINDS = ('collision', 'merger', 'graze', 'fragment', 'absorption', 'unmodeled')
_PALETTE = ('#8ecae6', '#f4a261', '#e9c46a', '#90be6d', '#f2cc8f', '#bde0fe', '#ffb4a2', '#cdb4db')


def load_playback(source):
    """Validate a ``sidereel-playback-v1`` document or a path to one."""
    if isinstance(source, (str, Path)):
        try:
            document = json.loads(Path(source).read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f'Invalid playback file: {exc}') from exc
    elif isinstance(source, dict):
        document = source
    else:
        raise ValueError('Playback must be a .sidereel path or a document.')
    if document.get('schema') != PLAYBACK_SCHEMA:
        raise ValueError(f'Unsupported playback schema. Expected {PLAYBACK_SCHEMA}.')
    runs = document.get('runs')
    if type(runs) is not list or not runs:
        raise ValueError('A playback file needs at least one run.')
    for run in runs:
        if run.get('status') not in ('complete', 'halted', 'cancelled', 'stopped', 'running'):
            raise ValueError('Each run needs a status of complete, halted, cancelled, stopped, or running.')
        samples = run.get('samples')
        if type(samples) is not list or not samples:
            raise ValueError('Each run needs recorded samples.')
        for sample in samples:
            if 'time' not in sample or 'ids' not in sample or 'r' not in sample:
                raise ValueError('Each sample needs time, ids, and positions.')
            rows = sample['r']
            if not rows or len(rows) != len(sample['ids']) or len(rows[0]) not in (2, 3):
                raise ValueError('Sample positions must align with ids and have two or three components.')
    return document


def equal_time(run):
    """A finished run can sit on the overlay with the others.

    A halt, a cancellation, or any other status that is not complete is a
    shorter member. A collision logged on a finished run is not that kind of
    early end.
    """
    return run.get('status') == 'complete'


def run_notes(run):
    """Collision marks, and an early-end mark when the recording itself stopped."""
    notes = []
    if run.get('status') == 'running':
        notes.append('Recorded calculation in progress (snapshot)')
    elif not equal_time(run):
        reason = (run.get('reason') or '').strip()
        status = run.get('status') or 'incomplete'
        notes.append(f'Ended early ({status}): {reason}' if reason else f'Ended early ({status})')
    for event in run.get('events') or ():
        if event.get('kind') not in _COLLISION_KINDS:
            continue
        message = event.get('message') or event.get('kind')
        notes.append(f'Collision at {float(event.get("time", 0)):.4g} yr: {message}')
    return tuple(notes)


def run_caption(run):
    """Gold run label, including the swept value when the file has one."""
    variable = run.get('variable') or None
    index = run.get('index', 0)
    if not variable:
        return f'Run {int(index) + 1}'
    parameter = variable.get('parameter') or 'value'
    body = str(variable.get('body') or 'body')
    value = variable.get('value')
    unit = variable.get('unit') or ''
    if parameter == 'mass' and body == 'sun' and isinstance(value, (int, float)) and math.isfinite(value):
        percent = value * 100.0
        text = f'{percent:.0f}%' if abs(percent - round(percent)) < 1e-6 else f'{percent:.4g}%'
        return f'Sun mass: {text}'
    if parameter == 'mass':
        shown = f'{float(value):.6g} solar masses'
    elif parameter == 'distance':
        shown = f'{float(value):.6g} AU'
    elif parameter == 'speed' and unit == 'AU/Julian_year':
        shown = f'{float(value):.4g} AU/yr'
    elif parameter == 'speed':
        shown = f'{float(value):.4g}× catalog'
    else:
        shown = f'{float(value):.6g} {unit}'.strip()
    return f'{body} {parameter}: {shown}'


class Player:
    """Transport over recorded runs. Speeds choose samples; they do not integrate."""

    def __init__(self, playback):
        self.playback = load_playback(playback)
        self.runs = self.playback['runs']
        self.view = VIEW_LIVE
        self.observable = 'radius'
        self.run_index = 0
        self.time = float(self.runs[0]['time_start'])
        self.paused = False
        self.years_per_second = 1.0
        self.runs_per_second = 1.0
        self.reset_each_run = True
        self.comparing = False
        self.left_index = 0
        self.right_index = 1 if len(self.runs) > 1 else 0
        self._dwell = 0.0
        self.selected_body = None
        self.sample_index = None
        self._comparison = None
        self._comparison_key = None

    @property
    def current(self):
        return self.runs[self.run_index]

    def comparison(self):
        from .observatory import RecordedComparison
        key = (self.left_index, self.right_index)
        if self._comparison_key != key:
            self._comparison = RecordedComparison(self.runs[key[0]], self.runs[key[1]])
            self._comparison_key = key
        return self._comparison

    def interval(self):
        if self.comparing:
            comparison = self.comparison()
            return comparison.common_start, comparison.common_end
        return float(self.current['time_start']), float(self.current['time_end'])

    def seek(self, time):
        if not math.isfinite(time):
            raise ValueError('Playback time must be finite.')
        start, end = self.interval()
        self.time = min(max(float(time), start), end)
        self.sample_index = None
        self.pause()
        if self.comparing:
            self.comparison().seek(self.time)

    def seek_sample(self, index):
        if index < 0 or index >= len(self.current['samples']):
            raise ValueError('Sample index is outside the recording.')
        self.seek(float(self.current['samples'][index]['time']))
        if not self.comparing:
            self.sample_index = index

    def set_view(self, view):
        if view not in {key for key, _label in VIEW_CHOICES}:
            raise ValueError(f'Unknown view {view!r}.')
        self.view = view

    def set_observable(self, name):
        if name not in OBSERVABLES:
            raise ValueError(f'Unknown observable {name!r}.')
        self.observable = name

    def set_years_per_second(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Speed within a run must be a positive finite number of years per second.')
        self.years_per_second = float(value)

    def set_runs_per_second(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Speed between runs must be a positive finite number of runs per second.')
        self.runs_per_second = float(value)

    def set_run(self, index):
        self._enter(self._index(index))
        self._dwell = 0.0

    def set_right(self, index):
        self.right_index = self._index(index)

    def pause(self):
        self.paused = True

    def play(self):
        self.paused = False

    def advance(self, wall_seconds):
        """Move the playhead. At the end of a run, wait, then step to the next."""
        if isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float)) or not math.isfinite(wall_seconds) or wall_seconds < 0:
            raise ValueError('Elapsed wall seconds must be nonnegative and finite.')
        if self.paused or wall_seconds == 0:
            return
        self.set_years_per_second(self.years_per_second)
        self.set_runs_per_second(self.runs_per_second)
        self.sample_index = None
        if self.comparing:
            comparison = self.comparison()
            comparison.seek(min(max(self.time, comparison.common_start), comparison.common_end))
            comparison.play()
            self.time = comparison.advance(wall_seconds, self.years_per_second).time
            self.paused = not comparison.playing
            return
        remaining = float(wall_seconds)
        for _ in range(10000):
            if remaining <= 0:
                return
            end = float(self.current['time_end'])
            if self.time < end:
                self._dwell = 0.0
                need = (end - self.time) / self.years_per_second
                if remaining < need:
                    self.time += remaining * self.years_per_second
                    return
                self.time = end
                remaining -= need
            dwell_need = (1.0 / self.runs_per_second) - self._dwell
            if remaining < dwell_need:
                self._dwell += remaining
                return
            remaining -= max(dwell_need, 0.0)
            self._dwell = 0.0
            if self.run_index + 1 >= len(self.runs):
                self.paused = True
                return
            self._enter(self.run_index + 1)

    def _index(self, index):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError('Run index must be an integer.')
        index = int(index)
        if index < 0 or index >= len(self.runs):
            raise ValueError('Run index is outside this playback.')
        return index

    def _enter(self, index):
        self.sample_index = None
        self.selected_body = None
        self.run_index = index
        self.left_index = index
        run = self.current
        start = float(run['time_start'])
        end = float(run['time_end'])
        if self.reset_each_run:
            self.time = start
        else:
            self.time = min(max(self.time, start), end)

    def _snapshot(self):
        return (self.run_index, self.time, self.view, self.paused, self._dwell, self.left_index, self.comparing)

    def _restore(self, snapshot):
        (self.run_index, self.time, self.view, self.paused, self._dwell, self.left_index, self.comparing) = snapshot


def _render(player, figure):
    """Draw the current view onto ``figure``. Comparison splits Live view or the inner/outer split."""
    if player.comparing:
        start, end = player.interval()
        player.time = min(max(player.time, start), end)
        player.comparison().seek(player.time)
    figure.clear()
    figure.patch.set_facecolor(BACKGROUND)
    if player.comparing and player.view == VIEW_INNER_OUTER:
        _comparison_split(figure, player)
        return
    if player.comparing and player.view == VIEW_LIVE:
        _comparison_live(figure, player)
        return
    if player.view == VIEW_OVERLAY:
        _overlay(figure, player)
        return
    if player.view == VIEW_INNER_OUTER:
        _split(figure, player.current, _shown_time(player.current, player.time))
        return
    figure.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.14)
    axis = figure.add_subplot(1, 1, 1)
    _draw_run(axis, player.current, _shown_time(player.current, player.time), axes_values=False,
              title=run_caption(player.current), stamp=True)


def render(player, figure):
    from .observatory import carriers
    if player.sample_index is None or player.comparing:
        _render(player, figure)
    else:
        preview = copy(player)
        preview.runs = list(player.runs)
        preview.runs[player.run_index] = dict(
            player.current, samples=player.current['samples'][:player.sample_index + 1])
        _render(preview, figure)
    if player.selected_body is None or player.view == VIEW_OVERLAY:
        return
    runs = [player.current]
    if player.comparing:
        runs = [player.runs[player.left_index], player.runs[player.right_index]]
    for i, axis in enumerate(figure.axes):
        run = runs[i // 2 if player.view == VIEW_INNER_OUTER and player.comparing else i % len(runs)]
        samples = _samples_until(run, player.time)
        sample = samples[-1]
        if run is player.current and player.sample_index is not None and not player.comparing:
            sample = run['samples'][player.sample_index]
        points, _ = _projected(sample, _bodies(run))
        selected = set(carriers(run, sample, player.selected_body)) | {player.selected_body}
        indices = [j for j, key in enumerate(sample['ids']) if key in selected]
        if indices:
            axis.scatter(points[indices, 0], points[indices, 1], s=100, facecolors='none',
                         edgecolors='#ffffff', linewidths=2, zorder=5, gid='ancestry')


def export_overlay_png(player, path):
    _export_view(player, path, VIEW_OVERLAY)


def export_overlay_svg(player, path):
    _export_view(player, path, VIEW_OVERLAY)


def export_split_png(player, path):
    _export_view(player, path, VIEW_INNER_OUTER)


def export_overlay_csv(player, path):
    """Every sample, including runs that are not drawn as equal-time trails."""
    path = Path(path)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'run_index', 'status', 'equal_time', 'parameter', 'swept_body', 'swept_value',
            'time_years', 'body_id', 'x_au', 'y_au', 'z_au', 'radius_au', 'speed_au_per_year', 'notes'])
        for run in player.runs:
            variable = run.get('variable') or {}
            notes = ' | '.join(run_notes(run))
            value = variable.get('value')
            for sample in run['samples']:
                rows, raw = _projected(sample, _bodies(run))
                velocity = np.asarray(sample.get('v') or [], dtype=float)
                primary = _primary_row(sample, _bodies(run))
                for index, body_id in enumerate(sample['ids']):
                    speed = ''
                    if velocity.ndim == 2 and index < len(velocity) and velocity.shape[1] >= 2:
                        relative = velocity[index, :2] - velocity[primary, :2]
                        speed = f'{float(np.linalg.norm(relative)):.8g}'
                    writer.writerow([
                        run.get('index', ''),
                        run.get('status', ''),
                        'yes' if equal_time(run) else 'no',
                        variable.get('parameter', ''),
                        variable.get('body', ''),
                        '' if value is None else f'{float(value):.8g}',
                        f'{float(sample["time"]):.8g}',
                        body_id,
                        f'{rows[index, 0]:.8g}',
                        f'{rows[index, 1]:.8g}',
                        '' if raw.shape[1] == 2 else f'{float(raw[index, 2]):.8g}',
                        f'{float(np.linalg.norm(rows[index])):.8g}',
                        speed,
                        notes,
                    ])


def export_split_gif(player, path):
    """One inner/outer frame per run, at the end of that recording, then the player is restored."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image
    snapshot = player._snapshot()
    images = []
    try:
        player.comparing = False
        player.view = VIEW_INNER_OUTER
        for index, run in enumerate(player.runs):
            player.set_run(index)
            player.time = float(run['time_end'])
            figure = Figure(figsize=(10.4, 5.2), dpi=80)
            render(player, figure)
            canvas = FigureCanvasAgg(figure)
            canvas.draw()
            buffer = np.asarray(canvas.buffer_rgba())
            images.append(Image.fromarray(buffer.copy(), 'RGBA'))
    finally:
        player._restore(snapshot)
    if not images:
        raise ValueError('A GIF needs at least one run.')
    frame_ms = max(50, int(round(1000.0 / player.runs_per_second)))
    images[0].save(path, save_all=True, append_images=images[1:], duration=frame_ms, loop=0, disposal=2)


def _export_view(player, path, view):
    saved = player.view
    player.view = view
    try:
        figure = Figure(figsize=(9.6, 5.4), dpi=120)
        render(player, figure)
        figure.savefig(path, facecolor=figure.get_facecolor())
    finally:
        player.view = saved


def _comparison_live(figure, player):
    figure.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.14, wspace=0.16)
    left_axis, right_axis = figure.subplots(1, 2)
    left = player.runs[player.left_index]
    right = player.runs[player.right_index]
    _draw_run(left_axis, left, _shown_time(left, player.time), axes_values=False,
              title='Left · ' + run_caption(left), stamp=True)
    _draw_run(right_axis, right, _shown_time(right, player.time), axes_values=False,
              title='Right · ' + run_caption(right), stamp=True)


def _comparison_split(figure, player):
    figure.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.08, wspace=0.28, hspace=0.38)
    axes = figure.subplots(2, 2)
    left = player.runs[player.left_index]
    right = player.runs[player.right_index]
    _draw_pair(axes[:, 0], left, _shown_time(left, player.time), 'Left · ' + run_caption(left))
    _draw_pair(axes[:, 1], right, _shown_time(right, player.time), 'Right · ' + run_caption(right))


def _overlay(figure, player):
    if player.observable == 'trajectory':
        figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14)
        trail_axis = figure.add_subplot(1, 1, 1)
        scalar_axis = None
    else:
        figure.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.12, hspace=0.45)
        trail_axis, scalar_axis = figure.subplots(2, 1)
    _style_dark(trail_axis, axes_values=True)
    trail_axis.set_aspect('equal', adjustable='datalim')
    trail_axis.set_xlabel('AU', color=MUTED)
    trail_axis.set_ylabel('AU', color=MUTED)
    excluded = []
    total = max(len(player.runs) - 1, 1)
    labeled = False
    for index, run in enumerate(player.runs):
        if not equal_time(run):
            excluded.append(run_caption(run))
            continue
        color = _run_color(index, total)
        first = True
        when = _shown_time(run, player.time)
        for body_id in _plot_ids(run):
            xs, ys, _times = _series(run, when, body_id)
            if len(xs) < 1:
                continue
            kind = (_bodies(run).get(body_id) or {}).get('kind')
            trail_axis.plot(xs, ys, color=color, linewidth=0.45 if kind == 'particle' else 1.25,
                            label=run_caption(run) if first else None, solid_capstyle='round')
            first = False
            labeled = True
        if scalar_axis is not None:
            for body_id in _observable_ids(run):
                times, values = _scalar(run, when, body_id, player.observable)
                if times:
                    scalar_axis.plot(times, values, color=color, linewidth=1.15)
    if labeled:
        legend = trail_axis.legend(frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color(DURATION)
    if excluded:
        trail_axis.text(0.02, 0.02, 'Not an equal-time result:\n' + '\n'.join(excluded),
                        transform=trail_axis.transAxes, color=NOTE, fontsize=8, va='bottom')
    trail_axis.set_title('Overlay', color=GOLD, fontsize=13)
    if scalar_axis is not None:
        _style_dark(scalar_axis, axes_values=True)
        scalar_axis.set_xlabel('yr', color=MUTED)
        scalar_axis.set_ylabel('AU' if player.observable == 'radius' else 'AU/yr', color=MUTED)
        scalar_axis.set_title(OBSERVABLE_LABELS[player.observable], color=DURATION, fontsize=11)
    else:
        figure.text(0.5, 0.02, _duration_text(player.current, player.time), ha='center', va='bottom',
                    color=DURATION, fontsize=11, gid='duration')


def _split(figure, run, time):
    figure.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.14, wspace=0.22)
    inner, outer = figure.subplots(1, 2)
    _draw_pair((inner, outer), run, time, '')
    figure.suptitle(run_caption(run), color=GOLD, fontsize=14)
    figure.text(0.5, 0.02, _duration_text(run, time), ha='center', va='bottom', color=DURATION, fontsize=11, gid='duration')


def _draw_pair(axes, run, time, caption):
    inner, outer = axes
    _draw_run(inner, run, time, axes_values=True, title='Inner solar system', limits=INNER_LIMIT_AU, stamp=False)
    _draw_run(outer, run, time, axes_values=True, title='Outer solar system', limits=OUTER_LIMIT_AU, stamp=False)
    inner.set_title('Inner solar system', color=DURATION, fontsize=11)
    outer.set_title('Outer solar system', color=DURATION, fontsize=11)
    if caption:
        inner.text(0.5, 1.14, caption, transform=inner.transAxes, ha='center', va='bottom',
                   color=GOLD, fontsize=12, clip_on=False)


def _draw_run(axis, run, time, *, axes_values, title, limits=None, stamp=False):
    _style_dark(axis, axes_values=axes_values)
    axis.set_aspect('equal', adjustable='box' if limits is not None else 'datalim')
    bodies = _bodies(run)
    for number, body_id in enumerate(_plot_ids(run)):
        xs, ys, _times = _series(run, time, body_id)
        if not xs:
            continue
        body = bodies.get(body_id) or {}
        color = body.get('color') if isinstance(body.get('color'), str) and body.get('color') else _PALETTE[number % len(_PALETTE)]
        kind = body.get('kind')
        axis.plot(xs, ys, color=color, linewidth=0.55 if kind == 'particle' else 1.35, solid_capstyle='round')
        if body_id in _samples_until(run, time)[-1]['ids']:
            axis.scatter([xs[-1]], [ys[-1]], s=8 if kind == 'particle' else 16, color=color, zorder=3)
    notes = run_notes(run)
    if notes:
        axis.text(0.02, 0.98, '\n'.join(notes), transform=axis.transAxes, color=NOTE, fontsize=8, va='top')
    if limits is None:
        axis.margins(0.12)
        if not axes_values:
            axis.set_xticks([])
            axis.set_yticks([])
    else:
        axis.set_xlim(-limits, limits)
        axis.set_ylim(-limits, limits)
    if axes_values:
        axis.set_xlabel('AU', color=MUTED)
        axis.set_ylabel('AU', color=MUTED)
    axis.set_title(title, color=GOLD, fontsize=13, pad=8)
    if stamp:
        axis.text(0.5, -0.08, _duration_text(run, time), transform=axis.transAxes, ha='center', va='top',
                  color=DURATION, fontsize=11, clip_on=False, gid='duration')


def _style_dark(axis, *, axes_values):
    axis.set_facecolor(BACKGROUND)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color('#31485f' if axes_values else BACKGROUND)


def _duration_text(run, time):
    shown = _shown_time(run, time)
    text = f'Duration {shown:.4g} yr'
    if not equal_time(run):
        text += ' · incomplete'
    return text


def _shown_time(run, time):
    return min(max(float(time), float(run['time_start'])), float(run['time_end']))


def _run_color(index, total):
    from matplotlib import colormaps
    return colormaps['viridis'](0.15 + 0.75 * (index / total))


def _bodies(run):
    return {body['id']: body for body in run.get('bodies') or () if isinstance(body, dict) and 'id' in body}


def _plot_ids(run):
    ordered = [body['id'] for body in run.get('bodies') or () if isinstance(body, dict) and 'id' in body]
    seen = set(ordered)
    for sample in run['samples']:
        for body_id in sample['ids']:
            if body_id not in seen:
                ordered.append(body_id)
                seen.add(body_id)
    return ordered


def _observable_ids(run):
    bodies = _bodies(run)
    variable = (run.get('variable') or {}).get('body')
    if variable in bodies and bodies[variable].get('kind') not in ('star', 'particle', None):
        return (variable,)
    chosen = []
    for body in run.get('bodies') or ():
        if not isinstance(body, dict) or 'id' not in body:
            continue
        if body.get('kind') not in ('star', 'particle'):
            chosen.append(body['id'])
    return tuple(chosen) or tuple(_plot_ids(run)[:1])


def _samples_until(run, time):
    chosen = [sample for sample in run['samples'] if float(sample['time']) <= float(time) + 1e-12]
    return chosen or run['samples'][:1]


def _series(run, time, body_id):
    xs, ys, times = [], [], []
    bodies = _bodies(run)
    for sample in _samples_until(run, time):
        if body_id not in sample['ids']:
            if xs:
                break
            continue
        rows, _raw = _projected(sample, bodies)
        point = rows[list(sample['ids']).index(body_id)]
        xs.append(float(point[0]))
        ys.append(float(point[1]))
        times.append(float(sample['time']))
    return xs, ys, times


def _scalar(run, time, body_id, kind):
    times, values = [], []
    bodies = _bodies(run)
    for sample in _samples_until(run, time):
        if body_id not in sample['ids']:
            if times:
                break
            continue
        index = list(sample['ids']).index(body_id)
        rows, _raw = _projected(sample, bodies)
        if kind == 'radius':
            value = float(np.linalg.norm(rows[index]))
        else:
            velocity = np.asarray(sample.get('v') or [], dtype=float)
            if velocity.ndim != 2 or index >= len(velocity) or velocity.shape[1] < 2:
                continue
            primary = _primary_row(sample, bodies)
            value = float(np.linalg.norm(velocity[index, :2] - velocity[primary, :2]))
        times.append(float(sample['time']))
        values.append(value)
    return times, values


def _projected(sample, bodies):
    """Horizontal plane. A third component is not drawn."""
    raw = np.asarray(sample['r'], dtype=float)
    if raw.ndim != 2 or raw.shape[1] not in (2, 3) or raw.shape[0] != len(sample['ids']):
        raise ValueError('Sample positions must align with ids and have two or three components.')
    rows = raw[:, :2] - raw[_primary_row(sample, bodies), :2]
    return rows, raw


def _primary_row(sample, bodies):
    stars = [index for index, body_id in enumerate(sample['ids']) if (bodies.get(body_id) or {}).get('kind') == 'star']
    if not stars:
        return 0
    masses = sample.get('mass') or ()
    def weight(index):
        if index < len(masses):
            try:
                return float(masses[index])
            except (TypeError, ValueError):
                return 0.0
        return 0.0
    return max(stars, key=weight)
