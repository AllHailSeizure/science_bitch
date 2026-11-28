"""Qt shell that plays a finished ``.sidereel`` file.

``sidereel`` with no arguments opens this window so a playback file can be
chosen. ``sidereel <file.sidereel>`` plays that file. Nothing here builds or
integrates an experiment.
"""
import sys
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QSlider, QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
)
from .player import (
    OBSERVABLE_LABELS, OBSERVABLES, VIEW_CHOICES, Player, export_overlay_csv, export_overlay_png,
    export_overlay_svg, export_split_gif, export_split_png, load_playback, render, run_caption, run_notes,
)
from .preset_sweeps import PRESETS

USAGE = """\
sidereel [file.sidereel]
sidereel --new [experiment.json]
sidereel --build experiment.json

With a .sidereel file, the player opens and plays that finished recording.
With no file, the player opens so you can choose one. It does not calculate.

Default view: Live view. A single top-down plot, no axis values, orbit trails
as lines, and the duration underneath.
Other views: Overlay, and the inner/outer split view (name TBD).
Control board: speed within a run, speed between runs, reset each run, pause,
and a comparison that puts one chosen run on the left and another on the right.
A run slider switches the recorded member without calculating it.
"""

STYLE = """
QWidget { background: #0d1725; color: #e7eff7; font-family: 'DejaVu Sans'; font-size: 13px; }
QLabel[role='title'] { font-size: 32px; font-weight: 600; }
QLabel[role='muted'] { color: #a9bdd0; }
QPushButton { background: #1a2c40; border: 1px solid #31485f; border-radius: 6px; padding: 7px 12px; }
QPushButton:hover { background: #28445c; border-color: #80cfcb; }
QComboBox, QDoubleSpinBox { background: #142437; border: 1px solid #344b62; border-radius: 4px; padding: 4px; }
QListWidget { background: #111e2e; border: 1px solid #263d52; border-radius: 6px; padding: 4px; }
QListWidget::item { padding: 8px; }
QListWidget::item:selected { background: #294e5e; }
"""


def launch(arguments):
    if any(argument in ('-h', '--help') for argument in arguments):
        print(USAGE)
        return 0
    flags = [argument for argument in arguments if argument.startswith('-')]
    files = [argument for argument in arguments if not argument.startswith('-')]
    if flags or len(files) > 1:
        print(USAGE, file=sys.stderr)
        return 2
    path = None if not files else Path(files[0])
    if path is not None:
        if path.suffix != '.sidereel':
            print(
                f'{path} is not a .sidereel playback file. '
                'Build an experiment with sidereel --build, then open the .sidereel file.',
                file=sys.stderr)
            return 2
        try:
            load_playback(path)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    application = QApplication.instance() or QApplication([sys.argv[0]])
    application.setApplicationName('Sidereel')
    application.setOrganizationName('Sidereel')
    window = PlayerShell(path)
    window.show()
    return application.exec()


class PlayerShell(QMainWindow):
    """Home list of preset sweep files, or the player for one playback."""

    def __init__(self, path=None):
        super().__init__()
        self.setStyleSheet(STYLE)
        self.setWindowTitle('Sidereel')
        self.resize(1280, 860)
        self.setMinimumSize(960, 680)
        self.player = None
        self._syncing = False
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.home = self._home()
        self.board = self._board()
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.board)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        if path is not None:
            self.open_playback(path)

    def _home(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 28, 36, 28)
        title = QLabel('Sidereel')
        title.setProperty('role', 'title')
        layout.addWidget(title)
        note = QLabel(
            'Open a finished .sidereel playback file. Experiments are built with '
            'sidereel --build. This window does not calculate them.')
        note.setWordWrap(True)
        note.setProperty('role', 'muted')
        layout.addWidget(note)
        self.open_button = QPushButton('Open playback file')
        self.open_button.clicked.connect(self._choose_file)
        layout.addWidget(self.open_button)
        layout.addWidget(QLabel('Preset sweep files'))
        self.preset_summary = QLabel('')
        self.preset_summary.setWordWrap(True)
        self.preset_command = QLabel('')
        self.preset_command.setProperty('role', 'muted')
        self.preset_list = QListWidget()
        self.preset_list.currentRowChanged.connect(self._describe_preset)
        for item in PRESETS:
            entry = QListWidgetItem(f'{item.title}  —  experiments/{item.key}.json')
            entry.setData(Qt.ItemDataRole.UserRole, item.key)
            entry.setToolTip(item.summary)
            self.preset_list.addItem(entry)
        layout.addWidget(self.preset_list, 1)
        layout.addWidget(self.preset_summary)
        layout.addWidget(self.preset_command)
        if self.preset_list.currentRow() < 0:
            self.preset_list.setCurrentRow(0)
        else:
            self._describe_preset(self.preset_list.currentRow())
        return page

    def _describe_preset(self, row):
        if row < 0:
            return
        item = PRESETS[row]
        self.preset_summary.setText(item.summary)
        self.preset_command.setText(f'sidereel --build experiments/{item.key}.json')

    def _board(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        self.figure = Figure(figsize=(9.6, 5.4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        self.view_combo = QComboBox()
        self.view_combo.setObjectName('view_combo')
        for key, label in VIEW_CHOICES:
            self.view_combo.addItem(label, key)
        self.view_combo.currentIndexChanged.connect(self._change_view)
        controls.addWidget(self.view_combo)
        self.observable_combo = QComboBox()
        self.observable_combo.setObjectName('observable_combo')
        for key in OBSERVABLES:
            self.observable_combo.addItem(OBSERVABLE_LABELS[key], key)
        self.observable_combo.setCurrentIndex(OBSERVABLES.index('radius'))
        self.observable_combo.currentIndexChanged.connect(self._change_observable)
        controls.addWidget(self.observable_combo)
        self.pause_button = QPushButton('Pause')
        self.pause_button.setObjectName('pause_button')
        self.pause_button.clicked.connect(self._toggle_pause)
        controls.addWidget(self.pause_button)
        controls.addWidget(QLabel('Within the run'))
        self.within_speed = self._speed_box(1.0, ' yr/s')
        self.within_speed.setObjectName('within_speed')
        self.within_speed.valueChanged.connect(self._change_within)
        controls.addWidget(self.within_speed)
        controls.addWidget(QLabel('Between runs'))
        self.between_speed = self._speed_box(1.0, ' runs/s')
        self.between_speed.setObjectName('between_speed')
        self.between_speed.valueChanged.connect(self._change_between)
        controls.addWidget(self.between_speed)
        self.reset_each = QCheckBox('Reset each run')
        self.reset_each.setObjectName('reset_each')
        self.reset_each.setChecked(True)
        self.reset_each.toggled.connect(self._change_reset)
        controls.addWidget(self.reset_each)
        layout.addLayout(controls)
        slider_row = QHBoxLayout()
        self.run_caption = QLabel('Run')
        slider_row.addWidget(self.run_caption)
        self.run_slider = QSlider(Qt.Orientation.Horizontal)
        self.run_slider.setObjectName('run_slider')
        self.run_slider.valueChanged.connect(self._slide_run)
        slider_row.addWidget(self.run_slider, 1)
        layout.addLayout(slider_row)
        compare = QHBoxLayout()
        self.compare_check = QCheckBox('Compare')
        self.compare_check.setObjectName('compare_check')
        self.compare_check.toggled.connect(self._change_compare)
        compare.addWidget(self.compare_check)
        compare.addWidget(QLabel('Left'))
        self.left_run = QComboBox()
        self.left_run.setObjectName('left_run')
        self.left_run.currentIndexChanged.connect(self._change_left)
        compare.addWidget(self.left_run)
        compare.addWidget(QLabel('Right'))
        self.right_run = QComboBox()
        self.right_run.setObjectName('right_run')
        self.right_run.currentIndexChanged.connect(self._change_right)
        compare.addWidget(self.right_run)
        self.export_png = QPushButton('Export overlay PNG')
        self.export_png.clicked.connect(lambda: self._export('overlay-png'))
        self.export_svg = QPushButton('Export overlay SVG')
        self.export_svg.clicked.connect(lambda: self._export('overlay-svg'))
        self.export_csv = QPushButton('Export overlay CSV')
        self.export_csv.clicked.connect(lambda: self._export('overlay-csv'))
        self.export_split_png = QPushButton('Export split PNG')
        self.export_split_png.clicked.connect(lambda: self._export('split-png'))
        self.export_gif = QPushButton('Export split GIF')
        self.export_gif.clicked.connect(lambda: self._export('split-gif'))
        for button in (self.export_png, self.export_svg, self.export_csv, self.export_split_png, self.export_gif):
            compare.addWidget(button)
        layout.addLayout(compare)
        scrub = QHBoxLayout()
        scrub.addWidget(QLabel('Recorded state'))
        self.state_slider = QSlider(Qt.Orientation.Horizontal)
        self.state_slider.valueChanged.connect(self._scrub_state)
        scrub.addWidget(self.state_slider, 1)
        self.state_label = QLabel('')
        scrub.addWidget(self.state_label)
        self.event_combo = QComboBox()
        self.event_combo.setMinimumWidth(220)
        self.event_combo.activated.connect(self._scrub_event)
        scrub.addWidget(self.event_combo)
        layout.addLayout(scrub)
        ancestry = QHBoxLayout()
        ancestry.addWidget(QLabel('Body / ancestry'))
        self.body_combo = QComboBox()
        self.body_combo.currentIndexChanged.connect(self._select_body)
        ancestry.addWidget(self.body_combo)
        self.history_combo = QComboBox()
        self.history_combo.activated.connect(self._scrub_history)
        ancestry.addWidget(self.history_combo, 1)
        layout.addLayout(ancestry)
        self.inspection = QTextEdit()
        self.inspection.setReadOnly(True)
        self.inspection.setMaximumHeight(125)
        layout.addWidget(self.inspection)
        self._inspection_run = None
        self.status = QLabel('')
        self.status.setWordWrap(True)
        self.status.setProperty('role', 'muted')
        layout.addWidget(self.status)
        return page

    def _speed_box(self, value, suffix):
        box = QDoubleSpinBox()
        box.setRange(0.05, 100.0)
        box.setDecimals(2)
        box.setSingleStep(0.25)
        box.setValue(value)
        box.setSuffix(suffix)
        return box

    def _choose_file(self):
        path, _filter = QFileDialog.getOpenFileName(self, 'Open playback', '', 'Playback (*.sidereel)')
        if path:
            self.open_playback(path)

    def open_playback(self, path):
        try:
            playback = load_playback(path)
        except ValueError as exc:
            QMessageBox.warning(self, 'Sidereel', str(exc))
            return
        self.player = Player(playback)
        self._inspection_run = None
        self.setWindowTitle(f'Sidereel — {Path(path).name}')
        self.player.set_years_per_second(float(self.within_speed.value()))
        self.player.set_runs_per_second(float(self.between_speed.value()))
        self.player.reset_each_run = self.reset_each.isChecked()
        self._fill_runs()
        self.stack.setCurrentWidget(self.board)
        self._redraw()
        self.timer.start()

    def _fill_runs(self):
        self._syncing = True
        try:
            count = len(self.player.runs)
            self.run_slider.setMaximum(max(count - 1, 0))
            self.run_slider.setValue(0)
            several = count > 1
            self.run_slider.setVisible(several)
            self.run_caption.setVisible(several)
            self.left_run.clear()
            self.right_run.clear()
            for run in self.player.runs:
                label = run_caption(run)
                self.left_run.addItem(label)
                self.right_run.addItem(label)
            self.right_run.setCurrentIndex(self.player.right_index)
            self.view_combo.setCurrentIndex(0)
            self.pause_button.setText('Pause')
            self.player.reset_each_run = True
            self.reset_each.setChecked(True)
            self.player.comparing = False
            self.compare_check.setChecked(False)
        finally:
            self._syncing = False

    def _tick(self):
        if self.player is None or self.player.paused:
            return
        self.player.advance(self.timer.interval() / 1000.0)
        self._sync_transport()
        self._redraw()

    def _sync_transport(self):
        self._syncing = True
        try:
            self.run_slider.setValue(self.player.run_index)
            self.left_run.setCurrentIndex(self.player.left_index)
            self.pause_button.setText('Play' if self.player.paused else 'Pause')
        finally:
            self._syncing = False

    def _redraw(self):
        if self.player is None:
            return
        try:
            render(self.player, self.figure)
        except ValueError as exc:
            self.player.comparing = False
            self.compare_check.setChecked(False)
            self.status.setText(str(exc))
            return
        self._refresh_inspection()
        self.canvas.draw_idle()
        notes = '  '.join(run_notes(self.player.current))
        self.status.setText(
            f'{run_caption(self.player.current)} · {self.player.current["status"]} · Time {self.player.time:.6g} yr. Playback uses the recording and does not integrate. {notes}'.strip())

    def _events(self):
        run = self.player.current
        events = run.get('formation_events', []) + run.get('events', [])
        if run.get('branch_point'):
            events = events + [dict(run['branch_point'], kind='branch', message='Disclosed branch change')]
        return sorted(events, key=lambda item: item['time'])

    def _refresh_inspection(self):
        import json
        from .observatory import contributions, history_indices
        run = self.player.current
        self._syncing = True
        try:
            if self._inspection_run is not run:
                self._inspection_run = run
                self.state_slider.setMaximum(len(run['samples']) - 1)
                self.body_combo.clear()
                self.body_combo.addItem('Choose a body', None)
                names = {body['id']: body.get('name', body['id']) for body in run.get('bodies', [])}
                for sample in run['samples']:
                    for key in sample['ids']:
                        names.setdefault(key, key)
                for key, name in names.items():
                    self.body_combo.addItem(f'{name} ({key})', key)
                self.event_combo.clear()
                self.event_combo.addItem('Choose a recorded event')
                for event in self._events():
                    self.event_combo.addItem(f"{event['time']:.6g} yr · {event['kind']} · {event.get('message', '')}")
            index = self.player.sample_index
            if index is None:
                index = max(i for i, sample in enumerate(run['samples']) if sample['time'] <= self.player.time)
            self.state_slider.setValue(index)
            sample = run['samples'][index]
            self.state_label.setText(f"{sample['time']:.6g} yr · {sample.get('phase') or 'sample'} [{index + 1}]")
            key = self.player.selected_body
            self.history_combo.clear()
            self.history_combo.addItem('Body history: existence → removal → following state', None)
            if key:
                for i in history_indices(run, key):
                    state = run['samples'][i]
                    self.history_combo.addItem(f"{state['time']:.6g} yr · {state.get('phase') or 'sample'} · "
                                               + ('present' if key in state['ids'] else 'absent after removal'), i)
            details = []
            if key:
                amounts = contributions(run, key)
                details.append('Parcel contributions (solar masses): ' + (', '.join(
                    f"{item['parcel_id']}: {item['mass']:.8g}" for item in amounts) or 'not recorded'))
                removed = next((body for body in run.get('removed_bodies', []) if body['id'] == key), None)
                details.append('Body status: ' + (removed['status'] if removed else
                               ('surviving' if key in run['samples'][-1]['ids'] else 'removed')))
                for event in self._events():
                    if key in event.get('parent_ids', event.get('bodies', [])):
                        details.append('Removal / event: ' + json.dumps(event, sort_keys=True))
            for side, member in ([('Left', self.player.runs[self.player.left_index]),
                                  ('Right', self.player.runs[self.player.right_index])] if self.player.comparing
                                 else [('Run', run)]):
                details.append(side + ' assumptions: ' + '; '.join(member.get('assumptions', ['not recorded'])))
                if member.get('branch_point'):
                    details.append(side + ' branch checkpoint state and changed state: '
                                   + json.dumps(member['branch_point'], sort_keys=True))
                details.append(side + ' provenance: ' + json.dumps(member.get('provenance', {
                    'setup_digest': member.get('setup_digest', 'not recorded')}), sort_keys=True))
            if self.player.comparing:
                start, end = self.player.interval()
                left, right = self.player.runs[self.player.left_index], self.player.runs[self.player.right_index]
                parent = left.get('provenance', {}).get('parent_checkpoint_id')
                common = parent and parent == right.get('provenance', {}).get('parent_checkpoint_id')
                details.append(f'Shared interval: {start:.6g}–{end:.6g} yr. Common checkpoint: '
                               + (str(parent) if common else 'not recorded / different checkpoints'))
                details.append('Starting states: ' + json.dumps([member['samples'][0] for member in (left, right)]))
            self.inspection.setPlainText('\n'.join(details))
        finally:
            self._syncing = False

    def _scrub_state(self, index):
        if self._syncing or self.player is None:
            return
        self.player.seek_sample(index)
        self._sync_transport()
        self._redraw()

    def _scrub_event(self, index):
        if index <= 0:
            return
        self.player.seek(float(self._events()[index - 1]['time']))
        self._sync_transport()
        self._redraw()
        import json
        self.inspection.append(json.dumps(self._events()[index - 1], sort_keys=True))

    def _select_body(self):
        if self._syncing or self.player is None:
            return
        self.player.selected_body = self.body_combo.currentData()
        self._redraw()

    def _scrub_history(self, index):
        sample = self.history_combo.itemData(index)
        if sample is not None:
            self._scrub_state(sample)

    def _change_view(self):
        if self._syncing or self.player is None:
            return
        self.player.set_view(self.view_combo.currentData())
        self._redraw()

    def _change_observable(self):
        if self._syncing or self.player is None:
            return
        self.player.set_observable(self.observable_combo.currentData())
        self._redraw()

    def _toggle_pause(self):
        if self.player is None:
            return
        if self.player.paused:
            self.player.play()
        else:
            self.player.pause()
        self.pause_button.setText('Play' if self.player.paused else 'Pause')

    def _change_within(self, value):
        if self.player is not None:
            self.player.set_years_per_second(float(value))

    def _change_between(self, value):
        if self.player is not None:
            self.player.set_runs_per_second(float(value))

    def _change_reset(self, checked):
        if self._syncing or self.player is None:
            return
        self.player.reset_each_run = bool(checked)

    def _slide_run(self, value):
        if self._syncing or self.player is None:
            return
        self.player.set_run(int(value))
        self._syncing = True
        try:
            self.left_run.setCurrentIndex(self.player.left_index)
        finally:
            self._syncing = False
        self._redraw()

    def _change_left(self, index):
        if self._syncing or self.player is None or index < 0:
            return
        self.player.set_run(index)
        self._syncing = True
        try:
            self.run_slider.setValue(self.player.run_index)
        finally:
            self._syncing = False
        self._redraw()

    def _change_right(self, index):
        if self._syncing or self.player is None or index < 0:
            return
        self.player.set_right(index)
        self._redraw()

    def _change_compare(self, checked):
        if self._syncing or self.player is None:
            return
        self.player.comparing = bool(checked)
        self._redraw()

    def _export(self, kind):
        if self.player is None:
            return
        filters = {
            'overlay-png': ('Export overlay PNG', 'PNG (*.png)', '.png', export_overlay_png),
            'overlay-svg': ('Export overlay SVG', 'SVG (*.svg)', '.svg', export_overlay_svg),
            'overlay-csv': ('Export overlay CSV', 'CSV (*.csv)', '.csv', export_overlay_csv),
            'split-png': ('Export split PNG', 'PNG (*.png)', '.png', export_split_png),
            'split-gif': ('Export split GIF', 'GIF (*.gif)', '.gif', export_split_gif),
        }
        title, pattern, suffix, writer = filters[kind]
        path, _filter = QFileDialog.getSaveFileName(self, title, '', pattern)
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != suffix:
            destination = destination.with_suffix(suffix)
        try:
            writer(self.player, destination)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Sidereel', str(exc))
