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
    QSlider, QStackedWidget, QVBoxLayout, QWidget,
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
        render(self.player, self.figure)
        self.canvas.draw_idle()
        notes = '  '.join(run_notes(self.player.current))
        self.status.setText(
            f'{run_caption(self.player.current)}. Playback uses the recording and does not integrate. {notes}'.strip())

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
