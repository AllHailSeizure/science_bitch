"""Experiment-first desktop application using native Qt controls and the shared solver."""
from collections import deque
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
from matplotlib import get_data_path
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFontDatabase, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QScrollArea, QListWidget, QListWidgetItem, QTabWidget, QLineEdit,
    QFileDialog, QMessageBox, QFormLayout, QGroupBox, QSpinBox,
    QProgressBar, QCheckBox,
)

from .collisions import advance_prepared, mapped_selection
from .context import selection_context
from .core import System
from .desktop_panels import AUYR_TO_KMS, BodyCatalog, BodyInformation, BodyProperties, Dropdown, action, label, spin
from .experiments import ExperimentSession
from .history import MAX_HISTORY, StateFrame
from .initialization import PreparedSystem
from .persistence import load_setup, save_setup, setup_digest
from .playback import Playback
from .recording import Calculation, CalculationRequest
from .rendering import BACKGROUND, ACCENT, draw_system
from .templates import EXPERIMENTS, get_experiment, place_body, remove_body, update_body

STYLE = """
QWidget { background: #0d1725; color: #e7eff7; font-family: 'DejaVu Sans'; font-size: 13px; }
QLabel[role='title'] { font-size: 42px; font-weight: 600; }
QLabel[role='heading'] { font-size: 24px; font-weight: 600; }
QLabel[role='subheading'] { font-size: 17px; font-weight: 600; }
QLabel[role='muted'] { color: #a9bdd0; }
QLabel[role='notice'] { color: #dfbd80; padding: 8px; }
QPushButton { background: #1a2c40; border: 1px solid #31485f; border-radius: 6px; padding: 7px 14px; }
QPushButton:hover { background: #28445c; border-color: #80cfcb; }
QPushButton[primary='true'] { background: #83d6cc; color: #10212c; border-color: #83d6cc; font-weight: 600; }
QPushButton:disabled { color: #6f8192; background: #152131; border-color: #223246; }
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox { background: #142437; border: 1px solid #344b62; border-radius: 4px; padding: 7px; }
QComboBox::drop-down { width: 24px; border: none; }
QComboBox QAbstractItemView { background: #1a2c40; selection-background-color: #315b69; }
QListWidget { background: #111e2e; border: 1px solid #263d52; border-radius: 6px; padding: 4px; }
QListWidget::item { padding: 8px; border-radius: 4px; }
QListWidget::item:selected { background: #294e5e; color: #dbfffa; }
QTabWidget::pane { border: 0; }
QTabBar::tab { background: #142437; padding: 9px 12px; margin-right: 2px; }
QTabBar::tab:selected { background: #294e5e; color: #b0f1e7; }
QGroupBox { border: 1px solid #2a4055; border-radius: 6px; margin-top: 14px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #b4c8d9; }
QScrollArea { border: none; }
QScrollBar:vertical { width: 10px; background: #101c2b; }
QScrollBar::handle:vertical { background: #355269; min-height: 30px; border-radius: 4px; }
QSplitter::handle { background: #243a4f; width: 5px; }
QProgressBar { border: 1px solid #31485f; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #416f75; }
"""


def scrollable(widget):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    return scroll


class Observatory(QMainWindow):
    _fonts_loaded = False

    def __init__(self):
        super().__init__()
        # Matplotlib already ships this font, including in the frozen bundle.
        # Loading it explicitly also supports machines without system fonts.
        if not Observatory._fonts_loaded:
            for filename in ('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf'):
                QFontDatabase.addApplicationFont(str(Path(get_data_path())/'fonts/ttf'/filename))
            Observatory._fonts_loaded = True
        self.setWindowTitle('Observatory')
        self.resize(1440, 940)
        self.setMinimumSize(960, 680)
        self.setStyleSheet(STYLE)
        self.session = None
        self.original_setup = None
        self.selected_id = None
        self.running = False
        self.editing = True
        self.dirty = False
        self.saved_digest = None
        self.pending_body = None
        self.calculation = None
        self.playback = None
        self.undo_stack = []
        self.history = deque(maxlen=MAX_HISTORY)
        self._drag = None
        self.following = False
        self.pending_time = 0.
        self.last_tick = time.monotonic()
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.menu_page = self.make_menu()
        self.library_page = self.make_library()
        self.workspace = self.make_workspace()
        for page in (self.menu_page, self.library_page, self.workspace):
            self.stack.addWidget(page)
        self.stack.setCurrentWidget(self.menu_page)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        QShortcut(QKeySequence('Escape'), self, activated=self.cancel_interaction)
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.save_dialog)

    def make_menu(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.addStretch()
        card = QWidget()
        card.setMinimumWidth(560)
        card.setMaximumWidth(590)
        layout = QVBoxLayout(card)
        layout.setSpacing(20)
        layout.addWidget(label('PHYSICS EXPERIMENTS', 'muted'))
        layout.addWidget(label('Observatory', 'title'))
        layout.addWidget(label('Choose a prepared experiment. Run it, change it, and see what happens.', 'subheading'))
        layout.addSpacing(12)
        self.choose_button = action('Choose experiment', self.show_library, primary=True)
        self.choose_button.setMinimumHeight(54)
        self.load_button = action('Load saved experiment', self.load_dialog)
        self.load_button.setMinimumHeight(54)
        layout.addWidget(self.choose_button)
        layout.addWidget(self.load_button)
        layout.addWidget(label('Every new experiment starts with the Solar System.', 'muted'))
        self.menu_error = label('', 'notice')
        layout.addWidget(self.menu_error)
        outer.addWidget(card, alignment=Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch()
        return page

    def make_library(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(55, 40, 55, 40)
        layout.setSpacing(22)
        layout.addWidget(action('‹ Starting menu', lambda: self.stack.setCurrentWidget(self.menu_page)))
        layout.addWidget(label('Choose an experiment', 'title'))
        layout.addWidget(label('Prepared starting points you can run unchanged or make your own.', 'subheading'))
        row = QHBoxLayout()
        self.experiment_list = QListWidget()
        self.experiment_list.setMinimumWidth(280)
        for experiment in EXPERIMENTS:
            self.experiment_list.addItem(experiment.title)
        row.addWidget(self.experiment_list, 1)
        details = QWidget()
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(30, 20, 30, 20)
        self.experiment_title = label('', 'heading')
        self.experiment_description = label('', 'subheading')
        self.experiment_observe = label('', 'muted')
        for widget in (self.experiment_title, self.experiment_description, self.experiment_observe):
            details_layout.addWidget(widget)
        details_layout.addStretch()
        self.open_button = action('Open frozen starting setup', self.open_selected, primary=True)
        details_layout.addWidget(self.open_button)
        row.addWidget(details, 2)
        layout.addLayout(row, 1)
        self.experiment_list.currentRowChanged.connect(self.describe_experiment)
        self.experiment_list.itemDoubleClicked.connect(lambda item: self.open_selected())
        self.experiment_list.setCurrentRow(0)
        return page

    def describe_experiment(self, index):
        item = EXPERIMENTS[max(0, index)]
        self.experiment_title.setText(item.title)
        self.experiment_description.setText(item.description)
        self.experiment_observe.setText(item.observe)

    def make_workspace(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 10, 12, 10)
        header = QHBoxLayout()
        header.addWidget(action('Starting menu', self.show_menu))
        self.name_field = QLineEdit()
        self.name_field.setAccessibleName('Experiment name')
        self.name_field.setPlaceholderText('Experiment name')
        self.name_field.editingFinished.connect(self.rename_experiment)
        header.addWidget(self.name_field, 1)
        self.save_button = action('Save experiment…', self.save_dialog)
        header.addWidget(self.save_button)
        layout.addLayout(header)
        self.splitter = QSplitter()
        sidebar = QWidget()
        sidebar.setMinimumWidth(320)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(4, 8, 12, 8)
        self.mode_label = label('Starting setup', 'subheading')
        side.addWidget(self.mode_label)
        side.addWidget(label('Bodies', 'muted'))
        self.body_list = QListWidget()
        self.body_list.setAccessibleName('Experiment bodies')
        self.body_list.setMinimumHeight(110)
        self.body_list.setMaximumHeight(150)
        self.body_list.currentItemChanged.connect(self.select_list_body)
        side.addWidget(self.body_list)
        self.tabs = QTabWidget()
        self.inspection = QWidget()
        inspection_layout = QVBoxLayout(self.inspection)
        self.body_information = BodyInformation()
        inspection_layout.addWidget(self.body_information)
        self.context_label = label('Select a body in the scene or the list to inspect it.', 'muted')
        inspection_layout.addWidget(self.context_label)
        self.focus_button = action('Focus selected and its orbiting bodies', self.focus_selected)
        inspection_layout.addWidget(self.focus_button)
        self.follow_button = action('Follow selected', self.toggle_follow)
        inspection_layout.addWidget(self.follow_button)
        self.remove_button = action('Remove selected body', self.remove_selected)
        inspection_layout.addWidget(self.remove_button)
        inspection_layout.addStretch()
        self.catalog = BodyCatalog()
        self.catalog.add_requested.connect(self.begin_placement)
        self.catalog.type_combo.currentTextChanged.connect(self.sync_catalog_references)
        self.properties = BodyProperties()
        self.properties.apply_requested.connect(self.apply_properties)
        self.tabs.addTab(scrollable(self.inspection), 'Inspect')
        for panel, button, title in [(self.catalog, self.catalog.add_button, 'Add body'),
                                      (self.properties, self.properties.apply_button, 'Edit body')]:
            tab = QWidget()
            content = QVBoxLayout(tab)
            content.setContentsMargins(0, 0, 0, 0)
            content.addWidget(scrollable(panel), 1)
            content.addWidget(button)
            self.tabs.addTab(tab, title)
        side.addWidget(self.tabs, 1)
        self.edit_button = action('Edit starting setup', self.edit_starting_setup)
        side.addWidget(self.edit_button)
        edits = QHBoxLayout()
        self.undo_button = action('Undo edit', self.undo_edit)
        self.original_button = action('Restore original', self.restore_original)
        edits.addWidget(self.undo_button)
        edits.addWidget(self.original_button)
        side.addLayout(edits)
        self.splitter.addWidget(sidebar)
        view = QWidget()
        view_layout = QVBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(10, 8), facecolor=BACKGROUND)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setAccessibleName('Simulation scene')
        self.canvas.setMinimumWidth(480)
        self.ax = self.figure.add_axes((0, 0, 1, 1))
        view_layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        self.run_button = action('Run', self.toggle_run, primary=True)
        self.reset_button = action('Reset', self.reset_run)
        self.step_button = action('Step', self.step_once)
        controls.addWidget(self.run_button)
        controls.addWidget(self.reset_button)
        controls.addWidget(self.step_button)
        controls.addWidget(action('Fit view', self.fit_view))
        controls.addWidget(label('Years / second', 'muted'))
        self.speed = spin(.5, .0001, 1000, 4)
        controls.addWidget(self.speed)
        self.arrows_checkbox = QCheckBox('Starting velocities')
        self.arrows_checkbox.setChecked(True)
        self.arrows_checkbox.toggled.connect(lambda: self.refresh_scene())
        controls.addStretch()
        view_layout.addLayout(controls)
        view_tools = QHBoxLayout()
        view_tools.addWidget(self.arrows_checkbox)
        view_tools.addStretch()
        view_layout.addLayout(view_tools)
        view_layout.addWidget(self.make_calculation_controls())
        recordings = QHBoxLayout()
        recordings.addWidget(label('Recorded results', 'muted'))
        self.result_combo = Dropdown()
        self.result_combo.setMinimumContentsLength(20)
        self.result_combo.currentIndexChanged.connect(self.select_recording)
        recordings.addWidget(self.result_combo, 1)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        recordings.addWidget(self.progress, 1)
        view_layout.addLayout(recordings)
        self.splitter.addWidget(view)
        self.splitter.setSizes([370, 1040])
        layout.addWidget(self.splitter, 1)
        self.notice = label('Choose an experiment to begin.', 'notice')
        layout.addWidget(self.notice)
        for event, callback in [('button_press_event', self.on_press), ('motion_notify_event', self.on_motion),
                                ('button_release_event', self.on_release), ('scroll_event', self.on_scroll)]:
            self.canvas.mpl_connect(event, callback)
        self.canvas.mpl_connect('resize_event', lambda event: self.refresh_scene())
        return page

    def make_calculation_controls(self):
        group = QGroupBox('Calculate a recording')
        group.setCheckable(True)
        group.setChecked(False)
        outer = QVBoxLayout(group)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(label('Calculate from the starting setup, then play the saved samples. Recordings stay in this session.', 'muted'))
        row = QHBoxLayout()
        self.duration = spin(1, .000001, 1e6, 6)
        self.stride = QSpinBox()
        self.stride.setRange(1, 1000000)
        self.stride.setValue(10)
        self.memory_budget = QSpinBox()
        self.memory_budget.setRange(1, 16384)
        self.memory_budget.setValue(256)
        for text, widget in [('Years', self.duration), ('Sample every N steps', self.stride), ('Budget (MiB)', self.memory_budget)]:
            form = QFormLayout()
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
            form.addRow(text, widget)
            row.addLayout(form)
        self.calculate_button = action('Calculate', self.start_calculation)
        row.addWidget(self.calculate_button)
        content_layout.addLayout(row)
        outer.addWidget(content)
        content.setVisible(False)
        group.toggled.connect(content.setVisible)
        return group

    @property
    def setup(self):
        return self.session.initial_setup if self.session else None

    @property
    def prepared(self):
        return self.session.prepared

    def tell(self, text):
        self.notice.setText(str(text))
        self.menu_error.setText(str(text))

    def confirm_leave(self):
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, 'Unsaved experiment', 'Save your edited starting setup before leaving?',
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Save:
            return self.save_dialog()
        return answer == QMessageBox.StandardButton.Discard

    def show_menu(self):
        if not self.confirm_leave():
            return
        self.stop_activity()
        self.cancel_interaction()
        self.stack.setCurrentWidget(self.menu_page)
        self.menu_error.setText('')

    def show_library(self):
        self.stack.setCurrentWidget(self.library_page)

    def open_selected(self):
        self.open_experiment(EXPERIMENTS[max(0, self.experiment_list.currentRow())].key)

    def open_experiment(self, key):
        self.install_experiment(get_experiment(key))

    def install_experiment(self, setup):
        # Construct and validate before replacing any active work.
        session = ExperimentSession(setup)
        self.stop_activity()
        self.session, self.original_setup = session, setup
        self.saved_digest = setup_digest(setup)
        self.selected_id = None
        self.editing, self.dirty = True, False
        self.pending_body = None
        self.following = False
        self.undo_stack.clear()
        self.playback = self.calculation = None
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        self.result_combo.blockSignals(False)
        self.name_field.setText(setup.name)
        self.stack.setCurrentWidget(self.workspace)
        self.history = deque([StateFrame.capture(self.prepared)], maxlen=MAX_HISTORY)
        self.create_scene()
        self.refresh_bodies()
        self.sync_controls()
        self.tell(setup.provenance.get('observe', 'Frozen starting setup. Run it, or select a body to edit.'))

    def create_scene(self):
        self.scene = draw_system(self.ax, self.prepared, self.history)
        self.scene.heading.set_position((.025, .97))
        self.scene.guide_note.set_visible(False)
        self.scene.trail_info.set_position((.025, .10))
        self.context_orbits = LineCollection([], colors='#537f91', linewidths=.85, alpha=.7, zorder=1.8)
        self.ax.add_collection(self.context_orbits)
        self.context_points = self.ax.scatter([], [], marker='D', s=65, facecolors='none', edgecolors='#c6a9ed', zorder=4)
        self.context_names = [self.ax.annotate(text, (0, 0), xytext=(8, 7), textcoords='offset points',
            color='#c6a9ed', fontsize=9, zorder=8) for text in ('L4', 'L5')]
        self.context_note = self.ax.text(.025, .06, '', transform=self.ax.transAxes, fontsize=8, color='#b0bbd2', zorder=10)
        self.preview_marker, = self.ax.plot([], [], 'o', markerfacecolor='none', markeredgecolor=ACCENT, markersize=20, zorder=20)
        self.velocity_artists = []
        self.refresh_scene()

    def refresh_bodies(self):
        self.body_list.blockSignals(True)
        self.body_list.clear()
        for key, spec in zip(self.prepared.body_ids, self.prepared.body_specs):
            name = spec.name if spec.kind != 'particle' else 'Test particle '+key.rsplit(':', 1)[-1]
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.body_list.addItem(item)
            if key == self.selected_id:
                self.body_list.setCurrentItem(item)
        self.body_list.blockSignals(False)
        self.sync_catalog_references()
        self.refresh_inspection()

    def sync_catalog_references(self):
        if self.session:
            self.catalog.set_references(self.prepared, self.selected_id)

    def select_list_body(self, item, previous):
        self.select_body(item.data(Qt.ItemDataRole.UserRole) if item else None)

    def select_body(self, body_id):
        self.selected_id = body_id if body_id in self.prepared.body_ids else None
        self.refresh_inspection()
        self.sync_catalog_references()
        self.sync_controls()
        self.refresh_scene()

    def refresh_inspection(self, update_editor=True):
        present = self.selected_id in self.prepared.body_ids
        self.body_information.setVisible(present)
        self.focus_button.setEnabled(present)
        self.follow_button.setEnabled(present)
        if not present:
            self.context_label.setText('The selected body is absent from this state.' if self.selected_id else
                                       'Select a body in the scene or the list to inspect it.')
            return
        i = self.prepared.body_ids.index(self.selected_id)
        spec = self.prepared.body_specs[i]
        self.body_information.set_spec(spec)
        reference = self.prepared.reference_ids[i]
        text = 'No parent is assigned.'
        if reference in self.prepared.body_ids:
            j = self.prepared.body_ids.index(reference)
            distance = np.linalg.norm(self.prepared.system.r[i]-self.prepared.system.r[j])
            speed = np.linalg.norm(self.prepared.system.v[i]-self.prepared.system.v[j])*AUYR_TO_KMS
            text = f'Relative to {self.prepared.body_specs[j].name}\nSeparation: {distance:,.6g} AU\nSpeed: {speed:,.6g} km/s'
        children = sum(ref == self.selected_id for ref in self.prepared.reference_ids)
        text += f'\nOrbiting bodies in this setup: {children}'
        text += '\nOrbit curves are instantaneous two-body estimates.'
        if self.editing:
            text += '\nDrag the selected arrowhead to edit its starting motion.'
        self.context_label.setText(text)
        body = next((b for b in self.setup.bodies if b.id == self.selected_id), None)
        if body and update_editor:
            self.properties.set_body(body)

    def sync_controls(self):
        if not self.session:
            return
        calculating = self.calculation is not None and not self.calculation.done
        placing = self.pending_body is not None
        can_edit = self.editing and not calculating and not placing
        authored = any(b.id == self.selected_id for b in self.setup.bodies)
        self.tabs.setTabEnabled(1, can_edit)
        self.tabs.setTabEnabled(2, can_edit and authored)
        self.remove_button.setEnabled(can_edit and authored and len(self.setup.bodies) > 1)
        self.edit_button.setEnabled(not calculating and not placing)
        self.edit_button.setText('Editing starting setup' if self.editing else 'Edit starting setup')
        self.undo_button.setEnabled(can_edit and bool(self.undo_stack))
        self.original_button.setEnabled(not calculating and not placing)
        self.name_field.setEnabled(can_edit)
        self.run_button.setEnabled(not calculating and not placing)
        self.step_button.setEnabled(not calculating and not placing)
        self.reset_button.setEnabled(not calculating and not placing)
        self.save_button.setEnabled(not calculating and not placing)
        self.calculate_button.setEnabled(not placing)
        self.calculate_button.setText('Cancel calculation' if calculating else 'Calculate')
        self.run_button.setText('Pause' if self.running else 'Play recording' if self.playback else 'Run')
        self.step_button.setText('Next sample' if self.playback else 'Step')
        self.mode_label.setText('Calculating…' if calculating else 'Recorded result' if self.playback else
            'Starting setup · frozen' if self.editing else 'Running' if self.running else 'Paused')
        self.arrows_checkbox.setEnabled(self.editing)
        self.progress.setVisible(calculating)
        self.result_combo.setEnabled(not calculating and not placing and bool(self.session.recordings))

    def refresh_scene(self):
        if not self.session or not hasattr(self, 'scene'):
            return
        self.scene.update(self.prepared, self.history)
        self.scene.select(self.selected_id)
        self.scene.highlight.set_visible(self.selected_id in self.prepared.body_ids)
        self.scene.particles.set_visible(any(spec.kind == 'particle' for spec in self.prepared.body_specs))
        # The legacy controller retains its compatibility guides. The desktop
        # deliberately replaces them with generic, selection-scoped overlays.
        self.scene.reference_guides.set_visible(False)
        self.scene.guide_note.set_visible(False)
        for item in self.scene.guide_labels:
            item.set_visible(False)
        frame = StateFrame.capture(self.prepared)
        context = selection_context(frame, self.selected_id)
        self.context_orbits.set_segments([self.scene.world_to_display(curve) for curve in context.orbits.values()])
        points = self.scene.world_to_display(context.guides) if len(context.guides) else np.empty((0, 2))
        self.context_points.set_offsets(points)
        self.context_points.set_visible(bool(len(points)))
        for index, item in enumerate(self.context_names):
            item.set_visible(bool(len(points)))
            if len(points):
                item.xy = points[index]
        text = ''
        if context.pair:
            names = [frame.body_specs[frame.body_ids.index(key)].name for key in context.pair]
            text = ' – '.join(names)+': L4/L5 geometric guides (ideal circular model).'
        if context.orbits:
            text += '\nOrbit curves: instantaneous two-body estimates, not predicted trajectories.'
        self.context_note.set_text(text.strip())
        self.draw_starting_velocities()
        if not self.editing:
            self.refresh_inspection(update_editor=False)
        if self.following and self.selected_id in self.prepared.body_ids:
            point = self.scene.positions[self.prepared.body_ids.index(self.selected_id)]
            width, height = np.ptp(self.ax.get_xlim()), np.ptp(self.ax.get_ylim())
            self.ax.set_xlim(point[0]-width/2, point[0]+width/2)
            self.ax.set_ylim(point[1]-height/2, point[1]+height/2)
        self.canvas.draw_idle()

    def draw_starting_velocities(self):
        for artist in self.velocity_artists:
            artist.remove()
        self.velocity_artists.clear()
        if not self.editing or not self.arrows_checkbox.isChecked():
            return
        self.velocity_tips = {}
        for i, (key, spec, reference) in enumerate(zip(self.prepared.body_ids, self.prepared.body_specs, self.prepared.reference_ids)):
            if spec.kind == 'particle' and key != self.selected_id:
                continue
            parent = self.prepared.body_ids.index(reference) if reference in self.prepared.body_ids else None
            velocity = self.prepared.system.v[i]-(self.prepared.system.v[parent] if parent is not None else 0)
            anchor = self.scene.positions[i]
            tip = self.ax.transData.inverted().transform(self.ax.transData.transform(anchor)+velocity*12)
            arrow = FancyArrowPatch(anchor, tip, arrowstyle='->', mutation_scale=11,
                color=ACCENT if key == self.selected_id else '#6593a5', linewidth=1.1, zorder=9)
            self.ax.add_patch(arrow)
            self.velocity_artists.append(arrow)
            self.velocity_tips[key] = tip

    def apply_setup(self, setup, *, preserve_definition=False):
        self.undo_stack.append(self.setup)
        if preserve_definition:
            self.session.select(setup)
        else:
            self.session.replace_initial(setup)
        self.running, self.editing, self.playback = False, True, None
        self.scene.result_info.set_text('')
        self.dirty = setup_digest(self.setup) != self.saved_digest
        self.pending_time = 0.
        self.history = deque([StateFrame.capture(self.prepared)], maxlen=MAX_HISTORY)
        self.name_field.setText(self.setup.name)
        self.refresh_bodies()
        self.refresh_scene()
        self.sync_controls()

    def begin_placement(self, payload):
        if not self.editing:
            return
        if payload['circular'] and payload['reference_id'] is None:
            self.tell('Choose a body for the initial orbit, or choose manual starting motion.')
            return
        self.pending_body = payload
        self.tabs.setCurrentIndex(0)
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)
        self.tell(f'Click in the scene to place {payload["spec"].name}. Escape cancels without changing the setup.')
        self.sync_controls()

    def place_at(self, world):
        if not self.pending_body:
            return
        payload = self.pending_body
        try:
            candidate = place_body(self.setup, payload['spec'], world, payload['reference_id'],
                circular=payload['circular'], velocity=payload['velocity'])
            self.selected_id = candidate.bodies[-1].id
            self.pending_body = None
            self.preview_marker.set_data([], [])
            self.canvas.unsetCursor()
            self.apply_setup(candidate)
            self.tell('Body added to the starting setup. Select it to edit, or press Run.')
        except (ValueError, OverflowError) as exc:
            self.tell(str(exc)+' Choose another position, or Escape to cancel.')

    def apply_properties(self, payload):
        if not self.editing:
            return
        try:
            body = next(b for b in self.setup.bodies if b.id == payload['body_id'])
            updated = replace(body, spec=payload['spec'], r=payload['r'], v=payload['v'])
            from .templates import _validated_edit
            candidate = _validated_edit(replace(self.setup,
                bodies=tuple(updated if b.id == body.id else b for b in self.setup.bodies)), self.setup)
            old_dt = self.setup.dt
            self.apply_setup(candidate)
            text = 'Starting setup updated. The original experiment is preserved.'
            if candidate.dt < old_dt:
                text += f' Fixed timestep reduced to {candidate.dt:.6g} yr for this starting setup.'
            self.tell(text)
        except (ValueError, OverflowError) as exc:
            self.tell(exc)

    def remove_selected(self):
        if not self.editing or self.selected_id is None:
            return
        try:
            candidate = remove_body(self.setup, self.selected_id)
            self.selected_id = None
            self.apply_setup(candidate)
            self.tell('Body removed. Undo edit restores it.')
        except ValueError as exc:
            self.tell(exc)

    def undo_edit(self):
        if not self.editing or not self.undo_stack:
            return
        previous = self.undo_stack.pop()
        self.session.select(previous)
        self.dirty = setup_digest(self.setup) != self.saved_digest
        self.history = deque([StateFrame.capture(self.prepared)], maxlen=MAX_HISTORY)
        self.name_field.setText(previous.name)
        self.refresh_bodies()
        self.refresh_scene()
        self.sync_controls()
        self.tell('Restored the previous starting setup.')

    def restore_original(self):
        if self.original_setup is None:
            return
        self.stop_activity()
        self.cancel_interaction()
        self.editing = True
        self.apply_setup(self.original_setup, preserve_definition=True)
        self.tell('Original frozen experiment restored. Previous edits remain available through Undo edit.')

    def rename_experiment(self):
        if not self.session or not self.editing or self.name_field.text() == self.setup.name:
            return
        name = self.name_field.text().strip()
        if not name:
            self.name_field.setText(self.setup.name)
            return
        self.apply_setup(replace(self.setup, name=name))

    def save_to(self, path):
        save_setup(self.setup, Path(path))
        self.saved_digest = setup_digest(self.setup)
        self.dirty = False
        self.tell(f'Saved starting setup: {path}. Recorded results remain in memory for this session.')

    def save_dialog(self):
        if not self.session:
            return False
        self.rename_experiment()
        filename, _ = QFileDialog.getSaveFileName(self, 'Save experiment starting setup', self.setup.name+'.json', 'Experiment (*.json)')
        if not filename:
            return False
        try:
            self.save_to(filename)
        except (ValueError, OSError) as exc:
            self.tell(exc)
            return False
        return True

    def load_from(self, path):
        setup = load_setup(Path(path))
        self.install_experiment(setup)

    def load_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Load saved experiment', '', 'Experiment (*.json)')
        if not filename or not self.confirm_leave():
            return
        try:
            self.load_from(filename)
        except (ValueError, OSError) as exc:
            self.tell(f'Could not load the experiment: {exc}')

    def stop_activity(self):
        self.running = False
        self.pending_time = 0.
        if self.playback:
            self.playback.pause()
        if self.calculation is not None and not self.calculation.done:
            self.calculation.cancel()

    def edit_starting_setup(self):
        if self.session:
            self.reset_run()
            self.tell('Frozen starting setup. Select a body to edit it, or use Add body.')

    def reset_run(self):
        if not self.session:
            return
        self.stop_activity()
        self.cancel_interaction()
        self.playback = None
        self.session.restart()
        self.editing = True
        self.history = deque([StateFrame.capture(self.prepared)], maxlen=MAX_HISTORY)
        self.scene.result_info.set_text('')
        self.refresh_bodies()
        self.refresh_scene()
        self.sync_controls()
        self.tell('Returned to the edited starting setup at time zero.')

    def toggle_run(self):
        if not self.session or self.pending_body is not None:
            return
        if self.playback:
            if self.running:
                self.playback.pause()
            else:
                if self.playback.index == len(self.playback.recording.frames)-1:
                    self.playback = Playback(self.playback.recording)
                self.playback.play()
            self.running = self.playback.playing
        else:
            self.running = not self.running
            self.editing = False
        self.last_tick = time.monotonic()
        self.sync_controls()
        self.refresh_scene()
        self.tell('Running. Pause to inspect; Edit starting setup returns to the frozen beginning.' if self.running else 'Paused.')

    def step_once(self):
        if not self.session or self.pending_body:
            return
        self.running, self.editing = False, False
        if self.playback:
            self.playback.next_sample()
            self.display_recorded_frame()
        else:
            self.advance_live(1)
        self.sync_controls()

    def advance_live(self, count, budget_seconds=None):
        started = time.monotonic()
        old_ids = self.prepared.body_ids
        completed = 0
        for _ in range(count):
            if completed and budget_seconds is not None and time.monotonic()-started >= budget_seconds:
                break
            try:
                result = advance_prepared(self.prepared)
                self.session.prepared = result.prepared
                self.history.extend(result.event_frames)
                completed += 1
            except ValueError as exc:
                self.running = False
                self.tell(f'Simulation halted at the last valid state: {exc}')
                break
        self.history.append(StateFrame.capture(self.prepared))
        if self.selected_id not in self.prepared.body_ids:
            mapped = mapped_selection(self.selected_id, self.prepared.events)
            if mapped != self.selected_id:
                self.tell(f'The selected body merged; selection transferred to {mapped}.')
            self.selected_id = mapped
        if old_ids != self.prepared.body_ids:
            self.refresh_bodies()
        self.refresh_scene()
        return completed

    def tick(self):
        now = time.monotonic()
        wall = min(max(0., now-self.last_tick), .25)
        self.last_tick = now
        if self.session is None or self.stack.currentWidget() is not self.workspace:
            return
        if self.calculation is not None and not self.calculation.done:
            self.calculation.advance(150, budget_seconds=.006)
            self.progress.setValue(round(self.calculation.progress*100))
            self.tell(f'Calculating {self.calculation.completed_steps:,} / {self.calculation.total_steps:,} steps '
                      f'({self.calculation.prepared.system.time:.6g} of {self.calculation.resolved_end_time:.6g} years).')
            if self.calculation.done:
                self.finish_calculation()
        elif self.running:
            if self.playback:
                old_index = self.playback.index
                self.playback.advance(wall, self.speed.value())
                self.running = self.playback.playing
                if old_index != self.playback.index:
                    self.display_recorded_frame()
            else:
                self.pending_time += wall*self.speed.value()
                count = min(150, int(self.pending_time/self.prepared.system.dt))
                if count:
                    completed = self.advance_live(count, budget_seconds=.006)
                    self.pending_time = max(0., self.pending_time-completed*self.prepared.system.dt)
                    if self.running and self.pending_time > max(.25, self.speed.value()):
                        self.tell('Calculation is slower than the requested playback speed; physical steps are preserved.')
            self.sync_controls()

    def start_calculation(self):
        if not self.session or self.pending_body:
            return
        if self.calculation is not None and not self.calculation.done:
            self.calculation.cancel()
            self.finish_calculation()
            return
        try:
            request = CalculationRequest(self.duration.value(), self.stride.value(), self.memory_budget.value()*1048576)
            calculation = Calculation(self.setup, request)
        except ValueError as exc:
            self.tell(exc)
            return
        self.stop_activity()
        self.calculation = calculation
        self.playback = None
        self.editing = False
        self.progress.setValue(0)
        self.sync_controls()
        self.tell(f'Calculating from the frozen starting setup: {calculation.total_steps:,} steps, '
                  f'fixed dt {self.setup.dt:.6g} yr; resolved end {calculation.resolved_end_time:.6g} yr.')

    def finish_calculation(self):
        result = self.calculation.result
        self.session.retain_recording(result)
        self.result_combo.blockSignals(True)
        suffix = 'original' if result.setup_digest == setup_digest(self.original_setup) else 'edited'
        self.result_combo.addItem(f'{len(self.session.recordings)}. {result.source_name} · {suffix} · {result.status}')
        self.result_combo.setCurrentIndex(len(self.session.recordings)-1)
        self.result_combo.blockSignals(False)
        self.show_recording(result)
        self.tell(f'{result.status.title()}: {result.reason or "Calculation finished."} '
                  f'Requested {result.requested_end_time:.6g} yr; recorded through {result.actual_end_time:.6g} yr.')

    def select_recording(self, index):
        if self.session and 0 <= index < len(self.session.recordings):
            self.show_recording(self.session.recordings[index])

    def show_recording(self, result):
        self.stop_activity()
        self.playback = Playback(result)
        self.editing = False
        self.display_recorded_frame()
        self.sync_controls()

    def display_recorded_frame(self):
        frame = self.playback.current_frame
        old_ids = self.prepared.body_ids
        self.session.prepared = PreparedSystem(System(frame.masses, frame.r, frame.v, frame.dt, frame.time),
            frame.body_ids, frame.body_specs, self.playback.recording.source_setup,
            frame.events, reference_ids=frame.reference_ids)
        self.history = deque(self.playback.recording.frames[max(0, self.playback.index-MAX_HISTORY+1):self.playback.index+1],
                             maxlen=MAX_HISTORY)
        if self.selected_id not in frame.body_ids:
            self.selected_id = mapped_selection(self.selected_id, frame.events)
        if old_ids != frame.body_ids:
            self.refresh_bodies()
        run = self.playback.recording
        self.scene.result_info.set_text(f'{run.source_name} · {run.status.upper()}\n'
            f'Sample {self.playback.index+1:,} / {len(run.frames):,}\n'
            f'Recorded through {run.actual_end_time:.6g} yr\nSource {run.setup_digest[:12]}')
        self.refresh_scene()

    def fit_view(self):
        if self.session:
            self.following = False
            self.scene.fit_view()
            self.refresh_scene()

    def toggle_follow(self):
        if self.selected_id:
            self.following = not self.following
            self.follow_button.setText('Stop following' if self.following else 'Follow selected')
            self.refresh_scene()

    def focus_selected(self):
        if self.selected_id not in self.prepared.body_ids:
            return
        i = self.prepared.body_ids.index(self.selected_id)
        frame = StateFrame.capture(self.prepared)
        context = selection_context(frame, self.selected_id)
        center = self.scene.positions[i]
        curves = list(context.orbits.values())
        if curves:
            points = self.scene.world_to_display(np.vstack(curves))
            extent = max(1e-5, float(np.max(np.abs(points-center))))*1.25
        else:
            extent = .5
        ratio = self.ax.bbox.height/max(self.ax.bbox.width, 1)
        width = extent/min(1., ratio)
        self.ax.set_xlim(center[0]-width, center[0]+width)
        self.ax.set_ylim(center[1]-width*ratio, center[1]+width*ratio)
        self.following = True
        self.follow_button.setText('Stop following')
        self.refresh_scene()

    def on_press(self, event):
        if not self.session or event.inaxes is not self.ax or event.button != 1 or event.xdata is None:
            return
        if self.pending_body:
            self.place_at(self.scene.display_to_world((event.xdata, event.ydata)))
            return
        point = np.array([event.x, event.y])
        if self.editing and self.selected_id in getattr(self, 'velocity_tips', {}):
            tip = self.ax.transData.transform(self.velocity_tips[self.selected_id])
            index = self.prepared.body_ids.index(self.selected_id)
            anchor = self.ax.transData.transform(self.scene.positions[index])
            if np.linalg.norm(tip-point) < 14 and np.linalg.norm(tip-anchor) > 12:
                self._drag = ('velocity', self.selected_id, None)
                self.tell('Drag to set the starting velocity. Release applies; Escape cancels.')
                return
        distances = np.linalg.norm(self.ax.transData.transform(self.scene.positions)-point, axis=1)
        index = int(np.argmin(distances))
        if distances[index] < 20:
            self.select_body(self.prepared.body_ids[index])
            for row in range(self.body_list.count()):
                item = self.body_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == self.selected_id:
                    self.body_list.blockSignals(True)
                    self.body_list.setCurrentItem(item)
                    self.body_list.scrollToItem(item)
                    self.body_list.blockSignals(False)
                    break
        else:
            self.select_body(None)
            self.following = False
            self._drag = ('pan', point, (self.ax.get_xlim(), self.ax.get_ylim()))

    def on_motion(self, event):
        if not self.session or event.inaxes is not self.ax or event.xdata is None:
            return
        if self.pending_body:
            self.preview_marker.set_data([event.xdata], [event.ydata])
            self.canvas.draw_idle()
        elif self._drag:
            if self._drag[0] == 'pan':
                _, point, limits = self._drag
                delta = np.array([event.x, event.y])-point
                for values, pixels, shift, setter in [(limits[0], self.ax.bbox.width, delta[0], self.ax.set_xlim),
                                                       (limits[1], self.ax.bbox.height, delta[1], self.ax.set_ylim)]:
                    setter(np.array(values)-shift*(values[1]-values[0])/pixels)
                self.canvas.draw_idle()
            elif self._drag[0] == 'velocity':
                self._drag = ('velocity', self._drag[1], np.array([event.x, event.y]))
                self.tell('Release to apply the new starting velocity, or Escape to cancel.')

    def on_release(self, event):
        drag, self._drag = self._drag, None
        if not drag or drag[0] != 'velocity' or drag[2] is None:
            return
        key, tip = drag[1], drag[2]
        if event.inaxes is not self.ax:
            self.tell('Velocity edit cancelled outside the scene.')
            return
        i = self.prepared.body_ids.index(key)
        reference = self.prepared.reference_ids[i]
        parent = self.prepared.body_ids.index(reference) if reference in self.prepared.body_ids else None
        velocity = (tip-self.ax.transData.transform(self.scene.positions[i]))/12
        world_velocity = velocity+(self.prepared.system.v[parent] if parent is not None else 0)
        try:
            self.apply_setup(update_body(self.setup, key, velocity=world_velocity))
            self.tell('Starting velocity updated. Undo edit restores the previous value.')
        except ValueError as exc:
            self.tell(exc)

    def on_scroll(self, event):
        if not self.session or event.inaxes is not self.ax or event.xdata is None:
            return
        scale = .8 if event.step > 0 else 1.25
        for center, limits, setter in [(event.xdata, self.ax.get_xlim(), self.ax.set_xlim),
                                       (event.ydata, self.ax.get_ylim(), self.ax.set_ylim)]:
            setter(center+(np.array(limits)-center)*scale)
        self.refresh_scene()

    def cancel_interaction(self):
        self._drag = None
        if self.pending_body:
            self.pending_body = None
            self.preview_marker.set_data([], [])
            self.canvas.unsetCursor()
            self.tell('Placement cancelled. Starting setup unchanged.')
        if self.session:
            self.sync_controls()
            self.refresh_scene()

    def closeEvent(self, event):
        if not self.confirm_leave():
            event.ignore()
            return
        self.stop_activity()
        self.timer.stop()
        super().closeEvent(event)


def main():
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName('Observatory')
    application.setOrganizationName('Science Bitch')
    window = Observatory()
    window.show()
    return application.exec()
