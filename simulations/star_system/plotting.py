"""Overhead observatory controls, using the existing Matplotlib desktop backend."""
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from .core import SOLAR_RADIUS_AU
from .data import STARS, PLANETS, AU_KM
from .initialization import build_system, legacy_setup, PreparedSystem
from .history import StateFrame, MAX_HISTORY, relative_state
from .rendering import BACKGROUND, PANEL, TEXT, MUTED, ACCENT, CanvasPainter, draw, draw_system
from .sidebar import Sidebar
from .experiments import ExperimentSession
from .scenarios import get_scenario
from .recording import Calculation, CalculationRequest
from .playback import Playback

FRAME_SECONDS = 1 / 60
PHYSICS_BUDGET_SECONDS = .006


def _style_axes(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)


class App:
    def __init__(self, star='sun', planets=('earth', 'jupiter'), population=None, setup=None):
        if population is not None and population.reference != 0:
            raise ValueError('The legacy population argument is star-relative; use InitialSetup for general references.')
        self.star, self.planets = star, list(planets)
        self.population = setup.population.settings if setup is not None and setup.population is not None else population
        self.population_editor = None
        self.body_editor = None
        self.experiment_editor = None
        self.calculation_editor = None
        self.calculation = None
        self.playback = None
        self._selected_id = None
        self.preparing = True
        self._legacy_configuration = setup is None
        initial = setup if setup is not None else (get_scenario('sun-earth-jupiter')
            if star == 'sun' and tuple(planets) == ('earth', 'jupiter') and population is None
            else legacy_setup(star, planets, population=population))
        self.session = ExperimentSession(initial)
        self.distances = [PLANETS[p].distance for p in planets]
        self.paused = True
        self.following = False
        self.frame_mode = 'sun-relative'
        self.selected_index = 0
        self._drag = None
        self.fig = plt.figure(figsize=(14.8, 9.2), facecolor=BACKGROUND)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title('Star system | Observatory')
        self.ax = self.fig.add_axes((0, 0, 1, 1), zorder=0)
        self._viewport_size = (self.fig.bbox.width, self.fig.bbox.height)
        self.sidebar = Sidebar(self)
        self.star_picker = self.sidebar.star_picker
        self.planet_picker = self.sidebar.planet_picker
        self.add = self.sidebar.catalog_add
        self.population_button = self.sidebar.population
        self.reset, self.pause, self.step = self.sidebar.reset, self.sidebar.pause, self.sidebar.step
        self.home, self.follow, self.speed = self.sidebar.home, self.sidebar.follow, self.sidebar.speed
        self.inspector, self.mode, self.status = self.sidebar.inspector, self.sidebar.mode, self.sidebar.status
        self.experiment_name = ''
        self.set_experiment_name(setup.name if setup is not None else self._setup_name())
        self.sliders = []
        self.star_picker.on_clicked(self.change_star)
        self.add.on_clicked(self.add_planet)
        self.reset.on_clicked(lambda event: self.restart())
        self.pause.on_clicked(self.toggle_pause)
        self.step.on_clicked(self.step_once)
        self.home.on_clicked(self.fit_view)
        self.follow.on_clicked(self.toggle_follow)
        self.population_button.on_clicked(self.edit_population)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('resize_event', self._resize_view)
        self.painter = CanvasPainter(self.fig.canvas, overlay_axes=self.sidebar.axes + [self.sidebar.menu.ax])
        self.rebuild_sliders()
        self.restart()
        self.speed.on_changed(lambda value: self._refresh())
        self.timer = self.fig.canvas.new_timer(interval=round(FRAME_SECONDS*1000))
        self.timer.add_callback(self._on_timer)
        self.fig.canvas.mpl_connect('close_event', self.close)
        self.timer.start()

    def _setup_name(self):
        if len(self.planets) > 3:
            return f'{STARS[self.star].name} / {len(self.planets)} planets'
        return ' / '.join([STARS[self.star].name, *(PLANETS[p].name for p in self.planets)])

    @property
    def initial_setup(self):
        return self.session.initial_setup

    @initial_setup.setter
    def initial_setup(self, setup):
        self.session.update_draft(setup)
        self.session.initial_setup = self.session.draft

    @property
    def prepared(self):
        return self.session.prepared

    @prepared.setter
    def prepared(self, prepared):
        self.session.prepared = prepared

    def edit_experiment(self, event=None):
        if self.body_editor is not None or self.population_editor is not None or self.experiment_editor is not None or self.calculation_editor is not None:
            return
        from .sidebar import ExperimentEditor
        self.experiment_editor = ExperimentEditor(self)
        self.paused = True
        if self.playback is not None:
            self.playback.pause()
        self.pending = 0.
        self._drag = None
        self.sidebar._sync()
        self._refresh()
        self.experiment_editor.fig.show()

    def select_setup(self, setup):
        if self.calculation is not None and not self.calculation.done:
            self.cancel_calculation()
        self.playback = None
        self._selected_id = None
        prepared = self.session.select(setup)
        self._legacy_configuration = False
        self.preparing = self.paused = True
        self.following = False
        self.selected_index = 0
        self._install_prepared(prepared, reset=True)

    def load_setup(self, path):
        from .persistence import load_setup
        self.select_setup(load_setup(path))

    def save_current_setup(self, path, name):
        if self.playback is None:
            return self.session.save_current(path, name)
        from .experiments import capture_current_setup
        from .persistence import save_setup
        source = self.playback.recording.source_setup
        if source is None:
            raise ValueError('This recording has no source setup definition for capture provenance.')
        captured = capture_current_setup(self.prepared, source, name)
        save_setup(captured, path)
        return captured

    def rename_setup(self, name):
        from dataclasses import replace
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Experiment name must be nonempty.')
        if self.playback is not None:
            self._enter_preparation()
        self.initial_setup = replace(self.initial_setup, name=name.strip())
        self.prepared = replace(self.prepared, setup=self.initial_setup)
        self.set_experiment_name(self.initial_setup.name)

    def set_experiment_name(self, name: str):
        if self.experiment_name == str(name):
            return
        self.experiment_name = str(name)
        from textwrap import fill
        self.sidebar.title.set_text(fill(self.experiment_name, 40, max_lines=2, placeholder='…'))
        if hasattr(self, 'painter'):
            self.painter.invalidate()
            self.fig.canvas.draw_idle()

    def toggle_sidebar(self, event=None):
        self._drag = None
        self.sidebar.set_visible(not self.sidebar.visible)

    def set_frame_mode(self, mode):
        if any((self.body_editor, self.population_editor, self.experiment_editor, self.calculation_editor)):
            return
        try:
            self.scene.set_frame_mode(mode)
        except ValueError as exc:
            self.status.set_text(str(exc))
        else:
            self.frame_mode = mode
            self._drag = None
        self._refresh()

    def toggle_inertial_frame(self, event=None):
        self.set_frame_mode('world' if self.scene.frame_mode == 'sun-relative' else 'sun-relative')

    def prepare_jupiter_control(self):
        from .scenarios import make_jupiter_comparison
        _, control = make_jupiter_comparison(self.initial_setup)
        self.select_setup(control)
        self.status.set_text('No-Jupiter control: identical resolved Sun/rock vectors and fixed dt. Save this actual control; calculate the same duration / stride for matched samples.')
        self._refresh()

    def _resize_view(self, event=None):
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        width, height = self.fig.bbox.width, self.fig.bbox.height
        if not width or not height:
            return
        scale = (xlim[1]-xlim[0]) / self._viewport_size[0]
        cx, cy = np.mean(xlim), np.mean(ylim)
        self._viewport_size = (width, height)
        self.ax.set_xlim(cx-scale*width/2, cx+scale*width/2)
        self.ax.set_ylim(cy-scale*height/2, cy+scale*height/2)
        if hasattr(self, 'painter'):
            self.painter.invalidate()

    def edit_population(self, event=None):
        if self.body_editor is not None or self.population_editor is not None or self.experiment_editor is not None or self.calculation_editor is not None:
            return
        if self.mode_name == "CALCULATING":
            return
        if self.playback is not None:
            self._enter_preparation()
        from .population_editor import PopulationEditor
        self.population_editor = PopulationEditor(self)
        self.paused = True
        self._drag = None
        self.pending = 0.
        self.last = time.monotonic()
        self._set_population_editing(True)
        self._refresh()
        self.population_editor.fig.show()

    def _set_population_editing(self, editing):
        self.sidebar._sync()
        for widget in self.sliders:
            widget.set_active(not editing and self.sidebar.visible and self.sidebar.catalog_open and self.preparing)

    def close(self, event=None):
        self.timer.stop()
        if self.body_editor is not None:
            self.body_editor.finish()
        if self.population_editor is not None:
            self.population_editor.finish(closing=True)
        if self.experiment_editor is not None:
            self.experiment_editor.finish()
        if self.calculation_editor is not None:
            self.calculation_editor.finish()

    def _on_timer(self):
        started = time.perf_counter()
        self.update(None)
        # Tk schedules its next tick after this callback completes. Account for
        # work already done instead of adding a full frame's sleep on top.
        remaining = FRAME_SECONDS - (time.perf_counter()-started)
        self.timer.interval = max(1, round(remaining*1000))

    def rebuild_sliders(self):
        for slider in self.sliders:
            self.sidebar.catalog_widgets.remove(slider)
            self.sidebar.axes.remove(slider.ax)
            slider.disconnect_events()
            slider.ax.remove()
        if hasattr(self, 'painter'):
            self.painter.overlay_axes = self.sidebar.axes + [self.sidebar.menu.ax]
        self.sliders = []
        minimum = max(.005, STARS[self.star].radius*SOLAR_RADIUS_AU*1.02)
        for i, (p, a) in enumerate(zip(self.planets, self.distances)):
            a = max(minimum, a)
            self.distances[i] = a
            slider = Slider(self.fig.add_axes((.105, .285-i*.023, .12, .009), zorder=14), PLANETS[p].name,
                            np.log10(minimum), np.log10(80), valinit=np.log10(a),
                            color=ACCENT, track_color='#22374b')
            _style_axes(slider.ax)
            slider.label.set_fontsize(8)
            slider.label.set_color(TEXT)
            slider.valtext.set_fontsize(8)
            slider.valtext.set_color(MUTED)
            slider.on_changed(lambda value, index=i: self.distance_changed(index, value))
            slider.valtext.set_text(f'{a:.3g} AU')
            self.sliders.append(slider)
            slider.ax.set_visible(self.sidebar.visible and self.sidebar.catalog_open)
            slider.set_active(self.sidebar.visible and self.sidebar.catalog_open and self.population_editor is None and self.body_editor is None and self.preparing)
            self.sidebar.catalog_widgets.append(slider)
            self.sidebar.axes.append(slider.ax)
        if hasattr(self, 'painter'):
            self.painter.overlay_axes = self.sidebar.axes + [self.sidebar.menu.ax]

    def change_star(self, key):
        if self.population is not None and self.population.inner <= STARS[key].radius*SOLAR_RADIUS_AU:
            self.status.set_text('Population would be inside this star; choose a smaller star or wider radii.')
            # Restore the picker without firing another change callback.
            self.star_picker.eventson = False
            self.star_picker.set_active(list(STARS).index(self.star))
            self.star_picker.eventson = True
            self.painter.invalidate()
            self.fig.canvas.draw_idle()
            return
        self.star = key
        self.set_experiment_name(self._setup_name())
        self.rebuild_sliders()
        self._restart_legacy()

    def add_planet(self, event):
        p = self.planet_picker.value_selected
        if p not in self.planets:
            self.planets.append(p)
            self.distances.append(PLANETS[p].distance)
            self.set_experiment_name(self._setup_name())
            self.rebuild_sliders()
            self._restart_legacy()

    def distance_changed(self, index, value):
        self.distances[index] = 10**value
        self.sliders[index].valtext.set_text(f'{10**value:.3g} AU')
        self._restart_legacy()

    def _restart_legacy(self):
        self.initial_setup = legacy_setup(self.star, self.planets, self.distances, self.population)
        self._legacy_configuration = True
        self.restart()

    @property
    def selected_id(self):
        if self._selected_id is not None and self._selected_id not in self.prepared.body_ids:
            return self._selected_id
        return self.prepared.body_ids[self.selected_index]

    @selected_id.setter
    def selected_id(self, value):
        self._selected_id = value
        if value in self.prepared.body_ids:
            self.selected_index = self.prepared.body_ids.index(value)

    @property
    def mode_name(self):
        return ('CALCULATING' if self.calculation is not None and not self.calculation.done else
                'RECORDED' if self.playback is not None else 'PREP' if self.preparing else 'LIVE')

    def _enter_preparation(self):
        if self.calculation is not None and not self.calculation.done:
            self.cancel_calculation()
        self.playback = None
        self.preparing = self.paused = True
        self._selected_id = None
        self._install_prepared(self.session.restart(), reset=True)

    def edit_calculation(self, event=None):
        if self.calculation is not None and not self.calculation.done:
            self.cancel_calculation()
            return
        if any((self.body_editor, self.population_editor, self.experiment_editor, self.calculation_editor)):
            return
        from .sidebar import CalculationEditor
        self.calculation_editor = CalculationEditor(self)
        self.paused = True
        if self.playback is not None:
            self.playback.pause()
        self.sidebar._sync()
        self._refresh()
        self.calculation_editor.fig.show()

    def start_calculation(self, request):
        calculation = Calculation(self.initial_setup, request)  # preflight before replacing the displayed state
        if self.calculation is not None and not self.calculation.done:
            self.cancel_calculation()
        self.playback = None
        self.calculation = calculation
        self.preparing, self.paused = False, True
        self._selected_id = None
        self._install_prepared(calculation.prepared, reset=True)
        self._calculation_status()
        self.sidebar._sync()
        self._refresh()

    def _calculation_status(self):
        calc = self.calculation
        self.status.set_text(f'{calc.completed_steps:,}/{calc.total_steps:,} steps ({calc.progress:.1%})\nCancel calculation available.')
        self.scene.result_info.set_text(f'CALCULATING / {calc.source_name}\n'
            f'{calc.completed_steps:,}/{calc.total_steps:,} steps ({calc.progress:.1%}) / '
            f'{calc.prepared.system.time:.8g} of {calc.resolved_end_time:.8g} yr\n'
            f'Requested {calc.request.duration_years:.8g} yr / stride {calc.request.sample_stride} / '
            f'budget {calc.request.max_bytes/1048576:g} MiB\nUse Cancel calculation to retain partial output.')

    def _finish_calculation(self):
        result = self.calculation.result
        self.session.retain_recording(result)
        self.show_recording(result)

    def cancel_calculation(self, event=None):
        if self.calculation is not None and not self.calculation.done:
            self.calculation.cancel()
            self._finish_calculation()

    def show_recording(self, result=None):
        if self.calculation is not None and not self.calculation.done:
            return
        if result is None:
            if not self.session.recordings:
                return
            # Repeated clicks cycle all retained results, including older sources.
            current = next((i for i, run in enumerate(self.session.recordings)
                            if self.playback is not None and run is self.playback.recording), len(self.session.recordings))
            result = self.session.recordings[(current-1) % len(self.session.recordings)]
        self.playback = Playback(result)
        self.preparing, self.paused = False, True
        self._selected_id = None
        self._display_recorded_frame(reset=True)

    def _display_recorded_frame(self, reset=False):
        from .core import System
        frame = self.playback.current_frame
        selected = self.selected_id if not reset else None
        membership_changed = frame.body_ids != self.prepared.body_ids
        prepared = PreparedSystem(System(frame.masses, frame.r, frame.v, frame.dt, frame.time),
            frame.body_ids, frame.body_specs, events=frame.events, reference_ids=frame.reference_ids)
        self.prepared, self.system = prepared, prepared.system
        self.history = list(self.playback.recording.frames[max(0, self.playback.index-MAX_HISTORY+1):self.playback.index+1])
        if reset:
            self.scene = draw_system(self.ax, prepared, self.history, frame_mode=self.frame_mode)
            self.selected_index = 0
            self.following = False
        elif selected is not None:
            self.selected_id = selected
        self.halted = self.playback.recording.status == 'halted'
        self.paused = not self.playback.playing
        run = self.playback.recording
        if reset or membership_changed:
            self.scene.update(prepared, self.history)
            self._sync_painter()
        if reset:
            self.set_experiment_name(run.source_name)
            self.sidebar._sync()
        self.status.set_text(f'{run.status.upper()}: {run.reason or "Calculation finished."}')
        from textwrap import fill
        self.scene.result_info.set_text(f'RECORDED / {run.status.upper()} / sample {self.playback.index+1:,}/{len(run.frames):,}\n'
            f'Source: {run.source_name} [{run.setup_digest[:12]}]\n'
            f'Requested {run.requested_end_time:.8g} yr / resolved {run.resolved_end_time:.8g} yr / '
            f'actual {run.actual_end_time:.8g} yr\n'+fill(run.reason or 'Calculation finished.', 78))
        from .scenarios import compare_recordings
        for other in reversed(self.session.recordings):
            if other is run:
                continue
            try:
                summary = compare_recordings(run, other)
            except ValueError:
                continue
            self.scene.result_info.set_text(self.scene.result_info.get_text()+'\n'+fill(summary.description, 100))
            break
        self._refresh()

    def run_live(self, event=None):
        if any((self.body_editor, self.population_editor, self.experiment_editor, self.calculation_editor)):
            return
        self._enter_preparation()
        self.preparing, self.paused = False, False
        self.last = time.monotonic()
        self.sidebar._sync()
        self._refresh()

    def prepare_setup(self, event=None):
        if self.body_editor is not None or self.population_editor is not None or self.experiment_editor is not None or self.calculation_editor is not None:
            return
        self._enter_preparation()
        self.following = False
        self.session.begin_draft()
        self.restart()
        self.status.set_text('PREPARATION: editing starting setup. Run live begins evolution.')
        self._refresh()

    def edit_body(self, event=None, body_id=None):
        if self.body_editor is not None or self.population_editor is not None or self.experiment_editor is not None or self.calculation_editor is not None:
            return
        if self.mode_name == "CALCULATING":
            return
        if self.playback is not None:
            self._enter_preparation()
        if body_id is not None and not self.preparing:
            self.status.set_text('Edit/remove applies to preparation. Choose Prepare setup first.')
            self._refresh()
            return
        if body_id is not None and body_id not in {b.id for b in self.initial_setup.bodies}:
            self.status.set_text('Generated particles are authored through population settings; starting setup preserved.')
            self._refresh()
            return
        from .authoring import BodyEditor
        self.body_editor = BodyEditor(self, body_id)
        self.paused = True
        self.pending = 0.
        self._drag = None
        self.sidebar.set_catalog(False)
        self.sidebar._sync()
        self._sync_painter()
        self._refresh()

    def remove_body(self, event=None):
        if not self.preparing or self.body_editor is not None or self.population_editor is not None or self.experiment_editor is not None or self.calculation_editor is not None:
            return
        if self.selected_id not in {b.id for b in self.initial_setup.bodies}:
            self.status.set_text('Generated particles are authored through population settings; starting setup preserved.')
            self._refresh()
            return
        from dataclasses import replace
        try:
            setup = replace(self.initial_setup, bodies=tuple(b for b in self.initial_setup.bodies if b.id != self.selected_id))
            prepared = build_system(setup)
        except ValueError as exc:
            self.status.set_text(str(exc)); self._refresh(); return
        self.initial_setup = setup
        self._legacy_configuration = False
        self.selected_index = 0
        self._install_prepared(prepared, reset=True)

    def _sync_painter(self):
        editor = self.body_editor
        self.painter.overlay_axes = self.sidebar.axes + [self.sidebar.menu.ax] + (editor.axes if editor else [])
        artists = self.scene.moving_artists + (editor.artists if editor else [])
        self.painter.set_artists(list(dict.fromkeys(artists)))

    def restart(self):
        if self.body_editor is not None or self.population_editor is not None or self.experiment_editor is not None or self.calculation_editor is not None:
            return
        if self.mode_name == 'CALCULATING':
            return
        if self.playback is not None:
            self.show_recording(self.playback.recording)
            return
        self._install_prepared(self.session.restart(), reset=True)

    def _install_prepared(self, prepared, reset=False):
        from dataclasses import replace
        if reset:
            prepared = replace(prepared, setup=self.initial_setup)
        self.prepared, self.system = prepared, prepared.system
        self.population = self.initial_setup.population.settings if self.initial_setup.population is not None else None
        self.pending = 0.
        self.halted = False
        self._drag = None
        self.last = time.monotonic()
        if reset:
            self._selected_id = None
            self.history, self.trails = [StateFrame.capture(prepared)], []
            self.status.set_text('')
            self.scene = draw_system(self.ax, prepared, frame_mode=self.frame_mode)
            self._viewport_size = (self.fig.bbox.width, self.fig.bbox.height)
            self.ax.callbacks.connect('xlim_changed', self.painter.invalidate)
            self.ax.callbacks.connect('ylim_changed', self.painter.invalidate)
        else:
            self.history.append(StateFrame.capture(prepared))
            self.history = self.history[-MAX_HISTORY:]
            self.scene.update(prepared, self.history)
        self.selected_index = min(self.selected_index, len(self.system.masses)-1)
        self._sync_painter()
        self.set_experiment_name(self.initial_setup.name)
        self.sidebar._sync()
        self._refresh()

    def _refresh(self):
        self.scene.update(self.prepared, self.history)
        self.frame_mode = self.scene.frame_mode
        self.scene.select(self.selected_id)
        index = self.selected_index
        missing = self.selected_id not in self.prepared.body_ids
        if missing:
            details = f"{self.selected_id}: missing / removed\nSee recorded events; no replacement assigned."
            if self.prepared.events:
                details += "\n" + str(self.prepared.events[-1].message)
        else:
            body = self.prepared.body_specs[index]
            mass = f'{self.system.masses[index]:.6g}'
            radius = 'unknown' if body.radius is None else f'{body.radius*AU_KM:,.5g} km'
            if body.kind == 'particle':
                distance = np.linalg.norm(self.system.r[index]-self.scene.origin)
                speed = np.linalg.norm(self.system.v[index]-self.system.v[self.prepared.body_ids.index(self.scene.origin_id)])
                origin_name = self.prepared.body_specs[self.prepared.body_ids.index(self.scene.origin_id)].name
                details = f'Test particle {self.selected_id}\n{distance:.4g} AU from {origin_name}\n{speed:.4g} AU / yr relative to {origin_name}'
                from .frames import reference_id
                try:
                    jupiter = self.prepared.body_ids.index(reference_id(self.prepared, 'jupiter'))
                except ValueError:
                    details += '\nJupiter: absent / ambiguous'
                else:
                    separation = np.linalg.norm(self.system.r[index]-self.system.r[jupiter])
                    relative_v = self.system.v[index]-self.system.v[jupiter]
                    details += f'\nJupiter separation: {separation:.6g} AU\nJupiter-relative physical inertial v\n({relative_v[0]:.5g}, {relative_v[1]:.5g}) AU/yr'
            else:
                details = f'{body.name}\n{mass} solar masses / radius {radius}'
                if body.temperature is not None:
                    details += f'\n{body.temperature:,.0f} K'
                details += f'\nID: {self.selected_id}\n{body.provenance}'
                if body.kind == 'moon':
                    relative = relative_state(StateFrame.capture(self.prepared), self.selected_id)
                    if relative is None:
                        details += '\nParent: no initial association'
                    elif relative.missing:
                        details += f'\nParent {relative.reference_id}: missing / removed\nSee state events; no replacement assigned.'
                    else:
                        details += f'\nParent: {relative.reference_name} [{relative.reference_id}]\nCurrent separation: {relative.separation*AU_KM:,.7g} km\n{relative.separation:.8g} AU'
            r, v = self.system.r[index], self.system.v[index]
            details += f'\nWorld r = ({r[0]:.5g}, {r[1]:.5g}) AU\nWorld v = ({v[0]:.5g}, {v[1]:.5g}) AU/yr'
        self.inspector.set_text(details)
        self.sidebar.set_bodies([(key, f'{spec.name} [{key if spec.kind == "particle" else key[:8]}]')
                                for key,spec in zip(self.prepared.body_ids,self.prepared.body_specs)], self.selected_id)
        self.sidebar.update(self.system.time, self.speed.val, None if missing else index)
        if self.following and not missing and self.body_editor is None:
            center = self.scene.positions[index]
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
            half_width, half_height = np.diff(xlim)[0]/2, np.diff(ylim)[0]/2
            new_x = (center[0]-half_width, center[0]+half_width)
            new_y = (center[1]-half_height, center[1]+half_height)
            if new_x != xlim: self.ax.set_xlim(new_x)
            if new_y != ylim: self.ax.set_ylim(new_y)
        self.mode.set_text('HALTED' if self.halted and self.playback is None else self.mode_name)
        self.mode.set_color('#efb17b' if self.halted else MUTED if self.paused else ACCENT)
        for label, text in ((self.pause.label, ('Play' if self.playback is not None else 'Run live' if self.preparing else 'Resume') if self.paused else 'Pause'),
                            (self.step.label, 'Next sample' if self.playback is not None else 'Step'),
                            (self.sidebar.calculate.label, 'Cancel calculation' if self.mode_name == 'CALCULATING' else 'Calculate...'),
                            (self.sidebar.results.label, f'Results ({len(self.session.recordings)})'),
                            (self.follow.label, 'Unfollow' if self.following else 'Follow body')):
            if label.get_text() != text:
                label.set_text(text)
                self.painter.invalidate()
        self.painter.paint()

    def toggle_pause(self, event=None):
        if self.population_editor is not None or self.body_editor is not None or self.experiment_editor is not None or self.calculation_editor is not None:
            return
        if self.mode_name == "CALCULATING":
            return
        if self.playback is not None:
            if self.playback.playing: self.playback.pause()
            else: self.playback.play()
            self._display_recorded_frame()
            return
        self.paused = not self.paused
        if not self.paused:
            self.preparing = False
        self.sidebar._sync()
        self.pending = 0.
        self.last = time.monotonic()
        self._refresh()

    def _advance(self, count, budget_seconds=None):
        if self.playback is not None or self.mode_name == 'CALCULATING':
            return 0
        deadline = None if budget_seconds is None else time.perf_counter()+budget_seconds
        completed = 0
        for _ in range(count):
            try:
                self.system.step()
            except ValueError as exc:
                self.halted = True
                self.status.set_text(str(exc))
                break
            self.preparing = False
            self.trails.append((self.system.r-self.system.r[0]).copy())
            self.history.append(StateFrame.capture(self.prepared))
            completed += 1
            if deadline is not None and time.perf_counter() >= deadline:
                break
        self.trails = self.trails[-MAX_HISTORY:]
        self.history = self.history[-MAX_HISTORY:]
        return completed

    def step_once(self, event=None):
        if self.population_editor is not None or self.body_editor is not None or self.experiment_editor is not None or self.calculation_editor is not None:
            return
        if self.mode_name == "CALCULATING":
            return
        if self.playback is not None:
            self.playback.next_sample()
            self._display_recorded_frame()
            return
        self.paused = True
        self.pending = 0.
        self.last = time.monotonic()
        if not self.halted:
            self._advance(1)
        self.sidebar._sync()
        self._refresh()

    def fit_view(self, event=None):
        self.following = False
        self.scene.fit_view()
        self._refresh()

    def toggle_follow(self, event=None):
        self.following = not self.following
        self._refresh()

    def _toolbar_active(self):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        return toolbar is not None and bool(toolbar.mode)

    def on_scroll(self, event):
        if self.population_editor is not None or self.experiment_editor is not None or self.calculation_editor is not None:
            return
        if self.sidebar.contains_event(event) or event.inaxes != self.ax or event.xdata is None or self._toolbar_active():
            return
        scale = 1.2**(-np.clip(event.step, -5, 5))
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        width = (xlim[1]-xlim[0])*scale
        if not .08 < width < 100:
            return
        self.ax.set_xlim(event.xdata+(np.asarray(xlim)-event.xdata)*scale)
        self.ax.set_ylim(event.ydata+(np.asarray(ylim)-event.ydata)*scale)
        if self.body_editor is not None: self.body_editor.preview()
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if self.population_editor is not None or self.experiment_editor is not None or self.calculation_editor is not None:
            return
        if self.body_editor is not None:
            try: self.body_editor.on_press(event)
            except ValueError as exc: self.body_editor.error.set_text(str(exc)); self.painter.paint()
            return
        if not self.sidebar.contains_event(event) and event.inaxes == self.ax and event.button == 1 and not self._toolbar_active():
            self._drag = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

    def on_motion(self, event):
        if self.population_editor is not None or self.experiment_editor is not None or self.calculation_editor is not None:
            return
        if self.body_editor is not None:
            self.body_editor.on_motion(event)
            return
        if self._drag is None:
            return
        if self.sidebar.contains_event(event):
            self._drag = None
            return
        x, y, xlim, ylim = self._drag
        if np.hypot(event.x-x, event.y-y) < 4:
            return
        self.following = False
        dx = (event.x-x)/self.ax.bbox.width*(xlim[1]-xlim[0])
        dy = (event.y-y)/self.ax.bbox.height*(ylim[1]-ylim[0])
        self.ax.set_xlim(np.asarray(xlim)-dx)
        self.ax.set_ylim(np.asarray(ylim)-dy)
        self.follow.label.set_text('Follow body')
        self.painter.invalidate()
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.population_editor is not None or self.experiment_editor is not None or self.calculation_editor is not None:
            return
        if self.body_editor is not None:
            self.body_editor.dragging = False
            return
        drag, self._drag = self._drag, None
        if drag is None or self.sidebar.contains_event(event) or event.inaxes != self.ax:
            return
        if np.hypot(event.x-drag[0], event.y-drag[1]) < 4:
            pixels = self.ax.transData.transform(self.scene.positions)
            distances = np.linalg.norm(pixels - (event.x, event.y), axis=1)
            index = int(np.argmin(distances))
            if distances[index] <= 22:
                self.selected_id = self.prepared.body_ids[index]
                self._refresh()

    def on_key(self, event):
        if self.calculation_editor is not None:
            if event.key == 'escape': self.calculation_editor.finish()
            return
        if self.experiment_editor is not None:
            if event.key == 'escape': self.experiment_editor.finish()
            return
        if self.body_editor is not None:
            if event.key == 'escape': self.body_editor.finish()
            return
        if self.population_editor is not None:
            if event.key == 'escape': self.population_editor.finish()
            return
        if event.key == 'm':
            self.toggle_sidebar()
        elif event.key in (' ', 'space'):
            self.toggle_pause()
        elif event.key == '.':
            self.step_once()
        elif event.key in ('home', '0'):
            self.fit_view()

    def update(self, frame):
        now = time.monotonic()
        elapsed = min(now-self.last, .1)
        self.last = now
        if self.mode_name == "CALCULATING":
            if any((self.body_editor, self.population_editor, self.experiment_editor, self.calculation_editor)):
                return ()
            self.calculation.advance(150, budget_seconds=PHYSICS_BUDGET_SECONDS)
            if self.calculation.done:
                self._finish_calculation()
            else:
                self._calculation_status()
                self._refresh()
            return ()
        if self.playback is not None:
            if self.playback.playing:
                self.playback.advance(elapsed, self.speed.val)
                self._display_recorded_frame()
            return ()
        if not self.halted and not self.paused:
            self.pending += elapsed*self.speed.val
            steps = min(150, int(self.pending/self.system.dt))
            if steps:
                completed = self._advance(steps, budget_seconds=PHYSICS_BUDGET_SECONDS)
                self.pending -= completed*self.system.dt
                if not self.halted:
                    self.status.set_text('Compute limit: running slower than requested' if self.pending > self.system.dt*300 else '')
                self.pending = min(self.pending, self.system.dt*600)
                self._refresh()
        return ()
