"""Overhead observatory controls, using the existing Matplotlib desktop backend."""
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, Slider, TextBox

from .core import make_system, SOLAR_RADIUS_AU
from .data import STARS, PLANETS
from .populations import Population, with_population
from .rendering import BACKGROUND, PANEL, TEXT, MUTED, ACCENT, CanvasPainter, draw
from .sidebar import Sidebar

FRAME_SECONDS = 1 / 60
PHYSICS_BUDGET_SECONDS = .006


def _style_axes(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _button(fig, bounds, label):
    button = Button(fig.add_axes(bounds), label, color='#1b3044', hovercolor='#294b60', useblit=False)
    button.label.set_color(TEXT)
    button.label.set_fontsize(9)
    for spine in button.ax.spines.values():
        spine.set_edgecolor('#294055')
    return button


class PopulationEditor:
    """Draft settings in a separate window; only Apply replaces initial state."""

    def __init__(self, app):
        self.app = app
        self.was_paused = app.paused
        self.finished = False
        settings = app.population or Population()
        self.fig = plt.figure(figsize=(6.4, 6.8), facecolor=PANEL)
        self.fig.canvas.manager.set_window_title('Population settings')
        self.fig.text(.08, .94, 'Population settings', fontsize=16, color=TEXT)
        self.fig.text(.08, .89, f'Reference: {STARS[app.star].name} / circular prograde',
                      fontsize=10, color=MUTED)
        self.kind = RadioButtons(self.fig.add_axes((.08, .70, .35, .14)),
                                 ['ring', 'disk', 'belt'], active=['ring', 'disk', 'belt'].index(settings.kind),
                                 activecolor=ACCENT, useblit=False)
        self.angles = RadioButtons(self.fig.add_axes((.57, .70, .35, .14)),
                                   ['random', 'even'], active=['random', 'even'].index(settings.angles),
                                   activecolor=ACCENT, useblit=False)
        for picker in (self.kind, self.angles):
            _style_axes(picker.ax)
            picker.set_radio_props({'edgecolor': MUTED})
            for label in picker.labels:
                label.set_color(TEXT)
        self.fields = {}
        for key, label, value, y in (
            ('count', 'Particles (1-2000)', settings.count, .61),
            ('inner', 'Inner radius (AU)', settings.inner, .52),
            ('outer', 'Outer radius (AU)', settings.outer, .43),
            ('seed', 'Seed', settings.seed, .34),
        ):
            field = TextBox(self.fig.add_axes((.47, y, .42, .055)), label+'  ',
                            initial=str(value), color='#1b3044', hovercolor='#294b60')
            field.label.set_color(TEXT)
            field.text_disp.set_color(TEXT)
            self.fields[key] = field
        self.error = self.fig.text(.08, .26, '', color='#efb17b', fontsize=9,
                                  va='top', wrap=True)
        self.apply = _button(self.fig, (.08, .13, .40, .055), 'Apply and restart')
        self.remove = _button(self.fig, (.52, .13, .40, .055), 'Remove and restart')
        self.remove.set_active(app.population is not None)
        self.remove.label.set_color(TEXT if app.population is not None else MUTED)
        self.cancel = _button(self.fig, (.62, .045, .30, .05), 'Cancel')
        self.apply.on_clicked(self.apply_settings)
        self.remove.on_clicked(lambda event: self.finish(None, apply=True))
        self.cancel.on_clicked(lambda event: self.finish())
        self.kind.on_clicked(self.kind_changed)
        self.kind_changed(settings.kind)
        self.fig.canvas.mpl_connect('close_event', lambda event: self.finish())
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key == 'escape':
            self.finish()

    def kind_changed(self, kind):
        # Rings have one radius; preserve the outer draft for disk/belt switches.
        enabled = kind != 'ring'
        self.fields['inner'].label.set_text('Inner radius (AU)  ' if enabled else 'Radius (AU)  ')
        self.fields['outer'].set_active(enabled)
        self.fields['outer'].text_disp.set_color(TEXT if enabled else MUTED)
        self.fig.canvas.draw_idle()

    def apply_settings(self, event=None):
        try:
            try:
                count = int(self.fields['count'].text)
                seed = int(self.fields['seed'].text)
                inner = float(self.fields['inner'].text)
                outer = inner if self.kind.value_selected == 'ring' else float(self.fields['outer'].text)
            except ValueError:
                raise ValueError('Count and seed must be integers; radii must be numbers.') from None
            settings = Population(kind=self.kind.value_selected, count=count, seed=seed,
                                  inner=inner, outer=outer, angles=self.angles.value_selected)
            with_population(make_system(self.app.star, self.app.planets, self.app.distances),
                            settings, STARS[self.app.star].radius*SOLAR_RADIUS_AU)
        except ValueError as exc:
            self.error.set_text(str(exc))
            self.fig.canvas.draw_idle()
            return
        self.finish(settings, apply=True)

    def finish(self, settings=None, apply=False):
        if self.finished:
            return
        self.finished = True
        self.app.population_editor = None
        self.app._set_population_editing(False)
        if apply:
            self.app.population = settings
            self.app.paused = True
            self.app.following = False
            self.app.selected_index = 0
            self.app.restart()
        else:
            self.app.paused = self.was_paused
            self.app.pending = 0.
            self.app.last = time.monotonic()
            self.app._refresh()
        plt.close(self.fig)


class App:
    def __init__(self, star='sun', planets=('earth', 'jupiter'), population=None):
        if population is not None and population.reference != 0:
            raise ValueError('The current observatory supports star-relative populations only.')
        self.star, self.planets = star, list(planets)
        self.population = population
        self.population_editor = None
        self.distances = [PLANETS[p].distance for p in planets]
        self.paused = False
        self.following = False
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
        self.set_experiment_name(self._setup_name())
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

    def set_experiment_name(self, name: str):
        self.experiment_name = str(name)
        self.sidebar.title.set_text(self.experiment_name)
        if hasattr(self, 'painter'):
            self.painter.invalidate()
            self.fig.canvas.draw_idle()

    def toggle_sidebar(self, event=None):
        self._drag = None
        self.sidebar.set_visible(not self.sidebar.visible)

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
        if self.population_editor is not None:
            return
        self.population_editor = PopulationEditor(self)
        self.paused = True
        self.pending = 0.
        self.last = time.monotonic()
        self._set_population_editing(True)
        self._refresh()
        self.population_editor.fig.show()

    def _set_population_editing(self, editing):
        self.sidebar._sync()
        for widget in self.sliders:
            widget.set_active(not editing and self.sidebar.visible and self.sidebar.catalog_open)

    def close(self, event=None):
        self.timer.stop()
        if self.population_editor is not None:
            self.population_editor.finished = True
            plt.close(self.population_editor.fig)
            self.population_editor = None

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
            slider.set_active(self.sidebar.visible and self.sidebar.catalog_open and self.population_editor is None)
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
        self.restart()

    def add_planet(self, event):
        p = self.planet_picker.value_selected
        if p not in self.planets:
            self.planets.append(p)
            self.distances.append(PLANETS[p].distance)
            self.set_experiment_name(self._setup_name())
            self.rebuild_sliders()
            self.restart()

    def distance_changed(self, index, value):
        self.distances[index] = 10**value
        self.sliders[index].valtext.set_text(f'{10**value:.3g} AU')
        self.restart()

    def restart(self):
        self.system = make_system(self.star, self.planets, self.distances)
        if self.population is not None:
            self.system = with_population(self.system, self.population,
                                          STARS[self.star].radius*SOLAR_RADIUS_AU)
        self.trails = []
        self.pending = 0.
        self.halted = False
        self._drag = None
        self.status.set_text('')
        self.last = time.monotonic()
        self.scene = draw(self.ax, self.system, self.star, self.planets)
        self._viewport_size = (self.fig.bbox.width, self.fig.bbox.height)
        self.ax.callbacks.connect('xlim_changed', self.painter.invalidate)
        self.ax.callbacks.connect('ylim_changed', self.painter.invalidate)
        self.painter.set_artists(self.scene.moving_artists)
        self.selected_index = min(self.selected_index, len(self.system.masses)-1)
        self._refresh()

    def _refresh(self):
        self.scene.update(self.system, self.trails)
        self.scene.select(self.selected_index)
        index = self.selected_index
        if index == 0:
            body = STARS[self.star]
            details = f'{body.name}\n{body.mass:g} solar masses\n{body.radius:g} solar radii   /   {body.temperature:,.0f} K'
        elif index <= len(self.planets):
            key = self.planets[index-1]
            body = PLANETS[key]
            distance = np.linalg.norm(self.system.r[index]-self.system.r[0])
            details = f'{body.name}\n{body.radius:,.0f} km radius\n{distance:.4g} AU from star'
        else:
            distance = np.linalg.norm(self.system.r[index]-self.system.r[0])
            speed = np.linalg.norm(self.system.v[index]-self.system.v[0])
            details = (f'Test particle {index-len(self.planets)}\n'
                       f'{distance:.4g} AU from star\n{speed:.4g} AU / yr relative to star')
        self.inspector.set_text(details)
        rows = [(0, STARS[self.star].name)] + [(i+1, PLANETS[p].name) for i, p in enumerate(self.planets)]
        rows += [(i, f'Test particle {i-len(self.planets)}') for i in range(1+len(self.planets), len(self.system.masses))]
        self.sidebar.set_bodies(rows, index)
        self.sidebar.update(self.system.time, self.speed.val, index)
        if self.following:
            center = self.scene.positions[index]
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
            half_width, half_height = np.diff(xlim)[0]/2, np.diff(ylim)[0]/2
            new_x = (center[0]-half_width, center[0]+half_width)
            new_y = (center[1]-half_height, center[1]+half_height)
            if new_x != xlim:
                self.ax.set_xlim(new_x)
            if new_y != ylim:
                self.ax.set_ylim(new_y)
        self.mode.set_text('HALTED' if self.halted else 'PAUSED' if self.paused else 'LIVE')
        self.mode.set_color('#efb17b' if self.halted else MUTED if self.paused else ACCENT)
        for label, text in ((self.pause.label, 'Resume' if self.paused else 'Pause'),
                            (self.follow.label, 'Unfollow' if self.following else 'Follow body')):
            if label.get_text() != text:
                label.set_text(text)
                self.painter.invalidate()
        self.painter.paint()

    def toggle_pause(self, event=None):
        if self.population_editor is not None:
            return
        self.paused = not self.paused
        self.pending = 0.
        self.last = time.monotonic()
        self._refresh()

    def _advance(self, count, budget_seconds=None):
        deadline = None if budget_seconds is None else time.perf_counter()+budget_seconds
        completed = 0
        for _ in range(count):
            try:
                self.system.step()
            except ValueError as exc:
                self.halted = True
                self.status.set_text(str(exc))
                break
            self.trails.append((self.system.r-self.system.r[0]).copy())
            completed += 1
            if deadline is not None and time.perf_counter() >= deadline:
                break
        self.trails = self.trails[-1500:]
        return completed

    def step_once(self, event=None):
        if self.population_editor is not None:
            return
        self.paused = True
        self.pending = 0.
        self.last = time.monotonic()
        if not self.halted:
            self._advance(1)
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
        if self.sidebar.contains_event(event) or event.inaxes != self.ax or event.xdata is None or self._toolbar_active():
            return
        scale = 1.2**(-np.clip(event.step, -5, 5))
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        width = (xlim[1]-xlim[0])*scale
        if not .08 < width < 100:
            return
        self.ax.set_xlim(event.xdata+(np.asarray(xlim)-event.xdata)*scale)
        self.ax.set_ylim(event.ydata+(np.asarray(ylim)-event.ydata)*scale)
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if not self.sidebar.contains_event(event) and event.inaxes == self.ax and event.button == 1 and not self._toolbar_active():
            self._drag = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

    def on_motion(self, event):
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
        drag, self._drag = self._drag, None
        if drag is None or self.sidebar.contains_event(event) or event.inaxes != self.ax:
            return
        if np.hypot(event.x-drag[0], event.y-drag[1]) < 4:
            pixels = self.ax.transData.transform(self.scene.positions)
            distances = np.linalg.norm(pixels - (event.x, event.y), axis=1)
            index = int(np.argmin(distances))
            if distances[index] <= 22:
                self.selected_index = index
                self._refresh()

    def on_key(self, event):
        if self.population_editor is not None:
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
