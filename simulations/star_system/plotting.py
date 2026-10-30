"""Overhead observatory controls, using the existing Matplotlib desktop backend."""
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import RadioButtons, Button, Slider

from .core import make_system, SOLAR_RADIUS_AU
from .data import STARS, PLANETS
from .populations import with_population
from .rendering import BACKGROUND, PANEL, TEXT, MUTED, ACCENT, CanvasPainter, draw

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


class App:
    def __init__(self, star='sun', planets=('earth', 'jupiter'), population=None):
        if population is not None and population.reference != 0:
            raise ValueError('The current observatory supports star-relative populations only.')
        self.star, self.planets = star, list(planets)
        self.population = population
        self.distances = [PLANETS[p].distance for p in planets]
        self.paused = False
        self.following = False
        self.selected_index = 0
        self._drag = None
        self.fig = plt.figure(figsize=(14.8, 9.2), facecolor=BACKGROUND)
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title('Star system | Observatory')
        self.fig.text(.033, .952, 'ORBITAL OBSERVATORY', color=ACCENT, fontsize=11, weight='bold')
        self.fig.text(.033, .927, 'Build a system. Watch it move.', color=MUTED, fontsize=9)
        self.fig.text(.321, .948, 'A system in motion', color=TEXT, fontsize=22)
        self.fig.text(.322, .918, 'An overhead view of gravity at work', color=MUTED, fontsize=10)
        self.mode = self.fig.text(.975, .949, 'LIVE', color=ACCENT, fontsize=9, ha='right')
        for bottom, height in ((.55, .35), (.245, .29), (.035, .195)):
            panel = FancyBboxPatch((.02, bottom), .272, height,
                                   boxstyle='round,pad=0.006,rounding_size=0.012',
                                   transform=self.fig.transFigure, facecolor=PANEL,
                                   edgecolor='#1e3043', linewidth=.7, zorder=-1)
            self.fig.add_artist(panel)
        self.fig.text(.035, .874, 'CENTRAL STAR', color=MUTED, fontsize=8, weight='bold')
        self.fig.text(.164, .874, 'ADD A BODY', color=MUTED, fontsize=8, weight='bold')
        self.star_picker = RadioButtons(self.fig.add_axes((.03, .69, .122, .17)), list(STARS),
                                        active=list(STARS).index(star), activecolor=ACCENT, useblit=False)
        self.planet_picker = RadioButtons(self.fig.add_axes((.158, .605, .12, .255)), list(PLANETS), activecolor=ACCENT, useblit=False)
        for picker in (self.star_picker, self.planet_picker):
            _style_axes(picker.ax)
            picker.set_radio_props({'edgecolor': MUTED})
            for label in picker.labels:
                label.set_color(TEXT)
                label.set_fontsize(9)
        self.add = _button(self.fig, (.035, .567, .112, .033), 'Add planet')
        self.reset = _button(self.fig, (.162, .567, .112, .033), 'Restart')
        self.fig.text(.035, .508, 'ORBITAL DISTANCES', color=MUTED, fontsize=8, weight='bold')
        self.fig.text(.035, .485, 'Changes restart the system', color=MUTED, fontsize=8)
        self.fig.text(.035, .203, 'INSPECTOR', color=MUTED, fontsize=8, weight='bold')
        self.inspector = self.fig.text(.035, .179, '', color=TEXT, fontsize=10, va='top', linespacing=1.65)
        self.follow = _button(self.fig, (.035, .052, .112, .032), 'Follow body')
        self.ax = self.fig.add_axes((.315, .18, .67, .71))
        self.pause = _button(self.fig, (.325, .083, .085, .039), 'Pause')
        self.step = _button(self.fig, (.421, .083, .065, .039), 'Step')
        self.home = _button(self.fig, (.497, .083, .078, .039), 'Fit view')
        self.speed = Slider(self.fig.add_axes((.666, .096, .252, .014)), 'yr / sec', .001, 10.,
                            valinit=.5, valfmt='%.3g', color=ACCENT, track_color='#22374b')
        _style_axes(self.speed.ax)
        self.speed.label.set_color(MUTED)
        self.speed.valtext.set_color(TEXT)
        self.speed.label.set_fontsize(9)
        self.speed.valtext.set_fontsize(9)
        self.status = self.fig.text(.325, .145, '', color='#efb17b', fontsize=9)
        self.fig.text(.325, .04, 'SCROLL  zoom     DRAG  pan     CLICK  inspect     SPACE  pause     .  step     HOME  fit',
                      color=MUTED, fontsize=8)
        self.sliders = []
        self.star_picker.on_clicked(self.change_star)
        self.add.on_clicked(self.add_planet)
        self.reset.on_clicked(lambda event: self.restart())
        self.pause.on_clicked(self.toggle_pause)
        self.step.on_clicked(self.step_once)
        self.home.on_clicked(self.fit_view)
        self.follow.on_clicked(self.toggle_follow)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.painter = CanvasPainter(self.fig.canvas)
        self.rebuild_sliders()
        self.restart()
        self.timer = self.fig.canvas.new_timer(interval=round(FRAME_SECONDS*1000))
        self.timer.add_callback(self._on_timer)
        self.fig.canvas.mpl_connect('close_event', lambda event: self.timer.stop())
        self.timer.start()

    def _on_timer(self):
        started = time.perf_counter()
        self.update(None)
        # Tk schedules its next tick after this callback completes. Account for
        # work already done instead of adding a full frame's sleep on top.
        remaining = FRAME_SECONDS - (time.perf_counter()-started)
        self.timer.interval = max(1, round(remaining*1000))

    def rebuild_sliders(self):
        for slider in self.sliders:
            slider.disconnect_events()
            slider.ax.remove()
        self.sliders = []
        minimum = max(.005, STARS[self.star].radius*SOLAR_RADIUS_AU*1.02)
        for i, (p, a) in enumerate(zip(self.planets, self.distances)):
            a = max(minimum, a)
            self.distances[i] = a
            slider = Slider(self.fig.add_axes((.104, .462-i*.022, .122, .009)), PLANETS[p].name,
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
        self.rebuild_sliders()
        self.restart()

    def add_planet(self, event):
        p = self.planet_picker.value_selected
        if p not in self.planets:
            self.planets.append(p)
            self.distances.append(PLANETS[p].distance)
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
        self.ax.callbacks.connect('xlim_changed', self.painter.invalidate)
        self.ax.callbacks.connect('ylim_changed', self.painter.invalidate)
        self.painter.set_artists([*self.scene.moving_artists,
                                  self.inspector, self.mode, self.status])
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
            details = (f'Particle {index-len(self.planets)} / massless\n'
                       f'{distance:.4g} AU from star\n{speed:.4g} AU / yr relative to star')
        self.inspector.set_text(details)
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
        if event.inaxes != self.ax or event.xdata is None or self._toolbar_active():
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
        if event.inaxes == self.ax and event.button == 1 and not self._toolbar_active():
            self._drag = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

    def on_motion(self, event):
        if self._drag is None:
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
        if drag is None or event.inaxes != self.ax:
            return
        if np.hypot(event.x-drag[0], event.y-drag[1]) < 4:
            pixels = self.ax.transData.transform(self.scene.positions)
            distances = np.linalg.norm(pixels - (event.x, event.y), axis=1)
            index = int(np.argmin(distances))
            if distances[index] <= 22:
                self.selected_index = index
                self._refresh()

    def on_key(self, event):
        if event.key in (' ', 'space'):
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
