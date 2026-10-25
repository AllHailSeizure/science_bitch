"""Matplotlib live controls and shared headless drawing."""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Button, Slider
from .core import make_system, SOLAR_RADIUS_AU
from .data import STARS, PLANETS, blackbody_rgb, star_size


def coordinates(r):
    """Star-relative radial asinh mapping; direction is preserved."""
    radius = np.linalg.norm(r, axis=-1)
    return r * (np.arcsinh(radius/.05)/np.maximum(radius, 1e-30))[:,None]


def draw(ax, system, star, planets, trails=None):
    ax.clear()
    ax.set_facecolor('#080f20')
    relative = system.r-system.r[0]
    pos = coordinates(relative)
    extent = max(2., np.max(np.linalg.norm(pos, axis=1))*1.2)
    for radius in (.05, .1, 1, 5, 30):
        rr = np.arcsinh(radius/.05)
        if rr < extent:
            ax.add_patch(plt.Circle((0,0), rr, fill=False, color='#344057', lw=.6))
            ax.text(0,rr, f'{radius:g} AU', color='#8190a8', fontsize=8)
    if trails:
        for i,p in enumerate(planets):
            trail = coordinates(np.array(trails)[:,i+1,:])
            ax.plot(*trail.T, color=PLANETS[p].color, alpha=.45, lw=1)
    s = STARS[star]
    ax.scatter([0],[0], s=star_size(s)**2, color=[blackbody_rgb(s.temperature)], edgecolors='white', linewidths=.5, zorder=4)
    for i,p in enumerate(planets,1):
        body = PLANETS[p]
        ax.scatter(*pos[i], s=(5+2*np.log10(1+body.radius/1000))**2, color=body.color, zorder=5)
        ax.annotate(body.name, pos[i], xytext=(8,8), textcoords='offset points', color='white', fontsize=9)
    ax.set(xlim=(-extent,extent), ylim=(-extent,extent), aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{s.name}  •  {s.temperature:g} K  •  {s.mass:g} M☉  •  {s.radius:g} R☉\n'
                 f'{system.time:.3f} years   |   compressed radial scale; marker sizes exaggerated', fontsize=11)


class App:
    def __init__(self, star='sun', planets=('earth','jupiter')):
        self.star, self.planets = star, list(planets)
        self.distances = [PLANETS[p].distance for p in planets]
        self.fig = plt.figure(figsize=(13,8))
        self.ax = self.fig.add_axes((.31,.12,.68,.80))
        self.star_picker = RadioButtons(self.fig.add_axes((.025,.72,.21,.20)), list(STARS), active=list(STARS).index(star))
        self.planet_picker = RadioButtons(self.fig.add_axes((.025,.32,.15,.36)), list(PLANETS))
        self.add = Button(self.fig.add_axes((.18,.47,.10,.05)), 'Add planet')
        self.reset = Button(self.fig.add_axes((.18,.39,.10,.05)), 'Reset')
        self.speed = Slider(self.fig.add_axes((.41,.045,.45,.025)), 'years / sec', .001, 10., valinit=.5)
        self.status = self.fig.text(.32,.94,'', color='#aa3333', fontsize=10)
        self.sliders = []
        self.star_picker.on_clicked(self.change_star)
        self.add.on_clicked(self.add_planet)
        self.reset.on_clicked(lambda event: self.restart())
        self.rebuild_sliders()
        self.restart()
        self.last = time.monotonic()
        self.animation = FuncAnimation(self.fig, self.update, interval=33, cache_frame_data=False)

    def rebuild_sliders(self):
        for slider in self.sliders:
            slider.ax.remove()
        self.sliders = []
        minimum = max(.005, STARS[self.star].radius*SOLAR_RADIUS_AU*1.02)
        for i,(p,a) in enumerate(zip(self.planets,self.distances)):
            a = max(minimum,a)
            self.distances[i] = a
            slider = Slider(self.fig.add_axes((.09,.27-i*.022,.17,.012)), PLANETS[p].name,
                            np.log10(minimum), np.log10(80), valinit=np.log10(a))
            slider.label.set_fontsize(8); slider.valtext.set_fontsize(8)
            slider.on_changed(lambda value, index=i: self.distance_changed(index,value))
            self.sliders.append(slider)
            slider.valtext.set_text(f'{a:.3g} AU')

    def change_star(self, key):
        self.star = key
        self.rebuild_sliders()
        self.restart()

    def add_planet(self, event):
        p = self.planet_picker.value_selected
        if p not in self.planets:
            self.planets.append(p); self.distances.append(PLANETS[p].distance)
            self.rebuild_sliders(); self.restart()

    def distance_changed(self, index, value):
        self.distances[index] = 10**value
        self.sliders[index].valtext.set_text(f'{10**value:.3g} AU')
        self.restart()

    def restart(self):
        self.system = make_system(self.star,self.planets,self.distances)
        self.trails = []
        self.pending = 0.
        self.halted = False
        self.status.set_text('')
        self.last = time.monotonic()
        draw(self.ax,self.system,self.star,self.planets)
        self.fig.canvas.draw_idle()

    def update(self, frame):
        now = time.monotonic()
        self.pending += min(now-self.last,.1)*self.speed.val
        self.last = now
        if not self.halted:
            steps = min(150, int(self.pending/self.system.dt))
            try:
                for _ in range(steps):
                    self.system.step()
                    self.trails.append((self.system.r-self.system.r[0]).copy())
                self.pending -= steps*self.system.dt
                self.trails = self.trails[-1500:]
                self.status.set_text('Compute limit: simulation running slower than requested' if self.pending > self.system.dt*300 else '')
                self.pending = min(self.pending,self.system.dt*600)
            except ValueError as exc:
                self.halted = True
                self.status.set_text(str(exc))
        draw(self.ax,self.system,self.star,self.planets,self.trails)
        return ()
