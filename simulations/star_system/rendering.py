"""Retained overhead rendering; lighting and sizes are illustrative, not photometry."""
from functools import lru_cache

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle
from matplotlib.transforms import Bbox

from .core import SOLAR_RADIUS_AU
from .data import STARS, PLANETS, blackbody_rgb, star_size

BACKGROUND = '#070d18'
PANEL = '#101b2b'
TEXT = '#e3edf7'
MUTED = '#8298b0'
ACCENT = '#79d7e8'
MAX_TRAIL_VERTICES = 400


class CanvasPainter:
    """Cache static pixels and repaint moving artists on blit-capable backends."""

    def __init__(self, canvas, overlay_axes=()):
        self.canvas = canvas
        self.artists = []
        self.background = None
        self.text_bounds = {}
        self.use_blit = canvas.supports_blit
        self.overlay_axes = overlay_axes
        canvas.mpl_connect('draw_event', self._on_draw)
        canvas.mpl_connect('resize_event', self.invalidate)

    @property
    def overlay_axes(self):
        return self._overlay_axes

    @overlay_axes.setter
    def overlay_axes(self, axes):
        for ax in getattr(self, '_overlay_axes', ()):
            ax.set_animated(False)
        self._overlay_axes = list(axes)
        for ax in self._overlay_axes:
            ax.set_animated(self.use_blit)
        self.invalidate()

    def invalidate(self, event=None):
        self.background = None

    def set_artists(self, artists):
        for artist in self.artists:
            artist.set_animated(False)
        self.artists = sorted(artists, key=lambda artist: artist.get_zorder())
        self.text_bounds.clear()
        for artist in self.artists:
            artist.set_animated(self.use_blit)
        self.overlay_axes = self.overlay_axes
        self.invalidate()

    def _on_draw(self, event):
        if not self.use_blit:
            return
        if self.canvas.is_saving():
            # Axes include animated artists in exports, but Figure excludes its
            # animated text. Add that text to the export, without caching its DPI.
            for artist in self.artists:
                if artist.axes is None:
                    artist.draw(event.renderer)
            self._draw_overlay(event.renderer)
            self.invalidate()
            return
        self.background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._draw_artists()
        self._blit_regions()

    def _draw_artists(self):
        for artist in self.artists:
            if artist.get_visible():
                self.canvas.figure.draw_artist(artist)
        self._draw_overlay(self.canvas.get_renderer())

    def _draw_overlay(self, renderer):
        # Keep overlays out of the cached background. Drawing their translucent
        # edges twice changes pixels, and moving scene artists must stay below.
        for ax in sorted(self.overlay_axes, key=lambda ax: ax.get_zorder()):
            if ax.get_visible():
                ax.draw(renderer)

    def _blit_regions(self):
        figure = self.canvas.figure
        renderer = self.canvas.get_renderer()
        axes = {artist.axes for artist in self.artists if artist.axes is not None}
        # Include marker/label overhang and the overlay repainted above it.
        regions = [ax.bbox.padded(.8*figure.dpi) for ax in axes]
        for artist in self.artists:
            if artist.axes is None:
                current = artist.get_window_extent(renderer).padded(3)
                previous = self.text_bounds.get(artist, current)
                regions.append(Bbox.union([previous, current]))
                self.text_bounds[artist] = current
        return [clipped for region in regions
                if (clipped := Bbox.intersection(region, figure.bbox)) is not None]

    def paint(self):
        if not self.use_blit or self.background is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self.background)
        self._draw_artists()
        for region in self._blit_regions():
            self.canvas.blit(region)


def coordinates(r):
    """Star-relative radial asinh mapping; direction is preserved."""
    r = np.asarray(r)
    radius = np.linalg.norm(r, axis=-1)
    return r * (np.arcsinh(radius / .05) / np.maximum(radius, 1e-30))[..., None]


@lru_cache(maxsize=1)
def _sphere():
    axis = np.linspace(-1., 1., 80)
    x, y = np.meshgrid(axis, axis)
    radius2 = x*x + y*y
    z = np.sqrt(np.maximum(0., 1. - radius2))
    alpha = np.clip((1. - np.sqrt(radius2)) * 40., 0., 1.)
    return x, y, z, alpha


@lru_cache(maxsize=16)
def _surface(planet):
    """Subtle stylized markings, not maps of observed surfaces."""
    x, y, z, _ = _sphere()
    color = np.broadcast_to(to_rgb(PLANETS[planet].color), (*x.shape, 3)).copy()
    if planet in ('jupiter', 'saturn'):
        bands = .86 + .12 * np.sin(28*y + .7*np.sin(7*x))
        color *= bands[..., None]
    elif planet == 'earth':
        land = np.sin(8*x + 3*y) + .65*np.cos(11*y - 4*x) > .8
        color[land] = (.22, .48, .35)
        clouds = np.maximum(0., np.sin(16*y + 5*x + 2*np.sin(7*x)) - .65) * .65
        color = color*(1-clouds[..., None]) + clouds[..., None]
    else:
        color *= (.93 + .04*np.sin(19*x + 7*y)*np.cos(23*y - 4*x))[..., None]
    return color


def planet_sprite(planet, toward_star):
    """RGBA sphere with the bright hemisphere facing the star in the orbital plane."""
    x, y, z, alpha = _sphere()
    direction = np.asarray(toward_star, dtype=float)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm else np.array([1., 0.])
    incidence = np.maximum(0., x*direction[0] + y*direction[1])
    # A little ambient light keeps the night hemisphere legible against space.
    illumination = .10 + .90*incidence**.7
    limb = .55 + .45*z**.35
    rgb = _surface(planet) * (illumination*limb)[..., None]
    rgb += .06*incidence[..., None]**5
    return np.dstack((np.clip(rgb, 0., 1.), alpha))


@lru_cache(maxsize=8)
def star_sprite(star):
    """Limb-darkened luminous disk surrounded by a translucent halo."""
    axis = np.linspace(-1., 1., 160)
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x, y)
    color = np.asarray(blackbody_rgb(STARS[star].temperature))
    rgba = np.zeros((*x.shape, 4))
    rgba[..., :3] = color
    rgba[..., 3] = .38*np.exp(-4.5*radius) * np.clip((1-radius)*8, 0, 1)
    disk_alpha = np.clip((.34-radius)*80 + .5, 0., 1.)
    surface = np.sqrt(np.maximum(0., 1 - (radius/.34)**2))
    disk = np.clip(color*(.68 + .32*surface[..., None]) + .15*surface[..., None], 0, 1)
    rgba[..., :3] = color*(1-disk_alpha[..., None]) + disk*disk_alpha[..., None]
    rgba[..., 3] = disk_alpha + rgba[..., 3]*(1-disk_alpha)
    return rgba


def planet_size(planet):
    return 4 + 3*np.log10(1 + PLANETS[planet].radius/1000)


class Scene:
    """Create artists once and update their data without resetting the camera."""

    def __init__(self, ax, system, star, planets, trails=None):
        self.ax, self.star, self.planets = ax, star, list(planets)
        self.selected = None
        ax.clear()
        ax.set_facecolor(BACKGROUND)
        ax.set_aspect('auto')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self._background()
        self._guides()
        self.trails = []
        for planet in planets:
            line = LineCollection([], linewidths=1.05, zorder=2)
            ax.add_collection(line)
            self.trails.append(line)
        sun = STARS[star]
        image = OffsetImage(star_sprite(star), zoom=star_size(sun)/(160*.34), origin='lower')
        self.star_artist = AnnotationBbox(image, (0, 0), frameon=False, pad=0, zorder=5)
        ax.add_artist(self.star_artist)
        self.bodies, self.labels = [], []
        for planet in planets:
            image = OffsetImage(planet_sprite(planet, (1., 0.)), zoom=planet_size(planet)/80, origin='lower')
            body = AnnotationBbox(image, (0, 0), frameon=False, pad=0, zorder=6)
            ax.add_artist(body)
            self.bodies.append(body)
            label = ax.annotate(PLANETS[planet].name, (0, 0), xytext=(16, 10),
                                textcoords='offset points', fontsize=9, color=MUTED,
                                annotation_clip=True, zorder=8)
            self.labels.append(label)
        self.highlight = ax.scatter([], [], s=900, facecolors='none', edgecolors=ACCENT,
                                    linewidths=.8, alpha=.65, zorder=4)
        self.particles = ax.scatter([], [], s=4, color='#d8dfb3', alpha=.8,
                                    linewidths=0, zorder=3)
        self.heading = ax.text(.025, .97, sun.name, transform=ax.transAxes,
                               color=TEXT, fontsize=15, va='top', zorder=10)
        ax.text(.025, .925, f'{sun.temperature:,.0f} K   /   {sun.mass:g} solar masses',
                transform=ax.transAxes, color=MUTED, fontsize=8, va='top', zorder=10)
        self.clock = ax.text(.975, .965, '', transform=ax.transAxes, ha='right', va='top',
                             color=TEXT, fontsize=11, fontfamily='monospace', zorder=10)
        ax.text(.025, .025, 'COMPRESSED DISTANCES  /  BODY SIZES EXAGGERATED',
                transform=ax.transAxes, color=MUTED, fontsize=7, zorder=10)
        self.update(system, trails)
        self.fit_view()

    @property
    def moving_artists(self):
        # Guide text can overlap trails. Paint it in the same z-order as a full
        # axes draw, rather than baking it below every moving artist.
        return [*self.trails, *self.ax.texts, self.particles, self.highlight,
                self.star_artist, *self.bodies]

    def _background(self):
        rng = np.random.default_rng(24)
        points = rng.uniform(.01, .99, (155, 2))
        colors = np.tile(to_rgb('#a8bed4'), (len(points), 1))
        rgba = np.column_stack((colors, rng.uniform(.12, .50, len(points))))
        self.ax.scatter(*points.T, s=rng.uniform(.25, 2., len(points)), c=rgba,
                        transform=self.ax.transAxes, linewidths=0, zorder=0)

    def _guides(self):
        for radius in (.05, .1, .5, 1, 5, 10, 30, 80):
            if radius <= STARS[self.star].radius*SOLAR_RADIUS_AU:
                continue
            rr = np.arcsinh(radius/.05)
            self.ax.add_patch(Circle((0, 0), rr, fill=False, edgecolor='#223247',
                                     linewidth=.6, linestyle=(0, (2, 5)), zorder=1))
            self.ax.text(rr*.707, -rr*.707, f'{radius:g} AU', fontsize=7,
                         color='#6b839e', ha='center', va='center', clip_on=True,
                         bbox=dict(facecolor=BACKGROUND, edgecolor='none', pad=2), zorder=3)

    def update(self, system, trails=None):
        relative = system.r - system.r[0]
        self.positions = coordinates(relative)
        self.particles.set_offsets(self.positions[1+len(self.planets):])
        self.clock.set_text(f'{system.time:,.3f} yr')
        for i, (planet, body, label) in enumerate(zip(self.planets, self.bodies, self.labels), 1):
            body.xy = self.positions[i]
            body.xybox = self.positions[i]
            body.offsetbox.set_data(planet_sprite(planet, -relative[i]))
            label.xy = self.positions[i]
        history = np.asarray(trails) if trails is not None and len(trails) else None
        for i, (planet, line) in enumerate(zip(self.planets, self.trails), 1):
            if history is None or len(history) < 2:
                line.set_segments([])
                continue
            # Keep the full integration history; only bound the number of
            # vertices sent to the display. Always retain both endpoints.
            indices = np.linspace(0, len(history)-1,
                                  min(len(history), MAX_TRAIL_VERTICES), dtype=int)
            trail = coordinates(history[indices, i, :])
            segments = np.stack((trail[:-1], trail[1:]), axis=1)
            colors = np.tile((*to_rgb(PLANETS[planet].color), 1.), (len(segments), 1))
            colors[:, 3] = np.linspace(.015, .65, len(segments))
            line.set_segments(segments)
            line.set_color(colors)
        self.select(self.selected)

    def select(self, index):
        self.selected = index
        self.highlight.set_offsets(np.empty((0, 2)) if index is None else self.positions[[index]])
        if index is not None:
            diameter = (star_size(STARS[self.star]) if index == 0 else
                        planet_size(self.planets[index-1]) if index <= len(self.planets) else 3)
            self.highlight.set_sizes([(diameter+12)**2])
        for i, label in enumerate(self.labels, 1):
            label.set_color(TEXT if i == index else MUTED)

    def fit_view(self):
        extent = max(2., np.max(np.linalg.norm(self.positions, axis=1))*1.25)
        ratio = self.ax.bbox.height / max(self.ax.bbox.width, 1)
        half_width = extent / min(1., ratio)
        self.ax.set_xlim(-half_width, half_width)
        self.ax.set_ylim(-half_width*ratio, half_width*ratio)


def draw(ax, system, star, planets, trails=None):
    """Shared still/live appearance. Return the scene for subsequent frame updates."""
    ax.figure.set_facecolor(BACKGROUND)
    return Scene(ax, system, star, planets, trails)
