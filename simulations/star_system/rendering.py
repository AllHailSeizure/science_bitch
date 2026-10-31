"""Retained overhead rendering; lighting and sizes are illustrative, not photometry."""
from functools import lru_cache

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle
from matplotlib.transforms import Bbox

from .core import SOLAR_RADIUS_AU
from .data import STARS, PLANETS, CATALOG, AU_KM, blackbody_rgb, star_size
from .model import BodySpec
from .initialization import PreparedSystem
from .history import StateFrame
from .frames import frame_transform, reference_id, lagrange_guides, LABELS

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
    """View-origin-relative radial asinh mapping; direction is preserved."""
    r = np.asarray(r)
    radius = np.linalg.norm(r, axis=-1)
    return r * (np.arcsinh(radius / .05) / np.maximum(radius, 1e-30))[..., None]


def inverse_coordinates(q):
    """Physical offset in AU from compressed display coordinates."""
    q = np.asarray(q, dtype=float)
    radius = np.linalg.norm(q, axis=-1)
    return q * (.05*np.sinh(radius)/np.maximum(radius, 1e-30))[..., None]


def dominant_star(prepared, index):
    """Illustrative bolometric flux; missing radius/temperature uses solar luminosity."""
    candidates = [i for i, spec in enumerate(prepared.body_specs) if spec.kind == 'star' and i != index]
    if not candidates:
        return None
    def flux(i):
        spec = prepared.body_specs[i]
        luminosity = ((spec.radius/SOLAR_RADIUS_AU)**2*(spec.temperature/5772)**4
                      if spec.radius is not None and spec.temperature is not None else 1.)
        distance2 = np.sum((prepared.system.r[i]-prepared.system.r[index])**2)
        return luminosity/max(distance2, 1e-30)
    return max(candidates, key=flux)


def body_size(spec):
    if spec.kind == 'star':
        return 12 + 16*np.log10(1 + 8*(spec.radius or SOLAR_RADIUS_AU)/SOLAR_RADIUS_AU)
    return 4 + 3*np.log10(1 + (spec.radius or 1000/AU_KM)*AU_KM/1000)


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
    spec = planet if isinstance(planet, BodySpec) else CATALOG[planet]
    color = np.broadcast_to(to_rgb(spec.color or '#a9b8cb'), (*x.shape, 3)).copy()
    planet = spec.catalog_key
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
    spec = star if isinstance(star, BodySpec) else CATALOG[star]
    color = np.asarray(blackbody_rgb(spec.temperature) if spec.temperature is not None else to_rgb(spec.color or '#fff0d9'))
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
    """Stable-ID artists and world-frame history; updates retain the camera."""

    def __init__(self, ax, prepared, history=(), legacy=False, frame_mode='sun-relative'):
        self.ax, self.prepared, self.legacy = ax, prepared, legacy
        self.frame_mode = frame_mode
        self.selected = None
        self.origin_id = next((key for key, spec in zip(prepared.body_ids, prepared.body_specs)
                               if spec.kind == 'star'), prepared.body_ids[0])
        self.artists_by_id, self.labels_by_id, self.trails_by_id = {}, {}, {}
        ax.clear()
        ax.set_facecolor(BACKGROUND)
        ax.set_aspect('auto')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self._background()
        self._guides()
        self.highlight = ax.scatter([], [], s=900, facecolors='none', edgecolors=ACCENT,
                                    linewidths=.8, alpha=.65, zorder=4)
        self.particles = ax.scatter([], [], s=4, color='#d8dfb3', alpha=.8, linewidths=0, zorder=3)
        self.selected_trail = LineCollection([], linewidths=1.25, zorder=2.5)
        ax.add_collection(self.selected_trail)
        self.selected_trail_times = ()
        self.selected_trail_vertices = 0
        self.reference_guides = ax.scatter([], [], s=90, marker='D', facecolors='none',
                                           edgecolors='#c6a9ed', linewidths=1.1, zorder=4)
        self.guide_labels = [ax.annotate(label, (0, 0), xytext=(8, 6), textcoords='offset points',
                                        color='#c6a9ed', fontsize=9, zorder=7) for label in ('L4 guide', 'L5 guide')]
        self.guide_note = ax.text(.025, .065, '', transform=ax.transAxes, color='#c6a9ed', fontsize=8, zorder=10)
        self.guide_note.set_x(.335)
        self.trail_info = ax.text(.335, .12, '', transform=ax.transAxes, color=ACCENT, fontsize=8, zorder=10)
        self.heading = ax.text(.40, .97, '', transform=ax.transAxes, color=TEXT, fontsize=12, va='top', zorder=10)
        self.clock = ax.text(.975, .965, '', transform=ax.transAxes, ha='right', va='top', color=TEXT,
                             fontsize=11, fontfamily='monospace', zorder=10)
        self.result_info = ax.text(.975, .915, '', transform=ax.transAxes, ha='right', va='top', color=TEXT,
                                   fontsize=9, zorder=10,
                                   bbox=dict(facecolor=BACKGROUND, alpha=.9, edgecolor='none', pad=5))
        ax.text(.025, .025, 'COMPRESSED DISTANCES / BODY SIZES EXAGGERATED / ILLUSTRATIVE LIGHTING',
                transform=ax.transAxes, color=MUTED, fontsize=7, zorder=10)
        self.update(prepared, history)
        self.fit_view()

    def _ensure_artists(self):
        for key, artist in self.artists_by_id.items():
            present = (key in self.prepared.body_ids and
                       self.prepared.body_specs[self.prepared.body_ids.index(key)].kind != 'particle')
            artist.set_visible(present)
            self.labels_by_id[key].set_visible(present)
            if key in self.trails_by_id:
                self.trails_by_id[key].set_visible(present)
        for key, spec in zip(self.prepared.body_ids, self.prepared.body_specs):
            if spec.kind == 'particle' or key in self.artists_by_id:
                continue
            stellar = spec.kind == 'star'
            image = OffsetImage(star_sprite(spec) if stellar else planet_sprite(spec, (1, 0)),
                                zoom=body_size(spec)/(160*.34 if stellar else 80), origin='lower')
            body = AnnotationBbox(image, (0, 0), frameon=False, pad=0, zorder=5 if stellar else 6)
            self.ax.add_artist(body)
            self.artists_by_id[key] = body
            self.labels_by_id[key] = self.ax.annotate(spec.name, (0, 0), xytext=(16, 10), textcoords='offset points',
                                                    fontsize=9, color=MUTED, annotation_clip=True, zorder=8)
            if key != self.origin_id:
                line = LineCollection([], linewidths=1.05, zorder=2)
                self.ax.add_collection(line)
                self.trails_by_id[key] = line
        stars = [self.artists_by_id[k] for k, spec in zip(self.prepared.body_ids, self.prepared.body_specs) if spec.kind == 'star']
        self.star_artist = stars[0] if stars else None  # legacy adapter
        self.bodies = [self.artists_by_id[k] for k, spec in zip(self.prepared.body_ids, self.prepared.body_specs)
                       if spec.kind not in ('star', 'particle')]
        self.labels = list(self.labels_by_id.values())
        self.trails = list(self.trails_by_id.values())

    @property
    def moving_artists(self):
        return [*self.trails, self.selected_trail, *self.ax.texts, self.reference_guides,
                self.particles, self.highlight, *self.artists_by_id.values()]

    def set_frame_mode(self, mode):
        # Validate before changing display state; no physical array is written.
        frame_transform(self.frame, mode, self.origin_id)
        self.frame_mode = mode
        self.update(self.prepared, self.history)

    def world_to_display(self, world):
        return coordinates(self.transform.positions(world))

    def display_to_world(self, displayed):
        return self.transform.inverse_positions(inverse_coordinates(displayed))

    def _background(self):
        rng = np.random.default_rng(24)
        points = rng.uniform(.01, .99, (155, 2))
        colors = np.tile(to_rgb('#a8bed4'), (len(points), 1))
        rgba = np.column_stack((colors, rng.uniform(.12, .50, len(points))))
        self.ax.scatter(*points.T, s=rng.uniform(.25, 2., len(points)), c=rgba,
                        transform=self.ax.transAxes, linewidths=0, zorder=0)

    def _guides(self):
        for radius in (.05, .1, .5, 1, 5, 10, 30, 80):
            rr = np.arcsinh(radius/.05)
            self.ax.add_patch(Circle((0, 0), rr, fill=False, edgecolor='#223247',
                                     linewidth=.6, linestyle=(0, (2, 5)), zorder=1))
            self.ax.text(rr*.707, -rr*.707, f'{radius:g} AU', fontsize=7,
                         color='#6b839e', ha='center', va='center', clip_on=True,
                         bbox=dict(facecolor=BACKGROUND, edgecolor='none', pad=2), zorder=3)

    def update(self, prepared, history=()):
        if not isinstance(prepared, PreparedSystem):
            from dataclasses import replace
            prepared = replace(self.prepared, system=prepared)
        self.prepared = prepared
        self.history = tuple(history) if history is not None else ()
        if self.origin_id not in prepared.body_ids:
            self.origin_id = next((key for key, spec in zip(prepared.body_ids, prepared.body_specs)
                                   if spec.kind == 'star'), prepared.body_ids[0])
        self._ensure_artists()
        system = prepared.system
        self.frame = StateFrame.capture(prepared)
        origin_index = prepared.body_ids.index(self.origin_id)
        self.origin = system.r[origin_index].copy()
        self.rotating_reason = ''
        try:
            frame_transform(self.frame, 'rotating')
        except ValueError as exc:
            self.rotating_reason = str(exc)
            if self.frame_mode == 'rotating':
                self.frame_mode = 'sun-relative'
        self.transform = frame_transform(self.frame, self.frame_mode, self.origin_id)
        self.positions = self.world_to_display(system.r)
        particle_indices = [i for i, s in enumerate(prepared.body_specs) if s.kind == 'particle']
        self.particles.set_offsets(self.positions[particle_indices])
        self.clock.set_text(f'{system.time:,.8g} yr')
        label = LABELS[self.frame_mode]
        if self.frame_mode == 'sun-relative':
            label = prepared.body_specs[origin_index].name+'-relative / inertial axes'
        self.heading.set_text(label)
        self._update_reference_guides()
        for i, (key, spec) in enumerate(zip(prepared.body_ids, prepared.body_specs)):
            if spec.kind == 'particle' or key not in self.artists_by_id:
                continue
            body = self.artists_by_id[key]
            body.xy = body.xybox = self.positions[i]
            body.offsetbox.set_zoom(body_size(spec)/(160*.34 if spec.kind == 'star' else 80))
            self.labels_by_id[key].set_text(spec.name)
            light = dominant_star(prepared, i)
            if spec.kind == 'star':
                body.offsetbox.set_data(star_sprite(spec))
            else:
                direction = system.r[light]-system.r[i] if light is not None else (1, 0)
                body.offsetbox.set_data(planet_sprite(spec, self.transform.vectors(direction)))
            self.labels_by_id[key].xy = self.positions[i]
        for key, line in self.trails_by_id.items():
            if key not in prepared.body_ids:
                line.set_segments([])
                continue
            spec = prepared.body_specs[prepared.body_ids.index(key)]
            self._set_trail(line, key, spec.color or '#b7c9df')
        self.select(self.selected)

    def _update_reference_guides(self):
        available = not self.rotating_reason
        self.reference_guides.set_visible(available)
        for label in self.guide_labels:
            label.set_visible(available)
        self.guide_note.set_text('L4/L5: ideal circular restricted-three-body reference geometry\n'
                                 'Geometric guides; occupancy / stability / capture not detected.' if available else self.rotating_reason)
        if available:
            sun, jupiter = reference_id(self.frame, 'sun'), reference_id(self.frame, 'jupiter')
            points = self.world_to_display(lagrange_guides(self.frame, sun, jupiter))
            self.reference_guides.set_offsets(points)
            for label, point in zip(self.guide_labels, points):
                label.xy = point

    def _set_trail(self, line, key, color):
        points, times = [], []
        for frame in self.history:
            if hasattr(frame, 'body_ids'):
                try:
                    if key not in frame.body_ids:
                        raise ValueError('Missing body')
                    transform = frame_transform(frame, self.frame_mode, self.origin_id)
                    points.append(transform.positions(frame.r[frame.body_ids.index(key)]))
                    times.append(frame.time)
                except ValueError:
                    # Do not connect a trajectory across missing identities/references.
                    points, times = [], []
            else:  # legacy relative-array history
                points.append(frame[self.prepared.body_ids.index(key)])
        indices = np.linspace(0, len(points)-1, min(len(points), MAX_TRAIL_VERTICES), dtype=int)
        trail = coordinates(np.asarray(points)[indices]) if len(points) else np.empty((0, 2))
        segments = np.stack((trail[:-1], trail[1:]), axis=1) if len(trail) > 1 else []
        colors = np.tile((*to_rgb(color), 1.), (len(segments), 1))
        colors[:, 3] = np.linspace(.015, .8, len(segments))
        line.set_segments(segments)
        line.set_color(colors)
        return tuple(times), len(trail)

    def select(self, body_id):
        if isinstance(body_id, (int, np.integer)):
            body_id = self.prepared.body_ids[body_id]
        self.selected = body_id
        index = self.prepared.body_ids.index(body_id) if body_id in self.prepared.body_ids else None
        self.highlight.set_offsets(np.empty((0, 2)) if index is None else self.positions[[index]])
        if index is not None:
            self.highlight.set_sizes([(body_size(self.prepared.body_specs[index])+12)**2])
        for key, label in self.labels_by_id.items():
            label.set_color(TEXT if key == body_id else MUTED)
        particle = index is not None and self.prepared.body_specs[index].kind == 'particle'
        self.selected_trail.set_visible(particle)
        if particle:
            self.selected_trail_times, self.selected_trail_vertices = self._set_trail(self.selected_trail, body_id, ACCENT)
            times = self.selected_trail_times
            self.trail_info.set_text((f'Selected {body_id}: {times[0]:.6g}–{times[-1]:.6g} yr / '
                f'{times[-1]-times[0]:.6g} yr duration\n{len(times)} retained samples → '
                f'{self.selected_trail_vertices} display vertices (max {MAX_TRAIL_VERTICES})'
                + (' / earlier history truncated' if self.history and hasattr(self.history[0], 'time')
                   and self.history[0].time > 0 else '')) if times else 'Selected trail: no retained samples')
        else:
            self.selected_trail.set_segments([])
            self.selected_trail_times, self.selected_trail_vertices = (), 0
            self.trail_info.set_text('')

    def fit_view(self):
        extent = max(2., np.max(np.linalg.norm(self.positions, axis=1))*1.25)
        ratio = self.ax.bbox.height / max(self.ax.bbox.width, 1)
        half_width = extent / min(1., ratio)
        self.ax.set_xlim(-half_width, half_width)
        self.ax.set_ylim(-half_width*ratio, half_width*ratio)


def draw_system(ax, prepared, history=(), frame_mode='sun-relative'):
    ax.figure.set_facecolor(BACKGROUND)
    return Scene(ax, prepared, history, frame_mode=frame_mode)


def draw(ax, system, star, planets, trails=None):
    """Legacy numeric/circular caller adapter; generic rendering has no row prefix rule."""
    from .model import BodySpec
    ax.figure.set_facecolor(BACKGROUND)
    keys = (star, *planets)
    count = len(system.masses)-len(keys)
    particle = BodySpec('Test particle', 'particle', 0., None, color='#d8dfb3')
    prepared = PreparedSystem(system, keys+tuple(f'population:{i}' for i in range(count)),
                              tuple(CATALOG[k] for k in keys)+(particle,)*count)
    return Scene(ax, prepared, trails, legacy=True)
