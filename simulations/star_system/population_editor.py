"""Population drafts rebuild the selected initial setup only on Apply/Remove."""
from dataclasses import replace
from textwrap import fill
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

from .authoring import reference_ids
from .core import G, kepler_period
from .initialization import build_system
from .model import PopulationSpec
from .populations import Population
from .rendering import PANEL, TEXT, MUTED, ACCENT


class PopulationEditor:
    """Separate-window editor with the shared reference-ID/cycle interaction."""

    def __init__(self, app):
        self.app, self.was_paused, self.finished = app, app.paused, False
        applied = app.initial_setup.population
        settings = applied.settings if applied is not None else Population()
        reference = applied.reference_id if applied is not None else (
            app.selected_id if app.system.masses[app.selected_index] > 0 else '')
        self.fig = plt.figure(figsize=(7.4, 8.5), facecolor=PANEL)
        self.fig.canvas.manager.set_window_title('Population settings')
        self.fig.text(.08, .95, 'Population settings / ' + ('preparation' if app.preparing else 'restart'),
                      fontsize=16, color=TEXT)
        self.reference_label = self.fig.text(.08, .91, '', fontsize=10, color=MUTED, va='top')
        self.fields, self.widgets = {}, []

        def field(key, label, value, y):
            widget = TextBox(self.fig.add_axes((.47, y, .42, .045)), label+'  ',
                             initial=str(value), color='#1b3044', hovercolor='#294b60')
            widget.label.set_color(TEXT)
            widget.text_disp.set_color(TEXT)
            self.fields[key] = widget
            self.widgets.append(widget)
            return widget

        field('reference', 'Reference ID', reference, .80)
        self.reference_next = self._button((.08, .746, .84, .04), 'Reference: next massive', self.cycle_reference)
        self.kind = RadioButtons(self.fig.add_axes((.08, .58, .35, .14)),
                                 ['ring', 'disk', 'belt'], active=['ring', 'disk', 'belt'].index(settings.kind),
                                 activecolor=ACCENT, useblit=False)
        self.angles = RadioButtons(self.fig.add_axes((.57, .58, .35, .14)),
                                   ['random', 'even'], active=['random', 'even'].index(settings.angles),
                                   activecolor=ACCENT, useblit=False)
        for picker in (self.kind, self.angles):
            picker.ax.set_facecolor(PANEL)
            for spine in picker.ax.spines.values(): spine.set_visible(False)
            picker.set_radio_props({'edgecolor': MUTED, 'facecolor': ACCENT})
            for label in picker.labels: label.set_color(TEXT)
            self.widgets.append(picker)
        for key, label, value, y in (
            ('count', 'Particles (1-2000 total)', settings.count, .52),
            ('inner', 'Inner radius (AU)', settings.inner, .45),
            ('outer', 'Outer radius (AU)', settings.outer, .38),
            ('seed', 'Seed', settings.seed, .31),
        ):
            field(key, label, value, y)
        self.timestep_label = self.fig.text(.08, .275, '', color=MUTED, fontsize=9, va='top')
        self.error = self.fig.text(.08, .205, '', color='#efb17b', fontsize=9, va='top')
        self.apply = self._button((.08, .095, .40, .05),
                                 'Apply' if app.preparing else 'Apply and restart', self.apply_settings)
        self.remove = self._button((.52, .095, .40, .05),
                                  'Remove' if app.preparing else 'Remove and restart', self.remove_settings)
        self.remove.set_active(applied is not None)
        self.remove.label.set_color(TEXT if applied is not None else MUTED)
        self.cancel = self._button((.62, .02, .30, .05), 'Cancel', lambda event: self.finish())
        self.kind.on_clicked(self.kind_changed)
        for key in ('reference', 'inner'):
            self.fields[key].on_text_change(lambda text: self.describe_reference())
        self.kind_changed(settings.kind)
        self.connections = [self.fig.canvas.mpl_connect('close_event', lambda event: self.finish()),
                            self.fig.canvas.mpl_connect('key_press_event', self.on_key)]

    def _button(self, bounds, label, callback):
        widget = Button(self.fig.add_axes(bounds), label, color='#1b3044', hovercolor='#294b60', useblit=False)
        widget.label.set_color(TEXT)
        widget.label.set_fontsize(9)
        for spine in widget.ax.spines.values(): spine.set_edgecolor('#294055')
        widget.on_clicked(callback)
        self.widgets.append(widget)
        return widget

    def on_key(self, event):
        if event.key == 'escape': self.finish()

    def cycle_reference(self, event=None):
        keys = reference_ids((body.id, body.spec) for body in self.app.initial_setup.bodies)
        current = self.fields['reference'].text.strip()
        if keys:
            self.fields['reference'].set_val(keys[(keys.index(current)+1)%len(keys)] if current in keys else keys[0])

    def reference_body(self):
        key = self.fields['reference'].text.strip()
        body = next((body for body in self.app.initial_setup.bodies if body.id == key), None)
        if body is None:
            raise ValueError('Choose an existing massive reference ID from the selected starting setup.')
        if body.spec.mass is None or body.spec.mass <= 0:
            raise ValueError(f'Population reference {key!r} must have a known positive mass.')
        if body.spec.radius is None:
            raise ValueError(f'Population reference {key!r} needs a known physical radius (AU).')
        return body

    def fixed_dt(self, inner, reference):
        # The edited setup records the disclosed fixed step; build_system never
        # silently overrides an explicit setup dt with the generator's choice.
        try:
            return min(self.app.initial_setup.dt, kepler_period(inner, G*reference.spec.mass)/600)
        except OverflowError:
            raise ValueError('Reference orbital period exceeds the numeric range; choose smaller radii.') from None

    def describe_reference(self):
        if self.finished: return
        try:
            body = self.reference_body()
            self.reference_label.set_text(fill(f'Reference: {body.spec.name} [{body.id}]', width=78)
                                          + '\nInitial reference frame / radii in AU / circular prograde velocities')
            inner = float(self.fields['inner'].text)
            if not 0 < inner < float('inf'): raise ValueError('Enter a finite positive inner radius.')
            dt = self.fixed_dt(inner, body)
            self.timestep_label.set_text(f'Fixed dt on Apply: {dt:.6g} yr\n'
                                         'Uses min(starting dt, reference inner orbital period / 600).')
        except ValueError as exc:
            self.reference_label.set_text(fill(str(exc), width=78))
            self.timestep_label.set_text('Fixed dt: choose a valid reference and inner radius.')
        self.fig.canvas.draw_idle()

    def kind_changed(self, kind):
        # Rings have one radius; retain the outer draft for disk/belt switches.
        enabled = kind != 'ring'
        self.fields['inner'].label.set_text('Inner radius (AU)  ' if enabled else 'Radius (AU)  ')
        self.fields['outer'].set_active(enabled)
        self.fields['outer'].text_disp.set_color(TEXT if enabled else MUTED)
        self.describe_reference()

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
            reference = self.reference_body()
            previous = self.app.initial_setup.population
            population = PopulationSpec(previous.id if previous is not None else 'population', reference.id, settings)
            setup = replace(self.app.initial_setup, population=population, dt=self.fixed_dt(inner, reference))
            prepared = build_system(setup)
        except ValueError as exc:
            self.error.set_text(fill(str(exc), width=79))
            self.fig.canvas.draw_idle()
            return
        self.finish(setup, prepared)

    def remove_settings(self, event=None):
        try:
            setup = replace(self.app.initial_setup, population=None)
            prepared = build_system(setup)
        except ValueError as exc:
            self.error.set_text(fill(str(exc), width=79))
            self.fig.canvas.draw_idle()
            return
        self.finish(setup, prepared)

    def finish(self, setup=None, prepared=None, closing=False):
        if self.finished: return
        self.finished = True
        self.app.population_editor = None
        for widget in self.widgets: widget.disconnect_events()
        for connection in self.connections: self.fig.canvas.mpl_disconnect(connection)
        if prepared is not None:
            self.app.initial_setup = setup
            self.app._legacy_configuration = self.app._legacy_configuration and (
                setup.population is None or setup.population.reference_id == self.app.star)
            self.app.paused, self.app.following, self.app.selected_index = True, False, 0
            self.app._install_prepared(prepared, reset=True)
            self.app.status.set_text(('PREPARATION: starting setup updated.' if self.app.preparing else 'Starting setup restarted.')
                                     + f' Fixed dt = {setup.dt:.6g} yr.')
        else:
            self.app.paused = self.was_paused
        self.app.pending, self.app.last = 0., time.monotonic()
        if not closing:
            self.app._set_population_editing(False)
            self.app._refresh()
        plt.close(self.fig)
