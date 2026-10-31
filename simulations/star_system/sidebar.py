"""Overlay controls for the full-window observatory."""
from textwrap import fill

import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

from .data import STARS, PLANETS
from .rendering import PANEL, TEXT, MUTED, ACCENT, planet_sprite, star_sprite, dominant_star


def button(fig, bounds, label):
    widget = Button(fig.add_axes(bounds, zorder=12), label, color='#1b3044',
                    hovercolor='#294b60', useblit=False)
    widget.label.set_color(TEXT)
    widget.label.set_fontsize(9)
    return widget


class Sidebar:
    """Own overlay visibility, catalog, and a fixed-height scrollable body list."""

    def __init__(self, app):
        self.app, self.visible, self.catalog_open = app, True, False
        self.rows, self.selected_id, self.offset = [], None, 0
        fig = app.fig
        self.background = fig.add_axes((0, 0, .31, 1), zorder=10, facecolor=PANEL)
        self.background.set_xticks([])
        self.background.set_yticks([])
        for spine in self.background.spines.values():
            spine.set_visible(False)
        self.title = self.background.text(.06, .965, '', color=TEXT, fontsize=13, va='top', wrap=True)
        self.readout = self.background.text(.06, .91, '', color=MUTED, fontsize=9, va='top')
        self.inspector = self.background.text(.06, .33, '', color=TEXT, fontsize=9, va='top')
        self.status = self.background.text(.06, .055, '', color='#efb17b', fontsize=8, va='top', wrap=True)
        self.mode = self.background.text(.94, .91, '', color=ACCENT, fontsize=9, ha='right', va='top')
        self.body_ax = fig.add_axes((.02, .40, .27, .19), zorder=12, facecolor='#17283a')
        self.body_ax.set_xticks([])
        self.body_ax.set_yticks([])
        self.body_ax.set_xlim(0, 1)
        self.body_ax.set_ylim(0, 1)
        self.preview = fig.add_axes((.205, .20, .075, .12), zorder=12, facecolor=PANEL)
        self.preview.set_axis_off()
        self.preview_image = self.preview.imshow(star_sprite(app.star), origin='lower')
        self.prepare = button(fig, (.02,.825,.13,.03), 'Prepare setup')
        self.legacy = button(fig, (.16,.825,.13,.03), 'Prepare circular')
        self.edit = button(fig, (.02,.37,.13,.027), 'Edit selected')
        self.remove = button(fig, (.16,.37,.13,.027), 'Remove selected')
        self.add = button(fig, (.02, .77, .13, .037), 'Add a body')
        self.population = button(fig, (.16, .77, .13, .037), 'Population...')
        self.reset = button(fig, (.02, .72, .13, .037), 'Restart')
        self.pause = button(fig, (.16, .72, .13, .037), 'Pause')
        self.step = button(fig, (.02, .67, .085, .037), 'Step')
        self.home = button(fig, (.112, .67, .085, .037), 'Fit view')
        self.follow = button(fig, (.204, .67, .086, .037), 'Follow')
        self.experiment = button(fig, (.02, .135, .27, .035), 'Experiment / Save / Load...')
        self.experiment.on_clicked(app.edit_experiment)
        self.speed = Slider(fig.add_axes((.09, .625, .15, .012), zorder=12), 'yr/sec', .001, 10., valinit=.5, valfmt='%.3g', color=ACCENT, track_color='#22374b')
        self.speed.label.set_color(MUTED)
        self.speed.valtext.set_color(TEXT)
        self.speed.label.set_fontsize(8)
        self.speed.valtext.set_fontsize(8)
        self.menu = button(fig, (.315, .95, .07, .04), 'Menu [M]')
        self.star_picker = RadioButtons(fig.add_axes((.02, .36, .12, .24), zorder=14, facecolor=PANEL), list(STARS), active=list(STARS).index(app.star), activecolor=ACCENT, useblit=False)
        self.planet_picker = RadioButtons(fig.add_axes((.15, .36, .14, .24), zorder=14, facecolor=PANEL), list(PLANETS), activecolor=ACCENT, useblit=False)
        for picker in (self.star_picker, self.planet_picker):
            picker.set_radio_props({'edgecolor': MUTED})
            for label in picker.labels:
                label.set_color(TEXT)
                label.set_fontsize(8)
        self.catalog_add = button(fig, (.02, .31, .13, .035), 'Add planet')
        self.catalog_close = button(fig, (.16, .31, .13, .035), 'Close catalog')
        self.catalog_widgets = [self.star_picker, self.planet_picker, self.catalog_add, self.catalog_close]
        self.widgets = [self.prepare, self.legacy, self.edit, self.remove, self.add, self.population, self.reset, self.pause, self.step, self.home, self.follow, self.speed, self.experiment]
        self.axes = [self.background, self.body_ax, self.preview, *(w.ax for w in self.widgets), *(w.ax for w in self.catalog_widgets)]
        self.add.on_clicked(app.edit_body)
        self.prepare.on_clicked(app.prepare_setup)
        self.legacy.on_clicked(self.prepare_circular)
        self.edit.on_clicked(lambda event: app.edit_body(body_id=app.selected_id))
        self.remove.on_clicked(app.remove_body)
        self.catalog_close.on_clicked(lambda event: self.set_catalog(False))
        self.menu.on_clicked(app.toggle_sidebar)
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.set_catalog(False)

    def prepare_circular(self, event=None):
        self.app.prepare_setup()
        self.set_catalog(True)

    def set_catalog(self, visible):
        self.catalog_open = bool(visible)
        self._sync()

    def _sync(self):
        authoring = self.app.body_editor is not None
        editing = authoring or self.app.population_editor is not None or self.app.experiment_editor is not None
        self.background.set_visible(self.visible)
        for artist in (self.body_ax, self.preview, self.inspector):
            artist.set_visible(self.visible and not self.catalog_open and not authoring)
        for widget in self.widgets:
            widget.ax.set_visible(self.visible and not authoring)
            widget.active = self.visible and not editing
            if widget in (self.edit,self.remove):
                widget.active = widget.active and self.app.preparing
                widget.ax.set_visible(self.visible and not self.catalog_open and not authoring)
            if widget == self.legacy:
                widget.active = widget.active and self.app._legacy_configuration
        for widget in self.catalog_widgets:
            enabled = self.visible and self.catalog_open and not authoring
            widget.ax.set_visible(enabled)
            widget.active = enabled and not editing and self.app.preparing
        if authoring:
            for ax in self.app.body_editor.axes: ax.set_visible(self.visible)
            for widget in self.app.body_editor.widgets:
                widget.active = self.visible and (widget != self.app.body_editor.fields['dt'] or self.app.preparing)
        if hasattr(self.app, 'painter'):
            self.app.painter.invalidate()
            self.app.fig.canvas.draw_idle()

    def set_visible(self, visible):
        self.visible = bool(visible)
        self._sync()

    def contains_event(self, event):
        return (self.visible and event.x is not None and event.y is not None
                and 0 <= event.x < self.app.fig.bbox.width*.31
                and 0 <= event.y <= self.app.fig.bbox.height)

    def set_bodies(self, rows, selected_id):
        rows = list(rows)
        if rows == self.rows and selected_id == self.selected_id:
            return
        self.rows, self.selected_id = rows, selected_id
        self.offset = min(self.offset, max(0, len(self.rows)-6))
        self._draw_rows()

    def _draw_rows(self):
        self.body_ax.clear()
        self.body_ax.set_facecolor('#17283a')
        self.body_ax.set_xticks([])
        self.body_ax.set_yticks([])
        self.body_ax.set_xlim(0, 1)
        self.body_ax.set_ylim(0, 1)
        for i, row in enumerate(self.rows[self.offset:self.offset+6]):
            key, label = row
            self.body_ax.text(.04, .92-i*.15, str(label), va='top', fontsize=9,
                              clip_on=True, color=ACCENT if key == self.selected_id else TEXT)
        if len(self.rows) > 6:
            self.body_ax.text(.97, .03,
                              f'{self.offset+1}-{min(self.offset+6,len(self.rows))}/{len(self.rows)}',
                              ha='right', color=MUTED, fontsize=7)

    def on_scroll(self, event):
        if self.visible and not self.catalog_open and self.app.population_editor is None and self.app.body_editor is None and self.app.experiment_editor is None and event.inaxes == self.body_ax:
            self.offset = max(0, min(max(0, len(self.rows)-6), self.offset-int(event.step)))
            self._draw_rows()
            self.app.painter.invalidate()
            self.app.fig.canvas.draw_idle()

    def on_click(self, event):
        if self.visible and not self.catalog_open and self.app.population_editor is None and self.app.body_editor is None and self.app.experiment_editor is None and event.inaxes == self.body_ax and event.button == 1 and event.ydata is not None:
            row = int(np.floor((.92-event.ydata)/.15))
            index = self.offset + row
            if 0 <= row < 6 and 0 <= index < len(self.rows):
                key = self.rows[index][0]
                if isinstance(key, int): self.app.selected_index = key
                else: self.app.selected_id = key
                self.app._refresh()

    def update(self, time, speed, selected_index):
        self.readout.set_text(f'{time:,.3f} yr   /   {speed:.3g} yr/sec')
        self.status.set_text(fill(self.status.get_text(), width=42))
        spec = self.app.prepared.body_specs[selected_index]
        if spec.kind == 'star':
            image = star_sprite(spec)
        elif spec.kind != 'particle':
            light = dominant_star(self.app.prepared, selected_index)
            direction = self.app.system.r[light]-self.app.system.r[selected_index] if light is not None else (1,0)
            image = planet_sprite(spec, direction)
        else:
            image = None
        self.preview_image.set_visible(image is not None)
        if image is not None:
            self.preview_image.set_data(image)


class ExperimentEditor:
    """Names, preset selection and explicit paths using the existing backend."""

    def __init__(self, app):
        import matplotlib.pyplot as plt
        from .scenarios import COMPATIBILITY, DESCRIPTIONS
        self.app, self.was_paused, self.finished = app, app.paused, False
        self.fig = plt.figure(figsize=(8.5, 8.5), facecolor=PANEL)
        self.fig.canvas.manager.set_window_title('Experiment / preparation and saves')
        self.fig.text(.07, .95, 'Experiment / preparation and saves', color=TEXT, fontsize=16)
        origin = app.initial_setup.preset_id or app.initial_setup.provenance.get('origin_preset')
        self.origin = self.fig.text(.07, .905, ('Preset: ' if app.initial_setup.preset_id else 'Custom copy / origin: ')
                                   + str(origin or 'free preparation'), color=MUTED, fontsize=10)
        self.widgets, self.fields = [], {}
        for key, label, value, y in [('name', 'Experiment name', app.initial_setup.name, .81),
                                    ('path', 'JSON path', 'output/experiment.json', .64),
                                    ('capture_name', 'New setup name', app.initial_setup.name+' / captured', .39),
                                    ('capture_path', 'Capture JSON path', 'output/experiment-captured.json', .33)]:
            box = TextBox(self.fig.add_axes((.28, y, .64, .045)), label+'  ', initial=value,
                          color='#1b3044', hovercolor='#294b60')
            box.label.set_color(TEXT); box.text_disp.set_color(TEXT)
            self.widgets.append(box); self.fields[key] = box
        self._button((.07, .74, .39, .045), 'Apply name', self.rename)
        self._button((.50, .74, .42, .045), 'Prepare selected setup', self.prepare)
        self._button((.07, .565, .39, .045), 'Save initial setup', self.save_initial)
        self._button((.50, .565, .42, .045), 'Load into preparation', self.load)
        self.fig.text(.07, .525, 'Initial save restores the configured beginning, including its original seed.\n'
                      'Evolution and live insertions do not change that starting setup.', color=MUTED, fontsize=10, va='top')
        self._button((.07, .26, .85, .045), 'Save current state as new setup', self.save_current)
        self.fig.text(.07, .245, 'Capture writes a separate named epoch at time zero, including explicit particles.\n'
                      'The current experiment stays selected. Load the file to use the new epoch.', color=MUTED, fontsize=10, va='top')
        self._button((.07, .155, .54, .045), 'Open Sun / Earth / Jupiter preset', self.preset)
        self._button((.65, .155, .27, .045), 'Close', lambda event: self.finish())
        self.fig.text(.07, .135, fill(DESCRIPTIONS[COMPATIBILITY], 104), color=MUTED, fontsize=8, va='top')
        self.error = self.fig.text(.07, .065, '', color='#efb17b', fontsize=9, va='top')
        self.connections = [self.fig.canvas.mpl_connect('close_event', lambda event: self.finish()),
                            self.fig.canvas.mpl_connect('key_press_event', self.on_key)]

    def _button(self, bounds, label, callback):
        widget = button(self.fig, bounds, label)
        widget.on_clicked(callback)
        self.widgets.append(widget)
        return widget

    def _action(self, callback, message):
        try:
            callback()
        except (ValueError, OSError) as exc:
            self.error.set_text(fill(str(exc), 105))
        else:
            self.error.set_text(fill(message, 105))
        self.fig.canvas.draw_idle()

    def rename(self, event=None):
        self._action(lambda: self.app.rename_setup(self.fields['name'].text), 'Name applied to your custom setup.')
        origin = self.app.initial_setup.provenance.get('origin_preset')
        self.origin.set_text('Custom copy / origin: '+str(origin or 'free preparation'))

    def prepare(self, event=None):
        self.finish()
        self.app.prepare_setup()

    def save_initial(self, event=None):
        self._action(lambda: self.app.session.save_initial(self.fields['path'].text),
                     'Saved initial setup. Restart/load restores the configured beginning.')

    def save_current(self, event=None):
        def save():
            from pathlib import Path
            path, initial = Path(self.fields['capture_path'].text), Path(self.fields['path'].text)
            if path.resolve() == initial.resolve() or (path.exists() and initial.exists() and path.samefile(initial)):
                raise ValueError('Choose a capture path different from the initial setup path.')
            captured = self.app.session.save_current(path, self.fields['capture_name'].text)
            self.error.set_text(fill(captured.provenance['capture_note']+f' Fixed dt = {captured.dt:.9g} yr.', 105))
        # Capture's detailed step/epoch disclosure must remain visible.
        try:
            save()
        except (ValueError, OSError) as exc:
            self.error.set_text(fill(str(exc), 105))
        self.fig.canvas.draw_idle()

    def load(self, event=None):
        try:
            self.app.load_setup(self.fields['path'].text)
        except (ValueError, OSError) as exc:
            self.error.set_text(fill(str(exc), 105)); self.fig.canvas.draw_idle()
            return
        self.was_paused = True
        self.finish()

    def preset(self, event=None):
        from .scenarios import COMPATIBILITY, get_scenario
        self.app.select_setup(get_scenario(COMPATIBILITY))
        self.was_paused = True
        self.finish()

    def on_key(self, event):
        if event.key == 'escape': self.finish()

    def finish(self):
        if self.finished: return
        import matplotlib.pyplot as plt
        import time
        self.finished = True
        for widget in self.widgets: widget.disconnect_events()
        for connection in self.connections: self.fig.canvas.mpl_disconnect(connection)
        self.app.experiment_editor = None
        self.app.paused = self.was_paused
        self.app.pending, self.app.last = 0., time.monotonic()
        self.app.sidebar._sync()
        self.app._refresh()
        plt.close(self.fig)
