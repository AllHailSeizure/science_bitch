"""Overlay controls for the full-window observatory."""
from textwrap import fill

import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from .data import STARS, PLANETS
from .rendering import PANEL, TEXT, MUTED, ACCENT, planet_sprite, star_sprite


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
        self.add = button(fig, (.02, .77, .13, .037), 'Add a body')
        self.population = button(fig, (.16, .77, .13, .037), 'Population...')
        self.reset = button(fig, (.02, .72, .13, .037), 'Restart')
        self.pause = button(fig, (.16, .72, .13, .037), 'Pause')
        self.step = button(fig, (.02, .67, .085, .037), 'Step')
        self.home = button(fig, (.112, .67, .085, .037), 'Fit view')
        self.follow = button(fig, (.204, .67, .086, .037), 'Follow')
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
        self.widgets = [self.add, self.population, self.reset, self.pause, self.step, self.home, self.follow, self.speed]
        self.axes = [self.background, self.body_ax, self.preview, *(w.ax for w in self.widgets), *(w.ax for w in self.catalog_widgets)]
        self.add.on_clicked(lambda event: self.set_catalog(True))
        self.catalog_close.on_clicked(lambda event: self.set_catalog(False))
        self.menu.on_clicked(app.toggle_sidebar)
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.set_catalog(False)

    def set_catalog(self, visible):
        self.catalog_open = bool(visible)
        self._sync()

    def _sync(self):
        self.background.set_visible(self.visible)
        for artist in (self.body_ax, self.preview, self.inspector):
            artist.set_visible(self.visible and not self.catalog_open)
        for widget in self.widgets:
            widget.ax.set_visible(self.visible)
            widget.active = self.visible and self.app.population_editor is None
        for widget in self.catalog_widgets:
            enabled = self.visible and self.catalog_open
            widget.ax.set_visible(enabled)
            widget.active = enabled and self.app.population_editor is None
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
        if self.visible and not self.catalog_open and self.app.population_editor is None and event.inaxes == self.body_ax:
            self.offset = max(0, min(max(0, len(self.rows)-6), self.offset-int(event.step)))
            self._draw_rows()
            self.app.painter.invalidate()
            self.app.fig.canvas.draw_idle()

    def on_click(self, event):
        if self.visible and not self.catalog_open and self.app.population_editor is None and event.inaxes == self.body_ax and event.button == 1 and event.ydata is not None:
            row = int(np.floor((.92-event.ydata)/.15))
            index = self.offset + row
            if 0 <= row < 6 and 0 <= index < len(self.rows):
                self.app.selected_index = self.rows[index][0]
                self.app._refresh()

    def update(self, time, speed, selected_index):
        self.readout.set_text(f'{time:,.3f} yr   /   {speed:.3g} yr/sec')
        self.status.set_text(fill(self.status.get_text(), width=42))
        if selected_index == 0:
            image = star_sprite(self.app.star)
        elif selected_index <= len(self.app.planets):
            direction = self.app.system.r[0] - self.app.system.r[selected_index]
            image = planet_sprite(self.app.planets[selected_index-1], direction)
        else:
            image = None
        self.preview_image.set_visible(image is not None)
        if image is not None:
            self.preview_image.set_data(image)
