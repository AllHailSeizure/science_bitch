"""Shared draft body editor. Scene placement and velocity use distinct transforms."""
from dataclasses import replace
from uuid import uuid4
from textwrap import fill
import time
import numpy as np
from matplotlib.widgets import TextBox, Button, RadioButtons
from matplotlib.patches import FancyArrowPatch
from .data import CATALOG
from .model import BodySpec, InitialBody, PropertySource
from .initialization import build_system, insert_body, circular_relative_velocity
from .rendering import PANEL, TEXT, MUTED, ACCENT, coordinates, inverse_coordinates

VELOCITY_PIXELS_PER_AUYR = 12.


def reference_ids(bodies, exclude=None, world=False):
    """Stable choices shared by body placement and initial population drafts."""
    keys = [key for key, spec in bodies if spec.mass is not None and spec.mass > 0 and key != exclude]
    return keys + (['world'] if world else [])


class BodyEditor:
    """Drafts never mutate the active trajectory before successful confirmation."""
    def __init__(self, app, body_id=None):
        self.app, self.body_id = app, body_id
        self.was_paused, self.finished, self.dragging = app.paused, False, False
        self.axes, self.widgets, self.fields = [], [], {}
        self.spec = CATALOG['earth']
        initial = next((b for b in app.initial_setup.bodies if b.id == body_id), None)
        if initial is not None:
            self.spec = initial.spec
        selected = app.prepared.body_ids[app.selected_index]
        reference = selected if app.prepared.body_specs[app.selected_index].mass > 0 else ''
        if initial is not None:
            reference = initial.reference_id or 'world'
        self.panel = self._axes((0, .005, .31, .89))
        self.panel.set_xticks([]); self.panel.set_yticks([])
        self.panel.text(.05, .985, ('Edit starting body' if body_id else 'Add body') + (' / preparation' if app.preparing else ' / live insertion'), color=TEXT, fontsize=12, va='top')
        self.notice = self.panel.text(.05, .94, '', color=MUTED, fontsize=8, va='top', wrap=True)
        self.category = RadioButtons(self._axes((.02,.727,.27,.065)), ['star','planet','moon','asteroid','custom'], active=(['star','planet','moon','asteroid'].index(self.spec.kind) if self.spec.catalog_key else 4), activecolor=ACCENT, useblit=False)
        for label in self.category.labels:
            label.set_color(TEXT); label.set_fontsize(8)
        self.widgets.append(self.category)
        self.category.on_clicked(self.change_category)
        def field(key, label, value, y):
            widget = TextBox(self._axes((.118,y,.17,.026)), label+' ', initial=str(value), color='#1b3044', hovercolor='#294b60')
            widget.label.set_color(TEXT); widget.label.set_fontsize(8)
            widget.text_disp.set_color(TEXT); widget.text_disp.set_fontsize(8)
            self.fields[key] = widget; self.widgets.append(widget)
            return widget
        field('catalog','Catalog key',self.spec.catalog_key or 'custom',.688).on_submit(self.load_catalog)
        self.previous = self._button((.02,.652,.065,.026),'<', lambda e:self.browse(-1))
        self.load = self._button((.09,.652,.125,.026),'Load definition',lambda e:self.load_catalog(self.fields['catalog'].text))
        self.next = self._button((.22,.652,.065,.026),'>',lambda e:self.browse(1))
        for key,label,value,y in [('name','Name',self.spec.name,.614),('kind','Kind',self.spec.kind,.579),('mass','Mass (solar)',self.spec.mass,.544),('radius','Radius (AU)',self.spec.radius,.509),('temperature','Star temp (K)',self.spec.temperature,.474),('reference','Reference ID',reference,.425),('x','x (AU)',initial.r[0] if initial else 1.,.387),('y','y (AU)',initial.r[1] if initial else 0.,.352),('vx','vx (AU/yr)',initial.v[0] if initial else 0.,.304),('vy','vy (AU/yr)',initial.v[1] if initial else 0.,.269),('dt','Fixed dt (yr)',app.initial_setup.dt if app.preparing else app.system.dt,.225)]:
            field(key,label,'' if value is None else value,y)
        self.fields['dt'].set_active(app.preparing)
        if not app.preparing:
            self.fields['dt'].label.set_text('dt auto <= (yr) ')
        self.reference_next = self._button((.02,.193,.27,.026),'Reference: next massive / world',self.cycle_reference)
        self.hint = self.panel.text(.05,.22,'', color=MUTED, fontsize=8, va='top')
        self.circular = self._button((.02,.133,.27,.03),'Circular aid about reference',self.circular_aid)
        self.confirm = self._button((.02,.087,.13,.032),'Confirm',self.commit)
        self.cancel = self._button((.16,.087,.13,.032),'Cancel',lambda e:self.finish())
        self.error = self.panel.text(.05,.065,'',color='#efb17b',fontsize=8,va='top',wrap=True)
        self.marker, = app.ax.plot([], [], 'o', color=ACCENT, markersize=7, zorder=20)
        self.arrow = FancyArrowPatch((0,0),(0,0),arrowstyle='->',shrinkA=0,shrinkB=0,mutation_scale=14,color=ACCENT,linewidth=1.5,zorder=21)
        app.ax.add_patch(self.arrow)
        self.legend = app.ax.text(.975,.91,'',transform=app.ax.transAxes,color=ACCENT,fontsize=9,ha='right',va='top',zorder=21)
        self.artists = [self.marker,self.arrow,self.legend]
        for key in ('reference','x','y','vx','vy'):
            self.fields[key].on_text_change(lambda text:self.preview())
        self._describe()
        self.preview()

    def _axes(self, bounds):
        ax = self.app.fig.add_axes(bounds,zorder=30,facecolor=PANEL)
        for spine in ax.spines.values(): spine.set_visible(False)
        self.axes.append(ax)
        return ax

    def _button(self,bounds,label,callback):
        widget = Button(self._axes(bounds),label,color='#1b3044',hovercolor='#294b60',useblit=False)
        widget.label.set_color(TEXT); widget.label.set_fontsize(8)
        widget.on_clicked(callback); self.widgets.append(widget)
        return widget

    def _describe(self):
        unknown = [key for key in (('mass','radius','temperature') if self.spec.kind == 'star' else ('mass','radius')) if getattr(self.spec,key) is None]
        status = '; '.join(f'{k}: {v.status}' for k,v in self.spec.sources.items() if v.status not in ('measurement','rounded legacy measurement'))
        text = self.spec.name + (' — unknown '+', '.join(unknown)+'. Supply custom values when needed.' if unknown else '')
        self.notice.set_text(fill(text,width=49)+'\n'+fill(status[:110],width=49))
        self.hint.set_position((.05,.197))
        self.hint.set_text('Click to place; drag arrow tip for velocity.')

    def change_category(self, category):
        if category == 'custom':
            self.spec = BodySpec('Custom body','planet',None,None)
            self._fill_spec()
            return
        keys = [k for k,s in CATALOG.items() if s.kind == category]
        self.load_catalog(keys[0])

    def browse(self, delta):
        keys = [k for k,s in CATALOG.items() if s.kind == self.category.value_selected]
        if keys:
            current = self.fields['catalog'].text
            self.load_catalog(keys[((keys.index(current) if current in keys else -1)+delta)%len(keys)])

    def load_catalog(self, key):
        key = key.strip().lower()
        if key not in CATALOG:
            self.error.set_text('Unknown catalog key. Use category and < / > to browse.'); self.app.painter.paint(); return
        self.spec = CATALOG[key]
        self._fill_spec()

    def _fill_spec(self):
        for key in ('name','kind','mass','radius','temperature'):
            value=getattr(self.spec,key)
            self.fields[key].set_val('' if value is None else str(value))
        self.fields['catalog'].eventson=False
        self.fields['catalog'].set_val(self.spec.catalog_key or 'custom')
        self.fields['catalog'].eventson=True
        self.error.set_text(''); self._describe(); self.preview()

    def cycle_reference(self,event=None):
        p=self.app.prepared
        keys=reference_ids(zip(p.body_ids,p.body_specs), exclude=self.body_id, world=True)
        current=self.fields['reference'].text.strip()
        self.fields['reference'].set_val(keys[(keys.index(current)+1)%len(keys)] if current in keys else keys[0])
        self.preview()

    def reference_state(self):
        key=self.fields['reference'].text.strip()
        if key == 'world': return np.zeros(2),np.zeros(2),None
        p=self.app.prepared
        if not key or key not in p.body_ids:
            raise ValueError('Choose a massive reference ID from the body list, or enter world.')
        if key == self.body_id: raise ValueError('A body cannot reference itself.')
        i=p.body_ids.index(key)
        if p.system.masses[i] <= 0: raise ValueError('Choose a massive reference body.')
        return p.system.r[i],p.system.v[i],key

    def vectors(self):
        try: r=np.array([float(self.fields[k].text) for k in ('x','y')]); v=np.array([float(self.fields[k].text) for k in ('vx','vy')])
        except ValueError: raise ValueError('Position and velocity components must be numbers.') from None
        if not np.isfinite([r,v]).all(): raise ValueError('Position and velocity must be finite.')
        return r,v

    def preview(self):
        if self.finished: return
        try:
            r,v=self.vectors(); rr,_,_=self.reference_state()
            anchor=coordinates(r+rr-self.app.scene.origin)
            pixel=self.app.ax.transData.transform(anchor)
            tip=self.app.ax.transData.inverted().transform(pixel+v*VELOCITY_PIXELS_PER_AUYR)
            self.marker.set_data([anchor[0]],[anchor[1]])
            self.arrow.set_positions(anchor,tip)
            self.anchor, self.velocity_tip=anchor,tip
            self.legend.set_text(f'VELOCITY / linear {VELOCITY_PIXELS_PER_AUYR:g} px per AU/yr\nv = ({v[0]:.4g}, {v[1]:.4g}) AU/yr / speed {np.linalg.norm(v):.4g}')
            for a in self.artists: a.set_visible(True)
        except ValueError:
            for a in self.artists: a.set_visible(False)
        if hasattr(self.app,'painter'): self.app.painter.paint()

    def on_press(self,event):
        if event.button != 1 or event.inaxes != self.app.ax or self.app.sidebar.contains_event(event) or event.xdata is None: return
        tip=self.app.ax.transData.transform(getattr(self,'velocity_tip',(np.inf,np.inf)))
        if np.linalg.norm(tip-(event.x,event.y)) < 15:
            self.dragging=True; return
        rr,_,_=self.reference_state()
        world=inverse_coordinates((event.xdata,event.ydata))+self.app.scene.origin
        r=world-rr
        for key,value in zip(('x','y'),r): self.fields[key].set_val(f'{value:.16g}')
        self.preview()

    def on_motion(self,event):
        if not self.dragging or event.x is None: return
        v=(np.array([event.x,event.y])-self.app.ax.transData.transform(self.anchor))/VELOCITY_PIXELS_PER_AUYR
        for key,value in zip(('vx','vy'),v): self.fields[key].set_val(f'{value:.16g}')
        self.preview()

    def circular_aid(self,event=None):
        try:
            r,_=self.vectors(); _,_,key=self.reference_state()
            if key is None: raise ValueError('Circular aid requires a massive reference body, not world.')
            mass=float(self.fields['mass'].text)
            ref_mass=self.app.system.masses[self.app.prepared.body_ids.index(key)]
            v=circular_relative_velocity(r,ref_mass,mass)
            for key,value in zip(('vx','vy'),v): self.fields[key].set_val(f'{value:.16g}')
            self.error.set_text('Two-body prograde initial aid; no stability guarantee.')
        except ValueError as exc: self.error.set_text(fill(str(exc),width=49))
        self.preview()

    def draft_body(self):
        r,v=self.vectors(); _,_,reference=self.reference_state()
        values={key:(float(self.fields[key].text) if self.fields[key].text.strip() else None) for key in ('mass','radius','temperature')}
        overrides={key:value for key,value in values.items() if value != getattr(self.spec,key)}
        sources=dict(self.spec.sources)
        for key,value in overrides.items():
            sources[key]=PropertySource('user://override','user input','',value,{'mass':'solar masses','radius':'AU','temperature':'K'}[key],'identity',status='user-authored override')
        spec=replace(self.spec,name=self.fields['name'].text,kind=self.fields['kind'].text.strip(),**values,sources=sources,
                     provenance='user overrides of '+str(self.spec.catalog_key or 'custom definition') if (overrides or self.fields['name'].text != self.spec.name or self.fields['kind'].text.strip() != self.spec.kind) else self.spec.provenance)
        return InitialBody(self.body_id or (spec.catalog_key or 'custom')+'-'+uuid4().hex[:10],spec,tuple(r),tuple(v),reference)

    def commit(self,event=None):
        try:
            body=self.draft_body()
            if self.app.preparing:
                bodies=tuple(body if b.id == self.body_id else b for b in self.app.initial_setup.bodies)
                if self.body_id is None: bodies += (body,)
                setup=replace(self.app.initial_setup,bodies=bodies,dt=float(self.fields['dt'].text))
                prepared=build_system(setup)
            else:
                prepared=insert_body(self.app.prepared,body)
        except ValueError as exc:
            self.error.set_text(fill(str(exc),width=49)); self.app.painter.paint(); return
        preparing = self.app.preparing
        self.finish(confirmed=True)
        if preparing:
            self.app._legacy_configuration=False
            self.app.initial_setup=setup
        self.app._install_prepared(prepared,reset=preparing)
        if not preparing:
            self.app.status.set_text(prepared.events[-1].message)
            self.app._refresh()

    def finish(self,confirmed=False):
        if self.finished: return
        self.finished=True
        self.app.body_editor=None
        for w in self.widgets: w.disconnect_events()
        for ax in self.axes: ax.set_animated(False); ax.remove()
        for artist in self.artists:
            if artist in self.app.ax.get_children(): artist.remove()
        self.app.paused=True if confirmed else self.was_paused
        self.app.pending=0.; self.app.last=time.monotonic()
        self.app._sync_painter()
        self.app.sidebar._sync()
        self.app._refresh()
