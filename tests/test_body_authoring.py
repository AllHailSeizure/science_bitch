from dataclasses import replace
import numpy as np
import pytest
from simulations.star_system.model import BodySpec, InitialBody, InitialSetup, PopulationSpec
from simulations.star_system.initialization import build_system, insert_body, circular_relative_velocity
from simulations.star_system.history import StateFrame
from simulations.star_system.core import G

SUN = BodySpec('Sun', 'star', 1., .00465, temperature=5772.)
EARTH = BodySpec('Earth', 'planet', 3e-6, 4.3e-5)

def setup(dt=.0001):
    return InitialSetup('Authored', bodies=(InitialBody('sun', SUN, (3., 2.), (.2, -.1)),
        InitialBody('earth', EARTH, (1., 0.), (0., 6.28), 'sun')), dt=dt)

def test_arbitrary_world_reference_chain_lone_body_binary():
    prepared = build_system(setup())
    np.testing.assert_array_equal(prepared.system.r, [[3, 2], [4, 2]])
    np.testing.assert_allclose(prepared.system.v, [[.2, -.1], [.2, 6.18]])
    lone = build_system(replace(setup(), bodies=setup().bodies[:1]))
    lone.system.step()
    np.testing.assert_allclose(lone.system.r[0], [3.00002, 1.99999])
    binary = build_system(replace(setup(), bodies=(InitialBody('a', SUN, (-1, 0), (0, -2)), InitialBody('b', SUN, (1, 0), (0, 2)))))
    binary.system.step(10)
    assert binary.body_ids == ('a', 'b')

@pytest.mark.parametrize('bodies', [
    (InitialBody('a', SUN, (0,0), (0,0), 'missing'),),
    (InitialBody('a', SUN, (1,0), (0,0), 'b'), InitialBody('b', SUN, (1,0), (0,0), 'a')),
    (InitialBody('a', SUN, (0,0), (0,0)), InitialBody('a', SUN, (2,0), (0,0))),
    (InitialBody('a', SUN, (0,0), (0,0)), InitialBody('b', SUN, (.001,0), (0,0))),
])
def test_invalid_complete_setup_rejected(bodies):
    with pytest.raises(ValueError):
        build_system(InitialSetup('Bad', bodies=bodies, dt=.0001))

def test_insertion_preserves_every_old_row_time_and_starting_setup():
    before = build_system(setup())
    before.system.step(5)
    r, v = before.system.r.copy(), before.system.v.copy()
    after = insert_body(before, InitialBody('new', EARTH, (2,1), (0,1), 'sun'))
    np.testing.assert_array_equal(after.system.r[:len(r)], r)
    np.testing.assert_array_equal(after.system.v[:len(v)], v)
    assert after.system.time == before.system.time
    assert not np.shares_memory(after.system.r, before.system.r)
    assert after.setup is before.setup
    np.testing.assert_array_equal(after.system.r[-1], r[0] + (2,1))
    assert after.events[-1].kind == 'insertion'
    with pytest.raises(ValueError):
        insert_body(before, InitialBody('earth', EARTH, (2,0), (0,0)))
    np.testing.assert_array_equal(before.system.r, r)

def test_circular_aid_is_relative_and_prograde():
    np.testing.assert_allclose(circular_relative_velocity((1,0), 1., 3e-6), [0, np.sqrt(G*(1+3e-6))])
    with pytest.raises(ValueError):
        circular_relative_velocity((0,0), 1., 0.)

def test_world_history_is_copied_readonly_and_identity_aligned():
    p = build_system(setup())
    frame = StateFrame.capture(p)
    saved = frame.r.copy()
    p.system.step()
    np.testing.assert_array_equal(frame.r, saved)
    assert frame.body_ids == p.body_ids
    with pytest.raises(ValueError):
        frame.r[0] = 0

def test_high_speed_unresolved_step_is_transactional_and_insertion_shortens_segment():
    p = build_system(setup())
    fast = insert_body(p, InitialBody('flyby', EARTH, (0, 4), (1e5, 0)))
    assert fast.system.dt < p.system.dt
    assert fast.events[-1].new_dt == fast.system.dt
    assert fast.segment == p.segment + 1
    fast.system.dt = 1.
    before = fast.system.r.copy(), fast.system.v.copy(), fast.system.time
    with pytest.raises(ValueError, match='crossing|encounter'):
        fast.system.step()
    np.testing.assert_array_equal(fast.system.r, before[0])
    np.testing.assert_array_equal(fast.system.v, before[1])
    assert fast.system.time == before[2]
# Headless events exercise the same Matplotlib callbacks as the live authoring UI.
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, KeyEvent
from simulations.star_system.plotting import App
from simulations.star_system.rendering import coordinates, inverse_coordinates, draw_system, dominant_star
from simulations.star_system.populations import Population

@pytest.fixture
def app():
    app = App()
    app.timer.stop()
    app.fig.canvas.draw()
    yield app
    app.close(); plt.close(app.fig)

def click(app, ax, point=None):
    pixel = ax.transAxes.transform((.5,.5)) if point is None else ax.transData.transform(point)
    for name in ('button_press_event','button_release_event'):
        app.fig.canvas.callbacks.process(name,MouseEvent(name,app.fig.canvas,*pixel,button=1))

@pytest.mark.parametrize('r',[(0,0),(1e-12,-1e-12),(1,0),(0,80)])
def test_inverse_placement_transform(r):
    np.testing.assert_allclose(inverse_coordinates(coordinates(r)),r,atol=1e-14)

def test_editor_live_placement_arrow_drag_confirm_retains_camera_and_restart(app):
    app._advance(5); app.selected_index=1; app.following=True; app._refresh()
    old=app.system; old_r=old.r.copy(); old_v=old.v.copy(); elapsed=old.time
    old_scene=app.scene; limits=app.ax.get_xlim(),app.ax.get_ylim()
    click(app,app.sidebar.add.ax)
    editor=app.body_editor
    assert app.paused and not app.pause.active
    assert editor.fields['reference'].text == 'earth'
    editor.fields['reference'].set_val('sun')
    editor.category.set_active(4)
    editor.fields['mass'].set_val('1e-8')
    editor.fields['radius'].set_val('1e-7')
    click(app,app.ax,coordinates(np.array((0,2))+old.r[0]-app.scene.origin))
    np.testing.assert_allclose([float(editor.fields[k].text) for k in ('x','y')],(0,2),atol=.02)
    editor.fields['vx'].set_val('2'); editor.fields['vy'].set_val('1')
    editor.preview(); anchor=app.ax.transData.transform(editor.anchor); tip=app.ax.transData.transform(editor.velocity_tip)
    old_length=np.linalg.norm(tip-anchor)
    editor.fields['vx'].set_val('4'); editor.fields['vy'].set_val('2'); editor.preview()
    assert np.linalg.norm(app.ax.transData.transform(editor.velocity_tip)-anchor) == pytest.approx(2*old_length)
    start=app.ax.transData.transform(editor.velocity_tip); end=anchor+np.array((48,-36))
    for name,pixel in [('button_press_event',start),('motion_notify_event',end),('button_release_event',end)]:
        app.fig.canvas.callbacks.process(name,MouseEvent(name,app.fig.canvas,*pixel,button=1))
    np.testing.assert_allclose([float(editor.fields[k].text) for k in ('vx','vy')],(4,-3),atol=.1)
    click(app,editor.confirm.ax)
    assert app.body_editor is None and app.paused
    assert app.scene is old_scene and app.following and app.selected_index==1
    assert (app.ax.get_xlim(),app.ax.get_ylim()) == limits
    np.testing.assert_array_equal(app.system.r[:-1],old_r)
    np.testing.assert_array_equal(app.system.v[:-1],old_v)
    assert app.system.time == elapsed
    np.testing.assert_allclose(app.system.v[-1]-old_v[0],(4,-3),atol=.1)
    new_id=app.prepared.body_ids[-1]
    assert app.scene.trails_by_id[new_id].get_segments()==[]
    app.restart()
    assert new_id not in app.prepared.body_ids and app.system.time==0

def test_editor_cancel_invalid_and_hidden_widgets_preserve_state(app):
    app.toggle_pause()  # Explicitly enter live mode from paused preparation.
    app._advance(3)
    previous=app.system; r=previous.r.copy(); v=previous.v.copy(); elapsed=previous.time
    click(app,app.sidebar.add.ax); editor=app.body_editor
    editor.fields['mass'].set_val('nan'); click(app,editor.confirm.ax)
    assert app.body_editor is editor and editor.error.get_text()
    assert app.system is previous
    app.toggle_sidebar()
    assert all(not ax.get_visible() for ax in editor.axes)
    assert all(not w.active for w in editor.widgets)
    app.toggle_sidebar()
    click(app,editor.cancel.ax)
    assert not app.paused and app.system is previous
    np.testing.assert_array_equal(app.system.r,r); np.testing.assert_array_equal(app.system.v,v)
    assert app.system.time==elapsed

def test_preparation_duplicate_catalog_custom_edit_remove_and_dependencies(app):
    click(app,app.sidebar.prepare.ax)
    assert app.preparing and app.paused
    click(app,app.sidebar.add.ax); editor=app.body_editor
    editor.fields['y'].set_val('2'); click(app,editor.confirm.ax)
    assert app.body_editor is None
    assert app.prepared.body_specs[-1].catalog_key=='earth'
    assert len(set(app.prepared.body_ids))==4
    new_id=app.prepared.body_ids[-1]; app.selected_id=new_id
    click(app,app.sidebar.edit.ax); editor=app.body_editor
    editor.fields['mass'].set_val('7e-6'); click(app,editor.confirm.ax)
    assert app.prepared.body_specs[-1].mass==7e-6
    assert app.prepared.body_specs[-1].sources['mass'].status=='user-authored override'
    app.selected_id='sun'; click(app,app.sidebar.remove.ax)
    assert 'sun' in app.prepared.body_ids and 'reference' in app.status.get_text().lower()
    app.selected_id=new_id; click(app,app.sidebar.remove.ax)
    assert new_id not in app.prepared.body_ids
    app.toggle_pause()
    assert not app.preparing
    app.edit_body(body_id='earth')
    assert app.body_editor is None

def test_insertion_after_population_preserves_particle_ids_and_history():
    from simulations.star_system.initialization import legacy_setup
    p=build_system(legacy_setup(population=Population(count=8)))
    p.system.step(3); frame=StateFrame.capture(p)
    after=insert_body(p,InitialBody('second-star',SUN,(20,0),(0,0)))
    assert after.body_ids[:len(p.body_ids)]==p.body_ids
    np.testing.assert_array_equal(after.system.r[:-1],p.system.r)
    fig,ax=plt.subplots()
    try:
        scene=draw_system(ax,after,[frame,StateFrame.capture(after)])
        assert 'second-star' in scene.artists_by_id
        assert scene.trails_by_id['second-star'].get_segments()==[]
        assert len(scene.particles.get_offsets())==8
    finally: plt.close(fig)

def test_multi_star_lighting_uses_dominant_flux_not_first_row():
    cold=replace(SUN,name='Faint',radius=.001,temperature=2000)
    bright=replace(SUN,name='Bright',radius=.02,temperature=8000)
    p=build_system(InitialSetup('Lighting',(InitialBody('faint',cold,(-2,0),(0,0)),InitialBody('earth',EARTH,(0,0),(0,0)),InitialBody('bright',bright,(2,0),(0,0))),.0001))
    assert dominant_star(p,1)==2
    fig,ax=plt.subplots()
    try:
        scene=draw_system(ax,p)
        assert len(scene.artists_by_id)==3
        image=np.asarray(scene.artists_by_id['earth'].offsetbox.get_data())
        assert image[25:55,48:68,:3].mean()>3*image[25:55,12:32,:3].mean()
    finally: plt.close(fig)

@pytest.mark.parametrize('case',['binary','flyby'])
def test_binary_and_fast_flyby_step_halving_converges(case):
    def evolve(dt):
        if case=='binary':
            speed=np.sqrt(G*2/2)/2
            bodies=(InitialBody('a',SUN,(-1,0),(0,-speed)),InitialBody('b',SUN,(1,0),(0,speed)))
            duration=2.
        else:
            bodies=(InitialBody('a',SUN,(0,0),(0,0)),InitialBody('b',EARTH,(-4,1),(40,0)))
            duration=.2
        p=build_system(InitialSetup(case,bodies,dt))
        p.system.step(round(duration/dt))
        return p.system.r
    coarse=evolve(.0005); fine=evolve(.00025); reference=evolve(.000125)
    error=np.max(np.abs(coarse-reference)); half_error=np.max(np.abs(fine-reference))
    print(f'{case}: coarse-reference {error:.6g} AU; half-reference {half_error:.6g} AU')
    assert error<1e-6
    assert half_error<error/5
@pytest.mark.parametrize('field,value',[('r',(float('nan'),0)),('v',(0,float('inf')))])
def test_nonfinite_authored_vectors_are_rejected(field,value):
    with pytest.raises(ValueError,match='finite'):
        replace(setup().bodies[0],**{field:value})

def test_reference_chain_resolves_position_and_velocity_once():
    root=InitialBody('root',SUN,(5,6),(.25,.5))
    child=InitialBody('child',EARTH,(1,2),(3,4),'root')
    grandchild=InitialBody('grandchild',EARTH,(2,3),(4,5),'child')
    p=build_system(InitialSetup('Chain',(grandchild,root,child),.00001))
    np.testing.assert_array_equal(p.system.r[0],(8,11))
    np.testing.assert_array_equal(p.system.v[0],(7.25,9.5))
    p.system.step(2)
    assert p.setup.bodies[0].r==(2,3)

def test_intermediate_rejection_rolls_back_all_kicks_drifts_and_time():
    from simulations.star_system.core import System
    s=System(np.array([1.,1.]),np.array([[-.5,0],[.5,0]]),np.array([[20.,0],[-20.,0]]),.0024)
    s.validate_step()  # initial state is resolved; an internal stage is not
    r,v=s.r.copy(),s.v.copy()
    with pytest.raises(ValueError,match='unresolved'): s.step()
    np.testing.assert_array_equal(s.r,r); np.testing.assert_array_equal(s.v,v)
    assert s.time==0

def test_invalid_insertion_preserves_active_state_and_explicit_dt_rejects():
    p=build_system(setup()); p.system.step(2)
    r,v,t=p.system.r.copy(),p.system.v.copy(),p.system.time
    for body in (InitialBody('bad',EARTH,(0,0),(0,0),'sun'),InitialBody('bad',EARTH,(1,0),(0,0),'missing')):
        with pytest.raises(ValueError): insert_body(p,body)
        np.testing.assert_array_equal(p.system.r,r); np.testing.assert_array_equal(p.system.v,v)
        assert p.system.time==t
    with pytest.raises(ValueError,match='unresolved'):
        build_system(replace(setup(),dt=1.))

def test_editor_particle_selection_requires_reference_and_cache_matches_full():
    app=App(population=Population(count=3)); app.timer.stop(); app.fig.canvas.draw()
    try:
        app.selected_index=len(app.system.masses)-1
        click(app,app.sidebar.add.ax); e=app.body_editor
        assert e.fields['reference'].text==''
        click(app,e.confirm.ax)
        assert 'reference' in e.error.get_text().lower()
        click(app,e.reference_next.ax)
        assert e.fields['reference'].text=='sun'
        e.fields['vx'].set_val('3'); e.preview()
        app.fig.canvas.draw(); e.fields['vy'].set_val('2'); e.preview()
        incremental=np.asarray(app.fig.canvas.buffer_rgba()).copy()
        app.fig.canvas.draw()
        np.testing.assert_array_equal(incremental,np.asarray(app.fig.canvas.buffer_rgba()))
        assert all(ax.get_animated() for ax in e.axes)
    finally:
        app.close(); plt.close(app.fig)

def test_generated_particle_preparation_edit_remove_is_explicitly_deferred():
    app=App(population=Population(count=3)); app.timer.stop()
    try:
        app.prepare_setup(); app.selected_index=len(app.system.masses)-1
        previous=app.system
        app.edit_body(body_id=app.selected_id)
        assert app.body_editor is None and 'population settings' in app.status.get_text()
        app.remove_body()
        assert app.system is previous
        assert 'population settings' in app.status.get_text()
    finally:
        app.close(); plt.close(app.fig)


def test_moon_editor_catalog_distance_vector_cancel_and_live_insertion(app):
    from simulations.star_system.data import AU_KM
    app.toggle_pause()  # Cancel must restore the live mode selected by the user.
    app._advance(3); app._refresh()
    click(app,app.sidebar.body_ax,(.3,.92-.15-.04))
    assert app.selected_id=='earth'
    app.following=True; app._refresh()
    r,v,t=app.system.r.copy(),app.system.v.copy(),app.system.time
    click(app,app.sidebar.add.ax); e=app.body_editor
    e.category.set_active(2)
    assert e.fields['kind'].text=='moon' and e.fields['reference'].text=='earth'
    e.fields['separation'].set_val('384400')
    assert float(e.fields['x'].text)==pytest.approx(384400/AU_KM)
    assert 'Earth [earth]' in e.legend.get_text() and 'system r' in e.legend.get_text()
    click(app,e.circular.ax)
    # Exercise the same linear velocity handle used for every authored body.
    anchor=app.ax.transData.transform(e.anchor); start=app.ax.transData.transform(e.velocity_tip)
    end=anchor+np.array((24,12))
    for name,pixel in [('button_press_event',start),('motion_notify_event',end),('button_release_event',end)]:
        app.fig.canvas.callbacks.process(name,MouseEvent(name,app.fig.canvas,*pixel,button=1))
    np.testing.assert_allclose([float(e.fields[k].text) for k in ('vx','vy')],(2,1),atol=.1)
    click(app,e.cancel.ax)
    assert not app.paused and app.following
    np.testing.assert_array_equal(app.system.r,r); np.testing.assert_array_equal(app.system.v,v)
    assert app.system.time==t
    click(app,app.sidebar.add.ax); e=app.body_editor; e.category.set_active(2)
    e.fields['separation'].set_val('384400'); click(app,e.circular.ax)
    relative_v=np.array([float(e.fields[k].text) for k in ('vx','vy')])
    click(app,e.confirm.ax)
    assert app.body_editor is None and app.paused and app.following
    np.testing.assert_array_equal(app.system.r[:-1],r); np.testing.assert_array_equal(app.system.v[:-1],v)
    np.testing.assert_array_equal(app.system.v[-1],v[1]+relative_v)
    assert app.system.time==t and app.prepared.reference_ids[-1]=='earth'
    assert app.system.dt < app.initial_setup.dt
    assert 'fixed-step segment' in app.status.get_text()


def test_moon_editor_custom_preparation_reference_prompt_invalid_and_cancel(app):
    app.prepare_setup(); app.selected_id='sun'
    click(app,app.sidebar.add.ax); e=app.body_editor
    e.category.set_active(4); e.fields['kind'].set_val('moon')
    assert e.fields['reference'].text==''
    click(app,e.confirm.ax)
    assert app.body_editor is e and 'planet' in e.error.get_text().lower()
    click(app,e.reference_next.ax)
    assert e.fields['reference'].text=='earth'
    e.fields['name'].set_val('Custom satellite'); e.fields['mass'].set_val('1e-9'); e.fields['radius'].set_val('1e-7')
    e.fields['separation'].set_val('500000'); click(app,e.circular.ax)
    assert float(e.fields['dt'].text)<app.initial_setup.dt
    assert 'period / 600' in e.legend.get_text()
    e.fields['reference'].set_val('sun'); before=app.system
    click(app,e.confirm.ax)
    assert app.system is before and 'planet' in e.error.get_text().lower()
    e.fields['reference'].set_val('earth'); click(app,e.confirm.ax)
    assert app.body_editor is None and app.preparing and app.paused and app.system.time==0
    assert app.initial_setup.bodies[-1].spec.name=='Custom satellite'
    assert app.initial_setup.bodies[-1].reference_id=='earth'
    click(app,app.sidebar.add.ax); e=app.body_editor; e.category.set_active(2)
    app.fig.canvas.callbacks.process('key_press_event',KeyEvent('key_press_event',app.fig.canvas,key='escape'))
    assert app.body_editor is None and app.preparing and app.paused
