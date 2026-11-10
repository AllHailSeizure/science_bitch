"""Selection geometry applies to generic identities, without editing dynamics."""
from dataclasses import replace

import numpy as np

from simulations.star_system.context import selection_context
from simulations.star_system.history import StateFrame
from simulations.star_system.initialization import build_system
from simulations.star_system.templates import get_experiment


def test_clear_selection_clears_all_context():
    frame = StateFrame.capture(build_system(get_experiment('solar-system')))
    result = selection_context(frame, None)
    assert result.pair is None and not result.orbits and len(result.guides) == 0


def test_selected_star_shows_child_orbits_without_an_arbitrary_lagrange_pair():
    frame = StateFrame.capture(build_system(get_experiment('solar-system')))
    result = selection_context(frame, 'sun')
    assert result.pair is None
    assert len(result.orbits) == 8
    for key, curve in result.orbits.items():
        radius = np.linalg.norm(frame.r[frame.body_ids.index(key)]-frame.r[0])
        np.testing.assert_allclose(np.linalg.norm(curve-frame.r[0], axis=1), radius, rtol=1e-9)


def test_planet_selection_uses_its_pair_and_its_moon_orbit():
    frame = StateFrame.capture(build_system(get_experiment('earth-moon')))
    before = frame.r.copy(), frame.v.copy()
    result = selection_context(frame, 'earth')
    assert result.pair == ('sun', 'earth')
    assert set(result.orbits) == {'moon'}
    assert result.guides.shape == (2, 2)
    r = np.linalg.norm(frame.r[frame.body_ids.index('earth')]-frame.r[0])
    np.testing.assert_allclose(np.linalg.norm(result.guides-frame.r[0], axis=1), r)
    np.testing.assert_array_equal(frame.r, before[0])
    np.testing.assert_array_equal(frame.v, before[1])


def test_geometry_has_no_named_body_assumptions_and_updates_after_selection():
    frame = StateFrame.capture(build_system(get_experiment('earth-moon')))
    ids = tuple('body-'+str(i) for i in range(len(frame.body_ids)))
    mapping = dict(zip(frame.body_ids, ids))
    renamed = replace(frame, body_ids=ids,
                      body_specs=tuple(replace(s, name='Object '+str(i), catalog_key=None) for i, s in enumerate(frame.body_specs)),
                      reference_ids=tuple(mapping.get(ref) for ref in frame.reference_ids))
    selected = mapping['moon']
    result = selection_context(renamed, selected)
    assert result.pair == (mapping['earth'], selected)
    assert not result.orbits
    assert not selection_context(renamed, 'missing').orbits


def test_unbound_child_is_not_drawn_as_a_closed_orbit():
    frame = StateFrame.capture(build_system(get_experiment('solar-system')))
    velocity = frame.v.copy()
    velocity[1] *= 4
    result = selection_context(replace(frame, v=velocity), 'sun')
    assert 'mercury' not in result.orbits
