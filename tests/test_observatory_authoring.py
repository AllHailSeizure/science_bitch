"""Observatory authoring: live insertion, custom bodies, and catalog sources."""
import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from simulations.star_system.data import SOLAR_KG
from simulations.star_system.persistence import setup_digest


@pytest.fixture(scope='module')
def qtapp():
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qtapp):
    from simulations.star_system.desktop import Observatory
    instance = Observatory()
    instance.timer.stop()
    instance.show()
    qtapp.processEvents()
    yield instance
    instance.dirty = False
    instance.close()
    qtapp.processEvents()


def open_solar(window, qtapp):
    window.choose_button.click()
    window.open_button.click()
    qtapp.processEvents()


def pointer(pixel, data=(1., 1.)):
    return SimpleNamespace(inaxes=None, button=1, x=pixel[0], y=pixel[1], xdata=data[0], ydata=data[1])


def test_live_insertion_drags_confirms_and_keeps_time_and_starting_setup(window, qtapp):
    open_solar(window, qtapp)
    window.step_button.click()
    elapsed = window.prepared.system.time
    old_r, old_v = window.prepared.system.r.copy(), window.prepared.system.v.copy()
    digest = setup_digest(window.setup)
    limits = (tuple(window.ax.get_xlim()), tuple(window.ax.get_ylim()))
    count = len(window.prepared.body_ids)
    window.tabs.setCurrentIndex(1)
    window.catalog.type_combo.setCurrentText('Other')
    window.catalog.body_combo.setCurrentIndex(window.catalog.body_combo.findData('vesta'))
    window.catalog.reference_combo.setCurrentIndex(window.catalog.reference_combo.findData('sun'))
    qtapp.processEvents()
    window.catalog.add_button.click()
    assert not window.running and window.pending_body is not None
    assert window.prepared.system.time == elapsed
    window.place_at(old_r[0]+(2.5, .4))
    assert window.live_draft is not None and len(window.prepared.body_ids) == count
    assert 'Sun' in window.catalog.draft_readout.text() and 'AU/yr' in window.catalog.draft_readout.text()
    anchor = window.ax.transData.transform(window._live_anchor)
    length = np.linalg.norm(window.ax.transData.transform(window.live_tip)-anchor)
    doubled = window.live_draft
    window.live_draft = replace(doubled, v=tuple(np.asarray(doubled.v)*2))
    window.refresh_scene()
    assert np.linalg.norm(window.ax.transData.transform(window.live_tip)-window.ax.transData.transform(window._live_anchor)) == pytest.approx(2*length)
    window.live_draft = doubled
    window.refresh_scene()
    tip = window.ax.transData.transform(window.live_tip)
    end = tip+np.array((36., -24.))
    for pixel, method in ((tip, window.on_press), (end, window.on_motion), (end, window.on_release)):
        event = pointer(pixel)
        event.inaxes = window.ax
        method(event)
    relative_r = np.asarray(window.live_draft.r)
    relative_v = np.asarray(window.live_draft.v)
    np.testing.assert_allclose(relative_v, (end-window.ax.transData.transform(window._live_anchor))/12., atol=1e-6)
    assert window.confirm_insertion_button.isVisible()
    window.confirm_insertion_button.click()
    qtapp.processEvents()
    new_id = window.prepared.body_ids[-1]
    assert window.live_draft is None and not window.running
    assert window.run_button.text() == 'Resume'
    assert 'Reset' in window.notice.text()
    assert window.prepared.system.time == elapsed
    np.testing.assert_array_equal(window.prepared.system.r[:-1], old_r)
    np.testing.assert_array_equal(window.prepared.system.v[:-1], old_v)
    np.testing.assert_allclose(window.prepared.system.r[-1], old_r[0]+relative_r)
    np.testing.assert_allclose(window.prepared.system.v[-1], old_v[0]+relative_v)
    assert setup_digest(window.setup) == digest
    assert (tuple(window.ax.get_xlim()), tuple(window.ax.get_ylim())) == limits
    assert len(window.scene.trails_by_id[new_id].get_segments()) == 0
    inserted = window.prepared.system.v[-1].copy()
    dt = window.prepared.system.dt
    window.step_button.click()
    assert window.prepared.system.time == pytest.approx(elapsed+dt)
    assert new_id in window.prepared.body_ids
    assert not np.allclose(window.prepared.system.v[-1], inserted)
    window.reset_button.click()
    assert new_id not in window.prepared.body_ids
    assert window.prepared.system.time == 0 and window.editing
    assert setup_digest(window.setup) == digest


def test_cancel_live_placement_resumes_without_adding(window, qtapp):
    open_solar(window, qtapp)
    window.run_button.click()
    assert window.running
    positions = window.prepared.system.r.copy()
    digest = setup_digest(window.setup)
    window.tabs.setCurrentIndex(1)
    window.catalog.add_button.click()
    assert not window.running and window.pending_body is not None
    window.cancel_interaction()
    assert window.running and window.pending_body is None and window.live_draft is None
    np.testing.assert_array_equal(window.prepared.system.r, positions)
    assert setup_digest(window.setup) == digest


def test_overlapping_live_insertion_stays_uncommitted(window, qtapp):
    open_solar(window, qtapp)
    window.step_button.click()
    elapsed = window.prepared.system.time
    count = len(window.prepared.body_ids)
    window.tabs.setCurrentIndex(1)
    window.catalog.type_combo.setCurrentText('Other')
    window.catalog.body_combo.setCurrentIndex(window.catalog.body_combo.findData('vesta'))
    window.catalog.reference_combo.setCurrentIndex(window.catalog.reference_combo.findData('sun'))
    window.catalog.add_button.click()
    window.place_at(window.prepared.system.r[0]+(0.001, 0.))
    assert window.live_draft is not None
    window.confirm_insertion_button.click()
    assert window.live_draft is not None
    assert 'overlap' in window.notice.text().lower()
    assert len(window.prepared.body_ids) == count
    assert window.prepared.system.time == elapsed


def test_custom_body_is_created_edited_and_keeps_unknown_radius(window, qtapp):
    open_solar(window, qtapp)
    window.tabs.setCurrentIndex(1)
    window.catalog.custom_check.setChecked(True)
    window.catalog.custom_name.setText('Rock')
    window.catalog.custom_kind.setCurrentIndex(window.catalog.custom_kind.findData('asteroid'))
    window.catalog.custom_mass.setText('1e18')
    window.catalog.custom_radius.setText('')
    window.catalog.reference_combo.setCurrentIndex(window.catalog.reference_combo.findData('sun'))
    qtapp.processEvents()
    assert 'Unknown' in window.catalog.information.details.text()
    window.catalog.add_button.click()
    window.place_at(window.prepared.system.r[0]+(3., 1.))
    body = window.setup.bodies[-1]
    assert body.spec.name == 'Rock' and body.spec.kind == 'asteroid'
    assert body.spec.provenance == 'custom' and body.spec.radius is None
    assert body.spec.sources['radius'].status == 'unknown'
    assert body.spec.mass == pytest.approx(1e18/SOLAR_KG)
    window.properties.fields['mass'].setText('2e18')
    window.properties.apply_button.click()
    edited = window.setup.bodies[-1].spec
    assert edited.mass == pytest.approx(2e18/SOLAR_KG)
    assert edited.sources['mass'].status == 'user-authored override'
    assert edited.radius is None


def test_inspector_shows_units_sources_and_unknown_radius(window, qtapp):
    open_solar(window, qtapp)
    window.tabs.setCurrentIndex(1)
    window.catalog.body_combo.setCurrentIndex(window.catalog.body_combo.findData('51_peg_b'))
    qtapp.processEvents()
    text = window.catalog.information.details.text()
    assert 'Unknown — not estimated' in text
    assert 'marker' in text
    assert 'https://exoplanetarchive.ipac.caltech.edu/overview/51%20Peg' in text
    window.select_body('jupiter')
    details = window.body_information.details.text()
    assert 'kg' in details and 'km' in details
    assert 'https://nssdc.gsfc.nasa.gov/planetary/factsheet/' in details
    assert 'rounded legacy measurement' in details


def test_second_star_is_a_normal_body_in_the_starting_setup(window, qtapp):
    open_solar(window, qtapp)
    window.tabs.setCurrentIndex(1)
    window.catalog.type_combo.setCurrentText('Stellar')
    qtapp.processEvents()
    assert 'binary' in window.catalog.note.text().lower()
    window.catalog.body_combo.setCurrentIndex(window.catalog.body_combo.findData('alpha_cen_a'))
    window.catalog.reference_combo.setCurrentIndex(window.catalog.reference_combo.findData('sun'))
    window.catalog.add_button.click()
    window.place_at(window.prepared.system.r[0]+(20., 0.))
    assert 'sun' in window.prepared.body_ids
    assert window.prepared.body_specs[-1].catalog_key == 'alpha_cen_a'
    assert sum(spec.kind == 'star' for spec in window.prepared.body_specs) == 2
    assert window.prepared.system.time == 0
