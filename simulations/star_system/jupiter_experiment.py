"""Sun–Jupiter ring inspection for the desktop observatory.

Display and comparison only. The shared Newtonian solver moves every body.
Nothing in this module attracts a rock toward L4 or L5, edits a recorded
trajectory, or invents a Jupiter for the matched control.
"""
from dataclasses import replace

import numpy as np

from .history import StateFrame
from .scenarios import compare_recordings, get_scenario, make_jupiter_comparison

RING_KEY = 'sun-jupiter-ring'


def ring_setup():
    """Independent Sun, Jupiter, and evenly spaced 3.2 AU ring."""
    setup = get_scenario(RING_KEY)
    provenance = dict(setup.provenance)
    provenance['starting_template'] = RING_KEY
    provenance['collision_assumption'] = (
        'none: no merge or bounce. A close encounter halts and keeps the last valid state.')
    provenance['model_assumption'] = (
        'Unsoftened planar Newtonian gravity. Test particles feel massive bodies and attract nothing. '
        'No scripted pull toward L4 or L5.')
    return replace(setup, name='Sun–Jupiter ring', provenance=provenance)


def is_jupiter_experiment(setup):
    if setup is None:
        return False
    origin = setup.preset_id or setup.provenance.get('origin_preset')
    return origin == RING_KEY or setup.provenance.get('comparison_role') in ('with-jupiter', 'without-jupiter')


def assumptions_text(setup):
    """Inspectable population, model, and collision assumptions for this experiment."""
    if not is_jupiter_experiment(setup):
        return ''
    lines = []
    population = setup.population
    if population is not None:
        settings = population.settings
        lines.append(
            f'Population {population.id} about {population.reference_id}: {settings.kind}, '
            f'count {settings.count}, radius {settings.inner:g}–{settings.outer:g} AU, '
            f'angles {settings.angles}, seed {settings.seed}.')
        lines.append('Rock velocities start as prograde circular two-body guesses about the Sun.')
    else:
        lines.append('Rocks keep the resolved vectors from the paired setup. The ring was not regenerated or recentered.')
    if setup.provenance.get('comparison_role') == 'without-jupiter':
        lines.append('Jupiter is omitted, so its gravity is absent. Nothing is substituted for it.')
    lines.append(
        f'Collision mode {setup.collision_mode}: no merge or bounce. '
        'A close encounter halts and keeps the last valid state.')
    lines.append(
        'Forces are the declared unsoftened Newtonian model. '
        'Test particles do not attract the Sun, Jupiter, or each other.')
    if setup.tides.enabled and setup.collision_mode == 'none':
        lines.append(
            f'Tidal dissipation is on ({setup.tides.version}). Close passes of massive bodies can lose orbital energy. '
            'This ring is not itself a tidal-capture result.')
    elif setup.tides.enabled:
        lines.append('Tidal dissipation is stored and is not applied inside the contact solver.')
    else:
        lines.append('Tidal dissipation is off.')
    lines.append('This fixture can show perturbation. It does not promise Trojan occupancy, stability, or capture.')
    lines.append(f'Fixed step {setup.dt:.6g} yr. Halts and cancelled runs stay labeled and are not treated as complete.')
    return '\n'.join(lines)


def matched_control(setup):
    """Resolved no-Jupiter setup with the same rock vectors, dt, and collision mode."""
    _treatment, control = make_jupiter_comparison(setup)
    return control


def comparison_readout(recordings):
    """Latest matched pair, at their common recorded time only."""
    recordings = tuple(recordings or ())
    if len(recordings) < 2:
        return ''
    message = ''
    for newer in range(len(recordings)-1, 0, -1):
        for older in range(newer):
            try:
                return compare_recordings(recordings[older], recordings[newer]).description
            except ValueError as exc:
                message = str(exc)
    return message


def rock_report(setup, prepared, selected_id, scene):
    """Jupiter-relative state and the selected rock's displayed history span."""
    if not is_jupiter_experiment(setup) or not selected_id or selected_id not in getattr(prepared, 'body_ids', ()):
        return ''
    index = prepared.body_ids.index(selected_id)
    if prepared.body_specs[index].kind != 'particle':
        return ''
    trail = scene.trail_info.get_text() if scene is not None else ''
    try:
        from .frames import reference_id
        jupiter = reference_id(StateFrame.capture(prepared), 'jupiter')
    except ValueError:
        return 'Jupiter is absent, so this rock has no Jupiter separation.\n'+trail
    jupiter_index = prepared.body_ids.index(jupiter)
    relative_v = prepared.system.v[index]-prepared.system.v[jupiter_index]
    separation = np.linalg.norm(prepared.system.r[index]-prepared.system.r[jupiter_index])
    return (
        f'Jupiter separation: {separation:.6g} AU\n'
        f'Jupiter-relative physical inertial v ({relative_v[0]:.5g}, {relative_v[1]:.5g}) AU/yr\n'
        f'{trail}\n'
        'Proximity or this displayed path is not permanent capture.')
