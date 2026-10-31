"""Preset factories return new immutable setups; editors never own the registry."""
from dataclasses import dataclass, replace
from types import MappingProxyType
import numpy as np

from .initialization import legacy_setup, build_system
from .model import InitialBody
from .populations import Population
from .frames import reference_id

COMPATIBILITY = 'sun-earth-jupiter'
JUPITER_RING = 'sun-jupiter-ring'
DESCRIPTIONS = MappingProxyType({COMPATIBILITY:
    'Sun, Earth and Jupiter with prograde circular two-body velocity guesses, '
    'separated initial angles and a barycentric shift. Illustrative, not an ephemeris or stability guarantee.',
    JUPITER_RING: 'Sun / Jupiter / 32 mass-zero test particles at 3.2 AU, evenly spaced, seed 7. '
    'Demonstrates gravitational perturbation; arbitrary starting conditions do not guarantee Trojan occupancy or capture.'})


def get_scenario(key):
    if key not in DESCRIPTIONS:
        raise ValueError(f'Unknown scenario: {key}.')
    setup = (legacy_setup(planets=('jupiter',), population=Population(kind='ring', inner=3.2, outer=3.2,
             count=32, angles='even', seed=7)) if key == JUPITER_RING else legacy_setup())
    return replace(setup, name='Sun / Jupiter / 3.2 AU ring (32, seed 7)' if key == JUPITER_RING else setup.name, preset_id=key,
                   provenance={'origin_preset': key, 'description': DESCRIPTIONS[key]})


def make_jupiter_comparison(setup):
    """Remove only Jupiter from an already resolved state; never regenerate/recenter."""
    from .persistence import setup_digest
    prepared = build_system(setup)
    jupiter = reference_id(prepared, 'jupiter')
    sun = reference_id(prepared, 'sun')
    source = setup_digest(setup)
    metadata = dict(setup.provenance, comparison_source=source, comparison_sun=sun)
    with_jupiter = replace(setup, provenance=dict(metadata, comparison_role='with-jupiter'))
    authored = {body.id: body for body in setup.bodies}
    if any(body.spec.kind == 'moon' and body.reference_id == jupiter for body in setup.bodies):
        raise ValueError('Cannot remove Jupiter while it is a moon parent; remove or reauthor that moon first.')
    bodies = tuple(authored[key] if spec.kind == 'moon' else InitialBody(key, spec, tuple(r), tuple(v))
                   for key, spec, r, v in zip(prepared.body_ids, prepared.body_specs, prepared.system.r, prepared.system.v)
                   if key != jupiter)
    control = replace(setup, name=setup.name+' / no Jupiter control', preset_id=None, bodies=bodies,
                      population=None, provenance=dict(metadata, comparison_role='without-jupiter',
                      description='Matched explicit world vectors; Jupiter removed. No recentering or regeneration.'))
    return with_jupiter, control


@dataclass(frozen=True)
class ComparisonSummary:
    start_time: float
    end_time: float
    particle_ids: tuple
    max_displacement: float
    description: str


def compare_recordings(first, second):
    """Compare Sun-relative rocks only at shared physical sample times, by ID."""
    setups = first.source_setup, second.source_setup
    if any(setup is None for setup in setups):
        raise ValueError('Comparison needs matched source setups.')
    metadata = [dict(setup.provenance) for setup in setups]
    # The shared UI can calculate the original before asking for its control.
    # Recognize its exact digest without rewriting immutable recording provenance.
    for i, run in enumerate((first, second)):
        other = metadata[1-i]
        if not metadata[i].get('comparison_source') and run.setup_digest == other.get('comparison_source'):
            metadata[i].update(comparison_source=run.setup_digest, comparison_role='with-jupiter',
                               comparison_sun=other['comparison_sun'])
    if (not metadata[0].get('comparison_source') or metadata[0].get('comparison_source') != metadata[1].get('comparison_source')
            or {m.get('comparison_role') for m in metadata} != {'with-jupiter', 'without-jupiter'}
            or setups[0].dt != setups[1].dt):
        raise ValueError('Results are not a matched Jupiter/control pair with the same fixed dt.')
    initial = [build_system(setup) for setup in setups]
    perturbation = next(i for i, m in enumerate(metadata) if m['comparison_role'] == 'with-jupiter')
    original, control = initial[perturbation], initial[1-perturbation]
    jupiter = reference_id(original, 'jupiter')
    expected = set(original.body_ids)-{jupiter}
    if (set(control.body_ids) != expected or setups[0].collision_mode != setups[1].collision_mode
            or setups[0].collision_policy != setups[1].collision_policy):
        raise ValueError('Matched initial bodies/model settings were changed.')
    for key in control.body_ids:
        i, j = original.body_ids.index(key), control.body_ids.index(key)
        if (original.body_specs[i] != control.body_specs[j]
                or not np.array_equal(original.system.r[i], control.system.r[j])
                or not np.array_equal(original.system.v[i], control.system.v[j])):
            raise ValueError('Matched initial physical vectors or body properties were changed.')
    a, b = ({frame.time: frame for frame in run.frames} for run in (first, second))
    common = sorted(a.keys() & b.keys())
    if not common:
        raise ValueError('Matched results have no common recorded physical time.')
    fa, fb = a[common[-1]], b[common[-1]]
    sun = metadata[0]['comparison_sun']
    if sun not in fa.body_ids or sun not in fb.body_ids:
        raise ValueError('Sun reference is missing at the common sample.')
    ids = tuple(key for key, spec in zip(fa.body_ids, fa.body_specs) if spec.kind == 'particle' and key in fb.body_ids)
    if not ids:
        raise ValueError('No common particle identities at the common sample.')
    offsets = []
    for key in ids:
        ra = fa.r[fa.body_ids.index(key)]-fa.r[fa.body_ids.index(sun)]
        rb = fb.r[fb.body_ids.index(key)]-fb.r[fb.body_ids.index(sun)]
        offsets.append(np.linalg.norm(ra-rb))
    maximum = float(max(offsets))
    incomplete = first.status != 'complete' or second.status != 'complete' or first.actual_end_time != second.actual_end_time
    description = (f'Matched common sampled interval {common[0]:.8g}–{common[-1]:.8g} yr; '
                   f'compared at {common[-1]:.8g} yr: max Sun-relative rock difference {maximum:.6g} AU.\n'
                   f'{first.source_name}: {first.status}, actual {first.actual_end_time:.8g} yr; '
                   f'{second.source_name}: {second.status}, actual {second.actual_end_time:.8g} yr.'
                   + (' Incomplete / unequal completion; only the common sampled interval is compared.' if incomplete else ''))
    return ComparisonSummary(common[0], common[-1], ids, maximum, description)
