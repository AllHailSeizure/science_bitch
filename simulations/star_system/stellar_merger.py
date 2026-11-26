"""One supported stellar merger: two catalog-Sun stars released from rest.

Only ``collision_mode='binary_star_merger'`` with model id
``solar-type-head-on-contact-v1`` uses this response. Other stellar contact,
including ``perfect_merge``, still halts. This is not a rule for every pair
of touching stars.
"""
import hashlib
import math

import numpy as np

from .collisions import CollisionEvent
from .core import G, System
from .data import CATALOG
from .initialization import PreparedSystem
from .model import BodySpec, InitialBody, InitialSetup

MODE = 'binary_star_merger'
MODEL_ID = 'solar-type-head-on-contact-v1'
SCENARIO_KEY = 'binary-star-merger'
TITLE = 'Binary-star merger'
# Initial center-to-center separation, in units of the sum of the two radii.
SEPARATION_FACTOR = 10.
# Published run length, in units of the analytic head-on contact time.
OBSERVATION_FACTOR = 1.5
# Homology-style main-sequence estimate, normalized to the catalog Sun.
# Not a stellar-structure integration.
REMNANT_RADIUS_EXPONENT = 0.8
DESCRIPTION = (
    'Two solar-type main-sequence stars, each with the catalog Sun\'s mass and radius, '
    'released from rest ten contact-radii apart. Newtonian gravity brings their surfaces '
    'together, and this scenario replaces them with one remnant.')
OBSERVE = (
    'Calculate through the published window (1.5 times the analytic contact time). '
    'A modeled merger names both stars, the contact time, and one remnant. '
    'Any other contact in this scenario halts as unsupported. '
    'A solver limit halts numerically and keeps the last accepted state. '
    'Mass is the sum of the parents. Radius is an M^0.8 estimate. '
    'Temperature, composition, mass loss, spin, tides, and stellar evolution are omitted.')
UNSUPPORTED = (
    'Unsupported contact: outside the supported binary-star merger scenario (#14); '
    'state preserved.')
_REMNANT_PROVENANCE = (
    'Estimated solar-type merger remnant\n'
    'mass = sum of the two parents (mass loss omitted)\n'
    'radius = R_sun * (M/M_sun)^0.8\n'
    'temperature, composition, spin, tides, and stellar evolution omitted\n'
    'illustrative color only')


def estimated_remnant_radius(mass):
    """Main-sequence radius estimate normalized to the catalog Sun."""
    sun = CATALOG['sun']
    return float(sun.radius * (mass / sun.mass) ** REMNANT_RADIUS_EXPONENT)


def head_on_rest_contact_time(separation, contact_separation, mass_1, mass_2):
    """Time for two point masses, released from rest, to reach ``contact_separation``.

    With total mass M and initial separation r0, the radial two-body fall is
    t = sqrt(r0^3 / (8 G M)) * (η + sin η), cos η = 2 r / r0 - 1.
    As r -> 0, η -> π, which is the textbook fall time to the center.
    """
    if not (0 < contact_separation < separation):
        raise ValueError('Contact separation must lie inside the initial separation.')
    if mass_1 <= 0 or mass_2 <= 0:
        raise ValueError('Head-on contact time requires two positive masses.')
    eta = math.acos(2 * contact_separation / separation - 1)
    gm = G * (mass_1 + mass_2)
    return math.sqrt(separation ** 3 / (8 * gm)) * (eta + math.sin(eta))


def _pair_geometry(setup):
    if len(setup.bodies) != 2:
        raise ValueError('The binary-star merger scenario has exactly two stars.')
    radii = []
    masses = []
    positions = []
    for body in setup.bodies:
        if body.spec.radius is None or body.spec.mass is None:
            raise ValueError('Both stars need a known mass and radius.')
        radii.append(body.spec.radius)
        masses.append(body.spec.mass)
        positions.append(np.asarray(body.r, dtype=float))
    # Reference offsets are resolved by build_system. This helper is for the
    # published world-frame initial conditions, which have no reference.
    if any(body.reference_id is not None for body in setup.bodies):
        raise ValueError('Published merger stars are placed in the world frame.')
    separation = float(np.linalg.norm(positions[0] - positions[1]))
    return separation, radii[0] + radii[1], masses[0], masses[1]


def analytic_contact_time(setup):
    """Contact time of the published from-rest initial conditions."""
    for body in setup.bodies:
        if body.v != (0., 0.):
            raise ValueError('The analytic contact time is for release from rest.')
    separation, contact, mass_1, mass_2 = _pair_geometry(setup)
    return head_on_rest_contact_time(separation, contact, mass_1, mass_2)


def observation_years(setup):
    """How long the published experiment is meant to be watched."""
    return analytic_contact_time(setup) * OBSERVATION_FACTOR


def binary_star_merger_setup():
    """Immutable starting setup for the one supported stellar merger."""
    sun = CATALOG['sun']
    contact = sun.radius + sun.radius
    separation = SEPARATION_FACTOR * contact
    half = separation / 2
    # Half the initial gravity/crossing resolution limit. The contact solver
    # may subdivide further as the stars fall; this is the saved macro step.
    dt = 0.5 * math.sqrt(separation ** 3 / (G * (sun.mass + sun.mass))) / 10
    stars = (
        InitialBody('primary', _named(sun, 'Solar-type primary'), (-half, 0.), (0., 0.)),
        InitialBody('secondary', _named(sun, 'Solar-type secondary'), (half, 0.), (0., 0.)),
    )
    provenance = {
        'origin_preset': SCENARIO_KEY,
        'starting_template': SCENARIO_KEY,
        'description': DESCRIPTION,
        'observe': OBSERVE,
        'merger_model': MODEL_ID,
        'stellar_types': 'two catalog-Sun main-sequence stars',
        'mechanism': 'head-on Newtonian fall from rest until the spherical surfaces touch',
        'separation_over_contact': SEPARATION_FACTOR,
        'remnant_radius_exponent': REMNANT_RADIUS_EXPONENT,
        'omitted': 'temperature, composition, mass loss, spin, tides, mass transfer, radiation, hydrodynamics, stellar evolution',
    }
    return InitialSetup(TITLE, stars, dt, SCENARIO_KEY, collision_mode=MODE, provenance=provenance)


def _named(spec, name):
    return BodySpec(name, spec.kind, spec.mass, spec.radius, spec.temperature, spec.color,
                    spec.catalog_key, spec.provenance, spec.sources)


def outcome_text(prepared):
    """One line for a state that already contains the remnant; empty otherwise."""
    events = getattr(prepared, 'events', ())
    body_ids = getattr(prepared, 'body_ids', ())
    for event in reversed(tuple(events)):
        if getattr(event, 'kind', None) == 'stellar_merger' and event.remnant_id in body_ids:
            return event.message
    return ''


def respond_if_stellar_scenario(prepared, contacts):
    """Apply the supported remnant, or refuse every other contact in this mode.

    Returns None when this mode is not selected, so nonstellar merging is unchanged.
    """
    setup = prepared.setup
    if setup is None or setup.collision_mode != MODE:
        return None
    if _supported(prepared, contacts):
        return _merge(prepared, contacts[0])
    raise ValueError(UNSUPPORTED)


def _supported(prepared, contacts):
    setup = prepared.setup
    if setup.provenance.get('merger_model') != MODEL_ID:
        return False
    if len(contacts) != 1 or len(prepared.body_ids) != 2:
        return False
    if {contacts[0][1], contacts[0][2]} != set(prepared.body_ids):
        return False
    sun = CATALOG['sun']
    for spec in prepared.body_specs:
        if spec.kind != 'star' or spec.mass != sun.mass or spec.radius != sun.radius:
            return False
    return True


def _remnant_id(prepared, ids):
    digest = hashlib.sha256((MODEL_ID + '\0' + '\0'.join(ids)).encode()).hexdigest()[:24]
    stem = 'stellar-merger:' + digest
    used = set(prepared.body_ids)
    for event in prepared.events:
        if isinstance(event, CollisionEvent):
            used.update(event.parent_ids)
            used.update(event.ancestry)
            if event.remnant_id:
                used.add(event.remnant_id)
    key, ordinal = stem, 0
    while key in used:
        ordinal += 1
        key = f'{stem}:{ordinal}'
    return key


def _merge(prepared, contact):
    ids = tuple(sorted((contact[1], contact[2])))
    indices = [prepared.body_ids.index(key) for key in ids]
    system = prepared.system
    masses = system.masses[indices]
    total = float(np.sum(masses))
    position = np.sum(masses[:, None] * system.r[indices], axis=0) / total
    velocity = np.sum(masses[:, None] * system.v[indices], axis=0) / total
    lost = float(.5 * np.sum(masses[:, None] * (system.v[indices] - velocity) ** 2))
    offsets = system.r[indices] - position
    relative = system.v[indices] - velocity
    spin = float(np.sum(masses * (offsets[:, 0] * relative[:, 1] - offsets[:, 1] * relative[:, 0])))
    radius = estimated_remnant_radius(total)
    key = _remnant_id(prepared, ids)
    names = [prepared.body_specs[prepared.body_ids.index(body_id)].name for body_id in ids]
    message = (
        f'Modeled stellar merger at {system.time:.8g} yr: '
        f'{names[0]} [{ids[0]}] and {names[1]} [{ids[1]}] -> {key}. '
        f'Outcome: one star; mass is the sum; '
        f'radius is R_sun*(M/M_sun)^{REMNANT_RADIUS_EXPONENT:g}; '
        f'temperature, composition, mass loss, and spin are omitted. '
        f'Scenario {MODEL_ID}.')
    spec = BodySpec('Solar-type merger remnant', 'star', total, radius, color='#fff0d9',
                    provenance=_REMNANT_PROVENANCE)
    keep = [i for i, body_id in enumerate(prepared.body_ids) if body_id not in ids]
    references = tuple(prepared.reference_ids[i] for i in keep) + (None,)
    event = CollisionEvent(system.time, 'stellar_merger', ids, key, ids, lost, spin, message)
    if keep:
        masses_kept = np.append(system.masses[keep], total)
        positions = np.vstack((system.r[keep], position))
        velocities = np.vstack((system.v[keep], velocity))
    else:
        masses_kept = np.array([total])
        positions = np.array([position])
        velocities = np.array([velocity])
    merged = PreparedSystem(
        System(masses_kept, positions, velocities, system.dt, system.time),
        tuple(prepared.body_ids[i] for i in keep) + (key,),
        tuple(prepared.body_specs[i] for i in keep) + (spec,),
        prepared.setup, prepared.events + (event,), prepared.segment, references)
    return merged
