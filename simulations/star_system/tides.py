"""Optional energy loss on resolved close pericenters.

``close-pass-energy-dissipation-v1`` is off unless an experiment stores an
enabled :class:`~simulations.star_system.model.TidalPolicy`. The fixed-step
Yoshida advance is unchanged when the policy is off. The kick is not part of
the contact solver, and it is not gas drag, fragmentation, or a 3D integrator.
"""
import math
from dataclasses import dataclass, replace

import numpy as np

from .core import G
from .data import CATALOG
from .initialization import build_system
from .model import TIDAL_MODEL_ID, TIDAL_REACH, TIDAL_STRENGTH, InitialBody, InitialSetup, TidalPolicy

KIND = 'close-pass-dissipation'
FLYBY_ECCENTRICITY = 1.01
BOUND_ECCENTRICITY = 0.65
# Pericenter in units of the sum of the catalog radii. Above 1 stays outside contact.
PERIAPSIS_CLEARANCE = 1.25
APPROACH_FACTOR = 2.5
_MAX_STEPS = 250000

MODEL_DISCLOSURE = (
    f'{TIDAL_MODEL_ID} removes orbital energy at a resolved pericenter of two positive-mass bodies '
    'when at least one has a positive radius and the estimated pericenter is within the reach, '
    'a multiple of the larger radius. '
    'Massless tracers are unchanged. '
    'The loss is strength * G * (m_j^2 * R_i^5 + m_i^2 * R_j^5) / q^6, '
    'with q floored at the larger radius. '
    'The relative velocity is scaled about the center of mass, so linear momentum is kept and the removed energy '
    'cannot exceed the relative kinetic energy. '
    'The change in orbital angular momentum is not stored as spin. '
    'The kick is applied to the velocity at the end of the fixed step that contains the inbound-to-outbound crossing, '
    'using a pericenter estimated by interpolating that radial-velocity sign change. '
    f'Strength {TIDAL_STRENGTH:g} is a dimensionless stand-in, not a measured k2/Q. '
    f'Reach {TIDAL_REACH:g} is in units of the larger radius. '
    'The choice is off unless the experiment stores an enabled tide policy. '
    'Outcome claims that a pass became bound or more circular are made only when that policy is on.'
)

INTERACTION = (
    'This kick is not a 3D or adaptive-step integrator and it does not call REBOUND (#70). '
    'A fixed step that skips the pericenter, or that estimates it poorly, understates the energy loss, '
    'so tides are not the dominant capture path for an unresolved encounter. '
    'It does not change collision rules (#69): perfect merge and the binary-star merger still use the contact solver, '
    'and a stored tide choice is not applied on that path. '
    'Fragmentation, hit-and-run, and merger remnants are not tidal outcomes. '
    'Where an impact would dominate, this model is not the capture path. '
    'It is not gas drag (#71). There is no disk, and a pass outside the tidal reach loses no energy here. '
    'Gas drag, fragmenting collisions, and a close-encounter integrator can outweigh this kick; '
    'none of those are implied by a tidal result. '
    'A longer encounter in this model is still planar, spin-free, and reversible by a later scattering. '
    'It is not a permanent capture.'
)


@dataclass(frozen=True)
class TideEvent:
    time: float
    kind: str
    body_ids: tuple[str, str]
    pericenter: float
    energy_lost: float
    unmodeled_angular: float
    message: str


@dataclass(frozen=True)
class PassSummary:
    steps: int
    time: float
    pericenters: int
    specific_energy: float
    eccentricity: float
    min_separation: float
    energy_lost: float
    tide_events: int


@dataclass(frozen=True)
class ClosePassComparison:
    flyby_off: PassSummary
    flyby_on: PassSummary
    eccentric_off: PassSummary
    eccentric_on: PassSummary
    disclosure: str


def pair_energy_loss(mass_i, radius_i, mass_j, radius_j, pericenter, strength):
    """Positive orbital energy removed from one pair at one pericenter.

    ``strength * G * (m_j**2 * R_i**5 + m_i**2 * R_j**5) / q**6``.
    ``q`` is the pericenter, floored at the larger radius so an overlap stays finite.
    A missing radius contributes nothing. This is not a collision outcome.
    """
    if min(mass_i, mass_j) <= 0 or pericenter <= 0 or strength <= 0:
        return 0.0
    radius_i = 0.0 if not radius_i else float(radius_i)
    radius_j = 0.0 if not radius_j else float(radius_j)
    if radius_i <= 0 and radius_j <= 0:
        return 0.0
    q = max(float(pericenter), radius_i, radius_j)
    raw = strength * G * (mass_j**2 * radius_i**5 + mass_i**2 * radius_j**5) / q**6
    if not np.isfinite(raw) or raw <= 0:
        return 0.0
    return float(raw)


def _crossing_distance(r0, v0, r1, v1):
    """Estimated pericenter when radial velocity goes from inbound to outbound.

    Positions are interpolated across the fixed step. A flat crossing, such as
    noise on a circular orbit, is not a pericenter.
    """
    vr0 = float(np.dot(r0, v0))
    vr1 = float(np.dot(r1, v1))
    if not vr0 < 0.0 <= vr1:
        return None
    span = vr0 - vr1
    if span == 0.0 or not np.isfinite(span):
        return None
    fraction = vr0 / span
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        return None
    distance = float(np.linalg.norm((1.0 - fraction) * r0 + fraction * r1))
    if not np.isfinite(distance) or distance <= 0.0:
        return None
    nearer_end = float(min(np.linalg.norm(r0), np.linalg.norm(r1)))
    if not distance < nearer_end:
        return None
    return distance


def _remove_energy(positions, velocities, masses, i, j, loss):
    """Scale the pair's relative velocity about its center of mass.

    Returns the positive energy actually removed and the change in the pair's
    orbital angular momentum. That change is not written onto either body as spin.
    """
    mi, mj = float(masses[i]), float(masses[j])
    total = mi + mj
    mu = mi * mj / total
    relative = velocities[i] - velocities[j]
    separation = positions[i] - positions[j]
    speed2 = float(np.dot(relative, relative))
    if speed2 <= 0.0 or not np.isfinite(speed2) or mu <= 0.0:
        return 0.0, 0.0
    kinetic = 0.5 * mu * speed2
    removed = kinetic if not np.isfinite(loss) else min(float(loss), kinetic)
    if removed <= 0.0:
        return 0.0, 0.0
    scale = math.sqrt(1.0 - removed / kinetic)
    before = mu * (separation[0] * relative[1] - separation[1] * relative[0])
    updated = relative * scale
    center = (mi * velocities[i] + mj * velocities[j]) / total
    velocities[i] = center + (mj / total) * updated
    velocities[j] = center - (mi / total) * updated
    after = mu * (separation[0] * updated[1] - separation[1] * updated[0])
    return removed, float(before - after)


def dissipate_close_passes(prepared, start_r, start_v):
    """Remove close-pass energy after one committed Yoshida step.

    ``start_r`` and ``start_v`` are the state at the beginning of that step.
    Returns the same prepared system when the policy is off or no pair qualifies.
    """
    policy = prepared.setup.tides if prepared.setup is not None else None
    if policy is None or not policy.enabled:
        return prepared
    system = prepared.system
    velocities = np.array(system.v, copy=True)
    events = []
    pairs = []
    for i in range(len(prepared.body_ids)):
        for j in range(i):
            if system.masses[i] <= 0 or system.masses[j] <= 0:
                continue
            distance = _crossing_distance(
                start_r[j] - start_r[i], start_v[j] - start_v[i],
                system.r[j] - system.r[i], system.v[j] - system.v[i])
            if distance is None:
                continue
            radius_i = prepared.body_specs[i].radius or 0.0
            radius_j = prepared.body_specs[j].radius or 0.0
            larger = max(radius_i, radius_j)
            if larger <= 0.0 or distance > policy.reach * larger:
                continue
            pairs.append((tuple(sorted((prepared.body_ids[j], prepared.body_ids[i]))), j, i, distance,
                          radius_j, radius_i))
    for pair, j, i, distance, radius_j, radius_i in sorted(pairs):
        loss = pair_energy_loss(system.masses[j], radius_j, system.masses[i], radius_i, distance, policy.strength)
        removed, unmodeled = _remove_energy(system.r, velocities, system.masses, j, i, loss)
        if removed <= 0.0 or not np.isfinite(removed):
            continue
        events.append(TideEvent(
            system.time, KIND, pair, distance, removed, unmodeled,
            f'{policy.version}: {pair[0]} and {pair[1]} lost {removed:.6g} '
            f'at estimated pericenter {distance:.6g} AU. '
            f'Linear momentum kept. Orbital angular-momentum change {unmodeled:.6g} is not stored as spin.'))
    if not events:
        return prepared
    system.v[:] = velocities
    return replace(prepared, events=prepared.events + tuple(events))


def _pericenter():
    return PERIAPSIS_CLEARANCE * (CATALOG['jupiter'].radius + CATALOG['earth'].radius)


def contact_separation():
    """Sum of the catalog Jupiter and Earth radii used by the documented pass."""
    return CATALOG['jupiter'].radius + CATALOG['earth'].radius


def _orbit_state(eccentricity):
    q = _pericenter()
    mu = G * (CATALOG['jupiter'].mass + CATALOG['earth'].mass)
    axis = q / (1.0 - eccentricity)
    energy = -mu / (2.0 * axis)
    angular = math.sqrt(mu * q * (1.0 + eccentricity))
    if eccentricity < 1.0:
        distance = 0.5 * (q + axis * (1.0 + eccentricity))
    else:
        distance = APPROACH_FACTOR * q
    radial2 = 2.0 * (energy + mu / distance) - (angular * angular) / (distance * distance)
    if radial2 <= 0.0 or not math.isfinite(radial2):
        raise ValueError('Close-pass start is not inbound on this orbit.')
    peri_speed = math.sqrt(mu * (1.0 + eccentricity) / q)
    limit = min(math.sqrt(q**3 / mu), q / peri_speed) / 10.0
    # A fifth of the pericenter resolution limit. The guard still applies every step.
    dt = 0.2 * limit
    return q, dt, (float(distance), 0.0), (-math.sqrt(radial2), angular / distance)


def jupiter_earth_pass(enabled, eccentricity=FLYBY_ECCENTRICITY):
    """Catalog Jupiter and Earth on one planar orbit, tides on or off.

    The Sun is omitted so the comparison isolates this energy sink. The same
    vectors are used for both choices. Enabling tides uses the default strength
    and reach.
    """
    if type(enabled) is not bool:
        raise ValueError('Tidal dissipation must be explicitly on or off.')
    q, dt, position, velocity = _orbit_state(eccentricity)
    role = 'with-tides' if enabled else 'without-tides'
    provenance = {
        'origin_preset': 'jupiter-earth-close-pass',
        'description': (
            'Catalog Jupiter and catalog Earth on a slow planar approach. '
            'The same vectors are used with tidal dissipation on and off. The Sun is omitted.'),
        'observe': (
            'Compare this run with the same setup after changing only the tide policy. '
            'A bound or more circular outcome is a claim only for the run that selects the model.'),
        'tidal_model': TIDAL_MODEL_ID if enabled else 'off',
        'comparison_role': role,
        'eccentricity': float(eccentricity),
        'pericenter_au': float(q),
        'omitted': 'Sun, gas drag, fragmentation, 3D, adaptive timesteps, spin, relativity, stellar structure',
    }
    return InitialSetup(
        'Jupiter–Earth close pass',
        (
            InitialBody('jupiter', CATALOG['jupiter'], (0.0, 0.0), (0.0, 0.0)),
            InitialBody('earth', CATALOG['earth'], position, (float(velocity[0]), float(velocity[1]))),
        ),
        float(dt), provenance=provenance, tides=TidalPolicy(enabled=True) if enabled else TidalPolicy())


def _specific(system, primary, secondary):
    separation = system.r[secondary] - system.r[primary]
    velocity = system.v[secondary] - system.v[primary]
    mu = G * (system.masses[primary] + system.masses[secondary])
    distance = float(np.linalg.norm(separation))
    energy = float(0.5 * np.dot(velocity, velocity) - mu / distance)
    angular = float(separation[0] * velocity[1] - separation[1] * velocity[0])
    eccentricity = math.sqrt(max(0.0, 1.0 + 2.0 * energy * angular * angular / (mu * mu)))
    radial = float(np.dot(separation, velocity))
    return energy, eccentricity, distance, radial


def _lost(prepared):
    total, count = 0.0, 0
    for event in prepared.events:
        if isinstance(event, TideEvent):
            total += event.energy_lost
            count += 1
    return total, count


def _advance(prepared):
    from .collisions import advance_prepared
    return advance_prepared(prepared).prepared


def integrate_pass(setup, *, steps=None, crossings=None, max_steps=_MAX_STEPS):
    """Integrate a Jupiter–Earth pass and summarize the relative orbit."""
    prepared = build_system(setup)
    primary = prepared.body_ids.index('jupiter')
    secondary = prepared.body_ids.index('earth')
    _energy, _ecc, distance, previous = _specific(prepared.system, primary, secondary)
    seen = 0
    minimum = distance
    taken = 0
    limit = max_steps if steps is None else steps
    while taken < limit:
        try:
            prepared = _advance(prepared)
        except ValueError as exc:
            raise ValueError(f'Close pass halted at step {taken}: {exc}') from exc
        taken += 1
        energy, eccentricity, distance, radial = _specific(prepared.system, primary, secondary)
        minimum = min(minimum, distance)
        if previous < 0.0 <= radial:
            seen += 1
            if crossings is not None and seen >= crossings:
                break
        previous = radial
    if crossings is not None and seen < crossings:
        raise ValueError(f'Close pass did not reach {crossings} pericenters within {limit} steps.')
    energy, eccentricity, distance, _radial = _specific(prepared.system, primary, secondary)
    lost, tide_events = _lost(prepared)
    return PassSummary(taken, float(prepared.system.time), seen, energy, eccentricity, minimum, lost, tide_events)


def _state_word(summary):
    return 'bound' if summary.specific_energy < 0.0 else 'unbound'


def compare_jupiter_earth_passes():
    """Same Jupiter–Earth vectors with the tide model off and on.

    The flyby is integrated until the tide run has returned for a second
    pericenter. The tide-free run uses that same step count. The eccentric
    pair stops at the first pericenter, after the kick if the model is on.
    """
    flyby_on = integrate_pass(jupiter_earth_pass(True, FLYBY_ECCENTRICITY), crossings=2)
    flyby_off = integrate_pass(jupiter_earth_pass(False, FLYBY_ECCENTRICITY), steps=flyby_on.steps)
    eccentric_on = integrate_pass(jupiter_earth_pass(True, BOUND_ECCENTRICITY), crossings=1)
    eccentric_off = integrate_pass(jupiter_earth_pass(False, BOUND_ECCENTRICITY), steps=eccentric_on.steps)
    disclosure = '\n'.join((
        MODEL_DISCLOSURE,
        (
            f'Flyby without tides, {flyby_off.steps} steps to {flyby_off.time:.6g} yr: '
            f'{flyby_off.pericenters} pericenter, specific energy {flyby_off.specific_energy:.6g}, '
            f'{_state_word(flyby_off)}, eccentricity {flyby_off.eccentricity:.6g}.'
        ),
        (
            f'Flyby with tides, same {flyby_on.steps} steps: '
            f'{flyby_on.pericenters} pericenters, specific energy {flyby_on.specific_energy:.6g}, '
            f'{_state_word(flyby_on)}, eccentricity {flyby_on.eccentricity:.6g}, '
            f'energy lost {flyby_on.energy_lost:.6g}.'
        ),
        (
            f'Eccentric pass without tides: eccentricity {eccentric_off.eccentricity:.6g}, '
            f'specific energy {eccentric_off.specific_energy:.6g}.'
        ),
        (
            f'Eccentric pass with tides, same {eccentric_on.steps} steps and one pericenter: '
            f'eccentricity {eccentric_on.eccentricity:.6g}, '
            f'specific energy {eccentric_on.specific_energy:.6g}, '
            f'energy lost {eccentric_on.energy_lost:.6g}.'
        ),
        'Minimum sampled separations stay outside the sum of the catalog radii. The Sun is omitted.',
        INTERACTION,
    ))
    return ClosePassComparison(flyby_off, flyby_on, eccentric_off, eccentric_on, disclosure)
