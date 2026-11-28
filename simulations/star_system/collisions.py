"""Contact detection and selectable collision rules.

No rendering coordinates enter this module. The swept path is each Verlet
drift's quadratic, not an assertion of exact continuous Newtonian trajectories.

A collision logs its time, participants, rule, and outcome, then the macro step
finishes. ``stop_on_contact`` is the exception: that choice ends the run at the
contact and does not build a remnant. Numerical failure still raises and leaves
the caller unchanged. Stellar contact is not a nonstellar merge: unless the
binary-star scenario handles it, or stop on contact was chosen, the contact is
logged as unmodeled and the bodies continue.
"""
from dataclasses import dataclass, replace
import hashlib
import math
from collections.abc import Mapping

import numpy as np

from .core import G
from .model import BodySpec
from .initialization import PreparedSystem
from .history import StateFrame

# Speed-and-angle map. b = 0 is head-on and b = 1 is a tangent graze.
# Merge at or below mutual escape. Above escape, b >= 0.5 grazes and b < 0.5 fragments.
GRAZE_IMPACT_PARAMETER = 0.5
FRAGMENT_KINETIC_RETAINED = 0.25
# A flyby inside this multiple of the contact radius, without contact, is a close encounter.
ENCOUNTER_FACTOR = 3.
ORBITAL_ECCENTRICITY_TOLERANCE = 0.02
ORBITAL_AXIS_RTOL = 0.05
_DYNAMIC_KINDS = ('merge', 'fragment', 'graze', 'unmodeled', 'absorption', 'stellar_merger', 'ignored', 'stop')


@dataclass(frozen=True)
class CollisionEvent:
    time: float
    kind: str
    parent_ids: tuple[str, ...]
    remnant_id: str | None
    ancestry: tuple[str, ...]
    lost_relative_kinetic: float
    unmodeled_spin: float
    message: str
    target_id: str | None = None
    rule: str = ''
    outcome: str = ''
    descendant_ids: tuple[str, ...] = ()
    lost_mass: float = 0.
    impact_speed: float = 0.
    impact_parameter: float = 0.
    approach_distance: float = 0.
    location: tuple[float, ...] | None = None


@dataclass(frozen=True)
class AdvanceResult:
    prepared: PreparedSystem
    event_frames: tuple[StateFrame, ...] = ()
    stopped: bool = False


def validate_collision_state(prepared):
    s = prepared.system
    if not np.isfinite(s.dt) or s.dt <= 0 or not np.isfinite(s.time):
        raise ValueError('Time and positive fixed dt must be finite.')
    s.resolution_limit()  # validates finite arrays without rejecting the macro dt
    for spec, mass in zip(prepared.body_specs, s.masses):
        if mass > 0 and spec.radius is None:
            raise ValueError(f'Physical collision mode requires a known radius for {spec.name}. Supply a radius in AU.')


def _record_ancestry(prior, remnant_ids, ancestry):
    recorded = tuple(ancestry)
    for key in remnant_ids:
        if isinstance(key, str):
            prior[key] = recorded


def _ancestors(prepared, ids):
    prior = {}
    historical = prepared.setup.provenance.get('collision_events', ()) if prepared.setup else ()
    if isinstance(historical, (tuple, list)):
        for event in historical:
            if not isinstance(event, Mapping):
                continue
            ancestry = event.get('ancestry', ())
            if not isinstance(ancestry, (tuple, list)) or not all(isinstance(key, str) for key in ancestry):
                continue
            kind = event.get('kind')
            if kind not in (None, 'merge', 'fragment', 'stellar_merger'):
                continue
            descendants = []
            if isinstance(event.get('remnant_id'), str):
                descendants.append(event['remnant_id'])
            extra = event.get('descendant_ids', ())
            if isinstance(extra, (tuple, list)):
                descendants.extend(key for key in extra if isinstance(key, str))
            _record_ancestry(prior, descendants, ancestry)
    for event in prepared.events:
        if isinstance(event, CollisionEvent) and event.kind in ('merge', 'fragment', 'stellar_merger'):
            descendants = [key for key in (event.remnant_id, *event.descendant_ids) if isinstance(key, str)]
            _record_ancestry(prior, descendants, event.ancestry)
    return tuple(sorted({ancestor for key in ids for ancestor in prior.get(key, (key,))}))


def _used_ids(prepared):
    used = set(prepared.body_ids)
    for event in prepared.events:
        if isinstance(event, CollisionEvent):
            used.update(event.parent_ids)
            used.update(event.ancestry)
            used.update(event.descendant_ids)
            if event.remnant_id is not None:
                used.add(event.remnant_id)
    # A captured epoch retains its earlier ancestry as provenance, not active
    # events. Respect those identities without interpreting arbitrary JSON.
    historical = prepared.setup.provenance.get('collision_events', ()) if prepared.setup else ()
    if isinstance(historical, (tuple, list)):
        for event in historical:
            if not isinstance(event, Mapping):
                continue
            if isinstance(event.get('remnant_id'), str):
                used.add(event['remnant_id'])
            for field in ('parent_ids', 'ancestry', 'descendant_ids'):
                values = event.get(field, ())
                if isinstance(values, (tuple, list)):
                    used.update(value for value in values if isinstance(value, str))
    return used


def _merge_group(prepared, ids):
    ids = tuple(sorted(ids))
    indices = [prepared.body_ids.index(key) for key in ids]
    s = prepared.system
    if any(prepared.body_specs[i].kind == 'star' for i in indices):
        raise ValueError('Unsupported star contact: stellar mergers are deferred to #14; state preserved.')
    if len(ids) < 2 or any(s.masses[i] <= 0 for i in indices):
        raise ValueError('A merge requires two or more positive-mass nonstellar bodies.')
    masses = s.masses[indices]
    total = float(np.sum(masses))
    r = np.sum(masses[:, None]*s.r[indices], axis=0)/total
    v = np.sum(masses[:, None]*s.v[indices], axis=0)/total
    radius = float(np.cbrt(sum(prepared.body_specs[i].radius**3 for i in indices)))
    lost = float(.5*np.sum(masses[:, None]*(s.v[indices]-v)**2))
    offsets, velocities = s.r[indices]-r, s.v[indices]-v
    if offsets.shape[1] == 2:
        spin = float(np.sum(masses*(offsets[:, 0]*velocities[:, 1]-offsets[:, 1]*velocities[:, 0])))
    else:
        # Magnitude of the residual angular momentum. Planar runs keep the signed z component above.
        spin = float(np.linalg.norm(np.sum(masses[:, None]*np.cross(offsets, velocities), axis=0)))
    ancestry = _ancestors(prepared, ids)
    key = 'merge:'+hashlib.sha256(('\0'.join(ids)).encode()).hexdigest()[:24]
    stem, ordinal = key, 0
    used_ids = _used_ids(prepared)
    while key in used_ids:
        ordinal += 1
        key = f'{stem}:{ordinal}'
    name = ' + '.join(prepared.body_specs[i].name for i in indices)
    rule = prepared.setup.collision_mode if prepared.setup is not None else 'perfect_merge'
    provenance = ('Computed perfect_merge\ntemperature and composition unknown.' if rule == 'perfect_merge'
                  else 'Computed speed_angle merge\nbelow mutual escape speed\ntemperature and composition unknown.')
    spec = BodySpec(name, 'planet', total, radius, color='#c9b28d', provenance=provenance)
    # Keep unaffected rows in their original relative order, append the new ID.
    keep = [i for i, body_id in enumerate(prepared.body_ids) if body_id not in ids]
    references = tuple(key if prepared.reference_ids[i] in ids else prepared.reference_ids[i] for i in keep)+(None,)
    message = (f'Merge ({rule}): {", ".join(ids)} -> {key}. '
               'One remnant keeps the mass and the linear momentum.')
    if rule == 'perfect_merge':
        message += ' Perfect merge is a disclosed option, not a general collision law.'
    event = CollisionEvent(s.time, 'merge', ids, key, ancestry, lost, spin, message,
                           rule=rule, outcome='merge', descendant_ids=(key,))
    merged = PreparedSystem(s.with_state(np.vstack((s.r[keep], r)), np.vstack((s.v[keep], v)), s.time,
                                         masses=np.append(s.masses[keep], total)),
        tuple(prepared.body_ids[i] for i in keep)+(key,), tuple(prepared.body_specs[i] for i in keep)+(spec,),
        prepared.setup, prepared.events+(event,), prepared.segment, references)
    return merged, event


def merge_pair(prepared, first_id, second_id):
    """Pure response primitive; the stepping boundary establishes actual contact."""
    return _merge_group(prepared, (first_id, second_id))


def _pairs(prepared):
    """O(N * massive-count), with no tracer/tracer candidate matrix."""
    s = prepared.system
    sources = np.flatnonzero(s.masses > 0)
    rows = np.arange(len(s.masses))[:, None]
    first, source_col = np.nonzero((s.masses[:, None] == 0) | (rows < sources[None, :]))
    second = sources[source_col]
    radii = np.array([spec.radius if mass > 0 else 0. for spec, mass in zip(prepared.body_specs, s.masses)])
    return first, second, radii[first]+radii[second]


def _first_root(q, u, w, radius, policy, x_tolerance):
    """Earliest contact on q + u*x + w*x², 0<=x<=1.

    Cubic stationary points partition the squared-distance quartic into monotone
    intervals. Bisection finds entry; a stationary tangent also counts as contact.
    """
    coefficients = np.array([np.dot(q, q)-radius**2, 2*np.dot(q, u),
                             np.dot(u, u)+2*np.dot(q, w), 2*np.dot(u, w), np.dot(w, w)])
    scale = max(float(np.max(np.abs(coefficients))), radius**2, np.finfo(float).tiny)
    coefficients /= scale
    tolerance = policy.distance_rtol*radius**2/scale + 32*np.finfo(float).eps
    derivative = np.polynomial.polynomial.polyder(coefficients)
    roots = np.polynomial.polynomial.polyroots(np.trim_zeros(derivative, 'b')) if np.any(derivative) else ()
    breaks = [0., *sorted(float(root.real) for root in roots if abs(root.imag) <= 1e-10 and 0 < root.real < 1), 1.]
    def value(x):
        # Evaluate geometry instead of subtracting nearly equal quartic terms.
        position = q+x*(u+x*w)
        return (float(np.dot(position, position))-radius**2)/scale
    if value(0.) <= 0:
        return 0.
    for low, high in zip(breaks, breaks[1:]):
        fh = value(high)
        if fh > tolerance:
            continue
        if fh > 0:  # a floating-point tangent, not a finite penetration
            return high
        for _ in range(policy.max_refinements):
            mid = (low+high)*.5
            if value(mid) > 0:
                low = mid
            else:
                high = mid
        if high-low > x_tolerance:
            raise ValueError('Collision numerical halt: bounded contact refinement tolerance unmet; state preserved.')
        return high
    return None


def _contacts(prepared, acceleration, h, policy):
    first, second, radii = _pairs(prepared)
    if not len(first):
        return []
    s = prepared.system
    q = s.r[first]-s.r[second]
    u = (s.v[first]-s.v[second])*h
    w = .5*(acceleration[first]-acceleration[second])*h*h
    chord = u+w
    norm2 = np.einsum('ij,ij->i', chord, chord)
    x = np.clip(np.divide(-np.einsum('ij,ij->i', q, chord), norm2,
                          out=np.zeros_like(norm2), where=norm2 > 0), 0., 1.)
    minimum_chord = np.linalg.norm(q+x[:, None]*chord, axis=1)
    # Quadratic/chord deviation is |a_rel| h²/8 = |w|/4.
    reach = radii+np.linalg.norm(w, axis=1)/4
    candidates = np.flatnonzero(minimum_chord <= reach*(1+policy.distance_rtol)+32*np.finfo(float).eps*np.maximum(1, np.linalg.norm(q, axis=1)))
    found = []
    for k in candidates:
        x_tolerance = policy.time_tolerance*max(1., abs(s.dt))/h if h > 0 else np.inf
        root = _first_root(q[k], u[k], w[k], radii[k], policy, x_tolerance)
        if root is not None:
            found.append((root*h, prepared.body_ids[first[k]], prepared.body_ids[second[k]]))
    return sorted(found, key=lambda item: (item[0], tuple(sorted(item[1:]))))


def mutual_escape_speed(mass_1, mass_2, radius_sum):
    """Escape speed at the contact separation, in AU/yr."""
    if not math.isfinite(radius_sum) or radius_sum <= 0 or mass_1 < 0 or mass_2 < 0:
        raise ValueError('Escape speed needs a positive contact radius and non-negative masses.')
    if not math.isfinite(mass_1) or not math.isfinite(mass_2):
        raise ValueError('Escape speed needs finite masses.')
    return math.sqrt(2*G*(mass_1+mass_2)/radius_sum)


def classify_impact(impact_speed, escape_speed, impact_parameter):
    """Merge, graze, or fragment from impact speed and the contact impact parameter.

    ``impact_parameter`` is ``|r × v| / (|r| |v|)`` at contact: 0 is head-on and 1
    is tangent. At or below mutual escape the outcome is merge. Above escape, a
    parameter of at least ``GRAZE_IMPACT_PARAMETER`` is a graze and a smaller one
    fragments. This is a disclosed speed-and-angle map, not the full Stewart and
    Leinhardt strength model.
    """
    if not math.isfinite(impact_speed) or impact_speed < 0:
        raise ValueError('Impact speed must be finite and non-negative.')
    if not math.isfinite(escape_speed) or escape_speed <= 0:
        raise ValueError('Mutual escape speed must be finite and positive.')
    if not math.isfinite(impact_parameter):
        raise ValueError('Impact parameter must be finite.')
    if impact_speed <= escape_speed:
        return 'merge'
    if min(1., max(0., float(impact_parameter))) >= GRAZE_IMPACT_PARAMETER:
        return 'graze'
    return 'fragment'


def fragment_mass_fractions(impact_speed, escape_speed, impact_parameter):
    """Largest-remnant, each tracked fragment, and lost-mass fractions. They sum to 1."""
    if classify_impact(impact_speed, escape_speed, impact_parameter) != 'fragment':
        raise ValueError('Fragment fractions apply only to the fragment outcome.')
    ratio = impact_speed/escape_speed
    severity = min(4., max(0., (ratio-1.)*(1.-min(1., max(0., float(impact_parameter))))))
    lost = min(.2, .05*severity)
    each = min(.25, .10+.05*severity)
    largest = 1.-lost-2.*each
    if largest < .2:
        scale = .8/(lost+2.*each)
        lost *= scale
        each *= scale
        largest = 1.-lost-2.*each
    return largest, each, lost


def _rule_name(prepared):
    return prepared.setup.collision_mode if prepared.setup is not None else 'perfect_merge'


def _body_index(prepared, body_id):
    return prepared.body_ids.index(body_id)


def _impact_geometry(prepared, first_id, second_id):
    i, j = _body_index(prepared, first_id), _body_index(prepared, second_id)
    delta = prepared.system.r[i]-prepared.system.r[j]
    relative = prepared.system.v[i]-prepared.system.v[j]
    distance = float(np.linalg.norm(delta))
    speed = float(np.linalg.norm(relative))
    if distance > 0 and np.isfinite(distance):
        direction = delta/distance
    else:
        direction = np.zeros(delta.shape[0])
        direction[0] = 1.
        distance = 0.
    if speed > 0 and distance > 0:
        # Planar runs keep the signed z component. A third component uses the full cross product.
        if delta.shape[0] == 2:
            cross = abs(float(delta[0]*relative[1]-delta[1]*relative[0]))
        else:
            cross = float(np.linalg.norm(np.cross(delta, relative)))
        impact_parameter = cross/(distance*speed)
    else:
        impact_parameter = 0.
    return i, j, direction, distance, speed, min(1., max(0., impact_parameter))


def _new_body_id(prepared, tag, parent_ids, role):
    stem = tag+':'+hashlib.sha256((role+'\0'+'\0'.join(parent_ids)).encode()).hexdigest()[:24]
    key, ordinal = stem, 0
    used = _used_ids(prepared)
    while key in used:
        ordinal += 1
        key = f'{stem}:{ordinal}'
    return key


def _kinetic_of(masses, velocities, indices):
    total = 0.
    for index in indices:
        total += .5*float(masses[index])*float(np.dot(velocities[index], velocities[index]))
    return total


def _spin_about(origin, masses, positions, velocities, indices):
    total = 0.
    for index in indices:
        offset = positions[index]-origin
        total += float(masses[index])*(float(offset[0])*float(velocities[index, 1])-float(offset[1])*float(velocities[index, 0]))
    return total


def _unmodeled_star(prepared, contact):
    """Log a stellar contact this rule does not model, and keep both bodies on their trajectories.

    The overlap stays open until the centers separate. While it is open, that pair's
    gravity and step limit use the contact separation so the fixed step can cross
    the point-mass singularity. Masses and velocities are not changed here, and no
    remnant is created. A stellar-merger model remains #14.
    """
    rule = _rule_name(prepared)
    first_id, second_id = contact[1], contact[2]
    i, j, _direction, distance, _speed, _parameter = _impact_geometry(prepared, first_id, second_id)
    system = prepared.system
    ids = tuple(sorted((first_id, second_id)))
    if ids in prepared.open_passages:
        return prepared
    origin = .5*(system.r[i]+system.r[j])
    message = (f'Unmodeled stellar contact ({rule}): {", ".join(ids)}. '
               'This rule does not merge a star and does not build a nonstellar remnant. '
               'The bodies keep their masses and velocities. While their surfaces stay in contact, '
               'that pair\'s gravity is evaluated at the contact separation so the run can finish. '
               'A stellar-merger model remains #14.')
    event = CollisionEvent(float(system.time), 'unmodeled', ids, None, ids, 0., 0., message,
                           rule=rule, outcome='unmodeled', descendant_ids=ids, approach_distance=float(distance),
                           location=tuple(float(component) for component in origin))
    return replace(prepared, events=prepared.events+(event,), open_passages=prepared.open_passages+(ids,))


def _graze_pair(prepared, first_id, second_id, speed, impact_parameter):
    """Hit-and-run: both bodies keep their mass and the normal relative velocity reverses."""
    rule = _rule_name(prepared)
    i, j, direction, distance, _, _ = _impact_geometry(prepared, first_id, second_id)
    system = prepared.system
    positions, velocities = system.r.copy(), system.v.copy()
    masses = system.masses
    radius = (prepared.body_specs[i].radius or 0.)+(prepared.body_specs[j].radius or 0.)
    target = radius*(1+1e-6)+1e-9
    mass_i, mass_j = float(masses[i]), float(masses[j])
    total = mass_i+mass_j
    if distance < target:
        shift = target-distance
        positions[i] = positions[i]+direction*shift*(mass_j/total)
        positions[j] = positions[j]-direction*shift*(mass_i/total)
    relative = velocities[i]-velocities[j]
    normal = float(np.dot(relative, direction))
    before = _kinetic_of(masses, velocities, (i, j))
    if normal < 0:
        change = -2*normal*direction
        velocities[i] = velocities[i]+change*(mass_j/total)
        velocities[j] = velocities[j]-change*(mass_i/total)
    lost = before-_kinetic_of(masses, velocities, (i, j))
    origin = (mass_i*system.r[i]+mass_j*system.r[j])/total
    spin = _spin_about(origin, masses, system.r, system.v, (i, j))-_spin_about(origin, masses, positions, velocities, (i, j))
    ids = tuple(sorted((first_id, second_id)))
    message = (f'Graze ({rule}): {", ".join(ids)}. Impact speed {speed:.6g} AU/yr exceeds mutual escape '
               f'and the impact parameter is {impact_parameter:.6g}. Both bodies keep their mass. '
               'The normal relative velocity is reversed and the tangential velocity is unchanged. '
               'No fragment body is created.')
    event = CollisionEvent(float(system.time), 'graze', ids, None, _ancestors(prepared, ids), float(lost), float(spin),
                           message, rule=rule, outcome='graze', descendant_ids=ids, impact_speed=float(speed),
                           impact_parameter=float(impact_parameter), approach_distance=float(distance),
                           location=tuple(float(component) for component in origin))
    return replace(prepared, system=system.with_state(positions, velocities, system.time, masses=masses.copy()),
                   events=prepared.events+(event,))


def _fragment_pair(prepared, first_id, second_id, speed, impact_parameter, escape):
    """Two tracked fragments plus one largest remnant. Lost mass is recorded, not dropped."""
    rule = _rule_name(prepared)
    ids = tuple(sorted((first_id, second_id)))
    indices = [_body_index(prepared, key) for key in ids]
    system = prepared.system
    masses = system.masses
    parent_masses = np.array([float(masses[index]) for index in indices])
    total = float(parent_masses.sum())
    largest_fraction, each_fraction, lost_fraction = fragment_mass_fractions(speed, escape, impact_parameter)
    lost_mass = total*lost_fraction
    each_mass = total*each_fraction
    largest_mass = total-lost_mass-2*each_mass
    origin = sum(parent_masses[n]*system.r[index] for n, index in enumerate(indices))/total
    com_velocity = sum(parent_masses[n]*system.v[index] for n, index in enumerate(indices))/total
    incoming = _kinetic_of(masses, system.v, indices)
    com_kinetic = .5*total*float(np.dot(com_velocity, com_velocity))
    relative_kinetic = incoming-com_kinetic
    volume = sum((prepared.body_specs[index].radius or 0.)**3 for index in indices)
    roles = (('largest', largest_mass), ('piece-1', each_mass), ('piece-2', each_mass))
    keys = []
    for role, _mass in roles:
        keys.append(_new_body_id(prepared, 'fragment', ids, role))
        # Reserve the id before the next hash so the three stems can still theoretically collide.
        prepared = replace(prepared, events=prepared.events+(
            CollisionEvent(float(system.time), 'reserve', ids, keys[-1], (), 0., 0., f'Reserved fragment id {keys[-1]}.'),))
    direction = _impact_geometry(prepared, ids[0], ids[1])[2]
    radii = tuple(float(np.cbrt(mass/total*volume)) for _role, mass in roles)
    gap = (radii[0]+radii[1])*(1+1e-2)
    retained = FRAGMENT_KINETIC_RETAINED*max(relative_kinetic, 0.)
    piece_speed = math.sqrt(retained/each_mass) if each_mass > 0 and retained > 0 else 0.
    width = int(np.asarray(origin).shape[0])
    offsets = (np.zeros(width), direction*gap, -direction*gap)
    piece_velocities = (np.zeros(width), direction*piece_speed, -direction*piece_speed)
    rows_r, rows_v, row_masses, specs = [], [], [], []
    ancestry = _ancestors(prepared, ids)
    for key, (_role, mass), radius, offset, velocity in zip(keys, roles, radii, offsets, piece_velocities):
        rows_r.append(origin+offset)
        rows_v.append(com_velocity+velocity)
        row_masses.append(mass)
        label = 'remnant' if _role == 'largest' else _role.replace('-', ' ')
        specs.append(BodySpec(f'{" + ".join(prepared.body_specs[index].name for index in indices)} {label}',
                              'planet', mass, radius, color='#c9b28d',
                              provenance=('Computed speed_angle fragment\n'
                                          'mass is a share of the parents\n'
                                          'lost mass is recorded and is not a body\n'
                                          'temperature and composition unknown.')))
    # Drop the temporary reservations now that the real event carries the ids.
    reserved = tuple(event for event in prepared.events if not (event.kind == 'reserve' and event.remnant_id in keys))
    outgoing = sum(.5*mass*float(np.dot(velocity, velocity)) for mass, velocity in zip(row_masses, piece_velocities))
    lost_kinetic = relative_kinetic-outgoing
    parent_spin = _spin_about(origin, masses, system.r, system.v, indices)
    child_positions = np.vstack(rows_r)
    child_velocities = np.vstack(rows_v)
    child_masses = np.array(row_masses)
    child_spin = _spin_about(origin, child_masses, child_positions, child_velocities, range(3))
    message = (f'Fragment ({rule}): {", ".join(ids)} -> {", ".join(keys)}. '
               f'Impact speed {speed:.6g} AU/yr, mutual escape {escape:.6g} AU/yr, '
               f'impact parameter {impact_parameter:.6g}. '
               f'Largest remnant {keys[0]} holds {largest_mass:.6g} solar masses. '
               f'Two fragment bodies continue. Lost mass {lost_mass:.6g} solar masses is recorded '
               'and is not a body; it leaves at the parents\' center-of-mass velocity. '
               'Parcel ancestry is shared by the tracked descendants.')
    event = CollisionEvent(float(system.time), 'fragment', ids, keys[0], ancestry, float(lost_kinetic),
                           float(parent_spin-child_spin), message, rule=rule, outcome='fragment',
                           descendant_ids=tuple(keys), lost_mass=float(lost_mass), impact_speed=float(speed),
                           impact_parameter=float(impact_parameter), approach_distance=float(np.linalg.norm(system.r[indices[0]]-system.r[indices[1]])),
                           location=tuple(float(component) for component in origin))
    keep = [index for index, body_id in enumerate(prepared.body_ids) if body_id not in ids]
    references = tuple(keys[0] if prepared.reference_ids[index] in ids else prepared.reference_ids[index] for index in keep)+(None, None, None)
    updated = system.with_state(np.vstack((system.r[keep], child_positions)),
                                np.vstack((system.v[keep], child_velocities)), system.time,
                                masses=np.append(system.masses[keep], row_masses))
    return PreparedSystem(updated, tuple(prepared.body_ids[index] for index in keep)+tuple(keys),
                          tuple(prepared.body_specs[index] for index in keep)+tuple(specs),
                          prepared.setup, reserved+(event,), prepared.segment, references)


def _speed_angle_pair(prepared, first_id, second_id):
    i, j, _direction, _distance, speed, impact_parameter = _impact_geometry(prepared, first_id, second_id)
    radius = (prepared.body_specs[i].radius or 0.)+(prepared.body_specs[j].radius or 0.)
    escape = mutual_escape_speed(float(prepared.system.masses[i]), float(prepared.system.masses[j]), radius)
    outcome = classify_impact(speed, escape, impact_parameter)
    if outcome == 'merge':
        return _merge_group(prepared, (first_id, second_id))[0]
    if outcome == 'graze':
        return _graze_pair(prepared, first_id, second_id, speed, impact_parameter)
    return _fragment_pair(prepared, first_id, second_id, speed, impact_parameter, escape)


def _absorb_tracers(prepared, absorbed, target_map):
    rule = _rule_name(prepared)
    for particle in sorted(absorbed):
        system = prepared.system
        target = target_map.get(min(absorbed[particle]), min(absorbed[particle]))
        event = CollisionEvent(system.time, 'absorption', (particle,), None, (particle,), 0., 0.,
            f'Point tracer {particle} absorbed by {target}; tracer removed with zero mass/momentum feedback; no remnant selection.',
            target, rule=rule, outcome='absorption', descendant_ids=())
        keep = [i for i, key in enumerate(prepared.body_ids) if key != particle]
        prepared = replace(prepared, system=system.with_state(system.r[keep], system.v[keep], system.time,
                                                              masses=system.masses[keep]),
            body_ids=tuple(prepared.body_ids[i] for i in keep), body_specs=tuple(prepared.body_specs[i] for i in keep),
            reference_ids=tuple(prepared.reference_ids[i] for i in keep), events=prepared.events+(event,))
    return prepared


def _ignore_contacts(prepared, contacts):
    """Let the bodies overlap. Masses and velocities stay as the gravity step left them."""
    specs = dict(zip(prepared.body_ids, prepared.body_specs))
    stellar = [contact for contact in contacts if specs[contact[1]].kind == 'star' or specs[contact[2]].kind == 'star']
    if stellar:
        return _unmodeled_star(prepared, stellar[0])
    passages = list(prepared.open_passages)
    events = []
    system = prepared.system
    for contact in contacts:
        ids = tuple(sorted((contact[1], contact[2])))
        if ids in passages:
            continue
        passages.append(ids)
        i, j, _direction, distance, _speed, _parameter = _impact_geometry(prepared, ids[0], ids[1])
        origin = .5*(system.r[i]+system.r[j])
        message = (f'Ignored contact ({_rule_name(prepared)}): {ids[0]} and {ids[1]} overlap at '
                   f'{distance:.6g} AU. Masses and velocities are unchanged and the run continues.')
        events.append(CollisionEvent(float(system.time), 'ignored', ids, None, ids, 0., 0., message,
                                     rule=_rule_name(prepared), outcome='ignored', descendant_ids=ids,
                                     approach_distance=float(distance),
                                     location=tuple(float(component) for component in origin)))
    if not events:
        return prepared
    return replace(prepared, events=prepared.events+tuple(events), open_passages=tuple(passages))


def _stop_contact(prepared, contacts):
    """End the run at the first contact. No remnant is built."""
    contact = min(contacts, key=lambda item: (item[0], tuple(sorted(item[1:]))))
    first_id, second_id = contact[1], contact[2]
    i, j, _direction, distance, _speed, _parameter = _impact_geometry(prepared, first_id, second_id)
    system = prepared.system
    ids = tuple(sorted((first_id, second_id)))
    origin = .5*(system.r[i]+system.r[j])
    message = (f'Stop on contact: {ids[0]} and {ids[1]} met at {float(distance):.6g} AU. '
               'The run ends at this contact. No remnant is built.')
    event = CollisionEvent(float(system.time), 'stop', ids, None, ids, 0., 0., message,
                           rule='stop_on_contact', outcome='stop', descendant_ids=ids,
                           approach_distance=float(distance),
                           location=tuple(float(component) for component in origin))
    return replace(prepared, events=prepared.events+(event,))


def _contains_stop(events):
    return any(isinstance(event, CollisionEvent) and event.kind == 'stop' for event in events)


def _contacted_ids(events):
    contacted = set()
    for event in events:
        if isinstance(event, CollisionEvent) and event.kind in (
                'merge', 'fragment', 'graze', 'unmodeled', 'absorption', 'stellar_merger', 'ignored', 'stop'):
            contacted.update(key for key in event.parent_ids if isinstance(key, str))
    return contacted


def _respond(prepared, contacts):
    """One disclosed response. Stars are logged and kept unless the #14 scenario handles them."""
    if prepared.system.detail is not None:
        from .body_detail import NO_REMNANT
        rule = _rule_name(prepared)
        if rule not in ('ignore', 'stop_on_contact', 'none'):
            raise ValueError(NO_REMNANT)
    from .stellar_merger import respond_if_stellar_scenario
    handled = respond_if_stellar_scenario(prepared, contacts)
    if handled is not None:
        return handled
    rule = _rule_name(prepared)
    if rule == 'ignore':
        return _ignore_contacts(prepared, contacts)
    if rule == 'stop_on_contact':
        return _stop_contact(prepared, contacts)
    masses = dict(zip(prepared.body_ids, prepared.system.masses))
    specs = dict(zip(prepared.body_ids, prepared.body_specs))
    stellar = [contact for contact in contacts if specs[contact[1]].kind == 'star' or specs[contact[2]].kind == 'star']
    if stellar:
        return _unmodeled_star(prepared, stellar[0])
    adjacency, absorbed = {}, {}
    for _, first_id, second_id in contacts:
        if masses[first_id] == 0 or masses[second_id] == 0:
            particle, target = (first_id, second_id) if masses[first_id] == 0 else (second_id, first_id)
            absorbed.setdefault(particle, set()).add(target)
        else:
            adjacency.setdefault(first_id, set()).add(second_id)
            adjacency.setdefault(second_id, set()).add(first_id)
    if _rule_name(prepared) == 'speed_angle' and adjacency:
        first_id = min(adjacency)
        second_id = min(adjacency[first_id])
        prepared = _speed_angle_pair(prepared, first_id, second_id)
        latest = next(event for event in reversed(prepared.events) if event.kind in ('merge', 'fragment', 'graze'))
        target_map = {} if latest.remnant_id is None else {first_id: latest.remnant_id, second_id: latest.remnant_id}
        return _absorb_tracers(prepared, absorbed, target_map)
    groups = []
    while adjacency:
        pending, group = [min(adjacency)], set()
        while pending:
            key = pending.pop()
            if key in group:
                continue
            group.add(key)
            pending.extend(adjacency.get(key, ()))
        for key in group:
            adjacency.pop(key, None)
        groups.append(tuple(sorted(group)))
    target_map = {}
    for group in sorted(groups):
        prepared, event = _merge_group(prepared, group)
        target_map.update({key: event.remnant_id for key in group})
    return _absorb_tracers(prepared, absorbed, target_map)


def _refresh_passages(prepared):
    """Drop unmodeled pairs once their centers are outside contact."""
    kept = []
    for pair in prepared.open_passages:
        if pair[0] not in prepared.body_ids or pair[1] not in prepared.body_ids:
            continue
        i, j = prepared.body_ids.index(pair[0]), prepared.body_ids.index(pair[1])
        radius = sum((prepared.body_specs[index].radius or 0.) if prepared.system.masses[index] > 0 else 0. for index in (i, j))
        if float(np.linalg.norm(prepared.system.r[i]-prepared.system.r[j])) <= radius*(1+1e-8):
            kept.append(pair)
    if tuple(kept) != prepared.open_passages:
        return replace(prepared, open_passages=tuple(kept))
    return prepared


def _passage_floor(prepared):
    if not prepared.open_passages:
        return None
    floor = np.zeros((len(prepared.body_ids), len(prepared.body_ids)))
    for pair in prepared.open_passages:
        i, j = prepared.body_ids.index(pair[0]), prepared.body_ids.index(pair[1])
        radius = sum((prepared.body_specs[index].radius or 0.) if prepared.system.masses[index] > 0 else 0. for index in (i, j))
        floor[i, j] = floor[j, i] = max(radius, 1e-12)
    return floor


def _without_passages(prepared, contacts):
    hidden = set(prepared.open_passages)
    return [contact for contact in contacts if tuple(sorted(contact[1:])) not in hidden]


def _note_approaches(prepared, approaches):
    first, second, radii = _pairs(prepared)
    if not len(first):
        return
    dist = np.linalg.norm(prepared.system.r[first]-prepared.system.r[second], axis=1)
    limit = ENCOUNTER_FACTOR*radii
    time = float(prepared.system.time)
    for k in np.flatnonzero((limit > 0) & (dist <= limit*(1+1e-12))):
        key = tuple(sorted((prepared.body_ids[int(first[k])], prepared.body_ids[int(second[k])])))
        distance = float(dist[k])
        midpoint = .5*(prepared.system.r[int(first[k])]+prepared.system.r[int(second[k])])
        previous = approaches.get(key)
        if previous is None or distance < previous[0]:
            approaches[key] = (distance, time, tuple(float(component) for component in midpoint), float(limit[k]))


def _remember_resolved(resolved, events):
    for event in events:
        if not isinstance(event, CollisionEvent) or event.kind not in _DYNAMIC_KINDS:
            continue
        ids = {key for key in event.parent_ids if isinstance(key, str)}
        if isinstance(event.target_id, str):
            ids.add(event.target_id)
        if len(ids) >= 2:
            resolved.add(frozenset(ids))


def _kepler_elements(position, velocity, mu):
    radius = float(np.linalg.norm(position))
    if radius == 0 or not math.isfinite(mu) or mu <= 0:
        return None
    speed2 = float(np.dot(velocity, velocity))
    energy = .5*speed2-mu/radius
    radial = float(np.dot(position, velocity))
    eccentricity = float(np.linalg.norm(((speed2-mu/radius)*position-radial*velocity)/mu))
    if abs(energy) <= 1e-14*max(1., mu/radius):
        semi_major = math.inf
    else:
        semi_major = -mu/(2.*energy)
    return semi_major, eccentricity


def _orbit_changed(before, after):
    if before is None or after is None:
        return False
    if abs(after[1]-before[1]) >= ORBITAL_ECCENTRICITY_TOLERANCE:
        return True
    scale = max(abs(before[0]), abs(after[0]), 1e-12)
    return math.isfinite(before[0]) and math.isfinite(after[0]) and abs(after[0]-before[0])/scale >= ORBITAL_AXIS_RTOL


def _primary_id(prepared):
    stars = [i for i, spec in enumerate(prepared.body_specs) if spec.kind == 'star' and prepared.system.masses[i] > 0]
    pool = stars or [i for i, mass in enumerate(prepared.system.masses) if mass > 0]
    if not pool:
        return None
    return prepared.body_ids[max(pool, key=lambda i: (float(prepared.system.masses[i]), prepared.body_ids[i]))]


def _append_passage_logs(prepared, start_state, approaches, resolved):
    """Close encounters and orbital changes for pairs that did not contact."""
    rule = _rule_name(prepared)
    survivors = set(prepared.body_ids)
    encounter_events = []
    interesting = []
    for key, (distance, time, location, limit) in sorted(approaches.items(), key=lambda item: (item[1][1], item[0])):
        if frozenset(key) in resolved or any(body_id not in survivors or body_id not in start_state for body_id in key):
            continue
        start = float(np.linalg.norm(start_state[key[0]][0]-start_state[key[1]][0]))
        end = float(np.linalg.norm(prepared.system.r[prepared.body_ids.index(key[0])]-prepared.system.r[prepared.body_ids.index(key[1])]))
        if not ((start > limit and distance <= limit) or (distance < start*(1-1e-6) and distance < end*(1-1e-6))):
            continue
        message = (f'Close encounter ({rule}): {key[0]}, {key[1]}. '
                   f'Closest approach {distance:.6g} AU at {time:.6g} yr, inside {limit:.6g} AU. No contact.')
        encounter_events.append(CollisionEvent(time, 'close_encounter', key, None, key, 0., 0., message,
                                               rule=rule, outcome='close_encounter', descendant_ids=key,
                                               approach_distance=distance, location=location))
        interesting.append(key)
    for event in prepared.events:
        if (isinstance(event, CollisionEvent) and event.kind in ('graze', 'unmodeled')
                and all(body_id in survivors and body_id in start_state for body_id in event.parent_ids)):
            interesting.append(tuple(sorted(event.parent_ids)))
    primary = _primary_id(prepared)
    orbital = []
    if primary in start_state and primary in survivors:
        pindex = prepared.body_ids.index(primary)
        start_pr, start_pv = start_state[primary]
        pr, pv = prepared.system.r[pindex], prepared.system.v[pindex]
        seen = set()
        for pair in interesting:
            for body_id, other_id in (pair, pair[::-1]):
                if body_id == primary or body_id not in start_state or (body_id, other_id) in seen:
                    continue
                seen.add((body_id, other_id))
                index = prepared.body_ids.index(body_id)
                mu = G*(float(prepared.system.masses[pindex])+float(prepared.system.masses[index]))
                before = _kepler_elements(start_state[body_id][0]-start_pr, start_state[body_id][1]-start_pv, mu)
                after = _kepler_elements(prepared.system.r[index]-pr, prepared.system.v[index]-pv, mu)
                if not _orbit_changed(before, after):
                    continue
                message = (f'Orbital change ({rule}): {body_id} after encountering {other_id}. '
                           f'semi-major {before[0]:.6g} -> {after[0]:.6g} AU, '
                           f'eccentricity {before[1]:.6g} -> {after[1]:.6g}.')
                orbital.append(CollisionEvent(float(prepared.system.time), 'orbital_change', (body_id, other_id), None,
                                              (body_id,), 0., 0., message, rule=rule, outcome='orbital_change',
                                              descendant_ids=(body_id,),
                                              location=tuple(float(component) for component in prepared.system.r[index])))
    if not encounter_events and not orbital:
        return prepared
    return replace(prepared, events=prepared.events+tuple(encounter_events)+tuple(orbital))


def advance_prepared(prepared):
    """One configured macro step, including opt-in relativistic terms and stellar mass loss.

    Both modules are off unless the setup enables them. The Newtonian step is
    unchanged when they are off.
    """
    setup = prepared.setup
    relativistic = setup is not None and setup.relativistic.enabled
    mass_loss = setup is not None and setup.stellar_mass_loss.enabled
    if not relativistic and not mass_loss:
        return _advance_prepared(prepared)
    from .mass_loss import apply_stellar_mass_loss
    from .relativity import relativistic_context
    started = float(prepared.system.time)
    with relativistic_context(setup):
        result = _advance_prepared(prepared)
    elapsed = float(result.prepared.system.time) - started
    if mass_loss and elapsed > 0.0:
        result = AdvanceResult(apply_stellar_mass_loss(result.prepared, elapsed), result.event_frames, result.stopped)
    return result


def _advance_prepared(prepared):
    """One configured macro dt, or no mutation on collision-mode rejection.

    All actual positive Verlet substeps obey half the existing start/end gravity
    and crossing limit. Contact is checked before computing endpoint forces.
    Event clipping uses the same drift polynomial and then integrates remainder.
    A resolved collision does not end the step. Numerical failure still raises.
    """
    if prepared.setup is None or prepared.setup.collision_mode == 'none':
        # Off, this is the same Yoshida step as before. On, the kick is a
        # separate velocity update and is not part of the contact solver.
        if prepared.setup is not None and prepared.setup.tides.enabled:
            from .tides import dissipate_close_passes
            start_r = prepared.system.r.copy()
            start_v = prepared.system.v.copy()
            prepared.system.step()
            prepared = dissipate_close_passes(prepared, start_r, start_v)
            return AdvanceResult(prepared)
        prepared.system.step()
        return AdvanceResult(prepared)
    validate_collision_state(prepared)
    policy = prepared.setup.collision_policy
    old = prepared.system
    trial = replace(prepared, system=old.with_state(old.r.copy(), old.v.copy(), old.time, masses=old.masses.copy()))
    trial.system.detail = None if old.detail is None else old.detail.copy()
    end = old.time+old.dt
    if not math.isfinite(end) or end <= old.time:
        raise ValueError('Collision numerical halt: macro timestep cannot advance clock; state preserved.')
    start_state = {body_id: (old.r[index].copy(), old.v[index].copy()) for index, body_id in enumerate(prepared.body_ids)}
    approaches, resolved = {}, set()
    event_frames = []
    attempts = 0
    stopped = False
    initial_events = len(prepared.events)
    while trial.system.time < end:
        trial = _refresh_passages(trial)
        floor = _passage_floor(trial)
        _note_approaches(trial, approaches)
        s = trial.system
        immediate = _without_passages(trial, _contacts(trial, np.zeros_like(s.r), 0., policy))
        if immediate:
            attempts += 1
            if attempts > policy.max_substeps:
                raise ValueError('Collision numerical halt: bounded response budget exceeded; state preserved.')
            before = StateFrame.capture(trial)
            old_count = len(trial.events)
            trial = _respond(trial, immediate)
            _remember_resolved(resolved, trial.events[old_count:])
            event_frames.append(replace(before, event_phase='pre', events=before.events+trial.events[old_count:]))
            event_frames.append(replace(StateFrame.capture(trial), event_phase='post'))
            if _contains_stop(trial.events[old_count:]):
                stopped = True
                break
            continue
        remaining = end-s.time
        limit = s.resolution_limit(floor)
        h = min(remaining, old.dt*policy.macro_fraction, limit*policy.guard_fraction)
        if not math.isfinite(h) or h <= 0 or s.time+h <= s.time:
            raise ValueError('Collision numerical halt: substep cannot resolve encounter / advance clock; state preserved.')
        acceleration = s.accelerations(floor)
        while True:
            attempts += 1
            if attempts > policy.max_substeps:
                raise ValueError('Collision numerical halt: bounded substep/refinement budget exceeded; state preserved.')
            contacts = _without_passages(trial, _contacts(trial, acceleration, h, policy))
            elapsed = contacts[0][0] if contacts else h
            r = s.r+s.v*elapsed+.5*acceleration*elapsed**2
            vhalf = s.v+.5*acceleration*elapsed
            candidate = s.with_state(r, vhalf, s.time+elapsed, masses=s.masses.copy(), dt=old.dt)
            # Endpoint force is safe only after the sweep has clipped first contact.
            candidate.v = vhalf+.5*elapsed*candidate.accelerations(floor)
            if not np.isfinite(candidate.r).all() or not np.isfinite(candidate.v).all():
                raise ValueError('Collision numerical halt: nonfinite Verlet state; state preserved.')
            if elapsed > candidate.resolution_limit(floor)*policy.guard_fraction*(1+1e-12):
                h *= .5
                if s.time+h <= s.time:
                    raise ValueError('Collision numerical halt: endpoint encounter unresolved; state preserved.')
                continue
            break
        if s.detail is not None and elapsed > 0:
            s.detail.advance(candidate, elapsed)
        trial = replace(trial, system=candidate)
        if contacts:
            tolerance = policy.time_tolerance*max(1., abs(old.dt))
            simultaneous = []
            for contact in contacts:
                if contact[0] > elapsed+tolerance:
                    continue
                a, b = (trial.body_ids.index(key) for key in contact[1:])
                radius = sum(trial.body_specs[i].radius if trial.system.masses[i] > 0 else 0. for i in (a, b))
                # World endpoints near 1 AU cannot preserve a sub-kilometer
                # separation. Verify the same relative drift the root search used.
                relative = ((s.r[a]-s.r[b])+(s.v[a]-s.v[b])*elapsed
                            +.5*(acceleration[a]-acceleration[b])*elapsed**2)
                distance = float(np.linalg.norm(relative))
                # Time proximity alone cannot put a still-separated pair into
                # the contact group (especially when h << time_tolerance).
                if distance <= radius*(1+policy.distance_rtol)+64*np.finfo(float).eps*max(radius, distance, float(np.linalg.norm(s.r[a]-s.r[b]))):
                    simultaneous.append(contact)
            if not simultaneous:
                raise ValueError('Collision numerical halt: swept contact geometry unresolved; state preserved.')
            before = StateFrame.capture(trial)
            old_count = len(trial.events)
            trial = _respond(trial, simultaneous)
            _remember_resolved(resolved, trial.events[old_count:])
            event_frames.append(replace(before, event_phase='pre', events=before.events+trial.events[old_count:]))
            event_frames.append(replace(StateFrame.capture(trial), event_phase='post'))
            if _contains_stop(trial.events[old_count:]):
                stopped = True
                break
        elif remaining == h:
            trial.system.time = end
    if not stopped:
        trial = _refresh_passages(trial)
        _note_approaches(trial, approaches)
        if prepared.setup is not None and prepared.setup.tides.enabled:
            from .tides import dissipate_surviving_passes
            trial = dissipate_surviving_passes(trial, start_state, _contacted_ids(trial.events[initial_events:]))
        trial = _append_passage_logs(trial, start_state, approaches, resolved)
    return AdvanceResult(trial, tuple(event_frames), stopped)


def mapped_selection(body_id, events):
    """An explicit merge or fragment remnant may carry a selected identity."""
    for event in events:
        if isinstance(event, CollisionEvent) and body_id in event.parent_ids and event.remnant_id is not None:
            body_id = event.remnant_id
    return body_id
