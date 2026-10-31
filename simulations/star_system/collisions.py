"""Idealized nonstellar contacts; positive-time, bounded, transactional advance.

No rendering coordinates enter this module. The swept path is each Verlet
drift's quadratic, not an assertion of exact continuous Newtonian trajectories.
"""
from dataclasses import dataclass, replace
import hashlib
import math
from collections.abc import Mapping

import numpy as np

from .core import System
from .model import BodySpec, CollisionPolicy
from .initialization import PreparedSystem
from .history import StateFrame


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


@dataclass(frozen=True)
class AdvanceResult:
    prepared: PreparedSystem
    event_frames: tuple[StateFrame, ...] = ()


def validate_collision_state(prepared):
    s = prepared.system
    if not np.isfinite(s.dt) or s.dt <= 0 or not np.isfinite(s.time):
        raise ValueError('Time and positive fixed dt must be finite.')
    s.resolution_limit()  # validates finite arrays without rejecting the macro dt
    for spec, mass in zip(prepared.body_specs, s.masses):
        if mass > 0 and spec.radius is None:
            raise ValueError(f'Physical collision mode requires a known radius for {spec.name}. Supply a radius in AU.')


def _ancestors(prepared, ids):
    prior = {}
    historical = prepared.setup.provenance.get('collision_events', ()) if prepared.setup else ()
    if isinstance(historical, (tuple, list)):
        for event in historical:
            if (isinstance(event, Mapping) and isinstance(event.get('remnant_id'), str)
                    and isinstance(event.get('ancestry'), (tuple, list))
                    and all(isinstance(key, str) for key in event['ancestry'])):
                prior[event['remnant_id']] = tuple(event['ancestry'])
    prior.update({event.remnant_id: event.ancestry for event in prepared.events
                  if isinstance(event, CollisionEvent) and event.remnant_id is not None})
    return tuple(sorted({ancestor for key in ids for ancestor in prior.get(key, (key,))}))


def _used_ids(prepared):
    used = set(prepared.body_ids)
    for event in prepared.events:
        if isinstance(event, CollisionEvent):
            used.update(event.parent_ids)
            used.update(event.ancestry)
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
            for field in ('parent_ids', 'ancestry'):
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
    spin = float(np.sum(masses*(offsets[:, 0]*velocities[:, 1]-offsets[:, 1]*velocities[:, 0])))
    ancestry = _ancestors(prepared, ids)
    key = 'merge:'+hashlib.sha256(('\0'.join(ids)).encode()).hexdigest()[:24]
    stem, ordinal = key, 0
    used_ids = _used_ids(prepared)
    while key in used_ids:
        ordinal += 1
        key = f'{stem}:{ordinal}'
    name = ' + '.join(prepared.body_specs[i].name for i in indices)
    spec = BodySpec(name, 'planet', total, radius, color='#c9b28d',
                    provenance='Computed perfect_merge\ntemperature and composition unknown.')
    # Keep unaffected rows in their original relative order, append the new ID.
    keep = [i for i, body_id in enumerate(prepared.body_ids) if body_id not in ids]
    references = tuple(key if prepared.reference_ids[i] in ids else prepared.reference_ids[i] for i in keep)+(None,)
    event = CollisionEvent(s.time, 'merge', ids, key, ancestry, lost, spin,
        f'Nonstellar perfect merge: {", ".join(ids)} -> {key}.')
    merged = PreparedSystem(System(np.append(s.masses[keep], total), np.vstack((s.r[keep], r)),
                            np.vstack((s.v[keep], v)), s.dt, s.time),
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


def _respond(prepared, contacts):
    """Connected massive groups, then point absorptions, all ordered by ID."""
    masses = dict(zip(prepared.body_ids, prepared.system.masses))
    specs = dict(zip(prepared.body_ids, prepared.body_specs))
    if any(specs[a].kind == 'star' or specs[b].kind == 'star' for _, a, b in contacts):
        raise ValueError('Unsupported star contact: stellar mergers are deferred to #14; state preserved.')
    adjacency, absorbed = {}, {}
    for _, a, b in contacts:
        if masses[a] == 0 or masses[b] == 0:
            particle, target = (a, b) if masses[a] == 0 else (b, a)
            absorbed.setdefault(particle, set()).add(target)
        else:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
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
    for particle in sorted(absorbed):
        s = prepared.system
        target = min(absorbed[particle])
        target = target_map.get(target, target)
        event = CollisionEvent(s.time, 'absorption', (particle,), None, (particle,), 0., 0.,
            f'Point tracer {particle} absorbed by {target}; tracer removed with zero mass/momentum feedback; no remnant selection.', target)
        keep = [i for i, key in enumerate(prepared.body_ids) if key != particle]
        prepared = replace(prepared, system=System(s.masses[keep], s.r[keep], s.v[keep], s.dt, s.time),
            body_ids=tuple(prepared.body_ids[i] for i in keep), body_specs=tuple(prepared.body_specs[i] for i in keep),
            reference_ids=tuple(prepared.reference_ids[i] for i in keep), events=prepared.events+(event,))
    return prepared


def advance_prepared(prepared):
    """One configured macro dt, or no mutation on collision-mode rejection.

    All actual positive Verlet substeps obey half the existing start/end gravity
    and crossing limit. Contact is checked before computing endpoint forces.
    Event clipping uses the same drift polynomial and then integrates remainder.
    """
    if prepared.setup is None or prepared.setup.collision_mode == 'none':
        prepared.system.step()
        return AdvanceResult(prepared)
    validate_collision_state(prepared)
    policy = prepared.setup.collision_policy
    old = prepared.system
    trial = replace(prepared, system=System(old.masses.copy(), old.r.copy(), old.v.copy(), old.dt, old.time))
    end = old.time+old.dt
    if not math.isfinite(end) or end <= old.time:
        raise ValueError('Collision numerical halt: macro timestep cannot advance clock; state preserved.')
    event_frames = []
    attempts = 0
    while trial.system.time < end:
        s = trial.system
        immediate = _contacts(trial, np.zeros_like(s.r), 0., policy)
        if immediate:
            attempts += 1
            if attempts > policy.max_substeps:
                raise ValueError('Collision numerical halt: bounded response budget exceeded; state preserved.')
            before = StateFrame.capture(trial)
            old_count = len(trial.events)
            trial = _respond(trial, immediate)
            event_frames.append(replace(before, event_phase='pre', events=before.events+trial.events[old_count:]))
            event_frames.append(replace(StateFrame.capture(trial), event_phase='post'))
            continue
        remaining = end-s.time
        h = min(remaining, old.dt*policy.macro_fraction, s.resolution_limit()*policy.guard_fraction)
        if not math.isfinite(h) or h <= 0 or s.time+h <= s.time:
            raise ValueError('Collision numerical halt: substep cannot resolve encounter / advance clock; state preserved.')
        acceleration = s.accelerations()
        while True:
            attempts += 1
            if attempts > policy.max_substeps:
                raise ValueError('Collision numerical halt: bounded substep/refinement budget exceeded; state preserved.')
            contacts = _contacts(trial, acceleration, h, policy)
            elapsed = contacts[0][0] if contacts else h
            r = s.r+s.v*elapsed+.5*acceleration*elapsed**2
            vhalf = s.v+.5*acceleration*elapsed
            candidate = System(s.masses.copy(), r, vhalf, old.dt, s.time+elapsed)
            # Endpoint force is safe only after the sweep has clipped first contact.
            candidate.v = vhalf+.5*elapsed*candidate.accelerations()
            if not np.isfinite(candidate.r).all() or not np.isfinite(candidate.v).all():
                raise ValueError('Collision numerical halt: nonfinite Verlet state; state preserved.')
            if elapsed > candidate.resolution_limit()*policy.guard_fraction*(1+1e-12):
                h *= .5
                if s.time+h <= s.time:
                    raise ValueError('Collision numerical halt: endpoint encounter unresolved; state preserved.')
                continue
            break
        trial = replace(trial, system=candidate)
        if contacts:
            tolerance = policy.time_tolerance*max(1., abs(old.dt))
            simultaneous = []
            for contact in contacts:
                if contact[0] > elapsed+tolerance:
                    continue
                a, b = (trial.body_ids.index(key) for key in contact[1:])
                radius = sum(trial.body_specs[i].radius if trial.system.masses[i] > 0 else 0. for i in (a, b))
                distance = float(np.linalg.norm(candidate.r[a]-candidate.r[b]))
                # Time proximity alone cannot put a still-separated pair into
                # the contact group (especially when h << time_tolerance).
                if distance <= radius*(1+policy.distance_rtol)+64*np.finfo(float).eps*max(radius, distance):
                    simultaneous.append(contact)
            if not simultaneous:
                raise ValueError('Collision numerical halt: swept contact geometry unresolved; state preserved.')
            before = StateFrame.capture(trial)
            old_count = len(trial.events)
            trial = _respond(trial, simultaneous)
            event_frames.append(replace(before, event_phase='pre', events=before.events+trial.events[old_count:]))
            event_frames.append(replace(StateFrame.capture(trial), event_phase='post'))
        elif remaining == h:
            trial.system.time = end
    return AdvanceResult(trial, tuple(event_frames))


def mapped_selection(body_id, events):
    """Only an explicit merge event may transfer a selected identity."""
    for event in events:
        if isinstance(event, CollisionEvent) and body_id in event.parent_ids and event.remnant_id is not None:
            body_id = event.remnant_id
    return body_id
