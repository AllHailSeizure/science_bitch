"""Copied, read-only world-frame snapshots shared by trails and future playback."""
from dataclasses import dataclass
import numpy as np

MAX_HISTORY = 1500

@dataclass(frozen=True)
class StateFrame:
    time: float
    dt: float
    body_ids: tuple
    body_specs: tuple
    masses: np.ndarray
    r: np.ndarray
    v: np.ndarray
    events: tuple = ()
    reference_ids: tuple[str | None, ...] = ()
    event_phase: str | None = None

    def __post_init__(self):
        if self.event_phase not in (None, 'pre', 'post'):
            raise ValueError('Event frame phase must be pre, post, or absent.')
        for name in ('masses', 'r', 'v'):
            # Immutable bytes backing prevents callers re-enabling write flags.
            value = np.asarray(getattr(self, name), dtype=float)
            frozen = np.frombuffer(value.tobytes(), dtype=float).reshape(value.shape)
            object.__setattr__(self, name, frozen)
        for name in ('body_ids', 'body_specs', 'events', 'reference_ids'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.reference_ids:
            object.__setattr__(self, 'reference_ids', (None,)*len(self.body_ids))
        if len(self.body_ids) != len(self.masses) or len(self.body_specs) != len(self.masses) or self.r.shape != (len(self.masses), 2) or self.v.shape != self.r.shape:
            raise ValueError('History identities, properties and numeric rows must align.')
        if len(self.reference_ids) != len(self.body_ids):
            raise ValueError('History reference metadata must align with body identities.')

    @classmethod
    def capture(cls, prepared):
        s = prepared.system
        return cls(s.time, s.dt, prepared.body_ids, prepared.body_specs, s.masses, s.r, s.v, prepared.events, prepared.reference_ids)


@dataclass(frozen=True)
class RelativeState:
    """Current world-frame difference; a missing reference remains explicit."""
    reference_id: str
    reference_name: str | None
    r: tuple[float, float] | None
    v: tuple[float, float] | None

    @property
    def missing(self):
        return self.r is None

    @property
    def separation(self):
        return None if self.missing else float(np.linalg.norm(self.r))


def relative_state(frame, body_id):
    """Inspect any displayed StateFrame without consulting an initial setup."""
    index = frame.body_ids.index(body_id)
    reference = frame.reference_ids[index]
    if reference is None:
        return None
    if reference not in frame.body_ids:
        return RelativeState(reference, None, None, None)
    parent = frame.body_ids.index(reference)
    return RelativeState(reference, frame.body_specs[parent].name,
                         tuple(frame.r[index]-frame.r[parent]), tuple(frame.v[index]-frame.v[parent]))
