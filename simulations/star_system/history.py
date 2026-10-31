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

    def __post_init__(self):
        for name in ('masses', 'r', 'v'):
            # Immutable bytes backing prevents callers re-enabling write flags.
            value = np.asarray(getattr(self, name), dtype=float)
            frozen = np.frombuffer(value.tobytes(), dtype=float).reshape(value.shape)
            object.__setattr__(self, name, frozen)
        for name in ('body_ids', 'body_specs', 'events'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.body_ids) != len(self.masses) or len(self.body_specs) != len(self.masses) or self.r.shape != (len(self.masses), 2) or self.v.shape != self.r.shape:
            raise ValueError('History identities, properties and numeric rows must align.')

    @classmethod
    def capture(cls, prepared):
        s = prepared.system
        return cls(s.time, s.dt, prepared.body_ids, prepared.body_specs, s.masses, s.r, s.v, prepared.events)
