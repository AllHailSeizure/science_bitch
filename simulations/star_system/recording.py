"""Bounded calculation of immutable, timestamped world states using the live solver."""
from dataclasses import dataclass, fields, is_dataclass
import math
import sys
import time
from collections.abc import Mapping

import numpy as np

from .core import System
from .history import StateFrame
from .initialization import build_system
from .persistence import setup_digest
from .model import InitialSetup

DEFAULT_MAX_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class CalculationRequest:
    duration_years: float
    sample_stride: int = 1
    max_bytes: int = DEFAULT_MAX_BYTES

    def __post_init__(self):
        try:
            valid_duration = not isinstance(self.duration_years, bool) and math.isfinite(self.duration_years) and self.duration_years > 0
        except (TypeError, OverflowError):
            valid_duration = False
        if not valid_duration:
            raise ValueError('Duration must be finite and positive, in years.')
        for name in ('sample_stride', 'max_bytes'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')


@dataclass(frozen=True)
class RecordedRun:
    setup_digest: str
    source_name: str
    requested_end_time: float
    resolved_end_time: float
    frames: tuple[StateFrame, ...]
    status: str
    reason: str = ''
    requested_steps: int = 0
    completed_steps: int = 0
    sample_stride: int = 1
    source_setup: InitialSetup | None = None

    def __post_init__(self):
        object.__setattr__(self, 'frames', tuple(self.frames))
        if self.status not in ('complete', 'halted', 'cancelled') or not self.frames:
            raise ValueError('A recording needs frames and an explicit terminal status.')
        if any(b.time < a.time for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError('Recorded sample times must be nondecreasing.')

    @property
    def actual_end_time(self):
        return self.frames[-1].time


def frame_storage_bytes(frame):
    """Conservative retained Python/NumPy storage, including immutable byte buffers.

    Shared definitions are charged again per frame, leaving margin rather than
    promising that raw numeric array sizes equal the entire recording cost.
    """
    seen = set()

    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        result = sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            return result + (size(value.base) if value.base is not None else 0)
        if is_dataclass(value):
            return result + sys.getsizeof(getattr(value, '__dict__', {})) + sum(size(getattr(value, f.name)) for f in fields(value))
        if isinstance(value, Mapping):
            # MappingProxyType does not report the size of the wrapped dict.
            return result + sys.getsizeof(dict(value)) + sum(size(k)+size(v) for k, v in value.items())
        if isinstance(value, (tuple, list)):
            return result + sum(size(v) for v in value)
        return result

    return size(frame) + 64  # frame-list/tuple slots and growth margin


class Calculation:
    """Incremental runner. advance is bounded by step count and optional wall budget.

    A checkpoint is kept for every accepted whole step, even off the output
    stride, so cancellation/halting always includes the last valid state.
    """
    def __init__(self, setup, request):
        self.request = request
        self.source_setup = setup
        self.prepared = build_system(setup)
        self.source_name, self.setup_digest = setup.name, setup_digest(setup)
        ratio = request.duration_years / self.prepared.system.dt
        if not math.isfinite(ratio):
            raise ValueError('Duration exceeds the supported step count; reduce duration or increase output stride.')
        self.total_steps = max(1, math.ceil(math.nextafter(ratio, -math.inf)))
        if self.total_steps*self.prepared.system.dt < request.duration_years:
            self.total_steps += 1
        self.resolved_end_time = self.total_steps*self.prepared.system.dt
        self.completed_steps, self.status, self.reason = 0, None, ''
        self._checkpoint = StateFrame.capture(self.prepared)
        self._frames = [self._checkpoint]
        self._checkpoint_bytes = frame_storage_bytes(self._checkpoint) + 256
        self._bytes = self._checkpoint_bytes
        # Reserve an off-stride terminal checkpoint as well as all output samples.
        self.estimated_bytes = self._bytes * (2 + math.ceil(self.total_steps/request.sample_stride))
        if self.estimated_bytes > request.max_bytes:
            raise ValueError(f'Estimated recording storage {self.estimated_bytes/1048576:.2f} MiB exceeds '
                             f'{request.max_bytes/1048576:.2f} MiB. Reduce duration or body/count settings, '
                             'increase sample stride, or choose a larger recording budget.')

    @property
    def done(self):
        return self.status is not None

    @property
    def progress(self):
        return self.completed_steps/self.total_steps

    @property
    def result(self):
        if not self.done:
            return None
        return RecordedRun(self.setup_digest, self.source_name, self.request.duration_years,
                           self.resolved_end_time, tuple(self._frames), self.status, self.reason,
                           self.total_steps, self.completed_steps, self.request.sample_stride, self.source_setup)

    def _finish(self, status, reason=''):
        if self._frames[-1] is not self._checkpoint:
            self._frames.append(self._checkpoint)
        self.status, self.reason = status, reason

    def cancel(self):
        if not self.done:
            self._finish('cancelled', 'Cancelled by user; partial recording retained.')
        return self.result

    def advance(self, max_steps, budget_seconds=None):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError('max_steps must be a nonnegative integer.')
        if budget_seconds is not None and (not math.isfinite(budget_seconds) or budget_seconds <= 0):
            raise ValueError('Wall budget must be finite and positive.')
        deadline = None if budget_seconds is None else time.perf_counter()+budget_seconds
        advanced = 0
        while not self.done and advanced < max_steps:
            try:
                self.prepared.system.step()
            except ValueError as exc:
                self._finish('halted', str(exc))
                break
            candidate = StateFrame.capture(self.prepared)
            changed = candidate.events != self._checkpoint.events or candidate.body_ids != self._checkpoint.body_ids
            same_metadata = all(getattr(candidate, name) is getattr(self._checkpoint, name)
                                for name in ('body_ids', 'body_specs', 'reference_ids', 'events'))
            # Fixed membership/metadata means fixed buffer sizes. Avoid walking
            # thousands of shared particle definitions on every physics step.
            candidate_bytes = self._checkpoint_bytes if same_metadata else frame_storage_bytes(candidate) + 256
            # Future event steps retain both sides even between stride samples.
            additions = ([self._checkpoint] if changed and self._frames[-1] is not self._checkpoint else [])
            sampled = changed or (self.completed_steps+1) % self.request.sample_stride == 0
            required = (self._checkpoint_bytes if additions else 0) + candidate_bytes
            if self._bytes + required > self.request.max_bytes:
                previous = self._checkpoint
                from dataclasses import replace
                self.prepared = replace(self.prepared, system=System(previous.masses.copy(), previous.r.copy(),
                    previous.v.copy(), previous.dt, previous.time), body_ids=previous.body_ids,
                    body_specs=previous.body_specs, events=previous.events, reference_ids=previous.reference_ids)
                self._finish('halted', 'Recording memory budget reached. Reduce duration or body/count settings, '
                             'increase sample stride, or choose a larger budget. Last accepted step retained.')
                break
            for frame in additions:
                self._frames.append(frame)
                self._bytes += self._checkpoint_bytes
            self._checkpoint = candidate
            self._checkpoint_bytes = candidate_bytes
            self.completed_steps += 1
            advanced += 1
            if sampled:
                self._frames.append(candidate)
                self._bytes += candidate_bytes
            if self.completed_steps == self.total_steps:
                self._finish('complete')
            if deadline is not None and time.perf_counter() >= deadline:
                break
        return advanced
