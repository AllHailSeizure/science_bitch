"""Recorded formation inspection adapters; no calculation is launched here."""
from dataclasses import asdict

import numpy as np

from .formation import SynchronizedPlayback
from .history import StateFrame
from .model import BodySpec
from .playback import Playback
from .recording import RecordedRun


def formation_metadata(experiment):
    """Optional v1 fields, also usable by headless checkpoint exporters."""
    provenance = dict(experiment.provenance)
    change = provenance.get('change')
    if change is not None and not isinstance(change, dict):
        provenance['change'] = dict(asdict(change), kind='velocity')
    branch_index = next((i for i in range(len(experiment.samples) - 1, 0, -1)
                         if experiment.samples[i].phase == 'branch'), None)
    branch_point = None if branch_index is None else {
        'time': experiment.samples[branch_index].time,
        'common_state': asdict(experiment.samples[branch_index - 1]),
        'changed_state': asdict(experiment.samples[branch_index]),
    }
    return {
        'branch_point': branch_point,
        'assumptions': list(experiment.assumptions),
        'provenance': provenance,
        'formation_events': [asdict(event) for event in experiment.events],
        'removed_bodies': [asdict(body) for body in experiment.removed.values()],
    }


def formation_run(experiment):
    """Export a calculated prototype or branch as an ordinary playback member."""
    from .experiment_file import _formation_bodies, _formation_sample
    samples = [_formation_sample(sample, experiment.dt_max) for sample in experiment.samples]
    return dict(formation_metadata(experiment), status=experiment.status, reason=experiment.reason,
                setup_digest=experiment.provenance['setup_digest'],
                time_start=samples[0]['time'], time_end=samples[-1]['time'],
                bodies=_formation_bodies(experiment.samples), samples=samples, events=[])


def formation_playback(*experiments):
    """Package already calculated branches without changing or advancing them."""
    return {'schema': 'sidereel-playback-v1',
            'runs': [dict(formation_run(experiment), index=i) for i, experiment in enumerate(experiments)]}


def recorded(run):
    definitions = {body['id']: body for body in run.get('bodies', ())}
    frames = []
    for sample in run['samples']:
        ids = tuple(sample['ids'])
        specs = tuple(BodySpec(definitions.get(key, {}).get('name', key),
                               definitions.get(key, {}).get('kind', 'planet'),
                               (sample.get('mass') or [None] * len(ids))[i], None) for i, key in enumerate(ids))
        positions = np.asarray(sample['r'], dtype=float)
        frames.append(StateFrame(sample['time'], sample.get('dt', 0.), ids, specs,
                                 np.asarray(sample.get('mass', [0.] * len(ids))), positions,
                                 np.asarray(sample.get('v', np.zeros_like(positions))),
                                 event_phase=sample.get('phase') if sample.get('phase') in ('pre', 'post') else None))
    return RecordedRun(run.get('setup_digest', ''), '', run['time_end'], run['time_end'], tuple(frames),
                       run['status'] if run['status'] in ('complete', 'halted', 'cancelled') else 'cancelled')


class RecordedComparison(SynchronizedPlayback):
    """Adapt playback-file runs to the prototype's synchronized transport."""

    def __init__(self, left, right):
        self.left_playback = Playback(recorded(left))
        self.right_playback = Playback(recorded(right))
        self.left_times = tuple(frame.time for frame in self.left_playback.recording.frames)
        self.right_times = tuple(frame.time for frame in self.right_playback.recording.frames)
        self.common_start = max(self.left_times[0], self.right_times[0])
        self.common_end = min(self.left_times[-1], self.right_times[-1])
        if self.common_end < self.common_start:
            raise ValueError('The runs have no common simulated-time interval.')
        self.time = self.common_start
        self.playing = False
        self._place()


def contributions(run, body_id):
    for sample in reversed(run['samples']):
        if body_id in sample['ids']:
            return (sample.get('contributions') or {}).get(body_id, [])
    return []


def carriers(run, sample, body_id):
    if body_id not in run['samples'][-1]['ids']:
        return [body_id] if body_id in sample['ids'] else []
    parcels = {item['parcel_id'] for item in contributions(run, body_id)}
    return [key for key, amounts in sample.get('contributions', {}).items()
            if parcels.intersection(item['parcel_id'] for item in amounts)]


def history_indices(run, body_id):
    indices = [i for i, sample in enumerate(run['samples']) if body_id in sample['ids']]
    if indices and indices[-1] + 1 < len(run['samples']):
        indices.append(indices[-1] + 1)
    return indices
