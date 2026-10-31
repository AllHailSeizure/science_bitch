"""Own the selected starting epoch separately from drafts and mutable live state."""
from dataclasses import replace

from .initialization import build_system, recommended_moon_step
from .model import InitialBody, InitialSetup
from .persistence import load_setup, save_setup, setup_digest, _population


def custom_copy(setup, original):
    origin = (setup.preset_id or original.preset_id
              or setup.provenance.get('origin_preset') or original.provenance.get('origin_preset'))
    provenance = dict(setup.provenance)
    if origin:
        provenance['origin_preset'] = origin
    return replace(setup, preset_id=None, provenance=provenance)


def capture_current_setup(prepared, initial_setup, name):
    """New epoch with explicit particles and current parent-relative vectors.

    This restarts a numerical segment at t=0; it is not trajectory continuation.
    """
    if not isinstance(name, str) or not name.strip() or name.strip() == initial_setup.name.strip():
        raise ValueError('Give the captured state a different, nonempty setup name.')
    system, bodies, dt = prepared.system, [], prepared.system.dt
    for i, (key, spec, reference) in enumerate(zip(prepared.body_ids, prepared.body_specs, prepared.reference_ids)):
        r, v = system.r[i].copy(), system.v[i].copy()
        if reference is not None:
            if reference not in prepared.body_ids:
                raise ValueError(f'Cannot capture {key}: parent {reference} is missing.')
            parent = prepared.body_ids.index(reference)
            r -= system.r[parent]
            v -= system.v[parent]
            if spec.kind == 'moon':
                dt = min(dt, recommended_moon_step(r, system.masses[parent], system.masses[i]))
        bodies.append(InitialBody(key, spec, tuple(r), tuple(v), reference))
    provenance = {'source_name': initial_setup.name, 'source_setup_digest': setup_digest(initial_setup),
                  'source_time': system.time, 'source_dt': system.dt,
                  'source_population': _population(initial_setup.population), 'captured_dt': dt,
                  'capture_note': 'New initial epoch at time zero; not uninterrupted trajectory continuation.'}
    if dt < system.dt:
        provenance['capture_note'] += ' Smaller fixed dt required by current moon period / 600.'
    if initial_setup.preset_id or initial_setup.provenance.get('origin_preset'):
        provenance['origin_preset'] = initial_setup.preset_id or initial_setup.provenance['origin_preset']
    captured = InitialSetup(name.strip(), tuple(bodies), dt, provenance=provenance)
    build_system(captured)
    return captured


class ExperimentSession:
    def __init__(self, setup):
        self.initial_setup = setup
        self.prepared = build_system(setup)
        self.draft = setup

    def select(self, setup):
        prepared = build_system(setup)
        self.initial_setup = self.draft = setup
        self.prepared = prepared
        return prepared

    def begin_draft(self):
        self.draft = self.initial_setup
        return self.draft

    def update_draft(self, setup):
        build_system(setup)
        self.draft = custom_copy(setup, self.initial_setup)

    def apply_draft(self):
        return self.select(self.draft)

    def cancel_draft(self):
        self.draft = self.initial_setup

    def replace_initial(self, setup):
        self.update_draft(setup)
        return self.apply_draft()

    def restart(self):
        self.prepared = build_system(self.initial_setup)
        self.draft = self.initial_setup
        return self.prepared

    def save_initial(self, path):
        save_setup(self.initial_setup, path)

    def save_current(self, path, name):
        captured = capture_current_setup(self.prepared, self.initial_setup, name)
        save_setup(captured, path)
        return captured

    def load(self, path):
        return self.select(load_setup(path))
