# Experiment preparation and persistence implementation plan

**Goal:** Implement #8's editable named scenarios and reproducible saved starting setups, including #6 populations and #9 moons.

**Architecture:** Make `InitialSetup` the immutable source for preparation, restart, and save/reload. Add a versioned JSON codec and preset registry around it; leave mutable evolved state and recorded results separate, and validate a loaded draft fully before replacing the current experiment.

**Tech Stack:** Python dataclasses, standard-library JSON/pathlib/hashlib, NumPy, Matplotlib, pytest; no new packages.

## Global Constraints

- “Editing a scenario produces the user's own experiment without altering the supplied preset.”
- “Saving after evolution begins makes clear which starting setup will be restored, and distinguishes that setup from recorded results or evolved state.”
- “Unsupported or invalid saves/configurations explain the problem and leave the current experiment recoverable.”
- Requires plans 2–4. Calculate-first playback is separate plan 6; this PR does not close #8.

## File map

| File | Responsibility |
| --- | --- |
| Create `simulations/star_system/persistence.py` | Versioned setup codec, validation, atomic replacement writes |
| Create `simulations/star_system/scenarios.py` | Immutable named preset factories and descriptions |
| Create `simulations/star_system/experiments.py` | Preparation/live state transitions and initial-setup ownership |
| Modify `simulations/star_system/model.py`, `initialization.py` | Complete model/settings identity and build validation |
| Modify `simulations/star_system/plotting.py`, `sidebar.py`, `__main__.py` | Prepare/load/save/preset controls and CLI loading |
| Create `tests/test_experiment_persistence.py`, `tests/test_experiment_controls.py` | Roundtrips, deterministic rebuild, invalid saves, UI ownership |
| Modify `simulations/star_system/README.md` | Save semantics, schema/model/version compatibility, controls |

### Task 1: Versioned starting-setup codec — judgment

**Files:** `persistence.py`, `model.py`, `initialization.py`, `tests/test_experiment_persistence.py`, README.

**Interfaces:** Produce `save_setup(setup: InitialSetup, path: Path) -> None`, `load_setup(path: Path) -> InitialSetup`, and `setup_digest(setup) -> str`. Consume the semantics settled in plan 2; do not decide live-insertion persistence implicitly here.

- [ ] Specify schema version 1 with explicit units (`AU`, `solar_mass`, `Julian_year`), name/preset provenance, instance IDs, all physical values and overrides, relative reference IDs/vectors, resolved numerical settings/integrator identifier, population generator/version/settings/seed, and collision mode. Save complete values, not just catalog keys whose values could change. Store provenance as data; never execute data from a save.
- [ ] Resolve plan 2's live-insertion save/restart decision before defining its schema. Whatever was chosen must be explicit in the save action and reproducible: either save the original initial setup with a clear account of excluded live changes, or save a separately named starting setup with declared epoch/state. Do not silently mix an original t=0 body list with later evolved positions or insertions. Update this plan to the chosen concrete contract before delegating codec work.
- [ ] Add parameterized roundtrips for an ordinary system, a two-star setup, a seeded population around the second star, and a planet/moon setup with an edited velocity. Require equality of restored initial `r`, `v`, masses, IDs and generation settings on the same supported environment. Advance rebuilt systems 50 identical steps and compare states with `rtol=1e-12, atol=1e-12`; document deterministic same-version expectations and no universal cross-version bitwise guarantee.

```python
save_setup(setup, path)
loaded = load_setup(path)
first, second = build_system(setup), build_system(loaded)
assert first.body_ids == second.body_ids
np.testing.assert_array_equal(first.system.r, second.system.r)
np.testing.assert_array_equal(first.system.v, second.system.v)
assert setup_digest(setup) == setup_digest(loaded)
```

- [ ] Run `python -m pytest -q tests/test_experiment_persistence.py`; expect missing codec. Implement canonical JSON with `allow_nan=False`, deterministic key ordering for the digest, strict finite/range/reference/type validation, and explicit unsupported-version errors. Unknown physical/model settings cannot be ignored. Reject malicious/non-JSON content without eval/pickle.
- [ ] Write a temporary sibling file and replace the destination only after encoding/writing succeeds; clean up only that tool-owned temporary file. Add interrupted-write/invalid-save tests proving an existing valid file and the running experiment survive. No recursive cleanup.
- [ ] Re-run the focused command; expect pass. Commit exact reviewed files with `git commit -m "Save and restore versioned experiment initial setups"`.

### Task 2: Shared preparation, editable presets, and UI integration — judgment

**Files:** `scenarios.py`, `experiments.py`, `plotting.py`, `sidebar.py`, `__main__.py`, `tests/test_experiment_controls.py`, README.

**Interfaces:** Produce `ExperimentSession.initial_setup`, `draft`, and `prepared` state ownership; `get_scenario(key) -> InitialSetup` returns an independent instance. Consume `App.set_experiment_name` and the shared body/population/moon editors.

- [ ] Add tests for opening a named preset, editing it into a custom experiment, canceling drafts, invalid load recovery, saving after time advances, and restarting from saved setup. Test that changing a copy cannot mutate the preset registry or another session.
- [ ] Run `python -m pytest -q tests/test_experiment_controls.py`; expect missing preparation/persistence flow.
- [ ] Register the current Sun/Earth/Jupiter configuration as the compatibility preset with a factual description of its circular initial guesses. Provide free preparation and a name field through the same sidebar. Plan 7 adds the required Sun–Jupiter experiment to this registry; do not build a separate scenario UI there.
- [ ] Start preparation paused with no time advancement. Body, population, and moon edits update a validated draft, not live arrays. Applying a new initial setup makes a custom experiment and invalidates its association with old calculations; preserve old recorded results for separate inspection once plan 6 exists. Keep a clear distinction between `Prepare`, `Run live`, and later `Calculate`.
- [ ] Wire Save/Load to the codec using explicit path controls or the existing GUI backend's file dialogs without adding a dependency. Failed load leaves draft/current state intact. Update the heading from setup name and distinguish preset origin from the editable user's copy.
- [ ] Add `--load PATH` to the existing CLI. Reject ambiguous combinations of `--load` with legacy body/population construction flags rather than silently overriding file contents. Keep `--years`/`--output` usable on a loaded setup through general `draw_system`; retain all old export commands.
- [ ] Run `python -m pytest -q tests/test_experiment_controls.py tests/test_experiment_persistence.py`; expect pass. Commit exact paths with `git commit -m "Add shared experiment preparation and editable presets"`.

### Task 3: Cross-issue persistence verification — mechanical

**Files:** Persistence/control tests and README.

**Interfaces:** Consume Tasks 1–2; deliver save/reload acceptance for bodies, seeds, multiple stars and moons, plus plan 6's setup identity boundary.

- [ ] Add subprocess roundtrip coverage: use `tmp_path` to save a moon/population setup; invoke `[sys.executable, '-m', 'simulations.star_system', '--load', str(path), '--years', '0.01', '--output', str(image)]` in a new process and assert return code 0 and readable nonempty PNG. A new process proves the save does not depend on in-memory restart configuration.
- [ ] Run `python -m pytest -q tests/test_experiment_persistence.py tests/test_experiment_controls.py`, then `python -m pytest -q --tb=short` and `git diff --check`; expect pass.
- [ ] Document exact restored content, unsupported schema behavior, numerical/model version constraints, and distinction between an initial setup, evolved state, and recorded results. Commit with `git commit -m "Verify experiment persistence across app restarts"` and deliver a PR referencing #8, #6, and #9. Collision mode `none` remains explicit until plan 8 adds its supported mode.
