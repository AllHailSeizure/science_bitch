# Population authoring integration implementation plan

**Goal:** Complete #6's general reference/preparation integration while retaining the merged reproducible test-particle generator and settings editor.

**Architecture:** Adapt existing `Population` and `PopulationEditor` to stable body references and the shared prepared system. Resolve an ID into the engine's numeric index only when building an initial state; continue to use the same generator and seeded PCG64 algorithm.

**Tech Stack:** Python, NumPy, Matplotlib, pytest; existing population APIs.

## Global Constraints

- “Test particles represent rocks whose gravitational influence is neglected: they feel every massive body but do not backreact or interact with one another.”
- “This issue should stay open until they are implemented and verified.”
- “Multiple-star support must state its limits rather than promise stable populations. 3D is deferred.”
- Requires the settled plan-2 authoring implementation. Reuse merged #11/#12; save/replay acceptance is owned by plans 5–6, not claimed complete here.

## File map

| File | Responsibility |
| --- | --- |
| Modify `simulations/star_system/model.py`, `initialization.py` | Stable `PopulationSpec` reference and prepared-state integration |
| Modify `simulations/star_system/populations.py` | Keep generator compatible; validate physical reference boundary |
| Create `simulations/star_system/population_editor.py` | Move the existing editor out of growing `plotting.py`; shared reference picker |
| Modify `simulations/star_system/plotting.py`, `sidebar.py`, `__main__.py` | Preparation flow and backward-compatible CLI defaults |
| Modify `tests/test_populations.py`, `tests/test_population_controls.py` | Multi-star/reference/identity and editor lifecycle checks |
| Modify `simulations/star_system/README.md` | Reference frame, seed, model and integration status |

### Task 1: Resolve population references by identity — mechanical

**Files:** `model.py`, `initialization.py`, `populations.py`, `tests/test_populations.py`.

**Interfaces:** Consume plan 2 `InitialSetup`/`PreparedSystem`; produce `PopulationSpec(reference_id, settings)` resolution before `with_population(system, settings, reference_radius=...)`.

- [ ] Write `test_population_reference_survives_body_reordering`, `test_population_around_second_star_uses_its_state`, and `test_missing_population_reference_is_recoverable`. Use two positive-mass stars at different positions/velocities; put the ring around the second and verify radii and relative speeds against that star, not row zero. Reorder metadata/bodies and compare outputs by stable ID.
- [ ] Run `python -m pytest -q tests/test_populations.py -k 'reference_survives or second_star or missing_population'`; expect unsupported general reference resolution.
- [ ] Resolve the stable ID to the prepared massive-body row, use its physical radius in AU, and pass a copied `Population` with the resolved integer reference. Preserve every existing massive body's position/velocity exactly. Reject deleted, unknown, zero-mass, or unknown-radius references with an actionable preparation error; retain the previous valid setup. Do not silently choose a new star.
- [ ] Preserve ring/disk/belt distribution, seeds, random/even angles, deterministic IDs, the 2,000-particle total limit, and no particle–particle forces. The shared wrapper remains an initialization operation: live population edits explicitly rebuild the initial system, as in #12. Do not mislabel it as live body insertion.
- [ ] Re-run the focused tests and `python -m pytest -q tests/test_populations.py`; expect pass, including the existing matched Jupiter perturbation check. Commit exact files with `git commit -m "Resolve population references through authored body identities"`.

### Task 2: Integrate the existing population editor — mechanical

**Files:** `population_editor.py`, `plotting.py`, `sidebar.py`, `__main__.py`, `tests/test_population_controls.py`, README.

**Interfaces:** Consume the settled shared reference selector and preparation model from plan 2; preserve existing PopulationEditor's apply/remove/cancel lifecycle.

- [ ] Extend editor tests to choose a second star, label its name and frame, reopen applied settings, edit a seed, cancel after time advances, and reject deletion of the selected reference without damaging the draft. Add an actual Matplotlib event test proving selection of the reference cannot also place a body in the scene.
- [ ] Run `python -m pytest -q tests/test_population_controls.py`; expect the new reference cases to fail before adaptation.
- [ ] Extract the existing editor into its own module without redesigning its settled controls. Populate the shared reference selector with eligible massive bodies. Show reference-relative AU and prograde circular velocity assumptions. In preparation, Apply updates the draft without starting time; in live mode retain clearly labeled `Apply and restart`/`Remove and restart`. Cancellation retains the evolved state and previous pause state.
- [ ] Keep the existing `--population`, `--inner`, `--outer`, `--seed`, `--angles`, and `--count` CLI behavior as the star-relative convenience path. General references arrive through the shared setup representation and later `--load`, rather than inventing a second CLI-only authoring system.
- [ ] Run `python -m pytest -q tests/test_population_controls.py tests/test_populations.py`, then `python -m pytest -q --tb=short`, `python -m simulations.star_system --population ring --inner 3.2 --count 64 --angles even --seed 7 --years 1 --output output/issue-6-ring.png`, and `git diff --check`; expect pass.
- [ ] Document the new supported reference flow and remaining persistence/replay dependencies. Report native multiwindow verification only if actually performed; no manual gate. Commit reviewed paths with `git commit -m "Integrate population settings with general system preparation"` and deliver a PR referencing #6 without closing it.

## Acceptance handoff

Plan 5 must save/reload the full settings and seed and compare regenerated initial particle states. Plan 6 must include populations in calculation/replay, with recorded time, complete/halted status, pause, and step. Preserve these requirements in those PR checklists; an API that merely accepts a nonzero reference index does not finish #6.
