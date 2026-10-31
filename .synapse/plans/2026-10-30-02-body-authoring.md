# Body authoring implementation plan

**Goal:** Resolve #5 by supporting sourced/custom bodies, arbitrary initial vectors, multiple stars, and state-preserving live insertion.

**Architecture:** Add immutable setup/identity metadata beside the numeric `System`; retain the existing solver and catalog construction as compatibility paths. A shared draft editor resolves display placement into physical world coordinates and commits only after validation; rendering uses body metadata rather than assuming one star followed by planets.

**Tech Stack:** Existing Python/NumPy/Matplotlib stack, dataclasses, pytest; no added packages.

## Global Constraints

- “Live insertion preserves all existing bodies' positions and velocities and the elapsed simulation time at insertion.”
- “Configured initial conditions remain distinguishable from evolved state.”
- “3D is deferred.”
- “No architecture or integrator is prescribed, and arbitrary starting conditions do not guarantee capture or stable orbits.”
- Requires plan 1. [Shared contracts](2026-10-30-00-issue-resolution-order.md) name the new APIs.

## File map

| File | Responsibility |
| --- | --- |
| Create `simulations/star_system/model.py` | `BodySpec`, `InitialBody`, `InitialSetup`, `PopulationSpec`, validation |
| Create `simulations/star_system/initialization.py` | Resolve references, build numeric states, insertion, timestep policy |
| Create `simulations/star_system/authoring.py` | Draft body editor, placement, velocity handles, confirmation/cancel |
| Create `simulations/star_system/history.py` | Copied world-frame history with stable identities |
| Modify `simulations/star_system/data.py` | General sourced catalog plus legacy adapters |
| Create `simulations/star_system/catalog-sources.json` | Field-level provenance, units, source versions and uncertainty |
| Modify `simulations/star_system/core.py` | Supported-input/encounter validation required by arbitrary states |
| Modify `simulations/star_system/plotting.py`, `sidebar.py`, `rendering.py` | Shared authoring, generic identities, multi-star presentation |
| Create `tests/test_body_authoring.py`, `tests/test_catalog_provenance.py` | Physical state preservation, validation, provenance, event interactions |
| Modify `tests/test_star_system.py`, `tests/test_star_system_display.py` | Compatibility and multi-star regressions |
| Modify `simulations/star_system/README.md` | Authoring semantics, units, limits, sourced catalog |

### Task 1: Settle authoring semantics and sourced definitions — judgment

**Files:** `model.py`, `data.py`, `catalog-sources.json`, `tests/test_catalog_provenance.py`, README, this plan if decisions change.

**Interfaces:** Produce immutable `BodySpec` and `InitialBody` contracts from the index; catalog keys identify definitions, instance IDs identify bodies.

- [ ] Resolve these issue-owned choices inline before delegating implementation: repeated catalog instances, default reference selection in multi-star systems, edit/remove behavior after evolution, and save/restart behavior after live insertion. Proposed defaults are unique instances for repeated choices, current massive selection as an explicitly shown reference (otherwise require selection), and edit/remove in preparation only. Live insertion modifies the active trajectory; initial-setup saving must visibly distinguish that from preparation. Ask the owner about the unresolved save/restart semantics rather than silently treating the evolved state as the original setup. Record the chosen behavior and update dependent persistence tests before continuing that part.
- [ ] Propose a concrete initial expansion: retain all current entries; add Alpha Centauri A/B and Sirius B, measured exoplanets 51 Pegasi b and HD 209458 b, the Moon/Io/Europa/Ganymede/Callisto/Titan, and Vesta/Eros/Bennu. Confirm entry choice if the owner wants different examples. Research authoritative NASA/JPL/ESA archives and primary measurement papers; capture source URL, retrieved/source version, original units, conversion, uncertainty, and estimated-vs-measured status per physical field. Do not invent missing exoplanet radii or replace minimum masses with true masses.
- [ ] Define missing-value behavior: show unknown values, allow the user to supply a clearly marked custom value, and block a capability that requires an unavailable physical value. A display marker may still exist without a collision radius, but physical contact cannot be claimed for it. Preserve source metadata when a catalog definition is copied; edited values become user-authored overrides.
- [ ] Write `test_catalog_categories_have_traceable_properties` and `test_unknown_property_is_not_silently_estimated`. Run `python -m pytest -q tests/test_catalog_provenance.py`; expect missing general definitions/provenance, then pass after adding reviewed records and validation. Check finite positive known mass/radius, required stellar temperature for temperature-based coloring, and source attribution rather than asserting unresearched numerical constants.
- [ ] Commit the settled metadata/catalog changes with `git add simulations/star_system/model.py simulations/star_system/data.py simulations/star_system/catalog-sources.json simulations/star_system/README.md tests/test_catalog_provenance.py`; `git commit -m "Add sourced body definitions and explicit authoring semantics"`.

### Task 2: General initial states and live insertion — judgment

**Files:** `model.py`, `initialization.py`, `history.py`, `core.py`, `tests/test_body_authoring.py`, `tests/test_star_system.py`, README.

**Interfaces:** Produce `build_system(setup) -> PreparedSystem`, `insert_body(prepared, body) -> PreparedSystem`, and `circular_relative_velocity(relative_position, reference_mass, body_mass) -> ndarray`. Consume immutable setup definitions. Legacy `make_system` remains callable.

- [ ] Write tests for arbitrary Cartesian states, one isolated body, a two-star system with no planet, reference chains/cycles, finite values, distinct IDs, shared catalog definitions, and exact live insertion preservation. Use this numerical assertion in the insertion test after advancing the original system five steps:

```python
before_r, before_v = before.system.r.copy(), before.system.v.copy()
after = insert_body(before, addition)
np.testing.assert_array_equal(after.system.r[:len(before_r)], before_r)
np.testing.assert_array_equal(after.system.v[:len(before_v)], before_v)
assert after.system.time == before.system.time
assert not np.shares_memory(after.system.r, before.system.r)
```

- [ ] Run `python -m pytest -q tests/test_body_authoring.py`; expect missing initialization/insertion APIs. Resolve relative input recursively once, reject absent/cyclic references, and keep arbitrary world coordinates unchanged. Do not reuse `make_system`'s automatic barycentric shift during insertion. Explicit binary presets may construct a barycentric state from the start.
- [ ] The circular aid uses the existing formula `sqrt(G * (M + m) / |r|)` in the perpendicular prograde direction, then adds the reference's current velocity. Show reference, physical speed, and the two-body assumption. Reject zero separation and invalid/missing positive mass. It is an initial aid, never a continuing constraint.
- [ ] Settle and document numerical resolution using isolated binary and high-speed flyby tests. Preserve the existing period/600 choice for legacy circular setups. For arbitrary setups use an explicit fixed step validated against pair gravitational and crossing timescales; reject unresolved inputs recoverably. A live insertion may require a smaller new fixed step: report the change, preserve positions/velocities/time exactly, record an insertion/step-change event, and start a new integration segment. Never claim the entire edited trajectory is one unmodified fixed-step Hamiltonian integration. Keep normal evolution fixed-step and retain the unresolved-encounter halt.
- [ ] Ensure rejected steps/insertion do not partly mutate state or advance time. Add a fast-flyby check that cannot bypass validation solely because the gravitational timescale is long. Use step-halving convergence on the explicit binary/flyby fixtures to establish supported tolerances; record measured bounds, not an arbitrary guarantee for every custom input.
- [ ] Store copied world-frame `StateFrame` samples, preserving the existing 1,500-sample history limit and 400-vertex display bound. Retain legacy trail adapters until their tests migrate. Clear/segment only the new body's nonexistent past; never draw a trail connecting different IDs across insertion.
- [ ] Re-run `python -m pytest -q tests/test_body_authoring.py tests/test_star_system.py tests/test_populations.py`; expect pass. Commit explicitly reviewed paths with `git commit -m "Support authored initial states and state-preserving insertion"`.

### Task 3: Shared placement, vectors, and multiple-star rendering — judgment

**Files:** `authoring.py`, `plotting.py`, `sidebar.py`, `rendering.py`, `tests/test_body_authoring.py`, `tests/test_star_system_display.py`, README.

**Interfaces:** Produce `BodyEditor` draft lifecycle and general `draw_system`; consume `PreparedSystem`, stable IDs, and world-frame history. Population and moon plans reuse this editor/reference interaction.

- [ ] Add event-based tests for Add→pause→catalog/custom selection→placement→arrow drag→confirm, cancel, invalid drafts, preparation edit/remove, and insertion while following an existing body. Run `python -m pytest -q tests/test_body_authoring.py -k 'editor or placement or arrow or cancel'`; expect missing UI behavior.
- [ ] Convert click coordinates back through the compressed radial mapping before setting physical position. The inverse of `coordinates(r)` is `q * (0.05 * sinh(norm(q)) / norm(q))`, with a zero-vector special case; then restore the current view origin/frame. Round-trip test near zero, at 1 AU, and at 80 AU. Do not interpret compressed axis units as AU.
- [ ] Draw velocity arrows in a dedicated local linear velocity scale, with an explicit AU/year legend and numeric components. Dragging the tip changes both magnitude and direction; double the represented speed and assert double arrow length at unchanged camera/scale. Position mapping and velocity scaling are different transforms. Add reference velocity exactly once when committing.
- [ ] Generalize renderer iteration over stable metadata. Render every star luminously, every nonstar with its own physical/display properties, and inspector values from the selected ID. Propose illumination from the dominant apparent stellar flux using available radius/temperature with an explicit illustrative fallback; settle this visual rule inline and test that it follows the chosen star rather than index zero. Do not add radiation forces or thermal evolution.
- [ ] Keep scene/camera intact on insertion; append artists/history identities as necessary. Cancel restores the previous running/paused state without changing physical state. Confirm leaves playback paused until Resume. Preparation edit/remove validates the complete draft, including dependent moon/population references, before replacing it.
- [ ] Run focused tests, then `python -m pytest -q --tb=short`, `python -m simulations.star_system --star sun --planet earth --planet jupiter --output output/issue-5-compatibility.png`, and `git diff --check`; expect pass. Document catalog sources, units, reference frames, numerical limits, and the settled edit/save semantics. Commit with explicit paths and `git commit -m "Add shared body placement and velocity authoring"`.
- [ ] Deliver the #5 PR and its settled interface/diff to plans 3–4. Do not implement those adaptations concurrently with this task.
