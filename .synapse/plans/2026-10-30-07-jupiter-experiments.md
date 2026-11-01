# Sun–Jupiter experiments implementation plan

**Goal:** Resolve #7 with reproducible Sun–Jupiter ring experiments, a rotating view, selected-rock history, and matched no-Jupiter comparisons.

**Architecture:** Register experiments through the shared scenario/setup system and use the existing population physics. Apply reference-frame transformations only when presenting copied world-frame states; comparison construction changes the declared gravitational system while preserving matched rock initial conditions and elapsed time.

**Tech Stack:** Existing Python/NumPy/Matplotlib stack, shared persistence and recordings, pytest.

## Global Constraints

- “Outcomes arise from the declared gravitational model, with verification that rules out scripted attraction, clustering, or relocation.”
- “Arbitrary starting conditions need not yield Trojan occupancy or permanent capture.”
- “The reference experiment remains Sun-Jupiter rather than expanding its acceptance criteria to every possible stellar configuration.”
- Requires plans 3, 5, 6 and their shared body authoring. Keep live simulation available. Collision assumptions are explicit; plan 8 may later add the optional response.

## File map

| File | Responsibility |
| --- | --- |
| Modify `simulations/star_system/scenarios.py` | Named Sun–Jupiter ring and matched control construction |
| Create `simulations/star_system/frames.py` | Pure inertial/Sun-relative/rotating display transforms |
| Modify `simulations/star_system/rendering.py`, `plotting.py`, `sidebar.py` | Frame control, L4/L5 guides, selected-rock trajectory/inspection |
| Modify `simulations/star_system/history.py`, `recording.py` | Read existing state/timestamps; no separate physics path |
| Create `tests/test_jupiter_experiments.py`, `tests/test_reference_frames.py` | Reproducibility, matched evolution, transform/history checks |
| Modify `tests/test_populations.py`, `tests/test_recorded_playback.py` | Keep perturbation fixture and verify recorded integration |
| Modify `simulations/star_system/README.md` | Reproducible example, history duration, physical interpretation |

### Task 1: Reproducible preset and matched comparison — judgment

**Files:** `scenarios.py`, `tests/test_jupiter_experiments.py`, `tests/test_populations.py`, README.

**Interfaces:** Produce `get_scenario('sun-jupiter-ring')` and `make_jupiter_comparison(setup) -> tuple[InitialSetup, InitialSetup]`; consume ordinary `PopulationSpec`, persistence, and calculation APIs.

- [ ] Base the first documented fixture on the existing checked perturbation example: Sun/Jupiter, 32 test particles at 3.2 AU, evenly spaced angles, seed 7. Show these settings in the shared editor and name/description. This demonstrates perturbation; it is not a promise of Trojan capture. Users can change radius/count/positions/vectors through the common tools.
- [ ] Add tests for scenario copy independence, exact reproducible initial populations, save/reload, and equal-duration matched controls. Run `python -m pytest -q tests/test_jupiter_experiments.py`; expect missing scenario/control helpers.
- [ ] Construct the perturbation run first, then make the comparison from the same resolved Sun and rock positions/velocities, omitting Jupiter's gravitational source. Do not independently regenerate or barycentrically recenter the control. Use the same dt and integer integration-step count in both runs; compare star-relative rock state by stable IDs. Removing a Jupiter row requires matching by IDs, not array offsets.
- [ ] Preserve the existing test's independent physical check: one initial ring period of matched evolution produces a maximum rock-position displacement difference above 0.01 AU for its fixed fixture. Keep the existing mass-zero control test as an independent reference; do not tune initial particles to make the result pass.
- [ ] Explicitly save the control as its own experiment with its actual bodies and model settings. If one run halts early, show unequal completion and compare only their common recorded interval; never present them as equal elapsed-time outcomes after one stops.
- [ ] Re-run `python -m pytest -q tests/test_jupiter_experiments.py tests/test_populations.py`; expect pass. Commit reviewed paths with `git commit -m "Add reproducible Sun-Jupiter experiments and matched controls"`.

### Task 2: Time-consistent rotating view and reference guides — judgment

**Files:** `frames.py`, `rendering.py`, `plotting.py`, `sidebar.py`, `tests/test_reference_frames.py`, README.

**Interfaces:** Produce `rotate_about_sun(frame, sun_id, jupiter_id)` and inverse display-position mapping. Consume world-frame history and stable IDs. This module must not import or call force/integration functions.

- [ ] Add tests for Jupiter remaining on the positive horizontal axis, rotation preserving physical distances before compression, inverse mapping, and per-sample historical transforms. Run `python -m pytest -q tests/test_reference_frames.py`; expect missing frame helpers.
- [ ] Implement the transformation from each sample's own Sun/Jupiter state:

```python
delta = frame.r[jupiter_index] - frame.r[sun_index]
theta = np.arctan2(delta[1], delta[0])
c, s = np.cos(theta), np.sin(theta)
rotation = np.array([[c, s], [-s, c]])
relative = frame.r - frame.r[sun_index]
rotating = relative @ rotation.T
```

- [ ] Apply radial compression only after physical frame transformation. Transform each historical frame using its historical angle; applying today's angle to all old samples produces the wrong rotating trajectory. Use the same path in live and recorded views. Switching views cannot alter `System.r`, `System.v`, recorded arrays, or time.
- [ ] Draw L4/L5 geometric reference guides at ±60 degrees and the instantaneous Sun–Jupiter separation, following the equilateral geometry described by [NASA](https://science.nasa.gov/solar-system/resources/faq/what-are-lagrange-points/). Label them as ideal circular restricted-three-body reference locations, not detected occupied/stable regions. Verify their geometry and distinguish their artists/legend from particle markers. Explain eccentric/perturbed systems do not inherit a guaranteed stationary equilibrium from these guides.
- [ ] If either reference body is absent (including the no-Jupiter control) or the pair is coincident, disable that rotating view with a clear reason. Do not synthesize a hidden Jupiter force/reference. World/Sun-relative comparison remains available.
- [ ] Keep velocity authoring units explicit: a physical reference-relative inertial velocity is not the derivative of rotating coordinates. Route placement through the inverse spatial transform; rotate a physical velocity arrow's orientation for presentation without subtracting a frame speed unless a separately labeled rotating-velocity readout is implemented. Test that switching views preserves the authored physical vector exactly.
- [ ] Re-run focused tests; expect pass. Commit exact files with `git commit -m "Add time-consistent Sun-Jupiter rotating views"`.

### Task 3: Selected-rock trajectories and reproducible workflow — mechanical

**Files:** `rendering.py`, `plotting.py`, `sidebar.py`, Jupiter/playback tests, README.

**Interfaces:** Consume Tasks 1–2, copied histories, and the shared inspector; expose actual current rock–Jupiter separation and the displayed trajectory's time interval.

- [ ] Add `test_selected_particle_history_uses_stable_identity`, `test_recorded_rotating_history_matches_live_samples`, and `test_jupiter_workflow_preserves_saved_initial_conditions`. Exercise selection/follow, pause/step/restart live, and play/pause/next-sample recorded.
- [ ] Run `python -m pytest -q tests/test_jupiter_experiments.py tests/test_reference_frames.py tests/test_recorded_playback.py`; expect the new selected-rock integration tests to fail before wiring them.
- [ ] Draw a trail only for the selected particle in addition to existing major-body trails; use copied retained samples, timestamps and a maximum 400 display vertices. Label the actual history duration, including truncation/sampling. Show rock–Jupiter physical separation and reference-relative velocity from the displayed sample. Do not call temporary proximity or a short trajectory permanent capture.
- [ ] Use the existing shared scenario picker, body/vector editors, save/load, calculation, and live controls. Provide a factual comparison readout of equal elapsed time and run status; a new dashboard is not required. Record that arbitrary capture attempts may escape or halt under the declared model.
- [ ] Run `python -m pytest -q --tb=short` and `git diff --check`; expect pass. Add a subprocess export of the saved preset to `test_jupiter_workflow_preserves_saved_initial_conditions`. Document actual measured perturbation and all limits. Commit with `git commit -m "Expose selected-rock histories in Jupiter experiments"` and deliver the #7 PR with references to #6/#8.
