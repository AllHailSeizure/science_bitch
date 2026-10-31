# Simplified inelastic collision implementation plan

**Goal:** Implement #8's physical-contact capability using an explicit, reproducible perfectly inelastic merge option.

**Architecture:** Separate physical contact/response from rendering and from the existing numerical close-encounter guard. Preserve the current collision-free Yoshida path; an opt-in collision-aware stepping path must resolve contact in forward physical time, emit membership-changing events, and feed the same live/recorded workflows.

**Tech Stack:** Existing Python/NumPy/Matplotlib stack and pytest; no new dependency is assumed.

## Global Constraints

- “Contact and its resulting response use declared physical properties/radii supplied by #5; illustrative marker overlap and numerical close-encounter limits are distinguishable from physical collision.”
- “Define the supported collision settings, interaction pairs, and response before claiming collision behavior is complete; preserve those choices in reproducible starting setups.”
- The original issue deliberately chose no response. The user's 2026-10-30 direction was “Inelastic merging I think. Does that reflect cosmological collision?” This plan proposes **perfectly inelastic merging**, with its limits explained; remaining interaction/remnant choices are a judgment task.
- Requires plans 2, 5, 6; sequence after 7 because shared history/controller/renderer files overlap. Collision-free work does not wait on this plan.

## Scientific interpretation

Perfect merging is an idealized accretion model. It conserves mass and linear momentum while removing relative translational kinetic energy; it does not calculate the heat, ejecta, material failure, or stellar structure that could determine a real outcome. Real planetary impacts include partial accretion, hit-and-run, erosion, and disruption as well as merging. See [Leinhardt & Stewart, outcome regimes](https://arxiv.org/abs/1106.6084) and [Stewart & Leinhardt, diversity of impact outcomes](https://arxiv.org/abs/1109.4588).

Do not describe this as a general model of cosmological collisions. Galaxy encounters have a different scale and usually do not involve individual stars striking one another; see [NASA's galaxy-collision explanation](https://www.nasa.gov/image-article/when-galaxies-collide-3/). The sandbox remains a 2D Newtonian body simulation.

## File map

| File | Responsibility |
| --- | --- |
| Create `simulations/star_system/collisions.py` | Contact candidates/events, deterministic merge response, event records |
| Modify `simulations/star_system/core.py` | Optional contact-aware stepping, transactional failure, numerical guards |
| Modify `simulations/star_system/model.py`, `initialization.py` | Supported physical radii, collision/model settings, validation |
| Modify `simulations/star_system/history.py`, `recording.py`, `playback.py` | Per-frame body membership, collision events, remnant metadata |
| Modify `simulations/star_system/persistence.py`, `experiments.py` | Persist model/pair/remnant policy and reproduce events |
| Modify `simulations/star_system/rendering.py`, `plotting.py`, `sidebar.py` | Event-aware selection/follow/readouts and model controls |
| Create `tests/test_collisions.py`, `tests/test_collision_workflows.py` | Contact accuracy, conservation, event replay, reproducibility |
| Modify `simulations/star_system/README.md` | Exact supported pairs, response, limits and tested tolerances |

### Task 1: Define the supported model and remnant contract — judgment

**Files:** `collisions.py`, `model.py`, `tests/test_collisions.py`, README; update downstream contracts if needed.

**Interfaces:** Produce `CollisionPolicy`, `CollisionEvent`, and `merge_pair(prepared, first_id, second_id) -> (PreparedSystem, CollisionEvent)` with deterministic remnant identity and provenance.

- [ ] Settle supported interactions with the owner before coding them. Proposed initial scope: positive-mass/positive-mass sphere contact merges; test-particle/positive-mass contact removes the test particle with zero mass/momentum feedback; test-particle/test-particle contact is ignored. Confirm whether stars participate under this same explicitly idealized rule or should report unsupported stellar contact. No planet/star structural realism is implied by generic merging.
- [ ] Settle remnant properties: recommend volume-additive physical radius `cbrt(R1**3 + R2**3)`, combined mass, center-of-mass position/velocity, a stable new ID with both parent IDs in an event, and user-visible combined naming. Volume addition is a chosen constant-volume approximation, not a measurement or stellar-evolution result. Temperature/composition are unknown after impact; don't propagate a source's measured values as measured remnant properties. A luminous display fallback, if needed, must be marked illustrative.
- [ ] Record the accepted pair/remnant rules, model version, and mode identifier `perfect_merge` in README and settings. Retain `none` for legacy behavior. Missing/unknown required radii block the physical collision mode recoverably; they never use sprite radii as a fallback.
- [ ] Write isolated response tests with known masses/positions/velocities. Run `python -m pytest -q tests/test_collisions.py -k merge`; expect missing response. Implement the response equations:

```python
mass = m1 + m2
position = (m1 * r1 + m2 * r2) / mass
velocity = (m1 * v1 + m2 * v2) / mass
radius = np.cbrt(radius1**3 + radius2**3)
lost_relative_kinetic = 0.5 * (m1 * m2 / mass) * np.dot(v1 - v2, v1 - v2)
```

- [ ] Assert mass/linear momentum conservation to floating precision and the stated kinetic-energy change. Record the orbital angular-momentum residual that would become internal spin; this model does not evolve spin, so do not assert total modeled orbital angular momentum is conserved through a merge. Do not claim total gravitational-plus-thermal energy conservation from the kinetic formula alone.
- [ ] Deterministically order simultaneous contact groups by stable IDs and define a group response, rather than letting array iteration choose a different remnant. Preserve event ancestry and noncolliding body states. Re-run the response tests; expect pass. Commit reviewed paths with `git commit -m "Define idealized inelastic collision responses"`.

### Task 2: Resolve physical contact without bypassing numerical limits — judgment

**Files:** `collisions.py`, `core.py`, `initialization.py`, `tests/test_collisions.py`, README.

**Interfaces:** Consume accepted policy and physical radii; produce a successful macro-step of exactly configured dt plus ordered physical-time events, or an unchanged state with a numerical failure reason.

- [ ] Add head-on, grazing, high-speed crossing, initial-overlap rejection, near-miss, simultaneous contact, and close-encounter-before-contact tests. Test both permitted and unsupported pair classes. Run `python -m pytest -q tests/test_collisions.py`; expect missing event handling.
- [ ] Use physical separation `norm(r1-r2) <= R1+R2` as the contact condition for supported massive spheres. Candidate sweep detection must catch bodies crossing between samples; endpoint overlap alone is insufficient. Particle contact uses the accepted point-tracer rule, not the marker diameter.
- [ ] Inspect the existing Yoshida composition's negative middle substep before choosing the event algorithm. Never merge merely because an internal negative-time stage overlaps. Proposed collision mode uses a positive-time velocity-Verlet stepping path with bounded subdivision/contact refinement, leaving `none` on the existing Yoshida path. This is a separately named numerical mode with its own accuracy checks, not a claim that irreversible merging preserves symplectic integration.
- [ ] Resolve the earliest forward-time contact, apply response once, and integrate the remainder of the macro-step with updated membership. A live Step still advances one configured macro timestep and ends paused. Internal event/subdivision times are recorded. If the bounded solver cannot resolve a contact, roll back the macro-step and report a numerical halt; do not falsely report a physical collision or skip the encounter guard globally.
- [ ] Run convergence tests at dt, dt/2, dt/4 on known supported fixtures. Require contact times to converge and no missed high-speed collision/false grazing contact; establish and document numerical tolerances before declaring the fixture supported. Run existing no-collision physics tests unchanged to guard against regression in mode `none`. Keep render scale entirely out of contact calculations.
- [ ] Re-run `python -m pytest -q tests/test_collisions.py tests/test_star_system.py tests/test_populations.py tests/test_moons.py`; expect pass for the supported model and explicit halts for unsupported cases. Commit exact paths with `git commit -m "Resolve physical contacts in collision-aware integration"`.

### Task 3: Save, record, replay, and inspect collision events — mechanical

**Files:** Persistence/history/recording/playback/session/rendering/controller files from the file map, `tests/test_collision_workflows.py`, README.

**Interfaces:** Consume Tasks 1–2, `StateFrame` per-frame identity metadata, and versioned saved settings. Produce reproducible event streams and truthful remnant inspection.

- [ ] Add save→reload→calculate tests that compare event times, parent/remnant IDs, physical properties, and post-event states within Task 2's documented tolerance. Run `python -m pytest -q tests/test_collision_workflows.py`; expect missing membership/settings integration.
- [ ] Persist accepted collision mode, supported-pair policy, remnant rule, numerical-mode version/settings, and initial radii. Reject an unsupported policy/version before replacing a current experiment. Old schema-version-1 saves with explicit mode `none` retain their behavior; introduce a migration/version only if the accepted fields require it, with tested old-file handling.
- [ ] Capture pre/post-event states with physical timestamps and events, even when a collision falls between normal output samples. Do not silently attach new IDs to old trajectories. Stop constituent trails at the event and start the remnant trail there. Record removal of an absorbed tracer if that rule was accepted.
- [ ] Follow a selected merged constituent to the recorded remnant only through the event's explicit mapping and display the change; an absorbed tracer has no massive remnant and selection should state its removal. Playback's no-integration guard must remain active. Restoring a previous recorded frame restores that frame's membership, never today's live body list.
- [ ] Run focused collision workflows plus recording/playback tests; expect pass. Commit exact paths with `git commit -m "Preserve collision events across saves and recorded playback"`.

### Task 4: Integration limits and issue closeout — mechanical

**Files:** Collision tests, existing integration tests, README.

**Interfaces:** Consume all preceding plans; deliver the final #8 collision acceptance evidence and updated #6/#7/#9 model disclosures.

- [ ] Run `python -m pytest -q --tb=short` and `git diff --check`; expect pass. Add a collision-mode saved-setup CLI export subprocess test and keep the existing collision-free export checks.
- [ ] Verify marker-size/zoom changes cannot alter collision event results. Verify the selected policy for population particles and moons; distinguish physical contact, unsupported contact, numerical halt, and normal completion in assertions on visible status text.
- [ ] Record only observed numerical tolerances, supported pairs, conservation properties, and performance limits. Explicitly exclude fragmentation, erosion, spin dynamics, tides, relativistic/stellar-impact physics, guaranteed capture, and 3D. Use the cited literature to explain why perfect merging is an approximation, without implying the toy model implements that literature's richer outcome model.
- [ ] Commit reviewed tests/docs with `git commit -m "Verify and document idealized collision workflows"`. Deliver the focused PR referencing #8. Reconcile all #8 acceptance against plans 5–8 and the #5/#6/#9 integrations before proposing closure; do not close it merely because the merge formula passes unit tests.
