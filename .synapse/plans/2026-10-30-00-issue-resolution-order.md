# Star-system issue resolution: plan order

Planning snapshot: 2026-10-30, `AllHailSeizure/science_bitch`. The user requested separate, dependency-ordered plans for all six open issues (#4–#9). These are implementation plans, not a claim that the issues have been resolved or authorization to merge PRs.

## Verified starting point

- Fetched `origin/Master`: `2d91df9ca552cffb807c233c5ac8769a636fcf36`.
- [PR #10](https://github.com/AllHailSeizure/science_bitch/pull/10) supplies the observatory, retained rendering, and frame pacing.
- [PR #11](https://github.com/AllHailSeizure/science_bitch/pull/11) supplies seeded ring/disk/belt test particles.
- [PR #12](https://github.com/AllHailSeizure/science_bitch/pull/12) **is merged**, including the population settings window. Issue #6 still describes it as open; do not repeat that work or add a merge prerequisite.
- The current checkout remains `feat/observatory-visuals` at `8267e62`. Plan against the fetched base, not that older implementation. The pre-existing untracked `2026-10-28-observatory-visuals.md` is historical and must be preserved.
- The merged PR reports 105 passing tests. That is historical PR evidence; this planning session did not run the application test suite or verify native desktop interaction.

## Delivery order

| Order | Independently reviewable deliverable | Requires | Issue completion |
| --- | --- | --- | --- |
| 1 | [Full-window observatory](2026-10-30-01-observatory-overlay.md) | Current Master | #4 presentation and controls; scenario naming integrates in 5 |
| 2 | [Body catalog, initial conditions, and authoring](2026-10-30-02-body-authoring.md) | 1 | #5; shared interfaces for later work |
| 3 | [Population authoring integration](2026-10-30-03-population-integration.md) | 2; reuse merged #11/#12 | #6 authoring; save/playback coverage in 5–6 |
| 4 | [Planet-relative moons](2026-10-30-04-moons.md) | 2; sequence after 3 to avoid shared-file conflicts | #9 live/preparation; save/playback coverage in 5–6 |
| 5 | [Experiment preparation and save/reload](2026-10-30-05-experiment-persistence.md) | 2–4 | #8 setup/persistence; completes relevant #6/#9 integration |
| 6 | [Calculate-first recording and playback](2026-10-30-06-calculated-playback.md) | 5 | #8 calculation; completes relevant #6/#9 integration |
| 7 | [Sun–Jupiter experiments and rotating view](2026-10-30-07-jupiter-experiments.md) | 3, 5, 6 | #7; provides #8's named Jupiter scenario |
| 8 | [Simplified inelastic collisions](2026-10-30-08-inelastic-collisions.md) | 2, 5, 6; sequence after 7 for shared renderer/controller files | #8 collision capability; recheck #6/#7/#9 limitations |

The collision-model decision can be resolved earlier without changing this implementation order. It does not block the collision-free setup and playback work. Keep #8 open until its separate deliverables and cross-issue integrations are verified.

Use `codex/issue-4-overlay`, `codex/issue-5-authoring`, `codex/issue-6-population-integration`, `codex/issue-9-moons`, `codex/issue-8-persistence`, `codex/issue-8-playback`, `codex/issue-7-jupiter`, and `codex/issue-8-collisions` as branch names. Start from the latest base. Preserve the current checkout; use isolated worktrees for implementation. Stack dependent PRs on the preceding implementation branch when it is awaiting merge; do not stop merely because the prior PR is open. Retarget after the base merges. Do not combine all eight deliverables in one branch.

The shared authoring interaction is developed first on a custom planet inserted into a Sun/Earth system. Population reference selection and moon setup reuse the resulting interface and diff. They are repeat/adaptation work, not concurrent redesigns. Do not start those adaptations alongside the unsettled body-authoring implementation; first land its implementation in the dependency branch, then carry that diff forward. The UI, renderer, and controller files overlap, so these plans are not parallel-safe implementation lanes.

## Shared contracts proposed by these plans

These names are new implementation targets, not existing APIs. Keep them consistent across the stack; if an earlier judgment task changes a contract, update its downstream plans before dispatching mechanical work.

- `model.py::BodySpec`: immutable physical/display definition: catalog key or custom provenance, name, kind, mass in solar masses, physical radius in AU, optional stellar temperature, illustrative color, and per-property source/uncertainty metadata. Unknown catalog values may be `None`; preparation blocks a capability needing a missing value rather than estimating it silently. A catalog definition is not an instance identity.
- `model.py::InitialBody`: `id: str`, `spec: BodySpec`, `r: tuple[float, float]`, `v: tuple[float, float]`, `reference_id: str | None`. Coordinates are AU and AU/year; when a reference is present they are relative to its initial state. Resolve references once at initialization; no force attaches a body to its reference.
- `model.py::PopulationSpec`: the existing `Population` generation parameters plus stable `id` and `reference_id` fields; the old integer reference is resolved only at the numeric-engine boundary. Keep one configured population initially, matching the current application; stable identity does not implicitly add a multiple-population UI.
- `model.py::InitialSetup`: `name`, optional `preset_id`, tuple of `InitialBody`, optional `PopulationSpec`, explicit fixed `dt` in years, and `collision_mode`. Initial collision mode is `none`; no new physics is implied by storing the field.
- `initialization.py::build_system(setup) -> PreparedSystem`: validates/resolves the setup and returns `system: System`, `body_ids: tuple[str, ...]`, and aligned `body_specs: tuple[BodySpec, ...]`. Particle identities are stable within a setup and derived from population identity/ordinal, not their current row number.
- `initialization.py::insert_body(prepared, body) -> PreparedSystem`: resolves placement against the current reference state and returns a new validated state, preserving existing rows and elapsed time. It never barycentrically recenters existing bodies.
- `history.py::StateFrame`: immutable copied `time`, `dt`, `body_ids`, `body_specs`, `masses`, `r`, and `v`, plus events. Store world-frame states before display compression. Read-only arrays or defensive copies prevent playback from modifying the simulation.
- `rendering.py::draw_system(ax, prepared, history=()) -> Scene`: general rendering entry point; retain existing `draw(ax, system, star, planets, trails=None)` as a compatibility adapter while callers migrate.
- `recording.py::RecordedRun`: setup digest, requested end time, tuple of `StateFrame`, status (`complete`, `halted`, `cancelled`), and reason. Frame timestamps, identities, and metadata are authoritative during replay.

Keep NumPy, Matplotlib, Pillow, pytest, PyQt6, Python 3.10+, the 2D model, and the existing fixed-step Yoshida solver for collision-free evolution. Introduce no new package or replacement UI framework in these plans. Resolve immutable setup data separately from mutable `System` arrays. Preserve the existing CLI and compatibility wrappers until all callers and existing behavior checks have migrated.

## Judgment tasks and owner decisions

1. **Collision response:** the user tentatively selected inelastic merging and asked about realism. Plan 8 targets perfectly inelastic merging as a deliberately simplified option. It must not be described as a general realistic model of astronomical impacts. Its first judgment task settles supported body pairs, remnant properties, and particle behavior before collision implementation.
2. **Authoring semantics:** distinct instance IDs (including repeated catalog choices), explicit reference selection, and preparation-only edit/remove are the execution defaults. The user selected saving the original setup plus offering the current state as a separately named new setup. Restart uses the selected starting setup; live insertion does not silently rewrite it. Capturing current state creates a new initial epoch, preserving the source time as provenance. Do not mix evolved population coordinates with regenerated particles in that captured setup.
3. **Recording policy:** plan 6 proposes user-entered duration and integer output stride, in-memory results, and forward recorded playback. These are a minimal proposal for the issue's open sampling/inspection choices; settle them before building UI. Scrubbing, reverse integration, persisted recordings, and preference saving are not presumed requirements.

Judgment tasks run inline. Mechanical tasks may be delegated only with the settled interfaces, the prerequisite diff, and their exact verification command. A passing structural check cannot settle a product or physics-model decision.

Plan 1 can start immediately. Later plans begin with the listed judgment work where their source issue leaves behavior unresolved; their downstream mechanical tasks start only after that contract is recorded. These are decision-bearing implementation plans, not a claim that every product choice has already been approved. This planning delivery itself only adds documents; it does not start implementation, publish plans, or change GitHub issue state.

Execution authorization, 2026-10-30: the user requested a PR for these plans followed by one agent per plan, launched sequentially, with every implementation PR stacked on the preceding branch. This overrides the earlier allocation of judgment tasks solely to the primary session; each plan agent owns its bounded implementation and reports genuine unresolved product choices. The collision-pair question has been reopened at the user's request; disregard the earlier answer until the new reply arrives.

## Verification and closure

Each plan names focused tests and commands. A new test is an implementation target, not an already passing check. Verify its intended failure before the implementation, then run the focused suite. Run `python -m pytest -q --tb=short`, the applicable headless CLI export, and `git diff --check` once at the deliverable's integration boundary. Do not repeat the full suite after documentation-only edits without a reason.

No plan requires a human playthrough. Existing Agg event/render checks establish only their tested behavior. Record native desktop input/multiwindow behavior as unverified unless an agent actually exercises it on a native backend; do not translate an offscreen Qt/Agg pass into native-interaction evidence. This reporting satisfies the distinction requested in #4/#6 without inventing a manual release gate.

| Issue | Acceptance ownership |
| --- | --- |
| [#4](https://github.com/AllHailSeizure/science_bitch/issues/4) | Plan 1 layout/catalog entry/size/readability/live controls/docs; plan 2 multi-star rendering; plan 5 scenario name |
| [#5](https://github.com/AllHailSeizure/science_bitch/issues/5) | Plan 2 catalog/provenance/custom bodies/placement/vectors/live insertion/multiple stars/units; plans 5–6 shared preparation/results; physical radii consumed by 4 and 8 |
| [#6](https://github.com/AllHailSeizure/science_bitch/issues/6) | Merged #11/#12 foundation; plan 3 reference integration; plans 5–6 reproducible saves/calculation; plan 8 explicit contact limits |
| [#7](https://github.com/AllHailSeizure/science_bitch/issues/7) | Plan 7 repeatable setup, rotating view, L4/L5 guides, rock trajectories, matched controls; uses 2–3 and 5–6 rather than duplicate subsystems |
| [#8](https://github.com/AllHailSeizure/science_bitch/issues/8) | Plan 5 setup/presets/persistence/recovery; plan 6 calculation/replay/progress; plan 7 Jupiter preset; plan 8 collision response/settings/reproducibility |
| [#9](https://github.com/AllHailSeizure/science_bitch/issues/9) | Plan 4 moon setup, parent-relative inspection, short-timescale checks; plans 5–6 save/replay; plan 8 stated contact model |

At issue closeout, compare the live acceptance list against the shipped dependency stack. Do not use `Closes #6`, `Closes #8`, or `Closes #9` on partial foundation PRs. PR descriptions should identify referenced issues, applicable existing labels, test checklist, measured results, and remaining limits. Merge and issue closure remain separate from writing these plans.
