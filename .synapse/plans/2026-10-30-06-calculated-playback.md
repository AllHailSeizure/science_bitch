# Calculated recording and playback implementation plan

**Goal:** Deliver #8's calculate-first workflow while preserving live simulation and completing calculated population/moon inspection.

**Architecture:** Run the same prepared numeric system to produce immutable, timestamped frames, then render those frames without calling the integrator. Keep setup, calculation progress, recorded result, and live mutable state distinct; use per-frame identity/metadata so later collisions can change body membership without corrupting playback.

**Tech Stack:** Python, NumPy, Matplotlib timers, existing solver, pytest; in-memory records initially.

## Global Constraints

- “Playback speed controls replay of calculated results without changing their physical evolution or requiring motion to be recalculated during playback.”
- “Calculation and playback make their simulation time, progress/completion, and any halt or incomplete result understandable.”
- “Changing starting conditions requires a new calculation and does not silently rewrite previously calculated results.”
- Requires plan 5. Timeline scrubbing, reverse navigation, comparison dashboards, persistent recordings, pause checkpoints, and saved camera preferences are open choices, not implicitly included.

## File map

| File | Responsibility |
| --- | --- |
| Create `simulations/star_system/recording.py` | Calculation requests, bounded runner, immutable `RecordedRun` |
| Create `simulations/star_system/playback.py` | Recorded-frame cursor, timing, pause and next-sample controls |
| Modify `simulations/star_system/history.py`, `experiments.py` | Frame capture, setup digests, state transitions |
| Modify `simulations/star_system/plotting.py`, `sidebar.py`, `rendering.py` | Progress/status, live-vs-recorded controls, shared inspection |
| Create `tests/test_recording.py`, `tests/test_recorded_playback.py` | Numerical consistency, immutability, progress, no reintegration |
| Modify `tests/test_experiment_controls.py` | Setup edits and run/result lifecycle |
| Modify `simulations/star_system/README.md` | Calculation policy, clocks, history, memory/numerical limits |

### Task 1: Settle recording policy and implement the calculator — judgment

**Files:** `recording.py`, `history.py`, `experiments.py`, `tests/test_recording.py`, README.

**Interfaces:** Produce `CalculationRequest(duration_years, sample_stride, max_bytes)`, `Calculation.advance(max_steps)`, `Calculation.cancel()`, and `RecordedRun`. Consume `build_system` and `setup_digest`; frames contain copied world states.

- [ ] Settle the issue's duration/output-sampling choices inline. Proposed first controls: positive finite duration in simulated years, positive integer sample stride in physics steps, in-memory records, and a visible 256 MiB default recording budget. Show the resolved integer step count and end time before starting; round up to the first complete fixed step at/after the requested duration. Record requested and actual times. Sampling is not a second physics timestep. Confirm any owner preference that changes this behavior before building its UI; record the accepted policy.
- [ ] Add tests for zero/nonfinite/negative input rejection, integer step resolution, sample stride, initial/final sample inclusion, cancellation, and a close-encounter halt that preserves the last completed step. Include population and moon setups from plans 3–5.
- [ ] Run `python -m pytest -q tests/test_recording.py`; expect missing calculator. Implement incremental step batches so the GUI can process progress/cancel events; no unbounded integration loop in a single UI callback. Headless use drives the same runner until terminal status.
- [ ] Verify numerical equivalence against a separately constructed live `System`: advance both exactly N steps using the same fixed dt and compare each recorded step index with `rtol=1e-12, atol=1e-12`. Sampling stride must not affect final physical state. Copy and protect arrays at capture; mutating the live system after capture cannot mutate a frame.
- [ ] Preflight estimated frame storage and enforce the declared budget during recording. If a run cannot fit, explain the required reduction in duration/count or increase in sample stride before calculation; do not silently drop samples. Preserve partial results and an explicit status if an error/cancel occurs. Never label halted or cancelled output complete.
- [ ] Use tuples of frames rather than a fixed `(time, fixed_body_count, 2)` array contract. Body IDs/specs and physical state travel together; plan 8 can record pre/post-merge membership and events. Add a synthetic variable-membership frame roundtrip check without implementing collisions here.
- [ ] Re-run `python -m pytest -q tests/test_recording.py`; expect pass. Commit exact paths with `git commit -m "Calculate bounded immutable experiment recordings"`.

### Task 2: Replay recorded states without physics calls — mechanical

**Files:** `playback.py`, `plotting.py`, `sidebar.py`, `rendering.py`, `tests/test_recorded_playback.py`.

**Interfaces:** Consume `RecordedRun`; produce `Playback.current_frame`, `play()`, `pause()`, `next_sample()`, and `advance(wall_seconds, years_per_second)`. Use the recorded timestamp as the displayed physical clock.

- [ ] Add tests for play/pause, speed changes, next sample, end-of-recording stop, status/reason display, and stable selected identity across frames. After generating a fixture recording, use this guard:

```python
def unexpected_step(*args, **kwargs):
    raise AssertionError('Recorded playback must not integrate')
monkeypatch.setattr(System, 'step', unexpected_step)
```

- [ ] Run `python -m pytest -q tests/test_recorded_playback.py`; expect missing playback controls. Implement forward sample selection driven by recorded time; do not invent interpolated physical state or display wall-clock time as simulated time. Label the replay step control `Next sample` so it cannot be confused with live `Step`.
- [ ] Render/inspect the frame's own positions, velocities, masses, radii, and identities. Selection/follow uses IDs and updates metadata from the displayed frame. A missing selected body reports its event/removal rather than silently selecting the next row. Camera settings persist through ordinary frame changes; fit/follow remains explicit.
- [ ] Build displayed trails from retained timestamped world frames up to the current sample. Report actual history start/end/duration, accounting for output stride and truncation. Preserve the existing bounded display vertex count. Do not infer duration as sample_count×current_dt when event segments differ.
- [ ] Re-run `python -m pytest -q tests/test_recorded_playback.py`; expect pass, including the `System.step` guard. Commit exact paths with `git commit -m "Replay recorded simulation states without reintegration"`.

### Task 3: Preparation/calculation/live integration — judgment

**Files:** `experiments.py`, `plotting.py`, `sidebar.py`, experiment/recording/playback tests, README.

**Interfaces:** Consume Tasks 1–2 and the preparation session; produce explicit preparation/calculating/recorded/live transitions and stored setup-digest association.

- [ ] Add event-driven tests for Prepare→Calculate→Recorded, cancel with a usable partial recording, halt with last valid frame, changing a draft while retaining the old immutable result, and switching to a fresh live run. Assert the UI identifies the setup used for a result and cannot relabel an old result as the edited setup.
- [ ] Run `python -m pytest -q tests/test_experiment_controls.py tests/test_recorded_playback.py`; expect missing transitions before wiring them. Bind both body/population/moon editors to the same preparation draft. Calculated-result inspection must not invoke live-insertion callbacks against a recorded frame; a starting-condition change returns to preparation and requires a new calculation.
- [ ] Preserve live pause/resume, exactly-one-physics-step, restart, and plan 2 live insertion behavior. Live speed affects requested physical advancement; replay speed affects only frame selection. Show mode and units clearly.
- [ ] Add end-to-end tests for save→reload→calculate→inspect on a seeded population and on a moon system. Compare initial particle arrays, recorded timestamps, moon/planet separation, follow selection, and terminal status. Test one intentionally unresolved setup and verify visible incomplete output rather than a silent reset.
- [ ] Run `python -m pytest -q --tb=short` and `git diff --check`; expect pass. Document actual timing/sampling/memory limits and what is not persisted. Commit reviewed files with `git commit -m "Connect preparation and live mode to calculated playback"`.
- [ ] Deliver the playback PR referencing #8/#6/#9. Re-audit #6/#9 acceptance across their complete dependency stack; #8 still needs the Jupiter preset and collision deliverable.
