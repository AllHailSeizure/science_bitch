# Full-window observatory implementation plan

**Goal:** Fulfill #4's presentation/control requirements with a full-window scene and a closable overlay sidebar.

**Architecture:** Keep the Matplotlib canvas and retained `Scene`/`CanvasPainter`. Overlay control axes above a scene whose viewport and camera do not change when the sidebar toggles; separate sidebar construction from the already large `App` controller.

**Tech Stack:** Python 3.10+, Matplotlib, NumPy, Pillow, pytest; existing GUI backends.

## Global Constraints

- “Opening or closing the sidebar reveals or covers the existing scene without resizing or recentering the simulation view.”
- “Keep the overhead view and physics-driven evolution. 3D is deferred.”
- “Record the outcome and limits of native desktop interaction verification without treating headless checks as a manual playthrough.”
- Base: fetched Master including #12. Follow the [shared plan contracts and delivery rules](2026-10-30-00-issue-resolution-order.md).

## File map

| File | Responsibility |
| --- | --- |
| Create `simulations/star_system/sidebar.py` | Overlay widgets, open/close state, scrollable body list, catalog panel |
| Modify `simulations/star_system/plotting.py` | Full-window scene, event routing, title/readouts, existing callback wiring |
| Modify `simulations/star_system/rendering.py` | Full-extent equal-scale viewport, overlay-aware painting, smaller planet markers |
| Modify `tests/test_star_system_display.py` | Layout, event isolation, camera, redraw, marker/detail regressions |
| Modify `tests/test_population_controls.py` | Merged population editor remains reachable and modal-safe |
| Modify `simulations/star_system/README.md` | Controls, sizes, scales, verification limits |

### Task 1: Overlay layout and event routing — judgment

**Files:** `sidebar.py`, `plotting.py`, `rendering.py`, `tests/test_star_system_display.py`.

**Interfaces:** Consume existing `App` callbacks and `CanvasPainter.invalidate`; produce `Sidebar.set_visible(visible)`, `Sidebar.contains_event(event)`, and `App.toggle_sidebar(event=None)`. Preserve `app.ax` as the scene axes.

- [ ] Add `test_sidebar_toggle_keeps_scene_transform`, `test_sidebar_blocks_scene_gestures`, and `test_scene_fills_wide_and_tall_canvas`. In the first, save camera limits and `ax.transData.transform(scene.positions)`, toggle twice through actual button events, and require both to remain unchanged. In the third, check `ax.bbox.bounds` against `fig.bbox.bounds` after drawing 12×8 and 8×12 figures.
- [ ] Run `python -m pytest -q tests/test_star_system_display.py -k 'sidebar or fills_wide'`; expect missing sidebar/full-viewport behavior.
- [ ] Give the scene `(0, 0, 1, 1)` figure bounds. Existing `set_aspect('equal', adjustable='box')` shrinks the active viewport; replace that policy with explicit resize handling that preserves center and world-units-per-pixel, adjusting limits to canvas aspect. Sidebar toggling must not call fit/restart or the resize handler.
- [ ] Put the overlay background and all its widgets in a single visibility/event group. Keep a visible `Menu` button outside that group, with a keyboard shortcut documented in the UI. Closing the sidebar disables its hidden widget handlers. Pointer events over the overlay must not pan, zoom, select, or place bodies behind it; typing in an editor must not trigger playback shortcuts.
- [ ] Update `CanvasPainter` so moving scene artists cannot be painted over cached sidebar pixels. Repaint the overlay last (including its widget axes) in both cached and full redraw paths; invalidate on visibility, resize, and control changes. Preserve export behavior.
- [ ] Re-run the focused command; expect pass. Extend `test_partial_redraw_matches_full_redraw_after_motion_and_resize` with sidebar open/closed cases and compare pixel buffers for identical state.
- [ ] Commit explicitly reviewed files: `git add simulations/star_system/sidebar.py simulations/star_system/plotting.py simulations/star_system/rendering.py tests/test_star_system_display.py`; `git commit -m "Make observatory scene fill the window behind overlay controls"`.

### Task 2: Useful controls, body list, and compact visual scale — judgment

**Files:** All plan files.

**Interfaces:** Consume Task 1 sidebar; produce `App.set_experiment_name(name: str)` and `Sidebar.set_bodies(rows, selected_id)`, keeping an adapter for current integer selections. Catalog selection invokes the existing add/restart behavior until plan 2 changes that behavior explicitly.

- [ ] Add `test_catalog_opens_only_from_add_body`, `test_sidebar_shows_elapsed_time_and_requested_speed`, `test_body_list_scroll_keeps_controls_reachable`, and `test_jupiter_marker_is_smaller_than_sun`. Update tests that assert obsolete fixed sidebar coordinates to assert accessibility/visibility instead.
- [ ] Run `python -m pytest -q tests/test_star_system_display.py -k 'catalog_opens or sidebar_shows or body_list or jupiter_marker'`; expect current permanent catalog/layout/size behavior to fail.
- [ ] Remove the three exact slogans in #4 and the fixed heading. Use an explicit experiment-name property, initially a descriptive name assembled from the selected setup; plan 5 supplies preset/custom names. Put elapsed simulated years, requested years per second, existing bodies, pause/step/restart, fit/follow, inspection, and Population access in the sidebar. Place catalog choices behind `Add a body`; preserve star selection access until plan 2 replaces it with multi-star authoring.
- [ ] Start the revised planet diameter rule at `4 + 3 * log10(1 + radius_km / 1000)` points. Assert `planet_size('jupiter') < 0.5 * star_size(STARS['sun'])` and monotonic size across catalog radii. Keep existing procedural surface and illumination. Add a larger selected-body preview in the inspector so shrinking overview markers does not remove inspectable surface detail. Scene zoom continues to expose spatial detail; the preview exposes surface detail.
- [ ] Use a clipped/scrollable present-body list rather than shrinking font sizes as bodies are added. Verify a 40-row synthetic list, the existing ten catalog planets, and access to population settings. Keep readable labels, restrained background, and fading trails; avoid changing their physical interpretation.
- [ ] Re-run the focused tests plus `python -m pytest -q tests/test_star_system_display.py tests/test_population_controls.py`; expect pass. Document keyboard access and illustrative scaling. Commit with `git add` naming only this plan's files, then `git commit -m "Expose experiment controls and compact body visuals in the sidebar"`.

### Task 3: Integration evidence and delivery — mechanical

**Files:** `tests/test_star_system_display.py`, `tests/test_population_controls.py`, `simulations/star_system/README.md`.

**Interfaces:** Consume Tasks 1–2; produce the issue-4 PR evidence and the title/sidebar interfaces used by later plans.

- [ ] Run `python -m pytest -q --tb=short`; expect all existing and new checks to pass. Existing pause/resume/one-step, pan/zoom/fit/follow, restart, speed budgeting, close-encounter reporting, and non-blit/export checks remain applicable.
- [ ] Run `python -m simulations.star_system --star sun --planet jupiter --years 1 --output output/issue-4-sun-jupiter.png` and `python -m simulations.star_system --population belt --count 200 --seed 42 --output output/issue-4-population.png`; expect successful nonempty exports. Use existing Pillow/render checks to inspect image dimensions and nonempty scene content; do not equate export with native input verification.
- [ ] Run `git diff --check`; expect no whitespace errors. Record actual outcomes and native verification limits in README. Commit the documentation and any reviewed regression updates with `git commit -m "Document and verify full-window observatory controls"`.
- [ ] Deliver the focused PR with a checklist and references to #4. Record that the name setter is ready for plan 5; later scenario/multiple-star integration is checked again in the owning plans.
