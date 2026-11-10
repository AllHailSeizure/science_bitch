# Experiment desktop implementation plan

**Goal:** Deliver a double-clickable Windows application built around choosing, editing, and running prepared experiments.

**Architecture:** Use the existing PyQt6 dependency for a native menu, dropdowns, and a resizable sidebar; embed the existing Matplotlib scene. Reuse immutable InitialSetup, ExperimentSession, Calculation, Playback, and persistence. The CLI and legacy controller remain available for compatibility and headless exports.

**Tech Stack:** Python 3.10+, PyQt6, Matplotlib, NumPy; PyInstaller for the Windows executable.

## Global constraints

- Open at a starting menu with Choose experiment and Load saved experiment.
- Solar System is the default template for every new experiment; prepared experiments are frozen variations.
- Experiments can be run or edited; preserve the original for reset and saving a separate variation.
- Add body uses type dropdown (Stellar, Planetary, Lunar, Other), body dropdown, information preview, Add, then scene placement.
- Contextual selection controls Lagrange guides and orbit paths; no globally displayed Jupiter-specific guides.
- Keep physics units and solver controls out of the ordinary creation flow; expose exact inputs under clearly labeled advanced controls.
- Build and verify a self-contained, windowed Windows executable.

## File map

- `simulations/star_system/templates.py`: Solar System baseline, prepared experiment library and immutable edits.
- `simulations/star_system/context.py`: selection-relative orbit and L4/L5 geometry independent of named catalog bodies.
- `simulations/star_system/desktop.py`: Qt menu, experiment library, workspace, run/replay, save/load and scene input.
- `simulations/star_system/desktop_panels.py`: body catalog dropdowns, information preview and edit form.
- `simulations/star_system/__main__.py`: default desktop entry; preserve explicit CLI construction and output behavior.
- `launch_observatory.pyw`, `Observatory.spec`, `requirements-build.txt`, `scripts/build-windows.ps1`: executable entry and reproducible build.
- `tests/test_experiment_templates.py`, `tests/test_context_overlays.py`, `tests/test_desktop.py`: behavior and Qt interaction regressions.
- `README.md`, `simulations/star_system/README.md`: launching, simple workflow and numerical disclosure.

## Tasks

### 1. Solar System baseline and immutable experiment edits (judgment)

Create all-eight-planet template using existing catalog values and circular/barycentric initializer. Preserve explicit parent relationships and the existing 2D illustrative-model disclosure. Include prepared variations with a specific stated observation, not unverified capture promises. Edits retain IDs, preserve starting vectors unless explicitly edited, validate atomically and keep original snapshots.

Verification: `.venv/Scripts/python.exe -m pytest -q tests/test_experiment_templates.py`.

### 2. Contextual scene geometry (judgment)

Compute guides from selected body and its reference pair; compute instantaneous bound two-body orbit curves for its referenced orbiting bodies. Use explicit labels distinguishing geometric guides and orbit estimates from predictions. Verify generic renamed bodies, clearing selection, moon relationships, and unchanged physical arrays.

Verification: `.venv/Scripts/python.exe -m pytest -q tests/test_context_overlays.py`.

### 3. Native desktop workflow (judgment)

Implement start menu and experiment library; open frozen setups in a resizable workspace. Add body dropdowns/preview and cancellable scene placement. Provide body replacement and property edits, starting velocity arrows, Run/Pause/Reset, recorded Calculate/Playback, native save/load and preserved baseline reset. Reuse existing rendering while hiding global legacy guides. Cover actual widget selection/click flows, invalid edits, run/reset and save/reload.

Verification: `.venv/Scripts/python.exe -m pytest -q tests/test_desktop.py tests/test_experiment_templates.py tests/test_context_overlays.py` plus executable UI inspection.

### 4. Build and deliver executable (mechanical)

Package PyQt6, Matplotlib, NumPy and catalog JSON into windowed `dist/Observatory.exe`; add a build script and Windows CI artifact workflow. Document launch and experiment editing. Run `.venv/Scripts/python.exe -m ruff check .`, `.venv/Scripts/python.exe -m pytest -q --basetemp output/pytest-desktop-final`, `powershell -File scripts/build-windows.ps1`, and launch the resulting application. Commit, push, and open a PR against master when checks pass.

## Execution results

- Tasks 1–4 implemented as a PR stack: foundations → desktop → Windows packaging.
- Baseline: 327 tests passed. Integration suite: 354 passed before the final camera-focus addition. Final focused suite: 28 passed, including the camera addition; Ruff and whitespace checks passed.
- The final Windows build produced the 78,501,936-byte single-file `dist/Observatory.exe`. Its archive includes Python, Qt Widgets, the physical catalog JSON and the DejaVu font.
- The final executable loaded a saved all-eight-planet setup with a Moon from a separate working directory and rendered 0.05 years successfully (exit 0, 162,170-byte PNG).
- Qt widget interaction tests and rendered 1280 × 800 previews were checked. The source application's native starting menu was observed through accessibility. The packaged app was launched minimized, but computer inspection was stopped with the user's Escape key before its menu could be inspected. A complete native packaged-app walkthrough is therefore unverified; no further Computer Use actions were taken.
