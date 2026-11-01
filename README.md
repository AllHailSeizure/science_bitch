# science_bitch

Small, well-established physics simulations in Python. Each one uses an integrator suited to the problem and checks itself against a known result.

## Install

From the repository root, with Python 3.10 or newer:

```bash
python3 -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt
```

The dependencies are numpy, matplotlib, pillow, pytest, and PyQt6 (GUI backend). The tests and the headless render need no display.

## Run

```bash
python -m simulations.star_system  # live interactive GUI
python -m simulations.star_system --star sun --planet earth --planet jupiter --output output/star-system.png
```

Details, presets, and the other command-line flags are in each simulation's README:

- [Interactive star system](simulations/star_system/README.md) — tunable stars and planets with full N-body gravity

## Development checks

Install the development tools in your virtual environment:

```bash
python -m pip install -r requirements-dev.txt
```

Run both checks from the repository root before opening a pull request:

```bash
python -m ruff check .
python -m pytest -q
```

The test suite covers physics, population generation, body authoring, persistence, recorded playback, collisions, controls, and rendering. Physics tolerances are listed in the [star system README](simulations/star_system/README.md). Tests select Matplotlib's Agg backend and need no display.

[Ruff](https://docs.astral.sh/ruff/) checks application and test code using the rules in `ruff.toml`: Python errors, undefined/unused names, and ambiguous statements. Its version is pinned in `requirements-dev.txt` so local checks and CI use the same rules. To apply supported automatic fixes, run `python -m ruff check . --fix`, review the diff, and rerun both checks. Formatting is not enforced.

The [CI workflow](.github/workflows/ci.yml) runs on every pull request (including PRs targeting another feature branch), pushes to `master`, and manual dispatch. Lint runs as a separate job; any lint or test failure fails CI. The complete pytest suite runs on Linux with Python 3.10 and 3.14, plus Windows with Python 3.14. Python 3.10 is the documented minimum; the Windows job also checks the desktop development platform. GitHub installs these interpreters on its runners; you only need one supported Python locally.

To enforce passing checks before merging, configure branch protection or a ruleset to require `Lint` and the three `Tests (...)` jobs after the workflow's first run.

## Adding a simulation

Create `simulations/<name>/` with:

- `core.py` — the equations and the integrator, importable without Matplotlib
- `__main__.py` — the command-line entry (`python -m simulations.<name>`)
- `plotting.py` — figures and the animation
- `README.md` — the equation, how to run it, and what is checked

Add the checks under `tests/`. Keep new packages in `requirements.txt` only when a simulation needs them.
