# Sidereel

Small, well-established physics simulations in Python. Each one uses an integrator suited to the problem and checks itself against a known result.

## Install

Python 3.10 or newer. Install Sidereel and open it:

```bash
pipx install sidereel
sidereel
```

`sidereel` launches the Qt desktop. It is registered as a GUI script, so on Windows the launcher uses `pythonw` and does not leave a console window open. `sidereel-cli` is the full command line, including flags and a headless `--output` render.

Until the package is on PyPI, the same commands install from this repository:

```bash
pipx install git+https://github.com/AllHailSeizure/sidereel
```

For development, from the repository root:

```bash
pip install -e ".[dev]"
```

That install provides numpy, matplotlib, pillow, and PyQt6, plus the development tools pytest and ruff.

A checkout can still use a virtual environment and the requirements files. `requirements.txt` is runtime only; tests need `requirements-dev.txt`:

```bash
python3 -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt
```

The tests and the headless render need no display.

## Run

```bash
sidereel
sidereel-cli --star sun --planet earth --planet jupiter --output output/sidereel.png
python -m simulations.star_system  # live interactive GUI
python -m simulations.star_system --star sun --planet earth --planet jupiter --output output/sidereel.png
```

With no arguments, `python -m simulations.star_system` and `sidereel-cli` open the Qt desktop. Explicit flags open the legacy Matplotlib UI, and `--output` saves a still image without a display. The import package is still `simulations.star_system`.

Details, presets, and the other command-line flags are in each simulation's README:

- [Sidereel](simulations/star_system/README.md) — tunable stars and planets with full N-body gravity

## Development checks

Install the development tools in your virtual environment. Either command is enough:

```bash
pip install -e ".[dev]"
python -m pip install -r requirements-dev.txt
```

Run both checks from the repository root before opening a pull request:

```bash
python -m ruff check .
python -m pytest -q
```

The test suite covers physics, population generation, body authoring, persistence, recorded playback, collisions, controls, and rendering. Physics tolerances are listed in the [Sidereel README](simulations/star_system/README.md). Tests select Matplotlib's Agg backend and need no display.

[Ruff](https://docs.astral.sh/ruff/) checks application and test code using the rules in `ruff.toml`: Python errors, undefined/unused names, and ambiguous statements. Its version is pinned in `requirements-dev.txt` so local checks and CI use the same rules. To apply supported automatic fixes, run `python -m ruff check . --fix`, review the diff, and rerun both checks. Formatting is not enforced.

The [CI workflow](.github/workflows/ci.yml) runs on every pull request (including PRs targeting another feature branch), pushes to `master`, and manual dispatch. Lint runs as a separate job; any lint or test failure fails CI. The complete pytest suite runs on Linux with Python 3.10 and 3.14, plus Windows with Python 3.14. Python 3.10 is the documented minimum; the Windows job also checks the desktop development platform. GitHub installs these interpreters on its runners; you only need one supported Python locally.

The `Wheel` job builds the sdist and wheel, installs that wheel into a clean virtual environment, and smoke-tests `sidereel-cli --help` plus a short headless `--output` render. Its check name is separate from the four jobs below.

To enforce passing checks before merging, configure branch protection or a ruleset to require `Lint` and the three `Tests (...)` jobs after the workflow's first run. The `Wheel` job can be required as well; it does not replace those names.

## Publishing

[`.github/workflows/release.yml`](.github/workflows/release.yml) publishes to PyPI when a `v*` tag is pushed. It uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OpenID Connect) and does not read a stored API token.

One-time setup, before the first tag:

1. Sign in to PyPI as the account that should own the project name `sidereel`.
2. The project is not on PyPI yet, so add a **pending publisher** (Account settings → Publishing):
   - PyPI project name: `sidereel`
   - Owner: `AllHailSeizure`
   - Repository name: `sidereel`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. In this GitHub repository, create an environment named `pypi` (Settings → Environments). No secret is required. Optional protection rules can limit who is allowed to publish.
4. Bump `version` in `pyproject.toml` and merge that change to `master`. Push a tag whose name is `v` plus that version exactly, for example version `0.1.0` and tag `v0.1.0`.

The workflow checks that the tag matches `version`, builds the sdist and wheel, smoke-tests the wheel, then publishes those files. A tag that does not match the version fails before upload.

## Adding a simulation

Create `simulations/<name>/` with:

- `core.py` — the equations and the integrator, importable without Matplotlib
- `__main__.py` — the command-line entry (`python -m simulations.<name>`)
- `plotting.py` — figures and the animation
- `README.md` — the equation, how to run it, and what is checked

Add the checks under `tests/`. Keep new runtime dependencies in `requirements.txt` and `pyproject.toml` only when a simulation needs them.
