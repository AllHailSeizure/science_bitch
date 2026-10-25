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

## Tests

```bash
pytest
```

The tests are the physics checks; the tolerances asserted are listed in the [star system README](simulations/star_system/README.md).

## Adding a simulation

Create `simulations/<name>/` with:

- `core.py` — the equations and the integrator, importable without Matplotlib
- `__main__.py` — the command-line entry (`python -m simulations.<name>`)
- `plotting.py` — figures and the animation
- `README.md` — the equation, how to run it, and what is checked

Add the checks under `tests/`. Keep new packages in `requirements.txt` only when a simulation needs them.
