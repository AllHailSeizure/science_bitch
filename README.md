# science_bitch

Small, well-established physics simulations in Python. Each one uses an integrator suited to the problem and checks itself against a known result.

## Install

From the repository root, with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependencies are numpy, matplotlib, pillow, and pytest. No display is required. The scripts use Matplotlib's Agg backend unless you pass `--show` and a display is available.

## Run

```bash
python -m simulations.harmonic
python -m simulations.harmonic --compare
python -m simulations.orbit
```

Plots and an animation are written under `output/`. Details, presets, and the other command-line flags are in each simulation's README:

- [Harmonic oscillator](simulations/harmonic/README.md) — a mass on a spring, with damping and a sinusoidal drive
- [Two-body orbit](simulations/orbit/README.md) — a planet and a star on elliptical orbits

## Tests

```bash
pytest
```

The tests are the physics checks. Tolerances below are the ones asserted in `tests/`.

| Simulation | Check | Tolerance |
| --- | --- | --- |
| Oscillator | max \|x − analytic\| over 20 periods, 2000 steps/period | 2×10⁻⁴ |
| Oscillator | undamped period against 2π√(m/k), 30 periods, 1000 steps/period | relative 1×10⁻⁵ |
| Oscillator | undamped relative energy excursion, 20 periods, 2000 steps/period | 1×10⁻⁵ |
| Orbit | relative energy drift, 20 orbits, 1000 steps/orbit | 1×10⁻⁶ |
| Orbit | relative angular-momentum drift, same run | 1×10⁻⁹ |
| Orbit | periapsis returns to its start, as a fraction of a | 1×10⁻⁴ |
| Orbit | departure from the polar ellipse, as a fraction of a | 2×10⁻⁴ |
| Orbit | Kepler: \|T²/a³ − 4π²/μ\| / (4π²/μ) at several a | 1×10⁻⁶ |

μ = G(M + m). Representative numbers from a test run are in the simulation READMEs.

## Adding a simulation

Create `simulations/<name>/` with:

- `core.py` — the equations and the integrator, importable without Matplotlib
- `__main__.py` — the command-line entry (`python -m simulations.<name>`)
- `plotting.py` — figures and the animation
- `README.md` — the equation, how to run it, and what is checked

Add the checks under `tests/`. Keep new packages in `requirements.txt` only when a simulation needs them.
