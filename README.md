# Sidereel

A Newtonian N-body sandbox. Regular starting experiments integrate in three dimensions, and the overhead view projects that motion onto the plane. Build a system, watch gravity work, keep the picture.

![Sun, Earth, and Jupiter through one year](docs/orbit.gif)

One year of the Sun, Earth, and Jupiter, rendered by `sidereel-cli`.

## Features

- Unsoftened Newtonian gravity. Regular presets store three components. An explicit star-and-planet system can stay planar. Masses pull on each other. Test particles only fall.
- A fixed-step Yoshida integrator. A close encounter halts the run and keeps the last good state.
- A PyQt6 window filled by the sky. Menu, or M, lays a sidebar over the scene. The view stays put.
- Six starting experiments: the Solar System, a star at twice the Sun's mass, Earth and the Moon, 32 asteroids in a ring at 3.2 AU, a Sun–Jupiter rock ring with seed 7, and a head-on merger of two solar-type stars.
- Add a moon by choosing its planet and a center-to-center distance in kilometers. Gravity moves the moon after that; the planet does not hold it in place.
- Add a catalog or custom body while a run is paused. Drop it in the scene, drag the velocity arrow, then confirm. The clock and the other bodies stay put. Reset returns to the starting setup.
- A CLI that integrates the system and writes a PNG. No display required.
- Save the initial setup, or save the current state as a new setup. Calculate a recording, then play the samples. Playback speed does not recalculate the motion.
- Optional perfect merge for nonstellar contact. Point tracers are absorbed with no mass feedback. Contact with a star halts. The default still halts on a close encounter and does not merge.
- Optional tidal dissipation on close passes of massive bodies. It is off unless the experiment turns it on, so a tide-free run is unchanged.
- Optional outside pulls for very wide orbits: a linear galactic tide from the solar neighborhood, and optional seeded stellar flybys. Off unless the experiment file turns the module on.
- An isolated planet holds its period to a relative 1e-7. At 1 AU, solar gravity sits about 30 ppm above JPL's GM, inside a 1e-4 bound.
- A late-stage formation prototype that follows rocky embryos, records which parcels built each body, and compares two runs from one checkpoint.

## Prerequisites and installation

Python 3.10 or newer.

```bash
pipx install sidereel
```

`sidereel` opens the desktop. `sidereel-cli` renders.

The project is not on PyPI yet. Until the first release, install from git:

```bash
pipx install git+https://github.com/AllHailSeizure/sidereel
```

From a checkout:

```bash
pip install -e .
```

On Debian and Ubuntu the desktop also needs `libegl1` and `libopengl0`.

## Usage

Open the GUI:

```bash
sidereel
```

Build an experiment from a file:

```bash
sidereel --new experiment.json
sidereel --build experiment.json
```

`--new` writes a blank experiment: dimensions, catalog bodies, opt-in physics, and the run. Newtonian gravity is what Sidereel uses. Edit that file, then `--build` checks it and writes `experiment.sidereel` beside it. The playback file holds the samples and duration for one run, or for each run of a sweep or an open scan. Starting `sidereel` with no arguments still opens the desktop. `sidereel-cli` still renders a catalog system, and it accepts the same two commands.

You get the experiment list, including the Solar System and a Sun–Jupiter ring. Save initial setup stores the beginning. Save current state as new setup stores the evolved system under a new name. `sidereel-cli` with no arguments opens the same window.

Sun–Jupiter ring places 32 mass-zero rocks evenly around the Sun at 3.2 AU (seed 7). Rotating view keeps the Sun–Jupiter direction fixed. Hollow L4 and L5 marks are geometric guides, not rocks. Select a rock to read its separation from Jupiter and how long the drawn trail is. Matched no-Jupiter run uses the same rock starting vectors. Collision mode is none, and a halt or a shorter run is labeled as incomplete.

Add body works in the starting setup and during a paused run. A click drops the body, the arrow sets its speed and direction, and confirm adds it without rewinding the clock. The circular arrow is only a two-body start around the body you pick, including a second star. Reset drops a live insertion and returns to the starting setup.

Render the year in the picture:

```bash
sidereel-cli --star sun --planet earth --planet jupiter --years 1 --output output/sidereel.png
```

Starting positions only. The default is the Sun, Earth, and Jupiter:

```bash
sidereel-cli --output output/start.png
```

Inner planets, half a year:

```bash
sidereel-cli --star sun --planet mercury --planet venus --planet earth --years 0.5 --output output/inner.png
```

`--star` is `proxima`, `sun`, `sirius`, or `rigel`. Repeat `--planet` for each body: `mercury`, `venus`, `earth`, `mars`, `jupiter`, `saturn`, `uranus`, `neptune`, `pluto`, `ceres`.

`--years` works only with `--output`. So do `--sample-stride` and `--max-recording-mib`. Name bodies without `--output` and the Matplotlib window opens instead.

In a checkout, `python -m simulations.star_system` is the same program. `sidereel-cli --help` lists every flag.

## Formation prototype

`simulations.star_system.formation` starts from a star and massive embryos, merges them on contact, and keeps the mass each parcel contributed. Save a checkpoint, change one velocity, and play both runs across the years they share. The bundled embryo radii are inflated, and the run does not include gas or fragmentation. The measured benchmark and the recommended scope are in [docs/formation-feasibility.md](docs/formation-feasibility.md).

## License

MIT. Copyright 2026 Nate Bosma. See [LICENSE](LICENSE).
