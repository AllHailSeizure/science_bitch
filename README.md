# Sidereel

A planar N-body sandbox. Build a system, watch gravity work, keep the picture.

![Sun, Earth, and Jupiter through one year](docs/orbit.gif)

One year of the Sun, Earth, and Jupiter, rendered by `sidereel-cli`.

## Features

- Unsoftened Newtonian gravity in the plane. Masses pull on each other. Test particles only fall.
- A fixed-step Yoshida integrator. A close encounter halts the run and keeps the last good state.
- A PyQt6 window filled by the sky. Menu, or M, lays a sidebar over the scene. The view stays put.
- Five starting experiments: the Solar System, a star at twice the Sun's mass, Earth and the Moon, 32 asteroids in a ring at 3.2 AU, and a Sun–Jupiter ring with seed 7.
- A CLI that integrates the system and writes a PNG. No display required.
- Save the initial setup, or save the current state as a new setup. Calculate a recording, then play the samples. Playback speed does not recalculate the motion.
- Optional perfect merge for nonstellar contact. Point tracers are absorbed with no mass feedback. Contact with a star halts. The default still halts on a close encounter and does not merge.
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

You get the experiment list, including the Solar System and a Sun–Jupiter ring. Save initial setup stores the beginning. Save current state as new setup stores the evolved system under a new name. `sidereel-cli` with no arguments opens the same window.

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
