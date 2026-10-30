# Interactive star system

Run from the repository root after installing `requirements.txt` into `../venv`:

```bash
source ../venv/bin/activate
python -m simulations.star_system
python -m simulations.star_system --star proxima --planet earth --planet neptune
# Display-free still image, evolved one Julian year:
python -m simulations.star_system --star sun --planet earth --planet jupiter --years 1 --output output/star-system.png
pytest -q
pytest -s tests/test_star_system.py
```

The GUI is a live Matplotlib observatory with dark control panels, illuminated planet disks, glowing stars, fading trails, and a body inspector.
PyQt6 provides a backend when Python lacks Tk. A desktop display is required
for the GUI; the output command forces Agg and needs no display. Stars are
`proxima`, `sun`, `sirius`, `rigel`; planet keys are their lowercase names.
Eight planets, Pluto, and Ceres are available. The Moon is omitted because its
usual orbital distance is geocentric, unlike this heliocentric catalog.

Pick a star, pick a planet, then click **Add planet**. Each body can be added
once; selecting an existing body is a no-op. Distance sliders are logarithmic,
labeled in AU, and start at catalog distances. They span 0.005–80 AU, with the
lower limit raised above the selected star's surface. Switching to a giant
clamps any now-interior orbit to 1.02 stellar radii. Moving a distance slider,
adding a planet, changing star, or pressing Restart rebuilds the initial system,
clears trails and resets time. Restart preserves current selections and distances.
The speed slider sets requested years per wall-clock second without changing
the physics timestep. Each frame advances at most 150 steps and yields after
approximately 6 ms of physics work (at step boundaries); a visible message
reports when the requested speed exceeds it. Close encounters that cannot be
resolved halt with a message; reset or adjust distances to resume.

## Navigation and playback

| Control | Action |
| --- | --- |
| Pause / Resume, or Space | Stop or resume evolution without rebuilding the system |
| Step, or `.` | Pause and advance one fixed physics timestep |
| Mouse wheel over the view | Zoom around the cursor |
| Left drag over the view | Pan; this also stops following a body |
| Click a star or planet | Select it and show its details in the inspector |
| Follow body / Unfollow | Keep the selected body centered at the current zoom |
| Fit view, Home, or `0` | Stop following and frame the current system |
| Restart | Clear time and trails, keeping selected bodies, distances, and pause state |

The speed slider requests simulated years per second. Zoom is still on the
compressed radial scale, not a linear AU scale. Shaded disks retain a readable
screen size while zooming. Adding a body or changing a star/distance rebuilds
the initial system as before. This update changes presentation and controls;
moons, swarm particles, and scenario saving remain part of the
[Evolving star-system sandbox milestone](https://github.com/AllHailSeizure/science_bitch/milestone/1).

The desktop timer targets 60 updates per second and accounts for time spent in
the callback. Actual frame rate depends on the system, viewport, and backend.
On backends that support blitting, static controls and guides are cached while
moving bodies, trails, and readouts are repainted. Camera changes, resizing,
and edited controls refresh that cache. Other backends use normal full redraws.

## Reproducible populations (Issue #6 Foundation)

Create a population through the command line, either in the live observatory or
as a display-free export:

```bash
python -m simulations.star_system --population belt --count 200 --inner 2 --outer 3.5 --seed 42
python -m simulations.star_system --planet jupiter --population ring --inner 3.2 --count 64 --angles even --seed 7 --years 1 --output output/ring.png
python -m simulations.star_system --population disk --inner 1.5 --outer 4 --count 1000 --seed 42 --output output/disk.png
```

`ring` uses a single radius (`--outer` defaults to `--inner` and must equal it).
`belt` samples radius uniformly between the bounds. `disk` samples uniformly in
annular area, so radius squared is uniform. Both require inner < outer; their
default outer radius is 3.5 AU. All radii are initial distances from the star's
center, must be positive and finite, and must lie outside its surface. These are
planar populations, without thickness, eccentricity dispersion, or inclination.
`--angles random` samples independent uniform angles; `--angles even` spaces
angles equally with a seeded random orientation. Quantity is 1-2000 particles
in total; seeds are nonnegative integers. The same settings, catalog bodies,
distances, and software versions reconstruct the same state using NumPy PCG64.
Restart regenerates that state and resets time and trails. A different seed
changes the realization. Changing a planet, distance, or star also restarts it.
Selecting a star whose surface contains the population is rejected recoverably.

These particles have zero mass: they feel every massive body's unsoftened
Newtonian gravity, but exert no gravity and do not interact with one another.
Their circular prograde starting velocities use `sqrt(G M_reference / radius)`
relative to the reference body's position and velocity. This is an initial guess,
not an equilibrium or a promise of stability in a perturbed system. No forces,
clustering, or capture outcomes are scripted. There is no physical collision,
accretion, bounce, or fragmentation response. Overlapping test particles do not
interact. An unresolved particle/massive-body or massive/massive close encounter
halts the whole run under the existing timescale guard; time reports completed
steps only. Visible dots have no physical contact radius.

Particles appear as small points alongside the major bodies. Pause, Resume,
Step, Fit view, click-to-inspect, and Follow apply to populated systems. Particle
inspection reports star-relative distance and speed; trails remain limited to
major bodies to bound display work. Restart reproduces the population while
retaining pause state. Native desktop interaction has not been manually verified
on this cloud workspace; Agg checks exercise playback, restart, selection,
following, and rendering, and CLI exports use the same scene.

The reusable Python API is `Population` plus
`with_population(system, settings, reference_radius=...)`. It returns an
independent `System`, preserves existing bodies and time, and appends test
particles. `reference` is the index of a positive-mass body; supply its known
physical radius in AU for surface validation (the API default is a point).
It supports arbitrary massive reference configurations, including multiple
stars supplied by a caller, but those systems have no stability guarantee.
The current catalog CLI/observatory offers star-relative populations only.
Initialization can shorten the fixed timestep to the inner-radius reference
period / 600. This constructor is not a live insertion operation.

Measured on the cloud workspace (Python 3.12, NumPy 2.3.5), 100 physics steps
with Sun/Earth/Jupiter took about 0.65, 0.82, and 1.59 ms/step for 200, 1000,
and 2000 particles respectively. These are indicative physics-only timings,
not desktop frame-rate promises. Gravity/encounter work scales with total bodies
times massive bodies, rather than with the square of the particle count. More
massive bodies, close inner radii, drawing, and stored history add cost. The
existing live work budget reports when it cannot achieve the requested speed.

Verification covers seeded initial states/evolution, radial distributions,
equally spaced angles, circular reference velocities, reference frame offsets,
timestep selection, one isolated ring revolution (position error < 1e-7 AU),
absence of particle backreaction, and close-encounter rejection. A matched
32-particle ring at 3.2 AU with seed 7 evolves for one initial ring period:
removing Jupiter's gravitational mass while preserving initial positions,
velocities, and timestep produces a maximum star-relative displacement
difference greater than 0.01 AU. This demonstrates perturbation, not Trojan
capture or long-term stability.

This completes the cloud-testable population foundation, not all of #6.
Interactive population authoring through #5 and preparation/save/reload and
calculate-first playback through #8 remain future integration work. Settings
are explicit immutable data ready for those workflows; persistence is not yet
implemented.

## Physics and display

`core.py` imports no plotting code. Every body, including the moving star, feels
all others via unsoftened Newtonian gravity:

`a_i = G Σ(j != i) m_j (r_j - r_i) / |r_j - r_i|³`.

Units are AU, solar masses, and Julian years. G is converted from SI
(6.67430e-11), using 1 AU = 149597870700 m, 1 solar mass = 1.98847e30 kg,
and 1 year = 365.25 days. Initial relative velocities are circular two-body
values `sqrt(G (M + m) / a)` about the chosen star; the whole system is then
shifted into its barycentric rest frame. In a multi-planet system these are
osculating circular initial conditions, not an exact N-body equilibrium.
Initial phases are spaced by the golden angle to avoid coincident placements.

The fourth-order Yoshida composition of velocity Verlet uses Yoshida's
coefficients (Yoshida 1990,
[doi:10.1016/0375-9601(90)90092-3](https://doi.org/10.1016/0375-9601(90)90092-3)).
The fixed step is the **shortest initial planet period / 600**, recomputed on
every GUI rebuild, including adding a close-in body. Fixed steps preserve
symplecticity; no frame-dependent or state-dependent adaptive integration is
used. Before each step we require every pair's local gravitational timescale
`sqrt(r³ / (G (m_i+m_j)))` to exceed 10 steps. This guard detects unresolved
encounters; it is not a guarantee for arbitrary unstable systems. There is no
collision/merger, tidal, relativistic, stellar evolution, or habitable-zone
model. Bodies are point masses; planets are not physically destroyed by heat.

The initial view is centered on the star; drag to pan, scroll to zoom, or follow a selected body. A radial `asinh(r / 0.05 AU)` mapping preserves
angles while compressing distance. Fit view frames the current positions. Rings
carry physical AU labels. Shapes and apparent speeds are distorted by this
mapping. Trails retain at most 1500 integration samples and draw at most 400
vertices each, including their first and latest samples. This display sampling
does not change integration or the retained history. **Marker sizes are
not to scale**: star diameter in points is `12 + 16 log10(1 + 8 R/Rsun)`;
planet diameter is `14 + 7 log10(1 + R/km / 1000)`. These monotonic compression
rules make small planets visible and giants visibly larger than dwarfs.
Planet shading faces the star in the orbital plane, with a small ambient term to keep the night hemisphere visible. Surface markings are stylized procedural illustrations, not geographic maps. Stellar halos, decorative background stars, and shading do not affect the physics. Camera limits persist between frames; **Fit view** recenters and frames the current system. Stellar colors approximate the Planckian locus using
[Tanner Helland's temperature-to-RGB fit](https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html).
A 3050 K dwarf appears warm orange-red, not saturated red; hotter stars appear
blue-white. Brightness is normalized. This is an approximate blackbody display,
not synthetic photometry with absorption lines. Planet colors are illustrative
natural-color choices, not computed atmospheres.

## Catalog and sources

Values are representative rounded measurements, not a precision ephemeris.
Stellar uncertainty (especially Rigel's mass/radius) exceeds numerical errors.

| Star | Mass (solar) | Radius (solar) | Effective temperature (K) |
| --- | ---: | ---: | ---: |
| Proxima Centauri | 0.120 | 0.141 | 3050 |
| Sun | 1 | 1 | 5772 |
| Sirius A | 2.063 | 1.7144 | 9845 |
| Rigel | 21 | 78.9 | 12100 |

Sources: [Anglada-Escudé et al. 2016](https://arxiv.org/abs/1609.03449)
(Proxima), [NASA Sun fact sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html)
(Sun), and [Bond et al. 2017](https://arxiv.org/abs/1703.10625) and
[Davis et al. 2011](https://arxiv.org/abs/1010.3790) (Sirius). Rigel uses
the representative parameters presented by
[Moravveji at KITP](https://www.on.kitp.ucsb.edu/online/asteroseismo-c11/moravveji/pdf/Moravveji_AsteroSeismoConf_KITP.pdf)
(radius and temperature, citing Przybilla et al.) and the 21-solar-mass
evolutionary estimate in the [Rigel compilation](https://en.wikipedia.org/wiki/Rigel).
Planet masses, mean radii and rounded semimajor axes use the
[NASA NSSDC fact sheets](https://nssdc.gsfc.nasa.gov/planetary/factsheet/) and
[JPL physical parameters](https://ssd.jpl.nasa.gov/planets/phys_par.html)
(including Pluto and Ceres); Ceres' distance is 2.767 AU. Actual eccentricities,
inclinations, epochs and satellite systems are intentionally discarded. Any
catalog planet can orbit any catalog star outside its physical surface.

## Display-free verification and tolerances

All reported errors are maximum excursions unless specified. Tests do not import
Matplotlib. All 40 star/body combinations are simulated for three orbits; the
first full unwrapped angular revolution is linearly interpolated in time. The
Earth/Jupiter test evolves for three Jupiter orbits and measures each body's
first revolution. Kepler's law is tested with the measured period, not merely
by comparing two analytic formulas.

| Check | Asserted tolerance | Representative result (Python 3.13) |
| --- | ---: | ---: |
| Single-body period vs `2π sqrt(a³ / G(M+m))`, every star/body pairing | relative 1e-7 | 1.114e-8 |
| Measured `T²/a³` vs `4π²/G(M+m)`, same pairings | relative 2e-7 | 2.228e-8 |
| Energy, Sun + Mercury + Earth + Jupiter, 30 Mercury orbits | relative 1e-7 | 6.239e-14 |
| Angular momentum, same run | relative 1e-11 | 3.175e-14 |
| Total linear momentum drift / initial sum of scalar momenta, same run | 1e-11 | 2.074e-14 |
| Earth period, Sun + Earth + Jupiter | relative 0.005 | 9.286e-6 |
| Jupiter period, same system | relative 0.005 | 1.018e-5 |
| Energy, Proxima + Earth at 0.05 AU + Neptune at 30 AU, 10 inner orbits | relative 1e-7 | passes |

The 0.5% multi-body period tolerance allows physical perturbations of osculating
orbits by the other planet; the isolated two-body period is only a reference in
that case. Momentum is normalized by a nonzero initial scalar scale because the
initial net vector momentum is zero. Catalog tests require positive masses,
radii and temperatures, strictly increasing blue/red color ratio with stellar
temperature, and increasing star marker diameter with physical radius. Invalid
orbits, duplicate bodies, and nonfinite distances are rejected.

The suite includes 45 physics/catalog checks and 21 display/control checks.
The display checks use Agg and Matplotlib mouse/key events to verify lighting
direction, sprite positions, fading trails, retained artists/camera, pause and
single-step, time advancement, zoom, pan, selection/follow, rebuilding controls,
ten-body layout, and close-encounter halting. Rendering checks also cover cached
frames, bounded display trails, actual step accounting under a work budget,
resize/full-redraw equivalence, exports, and the non-blitting fallback.
Run `python -m pytest -q`.
Still exports use the same renderer as the live window. These headless checks
do not establish native desktop input behavior on every GUI backend.
