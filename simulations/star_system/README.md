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

The GUI is a live Matplotlib window using RadioButtons, Buttons, and Sliders.
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
adding a planet, changing star, or pressing Reset rebuilds the initial system,
clears trails and resets time. Reset preserves current selections and distances.
The speed slider sets requested years per wall-clock second without changing
the physics timestep. Frames have a bounded compute budget; a visible message
reports when the requested speed exceeds it. Close encounters that cannot be
resolved halt with a message; reset or adjust distances to resume.

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

The view follows the star. A radial `asinh(r / 0.05 AU)` mapping preserves
angles while compressing distance, and the extent expands automatically. Rings
carry physical AU labels. Shapes and apparent speeds are distorted by this
mapping. Trails contain at most 1500 integration samples. **Marker sizes are
not to scale**: star diameter in points is `12 + 16 log10(1 + 8 R/Rsun)`;
planet diameter is `5 + 2 log10(1 + R/km / 1000)`. These monotonic compression
rules make small planets visible and giants visibly larger than dwarfs.
Stellar colors approximate the Planckian locus using
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

The full repository suite passes: 45 tests. GUI verification under `xvfb-run`
used mouse events to exercise star and planet radio selectors, Add, distance,
speed and Reset, checked timer-driven time advancement, and captured Proxima
and Rigel windows. Verification artifacts are outside the checkout under
`/workspace/science_bitch/artifacts/` and are not committed.
