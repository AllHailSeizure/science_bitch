# Interactive star system

Run from the repository root with `requirements.txt` installed:

```powershell
python -m simulations.star_system
python -m simulations.star_system --star sun --planet earth --planet jupiter --years 1 --output output/star-system.png
python -m pytest -q
```

The Matplotlib observatory has a full-window overhead scene, glowing stars,
illuminated bodies, trails, and a sidebar. **Menu [M]** hides the overlay without
changing the camera or simulation. The CLI retains its original four star and
ten planet/dwarf-planet selectors. General catalog bodies and custom states are
available through the authoring UI and Python API. A desktop backend is needed
for interactive input; `--output` uses Agg without a display.

## Authoring a body

**Add a body** pauses the active experiment and opens a draft in the sidebar.
Choose a category, browse with `<` / `>`, or enter a catalog key and load it.
The custom category starts with unknown physical values. Every addition creates
a distinct instance ID, including repeated copies of the same catalog definition.

The editor exposes name, kind, mass in **solar masses**, physical radius in
**AU**, optional stellar temperature in **K**, Cartesian position in **AU**, and
velocity in **AU per Julian year**. Known catalog properties retain field-level
provenance. Changed physical values are marked user-authored overrides. A blank
value means unknown. Gravity requires a known mass; radius can remain unknown
because contact is not modeled. No physical contact claim can be made for such a
body. Unknown radii use a fixed display marker without assigning a physical
value. Unknown temperature uses an explicitly illustrative color fallback.

The selected massive body's ID is shown as the default reference. If a particle
is selected, a reference must be chosen. **Reference: next massive / world**
cycles the choices; entering `world` explicitly uses the inertial world frame.
Position and velocity components are relative to that reference. They resolve
once when committed: the reference position and velocity are each added once.
References do not constrain subsequent motion. Preparation reference chains must
be acyclic and every reference must exist.

Click the scene to place the draft. Display coordinates are compressed by
`q = r * asinh(|r| / 0.05 AU) / |r|`; clicks invert that mapping and restore the
view origin before being interpreted as AU. The velocity arrow uses a separate
**linear scale of 12 pixels per AU/year**. Its tip changes both velocity
components when dragged; twice the speed produces twice the arrow length at
the same scale. The legend reports components, speed, and units. Arrow direction
and length describe velocity relative to the chosen reference, not compressed
position units.

**Circular aid about reference** supplies the prograde perpendicular two-body
initial guess `sqrt(G * (M_reference + m_body) / separation)`. It requires a known
positive reference mass and nonzero separation. Reference velocity is added at
commit time, once. This aid does not promise stability, binding, or capture in
an N-body system.

**Confirm** validates the draft and leaves playback paused until Resume.
**Cancel**, Escape, or closing the app discards the draft; cancel restores the
previous pause state and preserves the evolved positions, velocities, and time.
Invalid drafts remain open with a recoverable explanation and preserve the
active state. Keyboard playback shortcuts are disabled while authoring; the
Menu button can still hide/show the editor, making its widgets inactive.

## Starting setup and evolved state

The application owns two distinct objects: immutable `App.initial_setup` and the
mutable `App.prepared.system` trajectory. **Live insertion** changes only the
active trajectory, preserving every existing position, velocity, and elapsed
time exactly. It appends a new row even when particles already occupy earlier
rows. It retains the camera, existing scene artists, selected identity, and
follow target. An inserted body's history starts at insertion; it has no
invented past trail.

**Restart** reconstructs the selected starting setup, removes unsaved live
insertions, resets elapsed time and trails, and retains the pause state.
**Prepare setup** explicitly returns to that starting setup and pauses it.
Add, **Edit selected**, and **Remove selected** then update its initial bodies.
The complete candidate setup is validated before replacement, including body
and population references. Removing a referenced body fails recoverably.
Resume or Step leaves preparation and starts ordinary live evolution.

**Prepare circular** opens the retained star/planet preset controls after
entering preparation. These keep the original unique-planet, golden-angle,
circular, barycentric construction and logarithmic distance sliders. They
rebuild the starting state when changed. The circular controls are disabled
once the starting setup has been generically edited, so they cannot silently
discard custom bodies. General preparation supports arbitrary physical
positions and velocities, one isolated body, multiple stars, and stars-only
binaries. No single-star assumption is imposed by the general renderer.

Save/reload and calculated playback are later plans. The agreed save workflow
will offer both the original initial setup and the current evolved state as a
separately named new setup with a fresh initial epoch and source-time
provenance. Explicit captured particles must be preserved, rather than being
regenerated over an evolved state. No capture/save UI is implemented here.

## Navigation and evolution

| Control | Action |
| --- | --- |
| Menu [M], `m` outside editors | Show/hide sidebar, preserving state and camera |
| Pause / Resume, Space | Pause or resume; Resume exits preparation |
| Step, `.` | Pause and attempt one fixed step; exit preparation |
| Mouse wheel over scene | Zoom around cursor |
| Left drag outside authoring | Pan and stop following |
| Click body or body-list row | Inspect its stable identity and physical properties |
| Mouse wheel over body list | Scroll six visible rows |
| Follow body | Follow the selected body at current zoom |
| Fit view, Home, `0` | Frame current bodies and stop following |
| Restart | Restore the selected starting setup |

Sidebar gestures do not pass through to the scene. Stars and nonstars use their
own metadata, independent of numeric row order. The inspector gives a larger
surface preview; marker sizes are exaggerated and compressed monotonically
with physical radius. Surface textures and colors are procedural illustrations,
not measured surface maps.

Lighting chooses the star with the largest illustrative apparent bolometric
flux `R_star^2 * T_star^4 / distance^2`. A star with unavailable radius or
temperature uses one solar luminosity as a disclosed display fallback. All
stars are luminous. With no star, a fixed illustrative direction keeps a body
readable. No radiation force, photometry, thermal evolution, or destruction by
heat is implied. Blackbody colors use the approximate
[Tanner Helland fit](https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html).

The live timer targets 60 updates/sec and yields after roughly 6 ms of physics
work or 150 completed steps per callback. Requested years/sec changes wall-clock
pacing, never the integration step. A compute-limit message reports lag. Blit
backends cache static pixels; animated overlay axes are excluded from that
cache and painted once, last. Their text is not animated separately. Other
backends redraw fully. Resize preserves center and compressed units/pixel.

## Numerical model and limits

Gravity is unsoftened Newtonian gravity in 2D:
`a_i = G sum(j != i) m_j (r_j - r_i) / |r_j - r_i|^3`.
Positive masses mutually interact; zero-mass test particles feel massive bodies
without backreaction or particle/particle gravity. Units use 1 AU = 149597870700 m,
1 solar mass = 1.98847e30 kg, 1 Julian year = 365.25 days, and SI
G = 6.67430e-11. No integrator or dependency has been replaced.

The fixed-step fourth-order Yoshida composition of velocity Verlet uses
[Yoshida 1990](https://doi.org/10.1016/0375-9601(90)90092-3). The legacy circular
constructor retains shortest initial orbital period / 600. A general
`InitialSetup` carries an explicit positive fixed `dt` in years; unresolved
inputs fail instead of silently adjusting that requested step.

Before and throughout each candidate step, every interacting pair must have
both its gravitational timescale `sqrt(distance^3 / (G*(m_i+m_j)))` and relative
crossing time `distance / |v_i-v_j|` at least ten timesteps. This includes
massive/particle pairs and detects high-speed states even when gravity alone
would appear slow. Failure at any internal kick/drift leaves the whole step's
positions, velocities, and time unchanged. Earlier completed steps remain valid.
A halt explains that smaller fixed dt or wider spacing is needed. Preparation
also rejects missing/cyclic references, nonfinite states, duplicate IDs, unknown
required masses, and overlap of known physical surfaces.

Live insertion can shorten the step to half the validated resolution limit.
The displayed insertion event records old/new dt and starts a new integration
segment, while retaining old vectors/time exactly. Each uninterrupted segment
has a fixed step; an edited trajectory is not claimed to be one unmodified
fixed-step Hamiltonian integration. Encounter guards are conservative checks,
not an accuracy/stability guarantee for arbitrary custom systems.

Step-halving fixtures, measured on this implementation, compare dt = 0.0005 and
0.00025 yr with a 0.000125 yr reference:

| Fixture | Duration | Coarse/reference maximum coordinate difference | Half/reference |
| --- | ---: | ---: | ---: |
| Equal 1-solar-mass circular binary, 2 AU separation | 2 yr | 3.53e-11 AU | 2.01e-12 AU |
| 3e-6-solar-mass flyby, (-4, 1) AU and (40, 0) AU/yr past a 1-solar-mass star | 0.2 yr | 2.63e-10 AU | 1.55e-11 AU |

Legacy tests retain isolated orbital-period relative tolerance 1e-7, Kepler-law
2e-7, energy 1e-7, angular/linear momentum 1e-11, and multi-planet period 0.5%
(which permits real perturbations). These tests establish their fixtures only.
There are no physical collisions, mergers, tides, relativity, stellar evolution,
3D, or scripted capture outcomes. Future generic inelastic collisions are
nonstellar only and halt on stellar contact. Stellar mergers belong to the
separate deferred issue #14, including binaries.

## Sourced catalog

All original definitions remain. Added definitions include Alpha Centauri A/B,
Sirius B, 51 Peg b, HD 209458 b, Moon, Io, Europa, Ganymede, Callisto, Titan,
Vesta, Eros, and Bennu. These are physical definitions, not precision ephemerides;
no host-specific placement is silently assigned. General placements need an
explicit reference/vector, so a moon definition does not presume a parent.

[catalog-sources.json](catalog-sources.json) contains each physical field's
URL, source version, retrieval date, original value/units, conversion,
uncertainty in original units, and measurement/fit/estimate status. Null
uncertainties mean unavailable, not zero. Legacy rounded values remain for
compatibility. Rigel's mass is explicitly an evolutionary estimate. GM-derived
masses use the model's G convention; GM errors do not include G's conversion
uncertainty. Effective asteroid diameters become volume-equivalent spherical
radii, not resolved irregular surfaces. Illustrative colors are not measured.

Primary records checked on 2026-10-30:

- [JPL satellite physical parameters](https://ssd.jpl.nasa.gov/sats/phys_par/sep.html): DE440, JUP365, SAT441 GM solutions and mean radii with uncertainties.
- [JPL SBDB API](https://ssd-api.jpl.nasa.gov/doc/sbdb.html): API 1.3 GM and effective-diameter records for Vesta, Eros, Bennu, with original references and sigmas.
- [Kervella et al. 2017](https://arxiv.org/html/1610.06185v1): Alpha Centauri A/B Table 5 parameters and uncertainties.
- [Bond et al. 2017](https://arxiv.org/html/1703.10625v1): Sirius masses and Sirius B atmosphere-derived radius/temperature; quoted internal fit errors exclude larger systematics.
- [NASA Exoplanet Archive 51 Peg](https://exoplanetarchive.ipac.caltech.edu/overview/51%20Peg) and [HD 209458](https://exoplanetarchive.ipac.caltech.edu/overview/HD_209458): selected `ps` default solutions, including original publication references. The 2026-10-30 51 Peg b solution reports true mass (Cont et al. 2026), not RV minimum mass; its radius is unknown. HD 209458 b uses Stassun et al. 2017 mass/transit radius. No composite model-estimated radius is substituted.
- [NASA planetary fact sheets](https://nssdc.gsfc.nasa.gov/planetary/factsheet/), [JPL planets](https://ssd.jpl.nasa.gov/planets/phys_par.html), [Anglada-Escude et al. 2016](https://arxiv.org/html/1609.03449v1), and [Moravveji et al. 2012](https://arxiv.org/html/1202.1836v1): retained rounded legacy definitions; the latter identifies Rigel's mass as an evolutionary estimate.

## Reproducible population compatibility

The existing **Population...** window still supports one star-relative
ring/disk/belt in a legacy circular setup. Apply/remove explicitly restarts and
pauses; cancel preserves evolved state and prior pause status. Generic setup
population UI integration belongs to the next plan; the old window is disabled
for a generically edited starting setup so it cannot discard custom bodies.

```powershell
python -m simulations.star_system --population belt --count 200 --inner 2 --outer 3.5 --seed 42
python -m simulations.star_system --planet jupiter --population ring --inner 3.2 --count 64 --angles even --seed 7 --years 1 --output output/ring.png
```

Ring uses one radius; disk samples annular area uniformly; belt samples radius
uniformly. Random or evenly spaced angles use NumPy PCG64 and an explicit
nonnegative seed. Counts are 1-2000, radii finite/outside the reference surface,
and velocities are circular prograde guesses relative to the reference state.
Restart reproduces the population. Individual particles can be inspected and
followed; their own trails are omitted to bound display work. There is no
physical clustering/accretion response or stability promise.

## Shared APIs for the following plans

- `model.py`: frozen `BodySpec`, `PropertySource`, `InitialBody`, `PopulationSpec`, and `InitialSetup`. Catalog keys identify definitions; body IDs identify instances. `PopulationSpec(id, reference_id, settings)` maps the stable reference to the legacy numeric index at initialization.
- `initialization.py`: `build_system(setup) -> PreparedSystem(system, body_ids, body_specs, setup, events, segment)`, `insert_body(prepared, body)`, `circular_relative_velocity(r, M, m)`, and `legacy_setup(...)`. All vectors are world frame after one-time reference resolution. Insertion preserves old row order, including particle rows.
- `history.py`: `StateFrame.capture(prepared)` copies time/dt/IDs/specs/masses/world r/v/events into read-only arrays. `App.history` retains 1500 frames; each displayed trail has at most 400 vertices. `App.trails` remains a relative-array compatibility adapter.
- `rendering.py`: `draw_system(ax, prepared, history=()) -> Scene`; retained `draw(...)` accepts the old numeric/circular arguments. `Scene.artists_by_id`, `labels_by_id`, `trails_by_id`, `origin_id`, and `origin` support identity-based integration. `Scene.update(prepared, history)` appends new artists without resetting the camera.
- `authoring.py`: `BodyEditor` and its `reference_state`, `vectors`, `draft_body`, `commit`, `finish` draft lifecycle. `App.edit_body`, `prepare_setup`, `selected_id`, `_install_prepared`, and `_sync_painter` provide the controller integration points. Register new widget axes with sidebar visibility and the painter's overlay axes.

The body-authoring tests provide arbitrary/binary, reference-chain, populated
insertion, live draft events, invalid/cancel, inverse mapping, velocity scaling,
and multi-star lighting fixtures. The generic renderer never assumes a fixed
star/planet prefix. Future snapshot capture can store explicit particle bodies
with `population=None`, avoiding regeneration of evolved particle coordinates.

## Checked outcomes

Windows/Python 3.14 verification uses writable ignored `output/` basetemps because
the default Windows pytest temp directory is inaccessible in this environment.
Agg event tests exercise the real Matplotlib callback flow and normal exports;
native desktop input, perceived smoothness, and native multiwindow behavior
remain unverified.

Final commands on the completed tree:

```powershell
C:\Python314\python.exe -m pytest -q --tb=short --basetemp output/pytest-authoring-final-15
C:\Python314\python.exe -m simulations.star_system --star sun --planet earth --planet jupiter --years 1 --output output/issue-5-compatibility.png
git diff --check
git diff --check codex/issue-4-overlay...HEAD
```

The full suite passed **147 tests in 251.63 seconds**. Its 14 warnings are the
expected Agg warning that the separate population window cannot be shown.
The CLI command exited successfully and produced a visually inspected,
nonempty 1350x1350 PNG (102,842 bytes). Normal App exports at
`output/authoring-reviewed.png` (1480x919) and
`output/catalog-unknown-reviewed.png` were visually inspected for the draft
layout, physical units, velocity legend, and unknown-radius state. These
artifacts remain in ignored `output/`. Focused event tests also checked cached
versus full draws with the editor open. Whitespace validation passed.
