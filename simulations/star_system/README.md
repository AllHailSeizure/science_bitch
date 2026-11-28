# Sidereel

Run from the repository root with `requirements.txt` installed, or from an
install of this package. `sidereel` opens the Qt desktop (on Windows
that launcher uses `pythonw`, so no console window stays open).
`sidereel-cli` is the command line below, including headless `--output`.
`python -m simulations.star_system` is the same program; the import package name is unchanged.

```powershell
sidereel
sidereel-cli --star sun --planet earth --planet jupiter --years 1 --output output/sidereel.png
python -m simulations.star_system
python -m simulations.star_system --star sun --planet earth --planet jupiter --years 1 --output output/sidereel.png
python -m simulations.star_system --load output/experiment.json --years 0.01 --output output/restored.png
```

Dispatch is argument-dependent: no flags opens the Qt desktop; any explicit selector uses the
legacy Matplotlib controller or headless export described below.

| Entry | Example | First screen |
| --- | --- | --- |
| Qt desktop (default) | `python -m simulations.star_system` | **Choose experiment** / **Load saved experiment** |
| Legacy Matplotlib GUI | `python -m simulations.star_system --star sun --planet earth --planet jupiter` | Overhead scene in **PREP** at t=0; **Menu [M]** toggles the sidebar |
| Headless PNG | `python -m simulations.star_system --load output/experiment.json --years 0.01 --output output/restored.png` | No window; Agg writes the file |

`--years`, `--sample-stride`, and `--max-recording-mib` take effect only with
`--output`. They drive the shared incremental calculator before the PNG is saved.
Passing them without `--output` is rejected. Interactive GUIs open frozen at t=0 and
evolve through **Calculate...** (Qt or Matplotlib) or **Run live**.

```bash
python -m pytest -q
```

## Qt desktop (default)

`python -m simulations.star_system` with no arguments opens the native Qt
**Observatory** window. **Choose experiment** lists prepared templates (Solar
System, doubled stellar mass, added Moon, test-particle ring, the Sun–Jupiter
ring, and a head-on merger of two solar-type stars). **Load
saved experiment** opens validated JSON starting setups from disk.

**Sun–Jupiter ring** is the Sun, Jupiter, and 32 mass-zero rocks evenly spaced at 3.2 AU, seed 7. Collision mode is none. **Rotating view** keeps the current Sun–Jupiter direction on the positive horizontal axis; ordinary view uses inertial axes. Switching views does not change positions, velocities, or time. Hollow L4/L5 marks are ideal circular reference geometry, distinct from the rock markers, and they do not mean a rock is captured. Select a rock for its Jupiter separation, Jupiter-relative inertial velocity, and the trail's actual time span. **Matched no-Jupiter run** removes Jupiter and keeps the same rock vectors and fixed step. Compare only the shared recorded time; a halt or a cancelled run stays labeled and is not an equal-time result.

Each experiment opens at time zero with **Run**, **Pause**, **Reset**, and
**Edit starting setup**. The resizable sidebar has **Inspect**, **Add body**, and
**Edit body** tabs. **Add body** uses type/body dropdowns, information, **Add**,
then a scene click; Escape cancels. **Lunar** asks for the parent planet and the
moon's center-to-center distance in kilometers, separate from that planet's
position in the system; the click chooses only the direction. **Calculate a recording** uses the shared
bounded calculator; completed recordings appear under **Recorded results**.
**Save initial setup** writes that beginning, including population seeds and the collision mode.
**Save current state as new setup** writes evolved positions and velocities as a new named epoch and leaves the open experiment in place.
Choose `none`, `perfect_merge`, or `speed_angle`, then **Apply collisions + reset**, to rebuild the starting setup.
Perfect merge joins nonstellar bodies into one remnant and is not a general collision law.
Speed and angle merges, grazes, or fragments from impact speed and angle, and tracks fragments as bodies.
Both rules absorb point tracers with no mass feedback, ignore tracer–tracer contact, and log contact with a star as unmodeled without ending the run.
Native file dialogs save and load the existing validated JSON format.

**Focus selected and its orbiting bodies** zooms to a planet's local system;
**Fit view** returns to the full system. Select a body to see L4/L5 guides and
estimated orbit curves for its orbiting bodies. These templates use the
illustrative planar circular initializer, not current astronomical ephemerides.

## Legacy Matplotlib controller and headless export

Any explicit CLI selector (`--star`, `--planet`, `--load`, `--population`, and
related construction flags) skips the Qt menu and opens the compatibility
Matplotlib **App** in **PREP** at t=0, unless `--output` requests a headless PNG.

```bash
python -m simulations.star_system --star sun --planet earth --planet jupiter
python -m simulations.star_system --star sun --planet earth --planet jupiter --years 1 --output output/star-system.png
python -m simulations.star_system --load output/experiment.json --years 0.01 --output output/restored.png
```

With no arguments, the Qt Observatory opens. Its overhead scene fills the
window. **Menu** (keyboard **M**, unless a text field is focused) opens and
closes a sidebar drawn over that scene. Opening or closing the sidebar does
not resize the canvas or move the camera. The sidebar heading is the current
experiment name, such as **Solar System**. Passing selectors without `--output`
opens the legacy Matplotlib controller, which has the same full-window scene
and a **Menu [M]** overlay. The CLI retains its original four star and ten
planet/dwarf-planet selectors. General catalog bodies and custom states are
available through the authoring UI and Python API. A desktop backend is needed
for interactive input; `--output` uses Agg without a display.

## Desktop observatory

The sidebar shows elapsed simulated years, the requested simulated years per
second, the bodies currently in view, and the playback and inspection controls.

| Control | Action |
| --- | --- |
| Menu, `M` outside text fields | Show or hide the sidebar. The scene rectangle and camera stay as they are |
| Run | Start live evolution from the starting setup |
| Pause | Freeze simulated time and positions |
| Resume | Continue that same live trajectory without returning to the start |
| Step | Advance exactly one physics timestep and leave playback paused |
| Reset | Return to the edited starting setup at time zero |
| Simulated years per second | Change wall-clock pacing only. The integrator step stays fixed |
| Fit view | Frame the current bodies and stop following |
| Follow selected / Stop following | Keep the selected body centered at the current zoom, or leave the camera where it is |
| Mouse wheel over the scene | Zoom around the cursor |
| Left drag on empty space | Pan |
| Click a body, or a row in Bodies | Show that body's mass, radius, and other physical information |
| Add body | Open catalog choices. They are not listed on the main sidebar |
| Open planetary view | Show the selected planet and the moons assigned to it |

**Add body** is where catalog choices appear. Placement and starting-velocity
editing stay on that existing path. During a run, **Add body** pauses the clock.
Click the scene to drop the body, drag the arrow (its length is speed), then
**Confirm insertion**. Existing positions, velocities, and elapsed time stay as
they are, and the starting setup is not rewritten. **Reset** drops that insertion.
The circular arrow is a two-body aid around the chosen reference, including a
second star, and does not guarantee a stable orbit. **Custom body** takes a name,
kind, mass, and optional radius without editing a source file. Unknown catalog
values stay unknown until you supply one. The inspector shows units, status, and
the source for each value. A long body list scrolls inside the sidebar,
so Run, Pause, Resume, and Step remain on screen.

Select a planet and choose **Open planetary view**. The scene recenters on that
planet and shows it with the moons assigned to it; the body list follows the
same scope. **Return to system view** restores the full system. The switch
changes only the display; gravity still includes every body in the experiment.

Screen distances are compressed by `q = r * asinh(|r| / 0.05 AU) / |r|`. Marker
diameters stay larger than those compressed distances so bodies remain visible,
and they grow only logarithmically with physical radius. In the Sun/Jupiter
view, Jupiter's marker stays substantially smaller than the Sun's luminous
disk. Zooming in and selecting a body shows surface detail in the inspector.
Planet shading faces the illuminating star. Stars are drawn as luminous disks.
The background, labels, colors, and fading trails are illustrative: they are
not a measured sky, a surface map, or a photometric model. The scene caption
states the same limit: compressed distances, exaggerated body sizes, and
illustrative lighting.

Qt checks for this window use the offscreen platform. They click widgets and
deliver scene events in-process. They do not open a native window, move a
physical pointer, or judge smoothness. Passing them is not a manual playthrough.
Agg checks of the legacy controller are a separate headless path, not desktop
input.

## Legacy Matplotlib controller

Command-line selectors without `--output` open this controller. **Menu [M]**
hides its overlay without changing the camera or simulation. Its controls are
listed under Navigation and evolution.

## Calculate first, then inspect

In the Qt observatory, open **Calculate a recording** in the sidebar, then
**Calculate**. Playback of a finished recording uses **Play recording**,
**Pause**, and **Next sample**. The legacy Matplotlib controller uses the
control names in the rest of this section.

The primary flow is **PREP → CALCULATING → RECORDED**. Prepare bodies,
populations and moons with the same editors, then choose **Calculate...**.
Review the requested duration in years, positive integer output stride, and
recording budget before **Start calculation**. The default visible budget is
**256 MiB**. Duration must be finite and positive. The request rounds up to the
first whole configured physics step at or after that duration; the form shows
the resolved integer step count, fixed dt and resolved end time before start.
Output stride samples every N physics steps; it never changes physics dt.

Calculation uses `build_system` and the same transactional `advance_prepared`
boundary as live evolution. Each GUI tick accepts at most 150 steps and yields after a
6 ms work slice (a single whole solver step must finish or reject). Progress,
physical elapsed time and **Cancel calculation** remain visible. Initial and
final valid states are always included, even between output strides. Cancel
keeps usable partial output. An unresolved encounter halts with its reason and
last accepted whole step; neither cancellation nor halting is labeled complete.

Preflight conservatively estimates retained Python/NumPy frame storage, shared
metadata and an off-stride terminal checkpoint. An oversized request explains
how to reduce duration/body count, increase output stride or raise the budget.
Actual retained storage is checked as samples/events arrive; exceeding the
budget rejects the new step and retains the last accepted checkpoint instead
of silently dropping samples. This is a per-recording budget; other retained
results and the renderer also consume application memory.

**Play / Pause**, **Next sample**, and **yr/sec** select already calculated
samples forward in recorded time. Replay never calls the integrator. There is
no interpolation: the scene clock is the displayed frame's actual timestamp.
**Restart** in RECORDED returns to its first sample. Source name, setup digest,
execution environment, solver build, requested/resolved end times, actual
terminal time, completion state and reason stay with the result. Each frame
carries its own IDs, physical properties, parent associations and events;
body membership need not stay fixed. Missing selected bodies/parents report
removal. An explicit merge-event mapping can
transfer selection/follow to its remnant, with a visible disclosure; absorbed
tracers stay selected as removed and have no remnant mapping.
Ordinary samples retain the camera; fit and follow remain explicit actions.

Trails use timestamped world frames, retain at most 1500 samples and display at
most 400 vertices per body. The sidebar reports actual retained start/end and
duration, including sampling stride and history truncation. Duration is never
inferred from sample count times the current dt, including unequal time segments.

**Results (N)** cycles retained results, including older sources after editing
or calculating another setup. Editing a recorded setup's name, bodies or
population returns to PREP and needs a new calculation; prior immutable results
keep their source name/digest. The experiment window labels the selected starting
setup and current-state capture source separately. Capturing a displayed sample
as a new setup uses that recording's original source definition/digest and actual
sample time, even if another draft has since been edited. Setup JSON persists
starting conditions; recordings themselves remain in memory for this session.
Timeline scrubbing, reverse playback and comparison dashboards are not included.

**Run live from starting setup** explicitly starts a fresh **LIVE** trajectory.
Live Pause/Resume, one-dt **Step**, restart and insertion remain available.
Add/population actions from RECORDED enter preparation, so recorded arrays are
never insertion targets. On the headless CLI path, `--years` drives the same
incremental calculator; `--sample-stride` (default 10) and `--max-recording-mib`
(default 256) control its recording. `--years 0` still exports the initial setup
directly. These flags are not applied when opening an interactive GUI.

Python callers can use `Calculation(setup, CalculationRequest(duration_years,
sample_stride, max_bytes))`, call `advance(max_steps)` until `done`, or `cancel()`
for a partial `RecordedRun`. `result` is available at termination and includes
immutable `source_setup`, `frames`, status and actual timestamps. `Playback(run)`
exposes `current_frame`, `play()`, `pause()`, `next_sample()` and
`advance(wall_seconds, years_per_second)`. Event/membership changes preserve
exact-time PRE/POST collision frames even between output samples.

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
in collision mode `none`. The optional physical collision mode requires a known
radius for every massive body, including stars. Unknown radii use a fixed display marker without assigning a physical
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
the same scale. The legend reports relative and system components, speed,
separation in km/AU, and units. **Separation (km)** rescales the relative position
along its current direction (or +x from zero). Arrow direction
and length describe velocity relative to the chosen reference, not compressed
position units.

**Circular aid about reference** supplies the prograde perpendicular two-body
initial guess `sqrt(G * (M_reference + m_body) / separation)`. It requires a known
positive reference mass and nonzero separation. Reference velocity is added at
commit time, once. This aid does not promise stability, binding, or capture in
an N-body system.

**Confirm** validates the draft and leaves evolution paused until Run live/Resume.
**Cancel**, Escape, or closing the app discards the draft; cancel restores the
previous pause state and preserves the evolved positions, velocities, and time.
Invalid drafts remain open with a recoverable explanation and preserve the
active state. Keyboard playback shortcuts are disabled while authoring; the
Menu button can still hide/show the editor, making its widgets inactive.

## Starting setup and evolved state

`App.session` (`ExperimentSession`) owns the immutable `initial_setup`, a
validated `draft`, and the mutable `prepared` trajectory. `App.initial_setup`
and `App.prepared` expose those selected objects. The application opens in
**PREP**, paused at time zero. **Calculate...** starts a recorded calculation;
**Run live** or Step starts live evolution.
**Live insertion** changes only the
active trajectory, preserving every existing position, velocity, and elapsed
time exactly. It appends a new row even when particles already occupy earlier
rows. It retains the camera, existing scene artists, selected identity, and
follow target. An inserted body's history starts at insertion; it has no
invented past trail.

In live/preparation mode, **Restart** reconstructs the selected starting setup, removes unsaved live
insertions, resets elapsed time and trails, and retains the pause state.
**Prepare setup** explicitly returns to that starting setup and pauses it.
Add, **Edit selected**, and **Remove selected** then update its initial bodies.
The complete candidate setup is validated before replacement, including body
and population references. Removing a referenced body fails recoverably.
Run live or Step leaves preparation and starts ordinary live evolution.

**Prepare circular** opens the retained star/planet preset controls after
entering preparation. These keep the original unique-planet, golden-angle,
circular, barycentric construction and logarithmic distance sliders. They
rebuild the starting state when changed. The circular controls are disabled
once the starting setup has been generically edited, so they cannot silently
discard custom bodies. General preparation supports arbitrary physical
positions and velocities, one isolated body, multiple stars, and stars-only
binaries. No single-star assumption is imposed by the general renderer.

**Experiment / Save / Load...** opens a shared experiment window. Edit the name
and choose **Apply name** to update the heading and starting setup. The
Sun/Earth/Jupiter compatibility preset uses the original circular two-body
velocity guesses and barycentric shift; it is illustrative, not an ephemeris
or stability guarantee. Opening its factory returns an independent immutable
setup. Editing its name, bodies, population or moons creates a custom copy
with preset-origin metadata; the supplied preset remains unchanged. Use the
same body/population editors for free preparation.

**Save initial setup** writes the configured beginning to the explicit JSON
path, even after evolution or live insertion. Loading or restarting this
setup regenerates its original seeded population. **Save current state as new
setup** requires a different name and a separate capture path. It saves the
current coordinates, velocities and live additions as a new starting epoch
at time zero; it leaves the current experiment selected. Every current
particle becomes an explicit zero-mass initial body and `population` becomes
null, preventing duplicate generation. The original generator, seed and
settings survive only in capture provenance, together with source setup
digest, source name/time and step. Load the captured file to select it.

Captured reference associations, including moons, are retained by subtracting
each parent's **current** position and velocity once. Capture may need a
smaller fixed dt under the current moon period / 600 check; the new dt and
reason are saved and shown. This is a fresh numerical segment, not a promise
of uninterrupted trajectory continuation. General close-encounter validation
still applies; an invalid capture fails without changing the running state.

**Load into preparation** validates the complete file before replacing the
experiment, then pauses at time zero. An invalid file leaves the active
setup, vectors and draft recoverable. Escape/Close discards unapplied name
fields and restores the prior pause state. All experiment-window operations
use the existing Matplotlib backend, with no additional dependency.

### JSON compatibility and numerical identity

The public API is `save_setup(setup, path)`, `load_setup(path)` and
`setup_digest(setup)` in `persistence.py`. Version 1 stores explicit AU,
solar_mass, Julian_year, AU/Julian_year and K units; complete physical values
and property sources/overrides; stable IDs, relative reference vectors,
fixed dt and collision mode; preset/name provenance; source, model and
numerical version identifiers; and NumPy environment metadata. A generated
population stores PCG64, generator version, identity, reference ID and all
settings including seed. Catalog keys are descriptive; loading never looks
up replacement physical values in the current catalog.

The supported setup-file identities are `star-system-setup-v1`,
`planar-newtonian-point-mass-v1`, `fixed-yoshida4-v1`, and
`planar-circular-population-v1`. Unknown schemas, physical/model fields,
generators/settings, invalid references, invalid types/shapes/ranges,
nonfinite numbers and non-JSON content are rejected. Collision mode `none`
retains its original schema-1 encoding/digest; `perfect_merge` additionally stores
the explicit collision policy, rules, bounds and `positive-verlet-swept-parabola-v1`
numerical identity. Unsupported policy/numerical versions fail before install.
No pickle, eval, executable save data or silent migration
is used. Saves validate and encode before writing, then atomically replace
the destination through a uniquely owned sibling temporary file; failure
preserves an existing valid destination.

`setup_digest` is the SHA-256 identity of the physical starting conditions
and the source document that records them. Canonical JSON uses sorted keys
and disallows NaN/Infinity. The hashed document includes units, the
source/model/numerical version identifiers, the collision-mode encoding,
population settings, provenance, and the source `environment` block
(`environment.numpy`). It does not include the NumPy that last loaded the
file, and it does not include the solver that later integrates a trajectory.
Schema version remains 1. An existing version-1 file still loads with no
migration: its recorded NumPy version is validated and kept on the setup.
Saving a setup that carries that source environment writes the same
environment back, so an unedited load/resave under a different NumPy does
not change the document or the digest. A setup built in memory, with no
source environment yet, still records the current interpreter when it is
saved. Collision mode `none` therefore keeps its previous byte-for-byte
digest for a fresh save on a given NumPy. The same starting conditions keep
one digest for a given source document, including across loaders. A newly
produced document records the interpreter that produced it.

A produced run keeps that setup digest and records two further facts that
are not part of it. `execution_environment` is the NumPy that integrated the
trajectory, in the same `{"numpy": version}` shape as the setup document's
environment block. `solver_build` (`SOLVER_BUILD` in `persistence.py`)
identifies the integrator implementation. `numerical_version` remains the
numerical-method contract (`fixed-yoshida4-v1`, or the collision policy's
numerical id). Two solver builds can share a setup digest and a numerical
version; change `SOLVER_BUILD` when solver arithmetic changes even if that
method name stays the same. Recordings stay in memory and are not written
into setup JSON.

On the same supported environment, restored
initial arrays are exactly equal and deterministic evolution agrees to
`rtol=1e-12, atol=1e-12` after 50 identical steps in regression tests. There is
no universal cross-version/platform bitwise guarantee. Compare results with
the setup digest, the source environment, the run's execution environment,
and the solver build.

`--load PATH` accepts `--years` and `--output` (headless only) and uses the general renderer
for arbitrary bodies, multiple stars, particles and moons. It rejects any
explicit legacy construction flag (`--star`, `--planet`, `--population`,
`--count`, `--inner`, `--outer`, `--seed`, `--angles`), even with a default
value. Existing full CLI options and export commands remain available.
Initial setups and evolved captures are distinct from recorded results;
recordings remain in memory and are not serialized by setup saves.

## Navigation and evolution

| Control | Action |
| --- | --- |
| Menu [M], `m` outside editors | Show/hide sidebar, preserving state and camera |
| Run live / Pause / Resume, Space | Run live exits preparation; pause/resume an evolving system |
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

Gravity is unsoftened Newtonian gravity. The overhead presets are planar.
Formation can store a third component; that schedule is
[formation numerics](../../docs/formation-numerics.md). The force is
`a_i = G sum(j != i) m_j (r_j - r_i) / |r_j - r_i|^3`.
Positive masses mutually interact; zero-mass test particles feel massive bodies
without backreaction or particle/particle gravity. Units use 1 AU = 149597870700 m,
1 solar mass = 1.98847e30 kg, 1 Julian year = 365.25 days, and SI
G = 6.67430e-11. The collision-free integrator and dependencies are unchanged.

In collision mode `none`, the fixed-step fourth-order Yoshida composition of velocity Verlet uses
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

Live insertion can shorten the step to half the validated resolution limit,
or to the inserted moon's initial relative period / 600, whichever is smaller.
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
They still take their expected periods from the model constant and remain the
integration-accuracy checks.

Solar gravity is also compared with an external reference. The acceleration of
a massless body at 1 AU from one solar mass is checked against JPL DE440's
heliocentric gravitational constant GM_sun = 1.32712440041279419e20 m^3 s^-2
([astrodynamic parameters](https://ssd.jpl.nasa.gov/astro_par.html)), converted
with the IAU astronomical unit and a Julian year. Those reference constants
are hardcoded in the test. Rounded solar mass 1.98847e30 kg leaves the model
about 30.2 ppm high. The relative tolerance is 1e-4, which allows that
rounding and fails a 1% change in G or a unit conversion error.

Eccentric phase has a separate fixture, and the circular star/planet period
cases stay in place. An e=0.3 test particle integrated for one orbit at
period/1200 is compared with the position from Kepler's equation
M = E - e sin E, using the same implemented mu as the integrator. The measured
separation is about 2e-8 AU; the bound is 1e-6 AU.

These conservation/accuracy checks apply to collision-free, tide-free evolution. The optional
nonstellar contact model below has different numerics and conservation limits.
Relativity, stellar evolution, and scripted capture outcomes are not modeled.
Optional close-pass tides are a separate experiment choice and stay off in these checks.
The overhead presets stay planar. Formation's three-dimensional option is separate
and is not drawn by this view.

### Optional idealized nonstellar contacts

Open **Experiment / Save / Load...**, choose `none`, `perfect_merge`, or
`speed_angle`, and use **Apply collisions + RESET**. This explicitly rebuilds the
selected starting setup in paused preparation, including when invoked from
LIVE/RECORDED; existing recordings and their source setups remain immutable. An
invalid selection leaves the current state recoverable. The selected rule is stored
on the setup (`collision_mode` plus the versioned `collision_policy`) so a later
experiment file can keep it. Physical contact uses world AU and supplied physical
radii, independently of zoom, marker size, compression or rotating coordinates.
Initial surface overlap/contact is rejected for revision. Missing massive radii
block a physical mode; tracers are points and may have no radius.

A collision does not end a run. The macro step finishes, and every collision is
logged with its time, the participants, the selected rule, and the outcome.
A numerical failure or a user cancellation can still end a run early and is
labeled as such. Mode `none` still halts on an unresolved close encounter; that
guard is a fixed-step limit, not a collision outcome.

| Pair | `perfect_merge` | `speed_angle` |
| --- | --- | --- |
| Two positive-mass nonstellar bodies | Merge at resolved physical sphere contact | Merge, graze, or fragment from impact speed and impact parameter |
| Zero-mass tracer and positive-mass nonstellar body | Absorb/remove the point tracer; zero mass/momentum feedback | Same absorption |
| Two tracers | Ignore contact and mutual gravity | Ignore contact and mutual gravity |
| Any contact involving a star | Log `unmodeled`, keep both bodies, and continue | Log `unmodeled`, keep both bodies, and continue |

Ordinary binary gravity remains supported; an orbit is not a stellar-merger model.
Neither selectable rule merges a star. While an unmodeled stellar overlap remains,
that pair's gravity is evaluated at the contact separation so the fixed step can
cross the point-mass singularity. Masses and velocities are not replaced by a
remnant. The only modeled stellar merger is the separate
[Binary-star merger](#binary-star-merger) experiment.
Perfect merge's versioned policy is `nonstellar-perfect-merge-v1`, pair rule
`nonstellar-massive-merge-point-tracer-absorb-star-unmodeled-v1`, and remnant rule
`volume-com-planet-reference-map-v1`. Documents saved with the older pairs id
`nonstellar-massive-merge-point-tracer-absorb-star-halt-v1` still load and adopt
the continuation id. Simultaneous physical contacts form connected
massive groups, processed in sorted stable-ID order, with sorted-parent identities
and recursively retained ancestry. Unaffected rows retain their relative order.
Remnants get a deterministic hash ID and combined name, summed mass, center-of-mass
position/velocity, and radius `cbrt(sum(R**3))`. This is a volume-addition
approximation. Remnants are nonstellar `planet` bodies; temperature/composition are
unknown, catalog identity and measured sources are not inherited. Existing parent
references explicitly map to the new planet ID, so moons remain inspectable and
capturable without dangling or silently invalid saved references. This association
adds no force and guarantees no binding. Remnant parent reference is world/None.
Generated IDs reserve current and historical identities (including captured
epochs) and choose a deterministic suffix when a custom ID already occupies the
hash. Subsequent merges retain captured-epoch ancestry as well as active events.

Response conserves mass/linear momentum to floating-point precision. The recorded
lost relative translational kinetic energy is
`0.5*sum(m_i*|v_i-v_COM|**2)` (for a pair,
`0.5*m1*m2/(m1+m2)*|v1-v2|**2`). The orbital-angular-momentum residual about the
COM is recorded as **unmodeled internal spin**. Total energy and modeled orbital
angular momentum are not claimed conserved across merges. Heat, spin evolution,
ejecta are absent from perfect merge. Perfect merge is a disclosed
option with one remnant. It is unrealistic as a general law: Venus plus Earth
would not become one body. Optional close-pass tides are a separate choice
and are not applied inside this solver. Real planetary impacts have several outcome regimes; see
[Leinhardt & Stewart](https://arxiv.org/abs/1106.6084v3) and
[Stewart & Leinhardt](https://arxiv.org/abs/1109.4588v3). Perfect merge does not
implement those richer models or cosmological/galaxy collisions.

### Speed and angle

`speed_angle` is the other stored rule. Its policy version is
`speed-angle-merge-graze-fragment-v1`, pair rule
`nonstellar-speed-angle-merge-graze-fragment-star-unmodeled-v1`, and remnant rule
`fragments-tracked-losses-recorded-v1`. It uses the same positive-time contact
solver. At contact it measures the impact speed and the impact parameter
`b = |r × v| / (|r| |v|)` (0 is head-on, 1 is tangent) and the mutual escape speed
`sqrt(2 G (m1+m2) / (R1+R2))`.

| Condition | Outcome |
| --- | --- |
| Impact speed at or below mutual escape | Merge, same remnant as perfect merge |
| Faster, and `b >= 0.5` | Graze: both bodies keep their mass; the normal relative velocity reverses; the tangential velocity stays |
| Faster, and `b < 0.5` | Fragment |

A fragment replaces the parents with one largest remnant and two equal fragment
bodies. With `severity = min(4, (v/v_esc - 1) * (1-b))`, the lost-mass fraction is
`min(0.2, 0.05 * severity)` and each tracked fragment's fraction is
`min(0.25, 0.10 + 0.05 * severity)`. The largest remnant is the rest, and never
less than 0.2 of the parent mass. Lost mass is recorded on the event and is not a
body. It is accounted at the parents' center-of-mass velocity, so tracked mass plus
lost mass, and tracked momentum plus lost momentum, match the parents. Fragment
kinetic energy in the center-of-mass frame is 25% of the incoming relative kinetic
energy; the rest is the recorded loss. Parcel ancestry is the parents' parcels,
split across the tracked bodies and the recorded loss. This is a disclosed
speed-and-angle map. It is not the Stewart and Leinhardt strength model, and it
does not claim a unique real Venus–Earth outcome.

Further rules can be added as another named `collision_mode` and policy identity
on the same setup fields.

Close encounters and orbital changes are logged in addition to merges, grazes,
fragments, and ejections. A close encounter is a pair that comes inside three
contact radii without touching, either by entering that sphere or by reaching a
periapsis inside it. An orbital change is logged when a participant's star-centric
semi-major axis changes by at least 5% or its eccentricity changes by at least 0.02.
The log carries the time and the participating bodies. Histories of removed bodies
keep those events.

`perfect_merge` uses a separately named **positive-time velocity-Verlet** solver.
It never responds to Yoshida's negative middle stage. One live Step still advances
exactly the configured macro dt, applying the earliest contact then its remainder.
Every actual substep is at most macro dt/8 and half the original engine's
gravity/crossing limit at both start and accepted fully kicked endpoint. Endpoint
failures halve the candidate interval. Thus a larger initial macro dt can be
internally resolved; the close-encounter guard is not globally bypassed. A bound of
100,000 attempted subdivisions/responses per macro, 64 root bisections, finite
state checks and representable clock advancement give a clear numerical halt with
whole-macro rollback on failure, including metadata, events and membership.

The swept Verlet drift is `r(t)=r0+v0*t+a0*t²/2`. An O(N*massive-count)
vectorized chord broad phase expands by `|a_relative|*h²/8`; candidate quadratic
distance extrema partition the quartic into monotone intervals for entry-root
bisection or tangency detection. No tracer/tracer candidate matrix is constructed.
An insufficient configured refinement budget halts rather than accepting a root
whose bracket exceeds the declared time tolerance.
Detection precedes endpoint force evaluation, so a crossed singular endpoint is
not evaluated first. Time ties are within `1e-12*max(1,macro_dt)` years **and**
must satisfy the chosen-time sphere geometry within `R*1e-12 + 64*epsilon*max(R,d)`
AU; time proximity alone cannot merge still-separated pairs. The stationary-point
test uses squared-distance tolerance `R²*1e-12 + 32*epsilon*coefficient_scale`;
an inconsistent refined geometry halts. Settings and solver/pair/remnant versions
are saved explicitly. These bounds are safeguards, not arbitrary-trajectory accuracy
guarantees or a symplecticity claim for irreversible merging.

Observed fixtures use physical radii .1 AU and initial separation 2 AU. A
10 AU/yr head-on approach with masses 1e-14 and 2e-14 solar masses contacts at
0.18 yr within 1e-10 yr, including transit beyond both endpoint spheres. Grazing,
near-miss, earliest-event/remainder, point absorption, simultaneous groups and
row-reordering checks pass. For masses 1e-4/2e-4 solar and initial relative speed
1 AU/yr, measured contact times are:

| Macro dt (yr) | Contact time (yr) | Difference from dt=.0025 reference |
| ---: | ---: | ---: |
| .08 | 1.783919964365049 | 2.1593e-6 yr |
| .04 | 1.783918381350481 | 5.7625e-7 yr |
| .02 | 1.783917948979112 | 1.4387e-7 yr |
| .0025 reference | 1.783917805104718 | — |

This fixture/window shows convergence. A coarser .16 yr trial had nonmonotonic
error due to adaptive clipping; no universal monotone/error bound is asserted.
Same-environment save/reload/calculation reproduces checked contact times, IDs,
properties and arrays exactly. Cross-platform bitwise agreement is not promised.
PRE/POST frames carry physical timestamps and event metadata off output stride;
constituent trails end and remnant trails begin under their own IDs. Current-state
capture retains mode, policy, ancestry and remapped references as a new epoch.
Memory-budget rejection retains the last accepted whole macro and never reports
completion. Replay selects immutable frames and calls neither stepping path.

A supported 2,000-tracer Sun/Jupiter ring macro (2,002 bodies, dt=.009540592642 yr)
took **0.035611 s** in the measured Python 3.14 run. This exceeds the 6 ms GUI
target: cancellation and wall-budget yielding occur **between atomic macros**.
Pathological encounters can require many more subdivisions and take much longer
before the bounded halt. This measurement does not establish interactive pacing
for arbitrary body counts/encounters. The in-memory event frames also consume
budget beyond normal output samples.

### Binary-star merger

**Choose experiment** includes **Binary-star merger**. Two catalog-Sun
main-sequence stars start at rest, separated by ten times the sum of their
radii. The saved macro step is half the initial close-encounter limit. The run
is meant to be watched for 1.5 times the analytic contact time. Newtonian
point-mass gravity pulls them together until the spherical surfaces touch; the
positive-time contact solver then replaces that pair with one remnant. No other
pair uses this response.

The remnant mass is the sum of the parents, and its position and velocity are
the center of mass. Its radius is the estimate `R_sun * (M/M_sun)^0.8`,
normalized to the catalog Sun, not a stellar-structure integration. Temperature,
composition, mass loss, spin, tides, mass transfer, radiation, hydrodynamics,
and stellar evolution are omitted. The readout names both stars, the physical
contact time, and the remnant. The event also stores the lost relative kinetic
energy and the orbital angular momentum left as unmodeled spin. Total energy is
not claimed to be conserved across the merger.

The contact-time reference is the radial two-body fall from rest,
`t = sqrt(r0^3 / (8GM)) (η + sin η)` with `cos η = 2r/r0 - 1`. Point masses
would reach the center in `π sqrt(r0^3 / (8GM))`. On this fixture the analytic
contact time is 0.003496599597 yr. Halving the macro step moves the simulated
contact toward that value:

| Macro dt (yr) | Contact time (yr) | Difference from the analytic fall |
| ---: | ---: | ---: |
| 1.59612409795e-4 | 0.003496628450 | 2.89e-8 yr |
| half | 0.003496616391 | 1.68e-8 yr |
| quarter | 0.003496611297 | 1.17e-8 yr |

The test requires the quarter-step absolute error to stay under 1e-7 yr and
below the full-step error. These digits are from this implementation; they are
not a cross-platform bitwise guarantee. The checked case is this separation,
these two stellar types, and model id `solar-type-head-on-contact-v1`. A
different mass or radius, an extra body, or a missing model id halts as
unsupported contact and does not build a remnant. Exhausting the solver's
subdivision budget halts numerically, rolls back the unfinished step, and is
not labeled a merger. Other stellar types, compact objects, and random
encounters are outside this scenario.

### Optional close-pass tidal dissipation

`close-pass-energy-dissipation-v1` removes orbital energy when two positive-mass
bodies pass through a resolved pericenter inside a few radii of the larger body.
It is off unless the experiment stores an enabled tide policy, so existing
tide-free trajectories are unchanged. In the Qt observatory, check **Tidal
dissipation on close passes** and use **Apply collisions + reset**. That stores
the choice with the other experiment settings and rebuilds the starting setup.
The legacy controller keeps a saved choice; `App.set_tides(True)` selects the
default strength and reach and resets to the start. A captured epoch keeps the
policy that was active.

The loss at one pericenter is `strength * G * (m_j^2 * R_i^5 + m_i^2 * R_j^5) / q^6`,
with `q` floored at the larger radius. The default strength is 80 and the default
reach is 6 larger radii. Strength is a dimensionless stand-in for a
passage-integrated lag. It is not a measured k2/Q. Massless tracers do not
raise or feel this kick. The relative velocity is scaled about the center of
mass, so linear momentum is kept, and the removed energy cannot exceed the
relative kinetic energy. The change in the pair's orbital angular momentum is
recorded and is not stored as spin. Temperature, spin evolution, and stellar
structure are omitted.

The kick is applied to the velocity at the end of the step that contains the
inbound-to-outbound crossing. The pericenter is estimated by interpolating that
sign change across the step. On the Yoshida path (`collision_mode='none'`), an
unresolved step still halts and keeps the last accepted state, and this model
does not run. On the contact-solver path the same kick applies only to a pair
that does not become a contact in that step, including in three dimensions and
on an adaptive macro step. A contacting pair is left to the collision rule.
That rule is unchanged.

This kick does not call REBOUND (#70).
A step that skips the pericenter, or that estimates it poorly, understates the energy loss,
so tides are not the dominant capture path for an unresolved encounter.
It does not change collision rules (#69): perfect merge, speed and angle, ignore, stop on contact, and the binary-star merger keep their own responses.
Fragmentation, hit-and-run, and merger remnants are not tidal outcomes.
Where an impact would dominate, this model is not the capture path.
It is not gas drag (#71). There is no disk in this kick, and a pass outside the tidal reach loses no energy here.
Gas drag, fragmenting collisions, and a close-encounter integrator can outweigh this kick;
none of those are implied by a tidal result.
A longer encounter in this model is still spin-free and can be reversed by a later scattering.
It is not, by itself, a permanent capture.
Lasting tides are a separate module. Both may be on in one run, and when they are, the energy losses add.

The documented comparison is catalog Jupiter and catalog Earth, with the Sun
omitted so the energy change is only this kick. Both runs use the same vectors.
The flyby starts hyperbolic (eccentricity 1.01) at 1.25 times the sum of the
catalog radii. The eccentric case starts at eccentricity 0.65 and the same
pericenter distance. `compare_jupiter_earth_passes` integrates the tide run
until a second pericenter and gives the tide-free flyby that same step count.
The eccentric pair stops at the first pericenter. A bound flyby or a lower
eccentricity is a claim only for the run with the model on. Minimum sampled
separations stay outside contact, so this is not a collision.

```python
from simulations.star_system.tides import compare_jupiter_earth_passes
print(compare_jupiter_earth_passes().disclosure)
```

Measured on this implementation. The flyby rows share 6867 steps (0.00801789 yr). The eccentric rows share 286 steps. Specific energy is the Earth–Jupiter two-body value.

| Run | Pericenters | Specific energy | Eccentricity | Energy lost |
| --- | ---: | ---: | ---: | ---: |
| Flyby, tides off | 1 | +0.296525 | 1.01 | 0 |
| Flyby, tides on | 2 | -10.0444 | 0.661 | 3.095e-5 |
| Eccentric, tides off | 1 | -10.3784 | 0.650 | 0 |
| Eccentric, tides on | 1 | -15.5495 | 0.476 | 1.548e-5 |

The closest sampled separation is about 0.000637 AU, outside the 0.000510 AU sum of the catalog radii. These digits are from this implementation; they are not a cross-platform bitwise guarantee. The checked claims are the energy signs, the pericenter counts, and the drop in eccentricity.

### Optional body detail

`body-detail-v1` is off at point mass unless an experiment selects a richer level.
`physics.body_detail` is `null` or `point_mass` for the Newtonian point mass,
`fixed_oblateness` for a constant J2 bulge, or `spin` for a tracked spin and a
wobbling figure axis. A blank experiment file leaves the key null. Point mass
adds nothing to the force or the body state. The selected level is named on the
playback.

Fixed oblateness adds the zonal J2 acceleration of each catalog body that has a
J2. The bulge is constant. Its figure axis is fixed in inertial space: the
catalog obliquity, laid in the x–z plane, or +z when the body has no obliquity.
A planar run pins every axis to +z. There is no J3, J4, or tesseral field.
Bodies with no listed J2 stay spherical. The monopole is unchanged. The
companion feels the bulge, and the oblate body receives the opposite force
scaled by the mass ratio, so linear momentum is kept. Spin is not a state at
this level, so the torque that moves orbital angular momentum is not stored.

With the pole along +z and `s = p · r`, the acceleration of a companion is

```
a = (3/2) J2 G M R^2 / r^5 * ((5 s^2 / r^2 - 1) r - 2 s p)
```

Full spin and wobble needs a 3D state. Each body carries an angular-momentum
vector and a unit figure axis. The polar moment is `C = k M R^2` and the
equatorial moment is `A = C - J2 M R^2`. The same J2 acceleration is evaluated
on the current axis. The quadrupole torque on the body is minus the orbital
torque of that force,

```
N = -3 G M_other (C - A) (p · r) (p × r) / r^5
```

so the change in orbital angular momentum is stored as spin. The figure axis
rotates about the angular momentum at `|L| / A`. Catalog sidereal periods and
obliquities are the initial principal-axis spins. A body with no listed spin
starts at rest. A collision remnant, an inserted body, and a particle
population are not given a spin or a J2; those combinations stop with a named
reason. Close-pass tides stay spin-free and do not write into this state.
Playback samples store positions and velocities. A formation checkpoint stores
the spin and the figure axis, so a restored run continues the same rotation.

The checked case is a 400 km circular orbit at 51.6° inclination about the
catalog Earth, with the figure axis held on +z so the inclination is measured
from the equator. The secular node rate is `-(3/2) n J2 (R/a)^2 cos i`.
Earth's J2 in this comparison is 1.08263e-3. The analytic rate is -5.00868
deg/day. Eight orbits at 800 steps per orbit, on the Yoshida path, measured
-5.03634 deg/day for both fixed oblateness and spin. The same orbit with the
body-detail level left at point mass held its node to about 1e-13 deg/day.
Six orbits on the contact path (`collision_mode='ignore'`, 200 steps per
orbit) measured -5.04090 deg/day. These digits are from this implementation;
they are not a cross-platform bitwise guarantee.

### Optional lasting tides

`constant-time-lag-v1` is the secular tide. It is off unless
`physics.lasting_tides` is true, and that choice requires `physics.body_detail`
`spin`. Point mass and fixed oblateness have no spin rate for a lag to act on,
and a build refuses them by name. Turning this module on does not turn the
close-pass kick on, and turning the close-pass kick on does not turn this
module on.

The force is the quadrupole constant-time-lag tide, kept to first order in the
lag. For the tide raised on body i by body j, with `r` and `v` the position
and velocity of j relative to i and `Ω` the spin of i, the force on j is

```
F = -(3 k2 Δt G m_j^2 R_i^5 / r^10) (3 (r · v) r + r^2 (v - Ω × r))
```

Body i receives `-F`. The torque on i's spin is `-r × F`, so the orbital
angular momentum this force exchanges is stored as spin. Both directions are
evaluated for every pair of positive-mass bodies that have a radius, a Love
number, and a positive lag. On a circular orbit whose spin is aligned with the
orbit normal this reduces to

```
da/dt = 6 k2 Δt m R^5 n (Ω - n) / (M a^4)
torque = -3 k2 Δt G m^2 R^5 (Ω - n) / a^6
```

`Ω > n` expands the orbit and slows the spin. `Ω < n` does the opposite.
`Ω = n` is the tidal-lock fixed point: the torque and `da/dt` vanish.

Earth's default lag is the effective oceanic value that makes this circular,
spin-aligned, Earth-tide-only rate 3.82 cm/yr for the catalog Earth and Moon
at 384400 km, with `k2 = 0.299`. It is not a seismic quality factor. The
product `k2 Δt = 213 s` recedes at 4.466 cm/yr with these rounded masses, so
the stored lag is 609.274 s, the value that hits 3.82 cm/yr here. Other
catalog bodies, including the Moon, have a zero default lag and raise no
lasting tide until a lag is set.

Close-pass tides and lasting tides may both be on in one run. The close-pass
kick remains the impulsive spin-free sink at a resolved pericenter inside its
reach. Lasting tides are the continuous force and torque at every separation,
including inside that reach. The losses add. The sum is not one calibrated
tide. A run that wants only the secular migration leaves the close-pass kick off.

The checked case integrates the catalog Earth and Moon masses and radii for
four orbits in three dimensions. Earth's spin is the sidereal day, aligned
with the orbit. The Moon's spin is the mean motion and its lag is zero. J2 is
left off so the semi-major-axis change is the tide. The analytic rate is 3.82
cm/yr. The integrated rate was 3.81900 cm/yr. The instantaneous force on that
initial state gives the same 3.82 cm/yr, the aligned torque is zero when the
spin equals the mean motion, and a faster spin loses angular momentum. These
digits are from this implementation; they are not a cross-platform bitwise
guarantee.

### Final dependency-stack acceptance audit

Live acceptance lists for #4–#9 and deferred #14 were reread on 2026-10-31.
This audit describes the implemented stack and its verification boundary; it
does not merge PRs or close issues. #6 is already closed externally and remains so.

| Issue / plans | Implemented integration and acceptance evidence |
| --- | --- |
| #4 / 1 | Full-window scene and overlay sidebar in the default Qt observatory and the legacy Matplotlib controller; camera/input controls, physical inspection, retained rendering, and the current experiment name as the heading. Offscreen Qt and Agg checks cover those controls. Native desktop interaction remains unverified. |
| #5 / 2 | Sourced/custom bodies, relative placement/velocity vectors, repeated identities, multi-star gravity and preserved live insertion; authoring/catalog/physics tests. Physical radii feed contact, never marker sizes. |
| #6 / 3,5,6,8 | Seeded reference-relative populations save/reload/calculate/replay with stable IDs. Point-tracer absorption has no feedback; particle/particle interactions remain omitted. |
| #9 / 4,5,6,8 | Planet-relative initial states and timescale limits, live insertion, separation inspection, save/replay; contact uses the same nonstellar response and explicit remnant-parent mapping. No forced binding or tides. |
| #8 / 5,6,8 | Named/custom preparation, separate initial/current saves, strict version recovery, bounded calculation and immutable replay, retained live mode, versioned physical contact and PRE/POST events. Numerical/unsupported-contact/memory halts preserve partial accepted output and report incomplete status. |
| #7 / 7 | Sun/Jupiter ring preset and matched control, rotating historical view and labeled geometric guides; original collision-none behavior remains covered. Optional collision mode changes numerical identity; no Trojan/capture/stability guarantee. |
| #14 | The binary-star scenario is the only modeled stellar merger. Other stellar contact is logged as unmodeled and the run continues; it does not build a nonstellar remnant. |

The selected minimal recording controls remain duration/stride/budget, forward
Play/Pause/Next, and in-memory results. Timeline scrubbing, reverse navigation,
persisted recordings, preference persistence and richer comparison dashboards
remain outside this delivered scope. Real-impact outcomes and arbitrary-system
accuracy/performance are limited to the explicit approximation and fixtures above.

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

## Planet-relative moons

In the desktop, select a planet, open **Add body**, choose **Lunar**, and enter
the moon's distance from that planet in kilometers. The scene click sets the
direction only. That distance is center-to-center and is not the planet's
position from its star. Inspect the moon to see the parent and the current
separation. The parent does not pin the moon.

Select a planet, choose **Add a body**, then the **moon** category and a catalog
definition (Moon, Io, Europa, Ganymede, Callisto, or Titan). For a custom moon,
choose **custom**, set Kind to `moon`, and supply the required physical values.
The selected planet is the explicit default parent. If no planet is selected,
choose one with **Parent: next planet** or enter its stable ID; stars, particles,
missing IDs, and `world` are rejected as moon parents. The catalog identifies
physical properties, not a host or an ephemeris.

Enter the planet-relative separation in km or Cartesian x/y in AU, adjust the
relative velocity components/handle, or use the two-body circular aid. The
legend names the parent and shows both relative and system vectors. Physical
Radius (AU) is the body's surface radius; Separation (km) is center-to-center
orbital spacing. Neither is the planet's system position. During preparation,
the draft fixed dt decreases as necessary to the initial relative period / 600;
the field and legend disclose the chosen value. A directly supplied
`InitialSetup.dt` remains explicit and is rejected if coarser than that moon
bound or the general encounter guard. This is an initial circular accuracy aid,
not a guarantee for eccentric or rapidly perturbed orbits.

Confirmation resolves the parent's **initial** position/velocity in preparation
or its **current** position/velocity during live insertion, exactly once. Live
insertion never recenters the planet; every existing body/particle row and time
is retained exactly. A smaller live dt is reported in an insertion event and
starts a new fixed-step segment. Parent associations remain metadata after
initialization: the same ordinary mutual gravity includes all positive masses.
An excessive speed can carry the moon away; no reattachment, forced orbit, or
permanent binding is imposed.

The selected moon's inspector measures **current** separation from the displayed
world positions in km and AU. Use the body list to select close identities when
exaggerated markers overlap; follow, zoom, pause, and step use the common controls.
Lighting still uses stellar illumination, independently of the orbital parent.
Missing parents are reported by ID without selecting a substitute. The reusable
`relative_state(StateFrame, body_id)` path also supports later recorded frames.
Save/reload retains parent references and edited initial velocity; calculated
playback remains separate work.

An illustrative isolated circular Earth-Moon setup uses 384400 km from the
[NASA Moon Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html)
(mean semimajor axis, updated 2024-01-11; checked 2026-10-30), with physical
Earth/Moon masses and radii from `CATALOG`. This deliberately circular 2D fixture
is not the real Moon's eccentric/inclined ephemeris or its observed period:

```python
from simulations.star_system.data import CATALOG, AU_KM
from simulations.star_system.initialization import (
    build_system, circular_relative_velocity, recommended_moon_step,
)
from simulations.star_system.model import InitialBody, InitialSetup

earth, moon = CATALOG['earth'], CATALOG['moon']
relative_r = (384400 / AU_KM, 0.)
relative_v = circular_relative_velocity(relative_r, earth.mass, moon.mass)
setup = InitialSetup('Illustrative circular Earth-Moon', (
    InitialBody('earth', earth, (0., 0.), (0., 0.)),
    InitialBody('moon', moon, relative_r, tuple(relative_v), 'earth'),
), recommended_moon_step(relative_r, earth.mass, moon.mass))
prepared = build_system(setup)
```

The model's rounded catalog and G give a two-body period of **27.2895002371
days** (0.0747145797047 yr), dt **0.000124524299508 yr** (about 65.50 minutes).
Over three periods, angular interpolation at each of the first three full turns
measures the three successive orbital periods; normal Yoshida evolution gives:

| Fixed step | Maximum relative period error | Maximum relative energy drift | Maximum relative angular-momentum drift | Relative position error after three periods |
| --- | ---: | ---: | ---: | ---: |
| Period / 600 | 1.11364e-8 | 1.79856e-14 | 8.77076e-15 | 5.39384e-10 AU |
| Period / 1200 | 6.96010e-10 | 1.85407e-14 | 9.32587e-15 | 3.37102e-11 AU |

The supported fixture bounds are period error <1e-6, energy drift <1e-7,
angular-momentum drift <1e-10, and decreasing position error at dt/2 (measured
about 16 times smaller). Energy and angular drift are already near floating-point
roundoff; those values need not decrease monotonically with dt. Sun/Earth/Moon
and second-star fixtures verify finite mutually forced evolution at this step,
not long-term stability or guaranteed binding. Resolving a moon can require
many more steps per simulated year than the surrounding planets; the pacing
control does not bypass that cost. These moon checks leave tidal dissipation off.
Formation, stellar evolution, and 3D are not part of this fixture. Marker overlap has no contact effect. Optional `perfect_merge`
uses the shared nonstellar response for moons; collision-free moon accuracy
checks above do not validate a physical impact outcome. Saved/replayed moon
references and remnant-parent mappings use the same shared workflow.

## Reproducible populations

**Population...** supports one ring/disk/belt in any selected initial setup,
including custom bodies and multiple stars. Its reference-ID field and
**Reference: next massive** button use the body editor's selection interaction.
The selected massive body is the default for a new population; a selected
particle requires an explicit reference. Reopening an applied population uses
its applied reference and settings, independent of the current scene selection.
The window labels the reference's name, stable instance ID, and initial frame.
Choices come from the selected starting setup: an unsaved live insertion cannot
become a population reference until it belongs to a starting setup.

A reference needs positive known mass and a known physical radius in AU.
Missing/deleted references, particles, unknown radii, and radii inside the
reference surface fail recoverably. No replacement star is chosen silently.
Reference IDs resolve to numeric rows only at initialization; reordering the
initial bodies preserves generated states by ID. Particle IDs use
`<population-id>:<ordinal>`. Reference position and velocity are added exactly
once, so a ring around a second star inherits that star's initial world state.
The reference does not bind particles to it during evolution.

During preparation, **Apply** and **Remove** validate and update the starting
setup while time stays zero and paused. During live evolution, the labels are
**Apply and restart** and **Remove and restart**: they rebuild the selected
initial setup, preserving its custom/multiple-star definitions and removing
unsaved live insertions. Apply regenerates particles from the authored settings
and seed, rather than appending them to evolved arrays. Cancel, Escape in either
window, and closing the population window preserve evolved positions,
velocities, time, and the previous pause state exactly. Invalid drafts stay open
without changing the valid setup; conflicting controls are disabled.

The draft discloses the fixed step it will record on Apply:
`min(starting dt, inner reference orbital period / 600)`. This retains the
generator's initial resolution choice for short reference-relative orbits.
Normal evolution remains fixed-step, and the existing encounter/crossing guards
can reject a candidate or halt later evolution. Remove retains the selected
setup's fixed step. Direct `build_system(setup)` callers must supply a suitable
explicit `setup.dt`; initialization does not silently shorten it.

```powershell
python -m simulations.star_system --population belt --count 200 --inner 2 --outer 3.5 --seed 42
python -m simulations.star_system --planet jupiter --population ring --inner 3.2 --count 64 --angles even --seed 7 --years 1 --output output/ring.png
```

The CLI flags remain the original star-relative convenience path; general
references are authored through the shared setup/UI and `--load`. Ring uses one radius; disk samples annular area uniformly;
belt samples radius uniformly. Random or evenly spaced angles use NumPy PCG64 and an explicit
nonnegative seed. Counts are 1-2000, with at most 2000 total zero-mass particles
including explicitly authored ones; radii are finite/outside the reference
surface, and velocities are circular prograde guesses relative to the reference state.
Restart reproduces the population. Individual particles can be inspected and
followed; the selected particle has a bounded trail. There is no
particle self-gravity/accretion or stability promise, especially for
multiple-star systems. Test particles represent rocks whose gravitational
influence is neglected: they feel every massive body but do not backreact or
interact with one another. In `perfect_merge` and `speed_angle`, point tracers are absorbed
by nonstellar massive bodies without feedback, ignored by other tracers, and
logged without a remnant if they meet a star. The Jupiter preset defaults to collision mode `none`;
changing modes is an explicit reset and a different numerical model.

Persistence saves/reloads full settings, seed, population ID and reference ID,
with exact regenerated initial particle states checked in the same environment.
Calculated playback retains population identities, timestamps, complete or
halted status, pause, and sample stepping.

## Sun–Jupiter perturbation experiments

In the **legacy Matplotlib controller**, open **Experiment / Save / Load...**,
select **sun-jupiter-ring**, then **Open preset**. The Qt desktop lists the same
fixture as **Sun–Jupiter ring**, with Rotating view and Matched no-Jupiter run. The common setup contains the Sun,
Jupiter, and 32 mass-zero
test particles in a 3.2 AU ring with evenly spaced angles and seed 7. Its
positions and circular initial velocity guesses use the existing population
generator. Body/vector editors, population settings, initial saves, loading,
live stepping, and calculation work through the same interfaces as custom
systems. These initial conditions demonstrate perturbation; they do not promise
Trojan occupancy or capture. Arbitrary capture attempts may escape or halt.

For a matched comparison, save the initial setup and calculate it first. Then
open the same experiment dialog and choose **Prepare matched no-Jupiter
control**. Save this control to its own JSON path and calculate the same
duration and output stride. The control stores its actual explicit initial
bodies and physical vectors, with Jupiter removed and no regenerated ring or
barycentric recentering. Its fixed dt is identical. Results retains both runs;
click it to switch between them. The comparison matches particle IDs and shows
the maximum **Sun-relative** displacement difference at a common recorded
physical time, both run statuses/actual end times, and their common sampled
interval. An edited initial vector/property invalidates the matched comparison.
If either run halts, cancels, or ends at a different time, only common samples
are compared and the unequal/incomplete completion is explicit. Unrelated
moons retain their declared parent; remove or reauthor a Jupiter-relative moon
before removing its required parent.

The unchanged fixed fixture produces a maximum displacement difference of
**0.0730167 AU after 600 steps / 5.724355585 years**, one initial ring period,
with `dt = 0.009540592642 yr`. The separate existing mass-zero Jupiter-source
control remains an independent numerical check above 0.01 AU. Forces come
from the declared gravitational model; no attraction toward L4/L5, relocation,
clustering, capture rule, or altered solver is added.

**World axes**, ordinary origin-relative inertial axes (Sun-relative for this
preset), and **Rotating axes** change display coordinates only. In the rotating
view the current Sun is the origin and the current Sun–Jupiter direction is
the positive horizontal axis. Every historic point uses its own sample's Sun,
Jupiter, and angle, for both live trails and recorded playback. Radial distance
compression is applied afterward. Missing/ambiguous or coincident references
disable rotating axes with a reason, including the no-Jupiter control; no
synthetic reference body is supplied. Inverse click placement undoes compression
and then the selected spatial frame. Velocity fields always author physical
reference-relative **inertial AU/yr** vectors; arrows rotate their orientation
for display without subtracting an angular frame velocity.

The hollow violet L4/L5 diamonds are at ±60° and the instantaneous Sun–Jupiter
separation, following [NASA's equilateral reference geometry](https://science.nasa.gov/solar-system/resources/faq/what-are-lagrange-points/).
They are labeled **ideal circular restricted-three-body reference geometry**,
visually separate from particle markers. They do not detect occupied or stable
regions. Eccentric/perturbed systems do not inherit a guaranteed stationary
equilibrium, and proximity or a short trajectory does not establish permanent
capture.

Select a rock in the scene/body list to draw its trail alongside major-body
trails; other rocks have no individual trails. Selection uses stable IDs,
including after row reordering. The display keeps at most **400 vertices**
from retained world snapshots, labels the actual timestamp interval/duration,
sample/vertex counts and earlier-history truncation, and retains at most 1,500
samples in the live/replay viewing window. Recording stride samples output
without changing integration dt. The inspector reports actual rock–Jupiter
separation and Jupiter-relative physical inertial velocity from the displayed
sample, plus world vectors. Follow, live pause/step/restart, and recorded
play/pause/next-sample use the same shared controls.

For a reproducible saved export:

```python
from simulations.star_system.scenarios import get_scenario, make_jupiter_comparison
from simulations.star_system.persistence import save_setup

setup, control = make_jupiter_comparison(get_scenario('sun-jupiter-ring'))
save_setup(setup, 'output/sun-jupiter-ring.json')
save_setup(control, 'output/sun-jupiter-control.json')
```

```powershell
C:\Python314\python.exe -m simulations.star_system --load output/sun-jupiter-ring.json --years 5.724355585 --sample-stride 4 --output output/jupiter-saved-cli.png
C:\Python314\python.exe -m simulations.star_system --load output/sun-jupiter-control.json --years 5.724355585 --sample-stride 4 --output output/jupiter-control-cli.png
```

## Shared APIs for the following plans

- `model.py`: frozen `BodySpec`, `PropertySource`, `InitialBody`, `PopulationSpec`, and `InitialSetup`. Catalog keys identify definitions; body IDs identify instances. `PopulationSpec(id, reference_id, settings)` maps the stable reference to the legacy numeric index at initialization.
- `initialization.py`: `build_system(setup) -> PreparedSystem(system, body_ids, body_specs, setup, events, segment, reference_ids=())`, `insert_body(prepared, body)`, `circular_relative_velocity(r, M, m)`, `recommended_moon_step(r, M, m)`, and `legacy_setup(...)`. All vectors are world frame after one-time reference resolution. Insertion preserves old row order, including particle rows. `reference_ids` is a body-aligned tuple of initial reference IDs or `None`; its empty default derives from setup IDs.
- `history.py`: `StateFrame.capture(prepared)` copies time/dt/IDs/specs/masses/world r/v/events/reference_ids into read-only state. `StateFrame.reference_ids=()` defaults to all `None`. `relative_state(frame, body_id)` returns current relative vectors/separation or an explicit missing-reference result. Capture-current persistence must retain reference metadata while converting world vectors back relative to the captured parent once; storing world vectors plus an InitialBody reference would add the parent twice. `App.history` retains 1500 frames; each displayed trail has at most 400 vertices. `App.trails` remains a relative-array compatibility adapter.
- `rendering.py`: `draw_system(ax, prepared, history=()) -> Scene`; retained `draw(...)` accepts the old numeric/circular arguments. `Scene.artists_by_id`, `labels_by_id`, `trails_by_id`, `origin_id`, and `origin` support identity-based integration. `Scene.update(prepared, history)` appends new artists without resetting the camera.
- `authoring.py`: `BodyEditor` and its `reference_state`, `vectors`, `draft_body`, `commit`, `finish` draft lifecycle. `App.edit_body`, `prepare_setup`, `selected_id`, `_install_prepared`, and `_sync_painter` provide the controller integration points. Register new widget axes with sidebar visibility and the painter's overlay axes.
- `population_editor.py`: `PopulationEditor` validates a complete candidate `InitialSetup` before Apply/Remove, using the shared `authoring.reference_ids` choices. Its separate figure owns its widgets and close/Escape lifecycle; the main scene's overlay axes retain the existing painter contract.
- `persistence.py`: `save_setup`, `load_setup`, `setup_digest`; strict schema/model/numerical/generator versions above. `setup_digest` identifies the canonical starting conditions and the source-document NumPy environment, not the loading interpreter or `SOLVER_BUILD`. Produced runs record `execution_environment` and `solver_build` separately. Schema version 1 files load without migration.
- `experiments.py`: `ExperimentSession(initial_setup)` owns `initial_setup`, `draft`, `prepared`; `begin_draft`, `update_draft`, `apply_draft`, `cancel_draft`, `select`, `restart`, `save_initial`, `save_current`, `load` provide explicit transitions. `capture_current_setup(prepared, initial_setup, name)` returns an independent validated initial epoch. `App.initial_setup`/`prepared` delegate to session ownership, and `_install_prepared` remains the scene-installation path.
- `scenarios.py`: `get_scenario(key)` returns an independent immutable setup; `DESCRIPTIONS` holds preset descriptions. `sun-earth-jupiter` is the compatibility preset; `sun-jupiter-ring` is the checked perturbation fixture. `make_jupiter_comparison(setup)` resolves a matched explicit control, and `compare_recordings(first, second)` validates source vectors and reports Sun-relative differences only at common recorded times.
- `frames.py`: pure `frame_transform`, `rotate_about_sun`, and `lagrange_guides` transform snapshot copies without a dynamics import. `Scene.world_to_display` and `display_to_world` compose the selected frame with compression/inversion; historical points use their own frame transform.

The body-authoring tests provide arbitrary/binary, reference-chain, populated
insertion, live draft events, invalid/cancel, inverse mapping, velocity scaling,
and multi-star lighting fixtures. The generic renderer never assumes a fixed
star/planet prefix. Current-state capture stores explicit particle bodies with
`population=None`, avoiding regeneration of evolved particle coordinates.

## Historical integration logs

The sections below record past integration runs and command transcripts. The
[acceptance audit table](#final-dependency-stack-acceptance-audit) under
**Numerical model and limits** is historical evidence, not day-to-day operating
instructions. Use the launch table and desktop or compatibility sections above to
get started.

### Checked outcomes (integration transcripts)

Windows/Python 3.14 verification uses writable ignored `output/` basetemps because
the default Windows pytest temp directory is inaccessible in this environment.
Agg event tests exercise the real Matplotlib callback flow and normal exports;
native desktop input, perceived smoothness, and native multiwindow behavior
remain unverified.

Population integration commands:

```powershell
C:\Python314\python.exe -m pytest -q --tb=short --basetemp output/pytest-population-final-08
C:\Python314\python.exe -m simulations.star_system --population ring --inner 3.2 --count 64 --angles even --seed 7 --years 1 --output output/issue-6-ring.png
git diff --check
git diff --check codex/issue-5-authoring...HEAD
```

The full integration run passed **166 tests in 273.43 seconds**. Its 26 warnings
are the expected Agg notices that the separate population window cannot be
shown. A focused ring-control check also passed after the final radio styling
correction, and a refreshed preview verified the selected highlight. Whitespace
validation passed.

The required CLI command exited successfully and produced a visually inspected
1350x1350 PNG (106,275 bytes). Normal App/editor exports at
`output/population-scene-reviewed.png` (1480x919) and
`output/population-editor-reviewed.png` (740x850) were visually inspected for
the custom binary setup, reference name/ID/frame, units, fixed-step disclosure,
and restart controls. These artifacts remain in ignored `output/`.

Moon integration commands:

```powershell
C:\Python314\python.exe -m pytest -q tests/test_moons.py tests/test_body_authoring.py --basetemp output/pytest-moons-focused-05
C:\Python314\python.exe -m pytest -q --tb=short --basetemp output/pytest-moons-final-06
C:\Python314\python.exe -m pytest -q -s tests/test_moons.py -k orbit_converges --basetemp output/pytest-moons-periods-07
C:\Python314\python.exe -m simulations.star_system --years 1 --output output/issue-9-cli.png
git diff --check
git diff --check codex/issue-6-population-integration...HEAD
```

The focused moon/authoring run passed **42 tests in 92.15 seconds**. The full
integration run passed **182 tests in 302.42 seconds**, with the same 26 expected
Agg population-window notices. A final refinement to measure each of the three
successive periods separately passed its focused check (**1 passed**, 12
deselected, 5.94 seconds); the table above records that measurement. Whitespace
checks passed. The normal CLI export succeeded and its 1350x1350 PNG was
visually inspected. Normal App preparation and selected-moon inspector exports
at `output/moon-preparation-preview.png` and `output/moon-inspector-preview.png`
(1776x1104) were visually inspected for parent name/ID, relative/system units,
separation, fixed-step disclosure, body-list selection, and stellar illumination.
These artifacts remain in ignored `output/`; no native desktop interaction is
claimed by these Agg checks.

Persistence integration commands:

```powershell
C:\Python314\python.exe -m pytest -q --tb=short --basetemp output/pytest-persistence-final-08
C:\Python314\python.exe -m pytest -q tests/test_experiment_persistence.py tests/test_experiment_controls.py tests/test_populations.py::test_population_playback_restart_and_render tests/test_star_system_display.py::test_running_frames_reuse_background_without_full_redraw --tb=short --basetemp output/pytest-persistence-final-targeted-11
C:\Python314\python.exe -m simulations.star_system --load output/moon-population-setup.json --years 0.01 --output output/loaded-moon-population.png
C:\Python314\python.exe -m simulations.star_system --star sun --planet earth --planet jupiter --years 1 --output output/legacy-system.png
C:\Python314\python.exe -m simulations.star_system --population belt --count 24 --seed 42 --years 0.01 --output output/legacy-population.png
git diff --check
git diff --check codex/issue-9-moons...HEAD
```

The full integration run reported **239 passed, 2 failed in 329.81 seconds**,
with 29 expected Agg window notices. Both failures were legacy tests assuming
startup was already live: the population test toggled the new paused state
before checking unchanged coordinates, and the renderer test counted the new
PREP-to-LIVE label redraw as steady motion. Their setup now explicitly follows
the preparation/live contract. A small provenance refinement also retains
the origin preset through repeated circular rebuilds. The subsequent targeted
run passed **62 tests in 15.81 seconds**, including both corrected failures,
all persistence/control tests and the provenance regression, with three
expected Agg notices. The full suite was not rerun after those refinements.

All three CLI commands above succeeded. The loaded file combines an edited
Moon velocity with a seeded Earth-relative ring; both the automated subprocess
test and independent CLI command rebuild it in a fresh process. Its
1350x1350 PNG and the normal App/dialog previews at
`output/preparation-controls.png` and `output/experiment-controls.png` were
visually inspected for heading, preparation/live controls, separate save
paths and epoch/step disclosure. These are ignored local artifacts; native
desktop input and multiwindow behavior remain unverified.

Calculated-playback integration commands:

```powershell
C:\Python314\python.exe -m pytest -q --tb=short --basetemp output/pytest-playback-final-09
C:\Python314\python.exe -m pytest -q tests/test_recording.py tests/test_recorded_playback.py tests/test_experiment_controls.py tests/test_star_system_display.py::test_export_keeps_inspector_text_and_recovers_live_cache tests/test_star_system_display.py::test_partial_redraw_matches_full_redraw_after_motion_and_resize --tb=short --basetemp output/pytest-playback-post-10
C:\Python314\python.exe -m simulations.star_system --load output/playback-setup.json --years 0.03 --output output/playback-loaded-cli.png
C:\Python314\python.exe -m simulations.star_system --star sun --planet earth --planet jupiter --years 1 --output output/playback-legacy-cli.png
git diff --check
git diff --check codex/issue-8-persistence...HEAD
```

The full integration run passed **269 tests in 357.53 seconds**, with 33
expected Agg window notices. Final review then added displayed world r/v
vectors to the inspector and two regressions for bounded time slicing and a
naturally unresolved trajectory. The subsequent focused run passed **40 tests
in 37.98 seconds**, including recorded capture after a different draft was
edited, no-reintegration guards, full/cached/nonblit rendering, and live export
cache recovery. The full suite was not repeated after those small refinements.

The initially valid radial-infall setup (Sun plus Earth at 0.1 AU, zero initial
velocity, dt = 1e-5 yr) naturally halted after **552 of 1000 requested steps** at
**0.00552 yr**, retaining finite last-valid arrays and an explicit unresolved
encounter reason. It did not claim completion or silently reset the experiment.
Both CLI exports above succeeded and their 1350x1350 PNGs were visually inspected.
Normal App exports `output/playback-preparation.png`,
`output/calculation-progress.png`, `output/recorded-playback.png`, and
`output/natural-recorded-halt.png` (1776x1104), plus
`output/calculation-review.png` (960x600), were visually inspected for controls,
source association, physical clocks, interval readout and incomplete status.
These remain ignored local artifacts. Agg event checks and image review do not
establish native desktop input or multiwindow behavior.

Sun–Jupiter integration commands:

```powershell
C:\Python314\python.exe -m pytest -q --tb=short --basetemp output/pytest-jupiter-final-09
C:\Python314\python.exe -m pytest -q tests/test_jupiter_experiments.py tests/test_reference_frames.py tests/test_recorded_playback.py::test_recorded_rotating_history_matches_live_samples --tb=short --basetemp output/pytest-jupiter-post-12
C:\Python314\python.exe -m simulations.star_system --load output/sun-jupiter-ring.json --years 5.724355585 --sample-stride 4 --output output/jupiter-saved-cli.png
C:\Python314\python.exe -m simulations.star_system --load output/sun-jupiter-control.json --years 5.724355585 --sample-stride 4 --output output/jupiter-control-cli.png
C:\Python314\python.exe -m simulations.star_system --star sun --planet earth --planet jupiter --years 1 --output output/jupiter-legacy-cli.png
git diff --check
git diff --check codex/issue-8-playback...HEAD
```

The full integration run passed **287 tests in 400.24 seconds**, with 34 expected
Agg window notices. Two additional regressions (natural halting in both matched
runs and preserving/rejecting moon-parent controls), plus distinct particle-row
assertions, were added while that run was in progress. The final focused run
passed **18 tests in 18.32 seconds**, with one expected Agg notice. Production
code did not change after the full run started. Tests cover equal resolved
initial vectors, save/reload and a fresh-process saved-preset export, source
association through the actual shared-dialog workflow, common-time incomplete
results, historical rotating samples matching live samples, immutable view
switches, inverse placement/physical velocity arrows, stable particle identity,
400-vertex trails and rotating availability through sidebar/modal transitions.

All three CLI exports above succeeded and their 1350x1350 PNGs were visually
inspected. Normal App previews `output/jupiter-rotating-live.png`,
`output/jupiter-ordinary-live.png`, `output/jupiter-rotating-authoring.png`,
`output/jupiter-recorded-comparison.png`, and
`output/jupiter-control-recorded.png` (1776x1104), plus
`output/jupiter-experiment-picker.png` (1020x1020), were visually inspected for
distinct particle IDs, physical vectors, actual trail intervals/sampling,
matched source/time/status readouts, frame availability, and labeled geometric
guides. These artifacts remain ignored. Native desktop interaction is unverified.

Collision integration commands:

```powershell
C:\Python314\python.exe -m pytest -q --tb=short --basetemp output/pytest-collisions-final-14
C:\Python314\python.exe -m pytest -q tests/test_collisions.py tests/test_collision_workflows.py tests/test_star_system_display.py::test_export_keeps_inspector_text_and_recovers_live_cache --tb=short --basetemp output/pytest-collisions-post-17
C:\Python314\python.exe -m pytest -q tests/test_collisions.py::test_remnant_id_avoids_current_and_historical_identities_deterministically --tb=short --basetemp output/pytest-collisions-identity-18
C:\Python314\python.exe -m simulations.star_system --load output/collision-setup.json --years 0.4 --sample-stride 100 --output output/collision-saved-cli.png
C:\Python314\python.exe -m simulations.star_system --star sun --planet earth --planet jupiter --years 1 --output output/collision-legacy-cli.png
git diff --check
git diff --check codex/issue-7-jupiter...HEAD
```

The full integration run passed **322 tests in 428.77 seconds**, with 35 expected
Agg window notices. Review then tightened configurable contact-refinement failure,
reserved current/historical/captured remnant IDs, added explicit stellar pair and
natural close-approach checks, and refined caption spacing/provenance/units and
model-label layout. The focused run passed **39 tests in 32.59 seconds**, with one
expected Agg notice. A final captured-ancestry lookup assertion passed its focused
identity regression (**1 passed in 0.12 seconds**). This is the final verification
boundary; the full suite was not repeated after those scoped refinements.

Both CLI commands succeeded. The workflow tests also run saved collision exports
in fresh processes. A selectable-rule run that only meets a star completes and
exits 0; that contact is logged as unmodeled. The binary-star scenario still
exits 2 when a contact is outside its supported pair. The separately reviewed
CLI images are 1350x1350. Normal App previews `output/collision-preparation.png`,
`output/collision-live-remnant.png`, `output/collision-recorded-pre.png`, and
`output/collision-recorded-post.png` (1776x1104), plus
`output/collision-policy.png` (1020x1020), were visually inspected for explicit
mode/reset, exact event time, membership, remnant provenance, diagnostic units,
selection mapping and replay source/status. The policy caption and remnant
unknown-property text were separated from adjacent controls/preview, and the
model label was given its own line to preserve the CLI clock. These artifacts
remain ignored local outputs; no native desktop input or multiwindow validation
is claimed.
