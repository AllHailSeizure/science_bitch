# Formation numerics

This note is the dimensionality, radius, and close-encounter schedule for late-stage formation. The code is `simulations.star_system.formation`. Gravity is the base. Collision handling, gas drag, and close-pass tides are separate switches on that path. Perfect merge remains the default collision rule and its remnant is unchanged. The inflated planar benchmark and its measurements stay in [formation-feasibility.md](formation-feasibility.md).

The overhead observatory draws planar motion. A three-dimensional sample is refused there instead of being projected. Tree gravity is not in this schedule.

## Dimensionality

Presets and the inflated benchmark store two components. Their setup documents keep the model id `planar-newtonian-point-mass-v1`. A body may instead be given three components. Every position and velocity in one setup must share that length. The spatial document uses `newtonian-point-mass-v1`. Populations and the moon period/600 aid stay planar and are rejected on a three-dimensional setup.

Gravity is the same unsoftened sum in either length. A planar vector is not given a silent zero. An inclined orbit has to be written with three components or it stays in the plane.

One checked fixture is a circular embryo at 1 AU, inclination 30 degrees, mass 1e-6 solar masses, and a fixed step of the two-body period / 600. After a quarter period the star-centric z coordinate was 0.5000 AU. The direction of the specific angular momentum moved by about 1.2e-6 degrees. Those digits are from this host and this fixture. They are not a long-term stability bound.

## Rocky radii

`rocky_radius` is a constant-density sphere using the fact-sheet Earth in `PLANETS['earth']`: mass 5.97e24 kg and radius 6371 km. It is not an interior model.

At the benchmark embryo mass, 2e-6 solar masses, that radius is 3.719e-5 AU. The inflated encounter radius is 0.04 AU, 1075 times larger. The geometric cross section is about 1.2 million times smaller. The same trajectories therefore meet much less often. Mergers in the inflated benchmark belong to that encounter radius.

`make_late_stage_setup(..., radii='encounter')` is the default and still uses 0.04 AU, planar vectors, and the fixed step. `radii='rocky'` stores the constant-density radius. A positive `inclination` tilts each embryo by at most that many radians. Zero inclination does not draw extra random numbers, so the planar seed is the same sequence as before.

## Timestep

The authored `dt` is the fixed macro step, or the maximum step when adaptive control is on. Adaptive control is off unless the experiment is constructed with `adaptive=True`. Inflated runs, including `run_benchmark`, do not turn it on. Their contact times are the fixed-step solver's times.

On an adaptive run each macro step is the smaller of the authored step and the existing pair resolution limit. That limit is already `min(gravitational time, crossing time) / 10`, with the gravitational time `sqrt(distance^3 / (G (m_i+m_j)))` and the crossing time `distance / |v_i-v_j|`. After a pass the step at most doubles, until it is back at the authored step. It never drops below `1e-10` yr. If the limit is below that floor, the state is not advanced and the run halts. The inner positive-time Verlet still subdivides each macro, at most to 1/8 of the macro and half the limit at both ends. Whole macros still end the requested interval, so the stored clock can pass the request by less than one maximum step.

The collision numerical id remains `positive-verlet-swept-parabola-v1`. Choosing the macro step is not a new contact rule. A merge is still one remnant, with the mass and the linear momentum. In three dimensions the recorded unmodeled spin is the magnitude of the residual angular momentum. Planar runs keep the signed z component.

## Close-encounter integrator

REBOUND was the candidate named for a physical radius, if the fixed step started spending its budget on close misses or halted. IAS15 from REBOUND 5.2.3 was compared once on this host. REBOUND is not a package dependency, and the tests do not import it.

The comparison fixture is `make_close_pass_setup`: two rocky embryos, impact parameter 0.01 AU, relative speed 6 AU/yr, far from the star. Neither schedule merged them.

| Schedule | Stored clock (yr) | Macros | Smallest macro step (yr) | Wall time (s) |
| --- | ---: | ---: | ---: | ---: |
| Fixed step 0.02 yr | 0.4000000000000001 | 20 | 0.02 | 0.084 |
| Adaptive, maximum 0.02 yr | 0.4186331616484505 | 108 | 1.665e-4 | 0.180 |
| REBOUND IAS15, same end time as the adaptive run | 0.4186331616484505 | — | — | under 0.001 |

At that shared time the adaptive positions differed from IAS15 by 2.3e-8 AU on each embryo. The star differed by 1.5e-15 AU. On the inclined quarter-orbit fixture the embryo differed from IAS15 by 4.3e-7 AU. IAS15's wall time is a three-body C integration. It is not a parcel ledger.

The fixed step also finished this miss. It remains enough for the inflated benchmark and for a single pass that fits in the solver's subdivision budget. The adaptive step is what a hardened formation run uses, because the stored macro step follows the encounter instead of holding 0.02 yr across it. A pass that needs more than the inner budget of one authored macro can still halt the fixed schedule. The adaptive floor is the halt for a pass this solver will not take.

REBOUND is not adopted. Taking it would add a second gravity implementation beside the parcel ids, the checkpoint, and the perfect-merge sweep. IAS15 or MERCURIUS stays the candidate if a later run halts at the floor or leaves the two fixtures above. Those fixtures are not a general error bound.

## Physical-radius probe

`run_physical_probe` builds 8 embryos plus the star, seed 70, rocky radii, maximum inclination 0.1 rad, adaptive control, authored step 0.025 yr, and requests 0.25 yr. One measurement on this host:

Linux-6.12.113+-x86_64-with-glibc2.39; Intel(R) Xeon(R) Processor; 4 logical CPUs; Python 3.12.3; NumPy 2.5.5.

| Quantity | Result |
| --- | ---: |
| Bodies | 9 (1 star, 8 embryos) |
| Requested duration | 0.25 yr |
| Stored clock | 0.2744271871407581 yr |
| Status | complete |
| Calculation time | under 0.12 s |
| Smallest macro step | 0.01946 yr |
| Largest macro step | 0.025 yr |
| Mergers | 0 |

The clock is past 0.25 yr by less than one authored step. The smallest step stayed near 0.025 yr, so this window did not contain a deep encounter. Nothing merged. Repeats on this host finished in about 0.025 s after the first call; a cold call was closer to 0.1 s. The clock, the step sizes, and the merger count are the stable results. The run is cheap to calculate, and it does not produce the encounter history of the inflated benchmark. The inflated 80-embryo `run_benchmark` stays the planar fixed-step check. The rocky 80-embryo year on the combined adaptive path is measured below. Direct summation is what this probe used. Tree gravity stays a later optimization, and a grouped force would still have to keep parcel identity. This probe leaves gas drag and tides off.

## Combined switches

Gravity is the base. Collision handling, gas drag, and close-pass tides are separate switches on the same formation setup. Dimensions (2 or 3) and the adaptive step are separate from those switches. The default remains perfect merge, drag off, tides off, planar vectors, and a fixed step. Catalog presets stay on `collision_mode='none'` and the Yoshida step, which can still halt on an unresolved encounter. A formation run does not use that mode.

A formation run accepts `ignore`, `stop_on_contact`, `perfect_merge`, and `speed_angle`. Ignore lets an overlap continue with no remnant. Stop on contact is the only collision choice that ends the run, and that end is status `stopped`, not a numerical halt. Perfect merge and the speed-angle rule are the stored collision models. A contact does not stop the run unless stop on contact is selected. Stellar contact is logged as unmodeled and the requested duration still completes, including on a gas-bearing run. The standalone gas-and-dust prototype still halts on stellar contact; that prototype is unchanged.

Drag, when enabled, uses the existing quadratic body law and one of the existing annular disks (`annular`, `envelope`, or `giant`). The disk is evaluated at cylindrical radius. Height only multiplies the midplane density by the isothermal Gaussian. There is no vertical gas velocity and no new disk method. The annular fluid is subcycled at the existing Courant limit inside the macro step. If a subcycle would empty a zone, that piece is held and the gravity run continues. Dust swarms, grain growth, and the swarm-to-body rule stay in the gas-and-dust prototype. Bodies on this path use quadratic drag only.

Tides, when enabled, use `close-pass-energy-dissipation-v1`. On a contact-solver step the kick applies only to a pair that does not become a contact in that step. A contacting pair is left to the collision rule. In three dimensions the recorded angular-momentum change is a magnitude. Planar runs keep the signed z component. The Yoshida path (`collision_mode='none'`) is unchanged.

The combination in use is `FormationExperiment.combination()` and the switches line in `describe()`. Outcome claims about a longer-lived capture are made only for a run that has the relevant switch on. Checkpoints store `tides` and `drag` only when that switch is on. A checkpoint without those fields still loads. Parcel ids, checkpoints, and a branched velocity change follow the same formation files. A spatial adaptive checkpoint stores `dimensions`, `adaptive`, `dt_max`, and `dt_min`. A planar fixed-step checkpoint does not gain those fields.

## Rocky year

`run_rocky_year` is 80 embryos plus the star, seed 48, rocky radii, maximum inclination 0.1 rad, adaptive control, authored step 0.025 yr, and one simulated year. The measured combination is perfect merge, the annular disk, and tides. Each of those can be left off. Rocky radii make the quadratic drag cross-section tiny, so the dissipated kinetic energy on this year can be negligible. That is reported as measured, not adjusted.

One measurement on this host: Linux-6.12.113+-x86_64-with-glibc2.39; Intel(R) Xeon(R) Processor; 4 logical CPUs; Python 3.12.3; NumPy 2.5.5.

| Quantity | Result |
| --- | ---: |
| Bodies | 81 (1 star, 80 embryos) |
| Requested duration | 1 yr |
| Stored clock | 1.0051284839311654 yr |
| Status | complete |
| Calculation time | 2.93 s |
| Smallest macro step | 0.007999578262771059 yr |
| Largest macro step | 0.018060160830636794 yr |
| Collision | perfect_merge |
| Gas drag | on, annular disk |
| Tides | on, close-pass-energy-dissipation-v1 |
| Mergers | 0 |
| Tide events | 0 |
| Ejections | 0 |
| Unmodeled stellar contacts | 0 |
| Dissipated kinetic energy | 1.9262137457981554e-12 |

The stored clock is past 1 yr by less than one authored step. The run completed. Nothing merged, nothing was ejected, and no body met the star. No close pass reached the tidal reach, so the tide switch recorded no event. Drag removed 1.9262137457981554e-12 of kinetic energy. At a rocky radius that cross-section is about a million times smaller than the inflated encounter radius, so this year does not show drag changing an orbit. The same year on Windows with Python 3.14 also completed, with the same body count and the same event counts, and stored 1.005128483931164 yr. The two clocks agree through 1.005128483931 yr and differ in the last digit. Wall time was 2.93 s on the Linux host and can differ on another run. Direct summation is what this year used. Tree gravity stays deferred.

## Sun–Jupiter flyby

`run_sun_giant_flyby` places the Sun, catalog Jupiter, and one visitor on this combined path. The visitor mass is 1e-6 solar masses. Its radius is 0.05 AU, a disclosed drag cross-section, not a rocky surface. It starts unbound to the Sun, offset from Jupiter by (2.2, 0.5, 0.1) AU with an extra velocity of (-2.8, -0.6, -0.02) AU/yr. The disk, when enabled, is the existing annular isothermal disk placed from 4.2 to 6.4 AU with uniform surface density 0.02. The same initial state is run four times: gravity only, drag on, tides on, and drag and tides together. Collision stays perfect merge. Adaptive stepping and three-dimensional state are on for every run. The requested duration is 2 yr.

A body that is bound at the end of this window, and still a separate body, is a longer-lived capture for this window. It is not a permanent capture: a later scattering can still unbind it. Fragmenting collisions beyond the stored speed-angle map remain #69, a close-encounter integrator remains the candidate if a run halts at the adaptive floor (#70), and the tide formula is the optional close-pass model (#79). Gas drag is the existing quadratic law (#71).

gravity only: collision perfect_merge, drag off, tides off, status complete, Sun-specific binding unbound, initial Sun-specific energy 0.924109, last Sun-specific energy while the visitor exists 0.807138, last Jupiter-specific energy while the visitor exists 4.59125, minimum Jupiter distance 0.072645 AU, tide events 0
drag on: collision perfect_merge, drag giant, tides off, status complete, Sun-specific binding bound-at-end, initial Sun-specific energy 0.924109, last Sun-specific energy while the visitor exists -3.1648, last Jupiter-specific energy while the visitor exists 0.0954656, minimum Jupiter distance 1.0518 AU, tide events 0
tides on: collision perfect_merge, drag off, tides close-pass-energy-dissipation-v1, status complete, Sun-specific binding merged after bound-at-end, initial Sun-specific energy 0.924109, last Sun-specific energy while the visitor exists -3.69642, last Jupiter-specific energy while the visitor exists -0.518857, minimum Jupiter distance 0.0504673 AU, tide events 1
drag and tides: collision perfect_merge, drag giant, tides close-pass-energy-dissipation-v1, status complete, Sun-specific binding bound-at-end, initial Sun-specific energy 0.924109, last Sun-specific energy while the visitor exists -3.1648, last Jupiter-specific energy while the visitor exists 0.0954656, minimum Jupiter distance 1.0518 AU, tide events 0
A longer-lived capture, for this window, is claimed only for the runs that end bound and unmerged: drag on, drag and tides. No unmerged run was bound and then unbound again inside this window. A perfect merge removed the visitor as a separate body for: tides on. That merger is not a capture. On the drag-and-tides run the closest approach stays outside the tidal reach, so no tide event is recorded and the end state matches drag alone. The visitor starts unbound to the Sun. This is not a permanent-capture claim. A later scattering can still unbind a body, and the visitor radius of 0.05 AU is a drag cross-section, not a rocky surface.

```python
from simulations.star_system.formation import (
    FormationExperiment, make_close_pass_setup, make_inclined_circular_setup,
    run_physical_probe, run_rocky_year, run_sun_giant_flyby,
)

probe = run_physical_probe()
year = run_rocky_year()
flyby = run_sun_giant_flyby()
inclined = FormationExperiment(make_inclined_circular_setup(), ejection_radius=5.)
close = FormationExperiment(make_close_pass_setup(), ejection_radius=80., adaptive=True)
```
