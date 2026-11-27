# Formation numerics

This note is the dimensionality, radius, and close-encounter schedule for late-stage formation. The code is `simulations.star_system.formation`. It still calls the existing perfect-merge contact solver. Collision outcomes are unchanged. The inflated planar benchmark and its measurements stay in [formation-feasibility.md](formation-feasibility.md).

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

The clock is past 0.25 yr by less than one authored step. The smallest step stayed near 0.025 yr, so this window did not contain a deep encounter. Nothing merged. Repeats on this host finished in about 0.025 s after the first call; a cold call was closer to 0.1 s. The clock, the step sizes, and the merger count are the stable results. The run is cheap to calculate, and it does not produce the encounter history of the inflated benchmark. The checked large run is still the 80-embryo inflated `run_benchmark`. An 80-embryo year at the rocky radius was not measured. Direct summation is what this probe used. Tree gravity stays a later optimization, and a grouped force would still have to keep parcel identity.

## Checkpoints and captures

Parcel ids, checkpoints, and a branched velocity change follow the same formation files. A spatial adaptive checkpoint stores `dimensions`, `adaptive`, `dt_max`, and `dt_min`. A planar fixed-step checkpoint does not gain those fields. Resuming a checkpoint and applying a third velocity component were checked against an uninterrupted run. Perfect merge still has one descendant.

Three-dimensional state lets a flyby miss in the vertical direction, and the adaptive step follows that pass instead of halting at the floor or hiding it inside one long macro. That does not turn a temporary three-body capture into a permanent one. Nothing on this perfect-merge path removes energy except a merge, which replaces both bodies with one remnant. A longer-lived capture still needs another outcome than perfect merge (#69) or gas drag (#71). Optional close-pass tides (#79) are a separate planar fixed-step choice and stay off here, including when a tide policy is stored on the setup. A capture example that stays inside today's temporary three-body flybys is not blocked on this schedule. This note does not claim one.

```python
from simulations.star_system.formation import (
    FormationExperiment, make_close_pass_setup, make_inclined_circular_setup,
    run_physical_probe,
)

probe = run_physical_probe()
inclined = FormationExperiment(make_inclined_circular_setup(), ejection_radius=5.)
close = FormationExperiment(make_close_pass_setup(), ejection_radius=80., adaptive=True)
```
