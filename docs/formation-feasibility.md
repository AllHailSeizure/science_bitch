# Late-stage formation feasibility

This note is the first slice of a formation experience: a star and massive rocky embryos, a parcel ledger, checkpoints, and two runs branched from one saved state. The code is `simulations.star_system.formation`. It reuses the existing perfect-merge contact solver, `StateFrame`, `RecordedRun`, and `Playback`. The Qt observatory is unchanged. Scrubbing a finished run goes through those playback types, and the checkpoint file is the saved recording.

## Starting stage, material, and collisions

The candidate stage is late-stage assembly around a star that already exists. Each embryo is one body and one material parcel, with a stable id and a mass. There is no gas, no dust bin, and no tree that groups distant mass.

Contacts use the disclosed perfect-merge model `nonstellar-perfect-merge-v1`. One remnant keeps the mass and the linear momentum. The relative kinetic energy is recorded as lost, and the orbital spin residual is recorded as unmodeled. A collision has one descendant. A nonstellar body leaves the active set when its two-body specific energy about the star is positive and its star-centric distance has reached the ejection radius. The check runs at each macro step. The body, its parcels, and its last position stay in the history. Stellar contact halts the run.

The questions this prototype can answer are:

- Which initial parcels make up a surviving body, and how much mass came from each?
- When did mergers and ejections happen, including for a body that has left the region?
- If two runs share a checkpoint and one embryo velocity changes, how do their histories differ at the same simulated time?
- Do mass, parcel totals, and linear momentum follow the perfect-merge and ejection rules, and does the kinetic-energy drop at a merge match the recorded loss?
- How much does the first merger time move when the fixed step changes, while the remnant mass stays the combined parcel mass?

Outside this slice: a gas disk and drag, grain growth, a rule for turning a swarm into one body, fragmentation and hit-and-run, three-dimensional orbits, relativity, and stellar mergers. The gas-and-dust stage is a separate prototype, `simulations.star_system.gas_dust`, written up in `docs/gas-dust-formation.md`. It does not change this late-stage run. Optional close-pass tidal dissipation is a separate experiment choice and is not turned on here.

## Benchmark

`run_benchmark()` builds the seeded annulus and integrates it. Embryo radii are 0.04 AU, about 940 Earth radii using the catalog's 6371 km Earth radius, so they are encounter radii rather than rocky surfaces. Each embryo is 2e-6 solar masses, about two-thirds of an Earth mass. Initial spacing is at least 1.5 times the sum of the encounter radii, so the run does not start in contact. One extra unbound embryo starts outside the annulus so ejection has a known case. The annulus velocities are star-centric ellipses from PCG64 with seed 48.

Measured once on this host: Linux-6.12.113+-x86_64-with-glibc2.39; Intel(R) Xeon(R) Processor; 4 logical CPUs; Python 3.12.3; NumPy 2.5.5. Wall time includes building the setup and integrating. It does not include writing the checkpoint. Peak memory is Python's tracemalloc peak, not the operating-system resident set.

| Quantity | Result |
| --- | ---: |
| Bodies | 82 (1 star, 80 embryos, 1 unbound embryo) |
| Fixed step | 0.025 yr |
| Requested duration | 1 yr |
| Stored clock | 1.0000000000000004 yr |
| Status | complete |
| Calculation time | 1.637 s |
| Peak Python allocations | 14,759,635 bytes |
| Checkpoint size | 557,336 bytes |
| Samples kept | 25 |
| Mergers | 7 |
| Ejections | 1 |

The ejection is the disclosed unbound embryo at 0.300 yr. The merger times are 0.075910, 0.142517, 0.148492, 0.307739, 0.494739, 0.670525, and 0.819307 yr. Six mergers join two original embryos. One is a chain: embryo-009 and embryo-019 merge at 0.142517 yr, and that remnant merges with embryo-004 at 0.494739 yr. That survivor carries three parcels of 2e-6 solar masses. Closed mass stayed 1.000162 solar masses. Parcel totals matched. The largest momentum discrepancy on a merge was 2.7e-20, the kinetic-energy discrepancy against the recorded loss was 1.1e-19, and the ejection cut differed by 3.4e-21, in model units. Those residuals are roundoff.

At this inflated radius, a year of 82 bodies is practical to calculate and leaves several encounters to inspect. A physical rocky radius is about a thousand times smaller. Contacts would be much rarer, and the same fixed step would spend its effort on close misses or halt. The interesting history here belongs to the inflated encounter radius.

A separate 200-embryo trial, same step, year, seed, and radius, finished in 4.94 s with 37 mergers and the same disclosed ejection. That is the upper end of a 50–200 body first benchmark. It is one measurement on this host, not the regression fixture. The checked run is the 80-embryo `run_benchmark()`.

## History, ancestry, and branching

Parcel ids, event times, contribution masses, and descendant ids stay on the run. `save_checkpoint` writes `formation-checkpoint-v1` JSON. Loading that file and integrating again reproduced the uninterrupted positions, velocities, masses, and contribution amounts in the tests.

`replay_ancestry` on a survivor lists the parcel masses and can highlight, in each sample, the bodies still carrying those parcels. On a merged or ejected body it returns the samples from when that body existed, the removal event, and the next sample, when the region no longer holds it. The demonstration escaper crosses 6 AU and the following sample has no body at that position.

`branch` copies the checkpoint, stores the parent checkpoint id, the original setup digest, and one `VelocityChange` with its note, and does not edit the source run. `SynchronizedPlayback` steps both recordings only inside the simulated interval they share. `playback` builds a `RecordedRun` and returns the existing `Playback` type. Neither path calls the integrator. Setup JSON still stores the starting conditions only. The formation checkpoint is the saved history.

## Resolution

The same head-on pair (masses 1e-5 and 2e-5 solar masses, radii 0.08 AU, placed at 5.0 and 5.5 AU) was integrated at two fixed steps.

| Fixed step (yr) | First contact (yr) | Remnant mass (solar masses) |
| ---: | ---: | ---: |
| 0.02 | 0.06809379903882924 | 3.0000000000000004e-5 |
| 0.005 | 0.06809376872464797 | 3.0000000000000004e-5 |

The contact times differ by 3.03e-8 yr. The remnant mass is the sum of the two parcels at both steps. This head-on inflated encounter is converged across that step change. Grazing contacts in the existing perfect-merge solver move by more when the step changes. That behavior is part of the contact solver this prototype calls, and it is unchanged.

## Recommended scope

Keep the first implementation on late-stage embryos, one parcel per body, perfect merge, ejection records, ancestry amounts, and checkpoint branching. Calculate a run to completion, then scrub and compare it. Leave stellar contact as a halt.

That earlier gas-and-dust start is the annular isothermal prototype in `docs/gas-dust-formation.md`. It keeps a fluid disk, drag between gas and dust, grain growth, and a swarm-to-body rule that holds onto mass, momentum, origin, and the radius and ancestry the collision model needs. This late-stage run still does not include them. Partial accretion, erosion, hit-and-run, and fragmentation need an outcome map such as Stewart and Leinhardt, because perfect merge cannot send mass to more than one descendant. Tree gravity is a later optimization: at 200 bodies the measured cost was a few seconds of direct summation, and a grouped force would still have to keep parcel identity.

The in-repo Verlet contact solver was enough for these inflated-radius runs. A close-encounter integrator such as REBOUND is the piece to revisit if the radii become physical and the fixed step starts halting. It is not required for the questions above.

```python
from simulations.star_system.formation import (
    FormationExperiment, make_late_stage_setup, run_benchmark, save_checkpoint,
)

experiment = FormationExperiment(
    make_late_stage_setup(), ejection_radius=8.0, sample_stride=5, seed=48)
experiment.integrate(1.0, finish=True)
save_checkpoint(experiment, 'output/formation.json')
report = run_benchmark()
```
