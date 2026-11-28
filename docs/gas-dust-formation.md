# Gas-and-dust formation

This note is the earlier starting stage left open by the late-stage embryo prototype. The code is `simulations.star_system.gas_dust`, model `gas-dust-annular-isothermal-v1`. The late-stage run in `simulations.star_system.formation` is unchanged: it still begins with one rocky embryo per parcel. A formed body can be handed to that prototype with its parcel ledger intact.

The fluid method here is a planar annular isothermal disk. GADGET is a reference for a particle fluid, not the method this slice runs. The step stays fixed. Three-dimensional orbits, an adaptive step, fragmenting collisions, and tides stay in #70, #69, and #79.

## What each object is

| Object | What it represents |
| --- | --- |
| Star | A pinned point mass at the origin. It does not move and it does not merge. |
| Gas zone | One piecewise-constant annular fluid element. The zone id is stable. Mass can flow into a neighbor. The zone is not a molecule and not a solid parcel. |
| Dust swarm | One population of equal grains, with a Lagrangian parcel ledger. It is not yet a gravitating body, and it does not source gravity. |
| Body | One planetesimal produced by the swarm rule, or one authored intruder. It is one participant in the existing perfect-merge model. |

Solid parcels keep their ids. Gas identity is the zone. Total gas mass is conserved because the fluxes move mass between zones; a zone's own mass is not a closed parcel.

The seeded disk is a power law. Surface density is \(2\times10^{-4}\) solar masses per AU squared at 1.5 AU and falls as \(r^{-1}\), from 0.8 AU to 2.4 AU, in 16 zones. The scale height over radius is 0.05 at 1.5 AU, which sets the isothermal sound speed. Six ordinary swarms start on Keplerian orbits with millimeter grains, mass \(10^{-10}\) solar masses, and one parcel each. Their angles come from PCG64 with seed 71. One additional swarm, `swarm-ready`, starts at 1.5 AU already past the body rule, with mass \(10^{-8}\) solar masses and a 1 km grain. Grain and bulk density are 3000 kg/m³.

## Fluid, drag, and grain growth

Each fixed step updates the annular fluid first, then grows grains from the solid velocity at the start of the step, coagulates overlapping swarms, applies the swarm rule, and integrates the solids. The solid step is a velocity Verlet kick and drift. Gas drag sits between the half-kicks. Body-body contact is checked at the end of the step.

The fluid source is centrifugal support, the pinned star's gravity, and the isothermal pressure gradient \((1/\Sigma)\,dP/dr\). Advection is a donor-cell flux of mass, radial momentum, and angular momentum, with no-flux ends. Gas is not self-gravitating, and it does not feel the solids except through drag. There is no vertical velocity. A discrete equilibrium rotation is sub-Keplerian where the surface density declines.

Swarms use Epstein drag. The stopping time is \(\rho_s s / (\rho_g c_s)\), with the midplane density \(\Sigma / (\sqrt{2\pi} H)\) and \(H = c_s/\Omega\). Bodies use quadratic drag with coefficient 0.44. Both laws deposit the opposite radial momentum and the opposite angular momentum about the star on the zone that contains the solid. The star is pinned, and a zone stores only those two axisymmetric momenta, so the closed exchange is that pair rather than a free cartesian vector. The kinetic energy the exchange removes is accumulated as `dissipated_kinetic`. The isothermal model does not return it as heat.

Grain growth is representative sweep-up,

\[
\frac{ds}{dt} = \frac{\rho_\mathrm{dust}\,v_\mathrm{rel}}{4\rho_s}.
\]

The grain radius changes. The swarm mass and the parcel amounts do not. Two swarms whose clouds overlap coagulate when the combined internal speed, including their relative motion, stays below the cloud escape speed. Mass, momentum, and parcels add. The relative kinetic energy becomes internal kinetic energy.

The fixed step is not reduced. If it exceeds 0.4 of the zone width over \(|v_r| + c_s\), or if transport would empty a zone, the run halts. That limit is the one owned by #70.

Outside this model: three-dimensional motion, vertical settling, a turbulence spectrum, SPH or GADGET, gas self-gravity, the streaming instability, a multi-bin grain distribution, and tree-code grouping. Grouping, if it is added later, has to keep parcel identity.

## Swarm-to-body rule

The rule id is `hill-bound-stokes-v1`. A swarm becomes one body only when every gate passes:

- Stokes number \(t_\mathrm{stop}\Omega \ge 1\). Outside the gas the swarm is treated as decoupled, so this gate passes.
- Grain radius at least 1 km.
- Swarm mass at least \(10^{-9}\) solar masses.
- The swarm's spatial radius fits inside its Hill radius about the star.
- The internal speed is at most the swarm's escape speed.

The Hill check is the compactness condition for the swarm becoming one body. It is not the tidal evolution in #79.

The body keeps the swarm's mass, velocity, parcel amounts, grain radius, and internal speed. Its radius is the compact sphere at the disclosed bulk density, which is the radius `nonstellar-perfect-merge-v1` uses. Internal kinetic energy is stored on the body and on the event. It is not added to the orbit and it is not dropped. A later body-body contact calls the existing perfect-merge primitive and does not change that collision rule. The remnant keeps the collision model's volume radius, the recorded energy loss and spin residual, and the summed parcel ledger. The parents' internal kinetic energy is kept as the remnant's internal speed. Stellar contact halts the run.

`continue_as_late_stage` builds a late-stage `FormationExperiment` from the star and the current bodies. The composition argument on that experiment carries the parcel masses, and each body's provenance string carries the grain radius, internal speed, internal kinetic energy, and bulk density. Perfect merge still replaces a remnant's provenance string, because that text is part of the collision model this issue does not change. The parcel masses stay on the formation ledger.

## Checkpoints and branches

`save_checkpoint` writes `gas-dust-checkpoint-v1` JSON. The file holds the zones, swarms, bodies, removed swarms and bodies, events, samples, parcel amounts, and the running drag and transition residuals. Loading it and integrating again reproduces the uninterrupted zone masses, angular momenta, solid positions, and contribution amounts.

`branch` copies that state, stores the parent checkpoint id and one `VelocityChange`, and does not edit the source run. The change may name an active swarm or an active body.

## Benchmark

`run_disk_experiment()` builds the seeded disk, integrates it, and then runs the envelope comparison below. Wall time includes both. It does not include writing the checkpoint. Peak memory is Python's tracemalloc peak, not the operating-system resident set.

Measured once on this host: Linux-6.12.113+-x86_64-with-glibc2.39; Intel(R) Xeon(R) Processor; 4 logical CPUs; Python 3.12.3; NumPy 2.5.5.

| Quantity | Result |
| --- | ---: |
| Representation | annular fluid and dust swarms |
| Gas zones | 16 |
| Initial swarms / solid parcels | 7 |
| Bodies at the end | 1 |
| Fixed step | 0.02 yr |
| Requested disk duration | 0.5 yr |
| Stored disk clock | 0.5000000000000001 yr |
| Status | complete |
| Swarm-to-body events | 1 |
| Coagulations | 0 |
| Samples kept | 26 |
| Calculation time | 0.970 s |
| Peak Python allocations | 4,265,344 bytes |
| Checkpoint size | 131,768 bytes |

Gas mass stayed at 0.003013971866457235 solar masses and closed to \(4.34\times10^{-19}\). Parcel error was 0. Angular-momentum error about the star was \(3.47\times10^{-18}\). The promotion is `swarm-ready`. The six millimeter swarms are still swarms: this half-year of sweep-up does not carry them across the 1 km gate. The promoted body's mass is the swarm mass, \(10^{-8}\) solar masses.

## Envelope capture

Gas drag pulls a body toward the local gas velocity and removes hyperbolic excess. The comparison uses one Gaussian envelope, surface density \(0.05\exp(-(r/0.45\,\mathrm{AU})^2)\) from 0.2 to 1.6 AU, sound speed 0.35 AU/yr, and one intruder of mass \(10^{-6}\) solar masses. The drag radius is 0.04 AU. A radius this large is a demonstration cross-section, not a rocky surface. The intruder starts at \((-1.8, 0.6)\) AU with velocity \((7, 0)\) AU/yr. The only switch between the two runs is `drag_enabled`. The envelope's own gravity is not applied, so the energy difference is from drag. The fixed step is 0.002 yr for 0.6 yr. Both runs finished.

| Run | Initial specific energy | Final specific energy | Dissipated kinetic energy | Minimum radius |
| --- | ---: | ---: | ---: | ---: |
| Drag on | 3.6932037445701447 | -27.320252893836884 | \(9.38\times10^{-5}\) | 0.710 AU |
| Drag off | 3.6932037445701447 | 3.6930588943567493 | 0 | 0.219 AU |

With drag, the specific energy changes sign and the intruder is bound. Without drag, the same initial state stays unbound, and the specific energy changes by about \(1.4\times10^{-4}\). A separate one-step check places a fast body inside the envelope and records a drop in its outward speed, with the momentum deposited on the gas and the lost kinetic energy stored.

This is aerodynamic binding in a planar fixed-step model. It is not a permanent capture. Fragmenting collisions remain #69, three-dimensional orbits and an adaptive or close-encounter integrator remain #70, and tides remain #79. A sweep of today's temporary three-body captures can still be done without this drag result. Claims of permanent capture wait on those pieces.

The formation integrator can apply this same quadratic law and these same annular zones on a three-dimensional adaptive step. That path is opt-in, leaves swarms and grain growth here, and does not halt the gravity run when a body meets the star. This standalone prototype is unchanged: planar, fixed step, perfect merge, and a halt on stellar contact. A longer-lived capture on the formation path is claimed only when that run's drag or tide switch is on. See [formation-numerics.md](formation-numerics.md).

## Questions this stage can address

- Which solid parcels sit in a swarm or a body, and how much mass came from each?
- Does gas drag move angular momentum from a Keplerian dust swarm onto sub-Keplerian gas?
- Does representative grain growth change the grain radius while the swarm mass and parcel amounts stay the same?
- When the swarm rule passes, does the new body keep the swarm mass, momentum, parcels, grain radius, and internal speed, and a positive radius the perfect-merge model can use?
- Does a checkpoint resume the same gas, swarms, and parcels, and does one velocity branch leave the source run unchanged?
- Does quadratic drag from a gaseous envelope bind a hyperbolic flyby that stays unbound when the same drag is turned off?

```python
from simulations.star_system.gas_dust import (
    GasDustExperiment, make_disk_setup, run_disk_experiment, save_checkpoint,
)

experiment = GasDustExperiment(make_disk_setup())
experiment.integrate(0.5, finish=True)
save_checkpoint(experiment, 'output/gas-dust.json')
report = run_disk_experiment()
```
