"""Opt-in stellar wind. Off unless a mass-loss policy enables it.

Prescription ``isotropic-jeans-wind-v1``:

- Every body with kind ``star`` loses mass at ``M(t) = M(0) exp(−λ t)``.
- ``λ`` is ``StellarMassLossPolicy.rate_per_year``. The default is 1e-13 per
  year, the order of a weak stellar wind, not a Reimers integration and not a
  measured solar history.
- The wind is isotropic in the star's frame (Jeans mode). It leaves the
  system and is not kept as gas, dust, or another body. The star's velocity
  is unchanged. Planets, moons, and other non-stars are not reduced.
- A stellar-merger remnant is a star, so it loses mass only while this module
  is on. The merger rule itself still sets the contact mass to the sum of the
  parents and still omits mass loss from that rule. The published head-on
  scenario requires the catalog solar masses at contact; a rate that changes
  those masses before contact is outside that scenario.

For a test particle on a slow circular orbit, specific angular momentum is
conserved and the separation scales as ``a M = constant``. That adiabatic
Jeans result is the one in Hurley, Tout, and Pols, MNRAS 329, 897 (2000),
for an isotropic wind that carries the donor's specific angular momentum and
is not accreted. Mass loss evaluated here is one update per accepted step, on
the bodies present at the end of that step.
"""
import math
from dataclasses import replace


def apply_stellar_mass_loss(prepared, elapsed):
    """Reduce stellar masses over ``elapsed`` years. No-op when the module is off."""
    setup = prepared.setup
    policy = None if setup is None else setup.stellar_mass_loss
    if policy is None or not policy.enabled or elapsed == 0.0:
        return prepared
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError('Stellar mass loss needs a finite nonnegative step.')
    system = prepared.system
    specs = list(prepared.body_specs)
    updates = []
    for index, spec in enumerate(specs):
        if spec.kind != 'star':
            continue
        mass = float(system.masses[index])
        if mass <= 0.0:
            continue
        updated = mass * math.exp(-policy.rate_per_year * elapsed)
        if not math.isfinite(updated) or updated <= 0.0:
            raise ValueError(
                f'Stellar mass loss exhausted {spec.name} in one step. '
                'Use a smaller rate or a shorter step. Lost mass leaves the system.')
        updates.append((index, updated))
    if not updates:
        return prepared
    for index, updated in updates:
        system.masses[index] = updated
        specs[index] = replace(specs[index], mass=updated)
    return replace(prepared, body_specs=tuple(specs))
