"""Opt-in outside pulls: a solar-neighborhood galactic tide and seeded stellar flybys.

The module is for very wide orbits, the region where the Galaxy's smooth field
can compete with the system's own star. It is off unless an experiment turns it
on. Catalog planets on astronomical-unit scales barely feel it.

Galactic tide
-------------
The force is the linear tide of Heisler & Tremaine (1986, Icarus 65, 13), in
the form used for the Oort cloud by Levison, Dones & Duncan (2001) and written
out by Brasser, Higuchi & Kaib (2010, A&A 516, A72, eq. 1):

    a_x = (A - B) (3A + B) x
    a_y = -(A - B)^2 y
    a_z = -[4 π G ρ - 2 (B^2 - A^2)] z

x points away from the Galactic centre, y along Galactic rotation, and z toward
the south Galactic pole. The simulation axes are these axes, held fixed: a
sidereel run is short next to a Galactic year, so the field does not need to
rotate. A and B are the Kerr & Lynden-Bell (1986, MNRAS 221, 1023) solar
Oort constants, A = 14.4 km/s/kpc and B = -12.0 km/s/kpc. The midplane density
ρ = 0.185 Msun/pc^3 is the Bahcall (1984) value adopted by Heisler & Tremaine.
The vertical term is an order of magnitude stronger than the planar terms, which
is the published solar-neighborhood result. Coriolis acceleration is not part of
this force; the integrator's kick is the position-dependent tide above.

Stellar flybys
--------------
Passers are optional and are not drawn unless the experiment stores a seed.
The statistics follow García-Sánchez et al. (2001, A&A 379, 634):

- 11.7 encounters per Myr with a closest approach inside 1 pc, after their
  Hipparcos incompleteness correction. The count inside a smaller limit scales
  as the square of that limit, matching the geometric cross-section and their
  measured cumulative slope of about 2.
- Closest-approach distances are drawn from that geometric law, pdf p(q) ∝ q
  (Heisler, Tremaine & Alcock 1987, AJ 94, 1330). Gravitational focusing is not
  applied.
- Speeds are a Maxwellian whose mean is 45.7 km/s, the mean encounter speed of
  the 92 systems in their ±1 Myr, 5 pc sample. One Maxwellian stands in for the
  spectral-type mix; the corrected rate still counts every system.

Each passer has a fixed mass of 0.5 solar masses so the force has a scale. That
mass is a representative disk star, not a draw from an initial-mass function.
The passer moves in a straight line and is not a body in the collision list, so
the system's own collision rule does not fire on it. Encounters are a Poisson
process over the run. Run i of a sweep or an open scan uses NumPy's
SeedSequence([seed, i]), so the same file replays the same draws and each run
is its own sequence. One sequence is one draw, not a general rate.
"""
import math
from dataclasses import dataclass

import numpy as np

from .core import G

TIDE_MODEL = 'heisler-tremaine-1986'
FLYBY_MODEL = 'garcia-sanchez-2001'
# Kerr & Lynden-Bell (1986), km/s/kpc.
OORT_A = 14.4
OORT_B = -12.0
# Bahcall (1984), adopted by Heisler & Tremaine (1986). Solar masses per cubic parsec.
MIDPLANE_DENSITY = 0.185
# García-Sánchez et al. (2001), incompleteness-corrected.
ENCOUNTERS_PER_MYR_WITHIN_1_PC = 11.7
# García-Sánchez et al. (2001), mean speed of the 92 encounters inside 5 pc.
MEAN_ENCOUNTER_SPEED_KM_S = 45.7
# Representative passer mass. The rate already mixes spectral types.
PASSER_MASS_SOLAR = 0.5
# IAU parsec, and the same metre and Julian year used to build core.G.
_PC_AU = 648000.0 / math.pi
_AU_M = 149597870700.0
_YEAR_S = 365.25 * 86400.0
_AU_KM = _AU_M / 1000.0
_KM_S_TO_AU_YR = _YEAR_S / _AU_KM


def _gravitational_constant_pc():
    """core.G expressed in pc Msun^-1 (km/s)^2, so the tide uses the N-body G."""
    pc_km = _PC_AU * _AU_KM
    return G * (_AU_KM ** 3) / (_YEAR_S ** 2) / pc_km


def tidal_coefficients():
    """(K_x, K_y, K_z) in (km/s)^2 kpc^-2 for the solar-neighborhood tide.

    K_z is negative: the vertical tide restores a body toward the Galactic plane.
    K_x is positive: the radial tide stretches an outward displacement.
    """
    radial = (OORT_A - OORT_B) * (3.0 * OORT_A + OORT_B)
    azimuthal = -((OORT_A - OORT_B) ** 2)
    # ν^2 = 4πGρ - 2(B^2 - A^2), then K_z = -ν^2. Brasser, Higuchi & Kaib (2010), eq. 1.
    four_pi_g_rho = 4.0 * math.pi * _gravitational_constant_pc() * MIDPLANE_DENSITY * 1.0e6
    vertical = -(four_pi_g_rho - 2.0 * (OORT_B * OORT_B - OORT_A * OORT_A))
    return radial, azimuthal, vertical


def _coefficient_per_year2(coeff_per_kpc2):
    """Convert (km/s)^2 kpc^-2 into an AU/yr^2 acceleration per AU of offset."""
    kpc_au = 1000.0 * _PC_AU
    kpc_km = kpc_au * _AU_KM
    return coeff_per_kpc2 / (kpc_au * kpc_km) * (_YEAR_S ** 2) / _AU_KM


_TIDE_PER_YEAR2 = np.array([_coefficient_per_year2(coeff) for coeff in tidal_coefficients()], dtype=float)


def galactic_tide_acceleration(offset):
    """Tide at a galactic-frame offset from the primary, in AU/yr^2.

    ``offset`` is in AU and has shape (3,) or (n, 3). The same axis convention
    as :func:`tidal_coefficients` applies. A body in the Galactic midplane
    (z = 0) feels no vertical tide until it leaves the plane.
    """
    vector = np.asarray(offset, dtype=float)
    single = vector.shape == (3,)
    if single:
        vector = vector.reshape(1, 3)
    if vector.ndim != 2 or vector.shape[1] != 3 or not np.isfinite(vector).all():
        raise ValueError('Galactic tide offset must be finite and three-dimensional.')
    acc = vector * _TIDE_PER_YEAR2
    if single:
        return acc[0]
    return acc


def encounter_rate_per_year(closest_approach_limit_pc=1.0):
    """Encounters per Julian year inside ``closest_approach_limit_pc``.

    11.7 per Myr inside 1 pc, times the square of the limit in parsecs
    (García-Sánchez et al. 2001). One Myr is 1e6 Julian years.
    """
    limit = float(closest_approach_limit_pc)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('Closest-approach limit must be a finite distance in parsecs greater than 0.')
    return ENCOUNTERS_PER_MYR_WITHIN_1_PC * (limit ** 2) / 1_000_000.0


@dataclass(frozen=True)
class Flyby:
    """One straight-line passer. Vectors are relative to the primary at closest approach."""

    time_years: float
    closest_approach_au: float
    speed_au_per_year: float
    mass_solar: float
    impact_au: tuple[float, float, float]
    velocity_au_per_year: tuple[float, float, float]

    def document(self):
        return {
            'time_years': float(self.time_years),
            'closest_approach_au': float(self.closest_approach_au),
            'speed_au_per_year': float(self.speed_au_per_year),
            'mass_solar': float(self.mass_solar),
            'impact_au': [float(component) for component in self.impact_au],
            'velocity_au_per_year': [float(component) for component in self.velocity_au_per_year],
        }

    def watch_event(self):
        """One watch-for row. It records the pass and does not call the outcome a capture or a stripping."""
        return {
            'time': float(self.time_years),
            'kind': 'flyby',
            'bodies': ['field-star'],
            'message': (
                f'A field star passes {self.closest_approach_au:.6g} AU from the primary '
                f'at {self.speed_au_per_year:.6g} AU/yr. '
                'This is one draw from the encounter model.'
            ),
        }


def _generator(seed, run_index):
    """PCG64 stream for one (seed, run index) pair. The same pair replays the same draws."""
    sequence = np.random.SeedSequence([int(seed), int(run_index)])
    return np.random.Generator(np.random.PCG64(sequence))


def _nonnegative_index(value, label):
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ValueError(f'{label} must be a nonnegative integer.')
    return value


def draw_encounters(seed, run_index, duration_years, closest_approach_limit_pc=1.0):
    """Poisson encounters over ``duration_years``, sorted by time of closest approach.

    Draw order, which the seed fixes: Poisson count, then times, closest-approach
    distances, Maxwellian velocity vectors, and perpendicular impact directions.
    ``run_index`` selects the stream SeedSequence([seed, run_index]).
    """
    _nonnegative_index(seed, 'Flyby seed')
    _nonnegative_index(run_index, 'Flyby run index')
    if isinstance(duration_years, bool) or type(duration_years) not in (int, float):
        raise ValueError('Flyby duration must be a finite number of years greater than 0.')
    duration = float(duration_years)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Flyby duration must be a finite number of years greater than 0.')
    limit = float(closest_approach_limit_pc)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('Closest-approach limit must be a finite distance in parsecs greater than 0.')
    rng = _generator(seed, run_index)
    count = int(rng.poisson(encounter_rate_per_year(limit) * duration))
    if count == 0:
        return ()
    times = rng.random(count) * duration
    # cdf of p(q) = 2q/q_max^2 is (q/q_max)^2, so q = q_max * sqrt(U).
    closest_pc = limit * np.sqrt(rng.random(count))
    sigma = MEAN_ENCOUNTER_SPEED_KM_S * math.sqrt(math.pi / 8.0)
    velocity_km = sigma * rng.standard_normal((count, 3))
    speed_km = np.linalg.norm(velocity_km, axis=1)
    stalled = speed_km == 0
    while np.any(stalled):
        velocity_km[stalled] = sigma * rng.standard_normal((int(np.count_nonzero(stalled)), 3))
        speed_km = np.linalg.norm(velocity_km, axis=1)
        stalled = speed_km == 0
    reference = rng.standard_normal((count, 3))
    reference = reference - (np.sum(reference * velocity_km, axis=1) / (speed_km ** 2))[:, None] * velocity_km
    span = np.linalg.norm(reference, axis=1)
    parallel = span == 0
    while np.any(parallel):
        reference[parallel] = rng.standard_normal((int(np.count_nonzero(parallel)), 3))
        reference[parallel] = reference[parallel] - (
            np.sum(reference[parallel] * velocity_km[parallel], axis=1) / (speed_km[parallel] ** 2)
        )[:, None] * velocity_km[parallel]
        span = np.linalg.norm(reference, axis=1)
        parallel = span == 0
    order = np.argsort(times, kind='mergesort')
    times = times[order]
    closest_pc = closest_pc[order]
    velocity_km = velocity_km[order]
    reference = reference[order]
    span = span[order]
    flybys = []
    for index in range(count):
        impact = reference[index] / span[index] * (closest_pc[index] * _PC_AU)
        velocity = velocity_km[index] * _KM_S_TO_AU_YR
        flybys.append(Flyby(
            float(times[index]),
            float(closest_pc[index] * _PC_AU),
            float(np.linalg.norm(velocity)),
            PASSER_MASS_SOLAR,
            tuple(float(component) for component in impact),
            tuple(float(component) for component in velocity),
        ))
    return tuple(flybys)


def summarize_flybys(sequences):
    """Counts across runs. The note refuses to treat one sequence as the general result."""
    counts = tuple(len(sequence) for sequence in sequences)
    return {
        'runs': len(counts),
        'flyby_count': int(sum(counts)),
        'runs_with_a_flyby': int(sum(count > 0 for count in counts)),
        'note': (
            'Each run draws its own stellar encounters from seed and run index. '
            'One encounter sequence is not a general result; read the flybys across runs.'
        ),
    }


def make_acceleration(flybys=()):
    """Acceleration added to every body: the galactic tide, plus any straight-line passers.

    The tide is evaluated relative to the most massive body, which stands for the
    Sun. A passer is placed relative to that same body. The primary feels no
    self-tide, and it does feel the passer.
    """
    scheduled = tuple(flybys)

    def acceleration(system):
        position = np.asarray(system.r, dtype=float)
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError(
                'Outside pulls are a three-dimensional galactic field for very wide orbits. This state is planar.')
        primary = int(np.argmax(system.masses))
        acc = galactic_tide_acceleration(position - position[primary])
        if scheduled:
            star = position[primary]
            when = float(system.time)
            for flyby in scheduled:
                passer = (
                    star
                    + np.asarray(flyby.impact_au, dtype=float)
                    + np.asarray(flyby.velocity_au_per_year, dtype=float) * (when - flyby.time_years)
                )
                delta = passer - position
                dist2 = np.einsum('ij,ij->i', delta, delta)
                if not np.isfinite(dist2).all() or np.any(dist2 <= 0.0):
                    raise ValueError(
                        'A passing star reached an unresolved separation. '
                        'The straight-line flyby is an external force, not a collision, '
                        'and this step cannot represent the encounter.')
                dist = np.sqrt(dist2)
                acc = acc + (G * flyby.mass_solar) * delta / (dist2 * dist)[:, None]
        if not np.isfinite(acc).all():
            raise ValueError('Outside pull produced a nonfinite acceleration. The state was not advanced.')
        return acc

    return acceleration


def attach(system, policy, run_index, duration_years):
    """Attach this policy's force to ``system`` and return the flybys drawn for the run."""
    if not isinstance(policy, OutsidePulls):
        raise TypeError('Outside pulls need an OutsidePulls policy.')
    flybys = ()
    if policy.seed is not None:
        flybys = draw_encounters(policy.seed, run_index, duration_years)
    system.external_acceleration = make_acceleration(flybys)
    return flybys


@dataclass(frozen=True)
class OutsidePulls:
    """Galactic tide on. ``seed`` is None when stellar flybys stay off."""

    seed: int | None = None

    def __post_init__(self):
        if self.seed is not None:
            _nonnegative_index(self.seed, 'Flyby seed')

    def document(self):
        return {'flybys': None if self.seed is None else {'seed': self.seed}}


def parse_outside_pulls(value):
    """``None`` leaves the module off. An object turns on the galactic tide."""
    if value is None:
        return None
    if type(value) is not dict or set(value) != {'flybys'}:
        raise ValueError(
            'physics.outside_pulls must be null, or an object with field flybys. Null leaves the module off.')
    flybys = value['flybys']
    if flybys is None:
        return OutsidePulls(seed=None)
    if type(flybys) is not dict or set(flybys) != {'seed'}:
        raise ValueError(
            'physics.outside_pulls.flybys must be null, or an object with field seed. '
            'Null leaves stellar flybys off. A seed makes the encounter sequence reproducible.')
    seed = flybys['seed']
    if type(seed) is not int or isinstance(seed, bool) or seed < 0:
        raise ValueError('physics.outside_pulls.flybys.seed must be a nonnegative integer.')
    return OutsidePulls(seed=seed)


def run_document(policy, run_index, flybys):
    """Per-run record. Flybys are omitted from the physics claim when they were not drawn."""
    return {
        'galactic_tide': TIDE_MODEL,
        'flyby_model': None if policy.seed is None else FLYBY_MODEL,
        'seed': policy.seed,
        'run_index': int(run_index),
        'flybys': None if policy.seed is None else [flyby.document() for flyby in flybys],
    }


def playback_report(policy, runs):
    """Physics-section report. Counts appear only when flybys were actually drawn."""
    if policy is None:
        return {
            'enabled': False,
            'note': 'Outside pulls are off, so no galactic tide and no field star act.',
        }
    report = {
        'enabled': True,
        'galactic_tide': TIDE_MODEL,
        'regime': 'very wide orbits near the Sun',
        'flybys': policy.seed is not None,
        'note': (
            'Galactic tide heisler-tremaine-1986 is on. '
            'It is the linear solar-neighborhood field and is meant for very wide orbits.'
        ),
    }
    if policy.seed is None:
        report['note'] += ' Stellar flybys are off, so no field star was drawn.'
        return report
    sequences = []
    for run in runs:
        recorded = run.get('outside_pulls', {}).get('flybys')
        sequences.append(() if recorded is None else recorded)
    summary = summarize_flybys(sequences)
    report.update(summary)
    report['seed'] = policy.seed
    report['note'] = (
        'Galactic tide heisler-tremaine-1986 is on, for very wide orbits near the Sun. ' + summary['note']
    )
    return report
