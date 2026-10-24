# Two-body orbit

A star of mass M and a planet of mass m attract each other with Newton's inverse-square force. Both bodies move. In the barycentric frame each travels on an ellipse with the barycenter at one focus. The relative orbit has semi-major axis `a`, eccentricity `e`, and

    T² = 4 π² a³ / μ,    μ = G (M + m)

The two-body energy of a bound ellipse is −G M m / (2 a). The angular momentum and the center of mass of an isolated pair are constant.

## Integrator

Yoshida's fourth-order symplectic method (Yoshida, Physics Letters A 150, 262–268, 1990): three velocity-Verlet steps with weights `w1`, `w0`, `w1`, where `w0` is negative and the weights sum to 1. Velocity Verlet alone conserves energy and angular momentum, but its phase error makes a Keplerian ellipse precess. The fourth-order composition keeps the orbit closed over many revolutions.

## Run

From the repository root:

```bash
python -m simulations.orbit
python -m simulations.orbit --eccentricity 0.7 --orbits 8
python -m simulations.orbit --star-mass 1 --planet-mass 0.05 --semi-major 1 --g 1
```

Defaults are `G = 1`, `M = 1`, `m = 0.2`, `a = 1`, `e = 0.55`, four orbits, 600 steps per orbit. The planet is heavy enough that the star's ellipse is visible; both still share the barycenter as a focus. Outputs:

- `output/orbit/orbit.png` — both ellipses, with energy and angular-momentum drift
- `output/orbit/animation.gif` — the two bodies tracing those ellipses (`--animation-format mp4` when ffmpeg is available)

Pass `--show` to open the figure when a display is available. Units are whatever units you chose for `G`, the masses, and `a`.

## Checks

`pytest` integrates both bodies (masses 1 and 0.3, `G = 1.7`). Conservation and closure use 1000 steps per orbit for 20 orbits. Kepler's law uses 600 steps per orbit for 8 orbits at `a = 0.5, 1, 1.6, 2.4`.

| Check | Tolerance | Representative result |
| --- | --- | --- |
| max \|E − E0\| / \|E0\| | 1×10⁻⁶ | 6.8×10⁻⁸ |
| max \|L − L0\| / \|L0\| | 1×10⁻⁹ | 1.8×10⁻¹⁴ |
| periapsis displacement / a | 1×10⁻⁴ | 5.9×10⁻⁶ |
| max \|r − a(1−e²)/(1+e cos θ)\| / a | 2×10⁻⁴ | 1.3×10⁻⁵ |
| \|T²/a³ − 4π²/μ\| / (4π²/μ) | 1×10⁻⁶ | 5.0×10⁻⁸ |

The initial state is also checked against E = −G M m / (2 a) and against the angular momentum of the ellipse, each to a relative tolerance of 1×10⁻¹². The center of mass stays put to better than 1×10⁻⁸ of `a`.
