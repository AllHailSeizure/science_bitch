# Harmonic oscillator

A mass on a spring:

    m x'' + c x' + k x = F0 cos(ω t)

`c` is the damping coefficient and `F0` is the amplitude of a sinusoidal driving force. The undamped, unforced motion has period

    T = 2 π √(m / k)

and conserves the mechanical energy ½ m v² + ½ k x².

## Integrator

Velocity Verlet advances the spring and the drive. Linear friction is split off and applied with the exact half-step `v → v exp(−c Δt / 2m)` on either side of the Verlet step (Strang splitting). With friction and the drive turned off, the step is symplectic, so the energy error oscillates instead of drifting. The step is second-order.

The closed form used for the comparison covers underdamped, critical, and overdamped motion, a non-resonant drive, and the secular solution of an undamped oscillator driven exactly at `ω0`.

## Run

From the repository root:

```bash
python -m simulations.harmonic
python -m simulations.harmonic --compare
python -m simulations.harmonic --regime underdamped
python -m simulations.harmonic --regime critical
python -m simulations.harmonic --regime overdamped
python -m simulations.harmonic --regime resonance
python -m simulations.harmonic --mass 2 --spring 8 --cycles 10
```

The default regime is undamped, with `m = 1`, `k = 1`, `x(0) = 1`, `v(0) = 0`. Outputs:

- `output/harmonic/displacement.png` — numerical samples on the analytic curve, and the energy
- `output/harmonic/animation.gif` — the mass and the spring (`--animation-format mp4` if ffmpeg is installed)
- `output/harmonic/regimes.png` — written with `--compare`

Presets, expressed with the damping ratio ζ = c / (2 √(k m)):

| Regime | ζ | Drive |
| --- | --- | --- |
| `undamped` | 0 | none |
| `underdamped` | 0.1 | none |
| `critical` | 1 | none |
| `overdamped` | 2 | none |
| `resonance` | 0.05 | F0 = 0.4 at ω = ω0, released from rest at x = 0 |

`--damping`, `--drive-amplitude`, `--drive-frequency`, `--x0`, and `--v0` override the preset. `--steps-per-period` defaults to 800. Pass `--show` to open the figures when a display is available.

## Checks

`pytest` asserts the following. Position and energy use 2000 steps per undamped period for 20 periods. The period check uses 1000 steps per period for 30 periods, with `m = 2` and `k = 8`.

| Check | Tolerance | Representative result |
| --- | --- | --- |
| Undamped max \|x − analytic\| | 2×10⁻⁴ | 4.4×10⁻⁵ |
| Underdamped, critical, overdamped, driven, and resonant max \|x − analytic\| | 2×10⁻⁴ | 1×10⁻⁶ to 1×10⁻⁵ |
| Period against 2π√(m/k) | relative 1×10⁻⁵ | 1.6×10⁻⁶ |
| Undamped max \|E − E0\| / \|E0\| | 1×10⁻⁵ | 2.3×10⁻⁶ |

The analytic formula is also checked by substituting it back into the differential equation with finite differences (residual below 1×10⁻⁴; a wrong formula would be O(1)).
