"""Damped, driven harmonic oscillator.

The displacement x of a mass on a spring obeys

    m x'' + c x' + k x = F0 cos(omega t).

The spring and the driving force are advanced with velocity Verlet. Linear
friction is split off and applied with the exact map v -> v exp(-c dt / 2m)
on either side of the Verlet step (Strang splitting). With the friction and
the drive turned off, the step is the symplectic velocity-Verlet method, so
the mechanical energy stays bounded instead of drifting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Treat damping this close to critical as exactly critical. The closed form
# changes at zeta = 1, and a floating-point value of 2*sqrt(k*m) is not
# always bit-identical to that boundary.
_CRITICAL_ZETA_TOLERANCE = 1e-9
# Undriven oscillator whose drive frequency matches omega0 to this relative
# tolerance uses the resonant secular solution. The ordinary particular
# solution divides by zero in that case.
_RESONANCE_FREQUENCY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class OscillatorParams:
    """Parameters and initial conditions for one oscillator.

    ``spring`` is the stiffness k. ``damping`` is the coefficient c in
    c x', not the damping ratio. ``drive_amplitude`` is the force F0.
    """

    mass: float
    spring: float
    damping: float = 0.0
    drive_amplitude: float = 0.0
    drive_frequency: float = 0.0
    x0: float = 1.0
    v0: float = 0.0

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.spring <= 0.0:
            raise ValueError("spring constant must be positive")
        if self.damping < 0.0:
            raise ValueError("damping coefficient must be non-negative")


def natural_frequency(mass: float, spring: float) -> float:
    """omega0 = sqrt(k / m)."""
    return float(np.sqrt(spring / mass))


def natural_period(mass: float, spring: float) -> float:
    """Undamped period 2 pi sqrt(m / k)."""
    return float(2.0 * np.pi * np.sqrt(mass / spring))


def mechanical_energy(x: np.ndarray, v: np.ndarray, mass: float, spring: float) -> np.ndarray:
    """Energy 1/2 m v^2 + 1/2 k x^2. This is conserved only when c = 0 and F0 = 0."""
    displacement = np.asarray(x, dtype=np.float64)
    speed = np.asarray(v, dtype=np.float64)
    return 0.5 * mass * speed**2 + 0.5 * spring * displacement**2


def estimate_period(t: np.ndarray, x: np.ndarray) -> float:
    """Period from successive same-direction zero crossings.

    Linear interpolation locates each crossing. One full period is the gap
    between every second crossing, which does not depend on the phase.
    """
    times = np.asarray(t, dtype=np.float64)
    samples = np.asarray(x, dtype=np.float64)
    if times.shape != samples.shape or times.ndim != 1:
        raise ValueError("t and x must be one-dimensional arrays of the same length")
    crossings: list[float] = []
    for i in range(times.size - 1):
        left = samples[i]
        right = samples[i + 1]
        if left == 0.0:
            crossings.append(float(times[i]))
        elif left * right < 0.0:
            fraction = -left / (right - left)
            crossings.append(float(times[i] + fraction * (times[i + 1] - times[i])))
    if len(crossings) < 4:
        raise ValueError("need at least two full periods of zero crossings")
    marks = np.asarray(crossings)
    full_periods = marks[2:] - marks[:-2]
    return float(np.mean(full_periods))


def analytic_position(t: np.ndarray, params: OscillatorParams) -> np.ndarray:
    """Closed-form displacement, including damping and a sinusoidal drive."""
    times = np.asarray(t, dtype=np.float64)
    scalar = times.ndim == 0
    times = np.atleast_1d(times)

    mass = float(params.mass)
    spring = float(params.spring)
    omega0 = np.sqrt(spring / mass)
    gamma = float(params.damping) / mass
    force = float(params.drive_amplitude) / mass
    omega = float(params.drive_frequency)

    particular, x_particular_0, v_particular_0 = _particular(times, omega0, gamma, force, omega)
    homogeneous = _homogeneous(
        times,
        omega0,
        gamma,
        float(params.x0) - x_particular_0,
        float(params.v0) - v_particular_0,
    )
    position = homogeneous + particular
    if scalar:
        return position[0]
    return position


def integrate(params: OscillatorParams, t_end: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the oscillator. Returns ``t``, ``x``, ``v``.

    ``dt`` is adjusted slightly so that the last sample lands on ``t_end``.
    """
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    n_steps = max(1, int(round(t_end / dt)))
    step = t_end / n_steps
    # Arithmetic sequence, so the drive is sampled at the same instants the
    # Verlet step lands on. linspace is not always an exact multiple of dt.
    times = np.arange(n_steps + 1, dtype=np.float64) * step
    position = np.empty(n_steps + 1, dtype=np.float64)
    velocity = np.empty(n_steps + 1, dtype=np.float64)

    mass = float(params.mass)
    omega0_sq = float(params.spring) / mass
    gamma = float(params.damping) / mass
    force = float(params.drive_amplitude) / mass
    omega = float(params.drive_frequency)
    x = float(params.x0)
    v = float(params.v0)
    position[0] = x
    velocity[0] = v
    friction = np.exp(-0.5 * gamma * step)

    for i in range(n_steps):
        # Exact half-step of linear friction, then velocity Verlet for the
        # spring and the drive, then the other half-step of friction.
        v *= friction
        acceleration = -omega0_sq * x + force * np.cos(omega * times[i])
        v += 0.5 * step * acceleration
        x += step * v
        acceleration = -omega0_sq * x + force * np.cos(omega * times[i + 1])
        v += 0.5 * step * acceleration
        v *= friction
        position[i + 1] = x
        velocity[i + 1] = v

    return times, position, velocity


def diagnostics(params: OscillatorParams, t: np.ndarray, x: np.ndarray, v: np.ndarray) -> dict[str, float | None]:
    """Compare a numerical trajectory with the closed form and with energy conservation."""
    expected = analytic_position(t, params)
    position_error = float(np.max(np.abs(np.asarray(x) - expected)))
    energy = mechanical_energy(x, v, params.mass, params.spring)
    reference = float(energy[0])
    if abs(reference) > 1e-15:
        energy_drift = float(np.max(np.abs(energy - reference)) / abs(reference))
    else:
        energy_drift = float(np.max(np.abs(energy - reference)))

    period: float | None = None
    period_error: float | None = None
    analytic = natural_period(params.mass, params.spring)
    undamped_free = params.damping == 0.0 and params.drive_amplitude == 0.0
    if undamped_free:
        try:
            period = estimate_period(t, x)
        except ValueError:
            # A very short run may not contain two full periods of crossings.
            period = None
        else:
            period_error = abs(period - analytic) / analytic

    return {
        "position_error": position_error,
        "energy_drift": energy_drift,
        "period": period,
        "period_error": period_error,
        "analytic_period": analytic,
    }


def _particular(
    t: np.ndarray,
    omega0: float,
    gamma: float,
    force: float,
    omega: float,
) -> tuple[np.ndarray, float, float]:
    """Particular solution and its position and velocity at t = 0."""
    if force == 0.0:
        return np.zeros_like(t), 0.0, 0.0

    resonant = gamma == 0.0 and abs(omega - omega0) <= _RESONANCE_FREQUENCY_TOLERANCE * max(omega0, 1.0)
    if resonant:
        # x'' + omega^2 x = force * cos(omega t) has the secular solution
        # (force / (2 omega)) t sin(omega t). Its initial value and slope are zero.
        if omega == 0.0:
            raise ValueError("resonant drive at zero frequency is not a harmonic oscillator")
        position = (force / (2.0 * omega)) * t * np.sin(omega * t)
        return position, 0.0, 0.0

    det = (omega0 * omega0 - omega * omega) ** 2 + (gamma * omega) ** 2
    cosine_coeff = (omega0 * omega0 - omega * omega) * force / det
    sine_coeff = (gamma * omega) * force / det
    position = cosine_coeff * np.cos(omega * t) + sine_coeff * np.sin(omega * t)
    return position, cosine_coeff, sine_coeff * omega


def _homogeneous(t: np.ndarray, omega0: float, gamma: float, x0: float, v0: float) -> np.ndarray:
    """Solution of the homogeneous oscillator matching x(0) and v(0)."""
    zeta = gamma / (2.0 * omega0)
    if gamma == 0.0:
        return x0 * np.cos(omega0 * t) + (v0 / omega0) * np.sin(omega0 * t)
    if abs(zeta - 1.0) <= _CRITICAL_ZETA_TOLERANCE:
        return (x0 + (v0 + omega0 * x0) * t) * np.exp(-omega0 * t)
    if zeta < 1.0:
        omega_d = omega0 * np.sqrt(1.0 - zeta * zeta)
        sine_coeff = (v0 + zeta * omega0 * x0) / omega_d
        decay = np.exp(-zeta * omega0 * t)
        return decay * (x0 * np.cos(omega_d * t) + sine_coeff * np.sin(omega_d * t))
    discriminant = omega0 * np.sqrt(zeta * zeta - 1.0)
    root_slow = -zeta * omega0 + discriminant
    root_fast = -zeta * omega0 - discriminant
    coeff_slow = (v0 - root_fast * x0) / (root_slow - root_fast)
    coeff_fast = x0 - coeff_slow
    return coeff_slow * np.exp(root_slow * t) + coeff_fast * np.exp(root_fast * t)
