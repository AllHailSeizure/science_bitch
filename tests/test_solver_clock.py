"""Yoshida steps must advance a finite clock before committing state."""
import numpy as np
import pytest

from simulations.star_system.core import System


def _single(velocity, dt, time, position=(0., 0.)):
    return System(np.array([1.]), np.array([position], dtype=float), np.array([velocity], dtype=float), dt, time)


def test_underflowing_step_raises_and_preserves_position_and_time():
    system = _single((1., 0.), 1e-17, 1.0)
    position = system.r.copy()
    velocity = system.v.copy()
    masses = system.masses.copy()
    with pytest.raises(ValueError):
        system.step()
    assert system.time == 1.0
    assert system.dt == 1e-17
    np.testing.assert_array_equal(system.r, position)
    np.testing.assert_array_equal(system.v, velocity)
    np.testing.assert_array_equal(system.masses, masses)


def test_overflowing_stationary_step_raises_and_preserves_finite_time():
    system = _single((0., 0.), 1e308, 1e308)
    position = system.r.copy()
    with pytest.raises(ValueError):
        system.step()
    assert system.time == 1e308
    assert np.isfinite(system.time)
    np.testing.assert_array_equal(system.r, position)


def test_ordinary_single_body_drift_advances_clock_and_position():
    system = _single((1., 0.), 0.01, 0.0)
    system.step()
    assert system.time == 0.01
    assert system.r[0, 0] > 0
