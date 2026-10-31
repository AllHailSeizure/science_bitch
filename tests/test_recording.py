from dataclasses import replace

import numpy as np
import pytest

from simulations.star_system.initialization import build_system
from simulations.star_system.scenarios import get_scenario
from simulations.star_system.recording import Calculation, CalculationRequest


@pytest.mark.parametrize('duration,stride,budget', [(0,1,1024),(-1,1,1024),(float('nan'),1,1024),(float('inf'),1,1024),(1,0,1024),(1,1.5,1024),(1,True,1024),(1,1,0)])
def test_invalid_requests(duration, stride, budget):
    with pytest.raises(ValueError):
        CalculationRequest(duration, stride, budget)


def test_calculation_matches_live_and_stride_does_not_change_physics():
    setup = get_scenario('sun-earth-jupiter')
    dt = setup.dt
    live = build_system(setup)
    results = []
    for stride in (1, 4):
        calc = Calculation(setup, CalculationRequest(dt*10.25, stride))
        assert calc.total_steps == 11 and calc.resolved_end_time == dt*11
        assert calc.advance(3) == 3 and calc.completed_steps == 3
        while not calc.done:
            calc.advance(3)
        results.append(calc.result)
        assert calc.result.status == 'complete'
        assert calc.result.source_name == setup.name
        assert calc.result.requested_end_time == dt*10.25
        assert calc.result.actual_end_time == calc.result.frames[-1].time
    for frame in results[0].frames[1:]:
        live.system.step()
        np.testing.assert_allclose(frame.r, live.system.r, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(frame.v, live.system.v, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(results[0].frames[-1].r, results[1].frames[-1].r)
    np.testing.assert_allclose([f.time for f in results[1].frames], np.array([0,4,8,11])*dt)
    original = results[0].frames[-1].r.copy()
    live.system.r[:] = 0
    np.testing.assert_array_equal(results[0].frames[-1].r, original)
    with pytest.raises(ValueError):
        results[0].frames[-1].r.setflags(write=True)


def test_cancel_keeps_last_off_stride_step_and_preflight_budget():
    setup = get_scenario('sun-earth-jupiter')
    calc = Calculation(setup, CalculationRequest(setup.dt*20, 8))
    calc.advance(3)
    calc.cancel()
    assert calc.result.status == 'cancelled' and calc.result.reason
    assert len(calc.result.frames) == 2 and calc.result.frames[-1].time == calc.prepared.system.time
    assert calc.advance(50) == 0
    with pytest.raises(ValueError, match='stride'):
        Calculation(setup, CalculationRequest(1, 1, 100))


def test_transactional_halt_retains_final_valid_state(monkeypatch):
    setup = get_scenario('sun-earth-jupiter')
    calc = Calculation(setup, CalculationRequest(setup.dt*20, 8))
    calc.advance(3)
    expected = calc.prepared.system.r.copy()
    # A real unresolved step, rejected by the same solver used live.
    calc.prepared.system.v[1] = (1e8, 0)
    calc.advance(1)
    assert calc.result.status == 'halted' and 'unresolved' in calc.result.reason
    assert calc.result.frames[-1].time == calc.prepared.system.time
    np.testing.assert_array_equal(calc.result.frames[-1].r, expected)


def test_actual_budget_enforced_before_accepting_new_event_state():
    from simulations.star_system.initialization import IntegrationEvent
    from simulations.star_system.recording import frame_storage_bytes
    setup = get_scenario('sun-earth-jupiter')
    calc = Calculation(setup, CalculationRequest(setup.dt*4, 10))
    calc.advance(1)
    previous = calc.prepared.system.r.copy()
    budget = calc.estimated_bytes*2
    calc.request = CalculationRequest(setup.dt*4, 10, budget)
    event = IntegrationEvent(calc.prepared.system.time, 'synthetic', 'earth', setup.dt, setup.dt, 'x'*budget)
    calc.prepared = replace(calc.prepared, events=(event,))
    calc.advance(1)
    assert calc.result.status == 'halted' and 'budget' in calc.result.reason
    assert calc.completed_steps == 1 and not calc.result.frames[-1].events
    np.testing.assert_array_equal(calc.prepared.system.r, previous)
    assert sum(frame_storage_bytes(f) for f in calc.result.frames) <= budget


def test_event_boundaries_are_retained_between_output_strides():
    from simulations.star_system.initialization import IntegrationEvent
    setup = get_scenario('sun-earth-jupiter')
    calc = Calculation(setup, CalculationRequest(setup.dt*4, 10))
    calc.advance(1)
    event = IntegrationEvent(calc.prepared.system.time, 'synthetic', 'earth', setup.dt, setup.dt, 'Membership boundary')
    calc.prepared = replace(calc.prepared, events=(event,))
    calc.advance(1)
    calc.cancel()
    np.testing.assert_allclose([frame.time for frame in calc.result.frames], np.array([0,1,2])*setup.dt)
    assert not calc.result.frames[1].events and calc.result.frames[2].events == (event,)


def test_wall_budget_yields_after_a_completed_whole_step(monkeypatch):
    setup = get_scenario('sun-earth-jupiter')
    calc = Calculation(setup, CalculationRequest(setup.dt*100, 10))
    ticks = [0.]
    def now():
        ticks[0] += .01
        return ticks[0]
    monkeypatch.setattr('simulations.star_system.recording.time.perf_counter', now)
    assert calc.advance(100, budget_seconds=.006) == 1
    assert calc.completed_steps == 1 and not calc.done
    calc.cancel()
    assert calc.result.frames[-1].time == calc.prepared.system.time
