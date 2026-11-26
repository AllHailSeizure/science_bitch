"""No-display physics checks; all asserted tolerances documented here and README."""
import numpy as np
import pytest
from simulations.star_system.core import System, make_system, period, G
from simulations.star_system.data import STARS, PLANETS, blackbody_rgb, star_size


def measure(star, planets, distances=None, orbits=4):
    system = make_system(star,planets,distances)
    a = distances if distances is not None else [PLANETS[p].distance for p in planets]
    target = min(period(star,p,d) for p,d in zip(planets,a))
    times, positions, invariants = [], [], []
    for _ in range(round(orbits*target/system.dt)+1):
        times.append(system.time)
        positions.append(system.r-system.r[0])
        invariants.append(system.invariants())
        system.step()
    return system,np.array(times),np.array(positions),invariants


@pytest.mark.parametrize('star', STARS)
@pytest.mark.parametrize('planet', PLANETS)
def test_single_period_and_kepler(star,planet):
    system,t,r,_ = measure(star,[planet],orbits=3)
    theta = np.unwrap(np.arctan2(r[:,1,1],r[:,1,0]))
    measured = np.interp(2*np.pi,theta,t)
    expected = period(star,planet,PLANETS[planet].distance)
    error = abs(measured/expected-1)
    kepler = abs(measured**2/PLANETS[planet].distance**3 * G*(STARS[star].mass+PLANETS[planet].mass)/(4*np.pi**2)-1)
    print(f'{star}/{planet}: period {error:.3e}, Kepler {kepler:.3e}')
    assert error < 1e-7
    assert kepler < 2e-7


def test_many_orbit_conservation():
    system,t,r,inv = measure('sun',['mercury','earth','jupiter'],orbits=30)
    e = np.array([x[0] for x in inv])
    p = np.array([x[1] for x in inv])
    angular_momenta = np.array([x[2] for x in inv])
    energy = np.max(abs(e/e[0]-1))
    angular = np.max(abs(angular_momenta/angular_momenta[0]-1))
    # Normalize P by sum of initial scalar momenta; P starts at zero.
    initial = make_system('sun',['mercury','earth','jupiter'])
    momentum = np.max(np.linalg.norm(p-p[0],axis=1))/np.sum(initial.masses*np.linalg.norm(initial.v,axis=1))
    print(f'30 Mercury orbits: E={energy:.3e}, L={angular:.3e}, P={momentum:.3e}')
    assert energy < 1e-7
    assert angular < 1e-11
    assert momentum < 1e-11


def test_perturbed_periods():
    system = make_system('sun',['earth','jupiter'])
    expected = [period('sun',p,PLANETS[p].distance) for p in ['earth','jupiter']]
    t=[]
    r=[]
    for _ in range(round(3*max(expected)/system.dt)+1):
        t.append(system.time)
        r.append(system.r-system.r[0])
        system.step()
    r=np.array(r)
    for i,target in enumerate(expected,1):
        theta=np.unwrap(np.arctan2(r[:,i,1],r[:,i,0]))
        measured=np.interp(theta[0]+2*np.pi,theta,t)
        error=abs(measured/target-1)
        print(f'perturbed body {i}: period error={error:.3e}')
        assert error < .005  # 0.5%: physical perturbations, not integration error.


def test_close_in_and_wide():
    s=make_system('proxima',['earth','neptune'],[.05,30])
    assert s.dt == pytest.approx(period('proxima','earth',.05)/600)
    e=s.invariants()[0]
    s.step(6000)
    assert abs(s.invariants()[0]/e-1) < 1e-7


def test_catalog():
    colors=np.array([blackbody_rgb(s.temperature) for s in STARS.values()])
    assert np.all(np.diff(colors[:,2]/colors[:,0]) > 0)
    assert all(s.mass>0 and s.radius>0 and s.temperature>0 for s in STARS.values())
    assert np.all(np.diff([star_size(s) for s in STARS.values()])>0)
    assert all(p.mass>0 and p.radius>0 and p.distance>0 for p in PLANETS.values())


def test_invalid_inputs():
    for distances in ([0.], [float('nan')], [.001]):
        with pytest.raises(ValueError):
            make_system('sun',['earth'],distances)
    with pytest.raises(ValueError):
        make_system('sun',['earth','earth'])


# JPL DE440 heliocentric GM (Park et al. 2021), quoted at
# https://ssd.jpl.nasa.gov/astro_par.html. Independent of core.G.
JPL_SOLAR_GM = 1.32712440041279419e20  # m^3 s^-2
AU_METERS = 149597870700.  # IAU 2012 astronomical unit
JULIAN_YEAR_SECONDS = 365.25*86400.
# Rounded SOLAR_KG (1.98847e30 kg) leaves model GM ~30.2 ppm above JPL.
# 1e-4 covers that rounding and fails a 1% change in G or a unit conversion.
SOLAR_SCALE_TOLERANCE = 1e-4
# e=0.3 test particle, one orbit, dt = period/1200. Measured error ~2e-8 AU.
ECCENTRIC_PHASE_TOLERANCE_AU = 1e-6


def kepler_position(mean_anomaly, semi_major, eccentricity):
    """Periapsis-on-+x coordinates from Kepler's equation M = E - e sin E."""
    mean = float(np.mod(mean_anomaly, 2*np.pi))
    eccentric = mean
    for _ in range(50):
        residual = eccentric - eccentricity*np.sin(eccentric) - mean
        step = residual/(1 - eccentricity*np.cos(eccentric))
        eccentric -= step
        if abs(step) < 1e-15:
            break
    else:
        raise AssertionError('Kepler equation did not converge')
    return np.array((
        semi_major*(np.cos(eccentric) - eccentricity),
        semi_major*np.sqrt(1 - eccentricity**2)*np.sin(eccentric),
    ))


def test_one_au_solar_acceleration_matches_jpl_gm():
    """Acceleration at 1 AU from one solar mass, against JPL's GM."""
    system = System(np.array([1., 0.]), np.array([[0., 0.], [1., 0.]]), np.zeros((2, 2)), 1.)
    acceleration = system.accelerations()[1]
    expected = -JPL_SOLAR_GM/AU_METERS**2*JULIAN_YEAR_SECONDS**2/AU_METERS
    error = np.linalg.norm(acceleration - (expected, 0.))/abs(expected)
    print(f'1 AU solar acceleration vs JPL GM: relative {error:.3e}')
    assert error < SOLAR_SCALE_TOLERANCE


def test_eccentric_orbit_phase_matches_keplers_equation():
    """Eccentric orbit phase matches the position from Kepler's equation."""
    eccentricity, semi_major = .3, 1.
    mu = G  # implemented mu; the JPL scale check hardcodes its own GM
    period_years = 2*np.pi*np.sqrt(semi_major**3/mu)
    periapsis = semi_major*(1 - eccentricity)
    speed = np.sqrt(mu*(1 + eccentricity)/periapsis)
    steps = 1200  # periapsis is faster than the circular period/600 cases
    system = System(
        np.array([1., 0.]),
        np.array([[0., 0.], [periapsis, 0.]]),
        np.array([[0., 0.], [0., speed]]),
        period_years/steps,
    )
    mean_motion = np.sqrt(mu/semi_major**3)
    worst = 0.
    for _ in range(steps):
        system.step()
        predicted = kepler_position(mean_motion*system.time, semi_major, eccentricity)
        worst = max(worst, np.linalg.norm(system.r[1] - system.r[0] - predicted))
    print(f'e={eccentricity} Kepler phase position error {worst:.3e} AU')
    assert worst < ECCENTRIC_PHASE_TOLERANCE_AU
