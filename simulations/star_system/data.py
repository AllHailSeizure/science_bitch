"""Rounded physical catalog; sources and units in README.md."""
from dataclasses import dataclass
import numpy as np

SOLAR_KG = 1.98847e30
AU_KM = 149597870.7

@dataclass(frozen=True)
class Star:
    name: str
    mass: float  # solar masses
    radius: float  # solar radii
    temperature: float  # K

STARS = {
    'proxima': Star('Proxima Centauri', .120, .141, 3050),
    'sun': Star('Sun', 1., 1., 5772),
    'sirius': Star('Sirius A', 2.063, 1.7144, 9845),
    'rigel': Star('Rigel', 21., 78.9, 12100),
}

@dataclass(frozen=True)
class Planet:
    name: str
    mass: float  # solar masses
    radius: float  # km
    distance: float  # AU (heliocentric, including dwarf planets)
    color: str

# NASA NSSDC planetary fact sheet (mass in 10^24 kg, radius in km).
_ROWS = [
    ('Mercury', .330, 2439.7, .3871, '#b8aaa0'),
    ('Venus', 4.87, 6051.8, .7233, '#e9c78f'),
    ('Earth', 5.97, 6371, 1., '#51a9ef'),
    ('Mars', .642, 3389.5, 1.524, '#e47750'),
    ('Jupiter', 1898, 69911, 5.204, '#d5aa85'),
    ('Saturn', 568, 58232, 9.573, '#ead49a'),
    ('Uranus', 86.8, 25362, 19.165, '#8bdbdf'),
    ('Neptune', 102, 24622, 30.178, '#637aff'),
    ('Pluto', .0130, 1188.3, 39.482, '#d3bbaa'),
    ('Ceres', .000938, 469.7, 2.767, '#aaa9a5'),
]
PLANETS = {n.lower(): Planet(n, m*1e24/SOLAR_KG, r, a, c) for n,m,r,a,c in _ROWS}


def blackbody_rgb(temperature):
    """Approximate Planckian-locus sRGB, Tanner Helland's temperature fit.

    http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    Display approximation, normalized brightness; not a stellar spectrum.
    """
    t = np.clip(temperature, 1000, 40000) / 100
    r = 255 if t <= 66 else 329.698727446 * (t-60)**-.1332047592
    g = 99.4708025861*np.log(t)-161.1195681661 if t <= 66 else 288.1221695283*(t-60)**-.0755148492
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5177312231*np.log(t-10)-305.0447927307)
    return np.clip([r,g,b], 0, 255)/255


def star_size(star):
    """Marker diameter in points, compressed monotonically from physical radius."""
    return 12 + 16*np.log10(1 + 8*star.radius)


def _load_catalog():
    import json
    from pathlib import Path
    from .model import BodySpec, PropertySource
    records = json.loads(Path(__file__).with_name('catalog-sources.json').read_text())['bodies']
    result = {}
    for key, record in records.items():
        record['sources'] = {field: PropertySource(**value) for field, value in record['sources'].items()}
        result[key] = BodySpec(**record)
    return result

# General definitions are separate from the legacy circular CLI selectors.
CATALOG = _load_catalog()
