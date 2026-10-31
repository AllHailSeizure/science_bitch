"""Preset factories return new immutable setups; editors never own the registry."""
from dataclasses import replace
from types import MappingProxyType

from .initialization import legacy_setup

COMPATIBILITY = 'sun-earth-jupiter'
DESCRIPTIONS = MappingProxyType({COMPATIBILITY:
    'Sun, Earth and Jupiter with prograde circular two-body velocity guesses, '
    'separated initial angles and a barycentric shift. Illustrative, not an ephemeris or stability guarantee.'})


def get_scenario(key):
    if key != COMPATIBILITY:
        raise ValueError(f'Unknown scenario: {key}.')
    return replace(legacy_setup(), preset_id=key,
                   provenance={'origin_preset': key, 'description': DESCRIPTIONS[key]})
