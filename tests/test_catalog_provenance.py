import numpy as np
import pytest
from simulations.star_system.data import CATALOG, STARS, PLANETS
from simulations.star_system.model import InitialBody, InitialSetup
from simulations.star_system.initialization import build_system

def test_catalog_categories_have_traceable_properties():
    assert set(STARS) | set(PLANETS) <= set(CATALOG)
    assert {'star', 'planet', 'moon', 'asteroid'} <= {s.kind for s in CATALOG.values()}
    assert {'alpha_cen_a','alpha_cen_b','sirius_b','51_peg_b','hd_209458_b','moon','io','europa','ganymede','callisto','titan','vesta','eros','bennu'} <= set(CATALOG)
    for spec in CATALOG.values():
        for field in ('mass','radius') + (('temperature',) if spec.kind == 'star' else ()):
            value = getattr(spec, field)
            if value is not None:
                assert np.isfinite(value) and value > 0
            source = spec.sources[field]
            assert source.url.startswith('https://')
            assert source.version and source.retrieved and source.original_units and source.conversion
            assert source.status
    with pytest.raises(TypeError):
        CATALOG['sun'].sources['mass'] = None

def test_unknown_property_is_not_silently_estimated():
    peg = CATALOG['51_peg_b']
    assert peg.radius is None
    assert peg.sources['radius'].status == 'unknown'
    # Unknown radius does not block gravity; contact is not modeled.
    build_system(InitialSetup('Unknown radius', (InitialBody('peg', peg, (0,0), (0,0)),), .0001))
