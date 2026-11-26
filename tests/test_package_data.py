"""Catalog data is part of the package, not a file next to the working directory."""
from simulations.star_system.data import _load_catalog


def test_catalog_loads_from_package_data_outside_the_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    catalog = _load_catalog()
    assert catalog['sun'].name == 'Sun'
    assert catalog['earth'].mass > 0
    assert not (tmp_path / 'catalog-sources.json').exists()
