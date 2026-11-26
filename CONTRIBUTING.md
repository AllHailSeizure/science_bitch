# Contributing

Python 3.10 or newer. From the repository root:

```bash
pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
```

The `dev` extra adds pytest and ruff 0.17.3. `pip install -r requirements-dev.txt` is the same set. Tests use Matplotlib's Agg backend and need no display. On Linux, the Qt tests also need `libegl1` and `libopengl0`.

Ruff checks Python errors, undefined names, and ambiguous statements (`ruff.toml`). It does not format the code.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on pull requests, pushes to `master`, and manual dispatch.

Merging to `master` requires these four checks:

- `Lint`
- `Tests (ubuntu-latest, Python 3.10)`
- `Tests (ubuntu-latest, Python 3.14)`
- `Tests (windows-latest, Python 3.14)`

Pull requests must be up to date with `master`. History stays linear. Force-pushes are blocked, and the default branch cannot be deleted. Repository admins can bypass the ruleset.

A `Wheel` job builds the sdist and wheel and smoke-tests `sidereel-cli`. It is separate from the four checks above.

## Publishing

[`.github/workflows/release.yml`](.github/workflows/release.yml) publishes to PyPI on a `v*` tag. The tag has to match `version` in `pyproject.toml` (`v0.1.0` for `0.1.0`). It uses PyPI Trusted Publishing and the GitHub environment `pypi`. No API token is stored in the repo.

Model notes and tolerances are in [simulations/star_system/README.md](simulations/star_system/README.md).
