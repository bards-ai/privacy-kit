# Release scripts

Build and publish from a clean working tree.

```bash
python3 -m pip install build twine
scripts/publish.sh --repository pypi
```

Required token:

```bash
export PYPI_API_TOKEN="pypi-..."
```

For TestPyPI:

```bash
export TEST_PYPI_API_TOKEN="pypi-..."
scripts/publish.sh --repository testpypi
```

Do not commit tokens to this repository. For organization releases, prefer GitHub Actions with PyPI Trusted Publishing later.
