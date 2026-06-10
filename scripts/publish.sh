#!/usr/bin/env bash
set -euo pipefail

repository="pypi"
skip_tests=0
skip_build=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repository)
      repository="$2"
      shift 2
      ;;
    --skip-tests)
      skip_tests=1
      shift
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$repository" != "pypi" && "$repository" != "testpypi" ]]; then
  echo "--repository must be pypi or testpypi" >&2
  exit 2
fi

if [[ "$skip_tests" == "0" ]]; then
  python3 -m pytest -q
fi

if [[ "$skip_build" == "0" ]]; then
  rm -rf dist
  python3 -m build
fi

python3 -m twine check dist/*

if [[ "$repository" == "pypi" ]]; then
  : "${PYPI_API_TOKEN:?Set PYPI_API_TOKEN}"
  python3 -m twine upload --repository pypi -u __token__ -p "$PYPI_API_TOKEN" dist/*
else
  : "${TEST_PYPI_API_TOKEN:?Set TEST_PYPI_API_TOKEN}"
  python3 -m twine upload --repository testpypi -u __token__ -p "$TEST_PYPI_API_TOKEN" dist/*
fi
