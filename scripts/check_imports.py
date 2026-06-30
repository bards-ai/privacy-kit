"""Import every Python module under ``src/privacy_kit`` and report failures.

This is a lightweight smoke test for import-time errors. It walks the package
tree, imports each module by its package name, and exits non-zero if any import
fails.

Usage:
    python3 scripts/check_imports.py
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE_ROOT = SRC / "privacy_kit"


def iter_modules() -> list[str]:
    modules: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(SRC).with_suffix("")
        modules.append(".".join(rel.parts))
    return modules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    sys.path.insert(0, str(SRC))

    failures: list[str] = []
    for module_name in iter_modules():
        try:
            importlib.import_module(module_name)
            print(f"ok {module_name}")
        except Exception:
            failures.append(module_name)
            print(f"failed {module_name}", file=sys.stderr)
            traceback.print_exc()

    if failures:
        print(f"\n{len(failures)} import(s) failed", file=sys.stderr)
        return 1

    print(f"\nImported {len(iter_modules())} module(s) successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
