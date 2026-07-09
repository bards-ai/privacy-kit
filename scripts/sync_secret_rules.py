#!/usr/bin/env python3
"""Compare the vendored secret rules against the upstream gitleaks ruleset.

Dev tool, not shipped. Fetches gitleaks' ``config/gitleaks.toml`` for a given
release tag and reports, per vendored rule that carries an upstream id, whether
the upstream regex changed since we vendored it — so refreshing
``privacy_kit/core/secret_rules.py`` is a reviewed, mechanical diff instead of
guesswork. Rules with the ``privacy-kit-`` prefix are ours and are skipped.

Usage:
    uv run python scripts/sync_secret_rules.py [--tag v8.30.0]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from privacy_kit.core.secret_rules import SECRET_RULES

UPSTREAM = "https://raw.githubusercontent.com/gitleaks/gitleaks/{tag}/config/gitleaks.toml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="master", help="gitleaks git tag or branch")
    args = parser.parse_args()

    url = UPSTREAM.format(tag=args.tag)
    print(f"Fetching {url} ...")
    with urllib.request.urlopen(url, timeout=30) as response:
        upstream = tomllib.loads(response.read().decode("utf-8"))
    upstream_rules = {rule["id"]: rule for rule in upstream.get("rules", [])}
    print(f"Upstream has {len(upstream_rules)} rules.\n")

    drifted = 0
    for rule in SECRET_RULES:
        if rule.rule_id.startswith("privacy-kit-"):
            continue
        up = upstream_rules.get(rule.rule_id)
        if up is None:
            print(f"[GONE]    {rule.rule_id}: no longer in upstream — check rename")
            drifted += 1
            continue
        ours = rule.pattern.pattern
        theirs = up.get("regex", "")
        if theirs and theirs not in ours and ours not in theirs:
            print(f"[DRIFT]   {rule.rule_id}:")
            print(f"    vendored: {ours}")
            print(f"    upstream: {theirs}")
            drifted += 1
        else:
            print(f"[OK]      {rule.rule_id}")

    print(f"\n{drifted} rule(s) need review." if drifted else "\nAll vendored rules current.")
    return 1 if drifted else 0


if __name__ == "__main__":
    raise SystemExit(main())
