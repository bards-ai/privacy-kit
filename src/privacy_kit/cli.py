"""Console-script entry point.

The CLI drives the gateway, which needs the optional ``gateway`` extra. The
script itself installs with the base package, so fail with a clear hint rather
than a bare ImportError when the extra is missing.
"""

from __future__ import annotations

import sys


def main() -> None:
    try:
        from privacy_kit.gateway.cli import app
    except ImportError:
        sys.stderr.write(
            "The privacy-kit CLI drives the gateway, which needs optional "
            "dependencies.\nInstall them with: pip install 'privacy-kit[gateway]'\n"
        )
        raise SystemExit(1) from None
    app()


if __name__ == "__main__":
    main()
