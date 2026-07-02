"""Enable ``python -m privacy_kit`` as an alias for the console script.

Used as the fallback command for Cursor hooks when the ``privacy-kit`` script
isn't found on PATH (see ``gateway.cli._cursor_hook_command``).
"""

from privacy_kit.cli import main

if __name__ == "__main__":
    main()
