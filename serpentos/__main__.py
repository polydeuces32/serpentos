"""Entry point.

    python -m serpentos            # terminal UI
    python -m serpentos bot ...    # headless self-running agent
"""

from __future__ import annotations

import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] == "bot":
        from .bot import main as bot_main

        return bot_main(args[1:])

    if args and args[0] in ("-h", "--help"):
        print(__doc__.strip())
        print("\nRun 'python -m serpentos bot --help' for the agent options.")
        return 0

    from .serpentos import main as ui_main

    return ui_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
