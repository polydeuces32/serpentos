"""SerpentOS command line entry point.

    serpentos                 play (same as `serpentos run`)
    serpentos run             play in the terminal
    serpentos bot             train headlessly, no terminal needed
    serpentos bench           score a policy on the frozen benchmark

Also reachable as `python -m serpentos ...` without installing.
"""

from __future__ import annotations

import sys
from typing import List, Optional

USAGE = """SerpentOS — terminal snake with a Q-learning agent.

usage: serpentos [command] [options]

commands:
  run              Play in the terminal. The default when no command is given.
  bot              Run the agent headlessly: train, evaluate, export a policy.
  bench            Score the current policy on the frozen benchmark.
  help             Show this message.

examples:
  serpentos run                                  play, with the AI menus
  serpentos run --no-color                       plain monochrome
  serpentos bot --episodes 20000 --preset BOLD   train for a while
  serpentos bot --forever                        train until stopped
  serpentos bench                                score what it has learned

Every command takes --help. State lives in ~/.serpentos (override --data-dir).

The policy runtime has no CLI: it is a library. See docs/RUNTIME.md, or
  python -c "import serpentos; help(serpentos)"
"""

RUN_USAGE = """usage: serpentos run [-h] [--data-dir DATA_DIR] [--no-color]

Play SerpentOS in the terminal.

options:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   state directory (default: ~/.serpentos)
  --no-color            disable terminal colours
"""


def _version() -> str:
    from . import __version__

    return f"serpentos {__version__}"


def _wants_help(args: List[str]) -> bool:
    return "-h" in args or "--help" in args


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    # A leading flag means "no command", so `serpentos --data-dir X` still plays.
    command = args[0] if args and not args[0].startswith("-") else None
    rest = args[1:] if command is not None else args

    if command is None and args and args[0] in ("-h", "--help"):
        print(USAGE, end="")
        return 0
    if command is None and args and args[0] in ("-V", "--version"):
        print(_version())
        return 0

    if command == "help":
        print(USAGE, end="")
        return 0
    if command == "version":
        print(_version())
        return 0

    if command == "bot":
        from .bot import main as bot_main

        return bot_main(rest)

    if command == "bench":
        from .bot import main as bot_main

        return bot_main(["--bench", *rest])

    if command in (None, "run", "play"):
        # Help must work even on stock Windows Python where curses is absent.
        # Do not import the terminal UI merely to render its argument help.
        if _wants_help(rest):
            print(RUN_USAGE, end="")
            return 0

        from .serpentos import main as ui_main

        return ui_main(rest)

    sys.stderr.write(f"serpentos: unknown command {command!r}\n\n{USAGE}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
