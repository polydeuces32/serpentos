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
"""


def _version() -> str:
    from . import __version__

    return f"serpentos {__version__}"


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

    if command in ("help",):
        print(USAGE, end="")
        return 0
    if command in ("version",):
        print(_version())
        return 0

    if command == "bot":
        from .bot import main as bot_main

        return bot_main(rest)

    if command == "bench":
        from .bot import main as bot_main

        return bot_main(["--bench", *rest])

    if command in (None, "run", "play"):
        from .serpentos import main as ui_main

        return ui_main(rest)

    sys.stderr.write(f"serpentos: unknown command {command!r}\n\n{USAGE}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
