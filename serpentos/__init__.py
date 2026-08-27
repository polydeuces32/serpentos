"""SerpentOS — a terminal snake game with a Q-learning agent that can also run
itself headlessly.

The learning core (:mod:`serpentos.core`) has no curses dependency, so it
imports cleanly on any platform; the terminal UI lives in
:mod:`serpentos.serpentos` and the autonomous runner in :mod:`serpentos.bot`.
"""

__version__ = "0.2.0"
__all__ = ["__version__"]
