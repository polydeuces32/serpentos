"""Colour handling for the terminal UI.

Three palettes, picked from what the terminal reports: 256-colour, 8-colour and
monochrome. Every role resolves to *something* in all three, so the UI looks
intentional on a modern terminal and still runs on a VT100 or under
``TERM=dumb``.

Palette selection is a pure function so it can be tested without a terminal.
The module also remains importable on platforms where ``curses`` is not
available, notably stock Windows Python.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

Palette = Dict[str, Tuple[Optional[int], Optional[str]]]

ROLES = (
    "text", "dim", "border", "title", "accent",
    "head", "body", "body_dim", "food",
    "label", "value", "good", "bad", "spark",
    "glow_hot", "glow_warm", "glow_cool",
)

PALETTE_256: Palette = {
    "text": (252, None), "dim": (244, None), "border": (238, None),
    "title": (220, "A_BOLD"), "accent": (214, None),
    "head": (118, "A_BOLD"), "body": (41, None), "body_dim": (29, None),
    "food": (203, "A_BOLD"), "label": (245, None), "value": (81, None),
    "good": (84, None), "bad": (203, None), "spark": (79, None),
    "glow_hot": (226, "A_BOLD"), "glow_warm": (214, None),
    "glow_cool": (130, "A_DIM"),
}

PALETTE_8: Palette = {
    "text": (7, None), "dim": (7, "A_DIM"), "border": (7, "A_DIM"),
    "title": (3, "A_BOLD"), "accent": (3, None), "head": (2, "A_BOLD"),
    "body": (2, None), "body_dim": (2, "A_DIM"), "food": (1, "A_BOLD"),
    "label": (6, None), "value": (6, "A_BOLD"), "good": (2, None),
    "bad": (1, None), "spark": (6, None), "glow_hot": (3, "A_BOLD"),
    "glow_warm": (3, None), "glow_cool": (3, "A_DIM"),
}

PALETTE_MONO: Palette = {
    "text": (None, None), "dim": (None, "A_DIM"),
    "border": (None, "A_DIM"), "title": (None, "A_BOLD"),
    "accent": (None, "A_BOLD"), "head": (None, "A_REVERSE"),
    "body": (None, None), "body_dim": (None, "A_DIM"),
    "food": (None, "A_BOLD"), "label": (None, "A_DIM"),
    "value": (None, "A_BOLD"), "good": (None, "A_BOLD"),
    "bad": (None, "A_BOLD"), "spark": (None, None),
    "glow_hot": (None, "A_BOLD"), "glow_warm": (None, None),
    "glow_cool": (None, "A_DIM"),
}


class _FallbackCurses:
    """Attribute-only stand-in used when the real curses module is unavailable."""

    A_BOLD = 1 << 0
    A_DIM = 1 << 1
    A_REVERSE = 1 << 2
    A_UNDERLINE = 1 << 3
    COLOR_BLACK = 0
    COLORS = 0

    class error(Exception):
        pass

    def start_color(self) -> None:
        raise self.error("curses is unavailable")

    def use_default_colors(self) -> None:
        raise self.error("curses is unavailable")

    def init_pair(self, pair: int, fg: int, background: int) -> None:
        raise self.error("curses is unavailable")

    def color_pair(self, pair: int) -> int:
        return 0


def _load_curses():
    try:
        import curses
    except (ImportError, ModuleNotFoundError):
        return _FallbackCurses(), False
    return curses, True


def palette_for(colors: int) -> Palette:
    """Pick a palette for a terminal advertising ``colors`` colours."""
    if colors >= 256:
        return PALETTE_256
    if colors >= 8:
        return PALETTE_8
    return PALETTE_MONO


def color_disabled_by_env(environ=None) -> bool:
    """Honour the NO_COLOR convention (https://no-color.org)."""
    env = os.environ if environ is None else environ
    return bool(env.get("NO_COLOR", "")) or env.get("TERM", "") == "dumb"


class Theme:
    """Resolve semantic role names to terminal attributes.

    Construction is safe even when curses is unavailable. In that case the
    theme becomes monochrome and exposes synthetic attributes for non-UI tests;
    the actual terminal UI still requires the real curses module.
    """

    def __init__(self, enabled: bool = True, environ=None) -> None:
        curses, curses_available = _load_curses()
        self._curses = curses
        self.curses_available = curses_available
        self.enabled = enabled and curses_available and not color_disabled_by_env(environ)
        self.colors = 0
        self._attrs: Dict[str, int] = {}

        if self.enabled:
            try:
                curses.start_color()
                self.colors = getattr(curses, "COLORS", 0)
            except curses.error:
                self.enabled = False

        if not self.enabled:
            self.colors = 0

        palette = palette_for(self.colors)
        # Attempted even when colour is disabled. curses.wrapper() calls
        # start_color() before we get here, and a colour-capable ncurses that
        # has not been told to keep the terminal's defaults resets to its own
        # assumed white-on-black — which would repaint the screen of someone who
        # asked for --no-color on a light background.
        background = -1
        try:
            curses.use_default_colors()
        except curses.error:
            background = curses.COLOR_BLACK

        pair = 0
        for role in ROLES:
            fg, attr_name = palette.get(role, (None, None))
            attr = getattr(curses, attr_name, 0) if attr_name else 0
            if self.enabled and fg is not None:
                pair += 1
                try:
                    curses.init_pair(pair, fg, background)
                    attr |= curses.color_pair(pair)
                except (curses.error, ValueError):
                    pair -= 1
            self._attrs[role] = attr

    def __call__(self, role: str, extra: int = 0) -> int:
        return self._attrs.get(role, 0) | extra

    def body_role(self, index: int, length: int) -> str:
        """Shade a body segment by how far it sits from the head."""
        if length <= 1:
            return "body"
        return "body" if index <= max(1, length // 2) else "body_dim"
