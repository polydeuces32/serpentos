"""Tests for palette selection and colour fallback."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serpentos import theme  # noqa: E402


class PaletteTest(unittest.TestCase):
    def test_every_palette_defines_every_role(self):
        for name, palette in (
            ("256", theme.PALETTE_256),
            ("8", theme.PALETTE_8),
            ("mono", theme.PALETTE_MONO),
        ):
            self.assertEqual(set(palette), set(theme.ROLES), name)

    def test_selection_by_terminal_capability(self):
        self.assertIs(theme.palette_for(256), theme.PALETTE_256)
        self.assertIs(theme.palette_for(88), theme.PALETTE_8)
        self.assertIs(theme.palette_for(16), theme.PALETTE_8)
        self.assertIs(theme.palette_for(8), theme.PALETTE_8)
        self.assertIs(theme.palette_for(2), theme.PALETTE_MONO)
        self.assertIs(theme.palette_for(0), theme.PALETTE_MONO)

    def test_eight_colour_palette_stays_in_ansi_range(self):
        for role, (fg, _) in theme.PALETTE_8.items():
            if fg is not None:
                self.assertTrue(0 <= fg <= 7, f"{role} uses {fg}, outside the 8-colour range")

    def test_256_palette_stays_in_range(self):
        for role, (fg, _) in theme.PALETTE_256.items():
            if fg is not None:
                self.assertTrue(0 <= fg <= 255, f"{role} uses {fg}")

    def test_mono_palette_uses_no_colour(self):
        for role, (fg, _) in theme.PALETTE_MONO.items():
            self.assertIsNone(fg, f"{role} should not set a colour in monochrome")

    def test_attribute_names_exist_in_curses(self):
        import curses

        for palette in (theme.PALETTE_256, theme.PALETTE_8, theme.PALETTE_MONO):
            for role, (_, attr) in palette.items():
                if attr is not None:
                    self.assertTrue(hasattr(curses, attr), f"{role} wants curses.{attr}")


class NoColorTest(unittest.TestCase):
    def test_no_color_variable_disables_colour(self):
        self.assertTrue(theme.color_disabled_by_env({"NO_COLOR": "1"}))
        self.assertTrue(theme.color_disabled_by_env({"NO_COLOR": "anything"}))

    def test_empty_no_color_does_not_disable(self):
        self.assertFalse(theme.color_disabled_by_env({"NO_COLOR": ""}))

    def test_dumb_terminal_disables_colour(self):
        self.assertTrue(theme.color_disabled_by_env({"TERM": "dumb"}))
        self.assertFalse(theme.color_disabled_by_env({"TERM": "xterm-256color"}))

    def test_plain_environment_keeps_colour(self):
        self.assertFalse(theme.color_disabled_by_env({}))


class ThemeTest(unittest.TestCase):
    """A disabled theme needs no terminal, so it is testable off-screen."""

    def setUp(self):
        self.theme = theme.Theme(enabled=False, environ={})

    def test_disabled_theme_resolves_every_role(self):
        for role in theme.ROLES:
            self.assertIsInstance(self.theme(role), int)

    def test_unknown_role_is_plain_rather_than_an_error(self):
        self.assertEqual(self.theme("no-such-role"), 0)

    def test_extra_attributes_are_merged(self):
        import curses

        self.assertTrue(self.theme("text", curses.A_UNDERLINE) & curses.A_UNDERLINE)

    def test_disabled_theme_still_distinguishes_roles(self):
        import curses

        self.assertTrue(self.theme("title") & curses.A_BOLD)
        self.assertTrue(self.theme("dim") & curses.A_DIM)
        self.assertTrue(self.theme("head") & curses.A_REVERSE)

    def test_body_shading_runs_head_to_tail(self):
        self.assertEqual(self.theme.body_role(1, 10), "body")
        self.assertEqual(self.theme.body_role(9, 10), "body_dim")

    def test_body_shading_handles_a_one_cell_snake(self):
        self.assertEqual(self.theme.body_role(0, 1), "body")


if __name__ == "__main__":
    unittest.main()
