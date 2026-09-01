"""Tests for palette selection and colour fallback."""

import os
import re
import select
import subprocess
import sys
import tempfile
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from serpentos import theme  # noqa: E402

try:
    import pty
except ImportError:  # pragma: no cover - Windows has no pty
    pty = None


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

    def test_attribute_names_resolve_on_this_platform(self):
        disabled = theme.Theme(enabled=False, environ={})
        attrs = disabled._curses
        for palette in (theme.PALETTE_256, theme.PALETTE_8, theme.PALETTE_MONO):
            for role, (_, attr) in palette.items():
                if attr is not None:
                    self.assertTrue(hasattr(attrs, attr), f"{role} wants {attr}")


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
        underline = self.theme._curses.A_UNDERLINE
        self.assertTrue(self.theme("text", underline) & underline)

    def test_disabled_theme_still_distinguishes_roles(self):
        attrs = self.theme._curses
        self.assertTrue(self.theme("title") & attrs.A_BOLD)
        self.assertTrue(self.theme("dim") & attrs.A_DIM)
        self.assertTrue(self.theme("head") & attrs.A_REVERSE)

    def test_body_shading_runs_head_to_tail(self):
        self.assertEqual(self.theme.body_role(1, 10), "body")
        self.assertEqual(self.theme.body_role(9, 10), "body_dim")

    def test_body_shading_handles_a_one_cell_snake(self):
        self.assertEqual(self.theme.body_role(0, 1), "body")


@unittest.skipIf(pty is None, "needs a pty")
class LiveTerminalTest(unittest.TestCase):
    """Drive the real UI through a pty and inspect the bytes it emits.

    Palette selection is a pure function and tested above; what cannot be tested
    that way is what ncurses actually puts on the wire.
    """

    SGR = re.compile(r"\x1b\[[0-9;]*m")
    # Any SGR that sets a foreground or background colour.
    COLOUR_SGR = re.compile(r"\x1b\[[0-9;]*?(?:3[0-7]|4[0-7]|9[0-7]|10[0-7]|38;5;|48;5;)[0-9;]*m")

    def drive(self, *args, term="xterm-256color", seconds=2.5):
        env = dict(os.environ, TERM=term, LINES="30", COLUMNS="100")
        env.pop("NO_COLOR", None)
        master, slave = pty.openpty()
        process = subprocess.Popen(
            [sys.executable, "-m", "serpentos", "run", *args,
             "--data-dir", tempfile.mkdtemp()],
            stdin=slave, stdout=slave, stderr=slave, env=env, cwd=REPO,
        )
        os.close(slave)
        output = b""
        deadline = time.time() + seconds
        try:
            while time.time() < deadline:
                readable, _, _ = select.select([master], [], [], 0.2)
                if not readable:
                    continue
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output += chunk
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                process.kill()
            os.close(master)
        return output.decode("utf-8", "replace")

    def test_the_ui_paints_without_crashing(self):
        text = self.drive()
        self.assertNotIn("Traceback", text)
        self.assertTrue(self.COLOUR_SGR.search(text), "expected colour on a 256-colour terminal")

    def test_no_color_emits_no_colour_at_all(self):
        # curses.wrapper() starts colour before the theme is built, so a
        # colour-capable ncurses will happily reset to its own white-on-black
        # unless it is told to keep the terminal's defaults.
        text = self.drive("--no-color")
        self.assertNotIn("Traceback", text)
        found = self.COLOUR_SGR.findall(text)
        self.assertEqual(found, [], f"--no-color still emitted {sorted(set(found))}")

    def test_a_monochrome_terminal_does_not_crash(self):
        text = self.drive(term="vt100")
        self.assertNotIn("Traceback", text)


if __name__ == "__main__":
    unittest.main()
