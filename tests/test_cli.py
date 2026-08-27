"""Tests for the top-level command dispatcher."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serpentos import __main__ as cli  # noqa: E402


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = self._tmp.name

    def run_cli(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(args)
        return code, out.getvalue(), err.getvalue()

    def test_help_lists_every_command(self):
        code, out, _ = self.run_cli(["help"])
        self.assertEqual(code, 0)
        for command in ("run", "bot", "bench"):
            self.assertIn(command, out)

    def test_help_flag_without_a_command(self):
        code, out, _ = self.run_cli(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("usage: serpentos", out)

    def test_version(self):
        from serpentos import __version__

        for args in (["version"], ["--version"]):
            code, out, _ = self.run_cli(args)
            self.assertEqual(code, 0)
            self.assertIn(__version__, out)

    def test_unknown_command_is_rejected_with_usage(self):
        code, _, err = self.run_cli(["frobnicate"])
        self.assertEqual(code, 2)
        self.assertIn("unknown command", err)

    def test_bot_command_dispatches(self):
        code, out, _ = self.run_cli(
            ["bot", "--episodes", "5", "--data-dir", self.data_dir, "--quiet", "--json"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["episodes"], 5)

    def test_bench_command_is_bot_bench(self):
        self.run_cli(["bot", "--episodes", "100", "--data-dir", self.data_dir, "--quiet"])
        code, out, _ = self.run_cli(["bench", "--data-dir", self.data_dir, "--quiet"])
        self.assertEqual(code, 0)
        result = json.loads(out)
        self.assertIn("mean_score", result)
        self.assertIn("fingerprint", result)

    def test_run_command_reaches_the_ui_parser(self):
        # --help makes argparse exit before any terminal is touched.
        with self.assertRaises(SystemExit) as caught:
            self.run_cli(["run", "--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_play_is_an_alias_for_run(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_cli(["play", "--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_bare_flags_still_reach_the_ui(self):
        """`serpentos --data-dir X` kept working when subcommands were added."""
        with self.assertRaises(SystemExit) as caught:
            self.run_cli(["--data-dir", self.data_dir, "--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_ui_accepts_no_color(self):
        out = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stdout(out):
            cli.main(["run", "--help"])
        self.assertIn("--no-color", out.getvalue())


if __name__ == "__main__":
    unittest.main()
