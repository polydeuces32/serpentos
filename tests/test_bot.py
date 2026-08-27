"""Tests for the headless SerpentOS agent runner."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serpentos import bot, core  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BotCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = self._tmp.name

    def run_bot(self, *args):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = bot.main(["--data-dir", self.data_dir, "--quiet", *args])
        return code, buffer.getvalue()

    def test_training_run_writes_a_checkpoint_and_summary(self):
        code, out = self.run_bot("--episodes", "50", "--seed", "1", "--json")
        self.assertEqual(code, bot.EXIT_OK)
        summary = json.loads(out)
        self.assertEqual(summary["episodes"], 50)
        self.assertEqual(summary["stopped_by"], "completed")
        self.assertGreater(summary["states"], 0)

        storage = core.Storage(self.data_dir)
        q, meta = storage.load_checkpoint()
        self.assertGreater(len(q), 0)
        self.assertEqual(meta["episodes"], 50)
        self.assertEqual(meta["preset"], "DEFAULT")
        self.assertTrue(os.path.exists(storage.training_log_path))

    def test_second_run_resumes_the_first(self):
        self.run_bot("--episodes", "30", "--seed", "2")
        _, out = self.run_bot("--episodes", "30", "--seed", "2", "--json")
        summary = json.loads(out)
        self.assertEqual(summary["episodes"], 30)
        self.assertEqual(summary["lifetime_episodes"], 60)

    def test_fresh_discards_prior_learning(self):
        self.run_bot("--episodes", "30", "--seed", "3")
        _, out = self.run_bot("--episodes", "10", "--seed", "3", "--fresh", "--json")
        self.assertEqual(json.loads(out)["lifetime_episodes"], 10)

    def test_eval_does_not_write_a_checkpoint(self):
        self.run_bot("--episodes", "40", "--seed", "4")
        storage = core.Storage(self.data_dir)
        before, meta_before = storage.load_checkpoint()

        code, out = self.run_bot("--eval", "10", "--seed", "4", "--json")
        self.assertEqual(code, bot.EXIT_OK)
        self.assertEqual(json.loads(out)["mode"], "eval")

        after, meta_after = storage.load_checkpoint()
        self.assertEqual(after, before)
        self.assertEqual(meta_after["episodes"], meta_before["episodes"])

    def test_metrics_file_is_json_lines(self):
        path = os.path.join(self.data_dir, "metrics.jsonl")
        self.run_bot("--episodes", "5", "--seed", "5", "--metrics", path)
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["mode"], "bot")
        self.assertIn("score", rows[0])

    def test_same_seed_gives_the_same_summary(self):
        _, first = self.run_bot("--episodes", "40", "--seed", "42", "--fresh", "--json")
        _, second = self.run_bot("--episodes", "40", "--seed", "42", "--fresh", "--json")
        a, b = json.loads(first), json.loads(second)
        for field in ("avg_score", "best_score", "total_steps", "states"):
            self.assertEqual(a[field], b[field], field)

    def test_time_budget_stops_the_run(self):
        _, out = self.run_bot("--forever", "--max-seconds", "0.5", "--seed", "6", "--json")
        summary = json.loads(out)
        self.assertEqual(summary["stopped_by"], "time-budget")
        self.assertGreater(summary["episodes"], 0)

    def test_signal_request_stops_between_episodes(self):
        storage = core.Storage(self.data_dir)
        env = core.SnakeEnv(10, 20)
        agent = core.QAgent()
        stop = {"flag": False}

        def should_stop():
            if agent.episodes >= 3:
                stop["flag"] = True
            return stop["flag"]

        runner = bot.BotRunner(env, agent, storage, should_stop=should_stop, log_every=0)
        summary = runner.run(episodes=1000)
        self.assertEqual(summary["stopped_by"], "signal")
        self.assertEqual(summary["episodes"], 3)
        self.assertGreater(len(storage.load_checkpoint()[0]), 0)

    def test_data_dir_is_locked_against_a_second_writer(self):
        storage = core.Storage(self.data_dir)
        with storage.lock(owner="test"):
            code, _ = self.run_bot("--episodes", "1")
        self.assertEqual(code, bot.EXIT_LOCKED)

    def test_undersized_grid_is_rejected_cleanly(self):
        code, _ = self.run_bot("--episodes", "1", "--rows", "2", "--cols", "2")
        self.assertEqual(code, bot.EXIT_ERROR)

    def test_bench_reports_a_reproducible_score(self):
        self.run_bot("--episodes", "300", "--seed", "8")
        _, first = self.run_bot("--bench")
        _, second = self.run_bot("--bench")
        self.assertEqual(json.loads(first), json.loads(second))
        self.assertIn("fingerprint", json.loads(first))

    def test_bench_without_a_policy_fails_cleanly(self):
        code, _ = self.run_bot("--bench")
        self.assertEqual(code, bot.EXIT_ERROR)

    def test_export_then_import_preserves_the_score(self):
        path = os.path.join(self.data_dir, "policy.json")
        self.run_bot("--episodes", "300", "--seed", "9")
        _, exported = self.run_bot("--bench", "--export-policy", path, "--name", "tester")
        self.assertTrue(os.path.exists(path))

        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = bot.main(["--data-dir", other.name, "--quiet", "--bench", "--import-policy", path])
        self.assertEqual(code, bot.EXIT_OK)
        self.assertEqual(json.loads(buffer.getvalue()), json.loads(exported))

    def test_import_of_a_bad_policy_exits_with_an_error(self):
        path = os.path.join(self.data_dir, "junk.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"format": "nope"}')
        code, _ = self.run_bot("--bench", "--import-policy", path)
        self.assertEqual(code, bot.EXIT_ERROR)

    def test_module_entry_point_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "serpentos", "bot", "--episodes", "5",
             "--data-dir", os.path.join(self.data_dir, "sub"), "--quiet", "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["episodes"], 5)


if __name__ == "__main__":
    unittest.main()
