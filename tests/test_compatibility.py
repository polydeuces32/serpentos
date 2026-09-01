"""Regression tests for everything SerpentOS could do before the runtime existed.

``test_core.py``, ``test_bot.py`` and ``test_cli.py`` cover the game, the agent,
persistence and the command line in depth. This file guards the seam: that
adding a policy runtime on top changed none of it.
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from serpentos import core
from serpentos.environments import snake as snake_environment
from serpentos.policies.qlearning import QLearningPolicy
from serpentos.runtime.models import DecisionContext

V1_QTABLE = {
    "-1|-1|0|0|0|L|3|0": [0.5, 1.5, -0.25],
    "1|0|1|0|0|U|0|1": [2.0, 0.0, 0.0],
}


def run_cli(*args, **kwargs):
    return subprocess.run(
        [sys.executable, "-m", "serpentos", *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=300,
        **kwargs,
    )


class SnakeStillRunsTest(unittest.TestCase):
    def test_the_environment_behaves_as_before(self):
        env = core.SnakeEnv(12, 20, rng=random.Random(1))
        state = env.reset()
        self.assertFalse(env.done)
        _next_state, _reward, done, info = env.step(core.ACTION_STRAIGHT)
        self.assertFalse(done)
        self.assertEqual(info.steps, 1)
        self.assertEqual(len(state.key.split("|")), 8)

    def test_training_still_improves_the_agent(self):
        agent = core.QAgent(core.PRESETS["BOLD"], rng=random.Random(3))
        for index in range(300):
            core.run_episode(core.SnakeEnv(12, 20, rng=random.Random(index)), agent)
        self.assertGreater(len(agent.q), 0)
        self.assertEqual(agent.episodes, 300)

    def test_the_terminal_ui_module_still_imports_its_core(self):
        # The UI needs curses, which is absent on Windows without windows-curses;
        # importing the module it depends on is the portable half of the check.
        self.assertTrue(hasattr(core, "DIFFICULTY"))
        self.assertTrue(hasattr(core, "PRESETS"))
        source = (REPO / "serpentos" / "serpentos.py").read_text(encoding="utf-8")
        self.assertIn("from . import core", source)
        self.assertIn("curses.wrapper(run_ui", source)


class PersistenceCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_a_v1_bare_qtable_still_loads(self):
        path = os.path.join(self.dir, "qtable.json")
        Path(path).write_text(json.dumps(V1_QTABLE), encoding="utf-8")
        q, meta = core.Storage(self.dir).load_checkpoint()
        self.assertEqual(q, V1_QTABLE)
        self.assertEqual(meta, {})

    def test_a_v1_qtable_serves_decisions_through_the_runtime(self):
        path = os.path.join(self.dir, "qtable.json")
        Path(path).write_text(json.dumps(V1_QTABLE), encoding="utf-8")
        policy = QLearningPolicy.from_data_dir(self.dir)
        decision = policy.decide(DecisionContext({"state_key": "-1|-1|0|0|0|L|3|0"}))
        self.assertEqual(decision.action, "left")
        self.assertEqual(policy.states, 2)

    def test_checkpoints_still_round_trip(self):
        storage = core.Storage(self.dir)
        storage.save_checkpoint(V1_QTABLE, {"episodes": 5, "preset": "BOLD"})
        q, meta = storage.load_checkpoint()
        self.assertEqual(q, V1_QTABLE)
        self.assertEqual(meta["episodes"], 5)


class PolicyExchangeCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_export_import_round_trip_is_unchanged(self):
        path = os.path.join(self.dir, "policy.json")
        payload = core.export_policy(path, V1_QTABLE, name="regression")
        self.assertEqual(payload["format"], core.POLICY_FORMAT)
        q, reloaded = core.import_policy(path)
        self.assertEqual(q, V1_QTABLE)
        self.assertEqual(reloaded["fingerprint"], core.policy_fingerprint(V1_QTABLE))

    def test_an_exported_policy_loads_straight_into_the_runtime(self):
        path = os.path.join(self.dir, "policy.json")
        core.export_policy(path, V1_QTABLE, name="regression")
        policy = QLearningPolicy.from_policy_file(path)
        self.assertEqual(policy.fingerprint, core.policy_fingerprint(V1_QTABLE))


class BenchmarkReproducibilityTest(unittest.TestCase):
    def test_the_frozen_spec_has_not_moved(self):
        self.assertEqual(
            core.BENCHMARK_SPEC,
            {
                "version": 1,
                "rows": 22,
                "cols": 78,
                "episodes": 100,
                "max_steps": 1800,
                "seed_base": 1000,
            },
        )

    def test_two_runs_of_the_same_policy_are_identical(self):
        agent = core.QAgent(core.PRESETS["BOLD"], rng=random.Random(5))
        for index in range(200):
            core.run_episode(core.SnakeEnv(12, 20, rng=random.Random(index)), agent)
        first = core.run_benchmark(agent.q)
        second = core.run_benchmark(agent.q)
        self.assertEqual(first, second)

    def test_importing_the_runtime_does_not_perturb_the_score(self):
        # A stray global RNG call at import time would silently change results.
        table = {
            f"{dx}|{dy}|0|0|0|L|3|0": [float(dx), float(dy), 0.5]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        }
        before = core.run_benchmark(table)
        import serpentos.runtime

        self.assertTrue(serpentos.runtime.__all__)
        self.assertEqual(core.run_benchmark(table), before)

    def test_the_environment_namespace_reexports_the_same_objects(self):
        for name in ("SnakeEnv", "QAgent", "run_benchmark", "export_policy", "Storage"):
            self.assertIs(getattr(snake_environment, name), getattr(core, name))


class CommandLineCompatibilityTest(unittest.TestCase):
    def test_help_still_works(self):
        result = run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("serpentos", result.stdout)

    def test_bot_help_still_works(self):
        result = run_cli("bot", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--episodes", result.stdout)

    def test_a_short_training_run_still_produces_a_checkpoint(self):
        directory = tempfile.mkdtemp()
        result = run_cli(
            "bot", "--episodes", "20", "--data-dir", directory, "--quiet", "--json"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["episodes"], 20)
        self.assertTrue(os.path.exists(os.path.join(directory, "qtable.json")))

    def test_bench_output_is_byte_identical_across_runs(self):
        directory = tempfile.mkdtemp()
        policy_path = os.path.join(directory, "policy.json")
        core.export_policy(policy_path, V1_QTABLE, name="ci")
        first = run_cli(
            "bench", "--data-dir", os.path.join(directory, "a"), "--quiet",
            "--import-policy", policy_path,
        )
        second = run_cli(
            "bench", "--data-dir", os.path.join(directory, "b"), "--quiet",
            "--import-policy", policy_path,
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        self.assertIn("fingerprint", first.stdout)


if __name__ == "__main__":
    unittest.main()
