"""The non-Snake example must keep working, and keep being about retries.

This is the script a new reader runs first. If it breaks, or quietly stops
demonstrating the thing it claims to demonstrate, the claim that SerpentOS is
not just a game becomes untestable marketing.
"""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXAMPLE = ROOT / "examples" / "retry_policy.py"

from serpentos import ActionValidator, DecisionContext, DecisionEngine  # noqa: E402

sys.path.insert(0, str(ROOT / "examples"))
import retry_policy  # noqa: E402


class ExampleRunsTest(unittest.TestCase):
    def test_the_script_exists_and_is_runnable_as_a_file(self):
        self.assertTrue(EXAMPLE.is_file())

    def test_running_it_succeeds_and_explains_itself(self):
        # Run from a directory that is not the repository root, to prove the
        # script bootstraps its own import path rather than relying on cwd.
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)],
            cwd=str(ROOT.parent),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        for expected in ("DECIDING", "AUDIT RECORD", "REPLAY", "COMPARISON"):
            self.assertIn(expected, result.stdout)
        # It must show its work, not just an answer.
        self.assertIn("decision id", result.stdout)
        self.assertIn("because     : rule", result.stdout)

    def test_it_imports_nothing_that_could_reach_a_network_or_a_game(self):
        imported = set()
        for node in ast.walk(ast.parse(EXAMPLE.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        roots = {name.split(".")[0] for name in imported}
        self.assertEqual(roots - {"serpentos"}, {"__future__", "json", "os", "sys"})
        for forbidden in ("serpentos.core", "serpentos.bot", "serpentos.environments"):
            self.assertNotIn(forbidden, imported)


class RetryLogicTest(unittest.TestCase):
    """The example's policies are importable, so their logic is testable."""

    def setUp(self):
        self.policy = retry_policy.retry_rules()

    def decide(self, **values):
        return self.policy.decide(DecisionContext(values)).action

    def test_a_transient_server_error_is_retried(self):
        self.assertEqual(self.decide(attempt=1, status_code=503, latency_ms=100), "retry")

    def test_the_retry_budget_is_respected(self):
        self.assertEqual(self.decide(attempt=3, status_code=503, latency_ms=100), "fail")
        self.assertEqual(self.decide(attempt=9, status_code=503, latency_ms=100), "fail")

    def test_rate_limiting_means_back_off(self):
        self.assertEqual(self.decide(attempt=1, status_code=429, latency_ms=10), "wait")

    def test_our_own_bad_request_is_not_retried(self):
        self.assertEqual(self.decide(attempt=1, status_code=400, latency_ms=10), "fail")
        self.assertEqual(self.decide(attempt=1, status_code=404, latency_ms=10), "fail")

    def test_a_struggling_server_gets_a_pause_first(self):
        self.assertEqual(self.decide(attempt=1, status_code=500, latency_ms=3000), "wait")

    def test_success_is_not_the_policys_business(self):
        # Nothing routes a 200 through here, but the default must be safe.
        self.assertEqual(self.decide(attempt=1, status_code=200, latency_ms=10), "fail")

    def test_every_action_is_inside_the_allow_list(self):
        validator = ActionValidator(retry_policy.ACTIONS)
        engine = DecisionEngine(policy=self.policy, validator=validator)
        for status in range(200, 600, 7):
            for attempt in range(1, 6):
                context = DecisionContext(
                    {"attempt": attempt, "status_code": status, "latency_ms": 500}
                )
                self.assertIn(engine.decide(context).action, retry_policy.ACTIONS)

    def test_the_scoring_policy_covers_the_same_actions(self):
        scoring = retry_policy.retry_scoring()
        self.assertEqual(set(scoring.actions), retry_policy.ACTIONS)

    def test_both_policies_are_deterministic(self):
        for policy in (self.policy, retry_policy.retry_scoring()):
            context = DecisionContext({"attempt": 2, "status_code": 503, "latency_ms": 900})
            first = policy.decide(context).action
            for _ in range(20):
                self.assertEqual(policy.decide(context).action, first)

    def test_the_policy_is_pure_data_and_serialises(self):
        payload = self.policy.to_dict()
        self.assertEqual(len(payload["rules"]), 5)
        self.assertEqual(payload["default_action"], "fail")

    def test_tightening_the_budget_changes_only_what_it_should(self):
        strict = retry_policy.retry_rules(version="2.0", max_attempts=2)
        loose = DecisionContext({"attempt": 1, "status_code": 503, "latency_ms": 10})
        tight = DecisionContext({"attempt": 2, "status_code": 503, "latency_ms": 10})
        self.assertEqual(strict.decide(loose).action, self.policy.decide(loose).action)
        self.assertEqual(self.policy.decide(tight).action, "retry")
        self.assertEqual(strict.decide(tight).action, "fail")


if __name__ == "__main__":
    unittest.main()
