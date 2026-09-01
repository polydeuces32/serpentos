"""The public library surface.

If any of these break, someone's ``from serpentos import ...`` breaks with them.
"""

import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import serpentos
from serpentos import (
    ActionValidator,
    AuditRecord,
    Decision,
    DecisionContext,
    DecisionEngine,
    InMemoryAuditLog,
    Outcome,
    Policy,
    compare,
    replay,
)


def run_python(code):
    """Run ``code`` in a clean interpreter rooted at the repository."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=120,
    )


class SurfaceTest(unittest.TestCase):
    def test_everything_advertised_is_importable(self):
        for name in serpentos.__all__:
            self.assertTrue(hasattr(serpentos, name), f"serpentos.{name} is missing")

    def test_version_is_exposed(self):
        self.assertTrue(serpentos.__version__)
        self.assertEqual(serpentos.__version__.split(".")[0], "2")

    def test_policies_package_exports_the_implementations(self):
        from serpentos.policies import QLearningPolicy, RulePolicy, WeightedPolicy

        for policy_type in (RulePolicy, WeightedPolicy, QLearningPolicy):
            self.assertTrue(callable(policy_type))

    def test_policies_package_rejects_unknown_attributes(self):
        import serpentos.policies as policies

        with self.assertRaises(AttributeError):
            policies.NoSuchPolicy

    def test_public_types_are_the_runtime_types(self):
        from serpentos.runtime import models

        self.assertIs(Decision, models.Decision)
        self.assertIs(DecisionContext, models.DecisionContext)
        self.assertIs(Outcome, models.Outcome)

    def test_policy_protocol_recognises_a_conforming_object(self):
        from serpentos.policies import RulePolicy

        policy = RulePolicy(name="p", version="1.0", rules=[], default_action="fail")
        self.assertIsInstance(policy, Policy)


class DocumentedExampleTest(unittest.TestCase):
    def test_the_readme_example_runs(self):
        from serpentos import DecisionContext, DecisionEngine
        from serpentos.policies import Rule, RulePolicy, when
        from serpentos.runtime.validation import ActionValidator

        policy = RulePolicy(
            name="retry-policy",
            version="1.0",
            rules=[
                Rule("fail", when("attempts", "ge", 3), name="give-up"),
                Rule("retry", when("status_code", "in", [503, 504]), name="retry-5xx"),
            ],
            default_action="fail",
        )
        engine = DecisionEngine(
            policy=policy,
            validator=ActionValidator(allowed_actions={"retry", "fail"}),
        )
        decision = engine.decide(
            DecisionContext(values={"attempts": 2, "status_code": 503})
        )
        self.assertEqual(decision.action, "retry")

    def test_the_full_lifecycle_works_end_to_end(self):
        from serpentos.policies import Rule, RulePolicy, when

        policy = RulePolicy(
            name="retry-policy",
            version="1.0",
            rules=[Rule("retry", when("status_code", "ge", 500))],
            default_action="fail",
        )
        audit = InMemoryAuditLog()
        engine = DecisionEngine(
            policy, validator=ActionValidator({"retry", "fail"}), audit_sink=audit
        )
        cases = [DecisionContext({"status_code": code}) for code in (200, 500, 503)]
        for case in cases:
            engine.decide(case)

        self.assertEqual([r.action for r in audit.records], ["fail", "retry", "retry"])
        self.assertTrue(all(replay(policy, r).match for r in audit.records))
        report = compare([policy], cases)
        self.assertEqual(report.for_policy("retry-policy").action_counts, {"fail": 1, "retry": 2})
        self.assertIsInstance(audit.records[0], AuditRecord)


class IsolationTest(unittest.TestCase):
    """The runtime must not drag the game (or curses) in behind it."""

    def test_importing_serpentos_does_not_import_curses_or_the_game(self):
        result = run_python(
            "import sys, serpentos; "
            "print(sorted(m for m in sys.modules "
            "if m in ('curses', 'serpentos.core', 'serpentos.serpentos', 'serpentos.bot')))"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")

    def test_the_runtime_package_stands_alone(self):
        result = run_python(
            "import sys, serpentos.runtime; "
            "print('serpentos.core' in sys.modules, 'random' in dir(serpentos.runtime))"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False False")

    def test_rule_policies_do_not_pull_in_the_snake_environment(self):
        result = run_python(
            "import sys; from serpentos.policies import RulePolicy; "
            "print('serpentos.core' in sys.modules)"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")

    def test_the_qlearning_adapter_does_pull_in_the_core(self):
        result = run_python(
            "import sys; from serpentos.policies import QLearningPolicy; "
            "print('serpentos.core' in sys.modules)"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "True")

    def test_no_module_in_the_runtime_uses_eval_exec_or_pickle(self):
        banned = ("eval(", "exec(", "import pickle", "__import__(", "marshal", "shelve")
        for path in sorted((REPO / "serpentos" / "runtime").glob("*.py")) + sorted(
            (REPO / "serpentos" / "policies").glob("*.py")
        ):
            source = path.read_text(encoding="utf-8")
            for token in banned:
                self.assertNotIn(token, source, f"{path.name} contains {token!r}")


if __name__ == "__main__":
    unittest.main()
