"""The public API surface, pinned.

`docs/COMPATIBILITY.md` says exported names do not move casually. This is the
test that makes that true rather than merely written down: the surface is
enumerated below, and any change to it fails here.

Failing this test is not necessarily a bug. It means you changed the public
contract, and you now have to decide deliberately whether that is what you
meant:

* **Added a name?** Add it to the list, and record its tier in
  `docs/COMPATIBILITY.md`. Additions are fine in a minor release.
* **Removed or renamed one?** That is a breaking change. It needs a major
  version bump and a deprecation period first.
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serpentos
import serpentos.policies

DOCS = Path(__file__).resolve().parents[1] / "docs" / "COMPATIBILITY.md"

#: Everything `import serpentos` promises. Sorted, so diffs are readable.
TOP_LEVEL = {
    "ActionValidator",
    "AUDIT_SCHEMA_VERSION",
    "AuditError",
    "AuditRecord",
    "AuditSink",
    "BasePolicy",
    "ComparisonReport",
    "ConfigurationError",
    "Decision",
    "DecisionContext",
    "DecisionEngine",
    "DecisionValidationError",
    "DecisionValidator",
    "Disagreement",
    "DisagreementSummary",
    "InMemoryAuditLog",
    "JsonlAuditLog",
    "MetricSummary",
    "NullAuditSink",
    "Outcome",
    "OutcomeSummary",
    "PairDisagreement",
    "Policy",
    "PolicyError",
    "PolicyReport",
    "REDACTED",
    "ReplayError",
    "ReplayReport",
    "ReplayResult",
    "SUPPORTED_AUDIT_SCHEMA_VERSIONS",
    "SerpentOSError",
    "ValidationResult",
    "__version__",
    "compare",
    "is_deterministic",
    "read_jsonl",
    "replay",
    "replay_all",
}

#: Everything `serpentos.policies` promises.
POLICIES = {
    "AllOf",
    "Always",
    "AnyOf",
    "Comparison",
    "Condition",
    "LinearScorer",
    "Not",
    "Predicate",
    "QLearningPolicy",
    "Rule",
    "RulePolicy",
    "WeightedPolicy",
    "condition_from_dict",
    "when",
}

#: Named in docs/COMPATIBILITY.md as changeable within a major version.
EXPERIMENTAL = {
    "replay",
    "replay_all",
    "ReplayResult",
    "ReplayReport",
    "compare",
    "ComparisonReport",
    "PolicyReport",
    "OutcomeSummary",
    "MetricSummary",
    "DisagreementSummary",
    "Disagreement",
    "PairDisagreement",
}

#: The stable core. Breaking any of these requires a major version bump.
STABLE_CORE = {
    "DecisionContext",
    "Decision",
    "Outcome",
    "Policy",
    "DecisionEngine",
    "ActionValidator",
}


class TopLevelSurfaceTest(unittest.TestCase):
    def test_the_exported_names_are_exactly_the_pinned_set(self):
        self.assertEqual(
            set(serpentos.__all__),
            TOP_LEVEL,
            "the public surface changed; see this module's docstring",
        )

    def test_every_exported_name_actually_exists(self):
        for name in serpentos.__all__:
            self.assertTrue(hasattr(serpentos, name), f"{name} is advertised but missing")

    def test_nothing_is_exported_twice(self):
        self.assertEqual(len(serpentos.__all__), len(set(serpentos.__all__)))

    def test_no_private_name_is_exported(self):
        for name in serpentos.__all__:
            if name.startswith("__") and name.endswith("__"):
                continue
            self.assertFalse(name.startswith("_"), f"{name} is private")


class PoliciesSurfaceTest(unittest.TestCase):
    def test_the_exported_names_are_exactly_the_pinned_set(self):
        self.assertEqual(set(serpentos.policies.__all__), POLICIES)

    def test_every_exported_name_resolves(self):
        for name in serpentos.policies.__all__:
            self.assertIsNotNone(getattr(serpentos.policies, name))


class StabilityTierTest(unittest.TestCase):
    def test_the_stable_core_is_exported(self):
        self.assertTrue(STABLE_CORE.issubset(TOP_LEVEL))

    def test_experimental_names_are_all_real_exports(self):
        self.assertTrue(
            EXPERIMENTAL.issubset(TOP_LEVEL),
            EXPERIMENTAL - TOP_LEVEL,
        )

    def test_nothing_is_both_stable_core_and_experimental(self):
        self.assertEqual(STABLE_CORE & EXPERIMENTAL, set())

    def test_the_stable_core_signatures_are_what_callers_depend_on(self):
        import inspect

        # Parameter names and order are part of the promise: callers pass some
        # of these positionally. New parameters go on the end with a default.
        expected = {
            serpentos.DecisionContext: ["values", "request_id"],
            serpentos.Decision: [
                "action", "policy_name", "policy_version", "metadata", "decision_id",
            ],
            serpentos.Outcome: ["success", "score", "metrics", "metadata", "decision_id"],
        }
        for model, parameters in expected.items():
            actual = list(inspect.signature(model).parameters)
            self.assertEqual(actual, parameters, f"{model.__name__} signature moved")

    def test_the_engine_keeps_its_documented_parameters(self):
        import inspect

        parameters = list(inspect.signature(serpentos.DecisionEngine).parameters)
        self.assertEqual(parameters[0], "policy")
        for name in ("validator", "audit_sink", "fallback_policy", "clock", "id_factory"):
            self.assertIn(name, parameters)


class CompatibilityDocumentTest(unittest.TestCase):
    """The document and the code must not drift apart."""

    @classmethod
    def setUpClass(cls):
        cls.text = DOCS.read_text(encoding="utf-8")

    def test_the_document_exists_and_names_the_version(self):
        self.assertIn(serpentos.__version__, self.text)

    def test_every_experimental_name_is_listed_as_experimental(self):
        experimental_section = self.text.split("### Experimental", 1)[1].split("###", 1)[0]
        for name in EXPERIMENTAL:
            self.assertIn(name, experimental_section, f"{name} is not documented as experimental")

    def test_the_stable_core_is_described_as_stable(self):
        stable_section = self.text.split("### Stable", 1)[1].split("### Experimental", 1)[0]
        for name in STABLE_CORE:
            self.assertIn(name, stable_section, f"{name} is not documented as stable")

    def test_the_audit_schema_version_matches_the_document(self):
        row = re.search(
            r"^\|\s*Audit records \(JSONL\)\s*\|\s*`schema_version`\s*\|\s*(\d+)\s*\|$",
            self.text,
            re.MULTILINE,
        )
        self.assertIsNotNone(row, "the serialized-formats table lost its audit row")
        self.assertEqual(int(row.group(1)), serpentos.AUDIT_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
