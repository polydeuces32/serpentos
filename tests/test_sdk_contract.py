"""What an external application sees, using nothing but the public API.

Every other test in this suite reaches into `serpentos.runtime.*` and
`serpentos.policies.*` because it is testing those modules. This one deliberately
does not. It imports only from the two top-level packages a published SDK
promises, and runs the lifecycle a real integration runs:

    application context
        -> DecisionEngine -> policy -> validator -> Decision
        -> the application executes something
        -> Outcome -> audit -> replay -> comparison

If this test needs editing to accommodate an internal refactor, the refactor
broke the public contract. That is the entire point of it.

The scenario is a payment gateway choosing how to handle a declined charge —
picked because it is unrelated to Snake, unrelated to retries-as-such, and has
outcomes that are genuinely multi-dimensional.
"""

import ast
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Exactly what an external application is expected to import: the top-level
# package, and the policies package. Nothing below them.
from serpentos import (
    ActionValidator,
    Decision,
    DecisionContext,
    DecisionEngine,
    DecisionValidationError,
    InMemoryAuditLog,
    JsonlAuditLog,
    Outcome,
    Policy,
    PolicyError,
    compare,
    read_jsonl,
    replay_all,
)
from serpentos.policies import LinearScorer, Rule, RulePolicy, WeightedPolicy, when

#: What the gateway is permitted to do about a declined charge.
ACTIONS = {"retry_now", "retry_later", "request_new_card", "abandon"}


def gateway_rules(version="1.0", max_attempts=4):
    """The bank's decline code decides the response."""
    return RulePolicy(
        name="decline-handling",
        version=version,
        rules=[
            Rule("abandon", when("attempts", "ge", max_attempts), name="exhausted"),
            Rule(
                "request_new_card",
                when("decline_code", "in", ["expired_card", "stolen_card"]),
                name="card-is-unusable",
            ),
            Rule(
                "retry_later",
                when("decline_code", "eq", "insufficient_funds"),
                name="wait-for-payday",
            ),
            Rule("retry_now", when("decline_code", "eq", "issuer_unavailable"), name="transient"),
        ],
        default_action="abandon",
    )


def gateway_scoring():
    """The same problem expressed as scoring rather than branching."""
    return WeightedPolicy(
        name="decline-scoring",
        version="1.0",
        scorers={
            "retry_now": LinearScorer({"issuer_up": 5.0, "attempts": -2.0}, bias=1.0),
            "retry_later": LinearScorer({"balance_trend": 3.0, "attempts": -1.0}),
            "request_new_card": LinearScorer({"card_expires_days": -0.25}, bias=1.5),
            "abandon": LinearScorer({"attempts": 1.5}, bias=-4.0),
        },
    )


CHARGES = [
    {"attempts": 1, "decline_code": "issuer_unavailable", "amount_cents": 4999,
     "issuer_up": 1, "balance_trend": 0, "card_expires_days": 400},
    {"attempts": 2, "decline_code": "insufficient_funds", "amount_cents": 12500,
     "issuer_up": 1, "balance_trend": 1, "card_expires_days": 220},
    {"attempts": 1, "decline_code": "expired_card", "amount_cents": 899,
     "issuer_up": 1, "balance_trend": 0, "card_expires_days": -3},
    {"attempts": 5, "decline_code": "issuer_unavailable", "amount_cents": 25000,
     "issuer_up": 1, "balance_trend": 0, "card_expires_days": 90},
]


def execute(action, charge):
    """Stand-in for the host application actually doing the thing.

    Note what this proves: the engine returned an action and did nothing with
    it. Every side effect below happens here, in application code.
    """
    if action == "retry_now":
        settled = charge["decline_code"] == "issuer_unavailable"
        return Outcome(
            success=settled,
            metrics={"latency_ms": 240.0, "network_calls": 1, "amount_cents": charge["amount_cents"]},
            metadata={"channel": "immediate", "decline_code": charge["decline_code"]},
        )
    if action == "retry_later":
        return Outcome(
            success=True,
            metrics={"latency_ms": 5.0, "network_calls": 0, "scheduled_hours": 48.0},
            metadata={"channel": "queued"},
        )
    if action == "request_new_card":
        return Outcome(
            success=False,
            metrics={"latency_ms": 18.0, "network_calls": 0, "emails_sent": 1},
            metadata={"channel": "customer-contact"},
        )
    return Outcome(
        success=False,
        metrics={"latency_ms": 1.0, "network_calls": 0},
        metadata={"channel": "none", "reason": "written off"},
    )


class SdkImportSurfaceTest(unittest.TestCase):
    """This file must not reach past the published packages."""

    def test_this_test_imports_only_public_packages(self):
        source = Path(__file__).read_text(encoding="utf-8")
        modules = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
        serpentos_imports = {name for name in modules if name.startswith("serpentos")}
        self.assertEqual(serpentos_imports, {"serpentos", "serpentos.policies"})

    def test_it_touches_no_private_attributes(self):
        # Python's own dunders are fair game; a leading single underscore on a
        # SerpentOS object is not.
        source = Path(__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Attribute):
                continue
            name = node.attr
            if name.startswith("__") and name.endswith("__"):
                continue
            self.assertFalse(
                name.startswith("_"), f"reached a private attribute: {name}"
            )

    def test_the_names_this_application_uses_are_all_advertised(self):
        import serpentos

        for name in ("ActionValidator", "Decision", "DecisionContext", "DecisionEngine",
                     "Outcome", "Policy", "compare", "replay_all"):
            self.assertIn(name, serpentos.__all__, f"{name} is used here but not advertised")


class SdkLifecycleTest(unittest.TestCase):
    """The whole round trip, as an external integration would perform it."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.audit_path = os.path.join(self.directory, "decisions.jsonl")
        self.contexts = [
            DecisionContext(charge, request_id=f"charge-{index:03d}")
            for index, charge in enumerate(CHARGES)
        ]

    def test_decide_execute_report_audit_replay_compare(self):
        policy = gateway_rules()
        memory = InMemoryAuditLog(redact=["amount_cents"])
        engine = DecisionEngine(
            policy=policy,
            validator=ActionValidator(ACTIONS),
            audit_sink=memory,
        )

        # 1. Decide, and prove the engine only proposes.
        outcomes = []
        for context, charge in zip(self.contexts, CHARGES):
            decision = engine.decide(context)
            self.assertIsInstance(decision, Decision)
            self.assertIn(decision.action, ACTIONS)
            self.assertTrue(decision.decision_id)
            self.assertEqual(decision.policy_name, "decline-handling")

            # 2. The application executes. Nothing before this line did.
            outcome = execute(decision.action, charge)
            outcomes.append((decision, outcome))

        self.assertEqual(
            [decision.action for decision, _ in outcomes],
            ["retry_now", "retry_later", "request_new_card", "abandon"],
        )

        # 3. Outcomes are the application's to report, in whatever shape fits.
        settled = [outcome for _, outcome in outcomes if outcome.success]
        self.assertEqual(len(settled), 2)
        self.assertIn("latency_ms", outcomes[0][1].metrics)

        # 4. Every decision was audited, and the sensitive field is gone.
        self.assertEqual(len(memory.records), len(CHARGES))
        first = memory.records[0]
        self.assertEqual(first.request_id, "charge-000")
        self.assertEqual(first.context["amount_cents"], "[REDACTED]")
        self.assertEqual(first.context["decline_code"], "issuer_unavailable")
        self.assertTrue(first.validation_result.valid)

        # 5. Replay proves the policy is unchanged, then prices a proposed edit.
        unchanged = replay_all(policy, memory.records)
        self.assertEqual(unchanged.matched, len(CHARGES))
        self.assertTrue(unchanged.guaranteed)

        # The question an operator actually asks: if we gave up after two
        # attempts instead of four, which of yesterday's charges change?
        stricter = gateway_rules(version="2.0", max_attempts=2)
        changed = replay_all(stricter, memory.records, strict=False)
        self.assertEqual(changed.mismatched, 1)
        altered = [result for result in changed.results if not result.match]
        self.assertEqual(altered[0].original_action, "retry_later")
        self.assertEqual(altered[0].replayed_action, "abandon")

        # 6. Compare the two candidate policies over the same charges.
        report = compare(
            [policy, gateway_scoring()], self.contexts, validator=ActionValidator(ACTIONS)
        )
        self.assertEqual(report.cases, len(CHARGES))
        for policy_report in report.reports:
            self.assertEqual(policy_report.errors, 0)
            self.assertEqual(policy_report.validation_failures, 0)
        self.assertIsNotNone(report.disagreements)
        self.assertEqual(
            report.disagreements.compared, len(CHARGES)
        )
        # The whole report is JSON, so an application can store or ship it.
        json.dumps(report.to_dict())

    def test_outcomes_aggregate_through_the_public_comparison_api(self):
        by_id = {}

        def outcome_fn(context, decision):
            outcome = execute(decision.action, dict(context.values))
            by_id[decision.action] = outcome
            return outcome

        report = compare(
            [gateway_rules(), gateway_scoring()],
            self.contexts,
            validator=ActionValidator(ACTIONS),
            outcome_fn=outcome_fn,
        )
        for policy_report in report.reports:
            summary = policy_report.outcomes
            self.assertEqual(summary.count, len(CHARGES))
            self.assertIn("latency_ms", summary.metrics)
            self.assertGreater(summary.metrics["latency_ms"].count, 0)
            # No score was reported, so none is invented.
            self.assertIsNone(summary.score)

    def test_audit_survives_a_file_round_trip_for_an_external_reader(self):
        sink = JsonlAuditLog(self.audit_path, redact=["amount_cents"])
        engine = DecisionEngine(
            gateway_rules(), validator=ActionValidator(ACTIONS), audit_sink=sink
        )
        for context in self.contexts:
            engine.decide(context)
        sink.close()

        # A different program, reading the file cold.
        records = list(read_jsonl(self.audit_path))
        self.assertEqual(len(records), len(CHARGES))
        self.assertEqual(records[0].context["amount_cents"], "[REDACTED]")

        # And the records replay, which is the reason to keep them.
        self.assertEqual(replay_all(gateway_rules(), records).matched, len(CHARGES))

        # Every persisted line declares its schema version.
        with open(self.audit_path, encoding="utf-8") as handle:
            for line in handle:
                self.assertEqual(json.loads(line)["schema_version"], 1)

    def test_a_forbidden_action_is_refused_loudly(self):
        rogue = RulePolicy(
            name="rogue",
            version="1.0",
            rules=[Rule("refund_everything", when("attempts", "ge", 0))],
            default_action="refund_everything",
        )
        engine = DecisionEngine(rogue, validator=ActionValidator(ACTIONS))
        with self.assertRaises(DecisionValidationError) as caught:
            engine.decide(self.contexts[0])
        self.assertIn("refund_everything", str(caught.exception))
        self.assertFalse(caught.exception.result.valid)

    def test_a_broken_policy_surfaces_as_a_typed_error(self):
        class Broken:
            name = "broken"
            version = "1.0"

            def decide(self, context):
                raise ZeroDivisionError("bad arithmetic in a policy")

        self.assertIsInstance(Broken(), Policy)
        engine = DecisionEngine(Broken(), validator=ActionValidator(ACTIONS))
        with self.assertRaises(PolicyError) as caught:
            engine.decide(self.contexts[0])
        self.assertIsInstance(caught.exception.__cause__, ZeroDivisionError)

    def test_the_engine_never_executed_anything_itself(self):
        # The strongest form of the safety claim: run every decision with no
        # application executor at all, and confirm nothing happened.
        marker = os.path.join(self.directory, "side-effect")
        engine = DecisionEngine(gateway_rules(), validator=ActionValidator(ACTIONS))
        for context in self.contexts:
            engine.decide(context)
        self.assertFalse(os.path.exists(marker))
        self.assertEqual(os.listdir(self.directory), [])


if __name__ == "__main__":
    unittest.main()
