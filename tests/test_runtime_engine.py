"""Tests for the decision engine."""

import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serpentos.runtime.audit import AuditError, InMemoryAuditLog
from serpentos.runtime.engine import DecisionEngine
from serpentos.runtime.errors import (
    ConfigurationError,
    DecisionValidationError,
    PolicyError,
)
from serpentos.runtime.models import Decision, DecisionContext
from serpentos.runtime.policy import BasePolicy
from serpentos.runtime.validation import ActionValidator, ValidationResult


class FixedPolicy(BasePolicy):
    def __init__(self, action="go", name="fixed", version="1.0"):
        super().__init__(name, version)
        self.action = action
        self.calls = 0

    def decide(self, context):
        self.calls += 1
        return self.decision(self.action, {"seen": sorted(context.values)})


class ExplodingPolicy(BasePolicy):
    def __init__(self, exc=None):
        super().__init__("exploding", "1.0")
        self.exc = exc or ValueError("kaboom")

    def decide(self, context):
        raise self.exc


class WrongTypePolicy(BasePolicy):
    def __init__(self):
        super().__init__("wrong", "1.0")

    def decide(self, context):
        return "go"


def counting_ids():
    counter = itertools.count(1)
    return lambda: f"id-{next(counter)}"


class EngineBasicsTest(unittest.TestCase):
    def test_returns_the_policy_decision(self):
        engine = DecisionEngine(FixedPolicy("retry"))
        decision = engine.decide(DecisionContext({"a": 1}))
        self.assertEqual(decision.action, "retry")
        self.assertEqual(decision.policy_name, "fixed")
        self.assertEqual(decision.policy_version, "1.0")

    def test_assigns_a_decision_id(self):
        engine = DecisionEngine(FixedPolicy(), id_factory=counting_ids())
        self.assertEqual(engine.decide(DecisionContext()).decision_id, "id-1")
        self.assertEqual(engine.decide(DecisionContext()).decision_id, "id-2")

    def test_propagates_an_id_the_policy_already_set(self):
        class PreIdentified(BasePolicy):
            def __init__(self):
                super().__init__("pre", "1.0")

            def decide(self, context):
                return Decision("go", self.name, self.version, decision_id="chosen")

        engine = DecisionEngine(PreIdentified(), id_factory=counting_ids())
        self.assertEqual(engine.decide(DecisionContext()).decision_id, "chosen")

    def test_does_not_mutate_the_context(self):
        engine = DecisionEngine(FixedPolicy())
        context = DecisionContext({"a": [1, 2]})
        before = context.to_json()
        engine.decide(context)
        self.assertEqual(context.to_json(), before)

    def test_rejects_a_non_context_argument(self):
        engine = DecisionEngine(FixedPolicy())
        with self.assertRaises(ConfigurationError):
            engine.decide({"a": 1})

    def test_rejects_components_that_do_not_fit(self):
        with self.assertRaises(ConfigurationError):
            DecisionEngine(object())
        with self.assertRaises(ConfigurationError):
            DecisionEngine(FixedPolicy(), validator=object())
        with self.assertRaises(ConfigurationError):
            DecisionEngine(FixedPolicy(), audit_sink=object())
        with self.assertRaises(ConfigurationError):
            DecisionEngine(FixedPolicy(), clock="not callable")


class EngineFailureTest(unittest.TestCase):
    def test_policy_exceptions_become_policy_errors(self):
        engine = DecisionEngine(ExplodingPolicy())
        with self.assertRaises(PolicyError) as caught:
            engine.decide(DecisionContext())
        self.assertIn("ValueError", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_a_policy_error_is_not_double_wrapped(self):
        original = PolicyError("mine")
        engine = DecisionEngine(ExplodingPolicy(original))
        with self.assertRaises(PolicyError) as caught:
            engine.decide(DecisionContext())
        self.assertIs(caught.exception, original)

    def test_non_decision_return_is_a_policy_error(self):
        engine = DecisionEngine(WrongTypePolicy())
        with self.assertRaises(PolicyError) as caught:
            engine.decide(DecisionContext())
        self.assertIn("returned str", str(caught.exception))

    def test_validator_returning_the_wrong_type_is_a_configuration_error(self):
        class BadValidator:
            name = "bad"

            def validate(self, decision, context):
                return True

        engine = DecisionEngine(FixedPolicy(), validator=BadValidator())
        with self.assertRaises(ConfigurationError):
            engine.decide(DecisionContext())

    def test_bad_id_factory_is_a_configuration_error(self):
        engine = DecisionEngine(FixedPolicy(), id_factory=lambda: "")
        with self.assertRaises(ConfigurationError):
            engine.decide(DecisionContext())

    def test_bad_clock_is_a_configuration_error(self):
        engine = DecisionEngine(FixedPolicy(), clock=lambda: 12345)
        with self.assertRaises(ConfigurationError):
            engine.decide(DecisionContext())

    def test_audit_failure_fails_the_decision(self):
        class BrokenSink:
            def record(self, record):
                raise AuditError("disk on fire")

        engine = DecisionEngine(FixedPolicy(), audit_sink=BrokenSink())
        with self.assertRaises(AuditError):
            engine.decide(DecisionContext())


class EngineValidationTest(unittest.TestCase):
    def test_blocked_action_raises_with_the_result_attached(self):
        engine = DecisionEngine(FixedPolicy("nuke"), validator=ActionValidator({"go"}))
        with self.assertRaises(DecisionValidationError) as caught:
            engine.decide(DecisionContext())
        self.assertIsInstance(caught.exception.result, ValidationResult)
        self.assertFalse(caught.exception.result.valid)

    def test_rejections_are_still_audited(self):
        audit = InMemoryAuditLog()
        engine = DecisionEngine(
            FixedPolicy("nuke"), validator=ActionValidator({"go"}), audit_sink=audit
        )
        with self.assertRaises(DecisionValidationError):
            engine.decide(DecisionContext())
        self.assertEqual(len(audit), 1)
        self.assertFalse(audit.records[0].validation_result.valid)
        self.assertEqual(audit.records[0].action, "nuke")

    def test_no_validator_records_that_fact_explicitly(self):
        audit = InMemoryAuditLog()
        DecisionEngine(FixedPolicy(), audit_sink=audit).decide(DecisionContext())
        result = audit.records[0].validation_result
        self.assertTrue(result.valid)
        self.assertEqual(result.validator, "none")

    def test_validator_sees_the_context(self):
        seen = []

        class Spy:
            name = "spy"

            def validate(self, decision, context):
                seen.append((decision.action, dict(context.values)))
                return ValidationResult.accepted("spy")

        DecisionEngine(FixedPolicy("go"), validator=Spy()).decide(DecisionContext({"a": 1}))
        self.assertEqual(seen, [("go", {"a": 1})])


class EngineFallbackTest(unittest.TestCase):
    def test_fallback_is_used_only_after_a_rejection(self):
        primary = FixedPolicy("nuke", name="primary")
        fallback = FixedPolicy("go", name="fallback")
        engine = DecisionEngine(
            primary, validator=ActionValidator({"go"}), fallback_policy=fallback
        )
        decision = engine.decide(DecisionContext())
        self.assertEqual(decision.action, "go")
        self.assertEqual(decision.policy_name, "fallback")
        self.assertEqual(decision.metadata["fallback_for"], "nuke")

    def test_fallback_is_not_consulted_when_the_primary_passes(self):
        fallback = FixedPolicy("go", name="fallback")
        engine = DecisionEngine(
            FixedPolicy("go"), validator=ActionValidator({"go"}), fallback_policy=fallback
        )
        engine.decide(DecisionContext())
        self.assertEqual(fallback.calls, 0)

    def test_both_attempts_are_audited(self):
        audit = InMemoryAuditLog()
        engine = DecisionEngine(
            FixedPolicy("nuke"),
            validator=ActionValidator({"go"}),
            fallback_policy=FixedPolicy("go", name="fallback"),
            audit_sink=audit,
        )
        engine.decide(DecisionContext())
        self.assertEqual([r.action for r in audit.records], ["nuke", "go"])
        self.assertEqual([r.validation_result.valid for r in audit.records], [False, True])

    def test_a_rejected_fallback_still_raises(self):
        engine = DecisionEngine(
            FixedPolicy("nuke"),
            validator=ActionValidator({"go"}),
            fallback_policy=FixedPolicy("also-nuke", name="fallback"),
        )
        with self.assertRaises(DecisionValidationError) as caught:
            engine.decide(DecisionContext())
        self.assertIn("fallback policy", str(caught.exception))

    def test_fallback_decisions_get_their_own_id(self):
        engine = DecisionEngine(
            FixedPolicy("nuke"),
            validator=ActionValidator({"go"}),
            fallback_policy=FixedPolicy("go", name="fallback"),
            id_factory=counting_ids(),
        )
        self.assertEqual(engine.decide(DecisionContext()).decision_id, "id-2")


class EngineAuditContentTest(unittest.TestCase):
    def test_record_captures_the_whole_decision(self):
        audit = InMemoryAuditLog()
        engine = DecisionEngine(
            FixedPolicy("go"),
            validator=ActionValidator({"go"}),
            audit_sink=audit,
            clock=lambda: "2024-01-01T00:00:00+00:00",
            id_factory=counting_ids(),
        )
        engine.decide(DecisionContext({"a": 1}, request_id="req-9"))
        record = audit.records[0]
        self.assertEqual(record.decision_id, "id-1")
        self.assertEqual(record.timestamp, "2024-01-01T00:00:00+00:00")
        self.assertEqual(record.policy_name, "fixed")
        self.assertEqual(record.policy_version, "1.0")
        self.assertEqual(record.action, "go")
        self.assertEqual(record.request_id, "req-9")
        self.assertEqual(dict(record.context.values), {"a": 1})
        self.assertEqual(record.decision_metadata["seen"], ("a",))

    def test_decide_with_record_returns_both(self):
        engine = DecisionEngine(FixedPolicy(), id_factory=counting_ids())
        decision, record = engine.decide_with_record(DecisionContext())
        self.assertEqual(decision.decision_id, record.decision_id)

    def test_no_sink_means_nothing_is_persisted(self):
        engine = DecisionEngine(FixedPolicy())
        self.assertIsNone(engine.audit_sink)
        engine.decide(DecisionContext())


if __name__ == "__main__":
    unittest.main()
