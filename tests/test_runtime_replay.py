"""Tests for deterministic replay."""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serpentos.runtime.audit import AuditRecord, InMemoryAuditLog
from serpentos.runtime.engine import DecisionEngine
from serpentos.runtime.errors import ReplayError
from serpentos.runtime.models import DecisionContext
from serpentos.runtime.policy import BasePolicy
from serpentos.runtime.replay import replay, replay_all
from serpentos.runtime.validation import ValidationResult


class ThresholdPolicy(BasePolicy):
    """Deterministic: retries below a threshold, fails at or above it."""

    def __init__(self, threshold=3, version="1.0"):
        super().__init__("threshold", version)
        self.threshold = threshold

    def decide(self, context):
        attempts = context.get("attempts", 0)
        return self.decision("retry" if attempts < self.threshold else "fail")


class CoinFlipPolicy(BasePolicy):
    deterministic = False

    def __init__(self):
        super().__init__("coinflip", "1.0")
        self.rng = random.Random()

    def decide(self, context):
        return self.decision("heads" if self.rng.random() < 0.5 else "tails")


class BrokenPolicy(BasePolicy):
    def __init__(self):
        super().__init__("threshold", "1.0")

    def decide(self, context):
        raise RuntimeError("no")


def record_for(policy, context):
    audit = InMemoryAuditLog()
    DecisionEngine(policy, audit_sink=audit).decide(context)
    return audit.records[0]


class ReplayTest(unittest.TestCase):
    def test_same_policy_same_context_matches(self):
        policy = ThresholdPolicy()
        record = record_for(policy, DecisionContext({"attempts": 1}))
        result = replay(policy, record)
        self.assertTrue(result.match)
        self.assertTrue(result.guaranteed)
        self.assertEqual(result.original_action, "retry")
        self.assertEqual(result.replayed_action, "retry")
        self.assertEqual(result.decision_id, record.decision_id)

    def test_a_changed_policy_is_detected_as_a_mismatch(self):
        record = record_for(ThresholdPolicy(threshold=3), DecisionContext({"attempts": 2}))
        result = replay(ThresholdPolicy(threshold=1), record)
        self.assertFalse(result.match)
        self.assertEqual(result.original_action, "retry")
        self.assertEqual(result.replayed_action, "fail")

    def test_replay_does_not_mutate_the_recorded_context(self):
        policy = ThresholdPolicy()
        record = record_for(policy, DecisionContext({"attempts": 1, "nested": {"a": [1]}}))
        before = record.context.to_json()
        replay(policy, record)
        self.assertEqual(record.context.to_json(), before)

    def test_a_different_policy_is_refused_in_strict_mode(self):
        record = record_for(ThresholdPolicy(), DecisionContext({"attempts": 1}))

        class Other(BasePolicy):
            def __init__(self):
                super().__init__("other", "1.0")

            def decide(self, context):
                return self.decision("retry")

        with self.assertRaises(ReplayError) as caught:
            replay(Other(), record)
        self.assertIn("strict=False", str(caught.exception))

    def test_a_version_bump_is_refused_in_strict_mode(self):
        record = record_for(ThresholdPolicy(version="1.0"), DecisionContext({"attempts": 1}))
        with self.assertRaises(ReplayError):
            replay(ThresholdPolicy(version="2.0"), record)

    def test_non_strict_mode_replays_a_candidate_policy(self):
        record = record_for(ThresholdPolicy(threshold=3), DecisionContext({"attempts": 2}))
        result = replay(ThresholdPolicy(threshold=1, version="2.0"), record, strict=False)
        self.assertFalse(result.match)
        self.assertEqual(result.policy_version, "2.0")

    def test_a_record_without_a_context_cannot_be_replayed(self):
        audit = InMemoryAuditLog(include_context=False)
        policy = ThresholdPolicy()
        DecisionEngine(policy, audit_sink=audit).decide(DecisionContext({"attempts": 1}))
        with self.assertRaises(ReplayError) as caught:
            replay(policy, audit.records[0])
        self.assertIn("nothing to replay", str(caught.exception))

    def test_a_failing_policy_raises_replay_error(self):
        record = record_for(ThresholdPolicy(), DecisionContext({"attempts": 1}))
        with self.assertRaises(ReplayError) as caught:
            replay(BrokenPolicy(), record)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_nondeterministic_policies_are_never_certified(self):
        policy = CoinFlipPolicy()
        record = record_for(policy, DecisionContext())
        result = replay(policy, record)
        self.assertFalse(result.guaranteed)

    def test_result_serialises(self):
        policy = ThresholdPolicy()
        payload = replay(policy, record_for(policy, DecisionContext())).to_dict()
        self.assertEqual(payload["match"], True)
        self.assertEqual(payload["policy_name"], "threshold")


class ReplayAllTest(unittest.TestCase):
    def setUp(self):
        self.policy = ThresholdPolicy(threshold=3)
        self.audit = InMemoryAuditLog()
        engine = DecisionEngine(self.policy, audit_sink=self.audit)
        for attempts in range(5):
            engine.decide(DecisionContext({"attempts": attempts}))

    def test_all_records_match_the_unchanged_policy(self):
        report = replay_all(self.policy, self.audit.records)
        self.assertEqual(report.total, 5)
        self.assertEqual(report.matched, 5)
        self.assertEqual(report.mismatched, 0)
        self.assertTrue(report.guaranteed)

    def test_a_threshold_change_shows_up_as_partial_mismatch(self):
        # Originally attempts 0-2 retried; with threshold 1 only attempt 0 does.
        report = replay_all(ThresholdPolicy(threshold=1), self.audit.records)
        self.assertEqual(report.matched, 3)
        self.assertEqual(report.mismatched, 2)

    def test_one_bad_record_does_not_abort_the_run(self):
        broken = AuditRecord(
            decision_id="broken",
            timestamp="t",
            policy_name="threshold",
            policy_version="1.0",
            action="retry",
            context=None,
            validation_result=ValidationResult.accepted("v"),
        )
        report = replay_all(self.policy, list(self.audit.records) + [broken])
        self.assertEqual(report.matched, 5)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.errors[0][0], "broken")
        self.assertEqual(report.total, 6)

    def test_non_record_input_is_refused(self):
        with self.assertRaises(ReplayError):
            replay_all(self.policy, [{"decision_id": "x"}])

    def test_report_serialises(self):
        payload = replay_all(self.policy, self.audit.records).to_dict()
        self.assertEqual(payload["matched"], 5)
        self.assertEqual(len(payload["results"]), 5)


if __name__ == "__main__":
    unittest.main()
