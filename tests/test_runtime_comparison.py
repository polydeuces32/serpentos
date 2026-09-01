"""Tests for offline policy comparison."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serpentos.runtime.comparison import compare
from serpentos.runtime.errors import ConfigurationError
from serpentos.runtime.models import DecisionContext, Outcome
from serpentos.runtime.policy import BasePolicy
from serpentos.runtime.validation import ActionValidator

CASES = [DecisionContext({"attempts": n}) for n in range(5)]


class ThresholdPolicy(BasePolicy):
    def __init__(self, threshold, name):
        super().__init__(name, "1.0")
        self.threshold = threshold
        self.calls = 0

    def decide(self, context):
        self.calls += 1
        return self.decision("retry" if context["attempts"] < self.threshold else "fail")


class SometimesBrokenPolicy(BasePolicy):
    def __init__(self):
        super().__init__("flaky", "1.0")

    def decide(self, context):
        if context["attempts"] == 2:
            raise RuntimeError("bad input")
        return self.decision("retry")


class NotADecisionPolicy(BasePolicy):
    def __init__(self):
        super().__init__("wrong", "1.0")

    def decide(self, context):
        return "retry"


class RogueActionPolicy(BasePolicy):
    def __init__(self):
        super().__init__("rogue", "1.0")

    def decide(self, context):
        return self.decision("drop-table")


class ComparisonTest(unittest.TestCase):
    def test_counts_actions_per_policy(self):
        report = compare([ThresholdPolicy(2, "a"), ThresholdPolicy(4, "b")], CASES)
        self.assertEqual(report.cases, 5)
        self.assertEqual(report.for_policy("a").action_counts, {"retry": 2, "fail": 3})
        self.assertEqual(report.for_policy("b").action_counts, {"retry": 4, "fail": 1})

    def test_every_policy_sees_every_case(self):
        policy = ThresholdPolicy(2, "a")
        compare([policy], CASES)
        self.assertEqual(policy.calls, 5)

    def test_unknown_policy_lookup_raises(self):
        report = compare([ThresholdPolicy(2, "a")], CASES)
        with self.assertRaises(KeyError):
            report.for_policy("nope")

    def test_exceptions_are_isolated_to_one_policy(self):
        report = compare([SometimesBrokenPolicy(), ThresholdPolicy(9, "healthy")], CASES)
        flaky = report.for_policy("flaky")
        self.assertEqual(flaky.errors, 1)
        self.assertEqual(flaky.decisions, 4)
        self.assertIn("RuntimeError", flaky.error_samples[0])
        self.assertEqual(report.for_policy("healthy").errors, 0)

    def test_a_non_decision_return_is_an_error_not_a_crash(self):
        report = compare([NotADecisionPolicy()], CASES)
        self.assertEqual(report.for_policy("wrong").errors, 5)
        self.assertEqual(report.for_policy("wrong").decisions, 0)

    def test_error_samples_are_capped(self):
        report = compare([NotADecisionPolicy()], CASES, max_error_samples=2)
        self.assertEqual(len(report.for_policy("wrong").error_samples), 2)
        self.assertEqual(report.for_policy("wrong").errors, 5)

    def test_validation_failures_are_counted_not_raised(self):
        report = compare(
            [RogueActionPolicy(), ThresholdPolicy(9, "good")],
            CASES,
            validator=ActionValidator({"retry", "fail"}),
        )
        self.assertEqual(report.for_policy("rogue").validation_failures, 5)
        self.assertEqual(report.for_policy("rogue").decisions, 5)
        self.assertEqual(report.for_policy("good").validation_failures, 0)

    def test_evaluation_does_not_mutate_the_policies(self):
        policy = ThresholdPolicy(2, "a")
        before = (policy.name, policy.version, policy.threshold)
        compare([policy], CASES)
        self.assertEqual((policy.name, policy.version, policy.threshold), before)

    def test_contexts_are_not_mutated(self):
        before = [context.to_json() for context in CASES]
        compare([ThresholdPolicy(2, "a")], CASES)
        self.assertEqual([context.to_json() for context in CASES], before)

    def test_report_serialises_deterministically(self):
        report = compare([ThresholdPolicy(2, "a")], CASES)
        payload = report.to_dict()
        self.assertEqual(list(payload["policies"][0]["action_counts"]), ["fail", "retry"])
        self.assertEqual(payload["cases"], 5)

    def test_accepts_a_generator_of_cases(self):
        report = compare([ThresholdPolicy(2, "a")], (case for case in CASES))
        self.assertEqual(report.cases, 5)


class ComparisonOutcomeTest(unittest.TestCase):
    @staticmethod
    def outcome_fn(context, decision):
        return Outcome(
            success=decision.action == "retry",
            metrics={"cost": 1.0 if decision.action == "retry" else 5.0},
        )

    def test_outcomes_are_aggregated_when_supplied(self):
        report = compare([ThresholdPolicy(2, "a")], CASES, outcome_fn=self.outcome_fn)
        outcomes = report.for_policy("a").outcomes
        self.assertEqual(outcomes.count, 5)
        self.assertEqual(outcomes.successes, 2)
        self.assertAlmostEqual(outcomes.success_rate, 0.4)
        cost = outcomes.metrics["cost"]
        self.assertEqual(cost.count, 5)
        self.assertEqual(cost.total, 17.0)
        self.assertAlmostEqual(cost.mean, 3.4)
        self.assertEqual(cost.minimum, 1.0)
        self.assertEqual(cost.maximum, 5.0)

    def test_no_outcome_function_means_no_outcome_section(self):
        report = compare([ThresholdPolicy(2, "a")], CASES)
        self.assertIsNone(report.for_policy("a").outcomes)

    def test_returning_none_skips_a_case(self):
        report = compare(
            [ThresholdPolicy(2, "a")],
            CASES,
            outcome_fn=lambda context, decision: None if context["attempts"] else Outcome(True),
        )
        self.assertEqual(report.for_policy("a").outcomes.count, 1)

    def test_a_failing_outcome_function_is_isolated(self):
        def boom(context, decision):
            raise ValueError("nope")

        report = compare([ThresholdPolicy(2, "a")], CASES, outcome_fn=boom)
        self.assertEqual(report.for_policy("a").errors, 5)
        self.assertIn("outcome_fn raised", report.for_policy("a").error_samples[0])

    def test_a_wrong_outcome_type_is_a_configuration_error(self):
        with self.assertRaises(ConfigurationError):
            compare([ThresholdPolicy(2, "a")], CASES, outcome_fn=lambda c, d: "great")

    def test_outcome_summary_serialises(self):
        report = compare([ThresholdPolicy(2, "a")], CASES, outcome_fn=self.outcome_fn)
        payload = report.to_dict()["policies"][0]["outcomes"]
        self.assertEqual(payload["successes"], 2)
        self.assertEqual(payload["metrics"]["cost"]["count"], 5)


class ParityPolicy(BasePolicy):
    """Retries on half the cases — which half depends on ``flip``."""

    def __init__(self, name, flip):
        super().__init__(name, "1.0")
        self.flip = flip

    def decide(self, context):
        even = context["attempts"] % 2 == 0
        return self.decision("retry" if even != self.flip else "fail")


class DisagreementTest(unittest.TestCase):
    def test_a_single_policy_has_nothing_to_disagree_with(self):
        report = compare([ThresholdPolicy(2, "a")], CASES)
        self.assertIsNone(report.disagreements)

    def test_identical_policies_agree_on_everything(self):
        report = compare([ThresholdPolicy(2, "a"), ThresholdPolicy(2, "b")], CASES)
        summary = report.disagreements
        self.assertEqual(summary.compared, 5)
        self.assertEqual(summary.agreed, 5)
        self.assertEqual(summary.disagreed, 0)
        self.assertEqual(summary.agreement_rate, 1.0)
        self.assertEqual(summary.examples, ())

    def test_identical_distributions_can_still_disagree_on_every_case(self):
        # The whole reason disagreement is reported separately: both policies
        # retry on exactly half the cases, and never on the same one.
        cases = [DecisionContext({"attempts": n}) for n in range(6)]
        report = compare([ParityPolicy("a", False), ParityPolicy("b", True)], cases)
        self.assertEqual(
            report.for_policy("a").action_counts, report.for_policy("b").action_counts
        )
        summary = report.disagreements
        self.assertEqual(summary.disagreed, 6)
        self.assertEqual(summary.agreed, 0)
        self.assertEqual(summary.agreement_rate, 0.0)

    def test_partial_disagreement_is_counted_and_illustrated(self):
        report = compare([ThresholdPolicy(2, "a"), ThresholdPolicy(4, "b")], CASES)
        summary = report.disagreements
        self.assertEqual(summary.compared, 5)
        self.assertEqual(summary.agreed, 3)
        self.assertEqual(summary.disagreed, 2)
        self.assertEqual(
            [example.case_index for example in summary.examples], [2, 3]
        )
        self.assertEqual(summary.examples[0].actions, {"a": "fail", "b": "retry"})

    def test_pairs_are_reported_for_every_combination(self):
        report = compare(
            [ThresholdPolicy(2, "a"), ThresholdPolicy(4, "b"), ThresholdPolicy(2, "c")],
            CASES,
        )
        pairs = {(pair.first, pair.second): pair for pair in report.disagreements.pairs}
        self.assertEqual(set(pairs), {("a", "b"), ("a", "c"), ("b", "c")})
        self.assertEqual(pairs[("a", "c")].disagreed, 0)
        self.assertEqual(pairs[("a", "b")].disagreed, 2)
        self.assertAlmostEqual(pairs[("a", "b")].rate, 0.4)

    def test_a_case_a_policy_errored_on_is_not_a_disagreement(self):
        report = compare([ThresholdPolicy(9, "a"), SometimesBrokenPolicy()], CASES)
        summary = report.disagreements
        # attempts == 2 blew up, so only four cases can be compared at all.
        self.assertEqual(summary.compared, 4)
        self.assertEqual(summary.agreed, 4)
        self.assertEqual(summary.disagreed, 0)
        self.assertEqual(summary.pairs[0].compared, 4)

    def test_examples_are_capped_but_counts_are_not(self):
        report = compare(
            [ParityPolicy("a", False), ParityPolicy("b", True)],
            CASES,
            max_disagreement_samples=2,
        )
        self.assertEqual(report.disagreements.disagreed, 5)
        self.assertEqual(len(report.disagreements.examples), 2)

    def test_policies_sharing_a_name_are_distinguished_by_version(self):
        first = ThresholdPolicy(2, "retry")
        second = ThresholdPolicy(4, "retry")
        second.version = "2.0"
        report = compare([first, second], CASES)
        self.assertEqual(
            [item.label for item in report.reports], ["retry@1.0", "retry@2.0"]
        )
        self.assertEqual(
            set(report.disagreements.examples[0].actions), {"retry@1.0", "retry@2.0"}
        )
        self.assertEqual(report.for_policy("retry@2.0").policy_version, "2.0")

    def test_indistinguishable_policies_still_get_unique_labels(self):
        report = compare([ThresholdPolicy(2, "same"), ThresholdPolicy(4, "same")], CASES)
        labels = [item.label for item in report.reports]
        self.assertEqual(len(set(labels)), 2)

    def test_summary_serialises(self):
        report = compare([ThresholdPolicy(2, "a"), ThresholdPolicy(4, "b")], CASES)
        payload = report.to_dict()["disagreements"]
        self.assertEqual(payload["disagreed"], 2)
        self.assertEqual(payload["agreement_rate"], 0.6)
        self.assertEqual(payload["pairs"][0]["first"], "a")
        self.assertEqual(payload["examples"][0]["actions"], {"a": "fail", "b": "retry"})

    def test_rejects_a_negative_example_cap(self):
        with self.assertRaises(ConfigurationError):
            compare([ThresholdPolicy(2, "a")], CASES, max_disagreement_samples=-1)


class ComparisonScoreTest(unittest.TestCase):
    def test_scores_are_aggregated_separately_from_metrics(self):
        report = compare(
            [ThresholdPolicy(2, "a")],
            CASES,
            outcome_fn=lambda context, decision: Outcome(
                decision.action == "retry",
                score=1.0 if decision.action == "retry" else -0.5,
            ),
        )
        score = report.for_policy("a").outcomes.score
        self.assertEqual(score.count, 5)
        self.assertAlmostEqual(score.total, 0.5)
        self.assertAlmostEqual(score.mean, 0.1)
        self.assertEqual(score.minimum, -0.5)
        self.assertEqual(score.maximum, 1.0)

    def test_no_score_means_no_score_summary(self):
        report = compare(
            [ThresholdPolicy(2, "a")],
            CASES,
            outcome_fn=lambda context, decision: Outcome(True),
        )
        self.assertIsNone(report.for_policy("a").outcomes.score)
        self.assertIsNone(report.to_dict()["policies"][0]["outcomes"]["score"])


class ComparisonArgumentTest(unittest.TestCase):
    def test_needs_at_least_one_policy(self):
        with self.assertRaises(ConfigurationError):
            compare([], CASES)

    def test_rejects_a_non_policy(self):
        with self.assertRaises(ConfigurationError):
            compare([object()], CASES)

    def test_rejects_non_context_cases(self):
        with self.assertRaises(ConfigurationError) as caught:
            compare([ThresholdPolicy(2, "a")], [{"attempts": 1}])
        self.assertIn("cases[0]", str(caught.exception))

    def test_rejects_a_bad_validator(self):
        with self.assertRaises(ConfigurationError):
            compare([ThresholdPolicy(2, "a")], CASES, validator=object())

    def test_rejects_a_negative_sample_cap(self):
        with self.assertRaises(ConfigurationError):
            compare([ThresholdPolicy(2, "a")], CASES, max_error_samples=-1)


if __name__ == "__main__":
    unittest.main()
