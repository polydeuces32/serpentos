"""Tests for the weighted-scoring policy."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serpentos.policies.weighted import LinearScorer, WeightedPolicy
from serpentos.runtime.errors import ConfigurationError, PolicyError
from serpentos.runtime.models import DecisionContext


class LinearScorerTest(unittest.TestCase):
    def test_scores_a_weighted_sum(self):
        scorer = LinearScorer({"a": 2.0, "b": -1.0}, bias=10.0)
        self.assertEqual(scorer({"a": 3, "b": 4}), 12.0)

    def test_missing_keys_contribute_nothing(self):
        self.assertEqual(LinearScorer({"a": 2.0}, bias=1.0)({}), 1.0)

    def test_booleans_count_as_one_and_zero(self):
        scorer = LinearScorer({"healthy": 5.0})
        self.assertEqual(scorer({"healthy": True}), 5.0)
        self.assertEqual(scorer({"healthy": False}), 0.0)

    def test_non_numeric_values_are_a_policy_error(self):
        with self.assertRaises(PolicyError) as caught:
            LinearScorer({"a": 1.0})({"a": "three"})
        self.assertIn("expected a number", str(caught.exception))

    def test_configuration_is_validated(self):
        for weights, bias in (
            ({"a": "1"}, 0.0),
            ({"a": float("inf")}, 0.0),
            ({"": 1.0}, 0.0),
            ({}, float("nan")),
            ("not a mapping", 0.0),
        ):
            with self.assertRaises(ConfigurationError):
                LinearScorer(weights, bias)

    def test_serialises_as_data(self):
        scorer = LinearScorer({"b": 1.0, "a": 2.0}, bias=3.0)
        payload = json.loads(json.dumps(scorer.to_dict()))
        self.assertEqual(payload, {"bias": 3.0, "weights": {"a": 2.0, "b": 1.0}})


class WeightedPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = WeightedPolicy(
            name="router",
            version="1.0",
            scorers={
                "primary": LinearScorer({"primary_healthy": 10.0}),
                "secondary": LinearScorer({"secondary_healthy": 8.0}, bias=1.0),
            },
        )

    def test_highest_score_wins(self):
        self.assertEqual(
            self.policy.decide(DecisionContext({"primary_healthy": 1})).action, "primary"
        )
        self.assertEqual(
            self.policy.decide(DecisionContext({"secondary_healthy": 1})).action, "secondary"
        )

    def test_metadata_explains_the_choice(self):
        decision = self.policy.decide(DecisionContext({"primary_healthy": 1}))
        self.assertEqual(dict(decision.metadata["scores"]), {"primary": 10.0, "secondary": 1.0})
        self.assertEqual(decision.metadata["chosen_score"], 10.0)
        self.assertEqual(decision.metadata["reason"], "highest-score")

    def test_ties_break_by_declaration_order(self):
        policy = WeightedPolicy(
            name="tie",
            version="1.0",
            scorers={"first": lambda values: 1.0, "second": lambda values: 1.0},
        )
        self.assertEqual(policy.decide(DecisionContext()).action, "first")

    def test_the_same_context_always_gives_the_same_answer(self):
        context = DecisionContext({"primary_healthy": 1})
        self.assertEqual(
            self.policy.decide(context).to_json(), self.policy.decide(context).to_json()
        )

    def test_minimum_score_filters_candidates(self):
        policy = WeightedPolicy(
            name="gated",
            version="1.0",
            scorers={"go": lambda values: 0.5},
            minimum_score=1.0,
            default_action="stop",
        )
        decision = policy.decide(DecisionContext())
        self.assertEqual(decision.action, "stop")
        self.assertEqual(decision.metadata["reason"], "below-minimum")
        self.assertIsNone(decision.metadata["chosen_score"])

    def test_nothing_eligible_without_a_default_is_an_error(self):
        policy = WeightedPolicy(
            name="gated", version="1.0", scorers={"go": lambda values: 0.5}, minimum_score=1.0
        )
        with self.assertRaises(PolicyError) as caught:
            policy.decide(DecisionContext())
        self.assertIn("minimum_score", str(caught.exception))

    def test_a_raising_scorer_becomes_a_policy_error(self):
        def boom(values):
            raise KeyError("missing")

        policy = WeightedPolicy(name="p", version="1.0", scorers={"go": boom})
        with self.assertRaises(PolicyError) as caught:
            policy.decide(DecisionContext())
        self.assertIsInstance(caught.exception.__cause__, KeyError)

    def test_a_non_numeric_score_is_a_policy_error(self):
        policy = WeightedPolicy(name="p", version="1.0", scorers={"go": lambda values: "high"})
        with self.assertRaises(PolicyError):
            policy.decide(DecisionContext())

    def test_a_non_finite_score_is_a_policy_error(self):
        policy = WeightedPolicy(
            name="p", version="1.0", scorers={"go": lambda values: float("inf")}
        )
        with self.assertRaises(PolicyError) as caught:
            policy.decide(DecisionContext())
        self.assertIn("non-finite", str(caught.exception))

    def test_advertised_actions_include_the_default(self):
        policy = WeightedPolicy(
            name="p", version="1.0", scorers={"go": lambda v: 1.0}, default_action="stop"
        )
        self.assertEqual(policy.actions, ("go", "stop"))

    def test_construction_is_validated(self):
        with self.assertRaises(ConfigurationError):
            WeightedPolicy(name="p", version="1.0", scorers={})
        with self.assertRaises(ConfigurationError):
            WeightedPolicy(name="p", version="1.0", scorers={"go": "not callable"})
        with self.assertRaises(ConfigurationError):
            WeightedPolicy(
                name="p", version="1.0", scorers={"go": lambda v: 1.0}, minimum_score="high"
            )
        with self.assertRaises(ConfigurationError):
            WeightedPolicy(
                name="p", version="1.0", scorers={"go": lambda v: 1.0}, default_action=""
            )

    def test_does_not_mutate_the_context(self):
        context = DecisionContext({"primary_healthy": 1})
        before = context.to_json()
        self.policy.decide(context)
        self.assertEqual(context.to_json(), before)


class FromLinearTest(unittest.TestCase):
    SPEC = {
        "primary": {"weights": {"primary_healthy": 10.0}, "bias": 0.0},
        "secondary": {"weights": {}, "bias": 1.0},
    }

    def test_builds_an_equivalent_policy_from_data(self):
        policy = WeightedPolicy.from_linear("router", "1.0", json.loads(json.dumps(self.SPEC)))
        self.assertEqual(policy.decide(DecisionContext({"primary_healthy": 1})).action, "primary")
        self.assertEqual(policy.decide(DecisionContext({})).action, "secondary")

    def test_malformed_specs_are_refused(self):
        for spec in ({}, {"a": "weights"}, {"a": {"weights": {"k": "one"}}}):
            with self.assertRaises(ConfigurationError):
                WeightedPolicy.from_linear("p", "1.0", spec)


if __name__ == "__main__":
    unittest.main()
