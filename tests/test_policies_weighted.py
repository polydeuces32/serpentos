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

    def test_source_survives_the_data_round_trip(self):
        spec = {"ups": {"weights": {"cost": -1.0}, "source": "ups"}}
        policy = WeightedPolicy.from_linear("ship", "1.0", json.loads(json.dumps(spec)))
        self.assertEqual(policy.to_dict()["scorers"]["ups"]["source"], "ups")

    def test_a_policy_round_trips_through_json(self):
        policy = WeightedPolicy.from_linear(
            "router", "1.0", self.SPEC, minimum_score=0.5, default_action="secondary"
        )
        rebuilt = WeightedPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
        self.assertEqual(rebuilt.to_dict(), policy.to_dict())
        context = DecisionContext({"primary_healthy": 1})
        self.assertEqual(rebuilt.decide(context).action, policy.decide(context).action)

    def test_a_plain_callable_cannot_be_serialised(self):
        policy = WeightedPolicy("p", "1.0", {"go": lambda values: 1.0})
        with self.assertRaises(ConfigurationError):
            policy.to_dict()

    def test_malformed_payloads_are_refused(self):
        for payload in ("nope", {}, {"name": "p", "version": "1.0"}):
            with self.assertRaises(ConfigurationError):
                WeightedPolicy.from_dict(payload)


class ExplanationTest(unittest.TestCase):
    """Per-factor breakdowns: why this candidate, and by how much."""

    def test_explain_attributes_the_score_to_each_factor(self):
        scorer = LinearScorer({"a": 2.0, "b": -1.0}, bias=10.0)
        self.assertEqual(scorer.explain({"a": 3, "b": 4}), {"bias": 10.0, "a": 6.0, "b": -4.0})

    def test_contributions_always_sum_to_the_score(self):
        scorer = LinearScorer({"a": 2.5, "b": -1.25, "c": 0.5}, bias=-3.0)
        values = {"a": 4, "b": 8, "c": 2}
        self.assertAlmostEqual(sum(scorer.explain(values).values()), scorer(values))

    def test_a_zero_bias_is_left_out_of_the_breakdown(self):
        self.assertNotIn("bias", LinearScorer({"a": 1.0}).explain({"a": 1}))

    def test_missing_factors_are_absent_rather_than_zero(self):
        # Absent and "contributed nothing" are different claims.
        self.assertEqual(LinearScorer({"a": 1.0, "b": 1.0}).explain({"a": 2}), {"a": 2.0})

    def test_a_non_numeric_factor_is_an_error_not_a_zero(self):
        with self.assertRaises(PolicyError):
            LinearScorer({"a": 1.0}).explain({"a": "cheap"})

    def test_source_reads_factors_from_a_nested_object(self):
        scorer = LinearScorer({"cost": -1.0, "days": -3.0}, source="ups")
        quote = {"ups": {"cost": 12.5, "days": 2}, "fedex": {"cost": 18.0, "days": 1}}
        self.assertEqual(scorer(quote), -18.5)
        self.assertEqual(scorer.explain(quote), {"cost": -12.5, "days": -6.0})

    def test_a_missing_source_contributes_only_the_bias(self):
        scorer = LinearScorer({"cost": -1.0}, bias=2.0, source="dhl")
        self.assertEqual(scorer({"ups": {"cost": 1.0}}), 2.0)

    def test_a_source_that_is_not_an_object_is_an_error(self):
        with self.assertRaises(PolicyError):
            LinearScorer({"cost": -1.0}, source="ups")({"ups": 12.5})

    def test_an_empty_source_name_is_refused(self):
        for source in ("", 7):
            with self.assertRaises(ConfigurationError):
                LinearScorer({"cost": 1.0}, source=source)

    def test_the_decision_explains_every_candidate(self):
        # The shipping-carrier problem: same factors, different quotes.
        spec = {
            carrier: {
                "weights": {"cost_usd": -1.0, "days": -4.0, "reliability": 30.0},
                "source": carrier,
            }
            for carrier in ("ups", "fedex", "usps")
        }
        policy = WeightedPolicy.from_linear("carrier", "1.0", spec)
        decision = policy.decide(
            DecisionContext(
                {
                    "ups": {"cost_usd": 12.50, "days": 2, "reliability": 0.97},
                    "fedex": {"cost_usd": 18.00, "days": 1, "reliability": 0.99},
                    "usps": {"cost_usd": 8.25, "days": 4, "reliability": 0.91},
                }
            )
        )
        self.assertEqual(decision.action, "ups")
        factors = decision.metadata["factors"]
        self.assertEqual(set(factors), {"ups", "fedex", "usps"})
        self.assertEqual(factors["ups"]["days"], -8.0)
        self.assertAlmostEqual(factors["fedex"]["reliability"], 29.7)
        # The breakdown must reconcile with the headline score, or it is a lie.
        for carrier, breakdown in factors.items():
            self.assertAlmostEqual(
                sum(breakdown.values()), decision.metadata["scores"][carrier], places=5
            )

    def test_metadata_stays_json_serialisable(self):
        policy = WeightedPolicy.from_linear(
            "p", "1.0", {"go": {"weights": {"x": 1.5}, "bias": 1.0}}
        )
        payload = json.loads(policy.decide(DecisionContext({"x": 2})).to_json())
        self.assertEqual(payload["metadata"]["factors"], {"go": {"bias": 1.0, "x": 3.0}})

    def test_plain_callables_score_without_a_breakdown(self):
        policy = WeightedPolicy("p", "1.0", {"go": lambda values: 1.0})
        self.assertNotIn("factors", policy.decide(DecisionContext({})).metadata)

    def test_a_scorer_explaining_itself_as_nonsense_is_a_policy_error(self):
        class Bad:
            def explain(self, values):
                return "lots"

            def __call__(self, values):
                return 1.0

        with self.assertRaises(PolicyError):
            WeightedPolicy("p", "1.0", {"go": Bad()}).decide(DecisionContext({}))

    def test_a_raising_explain_is_retyped(self):
        class Bad:
            def explain(self, values):
                raise KeyError("missing")

            def __call__(self, values):
                return 1.0

        with self.assertRaises(PolicyError):
            WeightedPolicy("p", "1.0", {"go": Bad()}).decide(DecisionContext({}))


if __name__ == "__main__":
    unittest.main()
