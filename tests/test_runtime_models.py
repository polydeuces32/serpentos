"""Tests for the runtime data models."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serpentos.runtime.errors import ConfigurationError
from serpentos.runtime.models import (
    MAX_DEPTH,
    Decision,
    DecisionContext,
    Outcome,
    freeze_value,
    thaw_value,
    to_canonical_json,
)


class FreezeTest(unittest.TestCase):
    def test_scalars_pass_through(self):
        for value in (None, True, False, 0, -3, 2.5, "text", ""):
            self.assertEqual(freeze_value(value), value)

    def test_bool_does_not_become_int(self):
        self.assertIs(freeze_value(True), True)

    def test_containers_become_immutable(self):
        frozen = freeze_value({"a": [1, {"b": 2}]})
        self.assertEqual(frozen["a"][1]["b"], 2)
        with self.assertRaises(TypeError):
            frozen["a"] = 1
        with self.assertRaises(TypeError):
            frozen["a"][1]["b"] = 3

    def test_rejects_non_json_types(self):
        for value in (object(), {1, 2}, lambda: 1, b"bytes"):
            with self.assertRaises(ConfigurationError):
                freeze_value(value)

    def test_rejects_non_string_keys(self):
        with self.assertRaises(ConfigurationError):
            freeze_value({1: "one"})

    def test_rejects_non_finite_floats(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ConfigurationError):
                freeze_value(value)

    def test_rejects_excessive_nesting(self):
        payload = value = {}
        for _ in range(MAX_DEPTH + 2):
            value["next"] = {}
            value = value["next"]
        with self.assertRaises(ConfigurationError):
            freeze_value(payload)

    def test_error_names_the_offending_path(self):
        with self.assertRaises(ConfigurationError) as caught:
            freeze_value({"outer": {"inner": object()}})
        self.assertIn("outer.inner", str(caught.exception))

    def test_thaw_returns_plain_mutable_copies(self):
        thawed = thaw_value(freeze_value({"a": [1, 2]}))
        self.assertIsInstance(thawed, dict)
        self.assertIsInstance(thawed["a"], list)
        thawed["a"].append(3)


class DecisionContextTest(unittest.TestCase):
    def test_defaults_to_empty(self):
        self.assertEqual(dict(DecisionContext().values), {})

    def test_reads_like_a_mapping(self):
        ctx = DecisionContext({"attempts": 2})
        self.assertEqual(ctx["attempts"], 2)
        self.assertEqual(ctx.get("missing", "fallback"), "fallback")
        self.assertIn("attempts", ctx)
        self.assertEqual(list(ctx), ["attempts"])

    def test_is_frozen(self):
        ctx = DecisionContext({"a": 1})
        with self.assertRaises(Exception):
            ctx.values = {}

    def test_source_dict_mutation_does_not_leak_in(self):
        source = {"a": [1]}
        ctx = DecisionContext(source)
        source["a"].append(2)
        source["b"] = 3
        self.assertEqual(ctx.to_dict()["values"], {"a": [1]})

    def test_to_dict_is_a_detached_copy(self):
        ctx = DecisionContext({"a": [1]})
        snapshot = ctx.to_dict()
        snapshot["values"]["a"].append(2)
        self.assertEqual(ctx.to_dict()["values"], {"a": [1]})

    def test_with_values_does_not_mutate_the_original(self):
        ctx = DecisionContext({"a": 1})
        updated = ctx.with_values(b=2)
        self.assertEqual(dict(ctx.values), {"a": 1})
        self.assertEqual(dict(updated.values), {"a": 1, "b": 2})

    def test_equality_ignores_construction_order(self):
        self.assertEqual(
            DecisionContext({"a": 1, "b": [2, 3]}),
            DecisionContext({"b": [2, 3], "a": 1}),
        )

    def test_canonical_json_is_key_order_independent(self):
        first = DecisionContext({"a": 1, "b": 2}).to_json()
        second = DecisionContext({"b": 2, "a": 1}).to_json()
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["values"], {"a": 1, "b": 2})

    def test_roundtrip(self):
        ctx = DecisionContext({"a": 1, "nested": {"b": [1, 2]}}, request_id="req-1")
        self.assertEqual(DecisionContext.from_dict(ctx.to_dict()), ctx)

    def test_rejects_bad_request_id(self):
        with self.assertRaises(ConfigurationError):
            DecisionContext({}, request_id=7)

    def test_rejects_non_mapping_values(self):
        with self.assertRaises(ConfigurationError):
            DecisionContext([("a", 1)])

    def test_from_dict_rejects_malformed_payloads(self):
        for payload in ("not a dict", {"values": "nope"}, {"values": {}, "request_id": 5}):
            with self.assertRaises(ConfigurationError):
                DecisionContext.from_dict(payload)


class DecisionTest(unittest.TestCase):
    def test_requires_identity(self):
        for kwargs in (
            {"action": "", "policy_name": "p", "policy_version": "1"},
            {"action": "go", "policy_name": "", "policy_version": "1"},
            {"action": "go", "policy_name": "p", "policy_version": ""},
            {"action": 1, "policy_name": "p", "policy_version": "1"},
        ):
            with self.assertRaises(ConfigurationError):
                Decision(**kwargs)

    def test_decision_id_is_absent_until_assigned(self):
        decision = Decision("go", "p", "1")
        self.assertIsNone(decision.decision_id)
        identified = decision.with_decision_id("abc")
        self.assertEqual(identified.decision_id, "abc")
        self.assertIsNone(decision.decision_id)

    def test_metadata_is_frozen(self):
        decision = Decision("go", "p", "1", {"why": {"deep": 1}})
        with self.assertRaises(TypeError):
            decision.metadata["why"]["deep"] = 2

    def test_metadata_rejects_non_json(self):
        with self.assertRaises(ConfigurationError):
            Decision("go", "p", "1", {"fn": len})

    def test_roundtrip(self):
        decision = Decision("go", "p", "1", {"why": "because"}, "id-1")
        self.assertEqual(Decision.from_dict(decision.to_dict()), decision)

    def test_from_dict_requires_the_core_fields(self):
        with self.assertRaises(ConfigurationError):
            Decision.from_dict({"action": "go"})

    def test_canonical_json_is_stable(self):
        first = Decision("go", "p", "1", {"b": 1, "a": 2}).to_json()
        second = Decision("go", "p", "1", {"a": 2, "b": 1}).to_json()
        self.assertEqual(first, second)


class OutcomeTest(unittest.TestCase):
    def test_metrics_are_coerced_to_float(self):
        outcome = Outcome(True, metrics={"latency_ms": 12})
        self.assertEqual(outcome.metrics["latency_ms"], 12.0)
        self.assertIsInstance(outcome.metrics["latency_ms"], float)

    def test_success_must_be_a_bool(self):
        with self.assertRaises(ConfigurationError):
            Outcome(1)

    def test_metrics_must_be_numbers(self):
        for metrics in ({"a": "1"}, {"a": True}, {"a": None}):
            with self.assertRaises(ConfigurationError):
                Outcome(True, metrics=metrics)

    def test_metrics_reject_non_finite(self):
        with self.assertRaises(ConfigurationError):
            Outcome(True, metrics={"a": float("inf")})

    def test_roundtrip(self):
        outcome = Outcome(False, score=-1.0, metrics={"cost": 1.5},
                          metadata={"note": "timeout"}, decision_id="id-1")
        self.assertEqual(Outcome.from_dict(outcome.to_dict()), outcome)

    def test_from_dict_requires_success(self):
        with self.assertRaises(ConfigurationError):
            Outcome.from_dict({"metrics": {}})

    def test_score_is_optional(self):
        outcome = Outcome(True)
        self.assertIsNone(outcome.score)
        self.assertIsNone(outcome.to_dict()["score"])

    def test_score_is_coerced_to_float(self):
        outcome = Outcome(True, score=3)
        self.assertEqual(outcome.score, 3.0)
        self.assertIsInstance(outcome.score, float)

    def test_score_must_be_a_number(self):
        for score in ("1.0", True, [1], {"a": 1}):
            with self.assertRaises(ConfigurationError):
                Outcome(True, score=score)

    def test_score_rejects_non_finite(self):
        for score in (float("inf"), float("nan")):
            with self.assertRaises(ConfigurationError):
                Outcome(True, score=score)

    def test_a_negative_score_is_fine(self):
        # Rewards can be punishments.
        self.assertEqual(Outcome(False, score=-2.5).score, -2.5)

    def test_score_survives_the_json_round_trip(self):
        outcome = Outcome(True, score=0.25, metrics={"latency_ms": 12})
        self.assertEqual(Outcome.from_dict(json.loads(json.dumps(outcome.to_dict()))), outcome)


class OutcomeContractTest(unittest.TestCase):
    """Outcome is a long-term public contract, so prove it fits real shapes.

    Five deliberately dissimilar kinds of result. If any of these needed a
    subclass, a second model or a convention held only in comments, the
    representation would be wrong.
    """

    def assert_public_roundtrip(self, outcome):
        """Whatever shape it is, it must survive JSON unchanged."""
        rebuilt = Outcome.from_dict(json.loads(json.dumps(outcome.to_dict())))
        self.assertEqual(rebuilt, outcome)
        self.assertEqual(rebuilt.to_json(), outcome.to_json())
        return rebuilt

    def test_a_service_call_reports_latency_and_retries_with_no_score(self):
        # The common case: several measures, no honest way to combine them.
        outcome = Outcome(
            success=True,
            metrics={"latency_ms": 943.0, "retries": 2, "payload_bytes": 18_204},
            metadata={"endpoint": "POST /v1/charges", "upstream": "eu-west-1"},
            decision_id="d-1",
        )
        self.assertIsNone(outcome.score)
        self.assertEqual(outcome.metrics["retries"], 2.0)
        self.assertEqual(outcome.metadata["upstream"], "eu-west-1")
        self.assert_public_roundtrip(outcome)

    def test_a_trade_reports_profit_and_loss_as_the_score(self):
        # Here a scalar genuinely exists and is the objective, so it belongs in
        # score — with its components still separately visible.
        outcome = Outcome(
            success=False,
            score=-1284.55,
            metrics={"gross_pnl_usd": -1250.0, "fees_usd": 34.55, "holding_secs": 612},
            metadata={"currency": "USD", "venue": "XNYS", "reason": "stop-loss"},
        )
        self.assertLess(outcome.score, 0)
        self.assertFalse(outcome.success)
        # The score is the objective; the metrics explain how it got there.
        self.assertAlmostEqual(
            outcome.score,
            -(outcome.metrics["fees_usd"] - outcome.metrics["gross_pnl_usd"]),
            places=6,
        )
        self.assert_public_roundtrip(outcome)

    def test_a_shipping_choice_reports_multiple_competing_objectives(self):
        # Cost against speed against reliability. Any single number here would
        # be an opinion smuggled in as a measurement.
        outcome = Outcome(
            success=True,
            metrics={"cost_usd": 18.0, "transit_days": 1.0, "on_time": 1.0},
            metadata={"carrier": "fedex", "service": "priority-overnight"},
        )
        self.assertIsNone(outcome.score)
        self.assertEqual(len(outcome.metrics), 3)
        self.assert_public_roundtrip(outcome)

    def test_a_batch_job_reports_resource_consumption(self):
        outcome = Outcome(
            success=True,
            metrics={
                "cpu_seconds": 412.75,
                "peak_rss_bytes": 1_073_741_824,
                "rows_written": 2_400_000,
            },
            metadata={"queue": "nightly", "worker": "batch-07"},
        )
        self.assertEqual(outcome.metrics["peak_rss_bytes"], 1_073_741_824.0)
        self.assert_public_roundtrip(outcome)

    def test_an_unmeasurable_outcome_is_still_expressible(self):
        # A human overruled the decision. There is nothing to measure, and the
        # model must not invent something.
        outcome = Outcome(
            success=False,
            metadata={"overridden_by": "on-call", "reason": "customer escalation"},
        )
        self.assertIsNone(outcome.score)
        self.assertEqual(dict(outcome.metrics), {})
        self.assert_public_roundtrip(outcome)

    def test_a_bare_verdict_needs_nothing_else(self):
        self.assertEqual(Outcome(True).to_dict()["metrics"], {})
        self.assert_public_roundtrip(Outcome(False))

    def test_money_as_integer_minor_units_stays_exact(self):
        # Floats are documented as unsuitable for money. Minor units survive
        # exactly, which is the recommended workaround.
        outcome = Outcome(True, metrics={"cost_cents": 1899}, metadata={"currency": "USD"})
        rebuilt = self.assert_public_roundtrip(outcome)
        self.assertEqual(int(rebuilt.metrics["cost_cents"]), 1899)

    def test_metrics_and_metadata_do_not_bleed_into_each_other(self):
        outcome = Outcome(True, metrics={"cost_usd": 1.5}, metadata={"currency": "USD"})
        self.assertNotIn("currency", outcome.metrics)
        self.assertNotIn("cost_usd", outcome.metadata)

    def test_metadata_carries_structure_that_metrics_cannot(self):
        outcome = Outcome(
            True,
            metadata={"attempts": [{"code": 503}, {"code": 200}], "region": "eu"},
        )
        self.assertEqual(outcome.metadata["attempts"][1]["code"], 200)
        self.assert_public_roundtrip(outcome)

    def test_outcomes_are_immutable_whatever_shape_they_take(self):
        outcome = Outcome(True, metrics={"cost_usd": 1.0}, metadata={"tags": ["a"]})
        with self.assertRaises(TypeError):
            outcome.metrics["cost_usd"] = 2.0
        with self.assertRaises(TypeError):
            outcome.metadata["tags"] = ["b"]
        # Nested lists are frozen to tuples, so there is nothing to append to.
        self.assertEqual(outcome.metadata["tags"], ("a",))
        with self.assertRaises(AttributeError):
            outcome.metadata["tags"].append("b")


class CanonicalJsonTest(unittest.TestCase):
    def test_sorted_and_compact(self):
        self.assertEqual(to_canonical_json({"b": 1, "a": [1, 2]}), '{"a":[1,2],"b":1}')

    def test_refuses_non_finite(self):
        with self.assertRaises(ValueError):
            to_canonical_json({"a": float("nan")})


if __name__ == "__main__":
    unittest.main()
