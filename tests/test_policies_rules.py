"""Tests for the rule policy and its data-only condition language."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serpentos.policies.rules import (
    AllOf,
    Always,
    AnyOf,
    Not,
    Predicate,
    Rule,
    RulePolicy,
    condition_from_dict,
    when,
)
from serpentos.runtime.errors import ConfigurationError, PolicyError
from serpentos.runtime.models import DecisionContext


class ComparisonTest(unittest.TestCase):
    def test_equality_operators(self):
        self.assertTrue(when("a", "eq", 1).matches({"a": 1}))
        self.assertFalse(when("a", "eq", 1).matches({"a": 2}))
        self.assertTrue(when("a", "ne", 1).matches({"a": 2}))

    def test_ordering_operators(self):
        values = {"a": 5}
        self.assertTrue(when("a", "lt", 6).matches(values))
        self.assertTrue(when("a", "le", 5).matches(values))
        self.assertTrue(when("a", "gt", 4).matches(values))
        self.assertTrue(when("a", "ge", 5).matches(values))
        self.assertFalse(when("a", "lt", 5).matches(values))

    def test_membership_operators(self):
        self.assertTrue(when("code", "in", [503, 504]).matches({"code": 503}))
        self.assertFalse(when("code", "in", [503]).matches({"code": 500}))
        self.assertTrue(when("code", "not_in", [503]).matches({"code": 500}))
        self.assertTrue(when("tags", "contains", "urgent").matches({"tags": ["urgent"]}))

    def test_string_operators(self):
        self.assertTrue(when("path", "startswith", "/api").matches({"path": "/api/v1"}))
        self.assertTrue(when("path", "endswith", ".json").matches({"path": "a.json"}))
        self.assertFalse(when("path", "startswith", "/api").matches({"path": 7}))

    def test_presence_operators(self):
        self.assertTrue(when("a", "exists").matches({"a": None}))
        self.assertFalse(when("a", "exists").matches({}))
        self.assertTrue(when("a", "missing").matches({}))
        self.assertFalse(when("a", "missing").matches({"a": 1}))

    def test_a_missing_key_never_matches_a_value_comparison(self):
        for operator in ("eq", "ne", "lt", "gt", "in", "not_in", "contains"):
            operand = [] if operator in ("in", "not_in") else 1
            self.assertFalse(
                when("absent", operator, operand).matches({}),
                f"{operator} matched a missing key",
            )

    def test_incomparable_types_are_false_not_fatal(self):
        self.assertFalse(when("a", "lt", 5).matches({"a": "text"}))
        self.assertFalse(when("a", "contains", 1).matches({"a": 5}))

    def test_unknown_operator_is_refused_at_construction(self):
        with self.assertRaises(ConfigurationError) as caught:
            when("a", "regex", ".*")
        self.assertIn("unknown operator", str(caught.exception))

    def test_presence_operators_take_no_operand(self):
        with self.assertRaises(ConfigurationError):
            when("a", "exists", 1)

    def test_operand_must_be_json(self):
        with self.assertRaises(ConfigurationError):
            when("a", "eq", object())

    def test_key_must_be_a_non_empty_string(self):
        with self.assertRaises(ConfigurationError):
            when("", "eq", 1)


class CombinatorTest(unittest.TestCase):
    def test_all_of(self):
        condition = AllOf(when("a", "eq", 1), when("b", "eq", 2))
        self.assertTrue(condition.matches({"a": 1, "b": 2}))
        self.assertFalse(condition.matches({"a": 1}))

    def test_any_of(self):
        condition = AnyOf(when("a", "eq", 1), when("b", "eq", 2))
        self.assertTrue(condition.matches({"b": 2}))
        self.assertFalse(condition.matches({"c": 3}))

    def test_accepts_a_list_as_well_as_varargs(self):
        self.assertEqual(
            AllOf([when("a", "eq", 1)]).conditions, AllOf(when("a", "eq", 1)).conditions
        )

    def test_not(self):
        self.assertTrue(Not(when("a", "eq", 1)).matches({"a": 2}))

    def test_always(self):
        self.assertTrue(Always().matches({}))

    def test_empty_combinators_are_refused(self):
        with self.assertRaises(ConfigurationError):
            AllOf()
        with self.assertRaises(ConfigurationError):
            AnyOf()

    def test_combinators_require_conditions(self):
        with self.assertRaises(ConfigurationError):
            AllOf("a == 1")
        with self.assertRaises(ConfigurationError):
            Not("a == 1")

    def test_describe_is_readable(self):
        condition = AllOf(when("a", "eq", 1), Not(when("b", "in", [2, 3])))
        self.assertEqual(condition.describe(), "(a eq 1 and not b in [2, 3])")


class SerialisationTest(unittest.TestCase):
    def test_conditions_round_trip_through_json(self):
        original = AnyOf(
            AllOf(when("a", "ge", 1), Not(when("b", "exists"))),
            when("c", "in", ["x", "y"]),
            Always(),
        )
        rebuilt = condition_from_dict(json.loads(json.dumps(original.to_dict())))
        for values in ({"a": 1}, {"a": 1, "b": 2}, {"c": "x"}, {}):
            self.assertEqual(rebuilt.matches(values), original.matches(values))

    def test_unknown_condition_type_is_refused(self):
        with self.assertRaises(ConfigurationError):
            condition_from_dict({"type": "exec", "code": "import os"})

    def test_malformed_conditions_are_refused(self):
        for payload in ("nope", {}, {"type": "all"}, {"type": "all", "conditions": []}):
            with self.assertRaises(ConfigurationError):
                condition_from_dict(payload)

    def test_deeply_nested_conditions_are_refused(self):
        payload = {"type": "always"}
        for _ in range(20):
            payload = {"type": "not", "condition": payload}
        with self.assertRaises(ConfigurationError):
            condition_from_dict(payload)

    def test_a_serialised_operator_must_be_on_the_allow_list(self):
        with self.assertRaises(ConfigurationError):
            condition_from_dict({"type": "comparison", "key": "a", "op": "__import__"})

    def test_policy_round_trips_as_pure_data(self):
        policy = RulePolicy(
            name="retry",
            version="1.0",
            rules=[Rule("retry", when("code", "in", [503]), name="retry-5xx")],
            default_action="fail",
        )
        rebuilt = RulePolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
        self.assertEqual(rebuilt.name, "retry")
        self.assertEqual(
            rebuilt.decide(DecisionContext({"code": 503})).action,
            policy.decide(DecisionContext({"code": 503})).action,
        )

    def test_policy_from_malformed_data_is_refused(self):
        for payload in ("nope", {"rules": "all of them"}, {"name": "a", "version": "1"}):
            with self.assertRaises(ConfigurationError):
                RulePolicy.from_dict(payload)


class PredicateTest(unittest.TestCase):
    def test_wraps_a_callable(self):
        condition = Predicate(lambda values: values.get("a", 0) > 2, "a-over-two")
        self.assertTrue(condition.matches({"a": 3}))
        self.assertFalse(condition.matches({"a": 1}))

    def test_is_deliberately_not_serialisable(self):
        policy = RulePolicy(
            name="p",
            version="1.0",
            rules=[Rule("go", Predicate(lambda values: True, "always"))],
            default_action="stop",
        )
        with self.assertRaises(ConfigurationError) as caught:
            policy.to_dict()
        self.assertIn("cannot be serialised", str(caught.exception))

    def test_requires_a_callable(self):
        with self.assertRaises(ConfigurationError):
            Predicate("not callable")

    def test_a_raising_predicate_surfaces_as_a_policy_error(self):
        def boom(values):
            raise ZeroDivisionError

        policy = RulePolicy(
            name="p", version="1.0", rules=[Rule("go", Predicate(boom))], default_action="stop"
        )
        with self.assertRaises(PolicyError):
            policy.decide(DecisionContext())


class RuleTest(unittest.TestCase):
    def test_name_defaults_to_a_description(self):
        rule = Rule("retry", when("a", "eq", 1))
        self.assertEqual(rule.name, "retry-if-a eq 1")

    def test_rejects_bad_input(self):
        with self.assertRaises(ConfigurationError):
            Rule("", when("a", "eq", 1))
        with self.assertRaises(ConfigurationError):
            Rule("go", "a == 1")

    def test_round_trip(self):
        rule = Rule("retry", when("a", "eq", 1), name="named")
        self.assertEqual(Rule.from_dict(rule.to_dict()), rule)


class RulePolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = RulePolicy(
            name="retry-policy",
            version="1.0",
            rules=[
                Rule("fail", when("attempts", "ge", 3), name="give-up"),
                Rule("retry", when("status_code", "in", [503, 504]), name="retry-5xx"),
                Rule("fallback", when("status_code", "ge", 500), name="other-5xx"),
            ],
            default_action="fail",
        )

    def decide(self, **values):
        return self.policy.decide(DecisionContext(values))

    def test_first_matching_rule_wins(self):
        # Both give-up and retry-5xx match; the earlier rule takes it.
        decision = self.decide(attempts=3, status_code=503)
        self.assertEqual(decision.action, "fail")
        self.assertEqual(decision.metadata["rule"], "give-up")
        self.assertEqual(decision.metadata["rule_index"], 0)

    def test_later_rules_are_reachable(self):
        self.assertEqual(self.decide(attempts=0, status_code=500).action, "fallback")

    def test_default_applies_when_nothing_matches(self):
        decision = self.decide(attempts=0, status_code=200)
        self.assertEqual(decision.action, "fail")
        self.assertFalse(decision.metadata["matched"])
        self.assertIsNone(decision.metadata["rule"])

    def test_reordering_rules_changes_the_answer(self):
        reordered = RulePolicy(
            name="retry-policy",
            version="1.0",
            rules=list(reversed(self.policy.rules)),
            default_action="fail",
        )
        context = DecisionContext({"attempts": 3, "status_code": 503})
        self.assertEqual(self.policy.decide(context).action, "fail")
        self.assertEqual(reordered.decide(context).action, "fallback")

    def test_identical_contexts_always_decide_identically(self):
        first = self.decide(attempts=1, status_code=503)
        second = self.decide(attempts=1, status_code=503)
        self.assertEqual(first.to_json(), second.to_json())

    def test_advertised_actions_cover_every_outcome(self):
        self.assertEqual(self.policy.actions, ("fail", "fallback", "retry"))

    def test_decisions_carry_the_policy_identity(self):
        decision = self.decide(attempts=0)
        self.assertEqual(decision.policy_name, "retry-policy")
        self.assertEqual(decision.policy_version, "1.0")

    def test_does_not_mutate_the_context(self):
        context = DecisionContext({"attempts": 1, "status_code": 503})
        before = context.to_json()
        self.policy.decide(context)
        self.assertEqual(context.to_json(), before)

    def test_an_empty_rule_set_always_defaults(self):
        policy = RulePolicy(name="p", version="1", rules=[], default_action="fail")
        self.assertEqual(policy.decide(DecisionContext({"a": 1})).action, "fail")

    def test_construction_is_validated(self):
        with self.assertRaises(ConfigurationError):
            RulePolicy(name="", version="1", rules=[], default_action="fail")
        with self.assertRaises(ConfigurationError):
            RulePolicy(name="p", version="1", rules=[], default_action="")
        with self.assertRaises(ConfigurationError):
            RulePolicy(name="p", version="1", rules=["retry if a"], default_action="fail")

    def test_is_declared_deterministic(self):
        self.assertTrue(self.policy.deterministic)


if __name__ == "__main__":
    unittest.main()
