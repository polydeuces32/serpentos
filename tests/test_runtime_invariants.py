"""Property-style and security invariants for the runtime.

These are the claims the rest of the design leans on. Each is checked against a
deterministically generated spread of inputs rather than one hand-picked
example: the generator is seeded, so a failure here reproduces exactly.

No third-party property-testing library is used. Hypothesis would give better
shrinking, but the state space here is small and enumerable, and the runtime's
whole pitch is that it adds no dependencies to your project.
"""

import json
import os
import random
import string
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serpentos.policies.rules import Rule, RulePolicy, condition_from_dict, when
from serpentos.policies.weighted import LinearScorer, WeightedPolicy
from serpentos.runtime.audit import AuditRecord, InMemoryAuditLog, JsonlAuditLog, read_jsonl
from serpentos.runtime.comparison import compare
from serpentos.runtime.engine import DecisionEngine
from serpentos.runtime.errors import (
    AuditError,
    ConfigurationError,
    DecisionValidationError,
    PolicyError,
    SerpentOSError,
)
from serpentos.runtime.models import Decision, DecisionContext, Outcome
from serpentos.runtime.policy import BasePolicy
from serpentos.runtime.replay import replay_all
from serpentos.runtime.validation import ActionValidator

ACTIONS = ("retry", "fail", "fallback", "escalate")


def random_value(rng, depth=0):
    """A random JSON-representable value."""
    choices = ["int", "float", "str", "bool", "none"]
    if depth < 2:
        choices += ["list", "dict"]
    kind = rng.choice(choices)
    if kind == "int":
        return rng.randint(-1000, 1000)
    if kind == "float":
        return round(rng.uniform(-100, 100), 6)
    if kind == "str":
        return "".join(rng.choice(string.printable[:80]) for _ in range(rng.randint(0, 12)))
    if kind == "bool":
        return rng.choice([True, False])
    if kind == "none":
        return None
    if kind == "list":
        return [random_value(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {
        f"k{index}": random_value(rng, depth + 1) for index in range(rng.randint(0, 3))
    }


def random_context(rng):
    return DecisionContext(
        {f"field{index}": random_value(rng) for index in range(rng.randint(0, 5))},
        request_id=rng.choice([None, f"req-{rng.randint(0, 99)}"]),
    )


class ArbitraryActionPolicy(BasePolicy):
    """Proposes an action drawn from a seeded generator, including illegal ones."""

    deterministic = False

    def __init__(self, rng, pool):
        super().__init__("arbitrary", "1.0")
        self.rng = rng
        self.pool = pool

    def decide(self, context):
        return self.decision(self.rng.choice(self.pool))


class ValidatorInvariantTest(unittest.TestCase):
    def test_no_generated_action_outside_the_allow_list_is_ever_accepted(self):
        rng = random.Random(20240501)
        pool = list(ACTIONS) + ["nuke", "DROP TABLE", "retry ", "", "Retry", "../../etc"]
        validator = ActionValidator(ACTIONS)
        for _ in range(2000):
            action = rng.choice(pool)
            try:
                decision = Decision(action, "p", "1.0")
            except ConfigurationError:
                continue  # An empty action never becomes a Decision at all.
            result = validator.validate(decision, DecisionContext())
            self.assertEqual(result.valid, action in ACTIONS, f"action={action!r}")

    def test_the_engine_never_returns_a_rejected_action(self):
        rng = random.Random(7)
        policy = ArbitraryActionPolicy(rng, list(ACTIONS) + ["nuke", "wipe"])
        engine = DecisionEngine(policy, validator=ActionValidator(ACTIONS))
        allowed = 0
        for _ in range(500):
            try:
                decision = engine.decide(random_context(rng))
            except DecisionValidationError:
                continue
            self.assertIn(decision.action, ACTIONS)
            allowed += 1
        self.assertGreater(allowed, 0)

    def test_a_substituted_action_only_ever_comes_from_a_configured_fallback(self):
        rng = random.Random(11)
        engine = DecisionEngine(
            ArbitraryActionPolicy(rng, ["nuke"]),
            validator=ActionValidator(ACTIONS),
            fallback_policy=RulePolicy(
                name="safe", version="1.0", rules=[], default_action="fail"
            ),
        )
        for _ in range(50):
            decision = engine.decide(random_context(rng))
            self.assertEqual(decision.action, "fail")
            self.assertEqual(decision.policy_name, "safe")
            self.assertEqual(decision.metadata["fallback_for"], "nuke")


class ImmutabilityInvariantTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(4242)
        self.policy = RulePolicy(
            name="mixed",
            version="1.0",
            rules=[
                Rule("retry", when("field0", "exists")),
                Rule("fail", when("field1", "gt", 0)),
            ],
            default_action="fallback",
        )

    def test_deciding_never_mutates_the_context(self):
        engine = DecisionEngine(self.policy)
        for _ in range(300):
            context = random_context(self.rng)
            before = context.to_json()
            engine.decide(context)
            self.assertEqual(context.to_json(), before)

    def test_replay_never_mutates_the_context(self):
        audit = InMemoryAuditLog()
        engine = DecisionEngine(self.policy, audit_sink=audit)
        for _ in range(100):
            engine.decide(random_context(self.rng))
        before = [record.context.to_json() for record in audit.records]
        replay_all(self.policy, audit.records)
        self.assertEqual([record.context.to_json() for record in audit.records], before)

    def test_comparison_never_mutates_policies_or_cases(self):
        cases = [random_context(self.rng) for _ in range(50)]
        other = WeightedPolicy(
            name="weighted",
            version="1.0",
            scorers={"retry": LinearScorer({"field1": 1.0}), "fail": LinearScorer({}, 0.5)},
        )
        policies = [self.policy, other]
        before_policies = [(p.name, p.version, repr(vars(p))) for p in policies]
        before_cases = [case.to_json() for case in cases]
        compare(policies, cases, validator=ActionValidator(ACTIONS))
        self.assertEqual(
            [(p.name, p.version, repr(vars(p))) for p in policies], before_policies
        )
        self.assertEqual([case.to_json() for case in cases], before_cases)

    def test_a_deterministic_policy_always_replays_to_a_match(self):
        audit = InMemoryAuditLog()
        engine = DecisionEngine(self.policy, audit_sink=audit)
        for _ in range(200):
            engine.decide(random_context(self.rng))
        report = replay_all(self.policy, audit.records)
        self.assertEqual(report.mismatched, 0)
        self.assertEqual(len(report.errors), 0)
        self.assertTrue(report.guaranteed)


class SerialisationInvariantTest(unittest.TestCase):
    def test_every_generated_context_round_trips_byte_identically(self):
        rng = random.Random(99)
        for _ in range(500):
            context = random_context(rng)
            rebuilt = DecisionContext.from_dict(json.loads(context.to_json()))
            self.assertEqual(rebuilt, context)
            self.assertEqual(rebuilt.to_json(), context.to_json())

    def test_audit_records_survive_a_file_round_trip(self):
        rng = random.Random(123)
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "audit.jsonl")
        policy = RulePolicy(
            name="p", version="1.0", rules=[Rule("retry", when("field0", "exists"))],
            default_action="fail",
        )
        sink = JsonlAuditLog(path)
        engine = DecisionEngine(policy, validator=ActionValidator(ACTIONS), audit_sink=sink)
        written = [engine.decide_with_record(random_context(rng))[1] for _ in range(100)]
        sink.close()
        self.assertEqual(list(read_jsonl(path)), written)

    def test_json_serialisation_never_evaluates_its_input(self):
        # Strings that would be dangerous if anything ever eval'd them.
        payloads = [
            "__import__('os').system('touch /tmp/serpentos-pwned')",
            "{{7*7}}",
            "'; DROP TABLE decisions; --",
            "\\x00\\x01",
            "${jndi:ldap://example.invalid/a}",
        ]
        marker = Path("/tmp/serpentos-pwned")
        if marker.exists():  # pragma: no cover - only if a previous run misbehaved
            marker.unlink()
        for payload in payloads:
            context = DecisionContext({"input": payload})
            record = AuditRecord.from_json(
                AuditRecord.build(
                    decision=Decision(payload if payload else "x", "p", "1.0", {"echo": payload}),
                    context=context,
                    decision_id="d",
                    timestamp="t",
                ).to_json()
            )
            self.assertEqual(record.context["input"], payload)
            self.assertEqual(record.decision_metadata["echo"], payload)
        self.assertFalse(marker.exists())

    def test_a_rule_set_from_untrusted_json_stays_data(self):
        hostile = {
            "name": "hostile",
            "version": "1.0",
            "default_action": "fail",
            "rules": [
                {
                    "action": "retry",
                    "condition": {
                        "type": "comparison",
                        "key": "a",
                        "op": "eq",
                        "value": "__import__('os').system('true')",
                    },
                }
            ],
        }
        policy = RulePolicy.from_dict(hostile)
        self.assertEqual(policy.decide(DecisionContext({"a": 1})).action, "fail")
        self.assertEqual(
            policy.decide(
                DecisionContext({"a": "__import__('os').system('true')"})
            ).action,
            "retry",
        )

    def test_no_serialised_condition_can_name_a_python_callable(self):
        for operator in ("eval", "exec", "__import__", "system", "getattr", "call"):
            with self.assertRaises(ConfigurationError):
                condition_from_dict({"type": "comparison", "key": "a", "op": operator})


class MalformedInputInvariantTest(unittest.TestCase):
    def test_malformed_audit_lines_fail_as_audit_errors_not_crashes(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "audit.jsonl")
        broken = [
            "{",
            "[]",
            "null",
            "3",
            '"a string"',
            "{}",
            '{"decision_id": null}',
            '{"decision_id": "d", "timestamp": "t"}',
            '{"decision_id": "d", "timestamp": "t", "policy_name": "p", '
            '"policy_version": "1", "action": "a", "context": 5}',
            '{"decision_id": "d", "timestamp": "t", "policy_name": "p", '
            '"policy_version": "1", "action": "a", "validation_result": "yes"}',
        ]
        for line in broken:
            Path(path).write_text(line + "\n", encoding="utf-8")
            with self.assertRaises(AuditError, msg=line):
                list(read_jsonl(path))

    def test_every_malformed_model_payload_raises_a_typed_error(self):
        rng = random.Random(31337)
        for _ in range(300):
            payload = random_value(rng)
            for loader in (
                DecisionContext.from_dict,
                Decision.from_dict,
                Outcome.from_dict,
            ):
                try:
                    loader(payload)
                except SerpentOSError:
                    pass
                except Exception as exc:  # pragma: no cover - the bug we are hunting
                    self.fail(f"{loader} raised untyped {type(exc).__name__} for {payload!r}")

    def test_a_policy_that_returns_junk_never_escapes_as_junk(self):
        class JunkPolicy(BasePolicy):
            def __init__(self, value):
                super().__init__("junk", "1.0")
                self.value = value

            def decide(self, context):
                return self.value

        for value in (None, "retry", 42, [], {"action": "retry"}, object()):
            engine = DecisionEngine(JunkPolicy(value))
            with self.assertRaises(PolicyError):
                engine.decide(DecisionContext())


class BoundednessInvariantTest(unittest.TestCase):
    def test_in_memory_audit_never_exceeds_its_cap(self):
        log = InMemoryAuditLog(max_records=10)
        engine = DecisionEngine(
            RulePolicy(name="p", version="1.0", rules=[], default_action="fail"),
            audit_sink=log,
        )
        for _ in range(500):
            engine.decide(DecisionContext())
            self.assertLessEqual(len(log), 10)

    def test_a_rotating_jsonl_log_stays_bounded(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "audit.jsonl")
        log = JsonlAuditLog(path, max_bytes=4096)
        engine = DecisionEngine(
            RulePolicy(name="p", version="1.0", rules=[], default_action="fail"),
            audit_sink=log,
        )
        for index in range(2000):
            engine.decide(DecisionContext({"index": index}))
        log.close()
        total = os.path.getsize(path) + os.path.getsize(path + ".1")
        # Two generations only: the rotated file is replaced, never accumulated.
        self.assertLess(total, 4096 * 3)
        self.assertEqual(
            sorted(os.listdir(directory)), ["audit.jsonl", "audit.jsonl.1"]
        )


if __name__ == "__main__":
    unittest.main()
