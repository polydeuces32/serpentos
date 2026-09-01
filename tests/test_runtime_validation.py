"""Tests for action validation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serpentos.runtime.errors import ConfigurationError
from serpentos.runtime.models import Decision, DecisionContext
from serpentos.runtime.validation import ActionValidator, ValidationResult

CTX = DecisionContext()


def decision(action):
    return Decision(action, "p", "1")


class ValidationResultTest(unittest.TestCase):
    def test_constructors(self):
        self.assertTrue(ValidationResult.accepted("v").valid)
        rejected = ValidationResult.rejected("v", "because")
        self.assertFalse(rejected.valid)
        self.assertEqual(rejected.reason, "because")

    def test_roundtrip(self):
        result = ValidationResult.rejected("v", "because")
        self.assertEqual(ValidationResult.from_dict(result.to_dict()), result)

    def test_from_dict_rejects_malformed(self):
        for payload in ("nope", {}, {"valid": "yes"}, {"valid": True, "validator": 1}):
            with self.assertRaises(ConfigurationError):
                ValidationResult.from_dict(payload)


class ActionValidatorTest(unittest.TestCase):
    def setUp(self):
        self.validator = ActionValidator({"retry", "fail"})

    def test_allowed_action_passes(self):
        result = self.validator.validate(decision("retry"), CTX)
        self.assertTrue(result.valid)
        self.assertIsNone(result.reason)

    def test_blocked_action_is_rejected_with_a_reason(self):
        result = self.validator.validate(decision("delete-everything"), CTX)
        self.assertFalse(result.valid)
        self.assertIn("delete-everything", result.reason)
        self.assertIn("fail, retry", result.reason)

    def test_allow_list_is_immutable(self):
        allowed = self.validator.allowed_actions
        with self.assertRaises(AttributeError):
            allowed.add("anything")
        self.assertEqual(self.validator.allowed_actions, frozenset({"retry", "fail"}))

    def test_a_bare_string_is_not_an_allow_list(self):
        with self.assertRaises(ConfigurationError) as caught:
            ActionValidator("retry")
        self.assertIn("not a single string", str(caught.exception))

    def test_empty_allow_list_is_refused(self):
        with self.assertRaises(ConfigurationError):
            ActionValidator([])

    def test_non_string_actions_are_refused(self):
        with self.assertRaises(ConfigurationError):
            ActionValidator(["retry", 7])

    def test_malformed_decision_objects_are_rejected_not_crashed(self):
        class NotADecision:
            action = None

        result = self.validator.validate(NotADecision(), CTX)
        self.assertFalse(result.valid)
        self.assertIn("non-empty string", result.reason)

    def test_case_is_significant(self):
        self.assertFalse(self.validator.validate(decision("RETRY"), CTX).valid)

    def test_names_itself_in_the_result(self):
        validator = ActionValidator({"go"}, name="my-guardrail")
        self.assertEqual(validator.validate(decision("go"), CTX).validator, "my-guardrail")


if __name__ == "__main__":
    unittest.main()
