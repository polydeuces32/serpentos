"""Guardrails between a policy's proposal and the application acting on it.

A policy can propose anything. A validator decides whether the surrounding
system is willing to hear it. Keeping the two apart means a buggy or
maliciously-authored policy cannot widen its own permissions: the allow-list
lives with the host application, not with the policy.

Rejection is loud. The engine raises
:class:`~serpentos.runtime.errors.DecisionValidationError` rather than quietly
substituting a safe default, because silently rewriting a decision is how you
end up with a system whose audit log does not describe what happened. If you
want a fallback, configure one explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Optional

try:  # pragma: no cover - the fallback only runs on Python < 3.8
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment]

    def runtime_checkable(cls):  # type: ignore[misc]
        return cls


from .errors import ConfigurationError
from .models import Decision, DecisionContext

__all__ = ["ValidationResult", "DecisionValidator", "ActionValidator"]


@dataclass(frozen=True)
class ValidationResult:
    """The verdict on one proposed decision.

    Recorded verbatim on the audit record, so "why was this rejected?" is
    answerable from the log alone.
    """

    valid: bool
    validator: str
    reason: Optional[str] = None

    @classmethod
    def accepted(cls, validator: str) -> "ValidationResult":
        """A passing verdict from ``validator``."""
        return cls(True, validator, None)

    @classmethod
    def rejected(cls, validator: str, reason: str) -> "ValidationResult":
        """A failing verdict from ``validator``, with the reason."""
        return cls(False, validator, reason)

    def to_dict(self) -> Dict[str, Any]:
        """A mutable, JSON-ready copy."""
        return {"valid": self.valid, "validator": self.validator, "reason": self.reason}

    @classmethod
    def from_dict(cls, payload: Any) -> "ValidationResult":
        """Rebuild from :meth:`to_dict` output.

        :raises ConfigurationError: if the payload is malformed.
        """
        if not isinstance(payload, dict):
            raise ConfigurationError("validation_result must be a JSON object")
        if not isinstance(payload.get("valid"), bool):
            raise ConfigurationError("validation_result.valid must be a bool")
        validator = payload.get("validator", "")
        if not isinstance(validator, str):
            raise ConfigurationError("validation_result.validator must be a string")
        reason = payload.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ConfigurationError("validation_result.reason must be a string or null")
        return cls(payload["valid"], validator, reason)


#: The verdict used when an engine runs without a validator configured.
UNVALIDATED = ValidationResult(True, "none", "no validator configured")


@runtime_checkable
class DecisionValidator(Protocol):
    """Anything that can pass judgement on a proposed decision.

    Implementations return a :class:`ValidationResult`; they do not raise on
    rejection. The engine turns a failing result into an exception, which keeps
    validators usable in offline comparison where a rejection is a data point
    rather than a failure.
    """

    name: str

    def validate(self, decision: Decision, context: DecisionContext) -> ValidationResult:
        """Judge ``decision``. Must not mutate either argument."""
        ...


class ActionValidator:
    """Allow-list of actions the application is prepared to execute.

    >>> validator = ActionValidator({"retry", "fail"})
    >>> validator.validate(Decision("retry", "p", "1"), DecisionContext()).valid
    True
    >>> validator.validate(Decision("rm -rf /", "p", "1"), DecisionContext()).valid
    False

    The allow-list is fixed at construction and stored as a frozenset, so a
    policy holding a reference to the validator cannot extend it.
    """

    def __init__(self, allowed_actions: Iterable[str], *, name: str = "action-allowlist") -> None:
        if isinstance(allowed_actions, str):
            # "retry" would otherwise silently become {"r","e","t","y"}.
            raise ConfigurationError(
                "allowed_actions must be a collection of strings, not a single string"
            )
        try:
            actions = frozenset(allowed_actions)
        except TypeError as exc:
            raise ConfigurationError("allowed_actions must be iterable") from exc
        if not actions:
            raise ConfigurationError("allowed_actions must not be empty")
        for action in actions:
            if not isinstance(action, str) or not action:
                raise ConfigurationError(
                    f"allowed action {action!r} must be a non-empty string"
                )
        self._allowed: FrozenSet[str] = actions
        self.name = name

    @property
    def allowed_actions(self) -> FrozenSet[str]:
        """The immutable allow-list."""
        return self._allowed

    def validate(self, decision: Decision, context: DecisionContext) -> ValidationResult:
        """Accept ``decision`` only if its action is on the allow-list."""
        action = getattr(decision, "action", None)
        if not isinstance(action, str) or not action:
            return ValidationResult.rejected(
                self.name, f"action must be a non-empty string, got {action!r}"
            )
        if action not in self._allowed:
            allowed = ", ".join(sorted(self._allowed))
            return ValidationResult.rejected(
                self.name, f"action {action!r} is not allowed (allowed: {allowed})"
            )
        return ValidationResult.accepted(self.name)

    def __repr__(self) -> str:
        return f"ActionValidator({sorted(self._allowed)!r})"
