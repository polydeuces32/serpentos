"""Rule-based decisions: ordered conditions, first match wins.

The conditions here are *data*. ``when("status_code", "in", [503, 504])`` builds
a small frozen object that compares two values using one of a fixed set of
operators. There is no expression language, no ``eval``, no ``exec`` and no
import hook, so a rule set can be loaded from a JSON file written by someone you
do not trust with code execution on your servers.

Evaluation order is the order you wrote the rules in. The first matching rule
wins and the rest are never consulted, which makes the policy trivially
explainable: the audit record names the rule that fired.

A missing key never matches. ``when("attempts", "lt", 3)`` is false when there
is no ``attempts`` in the context, rather than raising or guessing a default.
Use ``when("attempts", "missing")`` if absence is what you mean. Failing closed
is the right default for a component that gates behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

from ..runtime.errors import ConfigurationError, PolicyError
from ..runtime.models import Decision, DecisionContext, freeze_value, thaw_value
from ..runtime.policy import BasePolicy

__all__ = [
    "Condition",
    "Comparison",
    "AllOf",
    "AnyOf",
    "Not",
    "Always",
    "Predicate",
    "Rule",
    "RulePolicy",
    "when",
    "condition_from_dict",
    "OPERATORS",
]


class _Missing:
    """Sentinel for "the context has no such key"."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


MISSING = _Missing()


def _ordered(op: Callable[[Any, Any], bool]) -> Callable[[Any, Any], bool]:
    """Wrap an ordering comparison so incomparable types are False, not fatal."""

    def compare(left: Any, right: Any) -> bool:
        try:
            return bool(op(left, right))
        except TypeError:
            return False

    return compare


def _contains(left: Any, right: Any) -> bool:
    try:
        return right in left
    except TypeError:
        return False


def _member_of(left: Any, right: Any) -> bool:
    try:
        return left in right
    except TypeError:
        return False


def _startswith(left: Any, right: Any) -> bool:
    return isinstance(left, str) and isinstance(right, str) and left.startswith(right)


def _endswith(left: Any, right: Any) -> bool:
    return isinstance(left, str) and isinstance(right, str) and left.endswith(right)


#: Every comparison a serialised rule is allowed to perform. Deliberately a
#: closed set: adding an operator is a code change and a review, not a config
#: change.
OPERATORS: Mapping[str, Callable[[Any, Any], bool]] = MappingProxyType(
    {
        "eq": lambda left, right: left == right,
        "ne": lambda left, right: left != right,
        "lt": _ordered(lambda left, right: left < right),
        "le": _ordered(lambda left, right: left <= right),
        "gt": _ordered(lambda left, right: left > right),
        "ge": _ordered(lambda left, right: left >= right),
        "in": _member_of,
        "not_in": lambda left, right: not _member_of(left, right),
        "contains": _contains,
        "startswith": _startswith,
        "endswith": _endswith,
    }
)

#: Operators that take no operand and are evaluated against key presence.
PRESENCE_OPERATORS = frozenset({"exists", "missing"})


class Condition:
    """Base class for rule conditions.

    Subclasses implement :meth:`matches` and, where the condition is pure data,
    :meth:`to_dict`. Conditions are immutable and side-effect free.
    """

    def matches(self, values: Mapping[str, Any]) -> bool:
        """Whether this condition holds for ``values``."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable form of this condition.

        :raises ConfigurationError: for conditions backed by Python callables,
            which cannot be represented as data.
        """
        raise NotImplementedError

    def describe(self) -> str:
        """A short human-readable rendering, used in error messages."""
        return repr(self)


@dataclass(frozen=True)
class Comparison(Condition):
    """Compare one context value against a fixed operand.

    >>> Comparison("attempts", "lt", 3).matches({"attempts": 2})
    True
    >>> Comparison("attempts", "lt", 3).matches({})
    False
    """

    key: str
    operator: str
    operand: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ConfigurationError("condition key must be a non-empty string")
        if self.operator in PRESENCE_OPERATORS:
            if self.operand is not None:
                raise ConfigurationError(
                    f"operator {self.operator!r} takes no operand"
                )
        elif self.operator not in OPERATORS:
            known = ", ".join(sorted(set(OPERATORS) | PRESENCE_OPERATORS))
            raise ConfigurationError(
                f"unknown operator {self.operator!r} (known: {known})"
            )
        object.__setattr__(self, "operand", freeze_value(self.operand, _path="operand"))

    def matches(self, values: Mapping[str, Any]) -> bool:
        """Whether the comparison holds for ``values``."""
        actual = values.get(self.key, MISSING) if isinstance(values, Mapping) else MISSING
        if self.operator == "exists":
            return actual is not MISSING
        if self.operator == "missing":
            return actual is MISSING
        if actual is MISSING:
            return False
        return bool(OPERATORS[self.operator](actual, self.operand))

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable form of this condition."""
        payload: Dict[str, Any] = {"type": "comparison", "key": self.key, "op": self.operator}
        if self.operator not in PRESENCE_OPERATORS:
            payload["value"] = thaw_value(self.operand)
        return payload

    def describe(self) -> str:
        """A short human-readable rendering."""
        if self.operator in PRESENCE_OPERATORS:
            return f"{self.key} {self.operator}"
        return f"{self.key} {self.operator} {thaw_value(self.operand)!r}"


def when(key: str, operator: str, operand: Any = None) -> Comparison:
    """Readable shorthand for :class:`Comparison`.

    >>> when("status_code", "in", [503, 504]).describe()
    "status_code in [503, 504]"
    """
    return Comparison(key, operator, operand)


def _as_conditions(conditions: Iterable[Condition], label: str) -> Tuple[Condition, ...]:
    items = tuple(conditions)
    if not items:
        raise ConfigurationError(f"{label} needs at least one condition")
    for item in items:
        if not isinstance(item, Condition):
            raise ConfigurationError(
                f"{label} takes Condition objects, got {type(item).__name__}"
            )
    return items


@dataclass(frozen=True)
class AllOf(Condition):
    """True when every sub-condition holds. Short-circuits."""

    conditions: Tuple[Condition, ...]

    def __init__(self, *conditions: Condition) -> None:
        flat = conditions[0] if len(conditions) == 1 and isinstance(conditions[0], (list, tuple)) else conditions
        object.__setattr__(self, "conditions", _as_conditions(flat, "AllOf"))

    def matches(self, values: Mapping[str, Any]) -> bool:
        """Whether every sub-condition holds."""
        return all(condition.matches(values) for condition in self.conditions)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable form of this condition."""
        return {"type": "all", "conditions": [c.to_dict() for c in self.conditions]}

    def describe(self) -> str:
        """A short human-readable rendering."""
        return "(" + " and ".join(c.describe() for c in self.conditions) + ")"


@dataclass(frozen=True)
class AnyOf(Condition):
    """True when at least one sub-condition holds. Short-circuits."""

    conditions: Tuple[Condition, ...]

    def __init__(self, *conditions: Condition) -> None:
        flat = conditions[0] if len(conditions) == 1 and isinstance(conditions[0], (list, tuple)) else conditions
        object.__setattr__(self, "conditions", _as_conditions(flat, "AnyOf"))

    def matches(self, values: Mapping[str, Any]) -> bool:
        """Whether any sub-condition holds."""
        return any(condition.matches(values) for condition in self.conditions)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable form of this condition."""
        return {"type": "any", "conditions": [c.to_dict() for c in self.conditions]}

    def describe(self) -> str:
        """A short human-readable rendering."""
        return "(" + " or ".join(c.describe() for c in self.conditions) + ")"


@dataclass(frozen=True)
class Not(Condition):
    """Negates a sub-condition."""

    condition: Condition

    def __post_init__(self) -> None:
        if not isinstance(self.condition, Condition):
            raise ConfigurationError(
                f"Not takes a Condition, got {type(self.condition).__name__}"
            )

    def matches(self, values: Mapping[str, Any]) -> bool:
        """Whether the sub-condition does not hold."""
        return not self.condition.matches(values)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable form of this condition."""
        return {"type": "not", "condition": self.condition.to_dict()}

    def describe(self) -> str:
        """A short human-readable rendering."""
        return f"not {self.condition.describe()}"


@dataclass(frozen=True)
class Always(Condition):
    """Always true. Useful as an explicit catch-all rule."""

    def matches(self, values: Mapping[str, Any]) -> bool:
        """Always true."""
        return True

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable form of this condition."""
        return {"type": "always"}

    def describe(self) -> str:
        """A short human-readable rendering."""
        return "always"


class Predicate(Condition):
    """Wraps a Python callable for in-process rule sets.

    This is the escape hatch for logic the operators cannot express. It is
    deliberately **not** serialisable: :meth:`to_dict` raises, so a rule set
    containing a predicate can never be round-tripped through a config file and
    a config file can never conjure one into existence.

    The callable must be pure. If it reads the clock or a database, replay and
    comparison stop meaning anything.
    """

    def __init__(self, func: Callable[[Mapping[str, Any]], bool], description: str = "") -> None:
        if not callable(func):
            raise ConfigurationError("Predicate needs a callable")
        self._func = func
        self._description = description or getattr(func, "__name__", "predicate")

    def matches(self, values: Mapping[str, Any]) -> bool:
        """Whether the callable returns something truthy for ``values``."""
        return bool(self._func(values))

    def to_dict(self) -> Dict[str, Any]:
        """Always raises: a callable is not data.

        :raises ConfigurationError: always.
        """
        raise ConfigurationError(
            f"Predicate({self._description}) cannot be serialised; "
            "rule sets that must round-trip through configuration may only use "
            "the built-in data conditions"
        )

    def describe(self) -> str:
        """A short human-readable rendering."""
        return f"predicate:{self._description}"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Predicate)
            and other._func is self._func
            and other._description == self._description
        )

    def __hash__(self) -> int:
        return hash((id(self._func), self._description))


def condition_from_dict(payload: Mapping[str, Any], *, _depth: int = 0) -> Condition:
    """Rebuild a condition from :meth:`Condition.to_dict` output.

    Only the built-in condition types are constructible, and the operator must
    be one of :data:`OPERATORS`. Nothing is imported, resolved by name or
    evaluated, so parsing an attacker-supplied rule file cannot execute code —
    the worst it can do is describe a comparison you did not want.

    :raises ConfigurationError: if the payload is not a well-formed condition.
    """
    if _depth > 16:
        raise ConfigurationError("condition nested more than 16 levels deep")
    if not isinstance(payload, Mapping):
        raise ConfigurationError(
            f"condition must be a JSON object, got {type(payload).__name__}"
        )
    kind = payload.get("type")
    if kind == "comparison":
        return Comparison(payload.get("key", ""), payload.get("op", ""), payload.get("value"))
    if kind in ("all", "any"):
        raw = payload.get("conditions")
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ConfigurationError(f"{kind!r} condition needs a non-empty conditions list")
        children = [condition_from_dict(item, _depth=_depth + 1) for item in raw]
        return AllOf(children) if kind == "all" else AnyOf(children)
    if kind == "not":
        return Not(condition_from_dict(payload.get("condition", {}), _depth=_depth + 1))
    if kind == "always":
        return Always()
    raise ConfigurationError(f"unknown condition type {kind!r}")


@dataclass(frozen=True)
class Rule:
    """One ``if condition then action`` clause.

    ``name`` is what shows up in the audit record. Give rules names you would
    want to read at three in the morning.
    """

    action: str
    condition: Condition
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or not self.action:
            raise ConfigurationError("rule action must be a non-empty string")
        if not isinstance(self.condition, Condition):
            raise ConfigurationError(
                f"rule condition must be a Condition, got {type(self.condition).__name__}"
            )
        if not isinstance(self.name, str):
            raise ConfigurationError("rule name must be a string")
        if not self.name:
            object.__setattr__(self, "name", f"{self.action}-if-{self.condition.describe()}")

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable form of this rule.

        :raises ConfigurationError: if the condition is not serialisable.
        """
        return {"action": self.action, "name": self.name, "condition": self.condition.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Rule":
        """Rebuild from :meth:`to_dict` output.

        :raises ConfigurationError: if the payload is malformed.
        """
        if not isinstance(payload, Mapping):
            raise ConfigurationError(
                f"rule must be a JSON object, got {type(payload).__name__}"
            )
        return cls(
            action=payload.get("action", ""),
            condition=condition_from_dict(payload.get("condition", {})),
            name=payload.get("name", "") or "",
        )


class RulePolicy(BasePolicy):
    """Ordered rules with an explicit default. Deterministic by construction.

    >>> policy = RulePolicy(
    ...     name="retry-policy",
    ...     version="1.0",
    ...     rules=[
    ...         Rule("fail", when("attempts", "ge", 3), name="give-up"),
    ...         Rule("retry", when("status_code", "in", [503, 504]), name="retry-5xx"),
    ...     ],
    ...     default_action="fail",
    ... )
    >>> policy.decide(DecisionContext({"attempts": 1, "status_code": 503})).action
    'retry'
    >>> policy.decide(DecisionContext({"attempts": 4, "status_code": 503})).action
    'fail'

    :param name: policy identity, recorded on every decision.
    :param version: policy version, recorded on every decision. Bump it when you
        change the rules, or your audit trail will lie to you.
    :param rules: evaluated in order; the first match wins.
    :param default_action: used when no rule matches. Required — a rule engine
        with no defined default is a rule engine with an undefined behaviour.

    :raises ConfigurationError: if the rules or default are unusable.
    """

    deterministic = True

    def __init__(
        self,
        name: str,
        version: str,
        rules: Sequence[Rule],
        default_action: str,
    ) -> None:
        super().__init__(name, version)
        items = tuple(rules)
        for index, rule in enumerate(items):
            if not isinstance(rule, Rule):
                raise ConfigurationError(
                    f"rules[{index}] must be a Rule, got {type(rule).__name__}"
                )
        if not isinstance(default_action, str) or not default_action:
            raise ConfigurationError("default_action must be a non-empty string")
        self._rules = items
        self.default_action = default_action

    @property
    def rules(self) -> Tuple[Rule, ...]:
        """The rules, in evaluation order."""
        return self._rules

    @property
    def actions(self) -> Tuple[str, ...]:
        """Every action this policy can possibly propose, sorted.

        Handy for building an :class:`~serpentos.runtime.validation.ActionValidator`
        that is guaranteed to accept everything the policy can say.
        """
        return tuple(sorted({rule.action for rule in self._rules} | {self.default_action}))

    def decide(self, context: DecisionContext) -> Decision:
        """Return the action of the first matching rule, or the default.

        :raises PolicyError: if a rule condition itself raises, which can only
            happen with a :class:`Predicate`.
        """
        values = context.values
        for index, rule in enumerate(self._rules):
            try:
                matched = rule.condition.matches(values)
            except Exception as exc:  # noqa: BLE001 - re-typed for the caller
                raise PolicyError(
                    f"rule {rule.name!r} raised {type(exc).__name__}: {exc}"
                ) from exc
            if matched:
                return self.decision(
                    rule.action,
                    {"rule": rule.name, "rule_index": index, "matched": True},
                )
        return self.decision(
            self.default_action, {"rule": None, "rule_index": -1, "matched": False}
        )

    # -- serialisation -----------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable form of this policy.

        :raises ConfigurationError: if any rule uses a :class:`Predicate`.
        """
        return {
            "name": self.name,
            "version": self.version,
            "default_action": self.default_action,
            "rules": [rule.to_dict() for rule in self._rules],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RulePolicy":
        """Rebuild a policy from data. Never executes anything from the payload.

        :raises ConfigurationError: if the payload is malformed.
        """
        if not isinstance(payload, Mapping):
            raise ConfigurationError(
                f"policy must be a JSON object, got {type(payload).__name__}"
            )
        raw_rules = payload.get("rules", [])
        if not isinstance(raw_rules, (list, tuple)):
            raise ConfigurationError("policy rules must be a list")
        return cls(
            name=payload.get("name", ""),
            version=payload.get("version", ""),
            rules=[Rule.from_dict(item) for item in raw_rules],
            default_action=payload.get("default_action", ""),
        )
