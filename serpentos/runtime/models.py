"""The four data models the runtime is built around.

::

    DecisionContext  ->  Policy  ->  Decision  ->  Outcome
      (what we know)              (what to do)   (what happened)

All three concrete models are frozen dataclasses holding deeply-immutable,
JSON-representable data. That combination is what makes the rest of the runtime
possible: a context that cannot be mutated and can be written to disk verbatim
is a context that can be replayed months later and produce the same decision.

Nothing here knows about Snake, HTTP, queues, machine learning or any other
domain. Values are ordinary JSON: objects, arrays, strings, numbers, booleans
and nulls. Anything else — a live socket, a lambda, a numpy array — is rejected
at construction time rather than at serialisation time, because a model that
only fails when you try to persist it is a model that fails in production.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping, Optional

from .errors import ConfigurationError

__all__ = [
    "DecisionContext",
    "Decision",
    "Outcome",
    "freeze_value",
    "thaw_value",
    "to_canonical_json",
]

# Depth limit for nested structures. Deeply recursive JSON is a classic way to
# blow the C stack during parsing or serialisation; refuse it up front.
MAX_DEPTH = 32

_EMPTY: Mapping[str, Any] = MappingProxyType({})


# =========================
# JSON-SAFE IMMUTABLE VALUES
# =========================
def freeze_value(value: Any, *, _depth: int = 0, _path: str = "value") -> Any:
    """Return a deeply-immutable, JSON-representable copy of ``value``.

    Mappings become read-only mappings, sequences become tuples, and scalars
    pass through. Anything that JSON cannot represent — arbitrary objects,
    callables, ``NaN``, non-string mapping keys — raises
    :class:`~serpentos.runtime.errors.ConfigurationError`.

    :raises ConfigurationError: if the value cannot be represented as JSON.
    """
    if _depth > MAX_DEPTH:
        raise ConfigurationError(f"{_path}: nested more than {MAX_DEPTH} levels deep")

    # bool before int: bool is a subclass of int and must stay a bool.
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # NaN and the infinities are not JSON. Allowing them here would produce
        # output that no other JSON parser accepts, and NaN != NaN breaks the
        # replay comparison in a way that is very hard to debug.
        if value != value or value in (float("inf"), float("-inf")):
            raise ConfigurationError(f"{_path}: {value!r} is not representable in JSON")
        return value
    if isinstance(value, Mapping):
        frozen: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigurationError(
                    f"{_path}: mapping keys must be strings, got {type(key).__name__}"
                )
            frozen[key] = freeze_value(item, _depth=_depth + 1, _path=f"{_path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_value(item, _depth=_depth + 1, _path=f"{_path}[{i}]")
            for i, item in enumerate(value)
        )
    raise ConfigurationError(
        f"{_path}: {type(value).__name__} is not JSON-representable; "
        "convert it before putting it in a context"
    )


def thaw_value(value: Any) -> Any:
    """Inverse of :func:`freeze_value`: plain dicts and lists, freshly copied.

    The result shares no mutable state with the model it came from, so callers
    can modify it without corrupting an audit record.
    """
    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    return value


def _freeze_mapping(value: Optional[Mapping[str, Any]], label: str) -> Mapping[str, Any]:
    if value is None:
        return _EMPTY
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a mapping, got {type(value).__name__}")
    return freeze_value(value, _path=label)


def to_canonical_json(payload: Any) -> str:
    """Serialise to JSON deterministically: sorted keys, no incidental spaces.

    Two equal payloads always produce byte-identical output, which is what lets
    audit records be diffed, hashed and compared across machines.
    """
    return json.dumps(
        thaw_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _require_str(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{label} must be a string, got {type(value).__name__}")
    if not allow_empty and not value:
        raise ConfigurationError(f"{label} must not be empty")
    return value


def _require_mapping(payload: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ConfigurationError(f"{label} must be a JSON object, got {type(payload).__name__}")
    return payload


# =========================
# CONTEXT
# =========================
@dataclass(frozen=True)
class DecisionContext:
    """Everything a policy is allowed to look at when making one decision.

    ``values`` is an arbitrary JSON object chosen by the host application:
    ``{"attempts": 2, "status_code": 503}`` for a retry policy, ``{"queue_depth":
    900}`` for a scheduler. The runtime never interprets it.

    ``request_id`` is an optional caller-supplied correlation identifier. It is
    copied onto audit records so a decision can be tied back to the request that
    triggered it. It is *not* the decision identifier — the engine assigns that.

    The instance is deeply immutable: ``values`` is a read-only mapping whose
    nested containers are also read-only, so handing the same context to five
    policies cannot let one of them corrupt it for the others.

    >>> ctx = DecisionContext({"attempts": 2})
    >>> ctx["attempts"]
    2
    >>> ctx.to_json()
    '{"request_id":null,"values":{"attempts":2}}'
    """

    values: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)
    request_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze_mapping(self.values, "values"))
        if self.request_id is not None:
            _require_str(self.request_id, "request_id")

    # -- read access -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __contains__(self, key: object) -> bool:
        return key in self.values

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def get(self, key: str, default: Any = None) -> Any:
        """Value for ``key``, or ``default`` when absent."""
        return self.values.get(key, default)

    def with_values(self, **updates: Any) -> "DecisionContext":
        """A new context with ``updates`` merged in. The original is untouched."""
        merged = dict(thaw_value(self.values))
        merged.update(updates)
        return DecisionContext(merged, self.request_id)

    # -- serialisation -----------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """A mutable, JSON-ready copy."""
        return {"values": thaw_value(self.values), "request_id": self.request_id}

    def to_json(self) -> str:
        """Canonical JSON. Equal contexts always serialise identically."""
        return to_canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionContext":
        """Rebuild from :meth:`to_dict` output.

        :raises ConfigurationError: if the payload is not a well-formed context.
        """
        payload = _require_mapping(payload, "context")
        request_id = payload.get("request_id")
        if request_id is not None:
            _require_str(request_id, "request_id")
        return cls(_require_mapping(payload.get("values", {}), "context.values"), request_id)


# =========================
# DECISION
# =========================
@dataclass(frozen=True)
class Decision:
    """What a policy proposes. Not what the application did.

    A ``Decision`` is inert: producing one has no effect on the world. The host
    application decides whether and how to act on ``action``, which keeps
    policies pure and therefore replayable.

    ``policy_name`` and ``policy_version`` record who proposed it, so an audit
    trail survives the policy being edited afterwards. ``decision_id`` is
    assigned by :class:`~serpentos.runtime.engine.DecisionEngine`; policies
    normally leave it unset.

    ``metadata`` is free-form JSON for explaining the choice — which rule
    matched, what the scores were. It is data, never code.
    """

    action: str
    policy_name: str
    policy_version: str
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)
    decision_id: Optional[str] = None

    def __post_init__(self) -> None:
        _require_str(self.action, "action")
        _require_str(self.policy_name, "policy_name")
        _require_str(self.policy_version, "policy_version")
        if self.decision_id is not None:
            _require_str(self.decision_id, "decision_id")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    def with_decision_id(self, decision_id: str) -> "Decision":
        """A copy carrying ``decision_id``. Used by the engine."""
        return replace(self, decision_id=_require_str(decision_id, "decision_id"))

    def to_dict(self) -> Dict[str, Any]:
        """A mutable, JSON-ready copy."""
        return {
            "action": self.action,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "metadata": thaw_value(self.metadata),
            "decision_id": self.decision_id,
        }

    def to_json(self) -> str:
        """Canonical JSON. Equal decisions always serialise identically."""
        return to_canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Decision":
        """Rebuild from :meth:`to_dict` output.

        :raises ConfigurationError: if required fields are missing or malformed.
        """
        payload = _require_mapping(payload, "decision")
        for required in ("action", "policy_name", "policy_version"):
            if required not in payload:
                raise ConfigurationError(f"decision is missing {required!r}")
        return cls(
            action=payload["action"],
            policy_name=payload["policy_name"],
            policy_version=payload["policy_version"],
            metadata=_require_mapping(payload.get("metadata", {}), "decision.metadata"),
            decision_id=payload.get("decision_id"),
        )


# =========================
# OUTCOME
# =========================
@dataclass(frozen=True)
class Outcome:
    """What happened after the application executed a decision.

    Three ways to say it, because one is never enough:

    * ``success`` — did it work? Means whatever the host application says it
      means. The runtime never infers it.
    * ``score`` — an optional single number, for the cases where one genuinely
      exists: a reward signal, a utility, a profit. Optional on purpose. A great
      many real decisions have no honest scalar summary, and forcing one is how
      you end up optimising the wrong thing.
    * ``metrics`` — a flat mapping of named numbers: latency, cost, retries.
      This is usually the truthful representation, and
      :func:`~serpentos.runtime.comparison.compare` aggregates each one
      separately rather than collapsing them.

    Outcomes are reported by the application, not produced by the runtime.
    Nothing consumes them automatically; they exist so comparison can aggregate
    real results instead of guessing at them.

    >>> Outcome(True, score=0.8, metrics={"latency_ms": 120})
    Outcome(success=True, score=0.8, ...)
    """

    success: bool
    score: Optional[float] = None
    metrics: Mapping[str, float] = field(default_factory=lambda: _EMPTY)
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)
    decision_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ConfigurationError(
                f"success must be a bool, got {type(self.success).__name__}"
            )
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise ConfigurationError(
                    f"score must be a number or None, got {type(self.score).__name__}"
                )
            object.__setattr__(
                self, "score", freeze_value(float(self.score), _path="score")
            )
        if not isinstance(self.metrics, Mapping):
            raise ConfigurationError(
                f"metrics must be a mapping, got {type(self.metrics).__name__}"
            )
        metrics: Dict[str, float] = {}
        for key, value in self.metrics.items():
            if not isinstance(key, str):
                raise ConfigurationError("metric names must be strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigurationError(
                    f"metric {key!r} must be a number, got {type(value).__name__}"
                )
            metrics[key] = freeze_value(float(value), _path=f"metrics.{key}")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))
        if self.decision_id is not None:
            _require_str(self.decision_id, "decision_id")

    def to_dict(self) -> Dict[str, Any]:
        """A mutable, JSON-ready copy."""
        return {
            "success": self.success,
            "score": self.score,
            "metrics": dict(self.metrics),
            "metadata": thaw_value(self.metadata),
            "decision_id": self.decision_id,
        }

    def to_json(self) -> str:
        """Canonical JSON. Equal outcomes always serialise identically."""
        return to_canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Outcome":
        """Rebuild from :meth:`to_dict` output.

        :raises ConfigurationError: if the payload is not a well-formed outcome.
        """
        payload = _require_mapping(payload, "outcome")
        if "success" not in payload:
            raise ConfigurationError("outcome is missing 'success'")
        return cls(
            success=payload["success"],
            score=payload.get("score"),
            metrics=_require_mapping(payload.get("metrics", {}), "outcome.metrics"),
            metadata=_require_mapping(payload.get("metadata", {}), "outcome.metadata"),
            decision_id=payload.get("decision_id"),
        )
