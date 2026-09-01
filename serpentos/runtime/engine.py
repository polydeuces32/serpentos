"""The decision engine: the one place a policy is actually invoked.

The engine is deliberately small. It takes a context, asks the policy what to
do, checks the answer against the guardrails, gives the decision an identity,
writes an audit record and hands the decision back.

It does not execute the action. Nothing in SerpentOS does. The host application
owns every side effect, which is what keeps policies replayable and keeps the
blast radius of a bad policy limited to "it proposed something we refused".
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Callable, Optional, Tuple

from .audit import AuditRecord, AuditSink, utc_now
from .errors import ConfigurationError, DecisionValidationError, PolicyError
from .models import Decision, DecisionContext
from .policy import Policy, ensure_policy
from .validation import UNVALIDATED, DecisionValidator, ValidationResult

__all__ = ["DecisionEngine"]


def _default_id_factory() -> str:
    return uuid.uuid4().hex


class DecisionEngine:
    """Runs one policy under guardrails and records what it decided.

    >>> from serpentos import ActionValidator, DecisionContext, DecisionEngine
    >>> from serpentos.policies import RulePolicy, Rule, when
    >>> policy = RulePolicy(
    ...     name="retry-policy",
    ...     version="1.0",
    ...     rules=[Rule("retry", when("status_code", "in", [503, 504]))],
    ...     default_action="fail",
    ... )
    >>> engine = DecisionEngine(policy, validator=ActionValidator({"retry", "fail"}))
    >>> engine.decide(DecisionContext({"status_code": 503})).action
    'retry'

    :param policy: the policy to run. Anything satisfying
        :class:`~serpentos.runtime.policy.Policy`.
    :param validator: optional guardrail. Without one, every proposed action is
        accepted and the audit record says so explicitly.
    :param audit_sink: optional destination for audit records. Without one,
        nothing is persisted anywhere.
    :param fallback_policy: optional second policy consulted only when the
        validator rejects the primary policy's proposal. This is the *only* way
        an action gets substituted, and it is never silent — the audit log
        contains both the rejection and the replacement.
    :param clock: returns the record timestamp. Injectable for deterministic
        tests; defaults to UTC now.
    :param id_factory: returns a fresh decision identifier. Injectable for
        deterministic tests; defaults to a random UUID hex.

    :raises ConfigurationError: if any component does not satisfy its interface.

    **Concurrency.** The engine holds no mutable state of its own, so a single
    instance can be reused for the life of a process and shared freely between
    threads. Both built-in sinks (:class:`~serpentos.runtime.audit.InMemoryAuditLog`
    and :class:`~serpentos.runtime.audit.JsonlAuditLog`) are internally
    synchronised, so the default configuration is thread-safe.

    That guarantee ends where your code begins. A shared engine is thread-safe
    only if the policy, validator and sink you supply are: a policy that honours
    the purity contract is stateless and therefore safe, but one that caches into
    a plain ``dict``, or a custom sink that appends to an unguarded list, is not.
    The runtime cannot check this for you.
    """

    def __init__(
        self,
        policy: Policy,
        *,
        validator: Optional[DecisionValidator] = None,
        audit_sink: Optional[AuditSink] = None,
        fallback_policy: Optional[Policy] = None,
        clock: Callable[[], str] = utc_now,
        id_factory: Callable[[], str] = _default_id_factory,
    ) -> None:
        self._policy = ensure_policy(policy)
        self._fallback = ensure_policy(fallback_policy) if fallback_policy is not None else None
        if validator is not None and not callable(getattr(validator, "validate", None)):
            raise ConfigurationError(
                f"{type(validator).__name__} is not a validator: no callable validate()"
            )
        if audit_sink is not None and not callable(getattr(audit_sink, "record", None)):
            raise ConfigurationError(
                f"{type(audit_sink).__name__} is not an audit sink: no callable record()"
            )
        if not callable(clock):
            raise ConfigurationError("clock must be callable")
        if not callable(id_factory):
            raise ConfigurationError("id_factory must be callable")
        self._validator = validator
        self._audit_sink = audit_sink
        self._clock = clock
        self._id_factory = id_factory

    # -- introspection -----------------------------------------------
    @property
    def policy(self) -> Policy:
        """The policy this engine runs."""
        return self._policy

    @property
    def validator(self) -> Optional[DecisionValidator]:
        """The configured guardrail, if any."""
        return self._validator

    @property
    def audit_sink(self) -> Optional[AuditSink]:
        """The configured audit destination, if any."""
        return self._audit_sink

    # -- the one operation -------------------------------------------
    def decide(self, context: DecisionContext) -> Decision:
        """Produce a validated decision for ``context``.

        :raises PolicyError: the policy raised, or returned a non-Decision.
        :raises DecisionValidationError: the proposal was rejected and no
            fallback policy is configured (or the fallback was also rejected).
        :raises AuditError: the audit record could not be written. Audit failure
            fails the decision; a decision nobody can account for is worse than
            no decision.
        """
        return self.decide_with_record(context)[0]

    def decide_with_record(
        self, context: DecisionContext
    ) -> Tuple[Decision, AuditRecord]:
        """Like :meth:`decide`, but also returns the audit record.

        Useful when the caller wants the decision identifier or wants to hand
        the record to :func:`~serpentos.runtime.replay.replay` later without
        going through a sink.
        """
        if not isinstance(context, DecisionContext):
            raise ConfigurationError(
                f"context must be a DecisionContext, got {type(context).__name__}"
            )

        decision = self._invoke(self._policy, context)
        decision = self._identify(decision)
        result = self._validate(decision, context)
        record = self._emit(decision, context, result)

        if result.valid:
            return decision, record

        if self._fallback is None:
            raise DecisionValidationError(
                f"{result.validator} rejected action {decision.action!r} "
                f"from policy {decision.policy_name!r}: {result.reason}",
                result,
            )

        substitute = self._invoke(self._fallback, context)
        substitute = self._identify(
            replace(
                substitute,
                metadata={
                    **dict(substitute.metadata),
                    "fallback_for": decision.action,
                    "fallback_reason": result.reason,
                },
                decision_id=None,
            )
        )
        substitute_result = self._validate(substitute, context)
        substitute_record = self._emit(substitute, context, substitute_result)
        if not substitute_result.valid:
            raise DecisionValidationError(
                f"fallback policy {substitute.policy_name!r} also proposed a rejected "
                f"action {substitute.action!r}: {substitute_result.reason}",
                substitute_result,
            )
        return substitute, substitute_record

    # -- steps -------------------------------------------------------
    def _invoke(self, policy: Policy, context: DecisionContext) -> Decision:
        try:
            decision = policy.decide(context)
        except PolicyError:
            raise
        except Exception as exc:  # noqa: BLE001 - deliberately broad, then re-typed
            raise PolicyError(
                f"policy {getattr(policy, 'name', '?')!r} raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(decision, Decision):
            raise PolicyError(
                f"policy {getattr(policy, 'name', '?')!r} returned "
                f"{type(decision).__name__}, expected a Decision"
            )
        return decision

    def _identify(self, decision: Decision) -> Decision:
        if decision.decision_id:
            return decision
        return decision.with_decision_id(self._require_id())

    def _require_id(self) -> str:
        decision_id = self._id_factory()
        if not isinstance(decision_id, str) or not decision_id:
            raise ConfigurationError("id_factory must return a non-empty string")
        return decision_id

    def _validate(self, decision: Decision, context: DecisionContext) -> ValidationResult:
        if self._validator is None:
            return UNVALIDATED
        result = self._validator.validate(decision, context)
        if not isinstance(result, ValidationResult):
            raise ConfigurationError(
                f"{type(self._validator).__name__}.validate must return a "
                f"ValidationResult, got {type(result).__name__}"
            )
        return result

    def _emit(
        self, decision: Decision, context: DecisionContext, result: ValidationResult
    ) -> AuditRecord:
        timestamp = self._clock()
        if not isinstance(timestamp, str) or not timestamp:
            raise ConfigurationError("clock must return a non-empty string")
        record = AuditRecord.build(
            decision=decision,
            context=context,
            decision_id=decision.decision_id or self._require_id(),
            timestamp=timestamp,
            validation_result=result,
        )
        if self._audit_sink is not None:
            self._audit_sink.record(record)
        return record

    def __repr__(self) -> str:
        return (
            f"DecisionEngine(policy={self._policy.name!r}, "
            f"validator={getattr(self._validator, 'name', None)!r})"
        )
