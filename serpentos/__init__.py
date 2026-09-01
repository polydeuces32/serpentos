"""SerpentOS — an embedded policy runtime, with a Snake game as its reference
environment.

Two things live here, and it is worth being clear about which is which.

:mod:`serpentos.runtime` is the product: a small, standard-library-only kernel
for defining decision logic, running it under guardrails, recording what it
decided and proving later that it still decides the same way::

    from serpentos import ActionValidator, DecisionContext, DecisionEngine
    from serpentos.policies import Rule, RulePolicy, when

    policy = RulePolicy(
        name="retry-policy",
        version="1.0",
        rules=[Rule("retry", when("status_code", "in", [503, 504]))],
        default_action="fail",
    )
    engine = DecisionEngine(policy, validator=ActionValidator({"retry", "fail"}))
    print(engine.decide(DecisionContext({"status_code": 503})).action)  # retry

The Snake game — :mod:`serpentos.core`, :mod:`serpentos.serpentos`,
:mod:`serpentos.bot` — is the reference environment: a complete, honest example
of an application with state, actions, a learned policy, persistence and a
reproducible benchmark. It is a consumer of the runtime, not a dependency of it.

Importing this package pulls in the runtime only. Nothing here imports curses,
and nothing here imports the game.
"""

from __future__ import annotations

from .runtime import (
    REDACTED,
    ActionValidator,
    AuditError,
    AuditRecord,
    AuditSink,
    BasePolicy,
    ComparisonReport,
    ConfigurationError,
    Decision,
    DecisionContext,
    DecisionEngine,
    DecisionValidationError,
    DecisionValidator,
    InMemoryAuditLog,
    JsonlAuditLog,
    MetricSummary,
    NullAuditSink,
    Outcome,
    OutcomeSummary,
    Policy,
    PolicyError,
    PolicyReport,
    ReplayError,
    ReplayReport,
    ReplayResult,
    SerpentOSError,
    ValidationResult,
    compare,
    is_deterministic,
    read_jsonl,
    replay,
    replay_all,
)

__version__ = "2.0.0"

__all__ = [
    "__version__",
    # models
    "DecisionContext",
    "Decision",
    "Outcome",
    # policy interface
    "Policy",
    "BasePolicy",
    "is_deterministic",
    # engine
    "DecisionEngine",
    # validation
    "ActionValidator",
    "DecisionValidator",
    "ValidationResult",
    # audit
    "AuditRecord",
    "AuditSink",
    "InMemoryAuditLog",
    "JsonlAuditLog",
    "NullAuditSink",
    "read_jsonl",
    "REDACTED",
    # replay
    "replay",
    "replay_all",
    "ReplayResult",
    "ReplayReport",
    # comparison
    "compare",
    "ComparisonReport",
    "PolicyReport",
    "OutcomeSummary",
    "MetricSummary",
    # errors
    "SerpentOSError",
    "ConfigurationError",
    "PolicyError",
    "DecisionValidationError",
    "ReplayError",
    "AuditError",
]
