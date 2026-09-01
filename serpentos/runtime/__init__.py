"""The SerpentOS policy runtime.

A small, dependency-free kernel for defining decision logic as data, running it
under guardrails, recording what it decided and proving later that it still
decides the same way.

::

    DecisionContext  ->  Policy  ->  Decision  ->  Outcome
                            |
                      DecisionEngine
                       /     |     \\
              validation  audit   identity
                            |
                    replay / comparison

Nothing in this package imports curses, touches the network, executes strings or
depends on the Snake environment. Snake is a consumer of this runtime, not part
of it.
"""

from __future__ import annotations

from .audit import (
    AUDIT_SCHEMA_VERSION,
    REDACTED,
    SUPPORTED_AUDIT_SCHEMA_VERSIONS,
    AuditRecord,
    AuditSink,
    InMemoryAuditLog,
    JsonlAuditLog,
    NullAuditSink,
    read_jsonl,
)
from .comparison import (
    ComparisonReport,
    Disagreement,
    DisagreementSummary,
    MetricSummary,
    OutcomeSummary,
    PairDisagreement,
    PolicyReport,
    compare,
)
from .engine import DecisionEngine
from .errors import (
    AuditError,
    ConfigurationError,
    DecisionValidationError,
    PolicyError,
    ReplayError,
    SerpentOSError,
)
from .models import Decision, DecisionContext, Outcome
from .policy import BasePolicy, Policy, is_deterministic
from .replay import ReplayReport, ReplayResult, replay, replay_all
from .validation import ActionValidator, DecisionValidator, ValidationResult

__all__ = [
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
    "AUDIT_SCHEMA_VERSION",
    "SUPPORTED_AUDIT_SCHEMA_VERSIONS",
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
    "DisagreementSummary",
    "Disagreement",
    "PairDisagreement",
    # errors
    "SerpentOSError",
    "ConfigurationError",
    "PolicyError",
    "DecisionValidationError",
    "ReplayError",
    "AuditError",
]
