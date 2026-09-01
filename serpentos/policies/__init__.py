"""Policy implementations shipped with SerpentOS.

Three of them, chosen to cover three genuinely different shapes of decision
logic rather than three variations on one:

* :class:`~serpentos.policies.rules.RulePolicy` — ordered conditions, first
  match wins. Serialisable as data.
* :class:`~serpentos.policies.weighted.WeightedPolicy` — score every candidate,
  take the highest. Callback-driven, or linear and serialisable.
* :class:`~serpentos.policies.qlearning.QLearningPolicy` — a read-only adapter
  over the trained tabular agent that powers the Snake reference environment.

The first two have nothing to do with machine learning, which is the point: the
runtime is not an ML framework with a general-purpose veneer.
"""

from __future__ import annotations

from .rules import (
    AllOf,
    Always,
    AnyOf,
    Comparison,
    Condition,
    Not,
    Predicate,
    Rule,
    RulePolicy,
    condition_from_dict,
    when,
)
from .weighted import LinearScorer, WeightedPolicy

__all__ = [
    "RulePolicy",
    "Rule",
    "Condition",
    "Comparison",
    "AllOf",
    "AnyOf",
    "Not",
    "Always",
    "Predicate",
    "when",
    "condition_from_dict",
    "WeightedPolicy",
    "LinearScorer",
    "QLearningPolicy",
]


def __getattr__(name: str):
    """Import the Q-learning adapter lazily.

    It pulls in :mod:`serpentos.core`, and there is no reason for a project
    using only rule policies to pay for the Snake environment.
    """
    if name == "QLearningPolicy":
        from .qlearning import QLearningPolicy

        return QLearningPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
