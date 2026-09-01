"""Run several policies over the same contexts and report what each one did.

This is an offline analysis tool. It never executes an action, never touches
production and never decides which policy is best — because "best" is a
property of your business, not of the runtime. What it gives you is the
evidence: how often each policy chose what, how often it failed, how often the
guardrails refused it, and, if you supply outcomes, how the numbers you care
about aggregate.

Resisting the urge to emit a single score is deliberate. A policy that retries
twice as often may be better or catastrophically worse depending on what a
retry costs you, and the runtime does not know.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .errors import ConfigurationError
from .models import Decision, DecisionContext, Outcome
from .policy import Policy, ensure_policy
from .validation import DecisionValidator, ValidationResult

__all__ = [
    "MetricSummary",
    "OutcomeSummary",
    "PolicyReport",
    "ComparisonReport",
    "compare",
]

#: Signature of the optional callback that reports what an action achieved.
OutcomeFn = Callable[[DecisionContext, Decision], Optional[Outcome]]


@dataclass(frozen=True)
class MetricSummary:
    """Aggregate of one named metric across the outcomes of one policy."""

    count: int
    total: float
    mean: float
    minimum: float
    maximum: float

    def to_dict(self) -> Dict[str, Any]:
        """A mutable, JSON-ready copy."""
        return {
            "count": self.count,
            "total": round(self.total, 6),
            "mean": round(self.mean, 6),
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass(frozen=True)
class OutcomeSummary:
    """Aggregate of the outcomes supplied for one policy."""

    count: int
    successes: int
    metrics: Mapping[str, MetricSummary]

    @property
    def success_rate(self) -> float:
        """Fraction of reported outcomes flagged successful, or 0.0 if none."""
        return self.successes / self.count if self.count else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """A mutable, JSON-ready copy."""
        return {
            "count": self.count,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 6),
            "metrics": {name: summary.to_dict() for name, summary in sorted(self.metrics.items())},
        }


@dataclass(frozen=True)
class PolicyReport:
    """What one policy did across the whole dataset."""

    policy_name: str
    policy_version: str
    decisions: int
    action_counts: Mapping[str, int]
    errors: int
    validation_failures: int
    error_samples: Tuple[str, ...] = ()
    outcomes: Optional[OutcomeSummary] = None

    def to_dict(self) -> Dict[str, Any]:
        """A mutable, JSON-ready copy with deterministic ordering."""
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "decisions": self.decisions,
            "action_counts": dict(sorted(self.action_counts.items())),
            "errors": self.errors,
            "validation_failures": self.validation_failures,
            "error_samples": list(self.error_samples),
            "outcomes": self.outcomes.to_dict() if self.outcomes is not None else None,
        }


@dataclass(frozen=True)
class ComparisonReport:
    """The result of one :func:`compare` run."""

    cases: int
    reports: Tuple[PolicyReport, ...]

    def for_policy(self, name: str) -> PolicyReport:
        """The report for the policy called ``name``.

        :raises KeyError: if no policy in the comparison had that name.
        """
        for report in self.reports:
            if report.policy_name == name:
                return report
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        """A mutable, JSON-ready copy."""
        return {
            "cases": self.cases,
            "policies": [report.to_dict() for report in self.reports],
        }


class _Tally:
    """Mutable accumulator; frozen into a PolicyReport at the end."""

    def __init__(self, policy: Policy, max_error_samples: int) -> None:
        self.name = policy.name
        self.version = policy.version
        self.decisions = 0
        self.actions: Dict[str, int] = {}
        self.errors = 0
        self.validation_failures = 0
        self.samples: List[str] = []
        self.max_error_samples = max_error_samples
        self.outcome_count = 0
        self.successes = 0
        self.metric_values: Dict[str, List[float]] = {}

    def note_error(self, message: str) -> None:
        self.errors += 1
        if len(self.samples) < self.max_error_samples:
            self.samples.append(message)

    def note_action(self, action: str) -> None:
        self.decisions += 1
        self.actions[action] = self.actions.get(action, 0) + 1

    def note_outcome(self, outcome: Outcome) -> None:
        self.outcome_count += 1
        if outcome.success:
            self.successes += 1
        for name, value in outcome.metrics.items():
            self.metric_values.setdefault(name, []).append(value)

    def finish(self, collected_outcomes: bool) -> PolicyReport:
        summary: Optional[OutcomeSummary] = None
        if collected_outcomes:
            metrics = {
                name: MetricSummary(
                    count=len(values),
                    total=sum(values),
                    mean=sum(values) / len(values),
                    minimum=min(values),
                    maximum=max(values),
                )
                for name, values in self.metric_values.items()
                if values
            }
            summary = OutcomeSummary(self.outcome_count, self.successes, metrics)
        return PolicyReport(
            policy_name=self.name,
            policy_version=self.version,
            decisions=self.decisions,
            action_counts=dict(self.actions),
            errors=self.errors,
            validation_failures=self.validation_failures,
            error_samples=tuple(self.samples),
            outcomes=summary,
        )


def compare(
    policies: Sequence[Policy],
    cases: Iterable[DecisionContext],
    *,
    validator: Optional[DecisionValidator] = None,
    outcome_fn: Optional[OutcomeFn] = None,
    max_error_samples: int = 5,
) -> ComparisonReport:
    """Run every policy over every case and tally the results.

    >>> report = compare([policy_a, policy_b], cases)          # doctest: +SKIP
    >>> report.for_policy("policy_a").action_counts            # doctest: +SKIP
    {'fail': 2, 'retry': 8}

    :param policies: the policies to evaluate. They are only asked to decide, so
        an evaluation-only policy is never mutated by this function.
    :param cases: the contexts to evaluate against. Consumed once and buffered,
        so a generator is fine.
    :param validator: optional guardrail. Rejections are counted rather than
        raised — in an offline comparison, "this policy proposes things we would
        refuse" is a finding, not a crash.
    :param outcome_fn: optional callback returning what a decision achieved.
        This is where you encode your definition of success. Return ``None`` to
        skip a case.
    :param max_error_samples: how many error messages to keep per policy, so a
        systematically broken policy does not produce a gigabyte of report.

    :raises ConfigurationError: if the arguments are unusable.

    A policy that raises is isolated: the exception is counted against that
    policy and the run continues, because the point of a comparison is to find
    out which policies are broken.
    """
    if not policies:
        raise ConfigurationError("compare() needs at least one policy")
    if max_error_samples < 0:
        raise ConfigurationError("max_error_samples must not be negative")
    for policy in policies:
        ensure_policy(policy)
    if validator is not None and not callable(getattr(validator, "validate", None)):
        raise ConfigurationError(
            f"{type(validator).__name__} is not a validator: no callable validate()"
        )
    if outcome_fn is not None and not callable(outcome_fn):
        raise ConfigurationError("outcome_fn must be callable")

    contexts = list(cases)
    for index, context in enumerate(contexts):
        if not isinstance(context, DecisionContext):
            raise ConfigurationError(
                f"cases[{index}] must be a DecisionContext, got {type(context).__name__}"
            )

    tallies = [_Tally(policy, max_error_samples) for policy in policies]

    for policy, tally in zip(policies, tallies):
        for context in contexts:
            try:
                decision = policy.decide(context)
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                tally.note_error(f"{type(exc).__name__}: {exc}")
                continue
            if not isinstance(decision, Decision):
                tally.note_error(
                    f"decide() returned {type(decision).__name__}, expected a Decision"
                )
                continue

            tally.note_action(decision.action)

            if validator is not None:
                result = validator.validate(decision, context)
                if not isinstance(result, ValidationResult):
                    raise ConfigurationError(
                        f"{type(validator).__name__}.validate must return a "
                        f"ValidationResult, got {type(result).__name__}"
                    )
                if not result.valid:
                    tally.validation_failures += 1

            if outcome_fn is not None:
                try:
                    outcome = outcome_fn(context, decision)
                except Exception as exc:  # noqa: BLE001 - caller code, isolated too
                    tally.note_error(f"outcome_fn raised {type(exc).__name__}: {exc}")
                    continue
                if outcome is None:
                    continue
                if not isinstance(outcome, Outcome):
                    raise ConfigurationError(
                        f"outcome_fn must return an Outcome or None, got "
                        f"{type(outcome).__name__}"
                    )
                tally.note_outcome(outcome)

    collected = outcome_fn is not None
    return ComparisonReport(
        cases=len(contexts),
        reports=tuple(tally.finish(collected) for tally in tallies),
    )
