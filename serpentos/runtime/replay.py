"""Re-run recorded decisions and check the answer still holds.

Replay answers a specific question: *given the same inputs, does this policy
still decide the same way?* That is the question you need before shipping a
policy change, and the question you need after an incident.

What replay does **not** do is prove the policy was right. It compares the
recorded action against a freshly computed one. Nothing more.

Determinism is a property of the policy, not of the replay machinery. A policy
that reads an unseeded random source, the clock or the network cannot be
replayed meaningfully, and the runtime will not pretend otherwise: such policies
set ``deterministic = False`` and every result they produce is flagged
``guaranteed=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from .audit import AuditRecord
from .errors import PolicyError, ReplayError
from .models import Decision, DecisionContext
from .policy import Policy, ensure_policy, is_deterministic

__all__ = ["ReplayResult", "ReplayReport", "replay", "replay_all"]


@dataclass(frozen=True)
class ReplayResult:
    """The outcome of replaying one audit record."""

    decision_id: str
    original_action: str
    replayed_action: str
    match: bool
    #: ``False`` when the policy declares itself nondeterministic, in which case
    #: a match is evidence but not proof.
    guaranteed: bool
    policy_name: str
    policy_version: str

    def to_dict(self) -> Dict[str, Any]:
        """A mutable, JSON-ready copy."""
        return {
            "decision_id": self.decision_id,
            "original_action": self.original_action,
            "replayed_action": self.replayed_action,
            "match": self.match,
            "guaranteed": self.guaranteed,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class ReplayReport:
    """Aggregate of replaying many records against one policy."""

    results: Tuple[ReplayResult, ...]
    errors: Tuple[Tuple[str, str], ...] = ()

    @property
    def total(self) -> int:
        """Records that produced a result, successfully or not."""
        return len(self.results) + len(self.errors)

    @property
    def matched(self) -> int:
        """Records whose replayed action equalled the recorded one."""
        return sum(1 for result in self.results if result.match)

    @property
    def mismatched(self) -> int:
        """Records whose replayed action differed."""
        return sum(1 for result in self.results if not result.match)

    @property
    def guaranteed(self) -> bool:
        """Whether every result came from a policy claiming determinism."""
        return all(result.guaranteed for result in self.results)

    def to_dict(self) -> Dict[str, Any]:
        """A mutable, JSON-ready copy."""
        return {
            "total": self.total,
            "matched": self.matched,
            "mismatched": self.mismatched,
            "errors": [{"decision_id": did, "error": msg} for did, msg in self.errors],
            "guaranteed": self.guaranteed,
            "results": [result.to_dict() for result in self.results],
        }


def _context_of(record: AuditRecord) -> DecisionContext:
    if record.context is None:
        raise ReplayError(
            f"decision {record.decision_id} was recorded without its context "
            "(the audit sink was configured with include_context=False); "
            "there is nothing to replay against"
        )
    return record.context


def replay(policy: Policy, record: AuditRecord, *, strict: bool = True) -> ReplayResult:
    """Re-run ``record``'s context through ``policy`` and compare the action.

    :param policy: the policy to re-run. To reproduce the original decision this
        must be the same policy, at the same version, with the same
        configuration — the runtime can check the name and version but cannot
        verify the configuration, so that part is on you.
    :param record: the recorded decision.
    :param strict: when true (the default), refuse to replay a record produced
        by a differently named or versioned policy. Set it to false to
        deliberately run a *new* policy against *old* traffic, which is how you
        see what a change would have done.

    :raises ReplayError: if the record has no context, if ``strict`` is set and
        the identities disagree, or if the policy fails during replay.
    """
    ensure_policy(policy)
    context = _context_of(record)

    if strict:
        if record.policy_name != policy.name or record.policy_version != policy.version:
            raise ReplayError(
                f"decision {record.decision_id} was made by "
                f"{record.policy_name}@{record.policy_version}, but replay was given "
                f"{policy.name}@{policy.version}; pass strict=False to compare "
                "different policies against recorded traffic"
            )

    try:
        decision = policy.decide(context)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, then re-typed
        raise ReplayError(
            f"policy {policy.name!r} raised {type(exc).__name__} while replaying "
            f"decision {record.decision_id}: {exc}"
        ) from exc

    if not isinstance(decision, Decision):
        raise ReplayError(
            f"policy {policy.name!r} returned {type(decision).__name__} while "
            f"replaying decision {record.decision_id}, expected a Decision"
        )

    return ReplayResult(
        decision_id=record.decision_id,
        original_action=record.action,
        replayed_action=decision.action,
        match=decision.action == record.action,
        guaranteed=is_deterministic(policy),
        policy_name=policy.name,
        policy_version=policy.version,
    )


def replay_all(
    policy: Policy, records: Iterable[AuditRecord], *, strict: bool = True
) -> ReplayReport:
    """Replay many records, isolating failures instead of aborting.

    One unreplayable record — a missing context, a policy that raises on some
    weird historical input — must not hide the verdict on the other ten
    thousand. Failures are collected in :attr:`ReplayReport.errors`.
    """
    ensure_policy(policy)
    results = []
    errors = []
    for record in records:
        if not isinstance(record, AuditRecord):
            raise ReplayError(
                f"records must be AuditRecord instances, got {type(record).__name__}"
            )
        try:
            results.append(replay(policy, record, strict=strict))
        except (ReplayError, PolicyError) as exc:
            errors.append((record.decision_id, str(exc)))
    return ReplayReport(tuple(results), tuple(errors))
