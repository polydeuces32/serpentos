#!/usr/bin/env python3
"""Deciding whether to retry a failed request. No Snake anywhere in sight.

This is the smallest honest demonstration that SerpentOS is a decision engine
rather than a game: a service call failed, and something has to choose between
retrying, waiting and giving up. That choice is exactly the kind of logic that
normally ends up as three nested ``if`` statements nobody wants to touch.

Run it::

    python examples/retry_policy.py

No network, no server, no database, no model, no API key. It finishes in well
under a second.

What it walks through, in order:

1. **Deciding.** A rule policy proposes an action for each of six situations,
   behind an allow-list that refuses anything it has not been told about.
2. **Auditing.** Every decision is recorded as JSON explaining which rule fired.
3. **Replaying.** The recorded decisions are re-run to prove the policy still
   behaves identically — and then re-run against a *revised* policy to see
   exactly which decisions the change would have altered.
4. **Comparing.** A rule policy and a weighted-scoring policy are measured over
   the same situations, including where they disagree.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

# Prefer an installed SerpentOS; fall back to the clone this file sits in, so
# the example runs before `pip install` as well as after it.
if importlib.util.find_spec("serpentos") is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serpentos import (  # noqa: E402
    ActionValidator,
    DecisionContext,
    DecisionEngine,
    InMemoryAuditLog,
    compare,
    replay_all,
)
from serpentos.policies import (  # noqa: E402
    AllOf,
    LinearScorer,
    Rule,
    RulePolicy,
    WeightedPolicy,
    when,
)

# The only three things this system is allowed to do about a failed request.
ACTIONS = {"retry", "wait", "fail"}

# Six situations, chosen to exercise every branch.
SITUATIONS = [
    {"attempt": 1, "status_code": 503, "latency_ms": 120},
    {"attempt": 2, "status_code": 503, "latency_ms": 950},
    {"attempt": 4, "status_code": 503, "latency_ms": 130},
    {"attempt": 1, "status_code": 429, "latency_ms": 80},
    {"attempt": 1, "status_code": 400, "latency_ms": 45},
    {"attempt": 2, "status_code": 500, "latency_ms": 2400},
]


def retry_rules(version: str = "1.0", max_attempts: int = 3) -> RulePolicy:
    """Ordered rules. First match wins, so the order below is the logic.

    Reading top to bottom: give up after too many attempts; back off when told
    to; do not retry our own bad requests; retry server errors, but wait first
    if the server is already struggling.
    """
    return RulePolicy(
        name="retry-rules",
        version=version,
        rules=[
            Rule("fail", when("attempt", "ge", max_attempts), name="attempts-exhausted"),
            Rule("wait", when("status_code", "eq", 429), name="rate-limited"),
            Rule(
                "fail",
                AllOf(when("status_code", "ge", 400), when("status_code", "lt", 500)),
                name="client-error-is-our-fault",
            ),
            Rule(
                "wait",
                AllOf(when("status_code", "ge", 500), when("latency_ms", "ge", 1000)),
                name="server-is-struggling",
            ),
            Rule("retry", when("status_code", "ge", 500), name="transient-server-error"),
        ],
        default_action="fail",
    )


def retry_scoring() -> WeightedPolicy:
    """The same problem as a scoring model rather than a rule ladder.

    Each action gets a score and the highest wins. This is quantitative, not
    machine learning: the weights are numbers a human chose and can defend.
    """
    return WeightedPolicy(
        name="retry-scoring",
        version="1.0",
        scorers={
            # Retrying is attractive for server errors, less so each attempt.
            "retry": LinearScorer({"status_code": 0.02, "attempt": -1.5}, bias=-9.0),
            # Waiting gets better the slower the service is responding.
            "wait": LinearScorer({"latency_ms": 0.004, "attempt": -0.5}, bias=-2.0),
            # Failing is the baseline the other two have to beat.
            "fail": LinearScorer({"attempt": 1.2}, bias=-3.0),
        },
    )


def describe(values: dict) -> str:
    return (
        f"attempt {values['attempt']}, HTTP {values['status_code']}, "
        f"{values['latency_ms']}ms"
    )


def main() -> int:
    contexts = [
        DecisionContext(values, request_id=f"req-{index + 1:03d}")
        for index, values in enumerate(SITUATIONS)
    ]

    # ---------------------------------------------------------------
    # 1. Decide
    # ---------------------------------------------------------------
    policy = retry_rules()
    audit = InMemoryAuditLog()
    engine = DecisionEngine(
        policy=policy,
        validator=ActionValidator(allowed_actions=ACTIONS),
        audit_sink=audit,
    )

    print("=" * 78)
    print("1. DECIDING")
    print("=" * 78)
    for context in contexts:
        decision = engine.decide(context)
        print(f"\n  {describe(dict(context.values))}")
        print(f"    action      : {decision.action}")
        print(f"    policy      : {decision.policy_name} v{decision.policy_version}")
        print(f"    decision id : {decision.decision_id}")
        print(f"    because     : rule {decision.metadata['rule']!r}")

    # The engine proposed those actions. It did not perform any of them —
    # retrying the request is the calling application's job, not the runtime's.
    print("\n  Note: nothing was retried. The engine chooses; the caller acts.")

    # ---------------------------------------------------------------
    # 2. Audit
    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("2. AUDIT RECORD (the full JSON written for one decision)")
    print("=" * 78)
    print(json.dumps(audit.records[1].to_dict(), indent=2, sort_keys=True))

    # ---------------------------------------------------------------
    # 3. Replay
    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("3. REPLAY")
    print("=" * 78)

    same = replay_all(policy, audit.records)
    print(f"\n  Unchanged policy : {same.matched}/{same.total} match, "
          f"guaranteed={same.guaranteed}")

    # Now the question you actually want answered before shipping a change:
    # if we tightened the retry budget from 3 attempts to 2, what would move?
    revised = retry_rules(version="2.0", max_attempts=2)
    changed = replay_all(revised, audit.records, strict=False)
    print(f"  Revised policy   : {changed.matched}/{changed.total} match, "
          f"{changed.mismatched} would change")
    for result in changed.results:
        if not result.match:
            print(f"      {result.decision_id[:8]}  "
                  f"{result.original_action} -> {result.replayed_action}")

    # ---------------------------------------------------------------
    # 4. Compare
    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("4. COMPARISON")
    print("=" * 78)

    report = compare(
        [policy, retry_scoring()],
        contexts,
        validator=ActionValidator(ACTIONS),
    )
    for policy_report in report.reports:
        counts = ", ".join(
            f"{action}={count}"
            for action, count in sorted(policy_report.action_counts.items())
        )
        print(f"\n  {policy_report.label} v{policy_report.policy_version}")
        print(f"    actions            : {counts}")
        print(f"    errors             : {policy_report.errors}")
        print(f"    validation failures: {policy_report.validation_failures}")

    summary = report.disagreements
    print(f"\n  Agreed on {summary.agreed}/{summary.compared} situations "
          f"({summary.agreement_rate:.0%}).")
    for example in summary.examples:
        actions = ", ".join(f"{name}={action}" for name, action in sorted(example.actions.items()))
        print(f"    {describe(SITUATIONS[example.case_index])}  ->  {actions}")

    print(
        "\n  No 'winner' is declared, deliberately. Which of those is better "
        "\n  depends on what a retry costs you, and only you know that."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
