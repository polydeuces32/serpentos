"""Pure HTTP-facing decision adapter used by the Cloudflare Worker.

This module deliberately contains no Cloudflare imports so its behavior can be
validated by the normal CPython test suite. The Worker entrypoint is only a
transport shim around :func:`handle_decision`.
"""

from __future__ import annotations

from typing import Any, Mapping

from serpentos import ActionValidator, DecisionContext, DecisionEngine
from serpentos.policies import AllOf, Rule, RulePolicy, when

ACTIONS = {"retry", "wait", "fail"}
_MAX_ATTEMPT = 1_000
_MAX_STATUS = 599
_MAX_LATENCY_MS = 3_600_000


def _retry_policy() -> RulePolicy:
    return RulePolicy(
        name="cloudflare-retry-policy",
        version="1.0",
        rules=[
            Rule("fail", when("attempt", "ge", 3), name="attempts-exhausted"),
            Rule("wait", when("status_code", "eq", 429), name="rate-limited"),
            Rule(
                "fail",
                AllOf(
                    when("status_code", "ge", 400),
                    when("status_code", "lt", 500),
                ),
                name="client-error",
            ),
            Rule(
                "wait",
                AllOf(
                    when("status_code", "ge", 500),
                    when("latency_ms", "ge", 1000),
                ),
                name="server-struggling",
            ),
            Rule(
                "retry",
                when("status_code", "ge", 500),
                name="transient-server-error",
            ),
        ],
        default_action="fail",
    )


_ENGINE = DecisionEngine(
    _retry_policy(),
    validator=ActionValidator(ACTIONS),
)


def _require_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def handle_decision(payload: Any) -> tuple[int, dict[str, Any]]:
    """Validate an HTTP JSON payload and return a SerpentOS decision.

    The endpoint intentionally exposes one fixed, auditable policy. It does not
    accept code, expressions, policy definitions, or arbitrary action names.
    """

    if not isinstance(payload, Mapping):
        return 400, {"error": "request body must be a JSON object"}

    try:
        attempt = _require_int(payload, "attempt", minimum=0, maximum=_MAX_ATTEMPT)
        status_code = _require_int(payload, "status_code", minimum=100, maximum=_MAX_STATUS)
        latency_ms = _require_int(payload, "latency_ms", minimum=0, maximum=_MAX_LATENCY_MS)

        request_id = payload.get("request_id")
        if request_id is not None:
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request_id must be a non-empty string")
            if len(request_id) > 128:
                raise ValueError("request_id must be at most 128 characters")

        context = DecisionContext(
            {
                "attempt": attempt,
                "status_code": status_code,
                "latency_ms": latency_ms,
            },
            request_id=request_id,
        )
        decision, record = _ENGINE.decide_with_record(context)
    except ValueError as exc:
        return 400, {"error": str(exc)}

    return 200, {
        "decision": decision.to_dict(),
        "audit": record.to_dict(),
    }


def health_payload() -> dict[str, str]:
    return {
        "service": "serpentos",
        "status": "ok",
        "policy": "cloudflare-retry-policy",
        "policy_version": "1.0",
    }
